from __future__ import annotations

import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]

BOOK_DIR = ROOT / "data" / "experiments" / "orderbook_tensor_audit_v1"
SEQ_DIR = ROOT / "data" / "experiments" / "complex_v1a_sequence_probe_v1"
BOOK_ARRAYS_PATH = BOOK_DIR / "orderbook_tensor_audit_arrays.npz"
BOOK_MANIFEST_PATH = BOOK_DIR / "manifest.json"
SEQ_ARRAYS_PATH = SEQ_DIR / "sequence_arrays.npz"
INDEX_PATH = SEQ_DIR / "sequence_index.csv"

OUT_DIR = ROOT / "data" / "experiments" / "book_only_conv1d_baseline_v1"
MODEL_DIR = OUT_DIR / "models"
NOTEBOOK_PATH = ROOT / "notebooks" / "27_book_only_conv1d_baseline_v1.ipynb"
DOC_PATH = ROOT / "docs" / "BOOK_ONLY_CONV1D_BASELINE_V1.md"

SEED = 53
TARGET = "y_last_cost0p5"
MAX_EPOCHS = 100
BATCH_SIZE = 192
PATIENCE = 18
LR = 0.0015
WEIGHT_DECAY = 1e-4

MIN_VALIDATION_ACTIONS = 80
MIN_VALIDATION_DAY_ACTIONS = 20
MIN_TEST_ACTIONS = 80

EV_THRESHOLDS = [-2.0, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PROBA_THRESHOLDS = [None, 0.50, 0.55, 0.60, 0.65]
TOP_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75]

CONTINUOUS_CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true).astype(bool)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, np.asarray(score)))


def split_metrics(y: np.ndarray, pred: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)) if len(y) >= 2 else np.nan,
        "spearman_corr": float(pd.Series(y).corr(pd.Series(pred), method="spearman")) if len(y) >= 2 else np.nan,
        "auc_positive_cost0p5": safe_auc(y > 0, pred),
        "auc_classifier_positive_cost0p5": safe_auc(y > 0, proba),
    }


def fit_transform_book_tensor(x: np.ndarray, mask: np.ndarray, split: pd.Series) -> tuple[np.ndarray, dict]:
    train_idx = split.eq("train_initial").to_numpy()
    valid_steps = mask[train_idx]
    train_values = x[train_idx][valid_steps]

    mean = np.zeros(x.shape[-1], dtype="float32")
    std = np.ones(x.shape[-1], dtype="float32")
    for channel in CONTINUOUS_CHANNELS:
        values = train_values[..., channel].reshape(-1)
        mean[channel] = float(values.mean())
        std[channel] = float(values.std() if values.std() > 1e-6 else 1.0)

    x_scaled = x.astype("float32").copy()
    for channel in CONTINUOUS_CHANNELS:
        x_scaled[..., channel] = (x_scaled[..., channel] - mean[channel]) / std[channel]
    x_scaled[~mask] = 0.0

    scaler = {
        "continuous_channels": CONTINUOUS_CHANNELS,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "note": "Only continuous book channels are standardized; presence masks stay as 0/1.",
    }
    return x_scaled, scaler


class BookEncoder(nn.Module):
    def __init__(self, n_channels: int, out_channels: int = 40) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Conv1d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.out_features = out_channels * 2

    def forward(self, book_steps: torch.Tensor) -> torch.Tensor:
        # book_steps: batch, level, channel.
        h = self.conv(book_steps.transpose(1, 2))
        return torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)


class BookLastSnapshotConv(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.encoder = BookEncoder(n_channels=n_channels, out_channels=40)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_features, 48),
            nn.ReLU(),
            nn.Dropout(0.12),
            nn.Linear(48, 24),
            nn.ReLU(),
        )
        self.reg_head = nn.Linear(24, 1)
        self.clf_head = nn.Linear(24, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = (lengths.long() - 1).clamp_min(0)
        last = x[torch.arange(x.shape[0], device=x.device), idx]
        h = self.head(self.encoder(last))
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


class BookSeqConvGRU(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.encoder = BookEncoder(n_channels=n_channels, out_channels=40)
        self.gru = nn.GRU(input_size=self.encoder.out_features, hidden_size=56, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(56, 32), nn.ReLU(), nn.Dropout(0.12))
        self.reg_head = nn.Linear(32, 1)
        self.clf_head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, levels, channels = x.shape
        emb = self.encoder(x.reshape(batch * steps, levels, channels)).reshape(batch, steps, -1)
        emb = emb * mask.unsqueeze(-1).float()
        out, _ = self.gru(emb)
        idx = (lengths.long() - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        h = out.gather(1, idx).squeeze(1)
        h = self.head(h)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


def make_model(name: str, n_channels: int) -> nn.Module:
    if name == "book_last_snapshot_conv":
        return BookLastSnapshotConv(n_channels)
    if name == "book_seq_conv_gru":
        return BookSeqConvGRU(n_channels)
    raise ValueError(name)


def train_one(
    model_name: str,
    x: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    split: pd.Series,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    _, _, _, n_channels = x.shape
    train_idx = split.eq("train_initial").to_numpy()
    val_idx = split.eq("validation_initial").to_numpy()

    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() if y[train_idx].std() > 1e-6 else 1.0)
    y_scaled = ((y - y_mean) / y_std).astype("float32")
    y_pos = (y > 0).astype("float32")

    train_ds = TensorDataset(
        torch.tensor(x[train_idx], dtype=torch.float32),
        torch.tensor(mask[train_idx], dtype=torch.bool),
        torch.tensor(lengths[train_idx], dtype=torch.long),
        torch.tensor(y_scaled[train_idx], dtype=torch.float32),
        torch.tensor(y_pos[train_idx], dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = make_model(model_name, n_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    reg_loss_fn = nn.SmoothL1Loss()
    clf_loss_fn = nn.BCEWithLogitsLoss()

    best_score = -math.inf
    best_state = None
    best_epoch = 0
    no_improve = 0

    val_tensors = (
        torch.tensor(x[val_idx], dtype=torch.float32, device=device),
        torch.tensor(mask[val_idx], dtype=torch.bool, device=device),
        torch.tensor(lengths[val_idx], dtype=torch.long, device=device),
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, mb, lb, yb, pb in train_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)
            pb = pb.to(device)
            optimizer.zero_grad(set_to_none=True)
            reg_scaled, logits = model(xb, mb, lb)
            loss = reg_loss_fn(reg_scaled, yb) + 0.35 * clf_loss_fn(logits, pb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reg_scaled, val_logits = model(*val_tensors)
            val_pred = (val_reg_scaled.detach().cpu().numpy() * y_std + y_mean).astype("float32")
            val_proba = torch.sigmoid(val_logits).detach().cpu().numpy().astype("float32")
        val_y = y[val_idx]
        auc = safe_auc(val_y > 0, val_proba)
        spearman = float(pd.Series(val_y).corr(pd.Series(val_pred), method="spearman"))
        score = (0.0 if np.isnan(auc) else auc) + 0.10 * (0.0 if np.isnan(spearman) else spearman)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    preds = []
    probas = []
    all_ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(lengths, dtype=torch.long),
    )
    all_loader = DataLoader(all_ds, batch_size=512, shuffle=False)
    with torch.no_grad():
        for xb, mb, lb in all_loader:
            xb = xb.to(device)
            mb = mb.to(device)
            lb = lb.to(device)
            reg_scaled, logits = model(xb, mb, lb)
            preds.append((reg_scaled.detach().cpu().numpy() * y_std + y_mean).astype("float32"))
            probas.append(torch.sigmoid(logits).detach().cpu().numpy().astype("float32"))
    pred = np.concatenate(preds)
    proba = np.concatenate(probas)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "state_dict": model.state_dict(),
            "target_mean": y_mean,
            "target_std": y_std,
            "best_epoch": best_epoch,
            "best_score": best_score,
        },
        MODEL_DIR / f"{model_name}.pt",
    )

    info = {
        "model_name": model_name,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "target_mean": y_mean,
        "target_std": y_std,
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    for split_name in ["train_initial", "validation_initial", "test_terminal"]:
        sm = split.eq(split_name).to_numpy()
        info.update({f"{split_name}_{key}": value for key, value in split_metrics(y[sm], pred[sm], proba[sm]).items()})
    return pred, proba, info


def summarize(index: pd.DataFrame, selected: np.ndarray) -> dict:
    part = index[selected].copy()
    if part.empty:
        return {
            "actions": 0,
            "mean_net_cost_0p25": np.nan,
            "mean_net_cost_0p5": np.nan,
            "mean_net_cost_1p0": np.nan,
            "positive_cost_0p5_pct": np.nan,
            "mean_pred": np.nan,
            "mean_proba": np.nan,
        }
    return {
        "actions": int(len(part)),
        "mean_net_cost_0p25": float(part["last_exec_net_cost_0p25_H60"].mean()),
        "mean_net_cost_0p5": float(part["last_exec_net_cost_0p5_H60"].mean()),
        "mean_net_cost_1p0": float(part["last_exec_net_cost_1p0_H60"].mean()),
        "positive_cost_0p5_pct": float((part["last_exec_net_cost_0p5_H60"] > 0).mean() * 100),
        "mean_pred": float(part["pred"].mean()),
        "mean_proba": float(part["proba"].mean()),
    }


def candidate_thresholds(train_pred: np.ndarray) -> list[dict]:
    rows = []
    for threshold in EV_THRESHOLDS:
        for proba in PROBA_THRESHOLDS:
            suffix = "" if proba is None else f"_p_ge_{proba:g}"
            rows.append({"policy_name": f"pred_ge_{threshold:g}{suffix}", "pred_threshold": threshold, "proba_threshold": proba})
    for fraction in TOP_FRACTIONS:
        threshold = float(np.quantile(train_pred, 1.0 - fraction))
        for proba in PROBA_THRESHOLDS:
            suffix = "" if proba is None else f"_p_ge_{proba:g}"
            rows.append({"policy_name": f"top_{fraction * 100:.3g}pct{suffix}", "pred_threshold": threshold, "proba_threshold": proba})
    deduped = {}
    for row in rows:
        deduped[row["policy_name"]] = row
    return list(deduped.values())


def policy_mask(index: pd.DataFrame, policy: dict) -> np.ndarray:
    out = index["pred"].ge(float(policy["pred_threshold"])).to_numpy()
    proba = policy["proba_threshold"]
    if proba is not None and pd.notna(proba):
        out = out & index["proba"].ge(float(proba)).to_numpy()
    return out


def evaluate_policies(index: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_pred = index.loc[index["terminal_split"].eq("train_initial"), "pred"].to_numpy()
    rows = []
    day_frames = []
    for policy in candidate_thresholds(train_pred):
        selected = policy_mask(index, policy)
        row = {
            "model_name": model_name,
            **policy,
            **{
                f"validation_{key}": value
                for key, value in summarize(index, selected & index["terminal_split"].eq("validation_initial").to_numpy()).items()
            },
            **{f"test_{key}": value for key, value in summarize(index, selected & index["terminal_split"].eq("test_terminal").to_numpy()).items()},
        }
        day_rows = []
        part = index[selected].copy()
        for (split_name, day), day_part in part.groupby(["terminal_split", "session_day"], observed=True):
            idx = part.index.isin(day_part.index)
            day_rows.append(
                {
                    "model_name": model_name,
                    "policy_name": policy["policy_name"],
                    "terminal_split": split_name,
                    "session_day": day,
                    **summarize(part.reset_index(drop=True), idx),
                }
            )
        day_df = pd.DataFrame(day_rows)
        if not day_df.empty:
            day_frames.append(day_df)
        validation_days = day_df[day_df["terminal_split"].eq("validation_initial")] if not day_df.empty else day_df
        test_days = day_df[day_df["terminal_split"].eq("test_terminal")] if not day_df.empty else day_df
        valid_day_eval = validation_days[validation_days["actions"].ge(MIN_VALIDATION_DAY_ACTIONS)] if not day_df.empty else day_df
        test_day_eval = test_days[test_days["actions"].gt(0)] if not day_df.empty else day_df
        row["validation_days"] = int(validation_days["session_day"].nunique()) if not day_df.empty else 0
        row["validation_days_ge_min_actions"] = int(valid_day_eval["session_day"].nunique()) if not day_df.empty else 0
        row["validation_negative_days_cost0p5"] = int(valid_day_eval["mean_net_cost_0p5"].le(0).sum()) if not valid_day_eval.empty else 0
        row["validation_min_day_cost0p5"] = float(valid_day_eval["mean_net_cost_0p5"].min()) if not valid_day_eval.empty else np.nan
        row["test_days"] = int(test_days["session_day"].nunique()) if not day_df.empty else 0
        row["test_negative_days_cost0p5"] = int(test_day_eval["mean_net_cost_0p5"].le(0).sum()) if not test_day_eval.empty else 0
        row["test_min_day_cost0p5"] = float(test_day_eval["mean_net_cost_0p5"].min()) if not test_day_eval.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()


def select_candidates(policy_results: pd.DataFrame) -> pd.DataFrame:
    candidates = policy_results[
        policy_results["validation_actions"].ge(MIN_VALIDATION_ACTIONS)
        & policy_results["validation_mean_net_cost_0p25"].gt(0)
        & policy_results["validation_mean_net_cost_0p5"].gt(0)
        & policy_results["validation_mean_net_cost_1p0"].gt(0)
        & policy_results["validation_negative_days_cost0p5"].eq(0)
        & policy_results["validation_days_ge_min_actions"].ge(2)
    ].copy()
    if candidates.empty:
        return candidates
    candidates["selection_score"] = (
        candidates["validation_mean_net_cost_1p0"] * np.sqrt(candidates["validation_actions"])
        + candidates["validation_min_day_cost0p5"].clip(lower=-5).fillna(-5)
    )
    return candidates.sort_values(["selection_score", "validation_actions"], ascending=[False, False])


def decide(selected: pd.DataFrame, policy_results: pd.DataFrame) -> dict:
    if selected.empty:
        return {"decision": "NO_GO_BOOK_ONLY_CONV1D", "reason": "No validation-stable book-only policy found."}
    best = selected.iloc[0].to_dict()
    selected_pass = (
        best["test_actions"] >= MIN_TEST_ACTIONS
        and best["test_mean_net_cost_0p25"] > 0
        and best["test_mean_net_cost_0p5"] > 0
        and best["test_mean_net_cost_1p0"] > 0
        and best["test_negative_days_cost0p5"] == 0
    )
    any_test_pass = policy_results[
        policy_results["test_actions"].ge(MIN_TEST_ACTIONS)
        & policy_results["test_mean_net_cost_0p25"].gt(0)
        & policy_results["test_mean_net_cost_0p5"].gt(0)
        & policy_results["test_mean_net_cost_1p0"].gt(0)
        & policy_results["test_negative_days_cost0p5"].eq(0)
    ]
    keys = [
        "model_name",
        "policy_name",
        "validation_actions",
        "validation_mean_net_cost_0p25",
        "validation_mean_net_cost_0p5",
        "validation_mean_net_cost_1p0",
        "validation_negative_days_cost0p5",
        "test_actions",
        "test_mean_net_cost_0p25",
        "test_mean_net_cost_0p5",
        "test_mean_net_cost_1p0",
        "test_negative_days_cost0p5",
        "test_min_day_cost0p5",
    ]
    return {
        "decision": "RESEARCH_PASS_BOOK_ONLY_CONV1D" if selected_pass else "NO_GO_BOOK_ONLY_CONV1D",
        "reason": (
            "Validation-selected book-only policy is test-positive with no negative test days."
            if selected_pass
            else "Validation-selected book-only policy fails terminal stability."
        ),
        "best_policy": {key: best.get(key) for key in keys},
        "any_test_pass_policy_count": int(len(any_test_pass)),
        "next_step": "If book-only ranks reasonably, train a fusion model: book encoder + tabular sequence GRU.",
    }


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Book-only Conv1D baseline v1\n",
                    "\n",
                    "Este notebook resume una prueba sencilla: usar solo el orderbook visible de Polymarket, convertido a tensor, para ver si contiene senal predictiva.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Que estamos probando\n",
                    "\n",
                    "El tensor tiene forma `(secuencia, paso, nivel, canal)`. Cada nivel representa un precio del libro y los canales guardan tamanos, acumulados, distancias al mid y mascaras de presencia.\n",
                    "\n",
                    "Entrenamos dos modelos:\n",
                    "\n",
                    "- `book_last_snapshot_conv`: mira solo la ultima foto antes de decidir.\n",
                    "- `book_seq_conv_gru`: resume cada foto con una Conv1D y luego mira la dinamica temporal con una GRU.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "import json\n",
                    "import pandas as pd\n",
                    "OUT = Path('../data/experiments/book_only_conv1d_baseline_v1')\n",
                    "decision = json.loads((OUT / 'decision.json').read_text(encoding='utf-8'))\n",
                    "metrics = pd.read_csv(OUT / 'model_metrics.csv')\n",
                    "selected = pd.read_csv(OUT / 'selected_candidates.csv')\n",
                    "days = pd.read_csv(OUT / 'day_breakdown.csv')\n",
                    "decision\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## Metricas de ranking\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["metrics.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Candidatos seleccionados por validacion\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["selected.head(20).round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Desglose diario del mejor candidato\n"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "best = decision.get('best_policy', {})\n",
                    "mask = days['model_name'].eq(best.get('model_name')) & days['policy_name'].eq(best.get('policy_name'))\n",
                    "days[mask].sort_values(['terminal_split','session_day']).round(5)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def format_float(value: object, digits: int = 4) -> str:
    try:
        if value is None or pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def write_doc(decision: dict, metrics: pd.DataFrame, selected: pd.DataFrame, days: pd.DataFrame, manifest: dict) -> None:
    best = decision.get("best_policy", {})
    lines = [
        "# Book-only Conv1D baseline v1",
        "",
        "Fecha: 2026-06-02",
        "",
        "## Objetivo",
        "",
        "Probar si el orderbook visible de Polymarket, tratado como un tensor, contiene senal predictiva por si solo.",
        "",
        "Esto no es aun el modelo final. Es una prueba intermedia para decidir si merece la pena pasar a fusion:",
        "",
        "```text",
        "book encoder + secuencia tabular + gestion de riesgo",
        "```",
        "",
        "## Dataset",
        "",
        "Tensor usado:",
        "",
        "```text",
        f"{tuple(manifest['book_tensor_shape'])}",
        "```",
        "",
        "Canales:",
        "",
        "```text",
        json.dumps(manifest["channels"], indent=2),
        "```",
        "",
        "Target:",
        "",
        "```text",
        "y_last_cost0p5",
        "```",
        "",
        "Interpretacion sencilla:",
        "",
        "```text",
        "edge neto del ultimo snapshot, despues de aplicar un coste teorico de 0.50.",
        "```",
        "",
        "## Modelos entrenados",
        "",
        "| Modelo | Parametros | Idea |",
        "|---|---:|---|",
    ]
    idea = {
        "book_last_snapshot_conv": "Conv1D sobre los 10 niveles del ultimo snapshot.",
        "book_seq_conv_gru": "Conv1D por snapshot + GRU sobre la secuencia temporal.",
    }
    for _, row in metrics.iterrows():
        lines.append(f"| `{row['model_name']}` | {int(row['n_parameters'])} | {idea.get(row['model_name'], '')} |")

    lines += [
        "",
        "## Ranking",
        "",
        "| Modelo | Spearman val | AUC val | Spearman test | AUC test |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row["model_name"],
                format_float(row["validation_initial_spearman_corr"]),
                format_float(row["validation_initial_auc_classifier_positive_cost0p5"]),
                format_float(row["test_terminal_spearman_corr"]),
                format_float(row["test_terminal_auc_classifier_positive_cost0p5"]),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        "Decision:",
        "",
        "```text",
        str(decision.get("decision")),
        "```",
        "",
        "Razon:",
        "",
        "```text",
        str(decision.get("reason")),
        "```",
        "",
    ]

    if best:
        lines += [
            "Mejor policy seleccionada por validacion:",
            "",
            "```text",
            f"{best.get('model_name')} + {best.get('policy_name')}",
            "```",
            "",
            "| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| validation | {int(best.get('validation_actions', 0))} | "
                f"{format_float(best.get('validation_mean_net_cost_0p25'))} | "
                f"{format_float(best.get('validation_mean_net_cost_0p5'))} | "
                f"{format_float(best.get('validation_mean_net_cost_1p0'))} | "
                f"{int(best.get('validation_negative_days_cost0p5', 0))} |"
            ),
            (
                f"| test | {int(best.get('test_actions', 0))} | "
                f"{format_float(best.get('test_mean_net_cost_0p25'))} | "
                f"{format_float(best.get('test_mean_net_cost_0p5'))} | "
                f"{format_float(best.get('test_mean_net_cost_1p0'))} | "
                f"{int(best.get('test_negative_days_cost0p5', 0))} |"
            ),
            "",
        ]
        best_days = days[
            days["model_name"].eq(best.get("model_name"))
            & days["policy_name"].eq(best.get("policy_name"))
            & days["terminal_split"].isin(["validation_initial", "test_terminal"])
        ].copy()
        if not best_days.empty:
            lines += [
                "Desglose diario del mejor candidato:",
                "",
                "| Dia | Split | Acciones | Net 0.50 | Net 1.00 |",
                "|---|---|---:|---:|---:|",
            ]
            for _, row in best_days.sort_values(["terminal_split", "session_day"]).iterrows():
                lines.append(
                    f"| {row['session_day']} | {row['terminal_split']} | {int(row['actions'])} | "
                    f"{format_float(row['mean_net_cost_0p5'])} | {format_float(row['mean_net_cost_1p0'])} |"
                )
            lines.append("")

    lines += [
        "## Lectura",
        "",
        "Este baseline responde a una pregunta concreta:",
        "",
        "```text",
        "si miro solo la forma del libro visible, saco ventaja estable?",
        "```",
        "",
        "Si el resultado queda cerca del GRU tabular, merece la pena fusionar ambos mundos.",
        "Si queda claramente peor, el orderbook visible se usara mas como contexto auxiliar/regimen que como fuente principal.",
        "",
        "## Comparacion cualitativa con el estado actual",
        "",
        "| Familia | Estado | Lectura corta |",
        "|---|---|---|",
        "| Flatten tabular | NO_GO | Buena AUC test agregada, pero falla estabilidad diaria. |",
        "| GRU tabular | NO_GO | Mejor policy que flatten; aun tiene un dia malo. |",
        "| Gate fijo de regimen | RESEARCH_PASS | Muy prometedor, pero no seleccionado de forma causal robusta. |",
        "| Book-only Conv1D | Pendiente de decision arriba | Test de si la imagen del libro aporta senal propia. |",
        "",
        "## Siguiente paso",
        "",
        "Si book-only no es desastroso, el siguiente paso natural es:",
        "",
        "```text",
        "fusion_v1 = encoder orderbook + GRU tabular + evaluacion de policy con costes y estabilidad diaria",
        "```",
        "",
        "No saltamos todavia a Transformer grande ni a bot. Primero queremos demostrar mejora incremental y estable.",
        "",
    ]

    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    book_npz = np.load(BOOK_ARRAYS_PATH)
    seq_npz = np.load(SEQ_ARRAYS_PATH)
    index_base = pd.read_csv(INDEX_PATH)
    book_manifest = read_json(BOOK_MANIFEST_PATH)

    x_book_raw = book_npz["tensor"].astype("float32")
    book_mask = book_npz["step_mask"].astype(bool)
    seq_mask = seq_npz["mask"].astype(bool)
    lengths = seq_npz["lengths"].astype("int64")
    y = seq_npz[TARGET].astype("float32")
    split = index_base["terminal_split"].astype(str)
    channels = [str(x) for x in book_npz["channel_names"].tolist()]

    if x_book_raw.shape[:2] != seq_mask.shape:
        raise ValueError(f"Book tensor shape {x_book_raw.shape[:2]} does not match sequence mask {seq_mask.shape}")
    if not np.array_equal(book_mask, seq_mask):
        raise ValueError("Book step mask does not match sequence mask.")

    x_book, scaler = fit_transform_book_tensor(x_book_raw, book_mask, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_rows = []
    policy_frames = []
    day_frames = []
    predictions = index_base[["sequence_id", "terminal_split", "session_day"]].copy()

    for model_name in ["book_last_snapshot_conv", "book_seq_conv_gru"]:
        pred, proba, info = train_one(model_name, x_book, book_mask, lengths, y, split, device)
        metric_rows.append(info)
        predictions[f"{model_name}_pred"] = pred
        predictions[f"{model_name}_proba"] = proba

        scored = index_base.copy()
        scored["pred"] = pred
        scored["proba"] = proba
        policies, days = evaluate_policies(scored, model_name)
        policy_frames.append(policies)
        if not days.empty:
            day_frames.append(days)

    metrics = pd.DataFrame(metric_rows)
    policies = pd.concat(policy_frames, ignore_index=True) if policy_frames else pd.DataFrame()
    days = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    selected = select_candidates(policies) if not policies.empty else pd.DataFrame()
    decision = decide(selected, policies) if not policies.empty else {"decision": "NO_GO_BOOK_ONLY_CONV1D"}

    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    policies.to_csv(OUT_DIR / "policy_results.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_candidates.csv", index=False)
    days.to_csv(OUT_DIR / "day_breakdown.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Book-only Conv1D baseline over Polymarket visible top-10 orderbook tensor.",
        "book_manifest": str(BOOK_MANIFEST_PATH),
        "sequence_arrays": str(SEQ_ARRAYS_PATH),
        "sequence_index": str(INDEX_PATH),
        "target": TARGET,
        "models": ["book_last_snapshot_conv", "book_seq_conv_gru"],
        "device": str(device),
        "book_tensor_shape": list(x_book_raw.shape),
        "channels": channels,
        "scaler": scaler,
        "train_config": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "policy_rows": int(len(policies)),
        "selected_candidates": int(len(selected)),
        "decision": decision,
        "reference_results": {
            "flatten_full_sklearn": {
                "test_auc": 0.5754,
                "selected_test_cost0p5": 0.3349,
                "selected_test_cost1p0": -0.0425,
                "selected_test_negative_days": 2,
            },
            "tabular_gru_torch": {
                "test_auc": 0.5542,
                "selected_test_cost0p5": 0.4550,
                "selected_test_cost1p0": 0.0828,
                "selected_test_negative_days": 1,
            },
            "fixed_regime_gate_research": {
                "test_cost0p5": 1.4021,
                "test_cost1p0": 1.0322,
                "test_negative_days": 0,
                "note": "Promising but not yet causally selected as final gate.",
            },
        },
        "outputs": {
            "model_metrics": str(OUT_DIR / "model_metrics.csv"),
            "policy_results": str(OUT_DIR / "policy_results.csv"),
            "selected_candidates": str(OUT_DIR / "selected_candidates.csv"),
            "day_breakdown": str(OUT_DIR / "day_breakdown.csv"),
            "predictions": str(OUT_DIR / "predictions.csv"),
            "decision": str(OUT_DIR / "decision.json"),
            "notebook": str(NOTEBOOK_PATH),
            "doc": str(DOC_PATH),
        },
        "runtime_seconds": round(time.time() - started, 3),
    }

    write_notebook()
    write_doc(decision=decision, metrics=metrics, selected=selected, days=days, manifest=run_manifest)
    (OUT_DIR / "manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run_manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
