from __future__ import annotations

import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.book_only_conv1d_baseline_v1 import (  # noqa: E402
    BookEncoder,
    evaluate_policies,
    fit_transform_book_tensor,
    format_float,
    safe_auc,
    select_candidates,
    split_metrics,
)


SEQ_DIR = ROOT / "data" / "experiments" / "complex_v1a_sequence_probe_v1"
BOOK_DIR = ROOT / "data" / "experiments" / "orderbook_tensor_audit_v1"
SEQ_ARRAYS_PATH = SEQ_DIR / "sequence_arrays.npz"
BOOK_ARRAYS_PATH = BOOK_DIR / "orderbook_tensor_audit_arrays.npz"
INDEX_PATH = SEQ_DIR / "sequence_index.csv"

OUT_DIR = ROOT / "data" / "experiments" / "fusion_v1_book_tabular_sequence"
MODEL_DIR = OUT_DIR / "models"
NOTEBOOK_PATH = ROOT / "notebooks" / "28_fusion_v1_book_tabular_sequence.ipynb"
DOC_PATH = ROOT / "docs" / "FUSION_V1_BOOK_TABULAR_SEQUENCE.md"

SEED = 59
TARGET = "y_last_cost0p5"
MAX_EPOCHS = 100
BATCH_SIZE = 192
PATIENCE = 18
LR = 0.0015
WEIGHT_DECAY = 1e-4
MIN_TEST_ACTIONS = 80


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TabularGRUControl(nn.Module):
    def __init__(self, n_tab_features: int, n_book_channels: int) -> None:
        super().__init__()
        del n_book_channels
        self.gru = nn.GRU(input_size=n_tab_features, hidden_size=56, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(56, 32), nn.ReLU(), nn.Dropout(0.12))
        self.reg_head = nn.Linear(32, 1)
        self.clf_head = nn.Linear(32, 1)

    def forward(
        self, x_tab: torch.Tensor, x_book: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del x_book
        out, _ = self.gru(x_tab)
        idx = (lengths.long() - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        h = out.gather(1, idx).squeeze(1)
        h = self.head(h)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


class FusionConcatGRU(nn.Module):
    def __init__(self, n_tab_features: int, n_book_channels: int) -> None:
        super().__init__()
        self.book_encoder = BookEncoder(n_channels=n_book_channels, out_channels=32)
        self.tab_proj = nn.Sequential(nn.Linear(n_tab_features, 48), nn.ReLU(), nn.Dropout(0.08))
        self.gru = nn.GRU(
            input_size=48 + self.book_encoder.out_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Linear(64, 40), nn.ReLU(), nn.Dropout(0.12))
        self.reg_head = nn.Linear(40, 1)
        self.clf_head = nn.Linear(40, 1)

    def forward(
        self, x_tab: torch.Tensor, x_book: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, levels, channels = x_book.shape
        book_emb = self.book_encoder(x_book.reshape(batch * steps, levels, channels)).reshape(batch, steps, -1)
        tab_emb = self.tab_proj(x_tab)
        emb = torch.cat([tab_emb, book_emb], dim=-1) * mask.unsqueeze(-1).float()
        out, _ = self.gru(emb)
        idx = (lengths.long() - 1).clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        h = out.gather(1, idx).squeeze(1)
        h = self.head(h)
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


class FusionDualGRU(nn.Module):
    def __init__(self, n_tab_features: int, n_book_channels: int) -> None:
        super().__init__()
        self.book_encoder = BookEncoder(n_channels=n_book_channels, out_channels=28)
        self.tab_gru = nn.GRU(input_size=n_tab_features, hidden_size=48, num_layers=1, batch_first=True)
        self.book_gru = nn.GRU(input_size=self.book_encoder.out_features, hidden_size=40, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(88, 48), nn.ReLU(), nn.Dropout(0.12), nn.Linear(48, 32), nn.ReLU())
        self.reg_head = nn.Linear(32, 1)
        self.clf_head = nn.Linear(32, 1)

    def forward(
        self, x_tab: torch.Tensor, x_book: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, levels, channels = x_book.shape
        book_emb = self.book_encoder(x_book.reshape(batch * steps, levels, channels)).reshape(batch, steps, -1)
        book_emb = book_emb * mask.unsqueeze(-1).float()
        tab_out, _ = self.tab_gru(x_tab)
        book_out, _ = self.book_gru(book_emb)
        idx_t = (lengths.long() - 1).clamp_min(0).view(-1, 1, 1)
        h_tab = tab_out.gather(1, idx_t.expand(-1, 1, tab_out.shape[-1])).squeeze(1)
        h_book = book_out.gather(1, idx_t.expand(-1, 1, book_out.shape[-1])).squeeze(1)
        h = self.head(torch.cat([h_tab, h_book], dim=-1))
        return self.reg_head(h).squeeze(-1), self.clf_head(h).squeeze(-1)


def make_model(name: str, n_tab_features: int, n_book_channels: int) -> nn.Module:
    if name == "tabular_gru_control":
        return TabularGRUControl(n_tab_features, n_book_channels)
    if name == "fusion_concat_gru":
        return FusionConcatGRU(n_tab_features, n_book_channels)
    if name == "fusion_dual_gru":
        return FusionDualGRU(n_tab_features, n_book_channels)
    raise ValueError(name)


def train_one(
    model_name: str,
    x_tab: np.ndarray,
    x_book: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    split: pd.Series,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    train_idx = split.eq("train_initial").to_numpy()
    val_idx = split.eq("validation_initial").to_numpy()
    y_mean = float(y[train_idx].mean())
    y_std = float(y[train_idx].std() if y[train_idx].std() > 1e-6 else 1.0)
    y_scaled = ((y - y_mean) / y_std).astype("float32")
    y_pos = (y > 0).astype("float32")

    train_ds = TensorDataset(
        torch.tensor(x_tab[train_idx], dtype=torch.float32),
        torch.tensor(x_book[train_idx], dtype=torch.float32),
        torch.tensor(mask[train_idx], dtype=torch.bool),
        torch.tensor(lengths[train_idx], dtype=torch.long),
        torch.tensor(y_scaled[train_idx], dtype=torch.float32),
        torch.tensor(y_pos[train_idx], dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = make_model(model_name, x_tab.shape[-1], x_book.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    reg_loss = nn.SmoothL1Loss()
    clf_loss = nn.BCEWithLogitsLoss()
    best_score = -math.inf
    best_state = None
    best_epoch = 0
    no_improve = 0
    val_tensors = (
        torch.tensor(x_tab[val_idx], dtype=torch.float32, device=device),
        torch.tensor(x_book[val_idx], dtype=torch.float32, device=device),
        torch.tensor(mask[val_idx], dtype=torch.bool, device=device),
        torch.tensor(lengths[val_idx], dtype=torch.long, device=device),
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for tb, bb, mb, lb, yb, pb in loader:
            tb, bb, mb, lb, yb, pb = tb.to(device), bb.to(device), mb.to(device), lb.to(device), yb.to(device), pb.to(device)
            opt.zero_grad(set_to_none=True)
            pred_scaled, logits = model(tb, bb, mb, lb)
            loss = reg_loss(pred_scaled, yb) + 0.35 * clf_loss(logits, pb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_scaled, val_logits = model(*val_tensors)
            val_pred = (val_scaled.cpu().numpy() * y_std + y_mean).astype("float32")
            val_proba = torch.sigmoid(val_logits).cpu().numpy().astype("float32")
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
    preds, probas = [], []
    all_ds = TensorDataset(
        torch.tensor(x_tab, dtype=torch.float32),
        torch.tensor(x_book, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.bool),
        torch.tensor(lengths, dtype=torch.long),
    )
    for tb, bb, mb, lb in DataLoader(all_ds, batch_size=512, shuffle=False):
        with torch.no_grad():
            scaled, logits = model(tb.to(device), bb.to(device), mb.to(device), lb.to(device))
        preds.append((scaled.cpu().numpy() * y_std + y_mean).astype("float32"))
        probas.append(torch.sigmoid(logits).cpu().numpy().astype("float32"))
    pred = np.concatenate(preds)
    proba = np.concatenate(probas)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_name": model_name, "state_dict": model.state_dict(), "target_mean": y_mean, "target_std": y_std},
        MODEL_DIR / f"{model_name}.pt",
    )
    info = {
        "model_name": model_name,
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    for split_name in ["train_initial", "validation_initial", "test_terminal"]:
        sm = split.eq(split_name).to_numpy()
        info.update({f"{split_name}_{k}": v for k, v in split_metrics(y[sm], pred[sm], proba[sm]).items()})
    return pred, proba, info


def decide(selected: pd.DataFrame, policy_results: pd.DataFrame) -> dict:
    if selected.empty:
        return {"decision": "NO_GO_FUSION_V1", "reason": "No validation-stable fusion/control policy found."}
    best = selected.iloc[0].to_dict()
    primary_pass = (
        best["test_actions"] >= MIN_TEST_ACTIONS
        and best["test_mean_net_cost_0p25"] > 0
        and best["test_mean_net_cost_0p5"] > 0
        and best["test_mean_net_cost_1p0"] > 0
        and best["test_negative_days_cost0p5"] == 0
    )
    best_is_fusion = str(best["model_name"]).startswith("fusion_")
    test_pass_mask = (
        policy_results["test_actions"].ge(MIN_TEST_ACTIONS)
        & policy_results["test_mean_net_cost_0p25"].gt(0)
        & policy_results["test_mean_net_cost_0p5"].gt(0)
        & policy_results["test_mean_net_cost_1p0"].gt(0)
        & policy_results["test_negative_days_cost0p5"].eq(0)
    )
    any_test_pass = policy_results[test_pass_mask]
    selected_fusion = selected[selected["model_name"].astype(str).str.startswith("fusion_")].copy()
    selected_fusion_test_pass = selected_fusion[
        selected_fusion["test_actions"].ge(MIN_TEST_ACTIONS)
        & selected_fusion["test_mean_net_cost_0p25"].gt(0)
        & selected_fusion["test_mean_net_cost_0p5"].gt(0)
        & selected_fusion["test_mean_net_cost_1p0"].gt(0)
        & selected_fusion["test_negative_days_cost0p5"].eq(0)
    ].copy()
    keys = [
        "model_name",
        "policy_name",
        "validation_actions",
        "validation_mean_net_cost_0p5",
        "validation_mean_net_cost_1p0",
        "validation_negative_days_cost0p5",
        "test_actions",
        "test_mean_net_cost_0p5",
        "test_mean_net_cost_1p0",
        "test_negative_days_cost0p5",
        "test_min_day_cost0p5",
    ]
    diagnostic_fusion = {}
    if not selected_fusion_test_pass.empty:
        row = selected_fusion_test_pass.sort_values(["selection_score", "validation_actions"], ascending=[False, False]).iloc[0].to_dict()
        diagnostic_fusion = {k: row.get(k) for k in keys}
    return {
        "decision": "RESEARCH_PASS_FUSION_V1_DIAGNOSTIC" if diagnostic_fusion else "NO_GO_FUSION_V1",
        "reason": (
            "A validation-stable fusion candidate passes terminal test, but the primary validation selector still chooses the tabular control."
            if diagnostic_fusion
            else "Fusion/control does not deliver a causal, stable improvement over the current tabular path."
        ),
        "primary_best_policy": {k: best.get(k) for k in keys},
        "primary_best_is_fusion": bool(best_is_fusion),
        "primary_best_passes_test": bool(primary_pass),
        "diagnostic_fusion_policy": diagnostic_fusion,
        "diagnostic_fusion_note": "Chosen only as retrospective diagnostic among validation-stable fusion policies; not a deployment selector.",
        "any_test_pass_policy_count": int(len(any_test_pass)),
    }


def write_notebook() -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Fusion v1 - book + tabular sequence\n"]},
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "Probamos si el tensor de orderbook mejora al GRU tabular. La comparacion usa el mismo split temporal, target y evaluacion de costes.\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\nimport json, pandas as pd\n",
                    "OUT = Path('../data/experiments/fusion_v1_book_tabular_sequence')\n",
                    "decision = json.loads((OUT/'decision.json').read_text(encoding='utf-8'))\n",
                    "metrics = pd.read_csv(OUT/'model_metrics.csv')\n",
                    "selected = pd.read_csv(OUT/'selected_candidates.csv')\n",
                    "days = pd.read_csv(OUT/'day_breakdown.csv')\n",
                    "decision\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## Metricas\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["metrics.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Candidatos por validacion\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["selected.head(20).round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## Desglose del mejor candidato\n"]},
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
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_doc(decision: dict, metrics: pd.DataFrame, days: pd.DataFrame) -> None:
    best = decision.get("primary_best_policy", {})
    diagnostic = decision.get("diagnostic_fusion_policy", {})
    lines = [
        "# Fusion v1 - book + tabular sequence",
        "",
        "Fecha: 2026-06-02",
        "",
        "## Objetivo",
        "",
        "Probar si el orderbook como tensor ayuda al GRU tabular.",
        "",
        "Comparacion justa:",
        "",
        "- mismo split temporal;",
        "- mismo target `y_last_cost0p5`;",
        "- mismas policies por validacion;",
        "- mismo criterio de coste 0.25/0.50/1.00 y estabilidad diaria.",
        "",
        "## Modelos",
        "",
        "| Modelo | Parametros | Idea |",
        "|---|---:|---|",
    ]
    ideas = {
        "tabular_gru_control": "Control: solo features tabulares secuenciales.",
        "fusion_concat_gru": "Concatena embedding book + tabular en cada paso y usa GRU.",
        "fusion_dual_gru": "GRU tabular y GRU book separados, fusionados al final.",
    }
    for _, row in metrics.iterrows():
        lines.append(f"| `{row['model_name']}` | {int(row['n_parameters'])} | {ideas[row['model_name']]} |")

    lines += [
        "",
        "## Ranking",
        "",
        "| Modelo | AUC val | AUC test | Spearman val | Spearman test |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in metrics.iterrows():
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                row["model_name"],
                format_float(row["validation_initial_auc_classifier_positive_cost0p5"]),
                format_float(row["test_terminal_auc_classifier_positive_cost0p5"]),
                format_float(row["validation_initial_spearman_corr"]),
                format_float(row["test_terminal_spearman_corr"]),
            )
        )

    lines += [
        "",
        "## Decision",
        "",
        "```text",
        str(decision.get("decision")),
        "```",
        "",
        str(decision.get("reason")),
        "",
    ]
    if best:
        lines += [
        "Policy primaria elegida por validacion:",
            "",
            "```text",
            f"{best.get('model_name')} + {best.get('policy_name')}",
            "```",
            "",
            "| Split | Acciones | Net 0.50 | Net 1.00 | Dias negativos |",
            "|---|---:|---:|---:|---:|",
            f"| validation | {int(best.get('validation_actions', 0))} | {format_float(best.get('validation_mean_net_cost_0p5'))} | {format_float(best.get('validation_mean_net_cost_1p0'))} | {int(best.get('validation_negative_days_cost0p5', 0))} |",
            f"| test | {int(best.get('test_actions', 0))} | {format_float(best.get('test_mean_net_cost_0p5'))} | {format_float(best.get('test_mean_net_cost_1p0'))} | {int(best.get('test_negative_days_cost0p5', 0))} |",
            "",
        ]
        if diagnostic:
            lines += [
                "Candidato diagnostico de fusion:",
                "",
                "```text",
                f"{diagnostic.get('model_name')} + {diagnostic.get('policy_name')}",
                "```",
                "",
                "Importante: este candidato se reporta para investigacion. No sustituye al selector primario porque seria facil caer en seleccion retrospectiva.",
                "",
                "| Split | Acciones | Net 0.50 | Net 1.00 | Dias negativos |",
                "|---|---:|---:|---:|---:|",
                f"| validation | {int(diagnostic.get('validation_actions', 0))} | {format_float(diagnostic.get('validation_mean_net_cost_0p5'))} | {format_float(diagnostic.get('validation_mean_net_cost_1p0'))} | {int(diagnostic.get('validation_negative_days_cost0p5', 0))} |",
                f"| test | {int(diagnostic.get('test_actions', 0))} | {format_float(diagnostic.get('test_mean_net_cost_0p5'))} | {format_float(diagnostic.get('test_mean_net_cost_1p0'))} | {int(diagnostic.get('test_negative_days_cost0p5', 0))} |",
                "",
            ]
        bd = days[days["model_name"].eq(best.get("model_name")) & days["policy_name"].eq(best.get("policy_name"))]
        if not bd.empty:
            bd = bd[bd["terminal_split"].isin(["validation_initial", "test_terminal"])]
            lines += ["Desglose diario de la policy primaria:", "", "| Dia | Split | Acciones | Net 0.50 | Net 1.00 |", "|---|---|---:|---:|---:|"]
            for _, row in bd.sort_values(["terminal_split", "session_day"]).iterrows():
                lines.append(
                    f"| {row['session_day']} | {row['terminal_split']} | {int(row['actions'])} | {format_float(row['mean_net_cost_0p5'])} | {format_float(row['mean_net_cost_1p0'])} |"
                )
            lines.append("")

    lines += [
        "## Lectura",
        "",
        "Esta iteracion no busca un modelo grande, sino responder una pregunta:",
        "",
        "```text",
        "el libro visible mejora al modelo secuencial tabular?",
        "```",
        "",
        "Si la fusion no mejora de forma estable, el siguiente paso no debe ser agrandar la red.",
        "Debe ser mejorar el target/regimen y la seleccion causal.",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    set_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    seq = np.load(SEQ_ARRAYS_PATH)
    book = np.load(BOOK_ARRAYS_PATH)
    index = pd.read_csv(INDEX_PATH)
    split = index["terminal_split"].astype(str)
    x_tab = seq["x"].astype("float32")
    x_book_raw = book["tensor"].astype("float32")
    mask = seq["mask"].astype(bool)
    book_mask = book["step_mask"].astype(bool)
    lengths = seq["lengths"].astype("int64")
    y = seq[TARGET].astype("float32")
    if not np.array_equal(mask, book_mask):
        raise ValueError("Book mask and tabular mask differ.")
    x_book, scaler = fit_transform_book_tensor(x_book_raw, mask, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_rows, policy_frames, day_frames = [], [], []
    predictions = index[["sequence_id", "terminal_split", "session_day"]].copy()
    models = ["tabular_gru_control", "fusion_concat_gru", "fusion_dual_gru"]
    for name in models:
        pred, proba, info = train_one(name, x_tab, x_book, mask, lengths, y, split, device)
        metric_rows.append(info)
        predictions[f"{name}_pred"] = pred
        predictions[f"{name}_proba"] = proba
        scored = index.copy()
        scored["pred"] = pred
        scored["proba"] = proba
        policies, days = evaluate_policies(scored, name)
        policy_frames.append(policies)
        if not days.empty:
            day_frames.append(days)

    metrics = pd.DataFrame(metric_rows)
    policies = pd.concat(policy_frames, ignore_index=True)
    days = pd.concat(day_frames, ignore_index=True) if day_frames else pd.DataFrame()
    selected = select_candidates(policies)
    decision = decide(selected, policies)

    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    policies.to_csv(OUT_DIR / "policy_results.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_candidates.csv", index=False)
    days.to_csv(OUT_DIR / "day_breakdown.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
    write_notebook()
    write_doc(decision, metrics, days)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compact fusion v1: tabular sequence GRU vs book+tabular fusion models.",
        "target": TARGET,
        "models": models,
        "device": str(device),
        "x_tab_shape": list(x_tab.shape),
        "x_book_shape": list(x_book.shape),
        "book_scaler": scaler,
        "train_config": {
            "seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "decision": decision,
        "runtime_seconds": round(time.time() - started, 3),
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
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
