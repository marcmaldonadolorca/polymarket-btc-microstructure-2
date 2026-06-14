from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_v1_book_tabular_sequence as fusion  # noqa: E402
import orderbook_regularized_sequence_v1 as obr  # noqa: E402


SEQ_DIR = ROOT / "data" / "experiments" / "complex_v1a_sequence_probe_v1"
BOOK_DIR = ROOT / "data" / "experiments" / "orderbook_tensor_audit_v1"
BOOK_FEATURES_PATH = ROOT / "data" / "experiments" / "orderbook_execution_feature_audit_v1" / "orderbook_features.csv"
SEQ_ARRAYS_PATH = SEQ_DIR / "sequence_arrays.npz"
BOOK_ARRAYS_PATH = BOOK_DIR / "orderbook_tensor_audit_arrays.npz"
INDEX_PATH = SEQ_DIR / "sequence_index.csv"
FEATURE_MANIFEST_PATH = SEQ_DIR / "sequence_feature_manifest.csv"

OUT_DIR = ROOT / "data" / "experiments" / "upstream_oof_fusion_stage2_v1"
DOC_PATH = ROOT / "docs" / "UPSTREAM_OOF_FUSION_STAGE2_V1.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "46_upstream_oof_fusion_stage2_v1.ipynb"

UPSTREAM_BLOCKS = [
    ("inner_1", ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14"], ["2026-05-15", "2026-05-16"]),
    (
        "inner_2",
        ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"],
        ["2026-05-17", "2026-05-18"],
    ),
    (
        "inner_3",
        [
            "2026-05-11",
            "2026-05-12",
            "2026-05-13",
            "2026-05-14",
            "2026-05-15",
            "2026-05-16",
            "2026-05-17",
            "2026-05-18",
        ],
        ["2026-05-19", "2026-05-20"],
    ),
    (
        "outer_validation",
        [
            "2026-05-11",
            "2026-05-12",
            "2026-05-13",
            "2026-05-14",
            "2026-05-15",
            "2026-05-16",
            "2026-05-17",
            "2026-05-18",
            "2026-05-19",
            "2026-05-20",
        ],
        ["2026-05-21", "2026-05-22"],
    ),
    (
        "test_terminal",
        [
            "2026-05-11",
            "2026-05-12",
            "2026-05-13",
            "2026-05-14",
            "2026-05-15",
            "2026-05-16",
            "2026-05-17",
            "2026-05-18",
            "2026-05-19",
            "2026-05-20",
            "2026-05-21",
            "2026-05-22",
        ],
        ["2026-05-23", "2026-05-24", "2026-05-25"],
    ),
]

STAGE2_TRAIN_BLOCKS = {
    "inner_2": ["inner_1"],
    "inner_3": ["inner_1", "inner_2"],
    "outer_validation": ["inner_1", "inner_2", "inner_3"],
    "test_terminal": ["inner_1", "inner_2", "inner_3", "outer_validation"],
}

UPSTREAM_TOP_FRACTION = 0.35
FIXED_UPSTREAM_EPOCHS = 1
RIDGE_ALPHAS = [10.0, 100.0]
LOGIT_CS = [0.10, 0.30]
DAILY_CAPS = [50, 75, 100, 150]
SEEDS = obr.SEEDS

TARGET_COST025 = "target_cost0p25"
TARGET_COST05 = "target_cost0p5"
TARGET_COST10 = "target_cost1p0"
TARGET_HEALTHY = "target_healthy"


def fmt(v: object, digits: int = 4) -> str:
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return "nan"
        return f"{float(v):.{digits}f}"
    return str(v)


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_Sin filas._"
    view = df[cols].head(limit) if limit else df[cols]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_tab_scaler(
    x_raw: np.ndarray, mask: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = x_raw[train_idx][mask[train_idx]]
    median = np.nanmedian(train_values, axis=0).astype("float32")
    median = np.nan_to_num(median, nan=0.0, posinf=0.0, neginf=0.0)
    train_values = np.where(np.isnan(train_values), median, train_values)
    mean = train_values.mean(axis=0).astype("float32")
    std = train_values.std(axis=0).astype("float32")
    std[std <= 1e-6] = 1.0
    return median, mean, std


def transform_tab(
    x_raw: np.ndarray,
    mask: np.ndarray,
    median: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    keep_idx: list[int],
) -> np.ndarray:
    imputed = np.where(np.isnan(x_raw), median, x_raw)
    x = ((imputed - mean) / std).astype("float32")
    x = x[..., keep_idx]
    x[~mask] = 0.0
    return x


def train_upstream_fixed(
    x_tab: np.ndarray,
    x_book: np.ndarray,
    mask: np.ndarray,
    lengths: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fusion.set_seed(seed)
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
    loader = DataLoader(train_ds, batch_size=fusion.BATCH_SIZE, shuffle=True)
    model = fusion.FusionConcatGRU(x_tab.shape[-1], x_book.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=fusion.LR, weight_decay=fusion.WEIGHT_DECAY)
    reg_loss = nn.SmoothL1Loss()
    clf_loss = nn.BCEWithLogitsLoss()

    for _ in range(FIXED_UPSTREAM_EPOCHS):
        model.train()
        for tb, bb, mb, lb, yb, pb in loader:
            tb, bb, mb, lb = tb.to(device), bb.to(device), mb.to(device), lb.to(device)
            yb, pb = yb.to(device), pb.to(device)
            opt.zero_grad(set_to_none=True)
            pred_scaled, logits = model(tb, bb, mb, lb)
            loss = reg_loss(pred_scaled, yb) + 0.35 * clf_loss(logits, pb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    model.eval()
    eval_ds = TensorDataset(
        torch.tensor(x_tab[eval_idx], dtype=torch.float32),
        torch.tensor(x_book[eval_idx], dtype=torch.float32),
        torch.tensor(mask[eval_idx], dtype=torch.bool),
        torch.tensor(lengths[eval_idx], dtype=torch.long),
    )
    preds, probas = [], []
    for tb, bb, mb, lb in DataLoader(eval_ds, batch_size=512, shuffle=False):
        with torch.no_grad():
            scaled, logits = model(tb.to(device), bb.to(device), mb.to(device), lb.to(device))
        preds.append((scaled.cpu().numpy() * y_std + y_mean).astype("float32"))
        probas.append(torch.sigmoid(logits).cpu().numpy().astype("float32"))
    return np.concatenate(preds), np.concatenate(probas), {
        "train_rows": int(train_idx.sum()),
        "eval_rows": int(eval_idx.sum()),
        "target_mean_train": y_mean,
        "target_std_train": y_std,
    }


def generate_upstream_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    seq = np.load(SEQ_ARRAYS_PATH)
    book = np.load(BOOK_ARRAYS_PATH)
    index = pd.read_csv(INDEX_PATH)
    feature_manifest = pd.read_csv(FEATURE_MANIFEST_PATH)

    x_raw = seq["x_raw"].astype("float32")
    mask = seq["mask"].astype(bool)
    lengths = seq["lengths"].astype("int64")
    y025 = seq["y_last_cost0p25"].astype("float32")
    y05 = seq["y_last_cost0p5"].astype("float32")
    y10 = seq["y_last_cost1p0"].astype("float32")
    x_book_raw = book["tensor"].astype("float32")
    book_mask = book["step_mask"].astype(bool)
    if not np.array_equal(mask, book_mask):
        raise ValueError("Book and tabular masks differ.")

    blocked_names = set(
        feature_manifest.loc[
            feature_manifest["feature_name"].str.contains(r"ev_pred|healthy_proba", regex=True),
            "feature_name",
        ].astype(str)
    )
    keep_idx = [
        int(row.feature_index)
        for row in feature_manifest.itertuples()
        if str(row.feature_name) not in blocked_names
    ]
    # This audit prioritizes exact reproducibility over speed. Small one-epoch
    # folds are cheap enough to run on CPU and avoid CUDA kernel variance.
    device = torch.device("cpu")
    rows, fold_rows = [], []

    for fold_no, (block_name, train_days, eval_days) in enumerate(UPSTREAM_BLOCKS, start=1):
        train_idx = index["session_day"].astype(str).isin(train_days).to_numpy()
        eval_idx = index["session_day"].astype(str).isin(eval_days).to_numpy()

        tab_median, tab_mean, tab_std = fit_tab_scaler(x_raw, mask, train_idx)
        x_tab = transform_tab(x_raw, mask, tab_median, tab_mean, tab_std, keep_idx)
        split_for_scaler = pd.Series(np.where(train_idx, "train_initial", "evaluation"))
        x_book, _ = fusion.fit_transform_book_tensor(x_book_raw, mask, split_for_scaler)
        pred, proba, info = train_upstream_fixed(
            x_tab,
            x_book,
            mask,
            lengths,
            y05,
            train_idx,
            eval_idx,
            device,
            seed=fusion.SEED + fold_no,
        )

        part = index.loc[eval_idx, ["sequence_id", "session_id", "market_id", "token_id", "session_day", "last_time_index_ns", "last_visible_entry_cost_ticks", "last_spread_ticks"]].copy()
        part["evaluation_block"] = block_name
        part["upstream_oof_pred"] = pred
        part["upstream_oof_proba"] = proba
        part[TARGET_COST025] = y025[eval_idx]
        part[TARGET_COST05] = y05[eval_idx]
        part[TARGET_COST10] = y10[eval_idx]
        part[TARGET_HEALTHY] = (y025[eval_idx] > 0).astype(int)
        proba_rank = part.groupby("session_day", observed=True)["upstream_oof_proba"].rank(
            pct=True, method="first"
        )
        part["upstream_selected"] = proba_rank.gt(1.0 - UPSTREAM_TOP_FRACTION)
        rows.append(part)
        fold_rows.append(
            {
                "evaluation_block": block_name,
                "train_start": train_days[0],
                "train_end": train_days[-1],
                "eval_start": eval_days[0],
                "eval_end": eval_days[-1],
                "tab_features_used": len(keep_idx),
                "tab_features_blocked": len(blocked_names),
                "device": str(device),
                "fixed_epochs": FIXED_UPSTREAM_EPOCHS,
                "selected_actions": int(part["upstream_selected"].sum()),
                "selected_rate": float(part["upstream_selected"].mean()),
                **info,
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame(fold_rows)


def stage2_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith("book_") and pd.api.types.is_numeric_dtype(df[c])
    ]
    cols += [
        "upstream_oof_pred",
        "upstream_oof_proba",
        "last_visible_entry_cost_ticks",
        "last_spread_ticks",
    ]
    return cols


def upstream_quality_summary(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, part in oof.groupby("evaluation_block", observed=True):
        selected = part[part["upstream_selected"]]
        y_pos = part[TARGET_COST05].gt(0).astype(int)
        auc = float(roc_auc_score(y_pos, part["upstream_oof_proba"])) if y_pos.nunique() > 1 else np.nan
        rows.append(
            {
                "evaluation_block": block,
                "rows": int(len(part)),
                "selected_actions": int(len(selected)),
                "auc_positive_cost0p5": auc,
                "spearman_pred_cost0p5": float(part["upstream_oof_pred"].corr(part[TARGET_COST05], method="spearman")),
                "all_mean_cost0p5": float(part[TARGET_COST05].mean()),
                "selected_mean_cost0p5": float(selected[TARGET_COST05].mean()),
                "selected_mean_cost1p0": float(selected[TARGET_COST10].mean()),
                "selected_healthy_rate": float(selected[TARGET_HEALTHY].mean()),
            }
        )
    return pd.DataFrame(rows)


def add_stage2_predictions(oof: pd.DataFrame) -> pd.DataFrame:
    out = oof.copy()
    features = stage2_feature_columns(out)
    score_cols = ["upstream_oof_pred", "upstream_oof_proba"]
    out["upstream_oof_combo"] = (
        0.50 * sigmoid_np(out["upstream_oof_pred"].to_numpy(dtype=float))
        + 0.50 * out["upstream_oof_proba"].to_numpy(dtype=float)
    )
    score_cols.append("upstream_oof_combo")

    for eval_block, train_blocks in STAGE2_TRAIN_BLOCKS.items():
        train_mask = out["evaluation_block"].isin(train_blocks) & out["upstream_selected"]
        eval_mask = out["evaluation_block"].eq(eval_block) & out["upstream_selected"]
        train = out[train_mask].copy()
        eval_part = out[eval_mask].copy()
        if len(train) < 40 or eval_part.empty:
            continue
        X_train, med = obr.prepare_X(train, features)
        X_eval, _ = obr.prepare_X(eval_part, features, med)
        y_mean = float(train[TARGET_COST05].mean())
        y_std = float(train[TARGET_COST05].std() if train[TARGET_COST05].std() > 1e-6 else 1.0)

        for alpha in RIDGE_ALPHAS:
            name = f"clean_stage2_ridge_a{str(alpha).replace('.', 'p')}"
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(X_train, train[TARGET_COST05].astype(float))
            pred = model.predict(X_eval)
            out.loc[eval_part.index, name] = pred
            ev_safe = sigmoid_np((pred - y_mean) / y_std)
            for C in LOGIT_CS:
                healthy_name = f"clean_stage2_healthy_c{str(C).replace('.', 'p')}"
                combo_name = f"clean_stage2_combo_a{str(alpha).replace('.', 'p')}_c{str(C).replace('.', 'p')}"
                if healthy_name not in out.columns or out.loc[eval_part.index, healthy_name].isna().all():
                    logit = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=C, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=97),
                    )
                    logit.fit(X_train, train[TARGET_HEALTHY].astype(int))
                    out.loc[eval_part.index, healthy_name] = logit.predict_proba(X_eval)[:, 1]
                healthy = out.loc[eval_part.index, healthy_name].to_numpy(dtype=float)
                out.loc[eval_part.index, combo_name] = (
                    0.50 * ev_safe
                    + 0.30 * healthy
                    + 0.20 * eval_part["upstream_oof_proba"].to_numpy(dtype=float)
                )
                score_cols.extend([healthy_name, combo_name])
            score_cols.append(name)
    return out, sorted(set(score_cols))


def select_daily(df: pd.DataFrame, score_col: str, daily_cap: int) -> pd.DataFrame:
    return (
        df.sort_values(["session_day", score_col, "upstream_oof_proba"], ascending=[True, False, False])
        .groupby("session_day", observed=True)
        .head(daily_cap)
        .copy()
    )


def summarize(filled: pd.DataFrame) -> dict:
    if filled.empty:
        return {
            "actions": 0,
            "sum_cost0p5": 0.0,
            "sum_cost1p0": 0.0,
            "worst_day_cost0p5": np.nan,
            "worst_day_cost1p0": np.nan,
            "negative_days_cost0p5": 0,
            "negative_days_cost1p0": 0,
        }
    daily05 = filled.groupby("session_day", observed=True)[TARGET_COST05].sum()
    daily10 = filled.groupby("session_day", observed=True)[TARGET_COST10].sum()
    return {
        "actions": int(len(filled)),
        "sum_cost0p5": float(filled[TARGET_COST05].sum()),
        "sum_cost1p0": float(filled[TARGET_COST10].sum()),
        "worst_day_cost0p5": float(daily05.min()),
        "worst_day_cost1p0": float(daily10.min()),
        "negative_days_cost0p5": int((daily05 <= 0).sum()),
        "negative_days_cost1p0": int((daily10 <= 0).sum()),
    }


def evaluate_policies(scored: pd.DataFrame, score_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, day_rows = [], []
    for block in STAGE2_TRAIN_BLOCKS:
        block_df = scored[scored["evaluation_block"].eq(block) & scored["upstream_selected"]].copy()
        for score_col in score_cols:
            available = block_df[block_df[score_col].notna()].copy()
            if available.empty:
                continue
            for cap in DAILY_CAPS:
                selected = select_daily(available, score_col, cap)
                for seed in SEEDS:
                    filled = obr.fill_random_by_day(selected, 0.5, seed)
                    summary = summarize(filled)
                    rows.append(
                        {
                            "evaluation_block": block,
                            "score_col": score_col,
                            "daily_cap": cap,
                            "fill_rate": 0.5,
                            "seed": seed,
                            **summary,
                            "pass_cost0p5": (
                                summary["actions"] >= 30
                                and summary["sum_cost0p5"] > 0
                                and summary["negative_days_cost0p5"] == 0
                            ),
                            "pass_cost1p0": (
                                summary["actions"] >= 30
                                and summary["sum_cost0p5"] > 0
                                and summary["negative_days_cost0p5"] == 0
                                and summary["sum_cost1p0"] > 0
                                and summary["negative_days_cost1p0"] == 0
                            ),
                        }
                    )
                    for day, d in filled.groupby("session_day", observed=True):
                        day_rows.append(
                            {
                                "evaluation_block": block,
                                "score_col": score_col,
                                "daily_cap": cap,
                                "seed": seed,
                                "session_day": day,
                                "actions": int(len(d)),
                                "sum_cost0p5": float(d[TARGET_COST05].sum()),
                                "sum_cost1p0": float(d[TARGET_COST10].sum()),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def aggregate_policies(raw: pd.DataFrame) -> pd.DataFrame:
    block_rows = []
    for keys, part in raw.groupby(["evaluation_block", "score_col", "daily_cap"], observed=True):
        block, score_col, cap = keys
        block_rows.append(
            {
                "evaluation_block": block,
                "score_col": score_col,
                "daily_cap": cap,
                "runs": int(len(part)),
                "actions_mean": float(part["actions"].mean()),
                "pass_cost0p5_rate": float(part["pass_cost0p5"].mean()),
                "pass_cost1p0_rate": float(part["pass_cost1p0"].mean()),
                "sum_cost0p5_mean": float(part["sum_cost0p5"].mean()),
                "sum_cost1p0_mean": float(part["sum_cost1p0"].mean()),
                "worst_day_cost0p5_mean": float(part["worst_day_cost0p5"].mean()),
                "worst_day_cost1p0_mean": float(part["worst_day_cost1p0"].mean()),
            }
        )
    block_agg = pd.DataFrame(block_rows)

    selection_blocks = ["inner_2", "inner_3", "outer_validation"]
    rows = []
    for keys, part in block_agg.groupby(["score_col", "daily_cap"], observed=True):
        score_col, cap = keys
        sel = part[part["evaluation_block"].isin(selection_blocks)].copy()
        test = part[part["evaluation_block"].eq("test_terminal")].iloc[0]
        row = {
            "score_col": score_col,
            "daily_cap": cap,
            "selection_blocks": int(len(sel)),
            "selection_pass_cost0p5_min": float(sel["pass_cost0p5_rate"].min()),
            "selection_pass_cost0p5_mean": float(sel["pass_cost0p5_rate"].mean()),
            "selection_worst_day_cost0p5_min": float(sel["worst_day_cost0p5_mean"].min()),
            "selection_sum_cost0p5_min": float(sel["sum_cost0p5_mean"].min()),
            "test_pass_cost0p5_rate": float(test["pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(test["pass_cost1p0_rate"]),
            "test_sum_cost0p5_mean": float(test["sum_cost0p5_mean"]),
            "test_sum_cost1p0_mean": float(test["sum_cost1p0_mean"]),
            "test_worst_day_cost0p5_mean": float(test["worst_day_cost0p5_mean"]),
            "test_worst_day_cost1p0_mean": float(test["worst_day_cost1p0_mean"]),
        }
        row["selection_eligible"] = (
            row["selection_blocks"] == len(selection_blocks)
            and row["selection_pass_cost0p5_min"] >= 0.50
            and row["selection_worst_day_cost0p5_min"] >= 0.0
        )
        row["selection_score"] = (
            1000.0 * row["selection_pass_cost0p5_min"]
            + 200.0 * row["selection_pass_cost0p5_mean"]
            + 0.50 * row["selection_worst_day_cost0p5_min"]
            + 0.05 * row["selection_sum_cost0p5_min"]
            - 0.01 * cap
        )
        row["test_robust_cost0p5"] = (
            row["test_pass_cost0p5_rate"] >= 0.60 and row["test_worst_day_cost0p5_mean"] >= 0.0
        )
        rows.append(row)
    return block_agg, pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Upstream OOF fusion + stage2 v1\n",
                    "\n",
                    "Primera y segunda etapa causales mediante walk-forward.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "base = '../data/experiments/upstream_oof_fusion_stage2_v1'\n",
                    "folds = pd.read_csv(f'{base}/upstream_fold_summary.csv')\n",
                    "folds\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "policies = pd.read_csv(f'{base}/policy_summary.csv')\n",
                    "policies.head(25)\n",
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
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_doc(
    decision: dict, folds: pd.DataFrame, quality: pd.DataFrame, policies: pd.DataFrame
) -> None:
    fold_cols = [
        "evaluation_block",
        "train_start",
        "train_end",
        "eval_start",
        "eval_end",
        "train_rows",
        "eval_rows",
        "selected_actions",
        "selected_rate",
        "tab_features_used",
        "tab_features_blocked",
    ]
    policy_cols = [
        "score_col",
        "daily_cap",
        "selection_eligible",
        "selection_pass_cost0p5_min",
        "selection_pass_cost0p5_mean",
        "selection_worst_day_cost0p5_min",
        "test_pass_cost0p5_rate",
        "test_pass_cost1p0_rate",
        "test_sum_cost0p5_mean",
        "test_worst_day_cost0p5_mean",
        "test_robust_cost0p5",
    ]
    quality_cols = [
        "evaluation_block",
        "rows",
        "selected_actions",
        "auc_positive_cost0p5",
        "spearman_pred_cost0p5",
        "all_mean_cost0p5",
        "selected_mean_cost0p5",
        "selected_mean_cost1p0",
        "selected_healthy_rate",
    ]
    text = f"""# Upstream OOF fusion + stage2 v1

Fecha: 2026-06-04

## Objetivo

Comprobar limpiamente la pista `book_plus_scores` sin reutilizar scores
in-sample.

## Protocolo causal

- upstream `fusion_concat_gru` entrenado solo con dias pasados;
- una epoca fija, elegida antes de esta prueba;
- se eliminan ocho features auxiliares `ev_pred/healthy_proba` de la entrada;
- cada bloque recibe predicciones upstream out-of-fold;
- como la calibracion cambia entre folds, el universo se selecciona por ranking
  con cobertura fija predeclarada, aproximadamente igual a la cobertura
  historica:

```text
top 35% diario por upstream_oof_proba
```

- stage2 se entrena solo con acciones OOF de bloques anteriores;
- seleccion con `inner_2`, `inner_3` y `outer_validation`;
- test terminal se abre al final.

## Decision

```text
{decision["decision"]}
```

{decision["reason"]}

## Bloques upstream OOF

{markdown_table(folds, fold_cols)}

## Calidad del upstream OOF

{markdown_table(quality, quality_cols)}

## Config seleccionada

```text
score={decision["selected_config"]["score_col"]}
cap={decision["selected_config"]["daily_cap"]}
```

## Politicas

{markdown_table(policies, policy_cols, 30)}

## Lectura sencilla

- Este experimento elimina el principal caveat de `book_plus_scores`.
- Si stage2 pasa, existe una pista causal para fusion/risk.
- Si no pasa, no debemos promover la fusion regularizada actual a modelo
  complejo solo porque alguna configuracion retrospectiva pareciera buena.
- El upstream OOF empieza debil en `inner_1/inner_2`, pero pasa a media
  seleccionada cost0.50 positiva desde `inner_3` y se mantiene positiva en
  validation/test.
- Esa mejora tardia sugiere estudiar cantidad minima de historial y regimen
  temporal antes de aumentar arquitectura.

## Siguiente paso recomendado

```text
upstream_oof_maturity_regime_audit_v1
```

Objetivo: comprobar si el ranking solo se vuelve util tras acumular suficiente
historial, y si una ventana expanding/rolling puede seleccionarse sin mirar
test. No entrenar aun CNN/Transformer mayor.
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    upstream_oof, fold_summary = generate_upstream_oof()
    upstream_quality = upstream_quality_summary(upstream_oof)
    book_features = pd.read_csv(BOOK_FEATURES_PATH)
    oof = upstream_oof.merge(book_features, on="sequence_id", how="left")
    scored, score_cols = add_stage2_predictions(oof)
    raw_policies, day_results = evaluate_policies(scored, score_cols)
    block_agg, policies = aggregate_policies(raw_policies)

    eligible = policies[policies["selection_eligible"]].copy()
    selection_pool = eligible if not eligible.empty else policies
    selected = selection_pool.sort_values("selection_score", ascending=False).iloc[0]
    selected_pass = bool(selected["test_robust_cost0p5"])
    robust_count = int(policies["test_robust_cost0p5"].sum())
    robust_stage2_count = int(
        policies[
            policies["test_robust_cost0p5"]
            & policies["score_col"].str.startswith("clean_stage2")
        ].shape[0]
    )

    if selected_pass and str(selected["score_col"]).startswith("clean_stage2"):
        decision_name = "RESEARCH_PASS_CLEAN_OOF_FUSION_STAGE2_COST05"
        reason = "The validation-selected clean OOF stage2 score passes the test cost0.50 robustness proxy."
    elif selected_pass:
        decision_name = "RESEARCH_PASS_CLEAN_OOF_UPSTREAM_BASELINE_COST05"
        reason = "The clean OOF selector passes test cost0.50, but selection falls back to the upstream baseline."
    else:
        decision_name = "NO_GO_CLEAN_OOF_FUSION_STAGE2_COST05"
        reason = "The validation-selected clean OOF policy does not close the test cost0.50 robustness proxy."

    decision = {
        "decision": decision_name,
        "reason": reason,
        "selected_config": {
            "score_col": str(selected["score_col"]),
            "daily_cap": int(selected["daily_cap"]),
            "selection_eligible": bool(selected["selection_eligible"]),
            "selection_pass_cost0p5_min": float(selected["selection_pass_cost0p5_min"]),
            "selection_pass_cost0p5_mean": float(selected["selection_pass_cost0p5_mean"]),
            "test_pass_cost0p5_rate": float(selected["test_pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(selected["test_pass_cost1p0_rate"]),
            "test_sum_cost0p5_mean": float(selected["test_sum_cost0p5_mean"]),
            "test_worst_day_cost0p5_mean": float(selected["test_worst_day_cost0p5_mean"]),
            "test_robust_cost0p5": bool(selected["test_robust_cost0p5"]),
        },
        "robust_config_count": robust_count,
        "robust_stage2_config_count": robust_stage2_count,
        "runtime_seconds": round(time.time() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "upstream_oof_predictions": str(OUT_DIR / "upstream_oof_predictions.csv"),
            "upstream_fold_summary": str(OUT_DIR / "upstream_fold_summary.csv"),
            "upstream_quality_summary": str(OUT_DIR / "upstream_quality_summary.csv"),
            "stage2_predictions": str(OUT_DIR / "stage2_predictions.csv"),
            "raw_policy_results": str(OUT_DIR / "raw_policy_results.csv"),
            "day_results": str(OUT_DIR / "day_results.csv"),
            "block_aggregate": str(OUT_DIR / "block_aggregate.csv"),
            "policy_summary": str(OUT_DIR / "policy_summary.csv"),
            "doc": str(DOC_PATH),
            "notebook": str(NOTEBOOK_PATH),
        },
    }

    upstream_oof.to_csv(OUT_DIR / "upstream_oof_predictions.csv", index=False)
    fold_summary.to_csv(OUT_DIR / "upstream_fold_summary.csv", index=False)
    upstream_quality.to_csv(OUT_DIR / "upstream_quality_summary.csv", index=False)
    stage2_keep = [
        "sequence_id",
        "session_day",
        "evaluation_block",
        "upstream_selected",
        TARGET_COST025,
        TARGET_COST05,
        TARGET_COST10,
        TARGET_HEALTHY,
    ] + score_cols
    scored[stage2_keep].to_csv(OUT_DIR / "stage2_predictions.csv", index=False)
    raw_policies.to_csv(OUT_DIR / "raw_policy_results.csv", index=False)
    day_results.to_csv(OUT_DIR / "day_results.csv", index=False)
    block_agg.to_csv(OUT_DIR / "block_aggregate.csv", index=False)
    policies.to_csv(OUT_DIR / "policy_summary.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_notebook()
    write_doc(decision, fold_summary, upstream_quality, policies)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
