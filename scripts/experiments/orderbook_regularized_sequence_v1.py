from __future__ import annotations

# Historical v1: raw logit_adverse probabilities were allowed as descending
# rankers. Use selection_stability_nested_v3_corrected.py for valid conclusions.

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = ROOT / "data" / "experiments" / "fill_model_v2_tabular_probe" / "predictions.csv"
BOOK_FEATURES_PATH = ROOT / "data" / "experiments" / "orderbook_execution_feature_audit_v1" / "orderbook_features.csv"
OUT_DIR = ROOT / "data" / "experiments" / "orderbook_regularized_sequence_v1"
DOC_PATH = ROOT / "docs" / "ORDERBOOK_REGULARIZED_SEQUENCE_V1.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "42_orderbook_regularized_sequence_v1.ipynb"

TARGET_COST05 = "exec_net_cost_0p5_H60"
TARGET_COST10 = "exec_net_cost_1p0_H60"
TARGET_HEALTHY = "healthy_fill_proxy_0p25_H60"
TARGET_ADVERSE = "adverse_fill_proxy_0p25_H60"

BASELINE_RANKERS = ["proba", "hybrid_score", "exec_score", "v2_safe_proba"]
DAILY_CAPS = [50, 75, 100, 150]
FILL_RATES = [1.0, 0.5]
SEEDS = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]

RIDGE_ALPHAS = [1.0, 10.0, 100.0, 300.0]
LOGIT_CS = [0.03, 0.10, 0.30, 1.0]


def fmt(x: object, digits: int = 4) -> str:
    try:
        if x is None or pd.isna(x):
            return "nan"
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_Sin filas._"
    view = df.copy()
    if max_rows is not None:
        view = view.head(max_rows)
    lines = [
        "| " + " | ".join([str(c) for c in view.columns]) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        cells = [fmt(v, 4) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    return (
        s.astype(str)
        .str.lower()
        .str.strip()
        .map({"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False})
        .fillna(False)
        .astype(bool)
    )


def load_actions() -> pd.DataFrame:
    actions = pd.read_csv(PREDICTIONS_PATH)
    for col in [TARGET_HEALTHY, TARGET_ADVERSE, "target_supported_H60"]:
        if col in actions.columns:
            actions[col] = parse_bool_series(actions[col])
    book = pd.read_csv(BOOK_FEATURES_PATH)
    out = actions.merge(book, on="sequence_id", how="left", suffixes=("", "_book"))
    out[TARGET_HEALTHY] = out[TARGET_HEALTHY].astype(int)
    out[TARGET_ADVERSE] = out[TARGET_ADVERSE].astype(int)
    return out


def book_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("book_") and pd.api.types.is_numeric_dtype(df[c])]


def score_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "pred",
        "proba",
        "exec_score",
        "hybrid_score",
        "last_ev_pred_full",
        "last_healthy_proba_full",
        "last_ev_pred_noclock",
        "last_healthy_proba_noclock",
        "last_visible_entry_cost_ticks",
        "last_spread_ticks",
    ]
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def prepare_X(df: pd.DataFrame, cols: list[str], medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    X = df[cols].replace([np.inf, -np.inf], np.nan).copy()
    if medians is None:
        medians = X.median(numeric_only=True).fillna(0.0)
    X = X.fillna(medians).fillna(0.0)
    return X, medians


def safe_auc(y: pd.Series, score: pd.Series) -> float:
    y = y.astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y, score))


def safe_ap(y: pd.Series, score: pd.Series) -> float:
    y = y.astype(int)
    if y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y, score))


def fit_predict_models(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    train = out[out["terminal_split"].eq("train_initial") & out["target_supported_H60"].astype(bool)].copy()
    feature_sets = {
        "book_only": book_feature_columns(out),
        "book_plus_scores": book_feature_columns(out) + score_feature_columns(out),
    }
    metric_rows = []
    for fs_name, cols in feature_sets.items():
        X_train, med = prepare_X(train, cols)
        for alpha in RIDGE_ALPHAS:
            name = f"obr_{fs_name}_ridge_a{str(alpha).replace('.', 'p')}"
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=73))
            model.fit(X_train, train[TARGET_COST05].astype(float))
            for split, part in out.groupby("terminal_split", observed=True):
                X_part, _ = prepare_X(part, cols, med)
                pred = model.predict(X_part)
                out.loc[part.index, name] = pred
                supported = part["target_supported_H60"].astype(bool)
                if supported.sum() >= 10:
                    metric_rows.append(
                        {
                            "model_name": name,
                            "model_family": "ridge_ev",
                            "feature_set": fs_name,
                            "terminal_split": split,
                            "rows": int(supported.sum()),
                            "mae_cost0p5_h60": float(mean_absolute_error(part.loc[supported, TARGET_COST05], pred[supported])),
                            "r2_cost0p5_h60": float(r2_score(part.loc[supported, TARGET_COST05], pred[supported])),
                            "spearman_cost0p5_h60": float(pd.Series(pred[supported]).corr(part.loc[supported, TARGET_COST05].reset_index(drop=True), method="spearman")),
                        }
                    )
        for C in LOGIT_CS:
            for target, suffix in [(TARGET_HEALTHY, "healthy"), (TARGET_ADVERSE, "adverse")]:
                name = f"obr_{fs_name}_logit_{suffix}_c{str(C).replace('.', 'p')}"
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=C, solver="lbfgs", class_weight="balanced", max_iter=1000, random_state=79),
                )
                y_train = train[target].astype(int)
                model.fit(X_train, y_train)
                for split, part in out.groupby("terminal_split", observed=True):
                    X_part, _ = prepare_X(part, cols, med)
                    proba = model.predict_proba(X_part)[:, 1]
                    out.loc[part.index, name] = proba
                    supported = part["target_supported_H60"].astype(bool)
                    if supported.sum() >= 10:
                        metric_rows.append(
                            {
                                "model_name": name,
                                "model_family": f"logit_{suffix}",
                                "feature_set": fs_name,
                                "terminal_split": split,
                                "rows": int(supported.sum()),
                                "auc": safe_auc(part.loc[supported, target], pd.Series(proba[supported])),
                                "ap": safe_ap(part.loc[supported, target], pd.Series(proba[supported])),
                            }
                        )

        # Combined scores use the strongest fixed regularization by convention, not test.
        for alpha in [10.0, 100.0]:
            ev_col = f"obr_{fs_name}_ridge_a{str(alpha).replace('.', 'p')}"
            for C in [0.10, 0.30]:
                healthy_col = f"obr_{fs_name}_logit_healthy_c{str(C).replace('.', 'p')}"
                adverse_col = f"obr_{fs_name}_logit_adverse_c{str(C).replace('.', 'p')}"
                combo_col = f"obr_{fs_name}_combo_a{str(alpha).replace('.', 'p')}_c{str(C).replace('.', 'p')}"
                # Rank EV by split to avoid scale drift dominating the probability terms.
                out[combo_col] = np.nan
                for split, part in out.groupby("terminal_split", observed=True):
                    ev_rank = part[ev_col].rank(pct=True, method="average").fillna(0.5)
                    out.loc[part.index, combo_col] = (
                        0.50 * ev_rank
                        + 0.35 * part[healthy_col].astype(float)
                        - 0.15 * part[adverse_col].astype(float)
                    )
    return out, pd.DataFrame(metric_rows)


def select_daily(df: pd.DataFrame, score_col: str, daily_cap: int) -> pd.DataFrame:
    return (
        df.sort_values(["terminal_split", "session_day", score_col, "proba"], ascending=[True, True, False, False])
        .groupby(["terminal_split", "session_day"], observed=True)
        .head(daily_cap)
        .copy()
    )


def fill_random_by_day(df: pd.DataFrame, fill_rate: float, seed: int) -> pd.DataFrame:
    if fill_rate >= 0.999:
        return df.copy()
    rng = np.random.default_rng(seed)
    parts = []
    for _, day in df.groupby("session_day", observed=True):
        n = int(np.floor(len(day) * fill_rate))
        if n <= 0:
            continue
        idx = rng.choice(day.index.to_numpy(), size=n, replace=False)
        parts.append(day.loc[idx])
    return pd.concat(parts, ignore_index=False) if parts else df.iloc[0:0].copy()


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
            "healthy_rate": np.nan,
            "adverse_rate": np.nan,
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
        "healthy_rate": float(filled[TARGET_HEALTHY].mean()),
        "adverse_rate": float(filled[TARGET_ADVERSE].mean()),
    }


def run_policy_grid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = BASELINE_RANKERS + [c for c in df.columns if c.startswith("obr_")]
    rows = []
    day_rows = []
    for score_col in score_cols:
        if score_col not in df.columns:
            continue
        for daily_cap in DAILY_CAPS:
            selected = select_daily(df, score_col, daily_cap)
            for fill_rate in FILL_RATES:
                seeds = [0] if fill_rate >= 0.999 else SEEDS
                for seed in seeds:
                    row = {"score_col": score_col, "daily_cap": daily_cap, "fill_rate": fill_rate, "seed": seed}
                    for split in ["validation_initial", "test_terminal"]:
                        part = selected[selected["terminal_split"].eq(split)]
                        filled = fill_random_by_day(part, fill_rate, seed)
                        summary = summarize(filled)
                        row.update({f"{split}_{k}": v for k, v in summary.items()})
                        if not filled.empty:
                            for day, d in filled.groupby("session_day", observed=True):
                                day_rows.append(
                                    {
                                        "score_col": score_col,
                                        "daily_cap": daily_cap,
                                        "fill_rate": fill_rate,
                                        "seed": seed,
                                        "terminal_split": split,
                                        "session_day": day,
                                        "actions": int(len(d)),
                                        "sum_cost0p5": float(d[TARGET_COST05].sum()),
                                        "sum_cost1p0": float(d[TARGET_COST10].sum()),
                                    }
                                )
                    row["validation_pass_cost0p5"] = (
                        row["validation_initial_actions"] >= 30
                        and row["validation_initial_sum_cost0p5"] > 0
                        and row["validation_initial_negative_days_cost0p5"] == 0
                    )
                    row["test_pass_cost0p5"] = (
                        row["test_terminal_actions"] >= 50
                        and row["test_terminal_sum_cost0p5"] > 0
                        and row["test_terminal_negative_days_cost0p5"] == 0
                    )
                    row["test_pass_cost1p0"] = (
                        row["test_pass_cost0p5"]
                        and row["test_terminal_sum_cost1p0"] > 0
                        and row["test_terminal_negative_days_cost1p0"] == 0
                    )
                    row["selection_score"] = (
                        row["validation_pass_cost0p5"] * 1000
                        + row["validation_initial_sum_cost0p5"]
                        + 0.25 * row["validation_initial_sum_cost1p0"]
                        + 0.50 * row["validation_initial_worst_day_cost0p5"]
                        - 0.01 * row["validation_initial_actions"]
                    )
                    rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(day_rows)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["score_col", "daily_cap", "fill_rate"]
    metrics = [
        "validation_initial_actions",
        "validation_initial_sum_cost0p5",
        "validation_initial_sum_cost1p0",
        "validation_initial_worst_day_cost0p5",
        "test_terminal_actions",
        "test_terminal_sum_cost0p5",
        "test_terminal_sum_cost1p0",
        "test_terminal_worst_day_cost0p5",
        "test_terminal_worst_day_cost1p0",
        "test_terminal_negative_days_cost0p5",
        "test_terminal_negative_days_cost1p0",
        "test_terminal_healthy_rate",
        "test_terminal_adverse_rate",
    ]
    rows = []
    for keys, part in results.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(part))
        for metric in metrics:
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_min"] = float(part[metric].min())
            row[f"{metric}_max"] = float(part[metric].max())
        row["validation_pass_cost0p5_rate"] = float(part["validation_pass_cost0p5"].mean())
        row["test_pass_cost0p5_rate"] = float(part["test_pass_cost0p5"].mean())
        row["test_pass_cost1p0_rate"] = float(part["test_pass_cost1p0"].mean())
        row["selection_score_mean"] = float(part["selection_score"].mean())
        row["is_orderbook_model"] = str(row["score_col"]).startswith("obr_")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_score_mean", ascending=False)


def select_config(agg: pd.DataFrame) -> pd.Series:
    focus = agg[
        agg["fill_rate"].eq(0.5)
        & agg["validation_pass_cost0p5_rate"].ge(0.6)
        & agg["validation_initial_actions_mean"].ge(30)
    ].copy()
    if focus.empty:
        focus = agg[agg["fill_rate"].eq(0.5)].copy()
    return focus.sort_values("selection_score_mean", ascending=False).iloc[0]


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Orderbook regularized sequence v1\n", "\n", "Modelo regularizado con features secuenciales interpretables del orderbook.\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "base = '../data/experiments/orderbook_regularized_sequence_v1'\n",
                    "metrics = pd.read_csv(f'{base}/model_metrics.csv')\n",
                    "agg = pd.read_csv(f'{base}/aggregate_results.csv')\n",
                    "metrics.head(20)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["agg[agg.fill_rate.eq(0.5)].head(25)\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_doc(decision: dict, selected: pd.Series, metrics: pd.DataFrame, agg: pd.DataFrame) -> None:
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "score_col",
        "daily_cap",
        "fill_rate",
        "runs",
        "validation_pass_cost0p5_rate",
        "test_pass_cost0p5_rate",
        "test_pass_cost1p0_rate",
        "test_terminal_sum_cost0p5_mean",
        "test_terminal_worst_day_cost0p5_mean",
        "test_terminal_sum_cost1p0_mean",
        "test_terminal_worst_day_cost1p0_mean",
        "is_orderbook_model",
    ]
    random50 = agg[agg["fill_rate"].eq(0.5)].copy()
    best_ob = random50[random50["is_orderbook_model"]].sort_values("selection_score_mean", ascending=False).head(12)
    lines = [
        "# Orderbook regularized sequence v1",
        "",
        "Fecha: 2026-06-03",
        "",
        "## Objetivo",
        "",
        "Probar un modelo pequeno y regularizado con features secuenciales interpretables del orderbook.",
        "",
        "No es CNN. No es bot. Es el paso intermedio entre gates manuales y encoder profundo.",
        "",
        "## Decision",
        "",
        "```text",
        decision["decision"],
        "```",
        "",
        decision["reason"],
        "",
        "## Config seleccionada por validation",
        "",
        "```text",
        f"score={selected['score_col']}, cap={int(selected['daily_cap'])}, fill={selected['fill_rate']}",
        "```",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| validation pass cost0.50 | {fmt(selected['validation_pass_cost0p5_rate'])} |",
        f"| test pass cost0.50 | {fmt(selected['test_pass_cost0p5_rate'])} |",
        f"| test pass cost1.00 | {fmt(selected['test_pass_cost1p0_rate'])} |",
        f"| test mean sum cost0.50 | {fmt(selected['test_terminal_sum_cost0p5_mean'])} |",
        f"| test mean worst day cost0.50 | {fmt(selected['test_terminal_worst_day_cost0p5_mean'])} |",
        f"| test mean sum cost1.00 | {fmt(selected['test_terminal_sum_cost1p0_mean'])} |",
        f"| test mean worst day cost1.00 | {fmt(selected['test_terminal_worst_day_cost1p0_mean'])} |",
        "",
        "## Metricas predictivas",
        "",
        markdown_table(metrics.head(40)),
        "",
        "## Top configs fill 50% random",
        "",
        markdown_table(random50[cols].head(25)),
        "",
        "## Mejores modelos orderbook por validation",
        "",
        markdown_table(best_ob[cols]),
        "",
        "## Lectura sencilla",
        "",
        "- Si validation selecciona un score `obr_*` y test aguanta, orderbook regularizado aporta algo operativo.",
        "- Si validation vuelve a baseline, las features de book son diagnosticas pero no mejoran policy en este protocolo.",
        "- Si un modelo orderbook va bien en test pero no validation, sigue siendo pista, no resultado final.",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    actions = load_actions()
    scored, metrics = fit_predict_models(actions)
    results, day_results = run_policy_grid(scored)
    agg = aggregate(results)
    selected = select_config(agg)

    selected_is_ob = bool(selected["is_orderbook_model"])
    pass_test = (
        selected_is_ob
        and selected["test_pass_cost0p5_rate"] >= 0.6
        and selected["test_terminal_worst_day_cost0p5_mean"] >= 0
    )
    if pass_test:
        decision_name = "RESEARCH_PASS_ORDERBOOK_REGULARIZED_SEQUENCE_COST05"
        reason = "Validation-selected regularized orderbook score passes test cost0.50 under random 50% fills."
    elif selected_is_ob:
        decision_name = "NO_GO_ORDERBOOK_REGULARIZED_SEQUENCE_COST05"
        reason = "Validation selects a regularized orderbook score, but it does not close test cost0.50 robustly."
    else:
        decision_name = "NO_GO_ORDERBOOK_REGULARIZED_SELECTED_BASELINE"
        reason = "Validation selection falls back to a non-orderbook baseline score."

    decision = {
        "decision": decision_name,
        "reason": reason,
        "selected_config": {
            "score_col": selected["score_col"],
            "daily_cap": int(selected["daily_cap"]),
            "fill_rate": float(selected["fill_rate"]),
            "selected_is_orderbook_model": selected_is_ob,
            "validation_pass_cost0p5_rate": float(selected["validation_pass_cost0p5_rate"]),
            "test_pass_cost0p5_rate": float(selected["test_pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(selected["test_pass_cost1p0_rate"]),
            "test_sum_cost0p5_mean": float(selected["test_terminal_sum_cost0p5_mean"]),
            "test_worst_day_cost0p5_mean": float(selected["test_terminal_worst_day_cost0p5_mean"]),
            "test_sum_cost1p0_mean": float(selected["test_terminal_sum_cost1p0_mean"]),
            "test_worst_day_cost1p0_mean": float(selected["test_terminal_worst_day_cost1p0_mean"]),
        },
        "runtime_seconds": round(time.time() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "predictions": str(OUT_DIR / "predictions.csv"),
            "model_metrics": str(OUT_DIR / "model_metrics.csv"),
            "policy_results": str(OUT_DIR / "policy_results.csv"),
            "aggregate_results": str(OUT_DIR / "aggregate_results.csv"),
            "day_results": str(OUT_DIR / "day_results.csv"),
            "doc": str(DOC_PATH),
            "notebook": str(NOTEBOOK_PATH),
        },
    }

    scored.to_csv(OUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "model_metrics.csv", index=False)
    results.to_csv(OUT_DIR / "policy_results.csv", index=False)
    agg.to_csv(OUT_DIR / "aggregate_results.csv", index=False)
    day_results.to_csv(OUT_DIR / "day_results.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_notebook()
    write_doc(decision, selected, metrics, agg)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
