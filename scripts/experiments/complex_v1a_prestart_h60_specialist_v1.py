from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = ROOT / "data" / "experiments" / "complex_v1a_execution_dataset"
DATASET = DATASET_DIR / "complex_v1a_execution_dataset.parquet"
DATASET_MANIFEST = DATASET_DIR / "manifest.json"

OUT_DIR = ROOT / "data" / "experiments" / "complex_v1a_prestart_h60_specialist_v1"
MODEL_DIR = OUT_DIR / "models"
NOTEBOOK_PATH = ROOT / "notebooks" / "15_complex_v1a_prestart_h60_specialist_v1.ipynb"

RANDOM_STATE = 44
TAG = "H60"
TARGET_COL = "exec_net_cost_0p25_H60"
HEALTHY_COL = "healthy_fill_proxy_0p25_H60"
POSITIVE_COL = "target_exec_positive_cost_0p25_H60"
MIN_VALIDATION_ACTIONS = 150
MIN_TEST_ACTIONS = 100

FIXED_EV_THRESHOLDS = [-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
HEALTHY_THRESHOLDS = [None, 0.50, 0.55, 0.60]

AUDIT_COLUMNS = [
    "session_id",
    "market_id",
    "token_id",
    "time_index_ns",
    "time_index_utc",
    "terminal_split",
    "session_day",
    "window_phase",
    "phase_bucket",
    "temporality",
    "seconds_from_window_start",
    "seconds_to_window_end",
    "age_ms",
    "spread_ticks",
    "visible_entry_cost_ticks",
    "full_book_ratio",
    "degraded_ratio",
    "practical_dead_early",
    "tradability_status",
    "polymarket_mid",
    "target_supported_H60",
    "exec_net_cost_0p25_H60",
    "exec_net_cost_0p5_H60",
    "exec_net_cost_1p0_H60",
    "target_exec_positive_cost_0p25_H60",
    "target_exec_buffered_cost_0p25_H60",
    "healthy_fill_proxy_0p25_H60",
    "adverse_fill_proxy_0p25_H60",
]

STRICT_FILTERS = {
    "window_phase": "prestart",
    "target_supported_H60": True,
    "spread_ticks_max": 2.0,
    "visible_entry_cost_ticks_max": 1.25,
    "age_ms_max": 1000.0,
    "full_book_ratio_min": 0.999,
    "degraded_ratio_max": 0.0,
}

UNIVERSES = {
    "strict_0_60_all": (0.0, 60.0),
    "strict_45_60_early": (45.0, 60.0),
}

CLOCK_FEATURES = {
    "seconds_from_window_start",
    "seconds_to_window_end",
    "window_progress",
    "phase_bucket",
    "window_phase",
    "temporality",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(feature_columns: list[str]) -> pd.DataFrame:
    columns = list(dict.fromkeys(AUDIT_COLUMNS + feature_columns))
    df = pd.read_parquet(DATASET, columns=columns)
    for col in feature_columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]) or isinstance(
            df[col].dtype, pd.CategoricalDtype
        ):
            df[col] = df[col].astype("category")
    return df


def add_prestart_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seconds_before_start"] = -pd.to_numeric(out["seconds_from_window_start"], errors="coerce")
    out["prestart_bucket"] = pd.cut(
        out["seconds_before_start"],
        bins=[0, 10, 30, 45, 60, np.inf],
        labels=["0-10s", "10-30s", "30-45s", "45-60s", ">60s"],
        right=True,
    ).astype("string")
    out["prestart_bucket"] = out["prestart_bucket"].fillna("not_prestart")
    return out


def strict_mask(df: pd.DataFrame, seconds_range: tuple[float, float]) -> pd.Series:
    lo, hi = seconds_range
    return (
        df["window_phase"].astype(str).eq("prestart")
        & df["target_supported_H60"].fillna(False)
        & df["seconds_before_start"].gt(lo)
        & df["seconds_before_start"].le(hi)
        & pd.to_numeric(df["spread_ticks"], errors="coerce").le(STRICT_FILTERS["spread_ticks_max"])
        & pd.to_numeric(df["visible_entry_cost_ticks"], errors="coerce").le(
            STRICT_FILTERS["visible_entry_cost_ticks_max"]
        )
        & pd.to_numeric(df["age_ms"], errors="coerce").le(STRICT_FILTERS["age_ms_max"])
        & pd.to_numeric(df["full_book_ratio"], errors="coerce").ge(STRICT_FILTERS["full_book_ratio_min"])
        & pd.to_numeric(df["degraded_ratio"], errors="coerce").le(STRICT_FILTERS["degraded_ratio_max"])
        & df[TARGET_COL].notna()
    )


def feature_sets(feature_columns: list[str]) -> dict[str, list[str]]:
    no_clock = [col for col in feature_columns if col not in CLOCK_FEATURES]
    return {
        "full_features": feature_columns,
        "no_clock": no_clock,
    }


def regressor_factory() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=0.10,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )


def classifier_factory() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.045,
        max_iter=220,
        max_leaf_nodes=15,
        min_samples_leaf=120,
        l2_regularization=0.10,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=RANDOM_STATE,
    )


def safe_auc(y_true: pd.Series, score: np.ndarray) -> float:
    y = y_true.astype("boolean")
    valid = y.notna()
    y = y[valid]
    if y.empty or y.nunique(dropna=True) < 2:
        return np.nan
    return float(roc_auc_score(y.astype(bool), np.asarray(score)[valid.to_numpy()]))


def regression_metrics(part: pd.DataFrame, pred: np.ndarray) -> dict:
    y = part[TARGET_COL].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "rows": int(len(part)),
        "target_mean": float(np.mean(y)),
        "pred_mean": float(np.mean(pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "pearson_corr": float(pd.Series(y).corr(pd.Series(pred), method="pearson")),
        "spearman_corr": float(pd.Series(y).corr(pd.Series(pred), method="spearman")),
        "auc_positive_cost_0p25": safe_auc(part[POSITIVE_COL], pred),
    }


def classifier_metrics(part: pd.DataFrame, healthy_proba: np.ndarray | None) -> dict:
    if healthy_proba is None:
        return {"auc_healthy_proxy": np.nan, "healthy_rate_pct": np.nan}
    y = part[HEALTHY_COL].astype("boolean")
    return {
        "auc_healthy_proxy": safe_auc(y, healthy_proba),
        "healthy_rate_pct": float(y.mean() * 100) if y.notna().any() else np.nan,
    }


def q(series: pd.Series, quantile: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    return float(valid.quantile(quantile))


def summarize_selected(part: pd.DataFrame) -> dict:
    if part.empty:
        return {
            "actions": 0,
            "mean_net_cost_0p25": np.nan,
            "median_net_cost_0p25": np.nan,
            "mean_net_cost_0p5": np.nan,
            "mean_net_cost_1p0": np.nan,
            "expected_total_net_cost_0p25": 0.0,
            "positive_cost_0p25_pct": np.nan,
            "buffered_cost_0p25_pct": np.nan,
            "healthy_proxy_pct": np.nan,
            "adverse_proxy_pct": np.nan,
            "mean_ev_pred": np.nan,
            "mean_healthy_proba": np.nan,
            "mean_spread_ticks": np.nan,
            "mean_visible_cost_ticks": np.nan,
            "mean_age_ms": np.nan,
        }
    healthy = part[HEALTHY_COL].astype("boolean")
    adverse = part["adverse_fill_proxy_0p25_H60"].astype("boolean")
    mean_net = float(part["exec_net_cost_0p25_H60"].mean())
    return {
        "actions": int(len(part)),
        "mean_net_cost_0p25": mean_net,
        "median_net_cost_0p25": float(part["exec_net_cost_0p25_H60"].median()),
        "mean_net_cost_0p5": float(part["exec_net_cost_0p5_H60"].mean()),
        "mean_net_cost_1p0": float(part["exec_net_cost_1p0_H60"].mean()),
        "expected_total_net_cost_0p25": mean_net * len(part),
        "positive_cost_0p25_pct": float(part["target_exec_positive_cost_0p25_H60"].mean() * 100),
        "buffered_cost_0p25_pct": float(part["target_exec_buffered_cost_0p25_H60"].mean() * 100),
        "healthy_proxy_pct": float(healthy.mean() * 100) if healthy.notna().any() else np.nan,
        "adverse_proxy_pct": float(adverse.mean() * 100) if adverse.notna().any() else np.nan,
        "mean_ev_pred": float(part["ev_pred"].mean()),
        "mean_healthy_proba": float(part["healthy_proba"].mean()) if "healthy_proba" in part else np.nan,
        "mean_spread_ticks": float(part["spread_ticks"].mean()),
        "mean_visible_cost_ticks": float(part["visible_entry_cost_ticks"].mean()),
        "mean_age_ms": float(part["age_ms"].mean()),
    }


def ev_thresholds(validation: pd.DataFrame) -> list[tuple[str, float]]:
    pred = validation["ev_pred"].to_numpy(dtype=float)
    out = []
    for fraction in TOP_FRACTIONS:
        out.append((f"top_{fraction * 100:.3g}pct", float(np.quantile(pred, 1.0 - fraction))))
    for threshold in FIXED_EV_THRESHOLDS:
        out.append((f"ev_gt_{threshold:g}", float(threshold)))
    return out


def evaluate_policies(universe: str, feature_set: str, part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = part[part["terminal_split"].eq("validation_initial")]
    test = part[part["terminal_split"].eq("test_terminal")]
    rows = []

    for ev_policy, ev_threshold in ev_thresholds(validation):
        for healthy_threshold in HEALTHY_THRESHOLDS:
            policy_name = ev_policy if healthy_threshold is None else f"{ev_policy}_healthy_ge_{healthy_threshold:g}"
            for split_name, split_part in [("validation_initial", validation), ("test_terminal", test)]:
                selected = split_part["ev_pred"].ge(ev_threshold)
                if healthy_threshold is not None:
                    selected = selected & split_part["healthy_proba"].ge(healthy_threshold)
                rows.append(
                    {
                        "universe": universe,
                        "feature_set": feature_set,
                        "terminal_split": split_name,
                        "policy_name": policy_name,
                        "ev_threshold": ev_threshold,
                        "healthy_threshold": healthy_threshold,
                        **summarize_selected(split_part[selected]),
                    }
                )

    policy_results = pd.DataFrame(rows)
    val = policy_results[
        policy_results["terminal_split"].eq("validation_initial")
        & policy_results["actions"].ge(MIN_VALIDATION_ACTIONS)
        & policy_results["mean_net_cost_0p25"].gt(0)
        & policy_results["mean_net_cost_0p5"].gt(0)
    ].copy()
    if val.empty:
        return policy_results, pd.DataFrame(
            [
                {
                    "universe": universe,
                    "feature_set": feature_set,
                    "selected_policy_name": "NO_TRADE",
                    "selection_reason": "No validation policy positive at cost 0.25 and 0.50 with enough actions.",
                }
            ]
        )
    val["selection_score"] = (
        val["mean_net_cost_0p5"] * np.sqrt(val["actions"]) - 0.01 * val["adverse_proxy_pct"].fillna(0)
    )
    best = val.sort_values(["selection_score", "expected_total_net_cost_0p25"], ascending=[False, False]).iloc[0]
    selected = policy_results[
        policy_results["policy_name"].eq(best["policy_name"])
        & policy_results["universe"].eq(universe)
        & policy_results["feature_set"].eq(feature_set)
    ].copy()
    selected.insert(0, "selected_policy_name", best["policy_name"])
    selected["selection_score_validation"] = float(best["selection_score"])
    return policy_results, selected


def train_and_score(universe: str, feature_set_name: str, features: list[str], part: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = part[part["terminal_split"].eq("train_initial")].copy()
    validation = part[part["terminal_split"].eq("validation_initial")].copy()
    test = part[part["terminal_split"].eq("test_terminal")].copy()
    if len(train) < 1_000 or len(validation) < 300 or len(test) < 300:
        raise RuntimeError(f"Low support for {universe}/{feature_set_name}")

    reg = regressor_factory()
    reg.fit(train[features], train[TARGET_COL])
    reg_path = MODEL_DIR / f"specialist_ev_{universe}_{feature_set_name}.joblib"
    joblib.dump(reg, reg_path)

    clf = classifier_factory()
    healthy_train = train[HEALTHY_COL].astype("boolean")
    clf_path = None
    healthy_train_valid = healthy_train.notna()
    has_classifier = healthy_train[healthy_train_valid].nunique(dropna=True) == 2
    if has_classifier:
        clf.fit(train.loc[healthy_train_valid, features], healthy_train.loc[healthy_train_valid].astype(bool))
        clf_path = MODEL_DIR / f"specialist_healthy_{universe}_{feature_set_name}.joblib"
        joblib.dump(clf, clf_path)

    scored_parts = []
    metric_rows = []
    for split_name, split_part in [
        ("train_initial", train),
        ("validation_initial", validation),
        ("test_terminal", test),
    ]:
        scored = split_part.copy()
        scored["ev_pred"] = reg.predict(scored[features])
        if has_classifier:
            scored["healthy_proba"] = clf.predict_proba(scored[features])[:, 1]
        else:
            scored["healthy_proba"] = np.nan
        scored_parts.append(scored)
        metric_rows.append(
            {
                "universe": universe,
                "feature_set": feature_set_name,
                "terminal_split": split_name,
                "model_iterations_ev": int(reg.n_iter_),
                "model_iterations_healthy": int(clf.n_iter_) if has_classifier else np.nan,
                **regression_metrics(scored, scored["ev_pred"].to_numpy()),
                **classifier_metrics(scored, scored["healthy_proba"].to_numpy() if has_classifier else None),
            }
        )

    scored_all = pd.concat(scored_parts, ignore_index=True)
    policies, selected = evaluate_policies(universe, feature_set_name, scored_all)
    model_info = {
        "universe": universe,
        "feature_set": feature_set_name,
        "ev_model_path": str(reg_path),
        "healthy_model_path": str(clf_path) if clf_path is not None else None,
        "features": features,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    return pd.DataFrame(metric_rows), policies, selected, model_info


def summarize_universe(universe: str, part: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, split_part in part.groupby("terminal_split", observed=True):
        rows.append(
            {
                "universe": universe,
                "terminal_split": split_name,
                "rows": int(len(split_part)),
                "sessions": int(split_part["session_id"].nunique()),
                "markets": int(split_part[["session_id", "market_id"]].drop_duplicates().shape[0]),
                "tokens": int(split_part["token_id"].nunique()),
                "mean_net_cost_0p25": float(split_part["exec_net_cost_0p25_H60"].mean()),
                "mean_net_cost_0p5": float(split_part["exec_net_cost_0p5_H60"].mean()),
                "mean_net_cost_1p0": float(split_part["exec_net_cost_1p0_H60"].mean()),
                "positive_cost_0p25_pct": float(split_part["target_exec_positive_cost_0p25_H60"].mean() * 100),
                "healthy_proxy_pct": float(split_part[HEALTHY_COL].astype("boolean").mean() * 100),
                "mean_spread_ticks": float(split_part["spread_ticks"].mean()),
                "p95_spread_ticks": q(split_part["spread_ticks"], 0.95),
                "mean_visible_cost_ticks": float(split_part["visible_entry_cost_ticks"].mean()),
                "p95_visible_cost_ticks": q(split_part["visible_entry_cost_ticks"], 0.95),
                "mean_age_ms": float(split_part["age_ms"].mean()),
                "p95_age_ms": q(split_part["age_ms"], 0.95),
            }
        )
    return pd.DataFrame(rows)


def bucket_breakdown(scored_parts: dict[tuple[str, str], pd.DataFrame], selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = selected[selected["selected_policy_name"].notna() & ~selected["selected_policy_name"].eq("NO_TRADE")]
    selected_policies = valid[
        ["universe", "feature_set", "selected_policy_name", "ev_threshold", "healthy_threshold"]
    ].drop_duplicates()
    for _, selected_row in selected_policies.iterrows():
        key = (selected_row["universe"], selected_row["feature_set"])
        scored = scored_parts[key]
        ev_threshold = float(selected_row["ev_threshold"])
        healthy_threshold = selected_row["healthy_threshold"]
        chosen = scored["ev_pred"].ge(ev_threshold)
        if pd.notna(healthy_threshold):
            chosen = chosen & scored["healthy_proba"].ge(float(healthy_threshold))
        sub = scored[chosen].copy()
        for (split, bucket), part in sub.groupby(["terminal_split", "prestart_bucket"], observed=True):
            rows.append(
                {
                    "universe": selected_row["universe"],
                    "feature_set": selected_row["feature_set"],
                    "selected_policy_name": selected_row["selected_policy_name"],
                    "terminal_split": split,
                    "prestart_bucket": bucket,
                    **summarize_selected(part),
                }
            )
    return pd.DataFrame(rows)


def day_breakdown(scored_parts: dict[tuple[str, str], pd.DataFrame], selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = selected[selected["selected_policy_name"].notna() & ~selected["selected_policy_name"].eq("NO_TRADE")]
    selected_policies = valid[
        ["universe", "feature_set", "selected_policy_name", "ev_threshold", "healthy_threshold"]
    ].drop_duplicates()
    for _, selected_row in selected_policies.iterrows():
        key = (selected_row["universe"], selected_row["feature_set"])
        scored = scored_parts[key]
        ev_threshold = float(selected_row["ev_threshold"])
        healthy_threshold = selected_row["healthy_threshold"]
        chosen = scored["ev_pred"].ge(ev_threshold)
        if pd.notna(healthy_threshold):
            chosen = chosen & scored["healthy_proba"].ge(float(healthy_threshold))
        sub = scored[chosen].copy()
        for (split, session_day), part in sub.groupby(["terminal_split", "session_day"], observed=True):
            rows.append(
                {
                    "universe": selected_row["universe"],
                    "feature_set": selected_row["feature_set"],
                    "selected_policy_name": selected_row["selected_policy_name"],
                    "terminal_split": split,
                    "session_day": session_day,
                    **summarize_selected(part),
                }
            )
    return pd.DataFrame(rows)


def decide(selected: pd.DataFrame) -> dict:
    candidate_rows = []
    if selected.empty:
        return {"decision": "NO_TRADE", "reason": "No selected policies.", "terminal_rows": candidate_rows}
    valid = selected[selected["selected_policy_name"].notna() & ~selected["selected_policy_name"].eq("NO_TRADE")]
    for (universe, feature_set, policy), group in valid.groupby(
        ["universe", "feature_set", "selected_policy_name"], observed=True
    ):
        val = group[group["terminal_split"].eq("validation_initial")]
        test = group[group["terminal_split"].eq("test_terminal")]
        if val.empty or test.empty:
            continue
        val_row = val.iloc[0]
        test_row = test.iloc[0]
        reasons = []
        if int(test_row["actions"]) < MIN_TEST_ACTIONS:
            reasons.append("low_test_actions")
        if float(test_row["mean_net_cost_0p25"]) <= 0:
            reasons.append("test_cost0p25_non_positive")
        if float(test_row["mean_net_cost_0p5"]) <= 0:
            reasons.append("test_cost0p5_non_positive")
        if float(test_row["mean_net_cost_1p0"]) <= 0:
            reasons.append("test_cost1p0_non_positive")
        status = "RESEARCH_PASS_STRONG" if not reasons else "WATCH_FAIL"
        candidate_rows.append(
            {
                "universe": universe,
                "feature_set": feature_set,
                "policy": policy,
                "status": status,
                "reasons": ",".join(reasons) if reasons else "test_positive_cost0p25_0p5_1p0",
                "validation_actions": int(val_row["actions"]),
                "validation_mean_net_cost_0p25": float(val_row["mean_net_cost_0p25"]),
                "validation_mean_net_cost_0p5": float(val_row["mean_net_cost_0p5"]),
                "validation_mean_net_cost_1p0": float(val_row["mean_net_cost_1p0"]),
                "test_actions": int(test_row["actions"]),
                "test_mean_net_cost_0p25": float(test_row["mean_net_cost_0p25"]),
                "test_mean_net_cost_0p5": float(test_row["mean_net_cost_0p5"]),
                "test_mean_net_cost_1p0": float(test_row["mean_net_cost_1p0"]),
            }
        )
    passes = [row for row in candidate_rows if row["status"] == "RESEARCH_PASS_STRONG"]
    if passes:
        best = sorted(
            passes,
            key=lambda r: (r["test_mean_net_cost_1p0"], r["test_mean_net_cost_0p5"], r["test_actions"]),
            reverse=True,
        )[0]
        no_clock_pass = any(row["feature_set"] == "no_clock" for row in passes)
        return {
            "decision": "RESEARCH_PASS_STRONG_NOT_BOT",
            "reason": "At least one specialist policy is validation-positive and test-positive through cost 1.0.",
            "best_candidate": best,
            "no_clock_pass": no_clock_pass,
            "terminal_rows": candidate_rows,
        }
    return {
        "decision": "WATCH_FAIL_NOT_APPROVED",
        "reason": "Specialist selected policies did not pass terminal test through cost 1.0.",
        "terminal_rows": candidate_rows,
    }


def write_notebook() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Complex v1a - Prestart H60 specialist v1\n",
                    "\n",
                    "Entrena modelos especialistas en `strict_0_60_all` y `strict_45_60_early`, con feature sets `full_features` y `no_clock`. La seleccion se hace solo con validation.\n",
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
                    "\n",
                    "OUT = Path('../data/experiments/complex_v1a_prestart_h60_specialist_v1')\n",
                    "manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))\n",
                    "decision = json.loads((OUT / 'decision.json').read_text(encoding='utf-8'))\n",
                    "universe = pd.read_csv(OUT / 'specialist_universe_summary.csv')\n",
                    "metrics = pd.read_csv(OUT / 'specialist_model_metrics.csv')\n",
                    "selected = pd.read_csv(OUT / 'specialist_selected_policies.csv')\n",
                    "buckets = pd.read_csv(OUT / 'specialist_selected_bucket_breakdown.csv')\n",
                    "days = pd.read_csv(OUT / 'specialist_selected_day_breakdown.csv')\n",
                    "decision\n",
                ],
            },
            {"cell_type": "markdown", "metadata": {}, "source": ["## 1. Universe summary\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["universe.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 2. Model metrics\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["metrics.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 3. Selected policies\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["selected.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 4. Bucket breakdown\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["buckets.round(5)\n"]},
            {"cell_type": "markdown", "metadata": {}, "source": ["## 5. Day breakdown\n"]},
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["days.sort_values(['universe','feature_set','terminal_split','session_day']).round(5)\n"],
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


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    manifest = read_json(DATASET_MANIFEST)
    features_by_set = feature_sets(manifest["feature_columns"])
    df = add_prestart_fields(load_dataset(manifest["feature_columns"]))

    universe_frames = []
    metric_frames = []
    policy_frames = []
    selected_frames = []
    model_infos = []
    scored_parts: dict[tuple[str, str], pd.DataFrame] = {}

    for universe, seconds_range in UNIVERSES.items():
        part = df[strict_mask(df, seconds_range)].copy()
        universe_frames.append(summarize_universe(universe, part))
        for feature_set_name, features in features_by_set.items():
            metrics, policies, selected, model_info = train_and_score(universe, feature_set_name, features, part)
            metric_frames.append(metrics)
            policy_frames.append(policies)
            selected_frames.append(selected)
            model_infos.append(model_info)

            # Re-score once for bucket reporting; keeps saved CSVs compact.
            ev_model = joblib.load(model_info["ev_model_path"])
            healthy_model_path = model_info["healthy_model_path"]
            scored = part.copy()
            scored["ev_pred"] = ev_model.predict(scored[features])
            if healthy_model_path:
                healthy_model = joblib.load(healthy_model_path)
                scored["healthy_proba"] = healthy_model.predict_proba(scored[features])[:, 1]
            else:
                scored["healthy_proba"] = np.nan
            scored_parts[(universe, feature_set_name)] = scored

    universe_summary = pd.concat(universe_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    policies = pd.concat(policy_frames, ignore_index=True)
    selected = pd.concat(selected_frames, ignore_index=True)
    buckets = bucket_breakdown(scored_parts, selected)
    days = day_breakdown(scored_parts, selected)
    decision = decide(selected)

    universe_summary.to_csv(OUT_DIR / "specialist_universe_summary.csv", index=False)
    metrics.to_csv(OUT_DIR / "specialist_model_metrics.csv", index=False)
    policies.to_csv(OUT_DIR / "specialist_policy_results.csv", index=False)
    selected.to_csv(OUT_DIR / "specialist_selected_policies.csv", index=False)
    buckets.to_csv(OUT_DIR / "specialist_selected_bucket_breakdown.csv", index=False)
    days.to_csv(OUT_DIR / "specialist_selected_day_breakdown.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    run_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Prestart H60 specialist v1.",
        "input_dataset": str(DATASET),
        "universes": {
            name: {"seconds_before_start_gt": lo, "seconds_before_start_le": hi}
            for name, (lo, hi) in UNIVERSES.items()
        },
        "strict_filters": STRICT_FILTERS,
        "feature_sets": {name: cols for name, cols in features_by_set.items()},
        "model": {
            "ev": "sklearn HistGradientBoostingRegressor",
            "healthy": "sklearn HistGradientBoostingClassifier",
        },
        "selection": {
            "split": "validation_initial",
            "min_validation_actions": MIN_VALIDATION_ACTIONS,
            "requires_validation_mean_net_cost_0p25_gt_0": True,
            "requires_validation_mean_net_cost_0p5_gt_0": True,
            "selection_score": "mean_net_cost_0p5 * sqrt(actions) - 0.01 * adverse_proxy_pct",
        },
        "decision": decision,
        "model_infos": model_infos,
        "outputs": {
            "specialist_universe_summary": str(OUT_DIR / "specialist_universe_summary.csv"),
            "specialist_model_metrics": str(OUT_DIR / "specialist_model_metrics.csv"),
            "specialist_policy_results": str(OUT_DIR / "specialist_policy_results.csv"),
            "specialist_selected_policies": str(OUT_DIR / "specialist_selected_policies.csv"),
            "specialist_selected_bucket_breakdown": str(OUT_DIR / "specialist_selected_bucket_breakdown.csv"),
            "specialist_selected_day_breakdown": str(OUT_DIR / "specialist_selected_day_breakdown.csv"),
            "decision": str(OUT_DIR / "decision.json"),
            "notebook": str(NOTEBOOK_PATH),
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    write_notebook()
    print(json.dumps({k: v for k, v in run_manifest.items() if k != "feature_sets"}, indent=2))


if __name__ == "__main__":
    main()
