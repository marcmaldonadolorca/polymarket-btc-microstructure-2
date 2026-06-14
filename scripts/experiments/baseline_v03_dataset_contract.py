from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/experiments/baseline_v0_full_core_robustness/baseline_v0_core_h16_parquet")
V02_DIR = Path("data/experiments/baseline_v02_conservative_backtest")
OUT_DIR = Path("data/experiments/baseline_v03_dataset_contract")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_COLS = ["session_id", "condition_id", "time_index_ns"]

PM_NUMERIC = [
    "polymarket_mid",
    "boundary_distance",
    "microprice_gap_ticks_clipped",
    "mid_delta_ticks_2s",
    "mid_delta_ticks_8s",
    "spread_ticks",
    "visible_entry_cost_ticks",
    "age_ms",
]
PM_NO_MICRO_NUMERIC = [x for x in PM_NUMERIC if x != "microprice_gap_ticks_clipped"]
SPOT_NUMERIC = [
    "external_mid_return_bps_2s_oriented",
    "external_mid_return_bps_8s_oriented",
    "external_trade_imbalance_oriented",
    "external_realized_vol_bps_5s",
]
PERP_NUMERIC = [
    "perp_mid_return_bps_2s_oriented",
    "perp_mid_return_bps_8s_oriented",
    "perp_mark_price_return_bps_2s_oriented",
    "perp_mark_price_return_bps_8s_oriented",
    "perp_taker_buy_sell_log_oriented",
    "perp_basis",
    "perp_basis_delta_8s",
    "perp_open_interest_delta_8s",
    "perp_realized_vol_bps_5s",
    "spot_perp_mid_gap_bps",
    "spot_perp_mark_gap_bps",
]
TIME_NUMERIC = [
    "seconds_to_window_end",
    "seconds_from_window_start",
    "window_progress",
    "tick_size",
    "fee_rate_bps",
    "min_order_size",
]
QUALITY_NUMERIC = [
    "chainlink_missing",
    "chainlink_staleness_ms",
    "freshness_gap_ms",
    "joint_age_ms",
    "missing_context_count",
    "stale_context_count",
    "full_book_ratio",
    "degraded_ratio",
]
INTERTEMP_NUMERIC = [
    "intertemporal_mid_vs_group_mean",
    "intertemporal_mid_vs_nearest_shorter",
    "intertemporal_mid_vs_nearest_longer",
    "intertemporal_group_range",
    "intertemporal_curve_residual",
]
BASE_CATS = ["temporality", "window_phase", "phase_bucket"]
AGREEMENT_CATS = ["pm_perp_agreement", "pm_spot_agreement"]
QUALITY_CATS = ["tradability_status"]

FEATURE_SETS = {
    "v03_general_full_v0": {
        "num": PM_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC,
        "cat": BASE_CATS + AGREEMENT_CATS,
        "purpose": "baseline general comparable con v0.1",
    },
    "v03_conservative_no_micro": {
        "num": PM_NO_MICRO_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC,
        "cat": BASE_CATS,
        "purpose": "politica conservadora candidata tras v0.2",
    },
    "v03_pm_perp_control": {
        "num": PM_NUMERIC + PERP_NUMERIC + TIME_NUMERIC,
        "cat": BASE_CATS + ["pm_perp_agreement"],
        "purpose": "control para medir cuanto aporta spot frente a PM+perp",
    },
    "v03_quality_diagnostic": {
        "num": QUALITY_NUMERIC,
        "cat": QUALITY_CATS,
        "purpose": "masks/diagnostico; no feature fuerte por defecto",
    },
    "v03_intertemporal_holdout": {
        "num": INTERTEMP_NUMERIC,
        "cat": [],
        "purpose": "holdout experimental; no entra en baseline v0.3",
    },
}

KEY_COLS = [
    "id",
    "session_id",
    "session_day",
    "time_index_ns",
    "time_index_utc",
    "market_id",
    "token_id",
    "condition_id",
    "outcome_label",
    "short_outcome_label",
    "outcome_sign",
]
TARGET_COLS = [
    "delta_ticks_16s",
    "delta_ticks_8s",
    "target_3c_16s_1tick",
    "target_3c_8s_1tick",
    "future_delta_ns_8s",
    "future_delta_ns_16s",
]
EVAL_COLS = [
    "entry_ask_est",
    "entry_fee_ticks",
    "visible_entry_cost_ticks",
    "spread_ticks",
    "tradability_status",
    "practical_dead_early",
    "full_book_ratio",
    "degraded_ratio",
]
BENCHMARK_COLS = [
    "predictor_label_name_8s",
    "predictor_economic_label_name_8s",
    "predictor_future_return_bps_8s",
    "predictor_economic_net_return_bps_8s",
]
LEAKAGE_COLS = [
    "future_mid_8s",
    "future_mid_16s",
    "future_microprice_16s",
    "future_time_index_ns_8s",
    "future_time_index_ns_16s",
    "future_stale_8s",
    "future_missing_8s",
    "future_stale_16s",
    "future_missing_16s",
]


def assign_terminal_split(day):
    day = str(day)
    if "2026-05-23" <= day <= "2026-05-25":
        return "test_terminal"
    if "2026-05-21" <= day <= "2026-05-22":
        return "validation_initial"
    if day <= "2026-05-20":
        return "train_initial"
    return "other"


def feature_block(feature):
    if feature in PM_NUMERIC or feature in ["pm_perp_agreement", "pm_spot_agreement"]:
        return "polymarket_microstructure"
    if feature in SPOT_NUMERIC:
        return "spot_reference"
    if feature in PERP_NUMERIC:
        return "perp_reference"
    if feature in TIME_NUMERIC or feature in BASE_CATS:
        return "time_contract"
    if feature in QUALITY_NUMERIC or feature in QUALITY_CATS:
        return "quality_masks"
    if feature in INTERTEMP_NUMERIC:
        return "intertemporal_holdout"
    return "other"


def load_dataset():
    all_feature_cols = sorted({col for fs in FEATURE_SETS.values() for col in fs["num"] + fs["cat"]})
    columns = sorted(set(KEY_COLS + TARGET_COLS + EVAL_COLS + BENCHMARK_COLS + LEAKAGE_COLS + all_feature_cols))
    df = pd.read_parquet(CACHE_DIR, columns=columns)
    df["terminal_split"] = df["session_day"].astype(str).map(assign_terminal_split)
    df["cost_bucket"] = pd.cut(
        pd.to_numeric(df["visible_entry_cost_ticks"], errors="coerce"),
        bins=[-np.inf, 0.75, 1.25, 2.0, np.inf],
        labels=["low_<=0.75", "normal_0.75_1.25", "high_1.25_2", "very_high_>2"],
    )
    return df


def pct(series):
    return float(series.mean() * 100) if len(series) else np.nan


def summarize_group(part):
    return {
        "rows": int(len(part)),
        "market_frames": int(part.drop_duplicates(FRAME_COLS).shape[0]),
        "sessions": int(part["session_id"].nunique()),
        "condition_ids": int(part["condition_id"].nunique()),
        "days": int(part["session_day"].nunique()),
        "target_up_pct": pct(part["target_3c_16s_1tick"].eq("up")),
        "target_flat_pct": pct(part["target_3c_16s_1tick"].eq("flat")),
        "target_down_pct": pct(part["target_3c_16s_1tick"].eq("down")),
        "delta_ticks_16s_mean": float(part["delta_ticks_16s"].mean()),
        "delta_ticks_16s_median": float(part["delta_ticks_16s"].median()),
        "visible_cost_median": float(part["visible_entry_cost_ticks"].median()),
        "spread_ticks_median": float(part["spread_ticks"].median()),
    }


def group_summary(df, keys):
    rows = []
    for key_values, part in df.groupby(keys, dropna=False, observed=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {key: value for key, value in zip(keys, key_values)}
        row.update(summarize_group(part))
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_contract(df):
    all_features = sorted({col for fs in FEATURE_SETS.values() for col in fs["num"] + fs["cat"]})
    general = set(FEATURE_SETS["v03_general_full_v0"]["num"] + FEATURE_SETS["v03_general_full_v0"]["cat"])
    conservative = set(FEATURE_SETS["v03_conservative_no_micro"]["num"] + FEATURE_SETS["v03_conservative_no_micro"]["cat"])
    control = set(FEATURE_SETS["v03_pm_perp_control"]["num"] + FEATURE_SETS["v03_pm_perp_control"]["cat"])
    diagnostic = set(FEATURE_SETS["v03_quality_diagnostic"]["num"] + FEATURE_SETS["v03_quality_diagnostic"]["cat"])
    holdout = set(FEATURE_SETS["v03_intertemporal_holdout"]["num"])

    rows = []
    for feature in all_features:
        s = df[feature] if feature in df.columns else pd.Series(dtype=float)
        is_numeric = pd.api.types.is_numeric_dtype(s)
        rows.append(
            {
                "column": feature,
                "role": "feature_candidate",
                "block": feature_block(feature),
                "dtype": str(s.dtype),
                "missing_pct": float(s.isna().mean() * 100) if len(s) else np.nan,
                "nunique": int(s.nunique(dropna=True)) if len(s) else 0,
                "mean": float(s.mean()) if is_numeric and len(s) else np.nan,
                "std": float(s.std()) if is_numeric and len(s) else np.nan,
                "include_v03_general_full_v0": feature in general,
                "include_v03_conservative_no_micro": feature in conservative,
                "include_v03_pm_perp_control": feature in control,
                "diagnostic_only": feature in diagnostic,
                "holdout_only": feature in holdout,
                "notes": "",
            }
        )

    for column in KEY_COLS:
        if column in df.columns:
            rows.append(
                {
                    "column": column,
                    "role": "key_or_metadata",
                    "block": "keys",
                    "dtype": str(df[column].dtype),
                    "missing_pct": float(df[column].isna().mean() * 100),
                    "nunique": int(df[column].nunique(dropna=True)),
                    "mean": np.nan,
                    "std": np.nan,
                    "include_v03_general_full_v0": False,
                    "include_v03_conservative_no_micro": False,
                    "include_v03_pm_perp_control": False,
                    "diagnostic_only": False,
                    "holdout_only": False,
                    "notes": "join/evaluacion, no input del modelo",
                }
            )
    for column in TARGET_COLS + BENCHMARK_COLS + LEAKAGE_COLS:
        if column in df.columns:
            rows.append(
                {
                    "column": column,
                    "role": "target_or_benchmark" if column not in LEAKAGE_COLS else "leakage_never_feature",
                    "block": "labels" if column not in LEAKAGE_COLS else "future_values",
                    "dtype": str(df[column].dtype),
                    "missing_pct": float(df[column].isna().mean() * 100),
                    "nunique": int(df[column].nunique(dropna=True)),
                    "mean": float(df[column].mean()) if pd.api.types.is_numeric_dtype(df[column]) else np.nan,
                    "std": float(df[column].std()) if pd.api.types.is_numeric_dtype(df[column]) else np.nan,
                    "include_v03_general_full_v0": False,
                    "include_v03_conservative_no_micro": False,
                    "include_v03_pm_perp_control": False,
                    "diagnostic_only": False,
                    "holdout_only": False,
                    "notes": "no usar como feature",
                }
            )
    return pd.DataFrame(rows)


def feature_set_health(feature_contract):
    rows = []
    for name, spec in FEATURE_SETS.items():
        cols = set(spec["num"] + spec["cat"])
        part = feature_contract[feature_contract["column"].isin(cols)]
        rows.append(
            {
                "feature_set": name,
                "purpose": spec["purpose"],
                "n_features": int(len(cols)),
                "numeric_features": int(len(spec["num"])),
                "categorical_features": int(len(spec["cat"])),
                "median_missing_pct": float(part["missing_pct"].median()),
                "max_missing_pct": float(part["missing_pct"].max()),
                "columns": ", ".join(spec["num"] + spec["cat"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_actions(path, label):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["score_bucket"] = pd.cut(
        pd.to_numeric(df["score_buy"], errors="coerce"),
        bins=[-np.inf, 0.55, 0.60, 0.65, 0.70, np.inf],
        labels=["<=0.55", "0.55_0.60", "0.60_0.65", "0.65_0.70", ">0.70"],
    )
    rows = []
    keys_options = [
        ["temporality"],
        ["phase_bucket"],
        ["temporality", "phase_bucket"],
        ["temporality", "score_bucket"],
    ]
    for keys in keys_options:
        for key_values, part in df.groupby(keys, dropna=False, observed=True):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            row = {"scenario_label": label, "segment": "+".join(keys)}
            row.update({key: value for key, value in zip(keys, key_values)})
            row.update(
                {
                    "actions": int(len(part)),
                    "hit_up_pct": pct(part["hit_up"]),
                    "flat_pct": pct(part["flat"]),
                    "wrong_down_pct": pct(part["wrong_down"]),
                    "net_after_buffer_mean": float(part["net_after_buffer_ticks"].mean()),
                    "net_after_buffer_median": float(part["net_after_buffer_ticks"].median()),
                    "net_after_buffer_sum": float(part["net_after_buffer_ticks"].sum()),
                    "net_positive_pct": pct(part["net_positive_after_buffer"]),
                    "score_median": float(part["score_buy"].median()),
                    "cost_median": float(part["visible_entry_cost_ticks"].median()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main():
    df = load_dataset()

    feature_contract = build_feature_contract(df)
    feature_contract.to_csv(OUT_DIR / "feature_contract.csv", index=False)

    fs_health = feature_set_health(feature_contract)
    fs_health.to_csv(OUT_DIR / "feature_set_health.csv", index=False)

    split_summary = group_summary(df, ["terminal_split"])
    split_summary.to_csv(OUT_DIR / "split_summary.csv", index=False)

    target_by_temporality = group_summary(df, ["terminal_split", "temporality"])
    target_by_temporality.to_csv(OUT_DIR / "target_by_split_temporality.csv", index=False)

    target_by_phase_cost = group_summary(df, ["terminal_split", "phase_bucket", "cost_bucket"])
    target_by_phase_cost.to_csv(OUT_DIR / "target_by_split_phase_cost.csv", index=False)

    strict_segments = summarize_actions(
        V02_DIR / "actions__full_no_micro_highconv__strict_score_cost_plus_0p5tick__cooldown_0s.csv",
        "strict_score_cost_plus_0p5tick_cd0",
    )
    cross_segments = summarize_actions(
        V02_DIR / "actions__full_no_micro_highconv__cross_visible__cooldown_0s.csv",
        "cross_visible_cd0",
    )
    action_segments = pd.concat([strict_segments, cross_segments], ignore_index=True)
    action_segments.to_csv(OUT_DIR / "v02_action_segments_for_v03.csv", index=False)

    manifest = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "source_cache": str(CACHE_DIR),
        "v02_source": str(V02_DIR),
        "out_dir": str(OUT_DIR),
        "rows": int(len(df)),
        "market_frames": int(df.drop_duplicates(FRAME_COLS).shape[0]),
        "columns_loaded": int(len(df.columns)),
        "feature_sets": FEATURE_SETS,
        "primary_target": "target_3c_16s_1tick",
        "primary_target_continuous": "delta_ticks_16s",
        "evaluation_unit": "session_id + condition_id + time_index_ns",
        "training_unit": "session_id + token_id + time_index_ns",
        "recommended_policy_from_v02": "full_no_micro_highconv",
        "outputs": {
            "feature_contract": "feature_contract.csv",
            "feature_set_health": "feature_set_health.csv",
            "split_summary": "split_summary.csv",
            "target_by_split_temporality": "target_by_split_temporality.csv",
            "target_by_split_phase_cost": "target_by_split_phase_cost.csv",
            "v02_action_segments_for_v03": "v02_action_segments_for_v03.csv",
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(make_json_safe(manifest), indent=2, allow_nan=False), encoding="utf-8")

    print("V0.3 DATASET CONTRACT")
    print(f"rows={len(df):,} market_frames={manifest['market_frames']:,}")
    print("\nSPLITS")
    print(split_summary.round(4).to_string(index=False))
    print("\nFEATURE SET HEALTH")
    print(fs_health[["feature_set", "n_features", "median_missing_pct", "max_missing_pct"]].round(4).to_string(index=False))
    print("\nACTION SEGMENTS WRITTEN")
    print(action_segments.groupby("scenario_label").size().rename("segments").reset_index().to_string(index=False))
if __name__ == "__main__":
    main()
