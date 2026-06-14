from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

IN_DIR = Path("data/experiments/baseline_laware_v01b_feature_runner")
OUT_DIR = Path("data/experiments/baseline_laware_v02_trade_diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACTION_FILES = {
    "v03_conservative_plus_micro": {
        "wide": IN_DIR / "test_actions__v03_conservative_plus_micro__selected_L2_visible.csv",
        "low_spread": IN_DIR / "test_actions__v03_conservative_plus_micro__selected_L2_low_spread_buffer_0p5.csv",
        "high_score_low_spread": IN_DIR / "test_actions__v03_conservative_plus_micro__score_ge_0p40_L2_low_spread_buffer_0p5.csv",
    },
    "v03_pm_perp_control": {
        "wide": IN_DIR / "test_actions__v03_pm_perp_control__selected_L2_visible.csv",
        "low_spread": IN_DIR / "test_actions__v03_pm_perp_control__selected_L2_low_spread_buffer_0p5.csv",
        "high_score_low_spread": IN_DIR / "test_actions__v03_pm_perp_control__score_ge_0p40_L2_low_spread_buffer_0p5.csv",
    },
}

NET = "net_buffer_0p5_ticks_L2_H16"
DELTA = "delta_ticks_L2_H16"
SCORE = "score_buy"
ENTRY_SPREAD = "entry_spread_ticks_L2"
ENTRY_COST = "entry_visible_cost_ticks_L2"

SCORE_BINS = [-np.inf, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, np.inf]
SCORE_LABELS = [
    "<0",
    "0-0.1",
    "0.1-0.2",
    "0.2-0.3",
    "0.3-0.4",
    "0.4-0.5",
    "0.5-0.55",
    "0.55-0.6",
    "0.6-0.65",
    "0.65-0.7",
    ">=0.7",
]


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    cum = np.r_[0.0, np.cumsum(values)]
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def summarize(part: pd.DataFrame) -> dict:
    if part.empty:
        return {
            "actions": 0,
            "hit_up_pct": np.nan,
            "wrong_down_pct": np.nan,
            "flat_pct": np.nan,
            "net_mean": np.nan,
            "net_sum": 0.0,
            "net_positive_pct": np.nan,
            "delta_mean": np.nan,
            "entry_spread_median": np.nan,
            "entry_cost_median": np.nan,
            "score_min": np.nan,
            "score_median": np.nan,
            "score_max": np.nan,
            "max_drawdown_ticks": np.nan,
        }
    ordered = part.sort_values(["entry_time_index_ns_L2", "condition_id"])
    return {
        "actions": int(len(part)),
        "hit_up_pct": part["hit_up"].mean() * 100,
        "wrong_down_pct": part["wrong_down"].mean() * 100,
        "flat_pct": part["flat"].mean() * 100,
        "net_mean": part[NET].mean(),
        "net_median": part[NET].median(),
        "net_sum": part[NET].sum(),
        "net_positive_pct": part[NET].gt(0).mean() * 100,
        "delta_mean": part[DELTA].mean(),
        "entry_spread_median": part[ENTRY_SPREAD].median(),
        "entry_cost_median": part[ENTRY_COST].median(),
        "score_min": part[SCORE].min(),
        "score_median": part[SCORE].median(),
        "score_max": part[SCORE].max(),
        "max_drawdown_ticks": max_drawdown(ordered[NET].to_numpy(dtype=float)),
    }


def non_overlapping(actions: pd.DataFrame, horizon_ns: int = 16_000_000_000) -> pd.DataFrame:
    kept = []
    last_by_condition: dict[str, int] = {}
    ordered = actions.sort_values(["condition_id", "entry_time_index_ns_L2"])
    for idx, row in ordered.iterrows():
        condition = str(row["condition_id"])
        t = int(row["entry_time_index_ns_L2"])
        last = last_by_condition.get(condition)
        if last is None or t >= last + horizon_ns:
            kept.append(idx)
            last_by_condition[condition] = t
    return actions.loc[kept].copy()


def bucket_summary(df: pd.DataFrame, feature_set: str, scenario: str) -> pd.DataFrame:
    frame = df.copy()
    frame["score_bucket"] = pd.cut(frame[SCORE], bins=SCORE_BINS, labels=SCORE_LABELS, right=False)
    rows = []
    for bucket, part in frame.groupby("score_bucket", observed=True):
        row = {
            "feature_set": feature_set,
            "scenario": scenario,
            "bucket_type": "score",
            "bucket": str(bucket),
        }
        row.update(summarize(part))
        rows.append(row)
    return pd.DataFrame(rows)


def segment_summary(df: pd.DataFrame, feature_set: str, scenario: str) -> pd.DataFrame:
    rows = []
    for segment in ["temporality", "phase_bucket", "session_day"]:
        for value, part in df.groupby(segment, observed=True, dropna=False):
            row = {
                "feature_set": feature_set,
                "scenario": scenario,
                "segment": segment,
                "segment_value": str(value),
            }
            row.update(summarize(part))
            rows.append(row)
    return pd.DataFrame(rows)


def combo_summary(df: pd.DataFrame, feature_set: str, scenario: str) -> pd.DataFrame:
    frame = df.copy()
    frame["score_bucket"] = pd.cut(frame[SCORE], bins=SCORE_BINS, labels=SCORE_LABELS, right=False)
    rows = []
    for keys, part in frame.groupby(["temporality", "score_bucket"], observed=True, dropna=False):
        temporality, score_bucket = keys
        row = {
            "feature_set": feature_set,
            "scenario": scenario,
            "temporality": str(temporality),
            "score_bucket": str(score_bucket),
        }
        row.update(summarize(part))
        rows.append(row)
    return pd.DataFrame(rows)


def candidate_rules(df: pd.DataFrame, feature_set: str, scenario: str) -> pd.DataFrame:
    rows = []
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    temporalities = ["all"] + sorted(str(x) for x in df["temporality"].dropna().unique())
    for temporality in temporalities:
        base = df if temporality == "all" else df[df["temporality"].astype(str).eq(temporality)]
        for threshold in thresholds:
            part = base[base[SCORE].ge(threshold)].copy()
            for min_actions in [30, 100]:
                row = {
                    "feature_set": feature_set,
                    "scenario": scenario,
                    "temporality_filter": temporality,
                    "score_threshold": threshold,
                    "min_actions_gate": min_actions,
                    "passes_gate": len(part) >= min_actions,
                }
                row.update(summarize(part))
                no = non_overlapping(part)
                no_summary = summarize(no)
                for key, value in no_summary.items():
                    row[f"nonoverlap_{key}"] = value
                rows.append(row)
    out = pd.DataFrame(rows)
    out["rule_score"] = (
        out["net_mean"].fillna(-999)
        + 0.01 * out["net_positive_pct"].fillna(0)
        - 0.03 * out["wrong_down_pct"].fillna(100)
        + 0.001 * np.minimum(out["actions"].fillna(0), 500)
    )
    return out.sort_values(["passes_gate", "net_mean", "rule_score"], ascending=[False, False, False])


def load_actions(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    t0 = time.time()
    bucket_frames = []
    segment_frames = []
    combo_frames = []
    rule_frames = []
    scenario_frames = []

    for feature_set, scenarios in ACTION_FILES.items():
        for scenario, path in scenarios.items():
            df = load_actions(path)
            row = {"feature_set": feature_set, "scenario": scenario}
            row.update(summarize(df))
            row["source_file"] = str(path)
            scenario_frames.append(row)
            bucket_frames.append(bucket_summary(df, feature_set, scenario))
            segment_frames.append(segment_summary(df, feature_set, scenario))
            combo_frames.append(combo_summary(df, feature_set, scenario))
            rule_frames.append(candidate_rules(df, feature_set, scenario))

    scenario_summary = pd.DataFrame(scenario_frames)
    bucket_df = pd.concat(bucket_frames, ignore_index=True)
    segment_df = pd.concat(segment_frames, ignore_index=True)
    combo_df = pd.concat(combo_frames, ignore_index=True)
    rules_df = pd.concat(rule_frames, ignore_index=True)

    promising_rules = rules_df[
        rules_df["passes_gate"]
        & rules_df["net_mean"].gt(0)
        & rules_df["wrong_down_pct"].le(15)
    ].copy()

    scenario_summary.to_csv(OUT_DIR / "scenario_summary.csv", index=False)
    bucket_df.to_csv(OUT_DIR / "score_bucket_summary.csv", index=False)
    segment_df.to_csv(OUT_DIR / "segment_summary.csv", index=False)
    combo_df.to_csv(OUT_DIR / "temporality_score_bucket_summary.csv", index=False)
    rules_df.to_csv(OUT_DIR / "candidate_rule_grid.csv", index=False)
    promising_rules.to_csv(OUT_DIR / "promising_rules.csv", index=False)

    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "input_dir": str(IN_DIR),
        "out_dir": str(OUT_DIR),
        "outputs": {
            "scenario_summary": "scenario_summary.csv",
            "score_bucket_summary": "score_bucket_summary.csv",
            "segment_summary": "segment_summary.csv",
            "temporality_score_bucket_summary": "temporality_score_bucket_summary.csv",
            "candidate_rule_grid": "candidate_rule_grid.csv",
            "promising_rules": "promising_rules.csv",
        },
        "promising_rules": int(len(promising_rules)),
        "wall_seconds": float(time.time() - t0),
        "notes": [
            "Diagnostic only: reuses saved action CSVs from v0.1b; no model refit.",
            "Rows are selected market-frame actions, not the full prediction universe.",
            "A positive small segment is a hypothesis, not a deployable policy.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not promising_rules.empty:
        cols = [
            "feature_set",
            "scenario",
            "temporality_filter",
            "score_threshold",
            "min_actions_gate",
            "actions",
            "hit_up_pct",
            "wrong_down_pct",
            "net_mean",
            "net_sum",
            "nonoverlap_actions",
            "nonoverlap_net_mean",
        ]
        print(promising_rules[cols].head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
