from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASELINE_V0_SPEC = importlib.util.spec_from_file_location(
    "baseline_v0",
    "scripts/experiments/baseline_v0_full_core_robustness.py",
)
baseline_v0 = importlib.util.module_from_spec(BASELINE_V0_SPEC)
BASELINE_V0_SPEC.loader.exec_module(baseline_v0)

CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "baseline_v03_contract",
    "scripts/experiments/baseline_v03_dataset_contract.py",
)
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
CONTRACT_SPEC.loader.exec_module(contract)

DATA_DIR = Path("data/experiments/baseline_laware_v01_dataset_contract")
DATASET = DATA_DIR / "baseline_laware_v01_core_L2.parquet"
OUT_DIR = Path("data/experiments/baseline_laware_v01_model_runner")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_KIND = "hgb_balanced_small"
FEATURE_SET = "v03_conservative_no_micro"
TARGET = "target_3c_L2_H16_1tick"
DELTA = "delta_ticks_L2_H16"
NET_VISIBLE = "net_visible_ticks_L2_H16"
NET_BUFFER = "net_buffer_0p5_ticks_L2_H16"
ENTRY_SPREAD = "entry_spread_ticks_L2"
ENTRY_COST = "entry_visible_cost_ticks_L2"
ENTRY_TIME = "entry_time_index_ns_L2"
TARGET_DELTA_NS = 16_000_000_000
THRESHOLDS = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]
FLOAT_TOL = 1e-9

FRAME_COLS = ["session_id", "condition_id", "time_index_ns"]
KEY_COLS = [
    "session_id",
    "session_day",
    "time_index_ns",
    "time_index_utc",
    "market_id",
    "token_id",
    "condition_id",
    "temporality",
    "window_phase",
    "phase_bucket",
    "outcome_sign",
    "terminal_split",
]
EVAL_COLS = [
    TARGET,
    DELTA,
    NET_VISIBLE,
    NET_BUFFER,
    ENTRY_COST,
    ENTRY_SPREAD,
    ENTRY_TIME,
    "entry_low_spread_L2",
    "entry_tradability_status_L2",
    "label_supported_L2",
    "target_3c_L4_H16_1tick",
    "delta_ticks_L4_H16",
    "net_buffer_0p5_ticks_L4_H16",
]


def make_json_safe(value):
    if isinstance(value, dict):
        return {key: make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def register_feature_set() -> list[str]:
    spec = contract.FEATURE_SETS[FEATURE_SET]
    baseline_v0.FEATURE_SETS[FEATURE_SET] = {"num": spec["num"], "cat": spec["cat"]}
    return spec["num"] + spec["cat"]


def load_dataset() -> pd.DataFrame:
    features = register_feature_set()
    columns = sorted(set(KEY_COLS + EVAL_COLS + features))
    df = pd.read_parquet(DATASET, columns=columns)
    return baseline_v0.optimize_loaded_df(df)


def build_splits(df: pd.DataFrame):
    train_initial = df[df["terminal_split"].eq("train_initial")].copy()
    validation_initial = df[df["terminal_split"].eq("validation_initial")].copy()
    test_terminal = df[df["terminal_split"].eq("test_terminal")].copy()
    pretest = df[df["terminal_split"].isin(["train_initial", "validation_initial"])].copy()
    pretest, purged_vs_test = baseline_v0.purge_train_conditions(pretest, test_terminal)
    return train_initial, validation_initial, pretest, test_terminal, purged_vs_test


def fit_and_score(train: pd.DataFrame, eval_data: pd.DataFrame):
    features = baseline_v0.FEATURE_SETS[FEATURE_SET]["num"] + baseline_v0.FEATURE_SETS[FEATURE_SET]["cat"]
    model = baseline_v0.build_model(MODEL_KIND, FEATURE_SET)
    t0 = time.time()
    model.fit(train[features], train[TARGET])
    fit_seconds = time.time() - t0

    proba = model.predict_proba(eval_data[features])
    pred = model.predict(eval_data[features])
    classes = list(model.named_steps["clf"].classes_)

    out = eval_data[KEY_COLS + EVAL_COLS].copy()
    out["pred"] = pred
    out["p_down"] = proba[:, classes.index("down")] if "down" in classes else 0.0
    out["p_flat"] = proba[:, classes.index("flat")] if "flat" in classes else 0.0
    out["p_up"] = proba[:, classes.index("up")] if "up" in classes else 0.0
    out["score_buy"] = out["p_up"] - out["p_down"]

    metrics = baseline_v0.token_metrics(eval_data[TARGET], pred, out["score_buy"], eval_data[DELTA])
    metrics["fit_seconds"] = fit_seconds
    return model, out, metrics


def best_token_per_frame(scored: pd.DataFrame) -> pd.DataFrame:
    keys = baseline_v0.complete_frame_keys(scored)
    complete = scored.merge(keys, on=FRAME_COLS, how="inner")
    idx = complete.groupby(FRAME_COLS, observed=True)["score_buy"].idxmax()
    chosen = complete.loc[idx].copy()
    chosen["hit_up"] = chosen[TARGET].eq("up")
    chosen["flat"] = chosen[TARGET].eq("flat")
    chosen["wrong_down"] = chosen[TARGET].eq("down")
    chosen["l4_hit_up"] = chosen["target_3c_L4_H16_1tick"].eq("up")
    chosen["l4_wrong_down"] = chosen["target_3c_L4_H16_1tick"].eq("down")
    return chosen


def max_drawdown(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    cum = np.r_[0.0, np.cumsum(values)]
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def summarize_actions(actions: pd.DataFrame, complete_frames: int, net_col: str) -> dict:
    if actions.empty:
        return {
            "complete_frames": int(complete_frames),
            "actions": 0,
            "coverage_pct": 0.0,
            "hit_up_pct": np.nan,
            "flat_pct": np.nan,
            "wrong_down_pct": np.nan,
            "delta_mean": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_sum": 0.0,
            "net_positive_pct": np.nan,
            "entry_cost_median": np.nan,
            "entry_spread_median": np.nan,
            "score_median": np.nan,
            "max_drawdown_ticks": np.nan,
            "l4_hit_up_pct": np.nan,
            "l4_wrong_down_pct": np.nan,
        }
    ordered = actions.sort_values([ENTRY_TIME, "condition_id"])
    return {
        "complete_frames": int(complete_frames),
        "actions": int(len(actions)),
        "coverage_pct": len(actions) / complete_frames * 100 if complete_frames else np.nan,
        "hit_up_pct": actions["hit_up"].mean() * 100,
        "flat_pct": actions["flat"].mean() * 100,
        "wrong_down_pct": actions["wrong_down"].mean() * 100,
        "delta_mean": actions[DELTA].mean(),
        "net_mean": actions[net_col].mean(),
        "net_median": actions[net_col].median(),
        "net_sum": actions[net_col].sum(),
        "net_positive_pct": actions[net_col].gt(0).mean() * 100,
        "entry_cost_median": actions[ENTRY_COST].median(),
        "entry_spread_median": actions[ENTRY_SPREAD].median(),
        "score_median": actions["score_buy"].median(),
        "max_drawdown_ticks": max_drawdown(ordered[net_col].to_numpy(dtype=float)),
        "l4_hit_up_pct": actions["l4_hit_up"].mean() * 100,
        "l4_wrong_down_pct": actions["l4_wrong_down"].mean() * 100,
    }


def selection_score(row: pd.Series) -> float:
    return (
        row["net_mean"]
        - 0.030 * row["wrong_down_pct"]
        + 0.010 * row["coverage_pct"]
        + 0.002 * row["net_positive_pct"]
    )


def apply_action_policy(chosen: pd.DataFrame, scenario: str, min_score: float, net_col: str, max_entry_spread_ticks=None):
    actions = chosen[chosen["score_buy"].ge(float(min_score))].copy()
    if max_entry_spread_ticks is not None:
        actions = actions[actions[ENTRY_SPREAD].le(float(max_entry_spread_ticks) + FLOAT_TOL)].copy()
    actions["scenario"] = scenario
    actions["min_score"] = float(min_score)
    actions["net_col"] = net_col
    actions["max_entry_spread_ticks"] = max_entry_spread_ticks
    return actions


def threshold_curve(chosen: pd.DataFrame) -> pd.DataFrame:
    rows = []
    complete_frames = len(chosen)
    for threshold in THRESHOLDS:
        actions = apply_action_policy(chosen, "validation_L2_buffer_0p5", threshold, NET_BUFFER)
        row = summarize_actions(actions, complete_frames, NET_BUFFER)
        row.update({"threshold": threshold, "feature_set": FEATURE_SET, "model_kind": MODEL_KIND})
        row["selection_score"] = selection_score(pd.Series(row)) if row["actions"] else -999.0
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame) -> pd.Series:
    candidates = curve[
        curve["actions"].ge(100)
        & curve["net_mean"].gt(0)
        & curve["wrong_down_pct"].le(25)
    ].copy()
    if candidates.empty:
        candidates = curve[curve["actions"].ge(30) & curve["net_mean"].gt(0)].copy()
    if candidates.empty:
        candidates = curve[curve["actions"].gt(0)].copy()
    if candidates.empty:
        candidates = curve.copy()
    return candidates.sort_values(["selection_score", "net_mean"], ascending=[False, False]).iloc[0]


def non_overlapping_actions(actions: pd.DataFrame) -> pd.DataFrame:
    kept = []
    last_by_condition = {}
    ordered = actions.sort_values(["condition_id", ENTRY_TIME])
    for idx, row in ordered.iterrows():
        condition = row["condition_id"]
        t = int(row[ENTRY_TIME])
        last = last_by_condition.get(condition)
        if last is None or t >= last + TARGET_DELTA_NS:
            kept.append(idx)
            last_by_condition[condition] = t
    return actions.loc[kept].copy()


def build_segments(actions: pd.DataFrame, complete_frames: int, scenario: str, net_col: str) -> pd.DataFrame:
    rows = []
    for segment_col in ["session_day", "temporality", "phase_bucket"]:
        for value, part in actions.groupby(segment_col, dropna=False, observed=True):
            row = {
                "feature_set": FEATURE_SET,
                "scenario": scenario,
                "segment": segment_col,
                "segment_value": str(value),
            }
            row.update(summarize_actions(part, max(len(part), 1), net_col))
            row["global_complete_frames"] = complete_frames
            rows.append(row)
    return pd.DataFrame(rows)


def save_model(model, threshold: float, train_rows: int, eval_rows: int, fit_seconds: float) -> dict:
    features = baseline_v0.FEATURE_SETS[FEATURE_SET]["num"] + baseline_v0.FEATURE_SETS[FEATURE_SET]["cat"]
    stem = f"{MODEL_KIND}__{FEATURE_SET}__L2_pretest_final"
    model_path = MODEL_DIR / f"{stem}.joblib"
    metadata_path = MODEL_DIR / f"{stem}.json"
    joblib.dump(model, model_path)
    metadata = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "model_kind": MODEL_KIND,
        "feature_set": FEATURE_SET,
        "stage": "L2_pretest_final",
        "target": TARGET,
        "continuous_target": DELTA,
        "score": "score_buy = P(up) - P(down)",
        "selected_threshold": float(threshold),
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "fit_seconds": float(fit_seconds),
        "features": list(features),
        "classes": list(model.named_steps["clf"].classes_),
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(make_json_safe(metadata), indent=2, allow_nan=False), encoding="utf-8")
    return {"model_path": str(model_path), "metadata_path": str(metadata_path), **metadata}


def main():
    t0 = time.time()
    df = load_dataset()
    train_initial, validation_initial, pretest, test_terminal, purged_vs_test = build_splits(df)

    print("BASELINE L-AWARE V0.1 MODEL RUNNER")
    print(f"rows={len(df):,}; train={len(train_initial):,}; validation={len(validation_initial):,}; test={len(test_terminal):,}")
    print(f"pretest_after_purge={len(pretest):,}; purged_vs_test={purged_vs_test:,}")

    print("\nVALIDATION FIT")
    _, scored_val, val_metrics = fit_and_score(train_initial, validation_initial)
    val_metrics.update({"stage": "validation_threshold_selection", "feature_set": FEATURE_SET, "model_kind": MODEL_KIND})
    chosen_val = best_token_per_frame(scored_val)
    curve = threshold_curve(chosen_val)
    choice = choose_threshold(curve)
    threshold = float(choice["threshold"])
    print(
        f"selected_threshold={threshold:.2f}; actions={int(choice['actions'])}; "
        f"hit={choice['hit_up_pct']:.2f}; wrong={choice['wrong_down_pct']:.2f}; net={choice['net_mean']:.3f}"
    )

    curve.to_csv(OUT_DIR / "validation_threshold_curve.csv", index=False)
    pd.DataFrame([choice.to_dict()]).to_csv(OUT_DIR / "threshold_choice.csv", index=False)

    print("\nFINAL FIT")
    model, scored_test, test_metrics = fit_and_score(pretest, test_terminal)
    test_metrics.update({"stage": "test_final", "feature_set": FEATURE_SET, "model_kind": MODEL_KIND})
    artifact = save_model(model, threshold, len(pretest), len(test_terminal), test_metrics["fit_seconds"])

    chosen_test = best_token_per_frame(scored_test)
    complete_frames = len(chosen_test)
    scenarios = [
        {
            "scenario": "selected_L2_visible",
            "min_score": threshold,
            "net_col": NET_VISIBLE,
            "max_entry_spread_ticks": None,
        },
        {
            "scenario": "selected_L2_buffer_0p5",
            "min_score": threshold,
            "net_col": NET_BUFFER,
            "max_entry_spread_ticks": None,
        },
        {
            "scenario": "selected_L2_low_spread_buffer_0p5",
            "min_score": threshold,
            "net_col": NET_BUFFER,
            "max_entry_spread_ticks": 1.0,
        },
        {
            "scenario": "score_ge_0p60_L2_low_spread_buffer_0p5",
            "min_score": 0.60,
            "net_col": NET_BUFFER,
            "max_entry_spread_ticks": 1.0,
        },
    ]

    test_rows = []
    segments = []
    for scenario in scenarios:
        actions = apply_action_policy(chosen_test, **scenario)
        non_overlap = non_overlapping_actions(actions)
        row = {
            "feature_set": FEATURE_SET,
            "model_kind": MODEL_KIND,
            **scenario,
            "selected_threshold": threshold,
        }
        row.update(summarize_actions(actions, complete_frames, scenario["net_col"]))
        non = summarize_actions(non_overlap, complete_frames, scenario["net_col"])
        for key, value in non.items():
            if key == "complete_frames":
                continue
            row[f"nonoverlap_{key}"] = value
        test_rows.append(row)
        actions.to_csv(OUT_DIR / f"test_actions__{scenario['scenario']}.csv", index=False)
        segments.append(build_segments(actions, complete_frames, scenario["scenario"], scenario["net_col"]))
        print(
            f"{scenario['scenario']}: actions={row['actions']}; hit={row['hit_up_pct']:.2f}; "
            f"wrong={row['wrong_down_pct']:.2f}; net={row['net_mean']:.3f}; "
            f"L4_hit={row['l4_hit_up_pct']:.2f}; L4_wrong={row['l4_wrong_down_pct']:.2f}"
        )

    token_metrics = pd.DataFrame([val_metrics, test_metrics])
    token_metrics.to_csv(OUT_DIR / "token_metrics.csv", index=False)
    test_summary = pd.DataFrame(test_rows).sort_values(["net_mean", "actions"], ascending=[False, False])
    test_summary.to_csv(OUT_DIR / "test_action_summary.csv", index=False)
    pd.concat(segments, ignore_index=True).to_csv(OUT_DIR / "test_action_segments.csv", index=False)

    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "dataset": str(DATASET),
        "out_dir": str(OUT_DIR),
        "model_kind": MODEL_KIND,
        "feature_set": FEATURE_SET,
        "target": TARGET,
        "continuous_target": DELTA,
        "rows": int(len(df)),
        "train_initial_rows": int(len(train_initial)),
        "validation_initial_rows": int(len(validation_initial)),
        "pretest_rows_after_purge": int(len(pretest)),
        "test_terminal_rows": int(len(test_terminal)),
        "purged_vs_test_rows": int(purged_vs_test),
        "selected_threshold": threshold,
        "model_artifact": make_json_safe(artifact),
        "top_test_scenarios": make_json_safe(test_summary.head(10).to_dict(orient="records")),
        "wall_seconds": float(time.time() - t0),
        "important_limitations": [
            "Offline model only.",
            "L=2s is still a sensitivity scenario until real p95 execution latency is measured.",
            "No fill, queue, size or slippage model beyond visible cost + optional 0.5 tick buffer.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(make_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    print("\nTEST SUMMARY")
    print(
        test_summary[
            [
                "scenario",
                "actions",
                "coverage_pct",
                "hit_up_pct",
                "wrong_down_pct",
                "net_mean",
                "net_sum",
                "nonoverlap_actions",
                "nonoverlap_hit_up_pct",
                "nonoverlap_wrong_down_pct",
                "nonoverlap_net_mean",
                "l4_hit_up_pct",
                "l4_wrong_down_pct",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
