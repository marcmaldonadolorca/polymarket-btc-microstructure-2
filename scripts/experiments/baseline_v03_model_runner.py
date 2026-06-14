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

CACHE_DIR = Path("data/experiments/baseline_v0_full_core_robustness/baseline_v0_core_h16_parquet")
CONTRACT_DIR = Path("data/experiments/baseline_v03_dataset_contract")
OUT_DIR = Path("data/experiments/baseline_v03_model_runner")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OUT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_DELTA_NS = 16_000_000_000
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
MODEL_KIND = "hgb_balanced_small"
FEATURE_SET_NAMES = [
    "v03_general_full_v0",
    "v03_conservative_no_micro",
    "v03_pm_perp_control",
]
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
]
TARGET_EVAL_COLS = [
    "target_3c_16s_1tick",
    "delta_ticks_16s",
    "visible_entry_cost_ticks",
    "spread_ticks",
    "tradability_status",
    "practical_dead_early",
    "full_book_ratio",
    "degraded_ratio",
]


def register_v03_feature_sets():
    for name in FEATURE_SET_NAMES:
        spec = contract.FEATURE_SETS[name]
        baseline_v0.FEATURE_SETS[name] = {"num": spec["num"], "cat": spec["cat"]}


def load_dataset():
    register_v03_feature_sets()
    feature_cols = sorted(
        {
            col
            for name in FEATURE_SET_NAMES
            for col in baseline_v0.FEATURE_SETS[name]["num"] + baseline_v0.FEATURE_SETS[name]["cat"]
        }
    )
    columns = sorted(set(KEY_COLS + TARGET_EVAL_COLS + feature_cols))
    df = pd.read_parquet(CACHE_DIR, columns=columns)
    df = baseline_v0.optimize_loaded_df(df)
    df["terminal_split"] = df["session_day"].astype(str).map(contract.assign_terminal_split).astype("category")
    df["cost_bucket"] = pd.cut(
        pd.to_numeric(df["visible_entry_cost_ticks"], errors="coerce"),
        bins=[-np.inf, 0.75, 1.25, 2.0, np.inf],
        labels=["low_<=0.75", "normal_0.75_1.25", "high_1.25_2", "very_high_>2"],
    )
    return df


def rows_between_days(data, start, end):
    day = data["session_day"].astype(str)
    return data[day.between(start, end)].copy()


def build_splits(df):
    train_initial = df[df["terminal_split"].eq("train_initial")].copy()
    validation_initial = df[df["terminal_split"].eq("validation_initial")].copy()
    test_terminal = df[df["terminal_split"].eq("test_terminal")].copy()
    pretest = df[df["terminal_split"].isin(["train_initial", "validation_initial"])].copy()
    pretest, purged_vs_test = baseline_v0.purge_train_conditions(pretest, test_terminal)
    return train_initial, validation_initial, pretest, test_terminal, purged_vs_test


def fit_and_score(train, eval_data, feature_set_name):
    fs = baseline_v0.FEATURE_SETS[feature_set_name]
    features = fs["num"] + fs["cat"]
    model = baseline_v0.build_model(MODEL_KIND, feature_set_name)
    t0 = time.time()
    model.fit(train[features], train["target_3c_16s_1tick"])
    fit_seconds = time.time() - t0

    proba = model.predict_proba(eval_data[features])
    pred = model.predict(eval_data[features])
    classes = list(model.named_steps["clf"].classes_)

    out = eval_data[KEY_COLS + TARGET_EVAL_COLS + ["cost_bucket"]].copy()
    out["pred"] = pred
    out["p_down"] = proba[:, classes.index("down")] if "down" in classes else 0.0
    out["p_flat"] = proba[:, classes.index("flat")] if "flat" in classes else 0.0
    out["p_up"] = proba[:, classes.index("up")] if "up" in classes else 0.0
    out["score_buy"] = out["p_up"] - out["p_down"]

    metrics = baseline_v0.token_metrics(
        eval_data["target_3c_16s_1tick"],
        out["pred"],
        out["score_buy"],
        eval_data["delta_ticks_16s"],
    )
    metrics["fit_seconds"] = fit_seconds
    return model, out, metrics


def best_token_per_frame(scored_df):
    keys = baseline_v0.complete_frame_keys(scored_df)
    complete = scored_df.merge(keys, on=FRAME_COLS, how="inner")
    idx = complete.groupby(FRAME_COLS, observed=True)["score_buy"].idxmax()
    chosen = complete.loc[idx].copy()
    chosen["hit_up"] = chosen["target_3c_16s_1tick"].eq("up")
    chosen["flat"] = chosen["target_3c_16s_1tick"].eq("flat")
    chosen["wrong_down"] = chosen["target_3c_16s_1tick"].eq("down")
    chosen["net_visible_ticks"] = chosen["delta_ticks_16s"] - chosen["visible_entry_cost_ticks"]
    return chosen


def max_drawdown(values):
    if len(values) == 0:
        return np.nan
    cum = np.r_[0.0, np.cumsum(values)]
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def summarize_actions(actions, complete_frames):
    if actions.empty:
        return {
            "complete_frames": int(complete_frames),
            "actions": 0,
            "coverage_pct": 0.0,
            "hit_up_pct": np.nan,
            "flat_pct": np.nan,
            "wrong_down_pct": np.nan,
            "delta_mean": np.nan,
            "delta_median": np.nan,
            "visible_cost_median": np.nan,
            "net_after_buffer_mean": np.nan,
            "net_after_buffer_median": np.nan,
            "net_after_buffer_sum": 0.0,
            "net_positive_pct": np.nan,
            "max_drawdown_ticks": np.nan,
            "score_median": np.nan,
        }
    net_col = "net_after_buffer_ticks"
    return {
        "complete_frames": int(complete_frames),
        "actions": int(len(actions)),
        "coverage_pct": len(actions) / complete_frames * 100 if complete_frames else np.nan,
        "hit_up_pct": actions["hit_up"].mean() * 100,
        "flat_pct": actions["flat"].mean() * 100,
        "wrong_down_pct": actions["wrong_down"].mean() * 100,
        "delta_mean": actions["delta_ticks_16s"].mean(),
        "delta_median": actions["delta_ticks_16s"].median(),
        "visible_cost_median": actions["visible_entry_cost_ticks"].median(),
        "net_after_buffer_mean": actions[net_col].mean(),
        "net_after_buffer_median": actions[net_col].median(),
        "net_after_buffer_sum": actions[net_col].sum(),
        "net_positive_pct": actions[net_col].gt(0).mean() * 100,
        "max_drawdown_ticks": max_drawdown(actions.sort_values(["time_index_ns", "condition_id"])[net_col].to_numpy(dtype=float)),
        "score_median": actions["score_buy"].median(),
    }


def apply_action_policy(chosen, scenario, min_score, extra_buffer_ticks=0.0, max_spread_ticks=None):
    actions = chosen[chosen["score_buy"].ge(min_score)].copy()
    if max_spread_ticks is not None:
        actions = actions[actions["spread_ticks"].le(max_spread_ticks)].copy()
    actions["scenario"] = scenario
    actions["min_score"] = float(min_score)
    actions["extra_buffer_ticks"] = float(extra_buffer_ticks)
    actions["max_spread_ticks"] = max_spread_ticks
    actions["net_after_buffer_ticks"] = (
        actions["delta_ticks_16s"] - actions["visible_entry_cost_ticks"] - float(extra_buffer_ticks)
    )
    actions["net_positive_after_buffer"] = actions["net_after_buffer_ticks"].gt(0)
    return actions


def selection_score(row):
    return (
        row["net_after_buffer_mean"]
        - 0.030 * row["wrong_down_pct"]
        + 0.010 * row["coverage_pct"]
        + 0.002 * row["net_positive_pct"]
    )


def threshold_curve(chosen, feature_set_name):
    rows = []
    complete_frames = len(chosen)
    for threshold in THRESHOLDS:
        actions = apply_action_policy(chosen, "validation_visible", threshold)
        row = summarize_actions(actions, complete_frames)
        row.update(
            {
                "feature_set": feature_set_name,
                "model_kind": MODEL_KIND,
                "threshold": threshold,
            }
        )
        row["selection_score"] = selection_score(row) if row["actions"] else -999.0
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(curve):
    candidates = curve[
        curve["actions"].ge(100)
        & curve["net_after_buffer_mean"].gt(0)
        & curve["wrong_down_pct"].le(20)
    ].copy()
    if candidates.empty:
        candidates = curve[curve["actions"].ge(30) & curve["net_after_buffer_mean"].gt(0)].copy()
    if candidates.empty:
        candidates = curve[curve["actions"].gt(0)].copy()
    return candidates.sort_values(["selection_score", "net_after_buffer_mean"], ascending=[False, False]).iloc[0]


def non_overlapping_actions(actions):
    kept = []
    last_by_condition = {}
    ordered = actions.sort_values(["condition_id", "time_index_ns"])
    for idx, row in ordered.iterrows():
        condition = row["condition_id"]
        t = int(row["time_index_ns"])
        last = last_by_condition.get(condition)
        if last is None or t >= last + TARGET_DELTA_NS:
            kept.append(idx)
            last_by_condition[condition] = t
    return actions.loc[kept].copy()


def segmented_summary(actions, complete_frames, keys, label):
    rows = []
    if actions.empty:
        return pd.DataFrame()
    for key_values, part in actions.groupby(keys, dropna=False, observed=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {"segment": "+".join(keys), **label}
        row.update({key: value for key, value in zip(keys, key_values)})
        row.update(summarize_actions(part, max(len(part), 1)))
        row["global_complete_frames"] = int(complete_frames)
        rows.append(row)
    return pd.DataFrame(rows)


def build_segment_tables(actions, complete_frames, label):
    keys_list = [
        ["temporality"],
        ["phase_bucket"],
        ["cost_bucket"],
        ["temporality", "phase_bucket"],
        ["temporality", "cost_bucket"],
    ]
    return pd.concat([segmented_summary(actions, complete_frames, keys, label) for keys in keys_list], ignore_index=True)


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


def save_model_artifact(model, feature_set, threshold, train_rows, eval_rows, fit_seconds):
    fs = baseline_v0.FEATURE_SETS[feature_set]
    features = fs["num"] + fs["cat"]
    artifact_stem = f"{MODEL_KIND}__{feature_set}__pretest_final"
    model_path = MODEL_DIR / f"{artifact_stem}.joblib"
    metadata_path = MODEL_DIR / f"{artifact_stem}.json"

    joblib.dump(model, model_path)
    metadata = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "model_kind": MODEL_KIND,
        "feature_set": feature_set,
        "stage": "pretest_final",
        "target": "target_3c_16s_1tick",
        "score": "score_buy = P(up) - P(down)",
        "selected_threshold": float(threshold),
        "train_rows": int(train_rows),
        "eval_rows": int(eval_rows),
        "fit_seconds": float(fit_seconds),
        "numeric_features": list(fs["num"]),
        "categorical_features": list(fs["cat"]),
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

    validation_rows = []
    token_rows = []
    threshold_choice_rows = []
    test_rows = []
    segment_frames = []
    model_artifacts = []

    print("V0.3 MODEL RUNNER")
    print(f"rows={len(df):,}; train_initial={len(train_initial):,}; validation={len(validation_initial):,}; test={len(test_terminal):,}")
    print(f"pretest_after_purge={len(pretest):,}; purged_vs_test={purged_vs_test:,}")

    selected_thresholds = {}
    for feature_set in FEATURE_SET_NAMES:
        print(f"\nVALIDATION FIT {feature_set}")
        _, scored_val, metrics = fit_and_score(train_initial, validation_initial, feature_set)
        metrics.update({"stage": "validation_threshold_selection", "model_kind": MODEL_KIND, "feature_set": feature_set})
        token_rows.append(metrics)
        chosen_val = best_token_per_frame(scored_val)
        curve = threshold_curve(chosen_val, feature_set)
        validation_rows.append(curve)
        choice = choose_threshold(curve)
        selected_thresholds[feature_set] = float(choice["threshold"])
        threshold_choice_rows.append(choice.to_dict())
        print(
            f"  selected_threshold={choice['threshold']:.2f}; "
            f"actions={int(choice['actions'])}; hit={choice['hit_up_pct']:.2f}; "
            f"wrong={choice['wrong_down_pct']:.2f}; net={choice['net_after_buffer_mean']:.3f}"
        )

    validation_curves = pd.concat(validation_rows, ignore_index=True)
    validation_curves.to_csv(OUT_DIR / "validation_threshold_curves.csv", index=False)
    threshold_choices = pd.DataFrame(threshold_choice_rows)
    threshold_choices.to_csv(OUT_DIR / "threshold_choices.csv", index=False)

    for feature_set in FEATURE_SET_NAMES:
        threshold = selected_thresholds[feature_set]
        print(f"\nFINAL FIT {feature_set} threshold={threshold:.2f}")
        model, scored_test, metrics = fit_and_score(pretest, test_terminal, feature_set)
        metrics.update({"stage": "test_final", "model_kind": MODEL_KIND, "feature_set": feature_set})
        token_rows.append(metrics)
        artifact = save_model_artifact(
            model,
            feature_set,
            threshold,
            train_rows=len(pretest),
            eval_rows=len(test_terminal),
            fit_seconds=metrics["fit_seconds"],
        )
        model_artifacts.append(artifact)

        chosen_test = best_token_per_frame(scored_test)
        complete_frames = len(chosen_test)
        scenarios = [
            {
                "scenario": "selected_visible",
                "min_score": threshold,
                "extra_buffer_ticks": 0.0,
                "max_spread_ticks": None,
            },
            {
                "scenario": "strict_score_cost_plus_0p5tick",
                "min_score": 0.60,
                "extra_buffer_ticks": 0.5,
                "max_spread_ticks": 1.0,
            },
        ]

        for scenario in scenarios:
            actions = apply_action_policy(chosen_test, **scenario)
            non_overlap = non_overlapping_actions(actions)
            summary = summarize_actions(actions, complete_frames)
            non_summary = summarize_actions(non_overlap, complete_frames)
            row = {
                "model_kind": MODEL_KIND,
                "feature_set": feature_set,
                "scenario": scenario["scenario"],
                "selected_threshold": threshold,
                "scenario_min_score": scenario["min_score"],
                "extra_buffer_ticks": scenario["extra_buffer_ticks"],
                "max_spread_ticks": scenario["max_spread_ticks"],
            }
            row.update(summary)
            for key, value in non_summary.items():
                if key in {"complete_frames"}:
                    continue
                row[f"nonoverlap_{key}"] = value
            test_rows.append(row)

            path = OUT_DIR / f"test_actions__{feature_set}__{scenario['scenario']}.csv"
            actions.to_csv(path, index=False)
            label = {
                "model_kind": MODEL_KIND,
                "feature_set": feature_set,
                "scenario": scenario["scenario"],
            }
            segment_frames.append(build_segment_tables(actions, complete_frames, label))
            print(
                f"  {scenario['scenario']}: actions={summary['actions']}; "
                f"hit={summary['hit_up_pct']:.2f}; wrong={summary['wrong_down_pct']:.2f}; "
                f"net={summary['net_after_buffer_mean']:.3f}"
            )

    token_metrics = pd.DataFrame(token_rows)
    token_metrics.to_csv(OUT_DIR / "token_metrics.csv", index=False)

    test_summary = pd.DataFrame(test_rows)
    test_summary = test_summary.sort_values(["net_after_buffer_mean", "actions"], ascending=[False, False])
    test_summary.to_csv(OUT_DIR / "test_action_summary.csv", index=False)

    if segment_frames:
        segments = pd.concat(segment_frames, ignore_index=True)
    else:
        segments = pd.DataFrame()
    segments.to_csv(OUT_DIR / "test_action_segments.csv", index=False)

    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "out_dir": str(OUT_DIR),
        "source_cache": str(CACHE_DIR),
        "contract_dir": str(CONTRACT_DIR),
        "rows": int(len(df)),
        "train_initial_rows": int(len(train_initial)),
        "validation_initial_rows": int(len(validation_initial)),
        "pretest_rows_after_purge": int(len(pretest)),
        "test_terminal_rows": int(len(test_terminal)),
        "purged_vs_test_rows": int(purged_vs_test),
        "model_kind": MODEL_KIND,
        "feature_sets": FEATURE_SET_NAMES,
        "thresholds": selected_thresholds,
        "model_artifacts": make_json_safe(model_artifacts),
        "top_test_scenarios": make_json_safe(test_summary.head(10).to_dict(orient="records")),
        "wall_seconds": float(time.time() - t0),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(make_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    print("\nTEST SUMMARY")
    cols = [
        "feature_set",
        "scenario",
        "actions",
        "coverage_pct",
        "hit_up_pct",
        "wrong_down_pct",
        "net_after_buffer_mean",
        "net_after_buffer_sum",
        "nonoverlap_actions",
        "nonoverlap_hit_up_pct",
        "nonoverlap_wrong_down_pct",
        "nonoverlap_net_after_buffer_mean",
    ]
    print(test_summary[cols].round(4).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
