from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("data/experiments/baseline_v0_full_core_robustness/baseline_v0_core_h16_parquet")
FOLD_DIR = Path("data/experiments/baseline_v03c_policy_fold_validation")
TEST_DIR = Path("data/experiments/baseline_v03b_error_calibration")
OUT_DIR = Path("data/experiments/baseline_v03d_execution_stress")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLOAT_TOL = 1e-9

FOLDS = ["F1", "F2", "F3", "F4"]
SIGNAL_SCENARIO = "score_ge_0p65_visible"
TEST_SIGNAL_FILE = TEST_DIR / "actions__test__selected_visible.csv"

LATENCIES_SECONDS = [0, 2, 4, 8]
COOLDOWNS_SECONDS = [0, 16, 60, 120]
EXECUTION_VARIANTS = [
    {
        "execution": "cross_plus_0p5tick",
        "description": "Entrada retrasada con coste visible + 0.5 tick.",
        "extra_buffer_ticks": 0.5,
        "max_entry_spread_ticks": None,
        "max_entry_visible_cost_ticks": None,
    },
    {
        "execution": "low_spread_plus_0p5tick",
        "description": "Entrada retrasada solo si spread <= 1 tick + 0.5 tick.",
        "extra_buffer_ticks": 0.5,
        "max_entry_spread_ticks": 1.0,
        "max_entry_visible_cost_ticks": None,
    },
    {
        "execution": "low_spread_plus_1p0tick",
        "description": "Entrada retrasada solo si spread <= 1 tick + 1.0 tick.",
        "extra_buffer_ticks": 1.0,
        "max_entry_spread_ticks": 1.0,
        "max_entry_visible_cost_ticks": None,
    },
    {
        "execution": "low_spread_plus_2p0tick",
        "description": "Stress severo: spread <= 1 tick + 2.0 ticks.",
        "extra_buffer_ticks": 2.0,
        "max_entry_spread_ticks": 1.0,
        "max_entry_visible_cost_ticks": None,
    },
]
LIMIT_POLICIES = [
    {
        "limit_policy": "none",
        "max_actions_per_day": None,
        "max_actions_per_session": None,
    },
    {
        "limit_policy": "day50_session3",
        "max_actions_per_day": 50,
        "max_actions_per_session": 3,
    },
    {
        "limit_policy": "day20_session1",
        "max_actions_per_day": 20,
        "max_actions_per_session": 1,
    },
]

META_COLS = [
    "session_id",
    "token_id",
    "condition_id",
    "time_index_ns",
    "time_index_utc",
    "session_day",
    "temporality",
    "phase_bucket",
    "polymarket_mid",
    "future_mid_16s",
    "delta_ticks_16s",
    "visible_entry_cost_ticks",
    "spread_ticks",
    "tick_size",
    "tradability_status",
    "practical_dead_early",
    "full_book_ratio",
    "degraded_ratio",
]


def normalize_keys(df):
    out = df.copy()
    out["session_id"] = pd.to_numeric(out["session_id"], errors="raise").astype("int64")
    out["time_index_ns"] = pd.to_numeric(out["time_index_ns"], errors="raise").astype("int64")
    out["token_id"] = out["token_id"].astype(str)
    out["condition_id"] = out["condition_id"].astype(str)
    return out


def load_signals():
    frames = []
    for fold in FOLDS:
        path = FOLD_DIR / f"actions__{fold}__{SIGNAL_SCENARIO}.csv"
        df = normalize_keys(pd.read_csv(path))
        df["split"] = fold
        df["split_group"] = "fold"
        frames.append(df)
    test = normalize_keys(pd.read_csv(TEST_SIGNAL_FILE))
    test["split"] = "TEST"
    test["split_group"] = "test_terminal"
    frames.append(test)
    signals = pd.concat(frames, ignore_index=True)
    signals = signals.rename(
        columns={
            "time_index_ns": "signal_time_index_ns",
            "time_index_utc": "signal_time_index_utc",
            "session_day": "signal_session_day",
            "delta_ticks_16s": "signal_delta_ticks_16s",
            "visible_entry_cost_ticks": "signal_visible_entry_cost_ticks",
            "spread_ticks": "signal_spread_ticks",
            "phase_bucket": "signal_phase_bucket",
        }
    )
    keep = [
        "split",
        "split_group",
        "session_id",
        "condition_id",
        "token_id",
        "signal_time_index_ns",
        "signal_time_index_utc",
        "signal_session_day",
        "market_id",
        "temporality",
        "signal_phase_bucket",
        "outcome_sign",
        "score_buy",
        "p_up",
        "p_down",
        "p_flat",
        "signal_delta_ticks_16s",
        "signal_visible_entry_cost_ticks",
        "signal_spread_ticks",
    ]
    return signals[[col for col in keep if col in signals.columns]].copy()


def load_execution_meta():
    meta = pd.read_parquet(CACHE_DIR, columns=META_COLS)
    meta = normalize_keys(meta)
    rename = {
        "time_index_ns": "entry_time_index_ns",
        "time_index_utc": "entry_time_index_utc",
        "session_day": "entry_session_day",
        "phase_bucket": "entry_phase_bucket",
        "polymarket_mid": "entry_polymarket_mid",
        "future_mid_16s": "entry_future_mid_16s",
        "delta_ticks_16s": "entry_delta_ticks_16s",
        "visible_entry_cost_ticks": "entry_visible_cost_ticks",
        "spread_ticks": "entry_spread_ticks",
        "tradability_status": "entry_tradability_status",
        "practical_dead_early": "entry_practical_dead_early",
        "full_book_ratio": "entry_full_book_ratio",
        "degraded_ratio": "entry_degraded_ratio",
    }
    meta = meta.rename(columns=rename)
    return meta


def build_latency_frame(signals, meta, latency_seconds):
    out = signals.copy()
    out["latency_seconds"] = int(latency_seconds)
    out["entry_time_index_ns"] = out["signal_time_index_ns"] + int(latency_seconds * 1_000_000_000)
    merged = out.merge(
        meta,
        on=["session_id", "condition_id", "token_id", "entry_time_index_ns"],
        how="left",
        suffixes=("", "_meta"),
    )
    merged["entry_supported"] = merged["entry_delta_ticks_16s"].notna()
    merged["entry_target_3c_16s_1tick"] = np.select(
        [merged["entry_delta_ticks_16s"].ge(1.0), merged["entry_delta_ticks_16s"].le(-1.0)],
        ["up", "down"],
        default="flat",
    )
    merged["hit_up"] = merged["entry_target_3c_16s_1tick"].eq("up")
    merged["flat"] = merged["entry_target_3c_16s_1tick"].eq("flat")
    merged["wrong_down"] = merged["entry_target_3c_16s_1tick"].eq("down")
    return merged


def apply_execution_filter(frame, variant):
    out = frame[frame["entry_supported"]].copy()
    if variant["max_entry_spread_ticks"] is not None:
        out = out[out["entry_spread_ticks"].le(variant["max_entry_spread_ticks"] + FLOAT_TOL)].copy()
    if variant["max_entry_visible_cost_ticks"] is not None:
        out = out[out["entry_visible_cost_ticks"].le(variant["max_entry_visible_cost_ticks"] + FLOAT_TOL)].copy()
    out["execution"] = variant["execution"]
    out["execution_description"] = variant["description"]
    out["extra_buffer_ticks"] = float(variant["extra_buffer_ticks"])
    out["max_entry_spread_ticks"] = variant["max_entry_spread_ticks"]
    out["max_entry_visible_cost_ticks"] = variant["max_entry_visible_cost_ticks"]
    out["net_execution_ticks"] = out["entry_delta_ticks_16s"] - out["entry_visible_cost_ticks"] - out["extra_buffer_ticks"]
    out["net_positive"] = out["net_execution_ticks"].gt(0)
    return out


def apply_cooldown(frame, cooldown_seconds):
    if frame.empty or cooldown_seconds <= 0:
        return frame.sort_values(["split", "entry_time_index_ns", "condition_id"]).copy()
    rows = []
    cooldown_ns = int(cooldown_seconds * 1_000_000_000)
    for split, part in frame.groupby("split", sort=False):
        last_by_condition = {}
        ordered = part.sort_values(["entry_time_index_ns", "condition_id"])
        keep = []
        for idx, row in ordered.iterrows():
            condition = row["condition_id"]
            t = int(row["entry_time_index_ns"])
            last = last_by_condition.get(condition)
            if last is None or t >= last + cooldown_ns:
                keep.append(idx)
                last_by_condition[condition] = t
        rows.append(ordered.loc[keep])
    return pd.concat(rows, ignore_index=True) if rows else frame.iloc[0:0].copy()


def apply_limits(frame, limit_policy):
    out = frame.sort_values(["split", "entry_time_index_ns", "condition_id"]).copy()
    max_day = limit_policy["max_actions_per_day"]
    max_session = limit_policy["max_actions_per_session"]
    if max_session is not None:
        out["_session_rank"] = out.groupby(["split", "session_id"], observed=True).cumcount() + 1
        out = out[out["_session_rank"].le(max_session)].copy()
    if max_day is not None:
        out["_day_rank"] = out.groupby(["split", "entry_session_day"], observed=True).cumcount() + 1
        out = out[out["_day_rank"].le(max_day)].copy()
    out["limit_policy"] = limit_policy["limit_policy"]
    out["max_actions_per_day"] = max_day
    out["max_actions_per_session"] = max_session
    return out.drop(columns=[col for col in ["_session_rank", "_day_rank"] if col in out.columns])


def max_drawdown(values):
    if len(values) == 0:
        return np.nan
    cum = np.r_[0.0, np.cumsum(values)]
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def summarize(frame, signal_count):
    if frame.empty:
        return {
            "signals": int(signal_count),
            "actions": 0,
            "retention_pct": 0.0,
            "hit_up_pct": np.nan,
            "flat_pct": np.nan,
            "wrong_down_pct": np.nan,
            "net_mean": np.nan,
            "net_median": np.nan,
            "net_sum": 0.0,
            "net_positive_pct": np.nan,
            "max_drawdown_ticks": np.nan,
            "score_median": np.nan,
            "entry_cost_median": np.nan,
            "entry_spread_median": np.nan,
            "entry_delta_median": np.nan,
        }
    ordered = frame.sort_values(["split", "entry_time_index_ns", "condition_id"])
    return {
        "signals": int(signal_count),
        "actions": int(len(frame)),
        "retention_pct": len(frame) / signal_count * 100 if signal_count else np.nan,
        "hit_up_pct": frame["hit_up"].mean() * 100,
        "flat_pct": frame["flat"].mean() * 100,
        "wrong_down_pct": frame["wrong_down"].mean() * 100,
        "net_mean": frame["net_execution_ticks"].mean(),
        "net_median": frame["net_execution_ticks"].median(),
        "net_sum": frame["net_execution_ticks"].sum(),
        "net_positive_pct": frame["net_positive"].mean() * 100,
        "max_drawdown_ticks": max_drawdown(ordered["net_execution_ticks"].to_numpy(dtype=float)),
        "score_median": frame["score_buy"].median(),
        "entry_cost_median": frame["entry_visible_cost_ticks"].median(),
        "entry_spread_median": frame["entry_spread_ticks"].median(),
        "entry_delta_median": frame["entry_delta_ticks_16s"].median(),
    }


def scenario_id(latency_seconds, execution, cooldown_seconds, limit_policy):
    return f"lat{latency_seconds}s__{execution}__cd{cooldown_seconds}s__{limit_policy}"


def run_scenarios(signals, meta):
    summaries = []
    split_summaries = []
    daily_summaries = []
    action_frames = {}
    support_rows = []

    signal_count_total = len(signals)
    signal_counts_by_split = signals.groupby("split").size().to_dict()

    for latency_seconds in LATENCIES_SECONDS:
        delayed = build_latency_frame(signals, meta, latency_seconds)
        support_rows.append(
            {
                "latency_seconds": latency_seconds,
                "signals": int(len(delayed)),
                "entry_supported": int(delayed["entry_supported"].sum()),
                "entry_supported_pct": float(delayed["entry_supported"].mean() * 100),
            }
        )
        for variant in EXECUTION_VARIANTS:
            filtered = apply_execution_filter(delayed, variant)
            for cooldown_seconds in COOLDOWNS_SECONDS:
                cooled = apply_cooldown(filtered, cooldown_seconds)
                for limit_policy in LIMIT_POLICIES:
                    executed = apply_limits(cooled, limit_policy)
                    sid = scenario_id(
                        latency_seconds,
                        variant["execution"],
                        cooldown_seconds,
                        limit_policy["limit_policy"],
                    )
                    executed = executed.copy()
                    executed["scenario_id"] = sid
                    executed["cooldown_seconds"] = cooldown_seconds
                    executed["latency_seconds"] = latency_seconds
                    row = {
                        "scenario_id": sid,
                        "latency_seconds": latency_seconds,
                        "execution": variant["execution"],
                        "execution_description": variant["description"],
                        "cooldown_seconds": cooldown_seconds,
                        "limit_policy": limit_policy["limit_policy"],
                        "extra_buffer_ticks": variant["extra_buffer_ticks"],
                        "max_entry_spread_ticks": variant["max_entry_spread_ticks"],
                        "max_actions_per_day": limit_policy["max_actions_per_day"],
                        "max_actions_per_session": limit_policy["max_actions_per_session"],
                    }
                    row.update(summarize(executed, signal_count_total))
                    row["positive_splits"] = int(
                        executed.groupby("split")["net_execution_ticks"].sum().gt(0).sum()
                    ) if len(executed) else 0
                    row["splits_with_actions"] = int(executed["split"].nunique()) if len(executed) else 0
                    summaries.append(row)

                    for split, part in executed.groupby("split", observed=True):
                        srow = {
                            "scenario_id": sid,
                            "split": split,
                            "split_group": part["split_group"].iloc[0],
                            "latency_seconds": latency_seconds,
                            "execution": variant["execution"],
                            "cooldown_seconds": cooldown_seconds,
                            "limit_policy": limit_policy["limit_policy"],
                        }
                        srow.update(summarize(part, signal_counts_by_split.get(split, len(part))))
                        split_summaries.append(srow)

                    for (split, day), part in executed.groupby(["split", "entry_session_day"], observed=True):
                        drow = {
                            "scenario_id": sid,
                            "split": split,
                            "entry_session_day": day,
                            "latency_seconds": latency_seconds,
                            "execution": variant["execution"],
                            "cooldown_seconds": cooldown_seconds,
                            "limit_policy": limit_policy["limit_policy"],
                        }
                        drow.update(summarize(part, max(len(part), 1)))
                        daily_summaries.append(drow)

                    # Keep action files for a small set of interpretable scenarios.
                    if (
                        variant["execution"] in {"low_spread_plus_0p5tick", "low_spread_plus_1p0tick"}
                        and latency_seconds in {0, 2, 4}
                        and cooldown_seconds in {16, 60}
                        and limit_policy["limit_policy"] in {"none", "day50_session3"}
                    ):
                        action_frames[sid] = executed

    return (
        pd.DataFrame(summaries),
        pd.DataFrame(split_summaries),
        pd.DataFrame(daily_summaries),
        pd.DataFrame(support_rows),
        action_frames,
    )


def recommend(summary):
    candidates = summary[
        summary["actions"].ge(100)
        & summary["net_mean"].gt(0)
        & summary["positive_splits"].ge(4)
        & summary["wrong_down_pct"].le(12)
    ].copy()
    if candidates.empty:
        candidates = summary[summary["actions"].ge(50) & summary["net_mean"].gt(0)].copy()
    candidates["decision_score"] = (
        candidates["net_mean"]
        + 0.001 * candidates["actions"]
        - 0.04 * candidates["wrong_down_pct"]
        + 0.004 * candidates["positive_splits"]
        + 0.001 * candidates["max_drawdown_ticks"].fillna(0)
    )
    return candidates.sort_values(["decision_score", "net_sum"], ascending=[False, False]).head(20)


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
    print("V0.3D EXECUTION STRESS")
    signals = load_signals()
    meta = load_execution_meta()
    print(f"signals={len(signals):,}; meta_rows={len(meta):,}")

    summary, split_summary, daily_summary, support, action_frames = run_scenarios(signals, meta)
    summary = summary.sort_values(["net_mean", "actions"], ascending=[False, False]).reset_index(drop=True)
    recommended = recommend(summary)

    summary.to_csv(OUT_DIR / "execution_stress_summary.csv", index=False)
    split_summary.to_csv(OUT_DIR / "execution_stress_by_split.csv", index=False)
    daily_summary.to_csv(OUT_DIR / "execution_stress_by_day.csv", index=False)
    support.to_csv(OUT_DIR / "latency_support.csv", index=False)
    recommended.to_csv(OUT_DIR / "recommended_execution_scenarios.csv", index=False)

    for sid in recommended["scenario_id"].head(5):
        if sid in action_frames:
            action_frames[sid].to_csv(OUT_DIR / f"actions__{sid}.csv", index=False)

    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "out_dir": str(OUT_DIR),
        "signals": int(len(signals)),
        "signal_scenario": SIGNAL_SCENARIO,
        "test_signal_file": str(TEST_SIGNAL_FILE),
        "latencies_seconds": LATENCIES_SECONDS,
        "cooldowns_seconds": COOLDOWNS_SECONDS,
        "execution_variants": EXECUTION_VARIANTS,
        "limit_policies": LIMIT_POLICIES,
        "recommended_top": make_json_safe(recommended.head(10).to_dict(orient="records")),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(make_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")

    print("\nLATENCY SUPPORT")
    print(support.round(4).to_string(index=False))
    print("\nTOP RECOMMENDED")
    cols = [
        "scenario_id",
        "actions",
        "retention_pct",
        "hit_up_pct",
        "wrong_down_pct",
        "net_mean",
        "net_sum",
        "net_positive_pct",
        "max_drawdown_ticks",
        "positive_splits",
    ]
    print(recommended[cols].head(12).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
