from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path.cwd()
SCRIPT_DIR = PROJECT / "scripts" / "experiments"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import baseline_v0_full_core_robustness as baseline_v0  # noqa: E402
import baseline_v03_dataset_contract as contract  # noqa: E402

CACHE_DIR = Path("data/experiments/baseline_v0_full_core_robustness/baseline_v0_core_h16_parquet")
OUT_DIR = Path("data/experiments/latency_v01_local_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = "v03_conservative_no_micro"
MODEL_KIND = "hgb_balanced_small"
TARGET = "target_3c_16s_1tick"
RANDOM_STATE = 42

TRAIN_N = 50_000
BENCH_N = 20_000
MAX_PARTS = 3
WARMUP_REPEATS = 5
REPEATS = 40
BATCH_SIZES = [1, 2, 10, 100, 1_000, 5_000]


def now_ms(start_ns: int) -> float:
    return (time.perf_counter_ns() - start_ns) / 1_000_000


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, q))


def summarize(values: list[float]) -> dict:
    return {
        "n": len(values),
        "min_ms": float(min(values)),
        "p50_ms": percentile(values, 50),
        "p75_ms": percentile(values, 75),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "max_ms": float(max(values)),
        "mean_ms": float(statistics.fmean(values)),
    }


def register_feature_set() -> tuple[list[str], list[str], list[str]]:
    spec = contract.FEATURE_SETS[FEATURE_SET]
    baseline_v0.FEATURE_SETS[FEATURE_SET] = {"num": spec["num"], "cat": spec["cat"]}
    num = list(spec["num"])
    cat = list(spec["cat"])
    features = num + cat
    return num, cat, features


def load_sample(columns: list[str]) -> tuple[pd.DataFrame, dict]:
    files = sorted(CACHE_DIR.glob("*.parquet"))[:MAX_PARTS]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {CACHE_DIR}")

    read_rows = []
    read_metrics = []
    for path in files:
        t0 = time.perf_counter_ns()
        part = pd.read_parquet(path, columns=columns)
        read_ms = now_ms(t0)
        read_rows.append(part)
        read_metrics.append(
            {
                "component": "parquet_part_read",
                "file": path.name,
                "rows": len(part),
                "columns": len(columns),
                "elapsed_ms": read_ms,
            }
        )
        if sum(len(x) for x in read_rows) >= TRAIN_N + BENCH_N:
            break

    df = pd.concat(read_rows, ignore_index=True)
    df = baseline_v0.optimize_loaded_df(df)
    df = df[df[TARGET].notna()].copy()
    if len(df) < TRAIN_N + BENCH_N:
        raise RuntimeError(f"Not enough rows for probe: {len(df)} rows loaded")

    payload = {
        "files_read": [p.name for p in files[: len(read_rows)]],
        "rows_loaded": int(len(df)),
        "read_metrics": read_metrics,
    }
    return df, payload


def fit_proxy_model(train: pd.DataFrame, features: list[str]):
    model = baseline_v0.build_model(MODEL_KIND, FEATURE_SET)
    t0 = time.perf_counter_ns()
    model.fit(train[features], train[TARGET])
    fit_ms = now_ms(t0)
    return model, fit_ms


def class_index(model, name: str) -> int | None:
    classes = list(model.named_steps["clf"].classes_)
    return classes.index(name) if name in classes else None


def benchmark_model(model, bench: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    up_idx = class_index(model, "up")
    down_idx = class_index(model, "down")
    rows = []
    repetition_rows = []

    for batch_size in BATCH_SIZES:
        if batch_size > len(bench):
            continue

        # Warmup: avoid measuring first-call overhead.
        for _ in range(WARMUP_REPEATS):
            sample_idx = rng.choice(len(bench), size=batch_size, replace=False)
            _ = model.predict_proba(bench.iloc[sample_idx][features])

        inference_ms = []
        decision_ms = []
        total_ms = []

        for repeat in range(REPEATS):
            sample_idx = rng.choice(len(bench), size=batch_size, replace=False)
            batch = bench.iloc[sample_idx][features]

            t_total = time.perf_counter_ns()
            t0 = time.perf_counter_ns()
            proba = model.predict_proba(batch)
            infer_ms = now_ms(t0)

            t1 = time.perf_counter_ns()
            if up_idx is None:
                p_up = np.zeros(len(proba), dtype=float)
            else:
                p_up = proba[:, up_idx]
            if down_idx is None:
                p_down = np.zeros(len(proba), dtype=float)
            else:
                p_down = proba[:, down_idx]
            score_buy = p_up - p_down
            _ = score_buy >= 0.65
            dec_ms = now_ms(t1)
            tot_ms = now_ms(t_total)

            inference_ms.append(infer_ms)
            decision_ms.append(dec_ms)
            total_ms.append(tot_ms)
            repetition_rows.append(
                {
                    "batch_size": batch_size,
                    "repeat": repeat,
                    "inference_ms": infer_ms,
                    "decision_rule_ms": dec_ms,
                    "total_local_score_ms": tot_ms,
                    "per_row_total_ms": tot_ms / batch_size,
                }
            )

        for component, values in [
            ("model_predict_proba_pipeline", inference_ms),
            ("decision_rule_score_threshold", decision_ms),
            ("total_local_score", total_ms),
        ]:
            row = {
                "component": component,
                "batch_size": batch_size,
                "repeats": REPEATS,
                "per_row_p50_ms": percentile([v / batch_size for v in values], 50),
                "per_row_p95_ms": percentile([v / batch_size for v in values], 95),
            }
            row.update(summarize(values))
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(repetition_rows)


def main():
    t_script = time.perf_counter_ns()
    num, cat, features = register_feature_set()
    columns = sorted(set(features + [TARGET, "session_id", "condition_id", "time_index_ns", "session_day"]))

    t0 = time.perf_counter_ns()
    df, load_payload = load_sample(columns)
    load_total_ms = now_ms(t0)

    sample = df.sample(TRAIN_N + BENCH_N, random_state=RANDOM_STATE).reset_index(drop=True)
    train = sample.iloc[:TRAIN_N].copy()
    bench = sample.iloc[TRAIN_N : TRAIN_N + BENCH_N].copy()

    model, fit_ms = fit_proxy_model(train, features)
    latency_summary, repetitions = benchmark_model(model, bench, features)

    parquet_read = pd.DataFrame(load_payload["read_metrics"])
    parquet_read.to_csv(OUT_DIR / "parquet_read_metrics.csv", index=False)
    latency_summary.to_csv(OUT_DIR / "local_latency_summary.csv", index=False)
    repetitions.to_csv(OUT_DIR / "local_latency_repetitions.csv", index=False)

    manifest = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "scope": "local_probe_only_not_end_to_end_execution",
        "feature_set": FEATURE_SET,
        "model_kind": MODEL_KIND,
        "target": TARGET,
        "cache_dir": str(CACHE_DIR),
        "out_dir": str(OUT_DIR),
        "train_rows": int(len(train)),
        "bench_rows": int(len(bench)),
        "numeric_features": num,
        "categorical_features": cat,
        "total_features": len(features),
        "files_read": load_payload["files_read"],
        "load_total_ms": load_total_ms,
        "fit_proxy_model_ms": fit_ms,
        "script_wall_ms": now_ms(t_script),
        "important_limitations": [
            "No mide input/websocket/API latency.",
            "No mide construccion incremental real de features.",
            "No mide firma de orden ni contrato.",
            "No mide envio, ack, fill parcial/total ni confirmacion.",
            "El modelo se entrena como proxy local porque v0.3 no guardaba modelo serializado.",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("LATENCY V01 LOCAL PROBE")
    print(f"- out_dir: {OUT_DIR}")
    print(f"- rows loaded: {load_payload['rows_loaded']:,}")
    print(f"- train rows: {len(train):,}; bench rows: {len(bench):,}")
    print(f"- load_total_ms: {load_total_ms:.2f}")
    print(f"- fit_proxy_model_ms: {fit_ms:.2f}")
    print("\nLOCAL SCORE LATENCY SUMMARY")
    print(latency_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
