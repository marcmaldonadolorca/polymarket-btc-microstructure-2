from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/experiments/latency_v03_event_budget")
DEFAULT_EVENT_LOG = OUT_DIR / "event_log.csv"

TIMESTAMP_COLS = [
    "data_source_utc",
    "data_seen_utc",
    "features_ready_utc",
    "model_start_utc",
    "model_done_utc",
    "decision_done_utc",
    "order_built_utc",
    "signed_utc",
    "sent_utc",
    "ack_utc",
    "fill_utc",
    "confirm_utc",
]

BASE_COLS = [
    "event_id",
    "event_type",
    "market_id",
    "condition_id",
    "token_id",
    "session_id",
    "time_index_ns",
    "score_buy",
    "threshold",
    "decision",
    "status",
    *TIMESTAMP_COLS,
    "notes",
]

COMPONENTS = [
    ("source_to_seen_ms", "data_source_utc", "data_seen_utc"),
    ("seen_to_features_ms", "data_seen_utc", "features_ready_utc"),
    ("features_to_model_start_ms", "features_ready_utc", "model_start_utc"),
    ("model_inference_ms", "model_start_utc", "model_done_utc"),
    ("model_to_decision_ms", "model_done_utc", "decision_done_utc"),
    ("decision_to_order_built_ms", "decision_done_utc", "order_built_utc"),
    ("order_build_to_signed_ms", "order_built_utc", "signed_utc"),
    ("signed_to_sent_ms", "signed_utc", "sent_utc"),
    ("sent_to_ack_ms", "sent_utc", "ack_utc"),
    ("ack_to_fill_ms", "ack_utc", "fill_utc"),
    ("fill_to_confirm_ms", "fill_utc", "confirm_utc"),
    ("total_seen_to_ack_ms", "data_seen_utc", "ack_utc"),
    ("total_seen_to_fill_ms", "data_seen_utc", "fill_utc"),
    ("total_seen_to_confirm_ms", "data_seen_utc", "confirm_utc"),
]


def percentile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float(np.percentile(clean.to_numpy(dtype=float), q))


def latency_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for component, start_col, end_col in COMPONENTS:
        values = pd.to_numeric(df.get(component), errors="coerce").dropna()
        row = {
            "component": component,
            "start_col": start_col,
            "end_col": end_col,
            "n": int(len(values)),
            "missing": int(len(df) - len(values)),
            "min_ms": float(values.min()) if len(values) else np.nan,
            "p50_ms": percentile(values, 50),
            "p75_ms": percentile(values, 75),
            "p90_ms": percentile(values, 90),
            "p95_ms": percentile(values, 95),
            "p99_ms": percentile(values, 99),
            "max_ms": float(values.max()) if len(values) else np.nan,
            "mean_ms": float(values.mean()) if len(values) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def create_schema(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    template = pd.DataFrame(
        [
            {
                "event_id": "example-001",
                "event_type": "paper",
                "market_id": "example_market",
                "condition_id": "example_condition",
                "token_id": "example_token",
                "session_id": 0,
                "time_index_ns": 0,
                "score_buy": 0.67,
                "threshold": 0.65,
                "decision": "buy",
                "status": "example",
                "data_source_utc": "2026-05-31T12:00:00.000000Z",
                "data_seen_utc": "2026-05-31T12:00:00.050000Z",
                "features_ready_utc": "2026-05-31T12:00:00.070000Z",
                "model_start_utc": "2026-05-31T12:00:00.071000Z",
                "model_done_utc": "2026-05-31T12:00:00.096000Z",
                "decision_done_utc": "2026-05-31T12:00:00.097000Z",
                "order_built_utc": "2026-05-31T12:00:00.105000Z",
                "signed_utc": "2026-05-31T12:00:00.155000Z",
                "sent_utc": "2026-05-31T12:00:00.165000Z",
                "ack_utc": "2026-05-31T12:00:00.260000Z",
                "fill_utc": "2026-05-31T12:00:00.410000Z",
                "confirm_utc": "2026-05-31T12:00:00.450000Z",
                "notes": "Example row. Replace with real timestamps.",
            }
        ],
        columns=BASE_COLS,
    )
    template.to_csv(out_dir / "event_log_template.csv", index=False)

    empty = pd.DataFrame(columns=BASE_COLS)
    if not DEFAULT_EVENT_LOG.exists():
        empty.to_csv(DEFAULT_EVENT_LOG, index=False)

    schema = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "event_log": str(DEFAULT_EVENT_LOG),
        "required_columns": BASE_COLS,
        "timestamp_columns": TIMESTAMP_COLS,
        "components": [
            {"component": name, "start_col": start, "end_col": end}
            for name, start, end in COMPONENTS
        ],
        "timestamp_format": "ISO-8601 UTC recommended, e.g. 2026-05-31T12:00:00.123456Z",
        "status_values_suggested": ["no_trade", "built", "signed", "sent", "acked", "filled", "partial", "rejected", "timeout"],
        "important_notes": [
            "Do not log private keys, raw signatures, secrets or wallet seed material.",
            "Use monotonic/perf_counter timestamps in runtime if possible, then export UTC wall time for analysis.",
            "Rows with missing start/end timestamps are excluded from that component only.",
        ],
    }
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")


def parse_event_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in BASE_COLS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in {path}: {missing}")
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def add_latency_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for component, start_col, end_col in COMPONENTS:
        out[component] = (out[end_col] - out[start_col]).dt.total_seconds() * 1000.0
    return out


def analyze(event_log: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    create_schema(out_dir)

    if not event_log.exists() or event_log.stat().st_size == 0:
        return write_pending(out_dir, event_log, reason="event_log_missing_or_empty")

    df = parse_event_log(event_log)
    if df.empty:
        return write_pending(out_dir, event_log, reason="event_log_has_no_rows")

    derived = add_latency_columns(df)
    summary = latency_summary(derived)

    derived_out = derived.copy()
    for col in TIMESTAMP_COLS:
        derived_out[col] = derived_out[col].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    derived_out.to_csv(out_dir / "event_latency_derived.csv", index=False)
    summary.to_csv(out_dir / "component_latency_summary.csv", index=False)

    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "status": "completed",
        "event_log": str(event_log),
        "rows": int(len(df)),
        "outputs": {
            "derived": "event_latency_derived.csv",
            "summary": "component_latency_summary.csv",
            "schema": "schema.json",
            "template": "event_log_template.csv",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def write_pending(out_dir: Path, event_log: Path, reason: str) -> dict:
    payload = {
        "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "status": "pending_real_event_log",
        "reason": reason,
        "event_log": str(event_log),
        "schema": str(out_dir / "schema.json"),
        "template": str(out_dir / "event_log_template.csv"),
        "next_step": "Fill event_log.csv with real end-to-end timestamps, then rerun this script.",
    }
    (out_dir / "manifest_pending.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Analyze end-to-end latency event logs.")
    parser.add_argument("--event-log", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    payload = analyze(args.event_log, args.out_dir)
    print("LATENCY V03 EVENT BUDGET")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
