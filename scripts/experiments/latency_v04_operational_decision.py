from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

LATENCY_V03_SPEC = importlib.util.spec_from_file_location(
    "latency_v03_event_budget",
    Path(__file__).with_name("latency_v03_event_budget.py"),
)
latency_v03 = importlib.util.module_from_spec(LATENCY_V03_SPEC)
LATENCY_V03_SPEC.loader.exec_module(latency_v03)

DEFAULT_OUT_DIR = Path("data/experiments/latency_v04_operational_decision")
DEFAULT_ANALYSIS_OUT_DIR = Path("data/experiments/latency_v03_event_budget")
DEFAULT_EVENT_LOG = latency_v03.DEFAULT_EVENT_LOG


def finite_number(value) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def round_to_grid_seconds(p95_ms: float, grid_seconds: float) -> float:
    seconds = max(float(p95_ms) / 1000.0, 0.0)
    return float(math.ceil(seconds / grid_seconds) * grid_seconds)


def row_for(summary: pd.DataFrame, component: str) -> dict | None:
    part = summary.loc[summary["component"] == component]
    if part.empty:
        return None
    return part.iloc[0].to_dict()


def choose_latency(summary: pd.DataFrame, min_rows: int, grid_seconds: float) -> dict:
    fill = row_for(summary, "total_seen_to_fill_ms")
    ack = row_for(summary, "total_seen_to_ack_ms")

    candidates = []
    for basis, row, finality in [
        ("fill", fill, "final_if_real_fills"),
        ("ack", ack, "provisional_ack_only"),
    ]:
        if row is None:
            continue
        n = int(row.get("n", 0))
        p95 = row.get("p95_ms")
        if n >= min_rows and finite_number(p95):
            candidates.append((basis, finality, n, float(p95), row))

    if not candidates:
        return {
            "status": "insufficient_real_latency_data",
            "reason": f"Need at least {min_rows} rows with total_seen_to_fill_ms or total_seen_to_ack_ms.",
            "recommended_l_seconds": None,
            "basis": None,
            "basis_finality": None,
            "p95_ms": None,
            "rows_used": 0,
            "grid_seconds": grid_seconds,
            "gate": "NO_GO_for_fixing_L_operativo",
        }

    basis, finality, n, p95_ms, row = candidates[0]
    l_seconds = round_to_grid_seconds(p95_ms, grid_seconds)
    if basis == "ack":
        gate = "CAUTION_ack_only_proxy_not_final"
    elif l_seconds <= 2.0:
        gate = "GO_to_validate_L2_labels"
    elif l_seconds <= 4.0:
        gate = "CAUTION_validate_L4_or_redesign_timing"
    else:
        gate = "NO_GO_current_taker_v0_without_timing_redesign"

    return {
        "status": "ready" if basis == "fill" else "provisional",
        "recommended_l_seconds": l_seconds,
        "basis": basis,
        "basis_finality": finality,
        "p95_ms": p95_ms,
        "rows_used": n,
        "grid_seconds": grid_seconds,
        "gate": gate,
        "basis_summary": row,
    }


def write_markdown(decision: dict, out_dir: Path) -> None:
    lines = [
        "# Latency v0.4 - Decision operativa",
        "",
        f"Estado: `{decision['status']}`",
        "",
        f"Base usada: `{decision.get('basis')}`",
        f"Finalidad: `{decision.get('basis_finality')}`",
        f"Filas usadas: `{decision.get('rows_used')}`",
        f"p95 ms: `{decision.get('p95_ms')}`",
        f"Grid segundos: `{decision.get('grid_seconds')}`",
        f"L recomendado: `{decision.get('recommended_l_seconds')}`",
        f"Gate: `{decision.get('gate')}`",
        "",
        "Lectura:",
        "",
    ]
    if decision["status"] == "insufficient_real_latency_data":
        lines.extend(
            [
                "Todavia no hay suficientes eventos reales para fijar `L_operativo`.",
                "Hay que instrumentar eventos paper/dry-run/live y volver a ejecutar.",
            ]
        )
    elif decision["status"] == "provisional":
        lines.extend(
            [
                "La decision usa `ack`, no `fill`. Sirve como proxy provisional,",
                "pero no cierra la latencia real de entrada ejecutada.",
            ]
        )
    else:
        lines.append("La decision usa fills reales y puede alimentar el contrato de labels.")
    (out_dir / "operational_latency_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Choose operational latency L from event logs.")
    parser.add_argument("--event-log", type=Path, default=DEFAULT_EVENT_LOG)
    parser.add_argument("--analysis-out-dir", type=Path, default=DEFAULT_ANALYSIS_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--grid-seconds", type=float, default=2.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    analysis_payload = latency_v03.analyze(args.event_log, args.analysis_out_dir)

    summary_path = args.analysis_out_dir / "component_latency_summary.csv"
    if not summary_path.exists():
        decision = {
            "status": "insufficient_real_latency_data",
            "reason": analysis_payload.get("reason", "component summary missing"),
            "event_log": str(args.event_log),
            "recommended_l_seconds": None,
            "basis": None,
            "basis_finality": None,
            "p95_ms": None,
            "rows_used": 0,
            "grid_seconds": args.grid_seconds,
            "gate": "NO_GO_for_fixing_L_operativo",
            "analysis_payload": analysis_payload,
        }
    else:
        summary = pd.read_csv(summary_path)
        decision = choose_latency(summary, min_rows=args.min_rows, grid_seconds=args.grid_seconds)
        decision["event_log"] = str(args.event_log)
        decision["analysis_out_dir"] = str(args.analysis_out_dir)
        decision["analysis_payload"] = analysis_payload

    (args.out_dir / "operational_latency_decision.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_markdown(decision, args.out_dir)
    print("LATENCY V04 OPERATIONAL DECISION")
    print(json.dumps(decision, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
