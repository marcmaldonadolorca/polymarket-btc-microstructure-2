"""
complex_v1a_prestart_h60_paper_shadow_v1.py

Paper trading shadow for the frozen prestart HGB specialist + vol gate.

Policy (frozen, DO NOT retune):
    perp_realized_vol_bps_5s <= 0.6657   (vol gate)
    ev_pred > 0.75                         (EV model)
    hp_pred >= 0.50                        (health classifier)
    strict_45_60_early filter              (prestart window, 45-60s before open)

Run after each fresh pipeline rebuild:
    python scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py

Outputs:
    data/experiments/complex_v1a_prestart_h60_paper_shadow_v1/trades.csv
    data/experiments/complex_v1a_prestart_h60_paper_shadow_v1/daily_summary.csv
    docs/COMPLEX_V1A_PRESTART_H60_PAPER_SHADOW_V1.md
    results/key_results.csv  (rows appended/updated)
    results/decision_register.csv  (row appended/updated)

Decision gates:
    PAPER_SHADOW_CANDIDATE  : net@0.5 > 0.50 AND n_trades >= 200 AND n_days >= 8
    PROMISING_NEEDS_DATA    : net@0.5 > 0.00 AND n_trades < 200
    REVIEW_POLICY           : net@0.5 <= 0.00 AND n_days >= 3
    INSUFFICIENT_DATA       : n_days < 3
"""
from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPERIMENT = "complex_v1a_prestart_h60_paper_shadow_v1"
OUT_DIR = ROOT / "data" / "experiments" / EXPERIMENT
TRADES_CSV = OUT_DIR / "trades.csv"
DAILY_CSV = OUT_DIR / "daily_summary.csv"
DOC_PATH = ROOT / "docs" / "COMPLEX_V1A_PRESTART_H60_PAPER_SHADOW_V1.md"
KEY_RESULTS_PATH = ROOT / "results" / "key_results.csv"
DECISION_REGISTER_PATH = ROOT / "results" / "decision_register.csv"

FRESH_DATASET = (
    ROOT
    / "data"
    / "experiments"
    / "large_edge_multiresolution_intraday_fresh_execution_prepare_v1"
    / "fresh_execution_dataset"
    / "complex_v1a_execution_dataset.parquet"
)
MODELS_DIR = ROOT / "data" / "experiments" / "complex_v1a_prestart_h60_specialist_v1" / "models"
MANIFEST = ROOT / "data" / "experiments" / "complex_v1a_prestart_h60_specialist_v1" / "manifest.json"

EV_MODEL_PATH = MODELS_DIR / "specialist_ev_strict_45_60_early_no_clock.joblib"
HP_MODEL_PATH = MODELS_DIR / "specialist_healthy_strict_45_60_early_no_clock.joblib"

# Frozen policy parameters — DO NOT CHANGE
VOL_GATE_THR = 0.6657      # perp_realized_vol_bps_5s <= this (q80 training vol)
EV_THR = 0.75              # ev_pred > this
HP_THR = 0.50              # hp_pred >= this
PAPER_SIZE_UNITS = 1.0     # 1x position size
COST_AT_ENTRY = 0.5        # ticks cost assumption (matches exec_net_cost_0p5_H60)

# Bot-readiness gates
BOT_CANDIDATE_MIN_DAYS = 12
BOT_CANDIDATE_MIN_TRADES = 400
BOT_CANDIDATE_MIN_NET = 0.50
PAPER_SHADOW_MIN_DAYS = 8
PAPER_SHADOW_MIN_TRADES = 200
PAPER_SHADOW_MIN_NET = 0.50


def finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def cast_categoricals(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col in df.columns:
            dt = df[col].dtype
            if pd.api.types.is_string_dtype(dt) or isinstance(dt, pd.CategoricalDtype):
                df[col] = df[col].astype("category")
    return df


def apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["window_phase"] == "prestart")
        & (df["target_supported_H60"] == True)
        & (df["spread_ticks"] <= 2.0)
        & (df["visible_entry_cost_ticks"] <= 1.25)
        & (df["age_ms"] <= 1000.0)
        & (df["full_book_ratio"] >= 0.999)
        & (df["degraded_ratio"] == 0)
        & (df["seconds_from_window_start"] >= -60.0)
        & (df["seconds_from_window_start"] < -45.0)
    ].copy()


def write_csv_full(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_key_results(rows: list[dict[str, Any]]) -> None:
    path = KEY_RESULTS_PATH
    existing: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("experiment") != EXPERIMENT:
                    existing.append(dict(row))
    all_rows = existing + rows
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def update_decision_register(row: dict[str, Any]) -> None:
    path = DECISION_REGISTER_PATH
    existing: list[dict[str, Any]] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if r.get("artifact") != row.get("artifact"):
                    existing.append(dict(r))
    all_rows = existing + [row]
    fieldnames = list(all_rows[0].keys())
    for r in all_rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def compute_drawdown(cumulative_pnl: list[float]) -> float:
    if not cumulative_pnl:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for v in cumulative_pnl:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return max_dd


def decide(n_days: int, n_trades: int, net: float, n_losses: int, worst_day: float) -> str:
    if n_days < 3:
        return "INSUFFICIENT_DATA"
    if net <= 0.0 and n_days >= 3:
        return "REVIEW_POLICY"
    if n_days >= BOT_CANDIDATE_MIN_DAYS and n_trades >= BOT_CANDIDATE_MIN_TRADES and net >= BOT_CANDIDATE_MIN_NET:
        if n_losses == 0 or (n_losses / n_trades) < 0.10:
            return "BOT_CANDIDATE"
    if n_days >= PAPER_SHADOW_MIN_DAYS and n_trades >= PAPER_SHADOW_MIN_TRADES and net >= PAPER_SHADOW_MIN_NET:
        return "PAPER_SHADOW_CANDIDATE"
    if net > 0.0:
        return "PROMISING_NEEDS_DATA"
    return "REVIEW_POLICY"


def render_doc(
    decision: str,
    n_days: int,
    n_trades: int,
    net_025: float,
    net_050: float,
    max_dd: float,
    worst_day_net: float,
    best_day_net: float,
    n_positive_days: int,
    days_range: str,
    ci_low: float,
    ci_high: float,
    p_positive: float,
    daily_ticks_estimate: float,
    days_to_paper_shadow: int,
    days_to_bot_candidate: int,
) -> str:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# COMPLEX_V1A_PRESTART_H60_PAPER_SHADOW_V1",
        "",
        f"Generated: `{ts}`",
        "",
        "## Strategy",
        "",
        "Frozen prestart HGB specialist + volatility gate.",
        "",
        "```text",
        f"Vol gate:  perp_realized_vol_bps_5s <= {VOL_GATE_THR}",
        f"EV model:  ev_pred > {EV_THR}",
        f"HP model:  hp_pred >= {HP_THR}",
        f"Window:    strict_45_60_early (45-60s before settlement open)",
        "```",
        "",
        "## Decision",
        "",
        f"```text",
        f"{decision}",
        "```",
        "",
        "## Summary Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Fresh days | {n_days} |",
        f"| Days range | {days_range} |",
        f"| Total trades (vol gate) | {n_trades} |",
        f"| Mean net@0.5 | {net_050:+.3f} ticks |",
        f"| Mean net@0.25 | {net_025:+.3f} ticks |",
        f"| Max drawdown (ticks) | {max_dd:.1f} |",
        f"| Worst day net/trade | {worst_day_net:+.3f} ticks |",
        f"| Best day net/trade | {best_day_net:+.3f} ticks |",
        f"| Positive days | {n_positive_days}/{n_days} |",
        f"| 90% CI net/trade | [{ci_low:+.3f}, {ci_high:+.3f}] |",
        f"| P(net > 0) | {p_positive:.1f}% |",
        f"| Daily ticks estimate | {daily_ticks_estimate:+.1f} |",
        "",
        "## Projection",
        "",
        f"| Gate | Status |",
        f"|------|--------|",
        f"| Days to PAPER_SHADOW_CANDIDATE ({PAPER_SHADOW_MIN_DAYS} days, {PAPER_SHADOW_MIN_TRADES} trades, net>{PAPER_SHADOW_MIN_NET}) | {days_to_paper_shadow} more days |",
        f"| Days to BOT_CANDIDATE ({BOT_CANDIDATE_MIN_DAYS} days, {BOT_CANDIDATE_MIN_TRADES} trades, net>{BOT_CANDIDATE_MIN_NET}) | {days_to_bot_candidate} more days |",
        "",
        "## Risk Controls",
        "",
        "| Control | Value |",
        "|---------|-------|",
        f"| Vol gate (skip HIGH_VOL) | vol > {VOL_GATE_THR} → no trade |",
        "| No position sizing (paper) | 1x flat always |",
        "| No intraday stop | collect more data first |",
        "",
        "## Models",
        "",
        f"- EV regressor: `{EV_MODEL_PATH.name}`",
        f"- HP classifier: `{HP_MODEL_PATH.name}`",
        f"- Manifest: `{MANIFEST.name}`",
        f"- Frozen since: 2026-05-25 (training days: May 11-20)",
        "",
        "## Run",
        "",
        "```bash",
        "python scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    import joblib

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{EXPERIMENT}] Loading models...")
    ev_m = joblib.load(EV_MODEL_PATH)
    hp_m = joblib.load(HP_MODEL_PATH)

    with open(MANIFEST) as fh:
        manifest = json.load(fh)
    nc = manifest["feature_sets"]["no_clock"]

    print(f"[{EXPERIMENT}] Loading fresh dataset: {FRESH_DATASET}")
    df_fr = pd.read_parquet(FRESH_DATASET)
    print(f"  Loaded {len(df_fr):,} rows")

    # Apply prestart strict filters
    filtered = apply_strict_filters(df_fr)
    print(f"  After strict filters: {len(filtered):,} rows")

    # Cast categoricals
    filtered = cast_categoricals(filtered, nc)

    # Score with frozen models
    filtered["ev_pred"] = ev_m.predict(filtered[nc])
    filtered["hp_pred"] = hp_m.predict_proba(filtered[nc])[:, 1]

    # Apply vol gate + policy
    filtered["vol_ok"] = filtered["perp_realized_vol_bps_5s"] <= VOL_GATE_THR
    filtered["policy_ok"] = (filtered["ev_pred"] > EV_THR) & (filtered["hp_pred"] >= HP_THR)
    selected = filtered[filtered["vol_ok"] & filtered["policy_ok"]].copy()

    print(f"  Policy selected (no vol gate): {(filtered['policy_ok']).sum():,}")
    print(f"  Policy + vol gate: {len(selected):,}")

    if selected.empty:
        print("[WARNING] No trades selected — check fresh dataset and models.")
        return

    # Build per-trade records
    selected = selected.reset_index(drop=True)
    selected["cumulative_net_0p5"] = selected["exec_net_cost_0p5_H60"].cumsum()
    selected["is_loss"] = selected["exec_net_cost_0p5_H60"] < -COST_AT_ENTRY

    trade_rows = []
    for i, row in selected.iterrows():
        trade_rows.append({
            "trade_idx": int(i) + 1,
            "session_day": str(row["session_day"]),
            "condition_id": str(row["condition_id"])[:24] + "...",
            "token_id": str(row["token_id"])[:20] + "...",
            "seconds_from_window_start": round(float(row["seconds_from_window_start"]), 1),
            "spread_ticks": round(float(row["spread_ticks"]), 2),
            "visible_entry_cost_ticks": round(float(row["visible_entry_cost_ticks"]), 3),
            "perp_vol_bps_5s": round(float(row["perp_realized_vol_bps_5s"]), 4) if not math.isnan(float(row["perp_realized_vol_bps_5s"])) else None,
            "ev_pred": round(float(row["ev_pred"]), 3),
            "hp_pred": round(float(row["hp_pred"]), 3),
            "paper_size": PAPER_SIZE_UNITS,
            "actual_net_0p5": round(float(row["exec_net_cost_0p5_H60"]), 4),
            "is_loss": bool(row["is_loss"]),
            "cumulative_net_0p5": round(float(row["cumulative_net_0p5"]), 4),
        })

    write_csv_full(TRADES_CSV, trade_rows)
    print(f"  Trades written: {TRADES_CSV}")

    # Daily summary
    days = sorted(selected["session_day"].astype(str).unique())
    n_days = len(days)
    daily_rows = []
    for day in days:
        day_sel = selected[selected["session_day"].astype(str) == day]
        day_fr = filtered[filtered["session_day"].astype(str) == day]
        hv_pct = (day_fr["perp_realized_vol_bps_5s"] > VOL_GATE_THR).mean() * 100
        net = float(day_sel["exec_net_cost_0p5_H60"].mean())
        total_ticks = float(day_sel["exec_net_cost_0p5_H60"].sum())
        n_loss = int(day_sel["is_loss"].sum())
        daily_rows.append({
            "session_day": day,
            "n_trades": len(day_sel),
            "hv_pct": round(hv_pct, 1),
            "mean_net_0p5": round(net, 3),
            "total_ticks_0p5": round(total_ticks, 1),
            "n_loss": n_loss,
            "worst_trade": round(float(day_sel["exec_net_cost_0p5_H60"].min()), 3),
            "best_trade": round(float(day_sel["exec_net_cost_0p5_H60"].max()), 3),
        })
    write_csv_full(DAILY_CSV, daily_rows)
    print(f"  Daily summary written: {DAILY_CSV}")

    # Aggregate stats
    outcomes = selected["exec_net_cost_0p5_H60"].values
    net_050 = float(np.mean(outcomes))
    net_025 = float(np.mean(outcomes - 0.25))  # more conservative cost assumption
    n_losses = int(selected["is_loss"].sum())
    cum_pnl = list(np.cumsum(outcomes))
    max_dd = compute_drawdown(cum_pnl)

    daily_nets = [r["mean_net_0p5"] for r in daily_rows]
    worst_day_net = min(daily_nets)
    best_day_net = max(daily_nets)
    n_positive_days = sum(1 for x in daily_nets if x > 0)
    days_range = f"{days[0]} to {days[-1]}" if days else "none"

    daily_ticks = [r["total_ticks_0p5"] for r in daily_rows]
    daily_ticks_estimate = float(np.mean(daily_ticks)) if daily_ticks else 0.0

    # Bootstrap CI
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(outcomes, size=len(outcomes), replace=True).mean() for _ in range(5000)]
    ci_low, ci_high = float(np.percentile(boot_means, 5)), float(np.percentile(boot_means, 95))
    p_positive = float((np.array(boot_means) > 0).mean() * 100)

    # Progress toward gates
    trades_per_day = len(selected) / max(n_days, 1)
    days_to_paper_shadow = max(0, math.ceil(
        max(
            (PAPER_SHADOW_MIN_DAYS - n_days),
            (PAPER_SHADOW_MIN_TRADES - len(selected)) / max(trades_per_day, 1),
        )
    )) if net_050 > 0 else 999
    days_to_bot_candidate = max(0, math.ceil(
        max(
            (BOT_CANDIDATE_MIN_DAYS - n_days),
            (BOT_CANDIDATE_MIN_TRADES - len(selected)) / max(trades_per_day, 1),
        )
    )) if net_050 > 0 else 999

    decision = decide(n_days, len(selected), net_050, n_losses, worst_day_net)

    print()
    print(f"=== RESULTS ===")
    print(f"  Days: {n_days}  ({days_range})")
    print(f"  Trades: {len(selected)}  ({trades_per_day:.1f}/day)")
    print(f"  Net@0.5: {net_050:+.3f}  Net@0.25: {net_025:+.3f}")
    print(f"  90% CI: [{ci_low:+.3f}, {ci_high:+.3f}]  P(>0): {p_positive:.1f}%")
    print(f"  Max drawdown: {max_dd:.1f} ticks  Worst day: {worst_day_net:+.3f}/trade")
    print(f"  Positive days: {n_positive_days}/{n_days}")
    print(f"  Daily ticks estimate: {daily_ticks_estimate:+.1f}")
    print(f"  ==> {decision}")
    print(f"  Days to PAPER_SHADOW_CANDIDATE: {days_to_paper_shadow}")
    print(f"  Days to BOT_CANDIDATE:          {days_to_bot_candidate}")

    # Write doc
    doc_content = render_doc(
        decision=decision,
        n_days=n_days,
        n_trades=len(selected),
        net_025=net_025,
        net_050=net_050,
        max_dd=max_dd,
        worst_day_net=worst_day_net,
        best_day_net=best_day_net,
        n_positive_days=n_positive_days,
        days_range=days_range,
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_positive,
        daily_ticks_estimate=daily_ticks_estimate,
        days_to_paper_shadow=days_to_paper_shadow,
        days_to_bot_candidate=days_to_bot_candidate,
    )
    DOC_PATH.write_text(doc_content, encoding="utf-8")
    print(f"  Doc written: {DOC_PATH}")

    # Key results
    kr_rows = [
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "decision",
            "value": decision,
            "unit": "str",
            "status": decision,
            "interpretation": f"Prestart specialist + vol gate paper shadow: {n_days} days, {len(selected)} trades",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "n_fresh_days",
            "value": n_days,
            "unit": "days",
            "status": "OK" if n_days >= 5 else "LOW",
            "interpretation": days_range,
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "n_trades_vol_gate",
            "value": len(selected),
            "unit": "trades",
            "status": "OK",
            "interpretation": f"{trades_per_day:.1f} trades/day average",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "mean_net_at_0p5",
            "value": round(net_050, 4),
            "unit": "ticks",
            "status": "POSITIVE" if net_050 > 0 else "NEGATIVE",
            "interpretation": f"90% CI [{ci_low:+.3f}, {ci_high:+.3f}]",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "p_net_positive_bootstrap",
            "value": round(p_positive, 1),
            "unit": "pct",
            "status": "STRONG" if p_positive >= 99 else ("OK" if p_positive >= 90 else "WEAK"),
            "interpretation": f"Bootstrap 5000 resamples",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "daily_ticks_estimate",
            "value": round(daily_ticks_estimate, 1),
            "unit": "ticks/day",
            "status": "POSITIVE" if daily_ticks_estimate > 0 else "NEGATIVE",
            "interpretation": f"Mean total ticks per fresh day",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "max_drawdown_ticks",
            "value": round(max_dd, 1),
            "unit": "ticks",
            "status": "OK" if max_dd < 200 else "RISK",
            "interpretation": "Cumulative peak-to-trough on paper trades",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "days_to_paper_shadow_candidate",
            "value": days_to_paper_shadow,
            "unit": "days",
            "status": "TARGET" if days_to_paper_shadow <= 5 else "WAITING",
            "interpretation": f"Need {PAPER_SHADOW_MIN_DAYS} days, {PAPER_SHADOW_MIN_TRADES} trades, net>{PAPER_SHADOW_MIN_NET}",
        },
        {
            "phase": EXPERIMENT,
            "experiment": EXPERIMENT,
            "metric": "days_to_bot_candidate",
            "value": days_to_bot_candidate,
            "unit": "days",
            "status": "TARGET" if days_to_bot_candidate <= 10 else "WAITING",
            "interpretation": f"Need {BOT_CANDIDATE_MIN_DAYS} days, {BOT_CANDIDATE_MIN_TRADES} trades, net>{BOT_CANDIDATE_MIN_NET}",
        },
    ]
    update_key_results(kr_rows)
    print(f"  Key results updated: {KEY_RESULTS_PATH}")

    # Decision register
    dr_row = {
        "order": 170,
        "phase": EXPERIMENT,
        "artifact": f"scripts/experiments/{EXPERIMENT}.py",
        "decision": decision,
        "public_role": "main",
        "reason": (
            f"Prestart specialist frozen + vol gate ({VOL_GATE_THR}) paper shadow. "
            f"{n_days} fresh days ({days_range}). "
            f"n={len(selected)} trades at {trades_per_day:.1f}/day. "
            f"Net@0.5={net_050:+.3f} ticks (90%CI [{ci_low:+.3f},{ci_high:+.3f}], P(>0)={p_positive:.0f}%). "
            f"Max DD={max_dd:.0f} ticks. "
            f"Worst day={worst_day_net:+.3f}. "
            f"Days to PAPER_SHADOW_CANDIDATE={days_to_paper_shadow}."
        ),
    }
    update_decision_register(dr_row)
    print(f"  Decision register updated: {DECISION_REGISTER_PATH}")
    print()
    print(f"[{EXPERIMENT}] DONE => {decision}")


if __name__ == "__main__":
    main()
