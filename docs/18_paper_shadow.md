# COMPLEX_V1A_PRESTART_H60_PAPER_SHADOW_V1

Generated: `2026-06-14T10:09:08+00:00`

## Strategy

Frozen prestart HGB specialist + volatility gate.

```text
Vol gate:  perp_realized_vol_bps_5s <= 0.6657
EV model:  ev_pred > 0.75
HP model:  hp_pred >= 0.5
Window:    strict_45_60_early (45-60s before settlement open)
```

## Decision

```text
PROMISING_NEEDS_DATA
```

## Summary Statistics

| Metric | Value |
|--------|-------|
| Fresh days | 5 |
| Days range | 2026-06-06 to 2026-06-10 |
| Total trades (vol gate) | 318 |
| Mean net@0.5 | +1.069 ticks |
| Mean net@0.25 | +0.819 ticks |
| Max drawdown (ticks) | 88.1 |
| Worst day net/trade | +0.025 ticks |
| Best day net/trade | +3.572 ticks |
| Positive days | 5/5 |
| 90% CI net/trade | [+0.472, +1.654] |
| P(net > 0) | 99.8% |
| Daily ticks estimate | +68.0 |

## Projection

| Gate | Status |
|------|--------|
| Days to PAPER_SHADOW_CANDIDATE (8 days, 200 trades, net>0.5) | 3 more days |
| Days to BOT_CANDIDATE (12 days, 400 trades, net>0.5) | 7 more days |

## Risk Controls

| Control | Value |
|---------|-------|
| Vol gate (skip HIGH_VOL) | vol > 0.6657 → no trade |
| No position sizing (paper) | 1x flat always |
| No intraday stop | collect more data first |

## Models

- EV regressor: `specialist_ev_strict_45_60_early_no_clock.joblib`
- HP classifier: `specialist_healthy_strict_45_60_early_no_clock.joblib`
- Manifest: `manifest.json`
- Frozen since: 2026-05-25 (training days: May 11-20)

## Run

```bash
python scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py
```
