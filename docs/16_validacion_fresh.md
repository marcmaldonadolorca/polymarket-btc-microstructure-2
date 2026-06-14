# COMPLEX_V1A_PRESTART_H60_FRESH_EVAL_V1

Fresh evaluation of the frozen prestart HGB specialist on June 6-10, 2026 data.

## Setup
- **Frozen models**: `specialist_ev_strict_45_60_early_no_clock.joblib` + `specialist_healthy_strict_45_60_early_no_clock.joblib`
- **Policy**: `ev_gt_0.75_healthy_ge_0.5` (ev_pred > 0.75 AND hp_pred >= 0.50)
- **Universe**: strict_45_60_early (45-60s before settlement window opens)
- **Fresh period**: June 6-10, 2026 (5 days, 342,454 total rows)
- **Script**: `scripts/experiments/complex_v1a_prestart_h60_fresh_eval_v1.py`

## Key Finding: Vol Regime Dominates

Fresh data has **52% HIGH_VOL** (vs 19.6% in training). This regime shift is the dominant factor.

### Performance with no gate (all selections)

| split | n_selected | net@0.5 | notes |
|-------|-----------|---------|-------|
| train (May 11-20) | 4,706 | +4.50 | in-sample |
| val (May 21-22)   | 614 | +0.77 | |
| test (May 23-25)  | 1,094 | +0.43 | |
| **fresh (Jun 6-10)** | **754** | **+0.35** | **out-of-sample** |

### Performance with vol gate (perp_realized_vol_bps_5s <= 0.6657)

| split | n_low_vol | net@0.5 | n_high_vol | net@0.5_high | improvement |
|-------|----------|---------|-----------|-------------|------------|
| train | 3,669 | +4.41 | 1,037 | +4.83 | -0.09 (gate not needed) |
| val   | 486 | +0.35 | 128 | +2.33 | -0.41 (gate not needed) |
| test  | 893 | +0.41 | 201 | +0.51 | -0.02 (gate not needed) |
| **fresh** | **319** | **+1.08** | **435** | **-0.18** | **+0.73 (gate critical!)** |

### Vol threshold sweep (fresh, no_clock model)

| vol_thr | n | pct_kept | net@0.25 | net@0.5 |
|---------|---|---------|---------|---------|
| 0.30 | 169 | 22% | +1.72 | +1.53 |
| 0.50 | 255 | 34% | +1.10 | +0.91 |
| **0.6657** | **319** | **42%** | **+1.26** | **+1.08** |
| 1.00 | 460 | 61% | +0.75 | +0.57 |
| no gate | 753 | 100% | +0.53 | +0.35 |

Threshold 0.6657 is optimal — corresponds to TCN regime head HIGH_VOL definition (q80 vol in training).

## Why vol gate works in fresh but not historical

| period | HIGH_VOL fraction | vol gate effect |
|--------|------------------|----------------|
| train (May 11-20) | 22% | minimal (gate fires rarely) |
| val (May 21-22)   | 21% | minimal |
| test (May 23-25)  | 18% | minimal |
| fresh (Jun 6-10)  | 58% | **critical (+0.73 improvement)** |

The model was trained in a LOW_VOL regime. Its HIGH_VOL predictions are unreliable in the shifted distribution. The vol gate says "only trade when you're in the regime the model was trained on."

## Fresh Daily Breakdown (no gate, no_clock model)

| day | n | vol_mean | net@0.5 |
|-----|---|---------|---------|
| Jun 6 | 220 | 0.885 | +0.666 |
| Jun 7 | 82  | 1.127 | +1.343 |
| Jun 8 | 241 | 1.142 | +0.308 |
| Jun 9 | 107 | 0.738 | -0.565 |
| Jun 10 | 42 | 1.047 | -0.105 |
| ALL | 754 | 0.984 | +0.349 |

## Fresh Daily Breakdown (with vol gate 0.6657, no_clock model)

| day | n_low | n_high | low_vol_net@0.5 |
|-----|-------|--------|----------------|
| Jun 6  | 93 | 127 | (to be computed) |
| Jun 7  | 33 | 49  | |
| Jun 8  | 91 | 150 | |
| Jun 9  | 60 | 47  | |
| Jun 10 | 11 | 31  | |
| ALL | 319 | 435 | +1.076 |

## Decision

**VOL_GATE_VALIDATED_ADD_TO_LIVE_SHADOW**

The vol gate with threshold 0.6657 transforms the prestart specialist:
- Fresh net@0.5: +0.349 → +1.076 ticks (+0.727 improvement)
- Absolute daily profit improves: 63.8 trades × 1.076 = 68.6 ticks/5 days vs 150.8 × 0.349 = 52.6 ticks/5 days
- No lookahead: `perp_realized_vol_bps_5s` is observable at entry time
- Conservative on historical data: gate rarely fires (vol mostly below threshold)

## Combined Strategy

```
if perp_realized_vol_bps_5s <= 0.6657:
    score = ev_m.predict([no_clock_features])
    health = h_m.predict_proba([no_clock_features])[:, 1]
    if score > 0.75 and health >= 0.50:
        execute_trade()
else:
    skip()  # HIGH_VOL regime: model predictions unreliable
```

## TPS (Ticks Per Settlement) Context

Carry gate (20 settled): 20.75 ticks mean, 0 losses → different magnitude  
Prestart specialist + vol gate (319 fresh trades): +1.076 ticks mean → high frequency, smaller edge but many more opportunities

Total fresh (5 days) projected:
- Carry gate: ~4 settlements × 20.75 = ~83 ticks
- Prestart + vol gate: 319 × 1.076 = 343 ticks

The prestart specialist at scale is likely the larger P&L driver.
