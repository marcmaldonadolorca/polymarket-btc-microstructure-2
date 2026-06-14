"""
complex_v1a_prestart_h60_fresh_eval_v1.py

Evaluate frozen HGB prestart specialist (strict_45_60_early, full_features + no_clock)
on fresh June 6-10 execution data. Zero model retraining — pure out-of-sample test.

Fresh data: data/experiments/large_edge_multiresolution_intraday_fresh_execution_prepare_v1/
            fresh_execution_dataset/complex_v1a_execution_dataset.parquet

Frozen models: data/experiments/complex_v1a_prestart_h60_specialist_v1/models/
  specialist_ev_strict_45_60_early_full_features.joblib
  specialist_healthy_strict_45_60_early_full_features.joblib
  specialist_ev_strict_45_60_early_no_clock.joblib
  specialist_healthy_strict_45_60_early_no_clock.joblib

Decision thresholds applied:
  - ev_gt_0.5_healthy_ge_0.55 (the validated best policy from training)

Output:
  data/experiments/complex_v1a_prestart_h60_fresh_eval_v1/
    fresh_eval_daily_breakdown.csv
    fresh_eval_summary.csv
    fresh_eval_regime_breakdown.csv
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ---- Paths ----------------------------------------------------------------
FRESH_PARQUET = Path(
    "data/experiments/large_edge_multiresolution_intraday_fresh_execution_prepare_v1/"
    "fresh_execution_dataset/complex_v1a_execution_dataset.parquet"
)
SPECIALIST_DIR = Path("data/experiments/complex_v1a_prestart_h60_specialist_v1")
MANIFEST_PATH = SPECIALIST_DIR / "manifest.json"
MODELS_DIR = SPECIALIST_DIR / "models"

OUT_DIR = Path("data/experiments/complex_v1a_prestart_h60_fresh_eval_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COSTS = [0.25, 0.50, 1.00]
HIGH_VOL_THRESHOLD = 0.6657  # q80 from training — same as TCN regime head


def load_models(feature_set: str) -> tuple:
    """Returns (ev_model, healthy_model)."""
    ev_path = MODELS_DIR / f"specialist_ev_strict_45_60_early_{feature_set}.joblib"
    healthy_path = MODELS_DIR / f"specialist_healthy_strict_45_60_early_{feature_set}.joblib"
    assert ev_path.exists(), f"Missing: {ev_path}"
    assert healthy_path.exists(), f"Missing: {healthy_path}"
    return joblib.load(ev_path), joblib.load(healthy_path)


def apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict_45_60_early universe filters.

    seconds_before_start = -seconds_from_window_start (fresh dataset uses signed value).
    Filter: 45 < seconds_before_start <= 60  =>  -60 <= sfw < -45
    """
    mask = (
        (df["window_phase"] == "prestart") &
        (df["target_supported_H60"] == True) &
        (df["spread_ticks"] <= 2.0) &
        (df["visible_entry_cost_ticks"] <= 1.25) &
        (df["age_ms"] <= 1000.0) &
        (df["full_book_ratio"] >= 0.999) &
        (df["degraded_ratio"] == 0) &
        (df["seconds_from_window_start"] >= -60.0) &
        (df["seconds_from_window_start"] < -45.0)
    )
    return df[mask].copy()


def score_and_evaluate(
    rows: pd.DataFrame,
    ev_model,
    healthy_model,
    feature_cols: list,
    feature_set_name: str,
    policy_ev_thr: float = 0.5,
    policy_healthy_thr: float = 0.55,
) -> pd.DataFrame:
    """Score rows and apply policy. Returns rows with pred columns added."""
    X = rows[feature_cols].values
    rows = rows.copy()
    rows["pred_ev_score"] = ev_model.predict(X)
    rows["pred_healthy_prob"] = healthy_model.predict_proba(X)[:, 1]

    rows["selected"] = (
        (rows["pred_ev_score"] > policy_ev_thr) &
        (rows["pred_healthy_prob"] >= policy_healthy_thr)
    )
    rows["high_vol"] = rows["perp_realized_vol_bps_5s"] > HIGH_VOL_THRESHOLD
    rows["feature_set"] = feature_set_name
    return rows


def daily_breakdown(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """Per-day stats for selected rows under various costs."""
    selected = rows[rows["selected"]]
    if len(selected) == 0:
        return pd.DataFrame()

    records = []
    for day, grp in selected.groupby("session_day"):
        rec = {
            "feature_set": label,
            "session_day": day,
            "n_selected": len(grp),
            "vol_mean": grp["perp_realized_vol_bps_5s"].mean(),
            "pct_high_vol": grp["high_vol"].mean(),
            "gross_h60_mean": grp["exec_net_cost_0p25_H60"].mean(),  # no friction
        }
        for cost in COSTS:
            col = f"exec_net_cost_{str(cost).replace('.', 'p')}_H60"
            if col in grp.columns:
                rec[f"net_{str(cost).replace('.', 'p')}_mean"] = grp[col].mean()
        records.append(rec)

    df_out = pd.DataFrame(records)
    # Add all-days aggregate
    agg = {
        "feature_set": label,
        "session_day": "ALL_FRESH",
        "n_selected": len(selected),
        "vol_mean": selected["perp_realized_vol_bps_5s"].mean(),
        "pct_high_vol": selected["high_vol"].mean(),
        "gross_h60_mean": selected["exec_net_cost_0p25_H60"].mean(),
    }
    for cost in COSTS:
        col = f"exec_net_cost_{str(cost).replace('.', 'p')}_H60"
        if col in selected.columns:
            agg[f"net_{str(cost).replace('.', 'p')}_mean"] = selected[col].mean()
    df_out = pd.concat([df_out, pd.DataFrame([agg])], ignore_index=True)
    return df_out


def regime_breakdown(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """Performance split by vol regime for selected rows."""
    selected = rows[rows["selected"]]
    if len(selected) == 0:
        return pd.DataFrame()

    records = []
    for regime, grp in selected.groupby("high_vol"):
        rec = {
            "feature_set": label,
            "vol_regime": "HIGH_VOL" if regime else "LOW_VOL",
            "n": len(grp),
            "vol_mean": grp["perp_realized_vol_bps_5s"].mean(),
        }
        for cost in COSTS:
            col = f"exec_net_cost_{str(cost).replace('.', 'p')}_H60"
            if col in grp.columns:
                rec[f"net_{str(cost).replace('.', 'p')}_mean"] = grp[col].mean()
        records.append(rec)
    return pd.DataFrame(records)


def print_table(df: pd.DataFrame, title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)
    print(df.to_string(index=False))


def main():
    print("Loading manifest...")
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    full_features = manifest["feature_sets"]["full_features"]
    no_clock_features = manifest["feature_sets"]["no_clock"]

    print(f"full_features: {len(full_features)} features")
    print(f"no_clock features: {len(no_clock_features)} features")

    print("\nLoading fresh execution dataset...")
    df = pd.read_parquet(FRESH_PARQUET)
    print(f"Total rows: {len(df):,}")

    print("\nApplying strict_45_60_early filters...")
    strict = apply_strict_filters(df)
    print(f"Filtered rows: {len(strict):,}")
    print("By day:", strict["session_day"].value_counts().sort_index().to_dict())

    # Historical context from training
    HIST_TRAIN_MEAN_NET_0p5 = 1.16    # validation_initial performance
    HIST_TEST_MEAN_NET_0p5 = 0.60     # test_terminal performance
    HIST_MEAN_VOL_TRAIN = 0.299       # approximate mean vol in train period

    print(f"\nHistorical reference:")
    print(f"  val  net@0.5: +{HIST_TRAIN_MEAN_NET_0p5:.2f} ticks")
    print(f"  test net@0.5: +{HIST_TEST_MEAN_NET_0p5:.2f} ticks")
    print(f"  train vol mean: ~{HIST_MEAN_VOL_TRAIN:.3f} bps")
    print(f"  fresh vol mean: {strict['perp_realized_vol_bps_5s'].mean():.3f} bps")

    daily_dfs = []
    regime_dfs = []

    for feature_set_name, feature_cols, label in [
        ("full_features", full_features, "full_features"),
        ("no_clock", no_clock_features, "no_clock"),
    ]:
        print(f"\n--- Evaluating feature_set={label} ---")
        ev_model, healthy_model = load_models(feature_set_name)
        print(f"  Loaded: ev={type(ev_model).__name__}, healthy={type(healthy_model).__name__}")

        # Validate features exist
        missing = [c for c in feature_cols if c not in strict.columns]
        if missing:
            print(f"  WARNING: Missing features: {missing}")
            feature_cols = [c for c in feature_cols if c in strict.columns]

        scored = score_and_evaluate(
            strict, ev_model, healthy_model, feature_cols, label
        )
        n_selected = scored["selected"].sum()
        selection_rate = n_selected / len(scored) * 100
        print(f"  Selected: {n_selected}/{len(scored)} ({selection_rate:.1f}%)")
        print(f"  pred_ev_score mean: {scored['pred_ev_score'].mean():.3f}")
        print(f"  pred_healthy_prob mean: {scored['pred_healthy_prob'].mean():.3f}")

        daily_df = daily_breakdown(scored, label)
        regime_df = regime_breakdown(scored, label)

        if not daily_df.empty:
            print_table(daily_df, f"Daily Breakdown — {label}")
            daily_dfs.append(daily_df)

        if not regime_df.empty:
            print_table(regime_df, f"Vol Regime Breakdown — {label}")
            regime_dfs.append(regime_df)

    # Merge and save
    if daily_dfs:
        all_daily = pd.concat(daily_dfs, ignore_index=True)
        out_daily = OUT_DIR / "fresh_eval_daily_breakdown.csv"
        all_daily.to_csv(out_daily, index=False)
        print(f"\nSaved: {out_daily}")

    if regime_dfs:
        all_regime = pd.concat(regime_dfs, ignore_index=True)
        out_regime = OUT_DIR / "fresh_eval_regime_breakdown.csv"
        all_regime.to_csv(out_regime, index=False)
        print(f"Saved: {out_regime}")

    # Summary table: comparison vs training splits
    print("\n" + "="*70)
    print(" SUMMARY: Fresh (June 6-10) vs Historical")
    print("="*70)
    print(f"  Training rows (strict_45_60_early): ~28,095")
    print(f"  Fresh rows (strict_45_60_early):     {len(strict):,}")
    print(f"  Fresh days: {sorted(strict['session_day'].unique())}")
    print(f"  Vol regime: {(strict['perp_realized_vol_bps_5s'] > HIGH_VOL_THRESHOLD).mean()*100:.1f}% HIGH_VOL")
    print(f"  (train was ~19.6% HIGH_VOL)")
    if daily_dfs:
        for df_d in daily_dfs:
            row = df_d[df_d["session_day"] == "ALL_FRESH"]
            if not row.empty and "net_0p5_mean" in row.columns:
                print(f"\n  [{row.iloc[0]['feature_set']}] ALL_FRESH n={int(row.iloc[0]['n_selected'])} "
                      f"net@0.5={row.iloc[0]['net_0p5_mean']:.3f} ticks")

    print("\nDone.")


if __name__ == "__main__":
    main()
