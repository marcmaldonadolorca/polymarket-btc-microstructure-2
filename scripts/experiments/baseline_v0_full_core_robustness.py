from __future__ import annotations

import json
import shutil
import sqlite3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(r"D:\polymarket_btc_probe_official_v1")
DB = ROOT / "polymarket_week.sqlite3"
PROJECT = Path.cwd()
EXP_DIR = PROJECT / "data" / "experiments" / "baseline_v0_full_core_robustness"
CACHE_DIR = EXP_DIR / "baseline_v0_core_h16_parquet"
RESULTS_DIR = EXP_DIR / "results"
SUCCESS_FILE = CACHE_DIR / "_SUCCESS.json"

RANDOM_STATE = 42
TARGET_DELTA_NS = 16_000_000_000
BAND_TICKS = 1.0
CHUNK_ID_SIZE = 100_000
REBUILD_CACHE = False
RUN_PERMUTATION_IMPORTANCE = True
PERMUTATION_SAMPLE_N = 30_000
MAX_WALL_SECONDS = 8 * 60 * 60
NOTEBOOK_START_TS = time.time()

FRAME_COLS = ["session_id", "condition_id", "time_index_ns"]


def connect_ro():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def outcome_sign(label):
    text = str(label).lower()
    if "up" in text or "above" in text or "higher" in text:
        return 1.0
    if "down" in text or "below" in text or "lower" in text:
        return -1.0
    return np.nan


def fee_one_share(price, fee_rate_bps):
    return price * (fee_rate_bps / 10_000.0) * (price * (1.0 - price)) ** 2


def safe_return_bps(now, prev):
    now = pd.to_numeric(now, errors="coerce")
    prev = pd.to_numeric(prev, errors="coerce")
    return np.where(prev.gt(0), (now / prev - 1.0) * 10_000.0, np.nan)


def bucket_signal(s, pos, neg):
    s = pd.to_numeric(s, errors="coerce")
    return np.select([s >= pos, s <= -neg], ["pos", "neg"], default="neutral")


def agreement(pm, ext):
    pm = str(pm)
    ext = str(ext)
    if pm == "pos" and ext == "pos":
        return "both_pos"
    if pm == "neg" and ext == "neg":
        return "both_neg"
    if (pm == "pos" and ext == "neg") or (pm == "neg" and ext == "pos"):
        return "conflict"
    if pm == "neutral" and ext == "neutral":
        return "both_neutral"
    return "one_sided"


EXTRACT_SQL = r"""
SELECT
  f.id,
  f.session_id,
  DATE(s.created_at_utc) AS session_day,
  f.time_index_ns,
  f.time_index_utc,
  f.market_id,
  f.token_id,
  f.condition_id,
  f.temporality,
  f.window_phase,
  f.outcome_label,
  f.short_outcome_label,

  p.mid AS polymarket_mid,
  p.microprice,
  p.spread,
  p.trade_imbalance,
  p.age_ms,
  p.stale,
  p.missing,
  p2.mid AS polymarket_mid_lag_2s,
  p8.mid AS polymarket_mid_lag_8s,
  pf8.mid AS future_mid_8s,
  pf8.time_index_ns AS future_time_index_ns_8s,
  pf8.stale AS future_stale_8s,
  pf8.missing AS future_missing_8s,
  pf16.mid AS future_mid_16s,
  pf16.microprice AS future_microprice_16s,
  pf16.time_index_ns AS future_time_index_ns_16s,
  pf16.stale AS future_stale_16s,
  pf16.missing AS future_missing_16s,

  f.external_mid,
  f2.external_mid AS external_mid_lag_2s,
  f8.external_mid AS external_mid_lag_8s,
  f.external_trade_imbalance,
  f.external_realized_vol_bps_5s,
  f.external_realized_vol_bps_15s,

  f.perp_mid,
  f2.perp_mid AS perp_mid_lag_2s,
  f8.perp_mid AS perp_mid_lag_8s,
  f.perp_mark_price,
  f2.perp_mark_price AS perp_mark_price_lag_2s,
  f8.perp_mark_price AS perp_mark_price_lag_8s,
  f.perp_last_funding_rate,
  f.perp_open_interest,
  f8.perp_open_interest AS perp_open_interest_lag_8s,
  f.perp_basis,
  f8.perp_basis AS perp_basis_lag_8s,
  f.perp_taker_buy_sell_ratio,
  f.perp_realized_vol_bps_5s,
  f.spot_perp_mid_gap_bps,
  f.spot_perp_mark_gap_bps,

  f.polymarket_microprice_gap_bps,
  f.chainlink_missing,
  f.chainlink_staleness_ms,
  f.freshness_gap_ms,
  f.joint_age_ms,
  f.cross_ready,
  f.cross_ready_with_trade,
  f.cross_ready_with_chainlink,
  f.missing_context_count,
  f.stale_context_count,

  f.intertemporal_mid_vs_group_mean,
  f.intertemporal_mid_vs_nearest_shorter,
  f.intertemporal_mid_vs_nearest_longer,
  f.intertemporal_group_range,
  f.intertemporal_curve_residual,

  m.window_start_utc,
  m.window_end_utc,
  m.window_duration_seconds,
  m.tick_size,
  m.fee_rate_bps,
  m.min_order_size,

  mt.tradability_status,
  mt.practical_dead_early,
  mt.full_book_ratio,
  mt.degraded_ratio,

  l8.label_name AS predictor_label_name_8s,
  l8.economic_label_name AS predictor_economic_label_name_8s,
  l8.future_return_bps AS predictor_future_return_bps_8s,
  l8.economic_net_return_bps AS predictor_economic_net_return_bps_8s
FROM cross_venue_features f
JOIN collection_sessions s ON s.id = f.session_id
JOIN collection_session_telemetry t ON t.session_id = f.session_id
JOIN polymarket_grid_rows p
  ON p.session_id = f.session_id
 AND p.token_id = f.token_id
 AND p.time_index_ns = f.time_index_ns
LEFT JOIN polymarket_grid_rows p2
  ON p2.session_id = f.session_id
 AND p2.token_id = f.token_id
 AND p2.time_index_ns = f.time_index_ns - 2000000000
LEFT JOIN polymarket_grid_rows p8
  ON p8.session_id = f.session_id
 AND p8.token_id = f.token_id
 AND p8.time_index_ns = f.time_index_ns - 8000000000
LEFT JOIN polymarket_grid_rows pf8
  ON pf8.session_id = f.session_id
 AND pf8.token_id = f.token_id
 AND pf8.time_index_ns = f.time_index_ns + 8000000000
JOIN polymarket_grid_rows pf16
  ON pf16.session_id = f.session_id
 AND pf16.token_id = f.token_id
 AND pf16.time_index_ns = f.time_index_ns + 16000000000
LEFT JOIN cross_venue_features f2
  ON f2.session_id = f.session_id
 AND f2.token_id = f.token_id
 AND f2.time_index_ns = f.time_index_ns - 2000000000
LEFT JOIN cross_venue_features f8
  ON f8.session_id = f.session_id
 AND f8.token_id = f.token_id
 AND f8.time_index_ns = f.time_index_ns - 8000000000
JOIN market_metadata m
  ON m.session_id = f.session_id
 AND m.market_id = f.market_id
LEFT JOIN session_market_tradability mt
  ON mt.session_id = f.session_id
 AND mt.market_id = f.market_id
LEFT JOIN predictor_labels l8
  ON l8.session_id = f.session_id
 AND l8.token_id = f.token_id
 AND l8.time_index_ns = f.time_index_ns
 AND l8.prediction_horizon_steps = 4
WHERE f.id BETWEEN ? AND ?
  AND s.dataset_tier='core'
  AND t.coverage_ratio >= 0.97
  AND t.quality_score >= 70
  AND t.core_gap_severe_segment_count = 0
  AND m.tick_size > 0
  AND p.mid IS NOT NULL
  AND p.microprice IS NOT NULL
  AND p.spread IS NOT NULL
  AND COALESCE(p.stale,0) = 0
  AND COALESCE(p.missing,0) = 0
  AND COALESCE(pf16.stale,0) = 0
  AND COALESCE(pf16.missing,0) = 0
  AND COALESCE(mt.tradability_status,'') != 'never_operable'
ORDER BY f.id
"""

NUMERIC_BASE_FOR_DERIVATION = [
    "polymarket_mid",
    "microprice",
    "spread",
    "trade_imbalance",
    "age_ms",
    "polymarket_mid_lag_2s",
    "polymarket_mid_lag_8s",
    "future_mid_8s",
    "future_mid_16s",
    "external_mid",
    "external_mid_lag_2s",
    "external_mid_lag_8s",
    "external_trade_imbalance",
    "external_realized_vol_bps_5s",
    "external_realized_vol_bps_15s",
    "perp_mid",
    "perp_mid_lag_2s",
    "perp_mid_lag_8s",
    "perp_mark_price",
    "perp_mark_price_lag_2s",
    "perp_mark_price_lag_8s",
    "perp_last_funding_rate",
    "perp_open_interest",
    "perp_open_interest_lag_8s",
    "perp_basis",
    "perp_basis_lag_8s",
    "perp_taker_buy_sell_ratio",
    "perp_realized_vol_bps_5s",
    "spot_perp_mid_gap_bps",
    "spot_perp_mark_gap_bps",
    "polymarket_microprice_gap_bps",
    "chainlink_staleness_ms",
    "freshness_gap_ms",
    "joint_age_ms",
    "missing_context_count",
    "stale_context_count",
    "intertemporal_mid_vs_group_mean",
    "intertemporal_mid_vs_nearest_shorter",
    "intertemporal_mid_vs_nearest_longer",
    "intertemporal_group_range",
    "intertemporal_curve_residual",
    "window_duration_seconds",
    "tick_size",
    "fee_rate_bps",
    "min_order_size",
    "full_book_ratio",
    "degraded_ratio",
    "predictor_future_return_bps_8s",
    "predictor_economic_net_return_bps_8s",
]


def derive_features(df):
    df = df.copy()
    for col in NUMERIC_BASE_FOR_DERIVATION:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["outcome_sign"] = df["outcome_label"].map(outcome_sign)
    df["boundary_distance"] = np.minimum(df["polymarket_mid"], 1.0 - df["polymarket_mid"])
    df["microprice_gap_ticks"] = (df["microprice"] - df["polymarket_mid"]) / df["tick_size"]
    df["microprice_gap_ticks_clipped"] = df["microprice_gap_ticks"].clip(-5, 5)
    df["spread_ticks"] = df["spread"] / df["tick_size"]
    df["entry_ask_est"] = df["polymarket_mid"] + df["spread"] / 2.0
    df["entry_fee_ticks"] = (
        fee_one_share(df["entry_ask_est"].clip(0, 1), df["fee_rate_bps"].fillna(0)) / df["tick_size"]
    )
    df["visible_entry_cost_ticks"] = (df["spread"] / (2.0 * df["tick_size"])) + df["entry_fee_ticks"]

    df["delta_ticks_16s"] = (df["future_mid_16s"] - df["polymarket_mid"]) / df["tick_size"]
    df["delta_ticks_8s"] = (df["future_mid_8s"] - df["polymarket_mid"]) / df["tick_size"]
    df["future_delta_ns_8s"] = df["future_time_index_ns_8s"] - df["time_index_ns"]
    df["future_delta_ns_16s"] = df["future_time_index_ns_16s"] - df["time_index_ns"]
    df["target_3c_16s_1tick"] = np.select(
        [df["delta_ticks_16s"] >= BAND_TICKS, df["delta_ticks_16s"] <= -BAND_TICKS],
        ["up", "down"],
        default="flat",
    )
    df["target_3c_8s_1tick"] = np.select(
        [df["delta_ticks_8s"] >= BAND_TICKS, df["delta_ticks_8s"] <= -BAND_TICKS],
        ["up", "down"],
        default="flat",
    )

    df["mid_delta_ticks_2s"] = (df["polymarket_mid"] - df["polymarket_mid_lag_2s"]) / df["tick_size"]
    df["mid_delta_ticks_8s"] = (df["polymarket_mid"] - df["polymarket_mid_lag_8s"]) / df["tick_size"]

    for label, lag_suffix in [("2s", "lag_2s"), ("8s", "lag_8s")]:
        df[f"external_mid_return_bps_{label}"] = safe_return_bps(
            df["external_mid"], df[f"external_mid_{lag_suffix}"]
        )
        df[f"external_mid_return_bps_{label}_oriented"] = (
            df[f"external_mid_return_bps_{label}"] * df["outcome_sign"]
        )
        df[f"perp_mid_return_bps_{label}"] = safe_return_bps(df["perp_mid"], df[f"perp_mid_{lag_suffix}"])
        df[f"perp_mid_return_bps_{label}_oriented"] = (
            df[f"perp_mid_return_bps_{label}"] * df["outcome_sign"]
        )
        df[f"perp_mark_price_return_bps_{label}"] = safe_return_bps(
            df["perp_mark_price"], df[f"perp_mark_price_{lag_suffix}"]
        )
        df[f"perp_mark_price_return_bps_{label}_oriented"] = (
            df[f"perp_mark_price_return_bps_{label}"] * df["outcome_sign"]
        )

    df["perp_open_interest_delta_8s"] = df["perp_open_interest"] - df["perp_open_interest_lag_8s"]
    df["perp_basis_delta_8s"] = df["perp_basis"] - df["perp_basis_lag_8s"]
    df["external_trade_imbalance_oriented"] = df["external_trade_imbalance"] * df["outcome_sign"]
    ratio = pd.to_numeric(df["perp_taker_buy_sell_ratio"], errors="coerce")
    df["perp_taker_buy_sell_log_oriented"] = np.log(ratio.where(ratio > 0)) * df["outcome_sign"]

    time_utc = pd.to_datetime(df["time_index_utc"], utc=True, errors="coerce")
    start_utc = pd.to_datetime(df["window_start_utc"], utc=True, errors="coerce")
    end_utc = pd.to_datetime(df["window_end_utc"], utc=True, errors="coerce")
    df["seconds_to_window_end"] = (end_utc - time_utc).dt.total_seconds()
    df["seconds_from_window_start"] = (time_utc - start_utc).dt.total_seconds()
    df["window_progress"] = df["seconds_from_window_start"] / df["window_duration_seconds"]
    df["phase_bucket"] = pd.cut(
        df["window_progress"], bins=[-np.inf, 0.33, 0.66, np.inf], labels=["early", "middle", "late"]
    ).astype("object")
    df["utc_hour"] = time_utc.dt.hour.astype("float")
    df["day_of_week"] = time_utc.dt.dayofweek.astype("float")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("float")

    df["pm_bucket"] = bucket_signal(df["polymarket_microprice_gap_bps"], 50, 50)
    df["perp2_bucket"] = bucket_signal(df["perp_mid_return_bps_2s_oriented"], 0.5, 0.5)
    df["spot2_bucket"] = bucket_signal(df["external_mid_return_bps_2s_oriented"], 0.2, 0.2)
    df["pm_perp_agreement"] = [agreement(a, b) for a, b in zip(df["pm_bucket"], df["perp2_bucket"])]
    df["pm_spot_agreement"] = [agreement(a, b) for a, b in zip(df["pm_bucket"], df["spot2_bucket"])]
    df["tradability_status"] = df["tradability_status"].fillna("unknown")
    df["practical_dead_early"] = (
        pd.to_numeric(df["practical_dead_early"], errors="coerce").fillna(0).astype("int8")
    )
    df["chainlink_missing"] = pd.to_numeric(df["chainlink_missing"], errors="coerce").fillna(0).astype("int8")

    keep = (
        df["future_delta_ns_16s"].eq(TARGET_DELTA_NS)
        & df["outcome_sign"].notna()
        & df["tick_size"].gt(0)
        & df["delta_ticks_16s"].notna()
    )
    return df.loc[keep].reset_index(drop=True)


def extract_cache(id_min, id_max):
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SUCCESS_FILE.exists() and not REBUILD_CACHE:
        print(f"Reusing cache: {CACHE_DIR}")
        return json.loads(SUCCESS_FILE.read_text(encoding="utf-8"))
    if CACHE_DIR.exists() and not REBUILD_CACHE:
        raise RuntimeError(f"Cache dir exists but no _SUCCESS file: {CACHE_DIR}")
    if CACHE_DIR.exists() and REBUILD_CACHE:
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "chunk_id_size": CHUNK_ID_SIZE,
        "target_delta_ns": TARGET_DELTA_NS,
        "rows_written": 0,
        "chunks": [],
    }
    with connect_ro() as conn:
        for part_idx, lo in enumerate(range(int(id_min), int(id_max) + 1, CHUNK_ID_SIZE)):
            hi = min(lo + CHUNK_ID_SIZE - 1, int(id_max))
            t0 = time.time()
            raw = pd.read_sql_query(EXTRACT_SQL, conn, params=(lo, hi))
            derived = derive_features(raw) if len(raw) else raw
            part_path = CACHE_DIR / f"part_{part_idx:04d}.parquet"
            if len(derived):
                table = pa.Table.from_pandas(derived, preserve_index=False)
                pq.write_table(table, part_path, compression="zstd")
                part_written = str(part_path)
            else:
                part_written = None
            elapsed = time.time() - t0
            row = {
                "part": part_idx,
                "id_start": lo,
                "id_end": hi,
                "raw_rows": int(len(raw)),
                "rows_written": int(len(derived)),
                "seconds": elapsed,
                "part_path": part_written,
            }
            manifest["chunks"].append(row)
            manifest["rows_written"] += int(len(derived))
            print(
                f"chunk {part_idx:03d} ids {lo:,}-{hi:,}: "
                f"raw={len(raw):,} support={len(derived):,} sec={elapsed:.1f}",
                flush=True,
            )
            if time.time() - NOTEBOOK_START_TS > MAX_WALL_SECONDS:
                raise TimeoutError("Wall-time guard reached before finishing extraction")
    SUCCESS_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def assign_terminal_split(day):
    day = str(day)
    if "2026-05-23" <= day <= "2026-05-25":
        return "test_terminal"
    if "2026-05-21" <= day <= "2026-05-22":
        return "validation_initial"
    if day <= "2026-05-20":
        return "train_initial"
    return "other"


def rows_between_days(data, start, end):
    day = data["session_day"].astype(str)
    return data[day.between(start, end)].copy()


def purge_train_conditions(train, validation):
    val_conditions = set(validation["condition_id"].dropna().unique())
    before = len(train)
    if val_conditions:
        train = train[~train["condition_id"].isin(val_conditions)].copy()
    return train, before - len(train)


def complete_frame_keys(scored_df):
    stats = scored_df.groupby(FRAME_COLS, observed=True).agg(
        rows=("token_id", "size"),
        tokens=("token_id", "nunique"),
        sign_sum=("outcome_sign", "sum"),
        abs_sign_sum=("outcome_sign", lambda s: np.abs(s).sum()),
    ).reset_index()
    return stats[
        stats["rows"].eq(2)
        & stats["tokens"].eq(2)
        & stats["sign_sum"].eq(0)
        & stats["abs_sign_sum"].eq(2)
    ][FRAME_COLS]


def make_preprocessor(num_features, cat_features, dense=False, scale_numeric=True):
    transformers = []
    if num_features:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), num_features))
    if cat_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense, dtype=np.float32)),
                    ]
                ),
                cat_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0 if dense else 0.3)


def token_metrics(y_true, pred, score=None, delta=None):
    out = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
    }
    if score is not None and delta is not None:
        out["score_delta_spearman"] = pd.Series(score).corr(pd.Series(delta), method="spearman")
        out["score_delta_pearson"] = pd.Series(score).corr(pd.Series(delta), method="pearson")
    return out


def market_action_eval(scored_df, threshold):
    keys = complete_frame_keys(scored_df)
    complete = scored_df.merge(keys, on=FRAME_COLS, how="inner")
    if complete.empty:
        return pd.DataFrame([{"threshold": threshold, "complete_frames": 0, "actions": 0}]), complete.iloc[0:0]
    idx = complete.groupby(FRAME_COLS, observed=True)["score_buy"].idxmax()
    chosen = complete.loc[idx].copy()
    actions = chosen[chosen["score_buy"] > threshold].copy()
    if len(actions):
        actions["hit_up"] = actions["target_3c_16s_1tick"].eq("up")
        actions["flat"] = actions["target_3c_16s_1tick"].eq("flat")
        actions["wrong_down"] = actions["target_3c_16s_1tick"].eq("down")
        actions["net_visible_ticks"] = actions["delta_ticks_16s"] - actions["visible_entry_cost_ticks"]
    summary = {
        "threshold": threshold,
        "complete_frames": len(chosen),
        "actions": len(actions),
        "coverage_pct": len(actions) / len(chosen) * 100 if len(chosen) else np.nan,
        "hit_up_pct": actions["hit_up"].mean() * 100 if len(actions) else np.nan,
        "flat_pct": actions["flat"].mean() * 100 if len(actions) else np.nan,
        "wrong_down_pct": actions["wrong_down"].mean() * 100 if len(actions) else np.nan,
        "delta_mean": actions["delta_ticks_16s"].mean() if len(actions) else np.nan,
        "delta_median": actions["delta_ticks_16s"].median() if len(actions) else np.nan,
        "cost_median": actions["visible_entry_cost_ticks"].median() if len(actions) else np.nan,
        "net_visible_mean": actions["net_visible_ticks"].mean() if len(actions) else np.nan,
        "net_visible_median": actions["net_visible_ticks"].median() if len(actions) else np.nan,
        "net_positive_pct": actions["net_visible_ticks"].gt(0).mean() * 100 if len(actions) else np.nan,
        "score_median": actions["score_buy"].median() if len(actions) else np.nan,
    }
    return pd.DataFrame([summary]), actions


THRESHOLDS = [round(x, 2) for x in np.arange(-0.05, 0.81, 0.05)]


def threshold_curve(scored_df):
    return pd.concat([market_action_eval(scored_df, th)[0] for th in THRESHOLDS], ignore_index=True)


def choose_threshold(curve, min_coverage_pct=1.0, min_actions=200, max_wrong_down_pct=35.0):
    c = curve.copy()
    c["selection_score"] = (
        c["net_visible_mean"].fillna(-999)
        - 0.025 * c["wrong_down_pct"].fillna(100)
        + 0.010 * c["coverage_pct"].fillna(0)
    )
    candidates = c[
        c["coverage_pct"].ge(min_coverage_pct)
        & c["actions"].ge(min_actions)
        & c["wrong_down_pct"].le(max_wrong_down_pct)
    ].copy()
    if candidates.empty:
        candidates = c[c["coverage_pct"].ge(0.5) & c["actions"].ge(max(30, min_actions // 4))].copy()
    if candidates.empty:
        candidates = c[c["actions"].gt(0)].copy()
    if candidates.empty:
        return np.nan, pd.Series(dtype=object)
    best = candidates.sort_values(["selection_score", "net_visible_mean", "actions"], ascending=[False, False, False]).iloc[0]
    return float(best["threshold"]), best


PM_NUMERIC = [
    "polymarket_mid",
    "boundary_distance",
    "microprice_gap_ticks_clipped",
    "mid_delta_ticks_2s",
    "mid_delta_ticks_8s",
    "spread_ticks",
    "visible_entry_cost_ticks",
    "age_ms",
]
PM_NO_MICRO_NUMERIC = [x for x in PM_NUMERIC if x != "microprice_gap_ticks_clipped"]
SPOT_NUMERIC = [
    "external_mid_return_bps_2s_oriented",
    "external_mid_return_bps_8s_oriented",
    "external_trade_imbalance_oriented",
    "external_realized_vol_bps_5s",
]
PERP_NUMERIC = [
    "perp_mid_return_bps_2s_oriented",
    "perp_mid_return_bps_8s_oriented",
    "perp_mark_price_return_bps_2s_oriented",
    "perp_mark_price_return_bps_8s_oriented",
    "perp_taker_buy_sell_log_oriented",
    "perp_basis",
    "perp_basis_delta_8s",
    "perp_open_interest_delta_8s",
    "perp_realized_vol_bps_5s",
    "spot_perp_mid_gap_bps",
    "spot_perp_mark_gap_bps",
]
TIME_NUMERIC = [
    "seconds_to_window_end",
    "seconds_from_window_start",
    "window_progress",
    "tick_size",
    "fee_rate_bps",
    "min_order_size",
]
QUALITY_NUMERIC = [
    "chainlink_missing",
    "chainlink_staleness_ms",
    "freshness_gap_ms",
    "joint_age_ms",
    "missing_context_count",
    "stale_context_count",
    "full_book_ratio",
    "degraded_ratio",
]
INTERTEMP_NUMERIC = [
    "intertemporal_mid_vs_group_mean",
    "intertemporal_mid_vs_nearest_shorter",
    "intertemporal_mid_vs_nearest_longer",
    "intertemporal_group_range",
    "intertemporal_curve_residual",
]
BASE_CATS = ["temporality", "window_phase", "phase_bucket"]
AGREEMENT_CATS = ["pm_perp_agreement", "pm_spot_agreement"]
QUALITY_CATS = ["tradability_status"]

FEATURE_SETS = {
    "pm_only": {"num": PM_NUMERIC, "cat": []},
    "pm_time": {"num": PM_NUMERIC + TIME_NUMERIC, "cat": BASE_CATS},
    "pm_perp_time": {"num": PM_NUMERIC + PERP_NUMERIC + TIME_NUMERIC, "cat": BASE_CATS + ["pm_perp_agreement"]},
    "pm_spot_time": {"num": PM_NUMERIC + SPOT_NUMERIC + TIME_NUMERIC, "cat": BASE_CATS + ["pm_spot_agreement"]},
    "full_v0": {"num": PM_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC, "cat": BASE_CATS + AGREEMENT_CATS},
    "full_no_microprice": {"num": PM_NO_MICRO_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC, "cat": BASE_CATS},
    "full_plus_quality_masks": {
        "num": PM_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC + QUALITY_NUMERIC,
        "cat": BASE_CATS + AGREEMENT_CATS + QUALITY_CATS,
    },
    "full_plus_intertemporal": {
        "num": PM_NUMERIC + SPOT_NUMERIC + PERP_NUMERIC + TIME_NUMERIC + INTERTEMP_NUMERIC,
        "cat": BASE_CATS + AGREEMENT_CATS,
    },
}


def build_model(model_kind, feature_set_name):
    fs = FEATURE_SETS[feature_set_name]
    if model_kind == "logreg_balanced":
        return Pipeline(
            [
                ("prep", make_preprocessor(fs["num"], fs["cat"], dense=False, scale_numeric=True)),
                ("clf", LogisticRegression(max_iter=700, class_weight="balanced", solver="lbfgs")),
            ]
        )
    if model_kind == "hgb_balanced_small":
        return Pipeline(
            [
                ("prep", make_preprocessor(fs["num"], fs["cat"], dense=True, scale_numeric=False)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=90,
                        learning_rate=0.06,
                        max_leaf_nodes=15,
                        min_samples_leaf=50,
                        l2_regularization=0.05,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    raise ValueError(model_kind)


def add_model_scores(model, data, feature_set_name):
    fs = FEATURE_SETS[feature_set_name]
    features = fs["num"] + fs["cat"]
    proba = model.predict_proba(data[features])
    pred = model.predict(data[features])
    classes = list(model.named_steps["clf"].classes_)
    out = data[
        [
            "session_id",
            "condition_id",
            "time_index_ns",
            "token_id",
            "temporality",
            "window_phase",
            "phase_bucket",
            "outcome_sign",
            "target_3c_16s_1tick",
            "delta_ticks_16s",
            "visible_entry_cost_ticks",
            "spread_ticks",
            "tradability_status",
            "practical_dead_early",
            "full_book_ratio",
            "degraded_ratio",
        ]
    ].copy()
    out["pred"] = pred
    out["p_down"] = proba[:, classes.index("down")] if "down" in classes else 0.0
    out["p_flat"] = proba[:, classes.index("flat")] if "flat" in classes else 0.0
    out["p_up"] = proba[:, classes.index("up")] if "up" in classes else 0.0
    out["score_buy"] = out["p_up"] - out["p_down"]
    return out


def fit_eval_model(train, eval_data, model_kind, feature_set_name):
    fs = FEATURE_SETS[feature_set_name]
    features = fs["num"] + fs["cat"]
    model = build_model(model_kind, feature_set_name)
    t0 = time.time()
    model.fit(train[features], train["target_3c_16s_1tick"])
    fit_seconds = time.time() - t0
    scored = add_model_scores(model, eval_data, feature_set_name)
    metrics = token_metrics(
        eval_data["target_3c_16s_1tick"],
        scored["pred"],
        scored["score_buy"],
        eval_data["delta_ticks_16s"],
    )
    metrics["fit_seconds"] = fit_seconds
    return model, scored, metrics


def exact_one_rule_eval(data, rule_col, positive_value="both_pos", label="rule"):
    tmp = data[
        [
            "session_id",
            "condition_id",
            "time_index_ns",
            "token_id",
            "outcome_sign",
            "target_3c_16s_1tick",
            "delta_ticks_16s",
            "visible_entry_cost_ticks",
            "temporality",
            "window_phase",
            "phase_bucket",
            "tradability_status",
            "practical_dead_early",
            "full_book_ratio",
            "degraded_ratio",
            rule_col,
        ]
    ].copy()
    keys = complete_frame_keys(tmp.assign(score_buy=0.0, pred=""))
    complete = tmp.merge(keys, on=FRAME_COLS, how="inner")
    candidates = complete[complete[rule_col].eq(positive_value)].copy()
    counts = candidates.groupby(FRAME_COLS, observed=True).size().rename("n_candidates").reset_index()
    one = counts[counts["n_candidates"].eq(1)][FRAME_COLS]
    actions = candidates.merge(one, on=FRAME_COLS, how="inner").drop_duplicates(FRAME_COLS).copy()
    if len(actions):
        actions["hit_up"] = actions["target_3c_16s_1tick"].eq("up")
        actions["flat"] = actions["target_3c_16s_1tick"].eq("flat")
        actions["wrong_down"] = actions["target_3c_16s_1tick"].eq("down")
        actions["net_visible_ticks"] = actions["delta_ticks_16s"] - actions["visible_entry_cost_ticks"]
    summary = pd.DataFrame(
        [
            {
                "label": label,
                "complete_frames": len(keys),
                "actions": len(actions),
                "coverage_pct": len(actions) / len(keys) * 100 if len(keys) else np.nan,
                "hit_up_pct": actions["hit_up"].mean() * 100 if len(actions) else np.nan,
                "flat_pct": actions["flat"].mean() * 100 if len(actions) else np.nan,
                "wrong_down_pct": actions["wrong_down"].mean() * 100 if len(actions) else np.nan,
                "delta_mean": actions["delta_ticks_16s"].mean() if len(actions) else np.nan,
                "delta_median": actions["delta_ticks_16s"].median() if len(actions) else np.nan,
                "cost_median": actions["visible_entry_cost_ticks"].median() if len(actions) else np.nan,
                "net_visible_mean": actions["net_visible_ticks"].mean() if len(actions) else np.nan,
                "net_visible_median": actions["net_visible_ticks"].median() if len(actions) else np.nan,
                "net_positive_pct": actions["net_visible_ticks"].gt(0).mean() * 100 if len(actions) else np.nan,
            }
        ]
    )
    return summary, actions


def optimize_loaded_df(df):
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        if col not in ["time_index_ns", "future_time_index_ns_8s", "future_time_index_ns_16s"]:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in [
        "session_day",
        "temporality",
        "window_phase",
        "phase_bucket",
        "pm_perp_agreement",
        "pm_spot_agreement",
        "tradability_status",
        "target_3c_16s_1tick",
        "target_3c_8s_1tick",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def serialize_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    return str(v)


def main():
    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 180)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"DB: {DB} exists={DB.exists()} size_gib={DB.stat().st_size / 1024**3:.2f}", flush=True)
    with connect_ro() as conn:
        id_min, id_max = conn.execute("SELECT MIN(id), MAX(id) FROM cross_venue_features").fetchone()
        core_sessions = pd.read_sql_query(
            """
            SELECT s.id AS session_id, DATE(s.created_at_utc) AS session_day
            FROM collection_sessions s
            JOIN collection_session_telemetry t ON t.session_id = s.id
            WHERE s.dataset_tier='core'
              AND t.coverage_ratio >= 0.97
              AND t.quality_score >= 70
              AND t.core_gap_severe_segment_count = 0
            """,
            conn,
        )
    print(f"id range {id_min:,}..{id_max:,}; qualified core sessions={len(core_sessions):,}", flush=True)
    manifest = extract_cache(id_min, id_max)

    t0 = time.time()
    df = pd.read_parquet(CACHE_DIR)
    df = optimize_loaded_df(df)
    print(f"loaded cache rows={len(df):,} cols={len(df.columns)} sec={time.time() - t0:.1f}", flush=True)

    df["terminal_split"] = df["session_day"].astype(str).map(assign_terminal_split).astype("category")

    qa_split = df.groupby(["terminal_split", "temporality"], observed=True).size().rename("rows").reset_index()
    target_dist = (df["target_3c_16s_1tick"].value_counts(normalize=True) * 100).rename("pct").reset_index()
    frame_stats = df.groupby(FRAME_COLS, observed=True).agg(
        rows=("token_id", "size"),
        tokens=("token_id", "nunique"),
        sign_sum=("outcome_sign", "sum"),
        abs_sign_sum=("outcome_sign", lambda s: np.abs(s).sum()),
    ).reset_index()
    complete_frame_mask = (
        frame_stats["rows"].eq(2)
        & frame_stats["tokens"].eq(2)
        & frame_stats["sign_sum"].eq(0)
        & frame_stats["abs_sign_sum"].eq(2)
    )
    print(
        f"complete market frames={complete_frame_mask.sum():,}/{len(frame_stats):,} "
        f"({complete_frame_mask.mean() * 100:.2f}%)",
        flush=True,
    )

    folds = [
        {"fold": "F1", "train_start": "2026-05-11", "train_end": "2026-05-14", "val_start": "2026-05-15", "val_end": "2026-05-16"},
        {"fold": "F2", "train_start": "2026-05-11", "train_end": "2026-05-16", "val_start": "2026-05-17", "val_end": "2026-05-18"},
        {"fold": "F3", "train_start": "2026-05-11", "train_end": "2026-05-18", "val_start": "2026-05-19", "val_end": "2026-05-20"},
        {"fold": "F4", "train_start": "2026-05-11", "train_end": "2026-05-20", "val_start": "2026-05-21", "val_end": "2026-05-22"},
    ]

    purge_rows = []
    for fold in folds:
        tr = rows_between_days(df, fold["train_start"], fold["train_end"])
        va = rows_between_days(df, fold["val_start"], fold["val_end"])
        tr_p, removed = purge_train_conditions(tr, va)
        purge_rows.append(
            {
                **fold,
                "train_rows_before": len(tr),
                "train_rows_after_purge": len(tr_p),
                "val_rows": len(va),
                "purged_rows": removed,
                "condition_overlap": len(set(tr["condition_id"]).intersection(set(va["condition_id"]))),
            }
        )

    train_initial = df[df["terminal_split"].eq("train_initial")].copy()
    validation_initial = df[df["terminal_split"].eq("validation_initial")].copy()
    test_terminal = df[df["terminal_split"].eq("test_terminal")].copy()
    train_purged, removed_final = purge_train_conditions(train_initial, validation_initial)
    purge_rows.append(
        {
            "fold": "FINAL",
            "train_start": "2026-05-11",
            "train_end": "2026-05-20",
            "val_start": "2026-05-21",
            "val_end": "2026-05-22",
            "train_rows_before": len(train_initial),
            "train_rows_after_purge": len(train_purged),
            "val_rows": len(validation_initial),
            "purged_rows": removed_final,
            "condition_overlap": len(set(train_initial["condition_id"]).intersection(set(validation_initial["condition_id"]))),
            "test_condition_overlap_after_purge": len(set(train_purged["condition_id"]).intersection(set(test_terminal["condition_id"]))),
        }
    )
    purge_audit = pd.DataFrame(purge_rows)

    missing_by_feature_set = []
    for name, fs in FEATURE_SETS.items():
        missing_by_feature_set.append({"feature_set": name, "missing": [c for c in fs["num"] + fs["cat"] if c not in df.columns]})
    missing_features = pd.DataFrame(missing_by_feature_set)
    if missing_features["missing"].map(len).sum() > 0:
        raise RuntimeError(f"Missing features: {missing_features.to_dict(orient='records')}")

    rule_rows = []
    for split_name, data in [("validation_initial", validation_initial), ("test_terminal", test_terminal)]:
        for rule_col, label in [
            ("pm_perp_agreement", "rule_pm_perp_both_pos"),
            ("pm_spot_agreement", "rule_pm_spot_both_pos"),
        ]:
            summary, _ = exact_one_rule_eval(data, rule_col, "both_pos", label)
            summary["split"] = split_name
            rule_rows.append(summary)
    rule_summary = pd.concat(rule_rows, ignore_index=True)
    print("rule summary")
    print(rule_summary[["label", "split", "coverage_pct", "hit_up_pct", "wrong_down_pct", "net_visible_mean"]].round(4).to_string(index=False), flush=True)

    walk_rows = []
    walk_curves = []
    for fold in folds:
        raw_train = rows_between_days(df, fold["train_start"], fold["train_end"])
        val = rows_between_days(df, fold["val_start"], fold["val_end"])
        train, purged = purge_train_conditions(raw_train, val)
        print(f"\nFold {fold['fold']}: train={len(train):,} val={len(val):,} purged={purged:,}", flush=True)
        for model_kind in ["logreg_balanced", "hgb_balanced_small"]:
            model, scored_val, tm = fit_eval_model(train, val, model_kind, "full_v0")
            curve = threshold_curve(scored_val)
            curve["fold"] = fold["fold"]
            curve["model_kind"] = model_kind
            curve["feature_set"] = "full_v0"
            selected_th, _ = choose_threshold(curve, min_coverage_pct=1.0, min_actions=150)
            action_summary, _ = market_action_eval(scored_val, selected_th)
            row = {
                **tm,
                "fold": fold["fold"],
                "model_kind": model_kind,
                "feature_set": "full_v0",
                "selected_threshold": selected_th,
                "purged_rows": purged,
            }
            for col in [
                "complete_frames",
                "actions",
                "coverage_pct",
                "hit_up_pct",
                "flat_pct",
                "wrong_down_pct",
                "delta_mean",
                "delta_median",
                "cost_median",
                "net_visible_mean",
                "net_visible_median",
                "net_positive_pct",
            ]:
                row[col] = action_summary.iloc[0].get(col, np.nan)
            walk_rows.append(row)
            walk_curves.append(curve)
            print(
                f"  {model_kind}: bal_acc={tm['balanced_accuracy']:.4f} "
                f"macro_f1={tm['macro_f1']:.4f} th={selected_th:.2f} actions={row['actions']:,} "
                f"hit={row['hit_up_pct']:.2f}% wrong={row['wrong_down_pct']:.2f}% net={row['net_visible_mean']:.3f}",
                flush=True,
            )
            del model, scored_val
    walkforward_summary = pd.DataFrame(walk_rows)
    walkforward_curves = pd.concat(walk_curves, ignore_index=True)

    final_runs = [("logreg_balanced", "full_v0")]
    final_runs += [("hgb_balanced_small", fs) for fs in FEATURE_SETS.keys()]

    final_rows = []
    final_curves = []
    final_models = {}
    final_actions = {}
    final_scored_val = {}
    for model_kind, feature_set in final_runs:
        label = f"{model_kind}__{feature_set}"
        print(f"\nFinal run {label}: train={len(train_purged):,} val={len(validation_initial):,} test={len(test_terminal):,}", flush=True)
        model, scored_val, val_tm = fit_eval_model(train_purged, validation_initial, model_kind, feature_set)
        scored_test = add_model_scores(model, test_terminal, feature_set)
        test_tm = token_metrics(
            test_terminal["target_3c_16s_1tick"],
            scored_test["pred"],
            scored_test["score_buy"],
            test_terminal["delta_ticks_16s"],
        )
        curve = threshold_curve(scored_val)
        curve["model_kind"] = model_kind
        curve["feature_set"] = feature_set
        curve["split"] = "validation_initial"
        selected_th, selected = choose_threshold(curve, min_coverage_pct=1.0, min_actions=200, max_wrong_down_pct=35.0)
        val_summary, _ = market_action_eval(scored_val, selected_th)
        test_summary, test_actions = market_action_eval(scored_test, selected_th)
        row = {
            "model_kind": model_kind,
            "feature_set": feature_set,
            "selected_threshold": selected_th,
            "validation_selection_score": selected.get("selection_score", np.nan),
            "fit_seconds": val_tm["fit_seconds"],
            "val_n": val_tm["n"],
            "val_accuracy": val_tm["accuracy"],
            "val_balanced_accuracy": val_tm["balanced_accuracy"],
            "val_macro_f1": val_tm["macro_f1"],
            "val_score_delta_spearman": val_tm.get("score_delta_spearman", np.nan),
            "test_n": test_tm["n"],
            "test_accuracy": test_tm["accuracy"],
            "test_balanced_accuracy": test_tm["balanced_accuracy"],
            "test_macro_f1": test_tm["macro_f1"],
            "test_score_delta_spearman": test_tm.get("score_delta_spearman", np.nan),
        }
        for prefix, summary in [("val_action", val_summary), ("test_action", test_summary)]:
            s = summary.iloc[0]
            for col in [
                "complete_frames",
                "actions",
                "coverage_pct",
                "hit_up_pct",
                "flat_pct",
                "wrong_down_pct",
                "delta_mean",
                "delta_median",
                "cost_median",
                "net_visible_mean",
                "net_visible_median",
                "net_positive_pct",
                "score_median",
            ]:
                row[f"{prefix}_{col}"] = s.get(col, np.nan)
        final_rows.append(row)
        final_curves.append(curve)
        final_models[label] = model
        final_actions[label] = test_actions
        final_scored_val[label] = scored_val
        print(
            f"  val_bal={row['val_balanced_accuracy']:.4f} test_bal={row['test_balanced_accuracy']:.4f} "
            f"th={selected_th:.2f} test_actions={row['test_action_actions']:,} "
            f"hit={row['test_action_hit_up_pct']:.2f}% wrong={row['test_action_wrong_down_pct']:.2f}% "
            f"net={row['test_action_net_visible_mean']:.3f}",
            flush=True,
        )
        del scored_test

    final_summary = pd.DataFrame(final_rows).sort_values(
        ["test_action_net_visible_mean", "test_action_actions"], ascending=[False, False]
    ).reset_index(drop=True)
    final_validation_curves = pd.concat(final_curves, ignore_index=True)

    candidates = final_summary[final_summary["test_action_actions"].ge(500)].copy()
    if candidates.empty:
        candidates = final_summary[final_summary["test_action_actions"].ge(200)].copy()
    if candidates.empty:
        candidates = final_summary.copy()
    best_final = candidates.sort_values(
        ["test_action_net_visible_mean", "test_action_wrong_down_pct"], ascending=[False, True]
    ).iloc[0]
    best_label = f"{best_final['model_kind']}__{best_final['feature_set']}"
    best_actions = final_actions[best_label].copy()
    print(f"\nBest label: {best_label}", flush=True)

    seg_rows = []
    if len(best_actions):
        best_actions["cost_bucket"] = pd.qcut(
            best_actions["visible_entry_cost_ticks"].rank(method="first"), 3, labels=["low_cost", "mid_cost", "high_cost"]
        )
        best_actions["score_bucket"] = pd.qcut(
            best_actions["score_buy"].rank(method="first"), 4, labels=["q1_low_score", "q2", "q3", "q4_high_score"]
        )
        for seg_col in [
            "temporality",
            "window_phase",
            "phase_bucket",
            "cost_bucket",
            "score_bucket",
            "tradability_status",
            "practical_dead_early",
        ]:
            for seg, part in best_actions.groupby(seg_col, observed=True, dropna=False):
                if len(part) < 30:
                    continue
                seg_rows.append(
                    {
                        "segment_type": seg_col,
                        "segment": str(seg),
                        "actions": len(part),
                        "coverage_within_actions_pct": len(part) / len(best_actions) * 100,
                        "hit_up_pct": part["hit_up"].mean() * 100,
                        "flat_pct": part["flat"].mean() * 100,
                        "wrong_down_pct": part["wrong_down"].mean() * 100,
                        "delta_mean": part["delta_ticks_16s"].mean(),
                        "delta_median": part["delta_ticks_16s"].median(),
                        "cost_median": part["visible_entry_cost_ticks"].median(),
                        "net_visible_mean": part["net_visible_ticks"].mean(),
                        "net_positive_pct": part["net_visible_ticks"].gt(0).mean() * 100,
                    }
                )
    segments = pd.DataFrame(seg_rows)
    if not segments.empty:
        segments = segments.sort_values(["segment_type", "actions"], ascending=[True, False])

    importance_rows = []
    logreg_label = "logreg_balanced__full_v0"
    if logreg_label in final_models:
        model = final_models[logreg_label]
        try:
            names = model.named_steps["prep"].get_feature_names_out()
            clf = model.named_steps["clf"]
            for class_idx, cls in enumerate(clf.classes_):
                coef = clf.coef_[class_idx]
                top_idx = np.argsort(np.abs(coef))[-25:][::-1]
                for i in top_idx:
                    importance_rows.append(
                        {
                            "source": "logreg_coef_abs_top",
                            "class": cls,
                            "feature": names[i],
                            "importance": float(coef[i]),
                            "abs_importance": float(abs(coef[i])),
                        }
                    )
        except Exception as exc:
            print(f"Could not extract logreg coefficients: {exc!r}", flush=True)
    top_logreg = pd.DataFrame(importance_rows).sort_values("abs_importance", ascending=False) if importance_rows else pd.DataFrame()

    perm_df = pd.DataFrame()
    if RUN_PERMUTATION_IMPORTANCE and best_label in final_models:
        model = final_models[best_label]
        fs = FEATURE_SETS[best_final["feature_set"]]
        features = fs["num"] + fs["cat"]
        sample_n = min(PERMUTATION_SAMPLE_N, len(validation_initial))
        sample = validation_initial.sample(sample_n, random_state=RANDOM_STATE)
        print(f"Permutation importance on {sample_n:,} validation rows for {best_label}", flush=True)
        t0 = time.time()
        perm = permutation_importance(
            model,
            sample[features],
            sample["target_3c_16s_1tick"],
            scoring="balanced_accuracy",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        perm_df = pd.DataFrame(
            {"feature": features, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}
        ).sort_values("importance_mean", ascending=False)
        print(f"Permutation seconds={time.time() - t0:.1f}", flush=True)

    qa_split.to_csv(RESULTS_DIR / "qa_split_rows.csv", index=False)
    target_dist.to_csv(RESULTS_DIR / "target_distribution.csv", index=False)
    purge_audit.to_csv(RESULTS_DIR / "purge_audit.csv", index=False)
    rule_summary.to_csv(RESULTS_DIR / "rule_summary.csv", index=False)
    walkforward_summary.to_csv(RESULTS_DIR / "walkforward_summary.csv", index=False)
    walkforward_curves.to_csv(RESULTS_DIR / "walkforward_curves.csv", index=False)
    final_summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)
    final_validation_curves.to_csv(RESULTS_DIR / "final_validation_curves.csv", index=False)
    segments.to_csv(RESULTS_DIR / "best_model_segments.csv", index=False)
    if not top_logreg.empty:
        top_logreg.to_csv(RESULTS_DIR / "logreg_top_coefficients.csv", index=False)
    if not perm_df.empty:
        perm_df.to_csv(RESULTS_DIR / "best_model_permutation_importance.csv", index=False)

    summary_payload = {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)),
        "complete_market_frames": int(complete_frame_mask.sum()),
        "market_frames": int(len(frame_stats)),
        "cache_dir": str(CACHE_DIR),
        "results_dir": str(RESULTS_DIR),
        "best_label": best_label,
        "best_final": {k: serialize_value(v) for k, v in best_final.to_dict().items()},
        "notebook_wall_seconds": time.time() - NOTEBOOK_START_TS,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("\nBASELINE V0 FULL-CORE ROBUSTNESS SUMMARY", flush=True)
    print(f"- rows: {len(df):,}", flush=True)
    print(f"- complete market frames: {complete_frame_mask.sum():,}/{len(frame_stats):,} ({complete_frame_mask.mean() * 100:.2f}%)", flush=True)
    print("\nTarget distribution:", flush=True)
    print(target_dist.round(4).to_string(index=False), flush=True)
    print("\nFinal ablations:", flush=True)
    print(
        final_summary[
            [
                "model_kind",
                "feature_set",
                "selected_threshold",
                "test_balanced_accuracy",
                "test_macro_f1",
                "test_action_actions",
                "test_action_coverage_pct",
                "test_action_hit_up_pct",
                "test_action_wrong_down_pct",
                "test_action_net_visible_mean",
                "test_action_net_positive_pct",
            ]
        ]
        .round(4)
        .to_string(index=False),
        flush=True,
    )
    print(f"\nBest label: {best_label}", flush=True)
    return summary_payload


if __name__ == "__main__":
    main()
