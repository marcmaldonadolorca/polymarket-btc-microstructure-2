"""Build the public, anonymised evidence used by the final report.

This script deliberately separates two scopes:

* ``base_oos``: the specialist and its EV/health thresholds, frozen before the
  6--10 June block;
* ``low_vol_posthoc``: the volatility diagnostic selected after inspecting that
  block.  It is useful evidence, but it is not a confirmatory OOS result.

The private corpus and fitted models are not published.  Pass the root of the
private research workspace explicitly; the generated action ledger contains
only hashed market identifiers and the outcome columns required to reproduce
the report tables.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EV_THRESHOLD = 0.75
HEALTH_THRESHOLD = 0.50
VOL_THRESHOLD = 0.6657
BOOTSTRAP_SEED = 42
N_BOOTSTRAP = 5_000

COST_COLUMNS = {
    "cost_0p25": "exec_net_cost_0p25_H60",
    "cost_0p5": "exec_net_cost_0p5_H60",
    "cost_1p0": "exec_net_cost_1p0_H60",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Private edgerunner workspace containing data/experiments.",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Public repository receiving results and report figures.",
    )
    return parser.parse_args()


def strict_prestart(rows: pd.DataFrame) -> pd.DataFrame:
    return rows[
        (rows["window_phase"] == "prestart")
        & (rows["target_supported_H60"] == True)
        & (rows["spread_ticks"] <= 2.0)
        & (rows["visible_entry_cost_ticks"] <= 1.25)
        & (rows["age_ms"] <= 1_000.0)
        & (rows["full_book_ratio"] >= 0.999)
        & (rows["degraded_ratio"] == 0)
        & (rows["seconds_from_window_start"] >= -60.0)
        & (rows["seconds_from_window_start"] < -45.0)
    ].copy()


def cast_categoricals(rows: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = rows.copy()
    for column in feature_columns:
        dtype = rows[column].dtype
        if pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            rows[column] = rows[column].astype("category")
    return rows


def short_hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def iid_bootstrap(values: np.ndarray) -> tuple[float, float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.asarray(
        [rng.choice(values, size=len(values), replace=True).mean() for _ in range(N_BOOTSTRAP)]
    )
    low, high = np.quantile(means, [0.05, 0.95])
    return float(low), float(high), float((means > 0).mean())


def cluster_bootstrap(rows: pd.DataFrame, group_column: str) -> tuple[float, float, float]:
    grouped = rows.groupby(group_column, sort=False)[COST_COLUMNS["cost_0p5"]]
    sums = grouped.sum().to_numpy(dtype=float)
    counts = grouped.size().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(N_BOOTSTRAP, dtype=float)
    for index in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(sums), size=len(sums))
        means[index] = sums[sampled].sum() / counts[sampled].sum()
    low, high = np.quantile(means, [0.05, 0.95])
    return float(low), float(high), float((means > 0).mean())


def drawdown(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])[1:]
    dd = cumulative - running_peak
    return cumulative, dd, float(-dd.min())


def configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
        }
    )


def save_figures(low: pd.DataFrame, high: pd.DataFrame, figures: Path) -> None:
    configure_plotting()
    figures.mkdir(parents=True, exist_ok=True)
    teal, coral, grey, dark = "#2A9D8F", "#E76F51", "#BFC3C9", "#264653"

    # Cost sensitivity of the post-hoc diagnostic.
    cost_values = [float(low[column].mean()) for column in COST_COLUMNS.values()]
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    bars = ax.bar(["0,25\n(optimista)", "0,5\n(referencia)", "1,0\n(estrés)"], cost_values,
                  color=[grey, teal, grey], width=0.58)
    ax.set_title("Diagnóstico post-hoc de baja volatilidad: sensibilidad al coste")
    ax.set_ylabel("Neto medio (ticks)")
    ax.set_xlabel("Nivel de coste")
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in cost_values], padding=3)
    ax.set_ylim(0, max(cost_values) * 1.25)
    fig.tight_layout()
    fig.savefig(figures / "fig_coste.png", bbox_inches="tight")
    plt.close(fig)

    # Daily means of the same diagnostic subset.
    daily = low.groupby("session_day", sort=True)[COST_COLUMNS["cost_0p5"]].mean()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    bars = ax.bar([str(day)[5:] for day in daily.index], daily.values, color=teal, width=0.58)
    ax.set_title("Diagnóstico post-hoc: neto medio por día (5/5 positivos)")
    ax.set_ylabel("Neto medio @0,5 (ticks)")
    ax.set_xlabel("Día (jun 2026)")
    ax.bar_label(bars, labels=[f"{value:+.2f}" for value in daily.values], padding=3)
    ax.set_ylim(0, max(daily.values) * 1.22)
    fig.tight_layout()
    fig.savefig(figures / "fig_neto_diario.png", bbox_inches="tight")
    plt.close(fig)

    # Diagnostic cumulative action units; not portfolio PnL.
    cumulative, dd, maximum_dd = drawdown(low[COST_COLUMNS["cost_0p5"]].to_numpy())
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.2, 4.7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    top.plot(cumulative, color=dark, linewidth=1.7)
    top.axhline(0, color="#777777", linewidth=0.8)
    top.fill_between(np.arange(len(cumulative)), 0, cumulative, color=dark, alpha=0.08)
    top.set_ylabel("Suma acumulada\n(ticks)")
    top.set_title("Suma de unidades de acción y drawdown diagnóstico (n=318)")
    bottom.fill_between(np.arange(len(dd)), dd, 0, color=coral, alpha=0.65)
    bottom.set_ylabel("Drawdown")
    bottom.set_xlabel("Señal (orden cronológico)")
    worst = int(np.argmin(dd))
    bottom.annotate(
        f"máx = {maximum_dd:.1f} ticks",
        xy=(worst, dd[worst]),
        xytext=(max(0, worst - 55), dd.min() * 0.72),
        arrowprops={"arrowstyle": "->", "color": coral},
        color=coral,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(figures / "fig_drawdown.png", bbox_inches="tight")
    plt.close(fig)

    # Regime contrast on the inspected fresh block.
    regime_values = [
        float(low[COST_COLUMNS["cost_0p5"]].mean()),
        float(high[COST_COLUMNS["cost_0p5"]].mean()),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    bars = ax.bar(
        [f"Baja volatilidad\n(n={len(low)})", f"Alta volatilidad\n(n={len(high)})"],
        regime_values,
        color=[teal, coral],
        width=0.58,
    )
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_title("Diagnóstico post-hoc por régimen de volatilidad")
    ax.set_ylabel("Neto medio @0,5 (ticks)")
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in regime_values], padding=3)
    ax.set_ylim(min(regime_values) - 0.25, max(regime_values) + 0.28)
    fig.tight_layout()
    fig.savefig(figures / "fig_regimen_neto.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source = args.source_root.resolve()
    public = args.public_root.resolve()
    experiment_root = source / "data" / "experiments"
    dataset = (
        experiment_root
        / "large_edge_multiresolution_intraday_fresh_execution_prepare_v1"
        / "fresh_execution_dataset"
        / "complex_v1a_execution_dataset.parquet"
    )
    specialist = experiment_root / "complex_v1a_prestart_h60_specialist_v1"
    manifest = json.loads((specialist / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["feature_sets"]["no_clock"]

    rows = strict_prestart(pd.read_parquet(dataset))
    rows = cast_categoricals(rows, features)
    ev_model = joblib.load(
        specialist / "models" / "specialist_ev_strict_45_60_early_no_clock.joblib"
    )
    health_model = joblib.load(
        specialist / "models" / "specialist_healthy_strict_45_60_early_no_clock.joblib"
    )
    rows["ev_pred"] = ev_model.predict(rows[features])
    rows["health_pred"] = health_model.predict_proba(rows[features])[:, 1]
    base = rows[(rows["ev_pred"] > EV_THRESHOLD) & (rows["health_pred"] >= HEALTH_THRESHOLD)].copy()

    volatility = base["perp_realized_vol_bps_5s"]
    base["volatility_regime"] = np.select(
        [volatility <= VOL_THRESHOLD, volatility > VOL_THRESHOLD],
        ["low", "high"],
        default="missing",
    )
    low = base[base["volatility_regime"] == "low"].copy()
    high = base[base["volatility_regime"] == "high"].copy()
    missing = base[base["volatility_regime"] == "missing"].copy()

    # Anonymised action ledger sufficient to recompute every public aggregate.
    ledger = pd.DataFrame(
        {
            "action_id": np.arange(1, len(base) + 1),
            "session_day": base["session_day"].astype(str),
            "market_hash": base["condition_id"].map(short_hash),
            "token_hash": base["token_id"].map(short_hash),
            "seconds_from_window_start": base["seconds_from_window_start"].round(3),
            "volatility_regime": base["volatility_regime"],
            "net_cost_0p25": base[COST_COLUMNS["cost_0p25"]].round(6),
            "net_cost_0p5": base[COST_COLUMNS["cost_0p5"]].round(6),
            "net_cost_1p0": base[COST_COLUMNS["cost_1p0"]].round(6),
        }
    )
    results = public / "results"
    results.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(results / "final_candidate_actions_anonymized.csv", index=False)

    iid_low, iid_high, iid_probability = iid_bootstrap(low[COST_COLUMNS["cost_0p5"]].to_numpy())
    cluster_low, cluster_high, cluster_probability = cluster_bootstrap(low, "condition_id")
    _, _, maximum_dd = drawdown(low[COST_COLUMNS["cost_0p5"]].to_numpy())
    base_daily = base.groupby("session_day")[COST_COLUMNS["cost_0p5"]].mean()
    low_daily = low.groupby("session_day")[COST_COLUMNS["cost_0p5"]].mean()
    market_counts = low.groupby("condition_id").size()

    summary_rows: list[dict[str, object]] = []

    def metric(scope: str, name: str, value: object, unit: str) -> None:
        summary_rows.append({"scope": scope, "metric": name, "value": value, "unit": unit})

    for scope, subset in [
        ("base_oos", base),
        ("low_vol_posthoc", low),
        ("high_vol_posthoc", high),
        ("missing_volatility", missing),
    ]:
        metric(scope, "n_actions", len(subset), "actions")
        for label, column in COST_COLUMNS.items():
            metric(scope, f"mean_net_{label}", float(subset[column].mean()), "ticks/action")
    metric("base_oos", "positive_days_cost_0p5", int((base_daily > 0).sum()), "days")
    metric("base_oos", "n_days", int(base_daily.size), "days")
    metric("low_vol_posthoc", "positive_days_cost_0p5", int((low_daily > 0).sum()), "days")
    metric("low_vol_posthoc", "n_days", int(low_daily.size), "days")
    metric("low_vol_posthoc", "iid_bootstrap_ci90_low", iid_low, "ticks/action")
    metric("low_vol_posthoc", "iid_bootstrap_ci90_high", iid_high, "ticks/action")
    metric("low_vol_posthoc", "iid_bootstrap_p_positive", iid_probability, "probability")
    metric("low_vol_posthoc", "market_cluster_ci90_low", cluster_low, "ticks/action")
    metric("low_vol_posthoc", "market_cluster_ci90_high", cluster_high, "ticks/action")
    metric("low_vol_posthoc", "market_cluster_p_positive", cluster_probability, "probability")
    metric("low_vol_posthoc", "diagnostic_max_drawdown", maximum_dd, "ticks")
    metric("low_vol_posthoc", "unique_markets", int(market_counts.size), "markets")
    metric("low_vol_posthoc", "max_actions_per_market", int(market_counts.max()), "actions/market")
    pd.DataFrame(summary_rows).to_csv(results / "final_candidate_summary.csv", index=False)

    save_figures(low, high, public / "reports" / "memoria" / "figures")
    print(f"Wrote {len(ledger)} anonymised actions and {len(summary_rows)} summary metrics.")
    print(f"Base OOS @0.5: n={len(base)}, mean={base[COST_COLUMNS['cost_0p5']].mean():+.3f}")
    print(f"Post-hoc low-vol @0.5: n={len(low)}, mean={low[COST_COLUMNS['cost_0p5']].mean():+.3f}")


if __name__ == "__main__":
    main()
