"""Modelo de costes en ticks, drawdown y bootstrap.

Extraído de ``scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py``.
Todo en ticks netos; nunca en dólares. El neto por trade ya incorpora el coste
de entrada (columna ``exec_net_cost_0p5_H60`` = neto con coste 0.5 ticks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 42        # reproducibilidad exacta del script de paper shadow
BOOTSTRAP_RESAMPLES = 5000


@dataclass(frozen=True)
class NetSummary:
    n_trades: int
    net_mean: float          # neto medio por trade (ticks)
    net_total: float         # ticks netos acumulados
    max_drawdown: float      # ticks
    ci_low: float            # IC bootstrap 5%
    ci_high: float           # IC bootstrap 95%
    p_positive: float        # P(neto medio > 0) por bootstrap, en %


def max_drawdown(cumulative_pnl) -> float:
    """Máximo drawdown sobre el PnL acumulado (en ticks)."""
    cum = list(cumulative_pnl)
    if not cum:
        return 0.0
    peak = cum[0]
    mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = max(mdd, peak - v)
    return mdd


def net_at_cost(net_outcomes, extra_cost_ticks: float = 0.0) -> float:
    """Neto medio aplicando un coste adicional uniforme (p. ej. 0.25 vs 0.5)."""
    arr = np.asarray(net_outcomes, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr - extra_cost_ticks))


def bootstrap_ci(net_outcomes, *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED):
    """IC bootstrap (5%, 95%) y P(media > 0), idéntico al script de paper shadow."""
    arr = np.asarray(net_outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(resamples)])
    return float(np.percentile(boot, 5)), float(np.percentile(boot, 95)), float((boot > 0).mean() * 100)


def summarize(net_outcomes) -> NetSummary:
    """Resumen completo del PnL en ticks de una serie de trades."""
    arr = np.asarray(net_outcomes, dtype=float)
    ci_low, ci_high, p_pos = bootstrap_ci(arr)
    return NetSummary(
        n_trades=int(arr.size),
        net_mean=float(np.mean(arr)) if arr.size else 0.0,
        net_total=float(np.sum(arr)),
        max_drawdown=max_drawdown(np.cumsum(arr)),
        ci_low=ci_low,
        ci_high=ci_high,
        p_positive=p_pos,
    )
