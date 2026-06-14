"""Tests de la lógica congelada extraída a ``edgerunner``.

Validan que el paquete reproduce fielmente la política, los gates y el modelo de
costes del script de paper shadow (la fuente de verdad). No reentrenan ni
recalibran nada.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edgerunner.config import load_frozen_policy
from edgerunner.eval import costs, gates
from edgerunner.models import policy


def test_frozen_policy_thresholds():
    p = load_frozen_policy()
    assert p.ev_pred_min == 0.75
    assert p.hp_pred_min == 0.50
    assert p.vol_gate_max == 0.6657
    assert p.horizon_seconds == 60
    assert p.reference_cost_ticks == 0.5


def test_policy_gates_select_expected_rows():
    df = pd.DataFrame(
        {
            "ev_pred": [0.80, 0.70, 0.90, 0.76],
            "hp_pred": [0.60, 0.99, 0.40, 0.50],
            "perp_realized_vol_bps_5s": [0.10, 0.10, 0.10, 0.70],
        }
    )
    # fila 0 pasa; 1 falla EV; 2 falla salud; 3 falla vol gate.
    selected = policy.apply_policy_gates(df)
    assert list(selected.index) == [0]


def test_decide_matches_arc_states():
    # Candidato final: 5 días, 318 trades, +1.069 -> faltan días para paper shadow.
    assert gates.decide(n_days=5, n_trades=318, net=1.069, n_losses=130) == "PROMISING_NEEDS_DATA"
    # Soporte suficiente -> paper shadow candidate.
    assert gates.decide(n_days=8, n_trades=250, net=0.60, n_losses=90) == "PAPER_SHADOW_CANDIDATE"
    # Neto no positivo con soporte -> revisar política.
    assert gates.decide(n_days=4, n_trades=100, net=-0.2, n_losses=60) == "REVIEW_POLICY"
    # Muy pocos días -> datos insuficientes.
    assert gates.decide(n_days=2, n_trades=50, net=1.0, n_losses=10) == "INSUFFICIENT_DATA"


def test_bootstrap_is_reproducible_and_positive():
    rng = np.random.default_rng(0)
    outcomes = rng.normal(loc=1.0, scale=2.0, size=300)
    s1 = costs.summarize(outcomes)
    s2 = costs.summarize(outcomes)
    assert s1.ci_low == s2.ci_low and s1.ci_high == s2.ci_high  # semilla fija
    assert s1.ci_low < s1.net_mean < s1.ci_high
    assert 0.0 <= s1.p_positive <= 100.0


def test_max_drawdown_simple():
    # PnL acumulado que sube a 10, baja a 4 -> drawdown 6.
    assert costs.max_drawdown([0, 5, 10, 7, 4, 8]) == 6.0
