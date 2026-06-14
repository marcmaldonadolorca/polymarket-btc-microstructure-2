"""Política congelada del especialista prestart H60 + vol gate.

Extraído de ``scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py``
(fuente de verdad). Aquí se expone de forma importable y testeable. La política
NO se reentrena ni se reajusta: se congeló antes de ver los datos fresh.

Dos etapas:
  1. Filtro estructural ``strict_45_60_early``: solo señales de la ventana
     prestart entre -60 s y -45 s antes de la apertura, con libro sano y coste
     visible acotado.
  2. Gates de política: EV (ev_pred > 0.75), salud (hp_pred >= 0.50) y
     volatilidad (perp_realized_vol_bps_5s <= 0.6657).
"""

from __future__ import annotations

import pandas as pd

from edgerunner.config import FrozenPolicy, load_frozen_policy


def apply_strict_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Filtro ``strict_45_60_early``: ventana prestart y calidad de libro.

    Reproduce exactamente el filtro del script de paper shadow.
    """
    return df[
        (df["window_phase"] == "prestart")
        & (df["target_supported_H60"] == True)  # noqa: E712 (columna booleana de pandas)
        & (df["spread_ticks"] <= 2.0)
        & (df["visible_entry_cost_ticks"] <= 1.25)
        & (df["age_ms"] <= 1000.0)
        & (df["full_book_ratio"] >= 0.999)
        & (df["degraded_ratio"] == 0)
        & (df["seconds_from_window_start"] >= -60.0)
        & (df["seconds_from_window_start"] < -45.0)
    ].copy()


def apply_policy_gates(
    df: pd.DataFrame,
    *,
    ev_col: str = "ev_pred",
    hp_col: str = "hp_pred",
    vol_col: str = "perp_realized_vol_bps_5s",
    policy: FrozenPolicy | None = None,
) -> pd.DataFrame:
    """Selecciona las filas que pasan EV, salud y vol gate (congelados)."""
    p = policy or load_frozen_policy()
    return df[
        (df[ev_col] > p.ev_pred_min)
        & (df[hp_col] >= p.hp_pred_min)
        & (df[vol_col] <= p.vol_gate_max)
    ].copy()


def select_trades(
    df: pd.DataFrame, *, policy: FrozenPolicy | None = None, **cols: str
) -> pd.DataFrame:
    """Pipeline completo: filtro estructural + gates de política."""
    return apply_policy_gates(apply_strict_filters(df), policy=policy, **cols)
