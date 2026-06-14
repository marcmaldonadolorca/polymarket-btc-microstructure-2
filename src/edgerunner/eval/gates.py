"""Gates de decisión cuantitativos hacia un posible bot.

Extraído de ``scripts/experiments/complex_v1a_prestart_h60_paper_shadow_v1.py``.
Los gates son deliberadamente conservadores: exigen soporte (días y trades) y no
solo un neto positivo. El candidato actual queda en PROMISING_NEEDS_DATA por
falta de días, no por falta de señal.
"""

from __future__ import annotations

# Umbrales de los gates (congelados).
PAPER_SHADOW_MIN_NET = 0.50
PAPER_SHADOW_MIN_TRADES = 200
PAPER_SHADOW_MIN_DAYS = 8

BOT_CANDIDATE_MIN_NET = 0.50
BOT_CANDIDATE_MIN_TRADES = 400
BOT_CANDIDATE_MIN_DAYS = 12
BOT_CANDIDATE_MAX_LOSS_RATE = 0.10


def decide(n_days: int, n_trades: int, net: float, n_losses: int, worst_day: float = 0.0) -> str:
    """Decisión de promoción. Réplica exacta del script de paper shadow.

    Devuelve uno de: INSUFFICIENT_DATA, REVIEW_POLICY, BOT_CANDIDATE,
    PAPER_SHADOW_CANDIDATE, PROMISING_NEEDS_DATA.
    """
    if n_days < 3:
        return "INSUFFICIENT_DATA"
    if net <= 0.0 and n_days >= 3:
        return "REVIEW_POLICY"
    if (
        n_days >= BOT_CANDIDATE_MIN_DAYS
        and n_trades >= BOT_CANDIDATE_MIN_TRADES
        and net >= BOT_CANDIDATE_MIN_NET
    ):
        if n_losses == 0 or (n_losses / n_trades) < BOT_CANDIDATE_MAX_LOSS_RATE:
            return "BOT_CANDIDATE"
    if (
        n_days >= PAPER_SHADOW_MIN_DAYS
        and n_trades >= PAPER_SHADOW_MIN_TRADES
        and net >= PAPER_SHADOW_MIN_NET
    ):
        return "PAPER_SHADOW_CANDIDATE"
    if net > 0.0:
        return "PROMISING_NEEDS_DATA"
    return "REVIEW_POLICY"
