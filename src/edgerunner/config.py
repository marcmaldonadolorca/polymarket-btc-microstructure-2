"""Rutas del repositorio y carga de la política congelada.

La fuente de verdad de los umbrales es ``config/frozen_policy_thresholds.yaml``.
Aquí solo se leen; no se modifican en código.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Raíz del repositorio (este fichero vive en src/edgerunner/).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FROZEN_POLICY_PATH = CONFIG_DIR / "frozen_policy_thresholds.yaml"


@dataclass(frozen=True)
class FrozenPolicy:
    """Umbrales congelados del especialista prestart H60 + vol gate.

    Los umbrales EV/HP se fijaron antes del bloque fresh. El vol gate se
    incorporó tras diagnosticar el bloque 6-10 jun 2026 y queda congelado para
    observaciones posteriores.
    Reflejan ``config/frozen_policy_thresholds.yaml`` y los scripts del arco.
    """

    horizon_seconds: int = 60
    ev_pred_min: float = 0.75          # ev_pred > ev_pred_min
    hp_pred_min: float = 0.50          # hp_pred >= hp_pred_min
    vol_gate_max: float = 0.6657       # perp_realized_vol_bps_5s <= vol_gate_max (q80 training)
    reference_cost_ticks: float = 0.5  # coste conservador de referencia
    secondary_cost_ticks: float = 0.25 # coste optimista de contraste
    latency_contract_seconds: int = 2  # contrato conservador (p95 medido ~0.43 s)


def load_frozen_policy() -> FrozenPolicy:
    """Devuelve la política congelada.

    Si PyYAML está disponible, lee el YAML; si no, usa los valores por defecto,
    que son idénticos a los del fichero (la política es inmutable).
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return FrozenPolicy()

    if not FROZEN_POLICY_PATH.exists():
        return FrozenPolicy()

    data = yaml.safe_load(FROZEN_POLICY_PATH.read_text(encoding="utf-8"))
    specialist = data.get("specialist", {})
    policy = data.get("policy", {})
    vol = data.get("vol_gate", {})
    cost = data.get("cost_model", {})
    return FrozenPolicy(
        horizon_seconds=int(specialist.get("horizon_seconds", 60)),
        ev_pred_min=float(policy.get("ev_pred_min", 0.75)),
        hp_pred_min=float(policy.get("hp_pred_min", 0.50)),
        vol_gate_max=float(vol.get("max", 0.6657)),
        reference_cost_ticks=float(cost.get("reference_cost_ticks", 0.5)),
        secondary_cost_ticks=float(cost.get("secondary_cost_ticks", 0.25)),
        latency_contract_seconds=int(cost.get("latency_contract_seconds", 2)),
    )
