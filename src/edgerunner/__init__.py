"""edgerunner — microestructura y predicción a corto plazo en Polymarket BTC.

Paquete de apoyo del TFM. Recoge la lógica compartida del arco principal
(política congelada, modelo de costes, gates de decisión) de forma importable y
testeable. Los experimentos completos viven en ``scripts/experiments/``; este
paquete extrae la parte reutilizable y la documenta.

NO es un sistema de trading en producción. Todos los resultados se expresan en
ticks netos, nunca en dólares, y la validación fresh tiene un soporte de 5 días.
"""

from __future__ import annotations

__version__ = "0.1.0"

from edgerunner import config  # noqa: F401
from edgerunner.eval import costs, gates  # noqa: F401
from edgerunner.models import policy  # noqa: F401

__all__ = ["config", "costs", "gates", "policy", "__version__"]
