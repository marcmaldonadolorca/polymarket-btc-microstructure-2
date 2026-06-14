"""Acceso a datos y construcción del dataset de ejecución.

El corpus completo (fuente SQLite del recolector, ~93 GB, y las cachés en
Parquet) no se publica por tamaño y privacidad. Los builders reproducibles del
dataset viven en ``scripts/experiments/`` (p. ej.
``baseline_v03_dataset_contract.py``, ``complex_v1a_prestart_h60_specialist_v1.py``)
y el esquema de la consulta base está en ``sql/core_training_base.sql``.

Este subpaquete queda como punto de extensión para la lógica de carga
compartida cuando se trabaje sobre datos equivalentes.
"""
