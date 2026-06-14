"""Ingeniería de características.

El especialista final usa el conjunto ``no_clock`` (36 features, sin variables
de reloj para evitar fuga temporal trivial). La definición operativa de los
conjuntos de features vive en los scripts del arco
(``baseline_v0_full_core_robustness.py`` define ``FEATURE_SETS``); este
subpaquete queda como punto de extensión para reutilizarla.
"""
