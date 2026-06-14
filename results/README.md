# results/

Estos dos CSV están **filtrados al arco principal del proyecto** (la escalera de
modelos que cuenta la memoria). El registro interno completo del proyecto era
mucho más extenso (cientos de experimentos, incluyendo ramas exploratorias que
no llegaron a la entrega); aquí se conserva solo la traza del arco.

- `key_results.csv` — métricas clave por experimento del arco
  (`phase, experiment, metric, value, unit, status, interpretation`).
  Todas las métricas económicas están en **ticks netos**, nunca en dólares.
- `decision_register.csv` — decisiones consecutivas asociadas a los documentos
  publicados (`order, phase, artifact, decision, public_role, reason`).

Las ramas exploratorias descartadas (maker, large-edge intraday, settlement
scouts, etc.) se resumen como negativos en la memoria, no se incluyen aquí.
