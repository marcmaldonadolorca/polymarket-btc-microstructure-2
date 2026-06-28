# Validación fresh y diagnóstico de régimen

Fecha de corte: 2026-06-10.

## 1. Qué estaba congelado

Antes del bloque del 6 al 10 de junio estaban congelados:

- los dos modelos HGB `no_clock`;
- el universo `strict_45_60_early`;
- `ev_pred > 0.75`;
- `hp_pred >= 0.50`.

La evaluación de esa política base, sin filtros añadidos después, es la única
estimación OOS confirmatoria de este documento.

## 2. Resultado OOS limpio

| Alcance | n | @0.25 | @0.5 | @1.0 | Días positivos @0.5 |
|---|---:|---:|---:|---:|---:|
| Especialista base | 754 | +0.537 | +0.349 | -0.026 | 3/5 |

Lectura: existe señal parcial al coste de referencia, pero no robustez bajo coste
completo ni consistencia diaria suficiente.

## 3. Diagnóstico post-hoc

Al inspeccionar el bloque se detectó un desplazamiento de volatilidad. Se analizó
el umbral `perp_realized_vol_bps_5s <= 0.6657`, que corresponde al percentil 80
del entrenamiento. El valor numérico procede de datos anteriores, pero la
decisión de incorporarlo como filtro se tomó después de ver el bloque fresh.

| Régimen | n | @0.25 | @0.5 | @1.0 |
|---|---:|---:|---:|---:|
| Baja volatilidad | 318 | +1.258 | +1.069 | +0.691 |
| Alta volatilidad | 435 | +0.004 | -0.183 | -0.558 |
| Volatilidad ausente | 1 | +3.327 | +3.154 | +2.807 |

El contraste es prometedor, pero es **descriptivo**. No debe etiquetarse como una
segunda validación OOS ni como prueba de rentabilidad.

## 4. Incertidumbre y dependencia

Las 318 unidades de baja volatilidad pertenecen a 156 mercados y hay hasta ocho
señales H60 solapadas por mercado.

| Remuestreo | IC90 @0.5 | P(media > 0) |
|---|---:|---:|
| IID por fila | [+0.472, +1.654] | 99.8% |
| Agrupado por mercado | [+0.193, +2.004] | 97.6% |

Ambos intervalos describen el subconjunto observado; la selección post-hoc impide
una interpretación confirmatoria.

## 5. Decisión

```text
POSTHOC_DIAGNOSTIC_PENDING_PROSPECTIVE_VALIDATION
```

La política completa queda congelada después del 10 de junio. Solo fechas
posteriores, sin retocar umbrales, pueden validarla.

Artefactos públicos:

- `results/final_candidate_actions_anonymized.csv`;
- `results/final_candidate_summary.csv`;
- `scripts/experiments/final_report_audit_artifacts_v1.py`.
