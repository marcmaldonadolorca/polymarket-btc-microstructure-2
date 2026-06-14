# Baseline latency-aware v0.2 - Diagnostico trade/no-trade L2

Fecha: `2026-06-01`

## Objetivo

Tras cerrar la latencia:

```text
L_operativo_real ~= 1s
L_dataset_principal = 2s
```

reutilizamos el dataset L2 como evaluacion conservadora. Esta iteracion no
entrena modelos nuevos. Solo mira los outputs ya generados en v0.1b para
responder:

```text
El problema es que el modelo no ve direccion, o que no sabe cuando compensa
tradear despues de costes?
```

## Artefactos

Script:

```text
scripts/experiments/baseline_laware_v02_trade_diagnostics.py
```

Input:

```text
data/experiments/baseline_laware_v01b_feature_runner/
```

Output:

```text
data/experiments/baseline_laware_v02_trade_diagnostics/
```

Archivos:

| Archivo | Uso |
|---|---|
| `scenario_summary.csv` | Resumen de escenarios ya guardados. |
| `score_bucket_summary.csv` | Calidad por bucket de `score_buy`. |
| `segment_summary.csv` | Segmentos por dia, temporalidad y fase. |
| `temporality_score_bucket_summary.csv` | Cruce temporalidad x score. |
| `candidate_rule_grid.csv` | Reglas simples score/temporalidad. |
| `promising_rules.csv` | Reglas positivas con filtro minimo. |

## Resultado global

Los escenarios principales siguen negativos:

| Feature set | Escenario | Acciones | Hit up | Wrong down | Neto medio |
|---|---|---:|---:|---:|---:|
| `v03_conservative_plus_micro` | high score + low spread | `1.278` | `43,19%` | `8,29%` | `-0,63` |
| `v03_pm_perp_control` | high score + low spread | `1.188` | `44,28%` | `7,74%` | `-0,64` |
| `v03_conservative_plus_micro` | low spread amplio | `201.685` | `32,83%` | `21,90%` | `-0,94` |
| `v03_pm_perp_control` | low spread amplio | `202.901` | `32,78%` | `21,93%` | `-0,95` |

Lectura:

```text
La mejora de microprice ayuda a reducir wrong-down, pero el movimiento medio
neto despues de coste + buffer sigue siendo negativo.
```

## Pistas positivas

Aparecen reglas positivas pequenas:

| Feature set | Filtro | Acciones | Hit up | Wrong down | Neto medio |
|---|---|---:|---:|---:|---:|
| `v03_pm_perp_control` | `1h`, score `>=0,40` | `61` | `62,30%` | `8,20%` | `0,23` |
| `v03_conservative_plus_micro` | `1h`, score `>=0,40` | `62` | `58,06%` | `9,68%` | `0,22` |
| `v03_pm_perp_control` | low spread, score `>=0,50` | `46` | `47,83%` | `2,17%` | `0,02` |

Pero:

```text
ninguna regla positiva alcanza 100 acciones.
```

Y una alerta importante:

```text
las reglas 1h positivas pasan a neto negativo al quitar solape H16.
```

Por ejemplo:

| Regla | Acciones | Neto medio | No solapado | Neto medio no solapado |
|---|---:|---:|---:|---:|
| plus micro `1h >=0,40` | `62` | `0,22` | `34` | `-0,29` |
| PM+perp `1h >=0,40` | `61` | `0,23` | `31` | `-0,26` |

Lectura:

```text
hay senal local, pero todavia no es robusta. El resultado positivo parece
depender de pocas acciones y/o de solape temporal.
```

## Decision

Decision:

```text
NO GO para aprobar una politica L2 ejecutable.
NO GO para bot.
GO para formular target economico trade/no_trade.
```

La conclusion no es "no hay edge". La conclusion es mas precisa:

```text
El score direccional ayuda, pero elegir trades solo por P(up)-P(down) no basta
para cubrir coste + buffer. Necesitamos entrenar directamente una decision
economica: trade si el markout neto esperado supera el coste.
```

## Siguiente iteracion recomendada

Crear baseline L-aware v0.2b con target economico:

```text
target_trade_L2_H16_buffer_0p5 = net_buffer_0p5_ticks_L2_H16 > 0
```

Primer experimento corto:

- feature set: `v03_conservative_plus_micro`;
- split temporal igual que v0.1b;
- evaluar por market-frame;
- metricas: precision de trade, coverage, neto medio, no solapado H16;
- comparar contra la regla direccional `score_buy >= threshold`;
- no usar columnas `entry_*`, `future_*`, `delta_*`, `net_*` como features.

Si ese target economico tampoco produce segmentos robustos, entonces habra que
pasar a diagnostico de timing/orderbook o aceptar `NO GO` para el enfoque
tabular taker v0.
