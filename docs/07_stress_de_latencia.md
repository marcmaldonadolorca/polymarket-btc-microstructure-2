# Baseline v0.3d - Stress offline de ejecucion y latencia

Fecha: `2026-05-31`

Estado: stress test offline de la politica candidata. No es backtest real de
fills y no autoriza bot.

## Pregunta

En v0.3c vimos que `score_buy >= 0,65` funcionaba bien en folds temporales.
La pregunta siguiente era mas dura:

> Si la senal aparece en `t`, sigue siendo buena si entramos algunos segundos
> despues y pagamos un coste mas conservador?

Esta pregunta es clave porque el target del modelo sigue mirando movimiento
desde el instante de la observacion. Un bot real no entra gratis en ese mismo
instante: tarda en calcular, enviar orden y conseguir fill.

## Artefactos

Script:

```text
scripts/experiments/baseline_v03d_execution_stress.py
```

Notebook:

```text
notebooks/20_baseline_v03d_execution_stress.ipynb
```

Resultados:

```text
data/experiments/baseline_v03d_execution_stress/
```

Archivos principales:

| Archivo | Uso |
|---|---|
| `execution_stress_summary.csv` | Resultado agregado por escenario. |
| `execution_stress_by_split.csv` | Resultado por fold/test. |
| `execution_stress_by_day.csv` | Resultado por dia. |
| `latency_support.csv` | Cuantas senales tienen fila de entrada retrasada. |
| `recommended_execution_scenarios.csv` | Ranking de escenarios. |
| `summary.json` | Resumen general. |

## Diseno

Senales de entrada:

- folds F1-F4 de v0.3c con `score_ge_0p65_visible`;
- test terminal de v0.3b con `selected_visible`;
- total combinado: `646` senales.

Unidad:

```text
session_id + condition_id + token_id + time_index_ns
```

Se evalua siempre como accion por market-frame, no como doble oportunidad de
tokens complementarios.

Latencias probadas:

```text
0s, 2s, 4s, 8s
```

Variantes de ejecucion:

| Variante | Regla |
|---|---|
| `cross_plus_0p5tick` | Entrar en la fila retrasada pagando coste visible + `0,5 tick`. |
| `low_spread_plus_0p5tick` | Igual, pero solo si el spread de entrada es `<= 1 tick`. |
| `low_spread_plus_1p0tick` | Spread `<= 1 tick` + buffer `1 tick`. |
| `low_spread_plus_2p0tick` | Stress severo: spread `<= 1 tick` + buffer `2 ticks`. |

Controles:

- cooldown por `condition_id`: `0s`, `16s`, `60s`, `120s`;
- limites: sin limite, `50` acciones/dia y `3` por sesion, o `20` acciones/dia
  y `1` por sesion.

Nota tecnica: se usa tolerancia flotante para `spread <= 1 tick`, porque en
coma flotante aparecen valores como `1.0000000000000009`.

## Soporte de latencia

| Latencia | Senales con fila de entrada | Soporte |
|---:|---:|---:|
| `0s` | 646 / 646 | 100,00% |
| `2s` | 640 / 646 | 99,07% |
| `4s` | 631 / 646 | 97,68% |
| `8s` | 617 / 646 | 95,51% |

Lectura: el problema no es falta de filas. Hay datos suficientes para medir el
efecto de entrar tarde.

## Resultado principal

Escenario conservador legible:

```text
low_spread_plus_0p5tick
cooldown = 16s
limite = 50 acciones/dia y 3 por sesion
```

| Latencia | Acciones | Hit up | Wrong down | Neto medio | Neto total | Splits positivos |
|---:|---:|---:|---:|---:|---:|---:|
| `0s` | 376 | 91,76% | 3,99% | 9,34 | 3.512,62 | 5/5 |
| `2s` | 343 | 59,48% | 26,24% | 2,96 | 1.014,47 | 4/5 |
| `4s` | 371 | 50,13% | 35,58% | 0,20 | 74,75 | 3/5 |
| `8s` | 369 | 42,82% | 36,86% | -0,76 | -280,11 | 0/5 |

Lectura sencilla:

- en `0s` la politica sigue siendo muy fuerte;
- con `2s` todavia queda neto agregado positivo, pero el error sube mucho;
- con `4s` queda casi plano;
- con `8s` se rompe.

## Test terminal bajo latencia

El punto mas importante esta en el test terminal, porque es el bloque temporal
que no usamos para elegir la politica.

Mismo escenario conservador:

| Latencia | Acciones TEST | Hit up | Wrong down | Neto medio | Neto total |
|---:|---:|---:|---:|---:|---:|
| `0s` | 64 | 92,19% | 1,56% | 6,79 | 434,44 |
| `2s` | 63 | 39,68% | 44,44% | -1,48 | -93,48 |
| `4s` | 66 | 45,45% | 43,94% | -0,74 | -48,83 |
| `8s` | 66 | 43,94% | 31,82% | -0,30 | -19,53 |

Esta tabla cambia la decision: la senal de `score >= 0,65` es real como
markout desde `t`, pero no queda aprobada como politica ejecutable con una
latencia simple de `2s` en el test terminal.

## Comparacion con entrada sin filtro de spread

Escenario:

```text
cross_plus_0p5tick
cooldown = 16s
limite = 50 acciones/dia y 3 por sesion
```

| Latencia | Acciones | Hit up | Wrong down | Neto medio | Neto total | Splits positivos |
|---:|---:|---:|---:|---:|---:|---:|
| `0s` | 399 | 91,23% | 5,01% | 8,95 | 3.571,58 | 5/5 |
| `2s` | 397 | 63,22% | 25,44% | 2,91 | 1.156,87 | 4/5 |
| `4s` | 394 | 53,55% | 33,76% | 0,39 | 153,63 | 2/5 |
| `8s` | 392 | 45,41% | 36,48% | -0,79 | -309,33 | 0/5 |

El patron es el mismo: la latencia destruye gran parte del edge. Filtrar spread
ayuda, pero no arregla el problema principal.

## Que queda demostrado

- La politica `score_buy >= 0,65` no era solo una casualidad del test: en
  latencia `0s` funciona en folds y test.
- El resultado es muy sensible al instante de entrada.
- `0s` debe tratarse como cota superior, no como simulacion real.
- `2s` ya es demasiado agresivo para aprobar bot: el agregado queda positivo,
  pero el test terminal es negativo.
- `4s` y `8s` no son aceptables bajo este contrato.

## Que NO queda demostrado

No hemos demostrado PnL real. Faltan:

- fills reales;
- cola/partial fills;
- tamano de orden;
- slippage dependiente de cantidad;
- comisiones finales exactas;
- reaccion del mercado a la orden;
- latencia real medida del sistema.

Este stress es mas realista que mirar solo `delta_ticks_16s`, pero sigue siendo
un proxy offline.

## Decision v0.3d

Decision:

```text
GO para medir latencia real end-to-end
GO para investigar target/modelo consciente de latencia despues de medirla
NO GO para bot
NO GO para modelos complejos todavia
```

La conclusion no es "no hay edge". La conclusion es mas precisa:

> Hay edge de prediccion a muy corto plazo, pero el baseline actual lo captura
> demasiado cerca del movimiento. Antes de pensar en bot, hay que entrenar y
> validar contra una entrada retrasada o demostrar ejecucion sub-grid.

## Siguiente paso recomendado

La siguiente iteracion debe ser corta y enfocada:

1. medir latencia real end-to-end: datos, features, inferencia, decision,
   orden, firma, envio, aceptacion/fill y confirmacion;
2. elegir `L_operativo` a partir de un percentil conservador (`p90/p95`), no a
   partir del resultado del modelo;
3. crear labels de ejecucion consciente de latencia:
   `net_execution_ticks_L2_H16` y `net_execution_ticks_L4_H16`;
4. mantener el mismo split temporal y la misma evaluacion por market-frame;
5. repetir un baseline tabular simple, no redes;
6. comparar el threshold elegido en folds contra el test terminal;
7. decidir si hay suficiente senal temprana para seguir.

Si el modelo consciente de `L=2s` no mejora el test terminal, el proyecto debe
volver a exploracion de timing/orderbook antes de cualquier bot.

Estrategia ampliada:
ESTRATEGIA_LATENCIA_Y_SIGUIENTE_FASE.md
