# Complex v1a - sequence probe v1

Fecha: 2026-06-02

## Objetivo

Hasta ahora vimos:

```text
hay edge en prestart H60,
pero los gates tabulares no estabilizan bien los dias malos.
```

La hipotesis nueva es:

```text
la dinamica corta de 7-8 snapshots por bloque puede contener informacion que
los modelos tabulares resumen demasiado mal.
```

Esta iteracion hace dos cosas:

1. construir un dataset secuencial pequeño y validado;
2. entrenar un baseline ligero `flatten_full` antes de pasar a CNN/GRU.

No se entrena todavia un modelo complejo.

## Artefactos

Dataset builder:

```text
scripts/experiments/complex_v1a_sequence_probe_v1_builder.py
```

Notebook dataset:

```text
notebooks/21_complex_v1a_sequence_probe_v1_dataset.ipynb
```

Output dataset:

```text
data/experiments/complex_v1a_sequence_probe_v1/
```

Baseline flatten:

```text
scripts/experiments/complex_v1a_sequence_probe_v1_flatten_baseline.py
```

Notebook baseline:

```text
notebooks/22_complex_v1a_sequence_probe_v1_flatten_baseline.ipynb
```

Output baseline:

```text
data/experiments/complex_v1a_sequence_probe_v1_flatten_baseline/
```

## Dataset secuencial

Unidad:

```text
session_id + market_id + token_id
```

Universo:

```text
strict_45_60_early
```

Es decir:

```text
prestart H60, 45-60s antes del inicio
```

Shape:

```text
x:    (5831, 8, 39)
mask: (5831, 8)
```

Lectura:

- `5831` secuencias;
- hasta `8` snapshots por secuencia;
- `39` features numericas;
- `40811` snapshots reales;
- la mascara suma exactamente `40811`, por tanto no se han perdido filas;
- `x` normalizado no contiene NaNs.

`x_raw` conserva algunos NaNs originales, pero `x` queda imputado a cero tras
normalizacion usando estadisticas solo de train.

## Features

Se usan features no-clock/core mas scores auxiliares:

- `seconds_before_start`;
- `age_ms`;
- spread/coste visible;
- Polymarket mid;
- microprice gap;
- momentum PM;
- retornos spot/perp;
- basis;
- gaps spot/perp;
- volumen/flujo externo disponible;
- `ev_pred_full`;
- `healthy_proba_full`;
- `ev_pred_noclock`;
- `healthy_proba_noclock`;
- desacuerdo entre modelos.

No se usan como features:

- `exec_net_*`;
- `target_*`;
- labels futuros;
- entry/exit futuros.

## Targets guardados

Se guardan dos familias de target:

```text
y_last_*
y_block_mean_*
```

Con costes:

- `0.25`;
- `0.50`;
- `1.00`.

La primera prueba usa:

```text
y_last_cost0p5
```

porque representa una accion en el ultimo snapshot de la secuencia.

## Split summary

| Split | Secuencias | Dias | Longitud media |
|---|---:|---:|---:|
| train | 4009 | 10 | 7.01 |
| validation | 756 | 2 | 7.01 |
| test | 1066 | 3 | 6.96 |

Importante:

```text
el universo completo tiene media negativa.
```

Esto es normal. El modelo no debe operar todo; debe seleccionar un subconjunto.

## Baseline entrenado

Se entrenaron dos variantes ligeras:

| Variante | Que ve |
|---|---|
| `last_step` | Solo el ultimo snapshot valido. |
| `flatten_full` | La secuencia completa aplanada, mas mask y longitud. |

Modelo:

```text
HistGradientBoostingRegressor + HistGradientBoostingClassifier
```

Esto no es aun CNN/GRU. Es una prueba puente.

## Calidad de ranking

| Feature mode | Spearman val | AUC val | Spearman test | AUC test |
|---|---:|---:|---:|---:|
| `last_step` | 0.0337 | 0.5177 | 0.0539 | 0.5143 |
| `flatten_full` | 0.0747 | 0.5253 | 0.1464 | 0.5754 |

Lectura:

```text
la secuencia completa aporta mas senal que mirar solo el ultimo snapshot.
```

Esto justifica probar CNN/GRU, pero no aprueba nada todavia.

## Decision del baseline

Decision:

```text
NO_GO_SEQUENCE_FLATTEN
```

Mejor policy seleccionada por validacion:

```text
feature_mode = flatten_full
policy       = pred_ge_2_p_ge_0.6
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|
| validation | 144 | +0.6129 | +0.4272 | +0.0559 | 0 |
| test | 212 | +0.5236 | +0.3349 | -0.0425 | 2 |

No pasa porque:

- test no aguanta coste 1.00;
- tiene `2` dias negativos a coste 0.50.

## Desglose test del mejor candidato

| Dia | Acciones | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|
| 2026-05-23 | 71 | +1.6540 | +1.2587 |
| 2026-05-24 | 72 | -0.3785 | -0.7500 |
| 2026-05-25 | 69 | -0.2782 | -0.6433 |

Lectura:

```text
el modelo encuentra buenos bloques en 2026-05-23, pero no estabiliza los dias
2026-05-24 y 2026-05-25.
```

## Policies que pasan test

Hay:

```text
14 policies que pasan test completo
```

La mejor retrospectiva:

```text
flatten_full + top_20pct
```

En test:

- `149` acciones;
- `+2.4196` coste 0.50;
- `+2.0405` coste 1.00;
- `0` dias negativos.

Pero en validacion:

- `102` acciones;
- `-0.2492` coste 0.50;
- `-0.6209` coste 1.00;
- `1` dia negativo.

Por tanto no se puede aceptar.

Esto repite el patron visto antes:

```text
hay informacion que limpia test, pero validacion no permite seleccionarla de
forma robusta.
```

## Conclusion

Esta iteracion extrae algo valioso:

```text
la secuencia aporta senal incremental frente a last_step.
```

Pero tambien confirma:

```text
un flatten tabular no basta para estabilizar la seleccion temporal.
```

Decision:

```text
NO GO bot.
NO GO sequence flatten como solucion.
GO CNN/GRU probe pequeno.
```

## Siguiente paso

Entrenar modelos secuenciales pequenos:

1. MLP regularizado sobre flatten, como control.
2. 1D CNN pequena.
3. GRU pequena.

Condicion para aprobar:

```text
seleccion en validacion positiva,
test positivo a coste 0.50 y 1.00,
0 dias negativos a coste 0.50,
y mejora sobre flatten_full y gates causales tabulares.
```

No empezamos por Transformer grande.

---

# Book-only Conv1D baseline v1

Fecha: 2026-06-02

## Objetivo

Probar si el orderbook visible de Polymarket, tratado como un tensor, contiene senal predictiva por si solo.

Esto no es aun el modelo final. Es una prueba intermedia para decidir si merece la pena pasar a fusion:

```text
book encoder + secuencia tabular + gestion de riesgo
```

## Dataset

Tensor usado:

```text
(5831, 8, 10, 10)
```

Canales:

```text
[
  "bid_log_size_rel_min_order",
  "ask_log_size_rel_min_order",
  "bid_cum_log_size_rel_min_order",
  "ask_cum_log_size_rel_min_order",
  "bid_distance_ticks",
  "ask_distance_ticks",
  "bid_present",
  "ask_present",
  "bid_gap_ticks_from_prev",
  "ask_gap_ticks_from_prev"
]
```

Target:

```text
y_last_cost0p5
```

Interpretacion sencilla:

```text
edge neto del ultimo snapshot, despues de aplicar un coste teorico de 0.50.
```

## Modelos entrenados

| Modelo | Parametros | Idea |
|---|---:|---|
| `book_last_snapshot_conv` | 9986 | Conv1D sobre los 10 niveles del ultimo snapshot. |
| `book_seq_conv_gru` | 29946 | Conv1D por snapshot + GRU sobre la secuencia temporal. |

## Ranking

| Modelo | Spearman val | AUC val | Spearman test | AUC test |
|---|---:|---:|---:|---:|
| `book_last_snapshot_conv` | 0.0863 | 0.5631 | 0.0241 | 0.5312 |
| `book_seq_conv_gru` | 0.0958 | 0.5607 | 0.0483 | 0.5355 |

## Decision

Decision:

```text
NO_GO_BOOK_ONLY_CONV1D
```

Razon:

```text
Validation-selected book-only policy fails terminal stability.
```

Mejor policy seleccionada por validacion:

```text
book_last_snapshot_conv + top_20pct
```

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|
| validation | 129 | 0.8457 | 0.6604 | 0.2898 | 0 |
| test | 151 | -0.8875 | -1.0830 | -1.4739 | 2 |

Desglose diario del mejor candidato:

| Dia | Split | Acciones | Net 0.50 | Net 1.00 |
|---|---|---:|---:|---:|
| 2026-05-23 | test_terminal | 58 | -1.4145 | -1.8117 |
| 2026-05-24 | test_terminal | 52 | 0.0664 | -0.3191 |
| 2026-05-25 | test_terminal | 41 | -2.0718 | -2.4606 |
| 2026-05-21 | validation_initial | 63 | 0.5504 | 0.1802 |
| 2026-05-22 | validation_initial | 66 | 0.7654 | 0.3944 |

## Lectura

Este baseline responde a una pregunta concreta:

```text
si miro solo la forma del libro visible, saco ventaja estable?
```

Si el resultado queda cerca del GRU tabular, merece la pena fusionar ambos mundos.
Si queda claramente peor, el orderbook visible se usara mas como contexto auxiliar/regimen que como fuente principal.

## Comparacion cualitativa con el estado actual

| Familia | Estado | Lectura corta |
|---|---|---|
| Flatten tabular | NO_GO | Buena AUC test agregada, pero falla estabilidad diaria. |
| GRU tabular | NO_GO | Mejor policy que flatten; aun tiene un dia malo. |
| Gate fijo de regimen | RESEARCH_PASS | Muy prometedor, pero no seleccionado de forma causal robusta. |
| Book-only Conv1D | NO_GO | La imagen del libro visible sola no generaliza en test. |

## Siguiente paso

Si book-only no es desastroso, el siguiente paso natural es:

```text
fusion_v1 = encoder orderbook + GRU tabular + evaluacion de policy con costes y estabilidad diaria
```

No saltamos todavia a Transformer grande ni a bot. Primero queremos demostrar mejora incremental y estable.

---

# Fusion v1 - book + tabular sequence

Fecha: 2026-06-02

## Objetivo

Probar si el orderbook como tensor ayuda al GRU tabular.

Comparacion justa:

- mismo split temporal;
- mismo target `y_last_cost0p5`;
- mismas policies por validacion;
- mismo criterio de coste 0.25/0.50/1.00 y estabilidad diaria.

## Modelos

| Modelo | Parametros | Idea |
|---|---:|---|
| `tabular_gru_control` | 18186 | Control: solo features tabulares secuenciales. |
| `fusion_concat_gru` | 42874 | Concatena embedding book + tabular en cada paso y usa GRU. |
| `fusion_dual_gru` | 34190 | GRU tabular y GRU book separados, fusionados al final. |

## Ranking

| Modelo | AUC val | AUC test | Spearman val | Spearman test |
|---|---:|---:|---:|---:|
| `tabular_gru_control` | 0.5698 | 0.5548 | 0.1442 | 0.1338 |
| `fusion_concat_gru` | 0.5703 | 0.5541 | 0.1368 | 0.1333 |
| `fusion_dual_gru` | 0.5694 | 0.5616 | 0.1309 | 0.1266 |

## Decision

```text
RESEARCH_PASS_FUSION_V1_DIAGNOSTIC
```

A validation-stable fusion candidate passes terminal test, but the primary validation selector still chooses the tabular control.

Policy primaria elegida por validacion:

```text
tabular_gru_control + top_20pct
```

| Split | Acciones | Net 0.50 | Net 1.00 | Dias negativos |
|---|---:|---:|---:|---:|
| validation | 112 | 1.8515 | 1.4843 | 0 |
| test | 185 | 0.5911 | 0.2227 | 1 |

Candidato diagnostico de fusion:

```text
fusion_concat_gru + pred_ge_-2_p_ge_0.55
```

Importante: este candidato se reporta para investigacion. No sustituye al selector primario porque seria facil caer en seleccion retrospectiva.

| Split | Acciones | Net 0.50 | Net 1.00 | Dias negativos |
|---|---:|---:|---:|---:|
| validation | 212 | 0.6412 | 0.2659 | 0 |
| test | 321 | 0.5373 | 0.1618 | 0 |

Desglose diario del candidato de fusion:

| Dia | Split | Acciones | Net 0.50 | Net 1.00 |
|---|---|---:|---:|---:|
| 2026-05-23 | test_terminal | 106 | 0.3745 | -0.0151 |
| 2026-05-24 | test_terminal | 108 | 0.3347 | -0.0390 |
| 2026-05-25 | test_terminal | 107 | 0.9031 | 0.5398 |
| 2026-05-21 | validation_initial | 108 | 0.5573 | 0.1840 |
| 2026-05-22 | validation_initial | 104 | 0.7284 | 0.3510 |

Lectura del candidato:

```text
elimina dias negativos a coste 0.50, pero coste 1.00 diario sigue justo.
```

Desglose diario de la policy primaria:

| Dia | Split | Acciones | Net 0.50 | Net 1.00 |
|---|---|---:|---:|---:|
| 2026-05-23 | test_terminal | 55 | 2.5516 | 2.1668 |
| 2026-05-24 | test_terminal | 66 | -0.7645 | -1.1351 |
| 2026-05-25 | test_terminal | 64 | 0.3043 | -0.0477 |
| 2026-05-21 | validation_initial | 55 | 0.6015 | 0.2393 |
| 2026-05-22 | validation_initial | 57 | 3.0578 | 2.6857 |

## Lectura

Esta iteracion no busca un modelo grande, sino responder una pregunta:

```text
el libro visible mejora al modelo secuencial tabular?
```

Si la fusion no mejora de forma estable, el siguiente paso no debe ser agrandar la red.
Debe ser mejorar el target/regimen y la seleccion causal.

Conclusion practica:

```text
seguir con fusion, pero orientada a selector/riesgo y coste 1.00, no a red mas grande.
```

---

