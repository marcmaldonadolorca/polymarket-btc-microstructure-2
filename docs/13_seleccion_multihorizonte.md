<!-- fuente original: UPSTREAM_OOF_H120_AUDIT_V1.md -->

# Upstream OOF H120 audit v1

Fecha: 2026-06-04

## Pregunta

¿El mismo modelo secuencial limpio funciona mejor si intentamos capturar un
movimiento a `120` segundos en lugar de `60`?

La comparación cambia solo el horizonte. No añadimos arquitectura, features ni
una segunda etapa. Esto permite saber si la mejora, si aparece, viene de dar
más tiempo a la señal.

## Construcción del target

El target H120 no estaba materializado dentro del tensor secuencial. Se une
desde el dataset de ejecución mediante la clave exacta:

```text
session_id + market_id + token_id + last_time_index_ns
```

Las `5.831` secuencias encuentran una única fila terminal. Solo se entrena y
evalúa donde `target_supported_H120 = true` y los tres costes están presentes.

Además se reentrena un control `H60_paired` usando exactamente esas mismas
secuencias. La comparación `H60_paired` frente a `H120` aísla el efecto del
horizonte; `H60_all` conserva la referencia histórica con toda la muestra.

## Protocolo causal

- features tabulares y tensor de orderbook idénticos al audit H60;
- control H60 emparejado con la misma muestra de train/evaluación que H120;
- se excluyen las ocho features auxiliares `ev_pred/healthy_proba`;
- `fusion_concat_gru`, una época fija y CPU determinista;
- scaler y modelo se ajustan solo con días pasados soportados;
- objetivo de entrenamiento: `exec_net_cost_0p5_H120`;
- ranking evaluado con `upstream_pred`, `upstream_proba` y su combinación;
- coste `0.50` es el criterio principal y coste `1.00` es stress;
- selección con `inner_1/2/3 + outer_validation`;
- test terminal se abre solo al final.

## Decisión

```text
NO_GO_CLEAN_OOF_H120_COST05
```

La policy H120 elegida con bloques previos no cierra el proxy de robustez cost0.50 en test terminal.

Configuración elegida antes de mirar test:

```text
score=upstream_proba
cap=75
selection_eligible=False
```

## Soporte H120 por split

| value | sequences | supported_sequences | support_rate | mean_cost0p5_supported | positive_cost0p5_rate_supported |
| --- | --- | --- | --- | --- | --- |
| test_terminal | 1066 | 718 | 0.6735 | -0.8686 | 0.4861 |
| train_initial | 4009 | 2835 | 0.7072 | -0.8914 | 0.4822 |
| validation_initial | 756 | 552 | 0.7302 | -0.8826 | 0.4819 |

## Soporte H120 por día

| value | sequences | supported_sequences | support_rate | mean_cost0p5_supported | positive_cost0p5_rate_supported |
| --- | --- | --- | --- | --- | --- |
| 2026-05-11 | 224 | 150 | 0.6696 | -0.8934 | 0.4800 |
| 2026-05-12 | 462 | 314 | 0.6797 | -0.8660 | 0.4841 |
| 2026-05-13 | 464 | 336 | 0.7241 | -0.8865 | 0.4911 |
| 2026-05-14 | 300 | 212 | 0.7067 | -0.8836 | 0.4858 |
| 2026-05-15 | 458 | 328 | 0.7162 | -0.8827 | 0.4726 |
| 2026-05-16 | 454 | 318 | 0.7004 | -0.9062 | 0.4906 |
| 2026-05-17 | 459 | 329 | 0.7168 | -0.9372 | 0.4742 |
| 2026-05-18 | 374 | 274 | 0.7326 | -0.8842 | 0.4891 |
| 2026-05-19 | 410 | 286 | 0.6976 | -0.8853 | 0.4860 |
| 2026-05-20 | 404 | 288 | 0.7129 | -0.8833 | 0.4688 |
| 2026-05-21 | 380 | 278 | 0.7316 | -0.8780 | 0.4820 |
| 2026-05-22 | 376 | 274 | 0.7287 | -0.8873 | 0.4818 |
| 2026-05-23 | 372 | 252 | 0.6774 | -0.8495 | 0.4921 |
| 2026-05-24 | 360 | 236 | 0.6556 | -0.8849 | 0.4788 |
| 2026-05-25 | 334 | 230 | 0.6886 | -0.8729 | 0.4870 |

## Folds

| horizon | evaluation_block | train_sequences_supported | train_support_rate | eval_sequences_supported | eval_support_rate | tab_features_used | tab_features_blocked |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H120 | inner_1 | 1012 | 0.6979 | 646 | 0.7083 | 31 | 8 |
| H120 | inner_2 | 1658 | 0.7019 | 603 | 0.7239 | 31 | 8 |
| H120 | inner_3 | 2261 | 0.7077 | 574 | 0.7052 | 31 | 8 |
| H120 | outer_validation | 2835 | 0.7072 | 552 | 0.7302 | 31 | 8 |
| H120 | test_terminal | 3387 | 0.7108 | 718 | 0.6735 | 31 | 8 |
| H60_paired | inner_1 | 1012 | 0.6979 | 646 | 0.7083 | 31 | 8 |
| H60_paired | inner_2 | 1658 | 0.7019 | 603 | 0.7239 | 31 | 8 |
| H60_paired | inner_3 | 2261 | 0.7077 | 574 | 0.7052 | 31 | 8 |
| H60_paired | outer_validation | 2835 | 0.7072 | 552 | 0.7302 | 31 | 8 |
| H60_paired | test_terminal | 3387 | 0.7108 | 718 | 0.6735 | 31 | 8 |

## Calidad del ranking H120

| evaluation_block | rows | auc_positive_cost0p5 | spearman_pred_cost0p5 | all_mean_cost0p5 | top0p35_mean_cost0p5 | top0p35_mean_cost1p0 |
| --- | --- | --- | --- | --- | --- | --- |
| inner_1 | 646 | 0.4768 | -0.0965 | -0.8942 | -2.1609 | -2.5333 |
| inner_2 | 603 | 0.4963 | -0.0301 | -0.9131 | -0.6214 | -1.0400 |
| inner_3 | 574 | 0.4848 | -0.0515 | -0.8843 | -0.0527 | -0.4297 |
| outer_validation | 552 | 0.5106 | -0.0518 | -0.8826 | -0.4789 | -0.8445 |
| test_terminal | 718 | 0.4951 | 0.0052 | -0.8686 | -1.1171 | -1.4910 |

## Comparación justa H60 frente a H120

| horizon | sequence_support_rate | selection_auc_mean | selection_spearman_mean | selection_top0p35_mean_cost0p5_min | selection_top0p35_mean_cost0p5_mean | test_auc | test_top0p35_mean_cost0p5 | test_top0p35_mean_cost1p0 | eligible_policy_count | robust_test_policy_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H60_all | 1.0000 | 0.5372 | 0.1148 | -0.7881 | -0.2352 | 0.5656 | 0.1778 | -0.1899 | 0 | 0 |
| H60_paired | 0.7040 | 0.5482 | 0.1368 | -1.1970 | -0.1828 | 0.5651 | 0.4262 | 0.0579 | 0 | 0 |
| H120 | 0.7040 | 0.4921 | -0.0575 | -2.1609 | -0.8285 | 0.4951 | -1.1171 | -1.4910 | 0 | 0 |

## Policies H120

| score_col | daily_cap | selection_eligible | selection_pass_cost0p5_min | selection_pass_cost0p5_mean | selection_worst_day_cost0p5_min | test_pass_cost0p5_rate | test_pass_cost1p0_rate | test_sum_cost0p5_mean | test_worst_day_cost0p5_mean | test_robust_cost0p5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| upstream_proba | 75 | False | 0.0000 | 0.1750 | -140.5006 | 0.0000 | 0.0000 | -194.2383 | -172.8870 | False |
| upstream_proba | 100 | False | 0.0000 | 0.1250 | -160.8453 | 0.0000 | 0.0000 | -280.4089 | -222.1817 | False |
| upstream_proba | 50 | False | 0.0000 | 0.1000 | -201.1444 | 0.0000 | 0.0000 | -130.1271 | -133.2283 | False |
| upstream_combo | 50 | False | 0.0000 | 0.0250 | -233.0813 | 0.0000 | 0.0000 | -26.2073 | -128.0463 | False |
| upstream_pred | 50 | False | 0.0000 | 0.0250 | -264.3779 | 0.0000 | 0.0000 | 28.4631 | -96.9383 | False |
| upstream_pred | 100 | False | 0.0000 | 0.0250 | -309.3361 | 0.0000 | 0.0000 | -6.1173 | -105.7034 | False |
| upstream_combo | 75 | False | 0.0000 | 0.0250 | -316.7853 | 0.1000 | 0.0000 | 40.8930 | -113.8697 | False |
| upstream_combo | 100 | False | 0.0000 | 0.0000 | -379.7059 | 0.1000 | 0.0000 | -0.6471 | -109.6632 | False |
| upstream_pred | 75 | False | 0.0000 | 0.0000 | -415.5707 | 0.3000 | 0.3000 | 116.8424 | -68.9227 | False |

## Lectura sencilla

- `AUC` y `Spearman` indican si el modelo ordena mejor las oportunidades, no
  si ya existe rentabilidad operativa.
- `top0p35_mean_cost...` mide el resultado medio del 35% mejor puntuado.
- Una policy solo es elegible si se puede escoger con los bloques anteriores y
  no esconde un día negativo en selección.
- El test terminal no participa en la elección.

## Conclusión metodológica

H120 no mejora de forma consistente los bloques previos y el test frente a H60. Alargar el horizonte por sí solo no resuelve la inestabilidad; conviene priorizar calidad de fill/labels y más folds.

No se debe promover todavía a bot ni aumentar a CNN/Transformer grande por
este resultado aislado. El siguiente paso depende de si H120 mejora de forma
causal y estable frente a H60, no de que exista una suma positiva puntual.


---

<!-- fuente original: COMPLEX_V1_LONG_HORIZON_PROBE.md -->

# Complex v1 - Probe de horizontes largos

Fecha: `2026-06-01`

## Pregunta

Queremos comprobar una idea razonable:

```text
quizas no hay que predecir solo a 16s; quizas a 60s, 120s o 240s el movimiento
es mas grande y cubre mejor el coste.
```

Esta iteracion no entrena modelos. Solo mide markout economico a varios
horizontes.

## Artefactos

Script:

```text
scripts/experiments/complex_v1_long_horizon_probe.py
```

Notebook:

```text
notebooks/08_complex_v1_long_horizon_probe.ipynb
```

Output:

```text
data/experiments/complex_v1_long_horizon_probe/
```

Figuras:

```text
data/experiments/complex_v1_long_horizon_probe/figures/
```

## Horizonte y latencia

Se mantiene la entrada principal:

```text
delay = 2s
```

Y se prueban horizontes despues de entrada:

```text
16s, 32s, 60s, 120s, 240s
```

Costes evaluados:

| Coste pagado | Lectura |
|---:|---|
| `0%` | Ejecucion teorica sin coste taker. |
| `25%` | Ejecucion muy barata / maker-proxy optimista. |
| `50%` | Ejecucion parcialmente barata. |
| `100%` | Taker completo. |

## Resultado global

Primero miramos todas las filas, sin seleccionar celdas.

Test terminal con `delay=2s`:

| Horizonte | Filas soportadas | Soporte | No-cost + buffer | Coste 25% | Coste 50% | Taker completo |
|---:|---:|---:|---:|---:|---:|---:|
| `16s` | `406.514` | `93,74%` | `-0,5000` | `-0,6790` | `-0,8580` | `-1,2159` |
| `32s` | `382.714` | `88,25%` | `-0,5000` | `-0,6795` | `-0,8589` | `-1,2178` |
| `60s` | `341.418` | `78,73%` | `-0,5000` | `-0,6802` | `-0,8604` | `-1,2209` |
| `120s` | `254.064` | `58,59%` | `-0,5000` | `-0,6815` | `-0,8630` | `-1,2260` |
| `240s` | `83.022` | `19,15%` | `-0,5000` | `-0,6842` | `-0,8683` | `-1,2367` |

Lectura:

```text
Globalmente, alargar horizonte no convierte el problema en rentable.
```

El `-0,5000` en no-cost aparece porque el promedio de movimiento entre tokens
se cancela aproximadamente y siempre restamos el buffer de `0,5 ticks`.

Por tanto:

```text
no hay edge global simplemente por esperar mas.
```

## Seleccion limpia de celdas

Despues seleccionamos celdas solo con:

```text
train_initial + validation_initial
```

Y miramos `test_terminal` despues.

Regla:

```text
delay >= 2s
rows_train >= 100
rows_validation >= 100
neto ajustado train > 0
neto ajustado validation > 0
```

Resultado:

| Coste | Horizonte | Celdas seleccionadas | Celdas robustas test | Positivas test | Neto test ponderado |
|---:|---:|---:|---:|---:|---:|
| `0%` | `16s` | `12` | `8` | `7` | `+0,0978` |
| `0%` | `32s` | `16` | `13` | `8` | `+0,0417` |
| `0%` | `60s` | `13` | `12` | `9` | `+0,1926` |
| `0%` | `120s` | `10` | `10` | `8` | `+0,3500` |
| `0%` | `240s` | `9` | `9` | `5` | `+0,4776` |
| `25%` | `16s` | `3` | `2` | `1` | `-0,2341` |
| `25%` | `32s` | `3` | `2` | `1` | `-0,1501` |
| `25%` | `60s` | `7` | `6` | `5` | `+0,3166` |
| `25%` | `120s` | `6` | `6` | `3` | `+0,2024` |
| `25%` | `240s` | `7` | `7` | `3` | `+0,3028` |
| `50%` | `16s` | `2` | `1` | `1` | `+0,1050` |
| `50%` | `60s` | `3` | `2` | `1` | `-0,4143` |
| `50%` | `120s` | `3` | `3` | `1` | `-0,2231` |
| `50%` | `240s` | `5` | `5` | `1` | `-0,4144` |
| `100%` | `60s` | `2` | `1` | `0` | `-2,7081` |
| `100%` | `120s` | `1` | `1` | `0` | `-1,1795` |
| `100%` | `240s` | `2` | `2` | `0` | `-1,2939` |

## Lectura sencilla

Hay tres conclusiones:

### 1. Mirar mas lejos si ayuda algo

Los horizontes `60s` y `120s` son mas interesantes que `16s/32s` cuando el
coste efectivo baja.

Especialmente:

```text
coste 25%, H60:  +0,3166 ticks ponderados en test
coste 25%, H120: +0,2024 ticks ponderados en test
```

Esto responde a la duda:

```text
si, predecir a mas tiempo puede ser parte de la solucion.
```

### 2. Pero no arregla taker completo

Con coste `100%` taker, los candidatos limpios que aparecen en train/validation
fallan en test.

Lectura:

```text
esperar mas no salva una ejecucion cara.
```

### 3. H240 es interesante pero peligroso

`240s` da algunos numeros positivos con coste bajo, pero tiene soporte global
mucho menor:

```text
19,15% en test
```

Ademas, en contratos `5m`, `240s` ocupa casi toda la vida del mercado. Eso ya no
es puro micro-edge de corto plazo: se parece mas a una prediccion de regimen o
de cierre.

Por eso:

```text
H240 queda como experimental, no como horizonte principal.
```

## Decision

Decision actual:

```text
incluir horizontes 60s y 120s en Complex v1a.
mantener 16s/32s como comparadores de corto plazo.
tratar 240s como experimental/holdout.
seguir en NO GO para bot.
```

No cambiamos la tesis principal:

```text
la solucion no es solo mas horizonte.
```

La tesis refinada queda:

```text
mejor horizonte + ejecucion barata + modelo de fill/adverse selection.
```

## Implicacion para el siguiente dataset

El dataset row-level de Complex v1a debe incluir targets por horizonte:

```text
H16, H32, H60, H120
```

Y opcional:

```text
H240 como experimental
```

Targets principales:

```text
exec_net_cost_0p25_H60
exec_net_cost_0p25_H120
exec_net_cost_0p50_H60
exec_net_cost_0p50_H120
healthy_fill_proxy_H60
healthy_fill_proxy_H120
adverse_fill_proxy_H60
adverse_fill_proxy_H120
```

Modelo recomendado:

```text
primero HGB/LightGBM tabular multitarget o varios modelos por horizonte.
despues encoder secuencial si el tabular no captura bien la persistencia.
```

## Frase corta para defender

```text
Alargar horizonte mejora algunas celdas con coste reducido, especialmente a
60s/120s, pero no genera edge global ni salva taker completo. Por eso el
siguiente modelo debe aprender valor esperado de ejecucion por horizonte y
riesgo de fill, no solo direccion.
```



---

