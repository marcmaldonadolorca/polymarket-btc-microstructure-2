# Baseline v0.3 - Contrato de dataset offline

Fecha: `2026-05-31`

Estado: contrato de dataset/modelado offline. No es pipeline productivo y no es
bot.

## Para que sirve

Hasta v0.2 hemos demostrado una cosa importante:

> Hay senal offline en zonas concretas del mercado, sobre todo en la politica
> `full_no_micro_highconv`.

El objetivo de v0.3 no es meter un modelo mas complejo. Es ordenar el terreno
para que los siguientes experimentos sean comparables:

- mismas filas;
- mismos splits;
- mismas features permitidas;
- mismas columnas prohibidas por leakage;
- misma unidad de evaluacion;
- mismos informes de control.

## Artefactos generados

Script:

```text
scripts/experiments/baseline_v03_dataset_contract.py
```

Resultados:

```text
data/experiments/baseline_v03_dataset_contract/
```

Archivos principales:

| Archivo | Uso |
|---|---|
| `manifest.json` | Resumen de fuente, filas, targets, feature sets y salidas. |
| `feature_contract.csv` | Diccionario tecnico de columnas: rol, bloque, missingness y si entra en cada feature set. |
| `feature_set_health.csv` | Salud de cada bloque de features. |
| `split_summary.csv` | Conteos y target por split temporal. |
| `target_by_split_temporality.csv` | Target por split y `temporality`. |
| `target_by_split_phase_cost.csv` | Target por split, fase de ventana y coste visible. |
| `v02_action_segments_for_v03.csv` | Segmentos de acciones v0.2 para decidir donde enfocar v0.3. |

Fuente de datos:

```text
data/experiments/baseline_v0_full_core_robustness/baseline_v0_core_h16_parquet
```

Este parquet viene de SQLite oficial y ya contiene el target H16 recalculado.
No se ha reabierto el recolector.

## Unidad de datos

Unidad de entrenamiento:

```text
session_id + token_id + time_index_ns
```

Unidad de evaluacion/accion:

```text
session_id + condition_id + time_index_ns
```

Esto se mantiene porque los dos tokens del mismo mercado son complementarios.
Entrenar por token es practico, pero evaluar por token inflaria resultados.

## Target v0.3

Target continuo principal:

```text
delta_ticks_16s
```

Target de clase principal:

```text
target_3c_16s_1tick
```

Regla:

```text
up   si delta_ticks_16s >= +1 tick
down si delta_ticks_16s <= -1 tick
flat en otro caso
```

La clase `flat` aqui significa "no se movio suficiente el mid". No significa
todavia `no_trade`. El `no_trade` pertenece a la capa economica.

## Splits consolidados

| Split | Filas | Market frames | Sesiones | Dias | Up | Flat | Down |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train_initial` | 1.594.712 | 797.356 | 1.765 | 10 | 26,70% | 46,80% | 26,50% |
| `validation_initial` | 313.860 | 156.930 | 345 | 2 | 26,45% | 47,33% | 26,22% |
| `test_terminal` | 436.712 | 218.356 | 480 | 3 | 25,47% | 49,31% | 25,22% |

Lectura:

- el test terminal tiene algo mas de `flat`;
- no es un problema, es precisamente drift temporal real;
- no debemos arreglarlo con shuffle aleatorio;
- el test terminal se mantiene bloqueado.

## Drift por temporality

El target no es igual en `5m`, `15m` y `1h`.

| Split | Temporality | Up | Flat | Down |
|---|---|---:|---:|---:|
| `test_terminal` | `15m` | 31,59% | 37,18% | 31,23% |
| `test_terminal` | `1h` | 23,10% | 54,05% | 22,85% |
| `test_terminal` | `5m` | 21,73% | 56,67% | 21,60% |

Lectura sencilla:

- `15m` se mueve mucho mas;
- `5m` y `1h` tienen mas `flat`;
- por eso todos los resultados deben reportarse por `temporality`;
- un promedio global puede esconder que el modelo funciona distinto por tipo
  de mercado.

## Feature sets v0.3

| Feature set | Uso | Num | Cat | Missing mediano | Missing max |
|---|---|---:|---:|---:|---:|
| `v03_general_full_v0` | baseline general comparable con v0.1 | 29 | 5 | 0,00% | 4,97% |
| `v03_conservative_no_micro` | politica conservadora candidata tras v0.2 | 28 | 3 | 0,00% | 4,97% |
| `v03_pm_perp_control` | control PM + perp | 25 | 4 | 0,00% | 4,97% |
| `v03_quality_diagnostic` | masks/diagnostico, no feature fuerte | 8 | 1 | 0,00% | 0,00% |
| `v03_intertemporal_holdout` | experimental, fuera del baseline | 5 | 0 | 33,42% | 68,75% |

Decision:

- `v03_general_full_v0` queda como baseline general;
- `v03_conservative_no_micro` queda como candidato principal de ejecucion
  offline;
- `v03_pm_perp_control` queda como control para no autoenganarnos;
- quality masks se reportan, pero no se tratan como edge fuerte;
- intertemporal queda en holdout porque tiene missingness alto y mezcla
  estructura de mercados.

## Columnas que nunca deben ser features

No entran como input:

- `future_mid_8s`;
- `future_mid_16s`;
- `future_microprice_16s`;
- `future_time_index_ns_8s`;
- `future_time_index_ns_16s`;
- `future_stale_*`;
- `future_missing_*`;
- `delta_ticks_16s`;
- `target_3c_16s_1tick`;
- `predictor_label_name_8s`;
- `predictor_economic_label_name_8s`;
- cualquier columna derivada del futuro.

Estas columnas sirven para calcular target, benchmark o auditoria, pero si
entran al modelo provocan leakage.

## Lectura de acciones v0.2 para v0.3

Escenario strict:

```text
full_no_micro_highconv + score >= 0,60 + spread <= 1 tick + buffer 0,5 tick
```

Por `temporality`:

| Temporality | Acciones | Hit up | Wrong down |
|---|---:|---:|---:|
| `15m` | 68 | 88,24% | 5,88% |
| `1h` | 98 | 88,78% | 1,02% |
| `5m` | 37 | 89,19% | 8,11% |

Lectura:

- la calidad aparece en las tres temporalidades;
- `5m` tiene pocas acciones y mas `wrong_down`;
- `1h` tiene buen hit pero muchos flats de base, asi que conviene mirar
  neto y no solo direccion;
- v0.3 debe reportar siempre por `temporality`, fase y score.

## Decision v0.3

Decision:

```text
GO para modelado offline v0.3 reproducible
NO GO para bot
NO GO para redes/orderbook encoder todavia
```

Primer modelo v0.3 recomendado:

- seguir con tabular interpretable;
- comparar `v03_general_full_v0` contra `v03_conservative_no_micro`;
- elegir threshold solo en validation/folds;
- reportar test terminal por `temporality`, fase y coste;
- mantener `full_no_micro_highconv` como politica conservadora candidata.

## Que viene despues

Siguiente iteracion razonable:

1. preparar un runner v0.3 que no reextraiga datos;
2. repetir baseline con contrato v0.3 y reporte segmentado;
3. comparar cualquier feature nueva contra `v03_conservative_no_micro`;
4. solo despues pensar en nuevas features grandes o modelos mas fuertes.

Notebook de lectura ya creado:

```text
notebooks/16_baseline_v03_dataset_contract.ipynb
```

---

# Complex v1a - Dataset row-level de ejecucion

Fecha: `2026-06-01`

## Objetivo

Construir el primer dataset entrenable para Complex v1a:

```text
features causales en t
entrada simulada en t+2s
targets economicos por horizonte
proxies de fill adverso
sin leakage
```

Este dataset todavia no entrena modelos. Es el punto de partida limpio para el
primer modelo tabular de valor esperado/fill-risk.

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_execution_dataset_builder.py
```

Notebook:

```text
notebooks/09_complex_v1a_execution_dataset_builder.ipynb
```

Output:

```text
data/experiments/complex_v1a_execution_dataset/
```

Archivos:

| Archivo | Uso |
|---|---|
| `complex_v1a_execution_dataset.parquet` | Dataset row-level completo. |
| `manifest.json` | Contrato de columnas, horizontes y targets. |
| `feature_manifest.csv` | Columnas permitidas como input. |
| `target_manifest.csv` | Targets y metadata prohibidos como input. |
| `target_summary.csv` | Soporte y distribuciones por split/horizonte. |
| `preview_2k.csv` | Vista ligera para inspeccion manual. |

## Tamano

Dataset:

```text
2.328.910 filas
~193 MB parquet
42 feature columns permitidas
111 columnas de target/metadata
```

Fuente:

```text
data/experiments/baseline_laware_v01_dataset_contract/baseline_laware_v01_core_L2.parquet
```

## Regla de causalidad

Features permitidas:

```text
solo informacion visible en t
```

Se excluyen como features:

```text
delta_*
net_*
exec_net_*
target_*
entry_*
exit_*
adverse_*
healthy_*
label_supported_*
future_*
```

Detalle importante:

```text
entry_spread_bucket_L2 NO es feature.
```

Aunque es util para analizar ejecucion, se conoce en `t+2s`, no en `t`.
Por eso se guarda como metadata/target, pero no entra en `feature_columns`.

Como sustituto causal se incluye:

```text
spread_bucket_t
```

que usa el spread visible en `t`.

## Horizontes

Entrada:

```text
L = 2s
```

Horizontes:

| Horizonte | Uso |
|---:|---|
| `H16` | Comparador corto heredado. |
| `H32` | Corto/medio. |
| `H60` | Principal. |
| `H120` | Principal. |
| `H240` | Experimental/holdout. |

Decision:

```text
H60 y H120 son los targets principales de Complex v1a.
H240 no debe usarse para aprobar la estrategia v1.
```

## Targets economicos

Para cada horizonte se calcula:

```text
delta_ticks_H
net_no_cost_ticks_H = delta_ticks_H - 0.5 buffer
exec_net_cost_0p25_H = net_no_cost_ticks_H - 0.25 * entry_cost_ticks_L2
exec_net_cost_0p50_H = net_no_cost_ticks_H - 0.50 * entry_cost_ticks_L2
exec_net_cost_1p0_H  = net_no_cost_ticks_H - 1.00 * entry_cost_ticks_L2
```

Tambien:

```text
target_exec_positive_cost_X_H = exec_net_cost_X_H > 0
target_exec_buffered_cost_X_H = exec_net_cost_X_H > +0.5 ticks
```

## Proxies de fill adverso

Para cada horizonte se calcula:

```text
adverse_fill_proxy_0p25_H
healthy_fill_proxy_0p25_H
```

El proxy adverse se define como:

```text
fila dentro del peor 25% de exec_net_cost_0p25 dentro de su grupo
```

Grupo:

```text
terminal_split
temporality
phase_bucket
entry_spread_bucket_L2
microprice_bucket_t
pm_2s_momentum_bucket_t
```

Solo se etiqueta si el grupo tiene al menos:

```text
100 filas
```

Esto no prueba fills reales. Es un proxy offline para entrenar al modelo a
distinguir condiciones sanas de condiciones peligrosas.

## Soporte por horizonte

Resumen `test_terminal`:

| Horizonte | Filas soportadas | Soporte | Neto medio coste 25% | Positivo coste 25% | Healthy proxy |
|---:|---:|---:|---:|---:|---:|
| `H16` | `406.514` | `93,74%` | `-0,6790` | `26,09%` | `26,05%` |
| `H32` | `382.714` | `88,25%` | `-0,6795` | `31,40%` | `31,37%` |
| `H60` | `341.418` | `78,73%` | `-0,6802` | `35,60%` | `35,57%` |
| `H120` | `254.064` | `58,59%` | `-0,6815` | `40,04%` | `39,99%` |
| `H240` | `83.022` | `19,15%` | `-0,6842` | `44,37%` | `44,26%` |

Lectura:

```text
el promedio global sigue siendo negativo, pero los horizontes largos aumentan
la proporcion de casos positivos.
```

Esto encaja con el probe anterior:

```text
no hay edge global, pero si puede haber subconjuntos aprendibles.
```

## Validacion contra baseline anterior

Se comparo:

```text
exec_net_cost_1p0_H16
```

contra:

```text
net_buffer_0p5_ticks_L2_H16
```

Resultado:

```text
filas comparadas: 2.183.856
max_abs_diff: 0,000057
mean_abs_diff: 0,0000009
99,83% de filas con diferencia < 1e-5
```

Conclusion:

```text
el builder reproduce el target L-aware H16 anterior; las diferencias son ruido
numerico de float.
```

## Como usarlo en el primer modelo

El entrenamiento debe leer:

```text
manifest.json
```

Y usar exclusivamente:

```text
manifest["feature_columns"]
```

Targets recomendados v1a:

```text
exec_net_cost_0p25_H60
exec_net_cost_0p25_H120
healthy_fill_proxy_0p25_H60
healthy_fill_proxy_0p25_H120
adverse_fill_proxy_0p25_H60
adverse_fill_proxy_0p25_H120
```

Targets auxiliares:

```text
exec_net_cost_0p50_H60
exec_net_cost_0p50_H120
target_exec_positive_cost_0p25_H60
target_exec_positive_cost_0p25_H120
```

## Siguiente paso

Entrenar el primer modelo Complex v1a:

```text
HGB/LightGBM tabular
```

Primera tarea:

```text
regresion de exec_net_cost_0p25_H60 y H120
```

Segunda tarea:

```text
clasificacion healthy_fill_proxy_0p25_H60/H120
```

Gates:

- validation positivo antes de mirar test;
- test terminal positivo;
- no depender de H240;
- no usar columnas `entry_*` como feature;
- comparar contra no operar y reglas por celda;
- revisar segmentos y feature importance.

## Addendum - Primer modelo tabular EV

El primer runner ya existe:

```text
docs/COMPLEX_V1A_TABULAR_EV_MODEL.md
```

Resultado:

```text
NO GO a bot.
NO GO a aprobar modelo tabular EV como estrategia.
GO a diagnostico de estabilidad temporal/regimen.
```

Resumen:

- `H60`: validation eligio `top_0.25pct`, pero test fue negativo.
- `H120`: no hubo policy positiva en validation, aunque test mostraba
  oportunidades fuertes.
- el modelo generaliza debilmente y depende mucho del reloj del contrato.

---

