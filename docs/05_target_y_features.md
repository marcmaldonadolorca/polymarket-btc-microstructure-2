# Target y mapa de features para EdgeRunner

Fecha: `2026-05-27`

Estado: documento de exploracion y decision previa a experimentos offline.

Guias pedagogicas actuales:

- [GUIA_01_FEATURES_Y_VARIABLES_SINTETICAS.md](GUIA_01_FEATURES_Y_VARIABLES_SINTETICAS.md)
- [GUIA_02_SPLITS_Y_VALIDACION.md](GUIA_02_SPLITS_Y_VALIDACION.md)
- [GUIA_03_PLAN_DE_ANALISIS_EXPLORATORIO.md](GUIA_03_PLAN_DE_ANALISIS_EXPLORATORIO.md)
- [GUIA_04_LABELS_TARGETS_Y_COSTES.md](GUIA_04_LABELS_TARGETS_Y_COSTES.md)
- [BASELINE_CONTRACT_V0.md](BASELINE_CONTRACT_V0.md)

Este documento queda como mapa tecnico detallado. Para entender el contexto
desde cero, empezar por las cuatro guias anteriores.

Notebooks asociados:

- [03_target_contract_and_feature_map.ipynb](../notebooks/03_target_contract_and_feature_map.ipynb)
- [04_alternative_target_exploration.ipynb](../notebooks/04_alternative_target_exploration.ipynb)
- [05_core_target_stability.ipynb](../notebooks/05_core_target_stability.ipynb)
- [06_pm_feature_coverage_and_distributions.ipynb](../notebooks/06_pm_feature_coverage_and_distributions.ipynb)
- [07_cross_venue_feature_audit.ipynb](../notebooks/07_cross_venue_feature_audit.ipynb)
- [08_pm_external_signal_stability_quick.ipynb](../notebooks/08_pm_external_signal_stability_quick.ipynb)
- [08b_pm_external_signal_stability_folds_quick.ipynb](../notebooks/08b_pm_external_signal_stability_folds_quick.ipynb)
- [09_action_unit_and_evaluation_quick.ipynb](../notebooks/09_action_unit_and_evaluation_quick.ipynb)

## Conclusion ejecutiva

Si, el target debe definirse y validarse especificamente. La existencia de
`predictor_labels` evita volver a capturar datos, pero no convierte
automaticamente sus labels en el objetivo final correcto para edge operable.

La recomendacion es:

1. usar los labels actuales solo como benchmark provisional y como auditoria
   de la trayectoria futura ya persistida;
2. recalcular offline familias de labels versionadas desde SQLite, sin tocar
   el recolector;
3. usar como target central de direccion un cambio de probabilidad del token
   en unidades absolutas/tick-normalizadas;
4. reservar el target realmente economico para una definicion de ejecucion
   explicita con bid/ask, fee, slippage y politica de entrada/salida.

## Que target tenemos hoy

La tabla `predictor_labels` actual se genera en
`src/sync/session_dataset_augmenter.py`, lineas `2587-2875`, del repositorio
de captura. La configuracion por defecto se fija en `config.py`, lineas
`159-161`.

Contrato materializado en la DB oficial:

| Elemento | Valor / formula actual |
|---|---|
| Grid | `2 s` |
| Horizonte | `4` pasos = `8 s` |
| Precio base | `polymarket_mid` del mismo `token_id` |
| `future_return_bps` | `((mid[t+8s] - mid[t]) / mid[t]) * 10000` |
| `label_name` | `up` si retorno `>= 5 bps`, `down` si `<= -5 bps`, si no `flat` |
| `microprice_label_name` | misma regla sobre microprice; fallback al retorno de mid |
| `economic_cost_bps` | `spread_bps_current / 2 + 2 bps` |
| `economic_net_return_bps` | retorno relativo menos coste en la direccion del movimiento |
| `economic_label_name` | misma clasificacion `up/flat/down` del retorno ajustado |
| `economic_target_bps` | `economic_cost_bps + 5 bps` |
| `target_up_hit/down_hit` | si el path de los siguientes cuatro frames toca el target economico |
| `first_target_hit_side` | primer lado del barrier hit en esos `8 s` |

El label no predice la resolucion final de BTC ni si BTC subira. Predice el
movimiento inmediato del precio del token de Polymarket observado.

## Evidencia de la auditoria

Se uso una muestra determinista del baseline:

- sesiones `core` aceptadas por calidad;
- `session_id % 100 = 0`;
- `24.288` labels.

| Comprobacion | Resultado |
|---|---:|
| Reproduccion de `label_name` con formula implementada | `100,00%` |
| Politica observada | `H=8 s`, `neutral=5 bps`, `buffer=2 bps` |
| `tick_size` observado | `0,01` |
| Umbral direccional mediano expresado en ticks | `0,025` ticks |
| Umbral economico mediano expresado en ticks | `0,535` ticks |
| Mercados de muestra con `fees_enabled=1` | todos |
| `fee_rate_bps` observado | `700` |
| Filas token de la muestra | `24.288` |
| Instantes-mercado unicos de la muestra | `12.144` |
| Pares direccionales `down + up` | `5.628` |
| Pares direccionales `flat + flat` | `6.514` |

Hay dos filas de la muestra cuyo `economic_label_name` no se reproduce desde
las columnas guardadas: almacenan `current_mid == future_mid` y
`future_return_bps == 0`, pero conservan un net return negativo. El codigo
local revisado no explica ese resultado para retorno cero, por lo que debe
registrarse como anomalia de generacion/versionado pendiente de reproducir
desde observaciones base. No afecta al volumen, pero confirma que una familia
nueva debe recomputarse desde observaciones, no transformando labels
materializados.

## Problemas del target actual

### 1. Retorno relativo del token

El movimiento se divide por `mid[t]`. En un contrato binario, el mismo cambio
absoluto de probabilidad pesa mucho mas si el token cotiza cerca de `0` que si
cotiza cerca de `1`.

Para un predictor transferible entre tokens y ventanas, interesa estudiar
primero:

```text
delta_prob_bp = (mid[t+H] - mid[t]) * 10000
```

o su version en ticks:

```text
delta_ticks = (mid[t+H] - mid[t]) / tick_size
```

### 2. Umbral direccional demasiado fino para edge

Con `tick_size = 0,01`, el umbral de `5 bps` relativo equivale en mediana a
`0,025` ticks. `label_name` es util para movimiento, pero no deberia venderse
como movimiento ejecutable.

### 3. Target economico incompleto

`economic_label_name` incorpora medio spread y un buffer fijo, pero no se ve
que incorpore:

- formula efectiva de fees;
- slippage segun tamano;
- probabilidad de fill;
- coste de salida;
- adverse selection;
- politica maker/taker.

Tambien requiere una auditoria de consistencia completa: ya se han observado
filas persistidas que no son reproducibles con la implementacion local
inspeccionada y los valores almacenados.

Ademas, un label `down` sobre un token no equivale automaticamente a una
operacion corta ejecutable; puede requerir operar el token complementario o
definir una politica de no entrada/salida.

### 4. Contrato documental previo distinto

`polymarket_btc_probe/docs/labeling.md` planteaba como contrato de diseno un
target de delta absoluto, horizontes `5/15/60 s` a grid de `1 s` y labels
cost-aware fuera del core. La DB oficial contiene otro contrato operativo:
retorno relativo a `8 s` sobre grid de `2 s`.

No es un fallo del corpus. Es una razon para versionar los labels que se
aprueben para EdgeRunner y no llamarlos todos "canonicos" sin apellido.

### 5. Dos tokens complementarios no son dos evidencias independientes

Cada instante de un mercado binario contiene dos tokens. En la muestra, todas
las agrupaciones `session_id + market_id + condition_id + time_index_ns`
contienen exactamente dos filas. La gran mayoria son parejas espejo:

- `down + up`;
- `flat + flat`.

Consecuencias:

- el split debe mantener completa cada `condition_id` en un solo fold;
- accuracy y class balance por fila deben complementarse con metricas
  agrupadas por instante-mercado;
- para una estrategia de compra de un lado, puede convenir reorientar el
  problema a "que token comprar / no operar" en vez de considerar ambos tokens
  como ejemplos independientes.

## Familias de target recomendadas

No conviene elegir un solo target demasiado pronto. Conviene generar familias
offline comparables desde las observaciones SQLite.

| Familia | Definicion | Uso | Prioridad |
|---|---|---|---|
| `existing_rel_return_8s_v1` | labels actuales de `predictor_labels` | Benchmark de continuidad y comparacion con lo ya capturado | Ya disponible |
| `prob_delta_3c_H_v1` | `(mid[t+H]-mid[t]) / tick_size`, clases por banda en ticks | Primer target direccional interpretable | Alta |
| `microprice_delta_H_v1` | delta de microprice, con mascara de fuente | Analisis auxiliar de presion de libro | Media |
| `path_barrier_prob_H_v1` | primer cruce de barreras en delta/ticks | Comparar movimiento terminal frente a oportunidad intrahorizonte | Alta tras direccional |
| `executable_markout_H_v1` | entrada a ask/bid observable y markout futuro neto de coste definido | Target economico principal | Alta, requiere especificacion |
| `fill_then_markout_H_v1` | fill simulado, slippage y PnL posterior | Modelo de ejecucion/riesgo | Posterior |

### Horizonte y bandas que hay que estudiar

El corpus permite recalcular sin recaptura dentro de cada sesion. Para la
primera comparacion:

| Dimension | Valores a explorar |
|---|---|
| Horizonte corto | `4 s`, `8 s`, `16 s` |
| Horizonte tactico posterior | `30 s`, `60 s`, limitado por borde de sesion/mercado |
| Banda absoluta | `0,5 tick`, `1 tick` |
| Output | regresion de delta y clasificacion `down/flat/up` |
| Censura | no usar paths incompletos, stale/missing ni ventanas que crucen frontera invalida |

La seleccion no debe hacerse por balance bonito de clases, sino por utilidad
economica y estabilidad walk-forward.

## EDA 04 - Primera comparacion desde observaciones

El notebook `04_alternative_target_exploration.ipynb` recalcula targets
alternativos desde `polymarket_grid_rows`. `predictor_labels` solo selecciona
las claves de instantes aceptados; no proporciona el precio futuro para el
nuevo label.

Metodo exploratorio:

- baseline `core` con el filtro de calidad de EDA 01;
- muestra temporal determinista `session_id % 250 = 0`, repartida en `10`
  sesiones entre `2026-05-12` y `2026-05-25`;
- extraccion estrecha e indexada sesion a sesion, apropiada para explorar una
  SQLite de `68,95 GiB` sin construir aun una pipeline;
- `9.140` observaciones recuperadas y `8.660` filas en soporte comun sin
  `missing/stale` y con futuro disponible hasta `H=16 s`;
- los desplazamientos conservan exactamente el horizonte: cero saltos
  invalidos entre los `9.020` futuros a `4 s`, `8.900` a `8 s` y `8.660` a
  `16 s`;
- `30` mercados y `60` tokens presentes en el soporte.

### Target direccional tick-normalizado

Se define:

```text
delta_ticks_H = (mid[t+H] - mid[t]) / tick_size
```

| Horizonte | Banda | Down | Flat | Up |
|---:|---:|---:|---:|---:|
| `4 s` | `0,5 tick` | `18,01%` | `63,93%` | `18,06%` |
| `4 s` | `1 tick` | `15,65%` | `68,55%` | `15,81%` |
| `8 s` | `0,5 tick` | `24,18%` | `51,56%` | `24,26%` |
| `8 s` | `1 tick` | `21,48%` | `56,56%` | `21,96%` |
| `16 s` | `0,5 tick` | `30,17%` | `39,54%` | `30,29%` |
| `16 s` | `1 tick` | `27,73%` | `43,95%` | `28,33%` |

El label observacional `H=8 s`, banda `0,5 tick`, coincide con el
`label_name` persistido en `99,56%` del soporte comun. Esto valida la
reconstruccion del movimiento, no el contrato economico existente.

### Primer markout economico exploratorio

Se calculo `taker_roundtrip_markout_H_v0` comprando a
`mid + spread/2`, saliendo a `future_mid - future_spread/2` y aplicando a
entrada y salida la formula de fee local existente en el repositorio de
captura. Es una aproximacion conservadora para explorar; no modela fill,
cola, slippage adicional, tamano ni politica de ejecucion.

| Horizonte | Compras netas `>= 0 ticks` | Compras netas `>= 1 tick` | Mediana neta |
|---:|---:|---:|---:|
| `4 s` | `8,65%` | `5,47%` | `-1,40 ticks` |
| `8 s` | `13,66%` | `9,48%` | `-1,41 ticks` |
| `16 s` | `20,07%` | `14,78%` | `-1,42 ticks` |

Para `H=8 s`, el fee roundtrip estimado equivale a una mediana de `68,55`
bps respecto al ask de entrada. La lectura importante es que predecir
direccion no basta: el target economico debe considerar costes desde su
definicion.

### Decision provisional

El primer baseline debe estudiar `prob_delta_3c_H_v1` calculado desde
observaciones para `H=8 s` y `H=16 s`, usando `1 tick` como banda principal y
`0,5 tick` como sensibilidad. `taker_roundtrip_markout_H_v0` queda como label
economico secundario para medir si la senal sobrevive a costes basicos, no
como PnL final.

Antes del modelo se debe extender esta misma extraccion estrecha a todo
`core`, con agregados por dia y temporality, manteniendo los splits
walk-forward purgados y agrupando la dependencia por `condition_id`.

Nota operativa: durante la exploracion la unidad `D:` se desconecto
transitoriamente y volvio a estar disponible sin modificar SQLite. Las
extracciones amplias deben ser reanudables y registrar el conjunto de claves
procesado; esto no justifica copiar ni transformar la fuente oficial antes de
necesitarlo.

## EDA 05 - Estabilidad en core completo

Informe asociado:
[EDA_05_TARGET_STABILITY.md](EDA_05_TARGET_STABILITY.md)

La EDA 05 recalculo los targets alternativos sobre todo el baseline core,
leyendo `4.999.680` filas de `polymarket_grid_rows` y reteniendo
`2.456.582` filas limpias con soporte comun hasta `H=16 s`.

Resultado principal con banda `1 tick`:

| Horizonte | Down | Flat | Up |
|---:|---:|---:|---:|
| `8 s` | `19,61%` | `60,58%` | `19,80%` |
| `16 s` | `25,18%` | `49,43%` | `25,39%` |

Markout taker roundtrip exploratorio:

| Horizonte | Neto `>= 0 ticks` | Neto `>= 1 tick` |
|---:|---:|---:|
| `8 s` | `12,69%` | `8,62%` |
| `16 s` | `18,02%` | `13,30%` |

Decision actual:

- `delta_ticks_16s` con banda `1 tick` pasa a ser candidato principal de
  clasificacion de movimiento;
- `delta_ticks_8s` con banda `1 tick` queda como benchmark comparable al label
  materializado actual;
- `0,5 tick` queda como sensibilidad;
- los labels economicos de markout se mantienen como metrica secundaria, no
  PnL final;
- el siguiente paso ejecutado fue EDA 06 de cobertura y distribucion de
  features PM.

## EDA 06 - Cobertura y senal inicial de features PM

Informe asociado:
[EDA_06_PM_FEATURE_COVERAGE.md](EDA_06_PM_FEATURE_COVERAGE.md)

La EDA 06 estudio `2.457.308` filas del soporte PM con `mid` y `spread`
validos. Las columnas PM basicas tienen cobertura completa en el soporte:
`mid`, `microprice`, `spread`, `spread_ticks`, `microprice_gap_ticks`,
`boundary_distance`, `visible_entry_cost_ticks`, `tick_size`, `fee_rate_bps`,
`age_ms`, `seconds_to_window_end`, `stale` y `missing`.

Hallazgos clave:

- `microprice_gap_ticks` es la senal univariante PM mas fuerte frente a
  `delta_ticks_16s` (`Spearman` de muestra `0,1958`);
- los deciles extremos de `microprice_gap_ticks` cambian fuertemente la
  probabilidad direccional: gap muy negativo da `39,16%` down y gap muy
  positivo da `39,48%` up;
- `mid_delta_ticks_2s` y `mid_delta_ticks_8s` aportan momentum interpretable,
  aunque menos limpio que microprice;
- `spread_ticks` y `visible_entry_cost_ticks` parecen mas utiles para decidir
  operabilidad/coste que direccion pura;
- `trade_imbalance` crudo esta saturado en `1,0` y debe tratarse como feature
  secundaria o transformada;
- `window_progress`, `seconds_to_window_end` y `phase_bucket` son candidatas
  fuertes porque EDA 05 ya mostro diferencias de movimiento por fase.

Decision actual de shortlist PM:

- alta: `mid`, `boundary_distance`, `microprice_gap_ticks` clipped,
  `mid_delta_ticks_2s`, `mid_delta_ticks_8s`, `spread_ticks`,
  `visible_entry_cost_ticks`, `seconds_to_window_end`, `window_progress`,
  `phase_bucket`, `age_ms` y masks;
- media: `spread_delta_ticks_8s`, `trade_imbalance` suavizado, calendario como
  ablation;
- pendiente: `cross_venue_features` en EDA 07, porque los escaneos anchos
  castigaron el disco y deben hacerse como extraccion estrecha.

## EDA 07 - Cross-venue externo y lead-lag inicial

Informe asociado:
[EDA_07_CROSS_VENUE_AUDIT.md](EDA_07_CROSS_VENUE_AUDIT.md)

La EDA 07 leyo `2.486.400` filas core de `cross_venue_features` y reconstruyo
un target H16 en `2.355.052` filas. La tabla cross es util, pero cara de
escanear: un escaneo estrecho de `81` columnas tardo `3.845,55 s`.

Hallazgos clave:

- spot externo y perp son densos y aportan senal si se orientan por outcome;
- `perp_mid_return_bps_2s_oriented` tuvo `Spearman` de muestra `0,1428`;
- `external_mid_return_bps_2s_oriented` tuvo `Spearman` de muestra `0,1351`;
- los retornos `8s` siguen siendo utiles, aunque mas debiles;
- Chainlink esta cubierto, pero su staleness mediana es alta y no aparece como
  senal tick-a-tick principal;
- gaps spot/perp, consenso, freshness y trade-rate son mas contexto/masks que
  predictores direccionales aislados;
- Coinbase, Hyperliquid y depth siguen fuera de v1.

Shortlist externa actual:

- alta: `external_mid_return_bps_2s_oriented`,
  `external_mid_return_bps_8s_oriented`,
  `perp_mid_return_bps_2s_oriented`,
  `perp_mid_return_bps_8s_oriented`;
- media: `perp_mark_price_return_bps_2s_oriented`,
  `external_trade_imbalance_oriented`,
  `perp_taker_buy_sell_log_oriented`;
- contexto/masks: `chainlink_staleness_ms`, `freshness_gap_ms`,
  `joint_age_ms`, `missing_context_count`, `reference_consensus_dispersion_bps`;
- enriquecido: `intertemporal_mid_vs_group_mean` y vecinos, con mask.

La siguiente EDA debe medir estabilidad combinada PM + externo antes de
entrenar.

## EDA 08 quick - Interaccion PM + externo

Informe asociado:
[EDA_08_PM_EXTERNAL_SIGNAL_STABILITY_QUICK.md](EDA_08_PM_EXTERNAL_SIGNAL_STABILITY_QUICK.md)

Esta EDA redujo el tamano de iteracion: muestra determinista por `id`, `99.040`
filas leidas y `93.496` filas con soporte H16 exacto.

Resultado principal:

- `polymarket_microprice_gap_bps`: `Spearman 0,1764`;
- `perp_mid_return_bps_2s_oriented`: `Spearman 0,1565`;
- `external_mid_return_bps_2s_oriented`: `Spearman 0,1338`;
- PM positivo + perp positivo: `50,27%` up;
- PM negativo + perp negativo: `51,52%` down;
- conflicto PM/perp vuelve a distribucion casi neutral.

Decision: las interacciones simples entre presion PM y movimiento externo son
candidatas reales para el primer baseline. El siguiente paso debe confirmar
estabilidad con muestra equilibrada y coste visible, no lanzar aun un full
scan ancho.

## EDA 08b quick - Balanceo y coste visible

Informe asociado:
[EDA_08B_PM_EXTERNAL_BALANCED_COST_QUICK.md](EDA_08B_PM_EXTERNAL_BALANCED_COST_QUICK.md)

La EDA 08b balanceo la muestra por `terminal_split + temporality`:

- `36.000` filas balanceadas;
- `4.000` por cada combinacion split/temporalidad;
- `spread` PM recuperado con cobertura `100%` para la muestra;
- `visible_entry_cost_ticks` mediano `0,7142`.

Resultados:

- `polymarket_microprice_gap_bps`: `Spearman 0,1938`;
- `perp_mid_return_bps_2s_oriented`: `Spearman 0,1553`;
- `external_mid_return_bps_2s_oriented`: `Spearman 0,1310`;
- PM+perp ambos positivos: `51,44%` up;
- PM+perp ambos negativos: `54,13%` down;
- senal alineada con coste visible bajo: `60,00%` de hit direccional.

Decision: coste visible entra en el contrato v0 como feature/filtro/regimen.
Antes de entrenar hay que cerrar unidad de accion y evaluacion por
market-frame para evitar doble conteo de tokens complementarios.

## EDA 09 quick - Unidad de accion y evaluacion

Informe asociado:
[EDA_09_ACTION_UNIT_EVALUATION_QUICK.md](EDA_09_ACTION_UNIT_EVALUATION_QUICK.md)

La EDA 09 confirmo que los pares complementarios de la muestra son completos:

- `46.748` market-frames completos;
- `100%` con dos tokens, uno `Up` y uno `Down`;
- mediana `abs(mid_up + mid_down - 1) = 0`;
- `down+up` aparece en `51,60%` de frames;
- `flat+flat` en `43,99%`.

Decision:

- el baseline v0 puede entrenarse por fila token;
- la evaluacion principal y la accion deben agregarse por
  `session_id + condition_id + time_index_ns`;
- una accion simple compra el token con senal positiva (`both_pos`) dentro del
  par;
- `both_neg` sobre un token no es una segunda oportunidad: es evidencia de que
  el complementario podria ser el lado comprable.

Con la regla simple `buy_one_token`, la muestra obtuvo:

| Metrica | Valor |
|---|---:|
| Cobertura de acciones sobre market-frames | `8,47%` |
| Hit `up` del token comprado | `50,27%` |
| Flat | `31,76%` |
| Wrong `down` | `17,98%` |

Siguiente paso: contrato del baseline v0.

## Contrato baseline v0

Documento asociado:
[BASELINE_CONTRACT_V0.md](BASELINE_CONTRACT_V0.md)

Decision cerrada:

- target principal: `delta_ticks_16s`, banda `1 tick`;
- benchmark: `delta_ticks_8s`, banda `1 tick`;
- entrenamiento: fila token;
- evaluacion/accion: market-frame;
- features v0: PM microprice/deltas/spread/coste/reloj + spot/perp retornos
  orientados;
- fuera de v0: contexto nativo holders/OI, sidecars, Coinbase, Hyperliquid,
  depth y modelos complejos;
- proximo paso: primer baseline offline rapido.

## Principios para features

Cada feature propuesta se clasifica asi:

- `v1 causal`: calculable con informacion disponible en o antes de `t`.
- `requiere auditoria as-of`: existe como contexto, pero hay que probar que no
  resume eventos posteriores a `t`.
- `filtro offline`: se usa para aceptar/excluir muestras, nunca como input
  online.
- `holdout`: no entra en v1 aunque pudiera calcularse.

Los ids (`session_id`, `market_id`, `token_id`, `condition_id`) son claves de
join, split y purga, no features predictivas por defecto.

## Bloque 1 - Microestructura Polymarket causal

Objetivo: representar el estado y dinamica inmediata del activo que realmente
se negocia.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `pm_mid`, `pm_mid_delta_1/2/4_steps`, `pm_mid_return_abs_bp` | `cross_venue_features.polymarket_mid` | v1 causal, alta |
| `pm_microprice`, `pm_microprice_minus_mid_bp/ticks` | `polymarket_grid_rows` y gap ya en `cross_venue_features` | alta; EDA 06 la marca como primera senal PM |
| `pm_spread_abs`, `pm_spread_ticks`, `pm_half_spread_cost_bp` | `polymarket_grid_rows.spread` + `market_metadata.tick_size` | esencial para target economico |
| `pm_trade_imbalance`, lags y EMA | `polymarket_grid_rows.trade_imbalance` | causal, pero cruda esta saturada; usar transforms |
| `mid_change_count_W`, `time_since_mid_change` | secuencia de `polymarket_mid` | alta |
| `pm_realized_range_W`, `pm_realized_vol_W`, `pm_momentum_W` | secuencia pasada del token | alta |
| `pm_microprice_signal_persistence_W` | microprice gap pasado | media |
| `stale/missing_prefix_W` | `polymarket_grid_rows.stale/missing` | mascara causal, alta |

Nota: el objeto de features original incluia spread, microprice y trade
imbalance de Polymarket, pero la tabla columnar `cross_venue_features` oficial
no expone todos esos campos. Para un dataset v1 serio deben promoverse desde
SQLite como observaciones causales, no tomarse desde `predictor_labels`.

## Bloque 2 - Semantica de contrato y reloj de ventana

Objetivo: distinguir el mismo estado de libro segun el contrato y su tiempo
restante.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `temporality` one-hot / embedding | `cross_venue_features`, `market_metadata` | v1 causal |
| `market_kind`, `outcome_label` | features/contexto token | v1 causal |
| `seconds_to_window_end`, `window_progress` | `time_index_utc` + `market_metadata.window_end_utc/start_utc` | v1 causal, alta |
| `phase_near_close` buckets | derivada anterior | alta |
| `tick_size`, `min_order_size` | `market_metadata` | alta |
| `fees_enabled`, fee-policy bucket | `market_metadata` | alta para ejecucion; verificar formula fee |
| `liquidity_num`, `volume_24h/1wk`, `open_interest` transformados | `market_metadata` | usar solo si snapshot es conocido en `t` |
| `price_distance_to_boundary_0_1` | `polymarket_mid` | alta |

## Bloque 3 - Referencias core: Binance spot, perp y Chainlink

Objetivo: medir repricing externo y contexto de derivados ya densos.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `external_mid`, retornos lag/rolling, momentum y acceleration | `cross_venue_features` | v1 causal, alta |
| `external_trade_imbalance` rolling/EMA/sign changes | `cross_venue_features` | causal; prioridad media tras EDA 07 |
| `external_realized_vol_bps_5s/15s` y extensiones `30s/60s` | materializado + rolling causal | alta |
| `perp_mid`, `perp_mark_price`, mark-mid gap | `cross_venue_features` | v1 causal; retornos `2s/8s` orientados son alta |
| `spot_perp_mid_gap_bps`, `spot_perp_mark_gap_bps` lags/z-score | materializado + rolling | alta |
| `perp_basis`, mask y cambio temporal | `cross_venue_features` | alta, missing-aware |
| `perp_open_interest` nivel/cambio/velocidad | `cross_venue_features` | media-alta si updates son causales |
| `perp_funding_rate`, `taker_buy_sell_ratio` regimen/lags | `cross_venue_features` | media-alta |
| `chainlink_price`, staleness y gaps spot/perp | `cross_venue_features` | cobertura alta; mas contexto/mask que senal directa |
| `gap_normalized_by_external_vol` | gaps / realized vol | alta |
| `lead_lag_external_to_pm_W` | lags de retorno externo frente a cambios pasados PM | investigacion alta, solo pasado |

## Bloque 4 - Incoherencias internas e intertemporales

Objetivo: explotar consistencia entre tokens complementarios y mercados de
distinta duracion sin usar futuro.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `complement_mid_sum_minus_one` | filas simultaneas por `market_id` | alta |
| `complement_spread/liquidity_asymmetry` | `polymarket_grid_rows` por pareja | alta tras promover BBO |
| `outcome_relative_mid` orientado a UP/YES | token context + mid | alta para comparabilidad |
| `best_side_signal` / `no_trade` por pareja | tokens simultaneos del mercado | alternativa de target/decision, evita doble conteo |
| `intertemporal_mid_vs_group_mean` | ya materializada | v1 enriquecida |
| `intertemporal_mid_vs_nearest_shorter/longer` | ya materializada | v1 enriquecida |
| `intertemporal_group_range`, `slope`, `curve_residual` | ya materializadas | media-alta |
| rolling/z-score de residual intertemporal | secuencia pasada | media-alta |

## Bloque 5 - Calidad causal y masks

Objetivo: que el modelo sepa cuando el dato visible en `t` es incompleto, sin
darle resumentes futuros de sesion.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `chainlink_missing`, `chainlink_staleness_ms` | `cross_venue_features` | v1 causal |
| `freshness_gap_ms`, `joint_age_ms` | `cross_venue_features` | v1 causal |
| `cross_ready`, `cross_ready_with_chainlink` | `cross_venue_features` | mascara/filtro |
| `missing_context_count`, `stale_context_count` | materializada | v1 causal si construida por frame |
| rolling past-only de missing/stale | secuencia pasada | alta |
| `coverage_ratio`, `quality_score`, gap counts de sesion | telemetry | filtro offline, nunca feature |
| `full_book_ratio`, `tradability_status` final | tradability | filtro/auditoria offline, no feature directa |

## Bloque 6 - Contexto nativo Polymarket

Objetivo: medir crowding y actividad estructural. Este bloque es prometedor,
pero no se promociona a input hasta comprobar su tiempo de observacion.

| Feature sintetizable | Fuente SQLite | Estado / uso |
|---|---|---|
| `holder_count`, `holder_total_amount`, `top_share`, `top3_share` | `session_token_context` | auditoria `as-of` necesaria |
| `holder_concentration_gap_between_outcomes` | contexto de tokens complementarios | auditoria `as-of` |
| `open_interest_rest/subgraph`, diferencia relativa | contextos market/token | auditoria `as-of` |
| `activity_split/merge/redemption_mix` | `session_market_context` | auditoria `as-of` |
| `order_filled/matched_volume` y ratios | contextos | auditoria `as-of` |
| `activity_recency_at_t` | timestamps de contexto vs frame | solo si evento es anterior a `t` |
| masks de cobertura de contexto | contextos | permitidas tras auditoria |

## Bloque 7 - Coste, ejecutabilidad y labels enriquecidos

Objetivo: transformar prediccion de movimiento en pregunta operable. Gran
parte de este bloque define labels o backtest, no inputs del predictor.

| Variable sintetizable | Fuente SQLite | Papel |
|---|---|---|
| `entry_ask_est`, `entry_bid_est` | mid + spread causal | simulacion/label |
| `spread_ticks`, `notional_min_order` | spread/tick/min size | input de riesgo y label |
| `fee_estimate` conforme a politica real | metadata + formula validada | label/backtest |
| `markout_after_entry_H` | precios futuros solo en labels | target economico |
| `barrier_hit_after_cost_H` | trayectoria futura solo en labels | target |
| `fill_proxy`, `time_to_fill`, `adverse_selection` | requiere politica de fill y libro/trades | fase posterior |
| `do_not_trade_due_to_cost` | coste visible en `t` | regla baseline/riesgo |

## Bloque 8 - Sidecars opcionales y holdout

Objetivo: mantener abierta la expansion sin contaminar v1.

| Familia | Estado actual | Decision |
|---|---|---|
| `liquidation_aggregates` | presente pero `trainable_rows = 0` | mascara/experimento separado |
| `binance_spot_depth`, `binance_perp_depth` | ausente en corpus entrenable | fuera v1 |
| `coinbase_spot_anchor`, `coinbase_exchange_depth` | ausente | fuera v1 |
| `deribit_volatility` | ausente | fuera v1 |
| `hyperliquid_*` en auditoria de muestra | presencia `0%` | holdout |

## Orden de ampliacion recomendado

### Ahora, dentro de exploracion

1. Auditar labels actuales y calcular distribucion de targets alternativos en
   delta absoluto/ticks.
2. Promover a una vista exploratoria causal los campos nativos de Polymarket
   necesarios para costes: `mid`, `microprice`, `spread`, `trade_imbalance`,
   `stale`, `missing`.
3. Auditar `as-of` de contextos de mercado/token.
4. Medir dependencia entre tokens complementarios y elegir la unidad de
   evaluacion (`token` frente a `market-frame`).
5. Medir cobertura y correlacion de los bloques 1 a 5, sin entrenar aun un
   predictor complejo.

### Primera lluvia de ideas / experimentos

1. Baseline tabular v0 con bloques PM + spot/perp + reloj y target
   `delta_ticks_16s` con banda `1 tick`: completado en modo quick en
   `docs/BASELINE_V0_QUICK_RESULTS.md` y ampliado a full-core robustness en
   `docs/BASELINE_V0_FULL_CORE_ROBUSTNESS.md`.
2. Repeticion con ablations: completada. Decision: PM solo no basta; PM+perp
   es el bloque clave; `time_only` queda descartado como explicacion del edge.
3. Variante con `economic_markout_8s_v1` cuando la formula de coste este
   definida.
4. Ablaciones ampliadas por bloques: microestructura, referencias externas, reloj de
   ventana e intertemporal.
5. Validacion `expanding walk-forward` purgada y test terminal sellado.

### Mucho despues

- contexto nativo promocionado tras auditoria temporal;
- encoder de libro;
- labels de fill/slippage/adverse selection;
- sidecars solo si se vuelve a disponer de cobertura entrenable real.
