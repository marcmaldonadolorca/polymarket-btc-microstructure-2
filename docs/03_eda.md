# Guia 03 - Plan de analisis exploratorio antes de modelar

Fecha: `2026-05-27`

## Por que estamos haciendo EDA

EDA significa analisis exploratorio de datos. En este proyecto no es una fase
decorativa: antes de entrenar debemos descubrir si:

- los datos son limpios y estan disponibles cuando creemos;
- los labels significan lo que necesitamos;
- las features tienen cobertura, variacion y relacion posible con el futuro;
- la senal parece estable en tiempo y por tipo de mercado;
- una senal de movimiento podria sobrevivir a costes.

Un modelo muy potente puede aprender errores, leakage o costes ignorados. La
EDA debe quitar esos riesgos antes de automatizar.

## Orden correcto del trabajo

```text
1. Entender tablas y calidad
2. Entender labels y alternativas
3. Entender features y sus relaciones
4. Comprobar estabilidad temporal y splits
5. Formular baselines simples
6. Entrenar offline
7. Solo mucho despues: dataset consolidado, pipeline o bot
```

## Que ya esta hecho

| Artefacto | Pregunta contestada | Hallazgos principales |
|---|---|---|
| `01_official_core_eda.ipynb` | Esta limpio el corpus y que capas existen? | `2.590` sesiones core, calidad alta, joins validos, sidecars no entrenables; tradability no debe filtrarse a ciegas. |
| `02_split_strategy_audit.ipynb` | Es suficiente un split fijo o random? | Hay drift de clase `flat`; se recomienda walk-forward purgado y test terminal. |
| `03_target_contract_and_feature_map.ipynb` | Que label materializado existe y que problemas tiene? | El label existente usa `H=8 s`, retorno relativo y un umbral muy fino; el economico no queda aprobado como final. |
| `04_alternative_target_exploration.ipynb` | Que pasa si recalculamos targets desde observaciones? | En ticks son interpretables; `H=8/16 s` y banda `1 tick` merecen estudio; costes reducen mucho las acciones atractivas. |
| `05_core_target_stability.ipynb` | Son estables esos targets en todo core? | `H=16 s`, banda `1 tick`, queda como candidato principal; `H=8 s` queda como benchmark; siguiente paso: features. |
| `06_pm_feature_coverage_and_distributions.ipynb` | Que features PM tienen cobertura y senal inicial? | Microestructura PM usable; `microprice_gap_ticks` es la senal univariante mas fuerte; `cross_venue_features` queda para EDA estrecha aparte. |
| `07_cross_venue_feature_audit.ipynb` | Que referencias externas son densas y tienen senal? | Spot/perp retornos orientados aportan senal; Chainlink/gaps son mas contexto que direccion; Coinbase/Hyperliquid fuera v1. |

Las cuatro guias actuales convierten esos hallazgos en un plan legible. No
sustituyen futuras mediciones sobre el corpus completo.

## Preguntas de la siguiente exploracion

### Pregunta A - Que target debemos modelar?

Necesitamos medir sobre todo `core`:

- distribucion de `delta_ticks` a `4`, `8` y `16 s`;
- clases `up/flat/down` para bandas `0.5` y `1 tick`;
- markout neto aproximado con distintos supuestos de coste;
- estabilidad por dia, `temporality` y fase de ventana.

Respuesta esperada:

- escoger uno o dos targets para baselines, no veinte.

### Pregunta B - Que features existen realmente y son utilizables?

Mediremos:

- cobertura no nula de cada feature;
- numero de valores unicos y variables constantes;
- staleness/missingness;
- distribuciones, extremos y escalas;
- si un dato requiere auditoria `as-of`.

Respuesta esperada:

- lista corta de features `v1 causal`;
- lista separada de contexto pendiente o features descartadas.

### Pregunta C - Hay relacion entre features presentes y futuro?

Si, aqui entran correlaciones, pero bien hechas.

Mediremos:

- correlacion entre features numericas en `t` y `delta_ticks_H` futuro;
- information coefficient por fold/dia/temporality;
- tablas binned: por ejemplo, que ocurre con el target cuando el spread esta
  bajo/alto o cuando el gap spot-PM es extremo;
- posibles no linealidades y relaciones condicionales.

Respuesta esperada:

- hipotesis razonables para un baseline tabular;
- no una promesa de edge basada en una sola correlacion global.

### Pregunta D - Que representa una operacion posible?

Mediremos:

- spread y fee visible;
- porcentaje de movimientos que supera coste aproximado;
- diferencia entre acertar direccion y obtener markout neto positivo;
- que lado del par complementario seria comprable.

Respuesta esperada:

- saber si tiene sentido seguir solo con prediccion direccional o introducir
  una politica de `no_trade`.

## Por que mirar correlaciones no basta

Una correlacion es util como primera linterna, pero no demuestra estrategia.

Ejemplos:

| Situacion | Por que una correlacion global puede enganar |
|---|---|
| Feature cambia segun la hora o el dia | La relacion puede venir de regimen, no de prediccion. |
| Muchisimas filas cada `2 s` | Las filas no son independientes; una significancia aparente puede estar inflada. |
| Tokens complementarios | Contar ambos como observaciones independientes duplica evidencia. |
| Relacion solo en `5m` | El promedio global puede ocultar que falla en `15m/1h`. |
| Relacion sin costes | Puede predecir pequenos movimientos imposibles de monetizar. |

Por eso usaremos correlaciones como diagnostico y validaremos la senal por
tiempo y coste mas adelante.

## Tipos de analisis y graficos que haremos

### 1. Inventario, calidad y cobertura

| Analisis | Graficos/tablas | Objetivo |
|---|---|---|
| Conteo de sesiones/filas por tier | Tabla resumen | Confirmar base `core`. |
| Cobertura por dia y temporality | Heatmap/tabla | Ver huecos o sesgos. |
| Quality, coverage, continuity, gaps | Histogramas/tabla | Decidir filtros offline. |
| Cobertura de contextos y sidecars | Tabla de presencia | No usar fuentes ausentes. |
| Tradability por estado/ratio | Tabla + distribucion | Entender, no excluir por nombre. |

Estado: iniciado en EDA 01; se ampliara solo donde afecte targets/features.

### 2. Labels y targets alternativos

| Analisis | Graficos/tablas | Objetivo |
|---|---|---|
| Distribucion de label actual | Barras por clase | Entender benchmark existente. |
| `delta_ticks_H` continuo | Histogramas/percentiles | Ver escala natural del movimiento. |
| `up/flat/down` por horizonte/banda | Barras apiladas | Ver equilibrio y dificultad. |
| Target por dia/temporality/fase | Heatmaps | Medir drift. |
| Net markout bajo coste | Distribucion y porcentaje positivo | Separar senal de rentabilidad. |

Estado: ejecutado en EDA 03/04 y extendido a todo `core` en EDA 05.

### 3. Distribucion de features y missingness

| Analisis | Ejemplos | Objetivo |
|---|---|---|
| Distribuciones univariantes | `mid`, `spread_ticks`, gaps, vol, imbalance | Escala, outliers y transformaciones. |
| Missing/null/stale por columna | porcentajes por dia/temporality | Elegir masks y excluir campos inutiles. |
| Variables constantes o casi constantes | nunique, desviacion | No entrenar ruido. |
| Distribucion cerca de limites | `mid` cerca de 0/1 | Entender tick/fee y sesgo de target. |
| Calendario | hora UTC, dia de semana, fin de semana | Identificar regimenes, no asumir causalidad. |

### 4. Dinamica temporal de Polymarket

| Analisis | Ejemplos | Objetivo |
|---|---|---|
| Autocorrelacion de `mid`/deltas | lags de `2-60 s` | Elegir ventanas, entender dependencia. |
| Spread frente a movimiento futuro | bins de `spread_ticks` | Saber cuando el mercado es operable. |
| Microprice gap frente a delta futuro | bins/correlacion por fold | Probar presion de libro. |
| Trade imbalance frente a delta | quantiles/curvas | Probar order flow. |
| Staleness frente a resultado | tabla | No premiar datos congelados. |

### 5. Lead-lag de referencias externas

| Analisis | Ejemplos | Objetivo |
|---|---|---|
| Retorno spot pasado frente a PM futuro | lags/horizontes | Detectar si PM reacciona tarde. |
| Perp/spot agreement | tabla por signos | Ver confirmacion externa. |
| Gaps normalizados por volatilidad | deciles vs target | Detectar desalineacion significativa. |
| Chainlink freshness/gap | condicionado por edad | Separar ancla fresca de vieja. |

Regla: una feature externa solo se alinea usando su valor disponible en `t` o
antes; nunca una actualizacion posterior.

### 6. Complementariedad e intertemporalidad

| Analisis | Ejemplos | Objetivo |
|---|---|---|
| `mid_A + mid_B - 1` | histograma/deciles | Consistencia de par. |
| Presion de ambos tokens | scatter/tabla | Elegir lado de accion. |
| Residuales de horizontes relacionados | distribucion/target por decil | Ver arbitraje de curva. |
| Metricas token vs market-frame | comparacion | Evitar doble conteo. |

### 7. Contexto nativo y auditoria `as-of`

Antes de correlacionar holders/OI/actividad con futuro, hay que contestar:

- la cifra estaba disponible en `t`?
- `latest_activity_utc <= time_index_utc`?
- el resumen es de toda la sesion o snapshot inicial?
- cuantas filas tendrian contexto causal valido?

| Resultado de auditoria | Decision |
|---|---|
| Timestamp compatible y buena cobertura | Probar feature en capa 2. |
| Snapshot posterior o ambiguo | Solo analisis descriptivo; no input. |
| Cobertura baja | Mask/experimento separado. |

### 8. Splits y estabilidad

| Analisis | Objetivo |
|---|---|
| Balance de targets por fold | Ver drift sin ocultarlo. |
| Distribuciones de features por fold | Detectar cambio de regimen/captura. |
| Conteo de `condition_id` cruzados | Aplicar purga. |
| Comparacion walk-forward vs random agrupado | Separar senal estable de interpolacion. |
| Resultados por temporality/dia | No depender del promedio. |

## Plan de notebooks siguientes

Los numeros son orientativos; cada notebook debe terminar con hallazgos,
decisiones y dudas restantes.

| Notebook propuesto | Contenido | Salida para decidir |
|---|---|---|
| `05_core_target_stability.ipynb` | Ejecutado. Targets alternativos para todo core; distribuciones por dia, temporality, fase y coste. | `H=16 s`, banda `1 tick`, candidato principal; `H=8 s` benchmark. |
| `06_pm_feature_coverage_and_distributions.ipynb` | Ejecutado. Cobertura, distribuciones, calendario y primera senal univariante de microestructura PM. | `microprice_gap_ticks`, deltas PM, spread/coste y reloj de ventana pasan a shortlist v1. |
| `07_cross_venue_feature_audit.ipynb` | Ejecutado. Extraccion estrecha de `cross_venue_features`: spot/perp/Chainlink, gaps, readiness e intertemporal. | Retornos spot/perp orientados pasan a shortlist; Coinbase/Hyperliquid fuera. |
| `08_pm_external_signal_stability_quick.ipynb` | Ejecutado en muestra corta. Combina PM microprice con retornos spot/perp orientados. | Cuando PM y perp apuntan juntos, `up/down` supera ~`50%`; merece ampliar por bloques. |
| `08b_pm_external_signal_stability_folds_quick.ipynb` | Ejecutado. Muestra equilibrada por split/temporalidad y coste visible. | PM+perp alineado se mantiene; coste visible bajo mejora el hit direccional. |
| `09_action_unit_and_evaluation_quick.ipynb` | Ejecutado. Pares complementarios y unidad de evaluacion en muestra corta. | Entrenar por token es aceptable; evaluar por market-frame es obligatorio. |
| `BASELINE_CONTRACT_V0.md` | Ejecutado. Contrato del primer baseline: target, features, filtros, splits y metricas. | Baseline v0 definido y probado en modo quick. |
| `11_first_offline_baseline_v0.ipynb` | Ejecutado. Primer baseline offline en muestra corta. | Senal inicial en test terminal; siguiente paso: robustez/ablations cortos. |
| `12_native_context_asof_audit_optional.ipynb` | Timestamps y cobertura de holders/OI/actividad. | Ampliacion posterior si el core ya muestra valor. |

La ruta corta al primer baseline es: EDA 09, contrato 10 y baseline 11. La
auditoria nativa `as-of` queda como ampliacion posterior si el core ya muestra
valor.

## EDA 05: completada

`05_core_target_stability.ipynb` ya se ejecuto sobre todo `core`.

Resultado de decision:

- candidato principal: `delta_ticks_16s`, banda `1 tick`;
- benchmark: `delta_ticks_8s`, banda `1 tick`;
- metrica economica secundaria: markout taker roundtrip a `8 s` y `16 s`.

## EDA 06: completada

Informe asociado:
[EDA_06_PM_FEATURE_COVERAGE.md](EDA_06_PM_FEATURE_COVERAGE.md)

`06_pm_feature_coverage_and_distributions.ipynb` ya se ejecuto sobre todo
`core`.

Resultado de decision:

- el bloque PM basico tiene cobertura suficiente para v1 exploratoria;
- `microprice_gap_ticks` es la primera senal PM seria;
- `mid_delta_ticks_2s/8s`, `spread_ticks`, `visible_entry_cost_ticks`,
  `seconds_to_window_end` y `window_progress` pasan a shortlist;
- `trade_imbalance` crudo queda como candidato debil por saturacion;
- calendario queda como EDA/ablation, no feature fuerte automatica.

## EDA 07: completada

Informe asociado:
[EDA_07_CROSS_VENUE_AUDIT.md](EDA_07_CROSS_VENUE_AUDIT.md)

`07_cross_venue_feature_audit.ipynb` ya se ejecuto sobre todo `core`.

Resultado de decision:

- `external_mid_return_bps_2s/8s_oriented` y
  `perp_mid_return_bps_2s/8s_oriented` pasan a shortlist;
- la orientacion por outcome es obligatoria;
- Chainlink, gaps spot/perp, consenso y freshness son contexto/masks, no
  senales direccionales fuertes por si solos;
- Coinbase, Hyperliquid y sidecars siguen fuera de v1;
- los escaneos anchos de `cross_venue_features` son caros y deben evitarse.

## EDA 08 quick: completada

Informe asociado:
[EDA_08_PM_EXTERNAL_SIGNAL_STABILITY_QUICK.md](EDA_08_PM_EXTERNAL_SIGNAL_STABILITY_QUICK.md)

`08_pm_external_signal_stability_quick.ipynb` ya se ejecuto en una muestra
corta de `93.496` filas con soporte H16 exacto.

Resultado de decision:

- PM microprice + perp `2s` orientado es una combinacion prometedora;
- ambos positivos: `50,27%` up, `17,98%` down;
- ambos negativos: `51,52%` down, `18,82%` up;
- conflicto o neutralidad vuelve a una distribucion mucho mas plana;
- el ritmo de trabajo pasa a iteraciones cortas antes de escaneos completos.

## EDA 08b: siguiente paso inmediato recomendado

Informe asociado:
[EDA_08B_PM_EXTERNAL_BALANCED_COST_QUICK.md](EDA_08B_PM_EXTERNAL_BALANCED_COST_QUICK.md)

`08b_pm_external_signal_stability_folds_quick.ipynb` ya se ejecuto con muestra
balanceada por `terminal_split + temporality` y coste visible.

Resultado de decision:

- `PM + perp` alineado mantiene mas de `51%` de clase correcta;
- `PM + spot` confirma el mismo patron;
- coste visible bajo mejora el hit direccional a `60,00%` en la muestra
  alineada;
- el siguiente paso corto debe resolver unidad de accion/evaluacion.

## EDA 09: siguiente paso inmediato recomendado

Informe asociado:
[EDA_09_ACTION_UNIT_EVALUATION_QUICK.md](EDA_09_ACTION_UNIT_EVALUATION_QUICK.md)

`09_action_unit_and_evaluation_quick.ipynb` ya se ejecuto en muestra corta.

Resultado de decision:

- los pares complementarios estan completos al `100%` en la muestra;
- la mediana de `abs(mid_up + mid_down - 1)` es `0`;
- entrenar por token es aceptable para v0;
- evaluar y accionar por `session_id + condition_id + time_index_ns` es
  obligatorio;
- no se deben contar `both_pos` y `both_neg` como oportunidades separadas.

## Baseline contract v0: completado

Documento asociado:
[BASELINE_CONTRACT_V0.md](BASELINE_CONTRACT_V0.md)

El contrato ya cierra:

1. target principal y benchmark;
2. features v0 y transforms permitidos;
3. filtros y masks;
4. unidad de entrenamiento y unidad de evaluacion;
5. split walk-forward/test terminal;
6. metricas predictivas y action metrics;
7. que queda fuera del primer modelo.

## Baseline offline v0: completado en modo quick

Informe asociado:
[BASELINE_V0_QUICK_RESULTS.md](BASELINE_V0_QUICK_RESULTS.md)

Notebook asociado:
[`../notebooks/11_first_offline_baseline_v0.ipynb`](../notebooks/11_first_offline_baseline_v0.ipynb)

Resultado de decision:

- el contrato v0 ejecuta de punta a punta sin full scan;
- `hist_gbdt_small` es el mejor candidato practico de esta corrida corta;
- en test terminal obtiene `63,46%` hit up, `18,59%` wrong down, `1,88`
  ticks de delta medio y `3,29%` coverage;
- la regla PM + perp `both_pos` cae en test a `46,54%` hit up y `1,41`
  ticks de delta medio;
- hay senal suficiente para una segunda prueba corta con ablations;
- no hay todavia evidencia suficiente para bot, pipeline productivo ni full
  core sin aprobacion.

## Baseline v0: siguiente paso inmediato recomendado

Hacer una iteracion corta de robustez:

1. repetir con 2 o 3 muestras estrechas distintas;
2. ablation por bloques: PM solo, PM + perp, PM + spot, PM + reloj;
3. seleccion de threshold con minimo de coverage y penalizacion por
   `wrong_down`/coste;
4. desglose por `5m/15m/1h` y fase de ventana con recuentos minimos.

## Baseline v0 full-core robustness: completado

Informe asociado:
[BASELINE_V0_FULL_CORE_ROBUSTNESS.md](BASELINE_V0_FULL_CORE_ROBUSTNESS.md)

Notebooks:

- [`../notebooks/12_baseline_v0_full_core_robustness.ipynb`](../notebooks/12_baseline_v0_full_core_robustness.ipynb)
- [`../notebooks/13_baseline_v0_feature_sanity_addendum.ipynb`](../notebooks/13_baseline_v0_feature_sanity_addendum.ipynb)

Resultado de decision:

- la senal core sobrevive a full-core experimental, folds temporales y test
  terminal;
- `full_v0` queda como baseline general v0;
- `full_no_microprice` queda como filtro high-conviction a calibrar;
- PM solo y `time_only` quedan descartados como explicaciones suficientes;
- PM + perp es el bloque incremental mas importante;
- quality masks/intertemporal no mejoran materialmente v0;
- siguiente paso: v0.1 offline con threshold walk-forward agregado,
  intervalos de confianza, evaluacion no solapada y backtest simple de costes.

## Baseline v0.1 threshold/bootstrap: completado

Informe asociado:
[BASELINE_V01_THRESHOLD_BOOTSTRAP.md](BASELINE_V01_THRESHOLD_BOOTSTRAP.md)

Notebook:
[`../notebooks/14_baseline_v01_threshold_bootstrap.ipynb`](../notebooks/14_baseline_v01_threshold_bootstrap.ipynb)

Resultado de decision:

- threshold `0,55` elegido por folds agregados;
- `full_v0` queda como baseline general calibrable;
- `full_no_microprice` queda como overlay high-conviction;
- el edge aguanta bootstrap y no-solapado H16;
- siguiente paso: v0.2 offline con backtest simple, cooldown por
  `condition_id`, bins finos de score por temporalidad y fill conservador.

## Baseline v0.2 conservative backtest: completado

Informe asociado:
[BASELINE_V02_CONSERVATIVE_BACKTEST.md](BASELINE_V02_CONSERVATIVE_BACKTEST.md)

Notebook:
[`../notebooks/15_baseline_v02_conservative_backtest.ipynb`](../notebooks/15_baseline_v02_conservative_backtest.ipynb)

Resultado de decision:

- se evaluaron acciones v0.1 sobre el test terminal con coste visible,
  buffers simples y cooldowns por `condition_id`;
- `full_no_micro_highconv` queda como politica conservadora principal;
- el escenario strict (`score >= 0,60`, `spread <= 1 tick`, buffer `0,5 tick`)
  obtiene `203` acciones, `88,67%` hit up, `3,94%` wrong down y `5,95` ticks
  netos medios;
- el escenario `cross_visible` sin cooldown obtiene mas acciones (`605`) y
  mayor neto total proxy, pero es menos conservador;
- los resultados diarios del escenario strict son positivos en los tres dias
  del test terminal;
- siguiente paso: consolidar dataset/modelado v0.3 offline, no bot.

## Baseline v0.3 dataset contract: completado

Informe asociado:
[BASELINE_V03_DATASET_CONTRACT.md](BASELINE_V03_DATASET_CONTRACT.md)

Notebook:
[`../notebooks/16_baseline_v03_dataset_contract.ipynb`](../notebooks/16_baseline_v03_dataset_contract.ipynb)

Resultado de decision:

- el dataset offline queda congelado sobre el cache H16 full-core:
  `2.345.284` filas y `1.172.642` market-frames;
- se mantienen target H16, split temporal y evaluacion por market-frame;
- se definen tres feature sets modelables:
  `v03_general_full_v0`, `v03_conservative_no_micro` y
  `v03_pm_perp_control`;
- `quality` queda como diagnostico y `intertemporal` como holdout;
- se audita drift por `temporality`: `15m` se mueve mas, `5m/1h` tienen mas
  `flat`;
- siguiente paso: modelado offline v0.3 reproducible con reporte segmentado.

## Baseline v0.3 model runner: completado

Informe asociado:
[BASELINE_V03_MODEL_RUNNER.md](BASELINE_V03_MODEL_RUNNER.md)

Notebook:
[`../notebooks/17_baseline_v03_model_runner.ipynb`](../notebooks/17_baseline_v03_model_runner.ipynb)

Resultado de decision:

- se entreno `hgb_balanced_small` con los tres feature sets v0.3;
- el threshold se eligio en validacion, no mirando test;
- `v03_conservative_no_micro` eligio threshold `0,65` y en test obtiene `68`
  acciones, `91,18%` hit up, `2,94%` wrong down y `7,10` ticks netos medios;
- su escenario strict da `203` acciones, `88,67%` hit up y `5,95` ticks netos
  medios;
- `v03_pm_perp_control` sigue siendo fuerte, confirmando el papel de PM+perp;
- siguiente paso: analisis de errores/calibracion v0.3b antes de ejecucion real.

## Baseline v0.3b error/calibration: completado

Informe asociado:
[BASELINE_V03B_ERROR_CALIBRATION.md](BASELINE_V03B_ERROR_CALIBRATION.md)

Notebook:
[`../notebooks/18_baseline_v03b_error_calibration.ipynb`](../notebooks/18_baseline_v03b_error_calibration.ipynb)

Resultado de decision:

- el score accionable `P(up)-P(down)` ordena bien el riesgo;
- `P(up)` aislado no basta como score operativo;
- en test, `score 0,65-0,70` tiene `90,77%` hit up, `3,08%` wrong down y
  `6,95` ticks netos medios;
- el escenario selected-visible tiene pocos errores absolutos: `4` flats y
  `2` wrong-down en test;
- `not_early` y `low_cost` son hipotesis a validar, no reglas aprobadas;
- siguiente paso: validar politica `score >= 0,65` en folds/segmentos.

## Baseline v0.3c policy fold validation: completado

Informe asociado:
[BASELINE_V03C_POLICY_FOLD_VALIDATION.md](BASELINE_V03C_POLICY_FOLD_VALIDATION.md)

Notebook:
[`../notebooks/19_baseline_v03c_policy_fold_validation.ipynb`](../notebooks/19_baseline_v03c_policy_fold_validation.ipynb)

Resultado de decision:

- `score_buy >= 0,65` aguanta los 4 folds temporales;
- politica visible: `578` acciones, `90,14%` hit up medio, `5,61%` wrong
  down medio y `4/4` folds positivos;
- politica con `spread <= 1 tick` + buffer `0,5`: `476` acciones, `90,92%`
  hit up medio, `4,88%` wrong down medio y `4/4` folds positivos;
- `score >= 0,60` aumenta cobertura, pero sube demasiado el wrong-down;
- `not_early` no queda aprobado como regla;
- siguiente paso: simulacion offline de ejecucion conservadora antes de bot.

## Baseline v0.3d execution stress: completado

Informe asociado:
[BASELINE_V03D_EXECUTION_STRESS.md](BASELINE_V03D_EXECUTION_STRESS.md)

Notebook:
[`../notebooks/20_baseline_v03d_execution_stress.ipynb`](../notebooks/20_baseline_v03d_execution_stress.ipynb)

Resultado de decision:

- se probaron latencias de `0s`, `2s`, `4s` y `8s` sobre `646` senales
  `score_buy >= 0,65`;
- el escenario conservador en `0s` sigue siendo fuerte: `376` acciones,
  `91,76%` hit up y `9,34` ticks netos medios;
- con `2s` el agregado aun queda positivo (`2,96` ticks), pero el TEST
  terminal queda negativo (`-1,48` ticks netos medios);
- con `4s` y `8s` la politica deja de ser defendible;
- decision: hay senal de movimiento, pero es muy sensible a timing;
- siguiente paso: medir latencia real end-to-end y despues construir
  target/modelo consciente de latencia, no bot.

Estrategia asociada:
[ESTRATEGIA_LATENCIA_Y_SIGUIENTE_FASE.md](ESTRATEGIA_LATENCIA_Y_SIGUIENTE_FASE.md)

## Notebook unificado de presentacion

Notebook asociado:
[`../notebooks/00_edgerunner_unified_eda_baselines.ipynb`](../notebooks/00_edgerunner_unified_eda_baselines.ipynb)

Guia:
[NOTEBOOK_UNIFICADO_EDA_BASELINES.md](NOTEBOOK_UNIFICADO_EDA_BASELINES.md)

Uso:

- no recalcula ejecuciones largas;
- carga resultados ya generados;
- explica dataset, target, splits, features, correlaciones, deciles,
  microestructura/orderbook, baselines y decision actual en una sola narrativa;
- es el notebook recomendado para entender/defender el proyecto.

## Como estudiaremos correlaciones

### Targets continuos antes que clases

Para estudiar si una feature se relaciona con el futuro, es mejor empezar con
`delta_ticks_H` continuo que con una clase. Las clases pierden informacion:
un movimiento de `1.1` ticks y otro de `8` ticks son ambos `up`.

### Correlaciones recomendadas

| Metodo | Para que |
|---|---|
| Pearson | Relacion aproximadamente lineal; sensible a extremos. |
| Spearman | Relacion monotona; mas robusta a distribuciones raras. |
| Deciles/quantiles de feature contra media/mediana del target | Interpretacion sencilla y no lineal. |
| Correlacion por fold/dia/temporality | Estabilidad; no premiar coincidencia global. |
| Correlacion con markout neto | Ver si la relacion es economica, no solo direccional. |

No escogeremos features solo porque su `p-value` sea pequeno: con millones de
frames autocorrelacionados casi cualquier diferencia pequena puede parecer
estadisticamente significativa.

### Reduccion de dependencia para interpretar

Para tablas y correlaciones podemos complementar el analisis por fila con:

- muestreo no solapado segun horizonte;
- agregacion por `condition_id + time_index_ns`;
- resultados por sesion/dia;
- intervalos de variabilidad entre folds.

## Criterios de salida antes de modelar

No pasaremos a baseline hasta poder responder por escrito:

| Pregunta | Respuesta requerida |
|---|---|
| Que target primario usamos? | Formula, horizonte y banda o target continuo. |
| Que target economico secundario usamos? | Supuestos de spread, fee y salida. |
| Que filas entran? | Filtros core, futuro completo, missing/stale y tratamiento tradability. |
| Que features entran? | Lista causal corta por bloques, con masks. |
| Que contexto queda fuera? | Campos sin auditoria `as-of` o sidecars ausentes. |
| Como validamos? | Walk-forward purgado, embargo y test terminal. |
| Que medira exito? | Metricas predictivas y markout neto por segmentos. |

## Relacion con las otras guias

- Inventario y synthetics:
  [GUIA_01_FEATURES_Y_VARIABLES_SINTETICAS.md](GUIA_01_FEATURES_Y_VARIABLES_SINTETICAS.md).
- Splits:
  [GUIA_02_SPLITS_Y_VALIDACION.md](GUIA_02_SPLITS_Y_VALIDACION.md).
- Labels, horizontes, `flat` y coste:
  [GUIA_04_LABELS_TARGETS_Y_COSTES.md](GUIA_04_LABELS_TARGETS_Y_COSTES.md).
