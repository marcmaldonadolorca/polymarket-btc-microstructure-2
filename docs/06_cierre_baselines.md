# Informe de cierre de baselines - EdgeRunner

Fecha: `2026-06-01`

> **Nota posterior, 2026-06-04.** Este informe conserva el cierre historico de
> los baselines taker. Un candidato maker positivo posterior fue invalidado por
> leakage de target futuro. Las repeticiones maker causales subset y full son
> negativas; sigue vigente el `NO_GO` para bot.

## Resumen ejecutivo

Este informe cierra la primera fase seria de baselines del proyecto EdgeRunner.

La conclusion corta es:

```text
Hay senal predictiva en el dataset.
La senal funciona muy bien si evaluamos sin latencia.
La senal se deteriora cuando simulamos entrada realista con L=2s.
Los baselines tabulares L2 simples no producen una politica economica robusta.
No hay autorizacion tecnica para bot.
```

La decision final de esta fase es:

```text
Cerrar baselines tabulares taker L2 simples.
Pasar a diagnostico de timing, microestructura y tipo de ejecucion.
```

Esto no significa que el proyecto falle. Significa que hemos evitado una trampa
habitual: quedarse con un buen accuracy o un buen backtest sin latencia y creer
que eso ya es edge ejecutable.

## Contexto del proyecto

Objetivo:

```text
construir un predictor de edge de muy corto plazo para mercados BTC de
Polymarket.
```

Fuente de verdad:

```text
D:\polymarket_btc_probe_official_v1\polymarket_week.sqlite3
```

Corpus:

- `2.604` sesiones importadas;
- `2.590` sesiones `core`;
- `2.499.840` filas en `cross_venue_features`;
- `2.433.124` filas en `predictor_labels`;
- coverage y continuity medias `1.0`;
- gaps severos core `0`.

Regla de trabajo:

```text
primero EDA, luego baselines offline, despues decidir si merece la pena
pipeline/bot/modelos complejos.
```

## Reglas metodologicas usadas

Durante toda la fase se han mantenido estas reglas:

- split temporal, no split aleatorio como baseline principal;
- test terminal bloqueado;
- purga de condiciones compartidas contra test;
- entrenamiento por token;
- evaluacion por market-frame:

```text
session_id + condition_id + time_index_ns
```

- features solo disponibles en tiempo `t`;
- labels/resultados futuros solo como targets o evaluacion;
- no usar `entry_*`, `future_*`, `delta_*`, `net_*`, `target_*` como features;
- coste visible y buffer incluidos en evaluacion economica;
- no aprobar reglas con pocas acciones aunque salgan positivas;
- version no solapada H16 para controlar autocorrelacion temporal.

## Dataset de modelado L-aware

Tras medir latencia real, el dataset L-aware principal queda:

```text
data/experiments/baseline_laware_v01_dataset_contract/baseline_laware_v01_core_L2.parquet
```

Filas:

```text
2.328.910
```

Split:

| Bloque | Filas |
|---|---:|
| Train inicial | `1.583.620` |
| Validacion inicial | `311.644` |
| Pretest tras purga | `1.894.928` |
| Test terminal | `433.646` |
| Filas purgadas vs test | `336` |

La latencia real medida fue alrededor de `0,43s` hasta ack/match CLOB, pero el
dataset esta en malla de `2s`. Por eso:

```text
L_operativo_real = 1s conceptual
L_dataset_principal = 2s por resolucion de captura
```

No interpolamos `t+1s` porque habria introducido supuestos nuevos.

## Fase 1 - Baselines sin latencia

Los primeros baselines evaluaban movimiento desde `t`, sin entrada retrasada.
Sirvieron para comprobar si habia senal en el corpus.

Resultado: si, habia senal fuerte.

Hitos principales:

| Fase | Resultado |
|---|---|
| v0 quick | HGB pequeno en test: `63,46%` hit up, `18,59%` wrong down, `1,88` ticks de delta medio. |
| full-core robustness | `full_v0`: `3.802` acciones, `66,91%` hit up, `1,60` ticks netos visibles medios. |
| v0.1 bootstrap | `full_no_microprice`: `605` acciones, `83,31%` hit up, `5,15` ticks netos medios. |
| v0.2 conservative backtest | escenario strict: `203` acciones, `88,67%` hit up, `3,94%` wrong down, `5,95` ticks netos medios. |
| v0.3c folds | `4/4` folds positivos; `578` acciones agregadas, `90,14%` hit up, `9,80` ticks netos medios. |

Lectura:

```text
El dataset contiene senal predictiva real si miramos el movimiento inmediato.
```

Pero aqui aparecio el primer aviso: una senal puede existir y aun asi no ser
ejecutable si desaparece antes de que llegue nuestra orden.

## Fase 2 - Stress de ejecucion

El stress v0.3d simulo retrasos de entrada.

Resultado:

- con `0s`, la politica era fuerte;
- con `2s`, el agregado todavia podia parecer aceptable;
- pero el test terminal paso a negativo;
- con mas latencia, la politica dejaba de ser defendible.

Decision:

```text
antes de bot, medir latencia real y construir labels conscientes de latencia.
```

## Fase 3 - Latencia real

Se midio el flujo operativo de Polymarket CLOB.

Resultado v1.1:

| Medida | Valor |
|---|---:|
| Live posts sigtype2 | `3` |
| Live fills sigtype2 | `1` |
| p95 total seen -> ack | `429,674 ms` |
| p95 sent -> ack | `144,137 ms` |
| fill matched seen -> fill | `426,737 ms` |

Decision:

```text
L_operativo_modelado = 1s conceptual
L_dataset_principal = 2s por grid del dataset
```

El bot siguio en `NO GO`, porque medir latencia no demuestra edge.

## Fase 4 - Labels L-aware

Se audito si podiamos construir labels exactos con `L=1s`.

Resultado:

| L | Soporte |
|---:|---:|
| `1s` | `0,00%` |
| `2s` | `99,30%` |

Por eso se usa `L=2s` como evaluacion conservadora:

```text
features en t
entrada evaluada en t+2s
salida H16 despues de la entrada
coste visible + buffer
```

## Fase 5 - Baselines L-aware direccionales

### v0.1 - Primer L2 direccional

Modelo:

```text
hgb_balanced_small
```

Feature set:

```text
v03_conservative_no_micro
```

Target:

```text
target_3c_L2_H16_1tick
```

Resultado:

| Escenario | Acciones | Hit up | Wrong down | Neto medio con buffer |
|---|---:|---:|---:|---:|
| selected L2 buffer | `68` | `61,76%` | `14,71%` | `-0,68` |

Decision:

```text
NO GO economico.
```

El modelo acierta direccion algo por encima de azar, pero no cubre costes.

### v0.1b - Comparativa de features

Se probaron:

- `v03_conservative_no_micro`;
- `v03_conservative_plus_micro`;
- `v03_pm_perp_control`.

Microprice ayudo a predecir:

| Feature set | Balanced accuracy test |
|---|---:|
| sin microprice | `56,55%` |
| plus microprice | `57,66%` |
| PM + perp | `57,63%` |

Pero la economia siguio negativa:

| Feature set | Acciones | Hit up | Wrong down | Neto medio |
|---|---:|---:|---:|---:|
| plus microprice, high score/low spread | `1.278` | `43,19%` | `8,29%` | `-0,63` |
| PM+perp, high score/low spread | `1.188` | `44,28%` | `7,74%` | `-0,64` |

Decision:

```text
Microprice aporta senal, pero no convierte la politica en rentable bajo L2.
```

## Fase 6 - Diagnostico trade/no-trade

### v0.2 - Buckets y segmentos

Se analizaron los outputs del modelo direccional por score, temporalidad y
reglas simples.

Conclusion:

```text
No habia regla robusta positiva con suficientes acciones.
```

Aparecieron pistas pequenas en `1h`, pero no eran suficientes.

### v0.2b - Target economico trade/no-trade

Se cambio la pregunta:

Antes:

```text
sube o baja?
```

Despues:

```text
merece la pena tradear despues de coste y latencia?
```

Target:

```text
target_trade_L2_H16_buffer_0p5 = net_buffer_0p5_ticks_L2_H16 > 0
```

Resultado predictivo:

| Stage | AUC | Balanced accuracy |
|---|---:|---:|
| Test | `0,7913` | `0,7298` |

Esto es importante: el modelo si aprende a ordenar oportunidades.

Pero al convertirlo en acciones:

| Escenario | Acciones | Trade hit | Wrong down | Neto medio |
|---|---:|---:|---:|---:|
| selected L2 buffer | `338` | `46,75%` | `39,35%` | `-2,134` |
| selected low spread | `329` | `47,11%` | `38,91%` | `-1,991` |

Decision:

```text
Hay ranking estadistico, pero no politica economica robusta.
```

### v0.2c - Estabilidad temporal

La pista `1h` parecia positiva en test:

```text
temporality = 1h
P(trade) >= 0,70
spread <= 1
```

Pero al comprobar validacion:

| Split | Acciones | Neto medio | Neto medio no solapado |
|---|---:|---:|---:|
| Validacion | `920` | `-0,582` | `-0,537` |
| Test | `1.099` | `+0,296` | `+0,064` |

Decision:

```text
NO GO para regla 1h.
```

La regla era una pista de test, no una politica validada.

### v0.2d - Targets estrictos

Ultima prueba de cierre:

```text
trade solo si net_buffer supera +1 o +2 ticks.
```

Resultados principales:

| Barrera | Escenario | Acciones | Strict hit | Trade positivo >0 | Wrong down | Neto medio |
|---:|---|---:|---:|---:|---:|---:|
| `> +1` | selected low spread | `4.343` | `37,51%` | `42,53%` | `47,23%` | `-1,695` |
| `> +2` | selected low spread | `8.256` | `33,30%` | `42,54%` | `46,56%` | `-1,406` |

No solapado:

| Barrera | Acciones | Neto medio |
|---:|---:|---:|
| `> +1` | `1.752` | `-1,648` |
| `> +2` | `2.407` | `-1,170` |

Decision:

```text
NO GO para targets estrictos.
```

## Decision final de baselines

La decision final es:

```text
Cerramos la familia de baselines tabulares taker L2 simples.
```

Motivos:

1. Sin latencia, la senal es fuerte.
2. Con entrada L2, la senal se degrada.
3. Microprice mejora prediccion, pero no neto economico.
4. El target trade/no-trade aprende ranking, pero no produce acciones
   rentables.
5. La pista `1h` no aguanta validacion.
6. Targets estrictos `> +1` y `> +2` tampoco pasan.
7. El test terminal y la version no solapada son negativos.

Por tanto:

```text
NO GO para bot.
NO GO para seguir probando pequenas variantes del mismo baseline.
GO para diagnostico de timing, microestructura y ejecucion.
```

## Explicacion sencilla

Una forma simple de entenderlo:

```text
El modelo ve algo.
Pero lo que ve ocurre demasiado pronto, es demasiado pequeno, o cuesta
demasiado capturarlo como taker despues de esperar hasta la siguiente foto L2.
```

El problema no es solo "accuracy". De hecho, algunos modelos tienen AUC bueno.
El problema es:

```text
convertir score en operacion rentable.
```

## Que no debemos hacer ahora

No conviene:

- pasar directamente a bot;
- seguir probando thresholds mirando test;
- meter redes profundas para maquillar el problema;
- optimizar infraestructura antes de tener politica positiva;
- reabrir el recolector salvo bug critico;
- meter sidecars opcionales como features fuertes sin contrato de presencia.

## Siguiente fase recomendada

La siguiente fase no deberia llamarse "otro baseline". Deberia llamarse:

```text
diagnostico de perdida de edge
```

Preguntas:

1. Donde muere la senal entre `t` y `t+2s`?
2. Que parte del neto negativo viene de coste/spread y que parte de direccion?
3. El problema es la prediccion o la ejecucion taker?
4. Hay patrones de orderbook que indiquen cuando no perseguir precio?
5. Tiene sentido una politica maker/quote en vez de taker buy?
6. Necesitamos datos live mas finos que la malla de `2s`?
7. Un encoder de orderbook resolveria algo o solo sobreajustaria?

Artefacto recomendado para la siguiente etapa:

```text
notebook de diagnostico timing/orderbook:
notebooks/02_laware_timing_orderbook_diagnostics.ipynb
```

Objetivo de ese notebook:

- comparar distribucion de `net_buffer` entre trades buenos/malos;
- estudiar `t -> t+2s` como periodo de perdida;
- ver spread, microprice, mid drift y fase de ventana;
- separar `5m`, `15m`, `1h`;
- decidir si el siguiente modelo necesita orderbook encoder, target distinto o
  cambio de politica de ejecucion.

## Lista de informes relevantes

Baselines y cierre:

- `docs/BASELINE_LAWARE_V01_MODEL_RUNNER.md`
- `docs/BASELINE_LAWARE_V01B_FEATURE_RUNNER.md`
- `docs/BASELINE_LAWARE_V02_TRADE_DIAGNOSTICS.md`
- `docs/BASELINE_LAWARE_V02B_TRADE_TARGET_RUNNER.md`
- `docs/BASELINE_LAWARE_V02C_TEMPORAL_STABILITY_PROBE.md`
- `docs/BASELINE_LAWARE_V02D_STRICT_TARGET_CLOSURE.md`

Latencia:

- `docs/LATENCY_V11_OPERATIVE_CLOSEOUT.md`
- `docs/LATENCY_AWARE_LABEL_AUDIT_V02_L1.md`

Estrategia:

- `docs/ESTRATEGIA_LATENCIA_Y_SIGUIENTE_FASE.md`
- `docs/PROJECT_BRIEF.md`

## Cierre

Resultado final:

```text
Baselines cerrados.
Edge estadistico detectado.
Edge economico L2 tabular taker no aprobado.
Siguiente paso: diagnostico profundo de timing/orderbook antes de modelar mas.
```

## Addendum 2026-06-01 - Decision tras diagnostico de ejecucion

Tras este informe se ejecutaron tres diagnosticos adicionales:

- `docs/LAWARE_TIMING_ORDERBOOK_DIAGNOSTICS_V01.md`
- `docs/LAWARE_MARKOUT_EXECUTION_DIAGNOSTICS_V01.md`
- `docs/LAWARE_EXECUTION_MODE_DIAGNOSTICS_V01.md`
- `docs/LAWARE_MAKER_PROXY_FILL_RISK_DIAGNOSTICS_V01.md`

La conclusion refinada es:

```text
GO a modelos complejos, pero solo si son modelos de ejecucion/microestructura.
NO GO a otro clasificador direccional generico.
NO GO a bot.
```

La razon:

- el edge inmediato existe, especialmente en celdas con microprice positivo;
- casi todas las celdas taker estables son `delay=0`;
- con `delay>=2s`, taker completo casi desaparece;
- si reducimos coste efectivo, reaparecen celdas delayed;
- pero bajo fill adverso no sobrevive ningun candidato.

Por tanto, el nuevo problema no es:

```text
predecir si sube o baja.
```

El nuevo problema es:

```text
predecir si una ejecucion maker/cheap-fill tendra esperanza positiva sin caer
en seleccion adversa.
```

Informe de decision:

```text
docs/INFORME_DECISION_PASO_MODELOS_COMPLEJOS.md
```

## Addendum 2026-06-02 - Estado tras fusion, risk y fills

Despues del cierre de baselines tabulares se avanzo a modelos secuenciales,
fusion con orderbook y gestion de riesgo proxy.

Mejor resultado proxy actual:

```text
fusion_concat_gru + risk manager cap100/dia
```

En test con fill completo:

- `300` acciones;
- sum cost0.50 `+179,6775`;
- sum cost1.00 `+67,3550`;
- `0` dias negativos en cost0.50 y cost1.00.

Esto confirma que la senal no ha desaparecido: hay ranking economico util bajo
el proxy offline.

Pero al introducir ejecucion imperfecta:

- fill aleatorio 50% solo pasa cost0.50 en `20%` de runs;
- fill adverso 50% rompe fuerte (`-1117,0699` cost0.50);
- el primer fill-aware risk manager v1, usando un proxy causal simple de fill,
  tampoco cierra test (`NO_GO_FILL_AWARE_RISK_MANAGER_V1`).

Decision refinada:

```text
GO a investigar modelos complejos, pero no como clasificador UP/DOWN generico.
GO a modelo de fill/ejecucion y microestructura.
NO GO a bot.
```

La narrativa correcta para el trabajo queda asi:

1. Primero encontramos senal direccional/economica.
2. Luego demostramos que coste y latencia cambian mucho el problema.
3. Despues vimos que fusion tabular + orderbook ayuda, pero no basta.
4. El risk manager proxy produce resultados fuertes con fill completo.
5. La ejecucion parcial/adversa es el cuello de botella.
6. Por tanto, el siguiente modelo debe predecir calidad de ejecucion/fill, no
   solo si el precio sube o baja.

Siguiente fase:

```text
fill model v2: orderbook, profundidad visible, spread, coste de entrada,
salud del libro, logs live si existen, y evaluacion con fills parciales.
```

Readiness audit ya ejecutado:

```text
docs/FILL_MODEL_V2_READINESS_AUDIT.md
```

Resultado:

- merge con dataset de ejecucion: `100%`;
- soporte H60: `100%`;
- soporte H120: `71,28%`;
- decision: `GO_FILL_MODEL_V2_DATA_READY_NO_MODEL_YET`.

Por tanto, el siguiente paso ya no es debatir si hay datos, sino entrenar un
primer `fill_model_v2_tabular_probe` interpretable.

Probe ejecutado:

```text
docs/FILL_MODEL_V2_TABULAR_PROBE.md
```

Resultado:

```text
NO_GO_FILL_MODEL_V2_TABULAR_COST05
```

El modelo tabular estrecho sobre las senales ya filtradas sobreajusta:

- train healthy AUC `0,9665`;
- validation healthy AUC `0,4589`;
- test healthy AUC `0,4894`;
- seleccion final vuelve a `hybrid_score`, no a un score v2;
- fill 50% random test pass cost0.50 `0,30`.

Conclusion refinada:

```text
no basta con un segundo modelo tabular sobre senales ya seleccionadas.
La siguiente fase debe ampliar universo y/o usar orderbook tensor/logs reales
de fill.
```

Se probo tambien ampliar universo con reglas causales simples:

```text
docs/FILL_MODEL_V2_EXPANDED_UNIVERSE_AUDIT.md
```

Resultado:

```text
NO_GO_EXPANDED_UNIVERSE_AUDIT
```

- `304` candidatos H60/H120;
- `0` pasan train+validation;
- `0` cierran test cost0.50 sin fallos diarios.

Conclusion final de esta rama:

```text
ni meta-modelo tabular estrecho ni reglas tabulares amplias bastan.
La siguiente informacion candidata es orderbook secuencial/profundidad real o
logs reales de fill.
```

Se audito el orderbook:

```text
docs/ORDERBOOK_EXECUTION_FEATURE_AUDIT_V1.md
```

Resultado:

```text
RESEARCH_PASS_ORDERBOOK_FEATURES_SELECTED
```

El libro visible aporta features estables dentro de las senales seleccionadas.
Ejemplo:

```text
book_ask_cum_top1_last
train +3,5610
validation +1,7782
test +2,8165
```

Pero el uso como gate simple fallo:

```text
docs/ORDERBOOK_FEATURE_RISK_GATE_V1.md
NO_GO_ORDERBOOK_RISK_GATE_COST05
```

Conclusion refinada:

```text
el orderbook merece entrar en el modelo, pero no como umbral manual simple.
```

Se probo tambien un modelo regularizado pequeno con orderbook:

```text
docs/ORDERBOOK_REGULARIZED_SEQUENCE_V1.md
```

Decision:

```text
NO_GO_ORDERBOOK_REGULARIZED_SEQUENCE_COST05
```

La validacion eligio un score de orderbook (`obr_book_plus_scores_logit_adverse_c1p0`),
lo cual es una buena noticia: el libro no es ruido. Pero al pasar a test con
fill aleatorio 50%, el resultado no cierra:

- test pass cost0.50 `0,20`;
- test pass cost1.00 `0,10`;
- peor dia medio cost0.50 `-15,2840`.

Hay una pista prometedora, aun no final:

```text
obr_book_only_logit_adverse_c0p3, cap=50
validation pass cost0.50 = 0,80
test pass cost0.50 = 0,60
test worst day cost0.50 = +1,0794
```

Conclusion actualizada:

```text
el cuello de botella ya no es solo crear mas features o meter una red mas grande.
Tambien necesitamos una forma mas estable de elegir configuraciones por periodo.
El siguiente paso limpio es auditar seleccion temporal/nested validation antes
de entrenar CNN/Transformer.
```

Ese audit de seleccion ya se ejecuto:

```text
docs/SELECTION_STABILITY_AUDIT_V1.md
```

Decision:

```text
RESEARCH_PASS_CANDIDATE_EXISTS_SELECTION_NO_GO
```

Resultado:

- `6` criterios simples de seleccion usando solo validation;
- `0` seleccionan una configuracion robusta en test;
- existe `1` candidata robusta validation-positive:
  `obr_book_only_logit_adverse_c0p3`, cap `50`;
- esa candidata tiene test pass cost0.50 `0,60`, test pass cost1.00 `0,50`
  y peor dia test cost0.50 `+1,0794`.

Lectura para la memoria:

```text
la familia orderbook/adverse-fill parece prometedora, pero el problema aun no
esta cerrado porque el selector no la elige de forma causal. La siguiente fase
debe fijar un protocolo de seleccion temporal antes de modelos mas complejos.
```

## Correccion metodologica posterior: direccion adverse

Referencia:

```text
docs/ORDERBOOK_ADVERSE_DIRECTION_CORRECTION.md
docs/SELECTION_STABILITY_NESTED_V3_CORRECTED.md
```

Se detecto que las primeras pruebas regularizadas permitian rankear directamente
`P(adverse)` de mayor a menor. Eso es incorrecto: `adverse=1` significa caer en
el peor 25% de ejecucion.

La correccion aplicada es:

```text
safe_adverse_score = 1 - P(adverse)
```

Resultado corregido:

```text
NO_GO_NESTED_V3_CORRECTED
```

- selector principal: `obr_book_only_logit_healthy_c0p3`, cap `150`;
- test pass cost0.50 `0,30`;
- test pass cost1.00 `0,10`;
- test worst day cost0.50 `-24,3385`;
- `0/32` configuraciones safe-adverse robustas.

Conclusion revisada:

```text
la antigua candidata adverse no es evidencia valida. Aun no existe una policy
regularizada de orderbook robusta y seleccionable. La siguiente fase debe crear
scores upstream out-of-fold o reformular el target antes de aumentar modelo.
```

## Cierre de la pista upstream OOF + stage2

La prueba out-of-fold ya se ejecuto:

```text
docs/UPSTREAM_OOF_FUSION_STAGE2_V1.md
```

Se reconstruyo la fusion sin scores in-sample y se entreno una segunda etapa
solo con acciones OOF historicas.

Decision:

```text
NO_GO_CLEAN_OOF_FUSION_STAGE2_COST05
```

Resultado:

- ninguna configuracion robusta;
- ninguna configuracion stage2 robusta;
- selector final `clean_stage2_ridge_a100p0`, cap `50`;
- test pass cost0.50 `0,40`;
- test worst day cost0.50 `-16,0244`.

La primera etapa limpia conserva ranking modesto:

- test AUC `0,5656`;
- top35 test mean cost0.50 `+0,1778`;
- top35 test mean cost1.00 `-0,1899`.

Conclusion para la memoria:

```text
el orderbook y la fusion contienen senal academica modesta, pero la segunda
etapa no produce una politica estable. La senal mejora al acumular historial,
por lo que el siguiente estudio debe centrarse en madurez y regimen temporal,
no en aumentar automaticamente la complejidad del modelo.
```

## Cierre de madurez, regimen y warm-up

Los estudios posteriores comprobaron esa hipotesis:

```text
docs/UPSTREAM_OOF_MATURITY_REGIME_AUDIT_V1.md
docs/UPSTREAM_OOF_WARMUP_GATE_AUDIT_V1.md
```

Resultados:

- ninguna ventana expanding/rolling produce una policy seleccionable y robusta;
- rolling6 deja un near-miss con suma test cost0.50 `+198,0718`, pero worst day
  `-1,6160`;
- validation selecciona no operar hasta disponer de 8 dias;
- incluso ese warm-up falla test con worst day cost0.50 `-28,9518`.

Conclusion final de la rama H60:

```text
la senal existe, pero no es suficientemente estable para justificar una policy
operativa. Seguir ajustando ventanas, caps o una segunda etapa simple aumenta
el riesgo de sobreajuste.
```

Siguiente fase defendible:

```text
H120 bajo protocolo OOF limpio, o labels/logs reales de fill.
```

## Cierre del horizonte H120

Se ejecuto la alternativa H120 sin cambiar arquitectura ni protocolo:

```text
docs/UPSTREAM_OOF_H120_AUDIT_V1.md
```

Para que la comparacion fuera justa se creo `H60_paired`: el mismo modelo H60
entrenado y evaluado solo sobre las secuencias donde H120 tambien existe.

Resultado:

```text
                         H60_paired     H120
selection AUC media         0,5482      0,4921
test AUC                    0,5651      0,4951
test top35 cost0.50        +0,4262     -1,1171
test top35 cost1.00        +0,0579     -1,4910
```

Decision:

```text
NO_GO_CLEAN_OOF_H120_COST05
```

La conclusion defendible para la memoria es:

```text
dar mas tiempo al movimiento no mejora automaticamente la prediccion. En este
corpus H120 destruye la modesta senal de ranking presente en H60. Ninguno de
los dos produce aun una politica estable y seleccionable, por lo que el
siguiente cuello de botella a estudiar es la validez ejecutable del label:
fill, latencia y coste, no una red mas grande.
```

## Nota posterior: del movimiento proxy al maker directo

La fase posterior confirmo la conclusion anterior. Cuando se amplio el dataset
secuencial maker a `24.344` endpoints, el baseline de movimiento H60 seguia
encontrando lift proxy:

```text
top5 test = +2,0456 ticks proxy
```

Pero al convertir esas acciones en un label maker ask-touch, la seleccion
perdia:

```text
-55,3545 ticks buffer0.50
```

Por eso el siguiente baseline serio no predijo `up/down`, sino directamente:

```text
P(fill observable) + E(markout neto | fill)
```

El modelo `maker_direct_two_stage_v2` selecciono una policy con train/validation
y obtuvo en test historico:

```text
57 senales, 9 fills proxy, +166,7763 ticks buffer0.50,
+162,2763 ticks buffer1.00, 3/3 dias positivos.
```

Lectura final:

```text
los baselines direccionales quedan cerrados; el camino prometedor es maker
directo fill/markout. El resultado es exploratorio y necesita holdout fresco
antes de hablar de rentabilidad real.
```
