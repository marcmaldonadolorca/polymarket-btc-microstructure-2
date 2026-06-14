<!-- fuente original: COMPLEX_V1A_PRESTART_H60_SPECIALIST_V1.md -->

# Complex v1a - prestart H60 specialist v1

Fecha: 2026-06-02

## Objetivo

La iteracion anterior encontro un candidato estricto:

```text
strict_45_60_early + pred_gt_0.5
```

Ese candidato funcionaba usando el modelo prestart global ya entrenado.

Esta iteracion hace el siguiente paso logico:

```text
entrenar modelos especialistas solo en el universo prestart estricto.
```

La pregunta ya no es:

```text
hay algo en prestart?
```

La pregunta ahora es:

```text
si entrenamos especificamente sobre esa zona, mejora la seleccion y aguanta
validation/test?
```

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_specialist_v1.py
```

Notebook:

```text
notebooks/15_complex_v1a_prestart_h60_specialist_v1.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_specialist_v1/
```

Modelos guardados:

```text
data/experiments/complex_v1a_prestart_h60_specialist_v1/models/
```

Runtime:

```text
25 segundos
```

## Universos entrenados

Se entrenaron modelos en dos universos:

| Universo | Rango | Train | Validation | Test |
|---|---:|---:|---:|---:|
| `strict_0_60_all` | 0-60s antes del inicio | 103.243 | 20.397 | 27.615 |
| `strict_45_60_early` | 45-60s antes del inicio | 28.095 | 5.301 | 7.415 |

Ambos mantienen filtros estrictos:

- `window_phase = prestart`;
- `target_supported_H60 = true`;
- `spread_ticks <= 2`;
- `visible_entry_cost_ticks <= 1.25`;
- `age_ms <= 1000`;
- `full_book_ratio >= 0.999`;
- `degraded_ratio = 0`.

## Modelos

Por cada universo se entrenaron dos modelos:

| Modelo | Target | Tipo |
|---|---|---|
| EV | `exec_net_cost_0p25_H60` | `HistGradientBoostingRegressor` |
| Healthy | `healthy_fill_proxy_0p25_H60` | `HistGradientBoostingClassifier` |

Y dos sets de features:

| Feature set | Que significa |
|---|---|
| `full_features` | Todas las features permitidas por el manifest. |
| `no_clock` | Quita variables directas de reloj/fase: `seconds_*`, `window_progress`, `phase_bucket`, `window_phase`, `temporality`. |

La ablation `no_clock` es importante porque pregunta:

```text
la senal depende solo de saber el segundo exacto de la ventana, o tambien hay
microestructura/contexto?
```

## Seleccion de policies

La seleccion se hizo solo en:

```text
validation_initial
```

Una policy debia cumplir:

- al menos `150` acciones en validation;
- `mean_net_cost_0p25 > 0` en validation;
- `mean_net_cost_0p5 > 0` en validation.

Score de seleccion:

```text
mean_net_cost_0p5 * sqrt(actions) - 0.01 * adverse_proxy_pct
```

Esto favorece:

- margen a coste 0.50;
- numero razonable de acciones;
- menor riesgo adverso.

## Resultado principal

Decision:

```text
RESEARCH_PASS_STRONG_NOT_BOT
```

Mejor candidato:

```text
universe = strict_45_60_early
feature_set = full_features
policy = ev_gt_0.5_healthy_ge_0.55
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|
| validation | 497 | +1.3412 | +1.1572 | +0.7893 |
| test | 834 | +0.7915 | +0.6022 | +0.2235 |

Esto mejora el candidato anterior porque:

- combina EV y probabilidad healthy;
- reduce acciones de baja calidad;
- mantiene resultado positivo incluso a coste 1.00;
- sigue pasando en test terminal.

## Ablation no_clock

La variante sin reloj tambien pasa:

```text
universe = strict_45_60_early
feature_set = no_clock
policy = ev_gt_0.75_healthy_ge_0.5
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|
| validation | 614 | +0.9521 | +0.7658 | +0.3932 |
| test | 1.094 | +0.6184 | +0.4297 | +0.0522 |

Lectura:

```text
la senal no desaparece al quitar las variables directas de reloj.
```

Esto aumenta la confianza: parece haber informacion en microestructura/contexto,
no solo en el timestamp relativo.

Pero la version sin reloj queda mas justa a coste 1.00:

```text
test cost 1.00 = +0.0522 ticks
```

No sobra margen.

## Universos que fallan

`strict_0_60_all` no pasa como candidato fuerte.

| Feature set | Policy | Validation 0.50 | Test 0.50 | Test 1.00 | Estado |
|---|---|---:|---:|---:|---|
| `full_features` | `ev_gt_2` | +2.4003 | -0.2158 | -0.5946 | Falla test |
| `no_clock` | `top_20pct_healthy_ge_0.5` | +1.7592 | +0.3098 | -0.0669 | Falla coste 1.00 |

Lectura:

```text
mezclar 0-60s sigue siendo demasiado amplio.
```

La zona robusta de momento es:

```text
45-60s antes del inicio
```

## Calidad predictiva

Metricas principales:

| Universo | Features | Split | AUC positivo 0.25 | AUC healthy |
|---|---|---|---:|---:|
| `strict_45_60_early` | `full_features` | validation | 0.5683 | 0.5867 |
| `strict_45_60_early` | `full_features` | test | 0.5467 | 0.5704 |
| `strict_45_60_early` | `no_clock` | validation | 0.5652 | 0.5755 |
| `strict_45_60_early` | `no_clock` | test | 0.5483 | 0.5576 |

No son AUC espectaculares.

La estrategia funciona por seleccion de cola:

```text
el modelo no predice perfecto; filtra situaciones mejores que la media.
```

Esto es normal en microestructura, pero obliga a controlar mucho la estabilidad.

## Advertencia temporal

El agregado validation/test pasa fuerte, pero el desglose diario muestra
inestabilidad.

Mejor candidato `full_features`:

| Split | Dia | Acciones | Net 0.50 |
|---|---|---:|---:|
| validation | 2026-05-21 | 258 | -0.0634 |
| validation | 2026-05-22 | 239 | +2.4749 |
| test | 2026-05-23 | 281 | +1.5299 |
| test | 2026-05-24 | 279 | -0.8403 |
| test | 2026-05-25 | 274 | +1.1195 |

La variante `no_clock` tambien tiene dias flojos:

| Split | Dia | Acciones | Net 0.50 |
|---|---|---:|---:|
| validation | 2026-05-21 | 321 | -0.4372 |
| validation | 2026-05-22 | 293 | +2.0838 |
| test | 2026-05-23 | 343 | +1.1952 |
| test | 2026-05-24 | 361 | -0.1256 |
| test | 2026-05-25 | 390 | +0.2703 |

Esto es crucial:

```text
el candidato pasa agregado, pero aun no pasa estabilidad diaria.
```

Por eso la decision sigue siendo:

```text
NO GO bot.
```

## Decision

### Estado

```text
RESEARCH_PASS_STRONG_NOT_BOT
```

### Que si podemos decir

Podemos decir:

- hay un candidato offline serio;
- la zona `45-60s` es mejor que mezclar `0-60s`;
- combinar EV + healthy mejora la seleccion;
- la senal sobrevive parcialmente sin variables directas de reloj;
- el resultado agregado test es positivo incluso a coste 1.00.

### Que no podemos decir

No podemos decir:

- que sea bot-ready;
- que aguante todos los dias;
- que la ejecucion real vaya a llenar;
- que no exista sesgo de regimen temporal;
- que sea rentable con slippage/adverse fill real.

## Siguiente paso recomendado

Antes de CNN/LSTM/Transformer:

```text
prestart_H60_specialist_stability_v2
```

Objetivo:

- walk-forward por dia o bloques de sesiones;
- seleccionar threshold en un dia/bloque y evaluar el siguiente;
- medir cuantos dias pasan y cuantos fallan;
- stress de coste adicional;
- comparar full vs no_clock;
- estudiar especialmente el dia negativo `2026-05-24`.

Solo si eso aguanta:

```text
GO a encoder/secuencia prestart H60.
```

## Conclusion corta

```text
El especialista prestart H60 mejora mucho la tesis: strict_45_60_early con EV
+ healthy pasa validation y test hasta coste 1.00, y la version no_clock tambien
pasa, lo que sugiere que no es solo reloj. Pero el desglose diario muestra dias
negativos, especialmente 2026-05-24. La decision correcta es continuar con
stability/walk-forward, no pasar todavia a bot ni a arquitectura compleja final.
```

## Actualizacion - stability v2

Se ejecuto:

```text
docs/COMPLEX_V1A_PRESTART_H60_SPECIALIST_STABILITY_V2.md
```

Resultado:

- walk-forward diario: `22/27` pares pasan hasta coste 1.00;
- pass rate global: `81,48%`;
- pass rate `full_features`: `85,71%`;
- pass rate `no_clock`: `76,92%`;
- pero la politica estatica falla en `2026-05-24`;
- `2026-05-23 -> 2026-05-24` tambien falla en walk-forward.

Decision:

```text
NO GO bot.
NO GO estrategia fija.
GO a risk gate de regimen.
```


---

<!-- fuente original: COMPLEX_V1A_PRESTART_H60_STRICT_CANDIDATE_V1.md -->

# Complex v1a - prestart H60 strict candidate v1

Fecha: 2026-06-02

## Objetivo

La auditoria anterior concluyo:

```text
prestart = data-operable but execution-unproven
```

Es decir:

- no parece libro roto;
- hay orderbook observable;
- hay senal H60;
- pero no tenemos fill real probado.

Esta iteracion convierte esa idea en un candidato mas estricto.

La pregunta es:

```text
si limpiamos mucho el universo prestart, queda una policy positiva en
validation y test?
```

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_strict_candidate_v1.py
```

Notebook:

```text
notebooks/14_complex_v1a_prestart_h60_strict_candidate_v1.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_strict_candidate_v1/
```

Ficheros:

| Fichero | Uso |
|---|---|
| `strict_universe_summary.csv` | Tamano y calidad de cada universo estricto. |
| `strict_policy_results.csv` | Todas las politicas evaluadas. |
| `strict_selected_policies.csv` | Politicas elegidas usando solo validation. |
| `strict_selected_bucket_breakdown.csv` | Desglose por bucket temporal. |
| `decision.json` | Decision automatica del experimento. |

Runtime:

```text
6 segundos
```

No se entreno modelo nuevo. Se reutilizo:

```text
hgb_ev_cost0p25_prestart_experimental_H60.joblib
```

## Filtros estrictos

Filtro base:

| Filtro | Valor |
|---|---:|
| `window_phase` | `prestart` |
| `target_supported_H60` | `true` |
| `seconds_before_start` | `0-60s` |
| `spread_ticks` | `<= 2` |
| `visible_entry_cost_ticks` | `<= 1.25` |
| `age_ms` | `<= 1000` |
| `full_book_ratio` | `>= 0.999` |
| `degraded_ratio` | `<= 0` |

Luego se evaluaron varios subuniversos:

| Universo | Rango |
|---|---|
| `strict_0_60_all` | 0-60s antes del inicio |
| `strict_0_30_near_start` | 0-30s |
| `strict_30_60_transition` | 30-60s |
| `strict_0_10_last_seconds` | 0-10s |
| `strict_10_30_near` | 10-30s |
| `strict_30_45_mid` | 30-45s |
| `strict_45_60_early` | 45-60s |

## Criterio de seleccion

La seleccion se hizo solo con:

```text
terminal_split = validation_initial
```

Una policy solo podia ser candidata si:

- tenia al menos `200` acciones en validation;
- `mean_net_cost_0p25 > 0` en validation;
- `mean_net_cost_0p5 > 0` en validation.

El score de seleccion fue:

```text
mean_net_cost_0p5 * sqrt(actions)
```

Esto fuerza una seleccion algo mas conservadora que mirar solo coste 0.25.

## Resultado principal

Decision del experimento:

```text
RESEARCH_PASS_NOT_BOT
```

Mejor candidato:

```text
universe = strict_45_60_early
policy = pred_gt_0.5
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|
| validation | 828 | +1.3805 | +1.1921 | +0.8154 |
| test | 1.286 | +0.6916 | +0.5049 | +0.1315 |

Esto es la primera vez que vemos un candidato prestart estricto que:

- pasa validation;
- pasa test;
- no solo aguanta coste 0.25;
- tambien aguanta coste 0.50;
- incluso queda positivo a coste 1.00.

Eso es buena noticia, pero todavia no aprueba bot.

## Comparacion con otros universos

| Universo | Policy | Validation Net 0.50 | Test Net 0.50 | Decision |
|---|---|---:|---:|---|
| `strict_0_60_all` | `pred_gt_1` | +0.3552 | +0.0222 | Pasa, pero muy justo en test. |
| `strict_30_60_transition` | `top_10pct` | +2.4099 | -0.0662 | Falla coste 0.50 en test. |
| `strict_30_45_mid` | `top_20pct` | +2.2157 | -0.8681 | Falla test. |
| `strict_45_60_early` | `pred_gt_0.5` | +1.1921 | +0.5049 | Mejor candidato. |
| `strict_0_30_near_start` | `NO_TRADE` | - | - | No pasa validation. |
| `strict_0_10_last_seconds` | `NO_TRADE` | - | - | No pasa validation. |
| `strict_10_30_near` | `NO_TRADE` | - | - | No pasa validation. |

Lectura:

```text
la senal no esta cerca del inicio inmediato, sino antes: 45-60s antes del
inicio de ventana.
```

Esto puede apuntar a posicionamiento previo, no a scalping de ultimo segundo.

## Universo base vs seleccion del modelo

El universo completo `strict_45_60_early` no es rentable de media:

| Split | Filas | Net 0.25 medio universo |
|---|---:|---:|
| train | 28.095 | -0.6872 |
| validation | 5.301 | -0.6890 |
| test | 7.415 | -0.6899 |

El edge aparece solo al seleccionar filas con el modelo:

```text
strict_45_60_early + pred_gt_0.5
```

Esto es importante porque evita una conclusion falsa:

```text
no basta con entrar siempre 45-60s antes; hay que seleccionar situaciones.
```

## Interpretacion

Lo que parece estar pasando:

1. El modelo global prestart aprende una zona temporal con informacion util.
2. El audit mostro que los buckets cercanos al inicio eran inestables.
3. Al imponer filtros de calidad y separar `45-60s`, aparece un candidato mas
   limpio.
4. Ese candidato mantiene edge incluso con coste 1.00 en validation/test.

Pero:

1. Sigue reutilizando un modelo entrenado con el universo prestart completo.
2. La decision sigue dependiendo de libro observable, no de fills reales.
3. Todavia no hay validacion walk-forward especifica de esta subzona.
4. El experimento prueba una hipotesis, no una estrategia lista para ejecutar.

## Decision

### Estado

```text
RESEARCH_PASS_NOT_BOT
```

### Que significa

Significa:

```text
hay suficiente evidencia offline para dedicarle la siguiente fase.
```

No significa:

```text
tenemos un bot rentable.
```

## Siguiente paso recomendado

Crear un modelo especialista:

```text
prestart_H60_specialist_v1
```

Condiciones:

- entrenar solo sobre prestart estricto;
- comparar dos universos:
  - `strict_0_60_all`;
  - `strict_45_60_early`;
- validation/test temporal;
- folds por dias o bloques de sesiones;
- target principal `exec_net_cost_0p25_H60`;
- reportar tambien coste 0.50 y 1.00;
- añadir target auxiliar `healthy_fill_proxy_0p25_H60`;
- no aprobar bot sin prueba o simulacion mas realista de fill.

## Conclusion corta

```text
El candidato estricto mejora la tesis: el edge prestart no vive en todo
prestart, sino sobre todo 45-60s antes del inicio y seleccionado por modelo.
La policy strict_45_60_early + pred_gt_0.5 pasa validation y test, incluso a
coste 1.00. Esto justifica pasar a un modelo especialista prestart H60, pero
todavia no justifica bot porque la ejecucion real sigue sin estar probada.
```

## Actualizacion - specialist v1

Se ejecuto:

```text
docs/COMPLEX_V1A_PRESTART_H60_SPECIALIST_V1.md
```

Resultado:

- decision: `RESEARCH_PASS_STRONG_NOT_BOT`;
- mejor candidato: `strict_45_60_early`, `full_features`,
  `ev_gt_0.5_healthy_ge_0.55`;
- validation: `497` acciones, `+1.3412` ticks coste 0.25,
  `+1.1572` coste 0.50, `+0.7893` coste 1.00;
- test: `834` acciones, `+0.7915` ticks coste 0.25,
  `+0.6022` coste 0.50, `+0.2235` coste 1.00;
- la variante `no_clock` tambien pasa, pero mas justa.

Limitacion nueva:

```text
el agregado pasa, pero hay dias negativos. Siguiente paso: estabilidad
walk-forward diaria.
```


---

