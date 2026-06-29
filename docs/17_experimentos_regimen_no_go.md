# Complex v1a - prestart H60 regime detector v1

Fecha: 2026-06-02

## Objetivo

Venimos de dos resultados:

```text
specialist v1: hay edge agregado.
risk gate v3: subir thresholds no elimina el dia malo.
```

El problema principal ya no es solo:

```text
predecir que fila tiene edge
```

sino:

```text
saber si el estado actual es operable o peligroso.
```

Esta iteracion prueba una primera version de `regime detector` sin usar
agregados futuros de dia/sesion. Es una puerta fila a fila:

1. primero una fila debe pasar una senal base;
2. despues un meta-modelo decide si esa fila sigue siendo aceptable;
3. los thresholds del meta-modelo se seleccionan en validacion;
4. solo despues se evalua test terminal.

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_regime_detector_v1.py
```

Notebook:

```text
notebooks/18_complex_v1a_prestart_h60_regime_detector_v1.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_regime_detector_v1/
```

Ficheros:

| Fichero | Uso |
|---|---|
| `regime_model_metrics.csv` | Calidad predictiva de los meta-modelos. |
| `regime_policy_results.csv` | Todas las combinaciones de base signal + meta-gate. |
| `regime_selected_candidates.csv` | Candidatos que pasan validacion. |
| `regime_day_breakdown.csv` | Desglose diario de cada candidato. |
| `decision.json` | Decision resumida. |
| `models/` | Modelos entrenados para reproducibilidad. |

Runtime:

```text
42 segundos
```

## Universo

Se mantiene:

```text
strict_45_60_early
```

Es decir:

```text
45-60 segundos antes del inicio de ventana
```

Con los filtros estrictos ya usados:

- prestart;
- H60 soportado;
- spread <= 2 ticks;
- coste visible <= 1.25 ticks;
- age <= 1000 ms;
- libro completo;
- no degradado.

## Que senales base se probaron

El detector no opera sobre todo el dataset. Opera solo sobre filas que ya pasan
una senal razonable.

Se probaron cuatro senales base:

| Base policy | Descripcion |
|---|---|
| `specialist_full_v1` | Mejor policy original del especialista: `full_ev >= 0.5` y `healthy >= 0.55`. |
| `riskgate_full_v3` | Mejor puerta simple de v3: `full_ev >= 1.0` y `healthy >= 0.55`. |
| `consensus_strict` | Exige acuerdo entre `full` y `no_clock`. |
| `noclock_v1` | Mejor policy sin reloj. |

## Que aprende el meta-modelo

Target:

```text
exec_net_cost_0p5_H60
```

Es decir:

```text
edge neto a 60s asumiendo coste medio 0.5 del spread visible.
```

Se entrenan:

- un `HistGradientBoostingRegressor` para estimar edge neto;
- un `HistGradientBoostingClassifier` para estimar probabilidad de edge positivo.

Feature sets:

| Feature set | Idea |
|---|---|
| `score_plus_core` | Scores del especialista + variables core observables. |
| `full_plus_scores` | Features completas permitidas + scores. |
| `noclock_plus_scores` | Features sin reloj + scores. |

## Por que esto no mete leakage

Este experimento usa:

- features de la fila actual;
- scores producidos por modelos entrenados antes;
- split temporal;
- seleccion de thresholds en validacion;
- evaluacion final en test.

No usa:

- resultado futuro como feature;
- informacion agregada de todo el dia;
- labels de test para elegir la policy;
- JSONL compactados;
- sidecars nuevos.

Por tanto es una prueba limpia de:

```text
puede una segunda capa fila a fila detectar peligro?
```

## Resultado principal

Decision:

```text
NO_GO_REGIME_DETECTOR
```

Mejor candidato seleccionado en validacion:

```text
base_policy      = riskgate_full_v3
risk_feature_set = noclock_plus_scores
policy           = risk_ev_ge_1.25
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|
| validation | 322 | +1.6148 | +1.4314 | +1.0646 | 0 |
| test | 543 | +0.8065 | +0.6157 | +0.2342 | 1 |

Lectura:

```text
el meta-gate mejora mucho el promedio, pero no elimina la inestabilidad diaria.
```

## Desglose test del mejor candidato

| Dia | Acciones | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|
| 2026-05-23 | 169 | +2.6011 | +2.4033 | +2.0078 |
| 2026-05-24 | 179 | -0.3610 | -0.5545 | -0.9414 |
| 2026-05-25 | 195 | +0.3229 | +0.1407 | -0.2237 |

El dia `2026-05-24` sigue siendo negativo.

Ademas, `2026-05-25` aguanta a coste 0.50, pero no a coste 1.00.

## Resultado mas importante

Se comprobo si existia alguna policy que pasara test completo:

```text
acciones suficientes
net positivo a coste 0.25
net positivo a coste 0.50
net positivo a coste 1.00
cero dias negativos en test a coste 0.50
```

Resultado:

```text
0 policies
```

Esto es fuerte:

```text
no es que el criterio de seleccion haya elegido mal;
es que el meta-gate fila a fila no encuentra una solucion robusta.
```

## Calidad predictiva del meta-modelo

Los meta-modelos no muestran poder predictivo estable fuera de train.

Ejemplos:

| Base | Feature set | Spearman val | AUC val | Spearman test | AUC test |
|---|---|---:|---:|---:|---:|
| `riskgate_full_v3` | `noclock_plus_scores` | +0.0090 | 0.5078 | +0.0118 | 0.5186 |
| `consensus_strict` | `score_plus_core` | +0.0594 | 0.5334 | -0.0626 | 0.4912 |
| `noclock_v1` | `score_plus_core` | +0.0701 | 0.5509 | -0.0276 | 0.4976 |

Lectura:

```text
la segunda capa no esta aprendiendo un patron fila a fila estable.
```

## Interpretacion

El problema del dia malo no parece visible en una fila aislada.

Es decir, una fila del `2026-05-24` puede parecer buena segun:

- EV previsto;
- healthy probability;
- spread;
- coste visible;
- age;
- microprice;
- momentum externo;
- agreement entre modelos.

Pero el resultado agregado del dia sale mal.

Eso sugiere que el fallo puede estar en una capa mas alta:

- regimen de mercado;
- estado por sesion/market;
- cambio de microestructura;
- latencia/fill no representada por el label;
- seleccion adversa no observable fila a fila;
- poca muestra temporal para distinguir dias buenos/malos.

## Decision

No se aprueba:

```text
bot
```

Tampoco se aprueba todavia:

```text
pasar directamente a CNN/LSTM/Transformer como solucion principal
```

Porque el fallo actual no es que falte un modelo mas expresivo por fila. El
fallo es que no sabemos detectar bien cuando el regimen completo deja de ser
operable.

## Siguiente estrategia

La siguiente iteracion deberia subir un nivel:

```text
session_block_regime_v1
```

Objetivo:

```text
detectar estados por sesion/mercado/bloque, no por fila aislada.
```

Pero hay que hacerlo con mucho cuidado para no meter futuro.

Dos caminos posibles:

### Camino A - live-safe

Crear features solo con informacion disponible antes de operar:

- estadisticas acumuladas hasta ese momento;
- ultimos N snapshots;
- desacuerdo reciente entre modelos;
- volatilidad reciente;
- cambios recientes de spread/coste;
- numero reciente de oportunidades;
- drift reciente entre Polymarket y perp/spot.

Esto es mas dificil, pero es el camino correcto para bot.

### Camino B - investigacion offline

Crear agregados por sesion/bloque usando toda la ventana `45-60s`.

Esto puede servir para entender si existe informacion de regimen, pero no debe
venderse como live-safe hasta reescribirlo con features causales.

Recomendacion:

```text
hacer primero una auditoria offline de session/block regime para saber si
existe senal de regimen. Si existe, rehacerla despues en version causal.
```

---

# Complex v1a - prestart H60 session/block regime v1

Fecha: 2026-06-02

## Objetivo

La iteracion anterior dejo una conclusion clara:

```text
el regimen malo no se detecta bien mirando filas aisladas.
```

Por eso esta iteracion sube un nivel.

En vez de decidir fila a fila, agrupamos por:

```text
session_id + market_id + token_id
```

dentro del universo:

```text
prestart H60, 45-60s antes del inicio
```

La pregunta es:

```text
si agregamos el bloque completo, aparece una senal de regimen?
```

## Advertencia importante

Este experimento es una auditoria offline.

Usa estadisticas agregadas del bloque completo `45-60s`. Eso sirve para saber
si hay informacion de regimen en el bloque, pero todavia no es una regla
live-safe.

Para bot habria que convertirlo despues a version causal:

```text
solo informacion disponible antes de mandar cada orden.
```

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_session_block_regime_v1.py
```

Notebook:

```text
notebooks/19_complex_v1a_prestart_h60_session_block_regime_v1.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_session_block_regime_v1/
```

Ficheros:

| Fichero | Uso |
|---|---|
| `block_table_scored.csv` | Tabla de bloques con features agregadas y prediccion. |
| `block_model_metrics.csv` | Calidad predictiva del modelo de bloque. |
| `block_policy_results.csv` | Todas las puertas de bloque evaluadas. |
| `block_selected_candidates.csv` | Candidatos que pasan validacion. |
| `block_day_breakdown.csv` | Desglose diario de cada gate. |
| `decision.json` | Decision resumida. |
| `models/` | Modelos entrenados. |

Runtime:

```text
13 segundos
```

## Que se agrego por bloque

Para cada bloque se calcularon estadisticas de:

- scores del especialista `full` y `no_clock`;
- probabilidad `healthy`;
- desacuerdo entre modelos;
- spread;
- coste visible;
- age;
- Polymarket mid;
- microprice;
- retornos Polymarket;
- retornos spot/perp;
- basis;
- gaps spot/perp;
- volumen/flujo externo disponible en el dataset.

Tambien se incluyo:

```text
selected_actions
selected_action_frac
```

Es decir, cuantas filas del bloque pasaban la senal base.

No se usan como features:

- `exec_net_*`;
- `target_*`;
- labels futuros;
- resultado de test.

## Senales base probadas

Se probo el block gate encima de cuatro senales base:

| Base policy | Descripcion |
|---|---|
| `specialist_full_v1` | Policy original del especialista full. |
| `riskgate_full_v3` | Mejor threshold simple previo. |
| `consensus_strict` | Exige acuerdo entre full y no_clock. |
| `noclock_v1` | Policy sin reloj. |

## Modelo de bloque

Se entrenaron modelos tabulares simples:

- `HistGradientBoostingRegressor`;
- `HistGradientBoostingClassifier`.

Target del bloque:

```text
media de exec_net_cost_0p5_H60 de las acciones seleccionadas en ese bloque.
```

Luego se probaron puertas del tipo:

```text
permitir operar solo si block_ev_pred >= threshold
```

y opcionalmente:

```text
block_positive_proba >= threshold
```

## Resultado principal

Decision:

```text
NO_GO_SESSION_BLOCK_REGIME
```

Mejor candidato seleccionado por validacion:

```text
base_policy = riskgate_full_v3
policy      = block_ev_ge_0_p_ge_0.5
```

Resultado:

| Split | Acciones | Bloques | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 360 | 123 | +1.7723 | +1.5891 | +1.2226 | 0 |
| test | 605 | 198 | +0.5021 | +0.3117 | -0.0691 | 1 |

Lectura:

```text
la puerta de bloque mejora, pero no pasa el criterio duro de test.
```

Falla por dos cosas:

- sigue teniendo `1` dia negativo en test a coste 0.50;
- no aguanta coste 1.00 en agregado de test.

## Desglose test del mejor candidato

| Dia | Acciones | Bloques | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|---:|
| 2026-05-23 | 192 | 66 | +1.6951 | +1.4997 | +1.1087 |
| 2026-05-24 | 197 | 64 | -0.6983 | -0.8940 | -1.2855 |
| 2026-05-25 | 216 | 68 | +0.5365 | +0.3554 | -0.0067 |

Otra vez:

```text
2026-05-24 rompe la estabilidad.
```

## Hay alguna puerta que arregle test?

Si, pero con trampa metodologica.

El experimento encontro:

```text
7 policies que pasan test completo
```

Ejemplo:

```text
riskgate_full_v3 + block_top_30pct
```

En test:

- 181 acciones;
- 51 bloques;
- `+2.4852` ticks a coste 0.50;
- `+2.1029` ticks a coste 1.00;
- `0` dias negativos.

Pero en validacion:

- solo 68 acciones;
- no cumple el minimo de acciones exigido;
- por tanto no se puede aceptar como regla principal.

Otro ejemplo:

```text
noclock_v1 + block_top_50pct
```

Pasa test, pero en validacion tiene `1` dia negativo a coste 0.50.

Conclusion:

```text
hay informacion de regimen en el bloque, pero aun no tenemos una regla
seleccionable de forma limpia y robusta.
```

## Calidad predictiva del modelo de bloque

Metricas relevantes:

| Base policy | Spearman val | AUC val | Spearman test | AUC test |
|---|---:|---:|---:|---:|
| `specialist_full_v1` | -0.0491 | 0.5016 | +0.0554 | 0.5330 |
| `riskgate_full_v3` | -0.1735 | 0.4486 | +0.1552 | 0.5855 |
| `consensus_strict` | -0.1649 | 0.4340 | +0.1039 | 0.5647 |
| `noclock_v1` | -0.0635 | 0.5035 | +0.0139 | 0.5016 |

Esto es mixto:

- en test aparece algo de senal para `riskgate_full_v3`;
- en validacion no es estable;
- por tanto no se puede aprobar como generalizable.

## Interpretacion sencilla

Esta iteracion nos dice tres cosas:

1. Agregar por bloque ayuda mas que mirar filas aisladas.
2. Hay reglas que limpian test, asi que la idea de regimen no esta muerta.
3. Pero esas reglas no fueron justificables desde validacion, asi que no son
   aprobables.

La conclusion no es:

```text
no hay edge
```

La conclusion es:

```text
hay edge, pero aun no tenemos una forma robusta de saber cuando activarlo.
```

## Decision

No se aprueba:

```text
bot
```

No se aprueba aun:

```text
encoder complejo como solucion directa
```

Si se aprueba como siguiente investigacion:

```text
causal_block_regime_v1
```

## Siguiente paso

Convertir esta auditoria offline en algo causal.

Ejemplos de features live-safe:

- estadisticas acumuladas solo hasta el snapshot actual;
- ultimos N snapshots del mismo bloque;
- cambio reciente de `ev_pred`;
- cambio reciente de `healthy_proba`;
- desacuerdo reciente full vs no_clock;
- spread/coste medio reciente;
- volatilidad reciente;
- drift reciente Polymarket vs perp/spot;
- numero reciente de oportunidades en el bloque.

Objetivo:

```text
aproximar la informacion util del bloque sin mirar el futuro del propio bloque.
```

Si `causal_block_regime_v1` tambien falla, entonces la opcion honesta sera:

- pasar a encoders solo como investigacion comparativa;
- o admitir que falta mas corpus/live execution para distinguir regimenes.

---

# Complex v1a - prestart H60 causal block regime v1

Fecha: 2026-06-02

## Objetivo

La auditoria anterior de bloque offline encontro algo prometedor pero peligroso:

```text
agregar el bloque completo ayuda a detectar regimen,
pero no es live-safe porque mira el bloque 45-60s entero.
```

Esta iteracion intenta convertir esa idea en algo causal:

```text
usar solo el snapshot actual y estadisticas acumuladas del bloque hasta ese
momento.
```

La pregunta es:

```text
podemos aproximar la informacion de regimen sin mirar el futuro del bloque?
```

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_causal_block_regime_v1.py
```

Notebook:

```text
notebooks/20_complex_v1a_prestart_h60_causal_block_regime_v1.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_causal_block_regime_v1/
```

Ficheros:

| Fichero | Uso |
|---|---|
| `causal_model_metrics.csv` | Calidad predictiva de los modelos causales. |
| `causal_policy_results.csv` | Todas las policies evaluadas. |
| `causal_selected_candidates.csv` | Candidatos que pasan validacion. |
| `causal_day_breakdown.csv` | Desglose diario de cada policy. |
| `causal_scored_rows_preview.csv` | Preview de filas puntuadas. |
| `decision.json` | Decision resumida. |
| `models/` | Modelos entrenados. |

Runtime:

```text
296 segundos
```

## Universo

Se mantiene el universo estricto:

```text
prestart H60
45-60s antes del inicio
spread <= 2
visible cost <= 1.25
age <= 1000 ms
libro completo
no degradado
```

Tamaño:

| Metrica | Valor |
|---|---:|
| Filas estrictas | 40.811 |
| Bloques `session_id + market_id + token_id` | 5.831 |
| Filas medias por bloque | ~7 |
| Train rows | 28.095 |
| Validation rows | 5.301 |
| Test rows | 7.415 |

## Features causales creadas

Para cada fila se crearon `243` features causales.

Regla:

```text
current row + prefix/rolling/delta stats inside block up to current snapshot only
```

Es decir, para cada variable numerica relevante:

- valor actual;
- media acumulada vista;
- media rolling de los ultimos 3 snapshots vistos;
- minimo visto;
- maximo visto;
- rango visto;
- delta contra primer snapshot del bloque;
- delta contra snapshot anterior.

Tambien:

- numero de snapshots vistos en el bloque;
- segundos transcurridos desde primer snapshot visto;
- cuantas filas de la senal base han aparecido hasta ahora;
- tasa acumulada de seleccion de la senal base.

Variables base usadas:

- scores `full_features`;
- scores `no_clock`;
- healthy probabilities;
- desacuerdo entre modelos;
- spread;
- coste visible;
- age;
- Polymarket mid;
- microprice;
- retornos PM;
- retornos spot/perp;
- basis;
- gaps spot/perp;
- volumen/flujo externo disponible.

No se usan:

- resultados futuros;
- `exec_net_*` como feature;
- `target_*` como feature;
- agregados del resto del bloque;
- labels de test para seleccionar policy.

## Senales base

El gate causal se probo encima de cuatro senales:

| Base policy | Descripcion |
|---|---|
| `specialist_full_v1` | Policy original del especialista full. |
| `riskgate_full_v3` | Mejor threshold simple previo. |
| `consensus_strict` | Acuerdo entre full y no_clock. |
| `noclock_v1` | Policy sin reloj. |

## Resultado principal

Decision:

```text
NO_GO_CAUSAL_BLOCK_REGIME
```

Mejor candidato seleccionado por validacion:

```text
base_policy = riskgate_full_v3
policy      = causal_ev_ge_-2
```

Resultado:

| Split | Acciones | Bloques | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 373 | 132 | +1.4852 | +1.3014 | +0.9339 | 0 |
| test | 631 | 216 | +0.7670 | +0.5767 | +0.1963 | 1 |

Lectura:

```text
el gate causal mejora margen agregado, pero no elimina el dia malo.
```

## Desglose del mejor candidato

| Dia | Split | Acciones | Net 0.50 | Net 1.00 |
|---|---|---:|---:|---:|
| 2026-05-21 | validation | 200 | +0.3783 | +0.0091 |
| 2026-05-22 | validation | 173 | +2.3686 | +2.0031 |
| 2026-05-23 | test | 206 | +2.0690 | +1.6793 |
| 2026-05-24 | test | 207 | -0.7650 | -1.1555 |
| 2026-05-25 | test | 218 | +0.4406 | +0.0784 |

Otra vez:

```text
2026-05-24 rompe la estabilidad.
```

## Hallazgo valioso: familia no_clock causal

Aunque la decision dura es `NO GO`, aparecio una familia importante:

```text
noclock_v1 + causal context
```

Ejemplo:

```text
noclock_v1 + causal_ev_ge_0.5
```

Resultado:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|
| validation | 518 | +0.9156 | +0.7299 | +0.3584 | 1 |
| test | 907 | +0.8523 | +0.6621 | +0.2818 | 0 |

En test pasa muy bien:

| Dia | Acciones | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|
| 2026-05-23 | 263 | +1.6583 | +1.2672 |
| 2026-05-24 | 313 | +0.1935 | -0.1945 |
| 2026-05-25 | 331 | +0.3137 | -0.0509 |

Pero falla validacion:

| Dia | Acciones | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|
| 2026-05-21 | 265 | -0.7258 | -1.0969 |
| 2026-05-22 | 253 | +2.2546 | +1.8828 |

Conclusion:

```text
esta familia es interesante, pero no aprobable bajo criterio duro.
```

Es interesante porque:

- elimina el dia malo de test a coste 0.50;
- mantiene muchas acciones;
- no depende del reloj fuerte;
- sugiere que el contexto causal ayuda.

No es aprobable porque:

- ya fallo un dia de validacion;
- a coste 1.00 algunos dias siguen flojos;
- aceptar esta regla seria relajar el protocolo despues de ver test.

## Politicas que pasan test

El experimento encontro:

```text
30 policies que pasan test completo
```

Pero ninguna de esas policies pasa tambien la regla dura de validacion.

Esto es muy informativo:

```text
hay informacion causal suficiente para limpiar test, pero no hay evidencia
estable suficiente para seleccionarla sin mirar test.
```

## Calidad predictiva de los modelos

Metricas por base:

| Base policy | Spearman val | AUC val | Spearman test | AUC test |
|---|---:|---:|---:|---:|
| `specialist_full_v1` | -0.0617 | 0.4796 | +0.0369 | 0.5246 |
| `riskgate_full_v3` | -0.1510 | 0.4122 | +0.0397 | 0.5314 |
| `consensus_strict` | -0.1483 | 0.4054 | -0.0110 | 0.4880 |
| `noclock_v1` | -0.0524 | 0.4952 | +0.0246 | 0.5185 |

No son modelos predictivos fuertes.

Esto refuerza la lectura:

```text
la informacion existe, pero el tabular causal no consigue aprenderla de forma
estable.
```

## Decision

No se aprueba:

```text
bot
```

No se aprueba:

```text
tabular causal gate como solucion final.
```

Si se aprueba:

```text
pasar a una prueba de modelos secuenciales, pero con objetivo muy concreto.
```

## Que hemos extraido de valor

La conclusion ya no es generica.

No vamos a probar CNN/LSTM/Transformer "porque si".

La hipotesis concreta es:

```text
el estado del bloque tiene una dinamica corta de 7-8 snapshots que el modelo
tabular con estadisticas de prefijo no captura bien.
```

Por tanto el siguiente experimento debe ser:

```text
sequence_encoder_probe_v1
```

Pero acotado:

- universo `strict_45_60_early`;
- horizonte H60;
- target `exec_net_cost_0p5_H60`;
- baseline a batir: `noclock_v1 + causal_ev_ge_0.5` y
  `riskgate_full_v3 + causal_ev_ge_-2`;
- input: secuencia corta de snapshots por bloque;
- split temporal igual;
- aprobacion solo si seleccion en validacion pasa test sin dia negativo a coste
  0.50 y con agregado positivo a coste 1.00.

## Siguiente paso recomendado

Preparar un dataset secuencial pequeño:

```text
data/experiments/sequence_probe_v1/
```

Con:

- `session_id + market_id + token_id` como bloque;
- secuencias de hasta 8 snapshots;
- masks de longitud;
- features core/no_clock;
- scores del especialista como canales auxiliares;
- labels por fila o por ultima fila seleccionada.

Primeros modelos razonables:

1. Logistic/MLP sobre flatten de secuencia.
2. 1D CNN pequena.
3. GRU pequena.

No empezaria aun por Transformer grande.

La vara de medir queda clara:

```text
debe superar a los gates causales tabulares en estabilidad temporal, no solo en
promedio.
```

---

