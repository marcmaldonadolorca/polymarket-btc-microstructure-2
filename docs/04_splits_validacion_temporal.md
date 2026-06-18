# Guia 02 - Splits y validacion sin leakage

Fecha: `2026-05-27`

## La pregunta sencilla

Queremos saber si un modelo entrenado con datos pasados funcionaria despues,
cuando llegue un mercado nuevo.

Un split es la forma de separar:

- `train`: datos con los que el modelo aprende;
- `validation`: datos con los que elegimos features, target e
  hiperparametros;
- `test`: datos reservados para la evaluacion final.

La duda razonable es:

> Si el split temporal tiene dias distintos y distribuciones diferentes,
> no seria mas justo barajar aleatoriamente para que train y test se parezcan?

La respuesta corta es:

- un split aleatorio puede ser mas equilibrado;
- pero para trading de corto plazo normalmente no es mas justo;
- es menos realista, porque deja al modelo aprender de frames casi gemelos o
  de dias futuros respecto a los que pretende predecir.

Si en produccion entrenamos con pasado y operamos en futuro, la prueba
principal debe respetar esa direccion del tiempo.

## Por que este dataset hace peligroso el random por filas

El corpus no son observaciones independientes como personas distintas en una
encuesta. Es una pelicula a intervalos de `2 s`.

### 1. Frames vecinos se parecen mucho

Una fila a las `12:00:00` y otra a las `12:00:02` suelen compartir precio,
spread, referencias y estado del mercado. Si una cae en train y la otra en
test, el modelo ya ha visto casi la respuesta.

### 2. El label mira hacia delante

Los labels actuales usan un futuro de `8 s`; los targets alternativos
estudian tambien `16 s`. Dos filas cercanas pueden compartir parte de la
misma trayectoria futura.

Ejemplo conceptual:

```text
fila t=00 usa futuro hasta t=08
fila t=02 usa futuro hasta t=10
```

Separarlas aleatoriamente mezcla informacion altamente solapada.

### 3. Dos tokens de un mercado son complementarios

El token de un lado y su complementario pertenecen a la misma oportunidad.
No seria honesto dejar uno en train y el otro en test.

### 4. Un mismo `condition_id` puede cruzar fronteras

La auditoria del notebook `02_split_strategy_audit.ipynb` detecto
`condition_id` que aparecen a ambos lados de cortes temporales. Por eso,
incluso con fechas, hay que purgar condiciones compartidas.

## Que significa realmente "justo"

Hay dos justicias diferentes:

| Pregunta | Split adecuado |
|---|---|
| Puede el modelo interpolar entre mercados parecidos del mismo periodo? | Agrupado aleatorio, solo diagnostico. |
| Puede el modelo generalizar a mercados que ocurren despues del entrenamiento? | Temporal walk-forward, evaluacion principal. |

EdgeRunner pretende la segunda pregunta. Por eso el resultado que decidira si
avanzar hacia estrategia/bot no puede ser un random split.

## Hallazgo ya medido: existe drift temporal

La primera division cronologica no tiene exactamente la misma distribucion de
labels en todos sus bloques:

| Bloque | Dias UTC | `flat` direccional | `flat` economico |
|---|---|---:|---:|
| Train inicial | `2026-05-11` a `2026-05-20` | `52,19%` | `53,57%` |
| Validation inicial | `2026-05-21` a `2026-05-22` | `53,21%` | `54,50%` |
| Test terminal | `2026-05-23` a `2026-05-25` | `56,53%` | `57,42%` |

Esto no es un defecto que debamos ocultar mezclando filas. Es precisamente un
riesgo de mercado que el modelo tendra que soportar: algunos dias tienen mas
movimiento y otros mas `flat`.

La validacion debe decirnos si el modelo resiste ese cambio.

## Estrategias consideradas

| Estrategia | Como funciona | Ventaja | Problema | Papel recomendado |
|---|---|---|---|---|
| Random por filas | Barajar todas las filas y repartir. | Balance de clases facil. | Leakage grave por frames vecinos, outcomes complementarios y futuro compartido. | Rechazado. |
| Random por `condition_id` | Mantener mercados completos juntos y asignarlos al azar. | Evita mezclar los dos tokens/mismo mercado. | Train puede contener dias posteriores al test; no simula despliegue. | Diagnostico secundario, nunca resultado principal. |
| Random por dia/bloque | Elegir dias completos al azar. | Menos autocorrelacion directa. | Sigue entrenando en el futuro para evaluar el pasado. | Diagnostico de dependencia de regimen. |
| Un corte temporal fijo | Pasado para train, futuro para validation/test. | Realista y simple. | Depende demasiado de un solo periodo de validation. | Util para bloquear test. |
| Expanding walk-forward purgado | Entrena con pasado creciente y valida en varios bloques siguientes. | Realista y prueba estabilidad. | Mas experimentos y purgas. | Protocolo principal de desarrollo. |
| Rolling window purgado | Entrena solo con una ventana reciente y valida en futuro. | Puede adaptarse al drift. | Decide otra longitud y desaprovecha historia. | Robustez posterior. |
| Leave-one-block-out temporal | Evalua distintos bloques como si fueran "desconocidos". | Diagnostica regimen dificil. | Si se entrena con futuro no mide despliegue cronologico. | Analisis, no metrica final. |

## Estrategia principal recomendada

### 1. Un test terminal bloqueado

No se consulta para escoger features, labels, bandas ni hiperparametros.

| Split final | Periodo | Sesiones core | Labels actuales |
|---|---|---:|---:|
| Test terminal | `2026-05-23` a `2026-05-25` | `480` | `448.562` |

Por que terminal:

- es el futuro mas cercano al supuesto despliegue;
- contiene el drift de mayor proporcion `flat`;
- evita obtener un resultado bonito seleccionando el tramo favorable.

### 2. Desarrollo mediante expanding walk-forward

El periodo anterior al test se usa varias veces, siempre entrenando en pasado
y validando en el bloque siguiente.

| Fold | Train | Validation | Labels train | Labels validation |
|---|---|---|---:|---:|
| F1 | `2026-05-11` a `2026-05-14` | `2026-05-15` a `2026-05-16` | `594.490` | `372.178` |
| F2 | `2026-05-11` a `2026-05-16` | `2026-05-17` a `2026-05-18` | `966.668` | `352.442` |
| F3 | `2026-05-11` a `2026-05-18` | `2026-05-19` a `2026-05-20` | `1.319.110` | `329.914` |
| F4 | `2026-05-11` a `2026-05-20` | `2026-05-21` a `2026-05-22` | `1.649.024` | `322.434` |

Con estos folds no confiamos en una sola validacion. Podremos ver:

- si un feature funciona en varios futuros o solo en uno;
- si un horizonte de label aguanta el cambio de regimen;
- si una mejora media viene acompanada de un fold desastroso.

## Purga y embargo: dos protecciones imprescindibles

### Purga por `condition_id`

En cada fold:

1. identificar `condition_id` presentes en validation;
2. retirar de train cualquier fila de esas condiciones;
3. hacer lo mismo entre entrenamiento final y test terminal.

Motivo: un mercado no debe estar parcialmente en aprendizaje y parcialmente
en evaluacion.

La auditoria encontro condiciones cruzadas antes de purgar:

| Frontera | Condiciones que cruzan |
|---|---:|
| Train inicial -> validation | `3` |
| Ajuste hasta `2026-05-22` -> test | `5` |

### Embargo temporal

Alrededor del final del train y comienzo de validation se deja una zona sin
usar cuando las ventanas puedan compartir informacion.

Regla inicial:

```text
embargo = max(horizonte_del_label, lookback_maximo_de_features)
```

Ejemplos:

| Caso | Embargo minimo |
|---|---:|
| Target `H=8 s` y features hasta `8 s` atras | `8 s` |
| Target `H=16 s` y rolling de `60 s` | `60 s` |

En la practica, si los folds son dias completos, este embargo no cambia mucho
el volumen; sigue siendo parte del contrato para evitar errores al construir
ventanas.

## Que hacer con el random split

No hay que prohibir cualquier aleatoriedad. Hay que usarla para la pregunta
correcta.

### Random agrupado permitido como diagnostico

Puede construirse un experimento secundario:

- unidad de reparto: `condition_id`, nunca filas;
- mantener juntos ambos tokens y todos los frames de una condicion;
- si se quiere mayor seguridad, agrupar por dia + `condition_id`;
- comparar su resultado con walk-forward.

Interpretacion:

| Resultado | Lectura posible |
|---|---|
| Random agrupado alto, walk-forward bajo | Hay senal interpolable, pero no estable en tiempo; posible drift/overfit. |
| Ambos bajos | No hay senal suficiente con ese target/features/modelo. |
| Ambos razonablemente altos | Candidato serio; aun falta test terminal/economia. |
| Walk-forward alto y random similar | Resultado robusto, menos dependiente de mezcla de regimenes. |

Lo que no haremos:

- escoger modelo por el score random;
- reportar random como expectativa de trading;
- balancear artificialmente el test para que parezca mas facil.

## Se puede estratificar por clases?

En clasificacion es habitual intentar que `up/flat/down` tengan porcentajes
similares en todos los splits. Aqui hay una trampa:

- para train, se pueden usar pesos de clase o muestreo controlado;
- para validation/test, no debemos alterar la distribucion real;
- si el futuro tiene mas `flat`, el modelo debe ser evaluado contra ese futuro
  real.

La distribucion desigual no invalida el split: es parte de la dificultad.

## Como trataremos temporalidades y tokens

### Temporalidad

El corpus esta balanceado globalmente por `5m`, `15m` y `1h`, pero las
metricas deben reportarse:

- globalmente;
- por `temporality`;
- por dia o fold;
- por fase de ventana (`window_progress`).

Un modelo que solo funcione para `5m` puede ser util, pero no debe esconderse
dentro de una media global.

### Tokens complementarios

Para evaluar una decision economica:

- ambos tokens de una `condition_id` permanecen en el mismo split;
- reportaremos metricas por fila-token para diagnostico;
- tambien reportaremos por `condition_id + time_index_ns`, que representa la
  oportunidad de mercado.

## Protocolo concreto para los primeros experimentos

| Paso | Accion |
|---:|---|
| 1 | Congelar las fechas del test terminal; no mirar sus resultados durante seleccion. |
| 2 | Generar cada familia de labels solo desde observaciones, dentro de cada split o antes de separar siempre sin usar labels como features. |
| 3 | Definir la unidad de grupo `condition_id` y purgar cruces. |
| 4 | Construir features rolling solo con pasado; aplicar embargo segun mayor lookback/horizonte. |
| 5 | Comparar targets y bloques de features mediante los cuatro folds walk-forward. |
| 6 | Usar random agrupado solo como tabla de diagnostico adicional. |
| 7 | Elegir una configuracion antes de abrir el test terminal. |
| 8 | Ejecutar test una vez y acompanarlo de metricas economicas aproximadas. |

## Metricas que acompanaran al split

No basta una accuracy.

| Metrica | Por que importa |
|---|---|
| Distribucion real de labels por fold | Entender si cambia el problema. |
| Balanced accuracy / macro F1 para `up/flat/down` | No premiar siempre predecir la clase dominante. |
| Matriz de confusion | Saber si falla confundiendo movimiento con `flat`. |
| Correlacion o error de `delta_ticks` si hay regresion | Medir magnitud, no solo clase. |
| Precision de senales operables | Una accion rara pero buena puede ser mejor que acertar flats. |
| Markout neto con coste aproximado | Comprobar si la senal sobreviviria al spread/fee. |
| Metricas por temporality, dia y market-frame | Evitar promedios enganosos. |

## Decision actual

La decision no es "temporal o aleatorio" como opciones equivalentes:

| Papel | Split decidido |
|---|---|
| Seleccion principal durante investigacion | `expanding walk-forward` temporal, purgado y con embargo. |
| Evaluacion final | Test terminal `2026-05-23` a `2026-05-25`, abierto una sola vez. |
| Diagnostico adicional de estabilidad | Random agrupado por `condition_id`, claramente etiquetado como no desplegable. |
| Robustez si vemos drift fuerte | Rolling window temporal, posterior al baseline. |

Esta estrategia es mas honesta que hacer random por filas y mas informativa
que depender de un unico corte cronologico.

## Relacion con las otras guias

- Las features y sus ventanas estan en
  GUIA_01_FEATURES_Y_VARIABLES_SINTETICAS.md.
- La revision de los analisis ejecutados y la estrategia actual estan en
  SEGUIMIENTO_Y_ESTRATEGIA_ACTUAL.md.
- Los targets e implicaciones de costes estan en
  GUIA_04_LABELS_TARGETS_Y_COSTES.md.
