# Complex v1a - prestart H60 risk gate v3

Fecha: 2026-06-02

## Objetivo

La iteracion anterior encontro algo importante:

```text
el especialista prestart H60 tiene edge agregado, pero falla algunos dias.
```

El dia critico es:

```text
2026-05-24
```

Esta iteracion prueba una pregunta concreta:

```text
podemos evitar esos dias malos con una puerta de riesgo simple y observable?
```

Por "observable" queremos decir:

- no usamos el resultado futuro;
- no usamos labels para decidir si operar en test;
- no miramos el dia malo para elegir la regla final;
- seleccionamos las reglas en validacion y luego evaluamos en test terminal.

## Artefactos

Script:

```text
scripts/experiments/complex_v1a_prestart_h60_risk_gate_v3.py
```

Notebook:

```text
notebooks/17_complex_v1a_prestart_h60_risk_gate_v3.ipynb
```

Output:

```text
data/experiments/complex_v1a_prestart_h60_risk_gate_v3/
```

Ficheros:

| Fichero | Uso |
|---|---|
| `risk_gate_policy_results.csv` | Resultado de todas las puertas candidatas. |
| `risk_gate_selected_candidates.csv` | Puertas que pasan los criterios de validacion. |
| `risk_gate_day_breakdown.csv` | Desglose por dia de cada puerta. |
| `decision.json` | Decision resumida. |

Runtime:

```text
12 segundos
```

## Universo usado

Se mantiene el mismo universo conservador:

```text
strict_45_60_early
```

Es decir:

```text
45-60 segundos antes del inicio de ventana
```

Con filtros ya definidos:

- prestart;
- target H60 soportado;
- spread <= 2 ticks;
- coste visible <= 1.25 ticks;
- age <= 1000 ms;
- libro completo;
- no degradado.

## Que modelos se usan

No se entrena ningun modelo nuevo.

Se reutilizan los dos modelos del especialista v1:

| Modelo | Tipo | Uso |
|---|---|---|
| `full_features` | HistGradientBoosting EV + healthy classifier | Modelo principal con reloj/contexto completo. |
| `no_clock` | HistGradientBoosting EV + healthy classifier | Ablacion sin variables fuertes de reloj. |

Cada fila recibe cuatro scores:

```text
ev_pred_full
healthy_proba_full
ev_pred_noclock
healthy_proba_noclock
```

Luego se prueban puertas basadas en esos scores.

## Que puertas se probaron

Se probaron 188 policies.

Bloques:

| Bloque | Idea |
|---|---|
| `full_only` | Solo usar el modelo completo, con thresholds mas estrictos. |
| `noclock_only` | Solo usar el modelo sin reloj, para reducir dependencia temporal. |
| `consensus` | Exigir que los dos modelos esten de acuerdo. |
| `min_ensemble` | Usar el minimo entre los dos modelos como score conservador. |
| `ev_gap` | Exigir que los dos modelos no discrepen demasiado. |

Ejemplo:

```text
full_ev_ge_1_healthy_ge_0.55
```

significa:

```text
operar solo si el modelo full predice EV >= 1 tick
y healthy_proba >= 0.55
```

## Criterios de seleccion

Una puerta solo puede ser candidata si en validacion cumple:

- al menos 150 acciones;
- net positivo a coste 0.25;
- net positivo a coste 0.50;
- net positivo a coste 1.00;
- cero dias negativos en validacion a coste 0.50;
- al menos 2 dias de validacion con acciones suficientes.

Esto es intencionadamente duro.

La razon es sencilla:

```text
si queremos resolver inestabilidad diaria, no basta con que el promedio suba.
```

## Resultado principal

Decision:

```text
NO_GO_RISK_GATE
```

La mejor puerta seleccionada por validacion fue:

```text
full_ev_ge_1_healthy_ge_0.55
```

Resultados:

| Split | Acciones | Net 0.25 | Net 0.50 | Net 1.00 | Dias negativos cost 0.50 |
|---|---:|---:|---:|---:|---:|
| validation | 378 | +1.4777 | +1.2940 | +0.9267 | 0 |
| test | 647 | +0.6314 | +0.4413 | +0.0611 | 1 |

Lectura:

```text
la puerta mejora el promedio, pero no elimina el dia malo.
```

## Desglose del mejor filtro en test

| Dia | Acciones | Net 0.25 | Net 0.50 | Net 1.00 |
|---|---:|---:|---:|---:|
| 2026-05-23 | 217 | +1.9415 | +1.7470 | +1.3580 |
| 2026-05-24 | 208 | -0.5990 | -0.7942 | -1.1847 |
| 2026-05-25 | 222 | +0.5037 | +0.3227 | -0.0393 |

Esto confirma el diagnostico anterior:

```text
2026-05-24 no se arregla simplemente subiendo el threshold del modelo.
```

## Punto importante: habia filtros que arreglaban test?

Si.

Se encontraron 12 filtros que en test tienen:

- acciones suficientes;
- promedio positivo;
- cero dias negativos a coste 0.50.

Pero casi todos tienen un problema:

```text
fallan los criterios de validacion.
```

Ejemplo:

```text
min_ensemble_ev_0.5_h_0.6_evgap_le_2
```

En test parece bueno, pero en validacion no cumple la estabilidad exigida.

Por tanto no se puede aceptar como solucion robusta. Seria elegir una regla
porque sabemos que arregla justo el test, y eso seria leakage metodologico.

## Conclusion sencilla

Hasta ahora sabemos esto:

1. Hay edge en `prestart H60`.
2. El edge se concentra en `45-60s` antes del inicio.
3. El especialista supera a los baselines simples en agregado.
4. El problema no es solo predecir mejor.
5. El problema principal ahora es saber cuando no operar.
6. Una puerta simple de thresholds no basta.

Conclusion tecnica:

```text
NO GO bot.
NO GO encoders todavia como solucion principal.
GO a deteccion explicita de regimen / risk state.
```

## Siguiente estrategia recomendada

El siguiente paso no deberia ser meter directamente CNN, LSTM o Transformer.

Primero necesitamos una capa de estabilidad:

```text
regime_detector_v1
```

Objetivo:

```text
clasificar si una sesion/dia/bloque parece operable antes de mandar senales.
```

Posibles entradas:

- distribucion reciente de `ev_pred`;
- distribucion reciente de `healthy_proba`;
- desacuerdo entre modelo `full` y `no_clock`;
- volatilidad externa;
- momentum externo/perp;
- microprice gap;
- spread/coste visible;
- calidad de libro;
- numero de oportunidades por sesion;
- estabilidad por market/time bucket.

La decision operativa futura deberia tener dos capas:

```text
1. risk gate: puedo operar este regimen?
2. signal model: que filas concretas tienen edge?
```

Solo si ambas capas dicen que si, tendria sentido enviar orden.

