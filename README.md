# Polymarket BTC — microestructura y predicción a corto plazo

> Trabajo Fin de Máster (Máster en Deep Learning, UPM, 2025/26).
> Autor: **Marc Maldonado Lorca**.
>
> Este repositorio acompaña a la memoria del TFM. Documenta una metodología
> completa de captura de datos, validación temporal y control de costes para
> mercados de BTC en Polymarket, y la **escalera de modelos** que va desde un
> baseline tabular hasta un especialista congelado y un diagnóstico posterior
> de cambio de régimen.

## Qué es y qué no es

Este es un trabajo **metodológico y académico**. No es un sistema de trading en
producción y **no afirma rentabilidad real**. Todos los resultados se reportan
en **ticks netos** (no en dólares ni ROI) y siempre acompañados de su soporte
(número de trades y días) y de su incertidumbre.

La validación fuera de muestra son **5 días** (6–10 de junio de 2026). El
especialista base estaba congelado antes de ese bloque; el filtro de volatilidad
se propuso después de observar el cambio de régimen y se reporta como análisis
post-hoc. Distinguir ambas evidencias es parte central del trabajo.

## La lección central

```text
Un baseline tabular acierta ~90% la dirección a 16 s... y aun así pierde dinero.
Con una latencia de entrada realista de 2 s, el neto en el conjunto terminal
cae a -1.48 ticks. Acertar la dirección no es ganar: el coste y la latencia
deciden. Todo el proyecto se reorganiza alrededor de esta lección.
```

## La escalera de modelos (el arco del proyecto)

El relato sigue una progresión incremental de modelos, exigiendo en cada paso que
la complejidad añadida se justifique con evidencia
(baseline → lineal → features → modelo profundo → modelo final):

```text
baseline tabular H16 (90% dir., parece excelente)
  -> muere con latencia realista (-1.48 ticks con 2 s)          [LECCIÓN CENTRAL]
  -> modelos latency-aware (AUC 0.79, política económica negativa)
  -> secuencias + orderbook (GRU, Conv1D, fusión, TCN): representación sí, política no
  -> corrección metodológica del score adverse                  [RIGOR]
  -> protocolo OOF estricto + selección de horizonte (H60 vive, H120/H240 colapsan)
  -> especialista prestart H60 CONGELADO antes de ver datos fresh
  -> OOS limpio, 5 días: +0.349 ticks/acción @0.5, 3/5 días positivos
  -> diagnóstico post-hoc: cambio de régimen y filtro de volatilidad
  -> subconjunto diagnóstico: +1.069 ticks/acción, 5/5 días (no confirmatorio)
  -> experimentos de adaptación al régimen: todos NO_GO (negativos valiosos)
  -> replay de sombra offline + diseño de validación prospectiva
```

## Resultado fuera de muestra y diagnóstico posterior

El especialista prestart H60 (HistGradientBoosting: regresor EV + clasificador
de sesión "healthy") se evaluó primero sin ningún filtro decidido sobre el
bloque nuevo:

```text
OOS limpio: n = 754 unidades, 5 días (6-10 jun 2026)
Net@0.5 = +0.349 ticks/unidad; 3/5 días positivos
Sensibilidad: @0.25 = +0.537   @0.5 = +0.349   @1.0 = -0.026

Diagnóstico post-hoc de baja volatilidad: n = 318
Net@0.5 = +1.069; 5/5 días positivos
IC90 agrupado por mercado = [+0.193, +2.004]
Estado = HIPÓTESIS CONGELADA PENDIENTE DE VALIDACIÓN PROSPECTIVA
```

La política completa queda congelada **después** del diagnóstico y no se
reentrena para mejorar estos números. Solo fechas posteriores pueden validarla
(ver [`config/frozen_policy_thresholds.yaml`](config/frozen_policy_thresholds.yaml)).


## Estructura del repositorio

```text
config/        Umbrales congelados de la política + configs del arco
docs/          18 documentos canónicos del arco (01..18), renombrados
notebooks/     8 notebooks pedagógicos ejecutados (EDA, splits, target, baseline,
               latencia, secuencias, corrección adverse, especialista)
scripts/       Entrypoints reproducibles (scripts/experiments/)
src/edgerunner/ Paquete con la lógica compartida (data, features, models, eval)
results/       key_results.csv y decision_register.csv filtrados al arco
sql/           Esquema de la consulta base de entrenamiento
data/samples/  Contrato y documentación de los datos publicados
reports/       Memoria canónica (`memoria.pdf`) y fuente LaTeX oficial
```

## Reproducibilidad

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Los scripts de `scripts/experiments/` son los entrypoints del arco. Por diseño
esperan ejecutarse desde la raíz del repositorio (resuelven rutas relativas).
El conjunto completo (~2.6M filas cross-venue y la fuente SQLite) **no se
publica** por tamaño. El repositorio incluye el esquema (`sql/`), el código y un
ledger anonimizado de 754 unidades en
`results/final_candidate_actions_anonymized.csv`, suficiente para recalcular las
tablas finales y la incertidumbre agrupada. El entrenamiento completo no es
reproducible sin el corpus privado y se declara como limitación.

Documentos clave para entender el pipeline, en orden:
`docs/01_sistema_de_datos.md` -> `docs/04_splits_validacion_temporal.md` ->
`docs/07_stress_de_latencia.md` -> `docs/14_especialista_prestart_h60.md` ->
`docs/16_validacion_fresh.md` -> `docs/18_paper_shadow.md`.

## Estado y trabajo futuro

Investigación **pausada** a fecha de entrega. Líneas abiertas: validar
prospectivamente la política completa sin retocar umbrales, definir exposición
máxima por mercado, comprobar el patrón en U (hipótesis) y revisitar el deep
learning con 60+ días de datos.

## Licencia

MIT — ver [`LICENSE`](LICENSE).
