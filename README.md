# Polymarket BTC — microestructura y predicción a corto plazo

> Trabajo Fin de Máster (Máster en Deep Learning, UPM, 2025/26).
> Autor: **Marc Maldonado Lorca**.
>
> Este repositorio acompaña a la memoria del TFM. Documenta una metodología
> completa de captura de datos, validación temporal y control de costes para
> mercados de BTC en Polymarket, y la **escalera de modelos** que va desde un
> baseline tabular hasta un especialista congelado con gate de volatilidad.

## Qué es y qué no es

Este es un trabajo **metodológico y académico**. No es un sistema de trading en
producción y **no afirma rentabilidad real**. Todos los resultados se reportan
en **ticks netos** (no en dólares ni ROI) y siempre acompañados de su soporte
(número de trades y días) y de su incertidumbre.

La validación fuera de muestra son **5 días** (6–10 de junio de 2026). Es un
soporte corto y se indica explícitamente en cada resultado. La honestidad del
protocolo —congelar antes de mirar, distinguir simulación de validación, admitir
los errores— es la tesis central, por encima de cualquier número concreto.

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
  -> vol gate (el régimen fresh cambió: 22% -> 52-71% de sesiones HIGH_VOL)
  -> validación fresh OOS, 5 días: +1.069 ticks/trade, 5/5 días positivos
  -> experimentos de adaptación al régimen: todos NO_GO (negativos valiosos)
  -> paper shadow ledger + gates cuantitativos hacia un posible bot
```

## Resultado del candidato final (congelado)

Especialista prestart H60 (HistGradientBoosting: regresor EV + clasificador de
sesión "healthy") con gate de volatilidad. Validación fresh out-of-sample sobre
datos no usados en la selección:

```text
n = 318 trades, 5 días (6-10 jun 2026), ~63.6 trades/día
Net@0.5  = +1.069 ticks/trade (referencia)
Sensibilidad al coste: @0.25 = +1.255   @0.5 = +1.069   @1.0 = +0.697 (estrés)
IC 90%   = [+0.472, +1.654]        P(neto > 0) = 99.8%
Días positivos = 5/5               Drawdown máx = 88.1 ticks
Sin vol gate: +0.349  ->  el gate aporta +0.72 ticks
Decisión = PROMISING_NEEDS_DATA (faltan días para PAPER_SHADOW_CANDIDATE)
```

La política está **congelada** y no se reentrena para "mejorar" estos números
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
data/samples/  Snapshots pequeños (nada pesado)
reports/       Memoria del TFM (fuente LaTeX + PDF)
```

## Reproducibilidad

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Los scripts de `scripts/experiments/` son los entrypoints del arco. Por diseño
esperan ejecutarse desde la raíz del repositorio (resuelven rutas relativas).
El conjunto de datos completo (~2.6M filas cross-venue del corpus principal y la
fuente SQLite del recolector) **no se publica** por tamaño y privacidad; este
repositorio incluye el esquema (`sql/`), snapshots pequeños (`data/samples/`) y
todo el código necesario para reconstruir el pipeline sobre datos equivalentes.

Documentos clave para entender el pipeline, en orden:
`docs/01_sistema_de_datos.md` -> `docs/04_splits_validacion_temporal.md` ->
`docs/07_stress_de_latencia.md` -> `docs/14_especialista_prestart_h60.md` ->
`docs/16_validacion_fresh.md` -> `docs/18_paper_shadow.md`.

## Estado y trabajo futuro

Investigación **pausada** a fecha de entrega. Líneas abiertas: ampliar el
soporte fresh a más días, validar el patrón en U de la volatilidad dentro del
gate (hipótesis, no resultado), y revisitar el deep learning con 60+ días de
datos.

## Licencia

MIT — ver [`LICENSE`](LICENSE).
