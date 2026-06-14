<!-- fuente original: LATENCY_V04_OPERATIVE_DECISION.md -->

# Latency v0.4 - Solucion para fijar L operativo

Fecha: `2026-06-01`

Estado: infraestructura implementada para medir latencia real y convertirla en
un `L_operativo`. Todavia no hay suficientes eventos reales.

## Respuesta corta

No, no tenemos clara la latencia real todavia.

Lo que si tenemos:

- inferencia local medida: p95 alrededor de `25 ms`;
- stress offline con escenarios `0s/2s/4s/8s`;
- labels L-aware ya construidos para `L=2s`;
- un primer baseline L2 que no cubre costes;
- ahora, una herramienta para registrar eventos reales y decidir `L_operativo`.

Lo que falta medir:

```text
input real -> features -> modelo -> decision -> orden -> firma -> envio
-> ack -> fill -> confirmacion
```

Especialmente faltan `firma`, `envio`, `ack` y `fill`.

## Que se ha anadido

### Logger runtime

Script:

```text
scripts/experiments/latency_v04_runtime_logger.py
```

Uso:

- se puede importar desde un runtime paper/dry-run/live;
- permite marcar cada etapa con timestamps UTC;
- escribe filas compatibles con `latency_v03_event_budget.py`;
- no guarda claves, firmas ni secretos.

Ejemplo conceptual de uso dentro de un runtime:

```python
from scripts.experiments.latency_v04_runtime_logger import LatencyEventLogger

logger = LatencyEventLogger(event_type="paper")
event = logger.new_event(market_id=market_id, token_id=token_id)

event.mark("data_seen")
event.mark("features_ready")

with event.measure("model_start", "model_done"):
    proba = model.predict_proba(row)

event.mark("decision_done")
event.mark("order_built")
event.mark("signed")
event.mark("sent")
event.mark("ack")
event.mark("fill")
event.mark("confirm")

logger.append(event)
```

Esto es solo instrumentacion. No es bot.

### Decision automatica de L

Script:

```text
scripts/experiments/latency_v04_operational_decision.py
```

Hace tres cosas:

1. lee `event_log.csv`;
2. calcula percentiles con el analizador v0.3;
3. recomienda `L_operativo` redondeando al grid de `2s`.

Regla:

```text
L_operativo = ceil(p95(total_seen_to_fill_ms) / grid)
```

Si no hay fills suficientes, puede usar `ack` como proxy provisional, pero no
como decision final.

## Estado real actual

Ejecutado contra el log real:

```text
python scripts\experiments\latency_v04_operational_decision.py
```

Resultado:

```text
status = insufficient_real_latency_data
reason = event_log_has_no_rows
gate = NO_GO_for_fixing_L_operativo
```

Lectura:

```text
Todavia no podemos fijar L_operativo.
```

Esto es correcto y honesto. El proyecto no debe fingir que `L=2s` ya esta
medido.

## Prueba tecnica con demo

Se ejecuto un demo local aislado:

```text
python scripts\experiments\latency_v04_runtime_logger.py ^
  --demo ^
  --events 5 ^
  --event-log data\experiments\latency_v04_runtime_logger_demo\event_log.csv

python scripts\experiments\latency_v04_operational_decision.py ^
  --event-log data\experiments\latency_v04_runtime_logger_demo\event_log.csv ^
  --analysis-out-dir data\experiments\latency_v04_runtime_logger_demo ^
  --out-dir data\experiments\latency_v04_runtime_logger_demo ^
  --min-rows 3
```

Resultado demo:

```text
status = ready
p95(total_seen_to_fill_ms) = 77,59 ms
recommended_l_seconds = 2,0
```

Importante:

```text
Este demo solo prueba que la maquinaria funciona. No mide Polymarket real.
```

## Como se soluciona de verdad

Necesitamos capturar eventos reales en tres niveles:

| Nivel | Que mide | Decision |
|---|---|---|
| `paper` | Datos + features + modelo + decision, sin orden real. | Sirve para aislar input/modelo. |
| `dry_run_order` | Construccion y firma de orden sin envio real, si la API lo permite. | Sirve para medir firma. |
| `live_tiny_or_safe` | Envio real controlado o entorno seguro, con ack/fill. | Solo esto cierra `L_operativo`. |

Minimos recomendados:

| Medicion | Minimo |
|---|---:|
| Paper events | `100` |
| Dry-run firma | `100` |
| Ack real | `100` |
| Fill real | `30-100`, segun disponibilidad |

Si solo hay `ack`, la decision sera provisional.
Si hay `fill`, la decision puede alimentar los labels.

## Gates

| Resultado medido | Decision |
|---|---|
| `p95 fill <= 2s` | Validar `L=2s`. |
| `2s < p95 fill <= 4s` | Validar `L=4s` o mejorar timing. |
| `p95 fill > 4s` | No forzar v0 taker; volver a timing/orderbook/maker. |
| Solo `ack`, sin `fill` | Caution: proxy, no contrato final. |
| Sin eventos | No go para fijar L. |

## Decision actual del proyecto

```text
NO tenemos L_operativo real cerrado.
SI tenemos la solucion para medirlo.
NO GO para bot.
GO para instrumentar paper/dry-run y recopilar eventos reales.
```

La parte importante para no perdernos:

```text
L=2s sigue siendo una sensibilidad offline, no una latencia real demostrada.
```

## Extension v0.5

Informe asociado:
[LATENCY_V05_POLYMARKET_SAFE_PROBE.md](LATENCY_V05_POLYMARKET_SAFE_PROBE.md)

Se anadio una prueba segura especifica de Polymarket:

- probe publico contra `https://clob.polymarket.com/time`;
- firma EIP-712 sintetica local;
- notebook `21_latency_end_to_end_probe.ipynb`;
- sin wallet real;
- sin orden real;
- sin fill.

Resultado orientativo:

| Medicion | p95 |
|---|---:|
| `total_seen_to_ack_ms` publico | `498,40 ms` |
| `order_build_to_signed_ms` dummy | `5,59 ms` |

Decision:

```text
L=2s es provisional si nos basamos solo en ack publico.
Para cerrar L_operativo falta total_seen_to_fill_ms real.
```


---

<!-- fuente original: LATENCY_V11_OPERATIVE_CLOSEOUT.md -->

# Latency v1.1 - Cierre operativo

Estado: `closed_for_next_modeling_phase`

## Decision

- Flujo de firma operativo: `signature_type=2`.
- Contrato de latencia para modelado: `L1s_operational_conservative`.
- `L=2s` queda como stress conservador, no como latencia live demostrada.
- No se espera confirmacion Polygon para decidir trading; se separa settlement de match CLOB.
- `NO GO` para bot: falta repetir muestras live y que la politica L-aware sea positiva tras costes.

## Evidencia principal

| Medida | Valor |
|---|---:|
| Live posts sigtype2 | `3` |
| Live fills sigtype2 | `1` |
| p95 `total_seen_to_ack_ms` sigtype2 | `429,674 ms` |
| p95 `sent_to_ack_ms` sigtype2 | `144,137 ms` |
| Fill matched `total_seen_to_fill_ms` | `426,737 ms` |
| Auth read p95 ack | `118,073 ms` |
| Public CLOB p95 ack | `498,397 ms` |
| Dry-create p95 sign, caliente | `7,042 ms` |

## Lectura sencilla

La decision ejecutable minima tardo unos `0,43s` hasta ack/match CLOB.
Redondeamos hacia arriba y usamos `1s` como contrato conservador de modelado.
Eso es suficiente para dejar cerrado que la latencia real no debe asumirse como
`2s` por defecto.

## Que significa para el proyecto

Para la siguiente fase:

```text
L_operativo_modelado = 1s
L_stress_conservador = 2s
```

Audit posterior: el cache H16 esta en malla de `2s`, asi que `L=1s` exacto no
tiene soporte (`0` filas). Por tanto:

```text
L_operativo_real = 1s conceptual
L_dataset_principal = 2s por resolucion de captura
```

Esto es conservador: en live esperamos llegar antes de la siguiente foto del
dataset, pero offline evaluamos contra la siguiente observacion disponible.

## Lo que no cierra

- No cierra riesgo de bot.
- No estima un p95 de produccion con muchas ejecuciones reales.
- No incluye una cadena completa con predictor live conectado al feed.
- No autoriza a optimizar infraestructura antes de validar senal L-aware.

## Siguiente fase

Pasar a diagnostico/modelado L-aware con:

```text
primario: L=1s
stress:   L=2s
```

El objetivo ya no es "medir si la latencia es de segundos", sino comprobar si
el edge offline sobrevive con una entrada retrasada realista.


---

