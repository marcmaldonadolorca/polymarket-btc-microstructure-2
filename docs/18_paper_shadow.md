# Registro de sombra del especialista H60

## Estado correcto

```text
OFFLINE_REPLAY_COMPLETE
PROSPECTIVE_VALIDATION_PENDING
```

El registro disponible reconstruye retrospectivamente la política de baja
volatilidad sobre el bloque del 6 al 10 de junio. No es una ejecución en vivo ni
una validación prospectiva, porque el filtro se adoptó tras inspeccionar ese mismo
bloque.

## Regla congelada para fechas posteriores

```text
Vol gate:  perp_realized_vol_bps_5s <= 0.6657
EV model:  ev_pred > 0.75
HP model:  hp_pred >= 0.50
Window:    strict_45_60_early
Freeze:    después de 2026-06-10
```

## Resumen retrospectivo

| Métrica | Valor |
|---|---:|
| Unidades de acción | 318 |
| Mercados | 156 |
| Máximo de acciones solapadas por mercado | 8 |
| Neto medio @0.25 | +1.258 ticks |
| Neto medio @0.5 | +1.069 ticks |
| Neto medio @1.0 | +0.691 ticks |
| Días positivos | 5/5 |
| IC90 agrupado por mercado | [+0.193, +2.004] |
| Drawdown diagnóstico | 88.1 ticks |

El drawdown y la suma acumulada corresponden a unidades de acción con tamaño
unitario. No incorporan capital, fills, ni una restricción de exposición
simultánea; no son PnL de cartera.

## Puertas prospectivas

Los cinco días retrospectivos no cuentan para estas puertas:

| Puerta | Requisito mínimo |
|---|---|
| Seguimiento ampliado | >0.5 ticks, 200 acciones, 8 días posteriores |
| Candidato a despliegue | >0.5 ticks, 400 acciones, 12 días posteriores, fills y exposición validados |

Soporte prospectivo actual de la política completa: **0 días**.
