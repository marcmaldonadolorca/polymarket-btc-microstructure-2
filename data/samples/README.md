# data/samples/

Snapshots pequeños para inspección y pruebas. **No** contiene el corpus completo
(la fuente SQLite del recolector, ~93 GB, y las cachés en Parquet no se publican
por tamaño y privacidad).

El esquema de la unidad de entrenamiento está en
[`../../sql/core_training_base.sql`](../../sql/core_training_base.sql). La unidad
es `session_id + token_id + time_index_ns` sobre una malla de 2 segundos, con
features de Polymarket CLOB, Binance spot/perp y Chainlink.

> Pendiente: añadir un snapshot anonimizado de pocas filas como ejemplo de forma
> del dataset. No se incluye dato pesado ni ninguna cifra en dólares.
