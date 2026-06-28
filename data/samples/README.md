# data/samples/

Este directorio documenta la forma de los datos. **No** contiene el corpus
completo (la fuente SQLite del recolector, de gran tamaño, ni las cachés
Parquet). Para auditar las cifras finales se publica un ledger anonimizado en
`results/final_candidate_actions_anonymized.csv`.

El esquema de la unidad de entrenamiento está en
[`../../sql/core_training_base.sql`](../../sql/core_training_base.sql). La unidad
es `session_id + token_id + time_index_ns` sobre una malla de 2 segundos, con
features de Polymarket CLOB, Binance spot/perp y Chainlink.

El ledger elimina los identificadores originales mediante SHA-256 truncado y
conserva solo fecha, instante relativo, régimen y resultados en ticks para los
tres niveles de coste. Permite reproducir las tablas agregadas, pero no volver a
entrenar los modelos; esa limitación se declara en la memoria.
