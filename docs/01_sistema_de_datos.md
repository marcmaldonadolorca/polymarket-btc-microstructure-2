# Data Model

## Fuente de verdad

- SQLite principal: `D:\polymarket_btc_probe_official_v1\polymarket_week.sqlite3`

## Tablas clave

### 1. `collection_sessions`

Rol:

- catalogo de sesiones importadas
- puerta de entrada para `dataset_tier`

Campos clave:

- `id`
- `created_at_utc`
- `run_purpose`
- `dataset_tier`
- `cross_venue_feature_rows_total`
- `source_output_dir`

### 2. `collection_session_telemetry`

Rol:

- calidad por sesion

Campos clave:

- `session_id`
- `quality_score`
- `coverage_ratio`
- `continuity_ratio_1p5x`
- `startup_latency_seconds`
- `predictor_core_ready_ratio`
- `core_gap_severe_segment_count`
- `session_continuity_status`
- `core_polymarket_health`
- `core_binance_spot_health`
- `core_binance_perp_health`
- `core_chainlink_health`

### 3. `cross_venue_features`

Rol:

- tabla principal de observaciones
- `94` columnas

Campos clave de join:

- `session_id`
- `time_index_ns`
- `time_index_utc`
- `market_id`
- `token_id`
- `condition_id`
- `temporality`

Familias de features visibles:

- precio Polymarket
  - `polymarket_mid`
- contexto externo
  - `external_mid`
  - `perp_mid`
  - `chainlink_price`
- gaps y basis
  - `chainlink_binance_gap_bps`
  - `chainlink_perp_gap_bps`
  - `spot_perp_mid_gap_bps`
- actividad/trade
  - `external_trade_imbalance`
  - `external_mid_return_bps_1s`
- metadata de ventana
  - `temporality`
  - `market_kind`
  - `window_phase`

### 4. `predictor_labels`

Rol:

- target canonico del predictor

Join clave:

- `session_id`
- `token_id`
- `time_index_ns`

Campos clave:

- `label_name`
- `microprice_label_name`
- `economic_label_name`
- `target_up_hit`
- `target_down_hit`
- `first_target_hit_side`
- `prediction_horizon_steps`
- `prediction_horizon_seconds`
- `neutral_threshold_bps`
- `execution_buffer_bps`

### 5. `market_metadata`

Rol:

- metadata estructural del mercado

Join recomendado:

- `session_id + market_id`

Campos utiles:

- `temporality`
- `question`
- `slug`
- `asset_symbol`
- `market_kind`
- `window_start_utc`
- `window_end_utc`
- `token_ids_json`

### 6. `session_market_tradability`

Rol:

- filtro de operabilidad real por mercado/sesion

Join recomendado:

- `session_id + market_id`

Campos clave:

- `full_book_ratio`
- `degraded_ratio`
- `practical_dead_early`
- `tradability_status`
- `transition_from_count`
- `transition_to_count`

### 7. `session_market_context`

Rol:

- contexto resumido por mercado

Join recomendado:

- `session_id + market_id`

Campos utiles:

- `open_interest_rest`
- `open_interest_subgraph`
- `open_interest_abs_diff`
- `open_interest_rel_diff`
- `activity_total_count`
- `activity_split_count`
- `activity_merge_count`
- `activity_redemption_count`
- holders agregados

### 8. `session_token_context`

Rol:

- contexto resumido por token

Join recomendado:

- `session_id + token_id`

Campos utiles:

- `outcome_label`
- `holder_count`
- `holder_total_amount`
- `holder_top_share`
- `holder_top3_share`
- `open_interest_rest`
- `open_interest_subgraph`
- `order_filled_count`
- `orders_matched_count`
- `total_net_balance`

### 9. `predictor_sidecar_availability`

Rol:

- mascara de sidecars opcionales

Join recomendado:

- por `session_id`

Campos utiles:

- `sidecar_name`
- `integration_mode`
- `present_mask`
- `comparison_ready`
- `available_for_training`
- `availability_status`

## Join canonico base

1. `collection_sessions s`
2. `collection_session_telemetry t on t.session_id = s.id`
3. `cross_venue_features f on f.session_id = s.id`
4. `predictor_labels l on l.session_id = f.session_id and l.token_id = f.token_id and l.time_index_ns = f.time_index_ns`
5. left joins de contexto:
   - `market_metadata m on m.session_id = f.session_id and m.market_id = f.market_id`
   - `session_market_tradability mt on mt.session_id = f.session_id and mt.market_id = f.market_id`
   - `session_market_context mc on mc.session_id = f.session_id and mc.market_id = f.market_id`
   - `session_token_context tc on tc.session_id = f.session_id and tc.token_id = f.token_id`

## Regla de consumo

Primer predictor:

- `core` only
- filtros de calidad por sesion
- filtros de tradability por mercado
- sidecars opcionales fuera salvo mascaras

---

# Official Corpus Snapshot

> Nota de estado (`2026-05-27`): este documento es una fotografia del corpus,
> no el contrato final de modelado. Tras la auditoria, `predictor_labels` se
> usa como benchmark provisional; los nuevos targets se calcularan desde
> observaciones. Consultar `GUIA_04_LABELS_TARGETS_Y_COSTES.md`.

Fecha de fotografia: `2026-05-26`

## Ubicacion canonica

- root oficial: `D:\polymarket_btc_probe_official_v1`
- base principal: `D:\polymarket_btc_probe_official_v1\polymarket_week.sqlite3`
- monitor snapshot: `D:\polymarket_btc_probe_official_v1\monitor\collection_status.json`

## Resumen ejecutivo

El corpus ya esta en estado fuerte para modelado inicial serio.

- ventana temporal real: `2026-05-11T10:00:36.756420Z` -> `2026-05-25T22:41:34.343307Z`
- duracion cubierta: `348.68 h` (`14.53` dias)
- sesiones importadas: `2604`
- sesiones `core`: `2590`
- sesiones `non_core`: `14`
- proporcion `core`: `99.46%`
- tamano DB: `74,039,296,000` bytes (`68.95 GiB`)
- disco libre al cierre de la foto: `1767.44 GB`

Lectura operativa:

- el corpus esta muy limpio
- la mayor parte del volumen ya pertenece a `core`
- no hace falta reconstruirlo para empezar a entrenar

## Volumen fisico

Tablas principales confirmadas:

| tabla | filas |
| --- | ---: |
| `collection_sessions` | `2604` |
| `market_metadata` | `15624` |
| `polymarket_grid_rows` | `4999680` |
| `external_grid_rows` | `1249920` |
| `cross_venue_rows` | `2499840` |
| `cross_venue_features` | `2499840` |
| `predictor_labels` | `2433124` |
| `predictor_sidecar_availability` | `15624` |
| `session_token_context` | `16448` |
| `session_market_context` | `15624` |
| `session_market_tradability` | `15624` |
| `session_continuity_context` | `20832` |
| `session_gap_segments` | `12` |
| `session_secondary_anchor_context` | `2604` |
| `session_external_regime_context` | `2604` |

Ratios utiles:

- `cross_venue_features` por sesion: `960.0`
- `predictor_labels` por sesion: `934.38`
- sesiones por dia observadas: `179.23`

Cobertura por `temporality` en `cross_venue_features`:

- `5m`: `833280`
- `15m`: `833280`
- `1h`: `833280`

Lectura:

- el corpus esta perfectamente balanceado por `5m / 15m / 1h`
- el throughput es alto y estable

## Calidad del corpus

Resumen global:

- `avg_quality = 99.775`
- `min_quality = 89.0`
- `max_quality = 100.0`
- `avg_coverage = 1.0`
- `avg_continuity = 1.0`
- `avg_startup = 9.737 s`
- `min_startup = 7.308 s`
- `max_startup = 103.813 s`
- `avg_predictor_core_ready = 0.999959`

Por tier:

| tier | sesiones | calidad media | startup medio |
| --- | ---: | ---: | ---: |
| `core` | `2590` | `99.815` | `9.682 s` |
| `non_core` | `14` | `92.357` | `19.907 s` |

Gap summary:

- `session_gap_segments.total_rows = 12`
- `small_rows = 8`
- `burst_rows = 4`
- `severe_rows = 0`

Lectura:

- no hay gaps severos del core
- los `non_core` son pocos y no parecen corrupcion del corpus, sino sesiones algo peores pero aun utiles

## Distribucion por dia

| dia | sesiones | core | non_core |
| --- | ---: | ---: | ---: |
| `2026-05-11` | `105` | `104` | `1` |
| `2026-05-12` | `204` | `203` | `1` |
| `2026-05-13` | `205` | `203` | `2` |
| `2026-05-14` | `130` | `127` | `3` |
| `2026-05-15` | `198` | `198` | `0` |
| `2026-05-16` | `200` | `200` | `0` |
| `2026-05-17` | `193` | `191` | `2` |
| `2026-05-18` | `186` | `186` | `0` |
| `2026-05-19` | `180` | `178` | `2` |
| `2026-05-20` | `175` | `175` | `0` |
| `2026-05-21` | `174` | `174` | `0` |
| `2026-05-22` | `172` | `171` | `1` |
| `2026-05-23` | `166` | `165` | `1` |
| `2026-05-24` | `165` | `165` | `0` |
| `2026-05-25` | `151` | `150` | `1` |

## Contrato minimo para predictor

Tablas que ya merecen uso directo:

- `collection_sessions`
- `collection_session_telemetry`
- `cross_venue_features`
- `predictor_labels`
- `market_metadata`
- `session_market_tradability`
- `session_market_context`
- `session_token_context`
- `session_secondary_anchor_context`
- `session_external_regime_context`
- `predictor_sidecar_availability`

Joins canonicos:

- `collection_sessions.id = collection_session_telemetry.session_id`
- `cross_venue_features.session_id = collection_sessions.id`
- `predictor_labels.session_id + token_id + time_index_ns` contra `cross_venue_features.session_id + token_id + time_index_ns`
- `session_market_tradability.session_id + market_id`
- `session_market_context.session_id + market_id`
- `session_token_context.session_id + token_id`

## Labels disponibles

En `predictor_labels` ya hay labels canonicos de predictor, no solo preview.

Muestra real:

- `prediction_horizon_steps = 4`
- `prediction_horizon_seconds = 8.0`
- `neutral_threshold_bps = 5.0`
- `execution_buffer_bps = 2.0`
- `label_name`
- `microprice_label_name`
- `economic_label_name`
- `target_up_hit`
- `target_down_hit`
- `first_target_hit_side`

Primera fila observada:

- `temporality = 5m`
- `label_name = flat`
- `microprice_label_name = flat`
- `economic_label_name = flat`
- `first_target_hit_side = none`

## Sidecars opcionales

Estado actual de `predictor_sidecar_availability`:

| sidecar | status | presente | trainable |
| --- | --- | ---: | ---: |
| `liquidation_aggregates` | `sparse` | `2604` | `0` |
| `binance_spot_depth` | `absent` | `0` | `0` |
| `binance_perp_depth` | `absent` | `0` | `0` |
| `coinbase_exchange_depth` | `absent` | `0` | `0` |
| `coinbase_spot_anchor` | `absent` | `0` | `0` |
| `deribit_volatility` | `absent` | `0` | `0` |

Lectura importante:

- el contrato missing-aware esta bien
- pero en este snapshot los sidecars opcionales no aportan todavia filas `available_for_training = 1`
- el primer predictor debe apoyarse en core + labels + contexto nativo/tradability

## Tradability

`session_market_tradability` ya esta poblada y debe tratarse como filtro, no solo como auditoria.

Columnas utiles:

- `any_book_ratio`
- `full_book_ratio`
- `degraded_ratio`
- `practical_dead_early`
- `tradability_status`
- `transition_from_count`
- `transition_to_count`

Lectura:

- muchos mercados seleccionados acaban etiquetados como `practical_dead_early` o `never_operable`
- eso no invalida el corpus entero
- si vas a entrenar predictor por fila, conviene usar esta tabla para excluir o marcar ventanas degradadas

## Limitaciones utiles de saber

1. `status.py --root` sobre esta raiz ya puede tardar bastante.
   - para lectura rapida es mejor `monitor\collection_status.json`
   - para analisis pesado, mejor SQL directo

2. `liquidation_aggregates` existe como sidecar, pero hoy no esta en modo realmente trainable.

3. `binance depth`, `coinbase` y `deribit` no deben asumirse como disponibles en el primer predictor.

4. El corpus serio por defecto sigue siendo `dataset_tier = core`.

## Como lo veo

Lo veo bien.

- ya hay suficiente volumen para entrenar serio
- la calidad media es muy alta
- la estructura esta bastante limpia
- el cuello de botella ya no es capturar mas para “tener algo”, sino decidir bien el corte de entrenamiento y el plan de muestreo

Mi recomendacion:

- arrancar primero con `core` puro
- usar `predictor_labels` como target canónico
- unir `cross_venue_features + market_metadata + tradability + contextos nativos`
- dejar sidecars opcionales fuera del primer modelo salvo como mascaras o analisis auxiliar

---

