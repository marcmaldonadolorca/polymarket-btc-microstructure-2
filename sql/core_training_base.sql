-- EdgeRunner: base canonica para el primer predictor
--
-- Filosofia:
-- 1. solo sesiones core
-- 2. sin gaps severos del core
-- 3. labels ya canonicos
-- 4. tradability como filtro de suciedad real

SELECT
    s.id AS session_id,
    s.created_at_utc AS session_created_at_utc,
    s.dataset_tier,
    t.quality_score,
    t.coverage_ratio,
    t.continuity_ratio_1p5x,
    t.startup_latency_seconds,
    t.predictor_core_ready_ratio,
    t.core_gap_severe_segment_count,

    f.time_index_ns,
    f.time_index_utc,
    f.market_id,
    f.token_id,
    f.condition_id,
    f.temporality,
    f.market_kind,
    f.window_phase,
    f.polymarket_mid,
    f.external_mid,
    f.external_trade_imbalance,
    f.external_mid_return_bps_1s,
    f.chainlink_price,
    f.chainlink_binance_gap_bps,
    f.chainlink_perp_gap_bps,
    f.perp_mid,
    f.perp_mark_price,
    f.perp_last_funding_rate,
    f.perp_open_interest,
    f.perp_basis,
    f.perp_taker_buy_sell_ratio,

    l.label_name,
    l.microprice_label_name,
    l.economic_label_name,
    l.target_up_hit,
    l.target_down_hit,
    l.first_target_hit_side,
    l.prediction_horizon_steps,
    l.prediction_horizon_seconds,
    l.neutral_threshold_bps,
    l.execution_buffer_bps,

    m.question,
    m.slug,
    m.asset_symbol,

    mt.full_book_ratio,
    mt.degraded_ratio,
    mt.practical_dead_early,
    mt.tradability_status,

    mc.open_interest_rest,
    mc.open_interest_subgraph,
    mc.open_interest_abs_diff,
    mc.open_interest_rel_diff,
    mc.activity_total_count,
    mc.activity_split_count,
    mc.activity_merge_count,

    tc.outcome_label,
    tc.holder_count,
    tc.holder_total_amount,
    tc.holder_top_share,
    tc.holder_top3_share

FROM collection_sessions s
JOIN collection_session_telemetry t
  ON t.session_id = s.id
JOIN cross_venue_features f
  ON f.session_id = s.id
JOIN predictor_labels l
  ON l.session_id = f.session_id
 AND l.token_id = f.token_id
 AND l.time_index_ns = f.time_index_ns
LEFT JOIN market_metadata m
  ON m.session_id = f.session_id
 AND m.market_id = f.market_id
LEFT JOIN session_market_tradability mt
  ON mt.session_id = f.session_id
 AND mt.market_id = f.market_id
LEFT JOIN session_market_context mc
  ON mc.session_id = f.session_id
 AND mc.market_id = f.market_id
LEFT JOIN session_token_context tc
  ON tc.session_id = f.session_id
 AND tc.token_id = f.token_id
WHERE s.dataset_tier = 'core'
  AND t.coverage_ratio >= 0.97
  AND t.quality_score >= 70
  AND t.core_gap_severe_segment_count = 0;
