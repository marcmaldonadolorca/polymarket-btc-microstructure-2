# Upstream OOF fusion + stage2 v1

Fecha: 2026-06-04

## Objetivo

Comprobar limpiamente la pista `book_plus_scores` sin reutilizar scores
in-sample.

## Protocolo causal

- upstream `fusion_concat_gru` entrenado solo con dias pasados;
- una epoca fija, elegida antes de esta prueba;
- se eliminan ocho features auxiliares `ev_pred/healthy_proba` de la entrada;
- cada bloque recibe predicciones upstream out-of-fold;
- como la calibracion cambia entre folds, el universo se selecciona por ranking
  con cobertura fija predeclarada, aproximadamente igual a la cobertura
  historica:

```text
top 35% diario por upstream_oof_proba
```

- stage2 se entrena solo con acciones OOF de bloques anteriores;
- seleccion con `inner_2`, `inner_3` y `outer_validation`;
- test terminal se abre al final.

## Decision

```text
NO_GO_CLEAN_OOF_FUSION_STAGE2_COST05
```

The validation-selected clean OOF policy does not close the test cost0.50 robustness proxy.

## Bloques upstream OOF

| evaluation_block | train_start | train_end | eval_start | eval_end | train_rows | eval_rows | selected_actions | selected_rate | tab_features_used | tab_features_blocked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner_1 | 2026-05-11 | 2026-05-14 | 2026-05-15 | 2026-05-16 | 1450 | 912 | 320 | 0.3509 | 31 | 8 |
| inner_2 | 2026-05-11 | 2026-05-16 | 2026-05-17 | 2026-05-18 | 2362 | 833 | 292 | 0.3505 | 31 | 8 |
| inner_3 | 2026-05-11 | 2026-05-18 | 2026-05-19 | 2026-05-20 | 3195 | 814 | 286 | 0.3514 | 31 | 8 |
| outer_validation | 2026-05-11 | 2026-05-20 | 2026-05-21 | 2026-05-22 | 4009 | 756 | 265 | 0.3505 | 31 | 8 |
| test_terminal | 2026-05-11 | 2026-05-22 | 2026-05-23 | 2026-05-25 | 4765 | 1066 | 374 | 0.3508 | 31 | 8 |

## Calidad del upstream OOF

| evaluation_block | rows | selected_actions | auc_positive_cost0p5 | spearman_pred_cost0p5 | all_mean_cost0p5 | selected_mean_cost0p5 | selected_mean_cost1p0 | selected_healthy_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inner_1 | 912 | 320 | 0.5154 | 0.0691 | -0.8954 | -0.4846 | -0.8614 | 0.4688 |
| inner_2 | 833 | 292 | 0.5035 | 0.0948 | -0.9066 | -0.7881 | -1.1858 | 0.4692 |
| inner_3 | 814 | 286 | 0.5720 | 0.1481 | -0.8837 | 0.2387 | -0.1310 | 0.5175 |
| outer_validation | 756 | 265 | 0.5577 | 0.1473 | -0.8791 | 0.0934 | -0.2680 | 0.5208 |
| test_terminal | 1066 | 374 | 0.5656 | 0.1386 | -0.8828 | 0.1778 | -0.1899 | 0.5080 |

## Config seleccionada

```text
score=clean_stage2_ridge_a100p0
cap=50
```

## Politicas

| score_col | daily_cap | selection_eligible | selection_pass_cost0p5_min | selection_pass_cost0p5_mean | selection_worst_day_cost0p5_min | test_pass_cost0p5_rate | test_pass_cost1p0_rate | test_sum_cost0p5_mean | test_worst_day_cost0p5_mean | test_robust_cost0p5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_stage2_ridge_a100p0 | 50 | False | 0.1000 | 0.3667 | -51.3717 | 0.4000 | 0.2000 | 17.3490 | -16.0244 | False |
| upstream_oof_pred | 75 | False | 0.1000 | 0.4000 | -65.1229 | 0.4000 | 0.2000 | 160.2113 | -7.9065 | False |
| upstream_oof_combo | 100 | False | 0.1000 | 0.3333 | -63.3243 | 0.1000 | 0.0000 | 28.7140 | -53.7249 | False |
| clean_stage2_combo_a10p0_c0p3 | 50 | False | 0.1000 | 0.2667 | -44.1752 | 0.0000 | 0.0000 | -9.2220 | -47.3686 | False |
| upstream_oof_pred | 50 | False | 0.1000 | 0.2667 | -45.1660 | 0.4000 | 0.3000 | 94.5862 | -6.7342 | False |
| clean_stage2_ridge_a100p0 | 75 | False | 0.1000 | 0.1000 | -68.8706 | 0.1000 | 0.1000 | 72.7635 | -29.0642 | False |
| clean_stage2_healthy_c0p1 | 150 | False | 0.1000 | 0.1667 | -114.1268 | 0.0000 | 0.0000 | 59.2256 | -64.3366 | False |
| upstream_oof_proba | 75 | False | 0.0000 | 0.4333 | -70.9539 | 0.4000 | 0.2000 | 97.4156 | -28.9518 | False |
| upstream_oof_combo | 50 | False | 0.0000 | 0.3667 | -50.3704 | 0.5000 | 0.4000 | 142.0253 | 1.0546 | False |
| clean_stage2_combo_a100p0_c0p1 | 50 | False | 0.0000 | 0.3000 | -42.4345 | 0.2000 | 0.2000 | 41.5905 | -23.7387 | False |
| clean_stage2_combo_a10p0_c0p1 | 50 | False | 0.0000 | 0.2667 | -47.2886 | 0.2000 | 0.2000 | -16.7128 | -56.1869 | False |
| upstream_oof_pred | 100 | False | 0.0000 | 0.3667 | -86.1535 | 0.2000 | 0.1000 | 39.0241 | -52.2535 | False |
| clean_stage2_healthy_c0p3 | 50 | False | 0.0000 | 0.3000 | -63.3186 | 0.0000 | 0.0000 | 0.0726 | -32.0075 | False |
| upstream_oof_proba | 50 | False | 0.0000 | 0.2333 | -40.4765 | 0.4000 | 0.2000 | 68.1569 | -10.7763 | False |
| upstream_oof_combo | 75 | False | 0.0000 | 0.3000 | -71.7566 | 0.1000 | 0.0000 | 142.8600 | -28.5315 | False |
| clean_stage2_combo_a10p0_c0p3 | 100 | False | 0.0000 | 0.2667 | -57.2283 | 0.4000 | 0.1000 | 109.1219 | -14.2825 | False |
| clean_stage2_combo_a100p0_c0p3 | 50 | False | 0.0000 | 0.3000 | -71.5925 | 0.2000 | 0.1000 | 35.3958 | -30.5314 | False |
| clean_stage2_healthy_c0p1 | 50 | False | 0.0000 | 0.2667 | -65.0832 | 0.2000 | 0.1000 | 45.9598 | -23.2678 | False |
| clean_stage2_ridge_a10p0 | 50 | False | 0.0000 | 0.2333 | -73.0506 | 0.0000 | 0.0000 | -2.5953 | -41.4987 | False |
| clean_stage2_healthy_c0p1 | 100 | False | 0.0000 | 0.2667 | -82.5098 | 0.1000 | 0.0000 | 19.4914 | -50.6299 | False |
| clean_stage2_combo_a100p0_c0p1 | 100 | False | 0.0000 | 0.3000 | -96.1207 | 0.0000 | 0.0000 | 81.5497 | -50.6242 | False |
| upstream_oof_proba | 150 | False | 0.0000 | 0.2667 | -85.4725 | 0.0000 | 0.0000 | 57.2332 | -59.3684 | False |
| clean_stage2_healthy_c0p3 | 100 | False | 0.0000 | 0.2333 | -77.0383 | 0.1000 | 0.0000 | 56.3894 | -34.0610 | False |
| clean_stage2_ridge_a10p0 | 150 | False | 0.0000 | 0.2667 | -87.9680 | 0.0000 | 0.0000 | 24.0417 | -51.4584 | False |
| clean_stage2_healthy_c0p3 | 75 | False | 0.0000 | 0.2333 | -78.5628 | 0.2000 | 0.0000 | 109.4826 | -44.8971 | False |
| upstream_oof_proba | 100 | False | 0.0000 | 0.2333 | -80.5647 | 0.1000 | 0.0000 | 71.0474 | -68.8903 | False |
| clean_stage2_ridge_a100p0 | 150 | False | 0.0000 | 0.2333 | -82.5251 | 0.0000 | 0.0000 | -19.9149 | -93.2353 | False |
| clean_stage2_combo_a100p0_c0p3 | 75 | False | 0.0000 | 0.1667 | -60.8676 | 0.1000 | 0.0000 | 65.0492 | -39.0711 | False |
| upstream_oof_pred | 150 | False | 0.0000 | 0.1667 | -70.9472 | 0.1000 | 0.0000 | 15.0312 | -79.6660 | False |
| clean_stage2_combo_a10p0_c0p3 | 75 | False | 0.0000 | 0.1333 | -63.3280 | 0.0000 | 0.0000 | 54.9197 | -49.3135 | False |

## Lectura sencilla

- Este experimento elimina el principal caveat de `book_plus_scores`.
- Si stage2 pasa, existe una pista causal para fusion/risk.
- Si no pasa, no debemos promover la fusion regularizada actual a modelo
  complejo solo porque alguna configuracion retrospectiva pareciera buena.
- El upstream OOF empieza debil en `inner_1/inner_2`, pero pasa a media
  seleccionada cost0.50 positiva desde `inner_3` y se mantiene positiva en
  validation/test.
- Esa mejora tardia sugiere estudiar cantidad minima de historial y regimen
  temporal antes de aumentar arquitectura.

## Siguiente paso recomendado

```text
upstream_oof_maturity_regime_audit_v1
```

Objetivo: comprobar si el ranking solo se vuelve util tras acumular suficiente
historial, y si una ventana expanding/rolling puede seleccionarse sin mirar
test. No entrenar aun CNN/Transformer mayor.

## Cierre posterior

Los audits recomendados ya fueron ejecutados:

```text
docs/UPSTREAM_OOF_MATURITY_REGIME_AUDIT_V1.md
docs/UPSTREAM_OOF_WARMUP_GATE_AUDIT_V1.md
```

Resultado:

- ninguna estrategia expanding/rolling produce una policy seleccionable y
  robusta;
- validation selecciona un warm-up de `8` dias;
- ese warm-up falla en test con worst day cost0.50 `-28,9518`.

Conclusion:

```text
cerrar ajustes de ventana/cap/stage2 sobre H60 actual.
Siguiente hipotesis: H120 limpio OOF o logs reales de fill.
```
