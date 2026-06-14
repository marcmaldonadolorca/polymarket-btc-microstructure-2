# Correccion de direccion del score adverse

Fecha: 2026-06-04

## Resumen

Durante el audit nested se detecto un error semantico en los experimentos
regularizados de orderbook.

El target:

```text
adverse_fill_proxy_0p25_H60 = 1
```

significa que la observacion pertenece al peor 25% de resultados de ejecucion.

Por tanto:

```text
P(adverse) alto = mayor riesgo = peor oportunidad
```

Sin embargo, `orderbook_regularized_sequence_v1` permitia usar directamente
`P(adverse)` como ranker descendente. Eso seleccionaba primero las observaciones
con mayor riesgo adverso estimado.

## Correccion

El score operativo correcto es:

```text
safe_adverse_score = 1 - P(adverse)
```

Tambien es correcto usar `P(adverse)` restando dentro de un score combinado.

## Artefactos afectados

Los siguientes resultados quedan superados para cualquier conclusion operativa
basada en scores raw `logit_adverse`:

- `docs/ORDERBOOK_REGULARIZED_SEQUENCE_V1.md`;
- `docs/SELECTION_STABILITY_AUDIT_V1.md`;
- `docs/SELECTION_STABILITY_NESTED_V2.md`.

Sus resultados siguen siendo utiles como trazabilidad del proceso, pero la
"candidata robusta" `obr_book_only_logit_adverse_c0p3` no es evidencia valida.

No quedan afectados:

- el audit descriptivo de features de orderbook;
- las regresiones Ridge;
- los scores `healthy`;
- los scores combinados que restan `P(adverse)`;
- el tensor de orderbook y su audit de calidad.

## Ejecucion corregida

Referencia:

```text
docs/SELECTION_STABILITY_NESTED_V3_CORRECTED.md
```

Decision:

```text
NO_GO_NESTED_V3_CORRECTED
```

Selector principal corregido:

```text
score=obr_book_only_logit_healthy_c0p3
cap=150
```

Resultado:

- inner pass cost0.50 minimo `1,00`;
- outer validation pass cost0.50 `0,90`;
- test pass cost0.50 `0,30`;
- test pass cost1.00 `0,10`;
- test worst day cost0.50 `-24,3385`.

Ademas:

- `0/32` configuraciones `safe_adverse` pasan el proxy robusto;
- solo `1/128` configuraciones corregidas pasa retrospectivamente;
- esa unica configuracion usa `book_plus_scores`;
- los scores upstream de esa familia son in-sample dentro de `train_initial`,
  por lo que queda como pista diagnostica, no validacion nested limpia.

## Conclusion

```text
no existe todavia una policy regularizada de orderbook seleccionable y robusta.
```

La siguiente fase no debe apoyarse en la antigua candidata adverse. Antes de
probar una fusion compleja hay que generar scores upstream out-of-fold o
reformular el target/universo de ejecucion.

## Cierre posterior de la pista OOF

La prueba recomendada ya fue ejecutada:

```text
docs/UPSTREAM_OOF_FUSION_STAGE2_V1.md
```

Decision:

```text
NO_GO_CLEAN_OOF_FUSION_STAGE2_COST05
```

Al generar scores upstream OOF limpios:

- `0` configuraciones resultan robustas;
- `0` configuraciones stage2 resultan robustas;
- la antigua pista `book_plus_scores` no sobrevive como policy seleccionable.

Queda una senal upstream modesta y dependiente de historial, que debe estudiarse
como madurez/regimen antes de aumentar arquitectura.
