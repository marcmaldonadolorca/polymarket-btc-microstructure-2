from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import selection_stability_nested_v2 as nested


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "experiments" / "selection_stability_nested_v3_corrected"
DOC_PATH = ROOT / "docs" / "SELECTION_STABILITY_NESTED_V3_CORRECTED.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "45_selection_stability_nested_v3_corrected.ipynb"


def add_direction_safe_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    adverse_cols = [
        c
        for c in out.columns
        if c.startswith("obr_") and "_logit_adverse_" in c and "_safe_adverse_" not in c
    ]
    for col in adverse_cols:
        safe_col = col.replace("_logit_adverse_", "_safe_adverse_")
        out[safe_col] = 1.0 - out[col].astype(float)
    return out


def valid_score_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("obr_") and "_logit_adverse_" not in c
    ]


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Selection stability nested v3 corrected\n",
                    "\n",
                    "Correccion semantica: las probabilidades adverse se convierten en score seguro `1 - P(adverse)`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "base = '../data/experiments/selection_stability_nested_v3_corrected'\n",
                    "selected = pd.read_csv(f'{base}/selected_configs.csv')\n",
                    "selected\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "configs = pd.read_csv(f'{base}/config_summary.csv')\n",
                    "configs[configs.score_col.str.contains('safe_adverse')].head(25)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def write_doc(decision: dict, selected: pd.DataFrame, configs: pd.DataFrame, families: pd.DataFrame) -> None:
    selected_cols = [
        "selector",
        "score_col",
        "daily_cap",
        "selection_eligible",
        "inner_pass_cost0p5_min",
        "inner_pass_cost0p5_mean",
        "outer_pass_cost0p5_rate",
        "joint_worst_floor",
        "test_pass_cost0p5_rate",
        "test_pass_cost1p0_rate",
        "test_worst_day_cost0p5_mean",
        "test_robust_cost0p5",
    ]
    config_cols = [
        "score_col",
        "daily_cap",
        "selection_eligible",
        "inner_pass_cost0p5_min",
        "inner_pass_cost0p5_mean",
        "outer_pass_cost0p5_rate",
        "joint_worst_floor",
        "nested_selection_score",
        "test_pass_cost0p5_rate",
        "test_pass_cost1p0_rate",
        "test_worst_day_cost0p5_mean",
        "test_robust_cost0p5",
    ]
    family_cols = [
        "selection_scope",
        "score_family",
        "config_count",
        "eligible_rate",
        "joint_pass_floor_median",
        "inner_pass_mean_median",
        "joint_worst_floor_median",
        "nested_selection_score_median",
    ]
    safe = configs[configs["score_col"].str.contains("safe_adverse")].copy()
    robust = configs[configs["test_robust_cost0p5"]].copy()
    text = f"""# Selection stability nested v3 corrected

Fecha: 2026-06-04

## Por que existe esta version

En v1/v2 se permitio rankear directamente por:

```text
P(adverse)
```

Eso es semanticamente incorrecto porque `adverse=1` significa pertenecer al peor
25% de resultados de ejecucion. Si seleccionamos scores altos, elegimos mayor
riesgo adverso.

La correccion es:

```text
safe_adverse_score = 1 - P(adverse)
```

Los scores raw `logit_adverse` quedan excluidos como rankers. Se mantienen en
los outputs anteriores solo para trazabilidad.

## Protocolo

- tres folds walk-forward internos dentro de train;
- validation externa `21-22 mayo`;
- test terminal `23-25 mayo` abierto solo despues de seleccionar;
- selector principal: consenso de familia `book_only`;
- fill aleatorio `50%`, cost0.50 principal y cost1.00 stress.

## Decision

```text
{decision["decision"]}
```

{decision["reason"]}

## Selectores corregidos

{nested.markdown_table(selected, selected_cols)}

## Familias corregidas

{nested.markdown_table(families, family_cols, 30)}

## Configuraciones `safe_adverse`

{nested.markdown_table(safe, config_cols, 30)}

## Configuraciones corregidas robustas encontradas

{nested.markdown_table(robust, config_cols, 20)}

Importante:

```text
las familias book_plus_scores usan scores de primera etapa que son in-sample
dentro de train_initial en los artefactos actuales. Por tanto, cualquier
resultado robusto de esa familia es diagnostico, no una validacion nested limpia.
```

## Lectura sencilla

- Esta version reemplaza las conclusiones operativas basadas en rankear
  directamente `P(adverse)`.
- Un modelo puede seguir usando `P(adverse)` como feature o head auxiliar, pero
  para seleccionar oportunidades debe restarse o invertirse.
- Ninguna configuracion `safe_adverse` pasa el proxy robusto.
- La unica configuracion corregida robusta usa `book_plus_scores` y queda como
  pista hasta generar scores upstream out-of-fold.
- El resultado sigue siendo investigacion offline, no bot.
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base = add_direction_safe_scores(nested.load_base())
    inner_oof = add_direction_safe_scores(nested.build_inner_oof(base))
    score_cols = valid_score_cols(inner_oof)

    raw_parts = []
    for block_name, _, _ in nested.INNER_FOLDS:
        part = inner_oof[inner_oof["evaluation_block"].eq(block_name)].copy()
        raw_parts.append(nested.evaluate_block(part, block_name, score_cols))
    raw_parts.append(
        nested.evaluate_block(
            base[base["terminal_split"].eq("validation_initial")].copy(),
            "outer_validation",
            score_cols,
        )
    )
    raw_parts.append(
        nested.evaluate_block(
            base[base["terminal_split"].eq("test_terminal")].copy(),
            "test_terminal",
            score_cols,
        )
    )

    raw = pd.concat(raw_parts, ignore_index=True)
    block_agg = nested.aggregate_blocks(raw)
    configs = nested.build_config_summary(block_agg)
    corrected_robust = configs[configs["test_robust_cost0p5"]].copy()
    safe_adverse_robust = corrected_robust[corrected_robust["score_col"].str.contains("safe_adverse")].copy()

    exact_book = nested.choose_exact(configs, "nested_exact_book_only_corrected", "obr_book_only")
    family_book, family_book_table = nested.choose_family(
        configs, "nested_family_book_only_corrected", "obr_book_only"
    )
    family_all, family_all_table = nested.choose_family(
        configs, "nested_family_all_orderbook_corrected", "obr_"
    )
    selected = pd.DataFrame([exact_book, family_book, family_all])

    family_book_table["selection_scope"] = "obr_book_only"
    family_all_table["selection_scope"] = "obr_all"
    families = pd.concat([family_book_table, family_all_table], ignore_index=True)

    primary = selected[selected["selector"].eq("nested_family_book_only_corrected")].iloc[0]
    primary_pass = bool(primary["test_robust_cost0p5"])
    secondary_pass = bool(selected["test_robust_cost0p5"].any())
    if primary_pass:
        decision_name = "RESEARCH_PASS_NESTED_V3_CORRECTED_BOOK_ONLY_COST05"
        reason = "The corrected predeclared nested book-only family selector passes the test cost0.50 robustness proxy."
    elif secondary_pass:
        decision_name = "RESEARCH_PASS_NESTED_V3_CORRECTED_SECONDARY_ONLY"
        reason = "The corrected primary selector fails, but a corrected secondary predeclared selector passes."
    else:
        decision_name = "NO_GO_NESTED_V3_CORRECTED"
        reason = "None of the corrected predeclared nested selectors closes the test cost0.50 robustness proxy."

    decision = {
        "decision": decision_name,
        "reason": reason,
        "semantic_correction": "safe_adverse_score = 1 - P(adverse); raw adverse rankers excluded",
        "corrected_robust_config_count": int(len(corrected_robust)),
        "safe_adverse_robust_config_count": int(len(safe_adverse_robust)),
        "robust_configs_are_clean_book_only": bool(
            not corrected_robust.empty
            and corrected_robust["score_col"].str.startswith("obr_book_only").all()
        ),
        "upstream_score_caveat": "book_plus_scores families use first-stage train_initial predictions that are in-sample inside inner folds",
        "primary_selector": "nested_family_book_only_corrected",
        "primary_selected_config": {
            "score_col": str(primary["score_col"]),
            "daily_cap": int(primary["daily_cap"]),
            "selection_eligible": bool(primary["selection_eligible"]),
            "inner_pass_cost0p5_min": float(primary["inner_pass_cost0p5_min"]),
            "inner_pass_cost0p5_mean": float(primary["inner_pass_cost0p5_mean"]),
            "outer_pass_cost0p5_rate": float(primary["outer_pass_cost0p5_rate"]),
            "test_pass_cost0p5_rate": float(primary["test_pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(primary["test_pass_cost1p0_rate"]),
            "test_worst_day_cost0p5_mean": float(primary["test_worst_day_cost0p5_mean"]),
            "test_robust_cost0p5": bool(primary["test_robust_cost0p5"]),
        },
        "runtime_seconds": round(time.time() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "inner_oof_predictions": str(OUT_DIR / "inner_oof_predictions.csv"),
            "raw_policy_results": str(OUT_DIR / "raw_policy_results.csv"),
            "block_aggregate": str(OUT_DIR / "block_aggregate.csv"),
            "config_summary": str(OUT_DIR / "config_summary.csv"),
            "family_summary": str(OUT_DIR / "family_summary.csv"),
            "selected_configs": str(OUT_DIR / "selected_configs.csv"),
            "doc": str(DOC_PATH),
            "notebook": str(NOTEBOOK_PATH),
        },
    }

    keep_inner = [
        "sequence_id",
        "session_day",
        "evaluation_block",
        nested.obr.TARGET_COST05,
        nested.obr.TARGET_COST10,
        nested.obr.TARGET_HEALTHY,
        nested.obr.TARGET_ADVERSE,
    ] + score_cols
    inner_oof[keep_inner].to_csv(OUT_DIR / "inner_oof_predictions.csv", index=False)
    raw.to_csv(OUT_DIR / "raw_policy_results.csv", index=False)
    block_agg.to_csv(OUT_DIR / "block_aggregate.csv", index=False)
    configs.to_csv(OUT_DIR / "config_summary.csv", index=False)
    families.to_csv(OUT_DIR / "family_summary.csv", index=False)
    selected.to_csv(OUT_DIR / "selected_configs.csv", index=False)
    (OUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_notebook()
    write_doc(decision, selected, configs, families)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
