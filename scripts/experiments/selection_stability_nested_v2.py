from __future__ import annotations

# Historical v2: raw P(adverse) rankers are semantically invalid.
# Use selection_stability_nested_v3_corrected.py.

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import orderbook_regularized_sequence_v1 as obr


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "experiments" / "orderbook_regularized_sequence_v1" / "predictions.csv"
OUT_DIR = ROOT / "data" / "experiments" / "selection_stability_nested_v2"
DOC_PATH = ROOT / "docs" / "SELECTION_STABILITY_NESTED_V2.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "44_selection_stability_nested_v2.ipynb"

DAILY_CAPS = [50, 75, 100, 150]
SEEDS = obr.SEEDS
FILL_RATE = 0.5

INNER_FOLDS = [
    ("inner_1", ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14"], ["2026-05-15", "2026-05-16"]),
    (
        "inner_2",
        ["2026-05-11", "2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"],
        ["2026-05-17", "2026-05-18"],
    ),
    (
        "inner_3",
        [
            "2026-05-11",
            "2026-05-12",
            "2026-05-13",
            "2026-05-14",
            "2026-05-15",
            "2026-05-16",
            "2026-05-17",
            "2026-05-18",
        ],
        ["2026-05-19", "2026-05-20"],
    ),
]


def fmt(v: object, digits: int = 4) -> str:
    if isinstance(v, (float, np.floating)):
        if pd.isna(v):
            return "nan"
        return f"{float(v):.{digits}f}"
    return str(v)


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_Sin filas._"
    view = df[cols].head(limit) if limit else df[cols]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def score_family(score_col: str) -> str:
    family = re.sub(r"_a\d+p\d+_c\d+p\d+$", "", score_col)
    family = re.sub(r"_a\d+p\d+$", "", family)
    family = re.sub(r"_c\d+p\d+$", "", family)
    return family


def load_base() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    for col in [obr.TARGET_HEALTHY, obr.TARGET_ADVERSE, "target_supported_H60"]:
        if col in df.columns:
            df[col] = obr.parse_bool_series(df[col])
    df["session_day"] = df["session_day"].astype(str)
    return df


def build_inner_oof(base: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fold_name, train_days, eval_days in INNER_FOLDS:
        fold = base[base["session_day"].isin(train_days + eval_days)].copy()
        fold["terminal_split"] = np.where(fold["session_day"].isin(train_days), "train_initial", "validation_initial")
        scored, _ = obr.fit_predict_models(fold)
        eval_part = scored[scored["session_day"].isin(eval_days)].copy()
        eval_part["evaluation_block"] = fold_name
        parts.append(eval_part)
    return pd.concat(parts, ignore_index=True)


def select_daily(df: pd.DataFrame, score_col: str, daily_cap: int) -> pd.DataFrame:
    return (
        df.sort_values(["session_day", score_col, "proba"], ascending=[True, False, False])
        .groupby("session_day", observed=True)
        .head(daily_cap)
        .copy()
    )


def evaluate_block(block_df: pd.DataFrame, block_name: str, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        if score_col not in block_df.columns:
            continue
        for cap in DAILY_CAPS:
            selected = select_daily(block_df, score_col, cap)
            for seed in SEEDS:
                filled = obr.fill_random_by_day(selected, FILL_RATE, seed)
                summary = obr.summarize(filled)
                rows.append(
                    {
                        "evaluation_block": block_name,
                        "score_col": score_col,
                        "score_family": score_family(score_col),
                        "daily_cap": cap,
                        "fill_rate": FILL_RATE,
                        "seed": seed,
                        **summary,
                        "pass_cost0p5": (
                            summary["actions"] >= 30
                            and summary["sum_cost0p5"] > 0
                            and summary["negative_days_cost0p5"] == 0
                        ),
                        "pass_cost1p0": (
                            summary["actions"] >= 30
                            and summary["sum_cost0p5"] > 0
                            and summary["negative_days_cost0p5"] == 0
                            and summary["sum_cost1p0"] > 0
                            and summary["negative_days_cost1p0"] == 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_blocks(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["evaluation_block", "score_col", "score_family", "daily_cap", "fill_rate"]
    for keys, part in raw.groupby(group_cols, observed=True):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "runs": int(len(part)),
                "actions_mean": float(part["actions"].mean()),
                "sum_cost0p5_mean": float(part["sum_cost0p5"].mean()),
                "sum_cost1p0_mean": float(part["sum_cost1p0"].mean()),
                "worst_day_cost0p5_mean": float(part["worst_day_cost0p5"].mean()),
                "worst_day_cost1p0_mean": float(part["worst_day_cost1p0"].mean()),
                "pass_cost0p5_rate": float(part["pass_cost0p5"].mean()),
                "pass_cost1p0_rate": float(part["pass_cost1p0"].mean()),
                "healthy_rate_mean": float(part["healthy_rate"].mean()),
                "adverse_rate_mean": float(part["adverse_rate"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_config_summary(block_agg: pd.DataFrame) -> pd.DataFrame:
    inner = block_agg[block_agg["evaluation_block"].str.startswith("inner_")].copy()
    outer = block_agg[block_agg["evaluation_block"].eq("outer_validation")].copy()
    test = block_agg[block_agg["evaluation_block"].eq("test_terminal")].copy()

    inner_rows = []
    for keys, part in inner.groupby(["score_col", "score_family", "daily_cap", "fill_rate"], observed=True):
        row = dict(zip(["score_col", "score_family", "daily_cap", "fill_rate"], keys))
        row.update(
            {
                "inner_blocks": int(len(part)),
                "inner_pass_cost0p5_min": float(part["pass_cost0p5_rate"].min()),
                "inner_pass_cost0p5_mean": float(part["pass_cost0p5_rate"].mean()),
                "inner_pass_cost1p0_mean": float(part["pass_cost1p0_rate"].mean()),
                "inner_sum_cost0p5_min": float(part["sum_cost0p5_mean"].min()),
                "inner_sum_cost0p5_mean": float(part["sum_cost0p5_mean"].mean()),
                "inner_worst_day_cost0p5_min": float(part["worst_day_cost0p5_mean"].min()),
                "inner_worst_day_cost0p5_mean": float(part["worst_day_cost0p5_mean"].mean()),
            }
        )
        inner_rows.append(row)
    summary = pd.DataFrame(inner_rows)

    keep = [
        "score_col",
        "score_family",
        "daily_cap",
        "fill_rate",
        "pass_cost0p5_rate",
        "pass_cost1p0_rate",
        "sum_cost0p5_mean",
        "sum_cost1p0_mean",
        "worst_day_cost0p5_mean",
        "worst_day_cost1p0_mean",
    ]
    outer = outer[keep].rename(
        columns={c: f"outer_{c}" for c in keep if c not in {"score_col", "score_family", "daily_cap", "fill_rate"}}
    )
    test = test[keep].rename(
        columns={c: f"test_{c}" for c in keep if c not in {"score_col", "score_family", "daily_cap", "fill_rate"}}
    )
    join_cols = ["score_col", "score_family", "daily_cap", "fill_rate"]
    summary = summary.merge(outer, on=join_cols, how="left").merge(test, on=join_cols, how="left")

    summary["joint_pass_floor"] = summary[["inner_pass_cost0p5_min", "outer_pass_cost0p5_rate"]].min(axis=1)
    summary["joint_worst_floor"] = summary[
        ["inner_worst_day_cost0p5_min", "outer_worst_day_cost0p5_mean"]
    ].min(axis=1)
    summary["joint_sum_floor"] = summary[["inner_sum_cost0p5_min", "outer_sum_cost0p5_mean"]].min(axis=1)
    summary["selection_eligible"] = (
        summary["inner_pass_cost0p5_min"].ge(0.50)
        & summary["outer_pass_cost0p5_rate"].ge(0.60)
        & summary["inner_worst_day_cost0p5_min"].ge(0.0)
        & summary["outer_worst_day_cost0p5_mean"].ge(0.0)
    )
    summary["nested_selection_score"] = (
        1000.0 * summary["joint_pass_floor"]
        + 200.0 * summary["inner_pass_cost0p5_mean"]
        + 0.50 * summary["joint_worst_floor"]
        + 0.05 * summary["joint_sum_floor"]
        - 0.01 * summary["daily_cap"]
    )
    summary["test_robust_cost0p5"] = (
        summary["test_pass_cost0p5_rate"].ge(0.60) & summary["test_worst_day_cost0p5_mean"].ge(0.0)
    )
    return summary.sort_values("nested_selection_score", ascending=False)


def choose_exact(configs: pd.DataFrame, selector: str, prefix: str) -> pd.Series:
    pool = configs[configs["score_col"].str.startswith(prefix)].copy()
    eligible = pool[pool["selection_eligible"]].copy()
    if not eligible.empty:
        pool = eligible
    selected = pool.sort_values(
        ["nested_selection_score", "joint_pass_floor", "joint_worst_floor", "daily_cap"],
        ascending=[False, False, False, True],
    ).iloc[0].copy()
    selected["selector"] = selector
    selected["selection_scope"] = prefix
    return selected


def choose_family(configs: pd.DataFrame, selector: str, prefix: str) -> tuple[pd.Series, pd.DataFrame]:
    pool = configs[configs["score_col"].str.startswith(prefix)].copy()
    family_rows = []
    for family, part in pool.groupby("score_family", observed=True):
        family_rows.append(
            {
                "score_family": family,
                "config_count": int(len(part)),
                "eligible_rate": float(part["selection_eligible"].mean()),
                "joint_pass_floor_median": float(part["joint_pass_floor"].median()),
                "joint_pass_floor_min": float(part["joint_pass_floor"].min()),
                "inner_pass_mean_median": float(part["inner_pass_cost0p5_mean"].median()),
                "joint_worst_floor_median": float(part["joint_worst_floor"].median()),
                "nested_selection_score_median": float(part["nested_selection_score"].median()),
            }
        )
    families = pd.DataFrame(family_rows).sort_values(
        [
            "eligible_rate",
            "joint_pass_floor_median",
            "inner_pass_mean_median",
            "joint_worst_floor_median",
            "nested_selection_score_median",
        ],
        ascending=False,
    )
    chosen_family = families.iloc[0]["score_family"]
    candidates = pool[pool["score_family"].eq(chosen_family)].copy()
    eligible = candidates[candidates["selection_eligible"]].copy()
    if not eligible.empty:
        candidates = eligible
    selected = candidates.sort_values(
        ["nested_selection_score", "joint_pass_floor", "joint_worst_floor", "daily_cap"],
        ascending=[False, False, False, True],
    ).iloc[0].copy()
    selected["selector"] = selector
    selected["selection_scope"] = prefix
    return selected, families


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Selection stability nested v2\n",
                    "\n",
                    "Selector walk-forward: entrena en pasado, evalua en bloques posteriores y deja test para auditoria final.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "base = '../data/experiments/selection_stability_nested_v2'\n",
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
                    "configs[configs.score_col.str.startswith('obr_book_only')].head(25)\n",
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
    best_book = configs[configs["score_col"].str.startswith("obr_book_only")].head(20)
    text = f"""# Selection stability nested v2

Fecha: 2026-06-04

## Objetivo

Seleccionar una policy de orderbook sin depender de solo dos dias de validation.

Se crean tres bloques walk-forward dentro de train:

```text
11-14 mayo -> 15-16 mayo
11-16 mayo -> 17-18 mayo
11-18 mayo -> 19-20 mayo
```

Despues se usa validation `21-22 mayo`. Test `23-25 mayo` solo se abre al final.

## Decision

```text
{decision["decision"]}
```

{decision["reason"]}

## Selectores fijados antes de test

{markdown_table(selected, selected_cols)}

## Familias auditadas

{markdown_table(families, family_cols, 30)}

## Mejores configuraciones book-only

{markdown_table(best_book, config_cols, 20)}

## Lectura sencilla

- Los modelos orderbook de cada bloque interno se entrenan solo con dias anteriores.
- El selector principal usa `book_only` para evitar leakage de scores upstream dentro de train.
- `selection_eligible` exige estabilidad tanto en bloques internos como en validation.
- El test se usa solo para comprobar si la regla seleccionada generaliza.

## Siguiente paso

Si el selector principal pasa:

```text
congelar selector y probar encoder CNN/Conv1D pequeno bajo el mismo protocolo.
```

Si falla:

```text
no aumentar arquitectura todavia; ampliar periodos o reformular target de ejecucion.
```
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base()

    inner_oof = build_inner_oof(base)
    score_cols = [c for c in inner_oof.columns if c.startswith("obr_")]

    raw_parts = []
    for block_name, _, eval_days in INNER_FOLDS:
        part = inner_oof[inner_oof["evaluation_block"].eq(block_name)].copy()
        raw_parts.append(evaluate_block(part, block_name, score_cols))

    outer = base[base["terminal_split"].eq("validation_initial")].copy()
    test = base[base["terminal_split"].eq("test_terminal")].copy()
    raw_parts.append(evaluate_block(outer, "outer_validation", score_cols))
    raw_parts.append(evaluate_block(test, "test_terminal", score_cols))

    raw = pd.concat(raw_parts, ignore_index=True)
    block_agg = aggregate_blocks(raw)
    configs = build_config_summary(block_agg)

    exact_book = choose_exact(configs, "nested_exact_book_only", "obr_book_only")
    family_book, family_book_table = choose_family(configs, "nested_family_book_only", "obr_book_only")
    family_all, family_all_table = choose_family(configs, "nested_family_all_orderbook", "obr_")
    selected = pd.DataFrame([exact_book, family_book, family_all])

    family_book_table["selection_scope"] = "obr_book_only"
    family_all_table["selection_scope"] = "obr_all"
    families = pd.concat([family_book_table, family_all_table], ignore_index=True)

    primary = selected[selected["selector"].eq("nested_family_book_only")].iloc[0]
    primary_pass = bool(primary["test_robust_cost0p5"])
    secondary_pass = bool(selected["test_robust_cost0p5"].any())
    if primary_pass:
        decision_name = "RESEARCH_PASS_NESTED_SELECTOR_BOOK_ONLY_COST05"
        reason = "The predeclared nested book-only family selector chooses a config that passes the test cost0.50 robustness proxy."
    elif secondary_pass:
        decision_name = "RESEARCH_PASS_SECONDARY_ONLY_NESTED_SELECTOR"
        reason = "The primary nested book-only family selector fails, but a secondary predeclared nested selector passes."
    else:
        decision_name = "NO_GO_NESTED_SELECTOR_V2"
        reason = "None of the predeclared nested selectors closes the test cost0.50 robustness proxy."

    decision = {
        "decision": decision_name,
        "reason": reason,
        "primary_selector": "nested_family_book_only",
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
        obr.TARGET_COST05,
        obr.TARGET_COST10,
        obr.TARGET_HEALTHY,
        obr.TARGET_ADVERSE,
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
