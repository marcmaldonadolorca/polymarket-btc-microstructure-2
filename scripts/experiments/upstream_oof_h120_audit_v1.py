from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_v1_book_tabular_sequence as fusion  # noqa: E402
import orderbook_regularized_sequence_v1 as obr  # noqa: E402
import upstream_oof_fusion_stage2_v1 as clean_oof  # noqa: E402


EXECUTION_DATASET = (
    ROOT
    / "data"
    / "experiments"
    / "complex_v1a_execution_dataset"
    / "complex_v1a_execution_dataset.parquet"
)
H60_AUDIT_DIR = ROOT / "data" / "experiments" / "upstream_oof_maturity_regime_audit_v1"
OUT_DIR = ROOT / "data" / "experiments" / "upstream_oof_h120_audit_v1"
DOC_PATH = ROOT / "docs" / "UPSTREAM_OOF_H120_AUDIT_V1.md"
NOTEBOOK_PATH = ROOT / "notebooks" / "49_upstream_oof_h120_audit_v1.ipynb"

TARGET_COLUMNS = {
    "exec_net_cost_0p25_H120": clean_oof.TARGET_COST025,
    "exec_net_cost_0p5_H120": clean_oof.TARGET_COST05,
    "exec_net_cost_1p0_H120": clean_oof.TARGET_COST10,
}
SCORE_COLS = ["upstream_pred", "upstream_proba", "upstream_combo"]
DAILY_CAPS = [50, 75, 100]
SEEDS = obr.SEEDS
SELECTION_BLOCKS = ["inner_1", "inner_2", "inner_3", "outer_validation"]


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return "nan"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    if df.empty:
        return "_Sin filas._"
    view = df[cols].head(limit) if limit else df[cols]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(fmt(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def load_h120_targets(index: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["session_id", "market_id", "token_id"]
    left = index[
        [
            "sequence_id",
            "session_id",
            "market_id",
            "token_id",
            "last_time_index_ns",
            "session_day",
            "terminal_split",
        ]
    ].copy()
    left["market_id"] = left["market_id"].astype(str)
    left["token_id"] = left["token_id"].astype(str)

    execution = pd.read_parquet(
        EXECUTION_DATASET,
        columns=[
            *key_cols,
            "time_index_ns",
            "target_supported_H120",
            *TARGET_COLUMNS,
        ],
    )
    execution["market_id"] = execution["market_id"].astype(str)
    execution["token_id"] = execution["token_id"].astype(str)

    if left.duplicated([*key_cols, "last_time_index_ns"]).any():
        raise ValueError("Sequence terminal keys are not unique.")
    if execution.duplicated([*key_cols, "time_index_ns"]).any():
        raise ValueError("Execution dataset keys are not unique.")

    merged = left.merge(
        execution,
        left_on=[*key_cols, "last_time_index_ns"],
        right_on=[*key_cols, "time_index_ns"],
        how="left",
        validate="one_to_one",
        indicator=True,
        sort=False,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("Some sequence terminals did not match the execution dataset.")
    if not merged["sequence_id"].equals(index["sequence_id"]):
        raise ValueError("H120 merge changed sequence order.")

    label_complete = merged[list(TARGET_COLUMNS)].notna().all(axis=1)
    merged["h120_supported"] = (
        merged["target_supported_H120"].fillna(False).astype(bool) & label_complete
    )
    for source, target in TARGET_COLUMNS.items():
        merged[target] = merged[source].astype("float32")
    return merged


def support_summary(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, column in [("day", "session_day"), ("split", "terminal_split")]:
        for value, part in targets.groupby(column, observed=True):
            supported = part[part["h120_supported"]]
            rows.append(
                {
                    "level": level,
                    "value": str(value),
                    "sequences": int(len(part)),
                    "supported_sequences": int(len(supported)),
                    "support_rate": float(part["h120_supported"].mean()),
                    "mean_cost0p5_supported": float(
                        supported[clean_oof.TARGET_COST05].mean()
                    ),
                    "positive_cost0p5_rate_supported": float(
                        supported[clean_oof.TARGET_COST05].gt(0).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def load_arrays() -> dict:
    seq = np.load(clean_oof.SEQ_ARRAYS_PATH)
    book = np.load(clean_oof.BOOK_ARRAYS_PATH)
    index = pd.read_csv(clean_oof.INDEX_PATH)
    feature_manifest = pd.read_csv(clean_oof.FEATURE_MANIFEST_PATH)
    targets = load_h120_targets(index)

    blocked_names = set(
        feature_manifest.loc[
            feature_manifest["feature_name"].str.contains(
                r"ev_pred|healthy_proba", regex=True
            ),
            "feature_name",
        ].astype(str)
    )
    keep_idx = [
        int(row.feature_index)
        for row in feature_manifest.itertuples()
        if str(row.feature_name) not in blocked_names
    ]
    return {
        "index": index,
        "targets": targets,
        "x_raw": seq["x_raw"].astype("float32"),
        "mask": seq["mask"].astype(bool),
        "lengths": seq["lengths"].astype("int64"),
        "y60_025": seq["y_last_cost0p25"].astype("float32"),
        "y60_05": seq["y_last_cost0p5"].astype("float32"),
        "y60_10": seq["y_last_cost1p0"].astype("float32"),
        "x_book_raw": book["tensor"].astype("float32"),
        "book_mask": book["step_mask"].astype(bool),
        "keep_idx": keep_idx,
        "blocked_count": len(blocked_names),
    }


def generate_oof(
    data: dict,
    y025: np.ndarray,
    y05: np.ndarray,
    y10: np.ndarray,
    supported: np.ndarray,
    horizon: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.array_equal(data["mask"], data["book_mask"]):
        raise ValueError("Book and tabular masks differ.")

    index = data["index"]
    device = torch.device("cpu")
    prediction_rows = []
    fold_rows = []

    for fold_no, (block_name, train_days, eval_days) in enumerate(
        clean_oof.UPSTREAM_BLOCKS, start=1
    ):
        train_base = index["session_day"].astype(str).isin(train_days).to_numpy()
        eval_base = index["session_day"].astype(str).isin(eval_days).to_numpy()
        train_idx = train_base & supported
        eval_idx = eval_base & supported
        if train_idx.sum() == 0 or eval_idx.sum() == 0:
            raise ValueError(f"Empty supported H120 fold: {block_name}")

        tab_median, tab_mean, tab_std = clean_oof.fit_tab_scaler(
            data["x_raw"], data["mask"], train_idx
        )
        x_tab = clean_oof.transform_tab(
            data["x_raw"],
            data["mask"],
            tab_median,
            tab_mean,
            tab_std,
            data["keep_idx"],
        )
        split_for_scaler = pd.Series(np.where(train_idx, "train_initial", "evaluation"))
        x_book, _ = fusion.fit_transform_book_tensor(
            data["x_book_raw"], data["mask"], split_for_scaler
        )
        pred, proba, info = clean_oof.train_upstream_fixed(
            x_tab,
            x_book,
            data["mask"],
            data["lengths"],
            y05,
            train_idx,
            eval_idx,
            device,
            seed=fusion.SEED + fold_no,
        )

        part = index.loc[
            eval_idx,
            [
                "sequence_id",
                "session_id",
                "market_id",
                "token_id",
                "session_day",
                "terminal_split",
                "last_time_index_ns",
            ],
        ].copy()
        part["horizon"] = horizon
        part["evaluation_block"] = block_name
        part["upstream_pred"] = pred
        part["upstream_proba"] = proba
        part["upstream_combo"] = (
            0.50 * clean_oof.sigmoid_np(pred.astype(float))
            + 0.50 * proba.astype(float)
        )
        part[clean_oof.TARGET_COST025] = y025[eval_idx]
        part[clean_oof.TARGET_COST05] = y05[eval_idx]
        part[clean_oof.TARGET_COST10] = y10[eval_idx]
        prediction_rows.append(part)

        fold_rows.append(
            {
                "horizon": horizon,
                "evaluation_block": block_name,
                "train_start": train_days[0],
                "train_end": train_days[-1],
                "eval_start": eval_days[0],
                "eval_end": eval_days[-1],
                "train_sequences_total": int(train_base.sum()),
                "train_sequences_supported": int(train_idx.sum()),
                "train_support_rate": float(train_idx.sum() / train_base.sum()),
                "eval_sequences_total": int(eval_base.sum()),
                "eval_sequences_supported": int(eval_idx.sum()),
                "eval_support_rate": float(eval_idx.sum() / eval_base.sum()),
                "tab_features_used": len(data["keep_idx"]),
                "tab_features_blocked": data["blocked_count"],
                "device": str(device),
                "fixed_epochs": clean_oof.FIXED_UPSTREAM_EPOCHS,
                **info,
            }
        )
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(fold_rows)


def quality_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, part in predictions.groupby("evaluation_block", observed=True):
        y_pos = part[clean_oof.TARGET_COST05].gt(0).astype(int)
        row = {
            "evaluation_block": block,
            "rows": int(len(part)),
            "auc_positive_cost0p5": (
                float(roc_auc_score(y_pos, part["upstream_proba"]))
                if y_pos.nunique() > 1
                else np.nan
            ),
            "spearman_pred_cost0p5": float(
                part["upstream_pred"].corr(
                    part[clean_oof.TARGET_COST05], method="spearman"
                )
            ),
            "all_mean_cost0p5": float(part[clean_oof.TARGET_COST05].mean()),
            "all_positive_cost0p5_rate": float(y_pos.mean()),
        }
        for fraction in [0.20, 0.35, 0.50]:
            rank = part.groupby("session_day", observed=True)["upstream_proba"].rank(
                pct=True, method="first"
            )
            selected = part[rank.gt(1.0 - fraction)]
            label = str(fraction).replace(".", "p")
            row[f"top{label}_actions"] = int(len(selected))
            row[f"top{label}_mean_cost0p5"] = float(
                selected[clean_oof.TARGET_COST05].mean()
            )
            row[f"top{label}_mean_cost1p0"] = float(
                selected[clean_oof.TARGET_COST10].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def select_daily(df: pd.DataFrame, score_col: str, daily_cap: int) -> pd.DataFrame:
    return (
        df.sort_values(["session_day", score_col], ascending=[True, False])
        .groupby("session_day", observed=True)
        .head(daily_cap)
        .copy()
    )


def evaluate_policies(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, part in predictions.groupby("evaluation_block", observed=True):
        for score_col in SCORE_COLS:
            for cap in DAILY_CAPS:
                selected = select_daily(part, score_col, cap)
                for seed in SEEDS:
                    filled = obr.fill_random_by_day(selected, 0.5, seed)
                    summary = clean_oof.summarize(filled)
                    rows.append(
                        {
                            "evaluation_block": block,
                            "score_col": score_col,
                            "daily_cap": cap,
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


def aggregate_policies(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    block_rows = []
    for keys, part in raw.groupby(
        ["evaluation_block", "score_col", "daily_cap"], observed=True
    ):
        block, score_col, cap = keys
        block_rows.append(
            {
                "evaluation_block": block,
                "score_col": score_col,
                "daily_cap": cap,
                "runs": int(len(part)),
                "actions_mean": float(part["actions"].mean()),
                "pass_cost0p5_rate": float(part["pass_cost0p5"].mean()),
                "pass_cost1p0_rate": float(part["pass_cost1p0"].mean()),
                "sum_cost0p5_mean": float(part["sum_cost0p5"].mean()),
                "sum_cost1p0_mean": float(part["sum_cost1p0"].mean()),
                "worst_day_cost0p5_mean": float(part["worst_day_cost0p5"].mean()),
                "worst_day_cost1p0_mean": float(part["worst_day_cost1p0"].mean()),
            }
        )
    blocks = pd.DataFrame(block_rows)

    rows = []
    for (score_col, cap), part in blocks.groupby(
        ["score_col", "daily_cap"], observed=True
    ):
        selection = part[part["evaluation_block"].isin(SELECTION_BLOCKS)]
        test = part[part["evaluation_block"].eq("test_terminal")].iloc[0]
        row = {
            "score_col": score_col,
            "daily_cap": cap,
            "selection_blocks": int(len(selection)),
            "selection_pass_cost0p5_min": float(
                selection["pass_cost0p5_rate"].min()
            ),
            "selection_pass_cost0p5_mean": float(
                selection["pass_cost0p5_rate"].mean()
            ),
            "selection_worst_day_cost0p5_min": float(
                selection["worst_day_cost0p5_mean"].min()
            ),
            "selection_sum_cost0p5_min": float(
                selection["sum_cost0p5_mean"].min()
            ),
            "test_pass_cost0p5_rate": float(test["pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(test["pass_cost1p0_rate"]),
            "test_sum_cost0p5_mean": float(test["sum_cost0p5_mean"]),
            "test_sum_cost1p0_mean": float(test["sum_cost1p0_mean"]),
            "test_worst_day_cost0p5_mean": float(test["worst_day_cost0p5_mean"]),
            "test_worst_day_cost1p0_mean": float(test["worst_day_cost1p0_mean"]),
        }
        row["selection_eligible"] = (
            row["selection_blocks"] == len(SELECTION_BLOCKS)
            and row["selection_pass_cost0p5_min"] >= 0.50
            and row["selection_worst_day_cost0p5_min"] >= 0.0
        )
        row["selection_score"] = (
            1000.0 * row["selection_pass_cost0p5_min"]
            + 200.0 * row["selection_pass_cost0p5_mean"]
            + 0.50 * row["selection_worst_day_cost0p5_min"]
            + 0.05 * row["selection_sum_cost0p5_min"]
            - 0.01 * cap
        )
        row["test_robust_cost0p5"] = (
            row["test_pass_cost0p5_rate"] >= 0.60
            and row["test_worst_day_cost0p5_mean"] >= 0.0
        )
        rows.append(row)
    return blocks, pd.DataFrame(rows).sort_values("selection_score", ascending=False)


def horizon_comparison(
    h120_quality: pd.DataFrame,
    h120_policies: pd.DataFrame,
    h60_paired_quality: pd.DataFrame,
    h60_paired_policies: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    h60_quality = pd.read_csv(H60_AUDIT_DIR / "quality_summary.csv")
    h60_quality = h60_quality[h60_quality["strategy"].eq("expanding")].copy()
    h60_policies = pd.read_csv(H60_AUDIT_DIR / "policy_summary.csv")
    h60_policies = h60_policies[h60_policies["strategy"].eq("expanding")].copy()
    rows = []
    for horizon, quality, policies, support_rate in [
        ("H60_all", h60_quality, h60_policies, 1.0),
        (
            "H60_paired",
            h60_paired_quality,
            h60_paired_policies,
            float(targets["h120_supported"].mean()),
        ),
        ("H120", h120_quality, h120_policies, float(targets["h120_supported"].mean())),
    ]:
        selection = quality[quality["evaluation_block"].isin(SELECTION_BLOCKS)]
        test = quality[quality["evaluation_block"].eq("test_terminal")].iloc[0]
        rows.append(
            {
                "horizon": horizon,
                "sequence_support_rate": support_rate,
                "selection_auc_mean": float(selection["auc_positive_cost0p5"].mean()),
                "selection_spearman_mean": float(
                    selection["spearman_pred_cost0p5"].mean()
                ),
                "selection_top0p35_mean_cost0p5_min": float(
                    selection["top0p35_mean_cost0p5"].min()
                ),
                "selection_top0p35_mean_cost0p5_mean": float(
                    selection["top0p35_mean_cost0p5"].mean()
                ),
                "test_auc": float(test["auc_positive_cost0p5"]),
                "test_spearman": float(test["spearman_pred_cost0p5"]),
                "test_top0p35_mean_cost0p5": float(test["top0p35_mean_cost0p5"]),
                "test_top0p35_mean_cost1p0": float(test["top0p35_mean_cost1p0"]),
                "eligible_policy_count": int(policies["selection_eligible"].sum()),
                "robust_test_policy_count": int(policies["test_robust_cost0p5"].sum()),
            }
        )
    return pd.DataFrame(rows)


def write_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Auditoria OOF limpia del horizonte H120\n",
                    "\n",
                    "Pregunta: ¿dar más tiempo al movimiento mejora la señal y la estabilidad frente a H60?\n",
                    "\n",
                    "El modelo, las features y el protocolo se mantienen. Solo cambia el target a H120. "
                    "Así evitamos atribuir a H120 una mejora causada por cambiar muchas cosas a la vez.\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Reglas contra leakage\n",
                    "\n",
                    "- Cada fold entrena únicamente con días anteriores.\n",
                    "- Scalers y modelo se ajustan solo con filas pasadas que tienen target H120.\n",
                    "- La configuración se elige con `inner_1/2/3 + outer_validation`.\n",
                    "- `test_terminal` se usa solo para la auditoría final.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "base = '../data/experiments/upstream_oof_h120_audit_v1'\n",
                    "support = pd.read_csv(f'{base}/support_summary.csv')\n",
                    "quality = pd.read_csv(f'{base}/quality_summary.csv')\n",
                    "policies = pd.read_csv(f'{base}/policy_summary.csv')\n",
                    "comparison = pd.read_csv(f'{base}/horizon_comparison.csv')\n",
                    "comparison\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "days = support[support.level == 'day']\n",
                    "days.plot(x='value', y='support_rate', kind='bar', figsize=(12, 4), legend=False)\n",
                    "plt.ylim(0, 1); plt.title('Soporte H120 por día'); plt.ylabel('proporción')\n",
                    "plt.show()\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "quality[['evaluation_block','auc_positive_cost0p5','spearman_pred_cost0p5',"
                    "'top0p35_mean_cost0p5','top0p35_mean_cost1p0']]\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "policies.head(15)\n",
                ],
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Cómo interpretar el resultado\n",
                    "\n",
                    "Una media positiva no basta. Pedimos que la policy pueda elegirse sin mirar test, "
                    "que sobreviva varias semillas de fill y que no esconda un día claramente negativo. "
                    "El informe Markdown contiene la decisión final y sus limitaciones.\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def write_doc(
    decision: dict,
    support: pd.DataFrame,
    folds: pd.DataFrame,
    quality: pd.DataFrame,
    policies: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    support_days = support[support["level"].eq("day")]
    support_splits = support[support["level"].eq("split")]
    support_cols = [
        "value",
        "sequences",
        "supported_sequences",
        "support_rate",
        "mean_cost0p5_supported",
        "positive_cost0p5_rate_supported",
    ]
    fold_cols = [
        "horizon",
        "evaluation_block",
        "train_sequences_supported",
        "train_support_rate",
        "eval_sequences_supported",
        "eval_support_rate",
        "tab_features_used",
        "tab_features_blocked",
    ]
    quality_cols = [
        "evaluation_block",
        "rows",
        "auc_positive_cost0p5",
        "spearman_pred_cost0p5",
        "all_mean_cost0p5",
        "top0p35_mean_cost0p5",
        "top0p35_mean_cost1p0",
    ]
    policy_cols = [
        "score_col",
        "daily_cap",
        "selection_eligible",
        "selection_pass_cost0p5_min",
        "selection_pass_cost0p5_mean",
        "selection_worst_day_cost0p5_min",
        "test_pass_cost0p5_rate",
        "test_pass_cost1p0_rate",
        "test_sum_cost0p5_mean",
        "test_worst_day_cost0p5_mean",
        "test_robust_cost0p5",
    ]
    comparison_cols = [
        "horizon",
        "sequence_support_rate",
        "selection_auc_mean",
        "selection_spearman_mean",
        "selection_top0p35_mean_cost0p5_min",
        "selection_top0p35_mean_cost0p5_mean",
        "test_auc",
        "test_top0p35_mean_cost0p5",
        "test_top0p35_mean_cost1p0",
        "eligible_policy_count",
        "robust_test_policy_count",
    ]
    text = f"""# Upstream OOF H120 audit v1

Fecha: 2026-06-04

## Pregunta

¿El mismo modelo secuencial limpio funciona mejor si intentamos capturar un
movimiento a `120` segundos en lugar de `60`?

La comparación cambia solo el horizonte. No añadimos arquitectura, features ni
una segunda etapa. Esto permite saber si la mejora, si aparece, viene de dar
más tiempo a la señal.

## Construcción del target

El target H120 no estaba materializado dentro del tensor secuencial. Se une
desde el dataset de ejecución mediante la clave exacta:

```text
session_id + market_id + token_id + last_time_index_ns
```

Las `5.831` secuencias encuentran una única fila terminal. Solo se entrena y
evalúa donde `target_supported_H120 = true` y los tres costes están presentes.

Además se reentrena un control `H60_paired` usando exactamente esas mismas
secuencias. La comparación `H60_paired` frente a `H120` aísla el efecto del
horizonte; `H60_all` conserva la referencia histórica con toda la muestra.

## Protocolo causal

- features tabulares y tensor de orderbook idénticos al audit H60;
- control H60 emparejado con la misma muestra de train/evaluación que H120;
- se excluyen las ocho features auxiliares `ev_pred/healthy_proba`;
- `fusion_concat_gru`, una época fija y CPU determinista;
- scaler y modelo se ajustan solo con días pasados soportados;
- objetivo de entrenamiento: `exec_net_cost_0p5_H120`;
- ranking evaluado con `upstream_pred`, `upstream_proba` y su combinación;
- coste `0.50` es el criterio principal y coste `1.00` es stress;
- selección con `inner_1/2/3 + outer_validation`;
- test terminal se abre solo al final.

## Decisión

```text
{decision["decision"]}
```

{decision["reason"]}

Configuración elegida antes de mirar test:

```text
score={decision["selected_config"]["score_col"]}
cap={decision["selected_config"]["daily_cap"]}
selection_eligible={decision["selected_config"]["selection_eligible"]}
```

## Soporte H120 por split

{markdown_table(support_splits, support_cols)}

## Soporte H120 por día

{markdown_table(support_days, support_cols)}

## Folds

{markdown_table(folds, fold_cols)}

## Calidad del ranking H120

{markdown_table(quality, quality_cols)}

## Comparación justa H60 frente a H120

{markdown_table(comparison, comparison_cols)}

## Policies H120

{markdown_table(policies, policy_cols)}

## Lectura sencilla

- `AUC` y `Spearman` indican si el modelo ordena mejor las oportunidades, no
  si ya existe rentabilidad operativa.
- `top0p35_mean_cost...` mide el resultado medio del 35% mejor puntuado.
- Una policy solo es elegible si se puede escoger con los bloques anteriores y
  no esconde un día negativo en selección.
- El test terminal no participa en la elección.

## Conclusión metodológica

{decision["conclusion"]}

No se debe promover todavía a bot ni aumentar a CNN/Transformer grande por
este resultado aislado. El siguiente paso depende de si H120 mejora de forma
causal y estable frente a H60, no de que exista una suma positiva puntual.
"""
    DOC_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    started = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_arrays()
    support = support_summary(data["targets"])
    h120_supported = data["targets"]["h120_supported"].to_numpy(dtype=bool)
    h120_y025 = data["targets"][clean_oof.TARGET_COST025].to_numpy(dtype="float32")
    h120_y05 = data["targets"][clean_oof.TARGET_COST05].to_numpy(dtype="float32")
    h120_y10 = data["targets"][clean_oof.TARGET_COST10].to_numpy(dtype="float32")
    predictions, h120_folds = generate_oof(
        data,
        h120_y025,
        h120_y05,
        h120_y10,
        h120_supported,
        "H120",
    )
    h60_paired_predictions, h60_paired_folds = generate_oof(
        data,
        data["y60_025"],
        data["y60_05"],
        data["y60_10"],
        h120_supported,
        "H60_paired",
    )
    folds = pd.concat([h120_folds, h60_paired_folds], ignore_index=True)
    quality = quality_summary(predictions)
    h60_paired_quality = quality_summary(h60_paired_predictions)
    raw = evaluate_policies(predictions)
    h60_paired_raw = evaluate_policies(h60_paired_predictions)
    blocks, policies = aggregate_policies(raw)
    h60_paired_blocks, h60_paired_policies = aggregate_policies(h60_paired_raw)
    comparison = horizon_comparison(
        quality,
        policies,
        h60_paired_quality,
        h60_paired_policies,
        data["targets"],
    )

    eligible = policies[policies["selection_eligible"]]
    selection_pool = eligible if not eligible.empty else policies
    selected = selection_pool.sort_values("selection_score", ascending=False).iloc[0]
    selected_pass = bool(selected["test_robust_cost0p5"])
    h60 = comparison[comparison["horizon"].eq("H60_paired")].iloc[0]
    h120 = comparison[comparison["horizon"].eq("H120")].iloc[0]
    stable_improvement = (
        h120["selection_top0p35_mean_cost0p5_min"]
        > h60["selection_top0p35_mean_cost0p5_min"]
        and h120["selection_top0p35_mean_cost0p5_mean"]
        > h60["selection_top0p35_mean_cost0p5_mean"]
        and h120["test_top0p35_mean_cost0p5"] > h60["test_top0p35_mean_cost0p5"]
    )

    if selected_pass and bool(selected["selection_eligible"]):
        decision_name = "RESEARCH_PASS_CLEAN_OOF_H120_COST05"
        reason = (
            "La policy H120 elegida solo con bloques previos supera el proxy "
            "de robustez cost0.50 en test terminal."
        )
    else:
        decision_name = "NO_GO_CLEAN_OOF_H120_COST05"
        reason = (
            "La policy H120 elegida con bloques previos no cierra el proxy de "
            "robustez cost0.50 en test terminal."
        )

    if stable_improvement:
        conclusion = (
            "H120 mejora de forma consistente el ranking medio/mínimo de selección "
            "y el test frente a H60. Merece una segunda prueba H60/H120 multitask, "
            "pero aún necesita resolver estabilidad y fills reales."
        )
    else:
        conclusion = (
            "H120 no mejora de forma consistente los bloques previos y el test "
            "frente a H60. Alargar el horizonte por sí solo no resuelve la "
            "inestabilidad; conviene priorizar calidad de fill/labels y más folds."
        )

    decision = {
        "decision": decision_name,
        "reason": reason,
        "conclusion": conclusion,
        "selected_config": {
            "score_col": str(selected["score_col"]),
            "daily_cap": int(selected["daily_cap"]),
            "selection_eligible": bool(selected["selection_eligible"]),
            "selection_pass_cost0p5_min": float(
                selected["selection_pass_cost0p5_min"]
            ),
            "selection_pass_cost0p5_mean": float(
                selected["selection_pass_cost0p5_mean"]
            ),
            "selection_worst_day_cost0p5_min": float(
                selected["selection_worst_day_cost0p5_min"]
            ),
            "test_pass_cost0p5_rate": float(selected["test_pass_cost0p5_rate"]),
            "test_pass_cost1p0_rate": float(selected["test_pass_cost1p0_rate"]),
            "test_sum_cost0p5_mean": float(selected["test_sum_cost0p5_mean"]),
            "test_sum_cost1p0_mean": float(selected["test_sum_cost1p0_mean"]),
            "test_worst_day_cost0p5_mean": float(
                selected["test_worst_day_cost0p5_mean"]
            ),
            "test_robust_cost0p5": bool(selected["test_robust_cost0p5"]),
        },
        "h120_support_rate": float(data["targets"]["h120_supported"].mean()),
        "eligible_policy_count": int(policies["selection_eligible"].sum()),
        "robust_test_policy_count": int(policies["test_robust_cost0p5"].sum()),
        "stable_h120_improvement_over_h60": bool(stable_improvement),
        "runtime_seconds": round(time.time() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "support_summary": str(OUT_DIR / "support_summary.csv"),
            "predictions": str(OUT_DIR / "predictions.csv"),
            "h60_paired_predictions": str(OUT_DIR / "h60_paired_predictions.csv"),
            "fold_summary": str(OUT_DIR / "fold_summary.csv"),
            "quality_summary": str(OUT_DIR / "quality_summary.csv"),
            "h60_paired_quality_summary": str(
                OUT_DIR / "h60_paired_quality_summary.csv"
            ),
            "raw_policy_results": str(OUT_DIR / "raw_policy_results.csv"),
            "block_aggregate": str(OUT_DIR / "block_aggregate.csv"),
            "policy_summary": str(OUT_DIR / "policy_summary.csv"),
            "h60_paired_block_aggregate": str(
                OUT_DIR / "h60_paired_block_aggregate.csv"
            ),
            "h60_paired_policy_summary": str(
                OUT_DIR / "h60_paired_policy_summary.csv"
            ),
            "horizon_comparison": str(OUT_DIR / "horizon_comparison.csv"),
            "doc": str(DOC_PATH),
            "notebook": str(NOTEBOOK_PATH),
        },
    }

    support.to_csv(OUT_DIR / "support_summary.csv", index=False)
    predictions.to_csv(OUT_DIR / "predictions.csv", index=False)
    h60_paired_predictions.to_csv(OUT_DIR / "h60_paired_predictions.csv", index=False)
    folds.to_csv(OUT_DIR / "fold_summary.csv", index=False)
    quality.to_csv(OUT_DIR / "quality_summary.csv", index=False)
    h60_paired_quality.to_csv(OUT_DIR / "h60_paired_quality_summary.csv", index=False)
    raw.to_csv(OUT_DIR / "raw_policy_results.csv", index=False)
    blocks.to_csv(OUT_DIR / "block_aggregate.csv", index=False)
    policies.to_csv(OUT_DIR / "policy_summary.csv", index=False)
    h60_paired_blocks.to_csv(OUT_DIR / "h60_paired_block_aggregate.csv", index=False)
    h60_paired_policies.to_csv(OUT_DIR / "h60_paired_policy_summary.csv", index=False)
    comparison.to_csv(OUT_DIR / "horizon_comparison.csv", index=False)
    (OUT_DIR / "decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    write_notebook()
    write_doc(decision, support, folds, quality, policies, comparison)
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
