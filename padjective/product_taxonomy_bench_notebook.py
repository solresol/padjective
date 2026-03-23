"""Generate the autonomous product-taxonomy-bench notebook."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from . import benchmark_runtime


NOTEBOOK_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}


def _markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def _code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


def render_notebook() -> dict:
    runtime_source = inspect.getsource(benchmark_runtime).strip()
    cells = [
        _markdown_cell(
            """
# product-taxonomy-bench: reproducible baselines + UMLLR ablations

This notebook loads an anonymised snapshot directly from Hugging Face and reruns:

- Dummy baseline
- Importance-optimised $p$-adic linear regression (UMLLR-style)
- UMLLR tag-order ablations
- Parameter-constrained logistic regression
- Parameter-constrained neural network
- Unconstrained logistic regression with L1
- Unconstrained neural network
- Decision tree
- Level-wise logistic regression
- Zubarev simulated-annealing $p$-adic regression

The benchmark runtime is embedded directly in this notebook so it runs without
Postgres access or local project imports. By default the notebook analyses the
fixed `paper` snapshot; set `PRODUCT_TAXONOMY_BENCH_SNAPSHOT=latest` to inspect
the rolling benchmark release instead.
            """
        ),
        _code_cell(runtime_source),
        _code_cell(
            """
import os

DATASET_ID = os.getenv("PRODUCT_TAXONOMY_BENCH_DATASET_ID", "gregb/product-taxonomy-bench")
REVISION = os.getenv("PRODUCT_TAXONOMY_BENCH_REVISION", "main")
SNAPSHOT = os.getenv("PRODUCT_TAXONOMY_BENCH_SNAPSHOT", "paper")
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_PRODUCTS = int(os.getenv("PRODUCT_TAXONOMY_BENCH_MAX_PRODUCTS", "0")) or None

snapshot_tables = load_snapshot_tables_from_hf(
    dataset_id=DATASET_ID,
    revision=REVISION,
    snapshot=SNAPSHOT,
    hf_token=HF_TOKEN,
    max_products=MAX_PRODUCTS,
)
benchmark = build_snapshot_benchmark_bundle(snapshot_tables)

model_results = pd.DataFrame(benchmark["models"]["rows"]).copy()
ablation_results = pd.DataFrame(benchmark["ablation"]["strategy_rows"]).copy()
random_runs = pd.DataFrame(
    [row for row in benchmark["ablation"]["runs"] if row["tag_order_strategy"] == "random"]
).copy()

print("snapshot", benchmark["snapshot"]["snapshot_name"])
print("filtered products", benchmark["snapshot"]["product_count_filtered"])
print("filtered tags", benchmark["snapshot"]["tag_count_filtered"])
print("filtered taxonomies", benchmark["snapshot"]["taxonomy_count_filtered"])
print("prime base", benchmark["snapshot"]["prime_base"])
print("tag battles", benchmark["snapshot"]["battle_count"])
            """
        ),
        _code_cell(
            """
# Model complexity vs p-adic loss (parsimoniousness baseline)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

for row in model_results.itertuples(index=False):
    scatter_kwargs = {
        "label": row.model_label,
        "color": row.color,
        "s": 150,
        "alpha": 0.8,
        "marker": row.marker,
    }
    if row.marker not in ["+", "x", ".", ","]:
        scatter_kwargs["edgecolors"] = "white"
        scatter_kwargs["linewidths"] = 2
    ax.scatter(row.params, row.mean_padic_loss, **scatter_kwargs)
    ax.annotate(
        row.short_label,
        (row.params, row.mean_padic_loss),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
        fontweight="bold",
        color=row.color,
    )

x_range = np.linspace(
    model_results["log10_params"].min() - 0.3,
    model_results["log10_params"].max() + 0.3,
    200,
)
baseline_log10_loss = (
    -0.1 * x_range
    + 0.3 * np.log10(max(float(benchmark["snapshot"]["taxonomy_count_filtered"]), 1.0) / 1000.0)
)
ax.plot(
    10 ** x_range,
    10 ** baseline_log10_loss,
    "-",
    color="#ef4444",
    linewidth=2.2,
    alpha=0.8,
    label=f"Parsimoniousness baseline ({benchmark['snapshot']['taxonomy_count_filtered']:,} taxonomies)",
)

ax.set_xlabel("Number of Parameters (non-zero)", fontsize=12, fontweight="bold")
ax.set_ylabel("P-adic Loss (lower is better)", fontsize=12, fontweight="bold")
ax.set_title("Model Complexity vs Performance (Parsimoniousness Baseline)", fontsize=14, fontweight="bold", pad=15)
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="lower left", frameon=True, shadow=True, fontsize=8)

plt.tight_layout()
plt.show()
            """
        ),
        _code_cell(
            """
# UMLLR tag-order ablation

ablation_plot = ablation_results.sort_values("mean_padic_loss", ascending=True).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
ax.bar(ablation_plot["tag_order_strategy"], ablation_plot["mean_padic_loss"], color="#0b6ce3")
ax.axhline(
    benchmark["ablation"]["baseline_mean_padic_loss"],
    color="#ef4444",
    linestyle="--",
    linewidth=1.5,
    label="battle_elo baseline",
)
ax.set_ylabel("Mean p-adic loss")
ax.set_title("UMLLR Tag-order Ablation")
ax.tick_params(axis="x", rotation=20)
ax.legend()
plt.tight_layout()
plt.show()

if not random_runs.empty:
    random_runs[["run_key", "mean_padic_loss", "mean_exact_accuracy", "mean_prefix2_accuracy"]]
            """
        ),
        _code_cell(
            """
# Final tables

results_table = model_results[
    [
        "model_label",
        "params",
        "mean_padic_loss",
        "mean_exact_accuracy",
        "mean_prefix1_accuracy",
        "mean_prefix2_accuracy",
        "mean_shared_prefix_depth",
        "mean_scoring_ops",
        "parsimony_score",
    ]
].copy()

results_table["params"] = results_table["params"].round(1)
results_table["mean_padic_loss"] = results_table["mean_padic_loss"].round(6)
results_table["mean_exact_accuracy"] = results_table["mean_exact_accuracy"].round(4)
results_table["mean_prefix1_accuracy"] = results_table["mean_prefix1_accuracy"].round(4)
results_table["mean_prefix2_accuracy"] = results_table["mean_prefix2_accuracy"].round(4)
results_table["mean_shared_prefix_depth"] = results_table["mean_shared_prefix_depth"].round(4)
results_table["mean_scoring_ops"] = results_table["mean_scoring_ops"].round(4)
results_table["parsimony_score"] = results_table["parsimony_score"].round(4)

ablation_table = ablation_results[
    [
        "tag_order_strategy",
        "mean_padic_loss",
        "loss_delta_vs_baseline",
        "wins_vs_baseline",
        "comparisons_vs_baseline",
        "mean_exact_accuracy",
        "mean_prefix2_accuracy",
        "mean_scoring_ops",
    ]
].copy()

ablation_table["mean_padic_loss"] = ablation_table["mean_padic_loss"].round(6)
ablation_table["loss_delta_vs_baseline"] = ablation_table["loss_delta_vs_baseline"].round(6)
ablation_table["mean_exact_accuracy"] = ablation_table["mean_exact_accuracy"].round(4)
ablation_table["mean_prefix2_accuracy"] = ablation_table["mean_prefix2_accuracy"].round(4)
ablation_table["mean_scoring_ops"] = ablation_table["mean_scoring_ops"].round(4)

print("Best ablation strategy:", benchmark["narrative"]["best_ablation_strategy"])
print("Best mean loss:", round(float(benchmark["narrative"]["best_ablation_mean_padic_loss"]), 6))
print("Delta vs battle_elo:", round(float(benchmark["narrative"]["best_ablation_delta_vs_baseline"]), 6))

results_table.sort_values("parsimony_score", ascending=False).reset_index(drop=True)
            """
        ),
        _code_cell(
            """
ablation_table.sort_values("mean_padic_loss", ascending=True).reset_index(drop=True)
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": NOTEBOOK_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(render_notebook(), indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate product_taxonomy_bench.ipynb")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/product_taxonomy_bench.ipynb"),
        help="Where to write the generated notebook.",
    )
    args = parser.parse_args()
    write_notebook(args.output)


if __name__ == "__main__":
    main()
