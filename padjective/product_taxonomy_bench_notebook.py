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
- Non-zero-parameter and active-parameter trade-off charts
- UMLLR tag-order ablations
- Parameter-constrained logistic regression
- Parameter-constrained neural network
- Unconstrained logistic regression with L1
- Unconstrained neural network
- Decision tree
- Level-wise logistic regression
- Zubarev-inspired stochastic continuation of the greedy linear fit

The benchmark runtime is embedded directly in this notebook so it runs without
Postgres access or local project imports. By default the notebook analyses the
fixed `paper` snapshot; set `PRODUCT_TAXONOMY_BENCH_SNAPSHOT=latest` to inspect
the rolling benchmark release instead (and select a revision containing it).
The paper uses 2,000 hidden units, selected after the width sweep, and a 10,000
iteration ceiling. The stochastic search uses raw training scores and fits the
reporting default afterwards. The separate bounded digitwise experiment is
documented at the end; its population is not the main benchmark population.
            """
        ),
        _code_cell(runtime_source),
        _code_cell(
            """
import os

DATASET_ID = os.getenv("PRODUCT_TAXONOMY_BENCH_DATASET_ID", "gregb/product-taxonomy-bench")
REVISION = os.getenv("PRODUCT_TAXONOMY_BENCH_REVISION", "paper-submission-2026-09-06")
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
# Paper reference: largest tested width, selected after the width sweep.
# Override explicitly for a shorter exploratory run; it is then not the paper run.
DEFAULT_UNN_HIDDEN = int(os.getenv("PRODUCT_TAXONOMY_BENCH_UNN_HIDDEN", "2000"))
DEFAULT_UNN_MAX_ITER = int(os.getenv("PRODUCT_TAXONOMY_BENCH_UNN_MAX_ITER", "10000"))
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
# Active parameters vs p-adic loss

from scipy import stats

# The matched-budget models are separate feature-dropping ablations. The
# stochastic continuation duplicates the greedy point in this paper run.
ACTIVE_PARAMS_EXCLUDED = {"pclr", "pcnn", "zubarev"}

active_results = model_results[
    ~model_results["model_key"].isin(ACTIVE_PARAMS_EXCLUDED)
].copy()
active_results["mean_scoring_ops"] = pd.to_numeric(
    active_results["mean_scoring_ops"],
    errors="coerce",
)
active_results["mean_padic_loss"] = pd.to_numeric(
    active_results["mean_padic_loss"],
    errors="coerce",
)
active_results = active_results[
    active_results["mean_scoring_ops"].gt(0)
    & active_results["mean_padic_loss"].gt(0)
].copy()
active_results["log10_mean_padic_loss"] = np.log10(active_results["mean_padic_loss"])

log10_active = np.log10(active_results["mean_scoring_ops"].to_numpy(dtype=float))
raw_regression = stats.linregress(
    log10_active,
    active_results["mean_padic_loss"].to_numpy(dtype=float),
)
log_regression = stats.linregress(
    log10_active,
    active_results["log10_mean_padic_loss"].to_numpy(dtype=float),
)

def regression_formula(regression, *, target_name):
    intercept_sign = "+" if regression.intercept >= 0 else "-"
    return (
        f"{target_name} = {regression.slope:.4f} log10(active params) "
        f"{intercept_sign} {abs(regression.intercept):.4f}"
    )

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150, sharex=True)
plot_specs = [
    (
        axes[0],
        "mean_padic_loss",
        "Mean p-adic loss (lower is better)",
        "Active Parameters vs Mean p-adic Loss",
        raw_regression,
        "mean p-adic loss",
    ),
    (
        axes[1],
        "log10_mean_padic_loss",
        "log10(mean p-adic loss)",
        "Active Parameters vs log10 Mean p-adic Loss",
        log_regression,
        "log10(mean p-adic loss)",
    ),
]

for ax, y_column, y_label, title, regression, target_name in plot_specs:
    for row in active_results.itertuples(index=False):
        scatter_kwargs = {
            "color": row.color,
            "s": 150,
            "alpha": 0.85,
            "marker": row.marker,
        }
        if row.marker not in ["+", "x", ".", ","]:
            scatter_kwargs["edgecolors"] = "white"
            scatter_kwargs["linewidths"] = 2
        y_value = getattr(row, y_column)
        ax.scatter(row.mean_scoring_ops, y_value, **scatter_kwargs)
        ax.annotate(
            row.short_label,
            (row.mean_scoring_ops, y_value),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            fontweight="bold",
            color=row.color,
        )

    x_range = np.linspace(log10_active.min() - 0.1, log10_active.max() + 0.1, 200)
    y_range = regression.intercept + regression.slope * x_range
    ax.plot(
        10 ** x_range,
        y_range,
        color="#111827",
        linestyle="--",
        linewidth=2.2,
        alpha=0.75,
        label=(
            f"y = {regression.slope:+.3f}x "
            f"{'+' if regression.intercept >= 0 else '-'} {abs(regression.intercept):.3f}; "
            f"R^2={regression.rvalue ** 2:.3f}, p={regression.pvalue:.3f}"
        ),
    )
    ax.set_xscale("log")
    ax.set_xlabel("Avg active params / classification (log scale)", fontsize=11, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, shadow=True, fontsize=8)

plt.tight_layout()
plt.show()

print("Excluded from active-parameter regressions:", ", ".join(sorted(ACTIVE_PARAMS_EXCLUDED)))
print("Fitted equation:", regression_formula(raw_regression, target_name="mean p-adic loss"))
print(
    "Raw-loss regression stats:",
    f"R^2={raw_regression.rvalue ** 2:.3f}",
    f"p={raw_regression.pvalue:.3f}",
)
print(
    "Fitted equation:",
    regression_formula(log_regression, target_name="log10(mean p-adic loss)"),
)
print(
    "Log-loss regression stats:",
    f"R^2={log_regression.rvalue ** 2:.3f}",
    f"p={log_regression.pvalue:.3f}",
)

pd.DataFrame(
    [
        {
            "target": "mean p-adic loss",
            "equation": regression_formula(raw_regression, target_name="mean p-adic loss"),
            "r_squared": round(float(raw_regression.rvalue ** 2), 4),
            "p_value": round(float(raw_regression.pvalue), 4),
        },
        {
            "target": "log10(mean p-adic loss)",
            "equation": regression_formula(log_regression, target_name="log10(mean p-adic loss)"),
            "r_squared": round(float(log_regression.rvalue ** 2), 4),
            "p_value": round(float(log_regression.pvalue), 4),
        },
    ]
)
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
        _markdown_cell(
            """
## Additional paper experiments

The frozen release includes a standalone runner and pinned dependencies in
`submission/2026-09-06/`. Download that directory from the same revision and run:

```sh
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python paper_replication.py --suite neural --output neural.json
.venv/bin/python paper_replication.py --suite digitwise --output digitwise.json
```

The neural sweep uses widths 4, 8, 12, 24, 48 and 2,000. The digitwise runner
uses the separately released **6,527-product, 2,747-tag, 308-class** matrix,
not the main **6,693-product, 2,542-tag, 363-class** matrix used above. It is a
bounded Mihara-inspired diagnostic with 96 candidate trials per digit; its
90% agreement threshold is an experiment setting, not a published Mihara
acceptance criterion. The full dimension sweep can take many hours.
Use `--caps 32 --folds 0` for a short smoke test, not a paper replication.
See the release README and manifest for settings, provenance and checksums.
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
