"""Plot the converged neural-width scaling experiment for the journal paper.

The source rows live in ``padjective.paper_revision_neural_sizes`` on the
Shopify Postgres database.  This module aggregates the five cross-validation
folds for each hidden-layer width, fits the same log-loss/log-parameter form
used by the paper's active-support figure, and renders a publication figure.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from psycopg import sql
from scipy import stats

from . import data_access, db
from .benchmark_bundle import load_bundle
from .build_site import (
    _build_active_params_regression_frame,
    _fit_active_params_regression,
)


FIGURE_ONE_ACTIVE_SUPPORT_SLOPE = -0.1916
PARAMETER_SCALE = 100_000.0


@dataclass(frozen=True)
class NeuralWidthSummary:
    """Fold-aggregated result for one hidden-layer width."""

    hidden_units: int
    fold_count: int
    mean_params: float
    mean_active_support: float
    mean_loss: float
    loss_sd: float
    mean_iterations: float
    min_iterations: int
    max_iterations: int
    all_converged: bool


@dataclass(frozen=True)
class ScalingFit:
    """Ordinary least-squares fit of log10 loss against a size transform."""

    slope: float
    intercept: float
    r_squared: float
    p_value: float
    slope_stderr: float
    slope_ci_low: float
    slope_ci_high: float
    observation_count: int
    x_transform: str


def load_neural_width_summaries(
    conn,
    *,
    schema: str,
    snapshot_ref: str,
    max_iterations: int,
    seed: int,
) -> list[NeuralWidthSummary]:
    """Load and fold-average one neural-width experiment from Postgres."""

    snapshot_id, _ = data_access._resolve_snapshot_id(
        conn,
        schema=schema,
        snapshot_ref=snapshot_ref,
    )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH input_dimension AS (
                    SELECT COUNT(*)::DOUBLE PRECISION AS feature_count
                    FROM {schema}.product_taxonomy_bench_tags
                    WHERE snapshot_id = %s
                ), per_product AS (
                    SELECT
                        products.cv_fold,
                        products.product_id_hash,
                        COUNT(DISTINCT product_tags.tag_id)::DOUBLE PRECISION
                            AS active_features
                    FROM {schema}.product_taxonomy_bench_products AS products
                    LEFT JOIN {schema}.product_taxonomy_bench_product_tags AS product_tags
                      ON product_tags.snapshot_id = products.snapshot_id
                     AND product_tags.product_id_hash = products.product_id_hash
                    WHERE products.snapshot_id = %s
                    GROUP BY products.cv_fold, products.product_id_hash
                ), fold_features AS (
                    SELECT cv_fold, AVG(active_features) AS mean_active_features
                    FROM per_product
                    GROUP BY cv_fold
                ), fold_results AS (
                    SELECT
                        results.*,
                        results.num_params
                            - results.hidden_units
                              * (input_dimension.feature_count - fold_features.mean_active_features)
                            AS mean_active_support
                    FROM {schema}.paper_revision_neural_sizes AS results
                    JOIN fold_features USING (cv_fold)
                    CROSS JOIN input_dimension
                    WHERE results.snapshot_ref = %s
                      AND results.max_iterations = %s
                      AND results.seed = %s
                )
                SELECT
                    hidden_units,
                    COUNT(*) AS fold_count,
                    AVG(num_params)::DOUBLE PRECISION AS mean_params,
                    AVG(mean_active_support)::DOUBLE PRECISION AS mean_active_support,
                    AVG(mean_loss)::DOUBLE PRECISION AS mean_loss,
                    COALESCE(STDDEV_SAMP(mean_loss), 0)::DOUBLE PRECISION AS loss_sd,
                    AVG(iterations_used)::DOUBLE PRECISION AS mean_iterations,
                    MIN(iterations_used) AS min_iterations,
                    MAX(iterations_used) AS max_iterations,
                    BOOL_AND(converged) AS all_converged
                FROM fold_results
                GROUP BY hidden_units
                ORDER BY hidden_units
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id, snapshot_id, snapshot_ref, max_iterations, seed),
        )
        rows = cur.fetchall()

    return [
        NeuralWidthSummary(
            hidden_units=int(row[0]),
            fold_count=int(row[1]),
            mean_params=float(row[2]),
            mean_active_support=float(row[3]),
            mean_loss=float(row[4]),
            loss_sd=float(row[5]),
            mean_iterations=float(row[6]),
            min_iterations=int(row[7]),
            max_iterations=int(row[8]),
            all_converged=bool(row[9]),
        )
        for row in rows
    ]


def validate_neural_width_summaries(
    summaries: Sequence[NeuralWidthSummary],
) -> None:
    """Reject a sparse, invalid, or unconverged scaling series."""

    if len(summaries) < 3:
        raise ValueError("Need at least three neural widths for a scaling fit")
    if len({row.mean_params for row in summaries}) != len(summaries):
        raise ValueError("Neural-width parameter counts must be distinct")
    if any(
        row.mean_params <= 0
        or row.mean_active_support <= 0
        or row.mean_loss <= 0
        for row in summaries
    ):
        raise ValueError(
            "Neural-width parameters, active support, and losses must be positive"
        )
    unconverged = [row.hidden_units for row in summaries if not row.all_converged]
    if unconverged:
        widths = ", ".join(str(width) for width in unconverged)
        raise ValueError(f"Neural fits did not converge for hidden widths: {widths}")


def fit_neural_scaling(
    summaries: Sequence[NeuralWidthSummary],
    *,
    log_parameters: bool,
) -> ScalingFit:
    """Fit log10(mean loss) against log10(params) or params/100,000."""

    validate_neural_width_summaries(summaries)
    parameters = np.asarray([row.mean_params for row in summaries], dtype=float)
    losses = np.asarray([row.mean_loss for row in summaries], dtype=float)
    if log_parameters:
        x_values = np.log10(parameters)
        x_transform = "log10(parameters)"
    else:
        x_values = parameters / PARAMETER_SCALE
        x_transform = "parameters/100000"
    y_values = np.log10(losses)
    regression = stats.linregress(x_values, y_values)
    degrees_of_freedom = len(summaries) - 2
    critical_t = float(stats.t.ppf(0.975, degrees_of_freedom))
    margin = critical_t * float(regression.stderr)
    return ScalingFit(
        slope=float(regression.slope),
        intercept=float(regression.intercept),
        r_squared=float(regression.rvalue**2),
        p_value=float(regression.pvalue),
        slope_stderr=float(regression.stderr),
        slope_ci_low=float(regression.slope - margin),
        slope_ci_high=float(regression.slope + margin),
        observation_count=len(summaries),
        x_transform=x_transform,
    )


def fit_neural_active_support_scaling(
    summaries: Sequence[NeuralWidthSummary],
) -> ScalingFit:
    """Fit log10(mean loss) against log10(mean active support)."""

    validate_neural_width_summaries(summaries)
    active_support = np.asarray(
        [row.mean_active_support for row in summaries],
        dtype=float,
    )
    losses = np.asarray([row.mean_loss for row in summaries], dtype=float)
    regression = stats.linregress(np.log10(active_support), np.log10(losses))
    degrees_of_freedom = len(summaries) - 2
    critical_t = float(stats.t.ppf(0.975, degrees_of_freedom))
    margin = critical_t * float(regression.stderr)
    return ScalingFit(
        slope=float(regression.slope),
        intercept=float(regression.intercept),
        r_squared=float(regression.rvalue**2),
        p_value=float(regression.pvalue),
        slope_stderr=float(regression.stderr),
        slope_ci_low=float(regression.slope - margin),
        slope_ci_high=float(regression.slope + margin),
        observation_count=len(summaries),
        x_transform="log10(active support)",
    )


def _format_p_value(value: float) -> str:
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def render_neural_scaling_figure(
    summaries: Sequence[NeuralWidthSummary],
    output_path: Path,
    *,
    reference_slope: float = FIGURE_ONE_ACTIVE_SUPPORT_SLOPE,
) -> ScalingFit:
    """Render the neural scaling plot and return its log-log fit."""

    fit = fit_neural_scaling(summaries, log_parameters=True)
    parameters = np.asarray([row.mean_params for row in summaries], dtype=float)
    losses = np.asarray([row.mean_loss for row in summaries], dtype=float)
    loss_sd = np.asarray([row.loss_sd for row in summaries], dtype=float)
    log_parameters = np.log10(parameters)
    log_losses = np.log10(losses)

    fig, ax = plt.subplots(figsize=(7.1, 4.5), dpi=180)
    ax.errorbar(
        parameters,
        losses,
        yerr=loss_sd,
        fmt="o",
        color="#0b6ce3",
        ecolor="#78aef2",
        elinewidth=1.3,
        capsize=3,
        markersize=7,
        markeredgecolor="#083b7a",
        markeredgewidth=0.8,
        label="Width means (error bars: fold SD)",
        zorder=3,
    )

    x_range = np.logspace(
        float(log_parameters.min()) - 0.06,
        float(log_parameters.max()) + 0.06,
        200,
    )
    fitted_loss = 10 ** (fit.intercept + fit.slope * np.log10(x_range))
    ax.plot(
        x_range,
        fitted_loss,
        color="#0b6ce3",
        linewidth=2.0,
        label=(
            f"Neural fit: slope {fit.slope:.3f}; "
            f"$R^2$={fit.r_squared:.3f}, p={_format_p_value(fit.p_value)}"
        ),
        zorder=2,
    )

    centre_x = float(log_parameters.mean())
    centre_y = float(log_losses.mean())
    reference_loss = 10 ** (centre_y + reference_slope * (np.log10(x_range) - centre_x))
    ax.plot(
        x_range,
        reference_loss,
        color="#4b5563",
        linewidth=1.6,
        linestyle="--",
        label=f"Figure 1 slope {reference_slope:.4f} (centred reference)",
        zorder=1,
    )

    for row in summaries:
        ax.annotate(
            f"{row.hidden_units} hidden",
            (row.mean_params, row.mean_loss),
            textcoords="offset points",
            xytext=(5, 7 if row.hidden_units != 48 else -15),
            fontsize=8,
            color="#111827",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total trained parameters (log scale)", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Mean p-adic loss (log scale; lower is better)", fontsize=10.5, fontweight="bold")
    ax.set_title(
        "Neural-network width, parameters and mean p-adic loss",
        fontsize=12.5,
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0.5,
        1.015,
        "Fixed paper snapshot; five widths x five cross-validation folds",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#4b5563",
    )
    ax.grid(True, which="major", color="#d1d5db", linewidth=0.8, linestyle=":")
    ax.grid(False, which="minor")
    ax.legend(loc="upper right", frameon=False, fontsize=7.7)
    ax.tick_params(axis="both", which="major", labelsize=8.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return fit


def _build_figure_one_cross_model_frame(
    bundle: dict[str, Any],
    summaries: Sequence[NeuralWidthSummary],
):
    """Return the six distinct cross-model points used by Figure 1."""

    width_12 = next((row for row in summaries if row.hidden_units == 12), None)
    if width_12 is None:
        raise ValueError("Figure 1 requires the 12-hidden-unit neural result")

    working_bundle = deepcopy(bundle)
    for row in working_bundle.get("models", {}).get("rows", []):
        if row.get("model_key") == "umllr":
            row["model_label"] = "Greedy taxonomy-count p-adic Linear Regression"
            row["short_label"] = "Greedy taxonomy-count"
        if row.get("model_key") == "unn":
            recorded_support = row.get("mean_scoring_ops")
            if recorded_support is not None and not math.isclose(
                float(recorded_support),
                width_12.mean_active_support,
                rel_tol=1e-5,
                abs_tol=0.05,
            ):
                raise ValueError(
                    "Derived 12-unit active support does not match the benchmark row: "
                    f"{width_12.mean_active_support:.6f} vs {float(recorded_support):.6f}"
                )
            row["mean_scoring_ops"] = width_12.mean_active_support
            row["mean_padic_loss"] = width_12.mean_loss

    chart_frame = _build_active_params_regression_frame(working_bundle)
    chart_frame = chart_frame.loc[chart_frame["model_key"] != "zubarev"].copy()
    if "unn" not in set(chart_frame["model_key"]):
        raise ValueError("Figure 1 benchmark bundle has no unconstrained neural row")
    return chart_frame


def render_figure_one_with_neural_widths(
    bundle: dict[str, Any],
    summaries: Sequence[NeuralWidthSummary],
    output_path: Path,
) -> tuple[dict[str, float], ScalingFit]:
    """Render Figure 1 with the neural-width sweep on its active-support axes."""

    validate_neural_width_summaries(summaries)
    chart_frame = _build_figure_one_cross_model_frame(bundle, summaries)
    cross_model_fit = _fit_active_params_regression(chart_frame, log_loss=True)
    if cross_model_fit is None:
        raise ValueError("Not enough distinct cross-model points for Figure 1")
    neural_fit = fit_neural_active_support_scaling(summaries)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    for row in chart_frame.itertuples(index=False):
        if row.model_key == "unn":
            continue
        active_value = float(row.mean_scoring_ops)
        log_loss = float(row.log10_mean_padic_loss)
        marker = getattr(row, "marker", "o")
        scatter_kwargs: dict[str, Any] = {
            "color": getattr(row, "color", "#0b6ce3"),
            "s": 150,
            "alpha": 0.9,
            "marker": marker,
            "zorder": 3,
        }
        if marker not in {"+", "x", ".", ","}:
            scatter_kwargs["edgecolors"] = "white"
            scatter_kwargs["linewidths"] = 2
        ax.scatter(active_value, log_loss, **scatter_kwargs)
        ax.annotate(
            str(getattr(row, "short_label", row.model_label)),
            (active_value, log_loss),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            fontweight="bold",
            color=getattr(row, "color", "#0b6ce3"),
        )

    cross_x = np.linspace(
        cross_model_fit["x_min"] - 0.1,
        cross_model_fit["x_max"] + 0.1,
        200,
    )
    cross_y = cross_model_fit["intercept"] + cross_model_fit["slope"] * cross_x
    ax.plot(
        10**cross_x,
        cross_y,
        color="#111827",
        linestyle="--",
        linewidth=2.2,
        alpha=0.8,
        label=(
            f"Cross-model fit: slope {cross_model_fit['slope']:.3f}; "
            f"$R^2$={cross_model_fit['r_squared']:.3f}, "
            f"p={_format_p_value(cross_model_fit['p_value'])}"
        ),
        zorder=1,
    )

    active_support = np.asarray(
        [row.mean_active_support for row in summaries],
        dtype=float,
    )
    losses = np.asarray([row.mean_loss for row in summaries], dtype=float)
    loss_sd = np.asarray([row.loss_sd for row in summaries], dtype=float)
    log_losses = np.log10(losses)
    lower_loss = np.maximum(losses - loss_sd, np.finfo(float).tiny)
    log_error = np.vstack(
        [
            log_losses - np.log10(lower_loss),
            np.log10(losses + loss_sd) - log_losses,
        ]
    )
    ax.errorbar(
        active_support,
        log_losses,
        yerr=log_error,
        fmt="o",
        linestyle="none",
        color="#db2777",
        ecolor="#f09ac3",
        elinewidth=1.4,
        capsize=3,
        markersize=8,
        markerfacecolor="white",
        markeredgecolor="#db2777",
        markeredgewidth=2,
        zorder=4,
    )

    neural_x = np.linspace(
        float(np.log10(active_support).min()) - 0.04,
        float(np.log10(active_support).max()) + 0.04,
        200,
    )
    neural_y = neural_fit.intercept + neural_fit.slope * neural_x
    ax.plot(
        10**neural_x,
        neural_y,
        color="#db2777",
        linewidth=2.2,
        label=(
            f"Neural-width fit: slope {neural_fit.slope:.3f}; "
            f"$R^2$={neural_fit.r_squared:.3f}, "
            f"p={_format_p_value(neural_fit.p_value)}"
        ),
        zorder=2,
    )

    label_offsets = {
        4: (8, 7),
        8: (8, 5),
        12: (8, -14),
        24: (-16, -15),
        48: (8, 6),
    }
    for row in summaries:
        ax.annotate(
            f"NN-{row.hidden_units}",
            (row.mean_active_support, math.log10(row.mean_loss)),
            textcoords="offset points",
            xytext=label_offsets.get(row.hidden_units, (8, 5)),
            fontsize=8.5,
            fontweight="bold",
            color="#db2777",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        "Avg active params / classification (log scale)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel(
        "log10(mean p-adic loss) (lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        "Active Parameters vs log10 Mean p-adic Loss",
        fontsize=14,
        fontweight="bold",
        pad=26,
    )
    ax.text(
        0.5,
        1.015,
        "Six cross-model configurations; neural widths are five-fold means with fold SD",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#4b5563",
    )
    ax.grid(True, which="major", color="#d1d5db", alpha=0.9, linestyle="--")
    ax.grid(True, which="minor", color="#e5e7eb", alpha=0.65, linestyle=":")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return cross_model_fit, neural_fit


def write_neural_scaling_tex(
    output_path: Path,
    *,
    log_log_fit: ScalingFit,
    semi_log_fit: ScalingFit,
    active_support_fit: ScalingFit,
) -> None:
    """Write stable LaTeX macros for the manuscript's reported statistics."""

    lines = [
        f"\\newcommand{{\\PadNeuralScalingSlope}}{{{log_log_fit.slope:.4f}}}",
        f"\\newcommand{{\\PadNeuralScalingIntercept}}{{{log_log_fit.intercept:.4f}}}",
        f"\\newcommand{{\\PadNeuralScalingRSquared}}{{{log_log_fit.r_squared:.3f}}}",
        f"\\newcommand{{\\PadNeuralScalingPValue}}{{{_format_p_value(log_log_fit.p_value)}}}",
        f"\\newcommand{{\\PadNeuralScalingSlopeCILow}}{{{log_log_fit.slope_ci_low:.4f}}}",
        f"\\newcommand{{\\PadNeuralScalingSlopeCIHigh}}{{{log_log_fit.slope_ci_high:.4f}}}",
        f"\\newcommand{{\\PadNeuralSemiLogSlope}}{{{semi_log_fit.slope:.4f}}}",
        f"\\newcommand{{\\PadNeuralSemiLogRSquared}}{{{semi_log_fit.r_squared:.3f}}}",
        f"\\newcommand{{\\PadNeuralSemiLogPValue}}{{{_format_p_value(semi_log_fit.p_value)}}}",
        f"\\newcommand{{\\PadNeuralActiveSlope}}{{{active_support_fit.slope:.4f}}}",
        f"\\newcommand{{\\PadNeuralActiveIntercept}}{{{active_support_fit.intercept:.4f}}}",
        f"\\newcommand{{\\PadNeuralActiveRSquared}}{{{active_support_fit.r_squared:.3f}}}",
        f"\\newcommand{{\\PadNeuralActivePValue}}{{{_format_p_value(active_support_fit.p_value)}}}",
        f"\\newcommand{{\\PadNeuralActiveSlopeCILow}}{{{active_support_fit.slope_ci_low:.4f}}}",
        f"\\newcommand{{\\PadNeuralActiveSlopeCIHigh}}{{{active_support_fit.slope_ci_high:.4f}}}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the converged paper neural-width scaling experiment from Postgres."
    )
    parser.add_argument("--dsn", help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)")
    parser.add_argument("--schema", default="padjective")
    parser.add_argument("--snapshot-ref", default="paper")
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tex-output", type=Path)
    parser.add_argument(
        "--figure-one-bundle",
        type=Path,
        help="Benchmark bundle used to render the combined active-support Figure 1.",
    )
    parser.add_argument(
        "--figure-one-slope",
        type=float,
        default=FIGURE_ONE_ACTIVE_SUPPORT_SLOPE,
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        summaries = load_neural_width_summaries(
            conn,
            schema=args.schema,
            snapshot_ref=args.snapshot_ref,
            max_iterations=args.max_iterations,
            seed=args.seed,
        )
    finally:
        conn.close()

    if args.figure_one_bundle is not None:
        cross_model_fit, active_support_fit = render_figure_one_with_neural_widths(
            load_bundle(args.figure_one_bundle),
            summaries,
            args.output,
        )
    else:
        cross_model_fit = None
        active_support_fit = fit_neural_active_support_scaling(summaries)
        render_neural_scaling_figure(
            summaries,
            args.output,
            reference_slope=args.figure_one_slope,
        )
    log_log_fit = fit_neural_scaling(summaries, log_parameters=True)
    semi_log_fit = fit_neural_scaling(summaries, log_parameters=False)
    if args.tex_output is not None:
        write_neural_scaling_tex(
            args.tex_output,
            log_log_fit=log_log_fit,
            semi_log_fit=semi_log_fit,
            active_support_fit=active_support_fit,
        )

    print(
        "Log-log fit: log10(mean loss) = "
        f"{log_log_fit.slope:.6f} log10(parameters) "
        f"{log_log_fit.intercept:+.6f}; R^2={log_log_fit.r_squared:.6f}; "
        f"p={log_log_fit.p_value:.6f}; 95% slope CI "
        f"[{log_log_fit.slope_ci_low:.6f}, {log_log_fit.slope_ci_high:.6f}]"
    )
    print(
        "Semi-log fit: log10(mean loss) = "
        f"{semi_log_fit.slope:.6f} (parameters/100000) "
        f"{semi_log_fit.intercept:+.6f}; R^2={semi_log_fit.r_squared:.6f}; "
        f"p={semi_log_fit.p_value:.6f}"
    )
    print(
        f"Figure 1 reference slope: {args.figure_one_slope:.6f}; "
        f"inside neural 95% slope CI: "
        f"{log_log_fit.slope_ci_low <= args.figure_one_slope <= log_log_fit.slope_ci_high}"
    )
    print(
        "Neural active-support fit: log10(mean loss) = "
        f"{active_support_fit.slope:.6f} log10(active support) "
        f"{active_support_fit.intercept:+.6f}; "
        f"R^2={active_support_fit.r_squared:.6f}; "
        f"p={active_support_fit.p_value:.6f}; 95% slope CI "
        f"[{active_support_fit.slope_ci_low:.6f}, "
        f"{active_support_fit.slope_ci_high:.6f}]"
    )
    if cross_model_fit is not None:
        print(
            "Figure 1 cross-model fit: log10(mean loss) = "
            f"{cross_model_fit['slope']:.6f} log10(active support) "
            f"{cross_model_fit['intercept']:+.6f}; "
            f"R^2={cross_model_fit['r_squared']:.6f}; "
            f"p={cross_model_fit['p_value']:.6f}"
        )


if __name__ == "__main__":  # pragma: no cover
    main()
