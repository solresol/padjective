"""Plot the converged neural-width scaling experiment for the journal paper.

The source rows live in ``padjective.paper_revision_neural_sizes`` on the
Shopify Postgres database.  This module aggregates the five cross-validation
folds for each hidden-layer width, fits the same log-loss/log-parameter form
used by the paper's active-support figure, and renders a publication figure.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from psycopg import sql
from scipy import stats

from . import db


FIGURE_ONE_ACTIVE_SUPPORT_SLOPE = -0.1916
PARAMETER_SCALE = 100_000.0


@dataclass(frozen=True)
class NeuralWidthSummary:
    """Fold-aggregated result for one hidden-layer width."""

    hidden_units: int
    fold_count: int
    mean_params: float
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

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    hidden_units,
                    COUNT(*) AS fold_count,
                    AVG(num_params)::DOUBLE PRECISION AS mean_params,
                    AVG(mean_loss)::DOUBLE PRECISION AS mean_loss,
                    COALESCE(STDDEV_SAMP(mean_loss), 0)::DOUBLE PRECISION AS loss_sd,
                    AVG(iterations_used)::DOUBLE PRECISION AS mean_iterations,
                    MIN(iterations_used) AS min_iterations,
                    MAX(iterations_used) AS max_iterations,
                    BOOL_AND(converged) AS all_converged
                FROM {schema}.paper_revision_neural_sizes
                WHERE snapshot_ref = %s
                  AND max_iterations = %s
                  AND seed = %s
                GROUP BY hidden_units
                ORDER BY hidden_units
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_ref, max_iterations, seed),
        )
        rows = cur.fetchall()

    return [
        NeuralWidthSummary(
            hidden_units=int(row[0]),
            fold_count=int(row[1]),
            mean_params=float(row[2]),
            mean_loss=float(row[3]),
            loss_sd=float(row[4]),
            mean_iterations=float(row[5]),
            min_iterations=int(row[6]),
            max_iterations=int(row[7]),
            all_converged=bool(row[8]),
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
    if any(row.mean_params <= 0 or row.mean_loss <= 0 for row in summaries):
        raise ValueError("Neural-width parameters and losses must be positive")
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


def write_neural_scaling_tex(
    output_path: Path,
    *,
    log_log_fit: ScalingFit,
    semi_log_fit: ScalingFit,
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

    log_log_fit = render_neural_scaling_figure(
        summaries,
        args.output,
        reference_slope=args.figure_one_slope,
    )
    semi_log_fit = fit_neural_scaling(summaries, log_parameters=False)
    if args.tex_output is not None:
        write_neural_scaling_tex(
            args.tex_output,
            log_log_fit=log_log_fit,
            semi_log_fit=semi_log_fit,
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


if __name__ == "__main__":  # pragma: no cover
    main()
