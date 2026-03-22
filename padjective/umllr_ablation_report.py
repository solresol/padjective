"""Generate paper-ready summary tables for UMLLR tag-order ablations."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:  # pragma: no cover - imported as a package
    from . import db


DEFAULT_BASELINE_RUN_KEY = "battle_elo"
DEFAULT_SCHEMA = "padjective"
RANDOM_AGGREGATE_RUN_KEY = "__random_aggregate__"
STRATEGY_ORDER: tuple[str, ...] = (
    "battle_elo",
    "frequency",
    "mean_title_position",
    "taxonomy_association",
    "random",
)


@dataclass(frozen=True)
class AblationFoldRow:
    run_key: str
    tag_order_strategy: str
    tag_order_seed: int | None
    cv_fold: int
    mean_loss: float | None
    exact_accuracy: float | None
    prefix1_accuracy: float | None
    prefix2_accuracy: float | None
    mean_shared_prefix_depth: float | None
    mean_scoring_ops: float | None
    prediction_count: int


@dataclass(frozen=True)
class AblationRunSummary:
    run_key: str
    tag_order_strategy: str
    tag_order_seed: int | None
    fold_count: int
    matched_fold_count: int
    source_run_count: int
    mean_loss: float | None
    loss_std: float | None
    mean_exact_accuracy: float | None
    mean_prefix1_accuracy: float | None
    mean_prefix2_accuracy: float | None
    mean_shared_prefix_depth: float | None
    mean_scoring_ops: float | None
    mean_loss_delta_vs_baseline: float | None
    mean_exact_accuracy_delta_vs_baseline: float | None
    mean_prefix2_delta_vs_baseline: float | None
    mean_scoring_ops_delta_vs_baseline: float | None
    loss_better_folds: int


def _table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


def _safe_mean(values: Iterable[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    return float(statistics.fmean(cleaned))


def _population_std(values: Sequence[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return 0.0
    return float(statistics.pstdev(cleaned))


def _summary_sort_key(summary: AblationRunSummary) -> tuple[int, int, int, str]:
    try:
        strategy_rank = STRATEGY_ORDER.index(summary.tag_order_strategy)
    except ValueError:
        strategy_rank = len(STRATEGY_ORDER)

    seed_rank = -1 if summary.tag_order_seed is None else summary.tag_order_seed
    aggregate_rank = 1 if summary.source_run_count > 1 else 0
    return (strategy_rank, aggregate_rank, seed_rank, summary.run_key)


def _display_strategy(summary: AblationRunSummary) -> str:
    if summary.run_key == RANDOM_AGGREGATE_RUN_KEY:
        return f"random ({summary.source_run_count} seeds avg.)"
    return summary.tag_order_strategy


def load_ablation_fold_rows(conn, schema: str = DEFAULT_SCHEMA) -> list[AblationFoldRow]:
    required_tables = (
        "umllr_order_ablation_fold_metrics",
        "umllr_order_ablation_predictions",
    )
    missing_tables = [
        table for table in required_tables if not _table_exists(conn, schema, table)
    ]
    if missing_tables:
        joined = ", ".join(f"{schema}.{table}" for table in missing_tables)
        raise RuntimeError(
            f"Missing UMLLR ablation tables: {joined}. "
            "Run `uv run -m padjective.umllr --tag-order-strategy all` first."
        )

    query = sql.SQL(
        """
        SELECT
            m.run_key,
            m.tag_order_strategy,
            m.tag_order_seed,
            m.cv_fold,
            p.mean_loss,
            p.prediction_count,
            m.exact_accuracy,
            m.prefix1_accuracy,
            m.prefix2_accuracy,
            m.mean_shared_prefix_depth,
            m.mean_scoring_ops
        FROM {schema}.umllr_order_ablation_fold_metrics AS m
        LEFT JOIN (
            SELECT run_key, cv_fold, AVG(loss) AS mean_loss, COUNT(*) AS prediction_count
            FROM {schema}.umllr_order_ablation_predictions
            GROUP BY run_key, cv_fold
        ) AS p
            ON p.run_key = m.run_key
           AND p.cv_fold = m.cv_fold
        ORDER BY m.tag_order_strategy, m.tag_order_seed NULLS FIRST, m.cv_fold
        """
    ).format(schema=sql.Identifier(schema))

    rows: list[AblationFoldRow] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            rows.append(
                AblationFoldRow(
                    run_key=str(row["run_key"]),
                    tag_order_strategy=str(row["tag_order_strategy"]),
                    tag_order_seed=int(row["tag_order_seed"])
                    if row["tag_order_seed"] is not None
                    else None,
                    cv_fold=int(row["cv_fold"]),
                    mean_loss=float(row["mean_loss"])
                    if row["mean_loss"] is not None
                    else None,
                    exact_accuracy=float(row["exact_accuracy"])
                    if row["exact_accuracy"] is not None
                    else None,
                    prefix1_accuracy=float(row["prefix1_accuracy"])
                    if row["prefix1_accuracy"] is not None
                    else None,
                    prefix2_accuracy=float(row["prefix2_accuracy"])
                    if row["prefix2_accuracy"] is not None
                    else None,
                    mean_shared_prefix_depth=float(row["mean_shared_prefix_depth"])
                    if row["mean_shared_prefix_depth"] is not None
                    else None,
                    mean_scoring_ops=float(row["mean_scoring_ops"])
                    if row["mean_scoring_ops"] is not None
                    else None,
                    prediction_count=int(row["prediction_count"] or 0),
                )
            )

    if not rows:
        raise RuntimeError(
            "No UMLLR ablation rows found. Run `uv run -m padjective.umllr --tag-order-strategy all` first."
        )
    return rows


def summarize_ablation_rows(
    rows: Sequence[AblationFoldRow],
    *,
    baseline_run_key: str = DEFAULT_BASELINE_RUN_KEY,
) -> list[AblationRunSummary]:
    grouped: dict[str, list[AblationFoldRow]] = defaultdict(list)
    for row in rows:
        grouped[row.run_key].append(row)

    if baseline_run_key not in grouped:
        available = ", ".join(sorted(grouped))
        raise ValueError(
            f"Baseline run {baseline_run_key!r} is unavailable. Available runs: {available}"
        )

    baseline_by_fold = {
        row.cv_fold: row for row in grouped[baseline_run_key]
    }

    summaries: list[AblationRunSummary] = []
    for run_key, run_rows in grouped.items():
        exemplar = run_rows[0]
        matched_rows = [
            (row, baseline_by_fold[row.cv_fold])
            for row in run_rows
            if row.cv_fold in baseline_by_fold
        ]

        loss_deltas = [
            row.mean_loss - baseline.mean_loss
            for row, baseline in matched_rows
            if row.mean_loss is not None and baseline.mean_loss is not None
        ]
        exact_accuracy_deltas = [
            row.exact_accuracy - baseline.exact_accuracy
            for row, baseline in matched_rows
            if row.exact_accuracy is not None and baseline.exact_accuracy is not None
        ]
        prefix2_deltas = [
            row.prefix2_accuracy - baseline.prefix2_accuracy
            for row, baseline in matched_rows
            if row.prefix2_accuracy is not None and baseline.prefix2_accuracy is not None
        ]
        scoring_ops_deltas = [
            row.mean_scoring_ops - baseline.mean_scoring_ops
            for row, baseline in matched_rows
            if row.mean_scoring_ops is not None and baseline.mean_scoring_ops is not None
        ]
        loss_better_folds = sum(1 for delta in loss_deltas if delta < 0)

        summaries.append(
            AblationRunSummary(
                run_key=run_key,
                tag_order_strategy=exemplar.tag_order_strategy,
                tag_order_seed=exemplar.tag_order_seed,
                fold_count=len(run_rows),
                matched_fold_count=len(loss_deltas),
                source_run_count=1,
                mean_loss=_safe_mean(row.mean_loss for row in run_rows),
                loss_std=_population_std([row.mean_loss for row in run_rows]),
                mean_exact_accuracy=_safe_mean(row.exact_accuracy for row in run_rows),
                mean_prefix1_accuracy=_safe_mean(row.prefix1_accuracy for row in run_rows),
                mean_prefix2_accuracy=_safe_mean(row.prefix2_accuracy for row in run_rows),
                mean_shared_prefix_depth=_safe_mean(
                    row.mean_shared_prefix_depth for row in run_rows
                ),
                mean_scoring_ops=_safe_mean(row.mean_scoring_ops for row in run_rows),
                mean_loss_delta_vs_baseline=_safe_mean(loss_deltas),
                mean_exact_accuracy_delta_vs_baseline=_safe_mean(exact_accuracy_deltas),
                mean_prefix2_delta_vs_baseline=_safe_mean(prefix2_deltas),
                mean_scoring_ops_delta_vs_baseline=_safe_mean(scoring_ops_deltas),
                loss_better_folds=loss_better_folds,
            )
        )

    return sorted(summaries, key=_summary_sort_key)


def aggregate_random_summaries(
    summaries: Sequence[AblationRunSummary],
) -> AblationRunSummary | None:
    random_runs = [
        summary for summary in summaries if summary.tag_order_strategy == "random"
    ]
    if not random_runs:
        return None

    return AblationRunSummary(
        run_key=RANDOM_AGGREGATE_RUN_KEY,
        tag_order_strategy="random",
        tag_order_seed=None,
        fold_count=sum(summary.fold_count for summary in random_runs),
        matched_fold_count=sum(summary.matched_fold_count for summary in random_runs),
        source_run_count=len(random_runs),
        mean_loss=_safe_mean(summary.mean_loss for summary in random_runs),
        loss_std=_population_std([summary.mean_loss for summary in random_runs]),
        mean_exact_accuracy=_safe_mean(
            summary.mean_exact_accuracy for summary in random_runs
        ),
        mean_prefix1_accuracy=_safe_mean(
            summary.mean_prefix1_accuracy for summary in random_runs
        ),
        mean_prefix2_accuracy=_safe_mean(
            summary.mean_prefix2_accuracy for summary in random_runs
        ),
        mean_shared_prefix_depth=_safe_mean(
            summary.mean_shared_prefix_depth for summary in random_runs
        ),
        mean_scoring_ops=_safe_mean(summary.mean_scoring_ops for summary in random_runs),
        mean_loss_delta_vs_baseline=_safe_mean(
            summary.mean_loss_delta_vs_baseline for summary in random_runs
        ),
        mean_exact_accuracy_delta_vs_baseline=_safe_mean(
            summary.mean_exact_accuracy_delta_vs_baseline for summary in random_runs
        ),
        mean_prefix2_delta_vs_baseline=_safe_mean(
            summary.mean_prefix2_delta_vs_baseline for summary in random_runs
        ),
        mean_scoring_ops_delta_vs_baseline=_safe_mean(
            summary.mean_scoring_ops_delta_vs_baseline for summary in random_runs
        ),
        loss_better_folds=sum(summary.loss_better_folds for summary in random_runs),
    )


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _format_percent(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _format_seed(value: int | None) -> str:
    if value is None:
        return "—"
    return str(value)


def _format_wins(summary: AblationRunSummary) -> str:
    if summary.matched_fold_count <= 0:
        return "—"
    return f"{summary.loss_better_folds}/{summary.matched_fold_count}"


def _markdown_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    escaped = text
    for needle, replacement in replacements.items():
        escaped = escaped.replace(needle, replacement)
    return escaped


def _latex_table(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""
    header, *body = rows
    alignment = "llrrrrrrr"
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(_latex_escape(cell) for cell in header) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(_latex_escape(cell) for cell in row) + r" \\"
        for row in body
    )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines)


def _build_table_rows(summaries: Sequence[AblationRunSummary]) -> list[list[str]]:
    rows: list[list[str]] = [
        [
            "Strategy",
            "Seed",
            "Mean p-adic loss",
            "Fold SD",
            "Δ loss vs baseline",
            "Loss wins",
            "Exact acc.",
            "Prefix-2 acc.",
            "Mean scoring ops",
        ]
    ]
    for summary in summaries:
        rows.append(
            [
                _display_strategy(summary),
                _format_seed(summary.tag_order_seed),
                _format_float(summary.mean_loss),
                _format_float(summary.loss_std),
                _format_float(summary.mean_loss_delta_vs_baseline),
                _format_wins(summary),
                _format_percent(summary.mean_exact_accuracy),
                _format_percent(summary.mean_prefix2_accuracy),
                _format_float(summary.mean_scoring_ops, digits=2),
            ]
        )
    return rows


def render_markdown_report(
    summaries: Sequence[AblationRunSummary],
    *,
    baseline_run_key: str = DEFAULT_BASELINE_RUN_KEY,
    show_random_seeds: bool = True,
) -> str:
    non_random = [
        summary for summary in summaries if summary.tag_order_strategy != "random"
    ]
    random_aggregate = aggregate_random_summaries(summaries)
    main_rows = list(non_random)
    if random_aggregate is not None:
        main_rows.append(random_aggregate)

    sections = [
        f"UMLLR tag-order ablation (`{baseline_run_key}` baseline)",
        "",
        "Lower p-adic loss is better. Deltas are paired fold means against the baseline run.",
        "",
        _markdown_table(_build_table_rows(main_rows)),
    ]

    if show_random_seeds:
        random_rows = [
            summary
            for summary in summaries
            if summary.tag_order_strategy == "random" and summary.source_run_count == 1
        ]
        if random_rows:
            sections.extend(
                [
                    "",
                    "Random seed details",
                    "",
                    _markdown_table(_build_table_rows(random_rows)),
                ]
            )

    return "\n".join(part for part in sections if part is not None)


def render_latex_report(
    summaries: Sequence[AblationRunSummary],
    *,
    baseline_run_key: str = DEFAULT_BASELINE_RUN_KEY,
    show_random_seeds: bool = True,
) -> str:
    non_random = [
        summary for summary in summaries if summary.tag_order_strategy != "random"
    ]
    random_aggregate = aggregate_random_summaries(summaries)
    main_rows = list(non_random)
    if random_aggregate is not None:
        main_rows.append(random_aggregate)

    sections = [
        f"% UMLLR tag-order ablation ({baseline_run_key} baseline)",
        "% Lower p-adic loss is better. Deltas are paired fold means against the baseline run.",
        _latex_table(_build_table_rows(main_rows)),
    ]

    if show_random_seeds:
        random_rows = [
            summary
            for summary in summaries
            if summary.tag_order_strategy == "random" and summary.source_run_count == 1
        ]
        if random_rows:
            sections.extend(
                [
                    "",
                    "% Random seed details",
                    _latex_table(_build_table_rows(random_rows)),
                ]
            )

    return "\n".join(sections)


def build_report(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    baseline_run_key: str = DEFAULT_BASELINE_RUN_KEY,
    output_format: str = "markdown",
    show_random_seeds: bool = True,
) -> str:
    rows = load_ablation_fold_rows(conn, schema=schema)
    summaries = summarize_ablation_rows(rows, baseline_run_key=baseline_run_key)
    if output_format == "markdown":
        return render_markdown_report(
            summaries,
            baseline_run_key=baseline_run_key,
            show_random_seeds=show_random_seeds,
        )
    if output_format == "latex":
        return render_latex_report(
            summaries,
            baseline_run_key=baseline_run_key,
            show_random_seeds=show_random_seeds,
        )
    raise ValueError(f"Unsupported output format: {output_format}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize UMLLR tag-order ablation results from Postgres.",
    )
    parser.add_argument(
        "--dsn",
        help=(
            "Postgres DSN for the Shopify stores database. If omitted, the script "
            "uses SHOPIFY_DB_DSN or DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Schema containing UMLLR ablation tables.",
    )
    parser.add_argument(
        "--baseline-run-key",
        default=DEFAULT_BASELINE_RUN_KEY,
        help="Run key to use as the paired baseline (default: battle_elo).",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "latex"),
        default="markdown",
        help="Output format for the report table.",
    )
    parser.add_argument(
        "--hide-random-seeds",
        action="store_true",
        help="Suppress the second table with per-seed random-order rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the rendered report. Defaults to stdout.",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        report = build_report(
            conn,
            schema=args.schema,
            baseline_run_key=args.baseline_run_key,
            output_format=args.format,
            show_random_seeds=not args.hide_random_seeds,
        )
    finally:
        conn.close()

    if args.output is not None:
        args.output.write_text(report + "\n", encoding="utf-8")
        return

    print(report)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
