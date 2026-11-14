"""Replay greedy tag orders on archived rolling benchmark snapshots.

The journal paper selects ``taxonomy_association`` on one fixed benchmark.
This module checks whether that choice is stable over time by replaying the
same benchmark implementation on a predeclared weekly sample of the archived
``latest-*`` snapshots.  Fold summaries, including valuation histograms, are
persisted in Postgres; compact CSV, TeX, and EPS outputs can then be generated
without retaining another copy of the product data.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import statistics
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from psycopg import sql
from psycopg.types.json import Jsonb

from . import benchmark_runtime, db
from .paper_revision_experiments import _load_paper_dataset


DEFAULT_SCHEMA = "padjective"
DEFAULT_SNAPSHOT_PREFIX = "latest-"
DEFAULT_BASELINE_STRATEGY = "taxonomy_association"
DETERMINISTIC_STRATEGIES = benchmark_runtime.DEFAULT_ABLATION_STRATEGIES
RANDOM_SEEDS = benchmark_runtime.DEFAULT_RANDOM_SEEDS
DISPLAY_ORDER = (
    "taxonomy_association",
    "random",
    "frequency",
    "battle_elo",
    "mean_title_position",
)
DISPLAY_LABELS = {
    "taxonomy_association": "Taxonomy association",
    "random": "Random (five-seed mean)",
    "frequency": "Frequency",
    "battle_elo": "Battle Elo",
    "mean_title_position": "Mean title position",
}
DISPLAY_COLOURS = {
    "taxonomy_association": "#0b6ce3",
    "random": "#7c3aed",
    "frequency": "#0891b2",
    "battle_elo": "#d97706",
    "mean_title_position": "#dc2626",
}


@dataclass(frozen=True)
class SnapshotSpec:
    snapshot_id: str
    snapshot_name: str
    snapshot_date: date
    created_at: datetime
    product_count: int
    tag_count: int
    taxonomy_count: int
    code_version: str | None


@dataclass(frozen=True)
class FoldHistoryRow:
    snapshot_id: str
    snapshot_name: str
    snapshot_date: date
    product_count: int
    tag_count: int
    taxonomy_count: int
    run_key: str
    strategy: str
    seed: int | None
    cv_fold: int
    mean_loss: float
    exact_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    valuation_counts: dict[str, int]


@dataclass(frozen=True)
class SnapshotStrategySummary:
    snapshot_id: str
    snapshot_name: str
    snapshot_date: date
    product_count: int
    tag_count: int
    taxonomy_count: int
    strategy: str
    source_run_count: int
    fold_count: int
    mean_loss: float
    fold_loss_sd: float
    mean_exact_accuracy: float
    mean_prefix2_accuracy: float


@dataclass(frozen=True)
class StrategyComparison:
    strategy: str
    snapshot_count: int
    lower_loss_snapshots: int
    tied_snapshots: int
    mean_delta_vs_baseline: float
    median_delta_vs_baseline: float
    min_delta_vs_baseline: float
    max_delta_vs_baseline: float


def _run_key(strategy: str, seed: int | None) -> str:
    if seed is None:
        return strategy
    return f"{strategy}_seed_{seed}"


def valuation_histogram(
    predictions: Iterable[benchmark_runtime.Prediction],
    *,
    prime_base: int,
) -> dict[str, int]:
    """Count exact predictions and finite valuations of nonzero residuals."""

    counts: dict[str, int] = {}
    for prediction in predictions:
        difference = abs(
            int(prediction.true_value) - int(prediction.predicted_value)
        )
        if difference == 0:
            key = "exact"
        else:
            valuation = 0
            while difference % prime_base == 0:
                difference //= prime_base
                valuation += 1
            key = str(valuation)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.paper_revision_order_history (
                    snapshot_id UUID NOT NULL,
                    snapshot_name TEXT NOT NULL,
                    snapshot_date DATE NOT NULL,
                    snapshot_created_at TIMESTAMPTZ NOT NULL,
                    product_count INTEGER NOT NULL,
                    tag_count INTEGER NOT NULL,
                    taxonomy_count INTEGER NOT NULL,
                    code_version TEXT,
                    prime_base INTEGER NOT NULL,
                    run_key TEXT NOT NULL,
                    tag_order_strategy TEXT NOT NULL,
                    tag_order_seed INTEGER,
                    cv_fold INTEGER NOT NULL,
                    num_test INTEGER NOT NULL,
                    mean_loss DOUBLE PRECISION NOT NULL,
                    exact_accuracy DOUBLE PRECISION NOT NULL,
                    prefix2_accuracy DOUBLE PRECISION NOT NULL,
                    mean_shared_prefix_depth DOUBLE PRECISION NOT NULL,
                    valuation_counts JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    paper_revision_order_history_run_idx
                ON {schema}.paper_revision_order_history
                    (snapshot_id, run_key, cv_fold)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS
                    paper_revision_order_history_date_strategy_idx
                ON {schema}.paper_revision_order_history
                    (snapshot_date, tag_order_strategy, tag_order_seed)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def load_weekly_snapshot_specs(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
) -> list[SnapshotSpec]:
    """Return the latest archived rolling snapshot in each Sydney week."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH ranked AS (
                    SELECT
                        snapshot_id,
                        snapshot_name,
                        created_at,
                        product_count,
                        tag_count,
                        taxonomy_count,
                        code_version,
                        ROW_NUMBER() OVER (
                            PARTITION BY DATE_TRUNC(
                                'week',
                                created_at AT TIME ZONE 'Australia/Sydney'
                            )
                            ORDER BY created_at DESC, snapshot_name DESC
                        ) AS recency_rank
                    FROM {schema}.product_taxonomy_bench_snapshots
                    WHERE snapshot_name LIKE %s
                )
                SELECT
                    snapshot_id,
                    snapshot_name,
                    (created_at AT TIME ZONE 'Australia/Sydney')::DATE,
                    created_at,
                    product_count,
                    tag_count,
                    taxonomy_count,
                    code_version
                FROM ranked
                WHERE recency_rank = 1
                ORDER BY created_at
                """
            ).format(schema=sql.Identifier(schema)),
            (f"{snapshot_prefix}%",),
        )
        rows = cur.fetchall()
    return [
        SnapshotSpec(
            snapshot_id=str(row[0]),
            snapshot_name=str(row[1]),
            snapshot_date=row[2],
            created_at=row[3],
            product_count=int(row[4]),
            tag_count=int(row[5]),
            taxonomy_count=int(row[6]),
            code_version=str(row[7]) if row[7] is not None else None,
        )
        for row in rows
    ]


def select_snapshot_specs(
    specs: Sequence[SnapshotSpec],
    *,
    through_date: date | None = None,
    max_snapshots: int | None = None,
) -> list[SnapshotSpec]:
    """Apply reproducible reporting bounds to the weekly snapshot series."""

    selected = [
        spec
        for spec in specs
        if through_date is None or spec.snapshot_date <= through_date
    ]
    if max_snapshots is not None:
        selected = selected[:max_snapshots]
    return selected


def _snapshot_is_complete(conn, schema: str, snapshot_id: str) -> bool:
    expected_rows = (
        len(DETERMINISTIC_STRATEGIES) + len(RANDOM_SEEDS)
    ) * 5
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*)
                FROM {schema}.paper_revision_order_history
                WHERE snapshot_id = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        row = cur.fetchone()
    return bool(row and int(row[0]) == expected_rows)


def run_snapshot_history(
    conn,
    spec: SnapshotSpec,
    *,
    schema: str = DEFAULT_SCHEMA,
) -> int:
    """Replay every paper tag order for one archived snapshot and persist folds."""

    dataset = _load_paper_dataset(
        conn,
        snapshot_ref=spec.snapshot_name,
        schema=schema,
        skip_incomplete_rows=True,
    )
    analyzed_product_count = len(dataset.records)
    analyzed_tag_count = len(dataset.feature_names)
    analyzed_taxonomy_count = len(
        {record.taxonomy_id for record in dataset.records}
    )
    if analyzed_product_count != spec.product_count:
        print(
            f"{spec.snapshot_name}: using {analyzed_product_count}/"
            f"{spec.product_count} products with assigned folds"
        )
    battles = benchmark_runtime._derive_battles(dataset.records)
    folds = sorted({record.cv_fold for record in dataset.records})
    run_specs = [
        *((strategy, None) for strategy in DETERMINISTIC_STRATEGIES),
        *(("random", seed) for seed in RANDOM_SEEDS),
    ]
    stored_rows: list[tuple[object, ...]] = []
    for strategy, seed in run_specs:
        for fold in folds:
            result = benchmark_runtime.umllr_run_fold(
                fold,
                dataset.records,
                battles,
                dataset.prime_base,
                tag_order_strategy=strategy,
                tag_order_seed=seed,
            )
            prediction_count = len(result.predictions)
            stored_rows.append(
                (
                    spec.snapshot_id,
                    spec.snapshot_name,
                    spec.snapshot_date,
                    spec.created_at,
                    analyzed_product_count,
                    analyzed_tag_count,
                    analyzed_taxonomy_count,
                    spec.code_version,
                    dataset.prime_base,
                    _run_key(strategy, seed),
                    strategy,
                    seed,
                    fold,
                    prediction_count,
                    result.loss / prediction_count,
                    result.exact_accuracy,
                    result.prefix2_accuracy,
                    result.mean_shared_prefix_depth,
                    Jsonb(
                        valuation_histogram(
                            result.predictions,
                            prime_base=dataset.prime_base,
                        )
                    ),
                )
            )

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.paper_revision_order_history
                    (snapshot_id, snapshot_name, snapshot_date,
                     snapshot_created_at, product_count, tag_count,
                     taxonomy_count, code_version, prime_base, run_key,
                     tag_order_strategy, tag_order_seed, cv_fold, num_test,
                     mean_loss, exact_accuracy, prefix2_accuracy,
                     mean_shared_prefix_depth, valuation_counts)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                     %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_id, run_key, cv_fold)
                DO UPDATE SET
                    snapshot_name = EXCLUDED.snapshot_name,
                    snapshot_date = EXCLUDED.snapshot_date,
                    snapshot_created_at = EXCLUDED.snapshot_created_at,
                    product_count = EXCLUDED.product_count,
                    tag_count = EXCLUDED.tag_count,
                    taxonomy_count = EXCLUDED.taxonomy_count,
                    code_version = EXCLUDED.code_version,
                    prime_base = EXCLUDED.prime_base,
                    tag_order_strategy = EXCLUDED.tag_order_strategy,
                    tag_order_seed = EXCLUDED.tag_order_seed,
                    num_test = EXCLUDED.num_test,
                    mean_loss = EXCLUDED.mean_loss,
                    exact_accuracy = EXCLUDED.exact_accuracy,
                    prefix2_accuracy = EXCLUDED.prefix2_accuracy,
                    mean_shared_prefix_depth =
                        EXCLUDED.mean_shared_prefix_depth,
                    valuation_counts = EXCLUDED.valuation_counts,
                    updated_at = NOW()
                """
            ).format(schema=sql.Identifier(schema)),
            stored_rows,
        )
    conn.commit()
    return len(stored_rows)


def run_missing_weekly_history(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    snapshot_prefix: str = DEFAULT_SNAPSHOT_PREFIX,
    max_snapshots: int | None = None,
) -> list[SnapshotSpec]:
    """Run incomplete weekly snapshots, returning the selected series."""

    _ensure_storage(conn, schema)
    specs = load_weekly_snapshot_specs(
        conn,
        schema=schema,
        snapshot_prefix=snapshot_prefix,
    )
    if max_snapshots is not None:
        specs = specs[:max_snapshots]
    for index, spec in enumerate(specs, start=1):
        if _snapshot_is_complete(conn, schema, spec.snapshot_id):
            print(
                f"[{index}/{len(specs)}] {spec.snapshot_name}: "
                "already complete"
            )
            continue
        print(f"[{index}/{len(specs)}] {spec.snapshot_name}: replaying orders")
        row_count = run_snapshot_history(conn, spec, schema=schema)
        print(f"[{index}/{len(specs)}] {spec.snapshot_name}: stored {row_count} folds")
    return specs


def load_fold_history_rows(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    snapshot_ids: Sequence[str] | None = None,
) -> list[FoldHistoryRow]:
    params: tuple[object, ...] = ()
    where_clause = sql.SQL("")
    if snapshot_ids is not None:
        where_clause = sql.SQL("WHERE snapshot_id = ANY(%s::uuid[])")
        params = (list(snapshot_ids),)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    snapshot_id, snapshot_name, snapshot_date, product_count,
                    tag_count, taxonomy_count, run_key, tag_order_strategy,
                    tag_order_seed, cv_fold, mean_loss, exact_accuracy,
                    prefix2_accuracy, mean_shared_prefix_depth,
                    valuation_counts
                FROM {schema}.paper_revision_order_history
                {where_clause}
                ORDER BY snapshot_date, snapshot_name, tag_order_strategy,
                         tag_order_seed NULLS FIRST, cv_fold
                """
            ).format(
                schema=sql.Identifier(schema),
                where_clause=where_clause,
            ),
            params,
        )
        rows = cur.fetchall()
    return [
        FoldHistoryRow(
            snapshot_id=str(row[0]),
            snapshot_name=str(row[1]),
            snapshot_date=row[2],
            product_count=int(row[3]),
            tag_count=int(row[4]),
            taxonomy_count=int(row[5]),
            run_key=str(row[6]),
            strategy=str(row[7]),
            seed=int(row[8]) if row[8] is not None else None,
            cv_fold=int(row[9]),
            mean_loss=float(row[10]),
            exact_accuracy=float(row[11]),
            prefix2_accuracy=float(row[12]),
            mean_shared_prefix_depth=float(row[13]),
            valuation_counts=dict(row[14]),
        )
        for row in rows
    ]


def summarize_history_rows(
    rows: Sequence[FoldHistoryRow],
) -> list[SnapshotStrategySummary]:
    """Aggregate deterministic folds and the five random runs by snapshot."""

    grouped: dict[tuple[str, str], list[FoldHistoryRow]] = {}
    for row in rows:
        grouped.setdefault((row.snapshot_id, row.strategy), []).append(row)

    summaries: list[SnapshotStrategySummary] = []
    for (_snapshot_id, strategy), group in grouped.items():
        exemplar = group[0]
        losses = [row.mean_loss for row in group]
        summaries.append(
            SnapshotStrategySummary(
                snapshot_id=exemplar.snapshot_id,
                snapshot_name=exemplar.snapshot_name,
                snapshot_date=exemplar.snapshot_date,
                product_count=exemplar.product_count,
                tag_count=exemplar.tag_count,
                taxonomy_count=exemplar.taxonomy_count,
                strategy=strategy,
                source_run_count=len({row.run_key for row in group}),
                fold_count=len(group),
                mean_loss=float(statistics.fmean(losses)),
                fold_loss_sd=float(statistics.pstdev(losses)),
                mean_exact_accuracy=float(
                    statistics.fmean(row.exact_accuracy for row in group)
                ),
                mean_prefix2_accuracy=float(
                    statistics.fmean(row.prefix2_accuracy for row in group)
                ),
            )
        )
    order = {strategy: index for index, strategy in enumerate(DISPLAY_ORDER)}
    return sorted(
        summaries,
        key=lambda row: (
            row.snapshot_date,
            row.snapshot_name,
            order.get(row.strategy, len(order)),
        ),
    )


def compare_history_to_baseline(
    summaries: Sequence[SnapshotStrategySummary],
    *,
    baseline_strategy: str = DEFAULT_BASELINE_STRATEGY,
) -> list[StrategyComparison]:
    """Compare each ordering with the baseline across matched snapshots."""

    by_snapshot: dict[str, dict[str, SnapshotStrategySummary]] = {}
    for row in summaries:
        by_snapshot.setdefault(row.snapshot_id, {})[row.strategy] = row
    comparisons: list[StrategyComparison] = []
    for strategy in DISPLAY_ORDER:
        deltas: list[float] = []
        for snapshot_rows in by_snapshot.values():
            baseline = snapshot_rows.get(baseline_strategy)
            candidate = snapshot_rows.get(strategy)
            if baseline is None or candidate is None:
                continue
            deltas.append(candidate.mean_loss - baseline.mean_loss)
        if not deltas:
            continue
        comparisons.append(
            StrategyComparison(
                strategy=strategy,
                snapshot_count=len(deltas),
                lower_loss_snapshots=sum(delta < 0 for delta in deltas),
                tied_snapshots=sum(np.isclose(delta, 0.0) for delta in deltas),
                mean_delta_vs_baseline=float(statistics.fmean(deltas)),
                median_delta_vs_baseline=float(statistics.median(deltas)),
                min_delta_vs_baseline=float(min(deltas)),
                max_delta_vs_baseline=float(max(deltas)),
            )
        )
    return comparisons


def write_history_csv(
    path: Path,
    summaries: Sequence[SnapshotStrategySummary],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "snapshot_date",
                "snapshot_name",
                "products",
                "tags",
                "taxonomies",
                "strategy",
                "source_runs",
                "fold_rows",
                "mean_padic_loss",
                "fold_loss_sd",
                "exact_accuracy",
                "prefix2_accuracy",
            )
        )
        for row in summaries:
            writer.writerow(
                (
                    row.snapshot_date.isoformat(),
                    row.snapshot_name,
                    row.product_count,
                    row.tag_count,
                    row.taxonomy_count,
                    row.strategy,
                    row.source_run_count,
                    row.fold_count,
                    f"{row.mean_loss:.9f}",
                    f"{row.fold_loss_sd:.9f}",
                    f"{row.mean_exact_accuracy:.9f}",
                    f"{row.mean_prefix2_accuracy:.9f}",
                )
            )


def write_history_tex(
    path: Path,
    comparisons: Sequence[StrategyComparison],
    summaries: Sequence[SnapshotStrategySummary],
) -> None:
    snapshot_dates = sorted({row.snapshot_date for row in summaries})
    if not snapshot_dates:
        raise ValueError("Cannot write an empty history summary")
    baseline = next(
        row
        for row in comparisons
        if row.strategy == DEFAULT_BASELINE_STRATEGY
    )
    lines = [
        "% Generated by padjective.paper_order_history",
        f"\\newcommand{{\\PadOrderHistorySnapshots}}{{{baseline.snapshot_count}}}",
        f"\\newcommand{{\\PadOrderHistoryStart}}{{{snapshot_dates[0].isoformat()}}}",
        f"\\newcommand{{\\PadOrderHistoryEnd}}{{{snapshot_dates[-1].isoformat()}}}",
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Order & Lower-loss dates & Mean $\\Delta$ & Median $\\Delta$ & Range of $\\Delta$ \\\\",
        "\\hline",
    ]
    for row in comparisons:
        label = row.strategy.replace("_", r"\_")
        if row.strategy == DEFAULT_BASELINE_STRATEGY:
            lower = "--"
            mean_delta = "0.000000"
            median_delta = "0.000000"
            delta_range = "--"
        else:
            lower = f"{row.lower_loss_snapshots}/{row.snapshot_count}"
            mean_delta = f"{row.mean_delta_vs_baseline:+.6f}"
            median_delta = f"{row.median_delta_vs_baseline:+.6f}"
            delta_range = (
                f"[{row.min_delta_vs_baseline:+.6f}, "
                f"{row.max_delta_vs_baseline:+.6f}]"
            )
        lines.append(
            f"\\texttt{{{label}}} & {lower} & {mean_delta} & "
            f"{median_delta} & {delta_range} \\\\"
        )
    lines.extend(("\\hline", "\\end{tabular}", ""))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def render_history_figure(
    path: Path,
    summaries: Sequence[SnapshotStrategySummary],
) -> None:
    """Render valuation-derived loss history with corpus size context."""

    if not summaries:
        raise ValueError("Cannot render an empty order history")
    by_strategy: dict[str, list[SnapshotStrategySummary]] = {}
    for row in summaries:
        by_strategy.setdefault(row.strategy, []).append(row)

    fig, (loss_ax, size_ax) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.3),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": (3.2, 1.0), "hspace": 0.08},
    )
    for strategy in DISPLAY_ORDER:
        rows = sorted(
            by_strategy.get(strategy, []),
            key=lambda row: row.snapshot_date,
        )
        if not rows:
            continue
        loss_ax.plot(
            [row.snapshot_date for row in rows],
            [row.mean_loss for row in rows],
            marker="o" if strategy == DEFAULT_BASELINE_STRATEGY else None,
            markersize=3.2,
            linewidth=2.4 if strategy == DEFAULT_BASELINE_STRATEGY else 1.35,
            color=DISPLAY_COLOURS[strategy],
            label=DISPLAY_LABELS[strategy],
            zorder=4 if strategy == DEFAULT_BASELINE_STRATEGY else 2,
        )

    snapshot_rows = sorted(
        (
            row
            for row in summaries
            if row.strategy == DEFAULT_BASELINE_STRATEGY
        ),
        key=lambda row: row.snapshot_date,
    )
    size_ax.plot(
        [row.snapshot_date for row in snapshot_rows],
        [row.product_count for row in snapshot_rows],
        color="#4b5563",
        linewidth=1.5,
    )
    size_ax.fill_between(
        [row.snapshot_date for row in snapshot_rows],
        [row.product_count for row in snapshot_rows],
        color="#d1d5db",
    )

    loss_ax.set_ylabel("Mean $p$-adic loss (lower is better)")
    loss_ax.set_title(
        "Greedy tag-order performance across archived rolling snapshots",
        fontsize=11.5,
        fontweight="bold",
    )
    loss_ax.grid(True, color="#d1d5db", linewidth=0.7, linestyle=":")
    loss_ax.legend(frameon=False, fontsize=7.3, ncol=2, loc="best")
    size_ax.set_ylabel("Products")
    size_ax.set_xlabel("Snapshot date (Australia/Sydney)")
    size_ax.grid(True, axis="y", color="#e5e7eb", linewidth=0.7, linestyle=":")
    size_ax.xaxis.set_major_locator(mdates.MonthLocator())
    size_ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    size_ax.tick_params(axis="x", rotation=25)
    for axis in (loss_ax, size_ax):
        axis.tick_params(labelsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.15)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _print_summary(comparisons: Sequence[StrategyComparison]) -> None:
    for row in comparisons:
        print(
            f"{row.strategy}: lower on {row.lower_loss_snapshots}/"
            f"{row.snapshot_count} snapshots; mean delta "
            f"{row.mean_delta_vs_baseline:+.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay and report weekly historical UMLLR tag-order comparisons."
        )
    )
    parser.add_argument("--dsn")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--snapshot-prefix", default=DEFAULT_SNAPSHOT_PREFIX)
    replay_group = parser.add_mutually_exclusive_group()
    replay_group.add_argument(
        "--run-missing",
        action="store_true",
        help="Replay weekly snapshots that do not yet have complete fold rows.",
    )
    replay_group.add_argument(
        "--force-rerun",
        action="store_true",
        help="Replay every selected weekly snapshot, replacing matching fold rows.",
    )
    parser.add_argument(
        "--through-date",
        type=date.fromisoformat,
        metavar="YYYY-MM-DD",
        help="Include only selected snapshots on or before this Sydney date.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        help="Optional leading-snapshot limit for a bounded test run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for CSV, TeX, and EPS report artifacts.",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        _ensure_storage(conn, args.schema)
        selected_specs = select_snapshot_specs(
            load_weekly_snapshot_specs(
                conn,
                schema=args.schema,
                snapshot_prefix=args.snapshot_prefix,
            ),
            through_date=args.through_date,
            max_snapshots=args.max_snapshots,
        )
        if args.run_missing or args.force_rerun:
            for index, spec in enumerate(selected_specs, start=1):
                if (
                    not args.force_rerun
                    and _snapshot_is_complete(conn, args.schema, spec.snapshot_id)
                ):
                    print(
                        f"[{index}/{len(selected_specs)}] {spec.snapshot_name}: "
                        "already complete"
                    )
                    continue
                print(
                    f"[{index}/{len(selected_specs)}] {spec.snapshot_name}: "
                    "replaying orders"
                )
                stored = run_snapshot_history(conn, spec, schema=args.schema)
                print(
                    f"[{index}/{len(selected_specs)}] {spec.snapshot_name}: "
                    f"stored {stored} folds"
                )

        rows = load_fold_history_rows(
            conn,
            schema=args.schema,
            snapshot_ids=[spec.snapshot_id for spec in selected_specs],
        )
        summaries = summarize_history_rows(rows)
        comparisons = compare_history_to_baseline(summaries)
        _print_summary(comparisons)

        if args.output_dir is not None:
            write_history_csv(args.output_dir / "umllr_order_history.csv", summaries)
            write_history_tex(
                args.output_dir / "umllr_order_history_summary.tex",
                comparisons,
                summaries,
            )
            render_history_figure(
                args.output_dir / "umllr_order_history.eps",
                summaries,
            )
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
