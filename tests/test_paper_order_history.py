from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from padjective.benchmark_runtime import Prediction
from padjective.paper_order_history import (
    FoldHistoryRow,
    compare_history_to_baseline,
    render_history_figure,
    summarize_history_rows,
    valuation_histogram,
    write_history_csv,
    write_history_tex,
)


def _row(
    snapshot: str,
    when: date,
    strategy: str,
    run_key: str,
    fold: int,
    loss: float,
) -> FoldHistoryRow:
    return FoldHistoryRow(
        snapshot_id=snapshot,
        snapshot_name=f"latest-{snapshot}",
        snapshot_date=when,
        product_count=1000 + fold,
        tag_count=200,
        taxonomy_count=50,
        run_key=run_key,
        strategy=strategy,
        seed=None,
        cv_fold=fold,
        mean_loss=loss,
        exact_accuracy=0.5,
        prefix2_accuracy=0.7,
        mean_shared_prefix_depth=2.1,
        valuation_counts={"exact": 50, "0": 50},
    )


def test_valuation_histogram_separates_exact_and_finite_depths() -> None:
    predictions = [
        Prediction(product_id=1, true_value=7, predicted_value=7, loss=0.0),
        Prediction(product_id=2, true_value=25, predicted_value=0, loss=0.04),
        Prediction(product_id=3, true_value=8, predicted_value=3, loss=0.2),
        Prediction(product_id=4, true_value=9, predicted_value=8, loss=1.0),
    ]

    assert valuation_histogram(predictions, prime_base=5) == {
        "exact": 1,
        "2": 1,
        "1": 1,
        "0": 1,
    }


def test_history_summary_averages_random_seeds_within_snapshot() -> None:
    rows = [
        _row("a", date(2026, 3, 1), "taxonomy_association", "taxonomy_association", 0, 0.20),
        _row("a", date(2026, 3, 1), "taxonomy_association", "taxonomy_association", 1, 0.22),
        _row("a", date(2026, 3, 1), "random", "random_seed_7", 0, 0.30),
        _row("a", date(2026, 3, 1), "random", "random_seed_7", 1, 0.32),
        _row("a", date(2026, 3, 1), "random", "random_seed_13", 0, 0.34),
        _row("a", date(2026, 3, 1), "random", "random_seed_13", 1, 0.36),
    ]

    summaries = summarize_history_rows(rows)
    random = next(row for row in summaries if row.strategy == "random")

    assert random.source_run_count == 2
    assert random.fold_count == 4
    assert random.mean_loss == pytest.approx(0.33)


def test_history_comparison_uses_taxonomy_association_as_baseline() -> None:
    rows = []
    for snapshot, when, baseline, frequency in (
        ("a", date(2026, 3, 1), 0.20, 0.24),
        ("b", date(2026, 3, 8), 0.30, 0.29),
    ):
        rows.extend(
            (
                _row(snapshot, when, "taxonomy_association", "taxonomy_association", 0, baseline),
                _row(snapshot, when, "frequency", "frequency", 0, frequency),
            )
        )

    comparisons = compare_history_to_baseline(summarize_history_rows(rows))
    frequency = next(row for row in comparisons if row.strategy == "frequency")

    assert frequency.snapshot_count == 2
    assert frequency.lower_loss_snapshots == 1
    assert frequency.mean_delta_vs_baseline == pytest.approx(0.015)
    assert frequency.min_delta_vs_baseline == pytest.approx(-0.01)
    assert frequency.max_delta_vs_baseline == pytest.approx(0.04)


def test_history_outputs_are_created(tmp_path: Path) -> None:
    strategies = (
        "taxonomy_association",
        "random",
        "frequency",
        "battle_elo",
        "mean_title_position",
    )
    rows = []
    for snapshot_index, when in enumerate((date(2026, 3, 1), date(2026, 3, 8))):
        for strategy_index, strategy in enumerate(strategies):
            rows.append(
                _row(
                    str(snapshot_index),
                    when,
                    strategy,
                    strategy,
                    0,
                    0.20 + snapshot_index * 0.01 + strategy_index * 0.02,
                )
            )
    summaries = summarize_history_rows(rows)
    comparisons = compare_history_to_baseline(summaries)
    csv_path = tmp_path / "history.csv"
    tex_path = tmp_path / "history.tex"
    eps_path = tmp_path / "history.eps"

    write_history_csv(csv_path, summaries)
    write_history_tex(tex_path, comparisons, summaries)
    render_history_figure(eps_path, summaries)

    assert "mean_padic_loss" in csv_path.read_text(encoding="utf-8")
    tex = tex_path.read_text(encoding="utf-8")
    assert "\\PadOrderHistorySnapshots" in tex
    assert "taxonomy\\_association" in tex
    assert eps_path.read_text(encoding="latin-1").startswith("%!PS-Adobe")
