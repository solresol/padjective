import pytest

from padjective.umllr_ablation_report import (
    AblationFoldRow,
    aggregate_random_summaries,
    render_latex_report,
    render_markdown_report,
    summarize_ablation_rows,
)


def _sample_rows() -> list[AblationFoldRow]:
    return [
        AblationFoldRow(
            run_key="battle_elo",
            tag_order_strategy="battle_elo",
            tag_order_seed=None,
            cv_fold=0,
            mean_loss=0.20,
            exact_accuracy=0.40,
            prefix1_accuracy=0.60,
            prefix2_accuracy=0.70,
            mean_shared_prefix_depth=2.0,
            mean_scoring_ops=3.0,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="battle_elo",
            tag_order_strategy="battle_elo",
            tag_order_seed=None,
            cv_fold=1,
            mean_loss=0.24,
            exact_accuracy=0.42,
            prefix1_accuracy=0.62,
            prefix2_accuracy=0.72,
            mean_shared_prefix_depth=2.1,
            mean_scoring_ops=3.1,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="frequency",
            tag_order_strategy="frequency",
            tag_order_seed=None,
            cv_fold=0,
            mean_loss=0.28,
            exact_accuracy=0.36,
            prefix1_accuracy=0.58,
            prefix2_accuracy=0.65,
            mean_shared_prefix_depth=1.8,
            mean_scoring_ops=2.8,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="frequency",
            tag_order_strategy="frequency",
            tag_order_seed=None,
            cv_fold=1,
            mean_loss=0.32,
            exact_accuracy=0.39,
            prefix1_accuracy=0.59,
            prefix2_accuracy=0.69,
            mean_shared_prefix_depth=1.9,
            mean_scoring_ops=2.9,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="random_seed_7",
            tag_order_strategy="random",
            tag_order_seed=7,
            cv_fold=0,
            mean_loss=0.33,
            exact_accuracy=0.34,
            prefix1_accuracy=0.55,
            prefix2_accuracy=0.61,
            mean_shared_prefix_depth=1.7,
            mean_scoring_ops=2.7,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="random_seed_7",
            tag_order_strategy="random",
            tag_order_seed=7,
            cv_fold=1,
            mean_loss=0.29,
            exact_accuracy=0.35,
            prefix1_accuracy=0.56,
            prefix2_accuracy=0.64,
            mean_shared_prefix_depth=1.75,
            mean_scoring_ops=2.75,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="random_seed_13",
            tag_order_strategy="random",
            tag_order_seed=13,
            cv_fold=0,
            mean_loss=0.31,
            exact_accuracy=0.35,
            prefix1_accuracy=0.56,
            prefix2_accuracy=0.63,
            mean_shared_prefix_depth=1.72,
            mean_scoring_ops=2.72,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="random_seed_13",
            tag_order_strategy="random",
            tag_order_seed=13,
            cv_fold=1,
            mean_loss=0.30,
            exact_accuracy=0.36,
            prefix1_accuracy=0.57,
            prefix2_accuracy=0.65,
            mean_shared_prefix_depth=1.78,
            mean_scoring_ops=2.78,
            prediction_count=100,
        ),
    ]


def test_summarize_ablation_rows_computes_paired_deltas() -> None:
    summaries = summarize_ablation_rows(_sample_rows())
    lookup = {summary.run_key: summary for summary in summaries}

    baseline = lookup["battle_elo"]
    assert baseline.mean_loss == pytest.approx(0.22)
    assert baseline.mean_loss_delta_vs_baseline == pytest.approx(0.0)
    assert baseline.loss_better_folds == 0
    assert baseline.matched_fold_count == 2

    frequency = lookup["frequency"]
    assert frequency.mean_loss == pytest.approx(0.30)
    assert frequency.loss_std == pytest.approx(0.02)
    assert frequency.mean_loss_delta_vs_baseline == pytest.approx(0.08)
    assert frequency.mean_prefix2_delta_vs_baseline == pytest.approx(-0.04)
    assert frequency.mean_scoring_ops_delta_vs_baseline == pytest.approx(-0.20)
    assert frequency.loss_better_folds == 0
    assert frequency.matched_fold_count == 2


def test_aggregate_random_summaries_averages_seed_runs() -> None:
    summaries = summarize_ablation_rows(_sample_rows())
    random_summary = aggregate_random_summaries(summaries)

    assert random_summary is not None
    assert random_summary.source_run_count == 2
    assert random_summary.mean_loss == pytest.approx(0.3075)
    assert random_summary.mean_loss_delta_vs_baseline == pytest.approx(0.0875)
    assert random_summary.loss_better_folds == 0
    assert random_summary.matched_fold_count == 4


def test_render_reports_include_main_and_random_sections() -> None:
    summaries = summarize_ablation_rows(_sample_rows())

    markdown = render_markdown_report(summaries)
    assert "UMLLR tag-order ablation (`battle_elo` baseline)" in markdown
    assert "random (2 seeds avg.)" in markdown
    assert "Random seed details" in markdown
    assert "| frequency | — | 0.300000 | 0.020000 | 0.080000 | 0/2 | 37.50% | 67.00% | 2.85 |" in markdown

    latex = render_latex_report(summaries)
    assert r"% UMLLR tag-order ablation (battle_elo baseline)" in latex
    assert r"battle\_elo" in latex
    assert r"random (2 seeds avg.)" in latex


def test_summarize_ablation_rows_accepts_snapshot_prefixed_baseline() -> None:
    rows = [
        AblationFoldRow(
            run_key="paper::battle_elo",
            tag_order_strategy="battle_elo",
            tag_order_seed=None,
            cv_fold=0,
            mean_loss=0.2,
            exact_accuracy=0.4,
            prefix1_accuracy=0.6,
            prefix2_accuracy=0.7,
            mean_shared_prefix_depth=2.0,
            mean_scoring_ops=3.0,
            prediction_count=100,
        ),
        AblationFoldRow(
            run_key="paper::frequency",
            tag_order_strategy="frequency",
            tag_order_seed=None,
            cv_fold=0,
            mean_loss=0.3,
            exact_accuracy=0.35,
            prefix1_accuracy=0.55,
            prefix2_accuracy=0.6,
            mean_shared_prefix_depth=1.8,
            mean_scoring_ops=2.8,
            prediction_count=100,
        ),
    ]

    summaries = summarize_ablation_rows(rows, baseline_run_key="battle_elo")
    lookup = {summary.run_key: summary for summary in summaries}

    assert lookup["paper::battle_elo"].mean_loss_delta_vs_baseline == pytest.approx(0.0)
    assert lookup["paper::frequency"].mean_loss_delta_vs_baseline == pytest.approx(0.1)
