from padjective.metrics import (
    decode_padic_digits,
    hierarchical_loss_score,
    parse_taxonomy_path,
    shared_prefix_depth,
    shared_prefix_depth_encoded,
    summarize_encoded_predictions,
    summarize_taxonomy_predictions,
)


def test_parse_taxonomy_path_splits_numeric_shopify_codes() -> None:
    assert parse_taxonomy_path("1.1.13.12") == ("1", "1", "13", "12")


def test_shared_prefix_depth_counts_root_segments() -> None:
    assert shared_prefix_depth(("1", "1", "13"), ("1", "1", "12")) == 2
    assert shared_prefix_depth(("1",), ("2",)) == 0


def test_hierarchical_loss_score_uses_numeric_segments() -> None:
    taxonomy_paths = {
        "true": ("1", "1", "13", "12"),
        "near": ("1", "1", "13", "11"),
        "far": ("9", "2", "1", "4"),
    }

    near_score = hierarchical_loss_score(["true"], ["near"], taxonomy_paths, base=2.0)
    far_score = hierarchical_loss_score(["true"], ["far"], taxonomy_paths, base=2.0)

    assert near_score == 0.5
    assert far_score == 0.0625


def test_summarize_taxonomy_predictions_reports_prefix_metrics() -> None:
    taxonomy_paths = {
        "a": ("1", "1", "13", "12"),
        "b": ("1", "1", "13", "11"),
        "c": ("1", "2", "5", "7"),
    }

    summary = summarize_taxonomy_predictions(
        ["a", "a", "a"],
        ["a", "b", "c"],
        taxonomy_paths,
        scoring_ops=[1.0, 2.0, 3.0],
    )

    assert summary.exact_accuracy == 1 / 3
    assert summary.prefix1_accuracy == 1.0
    assert summary.prefix2_accuracy == 2 / 3
    assert summary.mean_shared_prefix_depth == 8 / 3
    assert summary.mean_scoring_ops == 2.0


def test_decode_padic_digits_and_shared_prefix_for_raw_outputs() -> None:
    assert decode_padic_digits(86, 5) == (1, 2, 3)
    assert shared_prefix_depth_encoded(86, 6, 5, true_depth=3) == 1


def test_summarize_encoded_predictions_uses_raw_values_without_projection() -> None:
    summary = summarize_encoded_predictions(
        true_values=[86, 86, 86],
        pred_values=[86, 6, 61],
        base=5,
        true_depths=[3, 3, 3],
        scoring_ops=[1.0, 2.0, 3.0],
    )

    assert summary.exact_accuracy == 1 / 3
    assert summary.prefix1_accuracy == 1.0
    assert summary.prefix2_accuracy == 2 / 3
    assert summary.mean_shared_prefix_depth == 2.0
    assert summary.mean_scoring_ops == 2.0
