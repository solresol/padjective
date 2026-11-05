import pytest

from padjective.paper_q2_primary import check_scores, project
from padjective.q_loss_rescore import rescore


def test_projection_retains_valid_codes_and_original_tie_break():
    predictions = [72, 73, 143, 2, 1]
    assert project(predictions, {1, 2, 72}) == [72, 2, 1, 2, 1]
    assert predictions == [72, 73, 143, 2, 1]


def test_original_fold_metrics_and_prediction_hash_must_match():
    scores = rescore([1, 72, 2], [1, 1, 1], {"2": 2., "71": 71.})
    archived = dict(n=3, mean_padic_loss=(1 + 1/71)/3,
                    exact_accuracy=1/3, first_digit_accuracy=2/3)
    check_scores(scores, archived, scores["prediction_sha256"])
    with pytest.raises(ValueError, match="prediction hash"):
        check_scores(scores, archived, "wrong")
    with pytest.raises(ValueError, match="fold size"):
        check_scores(scores, dict(archived, n=4))
    with pytest.raises(ValueError, match="did not reproduce"):
        check_scores(scores, dict(archived, mean_padic_loss=.5))
