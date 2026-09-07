import numpy as np
import pytest

from padjective.paper_published_methods import (
    exact_rank_certificate, independent_scores, inclusion_certificate, training_feature_order,
)


def test_exact_rank_audit_does_not_remove_dependent_features():
    x = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = exact_rank_certificate(x, 5)
    assert result["status"] == "rank_obstruction"
    assert result["rank"] == 2 and result["columns"] == 3
    x = np.array([[0, 0], [1, 0], [0, 1]])
    assert exact_rank_certificate(x, 5)["status"] == "full_rank"


def test_independent_column_certificate_keeps_intercept_and_full_rank():
    x = np.array([[1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3]])
    result = exact_rank_certificate(x, 5)
    assert result["independent_feature_indices"] == [1]
    assert exact_rank_certificate(x[:, result["independent_feature_indices"]], 5)["status"] == "full_rank"


def test_duplicate_input_certificate_uses_full_rank_strict_threshold():
    x = np.array([[0]] * 10 + [[1]] * 10)
    y = np.array([0] * 9 + [1] + [1] * 9 + [0])
    result = inclusion_certificate(x, y, 5)
    assert result["maximum_agreement_count"] == 18
    assert result["status"] == "inclusion_obstruction"
    assert inclusion_certificate(x, np.array([0]*10 + [1]*10), 5)["status"] == "not_ruled_out"


def test_training_order_and_independent_metric():
    x = np.array([[1, 0, 1], [1, 1, 0]])
    assert training_feature_order(x, ("C", "B", "A")) == [0, 2, 1]
    result = independent_scores([0, 1, 5, 25], [0, 0, 0, 0], 5)
    assert result["mean_padic_loss"] == pytest.approx(.31)
    assert result["exact_accuracy"] == .25
    assert result["first_digit_accuracy"] == .75
