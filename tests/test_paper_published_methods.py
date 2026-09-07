import numpy as np
import pytest

from padjective.paper_published_methods import (
    exact_rank_certificate, independent_scores, training_feature_order,
)


def test_exact_rank_audit_does_not_remove_dependent_features():
    x = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    result = exact_rank_certificate(x, 5)
    assert result["status"] == "rank_obstruction"
    assert result["rank"] == 2 and result["columns"] == 3
    x = np.array([[0, 0], [1, 0], [0, 1]])
    assert exact_rank_certificate(x, 5)["status"] == "full_rank"


def test_training_order_and_independent_metric():
    x = np.array([[1, 0, 1], [1, 1, 0]])
    assert training_feature_order(x, ("C", "B", "A")) == [0, 2, 1]
    result = independent_scores([0, 1, 5, 25], [0, 0, 0, 0], 5)
    assert result["mean_padic_loss"] == pytest.approx(.31)
    assert result["exact_accuracy"] == .25
    assert result["first_digit_accuracy"] == .75
