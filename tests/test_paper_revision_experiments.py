from __future__ import annotations

import pytest

from padjective.paper_revision_experiments import q_weighted_distance, select_neural_folds


def test_q_weighted_distance_uses_base_valuation_and_q_weight() -> None:
    assert q_weighted_distance(0, 25, prime_base=5, q=10) == pytest.approx(0.01)
    assert q_weighted_distance(3, 8, prime_base=5, q=2) == pytest.approx(0.5)
    assert q_weighted_distance(7, 7, prime_base=5, q=2) == 0.0


def test_q_weighted_distance_rejects_non_metric_weight() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        q_weighted_distance(0, 5, prime_base=5, q=1)


def test_select_neural_folds_defaults_to_all_available_folds() -> None:
    assert select_neural_folds([2, 0, 1, 2]) == [0, 1, 2]


def test_select_neural_folds_validates_requested_subset() -> None:
    assert select_neural_folds([0, 1, 2, 3, 4], [4, 0, 4]) == [0, 4]
    with pytest.raises(ValueError, match="Unknown neural folds: 7"):
        select_neural_folds([0, 1, 2, 3, 4], [0, 7])
