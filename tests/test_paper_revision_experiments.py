from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from padjective.paper_revision_experiments import (
    multinomial_logistic_parameter_count,
    q_weighted_distance,
    select_fold_top_tags,
    select_snapshot_top_tags,
    single_hidden_layer_parameter_count,
)


def test_q_weighted_distance_uses_base_valuation_and_q_weight() -> None:
    assert q_weighted_distance(0, 25, prime_base=5, q=10) == pytest.approx(0.01)
    assert q_weighted_distance(3, 8, prime_base=5, q=2) == pytest.approx(0.5)
    assert q_weighted_distance(7, 7, prime_base=5, q=2) == 0.0


def test_q_weighted_distance_rejects_non_metric_weight() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        q_weighted_distance(0, 5, prime_base=5, q=1)


def test_select_fold_top_tags_ignores_evaluation_frequency() -> None:
    features = sparse.csr_matrix(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 100],
        ],
        dtype=np.float32,
    )
    selected = select_fold_top_tags(
        features,
        np.asarray([True, True, True, False]),
        max_tags=2,
    )
    assert selected.tolist() == [0, 1]


def test_select_fold_top_tags_rejects_nonpositive_budget() -> None:
    features = sparse.csr_matrix(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="positive"):
        select_fold_top_tags(features, np.asarray([True, False]), max_tags=0)


def test_select_snapshot_top_tags_includes_evaluation_frequency() -> None:
    features = sparse.csr_matrix(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 100],
        ],
        dtype=np.float32,
    )
    selected = select_snapshot_top_tags(features, max_tags=2)
    assert selected.tolist() == [1, 2]


def test_paper_snapshot_classical_models_match_coefficient_budget() -> None:
    assert multinomial_logistic_parameter_count(6, 361) == 2527
    assert single_hidden_layer_parameter_count(74, 5, 361) == 2541
