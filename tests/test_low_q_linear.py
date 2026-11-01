import math

import numpy as np
import pytest

from padjective import low_q_linear as low
from padjective import randomised_linear as original
from padjective.paper_revision_experiments import q_weighted_distance


def brute_choice(values, candidates, kind, p=71):
    return min(sorted(set(candidates)), key=lambda c: math.fsum(
        q_weighted_distance(int(v), int(c), prime_base=p, q=low.weight_value(kind)) for v in values))


def test_low_q_can_change_the_optimal_medoid():
    values = [1, 72, 143, 214, 2, 2, 2]
    assert low.best_choice(values, kind="71")[0] == 1
    assert low.best_choice(values, kind="sqrt2")[0] == 2
    assert low.best_choice(values, kind="2")[0] == 2


@pytest.mark.parametrize("kind", ["sqrt2", "2", "71"])
def test_prefix_optimizer_matches_direct_distance_search(kind):
    rng = np.random.default_rng(91)
    for _ in range(25):
        values = rng.integers(-200, 500, size=23)
        candidates = rng.integers(-250, 550, size=16)
        for options in [values, candidates]:
            candidate, _ = low.best_choice(values, candidates=options, kind=kind)
            assert candidate == brute_choice(values, options, kind)


def test_sqrt2_integer_surrogate_has_a_verified_ordering_bound():
    p, d = low.SQRT2_NUMERATOR, low.SQRT2_DENOMINATOR
    assert abs(p * p - 2 * d * d) == 1
    weights, scale = low.depth_weights("sqrt2", 7, 6693)
    assert np.allclose(weights / scale, [2 ** (-j / 2) for j in range(7)], atol=1e-15)
    with pytest.raises(ValueError, match="ordering bound"):
        low.depth_weights("sqrt2", 80, 6693)


def test_q71_coordinate_fit_reproduces_the_original_optimizer():
    rng = np.random.default_rng(12)
    features = rng.integers(0, 2, size=(18, 5))
    targets = rng.integers(1, 71 ** 2, size=18)
    reference = original.fit_coordinates(features, targets, p=71, precision=3, seed=42)
    actual = low.fit_coordinates(features, targets, kind="71", precision=3, seed=42)
    assert actual['status'] == reference.status == 'coordinate_optimum'
    assert np.array_equal(actual['coefficients'], reference.coefficients)


@pytest.mark.parametrize("kind", ["sqrt2", "2"])
def test_retrained_coordinates_are_locally_optimal_under_the_new_loss(kind):
    features = np.array([[1, 0], [1, 1], [0, 1], [1, 1], [1, 0], [0, 1]])
    targets = np.array([1, 72, 2, 143, 214, 2])
    fit = low.fit_coordinates(features, targets, kind=kind, precision=3, seed=42)
    assert fit['status'] == 'coordinate_optimum'
    assert low.certificate(features, targets, fit['coefficients'], kind=kind, precision=3)['status'] == 'coordinate_optimum'
    costs = [row['objective_units'] for row in fit['history']]
    assert all(b <= a for a, b in zip(costs, costs[1:]))


def test_q71_consensus_reproduces_original_valid_path_consensus():
    voters = np.array([[1, 72, 2, 2], [1, 1, 72, 143], [72, 143, 214, 2]])
    candidates = [1, 2, 72, 143, 214]
    expected, _ = original.consensus_predictions(voters, candidates, p=71, precision=3)
    actual = low.consensus_predictions(voters, candidates, kind="71", precision=3)
    assert np.array_equal(actual, expected)
