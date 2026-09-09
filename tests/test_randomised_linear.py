from itertools import product
from fractions import Fraction

import numpy as np
import pytest

from padjective.randomised_linear import (
    apply_default, consensus_predictions, coordinate_certificate, fit_coordinates,
    fit_default, linear_predictions, loss_units, medoid,
)


def distance(a, b, p):
    delta = abs(int(a) - int(b))
    if not delta:
        return Fraction(0)
    power = 1
    while delta % p == 0:
        delta //= p
        power *= p
    return Fraction(1, power)


@pytest.mark.parametrize("p,precision", [(2, 3), (3, 2), (71, 7)])
def test_loss_and_medoid_match_exhaustive_candidates(p, precision):
    q = p**precision
    rng = np.random.default_rng(19)
    for _ in range(15):
        values = rng.integers(0, q, size=12)
        candidate, cost = medoid(values, p, precision)
        possible = range(q) if q < 100 else sorted(set(values.tolist()))
        expected = min((sum(distance(v, c, p) for v in values), c) for c in possible)
        assert candidate == min(c for c in values if sum(distance(v, c, p) for v in values) == expected[0])
        assert cost == q * expected[0]
        assert loss_units(values-candidate, p, precision).sum() == cost


def test_all_small_multisets_have_correct_medoid():
    for values in product(range(4), repeat=3):
        c, units = medoid(values, 2, 2)
        expected = min((sum(distance(v, x, 2) for v in values), x) for x in set(values))
        assert (Fraction(units, 4), c) == expected


def test_revisits_improve_and_reproduce_without_changing_tied_weights():
    found_improvement = False
    for seed in range(6):
        rng = np.random.default_rng(seed)
        x = rng.integers(0, 2, (30, 8))
        y = rng.integers(0, 9, 30)
        fit = fit_coordinates(x, y, p=3, precision=2, seed=seed)
        same = fit_coordinates(x, y, p=3, precision=2, seed=seed)
        assert fit.status == "coordinate_optimum"
        assert np.array_equal(fit.coefficients, same.coefficients)
        assert all(a["training_loss_units"] >= b["training_loss_units"]
                   for a, b in zip(fit.history, fit.history[1:]))
        assert fit.history[-1]["changes"] == 0 and fit.history[-1]["complete"]
        assert fit.first_pass_coefficients is not None
        initial = linear_predictions(x, fit.first_pass_coefficients, 3, 2)
        first_cost = sum(distance(a, b, 3) for a, b in zip(y, initial))
        assert Fraction(fit.final_units, 9) <= first_cost
        found_improvement |= Fraction(fit.final_units, 9) < first_cost
        assert coordinate_certificate(x, y, fit.coefficients, p=3, precision=2)["status"] == "coordinate_optimum"
    assert found_improvement
    tied = fit_coordinates([[1], [1]], [1, 0], p=3, precision=1, seed=1)
    assert tied.coefficients.tolist() == [0]


def test_coordinate_optimum_is_not_global_optimum():
    x = np.array([[1, 0], [0, 1], [1, 1], [1, 1], [1, 1]])
    y = np.array([1, 2, 0, 0, 0])
    fit = fit_coordinates(x, y, p=3, precision=1, seed=42)
    assert fit.status == "coordinate_optimum" and fit.coefficients.tolist() == [0, 0]
    assert fit.final_units == 6
    assert linear_predictions(x, [1, 2], 3, 1).tolist() == y.tolist()


def test_incomplete_sweep_is_not_convergence():
    fit = fit_coordinates([[1, 0], [0, 1]], [1, 1], p=3, precision=1, seed=1, seconds=1e-12)
    assert fit.status == "time_limit"
    assert fit.first_pass_coefficients is None
    assert not fit.history[-1]["complete"]
    fit = fit_coordinates([[1, 0], [0, 1]], [1, 1], p=3, precision=1, seed=1, max_sweeps=1)
    assert fit.status == "sweep_limit" and fit.first_pass_coefficients is not None


def test_binary_and_empty_support_handling():
    fit = fit_coordinates([[1, 0], [0, 0]], [2, 1], p=3, precision=2, seed=1, first_order=[0])
    assert fit.coefficients.tolist() == [2, 0]
    with pytest.raises(ValueError, match="binary"):
        fit_coordinates([[2]], [1], p=3, precision=1, seed=1)
    with pytest.raises(ValueError, match="each supported"):
        fit_coordinates([[1, 1]], [1], p=3, precision=1, seed=1, first_order=[0])


def test_sparse_dot_avoids_overflow_within_protocol():
    q = 71**7
    x = np.ones((2, 2542), dtype=np.int64)
    w = np.full(2542, q-1, dtype=np.int64)
    assert linear_predictions(x, w, 71, 7).tolist() == [(-2542) % q]*2


def test_default_is_training_medoid_or_mode():
    y = np.array([1, 4, 4, 2])
    raw = np.array([0, 0, 1, 1])
    default = fit_default(y, raw, p=3, precision=2)
    assert default == 1
    assert apply_default(raw, default).tolist() == [1, 1, 1, 1]
    assert fit_default(y, np.ones(4), p=3, precision=2) == 4


@pytest.mark.parametrize("members", [1, 3, 9])
def test_consensus_matches_exhaustive_valid_path_scoring(members):
    rng = np.random.default_rng(52)
    predictions = rng.integers(0, 27, (20, members))
    candidates = [1, 4, 10, 20, 26]
    output, work = consensus_predictions(predictions, candidates, p=3, precision=3)
    expected = [min(candidates, key=lambda c: (sum(distance(c, v, 3) for v in row), c))
                for row in predictions]
    assert output.tolist() == expected
    assert work["members"] == members
    assert work["candidate_prefix_evaluations_per_prediction"] == 3*len(candidates)


def test_one_member_consensus_is_a_decoder_not_the_original_model():
    predictions, _ = consensus_predictions([[7]], [1, 4], p=3, precision=2)
    assert predictions.tolist() == [1]
