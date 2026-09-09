from fractions import Fraction

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from padjective.paper_validate_randomised import (
    direct_certificate, direct_consensus, direct_linear, direct_medoid,
    distance_matrix, summarise,
)
from padjective.randomised_linear import consensus_predictions, fit_coordinates, medoid


def scalar_distance(a, b, p):
    difference = abs(int(a)-int(b))
    if not difference:
        return Fraction(0)
    scale = 1
    while difference % p == 0:
        difference //= p
        scale *= p
    return Fraction(1, scale)


@pytest.mark.parametrize("p,precision", [(2, 3), (3, 2), (71, 7)])
def test_direct_distances_and_medoid_against_scalar_rationals(p, precision):
    q = p**precision
    values = np.random.default_rng(71).integers(0, q, 45)
    actual = distance_matrix(values, values, p, precision)
    expected = np.array([[int(q*scalar_distance(a, b, p)) for b in values] for a in values])
    assert np.array_equal(actual, expected)
    assert direct_medoid(values, p, precision) == medoid(values, p, precision)


def test_independent_coordinate_certificate_accepts_optimum_rejects_unfit():
    x = csr_matrix([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 2, 0])
    fit = fit_coordinates(x, y, p=3, precision=2, seed=42)
    pred = direct_linear(x, fit.coefficients, 9)
    assert direct_certificate(x, y, fit.coefficients, pred, 3, 2) == 2
    with pytest.raises(AssertionError):
        direct_certificate(csr_matrix([[1]]), np.array([2]), [0], [0], 3, 2)


def test_direct_integer_sums_do_not_overflow():
    q = 71**7
    x = csr_matrix(np.ones((1, 2542)))
    assert direct_linear(x, [q-1]*2542, q) == [(-2542) % q]


@pytest.mark.parametrize("members", [1, 3, 9])
def test_independent_consensus_matches_prefix_algorithm(members):
    rng = np.random.default_rng(11)
    predictions = rng.integers(0, 27, (15, members))
    candidates = [1, 2, 4, 9, 11, 21]
    direct = direct_consensus(predictions, candidates, 3, 3)
    fast, _ = consensus_predictions(predictions, candidates, p=3, precision=3)
    assert np.array_equal(direct, fast)


def test_summaries_keep_fold_averaging_and_three_seed_control():
    rows = []
    for method, seeds in (("linear_random", [42, 1729, 20260907]), ("linear_association", [42])):
        for fold in range(5):
            for seed in seeds:
                for label, loss in (("one_pass", .3), ("coordinate", .2)):
                    rows.append(dict(family=f"{method}/{label}", fold=fold, seed=seed+fold,
                        metrics=dict(held_out=dict(mean_padic_loss=loss, first_digit_accuracy=1-loss, exact_accuracy=.5),
                                     mean_nonzero_terms_consulted=2, stored_nonzero_coefficients=20),
                        fit_seconds=1, sweeps=1 if label == "one_pass" else 4))
    summaries, paired = summarise(rows, [])
    assert len(summaries) == 4
    assert paired[0]["pairs"] == paired[0]["improved"] == 15
    assert paired[1]["pairs"] == paired[1]["improved"] == 5
    assert paired[0]["mean_final_minus_first_loss"] == pytest.approx(-.1)
