from itertools import product
import random

import numpy as np
import pytest

from padjective.published_mihara import (
    Echelon, ResourceBudget, fit_published_mihara, noise_free_matrix,
    predict_published_mihara,
)


def test_dynamic_elimination_copy_and_inconsistency():
    form = Echelon.empty(2, 5)
    ok, first = form.add((1, 0, 1, 3))
    assert ok and first.rank == 1 and form.rank == 0
    ok, duplicate = first.add((1, 0, 1, 3))
    assert ok and duplicate is first
    ok, inconsistent = first.add((1, 0, 1, 4))
    assert not ok and inconsistent is first
    for row in ((0, 1, 1, 4), (0, 0, 1, 1)):
        ok, first = first.add(row)
        assert ok
    assert first.solution() == (2, 3, 1)


def test_row_space_membership_equals_all_affine_equations():
    p = 3
    for observations in [((0, 0, 1, 1),), ((0, 0, 1, 1), (1, 1, 1, 0))]:
        form = Echelon.empty(2, p)
        for row in observations:
            ok, form = form.add(row)
            assert ok
        # Independently enumerate every solution c of A c = y. The paper's C
        # is an affine basis of this same family of equations.
        equations = [c for c in product(range(p), repeat=3)
                     if all((sum(a*b for a, b in zip(row[:3], c)) - row[-1]) % p == 0
                            for row in observations)]
        rows = np.array([(a, b, 1, y) for a, b, y in product(range(p), repeat=3)])
        expected = [all((a*c[0] + b*c[1] + c[2] - y) % p == 0 for c in equations)
                    for a, b, _, y in rows]
        assert form.contains(rows).tolist() == expected


def test_rank_dependent_threshold_and_fitting_point_subtraction():
    form = Echelon.empty(1, 5)
    _, form = form.add((0, 1, 2))
    # L=1, D=1: threshold is .18, not .9. Four held-out matches out of 20 pass.
    rows = np.array([(0, 1, 2)] * 5 + [(1, 1, 0)] * 16)
    assert noise_free_matrix(rows, form, ResourceBudget())
    # Only the fitted point agrees: it must not provide its own validation.
    rows = np.array([(0, 1, 2)] + [(1, 1, 0)] * 20)
    assert not noise_free_matrix(rows, form, ResourceBudget())


def test_full_rank_threshold_is_strictly_greater_than_ninety_percent():
    form = Echelon.empty(0, 5)
    _, form = form.add((1, 2))
    assert not noise_free_matrix(np.array([(1, 2)] * 10 + [(1, 3)]), form, ResourceBudget())
    assert noise_free_matrix(np.array([(1, 2)] * 11 + [(1, 3)]), form, ResourceBudget())


@pytest.mark.parametrize("noise_rate", [0.0, 0.025])
def test_known_affine_recovery_with_higher_input_digits(noise_rate):
    p, precision = 5, 3
    modulus = p**precision
    rng = random.Random(73)
    x = np.array([[rng.randrange(modulus) for _ in range(2)] for _ in range(2000)])
    expected = (7, 19, 11)
    y = (x @ np.array(expected[:2]) + expected[-1]) % modulus
    for i in range(len(y)):
        for digit in range(precision):
            if rng.random() < noise_rate:
                old = int(y[i]) // p**digit % p
                new = rng.randrange(p)
                y[i] += (new - old) * p**digit
    fit = fit_published_mihara(x, y, p=p, precision=precision, seed=19,
                              budget=ResourceBudget(seconds=10, max_draws=100000))
    assert fit.completed, fit
    assert fit.coefficients == expected
    assert fit.completed_digits == precision
    assert predict_published_mihara(fit, x[:20], p=p) == ((x[:20] @ np.array(expected[:2]) + 11) % modulus).tolist()


def test_rank_obstruction_is_interrupted_not_fabricated_fit():
    x = np.array([(i % 5, i % 5) for i in range(50)])
    fit = fit_published_mihara(x, np.arange(50) % 5, p=5, precision=3,
                              budget=ResourceBudget(max_draws=100, seconds=10))
    assert not fit.completed and fit.status == "draw_limit"
    assert fit.draws == 100
    with pytest.raises(ValueError, match="interrupted"):
        predict_published_mihara(fit, x, p=5)


def test_external_clock_limit_and_undefined_validation_are_distinct():
    clock = [0.0]
    budget = ResourceBudget(clock=lambda: clock[0], seconds=1)
    clock[0] = 2
    fit = fit_published_mihara([[0], [1]], [1, 2], p=5, precision=1, budget=budget)
    assert fit.status == "time_limit" and fit.draws == 0
    fit = fit_published_mihara(np.empty((1, 0), dtype=int), [2], p=5, precision=1)
    assert fit.status == "insufficient_validation_rows"


def test_repeatability_and_input_validation():
    x = np.array(list(product(range(5), repeat=2)) * 20)
    y = (x[:, 0] * 3 + x[:, 1] * 2 + 1) % 5
    a = fit_published_mihara(x, y, p=5, precision=1, seed=17)
    b = fit_published_mihara(x, y, p=5, precision=1, seed=17)
    assert (a.coefficients, a.draws, a.restarts) == (b.coefficients, b.draws, b.restarts)
    with pytest.raises(ValueError, match="prime"):
        fit_published_mihara(x, y, p=4, precision=1)
