from collections import Counter
from itertools import product
import math

import numpy as np
import pytest

from padjective.published_zubarev import (
    GibbsProblem, MahlerDesign, binomial_basis_residues, draw_gibbs_transition,
    fit_published_zubarev, interleave_residue, modular_matvec, padic_losses,
)


def test_interleaving_uses_every_coordinate_and_higher_digits():
    # x=1+2*5+3*25, y=4+0*5+2*25; output digits 1,4,2,0,3,2.
    expected = sum(d * 5**i for i, d in enumerate((1, 4, 2, 0, 3, 2)))
    assert interleave_residue((86, 54), p=5, digits=6) == expected
    assert interleave_residue((1, 0, 1, 1), p=71, digits=4) == 1 + 71**2 + 71**3
    assert interleave_residue((-1, 0), p=5, digits=4) == 4 + 4*25


@pytest.mark.parametrize("p,precision,degree", [(2, 3, 40), (5, 1, 30), (71, 7, 150)])
def test_binomial_residues_match_direct_integer_arithmetic(p, precision, degree):
    for h in (0, 1, p, p**4 + 3*p + 1, 10**20 + 17):
        actual = binomial_basis_residues(h, degree, p=p, precision=precision)
        expected = [math.comb(h, k) % (p**precision) if k <= h else 0 for k in range(degree + 1)]
        assert actual.tolist() == expected
    # A coefficient that vanishes modulo p must not zero all later terms.
    assert binomial_basis_residues(p, p, p=p, precision=1)[p] == 1


def test_extra_input_precision_is_sufficient_for_modular_binomial_basis():
    x = np.array([[1, 4, 2], [7, 2, 8], [3, 9, 11]])
    design = MahlerDesign.build(x, p=5, precision=2, degree=26)
    for i, row in enumerate(x):
        full_h = interleave_residue(row, p=5, digits=15)
        expected = [math.comb(full_h, k) % 25 if k <= full_h else 0 for k in range(27)]
        assert design.basis[design.row_groups[i]].tolist() == expected


def test_literal_degree_resolution_does_not_silently_become_a_tag_sum():
    x = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]])
    small = MahlerDesign.build(x, p=5, precision=1, degree=4)
    roots = small.basis[small.row_groups] % 5
    assert np.array_equal(roots[0], roots[1])
    assert np.array_equal(roots[0], roots[2])
    assert not np.array_equal(roots[0], roots[3])
    larger = MahlerDesign.build(x, p=5, precision=1, degree=25)
    assert not np.array_equal(larger.basis[larger.row_groups[0]], larger.basis[larger.row_groups[1]])


def test_modular_matvec_near_overflow_matches_python_integers():
    rng = np.random.default_rng(5)
    for modulus in (71**7, 2**52):
        a = rng.integers(0, modulus, size=(9, 200), dtype=np.int64)
        w = rng.integers(0, modulus, size=200, dtype=np.int64)
        expected = [(sum(int(v) * int(c) for v, c in zip(row, w, strict=True)) % modulus) for row in a]
        assert modular_matvec(a, w, modulus).tolist() == expected


def test_sampler_matches_enumerated_finite_transition_mass():
    design = MahlerDesign.build([[0], [1]], p=2, precision=1, degree=1)
    problem = GibbsProblem.create(design, [0, 1])
    beta = 2.0
    states = list(product(range(2), repeat=2))
    weights = np.array([math.exp(-beta * problem.loss(np.array(state))) for state in states])
    expected = weights / weights.sum()
    rng = np.random.default_rng(81)
    counts = Counter()
    for _ in range(5000):
        draw = draw_gibbs_transition(problem, beta=beta, rng=rng, max_proposals=1000)
        assert draw.status == "accepted"
        counts[tuple(draw.coefficients)] += 1
    observed = np.array([counts[state] / 5000 for state in states])
    assert np.max(np.abs(observed - expected)) < 0.03


def test_root_lower_bound_is_a_valid_relaxation_and_sampling_can_be_interrupted():
    design = MahlerDesign.build([[0], [0], [1], [1]], p=3, precision=1, degree=1)
    problem = GibbsProblem.create(design, [0, 1, 1, 2])
    assert problem.root_lower_bound == pytest.approx(0.5)
    for state in product(range(3), repeat=2):
        assert problem.loss(np.array(state)) >= problem.root_lower_bound
    draw = draw_gibbs_transition(problem, beta=1, rng=np.random.default_rng(9), max_proposals=5, deadline=0)
    assert draw.status == "time_limit" and draw.coefficients is None


def test_independent_start_optimizer_recovery_and_reproducibility():
    design = MahlerDesign.build([[0], [1], [2], [3], [4]], p=5, precision=1, degree=1)
    problem = GibbsProblem.create(design, [1, 3, 0, 2, 4])
    kwargs = dict(seed=71, initialisation="zeros", betas=(0, 4, 16), draws_per_beta=20, proposals_per_beta=100000)
    a = fit_published_zubarev(problem, **kwargs)
    b = fit_published_zubarev(problem, **kwargs)
    assert a.status == "schedule_complete"
    assert a.best_loss == 0.0
    assert a.coefficients.tolist() == [1, 2]
    assert a.proposals == b.proposals
    assert np.array_equal(a.coefficients, b.coefficients)
    random_fit = fit_published_zubarev(problem, seed=71, initialisation="random", betas=(0,), draws_per_beta=20)
    assert random_fit.initialisation == "random"
    with pytest.raises(ValueError, match="independent"):
        fit_published_zubarev(problem, initialisation="umllr")


def test_metric_and_unseen_design_prediction():
    assert padic_losses([1, 5, 25, 125, 0], [0]*5, 5).tolist() == pytest.approx([1, .2, .04, .008, 0])
    design = MahlerDesign.build([[0], [1]], p=5, precision=2, degree=1)
    assert design.predict_new(np.array([[2], [3]]), np.array([1, 2])).tolist() == [5, 7]
