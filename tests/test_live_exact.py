from itertools import combinations_with_replacement

import numpy as np
import pytest
from scipy import sparse

from padjective import live_exact as live
from padjective import randomised_linear as paper


def distance(a, b, p, precision):
    if a == b:
        return 0
    d, scale = abs(int(a)-int(b)), p**(precision-1)
    while d % p == 0:
        d //= p
        scale //= p
    return scale


def test_medoid_exact_scale_and_ties():
    for values in combinations_with_replacement(range(9), 4):
        costs = [(sum(distance(v, w, 3, 2) for w in values), v) for v in set(values)]
        cost, pred = min(costs)
        assert live.medoid(values, 3, 2) == (pred, cost)
        old, old_cost = paper.medoid(values, 3, 2)
        assert pred == old and cost*3 == old_cost


def test_same_fits_as_paper_for_safe_inputs():
    rng = np.random.default_rng(51)
    x = sparse.csr_matrix(rng.integers(0, 2, (35, 8)))
    y = rng.integers(1, 71**3, 35)
    for seed in (1, 42, 1729):
        old = paper.fit_coordinates(x, y, p=71, precision=3, seed=seed)
        new = live.fit(x, y, p=71, precision=3, seed=seed)
        assert new['status'] == old.status == 'coordinate_optimum'
        assert new['coefficients'] == old.coefficients.tolist()
        raw = live.certificate(x, y, new['coefficients'], 71, 3)
        assert raw.tolist() == live.predict(x, new['coefficients'], 71, 3).tolist()


def test_live_depth_eight_and_previous_overflow_guard():
    p, e = 83, 8
    with pytest.raises(ValueError):
        paper.check_precision(p, e, 31204)
    values = np.array([1, 2, 1+p**7, 2+p**7]*7801)
    pred, cost = live.medoid(values, p, e)
    assert cost == sum(distance(pred, v, p, e) for v in values)
    assert live.total(live.units(values-pred, p, e)) == cost
    assert live.scores(values, np.full(len(values), pred), p, e)['n'] == 31204


def test_large_sums_and_linear_prediction_do_not_wrap():
    p, e = 83, 9
    values = np.array([1, 2]*20000)
    pred, cost = live.medoid(values, p, e)
    assert pred == 1 and cost == 20000*p**8
    assert live.total(live.units(values-pred, p, e)) == cost
    x = sparse.csr_matrix(np.ones((2, 100), dtype=np.int64))
    weights = [p**e-1]*100
    assert live.predict(x, weights, p, e).tolist() == [p**e-100]*2


def test_consensus_against_direct_distance_and_original():
    rng = np.random.default_rng(123)
    for p, e in ((3, 3), (83, 8)):
        bank = rng.integers(0, p**e, (12, 9))
        candidates = np.array([1, 2, 1+p, 2+p**(e-1)])
        result = live.consensus(bank, candidates, p, e)
        expected = [min(candidates, key=lambda c: (sum(distance(c, v, p, e) for v in row), c)) for row in bank]
        assert result.tolist() == expected
        original, _ = paper.consensus_predictions(bank, candidates, p=p, precision=e)
        assert result.tolist() == original.tolist()


def test_certificate_rejects_bad_fit():
    with pytest.raises(AssertionError):
        live.certificate(sparse.csr_matrix([[1], [1]]), [2, 2], [0], 83, 8)


def test_parameter_failures_and_zero_default():
    with pytest.raises(ValueError):
        live.parameters(83, 12)
    assert live.default_value([1, 2, 2], [0, 0, 0], 83, 8) == 2
    assert live.default_value([1, 2, 2], [1, 1, 1], 83, 8) == 2
