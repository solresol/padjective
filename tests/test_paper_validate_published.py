import numpy as np
import pytest

from padjective.paper_published_methods import independent_scores
from padjective.paper_validate_published import exact_scores, grouped_root_error_floor, reference_predictions
from padjective.published_zubarev import MahlerDesign


def test_integer_metric_agrees_with_separate_scalar_metric():
    targets = [0, 1, 71, 71**2, 71**6, 71**7-1]
    predictions = [0]*len(targets)
    expected = independent_scores(targets, predictions, 71)
    actual = exact_scores(targets, predictions, 71, 7)
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], abs=1e-14)


def test_reference_polynomial_sums_match_optimised_sums():
    design = MahlerDesign.build([[0, 0], [1, 1], [2, 4], [1, 0]], p=71, precision=7, degree=80)
    coefficients = np.random.default_rng(12).integers(0, 71**7, size=81).tolist()
    assert reference_predictions(design, coefficients) == design.predict(np.array(coefficients)).tolist()


def test_posthoc_group_bound_is_an_oracle_not_a_fitted_predictor():
    x = np.array([[0], [0], [1], [1]])
    y = np.array([0, 1, 2, 2])
    bound = grouped_root_error_floor(x, y, 71)
    assert bound == dict(n=4, groups=2, maximum_root_correct=3, root_error_floor=.25)
