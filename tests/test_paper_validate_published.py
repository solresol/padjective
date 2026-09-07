import numpy as np
import pytest

from padjective.paper_published_methods import independent_scores
from padjective.paper_validate_published import exact_scores, reference_predictions
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
