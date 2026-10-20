import numpy as np
import pytest

from padjective.randomised_linear import consensus_predictions, fit_coordinates, fit_default
from padjective.residual_ensemble import CumulativeConsensus, error_rows, fit_error_member


def test_mask_uses_current_whole_path_consensus_and_allows_reentry():
    actual = [1, 4, 2]
    assert error_rows(actual).tolist() == [0, 1, 2]
    # Same root (mod 3) is insufficient; a changed vote reintroduces row 0.
    assert error_rows(actual, [1, 1, 2]).tolist() == [1]
    assert error_rows(actual, [4, 4, 2]).tolist() == [0]
    assert error_rows(actual, actual).tolist() == []


@pytest.mark.parametrize('p,precision', [(3, 3), (71, 7)])
def test_every_cumulative_prefix_matches_existing_decoder(p, precision):
    rng = np.random.default_rng(812)
    candidates = rng.integers(0, p**precision, 13)
    bank = rng.integers(0, p**precision, (19, 27))
    state = CumulativeConsensus(len(bank), candidates, p=p, precision=precision, max_members=27)
    for index in range(bank.shape[1]):
        actual = state.add(bank[:, index])
        expected, _ = consensus_predictions(bank[:, :index+1], candidates, p=p, precision=precision)
        assert np.array_equal(actual, expected)
    with pytest.raises(ValueError, match='bound'):
        state.add(bank[:, 0])


def test_smallest_candidate_tie_and_int64_capacity():
    state = CumulativeConsensus(1, [2, 1], p=3, precision=2, max_members=2)
    state.add([2])
    assert state.add([1]).tolist() == [1]
    state = CumulativeConsensus(1, [0, 71**7-1])
    for _ in range(243):
        assert state.add([71**7-1]).tolist() == [71**7-1]


def test_first_member_matches_unmodified_fitter_and_predictions_cover_all_rows():
    x = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    y = np.array([1, 4, 2, 5])
    first = fit_error_member(x, y, x, seed=42, p=3, precision=2)
    control = fit_coordinates(x, y, seed=42, p=3, precision=2)
    assert np.array_equal(first.coefficients, control.coefficients)
    assert first.selected_rows.tolist() == [0, 1, 2, 3]
    second = fit_error_member(x, y, x, previous_predictions=[1, 1, 2, 5],
                              seed=1729, p=3, precision=2)
    assert second.selected_rows.tolist() == [1]
    assert len(second.predictions) == len(y)
    assert second.default == fit_default(y[[1]], second.training_raw[[1]], p=3, precision=2)
    # A one-row fit's default must not use the three excluded targets.
    assert second.default == 4
    assert fit_error_member(x, y, x, previous_predictions=y, seed=42, p=3, precision=2) is None


def test_incomplete_members_fail_instead_of_entering_ensemble():
    with pytest.raises(RuntimeError, match='Incomplete'):
        fit_error_member([[1]], [1], [[1]], seed=42, p=3, precision=2, max_sweeps=1)
