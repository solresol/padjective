import numpy as np
import pytest

from padjective.half_reference_ensemble import reference_members
from padjective.paper_validate_randomised import fingerprint
from padjective.reference_fraction_sweep import PERCENTS, full_reference_prefix, jobs


def test_original_half_sample_stream_is_byte_for_byte_preserved():
    values = [reference_members(n, fold=f, roster=r).tolist()
              for f in range(5) for r in (0, 9, 18) for n in range(243)]
    assert fingerprint(values) == '7447bd506a25db8b0ac9e9079f812c99a84377dd3b340c2908d455f6333a08a4'


@pytest.mark.parametrize('percent', [10, 25, 50, 75, 90, 100])
def test_sample_counts_round_up_and_never_repeat_members(percent):
    for n in (0, 1, 2, 3, 4, 9, 10, 79, 80, 81, 242):
        actual = reference_members(n, fold=1, roster=18, percent=percent)
        assert len(actual) == (percent*n+99)//100
        assert len(set(actual.tolist())) == len(actual)
        assert np.all((actual >= 0) & (actual < n))
        assert np.array_equal(actual, reference_members(n, fold=1, roster=18, percent=percent))
        if percent == 100:
            assert actual.tolist() == list(range(n))


@pytest.mark.parametrize('percent', [0, 101, 50.5])
def test_invalid_rates_are_rejected(percent):
    with pytest.raises(ValueError):
        reference_members(9, fold=0, roster=0, percent=percent)


def test_known_initial_prefix_identity_and_first_omitted_reference():
    assert {p: full_reference_prefix(p) for p in PERCENTS} == {10: 2, 25: 2, 75: 4, 90: 10}
    for p in PERCENTS:
        boundary = full_reference_prefix(p)
        for n in range(1, boundary):
            assert reference_members(n, fold=2, roster=9, percent=p).tolist() == list(range(n))
        assert len(reference_members(boundary, fold=2, roster=9, percent=p)) < boundary


def test_interleaved_grid_includes_each_fraction_fold_and_order_once():
    batches = {str(p): f'batch-{p}' for p in PERCENTS}
    actual = jobs(batches)
    assert len(actual) == len(set(actual)) == 60
    assert set(actual) == {(batches[str(p)], p, f, r) for p in PERCENTS
                           for f in range(5) for r in (0, 9, 18)}
