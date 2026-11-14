import numpy as np
import pytest

from padjective.half_reference_ensemble import reference_consensus, reference_members
from padjective.paper_validate_randomised import direct_consensus
from padjective.randomised_linear import consensus_predictions
from padjective.residual_ensemble import error_rows


def test_half_sample_is_reproducible_uniform_sized_and_without_replacement():
    for count in (0, 1, 2, 3, 8, 81, 242):
        selected = reference_members(count, fold=2, roster=9)
        assert len(selected) == (count+1)//2
        assert len(np.unique(selected)) == len(selected)
        assert np.all((selected >= 0) & (selected < count))
        assert np.array_equal(selected, reference_members(count, fold=2, roster=9))
    choices = {tuple(reference_members(12, fold=fold, roster=0)) for fold in range(12)}
    assert len(choices) > 1
    # Only the first member is available before fitting member two.
    assert reference_members(1, fold=0, roster=0).tolist() == [0]


@pytest.mark.parametrize('p,precision', [(2, 4), (3, 3), (71, 7)])
def test_vectorised_reference_vote_matches_two_existing_integer_implementations(p, precision):
    rng = np.random.default_rng(115)
    bank = rng.integers(0, p**precision, (31, 27))
    options = rng.integers(0, p**precision, 19)
    for count in (1, 2, 3, 9, 27):
        ids = reference_members(count, fold=3, roster=18)
        selected = bank[:, ids]
        vote = reference_consensus(selected, options, p=p, precision=precision)
        prefix, _ = consensus_predictions(selected, options, p=p, precision=precision)
        assert np.array_equal(vote, prefix)
        assert np.array_equal(vote, direct_consensus(selected, options, p, precision))


def test_unmatched_prefixes_ties_and_large_integer_bound():
    assert reference_consensus([[1, 2]], [2, 1], p=3, precision=2).tolist() == [1]
    assert reference_consensus([[8, 8]], [0, 1], p=3, precision=2).tolist() == [0]
    assert reference_consensus(np.full((2, 243), 71**7-1), [0, 71**7-1]).tolist() == [71**7-1]*2


def test_training_mask_uses_sampled_vote_instead_of_full_vote():
    # The sample can get a row wrong even when the whole ensemble is correct.
    bank = np.array([[1, 1, 2], [2, 2, 1]])
    subset = reference_consensus(bank[:, [2]], [1, 2], p=3, precision=2)
    full = reference_consensus(bank, [1, 2], p=3, precision=2)
    assert error_rows([1, 2], full).tolist() == []
    assert error_rows([1, 2], subset).tolist() == [0, 1]
