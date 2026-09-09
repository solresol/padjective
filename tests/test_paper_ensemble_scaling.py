from padjective.paper_ensemble_scaling_batch import BANK_SEEDS, SIZES, jobs
from padjective.paper_randomised_methods import SEED_BASES
from padjective.paper_validate_ensemble_scaling import nested_consensus
from padjective.paper_validate_randomised import direct_consensus
from padjective.randomised_linear import consensus_predictions
import numpy as np
import pytest


def test_extended_grid_preserves_old_bank_and_unique_new_fits():
    assert BANK_SEEDS[:9] == SEED_BASES
    assert len(BANK_SEEDS) == len(set(BANK_SEEDS)) == 243
    assert SIZES == (1, 3, 9, 15, 27, 45, 81, 135, 243)
    commands = jobs()
    assert len(commands) == len({name for name, _ in commands}) == 1170
    pairs = set()
    for name, args in commands:
        config = dict(zip(args[::2], args[1::2]))
        fold, seed = int(config["--fold"]), int(config["--seed"])
        assert seed-fold in BANK_SEEDS[9:]
        pairs.add((fold, seed))
    assert pairs == {(f, s+f) for f in range(5) for s in BANK_SEEDS[9:]}


@pytest.mark.parametrize("p,precision", [(2, 4), (3, 3), (71, 7)])
def test_nested_consensus_matches_two_independent_implementations(p, precision):
    rng = np.random.default_rng(12)
    bank = rng.integers(0, p**precision, (17, 15))
    candidates = rng.integers(0, p**precision, 23)
    orders = [np.arange(15), rng.permutation(15)]
    observed = {}
    for roster, size, predicted in nested_consensus(bank, candidates, (1, 3, 9, 15), orders, p, precision):
        selection = bank[:, orders[roster][:size]]
        assert np.array_equal(predicted, direct_consensus(selection, candidates, p, precision))
        prefix, _ = consensus_predictions(selection, candidates, p=p, precision=precision)
        assert np.array_equal(predicted, prefix)
        observed[(roster, size)] = predicted
    assert np.array_equal(observed[(0, 15)], observed[(1, 15)])


def test_nested_consensus_ties_choose_smallest_candidate():
    bank = np.array([[0, 1], [1, 0]])
    actual = list(nested_consensus(bank, [1, 0], (2,), [[0, 1]], 3, 2))
    assert actual[0][2].tolist() == [0, 0]


def test_nested_consensus_large_bank_integer_bound_and_bad_orders():
    bank = np.full((2, 243), 71**7-1)
    actual = list(nested_consensus(bank, [0, 71**7-1], (243,), [np.arange(243)]))
    assert actual[0][2].tolist() == [71**7-1, 71**7-1]
    with pytest.raises(AssertionError):
        list(nested_consensus(bank, [0], (243,), [[0]*243]))
