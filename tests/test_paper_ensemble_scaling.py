from padjective.paper_ensemble_scaling_batch import BANK_SEEDS, SIZES, jobs
from padjective.paper_randomised_methods import SEED_BASES


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
