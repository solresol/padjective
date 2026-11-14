import numpy as np
from scipy.sparse import csr_matrix

from padjective.paper_randomised_batch import jobs
from padjective.paper_randomised_methods import (
    SEED_BASES, association_order, model_evidence, supported_order,
)
from padjective.published_zubarev import MahlerDesign


def test_predeclared_grid_and_pilots():
    full, pilot = jobs(), jobs(True)
    assert len(full) == len({name for name, _ in full}) == 110
    assert len(pilot) == 4
    assert all("--pilot" in args for _, args in pilot)
    assert all("--pilot" not in args for _, args in full)
    assert sum(name.startswith("linear_random") for name, _ in full) == 45
    assert sum(name.startswith("linear_association") for name, _ in full) == 5
    assert sum("k357910" in name for name, _ in full) == 45
    assert sum("k357911" in name for name, _ in full) == 15
    for name, args in full:
        config = dict(zip(args[::2], args[1::2]))
        base = int(name.split("-s")[-1])
        assert base in SEED_BASES
        assert int(config["--seed"]) == base + int(config["--fold"])


def test_supported_order_uses_only_supported_features_first():
    x = csr_matrix([[0, 1, 1, 0], [0, 0, 1, 0]])
    names = ["z", "b", "a", "c"]
    order = supported_order(x, names, 42)
    assert set(order[:2]) == {1, 2}
    assert list(order[2:]) == [3, 0]
    assert np.array_equal(order, supported_order(x, names, 42))
    assert association_order(x, [1, 2], names) == [1, 2]


def test_pilot_never_predicts_or_scores_held_out():
    x = csr_matrix([[1, 0], [0, 1]])
    record, metrics = model_evidence(np.array([1, 0]), x, None, np.array([1, 2]), None,
                                    p=3, precision=2, pilot=True, completed=True)
    assert record["default"] == 2
    assert record["training_predictions"] == [1, 2]
    assert "predictions" not in record
    assert "held_out" not in metrics


def test_one_extra_mahler_term_exposes_fourth_root_coordinate():
    # Small-prime exhaustive analogue of the 71**3 boundary used in the study.
    from itertools import product
    x = np.array(list(product((0, 1), repeat=4)), dtype=np.int64)
    short = MahlerDesign.build(x, p=3, precision=2, degree=3**3-1)
    longer = MahlerDesign.build(x, p=3, precision=2, degree=3**3)
    weights = np.zeros(3**3+1, dtype=np.int64)
    weights[-1] = 1
    assert np.array_equal(longer.predict(weights) % 3, x[:, 3])
    roots = short.basis[short.row_groups] % 3
    # Every pair differs only in its fourth feature: all old root terms agree.
    assert np.array_equal(roots[::2], roots[1::2])
