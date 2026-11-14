from itertools import product
from copy import deepcopy
from types import SimpleNamespace

import pytest

from padjective.paper_refinement_comparison import (
    E, P, SNAPSHOT_DIGEST, aggregate, audit_checkpoint, classify, has_carry,
    occupied_powers, scalar_loss_units, score,
)


def test_distinct_valuations_do_not_imply_disjoint_powers_or_no_carries():
    assert classify([3, 2], 2, 3) == dict(active_count=2, valuation_separated=True,
                                          disjoint=False, hidden_contributions=0)
    assert has_carry([3, 2], 2, 3)
    assert classify([4, 3], 3, 3)["valuation_separated"]
    assert not classify([4, 3], 3, 3)["disjoint"]
    assert not has_carry([4, 3], 3, 3)
    assert classify([1, 2, 2], 2, 3)["valuation_separated"] is False


def test_paper_example_has_overlap_and_carry():
    z = [20*P**2+3*P+13, 2*P**5+17*P**4+8*P**3+53*P**2]
    assert occupied_powers(z[0], P, E) & occupied_powers(z[1], P, E) == {2}
    assert classify(z, P, E)["valuation_separated"]
    assert not classify(z, P, E)["disjoint"]
    assert has_carry(z, P, E)


def test_signed_digits_are_canonical_and_truncation_vacuity_is_visible():
    assert occupied_powers(-1, 3, 4) == {0, 1, 2, 3}
    assert occupied_powers(abs(-1), 3, 4) == {0}
    assert not classify([-1, 3], 3, 4)["disjoint"]
    low = classify([3, 6], 3, 1)
    assert low["disjoint"] and low["hidden_contributions"] == 2
    assert not low["valuation_separated"]
    assert not classify([3, 6], 3, 2)["disjoint"]
    with pytest.raises(ValueError):
        occupied_powers(1, 3, 0)


def test_zero_one_and_duplicate_contributions():
    for contributions in ([], [0, 0], [0, 3]):
        result = classify(contributions, 3, 3)
        assert result["disjoint"] and result["valuation_separated"]
    assert not classify([1, 1], 3, 3)["valuation_separated"]
    assert not classify([1, 1], 3, 3)["disjoint"]


@pytest.mark.parametrize("p,depth", [(2, 3), (3, 2)])
def test_full_depth_implication_and_no_cancellation_exhaustively(p, depth):
    q = p**depth
    for z in product(range(q), repeat=3):
        result = classify(z, p, depth)
        assert not result["disjoint"] or result["valuation_separated"]
        if result["active_count"] and result["valuation_separated"]:
            assert sum(z) % q != 0


def make_row(target, prediction, active, v, d):
    return dict(target=target, prediction=prediction, active_count=active,
                valuation_separated=v, disjoint=d, taxonomy_id="test", raw=prediction,
                integer_sum=prediction, carry=False,
                loss_units=scalar_loss_units(target, prediction, P, E))


def test_correctness_is_not_implied_and_wrong_only_losses_exclude_exact_hits():
    assert classify([1, 3], 3, 2)["disjoint"]
    assert scalar_loss_units(2, 4, 3, 2) == 9  # Maximal loss despite disjointness.
    assert not classify([1, 1], 3, 2)["valuation_separated"]
    assert scalar_loss_units(2, 2, 3, 2) == 0
    rows = [make_row(1, 1, 2, True, True), make_row(1, 2, 2, True, False)]
    result = aggregate(rows)
    assert result["multi"]["mean_padic_loss"] == .5
    assert result["multi"]["incorrect_mean_padic_loss"] == 1
    assert result["multi_disjoint"]["incorrect_mean_padic_loss"] is None
    assert result["zero"]["n"] == 0 and result["zero"]["exact_hit_rate"] is None
    assert score([])["mean_padic_loss"] is None


def test_strata_separate_degenerate_cases_from_the_primary_comparison():
    rows = [make_row(1, 1, 0, True, True), make_row(1, 2, 1, True, True),
            make_row(1, 1, 2, True, True), make_row(1, 2, 2, True, False),
            make_row(1, 2, 3, False, False)]
    result = aggregate(rows)
    assert result["all"]["n"] == 5
    assert result["zero"]["n"] == result["one"]["n"] == 1
    assert result["multi"]["n"] == 3
    assert result["multi_valuation_separated"]["n"] == 2
    assert result["multi_disjoint"]["n"] == result["multi_valuation_only"]["n"] == 1


def test_checkpoint_reconstruction_checks_saved_outputs_and_training_only_default():
    def record(key, fold, tags, target):
        return SimpleNamespace(product_key=key, cv_fold=fold, tags=tags,
                               encoded_path=target, taxonomy_id="test")
    records = [record("train0", 1, [], 1), record("train1", 1, ["A", "B"], 72),
               record("test0", 0, [], 1), record("test1", 0, ["A"], 1),
               record("test2", 0, ["A", "B"], 73)]
    evidence = dict(snapshot_digest=SNAPSHOT_DIGEST, feature_names=["A", "B"],
                    p=P, precision=E, models={"one_pass": dict(
                        coefficients=[1, 71], completed=True, default=1,
                        training_raw_predictions=[0, 72], training_predictions=[1, 72],
                        held_out_raw_predictions=[0, 1, 72], predictions=[1, 1, 72])})
    metrics = {"one_pass": {"held_out": dict(n=3, exact_accuracy=2/3, mean_padic_loss=1/3)}}
    rows, _ = audit_checkpoint(records, ("A", "B"), evidence, metrics, 0)
    assert [r["active_count"] for r in rows] == [0, 1, 2]
    assert rows[0]["raw"] == 0 and rows[0]["prediction"] == 1
    changed = deepcopy(evidence)
    changed["models"]["one_pass"]["predictions"][-1] = 73
    with pytest.raises(ValueError, match="held-out predictions"):
        audit_checkpoint(records, ("A", "B"), changed, metrics, 0)
    changed = deepcopy(evidence)
    changed["models"]["one_pass"]["default"] = 73
    with pytest.raises(ValueError, match="training-only default"):
        audit_checkpoint(records, ("A", "B"), changed, metrics, 0)


def test_modular_zero_is_distinct_from_no_contributions():
    z = [1, P**E-1]
    result = classify(z, P, E)
    assert result["active_count"] == 2 and sum(z) % P**E == 0
    assert not result["valuation_separated"] and not result["disjoint"]
    assert has_carry(z, P, E)
