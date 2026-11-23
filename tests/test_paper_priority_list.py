from itertools import permutations, product

import pytest

from padjective import paper_priority_list as plist
from padjective.benchmark_runtime import ProductRecord
from padjective.paper_revision_experiments import q_weighted_distance

P = 71


def record(index, tags, target):
    return ProductRecord(index, f"p{index}", list(tags), target, 0, str(target), 1, ())


def loss(targets, predictions, q):
    return sum(q_weighted_distance(a, b, prime_base=P, q=q) for a, b in zip(targets, predictions))


def test_fitting_weight_can_change_the_medoid():
    # Five distinct paths under root digit 1 against four copies of one path under 2.
    values = [1 + P * d for d in range(1, 6)] + [2] * 4
    assert plist.medoid(values, prime=P, q=71) == 1 + P
    assert plist.medoid(values, prime=P, q=2) == 2


def test_medoid_breaks_ties_by_integer_order_and_needs_an_integer_weight():
    assert plist.medoid([5, 3, 5, 3], prime=P, q=2) == 3
    with pytest.raises(ValueError, match="integer q"):
        plist.medoid([1, 2], prime=P, q=1.5)
    with pytest.raises(ValueError, match="at least one"):
        plist.medoid([], prime=P, q=2)


def test_each_tag_takes_the_medoid_of_the_products_it_heads():
    training = [record(0, ["a", "b"], 1), record(1, ["a"], 1), record(2, ["b"], 2),
                record(3, ["b", "c"], 2), record(4, ["c"], 3), record(5, ["a", "c"], 1)]
    rank, cells = plist.fit_cells(training, ["a", "b", "c"], prime=P, q=2)
    assert cells == {"a": 1, "b": 2, "c": 3}
    predictions, covered = plist.predict_priority(
        [record(9, ["c", "b"], 2), record(10, ["z"], 5)], rank, cells, default=7)
    assert predictions == [2, 7]
    assert covered == [True, False]


def test_tags_that_head_nothing_are_skipped_at_prediction():
    training = [record(0, ["a", "b"], 1), record(1, ["a", "b"], 1), record(2, ["c"], 4)]
    rank, cells = plist.fit_cells(training, ["a", "b", "c"], prime=P, q=2)
    assert set(cells) == {"a", "c"}
    assert plist.predict_priority([record(9, ["b", "c"], 4)], rank, cells, default=0)[0] == [4]


@pytest.mark.parametrize("q", [2, 71])
def test_cell_medoids_are_the_exact_optimum_for_every_order(q):
    labels = [1, 2, 1 + P, 1 + 2 * P, 2 + P]
    training = [record(i, tags, labels[i % len(labels)]) for i, tags in enumerate(
        [["a"], ["a", "b"], ["b"], ["b", "c"], ["c"], ["a", "c"], ["a", "b", "c"], ["c"], ["b"]])]
    targets = [r.encoded_path for r in training]
    for order in permutations("abc"):
        rank, cells = plist.fit_cells(training, order, prime=P, q=q)
        fitted = loss(targets, plist.predict_priority(training, rank, cells, 0)[0], q)
        best = min(loss(targets, plist.predict_priority(
            training, rank, dict(zip("abc", values)), 0)[0], q)
            for values in product(set(labels), repeat=3))
        assert fitted == pytest.approx(best)


def test_pooled_values_ignore_the_order_and_vote_by_medoid():
    training = [record(0, ["a", "b"], 1), record(1, ["a"], 1), record(2, ["b"], 2),
                record(3, ["b"], 2), record(4, ["c"], 1 + P), record(5, ["c"], 1 + P)]
    values, default = plist.fit_pooled(training, prime=P, q=71)
    assert values == {"a": 1, "b": 2, "c": 1 + P}
    assert default == 1
    # Two of the three active values share root digit 1, so one of those wins.
    predictions, covered = plist.predict_vote(
        [record(9, ["a", "b", "c"], 1), record(10, [], 1)], values, default, prime=P)
    assert predictions == [1, 1]
    assert covered == [True, False]
