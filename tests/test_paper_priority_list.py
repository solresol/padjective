from collections import Counter
from fractions import Fraction
from itertools import permutations, product
import random

import pytest

from padjective import paper_priority_list as plist
from padjective.benchmark_runtime import ProductRecord
from padjective.paper_revision_experiments import q_weighted_distance

P = 71


def record(index, tags, target):
    return ProductRecord(index, f"p{index}", list(tags), target, 0, str(target), 1, ())


def loss(targets, predictions, q):
    return sum(q_weighted_distance(a, b, prime_base=P, q=q) for a, b in zip(targets, predictions))


def pairwise_medoid(values, q):
    return min(sorted(set(values)), key=lambda c: sum(
        Fraction(1, q ** plist_depth(c, v)) for v in values if v != c))


def plist_depth(a, b):
    delta, depth = abs(a - b), 0
    while delta % P == 0:
        delta //= P
        depth += 1
    return depth


@pytest.mark.parametrize("q", [2, 71])
def test_residue_counting_matches_pairwise_distances(q):
    rng = random.Random(5)
    digits = [0, 1, 2, 3]
    for _ in range(200):
        values = [sum(rng.choice(digits) * P**k for k in range(rng.randint(1, 4)))
                  for _ in range(rng.randint(1, 12))]
        best, cost = plist.medoid_cost(Counter(values), prime=P, q=q)
        assert best == pairwise_medoid(values, q)
        assert Fraction(cost, q ** plist.DEPTH_CAP) == sum(
            Fraction(1, q ** plist_depth(best, v)) for v in values if v != best)


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
    with pytest.raises(ValueError, match="nonnegative"):
        plist.medoid([-1, 2], prime=P, q=2)


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


def test_backfill_gives_pooled_values_to_tags_that_head_nothing():
    training = [record(0, ["a", "b"], 1), record(1, ["a", "b"], 1), record(2, ["c"], 4)]
    testing = [record(9, ["b"], 1), record(10, ["b", "c"], 4), record(11, ["z"], 1)]
    values, default = plist.fit_pooled(training, prime=P, q=2)
    rows = plist.list_rows(training, testing, ["a", "b", "c"], values, default, prime=P, q=2)
    assert rows["cell"]["stored"] == 2 and rows["cell_backfill"]["stored"] == 3
    assert rows["cell"]["coverage"] == pytest.approx(1 / 3)
    assert rows["cell_backfill"]["coverage"] == pytest.approx(2 / 3)
    # The backfilled tag b outranks c, so the second product is now predicted as 1.
    assert rows["cell"]["exact_accuracy"] == 1
    assert rows["cell_backfill"]["exact_accuracy"] == pytest.approx(2 / 3)
    assert rows["cell"]["train_loss_q2"] == rows["cell_backfill"]["train_loss_q2"] == 0
    assert rows["pooled_priority"]["train_loss_q2"] is None


def test_greedy_order_takes_the_purest_remaining_tag():
    training = [record(0, ["pure"], 1), record(1, ["pure", "mixed", "late"], 1),
                record(2, ["pure", "mixed"], 1), record(3, ["mixed"], 2), record(4, ["mixed"], 3),
                record(5, ["mixed", "late"], 3), record(6, ["solo"], 5)]
    fallback = ["late", "mixed", "pure", "solo"]
    order = plist.greedy_order(training, fallback, prime=P, q=2, smoothing=0)
    # "late" starts with two different labels. Once "pure" has taken one of its
    # products it is a pure singleton, and it ties with "solo" on purity and size.
    assert order == ["pure", "late", "solo", "mixed"]
    rank, cells = plist.fit_cells(training, order, prime=P, q=2)
    assert cells == {"pure": 1, "solo": 5, "late": 3, "mixed": 2}


def test_smoothing_prefers_a_large_nearly_pure_tag_and_unused_tags_follow_the_fallback():
    training = [record(i, ["big"], 1) for i in range(30)] + [record(30, ["big"], 2)]
    training += [record(31, ["one"], 3), record(32, ["big", "shadow"], 1)]
    fallback = ["shadow", "one", "big"]
    assert plist.greedy_order(training, fallback, prime=P, q=2, smoothing=0)[0] == "one"
    order = plist.greedy_order(training, fallback, prime=P, q=2, smoothing=4)
    assert order == ["big", "one", "shadow"]


def naive_greedy(training, fallback, q, smoothing):
    """Rescore every remaining tag from scratch at each step."""
    labels = [r.encoded_path for r in training]
    prior = Fraction(plist.medoid_cost(Counter(labels), prime=P, q=q)[1], len(labels))
    remaining, order = set(range(len(training))), []
    while remaining:
        scores = {}
        for tag in {t for i in remaining for t in training[i].tags}:
            rows = [i for i in remaining if tag in training[i].tags]
            cost = plist.medoid_cost(Counter(labels[i] for i in rows), prime=P, q=q)[1]
            scores[tag] = ((cost + smoothing * prior) / (len(rows) + smoothing), -len(rows), tag)
        if not scores:
            break
        tag = min(scores.values())[2]
        order.append(tag)
        remaining -= {i for i in remaining if tag in training[i].tags}
    return order + [tag for tag in fallback if tag not in order]


@pytest.mark.parametrize("q,smoothing", [(2, 0), (2, 1), (71, 0), (71, 4)])
def test_incremental_greedy_order_matches_rescoring_from_scratch(q, smoothing):
    rng = random.Random(17)
    tags = [f"t{k}" for k in range(9)]
    labels = [1, 2, 1 + P, 1 + 2 * P, 2 + P, 1 + P + 3 * P * P]
    for _ in range(40):
        training = [record(i, rng.sample(tags, rng.randint(1, 4)), rng.choice(labels))
                    for i in range(rng.randint(5, 30))]
        assert plist.greedy_order(training, tags, prime=P, q=q, smoothing=smoothing) == naive_greedy(
            training, tags, q, smoothing)
