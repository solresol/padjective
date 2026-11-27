from collections import Counter
from itertools import permutations
import random

import pytest

from padjective import benchmark_runtime as runtime
from padjective import paper_priority_list as plist
from padjective import paper_priority_list_levels as levels
from padjective import paper_priority_list_purity as purity
from padjective.benchmark_runtime import ProductRecord

P = purity.P
LABELS = [1, 2, 1 + P, 1 + 2 * P, 2 + P, 1 + P + 3 * P * P, 1 + P + 4 * P * P]


def record(index, tags, target, fold=0):
    return ProductRecord(index, f"p{index}", list(tags), target, fold, str(target), 1, ())


def toy(seed, n=120, tags=12):
    rng = random.Random(seed)
    names = [f"t{k}" for k in range(tags)]
    return [record(i, rng.sample(names, rng.randint(1, 4)), rng.choice(LABELS), i % 5) for i in range(n)]


def distance(a, b, q):
    if a == b:
        return 0.0
    k = 0
    while a % P ** (k + 1) == b % P ** (k + 1):
        k += 1
    return q ** -k


@pytest.mark.parametrize("q", [1.25, 2, 71])
def test_spread_is_the_mean_distance_to_the_best_member(q):
    rng = random.Random(7)
    codes = [rng.choice(LABELS) for _ in range(25)]
    direct = min(sum(distance(c, x, q) for x in codes) for c in set(codes)) / len(codes)
    assert purity.spread(Counter(codes), q) == pytest.approx(direct)


@pytest.mark.parametrize("q", [2, 71])
def test_spread_matches_the_exact_integer_medoid_cost(q):
    counts = Counter({1: 3, 1 + P: 2, 2 + P: 1, 1 + P + 3 * P * P: 4})
    exact = plist.medoid_cost(counts, prime=P, q=q)[1] / q ** plist.DEPTH_CAP / sum(counts.values())
    assert purity.spread(counts, q) == pytest.approx(exact)


def test_pairwise_is_the_mean_over_different_products():
    codes = [1, 1, 1 + P, 2, 1 + P + 3 * P * P]
    pairs = list(permutations(range(len(codes)), 2))
    direct = sum(distance(codes[i], codes[j], 2) for i, j in pairs) / len(pairs)
    assert purity.pairwise(Counter(codes), 2) == pytest.approx(direct)
    assert purity.pairwise(Counter([5]), 2) == 0.0


def test_spread_tends_to_exact_match_noise_as_q_falls_to_one():
    counts = Counter({1: 5, 1 + P: 3, 2: 2})
    assert purity.spread(counts, 1 + 1e-9) == pytest.approx(purity.exact_match_noise(counts), abs=1e-6)
    # At a large weight only the top digit matters: two of ten products are under another root.
    assert purity.spread(counts, 1e9) == pytest.approx(0.2, abs=1e-6)


def test_ball_depth_counts_the_digits_a_share_of_labels_agree_on():
    counts = Counter({1 + P: 9, 2: 1})
    assert purity.ball_depth(counts, 1.0) == 0
    assert purity.ball_depth(counts, 0.9) == purity.DEPTH
    mixed = Counter({1 + P: 5, 1 + 2 * P: 5})
    assert purity.ball_depth(mixed, 1.0) == 1
    assert purity.ball_depth(Counter({1 + P: 4}), 1.0) == purity.DEPTH


def test_codes_longer_than_the_depth_are_rejected():
    with pytest.raises(ValueError):
        purity.residues(Counter([P ** purity.DEPTH]))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_match_order_is_taxonomy_association_and_weight_71_is_taxonomy_spread(seed):
    training = toy(seed)
    orders = purity.purity_orders(training)
    assert orders["exact_match"] == runtime._tag_order(training, [], 0, strategy="taxonomy_association")
    assert orders["spread_q71"] == levels.spread_order(training)
    tags = {tag for r in training for tag in r.tags}
    assert all(set(order) == tags and len(order) == len(tags) for order in orders.values())


def test_a_tag_that_is_tight_in_the_tree_outranks_a_tag_with_a_larger_exact_share():
    # "sock" always lands under 1.1 but on two leaves; "misc" has a bigger exact share and strays to another root.
    training = ([record(i, ["sock"], 1 + P + (3 + i % 2) * P * P) for i in range(6)]
                + [record(10 + i, ["misc"], 1) for i in range(4)] + [record(20, ["misc"], 2), record(21, ["misc"], 2)])
    orders = purity.purity_orders(training)
    assert orders["exact_match"].index("misc") < orders["exact_match"].index("sock")
    for name in ("spread_q2", "spread_q71", "pairwise_q2", "ball_1"):
        assert orders[name].index("sock") < orders[name].index("misc")
