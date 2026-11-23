import random

import pytest

from padjective import benchmark_runtime as runtime
from padjective import paper_priority_list as plist
from padjective import paper_priority_list_dynamic as dynamic
from padjective.benchmark_runtime import ProductRecord

P = 71
TAGS = [f"t{k:02d}" for k in range(10)]


def record(index, tags, target, positions=()):
    return ProductRecord(index, f"p{index}", list(tags), target, 1, str(target), 1, tuple(positions))


def toy(seed, n=50):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        tags = rng.sample(TAGS, rng.randint(1, 4))
        in_title = rng.sample(tags, rng.randint(0, len(tags)))
        positions = [(tag, rng.randint(0, 1), rng.randint(0, 40)) for tag in in_title]
        rows.append(record(i, tags, rng.choice([1, 2, 1 + P, 2 + P, 1 + 2 * P]), positions))
    return rows


def test_a_mixed_tag_moves_up_once_a_purer_tag_has_claimed_its_products():
    # "gift" is on ten sock products, all also tagged "socks", and on three mugs.
    training = [record(i, ["socks", "gift"], 1) for i in range(10)]
    training += [record(10 + i, ["gift"], 2) for i in range(3)]
    training += [record(13 + i, ["mug"], 2) for i in range(2)] + [record(15, ["mug"], 3)]
    rule = dynamic.static_rule("taxonomy_association", 0)
    static = rule(training)
    assert static == ["socks", "gift", "mug"]
    # "gift" already sits above "mug" here, so make "mug" purer than "gift" overall.
    training += [record(16 + i, ["mug"], 2) for i in range(8)]
    static = rule(training)
    assert static == ["socks", "mug", "gift"]
    # Once "socks" has taken the ten sock products, "gift" is left with three mugs.
    assert plist.dynamic_order(training, rule, static) == ["socks", "gift", "mug"]


@pytest.mark.parametrize("strategy", dynamic.RULES)
def test_dynamic_order_reapplies_the_static_rule_to_what_is_left(strategy):
    for seed in range(8):
        training = toy(seed)
        rule = dynamic.static_rule(strategy, 0)
        static = rule(training)
        order = plist.dynamic_order(training, rule, static)
        assert order[0] == static[0] and sorted(order) == sorted(static)
        remaining, placed = list(training), 0
        while remaining:
            battles = runtime._derive_battles(remaining) if strategy == "battle_elo" else []
            expected = runtime._tag_order(remaining, battles, 0, strategy=strategy)[0]
            assert order[placed] == expected
            before = len(remaining)
            remaining = [r for r in remaining if expected not in r.tags]
            assert len(remaining) < before
            placed += 1
        # Every tag after the placed ones heads nothing and keeps its static order.
        tail = order[placed:]
        assert tail == [tag for tag in static if tag in tail]
        rank, cells = plist.fit_cells(training, order, prime=P, q=71)
        assert set(cells) == set(order[:placed])


def test_heads_follow_the_order():
    rows = [record(0, ["a", "b"], 1), record(1, ["b"], 1), record(2, ["z"], 1)]
    assert dynamic.heads(rows, ["b", "a"]) == ["b", "b", None]
