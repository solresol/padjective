from collections import Counter
import random

import networkx as nx
import pytest

from padjective import paper_priority_list as plist
from padjective import paper_priority_list_transductive as trans
from padjective.benchmark_runtime import ProductRecord

P = 71
LABELS = [1, 2, 1 + P, 1 + 2 * P, 2 + P]


def record(index, tags, target, fold=0):
    return ProductRecord(index, f"p{index}", list(tags), target, fold, str(target), 1, ())


def test_values_pass_one_step_per_round():
    # Product 0 is covered. Products 1 and 2 are one and two steps away; 3 shares nothing.
    tags = [["u"], ["u", "w"], ["w"], ["z"]]
    base, covered = [5, 9, 9, 9], [True, False, False, False]
    values, reached, local = trans.spread(tags, base, covered, prime=P, q=71)
    assert values == [5, 5, 5, 9]
    assert reached == [0, 1, 2, None]
    assert {tag: entry[3] for tag, entry in local.items()} == {"u": 5, "w": 5}
    values, reached, _ = trans.spread(tags, base, covered, prime=P, q=71, rounds=1)
    assert values == [5, 5, 9, 9]
    assert reached == [0, 1, None, None]


def test_the_tightest_tag_wins_and_the_vote_takes_the_medoid():
    # "pure" sits on one covered product; "wide" and "also" each have a majority value of 2.
    tags = [["pure"], ["wide", "also"], ["wide", "also"], ["wide"], ["also"], ["pure", "wide", "also"]]
    base = [1, 2, 2, 3, 3, 0]
    covered = [True] * 5 + [False]
    tight, _, local = trans.spread(tags, base, covered, prime=P, q=71)
    assert local["pure"][0] == 0 and local["wide"][0] > 0
    assert tight[5] == 1
    voted, _, _ = trans.spread(tags, base, covered, prime=P, q=71, choice="vote")
    assert voted[5] == 2
    assert tight[:5] == voted[:5] == base[:5]


def test_ties_in_tightness_go_to_the_tag_on_more_products():
    tags = [["small"], ["large"], ["large"], ["small", "large"]]
    values, _, _ = trans.spread(tags, [1, 2, 2, 0], [True, True, True, False], prime=P, q=71)
    assert values[3] == 2


def test_tag_values_are_not_revised_by_later_rounds():
    # "u" is valued from product 0 alone. Product 3 also carries it but takes the value of "v".
    tags = [["u"], ["v"], ["v"], ["v", "u"], ["w", "v"], ["w"]]
    base, covered = [1, 2, 2, 0, 0, 0], [True, True, True, False, False, False]
    values, reached, local = trans.spread(tags, base, covered, prime=P, q=71)
    assert reached == [0, 0, 0, 1, 1, 2]
    assert values == [1, 2, 2, 2, 2, 2]
    assert local["u"] == (0, -1, "u", 1)
    assert local["v"] == (0, -2, "v", 2) and local["w"] == (0, -1, "w", 2)


def test_rounds_equal_distance_from_a_covered_product():
    rng = random.Random(11)
    names = [f"t{k}" for k in range(14)]
    for _ in range(60):
        n = rng.randint(3, 16)
        tags = [rng.sample(names, rng.randint(0, 3)) for _ in range(n)]
        covered = [rng.random() < .3 for _ in range(n)]
        base = [rng.choice(LABELS) for _ in range(n)]
        values, reached, _ = trans.spread(tags, base, covered, prime=P, q=71)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        graph.add_edges_from((i, ("tag", tag)) for i, row in enumerate(tags) for tag in row)
        sources = [i for i, flag in enumerate(covered) if flag]
        distance = nx.multi_source_dijkstra_path_length(graph, sources) if sources else {}
        assert reached == [distance[i] // 2 if i in distance else None for i in range(n)]
        for i in range(n):
            if covered[i] or reached[i] is None:
                assert values[i] == base[i]
            else:
                assert values[i] in {base[j] for j in sources}


def test_local_first_changes_only_products_whose_unfitted_tags_disagree():
    tags = [["mixed"], ["mixed"], ["mixed"], ["mixed", "pure"], [], ["mixed"]]
    base, covered = [2, 2, 3, 3, 4, 0], [True] * 5 + [False]
    values, reached = trans.smooth(tags, base, covered, prime=P, q=71)
    # Product 2 moves to the majority. Product 3 keeps its value through its pure tag.
    assert values == [2, 2, 2, 3, 4, 2]
    assert reached == [0, 0, 0, 0, 0, 1]


def test_store_variants_fall_back_in_order_and_leave_covered_products_alone():
    tags = [["u"], [], [], ["u"], ["z"], []]
    base, covered = [1, 2, 2, 9, 9, 9], [True, True, True, False, False, False]
    targets = [1, 2, 2, 1, 1 + P, 1 + P]
    out = trans.store_variants(tags, base, covered, targets, prime=P, q=71)
    assert out["base"] == (base, covered)
    assert out["store_default"] == ([1, 2, 2, 2, 2, 2], [True] * 6)
    assert out["tags_one_round"] == out["tags"] == ([1, 2, 2, 1, 9, 9], [True] * 4 + [False] * 2)
    assert out["tags_then_store"] == ([1, 2, 2, 1, 2, 2], [True] * 6)
    assert out["local_first"][0][:3] == base[:3]
    # The diagnostic takes the best single code for the three uncovered labels.
    assert out["store_oracle_constant"][0] == [1, 2, 2, 1 + P, 1 + P, 1 + P]


def test_a_store_with_no_covered_product_stays_on_the_default():
    out = trans.store_variants([["u"], ["u"]], [9, 9], [False, False], [1, 1], prime=P, q=71)
    for name in ("store_default", "tags", "tags_then_store", "local_first"):
        assert out[name] == ([9, 9], [False, False])


def test_fold_rows_keep_stores_apart_except_at_fold_scope():
    fitted = {"known": 1}
    testing = [record(0, ["known", "u"], 1), record(1, ["u"], 1), record(2, ["u"], 1),
               record(3, ["known"], 1), record(4, [], 2)]
    stores = {"p0": "a", "p1": "a", "p2": "b", "p3": "b", "p4": "c"}
    base, covered = plist.predict_priority(testing, {"known": 0}, fitted, 7)
    assert base == [1, 7, 7, 1, 7] and covered == [True, False, False, True, False]
    rows = trans.fold_rows(testing, stores, base, covered, fitted, prime=P, q=71)
    assert rows["base"]["exact_accuracy"] == pytest.approx(2 / 5)
    assert rows["base"]["reached_share"] == 0 and rows["base"]["uncovered_loss_q2"] == 1
    # Store b has no covered product with "u", and store c has no covered product at all.
    assert rows["tags"]["reached_share"] == pytest.approx(1 / 3)
    assert rows["tags"]["reached_loss_q2"] == 0 and rows["tags"]["reached_default_loss_q2"] == 1
    assert rows["tags"]["uncovered_loss_q2"] == pytest.approx(2 / 3)
    assert rows["tags_fold_scope"]["reached_share"] == pytest.approx(2 / 3)
    assert rows["store_default"]["exact_accuracy"] == pytest.approx(4 / 5)
    assert rows["store_oracle_constant"]["exact_accuracy"] == 1
    assert rows["local_first"]["changed_share"] == 0

    facts = trans.facts(testing, stores, covered, fitted, prime=P, q=71)
    assert facts["stores"] == 3 and facts["stores_without_covered_product"] == 1
    assert facts["uncovered_in_store_without_covered_product"] == pytest.approx(1 / 3)
    assert facts["unfitted_tags"] == 1 and facts["unfitted_tags_in_several_stores"] == 1
    assert facts["uncovered_share"] == pytest.approx(3 / 5)
    assert facts["uncovered_without_tags"] == pytest.approx(1 / 3)
    assert facts["reached_in_round_1"] == pytest.approx(1 / 3)
    assert facts["reached_in_later_rounds"] == 0 and facts["rounds"] == 1


def test_collect_tests_every_variant_against_the_base_list():
    rng = random.Random(4)
    results = []
    for fold in range(5):
        testing = [record(i, rng.sample(["known", "u", "v", "w"], rng.randint(0, 3)), rng.choice(LABELS))
                   for i in range(40)]
        stores = {r.product_key: rng.choice("abc") for r in testing}
        fitted = {"known": 1}
        base, covered = plist.predict_priority(testing, {"known": 0}, fitted, 2)
        rows = trans.fold_rows(testing, stores, base, covered, fitted, prime=P, q=71)
        results.append(dict(fold=fold, n=len(testing), rows={fill: rows for fill in trans.FILLS},
                            facts=trans.facts(testing, stores, covered, fitted, prime=P, q=71)))
    report = trans.collect(results)
    names = set(report["fits"]["cell_backfill"])
    assert {"base", "store_default", "tags", "tags_fold_scope", "store_oracle_constant"} <= names
    tested = Counter(key.split("|")[1] for key in report["tests"]["minus_base"])
    assert tested == {fill: len(names) - 2 for fill in trans.FILLS}
    assert len(report["tests"]["contrasts"]) == len(trans.CONTRASTS) * len(trans.FILLS)
    base = report["fits"]["cell_backfill"]["base"]
    assert report["fits"]["cell_backfill"]["store_oracle_constant"]["loss_q2"] <= base["loss_q2"]
    assert all(row["holm_p_value"] >= row["p_value"] for row in report["tests"]["minus_base"].values())
