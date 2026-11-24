import random

import pytest

from padjective import benchmark_runtime as runtime
from padjective import paper_priority_list as plist
from padjective import paper_priority_list_levels as levels
from padjective.benchmark_runtime import ProductRecord

P = levels.P
LABELS = [1, 2, 1 + P, 1 + 2 * P, 2 + P, 1 + P + 3 * P * P, 1 + P + 4 * P * P]


def record(index, tags, target, fold=0):
    return ProductRecord(index, f"p{index}", list(tags), target, fold, str(target), 1, ())


def toy(seed, n=80, tags=10):
    rng = random.Random(seed)
    names = [f"t{k}" for k in range(tags)]
    return [record(i, rng.sample(names, rng.randint(1, 4)), rng.choice(LABELS), i % 5) for i in range(n)]


def test_digits_read_the_path_from_the_root():
    code = 3 + 5 * P + 7 * P * P
    assert [levels.digit(code, level) for level in range(4)] == [3, 5, 7, 0]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_limit_zero_is_the_plain_backfilled_list(seed):
    records = toy(seed)
    training = [r for r in records if r.cv_fold != 0]
    testing = [r for r in records if r.cv_fold == 0] + [record(999, ["unseen"], 1)]
    order = runtime._tag_order(training, [], 0, strategy="taxonomy_association")
    pooled, default = plist.fit_pooled(training, prime=P, q=71)
    rank, filled = plist.fill(training, order, pooled, prime=P, q=71)
    expected = plist.predict_priority(testing, rank, filled["cell_backfill"], default)
    model = levels.fit_node(training, limit=0)
    got = [levels.predict(model, r.tags) for r in testing]
    assert [value for value, _ in got] == expected[0]
    assert [flag for _, flag in got] == expected[1]


def test_a_tag_that_is_pure_at_the_top_level_decides_the_top_level():
    # "men" is always under root 1 but splits below it. "odd" was seen once, under root 2.
    training = [record(i, ["men"], 1 + P * (1 + i % 2)) for i in range(6)] + [record(6, ["odd"], 2)]
    product = ["men", "odd"]
    plain = levels.fit_node(training, limit=0)
    assert levels.predict(plain, product) == (2, True)
    by_level = levels.fit_node(training, limit=1)
    assert by_level["kind"] == "split" and by_level["rank"]["men"] < by_level["rank"]["odd"]
    # Under root 1 the plain list knows only "men"; its two labels tie and the smaller wins.
    assert levels.predict(by_level, product) == (1 + P, True)
    assert levels.predict(levels.fit_node(training, limit=levels.FULL), product) == (1 + P, True)


def test_a_path_that_stops_early_is_its_own_child():
    training = [record(0, ["short"], 1), record(1, ["short"], 1), record(2, ["long"], 1 + P),
                record(3, ["long"], 1 + P), record(4, ["long", "deep"], 1 + P + 2 * P * P)]
    model = levels.fit_node(training, limit=levels.FULL)
    assert model["kind"] == "split" and set(model["children"]) == {1}
    below = model["children"][1]
    assert set(below["children"]) == {0, 1}
    assert below["children"][0] == dict(kind="leaf", value=1)
    assert levels.predict(model, ["short"]) == (1, True)
    assert levels.predict(model, ["long"]) == (1 + P, True)
    assert levels.predict(model, ["long", "deep"]) == (1 + P + 2 * P * P, True)
    # An unknown tag takes the most common digit at each split.
    assert levels.predict(model, ["new"]) == (1 + P, False)
    assert levels.size(model) == 3


def test_vote_nodes_take_the_plurality_and_break_ties_downwards():
    training = [record(0, ["a"], 1), record(1, ["a"], 1), record(2, ["b"], 2), record(3, ["c"], 2),
                record(4, ["d"], 3)]
    model = levels.fit_node(training, limit=1, select="vote")
    assert model["values"] == {"a": 1, "b": 2, "c": 2, "d": 3}
    assert levels.predict(model, ["a", "b", "c"]) == (2, True)
    assert levels.predict(model, ["a", "d"]) == (1, True)
    assert levels.predict(model, ["zzz"]) == (1, False)
    # The priority node follows the first tag in the order instead.
    assert levels.predict(levels.fit_node(training, limit=1), ["a", "b", "c"]) == (1, True)


def test_consistent_children_see_only_the_products_sent_to_them_correctly():
    # "mix" heads three products: two under root 1 and one under root 2 with a deeper label.
    training = [record(0, ["mix"], 1 + P), record(1, ["mix"], 1 + P), record(2, ["mix"], 2 + 5 * P),
                record(3, ["two"], 2 + P), record(4, ["two"], 2 + 2 * P), record(5, ["two"], 2 + 2 * P)]
    teacher = levels.fit_node(training, limit=levels.FULL)
    kept = levels.fit_node(training, limit=levels.FULL, consistent=True)
    # Product 2 is sent to root 1, so the consistent child for root 2 never sees digit 5.
    assert set(teacher["children"][2]["children"]) == {1, 2, 5}
    assert set(kept["children"][2]["children"]) == {1, 2}
    assert levels.predict(teacher, ["two"]) == levels.predict(kept, ["two"]) == (2 + 2 * P, True)


def test_a_child_with_no_correctly_sent_product_falls_back_to_all_of_its_products():
    training = [record(0, ["mix"], 1), record(1, ["mix"], 1), record(2, ["mix"], 2 + P),
                record(3, ["mix"], 2 + 2 * P)]
    model = levels.fit_node(training, limit=levels.FULL, consistent=True)
    assert set(model["children"]) == {1, 2}
    assert model["children"][2]["kind"] == "split"


def test_the_shared_order_is_used_at_every_split():
    training = [record(i, ["men"], 1 + P * (1 + i % 2)) for i in range(6)] + [record(6, ["odd"], 2)]
    shared = ["odd", "men"]
    model = levels.fit_node(training, limit=1, shared_order=shared)
    assert model["rank"] == {"odd": 0, "men": 1}
    assert levels.predict(model, ["men", "odd"]) == (2, True)


@pytest.mark.parametrize("name", list(levels.VARIANTS))
def test_every_variant_fits_products_with_their_own_tag_exactly(name):
    rng = random.Random(9)
    training = [record(i, [f"own{i}", rng.choice(["x", "y", "z"])], rng.choice(LABELS)) for i in range(40)]
    options = dict(levels.VARIANTS[name])
    if options.pop("shared", False):
        options["shared_order"] = runtime._tag_order(training, [], 0, strategy="taxonomy_association")
    model = levels.fit_node(training, **options)
    if options.get("select") == "vote":
        return
    assert [levels.predict(model, r.tags)[0] for r in training] == [r.encoded_path for r in training]


def test_split_nodes_minimise_top_level_errors_for_their_order():
    records = toy(5, n=60, tags=6)
    model = levels.fit_node(records, limit=1)
    order = sorted(model["rank"], key=model["rank"].get)
    fitted = sum(levels.choose(model, r.tags)[0] != levels.digit(r.encoded_path, 0) for r in records)
    heads = [plist.head(r.tags, model["rank"], model["rank"]) for r in records]
    best = 0
    for tag in order:
        cell = [levels.digit(r.encoded_path, 0) for r, top in zip(records, heads) if top == tag]
        best += len(cell) - max((cell.count(value) for value in set(cell)), default=0)
    assert fitted == best


def test_rows_report_prefix_accuracy_and_collect_tests_both_references():
    records = toy(7, n=200)
    results = []
    for fold in range(5):
        training = [r for r in records if r.cv_fold != fold]
        testing = [r for r in records if r.cv_fold == fold]
        shared = runtime._tag_order(training, [], fold, strategy="taxonomy_association")
        rows = {name: levels.variant_row(training, testing, options, shared)
                for name, options in levels.VARIANTS.items()}
        for row in rows.values():
            chain = [row[key] for key in ("root_accuracy", "prefix2_accuracy", "prefix3_accuracy", "exact_accuracy")]
            assert all(a >= b - 1e-12 for a, b in zip(chain, chain[1:]))
        assert rows["list"]["nodes"] == 1 and rows["levels_all"]["nodes"] > rows["levels_1"]["nodes"] > 1
        results.append(dict(split="product", fold=fold, n=len(testing), rows=rows))
    report = levels.collect(results)
    assert set(report["tests"]["minus_list"]) == set(levels.VARIANTS) - {"list"}
    assert set(report["tests"]["minus_spread_list"]) == set(levels.VARIANTS) - set(levels.REFERENCES)
    assert report["fits"]["levels_all"]["nodes"] > 1


def test_spread_order_puts_tight_tags_first_and_larger_tags_before_smaller():
    training = [record(0, ["tight", "big"], 1), record(1, ["tight"], 1), record(2, ["big"], 1),
                record(3, ["big"], 1), record(4, ["loose"], 1), record(5, ["loose"], 2)]
    assert levels.spread_order(training) == ["big", "tight", "loose"]
