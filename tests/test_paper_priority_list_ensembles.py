from dataclasses import replace
import random

import numpy as np
import pytest

from padjective import benchmark_runtime as runtime
from padjective import paper_priority_list as plist
from padjective import paper_priority_list_ensembles as ensembles
from padjective.benchmark_runtime import ProductRecord
from padjective.randomised_linear import consensus_predictions
from padjective.subspace_voting import feature_mask

P = 71
TAGS = [f"t{k:02d}" for k in range(12)]
LABELS = [1, 2, 1 + P, 1 + 2 * P, 2 + P, 1 + P + 3 * P * P]


def toy(seed, n=60):
    rng = random.Random(seed)
    return [ProductRecord(i, f"p{i}", rng.sample(TAGS, rng.randint(1, 4)), rng.choice(LABELS),
                          i % 5, "x", 1, ()) for i in range(n)]


@pytest.mark.parametrize("scheme", ["mask_0.75", "mask_0.5"])
def test_masked_member_matches_a_fit_on_masked_records(scheme):
    records = toy(3)
    fold, seed = 0, 12345
    training = [r for r in records if r.cv_fold != fold]
    testing = [r for r in records if r.cv_fold == fold]
    present = {tag for r in training for tag in r.tags}
    supported = np.array([j for j, tag in enumerate(TAGS) if tag in present])
    static = runtime._tag_order(training, [], fold, strategy="taxonomy_association")
    pooled, default = plist.fit_pooled(training, prime=P, q=71)
    order = ensembles.member_order(scheme, fold, seed, static, supported, TAGS)

    kept = {TAGS[j] for j in feature_mask(supported, ensembles.MASKS[scheme], fold, seed)}
    assert set(order) == kept

    def masked(rows):
        return [replace(r, tags=[tag for tag in r.tags if tag in kept]) for r in rows]

    assert order == runtime._tag_order(masked(training), [], fold, strategy="taxonomy_association")
    direct_pooled, _ = plist.fit_pooled(masked(training), prime=P, q=71)
    assert ensembles.member_predictions(training, testing, order, pooled, default) == \
        ensembles.member_predictions(masked(training), masked(testing), order, direct_pooled, default)


def test_random_order_members_keep_every_tag_and_differ_by_seed_and_fold():
    static = sorted(TAGS)
    supported = np.arange(len(TAGS))
    first = ensembles.member_order("random_order", 0, 7, static, supported, TAGS)
    assert sorted(first) == static
    assert first != ensembles.member_order("random_order", 0, 8, static, supported, TAGS)
    assert first != ensembles.member_order("random_order", 1, 7, static, supported, TAGS)
    assert first == ensembles.member_order("random_order", 0, 7, static[::-1], supported, TAGS)


def test_prefix_scores_follow_the_valid_path_consensus():
    y_train = np.array([1, 2, 1 + P, 2 + P])
    y_test = np.array([1, 1 + P, 2])
    bank = np.array([[1, 2, 1, 1], [1 + P, 1, 1 + P, 1 + P], [2 + P, 2 + P, 2, 2]], dtype=np.int64)
    context = dict(y_train=y_train, y_test=y_test)
    prefixes = ensembles.score_prefixes(bank, context)
    assert len(prefixes) == bank.shape[1]
    for size, row in enumerate(prefixes, start=1):
        direct, _ = consensus_predictions(bank[:, :size], np.unique(y_train), p=P, precision=7)
        assert row["loss_q2"] == pytest.approx(
            sum(plist.oracle.DISTANCE(int(a), int(b)) for a, b in zip(y_test, direct)) / 3)
    assert prefixes[-1]["exact_accuracy"] == 1


def test_family_adjusts_within_the_family():
    sizes = [10, 10, 10, 10, 10]
    tests = ensembles.family({
        "better": ([.10, .11, .09, .10, .12], [.20, .22, .19, .21, .20]),
        "same": ([.10, .12, .11, .10, .13], [.11, .11, .12, .10, .12])}, sizes)
    assert tests["better"]["mean_difference"] == pytest.approx(-.1)
    assert tests["better"]["holm_p_value"] >= tests["better"]["p_value"]
    assert tests["same"]["holm_p_value"] > .05
