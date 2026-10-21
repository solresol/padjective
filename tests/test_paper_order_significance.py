import copy
import json
from pathlib import Path

import pytest

from padjective.paper_order_significance import analyse


@pytest.fixture
def runs():
    path = Path(__file__).parents[1] / "docs/tag-order-significance/results.json"
    return json.loads(path.read_text())["runs"]


def test_published_fold_replay_and_corrected_contrasts(runs):
    result = analyse(runs)
    rows = {r["strategy"]: r for r in result["contrasts"]}
    assert rows["random"]["holm_p"] == pytest.approx(.0022964598711849387)
    assert rows["frequency"]["holm_p"] == pytest.approx(.019193480317377068)
    assert all(r["df"] == 4 and r["ci95"][0] > 0 for r in rows.values())


def test_random_seed_variation_is_not_extra_replication(runs):
    original = analyse(runs)
    altered = copy.deepcopy(runs)
    random = [r for r in altered if r["tag_order_strategy"] == "random"]
    random[0]["folds"][0]["padic_loss_mean"] += .01
    random[1]["folds"][0]["padic_loss_mean"] -= .01
    changed = analyse(altered)
    assert changed["contrasts"][0]["p_value"] == pytest.approx(original["contrasts"][0]["p_value"])


def test_reject_incomplete_seed_bank(runs):
    with pytest.raises(ValueError, match="strategy/seed grid"):
        analyse(runs[:-1])


def test_reject_other_snapshot_denominators(runs):
    runs[0]["folds"][0]["num_test_samples"] = 1268
    with pytest.raises(ValueError, match="frozen fold denominator"):
        analyse(runs)


def test_reject_unreproduced_published_mean(runs):
    runs[0]["folds"][0]["padic_loss_mean"] += .01
    with pytest.raises(ValueError, match="Published mean not reproduced"):
        analyse(runs)
