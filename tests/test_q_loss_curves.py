import copy
import math
from statistics import mean, stdev

import pytest

from padjective.q_loss_curves import FOLD_SIZES, STYLES, build_data


def reports():
    grid = {"sqrt2": 2**.5, "2": 2., "3": 3., "5": 5., "10": 10., "71": 71.}
    meta = dict(snapshot_id="frozen", snapshot_digest="digest", products=6693,
                prime=71, q_values=grid)
    def series(key, training, factor=1):
        folds = [dict(n=n, losses={q: factor * (i + 1) / (10 * value**.5)
                                  for q, value in grid.items()})
                 for i, n in enumerate(FOLD_SIZES)]
        return dict(configuration=key, training_q=training, folds=folds,
                    mean_losses={q: mean(f["losses"][q] for f in folds) for q in grid},
                    fold_sd={q: stdev(f["losses"][q] for f in folds) for q in grid})
    native_keys = {"taxonomy_association", "ensemble_0.75_243", "ensemble_1_243"}
    native = dict(meta, experiment_key="native", series=[
        series(key, training, factor) for key in sorted(native_keys)
        for training, factor in [("71", 1), ("2", .01)]])
    classical = dict(meta, experiment_key="classical", series=[
        series(key, "classification_objective_unchanged") for key in sorted(set(STYLES) - native_keys)])
    return native, classical


def test_plot_uses_log_of_fold_mean_and_does_not_mix_training_q():
    native, classical = reports()
    data = build_data(native, classical)
    curve = next(c for c in data["curves"] if c["configuration"] == "taxonomy_association")
    assert curve["training_q"] == "71"
    point = next(p for p in curve["points"] if p["q_label"] == "2")
    expected = [i / (10 * 2**.5) for i in range(1, 6)]
    assert point["mean_loss"] == pytest.approx(sum(expected) / 5)
    assert point["log10_mean_loss"] == pytest.approx(math.log10(sum(expected) / 5))
    assert point["log10_mean_loss"] != pytest.approx(sum(math.log10(v) for v in expected) / 5)
    assert point["log10_loss_minus_sd"] == pytest.approx(math.log10(mean(expected) - stdev(expected)))
    assert len(data["curves"]) == 8


def test_plot_rejects_different_population_or_snapshot():
    native, classical = reports()
    other = copy.deepcopy(classical)
    other["snapshot_digest"] = "different"
    with pytest.raises(ValueError, match="snapshot_digest"):
        build_data(native, other)
    classical["series"][0]["folds"][0]["n"] = 1301
    with pytest.raises(ValueError, match="Fold populations"):
        build_data(native, classical)


def test_plot_rejects_silent_changes_to_stored_aggregates():
    native, classical = reports()
    classical["series"][0]["mean_losses"]["sqrt2"] += .001
    with pytest.raises(ValueError, match="Stored aggregates"):
        build_data(native, classical)
