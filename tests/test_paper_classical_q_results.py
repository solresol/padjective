import copy

import pytest

from padjective.paper_classical_q_results import LABELS, combine


def reports():
    meta = dict(snapshot_id="frozen", snapshot_digest="digest", products=6693,
                prime=71, q_values={"71": 71., "sqrt2": 2**.5, "2": 2.})
    def series(key, training, loss):
        return dict(configuration=key, training_q=training,
                    mean_losses={q: loss for q in meta["q_values"]},
                    fold_sd={q: .01 for q in meta["q_values"]},
                    folds=[dict(n=n, losses={q: loss for q in meta["q_values"]})
                           for n in [1300, 1337, 1348, 1372, 1336]],
                    exact_accuracy=.5, root_accuracy=.8)
    classical_keys = {"unn", "ulr", "dt", "levelwise", "dummy"}
    native = dict(meta, experiment_key="native", series=[
        series(k, q, .6 if q == "71" else .3)
        for k in sorted(set(LABELS) - classical_keys) for q in ["71", "sqrt2", "2"]])
    classical = dict(meta, experiment_key="classical", series=[
        series(k, "classification_objective_unchanged", .4) for k in sorted(classical_keys)])
    return native, classical


def test_retrained_comparison_changes_only_the_native_fits():
    native, classical = reports()
    result = combine(native, classical)["comparisons"]
    assert result["fixed_sqrt2"][0]["training_q"] == "classification_objective_unchanged"
    assert result["retrained_sqrt2"][0]["training_q"] == "sqrt2"
    for arm in result.values():
        row = next(r for r in arm if r["configuration"] == "unn")
        assert row["mean_loss"] == .4
        assert len(arm) == 16


def test_combination_rejects_a_different_snapshot_or_fold_population():
    native, classical = reports()
    other = copy.deepcopy(classical)
    other["snapshot_digest"] = "another-digest"
    with pytest.raises(ValueError, match="snapshot_digest"):
        combine(native, other)
    classical["series"][0]["folds"][0]["n"] = 1301
    with pytest.raises(ValueError, match="Fold populations"):
        combine(native, classical)
