import pytest

from padjective.paper_classical_q_experiment import make_estimator, reference_check
from padjective.paper_low_q_experiment import score


def test_reference_check_rejects_wrong_population_or_predictions():
    actual = score([1, 1, 1], [1, 72, 2])
    ref = dict(padic_loss_mean=actual["losses"]["71"],
               exact_accuracy=1 / 3, num_test_samples=3)
    assert reference_check(actual, ref)["matches"]
    assert not reference_check(actual, dict(ref, num_test_samples=4))["matches"]
    assert not reference_check(score([1, 1, 1], [1, 1, 2]), ref)["matches"]


def test_neural_reference_uses_papers_largest_width():
    model = make_estimator("unn")
    assert model.hidden_layer_sizes == (2000,)
    assert model.max_iter == 10000
    assert model.batch_size == 256
    assert model.random_state == 42


def test_scores_keep_encoding_base_separate_from_depth_weight():
    values = score([1, 1, 1, 1], [1, 2, 72, 5042])
    assert values["first_disagreement_counts"] == {0: 1, 1: 1, 2: 1}
    assert values["losses"]["2"] == pytest.approx((1 + .5 + .25) / 4)
    assert values["losses"]["sqrt2"] == pytest.approx((1 + 2**(-.5) + .5) / 4)
