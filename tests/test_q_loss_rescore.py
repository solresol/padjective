import pytest

from padjective.q_loss_rescore import rescore


def test_near_one_weights_approach_exact_error_with_fixed_encoding():
    # One exact prediction, one root error, and one error at each of the next
    # two depths. The valuation base remains 71 at every scoring Q.
    qs = {"1.01": 1.01, "1.1": 1.1, "limit": 1 + 1e-8}
    result = rescore([1, 1, 1, 1], [1, 2, 72, 5042], qs)
    for key, q in qs.items():
        assert result["losses"][key] == pytest.approx((1 + q**-1 + q**-2) / 4)
    assert result["losses"]["limit"] == pytest.approx(.75, abs=1e-8)
    assert result["root_accuracy"] == .75
    assert result["exact_accuracy"] == .25
    assert result["losses"]["1.01"] > result["losses"]["1.1"]
