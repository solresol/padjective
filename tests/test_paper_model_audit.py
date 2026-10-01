import numpy as np
import pytest
from scipy.sparse import csr_matrix

from padjective.paper_model_audit import member_check


def test_modular_zero_changes_the_default_trigger():
    q = 71**7
    features = csr_matrix([[1, 1], [1, 0], [0, 0]])
    counts = member_check(features, [q-1, 1], 4, [0, q-1, 0], [4, q-1, 4])
    assert counts == dict(rows=3, wrapped_integer_sums=1,
        modular_zero_with_nonzero_integer_sum=1,
        integer_and_quotient_reported_outputs_differ=1)


@pytest.mark.parametrize('raw,prediction', [([1], [1]), ([0], [0])])
def test_audit_rejects_incorrect_saved_predictions(raw, prediction):
    with pytest.raises(AssertionError):
        member_check(np.array([[1, 1]]), [71**7-1, 1], 4, raw, prediction)


def test_exact_integer_cancellation_uses_default_in_both_definitions():
    counts = member_check(np.array([[1, 1]]), [-2, 2], 4, [0], [4])
    assert counts['modular_zero_with_nonzero_integer_sum'] == 0
    assert counts['integer_and_quotient_reported_outputs_differ'] == 0
