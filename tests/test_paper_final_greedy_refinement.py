from fractions import Fraction
from itertools import product

from padjective.paper_final_greedy_refinement import (
    P, digit_length, infinite_support_disjoint, integer_distance, statistics,
)
from padjective.paper_refinement_comparison import classify


def test_negative_tails_and_positive_finite_support_are_exact():
    assert infinite_support_disjoint([-3, 1], 3)
    assert not infinite_support_disjoint([-1, 3], 3)
    assert not infinite_support_disjoint([-1, -3], 3)
    assert not infinite_support_disjoint([3, 2], 2)
    assert infinite_support_disjoint([1, 3], 3)
    assert infinite_support_disjoint([0, 0], 3)
    assert infinite_support_disjoint([-9, 0], 3)


def test_finite_residue_support_can_disagree_with_signed_infinite_support():
    # Truncation can hide an entire nonzero contribution.
    assert classify([-1, -2], 2, 1)["disjoint"]
    assert not infinite_support_disjoint([-1, -2], 2)
    assert classify([-1, -2], 2, 1)["hidden_contributions"] == 1
    # Even with both terms visible, canonical lifts are different actual Z_2 integers.
    assert classify([-3, -2], 2, 2)["disjoint"]
    assert classify([-3, -2], 2, 2)["hidden_contributions"] == 0
    assert not infinite_support_disjoint([-3, -2], 2)
    assert infinite_support_disjoint([1, 2], 2)


def test_signed_support_matches_a_provably_sufficient_finite_depth_exhaustively():
    for p in (2, 3):
        for terms in product(range(-7, 8), repeat=3):
            bound = max(digit_length(z, p) for z in terms)+1
            exact = infinite_support_disjoint(terms, p)
            finite = classify(terms, p, bound)
            assert exact == finite["disjoint"]
            assert not exact or finite["valuation_separated"]


def test_integer_distance_does_not_drop_nonzero_high_precision_errors():
    assert integer_distance(1, 1+P**7) == Fraction(1, P**7)
    assert integer_distance(1, 1-P**8) == Fraction(1, P**8)
    assert integer_distance(-3, -3) == 0
    assert integer_distance(1, -2, 3) == Fraction(1, 3)


def test_wrong_only_summary_and_reference_depth_accounting():
    rows = [dict(target=1, prediction=1, loss=Fraction(0), taxonomy_id="t", raw=1,
                 active_count=2, negative_count=0, target_depth=1),
            dict(target=1, prediction=1+P**7, loss=Fraction(1, P**7), taxonomy_id="t", raw=1+P**7,
                 active_count=2, negative_count=1, target_depth=1)]
    result = statistics(rows)
    assert result["exact_hit_rate"] == .5
    assert result["incorrect_mean_padic_loss"] == float(Fraction(1, P**7))
    assert result["mean_padic_loss"] == float(Fraction(1, 2*P**7))
    assert result["incorrect_matching_entire_reference_path"] == 1
    assert result["predictions_outside_canonical_quotient"] == 1
    assert statistics([])["incorrect_mean_padic_loss"] is None
