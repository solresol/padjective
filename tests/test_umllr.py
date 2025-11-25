from padjective.umllr import (
    BattleRecord,
    ProductRecord,
    _p_adic_distance,
    _run_fold,
    _select_coefficient,
    _select_default_prediction,
)


def test_p_adic_distance_basic_properties() -> None:
    assert _p_adic_distance(10, 10, 5) == 0.0
    assert _p_adic_distance(10, 15, 5) == 0.2
    assert _p_adic_distance(1, 6, 5) == 0.2
    assert _p_adic_distance(1, 2, 5) == 1.0


def test_select_coefficient_prefers_smallest_candidate() -> None:
    values = [0, 5]
    base = 5
    # Candidates 0 and 5 produce the same loss; the smaller is chosen.
    assert _select_coefficient(values, base) == 0


def test_select_default_prediction_minimizes_padic_loss() -> None:
    # Products with no tags have encoded paths 5, 10, 15 (all divisible by 5)
    # Candidates are 1, 5, 10, 15
    # p-adic distance from 5 to {5, 10, 15}: 0 + 0.2 + 0.2 = 0.4
    # p-adic distance from 10 to {5, 10, 15}: 0.2 + 0 + 0.2 = 0.4
    # p-adic distance from 15 to {5, 10, 15}: 0.2 + 0.2 + 0 = 0.4
    # p-adic distance from 1 to {5, 10, 15}: 1 + 1 + 1 = 3.0
    # Tie between 5, 10, 15 - should pick smallest (5)
    no_tag_values = [5, 10, 15]
    candidate_values = [1, 5, 10, 15]
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 5


def test_select_default_prediction_empty_no_tag_values() -> None:
    # When no products have missing tags, fall back to most common
    no_tag_values = []
    candidate_values = [1, 5, 5, 10]  # 5 is most common
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 5


def test_select_default_prediction_picks_best_for_ultrametric() -> None:
    # Test case where one candidate is clearly better due to ultrametricity
    # Products with no tags: 25, 50 (both divisible by 25)
    # p-adic distance from 25 to {25, 50}: 0 + 0.04 = 0.04
    # p-adic distance from 50 to {25, 50}: 0.04 + 0 = 0.04
    # p-adic distance from 1 to {25, 50}: 1 + 1 = 2.0
    # Tie: pick smallest (25)
    no_tag_values = [25, 50]
    candidate_values = [1, 25, 50]
    base = 5
    assert _select_default_prediction(no_tag_values, candidate_values, base) == 25


def test_run_fold_returns_coefficients_and_predictions() -> None:
    records = [
        ProductRecord(product_id=1, tags=["ALPHA", "BETA"], encoded_path=11, cv_fold=0),
        ProductRecord(product_id=2, tags=["ALPHA"], encoded_path=4, cv_fold=1),
        ProductRecord(product_id=3, tags=["BETA"], encoded_path=9, cv_fold=0),
    ]
    battles = [BattleRecord(winner_tag="ALPHA", loser_tag="BETA", cv_fold=0)]

    fold_result = _run_fold(1, records, battles, base=5)

    assert fold_result.cv_fold == 1
    assert [coeff.tag for coeff in fold_result.coefficients] == ["ALPHA", "BETA"]
    assert fold_result.coefficients[0].coefficient == 11
    assert fold_result.coefficients[1].coefficient == 0

    assert len(fold_result.predictions) == 1
    prediction = fold_result.predictions[0]
    assert prediction.product_id == 2
    assert prediction.predicted_value == 11
    assert prediction.true_value == 4
    assert prediction.loss == 1.0
    assert fold_result.loss == 1.0
