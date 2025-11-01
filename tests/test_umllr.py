from padjective.umllr import (
    BattleRecord,
    ProductRecord,
    _p_adic_distance,
    _run_fold,
    _select_coefficient,
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
