"""Tests for the Mihara taxonomy comparison."""

from __future__ import annotations

import random

import pytest

from padjective.taxonomy_mihara_comparison import (
    fit_mihara_digitwise,
    fit_modulo_p_consensus,
    run_fold,
    select_fold_tags,
    solve_square_system_mod_p,
)
from padjective.umllr import ProductRecord


def test_solve_square_system_mod_p_handles_invertible_and_singular_systems() -> None:
    assert solve_square_system_mod_p([[1, 1], [1, 2]], [3, 0], 5) == (1, 2)
    assert solve_square_system_mod_p([[1, 1], [2, 2]], [3, 1], 5) is None


def test_digitwise_fit_recovers_sparse_digit_noise() -> None:
    p = 5
    precision = 3
    modulus = p**precision
    expected_coefficients = (7, 19)
    expected_intercept = 11
    rng = random.Random(13)
    features: list[tuple[int, int]] = []
    targets: list[int] = []

    for _ in range(500):
        row = (rng.randrange(modulus), rng.randrange(modulus))
        clean_target = (
            sum(
                feature * coefficient
                for feature, coefficient in zip(
                    row, expected_coefficients, strict=True
                )
            )
            + expected_intercept
        ) % modulus
        remaining = clean_target
        noisy_target = 0
        for exponent in range(precision):
            digit = remaining % p
            remaining //= p
            if rng.random() < 0.03:
                digit = (digit + rng.randrange(1, p)) % p
            noisy_target += digit * p**exponent
        features.append(row)
        targets.append(noisy_target)

    fit = fit_mihara_digitwise(
        features,
        targets,
        p=p,
        precision=precision,
        trials=64,
        seed=7,
    )

    assert fit.coefficients == expected_coefficients
    assert fit.intercept == expected_intercept
    assert fit.digits_fitted == precision
    assert fit.accepted_prefix_digits == precision
    assert fit.all_digits_accepted
    assert fit.stop_reason == "precision_complete"
    assert all(diagnostic.validation_inlier_fraction > 0.9 for diagnostic in fit.diagnostics)


def test_modulo_fit_marks_non_affine_targets_unaccepted() -> None:
    features = [(index % 5, (index // 5) % 5) for index in range(100)]
    targets = [(index * index + 2 * index + 1) % 5 for index in range(100)]

    fit = fit_modulo_p_consensus(
        features,
        targets,
        p=5,
        trials=24,
        seed=3,
    )

    assert fit is not None
    assert not fit.accepted
    assert fit.validation_inlier_fraction < 0.9


def test_digitwise_fit_reports_rank_deficient_design() -> None:
    fit = fit_mihara_digitwise(
        [(0, 0), (1, 1), (2, 2), (3, 3)],
        [0, 1, 2, 3],
        p=5,
        precision=2,
        trials=4,
    )

    assert fit.digits_fitted == 0
    assert fit.stop_reason == "rank_deficient_active_design"
    assert not fit.all_digits_accepted


def test_tag_selection_is_fold_local_and_deterministic() -> None:
    training = [
        ProductRecord(1, ["PURE", "COMMON"], 1, 1, taxonomy_id="A"),
        ProductRecord(2, ["PURE", "COMMON"], 1, 1, taxonomy_id="A"),
        ProductRecord(3, ["MIXED", "COMMON"], 2, 1, taxonomy_id="A"),
        ProductRecord(4, ["MIXED", "COMMON"], 3, 1, taxonomy_id="B"),
    ]

    assert select_fold_tags(
        training, max_tags=2, strategy="taxonomy_association"
    ) == ("PURE", "COMMON")
    assert select_fold_tags(training, max_tags=2, strategy="frequency") == (
        "COMMON",
        "MIXED",
    )


def test_run_fold_scores_held_out_raw_padic_predictions() -> None:
    records = [
        ProductRecord(1, ["A"], 3, 1, taxonomy_id="T3", taxonomy_depth=2),
        ProductRecord(2, [], 1, 1, taxonomy_id="T1", taxonomy_depth=2),
        ProductRecord(3, ["A"], 3, 2, taxonomy_id="T3", taxonomy_depth=2),
        ProductRecord(4, [], 1, 2, taxonomy_id="T1", taxonomy_depth=2),
        ProductRecord(5, ["A", "HELDOUT_ONLY"], 3, 0, taxonomy_id="T3", taxonomy_depth=2),
        ProductRecord(6, ["HELDOUT_ONLY"], 1, 0, taxonomy_id="T1", taxonomy_depth=2),
    ]

    result = run_fold(
        0,
        records,
        p=5,
        precision=2,
        max_tags=1,
        trials=8,
        seed=2,
    )

    assert result.selected_tags == ("A",)
    assert "HELDOUT_ONLY" not in result.selected_tags
    assert result.fit.coefficients == (2,)
    assert result.fit.intercept == 1
    assert result.fit.stop_reason == "precision_complete"
    assert [prediction.predicted_value for prediction in result.predictions] == [3, 1]
    assert result.total_loss == pytest.approx(0.0)
    assert result.exact_accuracy == pytest.approx(1.0)


def test_invalid_prime_is_rejected() -> None:
    with pytest.raises(ValueError, match="prime"):
        fit_modulo_p_consensus([(0,), (1,)], [0, 1], p=4)
