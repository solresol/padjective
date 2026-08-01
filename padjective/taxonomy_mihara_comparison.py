"""Compare a bounded Mihara-style digitwise regressor on taxonomy data.

Mihara's published method assumes samples from one hidden affine graph over
``Z_p`` with sparse, digitwise response noise.  Product-taxonomy data do not
necessarily satisfy those assumptions: tag vectors are sparse binary vectors,
and the encoded taxonomy may not be an affine function of them.  This module
therefore reports both held-out predictive performance and the diagnostic that
matters for interpreting it: whether the fitted consensus passed Mihara's
greater-than-90-percent noise-free acceptance test at each p-adic digit.

The modulo-p inner fit is a bounded RANSAC-style adaptation of Mihara's
probabilistic ``LinearRegressionModulo`` routine.  It samples full-rank affine
systems, retains the largest congruence consensus, and caps the number of
trials; the published routine instead uses intermediate affine-inclusion tests
and an unbounded restart loop.  When the acceptance test fails, this comparison
continues with the best bounded consensus so that the model can still be scored,
but it records the digit as unaccepted.  Such a score must not be reported as a
successful run of the published algorithm.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from psycopg import sql

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
    from padjective.cv import calculate_cv_folds
    from padjective.metrics import summarize_encoded_predictions
    from padjective.umllr import (
        DEFAULT_TAG_ORDER_STRATEGY,
        ProductRecord,
        _load_products,
        _p_adic_distance,
        snapshot_label,
    )
else:
    from . import db
    from .cv import calculate_cv_folds
    from .metrics import summarize_encoded_predictions
    from .umllr import (
        DEFAULT_TAG_ORDER_STRATEGY,
        ProductRecord,
        _load_products,
        _p_adic_distance,
        snapshot_label,
    )


FEATURE_SELECTION_STRATEGIES = ("taxonomy_association", "frequency")
DEFAULT_FEATURE_SELECTION = "frequency"
DEFAULT_MAX_TAGS = 32
DEFAULT_TRIALS = 96
DEFAULT_ACCEPTANCE_THRESHOLD = 0.9


@dataclass(frozen=True)
class ModuloFit:
    """Best bounded modulo-p affine consensus found in one digit step."""

    coefficients: tuple[int, ...]
    inliers: int
    total_observations: int
    validation_inlier_fraction: float
    accepted: bool
    successful_trials: int
    singular_trials: int


@dataclass(frozen=True)
class DigitDiagnostic:
    """One step of the trailing-digit recursion."""

    digit: int
    active_before: int
    inliers_after: int
    validation_inlier_fraction: float
    accepted: bool
    successful_trials: int
    singular_trials: int


@dataclass(frozen=True)
class MiharaFit:
    """Bounded digitwise affine fit modulo ``p ** precision``."""

    coefficients: tuple[int, ...]
    intercept: int
    precision_requested: int
    digits_fitted: int
    accepted_prefix_digits: int
    stop_reason: str
    diagnostics: tuple[DigitDiagnostic, ...]

    @property
    def all_digits_accepted(self) -> bool:
        return (
            self.digits_fitted == self.precision_requested
            and self.accepted_prefix_digits == self.precision_requested
        )


@dataclass(frozen=True)
class MiharaPrediction:
    product_id: int
    true_value: int
    predicted_value: int
    loss: float


@dataclass(frozen=True)
class MiharaFoldResult:
    cv_fold: int
    selected_tags: tuple[str, ...]
    fit: MiharaFit
    predictions: tuple[MiharaPrediction, ...]
    train_samples: int
    test_samples: int
    total_loss: float
    mean_loss: float
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float
    nonzero_parameters: int


@dataclass(frozen=True)
class BaselineFold:
    cv_fold: int
    total_loss: float
    mean_loss: float
    exact_accuracy: float | None
    prefix2_accuracy: float | None


def _validate_prime(p: int) -> None:
    if p < 2:
        raise ValueError("p must be prime and at least 2")
    if p == 2:
        return
    if p % 2 == 0:
        raise ValueError("p must be prime")
    for factor in range(3, math.isqrt(p) + 1, 2):
        if p % factor == 0:
            raise ValueError("p must be prime")


def _inverse_mod(value: int, p: int) -> int:
    value %= p
    if value == 0:
        raise ZeroDivisionError("zero has no inverse modulo p")
    return pow(value, -1, p)


def solve_square_system_mod_p(
    matrix: Sequence[Sequence[int]],
    targets: Sequence[int],
    p: int,
) -> tuple[int, ...] | None:
    """Solve a square system over ``F_p`` or return ``None`` if singular."""

    dimension = len(matrix)
    if dimension == 0 or len(targets) != dimension:
        raise ValueError("Expected a non-empty square system")
    if any(len(row) != dimension for row in matrix):
        raise ValueError("Expected a square matrix")

    augmented = [
        [int(value) % p for value in row] + [int(target) % p]
        for row, target in zip(matrix, targets, strict=True)
    ]
    for column in range(dimension):
        pivot = next(
            (
                row
                for row in range(column, dimension)
                if augmented[row][column] % p
            ),
            None,
        )
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        inverse = _inverse_mod(augmented[column][column], p)
        augmented[column] = [(value * inverse) % p for value in augmented[column]]

        for row in range(dimension):
            if row == column:
                continue
            factor = augmented[row][column] % p
            if factor:
                augmented[row] = [
                    (left - factor * right) % p
                    for left, right in zip(
                        augmented[row], augmented[column], strict=True
                    )
                ]

    return tuple(augmented[row][-1] % p for row in range(dimension))


def _independent_row_indices(
    matrix: Sequence[Sequence[int]],
    *,
    p: int,
    rng: random.Random,
) -> tuple[int, ...] | None:
    """Draw a random full-rank square subsystem, if the design has full rank."""

    if not matrix:
        return None
    dimension = len(matrix[0])
    if dimension == 0 or len(matrix) < dimension:
        return None
    if any(len(row) != dimension for row in matrix):
        raise ValueError("All feature rows must have the same dimension")

    indices = list(range(len(matrix)))
    rng.shuffle(indices)
    basis: list[list[int]] = []
    pivots: list[int] = []
    selected: list[int] = []

    for index in indices:
        row = [int(value) % p for value in matrix[index]]
        for pivot, basis_row in zip(pivots, basis, strict=True):
            factor = row[pivot] % p
            if factor:
                row = [
                    (left - factor * right) % p
                    for left, right in zip(row, basis_row, strict=True)
                ]
        pivot = next((column for column, value in enumerate(row) if value % p), None)
        if pivot is None:
            continue
        inverse = _inverse_mod(row[pivot], p)
        row = [(value * inverse) % p for value in row]

        for basis_index, basis_row in enumerate(basis):
            factor = basis_row[pivot] % p
            if factor:
                basis[basis_index] = [
                    (left - factor * right) % p
                    for left, right in zip(basis_row, row, strict=True)
                ]

        insertion = next(
            (position for position, old_pivot in enumerate(pivots) if pivot < old_pivot),
            len(pivots),
        )
        pivots.insert(insertion, pivot)
        basis.insert(insertion, row)
        selected.append(index)
        if len(selected) == dimension:
            return tuple(selected)

    return None


def _affine_residual(
    features: Sequence[int],
    target: int,
    coefficients: Sequence[int],
) -> int:
    return int(target) - sum(
        int(feature) * int(coefficient)
        for feature, coefficient in zip(features, coefficients[:-1], strict=True)
    ) - int(coefficients[-1])


def fit_modulo_p_consensus(
    features: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    p: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> ModuloFit | None:
    """Fit the largest sampled affine congruence consensus modulo ``p``.

    The returned ``accepted`` field applies Mihara's final full-rank
    ``NoiseFreeMatrix`` criterion.  The fitted rows are removed from the
    numerator and denominator before comparing the validation inlier fraction
    with the threshold, matching the bias correction in the paper.
    """

    _validate_prime(p)
    if not features or len(features) != len(targets):
        raise ValueError("features and targets must be non-empty and aligned")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if not 0.0 < acceptance_threshold < 1.0:
        raise ValueError("acceptance_threshold must be between 0 and 1")

    feature_dimension = len(features[0])
    if any(len(row) != feature_dimension for row in features):
        raise ValueError("All feature rows must have the same dimension")
    design = [tuple(int(value) % p for value in row) + (1,) for row in features]
    coefficient_dimension = feature_dimension + 1
    if len(design) < coefficient_dimension:
        return None

    rng = random.Random(seed)
    best_coefficients: tuple[int, ...] | None = None
    best_inliers = -1
    successful_trials = 0
    singular_trials = 0

    for _ in range(trials):
        selected = _independent_row_indices(design, p=p, rng=rng)
        if selected is None:
            singular_trials += 1
            break
        candidate = solve_square_system_mod_p(
            [design[index] for index in selected],
            [targets[index] for index in selected],
            p,
        )
        if candidate is None:  # pragma: no cover - rank builder is defensive
            singular_trials += 1
            continue
        successful_trials += 1
        inliers = sum(
            _affine_residual(row, target, candidate) % p == 0
            for row, target in zip(features, targets, strict=True)
        )
        if inliers > best_inliers or (
            inliers == best_inliers
            and (best_coefficients is None or candidate < best_coefficients)
        ):
            best_coefficients = candidate
            best_inliers = inliers

    if best_coefficients is None:
        return None

    validation_observations = len(features) - coefficient_dimension
    if validation_observations <= 0:
        validation_fraction = 0.0
        accepted = False
    else:
        validation_inliers = max(0, best_inliers - coefficient_dimension)
        validation_fraction = validation_inliers / validation_observations
        accepted = validation_fraction > acceptance_threshold

    return ModuloFit(
        coefficients=best_coefficients,
        inliers=best_inliers,
        total_observations=len(features),
        validation_inlier_fraction=validation_fraction,
        accepted=accepted,
        successful_trials=successful_trials,
        singular_trials=singular_trials,
    )


def fit_mihara_digitwise(
    features: Sequence[Sequence[int]],
    targets: Sequence[int],
    *,
    p: int,
    precision: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> MiharaFit:
    """Fit trailing p-adic coefficient digits and retain only digit inliers."""

    _validate_prime(p)
    if not features or len(features) != len(targets):
        raise ValueError("features and targets must be non-empty and aligned")
    if precision < 1:
        raise ValueError("precision must be at least 1")

    feature_dimension = len(features[0])
    if any(len(row) != feature_dimension for row in features):
        raise ValueError("All feature rows must have the same dimension")

    coefficients = [0] * (feature_dimension + 1)
    active_indices = list(range(len(features)))
    diagnostics: list[DigitDiagnostic] = []
    accepted_prefix_digits = 0
    accepted_prefix_open = True
    stop_reason = "precision_complete"

    for exponent in range(precision):
        scale = p**exponent
        active_features = [features[index] for index in active_indices]
        if len(active_features) < feature_dimension + 1:
            stop_reason = "insufficient_active_samples"
            break
        digit_targets = [
            (
                _affine_residual(features[index], targets[index], coefficients)
                // scale
            )
            % p
            for index in active_indices
        ]
        digit_fit = fit_modulo_p_consensus(
            active_features,
            digit_targets,
            p=p,
            trials=trials,
            seed=seed + exponent * 1_000_003,
            acceptance_threshold=acceptance_threshold,
        )
        if digit_fit is None:
            stop_reason = "rank_deficient_active_design"
            break

        coefficients = [
            coefficient + scale * digit
            for coefficient, digit in zip(
                coefficients, digit_fit.coefficients, strict=True
            )
        ]
        modulus = p ** (exponent + 1)
        active_indices = [
            index
            for index in active_indices
            if _affine_residual(features[index], targets[index], coefficients) % modulus
            == 0
        ]
        if accepted_prefix_open and digit_fit.accepted:
            accepted_prefix_digits += 1
        else:
            accepted_prefix_open = False
        diagnostics.append(
            DigitDiagnostic(
                digit=exponent,
                active_before=digit_fit.total_observations,
                inliers_after=len(active_indices),
                validation_inlier_fraction=digit_fit.validation_inlier_fraction,
                accepted=digit_fit.accepted,
                successful_trials=digit_fit.successful_trials,
                singular_trials=digit_fit.singular_trials,
            )
        )

    modulus = p**precision
    lifted = tuple(coefficient % modulus for coefficient in coefficients)
    return MiharaFit(
        coefficients=lifted[:-1],
        intercept=lifted[-1],
        precision_requested=precision,
        digits_fitted=len(diagnostics),
        accepted_prefix_digits=accepted_prefix_digits,
        stop_reason=stop_reason,
        diagnostics=tuple(diagnostics),
    )


def select_fold_tags(
    training: Sequence[ProductRecord],
    *,
    max_tags: int,
    strategy: str,
) -> tuple[str, ...]:
    """Select a fold-local feature subset without holdout leakage."""

    if max_tags < 0:
        raise ValueError("max_tags must be non-negative")
    if strategy not in FEATURE_SELECTION_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {', '.join(FEATURE_SELECTION_STRATEGIES)}"
        )

    counts: Counter[str] = Counter()
    taxonomy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in training:
        for tag in set(record.tags):
            counts[tag] += 1
            taxonomy_counts[tag][record.taxonomy_id] += 1

    if strategy == "frequency":
        ordered = sorted(counts, key=lambda tag: (-counts[tag], tag))
    else:
        ordered = sorted(
            counts,
            key=lambda tag: (
                -max(taxonomy_counts[tag].values(), default=0) / counts[tag],
                -counts[tag],
                tag,
            ),
        )
    return tuple(ordered[:max_tags])


def _feature_rows(
    records: Sequence[ProductRecord], selected_tags: Sequence[str]
) -> list[tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    for record in records:
        record_tags = set(record.tags)
        rows.append(tuple(1 if tag in record_tags else 0 for tag in selected_tags))
    return rows


def run_fold(
    fold: int,
    records: Sequence[ProductRecord],
    *,
    p: int,
    precision: int,
    max_tags: int = DEFAULT_MAX_TAGS,
    feature_selection: str = DEFAULT_FEATURE_SELECTION,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
) -> MiharaFoldResult:
    """Fit one training fold and score its held-out taxonomy products."""

    training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]
    if not training or not testing:
        raise ValueError(f"fold {fold} needs non-empty training and test partitions")

    selected_tags = select_fold_tags(
        training,
        max_tags=max_tags,
        strategy=feature_selection,
    )
    training_features = _feature_rows(training, selected_tags)
    fit = fit_mihara_digitwise(
        training_features,
        [record.encoded_path for record in training],
        p=p,
        precision=precision,
        trials=trials,
        seed=seed + fold * 10_000_019,
        acceptance_threshold=acceptance_threshold,
    )

    modulus = p**precision
    predictions: list[MiharaPrediction] = []
    scoring_ops: list[float] = []
    for record, feature_row in zip(
        testing, _feature_rows(testing, selected_tags), strict=True
    ):
        predicted = (
            sum(
                feature * coefficient
                for feature, coefficient in zip(
                    feature_row, fit.coefficients, strict=True
                )
            )
            + fit.intercept
        ) % modulus
        loss = _p_adic_distance(predicted, record.encoded_path, p)
        predictions.append(
            MiharaPrediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )
        active = sum(
            bool(feature and coefficient)
            for feature, coefficient in zip(
                feature_row, fit.coefficients, strict=True
            )
        )
        scoring_ops.append(float(active + bool(fit.intercept)))

    summary = summarize_encoded_predictions(
        [prediction.true_value for prediction in predictions],
        [prediction.predicted_value for prediction in predictions],
        base=p,
        true_depths=[record.taxonomy_depth for record in testing],
        scoring_ops=scoring_ops,
    )
    total_loss = sum(prediction.loss for prediction in predictions)
    nonzero_parameters = sum(bool(value) for value in fit.coefficients) + int(
        bool(fit.intercept)
    )
    return MiharaFoldResult(
        cv_fold=fold,
        selected_tags=selected_tags,
        fit=fit,
        predictions=tuple(predictions),
        train_samples=len(training),
        test_samples=len(testing),
        total_loss=total_loss,
        mean_loss=total_loss / len(testing),
        exact_accuracy=summary.exact_accuracy,
        prefix1_accuracy=summary.prefix1_accuracy,
        prefix2_accuracy=summary.prefix2_accuracy,
        mean_shared_prefix_depth=summary.mean_shared_prefix_depth,
        mean_scoring_ops=summary.mean_scoring_ops or 0.0,
        nonzero_parameters=nonzero_parameters,
    )


def _ensure_storage(conn: Any, schema: str) -> None:
    """Create append-only comparison tables in ``pg_default``."""

    db.ensure_schema(conn, schema)
    db.ensure_table(
        conn,
        schema,
        "taxonomy_mihara_runs",
        [
            "run_id UUID PRIMARY KEY",
            "created_at TIMESTAMPTZ NOT NULL",
            "snapshot_ref TEXT NOT NULL",
            "product_table TEXT NOT NULL",
            "prime_base INTEGER NOT NULL",
            "precision_requested INTEGER NOT NULL",
            "max_tags INTEGER NOT NULL",
            "feature_selection TEXT NOT NULL",
            "trials INTEGER NOT NULL",
            "seed BIGINT NOT NULL",
            "acceptance_threshold DOUBLE PRECISION NOT NULL",
            "product_count INTEGER NOT NULL",
            "taxonomy_count INTEGER NOT NULL",
            "available_tag_count INTEGER NOT NULL",
        ],
    )
    db.ensure_table(
        conn,
        schema,
        "taxonomy_mihara_fold_results",
        [
            "run_id UUID NOT NULL",
            "cv_fold INTEGER NOT NULL",
            "train_samples INTEGER NOT NULL",
            "test_samples INTEGER NOT NULL",
            "selected_tag_count INTEGER NOT NULL",
            "nonzero_parameters INTEGER NOT NULL",
            "digits_fitted INTEGER NOT NULL",
            "accepted_prefix_digits INTEGER NOT NULL",
            "all_digits_accepted BOOLEAN NOT NULL",
            "stop_reason TEXT NOT NULL",
            "total_loss DOUBLE PRECISION NOT NULL",
            "mean_loss DOUBLE PRECISION NOT NULL",
            "exact_accuracy DOUBLE PRECISION NOT NULL",
            "prefix1_accuracy DOUBLE PRECISION NOT NULL",
            "prefix2_accuracy DOUBLE PRECISION NOT NULL",
            "mean_shared_prefix_depth DOUBLE PRECISION NOT NULL",
            "mean_scoring_ops DOUBLE PRECISION NOT NULL",
            "baseline_mean_loss DOUBLE PRECISION",
            "loss_delta_vs_umllr DOUBLE PRECISION",
            "digit_diagnostics JSONB NOT NULL",
            "PRIMARY KEY (run_id, cv_fold)",
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {schema}.{table} (cv_fold)"
            ).format(
                index=sql.Identifier(f"{schema}_taxonomy_mihara_fold_idx"),
                schema=sql.Identifier(schema),
                table=sql.Identifier("taxonomy_mihara_fold_results"),
            )
        ],
    )
    db.ensure_table(
        conn,
        schema,
        "taxonomy_mihara_coefficients",
        [
            "run_id UUID NOT NULL",
            "cv_fold INTEGER NOT NULL",
            "tag TEXT NOT NULL",
            "coefficient NUMERIC NOT NULL",
            "sequence INTEGER NOT NULL",
            "is_intercept BOOLEAN NOT NULL",
            "PRIMARY KEY (run_id, cv_fold, tag)",
        ],
    )
    db.ensure_table(
        conn,
        schema,
        "taxonomy_mihara_predictions",
        [
            "run_id UUID NOT NULL",
            "cv_fold INTEGER NOT NULL",
            "product_id BIGINT NOT NULL",
            "true_value NUMERIC NOT NULL",
            "predicted_value NUMERIC NOT NULL",
            "loss DOUBLE PRECISION NOT NULL",
            "PRIMARY KEY (run_id, cv_fold, product_id)",
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {schema}.{table} (product_id)"
            ).format(
                index=sql.Identifier(f"{schema}_taxonomy_mihara_prediction_product_idx"),
                schema=sql.Identifier(schema),
                table=sql.Identifier("taxonomy_mihara_predictions"),
            )
        ],
    )


def load_umllr_baseline(
    conn: Any,
    *,
    schema: str,
    snapshot_ref: str | None,
    fold_sizes: Mapping[int, int],
) -> dict[int, BaselineFold]:
    """Load the matching default UMLLR fold metrics when they are available."""

    with conn.cursor() as cur:
        if snapshot_ref is None:
            cur.execute(
                sql.SQL(
                    """
                    SELECT cv_fold, loss, exact_accuracy, prefix2_accuracy
                    FROM {schema}.umllr_fold_metrics
                    ORDER BY cv_fold
                    """
                ).format(schema=sql.Identifier(schema))
            )
        else:
            cur.execute(
                sql.SQL(
                    """
                    SELECT cv_fold, loss, exact_accuracy, prefix2_accuracy
                    FROM {schema}.umllr_order_ablation_fold_metrics
                    WHERE snapshot_ref = %s
                      AND tag_order_strategy = %s
                      AND tag_order_seed IS NULL
                    ORDER BY cv_fold
                    """
                ).format(schema=sql.Identifier(schema)),
                (snapshot_label(snapshot_ref), DEFAULT_TAG_ORDER_STRATEGY),
            )
        rows = cur.fetchall()

    baseline: dict[int, BaselineFold] = {}
    for row in rows:
        fold = int(row[0])
        test_samples = fold_sizes.get(fold, 0)
        if not test_samples:
            continue
        total_loss = float(row[1])
        baseline[fold] = BaselineFold(
            cv_fold=fold,
            total_loss=total_loss,
            mean_loss=total_loss / test_samples,
            exact_accuracy=float(row[2]) if row[2] is not None else None,
            prefix2_accuracy=float(row[3]) if row[3] is not None else None,
        )
    return baseline


def save_results(
    conn: Any,
    *,
    schema: str,
    run_id: uuid.UUID,
    created_at: datetime,
    snapshot_ref: str,
    product_table: str,
    prime_base: int,
    precision: int,
    max_tags: int,
    feature_selection: str,
    trials: int,
    seed: int,
    acceptance_threshold: float,
    product_count: int,
    taxonomy_count: int,
    available_tag_count: int,
    results: Sequence[MiharaFoldResult],
    baseline: Mapping[int, BaselineFold],
) -> None:
    """Persist one comparison run and its fold-level evidence."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_mihara_runs (
                    run_id, created_at, snapshot_ref, product_table, prime_base,
                    precision_requested, max_tags, feature_selection, trials,
                    seed, acceptance_threshold, product_count, taxonomy_count,
                    available_tag_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            ).format(schema=sql.Identifier(schema)),
            (
                run_id,
                created_at,
                snapshot_ref,
                product_table,
                prime_base,
                precision,
                max_tags,
                feature_selection,
                trials,
                seed,
                acceptance_threshold,
                product_count,
                taxonomy_count,
                available_tag_count,
            ),
        )

        fold_rows = []
        coefficient_rows = []
        prediction_rows = []
        for result in results:
            baseline_fold = baseline.get(result.cv_fold)
            baseline_mean = baseline_fold.mean_loss if baseline_fold else None
            fold_rows.append(
                (
                    run_id,
                    result.cv_fold,
                    result.train_samples,
                    result.test_samples,
                    len(result.selected_tags),
                    result.nonzero_parameters,
                    result.fit.digits_fitted,
                    result.fit.accepted_prefix_digits,
                    result.fit.all_digits_accepted,
                    result.fit.stop_reason,
                    result.total_loss,
                    result.mean_loss,
                    result.exact_accuracy,
                    result.prefix1_accuracy,
                    result.prefix2_accuracy,
                    result.mean_shared_prefix_depth,
                    result.mean_scoring_ops,
                    baseline_mean,
                    result.mean_loss - baseline_mean
                    if baseline_mean is not None
                    else None,
                    json.dumps([asdict(item) for item in result.fit.diagnostics]),
                )
            )
            coefficient_rows.extend(
                (
                    run_id,
                    result.cv_fold,
                    tag,
                    coefficient,
                    sequence,
                    False,
                )
                for sequence, (tag, coefficient) in enumerate(
                    zip(
                        result.selected_tags,
                        result.fit.coefficients,
                        strict=True,
                    )
                )
            )
            coefficient_rows.append(
                (
                    run_id,
                    result.cv_fold,
                    "__INTERCEPT__",
                    result.fit.intercept,
                    len(result.selected_tags),
                    True,
                )
            )
            prediction_rows.extend(
                (
                    run_id,
                    result.cv_fold,
                    prediction.product_id,
                    prediction.true_value,
                    prediction.predicted_value,
                    prediction.loss,
                )
                for prediction in result.predictions
            )

        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_mihara_fold_results (
                    run_id, cv_fold, train_samples, test_samples,
                    selected_tag_count, nonzero_parameters, digits_fitted,
                    accepted_prefix_digits, all_digits_accepted, stop_reason, total_loss,
                    mean_loss, exact_accuracy, prefix1_accuracy, prefix2_accuracy,
                    mean_shared_prefix_depth, mean_scoring_ops, baseline_mean_loss,
                    loss_delta_vs_umllr, digit_diagnostics
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                )
                """
            ).format(schema=sql.Identifier(schema)),
            fold_rows,
        )
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_mihara_coefficients (
                    run_id, cv_fold, tag, coefficient, sequence, is_intercept
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
            ).format(schema=sql.Identifier(schema)),
            coefficient_rows,
        )
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_mihara_predictions (
                    run_id, cv_fold, product_id, true_value, predicted_value, loss
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
            ).format(schema=sql.Identifier(schema)),
            prediction_rows,
        )
    conn.commit()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def render_summary(
    results: Sequence[MiharaFoldResult],
    baseline: Mapping[int, BaselineFold],
) -> str:
    """Render a compact terminal comparison table."""

    lines = [
        "fold  digits  accepted  mean_loss  exact_acc  prefix2_acc  "
        "UMLLR_mean  delta  status",
    ]
    for result in results:
        baseline_fold = baseline.get(result.cv_fold)
        baseline_mean = baseline_fold.mean_loss if baseline_fold else None
        lines.append(
            f"{result.cv_fold:>4}  "
            f"{result.fit.digits_fitted:>2}/{result.fit.precision_requested:<2}  "
            f"{result.fit.accepted_prefix_digits:>2}/{result.fit.precision_requested:<2}  "
            f"{result.mean_loss:>9.6f}  "
            f"{result.exact_accuracy:>9.4f}  "
            f"{result.prefix2_accuracy:>11.4f}  "
            + (
                f"{baseline_mean:>10.6f}  {result.mean_loss - baseline_mean:>+9.6f}"
                if baseline_mean is not None
                else "         —          —"
            )
            + f"  {result.fit.stop_reason}"
        )
    lines.extend(
        [
            "",
            f"Mihara bounded mean loss: {_mean([item.mean_loss for item in results]):.6f}",
            "Mihara bounded exact accuracy: "
            f"{_mean([item.exact_accuracy for item in results]):.4f}",
            "Mihara bounded prefix-2 accuracy: "
            f"{_mean([item.prefix2_accuracy for item in results]):.4f}",
            "Published acceptance completed in "
            f"{sum(item.fit.all_digits_accepted for item in results)}/{len(results)} folds.",
        ]
    )
    if baseline:
        lines.append(
            f"Matching UMLLR mean loss: {_mean([item.mean_loss for item in baseline.values()]):.6f}"
        )
    return "\n".join(lines)


def process_database(
    dsn: str | None,
    *,
    schema: str = "padjective",
    product_table: str = "cantbuymelove.product",
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
    cv_splits: int = 5,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    max_tags: int = DEFAULT_MAX_TAGS,
    feature_selection: str = DEFAULT_FEATURE_SELECTION,
    precision: int | None = None,
    trials: int = DEFAULT_TRIALS,
    seed: int = 0,
    acceptance_threshold: float = DEFAULT_ACCEPTANCE_THRESHOLD,
    folds: Sequence[int] | None = None,
    persist: bool = True,
) -> tuple[uuid.UUID | None, list[MiharaFoldResult], dict[int, BaselineFold]]:
    """Load Postgres data, run the comparison, and persist an auditable run."""

    conn = db.get_connection(dsn)
    try:
        fold_assignments = None
        if snapshot_ref is None:
            fold_assignments = calculate_cv_folds(
                conn, product_table, n_splits=cv_splits
            )
            if not fold_assignments:
                raise ValueError("No cross-validation fold assignments were produced")

        records, prime_base, _, taxonomy_encodings, dataset = _load_products(
            conn,
            product_table,
            fold_assignments,
            min_tag_count=min_tag_count,
            min_samples_per_taxonomy=min_samples_per_taxonomy,
            snapshot_ref=snapshot_ref,
            snapshot_schema=snapshot_schema,
        )
        if not records:
            raise ValueError("No taxonomy products were loaded")

        effective_precision = (
            precision
            if precision is not None
            else max(record.taxonomy_depth for record in records)
        )
        available_folds = sorted({record.cv_fold for record in records})
        requested_folds = (
            list(dict.fromkeys(folds)) if folds is not None else available_folds
        )
        unknown_folds = sorted(set(requested_folds) - set(available_folds))
        if unknown_folds:
            raise ValueError(f"Unknown folds requested: {unknown_folds}")

        results = [
            run_fold(
                fold,
                records,
                p=prime_base,
                precision=effective_precision,
                max_tags=max_tags,
                feature_selection=feature_selection,
                trials=trials,
                seed=seed,
                acceptance_threshold=acceptance_threshold,
            )
            for fold in requested_folds
        ]
        fold_sizes = {result.cv_fold: result.test_samples for result in results}
        try:
            baseline = load_umllr_baseline(
                conn,
                schema=schema,
                snapshot_ref=snapshot_ref,
                fold_sizes=fold_sizes,
            )
        except Exception as exc:
            # The benchmark is still useful on a newly provisioned database
            # before the UMLLR tables have been populated.
            print(f"UMLLR baseline unavailable: {exc}", file=sys.stderr)
            conn.rollback()
            baseline = {}

        run_id: uuid.UUID | None = None
        if persist:
            _ensure_storage(conn, schema)
            run_id = uuid.uuid4()
            save_results(
                conn,
                schema=schema,
                run_id=run_id,
                created_at=datetime.now(timezone.utc),
                snapshot_ref=snapshot_label(snapshot_ref),
                product_table=product_table,
                prime_base=prime_base,
                precision=effective_precision,
                max_tags=max_tags,
                feature_selection=feature_selection,
                trials=trials,
                seed=seed,
                acceptance_threshold=acceptance_threshold,
                product_count=len(records),
                taxonomy_count=len(taxonomy_encodings),
                available_tag_count=len(dataset.feature_names),
                results=results,
                baseline=baseline,
            )
        return run_id, results, baseline
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a bounded Mihara-style digitwise p-adic regressor with "
            "UMLLR on Postgres taxonomy data."
        )
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument("--schema", default="padjective")
    parser.add_argument("--product-table", default="cantbuymelove.product")
    parser.add_argument(
        "--snapshot-ref",
        help="Benchmark snapshot alias/name/UUID; omit for the live catalog",
    )
    parser.add_argument("--snapshot-schema", default="padjective")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--min-tag-count", type=int, default=5)
    parser.add_argument("--min-samples-per-taxonomy", type=int, default=5)
    parser.add_argument(
        "--max-tags",
        type=int,
        default=DEFAULT_MAX_TAGS,
        help=(
            "Fold-local tag dimensions to fit (default: 32, matching the "
            "parameter-constrained baselines)"
        ),
    )
    parser.add_argument(
        "--feature-selection",
        choices=FEATURE_SELECTION_STRATEGIES,
        default=DEFAULT_FEATURE_SELECTION,
    )
    parser.add_argument(
        "--precision",
        type=int,
        help="Number of p-adic digits; defaults to the maximum taxonomy depth",
    )
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--acceptance-threshold",
        type=float,
        default=DEFAULT_ACCEPTANCE_THRESHOLD,
        help="Noise-free validation fraction required at each digit",
    )
    parser.add_argument(
        "--fold",
        type=int,
        action="append",
        dest="folds",
        help="Run only this fold; repeat for multiple folds",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Run and print the comparison without writing result tables",
    )
    args = parser.parse_args()

    run_id, results, baseline = process_database(
        args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        snapshot_ref=args.snapshot_ref,
        snapshot_schema=args.snapshot_schema,
        cv_splits=args.cv_splits,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        max_tags=args.max_tags,
        feature_selection=args.feature_selection,
        precision=args.precision,
        trials=args.trials,
        seed=args.seed,
        acceptance_threshold=args.acceptance_threshold,
        folds=args.folds,
        persist=not args.no_persist,
    )
    print(render_summary(results, baseline))
    if run_id is not None:
        print(f"Persisted run_id: {run_id}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
