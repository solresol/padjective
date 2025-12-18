"""Zubarev's p-adic polynomial regression for taxonomy prediction.

This implements Zubarev's method from "p-Adic Polynomial Regression as Alternative
to Neural Network for Approximating p-Adic Functions of Many Variables"
(arXiv:2503.23488), adapted for sparse one-hot tag data.

Key differences from UMLLR:
- Joint stochastic optimization (vs. greedy sequential)
- Optional Mahler polynomial basis (vs. purely linear)
- Temperature-controlled exploration (vs. deterministic)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db
    from padjective.cv import calculate_cv_folds
    from padjective.metrics import parse_taxonomy_path
    from padjective.tagbattle import filter_nested_tags
else:  # pragma: no cover - imported as a package
    from . import data_access, db
    from .cv import calculate_cv_folds
    from .metrics import parse_taxonomy_path
    from .tagbattle import filter_nested_tags


@dataclass(frozen=True)
class ProductRecord:
    product_id: int
    tags: List[str]
    encoded_path: int
    cv_fold: int


@dataclass(frozen=True)
class BattleRecord:
    winner_tag: str
    loser_tag: str
    cv_fold: int | None


@dataclass(frozen=True)
class TagCoefficient:
    tag: str
    coefficient: int
    sequence: int


@dataclass(frozen=True)
class Prediction:
    product_id: int
    true_value: int
    predicted_value: int
    loss: float


@dataclass(frozen=True)
class IterationRecord:
    """Record of optimization state at a specific iteration."""
    iteration: int
    train_loss: float
    validation_loss: float
    best_loss: float
    temperature: float
    acceptance_rate: float  # Recent acceptance rate


@dataclass(frozen=True)
class FoldResult:
    cv_fold: int
    coefficients: List[TagCoefficient]
    mahler_weights: List[int]  # Weights for Mahler polynomial terms
    predictions: List[Prediction]
    loss: float
    default_prediction: int
    iterations_used: int
    final_temperature: float
    iteration_history: List[IterationRecord]  # Loss tracking over iterations


def _next_prime(min_value: int) -> int:
    candidate = max(2, min_value + 1)
    while True:
        if _is_prime(candidate):
            return candidate
        candidate += 1


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in {2, 3}:
        return True
    if value % 2 == 0:
        return False
    limit = int(math.isqrt(value)) + 1
    for factor in range(3, limit, 2):
        if value % factor == 0:
            return False
    return True


def _p_adic_distance(a: int, b: int, base: int) -> float:
    """Compute p-adic distance between two integers."""
    if a == b:
        return 0.0

    diff = abs(a - b)
    valuation = 0
    while diff % base == 0:
        diff //= base
        valuation += 1
    return base ** (-valuation)


def _p_adic_valuation(x: int, base: int) -> int:
    """Compute p-adic valuation v_p(x) = max k such that p^k divides x."""
    if x == 0:
        return float('inf')  # Convention: v_p(0) = infinity
    x = abs(x)
    v = 0
    while x % base == 0:
        x //= base
        v += 1
    return v


def _binomial(n: int, k: int) -> int:
    """Compute binomial coefficient (n choose k).

    This is the Mahler basis function omega_k(n) = C(n, k).
    Works for any integer n (including negative).
    """
    if k < 0:
        return 0
    if k == 0:
        return 1
    if k > abs(n) and n >= 0:
        return 0

    # General formula for any integer n:
    # C(n, k) = n * (n-1) * ... * (n-k+1) / k!
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _mahler_predict(s: int, weights: Sequence[int]) -> int:
    """Apply Mahler polynomial: y = sum_k w_k * C(s, k)."""
    result = 0
    for k, w in enumerate(weights):
        result += w * _binomial(s, k)
    return result


def _parse_taxonomy_digits(path_value: str | None) -> Tuple[int, ...]:
    """Parse taxonomy path into sequence of digits."""
    import re
    _SEGMENT_NUMBER_RE = re.compile(r"-?\d+")

    if not path_value:
        return ()

    digits: List[int] = []
    for segment in parse_taxonomy_path(path_value):
        segment = segment.strip()
        if not segment:
            continue
        try:
            digits.append(int(segment))
            continue
        except ValueError:
            matches = _SEGMENT_NUMBER_RE.findall(segment)
            if matches:
                digits.extend(int(match) for match in matches)
                continue
    return tuple(digits)


def _encode_path(digits: Sequence[int], base: int) -> int:
    """Encode taxonomy path as p-adic integer."""
    value = 0
    for power, digit in enumerate(digits):
        value += digit * (base ** power)
    return value


def _load_battles(conn, schema: str) -> List[BattleRecord]:
    query = sql.SQL(
        "SELECT winner_tag, loser_tag, cv_fold FROM {schema}.battles"
    ).format(schema=sql.Identifier(schema))

    records: List[BattleRecord] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            records.append(
                BattleRecord(
                    winner_tag=row.get("winner_tag"),
                    loser_tag=row.get("loser_tag"),
                    cv_fold=row.get("cv_fold"),
                )
            )
    return records


def _load_products(
    conn,
    product_table: str,
    fold_assignments: Dict[int, int],
    *,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
) -> tuple[List[ProductRecord], int, int, Dict[str, Tuple[str, int]], data_access.ProductDataset]:
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )

    records: List[ProductRecord] = []
    max_digit = 0
    raw_entries: List[tuple[int, List[str], Tuple[int, ...], int, str, str]] = []

    valid_tags = set(dataset.feature_names)

    for record in dataset.records:
        cv_fold = fold_assignments.get(record.product_id)
        if cv_fold is None:
            continue
        nested_filtered = filter_nested_tags(record.tags)
        filtered_tags = [tag.upper() for tag in nested_filtered if tag in valid_tags]
        taxonomy_id = record.taxonomy_id or ""
        taxonomy_path = record.taxonomy_path or ""
        digits = _parse_taxonomy_digits(taxonomy_path)
        if digits:
            max_digit = max(max_digit, max(digits))
        raw_entries.append((record.product_id, filtered_tags, digits, cv_fold, taxonomy_id, taxonomy_path))

    prime_base = _next_prime(max_digit)
    taxonomy_encodings: Dict[str, Tuple[str, int]] = {}

    for product_id, tags, digits, cv_fold, taxonomy_id, taxonomy_path in raw_entries:
        encoded = _encode_path(digits, prime_base)
        records.append(ProductRecord(product_id, tags, encoded, cv_fold))
        if taxonomy_id and taxonomy_id not in taxonomy_encodings:
            taxonomy_encodings[taxonomy_id] = (taxonomy_path, encoded)

    return records, prime_base, max_digit, taxonomy_encodings, dataset


def _tag_order(
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    training_tags: set[str],
) -> List[str]:
    """Order tags by battle win/loss ratio (same as UMLLR for initialization)."""
    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}

    for battle in battles:
        if battle.cv_fold == holdout_fold:
            continue
        wins[battle.winner_tag] = wins.get(battle.winner_tag, 0) + 1
        losses[battle.loser_tag] = losses.get(battle.loser_tag, 0) + 1

    ordered_tags = list(training_tags)
    for tag in ordered_tags:
        wins.setdefault(tag, 0)
        losses.setdefault(tag, 0)

    ordered_tags.sort(key=lambda tag: (-wins[tag], losses[tag], tag))
    return ordered_tags


def _compute_loss(
    records: Sequence[ProductRecord],
    coefficients: Dict[str, int],
    mahler_weights: Sequence[int],
    default_prediction: int,
    base: int,
) -> float:
    """Compute total p-adic loss for all records."""
    total_loss = 0.0
    for record in records:
        # Linear aggregation
        s = sum(coefficients.get(tag, 0) for tag in record.tags)
        # Mahler polynomial transform
        if mahler_weights:
            predicted = _mahler_predict(s, mahler_weights)
        else:
            predicted = s
        # Use default for products with no contribution
        if predicted == 0 and not any(coefficients.get(tag, 0) != 0 for tag in record.tags):
            predicted = default_prediction
        total_loss += _p_adic_distance(predicted, record.encoded_path, base)
    return total_loss


def _initialize_coefficients_umllr_style(
    training: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    base: int,
) -> Dict[str, int]:
    """Initialize coefficients using UMLLR's greedy method.

    This provides a good starting point for stochastic optimization.
    """
    tag_to_products: Dict[str, List[int]] = {}
    for record in training:
        for tag in record.tags:
            tag_to_products.setdefault(tag, []).append(record.product_id)

    tag_order = _tag_order(battles, holdout_fold, set(tag_to_products.keys()))

    product_residuals: Dict[int, int] = {record.product_id: record.encoded_path for record in training}
    coefficients: Dict[str, int] = {}

    for tag in tag_order:
        product_ids = tag_to_products.get(tag, [])
        values = [product_residuals[pid] for pid in product_ids]

        if values:
            # Select coefficient minimizing p-adic loss on residuals
            unique_values = sorted(set(values))
            best_value = unique_values[0]
            best_loss = math.inf

            for candidate in unique_values:
                total_distance = sum(_p_adic_distance(candidate, v, base) for v in values)
                if total_distance < best_loss or (
                    math.isclose(total_distance, best_loss) and candidate < best_value
                ):
                    best_loss = total_distance
                    best_value = candidate

            coefficients[tag] = best_value
            # Update residuals
            for pid in product_ids:
                product_residuals[pid] -= best_value
        else:
            coefficients[tag] = 0

    return coefficients


def _stochastic_optimize(
    training: Sequence[ProductRecord],
    validation: Sequence[ProductRecord],
    initial_coefficients: Dict[str, int],
    base: int,
    *,
    mahler_degree: int = 0,
    max_iterations: int = 10000,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.9995,
    min_temperature: float = 0.001,
    perturbation_scale: int = 1000,
    seed: int | None = None,
    log_interval: int = 100,
) -> Tuple[Dict[str, int], List[int], float, int, List[IterationRecord]]:
    """Zubarev-style stochastic optimization.

    Uses p-adic random walk with simulated annealing.

    Args:
        training: Training records (for optimization)
        validation: Validation records (for tracking generalization)
        initial_coefficients: Starting point (e.g., from UMLLR initialization)
        base: Prime base for p-adic arithmetic
        mahler_degree: Degree of Mahler polynomial (0 = linear only)
        max_iterations: Maximum optimization iterations
        initial_temperature: Starting temperature for annealing
        cooling_rate: Temperature decay per iteration
        min_temperature: Stop when temperature falls below this
        perturbation_scale: Scale of random perturbations
        seed: Random seed for reproducibility
        log_interval: How often to record iteration history

    Returns:
        (optimized_coefficients, mahler_weights, final_loss, iterations_used, iteration_history)
    """
    if seed is not None:
        random.seed(seed)

    # Initialize
    coefficients = dict(initial_coefficients)
    tags = list(coefficients.keys())

    # Initialize Mahler weights (w_0 = 0, w_1 = 1 for identity transform)
    if mahler_degree > 0:
        mahler_weights = [0] + [1] + [0] * (mahler_degree - 1)
    else:
        mahler_weights = []

    # Compute unique taxonomy values for default prediction
    all_values = sorted(set(r.encoded_path for r in training))
    default_prediction = all_values[0] if all_values else 0

    # Current best
    current_loss = _compute_loss(training, coefficients, mahler_weights, default_prediction, base)
    best_coefficients = dict(coefficients)
    best_mahler = list(mahler_weights)
    best_loss = current_loss

    # Iteration history tracking
    iteration_history: List[IterationRecord] = []
    recent_accepts = 0
    recent_proposals = 0

    temperature = initial_temperature
    iteration = 0

    # Record initial state
    val_loss = _compute_loss(validation, coefficients, mahler_weights, default_prediction, base) if validation else 0.0
    iteration_history.append(IterationRecord(
        iteration=0,
        train_loss=current_loss / len(training) if training else 0.0,
        validation_loss=val_loss / len(validation) if validation else 0.0,
        best_loss=best_loss / len(training) if training else 0.0,
        temperature=temperature,
        acceptance_rate=1.0,
    ))

    while iteration < max_iterations and temperature > min_temperature:
        # Choose what to perturb: coefficient or Mahler weight
        if mahler_weights and random.random() < 0.2:
            # Perturb a Mahler weight
            k = random.randint(0, len(mahler_weights) - 1)
            old_weight = mahler_weights[k]

            # P-adic-style perturbation: prefer powers of p
            if random.random() < 0.5:
                # Perturb by a power of p
                power = random.randint(0, 5)
                sign = random.choice([-1, 1])
                delta = sign * (base ** power)
            else:
                # Random perturbation
                delta = random.randint(-perturbation_scale, perturbation_scale)

            mahler_weights[k] = old_weight + delta
            new_loss = _compute_loss(training, coefficients, mahler_weights, default_prediction, base)

            # Metropolis acceptance criterion
            recent_proposals += 1
            accepted = False
            if new_loss < current_loss:
                current_loss = new_loss
                accepted = True
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_coefficients = dict(coefficients)
                    best_mahler = list(mahler_weights)
            elif temperature > 0:
                delta_loss = new_loss - current_loss
                # Use p-adic-scaled acceptance probability
                acceptance_prob = math.exp(-delta_loss / temperature)
                if random.random() < acceptance_prob:
                    current_loss = new_loss
                    accepted = True
                else:
                    mahler_weights[k] = old_weight
            else:
                mahler_weights[k] = old_weight
            if accepted:
                recent_accepts += 1
        else:
            # Perturb a coefficient
            if not tags:
                iteration += 1
                temperature *= cooling_rate
                continue

            tag = random.choice(tags)
            old_coeff = coefficients[tag]

            # P-adic-style perturbation: prefer changes by powers of p
            if random.random() < 0.3:
                # Set to a power of p
                power = random.randint(0, 8)
                sign = random.choice([-1, 1])
                coefficients[tag] = sign * (base ** power)
            elif random.random() < 0.5:
                # Perturb by a power of p
                power = random.randint(0, 5)
                sign = random.choice([-1, 1])
                delta = sign * (base ** power)
                coefficients[tag] = old_coeff + delta
            else:
                # Try a value from training data (exploitation)
                relevant_values = [r.encoded_path for r in training if tag in r.tags]
                if relevant_values:
                    coefficients[tag] = random.choice(relevant_values)
                else:
                    coefficients[tag] = old_coeff + random.randint(-perturbation_scale, perturbation_scale)

            new_loss = _compute_loss(training, coefficients, mahler_weights, default_prediction, base)

            # Metropolis acceptance criterion
            recent_proposals += 1
            accepted = False
            if new_loss < current_loss:
                current_loss = new_loss
                accepted = True
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_coefficients = dict(coefficients)
                    best_mahler = list(mahler_weights)
            elif temperature > 0:
                delta_loss = new_loss - current_loss
                acceptance_prob = math.exp(-delta_loss / temperature)
                if random.random() < acceptance_prob:
                    current_loss = new_loss
                    accepted = True
                else:
                    coefficients[tag] = old_coeff
            else:
                coefficients[tag] = old_coeff
            if accepted:
                recent_accepts += 1

        iteration += 1
        temperature *= cooling_rate

        # Record history at intervals
        if iteration % log_interval == 0:
            val_loss = _compute_loss(validation, coefficients, mahler_weights, default_prediction, base) if validation else 0.0
            accept_rate = recent_accepts / recent_proposals if recent_proposals > 0 else 0.0
            iteration_history.append(IterationRecord(
                iteration=iteration,
                train_loss=current_loss / len(training) if training else 0.0,
                validation_loss=val_loss / len(validation) if validation else 0.0,
                best_loss=best_loss / len(training) if training else 0.0,
                temperature=temperature,
                acceptance_rate=accept_rate,
            ))
            # Reset acceptance tracking for next interval
            recent_accepts = 0
            recent_proposals = 0

    # Record final state (only if not already recorded at a log interval)
    if iteration % log_interval != 0:
        val_loss = _compute_loss(validation, coefficients, mahler_weights, default_prediction, base) if validation else 0.0
        accept_rate = recent_accepts / recent_proposals if recent_proposals > 0 else 0.0
        iteration_history.append(IterationRecord(
            iteration=iteration,
            train_loss=current_loss / len(training) if training else 0.0,
            validation_loss=val_loss / len(validation) if validation else 0.0,
            best_loss=best_loss / len(training) if training else 0.0,
            temperature=temperature,
            acceptance_rate=accept_rate,
        ))

    return best_coefficients, best_mahler, best_loss, iteration, iteration_history


def _select_default_prediction(
    no_tag_values: Sequence[int],
    candidate_values: Sequence[int],
    base: int,
) -> int:
    """Select optimal default prediction for products with no contributing tags."""
    if not no_tag_values:
        from collections import Counter
        if candidate_values:
            return Counter(candidate_values).most_common(1)[0][0]
        return 0

    unique_candidates = sorted(set(candidate_values)) if candidate_values else [0]
    best_value = unique_candidates[0]
    best_loss = float("inf")

    for candidate in unique_candidates:
        total_loss = sum(_p_adic_distance(candidate, value, base) for value in no_tag_values)
        if total_loss < best_loss or (total_loss == best_loss and candidate < best_value):
            best_loss = total_loss
            best_value = candidate

    return best_value


def _run_fold(
    fold: int,
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
    *,
    mahler_degree: int = 0,
    max_iterations: int = 10000,
    seed: int | None = None,
    validation_fraction: float = 0.2,
    initialization_method: str = 'umllr',
) -> FoldResult:
    """Run Zubarev regression for a single CV fold."""
    all_training = [r for r in records if r.cv_fold != fold]
    testing = [r for r in records if r.cv_fold == fold]

    # Split training into train/validation for monitoring
    fold_seed = seed + fold if seed is not None else None
    if fold_seed is not None:
        random.seed(fold_seed)

    shuffled = list(all_training)
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * validation_fraction)
    validation = shuffled[:val_size]
    training = shuffled[val_size:]

    # Initialize coefficients based on method
    if initialization_method == 'umllr':
        # Initialize using UMLLR's greedy method (on full training data for better init)
        initial_coefficients = _initialize_coefficients_umllr_style(
            all_training, battles, fold, base
        )
    elif initialization_method == 'zeros':
        # Initialize all coefficients to zero
        all_tags = set()
        for record in all_training:
            all_tags.update(record.tags)
        initial_coefficients = {tag: 0 for tag in all_tags}
    else:
        raise ValueError(f"Unknown initialization_method: {initialization_method}")

    # Stochastic optimization with validation tracking
    optimized_coefficients, mahler_weights, train_loss, iterations_used, iteration_history = _stochastic_optimize(
        training,
        validation,
        initial_coefficients,
        base,
        mahler_degree=mahler_degree,
        max_iterations=max_iterations,
        seed=fold_seed,
    )

    # Compute default prediction
    all_training_values = [r.encoded_path for r in training]
    no_tag_training_values = [
        r.encoded_path
        for r in training
        if sum(optimized_coefficients.get(tag, 0) for tag in r.tags) == 0
    ]
    default_prediction = _select_default_prediction(
        no_tag_training_values, all_training_values, base
    )

    # Make predictions on test set
    predictions: List[Prediction] = []
    total_loss = 0.0

    for record in testing:
        s = sum(optimized_coefficients.get(tag, 0) for tag in record.tags)
        if mahler_weights:
            predicted = _mahler_predict(s, mahler_weights)
        else:
            predicted = s
        if predicted == 0 and not any(optimized_coefficients.get(tag, 0) != 0 for tag in record.tags):
            predicted = default_prediction

        loss = _p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss
        predictions.append(Prediction(
            product_id=record.product_id,
            true_value=record.encoded_path,
            predicted_value=predicted,
            loss=loss,
        ))

    # Build coefficient list with sequence numbers
    tag_order = _tag_order(battles, fold, set(optimized_coefficients.keys()))
    coefficients = [
        TagCoefficient(tag=tag, coefficient=optimized_coefficients.get(tag, 0), sequence=i)
        for i, tag in enumerate(tag_order)
    ]

    final_temp = 1.0 * (0.9995 ** iterations_used)

    return FoldResult(
        cv_fold=fold,
        coefficients=coefficients,
        mahler_weights=mahler_weights,
        predictions=predictions,
        loss=total_loss,
        default_prediction=default_prediction,
        iterations_used=iterations_used,
        final_temperature=final_temp,
        iteration_history=iteration_history,
    )


def _ensure_storage(conn, schema: str) -> None:
    """Verify Zubarev tables exist."""
    required_tables = [
        "zubarev_tag_coefficients",
        "zubarev_fold_metrics",
        "zubarev_predictions",
        "zubarev_mahler_weights",
        "zubarev_iteration_history",
    ]

    with conn.cursor() as cur:
        for table in required_tables:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            if not cur.fetchone():
                raise RuntimeError(
                    f"Table {schema}.{table} does not exist. "
                    f"Please run create_zubarev_tables.sql first."
                )


def _save_results(
    conn,
    schema: str,
    results: Sequence[FoldResult],
    prime_base: int,
    max_digit: int,
    cv_splits: int,
    initialization_method: str = 'umllr',
    mahler_degree: int = 0,
) -> None:
    """Save Zubarev results to database."""
    coeff_rows: List[Tuple[int, str, int, int, str, int]] = []
    prediction_rows: List[Tuple[int, int, int, int, float, str, int]] = []
    metrics_rows: List[Tuple[int, float, int, int, int, int, float, str, int]] = []
    mahler_rows: List[Tuple[int, int, int, str, int]] = []
    history_rows: List[Tuple[int, int, float, float, float, float, float, str, int]] = []

    for result in results:
        metrics_rows.append((
            result.cv_fold,
            result.loss,
            prime_base,
            max_digit,
            result.default_prediction,
            result.iterations_used,
            result.final_temperature,
            initialization_method,
            mahler_degree,
        ))
        for entry in result.coefficients:
            coeff_rows.append((result.cv_fold, entry.tag, entry.coefficient, entry.sequence, initialization_method, mahler_degree))
        for prediction in result.predictions:
            prediction_rows.append((
                result.cv_fold,
                prediction.product_id,
                prediction.true_value,
                prediction.predicted_value,
                prediction.loss,
                initialization_method,
                mahler_degree,
            ))
        for k, weight in enumerate(result.mahler_weights):
            mahler_rows.append((result.cv_fold, k, weight, initialization_method, mahler_degree))
        for record in result.iteration_history:
            history_rows.append((
                result.cv_fold,
                record.iteration,
                record.train_loss,
                record.validation_loss,
                record.best_loss,
                record.temperature,
                record.acceptance_rate,
                initialization_method,
                mahler_degree,
            ))

    with conn.cursor() as cur:
        fold_list = list(range(cv_splits))

        if fold_list:
            cur.execute(
                sql.SQL("DELETE FROM {schema}.zubarev_tag_coefficients WHERE cv_fold = ANY(%s) AND initialization_method = %s AND mahler_degree = %s").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list, initialization_method, mahler_degree)
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.zubarev_fold_metrics WHERE cv_fold = ANY(%s) AND initialization_method = %s AND mahler_degree = %s").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list, initialization_method, mahler_degree)
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.zubarev_predictions WHERE cv_fold = ANY(%s) AND initialization_method = %s AND mahler_degree = %s").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list, initialization_method, mahler_degree)
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.zubarev_mahler_weights WHERE cv_fold = ANY(%s) AND initialization_method = %s AND mahler_degree = %s").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list, initialization_method, mahler_degree)
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.zubarev_iteration_history WHERE cv_fold = ANY(%s) AND initialization_method = %s AND mahler_degree = %s").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list, initialization_method, mahler_degree)
            )

        if coeff_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.zubarev_tag_coefficients (cv_fold, tag, coefficient, sequence, initialization_method, mahler_degree) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                coeff_rows,
            )
        if metrics_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.zubarev_fold_metrics "
                    "(cv_fold, loss, prime_base, max_digit, default_prediction, iterations_used, final_temperature, initialization_method, mahler_degree) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                metrics_rows,
            )
        if prediction_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.zubarev_predictions "
                    "(cv_fold, product_id, true_value, predicted_value, loss, initialization_method, mahler_degree) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                prediction_rows,
            )
        if mahler_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.zubarev_mahler_weights (cv_fold, k, weight, initialization_method, mahler_degree) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                mahler_rows,
            )
        if history_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.zubarev_iteration_history "
                    "(cv_fold, iteration, train_loss, validation_loss, best_loss, temperature, acceptance_rate, initialization_method, mahler_degree) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                history_rows,
            )
    conn.commit()


def process_database(
    dsn: str | None,
    schema: str,
    product_table: str = "cantbuymelove.product",
    cv_splits: int = 5,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    mahler_degree: int = 0,
    max_iterations: int = 10000,
    seed: int = 42,
    initialization_method: str = 'umllr',
) -> None:
    """Main entry point for Zubarev regression."""
    conn = db.get_connection(dsn)
    try:
        _ensure_storage(conn, schema)

        fold_assignments = calculate_cv_folds(conn, product_table, n_splits=cv_splits)
        if not fold_assignments:
            return

        (
            records,
            prime_base,
            max_digit,
            taxonomy_encodings,
            dataset,
        ) = _load_products(
            conn,
            product_table,
            fold_assignments,
            min_tag_count=min_tag_count,
            min_samples_per_taxonomy=min_samples_per_taxonomy,
        )
        if not records:
            return

        battles = _load_battles(conn, schema)
        results = [
            _run_fold(
                fold,
                records,
                battles,
                prime_base,
                mahler_degree=mahler_degree,
                max_iterations=max_iterations,
                seed=seed,
                initialization_method=initialization_method,
            )
            for fold in range(cv_splits)
        ]

        _save_results(conn, schema, results, prime_base, max_digit, cv_splits, initialization_method, mahler_degree)

        # Print summary
        print(f"\nZubarev P-adic Polynomial Regression Results ({initialization_method} initialization)")
        print(f"=" * 50)
        print(f"Prime base: {prime_base}")
        print(f"Mahler degree: {mahler_degree}")
        print(f"Max iterations: {max_iterations}")
        print(f"Initialization: {initialization_method}")
        print()
        for result in results:
            avg_loss = result.loss / len(result.predictions) if result.predictions else 0
            print(f"Fold {result.cv_fold}: loss={result.loss:.4f}, avg={avg_loss:.4f}, "
                  f"iters={result.iterations_used}")

    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Zubarev p-adic polynomial regression for taxonomy prediction.",
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN. If omitted, uses SHOPIFY_DB_DSN or DATABASE_URL.",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing battles data and storing results.",
    )
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table to read from.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=5,
        help="Minimum occurrences for a tag to participate.",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum products per taxonomy required.",
    )
    parser.add_argument(
        "--mahler-degree",
        type=int,
        default=0,
        help="Degree of Mahler polynomial (0=linear, 1=affine, 2+=polynomial).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum stochastic optimization iterations per fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--initialization-method",
        choices=['umllr', 'zeros'],
        default='umllr',
        help="Initialization method: 'umllr' (greedy UMLLR coefficients) or 'zeros' (all zeros).",
    )
    args = parser.parse_args()

    process_database(
        dsn=args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        cv_splits=args.cv_splits,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        mahler_degree=args.mahler_degree,
        max_iterations=args.max_iterations,
        seed=args.seed,
        initialization_method=args.initialization_method,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
