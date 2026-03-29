"""P-adic tag regression (umllr) trainer and reporting utilities."""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple, TypeVar

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db
    from padjective.cv import calculate_cv_folds
    from padjective.metrics import parse_taxonomy_path, summarize_encoded_predictions
    from padjective.tagbattle import build_battles, filter_nested_tags, split_title, tag_positions
else:  # pragma: no cover - imported as a package
    from . import data_access, db
    from .cv import calculate_cv_folds
    from .metrics import parse_taxonomy_path, summarize_encoded_predictions
    from .tagbattle import build_battles, filter_nested_tags, split_title, tag_positions


DEFAULT_TAG_ORDER_STRATEGY = "taxonomy_association"
TAG_ORDER_STRATEGIES: tuple[str, ...] = (
    "battle_elo",
    "frequency",
    "mean_title_position",
    "taxonomy_association",
    "random",
)
RANDOM_ABLATION_SEEDS: tuple[int, ...] = (7, 13, 23, 37, 101)
LIVE_SNAPSHOT_LABEL = "live"
_RowT = TypeVar("_RowT")
_LOCK_FAMILY = "padjective.umllr"


@dataclass(frozen=True)
class ProductRecord:
    product_id: int
    tags: List[str]
    encoded_path: int
    cv_fold: int
    title: str = ""
    taxonomy_id: str = ""
    taxonomy_depth: int = 0


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
class CoefficientCandidate:
    """Debug info for a candidate coefficient value."""
    candidate_value: int
    total_loss: float
    was_selected: bool


@dataclass(frozen=True)
class TagProductResidual:
    """Debug info for a product's residual before/after a tag is processed."""
    product_id: int
    residual_before: int
    residual_after: int


@dataclass(frozen=True)
class TagDebugInfo:
    """Debug info for coefficient selection for a single tag."""
    tag: str
    candidates: List[CoefficientCandidate]
    products: List[TagProductResidual]


@dataclass(frozen=True)
class FoldResult:
    cv_fold: int
    coefficients: List[TagCoefficient]
    predictions: List[Prediction]
    loss: float
    default_prediction: int
    tag_debug: List[TagDebugInfo]  # Debug info for each tag
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float


def _ensure_storage(conn, schema: str) -> None:
    """Verify Postgres tables required for umllr outputs exist.

    Tables must be created by an admin using create_umllr_tables.sql.
    This function only verifies they exist.
    """

    required_tables = [
        "umllr_tag_coefficients",
        "umllr_fold_metrics",
        "umllr_predictions",
        "umllr_taxonomy_encodings",
        "umllr_coefficient_candidates",
        "umllr_tag_products",
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
                    f"Please run create_umllr_tables.sql with admin privileges first."
                )


def _table_owner(conn, schema: str, table: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tableowner
            FROM pg_tables
            WHERE schemaname = %s AND tablename = %s
            """,
            (schema, table),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return str(row[0])


def _table_columns(conn, schema: str, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        rows = cur.fetchall()
    return {str(row[0]) for row in rows}


def _index_exists(conn, schema: str, index_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = %s AND indexname = %s
            """,
            (schema, index_name),
        )
        return cur.fetchone() is not None


def _ensure_ablation_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    db.ensure_table(
        conn,
        schema,
        "umllr_order_ablation_fold_metrics",
        columns_sql=(
            "run_key TEXT NOT NULL",
            "snapshot_ref TEXT NOT NULL DEFAULT 'live'",
            "tag_order_strategy TEXT NOT NULL",
            "tag_order_seed INTEGER",
            "cv_fold INTEGER NOT NULL",
            "loss DOUBLE PRECISION NOT NULL",
            "prime_base INTEGER NOT NULL",
            "max_digit INTEGER NOT NULL",
            "default_prediction NUMERIC",
            "exact_accuracy DOUBLE PRECISION",
            "prefix1_accuracy DOUBLE PRECISION",
            "prefix2_accuracy DOUBLE PRECISION",
            "mean_shared_prefix_depth DOUBLE PRECISION",
            "mean_scoring_ops DOUBLE PRECISION",
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            "PRIMARY KEY (run_key, cv_fold)",
        ),
    )
    db.ensure_table(
        conn,
        schema,
        "umllr_order_ablation_predictions",
        columns_sql=(
            "run_key TEXT NOT NULL",
            "snapshot_ref TEXT NOT NULL DEFAULT 'live'",
            "tag_order_strategy TEXT NOT NULL",
            "tag_order_seed INTEGER",
            "cv_fold INTEGER NOT NULL",
            "product_id BIGINT NOT NULL",
            "true_value NUMERIC NOT NULL",
            "predicted_value NUMERIC NOT NULL",
            "loss DOUBLE PRECISION NOT NULL",
            "PRIMARY KEY (run_key, cv_fold, product_id)",
        ),
    )
    metrics_table = "umllr_order_ablation_fold_metrics"
    predictions_table = "umllr_order_ablation_predictions"
    metrics_columns = _table_columns(conn, schema, metrics_table)
    prediction_columns = _table_columns(conn, schema, predictions_table)
    metrics_owner = _table_owner(conn, schema, metrics_table)
    predictions_owner = _table_owner(conn, schema, predictions_table)
    metrics_index_name = f"{schema}_umllr_ablation_snapshot_strategy_idx"
    predictions_index_name = f"{schema}_umllr_ablation_predictions_snapshot_idx"

    with conn.cursor() as cur:
        cur.execute("SELECT CURRENT_USER")
        current_user = str(cur.fetchone()[0])

        if "snapshot_ref" not in metrics_columns:
            if metrics_owner != current_user:
                raise RuntimeError(
                    f"{schema}.{metrics_table} is missing snapshot_ref, "
                    f"but the current database role {current_user!r} does not own the table "
                    f"(owner: {metrics_owner!r}). Transfer ownership or run the migration "
                    "as the owning role before retrying."
                )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {schema}.umllr_order_ablation_fold_metrics "
                    "ADD COLUMN IF NOT EXISTS snapshot_ref TEXT NOT NULL DEFAULT 'live'"
                ).format(schema=sql.Identifier(schema))
            )
            metrics_columns.add("snapshot_ref")

        if "snapshot_ref" not in prediction_columns:
            if predictions_owner != current_user:
                raise RuntimeError(
                    f"{schema}.{predictions_table} is missing snapshot_ref, "
                    f"but the current database role {current_user!r} does not own the table "
                    f"(owner: {predictions_owner!r}). Transfer ownership or run the migration "
                    "as the owning role before retrying."
                )
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {schema}.umllr_order_ablation_predictions "
                    "ADD COLUMN IF NOT EXISTS snapshot_ref TEXT NOT NULL DEFAULT 'live'"
                ).format(schema=sql.Identifier(schema))
            )
            prediction_columns.add("snapshot_ref")

        if "snapshot_ref" in metrics_columns:
            cur.execute(
                sql.SQL(
                    "UPDATE {schema}.umllr_order_ablation_fold_metrics "
                    "SET snapshot_ref = 'live' WHERE snapshot_ref IS NULL"
                ).format(schema=sql.Identifier(schema))
            )

        if "snapshot_ref" in prediction_columns:
            cur.execute(
                sql.SQL(
                    "UPDATE {schema}.umllr_order_ablation_predictions "
                    "SET snapshot_ref = 'live' WHERE snapshot_ref IS NULL"
                ).format(schema=sql.Identifier(schema))
            )

        if metrics_owner == current_user:
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {schema}.umllr_order_ablation_fold_metrics (snapshot_ref, tag_order_strategy, tag_order_seed) "
                    "TABLESPACE pg_default"
                ).format(
                    index=sql.Identifier(metrics_index_name),
                    schema=sql.Identifier(schema),
                )
            )
        elif not _index_exists(conn, schema, metrics_index_name):
            warnings.warn(
                f"Skipping creation of index {metrics_index_name!r}: "
                f"{schema}.{metrics_table} is owned by {metrics_owner!r}, not {current_user!r}.",
                RuntimeWarning,
            )

        if predictions_owner == current_user:
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {index} "
                    "ON {schema}.umllr_order_ablation_predictions (snapshot_ref, tag_order_strategy, tag_order_seed, cv_fold) "
                    "TABLESPACE pg_default"
                ).format(
                    index=sql.Identifier(predictions_index_name),
                    schema=sql.Identifier(schema),
                )
            )
        elif not _index_exists(conn, schema, predictions_index_name):
            warnings.warn(
                f"Skipping creation of index {predictions_index_name!r}: "
                f"{schema}.{predictions_table} is owned by {predictions_owner!r}, not {current_user!r}.",
                RuntimeWarning,
            )
    conn.commit()


def snapshot_label(snapshot_ref: str | None) -> str:
    cleaned = (snapshot_ref or "").strip()
    return cleaned or LIVE_SNAPSHOT_LABEL


def tag_order_run_key(
    strategy: str,
    seed: int | None = None,
    *,
    snapshot_ref: str | None = None,
) -> str:
    if strategy == "random":
        if seed is None:
            raise ValueError("random tag ordering requires a seed")
        base_key = f"{strategy}_seed_{seed}"
    else:
        base_key = strategy
    return f"{snapshot_label(snapshot_ref)}::{base_key}"


def _acquire_session_lock(conn, lock_name: str) -> None:
    """Serialize overlapping UMLLR runs that target the same outputs."""

    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_lock(hashtext(%s), hashtext(%s))",
            (_LOCK_FAMILY, lock_name),
        )


def _truncate_outputs(conn, schema: str) -> None:
    db.truncate_table(conn, schema, "umllr_tag_coefficients")
    db.truncate_table(conn, schema, "umllr_fold_metrics")
    db.truncate_table(conn, schema, "umllr_predictions")
    db.truncate_table(conn, schema, "umllr_taxonomy_encodings")
    db.truncate_table(conn, schema, "umllr_coefficient_candidates")
    db.truncate_table(conn, schema, "umllr_tag_products")


def _dedupe_rows_by_key(
    rows: Sequence[_RowT],
    *,
    key_fn: Callable[[_RowT], object],
    description: str,
) -> list[_RowT]:
    deduped: list[_RowT] = []
    seen: set[object] = set()
    duplicate_count = 0

    for row in rows:
        key = key_fn(row)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(row)

    if duplicate_count:
        warnings.warn(
            f"Dropped {duplicate_count} duplicate {description} row(s) before insert.",
            RuntimeWarning,
            stacklevel=2,
        )

    return deduped


def _parse_tags(tag_string: str | None) -> List[str]:
    if not tag_string:
        return []
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]
    if not tags:
        return []
    filtered = filter_nested_tags(tags)
    return [tag.upper() for tag in filtered]


_SEGMENT_NUMBER_RE = re.compile(r"-?\d+")


def _parse_taxonomy_digits(path_value: str | None) -> Tuple[int, ...]:
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
        # If the segment has no numeric information we skip it entirely.
    return tuple(digits)


def _encode_path(digits: Sequence[int], base: int) -> int:
    value = 0
    for power, digit in enumerate(digits):
        value += digit * (base ** power)
    return value


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
    if a == b:
        return 0.0

    diff = abs(a - b)
    valuation = 0
    while diff % base == 0:
        diff //= base
        valuation += 1
    return base ** (-valuation)


def _select_coefficient(
    values: Sequence[int], base: int
) -> Tuple[int, List[CoefficientCandidate]]:
    """Select the best coefficient and return debug info about all candidates tried.

    Returns:
        Tuple of (selected_coefficient, list of CoefficientCandidate debug info)
    """
    unique_values = sorted(set(values))
    best_value = unique_values[0]
    best_loss = math.inf

    # Track all candidates and their losses for debugging
    candidate_losses: List[Tuple[int, float]] = []

    for candidate in unique_values:
        total_distance = sum(_p_adic_distance(candidate, value, base) for value in values)
        candidate_losses.append((candidate, total_distance))
        if total_distance < best_loss or (
            math.isclose(total_distance, best_loss) and candidate < best_value
        ):
            best_loss = total_distance
            best_value = candidate

    # Build debug info
    candidates = [
        CoefficientCandidate(
            candidate_value=cand,
            total_loss=loss,
            was_selected=(cand == best_value),
        )
        for cand, loss in candidate_losses
    ]

    return best_value, candidates


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


def _derive_battles_from_records(records: Sequence[ProductRecord]) -> List[BattleRecord]:
    derived: List[BattleRecord] = []
    for record in records:
        if not record.title or len(record.tags) < 2:
            continue
        for winner_tag, loser_tag in build_battles(record.title, ",".join(record.tags)):
            derived.append(
                BattleRecord(
                    winner_tag=winner_tag,
                    loser_tag=loser_tag,
                    cv_fold=record.cv_fold,
                )
            )
    return derived


def _load_products(
    conn,
    product_table: str,
    fold_assignments: Dict[int, int] | None,
    *,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
) -> tuple[List[ProductRecord], int, int, Dict[str, Tuple[str, int]], data_access.ProductDataset]:
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
        snapshot_ref=snapshot_ref,
        snapshot_schema=snapshot_schema,
    )

    records: List[ProductRecord] = []
    max_digit = 0
    raw_entries: List[tuple[int, str, List[str], Tuple[int, ...], int, str, str]] = []

    # Use valid tags from dataset (respects min_tag_count)
    valid_tags = set(dataset.feature_names)

    for record in dataset.records:
        cv_fold = record.cv_fold if fold_assignments is None else fold_assignments.get(record.product_id)
        if cv_fold is None:
            continue
        # Filter nested tags and then filter by valid_tags (min_tag_count)
        nested_filtered = filter_nested_tags(record.tags)
        filtered_tags = [tag.upper() for tag in nested_filtered if tag in valid_tags]
        taxonomy_id = record.taxonomy_id or ""
        taxonomy_path = record.taxonomy_path or ""
        digits = _parse_taxonomy_digits(taxonomy_path)
        if digits:
            max_digit = max(max_digit, max(digits))
        raw_entries.append(
            (
                record.product_id,
                record.title,
                filtered_tags,
                digits,
                cv_fold,
                taxonomy_id,
                taxonomy_path,
            )
        )

    prime_base = _next_prime(max_digit)
    taxonomy_encodings: Dict[str, Tuple[str, int]] = {}

    for product_id, title, tags, digits, cv_fold, taxonomy_id, taxonomy_path in raw_entries:
        encoded = _encode_path(digits, prime_base)
        records.append(
            ProductRecord(
                product_id=product_id,
                title=title,
                tags=tags,
                encoded_path=encoded,
                cv_fold=cv_fold,
                taxonomy_id=taxonomy_id,
                taxonomy_depth=len(digits),
            )
        )
        if taxonomy_id and taxonomy_id not in taxonomy_encodings:
            taxonomy_encodings[taxonomy_id] = (taxonomy_path, encoded)

    return records, prime_base, max_digit, taxonomy_encodings, dataset


def _battle_elo_order(
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    training_tags: Sequence[str],
) -> List[str]:
    ordered_tags = sorted(set(training_tags))
    if not ordered_tags:
        return []

    ratings = {tag: 0.0 for tag in ordered_tags}
    filtered_battles = [
        battle
        for battle in battles
        if battle.cv_fold != holdout_fold
        and battle.winner_tag in ratings
        and battle.loser_tag in ratings
    ]
    for battle in filtered_battles:
        winner_rating = ratings[battle.winner_tag]
        loser_rating = ratings[battle.loser_tag]
        expected_win = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
        expected_loss = 1.0 - expected_win
        ratings[battle.winner_tag] = winner_rating + 32.0 * (1.0 - expected_win)
        ratings[battle.loser_tag] = loser_rating + 32.0 * (0.0 - expected_loss)

    return sorted(ordered_tags, key=lambda tag: (-ratings[tag], tag))


def _frequency_order(training: Sequence[ProductRecord]) -> List[str]:
    counts: Counter[str] = Counter()
    for record in training:
        counts.update(record.tags)
    return [tag for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _mean_title_position_order(training: Sequence[ProductRecord]) -> List[str]:
    scores: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()

    for record in training:
        counts.update(record.tags)
        for part_idx, part in enumerate(split_title(record.title or "")):
            positions = tag_positions(part, record.tags)
            for tag, position in positions.items():
                scores[tag].append(float(part_idx * 100000 + position))

    def mean_position(tag: str) -> float:
        values = scores.get(tag)
        if not values:
            return float("-inf")
        return float(sum(values) / len(values))

    ordered_tags = sorted(set(counts), key=lambda tag: (-mean_position(tag), -counts[tag], tag))
    return ordered_tags


def _taxonomy_association_order(training: Sequence[ProductRecord]) -> List[str]:
    tag_totals: Counter[str] = Counter()
    tag_taxonomy_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for record in training:
        if not record.taxonomy_id:
            continue
        for tag in record.tags:
            tag_totals[tag] += 1
            tag_taxonomy_counts[tag][record.taxonomy_id] += 1

    def association_strength(tag: str) -> float:
        total = tag_totals[tag]
        if total <= 0:
            return 0.0
        return max(tag_taxonomy_counts[tag].values(), default=0) / total

    return [
        tag
        for tag in sorted(
            tag_totals,
            key=lambda tag: (-association_strength(tag), -tag_totals[tag], tag),
        )
    ]


def _random_order(training_tags: Sequence[str], seed: int | None) -> List[str]:
    if seed is None:
        raise ValueError("random tag ordering requires an explicit seed")
    ordered_tags = sorted(set(training_tags))
    rng = random.Random(seed)
    rng.shuffle(ordered_tags)
    return ordered_tags


def _tag_order(
    training: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    *,
    strategy: str,
    seed: int | None = None,
) -> List[str]:
    training_tags = sorted({tag for record in training for tag in record.tags})
    if strategy == "battle_elo":
        return _battle_elo_order(battles, holdout_fold, training_tags)
    if strategy == "frequency":
        return _frequency_order(training)
    if strategy == "mean_title_position":
        return _mean_title_position_order(training)
    if strategy == "taxonomy_association":
        return _taxonomy_association_order(training)
    if strategy == "random":
        return _random_order(training_tags, seed)
    raise ValueError(f"Unknown tag order strategy: {strategy}")


def _select_default_prediction(
    no_tag_values: Sequence[int],
    candidate_values: Sequence[int],
    base: int,
) -> int:
    """Select the default prediction that minimizes p-adic loss for products with no tags.

    Due to ultrametricity, the optimal default is one of the existing taxonomy values.
    We check each unique candidate and pick the one with minimum total p-adic loss.
    """
    if not no_tag_values:
        # No products without tags to optimize for; fall back to most common
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
    tag_order_strategy: str = DEFAULT_TAG_ORDER_STRATEGY,
    tag_order_seed: int | None = None,
) -> FoldResult:
    training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]

    product_residuals: Dict[int, int] = {record.product_id: record.encoded_path for record in training}
    tag_to_products: Dict[str, List[int]] = {}
    for record in training:
        for tag in record.tags:
            tag_to_products.setdefault(tag, []).append(record.product_id)

    tag_order = _tag_order(
        training,
        battles,
        fold,
        strategy=tag_order_strategy,
        seed=tag_order_seed,
    )

    coefficients: List[TagCoefficient] = []
    tag_debug: List[TagDebugInfo] = []

    for sequence, tag in enumerate(tag_order):
        product_ids = tag_to_products.get(tag, [])
        values = [product_residuals[pid] for pid in product_ids]

        if values:
            coefficient, candidates = _select_coefficient(values, base)

            # Capture residuals before and after for each product
            product_residual_info = []
            for pid in product_ids:
                residual_before = product_residuals[pid]
                residual_after = residual_before - coefficient
                product_residual_info.append(
                    TagProductResidual(
                        product_id=pid,
                        residual_before=residual_before,
                        residual_after=residual_after,
                    )
                )
                # Update the residual
                product_residuals[pid] = residual_after
        else:
            coefficient = 0
            candidates = []
            product_residual_info = []

        coefficients.append(TagCoefficient(tag=tag, coefficient=coefficient, sequence=sequence))
        tag_debug.append(TagDebugInfo(tag=tag, candidates=candidates, products=product_residual_info))

    coefficient_lookup = {entry.tag: entry.coefficient for entry in coefficients}

    # Find training products with no contributing tags (prediction would be 0)
    no_tag_training_values = [
        record.encoded_path
        for record in training
        if sum(coefficient_lookup.get(tag, 0) for tag in record.tags) == 0
    ]
    all_training_values = [record.encoded_path for record in training]

    # Select default that minimizes p-adic loss for products with no tags
    default_prediction = _select_default_prediction(
        no_tag_training_values, all_training_values, base
    )

    predictions: List[Prediction] = []
    total_loss = 0.0
    scoring_ops: list[float] = []
    for record in testing:
        active_coefficients = [
            coefficient_lookup.get(tag, 0)
            for tag in record.tags
            if coefficient_lookup.get(tag, 0) != 0
        ]
        predicted = sum(active_coefficients)
        # Use default prediction when no tags contributed
        if predicted == 0:
            predicted = default_prediction
        loss = _p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss
        scoring_ops.append(float(len(active_coefficients)))
        predictions.append(
            Prediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )

    summary = summarize_encoded_predictions(
        [record.encoded_path for record in testing],
        [prediction.predicted_value for prediction in predictions],
        base=base,
        true_depths=[record.taxonomy_depth for record in testing],
        scoring_ops=scoring_ops,
    )

    return FoldResult(
        cv_fold=fold,
        coefficients=coefficients,
        predictions=predictions,
        loss=total_loss,
        default_prediction=default_prediction,
        tag_debug=tag_debug,
        exact_accuracy=summary.exact_accuracy,
        prefix1_accuracy=summary.prefix1_accuracy,
        prefix2_accuracy=summary.prefix2_accuracy,
        mean_shared_prefix_depth=summary.mean_shared_prefix_depth,
        mean_scoring_ops=summary.mean_scoring_ops or 0.0,
    )


def _save_results(
    conn,
    schema: str,
    results: Sequence[FoldResult],
    prime_base: int,
    max_digit: int,
    taxonomy_encodings: Dict[str, Tuple[str, int]],
    cv_splits: int,
    *,
    tag_order_strategy: str = DEFAULT_TAG_ORDER_STRATEGY,
    tag_order_seed: int | None = None,
) -> None:
    coeff_rows: List[Tuple[int, str, int, int]] = []
    prediction_rows: List[Tuple[int, int, int, int, float]] = []
    metrics_rows: List[Tuple[int, float, int, int, int, float, float, float, float, float, str, int | None]] = []
    encoding_rows: List[Tuple[int, str, str, int]] = []
    candidate_rows: List[Tuple[int, str, int, float, int, bool]] = []
    tag_product_rows: List[Tuple[int, str, int, int, int]] = []

    for result in results:
        metrics_rows.append(
            (
                result.cv_fold,
                result.loss,
                prime_base,
                max_digit,
                result.default_prediction,
                result.exact_accuracy,
                result.prefix1_accuracy,
                result.prefix2_accuracy,
                result.mean_shared_prefix_depth,
                result.mean_scoring_ops,
                tag_order_strategy,
                tag_order_seed,
            )
        )
        for entry in result.coefficients:
            coeff_rows.append((result.cv_fold, entry.tag, entry.coefficient, entry.sequence))
        for prediction in result.predictions:
            prediction_rows.append(
                (
                    result.cv_fold,
                    prediction.product_id,
                    prediction.true_value,
                    prediction.predicted_value,
                    prediction.loss,
                )
            )
        # Collect debug info
        for tag_debug in result.tag_debug:
            for candidate in tag_debug.candidates:
                candidate_rows.append(
                    (
                        result.cv_fold,
                        tag_debug.tag,
                        candidate.candidate_value,
                        candidate.total_loss,
                        len(tag_debug.products),  # product_count
                        candidate.was_selected,
                    )
                )
            for product_info in tag_debug.products:
                tag_product_rows.append(
                    (
                        result.cv_fold,
                        tag_debug.tag,
                        product_info.product_id,
                        product_info.residual_before,
                        product_info.residual_after,
                    )
                )

    # Save taxonomy encodings for each fold
    for fold in range(cv_splits):
        for taxonomy_id, (taxonomy_path, encoded_value) in taxonomy_encodings.items():
            encoding_rows.append((fold, taxonomy_id, taxonomy_path, encoded_value))

    coeff_rows = _dedupe_rows_by_key(
        coeff_rows,
        key_fn=lambda row: (row[0], row[1]),
        description="UMLLR coefficient",
    )
    prediction_rows = _dedupe_rows_by_key(
        prediction_rows,
        key_fn=lambda row: (row[0], row[1]),
        description="UMLLR prediction",
    )
    metrics_rows = _dedupe_rows_by_key(
        metrics_rows,
        key_fn=lambda row: row[0],
        description="UMLLR fold metric",
    )
    encoding_rows = _dedupe_rows_by_key(
        encoding_rows,
        key_fn=lambda row: (row[0], row[1]),
        description="UMLLR taxonomy encoding",
    )
    candidate_rows = _dedupe_rows_by_key(
        candidate_rows,
        key_fn=lambda row: (row[0], row[1], row[2]),
        description="UMLLR coefficient candidate",
    )
    tag_product_rows = _dedupe_rows_by_key(
        tag_product_rows,
        key_fn=lambda row: (row[0], row[1], row[2]),
        description="UMLLR tag-product",
    )

    with conn.cursor() as cur:
        # Clean up old data before inserting new results
        # This ensures cronscript can be re-run without conflicts
        fold_list = list(range(cv_splits))

        if fold_list:
            # Delete old coefficients for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_tag_coefficients WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            # Delete old metrics for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_fold_metrics WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            # Delete old predictions for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_predictions WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            # Delete old encodings for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_taxonomy_encodings WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            # Delete old coefficient candidates for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_coefficient_candidates WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            # Delete old tag products for all folds
            cur.execute(
                sql.SQL("DELETE FROM {schema}.umllr_tag_products WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )

        # Insert new results
        if coeff_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_tag_coefficients (cv_fold, tag, coefficient, sequence) "
                    "VALUES (%s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                coeff_rows,
            )
        if metrics_rows:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'umllr_fold_metrics'
                """,
                (schema,)
            )
            available_columns = {row[0] for row in cur.fetchall()}

            if {"default_prediction", "exact_accuracy", "prefix1_accuracy", "prefix2_accuracy",
                "mean_shared_prefix_depth", "mean_scoring_ops", "tag_order_strategy", "tag_order_seed"} <= available_columns:
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {schema}.umllr_fold_metrics "
                        "(cv_fold, loss, prime_base, max_digit, default_prediction, "
                        "exact_accuracy, prefix1_accuracy, prefix2_accuracy, mean_shared_prefix_depth, "
                        "mean_scoring_ops, tag_order_strategy, tag_order_seed) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(schema=sql.Identifier(schema)),
                    metrics_rows,
                )
            elif "default_prediction" in available_columns:
                metrics_rows_with_default = [(row[0], row[1], row[2], row[3], row[4]) for row in metrics_rows]
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {schema}.umllr_fold_metrics (cv_fold, loss, prime_base, max_digit, default_prediction) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(schema=sql.Identifier(schema)),
                    metrics_rows_with_default,
                )
            else:
                metrics_rows_compat = [(row[0], row[1], row[2], row[3]) for row in metrics_rows]
                cur.executemany(
                    sql.SQL(
                        "INSERT INTO {schema}.umllr_fold_metrics (cv_fold, loss, prime_base, max_digit) "
                        "VALUES (%s, %s, %s, %s)"
                    ).format(schema=sql.Identifier(schema)),
                    metrics_rows_compat,
                )
        if prediction_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_predictions (cv_fold, product_id, true_value, predicted_value, loss) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                prediction_rows,
            )
        if encoding_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_taxonomy_encodings (cv_fold, taxonomy_id, taxonomy_path, encoded_value) "
                    "VALUES (%s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                encoding_rows,
            )
        if candidate_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_coefficient_candidates "
                    "(cv_fold, tag, candidate_value, total_loss, product_count, was_selected) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                candidate_rows,
            )
        if tag_product_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_tag_products "
                    "(cv_fold, tag, product_id, residual_before, residual_after) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                tag_product_rows,
            )
    conn.commit()


def _save_ablation_results(
    conn,
    schema: str,
    results: Sequence[FoldResult],
    *,
    run_key: str,
    snapshot_ref: str,
    tag_order_strategy: str,
    tag_order_seed: int | None,
    prime_base: int,
    max_digit: int,
) -> None:
    _ensure_ablation_storage(conn, schema)

    metric_rows = [
        (
            run_key,
            snapshot_ref,
            tag_order_strategy,
            tag_order_seed,
            result.cv_fold,
            result.loss,
            prime_base,
            max_digit,
            result.default_prediction,
            result.exact_accuracy,
            result.prefix1_accuracy,
            result.prefix2_accuracy,
            result.mean_shared_prefix_depth,
            result.mean_scoring_ops,
        )
        for result in results
    ]
    prediction_rows = [
        (
            run_key,
            snapshot_ref,
            tag_order_strategy,
            tag_order_seed,
            result.cv_fold,
            prediction.product_id,
            prediction.true_value,
            prediction.predicted_value,
            prediction.loss,
        )
        for result in results
        for prediction in result.predictions
    ]

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DELETE FROM {schema}.umllr_order_ablation_fold_metrics WHERE run_key = %s").format(
                schema=sql.Identifier(schema)
            ),
            (run_key,),
        )
        cur.execute(
            sql.SQL("DELETE FROM {schema}.umllr_order_ablation_predictions WHERE run_key = %s").format(
                schema=sql.Identifier(schema)
            ),
            (run_key,),
        )
        if metric_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_order_ablation_fold_metrics "
                    "(run_key, snapshot_ref, tag_order_strategy, tag_order_seed, cv_fold, loss, prime_base, max_digit, "
                    "default_prediction, exact_accuracy, prefix1_accuracy, prefix2_accuracy, "
                    "mean_shared_prefix_depth, mean_scoring_ops) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_key, cv_fold) DO UPDATE SET "
                    "snapshot_ref = EXCLUDED.snapshot_ref, "
                    "tag_order_strategy = EXCLUDED.tag_order_strategy, "
                    "tag_order_seed = EXCLUDED.tag_order_seed, "
                    "loss = EXCLUDED.loss, "
                    "prime_base = EXCLUDED.prime_base, "
                    "max_digit = EXCLUDED.max_digit, "
                    "default_prediction = EXCLUDED.default_prediction, "
                    "exact_accuracy = EXCLUDED.exact_accuracy, "
                    "prefix1_accuracy = EXCLUDED.prefix1_accuracy, "
                    "prefix2_accuracy = EXCLUDED.prefix2_accuracy, "
                    "mean_shared_prefix_depth = EXCLUDED.mean_shared_prefix_depth, "
                    "mean_scoring_ops = EXCLUDED.mean_scoring_ops, "
                    "updated_at = now()"
                ).format(schema=sql.Identifier(schema)),
                metric_rows,
            )
        if prediction_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_order_ablation_predictions "
                    "(run_key, snapshot_ref, tag_order_strategy, tag_order_seed, cv_fold, product_id, true_value, predicted_value, loss) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_key, cv_fold, product_id) DO UPDATE SET "
                    "snapshot_ref = EXCLUDED.snapshot_ref, "
                    "tag_order_strategy = EXCLUDED.tag_order_strategy, "
                    "tag_order_seed = EXCLUDED.tag_order_seed, "
                    "true_value = EXCLUDED.true_value, "
                    "predicted_value = EXCLUDED.predicted_value, "
                    "loss = EXCLUDED.loss, "
                    "updated_at = now()"
                ).format(schema=sql.Identifier(schema)),
                prediction_rows,
            )
    conn.commit()


@dataclass(frozen=True)
class DummyFoldResult:
    """Results from a dummy classifier (always predicts most common class)."""
    cv_fold: int
    predictions: List[Prediction]
    loss: float
    accuracy: float
    most_common_value: int
    most_common_taxonomy_id: str


def _run_dummy_fold(
    fold: int,
    records: Sequence[ProductRecord],
    taxonomy_encodings: Dict[str, Tuple[str, int]],
    base: int,
) -> DummyFoldResult:
    """Run dummy classifier that always predicts the most common class."""
    from collections import Counter

    training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]

    # Find most common taxonomy encoding in training data
    training_values = [record.encoded_path for record in training]
    if training_values:
        most_common_value = Counter(training_values).most_common(1)[0][0]
    else:
        most_common_value = 0

    # Find the taxonomy_id for the most common value
    most_common_taxonomy_id = ""
    for taxonomy_id, (_, encoded_value) in taxonomy_encodings.items():
        if encoded_value == most_common_value:
            most_common_taxonomy_id = taxonomy_id
            break

    # Make predictions
    predictions: List[Prediction] = []
    total_loss = 0.0
    correct_count = 0

    for record in testing:
        predicted = most_common_value
        loss = _p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss

        if predicted == record.encoded_path:
            correct_count += 1

        predictions.append(
            Prediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )

    accuracy = correct_count / len(testing) if testing else 0.0
    average_loss = total_loss / len(testing) if testing else 0.0

    return DummyFoldResult(
        cv_fold=fold,
        predictions=predictions,
        loss=average_loss,
        accuracy=accuracy,
        most_common_value=most_common_value,
        most_common_taxonomy_id=most_common_taxonomy_id,
    )


def _save_dummy_results(
    conn,
    schema: str,
    results: Sequence[DummyFoldResult],
    cv_splits: int,
) -> None:
    """Save dummy classifier results to database."""
    # Check if dummy tables exist
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = 'dummy_fold_metrics'
            """,
            (schema,)
        )
        if not cur.fetchone():
            # Tables don't exist, skip saving
            return

    prediction_rows: List[Tuple[int, int, int, int, float]] = []
    metrics_rows: List[Tuple[int, float, float, int, str]] = []

    for result in results:
        metrics_rows.append((
            result.cv_fold,
            result.loss,
            result.accuracy,
            result.most_common_value,
            result.most_common_taxonomy_id,
        ))
        for prediction in result.predictions:
            prediction_rows.append((
                result.cv_fold,
                prediction.product_id,
                prediction.true_value,
                prediction.predicted_value,
                prediction.loss,
            ))

    with conn.cursor() as cur:
        # Clean up old data
        fold_list = list(range(cv_splits))
        if fold_list:
            cur.execute(
                sql.SQL("DELETE FROM {schema}.dummy_fold_metrics WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.dummy_predictions WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_list,)
            )

        # Insert new results
        if metrics_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.dummy_fold_metrics "
                    "(cv_fold, loss, accuracy, most_common_value, most_common_taxonomy_id) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                metrics_rows,
            )
        if prediction_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.dummy_predictions "
                    "(cv_fold, product_id, true_value, predicted_value, loss) "
                    "VALUES (%s, %s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                prediction_rows,
            )
    conn.commit()


def process_database(
    dsn: str | None,
    schema: str,
    product_table: str = "cantbuymelove.product",
    cv_splits: int = 5,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    tag_order_strategy: str = DEFAULT_TAG_ORDER_STRATEGY,
    tag_order_seed: int | None = None,
    ablation_only: bool = False,
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
) -> None:
    conn = db.get_connection(dsn)
    try:
        # Session advisory locks survive commits, which lets us serialize the
        # delete/rewrite flow across the whole job while reusing one connection.
        _acquire_session_lock(conn, f"{schema}.process_database")
        _ensure_storage(conn, schema)

        fold_assignments: Dict[int, int] | None
        if snapshot_ref is None:
            fold_assignments = calculate_cv_folds(conn, product_table, n_splits=cv_splits)
            if not fold_assignments:
                return
        else:
            fold_assignments = None

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
            snapshot_ref=snapshot_ref,
            snapshot_schema=snapshot_schema,
        )
        if not records:
            return

        if snapshot_ref is None:
            battles = _load_battles(conn, schema)
        else:
            battles = _derive_battles_from_records(records)
        current_snapshot_label = snapshot_label(snapshot_ref)
        run_specs: list[tuple[str, int | None]] = []
        if tag_order_strategy == "all":
            run_specs.extend((strategy, None) for strategy in TAG_ORDER_STRATEGIES if strategy != "random")
            run_specs.extend(("random", seed) for seed in RANDOM_ABLATION_SEEDS)
        else:
            if tag_order_strategy not in TAG_ORDER_STRATEGIES:
                raise ValueError(
                    f"tag_order_strategy must be one of {', '.join(TAG_ORDER_STRATEGIES)} or 'all'"
                )
            run_specs.append((tag_order_strategy, tag_order_seed))

        default_results: Sequence[FoldResult] | None = None

        for strategy, seed in run_specs:
            results = [
                _run_fold(
                    fold,
                    records,
                    battles,
                    prime_base,
                    tag_order_strategy=strategy,
                    tag_order_seed=seed,
                )
                for fold in range(cv_splits)
            ]
            run_key = tag_order_run_key(strategy, seed, snapshot_ref=snapshot_ref)
            _save_ablation_results(
                conn,
                schema,
                results,
                run_key=run_key,
                snapshot_ref=current_snapshot_label,
                tag_order_strategy=strategy,
                tag_order_seed=seed,
                prime_base=prime_base,
                max_digit=max_digit,
            )

            if strategy == DEFAULT_TAG_ORDER_STRATEGY and seed is None:
                default_results = results

        if ablation_only:
            return

        if default_results is None:
            default_results = [
                _run_fold(
                    fold,
                    records,
                    battles,
                    prime_base,
                    tag_order_strategy=DEFAULT_TAG_ORDER_STRATEGY,
                )
                for fold in range(cv_splits)
            ]

        _truncate_outputs(conn, schema)
        _save_results(
            conn,
            schema,
            default_results,
            prime_base,
            max_digit,
            taxonomy_encodings,
            cv_splits,
            tag_order_strategy=DEFAULT_TAG_ORDER_STRATEGY,
            tag_order_seed=None,
        )

        if tag_order_strategy in {DEFAULT_TAG_ORDER_STRATEGY, "all"} and tag_order_seed is None:
            dummy_results = [
                _run_dummy_fold(fold, records, taxonomy_encodings, prime_base)
                for fold in range(cv_splits)
            ]
            _save_dummy_results(conn, schema, dummy_results, cv_splits)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train p-adic tag coefficients and evaluate them via cross-validation.",
    )
    parser.add_argument(
        "--dsn",
        help=(
            "Postgres DSN for the Shopify stores database. If omitted, the script "
            "uses SHOPIFY_DB_DSN or DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing battles data and storing umllr results.",
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
        help="Number of cross-validation folds to evaluate.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=5,
        help="Minimum occurrences required for a tag to participate in training.",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum number of products per taxonomy required for training.",
    )
    parser.add_argument(
        "--tag-order-strategy",
        default=DEFAULT_TAG_ORDER_STRATEGY,
        help=(
            "Tag ordering strategy: battle_elo, frequency, mean_title_position, "
            "taxonomy_association, random, or all."
        ),
    )
    parser.add_argument(
        "--tag-order-seed",
        type=int,
        help="Random seed used when --tag-order-strategy=random.",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="Persist only tag-order ablation tables and leave the primary UMLLR outputs untouched.",
    )
    parser.add_argument(
        "--snapshot-ref",
        help="Optional benchmark snapshot alias/name/UUID to use instead of the live catalog.",
    )
    parser.add_argument(
        "--snapshot-schema",
        default="padjective",
        help="Schema containing product_taxonomy_bench snapshot tables.",
    )
    args = parser.parse_args()

    process_database(
        dsn=args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        cv_splits=args.cv_splits,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        tag_order_strategy=args.tag_order_strategy,
        tag_order_seed=args.tag_order_seed,
        ablation_only=args.ablation_only,
        snapshot_ref=args.snapshot_ref,
        snapshot_schema=args.snapshot_schema,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
