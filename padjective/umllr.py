"""P-adic tag regression (umllr) trainer and reporting utilities."""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
    from padjective.cv import calculate_cv_folds
    from padjective.metrics import parse_taxonomy_path
    from padjective.tagbattle import filter_nested_tags
else:  # pragma: no cover - imported as a package
    from . import db
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
class FoldResult:
    cv_fold: int
    coefficients: List[TagCoefficient]
    predictions: List[Prediction]
    loss: float


def _ensure_storage(conn, schema: str) -> None:
    """Create Postgres tables required for umllr outputs."""

    coeff_columns = (
        "cv_fold INTEGER NOT NULL",
        "tag TEXT NOT NULL",
        "coefficient NUMERIC NOT NULL",
        "sequence INTEGER NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        "PRIMARY KEY (cv_fold, tag)",
    )
    coeff_indexes = (
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON {schema}.umllr_tag_coefficients (sequence) TABLESPACE pg_default"
        ).format(
            name=sql.Identifier(f"{schema}_umllr_coeff_sequence_idx"),
            schema=sql.Identifier(schema),
        ),
    )

    metrics_columns = (
        "cv_fold INTEGER PRIMARY KEY",
        "loss DOUBLE PRECISION NOT NULL",
        "prime_base INTEGER NOT NULL",
        "max_digit INTEGER NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )

    predictions_columns = (
        "cv_fold INTEGER NOT NULL",
        "product_id BIGINT NOT NULL",
        "true_value NUMERIC NOT NULL",
        "predicted_value NUMERIC NOT NULL",
        "loss DOUBLE PRECISION NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        "PRIMARY KEY (cv_fold, product_id)",
    )
    predictions_indexes = (
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {name} ON {schema}.umllr_predictions (cv_fold) TABLESPACE pg_default"
        ).format(
            name=sql.Identifier(f"{schema}_umllr_predictions_fold_idx"),
            schema=sql.Identifier(schema),
        ),
    )

    db.ensure_schema(conn, schema)
    db.ensure_table(conn, schema, "umllr_tag_coefficients", coeff_columns, coeff_indexes)
    db.ensure_table(conn, schema, "umllr_fold_metrics", metrics_columns, ())
    db.ensure_table(conn, schema, "umllr_predictions", predictions_columns, predictions_indexes)


def _truncate_outputs(conn, schema: str) -> None:
    db.truncate_table(conn, schema, "umllr_tag_coefficients")
    db.truncate_table(conn, schema, "umllr_fold_metrics")
    db.truncate_table(conn, schema, "umllr_predictions")


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


def _select_coefficient(values: Sequence[int], base: int) -> int:
    unique_values = sorted(set(values))
    best_value = unique_values[0]
    best_loss = math.inf

    for candidate in unique_values:
        total_distance = sum(_p_adic_distance(candidate, value, base) for value in values)
        if total_distance < best_loss or (
            math.isclose(total_distance, best_loss) and candidate < best_value
        ):
            best_loss = total_distance
            best_value = candidate
    return best_value


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
    conn, product_table: str, fold_assignments: Dict[int, int]
) -> tuple[List[ProductRecord], int, int]:
    product_identifier = db.qualified_identifier(product_table)
    query = sql.SQL(
        """
        SELECT
            p.id,
            pd.product_detail->'product'->>'tags' AS tags,
            pt.taxonomy_path
        FROM {products} AS p
        JOIN public.product_details pd ON (
            p.myshopify_domain = pd.myshopify_domain
            AND p.run_name = pd.run_name
            AND p.product_handle = pd.product_handle
        )
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        WHERE p.product_title IS NOT NULL
        ORDER BY p.id
        """
    ).format(products=product_identifier)

    records: List[ProductRecord] = []
    max_digit = 0
    raw_entries: List[tuple[int, List[str], Tuple[int, ...], int]] = []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            product_id = row.get("id")
            if product_id is None:
                continue
            cv_fold = fold_assignments.get(product_id)
            if cv_fold is None:
                continue
            tags = _parse_tags(row.get("tags"))
            digits = _parse_taxonomy_digits(row.get("taxonomy_path"))
            if digits:
                max_digit = max(max_digit, max(digits))
            raw_entries.append((product_id, tags, digits, cv_fold))

    prime_base = _next_prime(max_digit)

    for product_id, tags, digits, cv_fold in raw_entries:
        encoded = _encode_path(digits, prime_base)
        records.append(ProductRecord(product_id, tags, encoded, cv_fold))

    return records, prime_base, max_digit


def _tag_order(
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    training_tags: Iterable[str],
) -> List[str]:
    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}

    for battle in battles:
        if battle.cv_fold == holdout_fold:
            continue
        wins[battle.winner_tag] = wins.get(battle.winner_tag, 0) + 1
        losses[battle.loser_tag] = losses.get(battle.loser_tag, 0) + 1

    ordered_tags = list({tag for tag in training_tags})
    for tag in ordered_tags:
        wins.setdefault(tag, 0)
        losses.setdefault(tag, 0)

    ordered_tags.sort(key=lambda tag: (-wins[tag], losses[tag], tag))
    return ordered_tags


def _run_fold(
    fold: int,
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
) -> FoldResult:
    training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]

    product_residuals: Dict[int, int] = {record.product_id: record.encoded_path for record in training}
    tag_to_products: Dict[str, List[int]] = {}
    for record in training:
        for tag in record.tags:
            tag_to_products.setdefault(tag, []).append(record.product_id)

    tag_order = _tag_order(battles, fold, tag_to_products.keys())

    coefficients: List[TagCoefficient] = []
    for sequence, tag in enumerate(tag_order):
        values = [product_residuals[pid] for pid in tag_to_products.get(tag, [])]
        if values:
            coefficient = _select_coefficient(values, base)
            for pid in tag_to_products[tag]:
                product_residuals[pid] -= coefficient
        else:
            coefficient = 0
        coefficients.append(TagCoefficient(tag=tag, coefficient=coefficient, sequence=sequence))

    coefficient_lookup = {entry.tag: entry.coefficient for entry in coefficients}

    predictions: List[Prediction] = []
    total_loss = 0.0
    for record in testing:
        predicted = sum(coefficient_lookup.get(tag, 0) for tag in record.tags)
        loss = _p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss
        predictions.append(
            Prediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )

    return FoldResult(cv_fold=fold, coefficients=coefficients, predictions=predictions, loss=total_loss)


def _save_results(
    conn,
    schema: str,
    results: Sequence[FoldResult],
    prime_base: int,
    max_digit: int,
) -> None:
    coeff_rows: List[Tuple[int, str, int, int]] = []
    prediction_rows: List[Tuple[int, int, int, int, float]] = []
    metrics_rows: List[Tuple[int, float, int, int]] = []

    for result in results:
        metrics_rows.append((result.cv_fold, result.loss, prime_base, max_digit))
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

    with conn.cursor() as cur:
        if coeff_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_tag_coefficients (cv_fold, tag, coefficient, sequence) "
                    "VALUES (%s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                coeff_rows,
            )
        if metrics_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_fold_metrics (cv_fold, loss, prime_base, max_digit) "
                    "VALUES (%s, %s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                metrics_rows,
            )
        if prediction_rows:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.umllr_predictions (cv_fold, product_id, true_value, predicted_value, loss) "
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
) -> None:
    conn = db.get_connection(dsn)
    try:
        _ensure_storage(conn, schema)
        _truncate_outputs(conn, schema)

        fold_assignments = calculate_cv_folds(conn, product_table, n_splits=cv_splits)
        if not fold_assignments:
            return

        records, prime_base, max_digit = _load_products(conn, product_table, fold_assignments)
        if not records:
            return

        battles = _load_battles(conn, schema)
        results = [
            _run_fold(fold, records, battles, prime_base)
            for fold in range(cv_splits)
        ]

        _save_results(conn, schema, results, prime_base, max_digit)
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
    args = parser.parse_args()

    process_database(
        dsn=args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        cv_splits=args.cv_splits,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
