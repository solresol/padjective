"""Build reproducible, anonymized examples requested in the paper review.

The examples are derived directly from the fixed Postgres paper snapshot.  The
module finds a four-product coordinate update, samples three-product feature
cycles, and contrasts two tags whose frequency and taxonomy-association ranks
pull strongly in opposite directions.  Results are stored as JSON in the
``padjective`` schema so that prose examples can be audited later.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
import random
from typing import Sequence

from psycopg import sql
from psycopg.types.json import Jsonb

from . import benchmark_runtime, db
from .paper_revision_experiments import PaperDataset, _load_paper_dataset


DEFAULT_SCHEMA = "padjective"
DEFAULT_SNAPSHOT_REF = "paper"
DEFAULT_FOLD = 0
DEFAULT_CYCLE_DRAWS = 100_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class CycleWitness:
    product_keys: tuple[str, str, str]
    taxonomy_ids: tuple[str, str, str]
    cycle_tags: tuple[str, str, str]
    incidence: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class CycleSample:
    draws: int
    valid_chains: int
    closed_chains: int
    closure_rate: float
    eligible_middle_products: int
    witness: CycleWitness | None


@dataclass(frozen=True)
class CoordinateExample:
    cv_fold: int
    sequence: int
    tag: str
    product_keys: tuple[str, str, str, str]
    target_digits: tuple[tuple[int, ...], ...]
    residual_digits_before: tuple[tuple[int, ...], ...]
    coefficient_digits: tuple[int, ...]
    residual_digits_after: tuple[tuple[int, ...], ...]
    differing_valuation: int


@dataclass(frozen=True)
class OrderTagExample:
    tag: str
    fitting_count: int
    taxonomy_count: int
    association_strength: float
    taxonomy_association_rank: int
    frequency_rank: int
    battle_elo_rank: int
    mean_title_position_rank: int


def valuation(value: int, *, prime_base: int) -> int:
    """Return the finite p-adic valuation of a nonzero integer."""

    if value == 0:
        raise ValueError("The valuation of zero is not finite")
    value = abs(int(value))
    result = 0
    while value % prime_base == 0:
        value //= prime_base
        result += 1
    return result


def decode_base_digits(value: int, *, prime_base: int) -> tuple[int, ...]:
    """Return least-significant-first base-p digits for a nonnegative value."""

    if value < 0:
        raise ValueError("Digit display expects a nonnegative value")
    if value == 0:
        return (0,)
    digits: list[int] = []
    remaining = int(value)
    while remaining:
        digits.append(remaining % prime_base)
        remaining //= prime_base
    return tuple(digits)


def sample_three_product_cycles(
    records: Sequence[benchmark_runtime.ProductRecord],
    *,
    draws: int = DEFAULT_CYCLE_DRAWS,
    seed: int = DEFAULT_SEED,
) -> CycleSample:
    """Estimate closure among connected three-product chains.

    A draw chooses an eligible middle product, two recurrent tags on that
    product, and one distinct neighbouring product through each tag.  The chain
    closes when the two endpoint products share a third tag.  This is a
    chain-conditioned rate, not the fraction of all unordered product triples.
    """

    if draws <= 0:
        raise ValueError("draws must be positive")
    by_id = {record.product_id: record for record in records}
    tag_to_products: dict[str, list[int]] = defaultdict(list)
    for record in records:
        for tag in record.tags:
            tag_to_products[tag].append(record.product_id)
    eligible_middle = [
        record
        for record in records
        if len(
            [tag for tag in record.tags if len(tag_to_products[tag]) >= 2]
        )
        >= 2
    ]
    if not eligible_middle:
        raise ValueError("No product has two recurrent tags")

    rng = random.Random(seed)
    valid_chains = 0
    closed_chains = 0
    witness: CycleWitness | None = None
    for _ in range(draws):
        middle = rng.choice(eligible_middle)
        eligible_tags = [
            tag for tag in middle.tags if len(tag_to_products[tag]) >= 2
        ]
        left_tag, right_tag = rng.sample(eligible_tags, 2)
        left_id = rng.choice(
            [
                product_id
                for product_id in tag_to_products[left_tag]
                if product_id != middle.product_id
            ]
        )
        right_candidates = [
            product_id
            for product_id in tag_to_products[right_tag]
            if product_id not in {middle.product_id, left_id}
        ]
        if not right_candidates:
            continue
        right_id = rng.choice(right_candidates)
        valid_chains += 1
        closing_tags = sorted(
            (set(by_id[left_id].tags) & set(by_id[right_id].tags))
            - {left_tag, right_tag}
        )
        if not closing_tags:
            continue
        closed_chains += 1
        if witness is None:
            closing_tag = closing_tags[0]
            product_keys = (
                by_id[left_id].product_key,
                middle.product_key,
                by_id[right_id].product_key,
            )
            witness = CycleWitness(
                product_keys=product_keys,
                taxonomy_ids=(
                    by_id[left_id].taxonomy_id,
                    middle.taxonomy_id,
                    by_id[right_id].taxonomy_id,
                ),
                cycle_tags=(left_tag, right_tag, closing_tag),
                incidence={
                    product_keys[0]: (left_tag, closing_tag),
                    product_keys[1]: (left_tag, right_tag),
                    product_keys[2]: (right_tag, closing_tag),
                },
            )

    if valid_chains == 0:
        raise ValueError("No draw produced three distinct connected products")
    return CycleSample(
        draws=draws,
        valid_chains=valid_chains,
        closed_chains=closed_chains,
        closure_rate=closed_chains / valid_chains,
        eligible_middle_products=len(eligible_middle),
        witness=witness,
    )


def find_coordinate_example(
    dataset: PaperDataset,
    *,
    cv_fold: int = DEFAULT_FOLD,
) -> CoordinateExample:
    """Find a four-product, two-residual update with visible digit refinement."""

    records = dataset.records
    training = [record for record in records if record.cv_fold != cv_fold]
    by_id = {record.product_id: record for record in records}
    battles = benchmark_runtime._derive_battles(records)
    order = benchmark_runtime._tag_order(
        training,
        battles,
        cv_fold,
        strategy="taxonomy_association",
    )
    tag_to_products: dict[str, list[int]] = defaultdict(list)
    for record in training:
        for tag in record.tags:
            tag_to_products[tag].append(record.product_id)
    residuals = {
        record.product_id: record.encoded_path for record in training
    }
    target_values = {record.encoded_path for record in training}

    for sequence, tag in enumerate(order):
        product_ids = tag_to_products[tag]
        values = [residuals[product_id] for product_id in product_ids]
        coefficient = (
            benchmark_runtime._select_coefficient(values, dataset.prime_base)
            if values
            else 0
        )
        value_counts = Counter(values)
        unique_values = sorted(value_counts)
        is_example = (
            len(product_ids) == 4
            and len(unique_values) == 2
            and all(value > 0 for value in unique_values)
            and all(value in target_values for value in unique_values)
            and coefficient == unique_values[0]
            and value_counts[coefficient] == 3
            and valuation(
                unique_values[1] - unique_values[0],
                prime_base=dataset.prime_base,
            )
            >= 2
        )
        if is_example:
            after = [value - coefficient for value in values]
            return CoordinateExample(
                cv_fold=cv_fold,
                sequence=sequence,
                tag=tag,
                product_keys=tuple(
                    by_id[product_id].product_key for product_id in product_ids
                ),
                target_digits=tuple(
                    decode_base_digits(
                        by_id[product_id].encoded_path,
                        prime_base=dataset.prime_base,
                    )
                    for product_id in product_ids
                ),
                residual_digits_before=tuple(
                    decode_base_digits(value, prime_base=dataset.prime_base)
                    for value in values
                ),
                coefficient_digits=decode_base_digits(
                    coefficient,
                    prime_base=dataset.prime_base,
                ),
                residual_digits_after=tuple(
                    decode_base_digits(value, prime_base=dataset.prime_base)
                    for value in after
                ),
                differing_valuation=valuation(
                    unique_values[1] - unique_values[0],
                    prime_base=dataset.prime_base,
                ),
            )
        for product_id in product_ids:
            residuals[product_id] -= coefficient
    raise ValueError("No suitable four-product coordinate example was found")


def find_order_contrast(
    dataset: PaperDataset,
    *,
    cv_fold: int = DEFAULT_FOLD,
) -> tuple[OrderTagExample, OrderTagExample]:
    """Return concentrated and frequent tags with sharply different ranks."""

    training = [record for record in dataset.records if record.cv_fold != cv_fold]
    battles = benchmark_runtime._derive_battles(dataset.records)
    orders = {
        strategy: benchmark_runtime._tag_order(
            training,
            battles,
            cv_fold,
            strategy=strategy,
        )
        for strategy in benchmark_runtime.DEFAULT_ABLATION_STRATEGIES
    }
    ranks = {
        strategy: {tag: rank + 1 for rank, tag in enumerate(order)}
        for strategy, order in orders.items()
    }
    tag_counts = Counter(tag for record in training for tag in record.tags)
    taxonomy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in training:
        for tag in record.tags:
            taxonomy_counts[tag][record.taxonomy_id] += 1
    strengths = {
        tag: max(counts.values()) / sum(counts.values())
        for tag, counts in taxonomy_counts.items()
    }

    concentrated_tag = max(
        orders["taxonomy_association"][:100],
        key=lambda tag: ranks["frequency"][tag],
    )
    frequent_tag = max(
        orders["frequency"][:20],
        key=lambda tag: ranks["taxonomy_association"][tag],
    )

    def build(tag: str) -> OrderTagExample:
        return OrderTagExample(
            tag=tag,
            fitting_count=tag_counts[tag],
            taxonomy_count=len(taxonomy_counts[tag]),
            association_strength=strengths[tag],
            taxonomy_association_rank=ranks["taxonomy_association"][tag],
            frequency_rank=ranks["frequency"][tag],
            battle_elo_rank=ranks["battle_elo"][tag],
            mean_title_position_rank=ranks["mean_title_position"][tag],
        )

    return build(concentrated_tag), build(frequent_tag)


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.paper_revision_examples (
                    snapshot_ref TEXT NOT NULL,
                    analysis_key TEXT NOT NULL,
                    analysis_version INTEGER NOT NULL,
                    result JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS
                    paper_revision_examples_key_idx
                ON {schema}.paper_revision_examples
                    (snapshot_ref, analysis_key, analysis_version)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def _persist_results(
    conn,
    *,
    schema: str,
    snapshot_ref: str,
    results: dict[str, object],
) -> None:
    rows = [
        (snapshot_ref, key, 1, Jsonb(value))
        for key, value in results.items()
    ]
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.paper_revision_examples
                    (snapshot_ref, analysis_key, analysis_version, result)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (snapshot_ref, analysis_key, analysis_version)
                DO UPDATE SET result = EXCLUDED.result, updated_at = NOW()
                """
            ).format(schema=sql.Identifier(schema)),
            rows,
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate anonymized examples for the Padjective paper revision."
    )
    parser.add_argument("--dsn")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--snapshot-ref", default=DEFAULT_SNAPSHOT_REF)
    parser.add_argument("--fold", type=int, default=DEFAULT_FOLD)
    parser.add_argument("--cycle-draws", type=int, default=DEFAULT_CYCLE_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        dataset = _load_paper_dataset(
            conn,
            snapshot_ref=args.snapshot_ref,
            schema=args.schema,
        )
        cycle_sample = sample_three_product_cycles(
            dataset.records,
            draws=args.cycle_draws,
            seed=args.seed,
        )
        coordinate = find_coordinate_example(dataset, cv_fold=args.fold)
        concentrated, frequent = find_order_contrast(
            dataset,
            cv_fold=args.fold,
        )
        results = {
            "three_product_cycle_sample": asdict(cycle_sample),
            "four_product_coordinate": asdict(coordinate),
            "tag_order_contrast": {
                "cv_fold": args.fold,
                "concentrated": asdict(concentrated),
                "frequent": asdict(frequent),
            },
        }
        _ensure_storage(conn, args.schema)
        _persist_results(
            conn,
            schema=args.schema,
            snapshot_ref=args.snapshot_ref,
            results=results,
        )
        print(json.dumps(results, indent=2, sort_keys=True))
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
