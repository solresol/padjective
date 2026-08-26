"""Targeted follow-up experiments for the p-adic journal paper revision.

The experiments read the fixed paper snapshot directly from Postgres and write
compact, reproducible summaries back to dedicated tables in the ``padjective``
schema.  They answer three reviewer-style questions without modifying the
production model tables:

* does Zubarev annealing improve when initialised by the best greedy order;
* how does the neural baseline change with hidden-layer width; and
* does the ranking of greedy tag orders change when the depth weight ``q`` is
  varied while the base-``p`` predictions are held fixed?
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import math
import warnings
from typing import Iterable, Sequence

import numpy as np
from psycopg import sql
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from . import benchmark_runtime, data_access, db, umllr


@dataclass(frozen=True)
class PaperDataset:
    records: list[benchmark_runtime.ProductRecord]
    features: sparse.csr_matrix
    labels: np.ndarray
    encoded_by_taxonomy: dict[str, int]
    prime_base: int
    max_digit: int


def q_weighted_distance(a: int, b: int, *, prime_base: int, q: float) -> float:
    """Return ``q^-v_p(a-b)`` (and zero for equality)."""

    if q <= 1:
        raise ValueError("q must be greater than one")
    if a == b:
        return 0.0
    difference = abs(int(a) - int(b))
    valuation = 0
    while difference % prime_base == 0:
        difference //= prime_base
        valuation += 1
    return float(q ** (-valuation))


def _next_prime(value: int) -> int:
    candidate = max(2, int(value) + 1)
    while True:
        if all(candidate % divisor for divisor in range(2, int(math.sqrt(candidate)) + 1)):
            return candidate
        candidate += 1


def _load_paper_dataset(
    conn,
    *,
    snapshot_ref: str,
    schema: str,
) -> PaperDataset:
    snapshot_id, _ = data_access._resolve_snapshot_id(
        conn,
        schema=schema,
        snapshot_ref=snapshot_ref,
    )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT product_id_hash, taxonomy_id, taxonomy_path, cv_fold
                FROM {schema}.product_taxonomy_bench_products
                WHERE snapshot_id = %s
                ORDER BY product_id_hash
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        product_rows = cur.fetchall()
        cur.execute(
            sql.SQL(
                """
                SELECT tag_id
                FROM {schema}.product_taxonomy_bench_tags
                WHERE snapshot_id = %s
                ORDER BY tag_rank
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        feature_names = [str(row[0]) for row in cur.fetchall()]
        cur.execute(
            sql.SQL(
                """
                SELECT product_id_hash, tag_id, title_part, title_position
                FROM {schema}.product_taxonomy_bench_product_tags
                WHERE snapshot_id = %s
                ORDER BY product_id_hash, tag_id
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        product_tag_rows = cur.fetchall()

    if not product_rows:
        raise ValueError(f"No products found for snapshot {snapshot_ref!r}")
    feature_index = {tag: index for index, tag in enumerate(feature_names)}

    tags_by_product: dict[str, list[str]] = defaultdict(list)
    title_positions_by_product: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for product_hash, tag_id, title_part, title_position in product_tag_rows:
        product_hash = str(product_hash)
        tag_id = str(tag_id)
        tags_by_product[product_hash].append(tag_id)
        if title_part is not None and title_position is not None:
            title_positions_by_product[product_hash].append(
                (tag_id, int(title_part), int(title_position))
            )

    paths = {
        str(taxonomy_id): benchmark_runtime.parse_taxonomy_digits(str(taxonomy_path))
        for _product_hash, taxonomy_id, taxonomy_path, _cv_fold in product_rows
    }
    max_digit = max((max(path) for path in paths.values() if path), default=1)
    prime_base = _next_prime(max_digit)
    encoded_by_taxonomy = {
        taxonomy_id: benchmark_runtime.encode_path(path, prime_base)
        for taxonomy_id, path in paths.items()
    }

    rows: list[int] = []
    columns: list[int] = []
    runtime_records: list[benchmark_runtime.ProductRecord] = []
    labels: list[str] = []
    for row_index, (product_hash, taxonomy_id, taxonomy_path, cv_fold) in enumerate(product_rows):
        product_hash = str(product_hash)
        taxonomy_id = str(taxonomy_id)
        if taxonomy_id not in encoded_by_taxonomy or cv_fold is None:
            raise ValueError(f"Incomplete paper snapshot row for product {product_hash}")
        product_tags = sorted(set(tags_by_product.get(product_hash, [])))
        for tag in product_tags:
            rows.append(row_index)
            columns.append(feature_index[tag])
        labels.append(taxonomy_id)
        runtime_records.append(
            benchmark_runtime.ProductRecord(
                product_id=row_index,
                product_key=product_hash,
                tags=product_tags,
                encoded_path=int(encoded_by_taxonomy[taxonomy_id]),
                cv_fold=int(cv_fold),
                taxonomy_id=taxonomy_id,
                taxonomy_depth=len(benchmark_runtime.parse_taxonomy_digits(str(taxonomy_path))),
                title_tag_positions=tuple(
                    sorted(title_positions_by_product.get(product_hash, []), key=lambda item: (item[1], item[2], item[0]))
                ),
            )
        )

    features = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(runtime_records), len(feature_names)),
        dtype=np.float32,
    )
    return PaperDataset(
        records=runtime_records,
        features=features,
        labels=np.asarray(labels, dtype=object),
        encoded_by_taxonomy=encoded_by_taxonomy,
        prime_base=prime_base,
        max_digit=max_digit,
    )


def _ensure_storage(conn, schema: str) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(schema=sql.Identifier(schema)))
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.paper_revision_zubarev (
                    snapshot_ref TEXT NOT NULL,
                    cv_fold INTEGER NOT NULL,
                    tag_order_strategy TEXT NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    num_train INTEGER NOT NULL,
                    num_test INTEGER NOT NULL,
                    initial_mean_loss DOUBLE PRECISION NOT NULL,
                    annealed_mean_loss DOUBLE PRECISION NOT NULL,
                    initial_exact_accuracy DOUBLE PRECISION NOT NULL,
                    annealed_exact_accuracy DOUBLE PRECISION NOT NULL,
                    initial_nonzero_params INTEGER NOT NULL,
                    annealed_nonzero_params INTEGER NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS paper_revision_zubarev_run_idx
                ON {schema}.paper_revision_zubarev
                    (snapshot_ref, cv_fold, tag_order_strategy, max_iterations, seed)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.paper_revision_neural_sizes (
                    snapshot_ref TEXT NOT NULL,
                    cv_fold INTEGER NOT NULL,
                    hidden_units INTEGER NOT NULL,
                    max_iterations INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    num_train INTEGER NOT NULL,
                    num_test INTEGER NOT NULL,
                    mean_loss DOUBLE PRECISION NOT NULL,
                    exact_accuracy DOUBLE PRECISION NOT NULL,
                    num_params BIGINT NOT NULL,
                    iterations_used INTEGER NOT NULL,
                    converged BOOLEAN NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS paper_revision_neural_sizes_run_idx
                ON {schema}.paper_revision_neural_sizes
                    (snapshot_ref, cv_fold, hidden_units, max_iterations, seed)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.paper_revision_q_sensitivity (
                    snapshot_ref TEXT NOT NULL,
                    run_key TEXT NOT NULL,
                    tag_order_strategy TEXT NOT NULL,
                    tag_order_seed INTEGER,
                    q DOUBLE PRECISION NOT NULL,
                    num_products INTEGER NOT NULL,
                    mean_loss DOUBLE PRECISION NOT NULL,
                    exact_accuracy DOUBLE PRECISION NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS paper_revision_q_sensitivity_run_idx
                ON {schema}.paper_revision_q_sensitivity (snapshot_ref, run_key, q)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def _predict_with_coefficients(
    records: Sequence[benchmark_runtime.ProductRecord],
    coefficients: dict[str, int],
    default_prediction: int,
) -> list[int]:
    predictions: list[int] = []
    for record in records:
        active = [coefficients.get(tag, 0) for tag in record.tags if coefficients.get(tag, 0) != 0]
        prediction = int(sum(active))
        predictions.append(default_prediction if prediction == 0 else prediction)
    return predictions


def run_zubarev_followup(
    conn,
    dataset: PaperDataset,
    *,
    snapshot_ref: str,
    schema: str,
    max_iterations: int,
    seed: int,
) -> None:
    insert_rows: list[tuple[object, ...]] = []
    folds = sorted({record.cv_fold for record in dataset.records})
    for fold in folds:
        training = [record for record in dataset.records if record.cv_fold != fold]
        testing = [record for record in dataset.records if record.cv_fold == fold]
        initial = benchmark_runtime.umllr_run_fold(
            fold,
            dataset.records,
            [],
            dataset.prime_base,
            tag_order_strategy="taxonomy_association",
        )
        initial_coefficients = {item.tag: item.coefficient for item in initial.coefficients}
        annealed_coefficients, _, _, _ = benchmark_runtime._stochastic_optimize(
            training,
            initial_coefficients,
            dataset.prime_base,
            mahler_degree=0,
            max_iterations=max_iterations,
            seed=seed + fold,
        )
        training_values = [record.encoded_path for record in training]
        zero_sum_values = [
            record.encoded_path
            for record in training
            if sum(annealed_coefficients.get(tag, 0) for tag in record.tags) == 0
        ]
        default_prediction = benchmark_runtime._select_default_prediction(
            zero_sum_values,
            training_values,
            dataset.prime_base,
        )
        predicted = _predict_with_coefficients(testing, annealed_coefficients, default_prediction)
        true_values = [record.encoded_path for record in testing]
        annealed_losses = [
            benchmark_runtime.p_adic_distance(actual, estimate, dataset.prime_base)
            for actual, estimate in zip(true_values, predicted)
        ]
        insert_rows.append(
            (
                snapshot_ref,
                fold,
                "taxonomy_association",
                max_iterations,
                seed,
                len(training),
                len(testing),
                initial.loss / len(testing),
                float(np.mean(annealed_losses)),
                initial.exact_accuracy,
                float(np.mean(np.asarray(true_values) == np.asarray(predicted))),
                sum(item.coefficient != 0 for item in initial.coefficients),
                sum(value != 0 for value in annealed_coefficients.values()),
            )
        )
        print(
            f"Zubarev fold {fold}: {initial.loss / len(testing):.6f} -> "
            f"{np.mean(annealed_losses):.6f}"
        )

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.paper_revision_zubarev
                    (snapshot_ref, cv_fold, tag_order_strategy, max_iterations, seed,
                     num_train, num_test, initial_mean_loss, annealed_mean_loss,
                     initial_exact_accuracy, annealed_exact_accuracy,
                     initial_nonzero_params, annealed_nonzero_params)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_ref, cv_fold, tag_order_strategy, max_iterations, seed)
                DO UPDATE SET
                    num_train = EXCLUDED.num_train,
                    num_test = EXCLUDED.num_test,
                    initial_mean_loss = EXCLUDED.initial_mean_loss,
                    annealed_mean_loss = EXCLUDED.annealed_mean_loss,
                    initial_exact_accuracy = EXCLUDED.initial_exact_accuracy,
                    annealed_exact_accuracy = EXCLUDED.annealed_exact_accuracy,
                    initial_nonzero_params = EXCLUDED.initial_nonzero_params,
                    annealed_nonzero_params = EXCLUDED.annealed_nonzero_params,
                    updated_at = NOW()
                """
            ).format(schema=sql.Identifier(schema)),
            insert_rows,
        )
    conn.commit()


def run_neural_size_followup(
    conn,
    dataset: PaperDataset,
    *,
    snapshot_ref: str,
    schema: str,
    hidden_sizes: Iterable[int],
    max_iterations: int,
    seed: int,
) -> None:
    folds = np.asarray([record.cv_fold for record in dataset.records], dtype=int)
    insert_rows: list[tuple[object, ...]] = []
    for hidden_units in hidden_sizes:
        for fold in sorted(set(folds.tolist())):
            train_mask = folds != fold
            test_mask = folds == fold
            model = MLPClassifier(
                hidden_layer_sizes=(hidden_units,),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                max_iter=max_iterations,
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(dataset.features[train_mask], dataset.labels[train_mask])
            predictions = model.predict(dataset.features[test_mask])
            true_labels = dataset.labels[test_mask]
            losses = [
                benchmark_runtime.p_adic_distance(
                    dataset.encoded_by_taxonomy[str(actual)],
                    dataset.encoded_by_taxonomy[str(predicted)],
                    dataset.prime_base,
                )
                for actual, predicted in zip(true_labels, predictions)
            ]
            converged = not any(isinstance(item.message, ConvergenceWarning) for item in caught)
            parameter_count = sum(array.size for array in model.coefs_) + sum(
                array.size for array in model.intercepts_
            )
            insert_rows.append(
                (
                    snapshot_ref,
                    fold,
                    hidden_units,
                    max_iterations,
                    seed,
                    int(train_mask.sum()),
                    int(test_mask.sum()),
                    float(np.mean(losses)),
                    float(np.mean(predictions == true_labels)),
                    int(parameter_count),
                    int(model.n_iter_),
                    converged,
                )
            )
            print(
                f"Neural {hidden_units:>2} fold {fold}: loss={np.mean(losses):.6f}, "
                f"accuracy={np.mean(predictions == true_labels):.4f}"
            )

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.paper_revision_neural_sizes
                    (snapshot_ref, cv_fold, hidden_units, max_iterations, seed,
                     num_train, num_test, mean_loss, exact_accuracy, num_params,
                     iterations_used, converged)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_ref, cv_fold, hidden_units, max_iterations, seed)
                DO UPDATE SET
                    num_train = EXCLUDED.num_train,
                    num_test = EXCLUDED.num_test,
                    mean_loss = EXCLUDED.mean_loss,
                    exact_accuracy = EXCLUDED.exact_accuracy,
                    num_params = EXCLUDED.num_params,
                    iterations_used = EXCLUDED.iterations_used,
                    converged = EXCLUDED.converged,
                    updated_at = NOW()
                """
            ).format(schema=sql.Identifier(schema)),
            insert_rows,
        )
    conn.commit()


def run_q_sensitivity_followup(
    conn,
    dataset: PaperDataset,
    *,
    snapshot_ref: str,
    schema: str,
    q_values: Iterable[float],
) -> None:
    insert_rows: list[tuple[object, ...]] = []
    battles = benchmark_runtime._derive_battles(dataset.records)
    run_specs = [
        ("battle_elo", None),
        ("frequency", None),
        ("mean_title_position", None),
        ("taxonomy_association", None),
        *(("random", seed) for seed in benchmark_runtime.DEFAULT_RANDOM_SEEDS),
    ]
    folds = sorted({record.cv_fold for record in dataset.records})
    for strategy, order_seed in run_specs:
        fold_pairs: dict[int, list[tuple[int, int]]] = {}
        for fold in folds:
            result = benchmark_runtime.umllr_run_fold(
                fold,
                dataset.records,
                battles,
                dataset.prime_base,
                tag_order_strategy=strategy,
                tag_order_seed=order_seed,
            )
            fold_pairs[fold] = [
                (prediction.true_value, prediction.predicted_value)
                for prediction in result.predictions
            ]
        run_key = umllr.tag_order_run_key(
            strategy,
            order_seed,
            snapshot_ref=snapshot_ref,
        )
        exact_accuracy = float(
            np.mean(
                [
                    np.mean([actual == estimate for actual, estimate in fold_pairs[fold]])
                    for fold in folds
                ]
            )
        )
        for q in q_values:
            mean_loss = float(
                np.mean(
                    [
                        np.mean(
                            [
                                q_weighted_distance(
                                    actual,
                                    estimate,
                                    prime_base=dataset.prime_base,
                                    q=float(q),
                                )
                                for actual, estimate in fold_pairs[fold]
                            ]
                        )
                        for fold in folds
                    ]
                )
            )
            insert_rows.append(
                (
                    snapshot_ref,
                    run_key,
                    strategy,
                    order_seed,
                    float(q),
                    len(dataset.records),
                    mean_loss,
                    exact_accuracy,
                )
            )

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {schema}.paper_revision_q_sensitivity
                    (snapshot_ref, run_key, tag_order_strategy, tag_order_seed,
                     q, num_products, mean_loss, exact_accuracy)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_ref, run_key, q)
                DO UPDATE SET
                    tag_order_strategy = EXCLUDED.tag_order_strategy,
                    tag_order_seed = EXCLUDED.tag_order_seed,
                    num_products = EXCLUDED.num_products,
                    mean_loss = EXCLUDED.mean_loss,
                    exact_accuracy = EXCLUDED.exact_accuracy,
                    updated_at = NOW()
                """
            ).format(schema=sql.Identifier(schema)),
            insert_rows,
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted experiments for the paper revision.")
    parser.add_argument("--dsn")
    parser.add_argument("--schema", default="padjective")
    parser.add_argument("--snapshot-ref", default="paper")
    parser.add_argument("--hidden-sizes", default="4,8,12,24,48")
    parser.add_argument("--neural-max-iterations", type=int, default=80)
    parser.add_argument("--zubarev-max-iterations", type=int, default=2000)
    parser.add_argument("--q-values", default="2,3,5,10,71")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-zubarev", action="store_true")
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--skip-q", action="store_true")
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        dataset = _load_paper_dataset(
            conn,
            snapshot_ref=args.snapshot_ref,
            schema=args.schema,
        )
        print(
            f"Loaded {len(dataset.records)} products, {dataset.features.shape[1]} tags, "
            f"max digit {dataset.max_digit}, prime {dataset.prime_base}."
        )
        _ensure_storage(conn, args.schema)
        if not args.skip_q:
            run_q_sensitivity_followup(
                conn,
                dataset,
                snapshot_ref=args.snapshot_ref,
                schema=args.schema,
                q_values=[float(value) for value in args.q_values.split(",")],
            )
        if not args.skip_neural:
            run_neural_size_followup(
                conn,
                dataset,
                snapshot_ref=args.snapshot_ref,
                schema=args.schema,
                hidden_sizes=[int(value) for value in args.hidden_sizes.split(",")],
                max_iterations=args.neural_max_iterations,
                seed=args.seed,
            )
        if not args.skip_zubarev:
            run_zubarev_followup(
                conn,
                dataset,
                snapshot_ref=args.snapshot_ref,
                schema=args.schema,
                max_iterations=args.zubarev_max_iterations,
                seed=args.seed,
            )
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
