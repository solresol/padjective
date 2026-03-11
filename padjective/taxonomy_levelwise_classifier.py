"""Train a simple hierarchy-aware level-wise logistic taxonomy classifier."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from psycopg import sql
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db
    from padjective.cv import calculate_cv_folds
    from padjective.metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
        summarize_taxonomy_predictions,
    )
else:
    from . import data_access, db
    from .cv import calculate_cv_folds
    from .metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
        summarize_taxonomy_predictions,
    )


@dataclass(frozen=True)
class NodeModel:
    prefix: tuple[str, ...]
    classifier: LogisticRegression | None
    majority_child: str | None
    majority_leaf_taxonomy_id: str
    children: tuple[str, ...]


@dataclass(frozen=True)
class FoldResult:
    cv_fold: int
    test_accuracy: float
    test_f1: float
    test_hierarchical_loss: float
    padic_loss_total: float
    padic_loss_mean: float
    prime_base: int
    num_train_samples: int
    num_test_samples: int
    num_nodes: int
    num_classifiers: int
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float
    predictions: list[tuple[int, str, str, float]]


def _load_padic_encodings(conn, cv_fold: int, schema: str) -> tuple[dict[str, int], int]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT prime_base FROM {schema}.umllr_fold_metrics WHERE cv_fold = %s").format(
                schema=sql.Identifier(schema)
            ),
            (cv_fold,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(
                f"No umllr fold metrics found for cv_fold={cv_fold}; run umllr first."
            )
        prime_base = int(row[0])

        cur.execute(
            sql.SQL(
                "SELECT taxonomy_id, encoded_value FROM {schema}.umllr_taxonomy_encodings WHERE cv_fold = %s"
            ).format(schema=sql.Identifier(schema)),
            (cv_fold,),
        )
        encodings = {str(row[0]): int(row[1]) for row in cur.fetchall()}

    return encodings, prime_base


def _padic_valuation(value: int, base: int) -> int:
    if value == 0:
        return 10**9
    valuation = 0
    current = abs(int(value))
    while current % base == 0:
        current //= base
        valuation += 1
    return valuation


def _padic_distance(a: int, b: int, base: int) -> float:
    if a == b:
        return 0.0
    return float(base ** (-_padic_valuation(abs(int(a) - int(b)), base)))


def train_levelwise_models(
    features: sparse.csr_matrix,
    labels: Sequence[str],
    taxonomy_paths: Dict[str, Sequence[str]],
    *,
    max_iter: int = 1000,
) -> dict[tuple[str, ...], NodeModel]:
    """Train one logistic model per internal taxonomy node."""

    prefix_indices: dict[tuple[str, ...], list[int]] = defaultdict(list)
    prefix_targets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    prefix_leaf_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    prefix_child_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for idx, label in enumerate(labels):
        label_str = str(label)
        path = tuple(taxonomy_paths[label_str])
        for depth in range(len(path) + 1):
            prefix = path[:depth]
            prefix_leaf_counts[prefix][label_str] += 1
            if depth < len(path):
                prefix_indices[prefix].append(idx)
                prefix_targets[prefix].append(path[depth])
                prefix_child_counts[prefix][path[depth]] += 1

    models: dict[tuple[str, ...], NodeModel] = {}
    for prefix, leaf_counts in prefix_leaf_counts.items():
        majority_leaf = leaf_counts.most_common(1)[0][0]
        child_counts = prefix_child_counts.get(prefix, Counter())
        children = tuple(sorted(child_counts))
        majority_child = child_counts.most_common(1)[0][0] if child_counts else None
        classifier: LogisticRegression | None = None

        if len(children) > 1:
            classifier = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
            )
            classifier.fit(features[prefix_indices[prefix]], prefix_targets[prefix])

        models[prefix] = NodeModel(
            prefix=prefix,
            classifier=classifier,
            majority_child=majority_child,
            majority_leaf_taxonomy_id=majority_leaf,
            children=children,
        )

    return models


def _active_sparse_indices(sample: sparse.spmatrix) -> np.ndarray:
    csr = sample.tocsr()
    return csr.indices


def _logistic_scoring_ops(model: LogisticRegression, sample: sparse.spmatrix) -> int:
    active_indices = _active_sparse_indices(sample)
    if active_indices.size == 0:
        return 0
    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return int(np.count_nonzero(coef[:, active_indices]))


def predict_levelwise_taxonomy(
    sample: sparse.spmatrix,
    models: dict[tuple[str, ...], NodeModel],
) -> tuple[str, int]:
    """Predict a valid taxonomy leaf using the trained hierarchy."""

    if () not in models:
        raise ValueError("Level-wise model is missing the root node")

    prefix: tuple[str, ...] = ()
    scoring_ops = 0
    while True:
        node = models[prefix]
        if node.classifier is not None:
            child = str(node.classifier.predict(sample)[0])
            scoring_ops += _logistic_scoring_ops(node.classifier, sample)
        elif node.majority_child is not None:
            child = node.majority_child
        else:
            return node.majority_leaf_taxonomy_id, scoring_ops

        next_prefix = prefix + (child,)
        if next_prefix not in models:
            return node.majority_leaf_taxonomy_id, scoring_ops
        prefix = next_prefix


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    db.ensure_table(
        conn,
        schema,
        "taxonomy_levelwise_fold_results",
        columns_sql=(
            "cv_fold INTEGER PRIMARY KEY",
            "test_accuracy DOUBLE PRECISION NOT NULL",
            "test_f1 DOUBLE PRECISION NOT NULL",
            "test_hierarchical_loss DOUBLE PRECISION NOT NULL",
            "padic_loss_total DOUBLE PRECISION NOT NULL",
            "padic_loss_mean DOUBLE PRECISION NOT NULL",
            "prime_base INTEGER NOT NULL",
            "num_train_samples INTEGER NOT NULL",
            "num_test_samples INTEGER NOT NULL",
            "num_nodes INTEGER NOT NULL",
            "num_classifiers INTEGER NOT NULL",
            "exact_accuracy DOUBLE PRECISION NOT NULL",
            "prefix1_accuracy DOUBLE PRECISION NOT NULL",
            "prefix2_accuracy DOUBLE PRECISION NOT NULL",
            "mean_shared_prefix_depth DOUBLE PRECISION NOT NULL",
            "mean_scoring_ops DOUBLE PRECISION NOT NULL",
            "trained_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        ),
    )
    db.ensure_table(
        conn,
        schema,
        "taxonomy_levelwise_predictions",
        columns_sql=(
            "cv_fold INTEGER NOT NULL",
            "product_id INTEGER NOT NULL",
            "true_taxonomy_id TEXT NOT NULL",
            "predicted_taxonomy_id TEXT NOT NULL",
            "loss DOUBLE PRECISION NOT NULL",
            "PRIMARY KEY (cv_fold, product_id)",
        ),
    )


def _load_training_data(
    conn,
    *,
    product_table: str,
    min_tag_count: int,
    min_samples_per_taxonomy: int,
) -> tuple[sparse.csr_matrix, np.ndarray, pd.DataFrame]:
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )
    metadata = dataset.metadata.copy()
    if metadata.empty:
        raise ValueError("No products with taxonomy classifications found")
    labels = metadata["taxonomy_id"].astype(str).to_numpy()
    return dataset.features, labels, metadata


def _run_fold(
    *,
    fold: int,
    features: sparse.csr_matrix,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    taxonomy_paths: dict[str, Sequence[str]],
    schema: str,
    conn,
    max_iter: int,
) -> FoldResult:
    cv_folds = metadata["cv_fold"].to_numpy()
    train_mask = cv_folds != fold
    test_mask = cv_folds == fold

    X_train = features[train_mask]
    X_test = features[test_mask]
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    test_metadata = metadata[test_mask].reset_index(drop=True)

    models = train_levelwise_models(X_train, y_train, taxonomy_paths, max_iter=max_iter)
    encodings, prime_base = _load_padic_encodings(conn, fold, schema)

    predictions: list[str] = []
    scoring_ops: list[float] = []
    for row_idx in range(X_test.shape[0]):
        predicted_label, ops = predict_levelwise_taxonomy(X_test[row_idx], models)
        predictions.append(predicted_label)
        scoring_ops.append(float(ops))

    padic_losses = [
        _padic_distance(encodings.get(str(true_label), 0), encodings.get(str(pred_label), 0), prime_base)
        for true_label, pred_label in zip(y_test, predictions)
    ]
    padic_loss_total = float(sum(padic_losses))
    padic_loss_mean = padic_loss_total / len(padic_losses) if padic_losses else 0.0

    summary = summarize_taxonomy_predictions(
        y_test,
        predictions,
        taxonomy_paths,
        scoring_ops=scoring_ops,
    )

    per_prediction_rows = [
        (
            int(test_metadata.iloc[idx]["product_id"]),
            str(y_test[idx]),
            str(predictions[idx]),
            float(padic_losses[idx]),
        )
        for idx in range(len(predictions))
    ]

    num_classifiers = sum(1 for model in models.values() if model.classifier is not None)

    return FoldResult(
        cv_fold=fold,
        test_accuracy=float(np.mean(np.asarray(predictions) == y_test)) if len(y_test) else 0.0,
        test_f1=float(f1_score(y_test, predictions, average="weighted")) if len(y_test) else 0.0,
        test_hierarchical_loss=float(hierarchical_loss_score(y_test, predictions, taxonomy_paths, base=1.1)),
        padic_loss_total=padic_loss_total,
        padic_loss_mean=padic_loss_mean,
        prime_base=prime_base,
        num_train_samples=int(X_train.shape[0]),
        num_test_samples=int(X_test.shape[0]),
        num_nodes=len(models),
        num_classifiers=num_classifiers,
        exact_accuracy=summary.exact_accuracy,
        prefix1_accuracy=summary.prefix1_accuracy,
        prefix2_accuracy=summary.prefix2_accuracy,
        mean_shared_prefix_depth=summary.mean_shared_prefix_depth,
        mean_scoring_ops=summary.mean_scoring_ops or 0.0,
        predictions=per_prediction_rows,
    )


def _save_results(conn, schema: str, results: Sequence[FoldResult]) -> None:
    with conn.cursor() as cur:
        fold_ids = [result.cv_fold for result in results]
        if fold_ids:
            cur.execute(
                sql.SQL("DELETE FROM {schema}.taxonomy_levelwise_fold_results WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_ids,),
            )
            cur.execute(
                sql.SQL("DELETE FROM {schema}.taxonomy_levelwise_predictions WHERE cv_fold = ANY(%s)").format(
                    schema=sql.Identifier(schema)
                ),
                (fold_ids,),
            )
        if results:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_levelwise_fold_results
                    (cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                     padic_loss_total, padic_loss_mean, prime_base,
                     num_train_samples, num_test_samples, num_nodes, num_classifiers,
                     exact_accuracy, prefix1_accuracy, prefix2_accuracy,
                     mean_shared_prefix_depth, mean_scoring_ops)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                [
                    (
                        result.cv_fold,
                        result.test_accuracy,
                        result.test_f1,
                        result.test_hierarchical_loss,
                        result.padic_loss_total,
                        result.padic_loss_mean,
                        result.prime_base,
                        result.num_train_samples,
                        result.num_test_samples,
                        result.num_nodes,
                        result.num_classifiers,
                        result.exact_accuracy,
                        result.prefix1_accuracy,
                        result.prefix2_accuracy,
                        result.mean_shared_prefix_depth,
                        result.mean_scoring_ops,
                    )
                    for result in results
                ],
            )
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_levelwise_predictions
                    (cv_fold, product_id, true_taxonomy_id, predicted_taxonomy_id, loss)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                [
                    (result.cv_fold, product_id, true_id, predicted_id, loss)
                    for result in results
                    for product_id, true_id, predicted_id, loss in result.predictions
                ],
            )
    conn.commit()


def process_database(
    dsn: str | None,
    *,
    schema: str,
    product_table: str,
    cv_splits: int,
    min_tag_count: int,
    min_samples_per_taxonomy: int,
    max_iter: int,
) -> None:
    conn = db.get_connection(dsn)
    try:
        _ensure_storage(conn, schema)
        features, labels, metadata = _load_training_data(
            conn,
            product_table=product_table,
            min_tag_count=min_tag_count,
            min_samples_per_taxonomy=min_samples_per_taxonomy,
        )

        fold_assignments = calculate_cv_folds(conn, product_table, n_splits=cv_splits)
        metadata = metadata.copy()
        metadata["cv_fold"] = metadata["product_id"].map(fold_assignments)
        valid_mask = metadata["cv_fold"].notna().to_numpy()
        features = features[valid_mask]
        metadata = metadata[valid_mask].reset_index(drop=True)
        labels = metadata["taxonomy_id"].astype(str).to_numpy()

        taxonomy_paths = build_taxonomy_path_map(metadata)
        ensure_taxonomy_paths_cover_labels(np.unique(labels), taxonomy_paths)

        results = [
            _run_fold(
                fold=fold,
                features=features,
                labels=labels,
                metadata=metadata,
                taxonomy_paths=taxonomy_paths,
                schema=schema,
                conn=conn,
                max_iter=max_iter,
            )
            for fold in range(cv_splits)
        ]
        _save_results(conn, schema, results)
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a top-down level-wise logistic taxonomy classifier.",
    )
    parser.add_argument("--dsn", help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)")
    parser.add_argument("--schema", default="padjective", help="Schema for reading/writing results")
    parser.add_argument("--product-table", default="cantbuymelove.product", help="Qualified product table name")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--min-tag-count", type=int, default=5, help="Minimum tag frequency")
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum products per taxonomy",
    )
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum solver iterations")
    args = parser.parse_args()

    process_database(
        args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        cv_splits=args.cv_splits,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
