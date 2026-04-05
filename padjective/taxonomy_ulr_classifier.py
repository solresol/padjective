"""Train unconstrained logistic regression classifier with L1 regularization.

This module provides an unconstrained logistic regression model that uses L1 (Lasso)
regularization to predict taxonomy IDs from product tags. Unlike the parameter
constrained (PCLR) model, this classifier uses ALL available tags and relies on
L1 regularization to achieve sparsity. The number of non-zero parameters is tracked
as a key metric.
"""

from __future__ import annotations

import argparse
import html
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from psycopg import sql
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db, tag_features
    from padjective.metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
    )
else:
    from . import data_access, db, tag_features
    from .metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
    )


def _load_padic_encodings(conn, cv_fold: int) -> tuple[dict[str, int], int]:
    """Load p-adic encodings for a specific CV fold.

    Returns:
        tuple: (taxonomy_id -> encoded_value mapping, prime_base)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT prime_base FROM padjective.umllr_fold_metrics WHERE cv_fold = %s"),
            (cv_fold,)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No umllr fold metrics found for cv_fold={cv_fold}")
        prime_base = int(row[0])

        cur.execute(
            sql.SQL(
                "SELECT taxonomy_id, encoded_value FROM padjective.umllr_taxonomy_encodings WHERE cv_fold = %s"
            ),
            (cv_fold,)
        )
        encodings = {row[0]: int(row[1]) for row in cur.fetchall()}

    return encodings, prime_base


def _padic_valuation(n: int, p: int) -> int:
    """Calculate p-adic valuation of n (how many times p divides n)."""
    if n == 0:
        return float('inf')
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def _padic_distance(a: int, b: int, p: int) -> float:
    """Calculate p-adic distance between two integers."""
    if a == b:
        return 0.0
    diff = abs(a - b)
    v = _padic_valuation(diff, p)
    return p ** (-v)


def calculate_padic_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    encodings: dict[str, int],
    prime_base: int,
) -> tuple[float, list[float]]:
    """Calculate p-adic loss for predictions."""
    losses = []
    for true_id, pred_id in zip(y_true, y_pred):
        true_enc = encodings.get(true_id, 0)
        pred_enc = encodings.get(pred_id, 0)
        loss = _padic_distance(true_enc, pred_enc, prime_base)
        losses.append(loss)

    return sum(losses), losses


@dataclass(slots=True)
class TrainingStats:
    """Metadata about the trained classifier."""

    samples: int
    taxonomies: int
    unique_tags: int
    training_accuracy: float
    num_nonzero_params: int = 0
    num_total_params: int = 0
    l1_alpha: float = 1.0
    training_f1: float | None = None
    training_hierarchical_loss: float | None = None


def load_training_data(
    conn,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    list[str],
    pd.DataFrame,
    data_access.ProductDataset,
]:
    """Load product tags and taxonomy labels for training."""
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
        snapshot_ref=snapshot_ref,
        snapshot_schema=snapshot_schema,
    )

    metadata = dataset.metadata.copy()
    if metadata.empty:
        raise ValueError("No products with taxonomy classifications found")

    if "taxonomy_path" not in metadata.columns:
        raise ValueError("taxonomy_path column is required in metadata")

    labels = metadata["taxonomy_id"].to_numpy()
    return dataset.features, labels, dataset.feature_names, metadata, dataset


def count_nonzero_params(model: LogisticRegression) -> tuple[int, int]:
    """Count the number of non-zero parameters in the model.

    Returns:
        tuple: (num_nonzero, num_total)
    """
    coef_matrix = model.coef_
    intercepts = model.intercept_

    # Handle binary case where coef is 1D
    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)
    if np.isscalar(intercepts):
        intercepts = np.array([intercepts])

    # Count non-zero coefficients
    nonzero_coef = np.count_nonzero(coef_matrix)
    nonzero_intercept = np.count_nonzero(intercepts)

    # Total possible parameters
    total_coef = coef_matrix.size
    total_intercept = len(intercepts)

    return nonzero_coef + nonzero_intercept, total_coef + total_intercept


def train_l1_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 2000,
) -> tuple[LogisticRegression, TrainingStats]:
    """Train an L1-regularized logistic regression classifier.

    Args:
        features: Sparse feature matrix (n_samples x n_features)
        labels: Target labels (n_samples,)
        C: Inverse regularization strength (smaller = more regularization)
        max_iter: Maximum iterations for solver

    Returns:
        tuple: (trained_model, training_stats)
    """
    if len(labels) == 0:
        raise ValueError("Cannot train on empty dataset")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 taxonomies to train")

    # Use L1 penalty with saga solver (required for L1 + multiclass)
    model = LogisticRegression(
        penalty="l1",
        C=C,
        solver="saga",
        max_iter=max_iter,
        class_weight="balanced",
        n_jobs=-1,
        tol=1e-4,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(features, labels)
    accuracy = float(model.score(features, labels))

    num_nonzero, num_total = count_nonzero_params(model)

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=len(unique_labels),
        unique_tags=features.shape[1],
        training_accuracy=accuracy,
        num_nonzero_params=num_nonzero,
        num_total_params=num_total,
        l1_alpha=1.0 / C,  # Convert C to alpha (regularization strength)
    )

    return model, stats


def main() -> None:
    """Command-line interface for training L1-regularized logistic regression."""
    parser = argparse.ArgumentParser(
        description="Train unconstrained L1-regularized logistic regression to predict product taxonomy from tags"
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table name",
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
    parser.add_argument(
        "--results-schema",
        default="padjective",
        help="Schema where classifier results are stored",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/unconstrained_logistic_regression"),
        help="Directory for HTML reports",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=5,
        help="Minimum tag occurrences to include",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum samples per taxonomy",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train on specific CV fold (0-4). Uses umllr fold assignments.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength (smaller = more L1 regularization)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum solver iterations",
    )
    parser.add_argument(
        "--hierarchical-base",
        type=float,
        default=1.1,
        help="Base M used for hierarchical loss (loss = M^{-T})",
    )

    args = parser.parse_args()

    # Load data - NO max-tags constraint
    data_conn = db.get_connection(args.dsn)
    try:
        features, labels, feature_names, metadata, dataset = load_training_data(
            data_conn,
            product_table=args.product_table,
            min_tag_count=args.min_tag_count,
            min_samples_per_taxonomy=args.min_samples_per_taxonomy,
            snapshot_ref=args.snapshot_ref,
            snapshot_schema=args.snapshot_schema,
        )
    finally:
        data_conn.close()

    print(f"Loaded {len(labels)} products with {len(feature_names)} tags (using ALL tags)")

    taxonomy_paths = build_taxonomy_path_map(metadata)
    ensure_taxonomy_paths_cover_labels(np.unique(labels), taxonomy_paths)

    # Handle fold-based training if --fold is specified
    if args.fold is not None:
        cv_fold_data = metadata["cv_fold"].to_numpy()
        has_fold = ~pd.isna(cv_fold_data)

        if not has_fold.any():
            raise ValueError("No cv_fold data found. Run umllr first to generate fold assignments.")

        features = features[has_fold]
        labels = labels[has_fold]
        metadata = metadata[has_fold].reset_index(drop=True)
        cv_fold_data = cv_fold_data[has_fold]

        test_mask = cv_fold_data == args.fold
        train_mask = ~test_mask

        X_train, X_test = features[train_mask], features[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        print(f"\nFold {args.fold}: Training on {len(y_train)} products, testing on {len(y_test)} products")

        # Train model with L1 regularization
        model, stats = train_l1_classifier(
            X_train,
            y_train,
            C=args.C,
            max_iter=args.max_iter,
        )

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        test_accuracy = float(np.mean(y_pred == y_test))
        test_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        test_hierarchical = float(hierarchical_loss_score(
            y_test,
            y_pred,
            taxonomy_paths,
            base=args.hierarchical_base,
        ))

        # Calculate p-adic loss
        padic_conn = db.get_connection(args.dsn)
        try:
            encodings, prime_base = _load_padic_encodings(padic_conn, args.fold)
            padic_loss, _ = calculate_padic_loss(y_test, y_pred, encodings, prime_base)
            padic_loss_mean = padic_loss / len(y_test) if len(y_test) > 0 else 0.0
        finally:
            padic_conn.close()

        # Recalculate non-zero params for test model
        num_nonzero, num_total = count_nonzero_params(model)
        sparsity_pct = (1 - num_nonzero / num_total) * 100 if num_total > 0 else 0

        print(f"\nFold {args.fold} Results:")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test F1 (weighted): {test_f1:.4f}")
        print(f"  Test hierarchical loss (M={args.hierarchical_base:.2f}): {test_hierarchical:.4f}")
        print(f"  P-adic loss (total): {padic_loss:.4f}")
        print(f"  P-adic loss (mean): {padic_loss_mean:.6f}")
        print(f"  Prime base: {prime_base}")
        print(f"  Non-zero parameters: {num_nonzero:,} / {num_total:,} ({sparsity_pct:.1f}% sparse)")
        print(f"  L1 regularization (C={args.C}, alpha={1/args.C:.4f})")

        # Save fold results to database
        save_conn = db.get_connection(args.dsn)
        try:
            with save_conn.cursor() as cur:
                # Override default tablespace temporarily to avoid permission issues
                cur.execute("SET LOCAL default_tablespace = ''")

                # Create tables if they don't exist
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.taxonomy_ulr_fold_results (
                            cv_fold INTEGER PRIMARY KEY,
                            test_accuracy DOUBLE PRECISION NOT NULL,
                            test_f1 DOUBLE PRECISION NOT NULL,
                            test_hierarchical_loss DOUBLE PRECISION NOT NULL,
                            padic_loss_total DOUBLE PRECISION NOT NULL,
                            padic_loss_mean DOUBLE PRECISION NOT NULL,
                            prime_base INTEGER NOT NULL,
                            num_train_samples INTEGER NOT NULL,
                            num_test_samples INTEGER NOT NULL,
                            num_tags INTEGER NOT NULL,
                            num_nonzero_params INTEGER NOT NULL,
                            num_total_params INTEGER NOT NULL,
                            l1_C DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                            trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    ).format(schema=sql.Identifier(args.results_schema))
                )

                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {schema}.taxonomy_ulr_predictions (
                            cv_fold INTEGER NOT NULL,
                            product_id INTEGER NOT NULL,
                            true_taxonomy_id TEXT NOT NULL,
                            predicted_taxonomy_id TEXT NOT NULL,
                            loss DOUBLE PRECISION NOT NULL,
                            PRIMARY KEY (cv_fold, product_id)
                        )
                        """
                    ).format(schema=sql.Identifier(args.results_schema))
                )

                # Delete existing results for this fold
                cur.execute(
                    sql.SQL("DELETE FROM {schema}.taxonomy_ulr_fold_results WHERE cv_fold = %s").format(
                        schema=sql.Identifier(args.results_schema)
                    ),
                    (args.fold,)
                )

                # Insert new results
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.taxonomy_ulr_fold_results
                        (cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                         padic_loss_total, padic_loss_mean, prime_base,
                         num_train_samples, num_test_samples, num_tags,
                         num_nonzero_params, num_total_params, l1_C)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    ).format(schema=sql.Identifier(args.results_schema)),
                    (args.fold, test_accuracy, test_f1, test_hierarchical,
                     padic_loss, padic_loss_mean, prime_base,
                     len(y_train), len(y_test), len(feature_names),
                     num_nonzero, num_total, args.C)
                )

                # Save individual predictions
                test_metadata = metadata[test_mask].reset_index(drop=True)
                cur.execute(
                    sql.SQL("DELETE FROM {schema}.taxonomy_ulr_predictions WHERE cv_fold = %s").format(
                        schema=sql.Identifier(args.results_schema)
                    ),
                    (args.fold,)
                )

                for idx in range(len(y_test)):
                    product_id = int(test_metadata.iloc[idx]["product_id"])
                    true_tax_id = y_test[idx]
                    pred_tax_id = y_pred[idx]

                    true_encoding = encodings.get(true_tax_id, 0)
                    pred_encoding = encodings.get(pred_tax_id, 0)
                    ind_loss = _padic_distance(true_encoding, pred_encoding, prime_base)

                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {schema}.taxonomy_ulr_predictions
                            (cv_fold, product_id, true_taxonomy_id, predicted_taxonomy_id, loss)
                            VALUES (%s, %s, %s, %s, %s)
                            """
                        ).format(schema=sql.Identifier(args.results_schema)),
                        (args.fold, product_id, true_tax_id, pred_tax_id, ind_loss)
                    )

            save_conn.commit()
            print(f"\nResults saved to {args.results_schema}.taxonomy_ulr_fold_results")
        finally:
            save_conn.close()

        return

    # If no fold specified, train on full dataset
    print("\nTraining on full dataset...")
    model, stats = train_l1_classifier(
        features,
        labels,
        C=args.C,
        max_iter=args.max_iter,
    )

    train_predictions = model.predict(features)
    stats.training_f1 = float(f1_score(labels, train_predictions, average="weighted"))
    stats.training_hierarchical_loss = float(
        hierarchical_loss_score(
            labels,
            train_predictions,
            taxonomy_paths,
            base=args.hierarchical_base,
        )
    )

    sparsity_pct = (1 - stats.num_nonzero_params / stats.num_total_params) * 100 if stats.num_total_params > 0 else 0

    print(f"\nTrained on {stats.samples:,} samples covering {stats.taxonomies:,} taxonomies")
    print(f"Using {stats.unique_tags:,} tags (ALL available)")
    print(f"Training accuracy: {stats.training_accuracy:.3f}")
    if stats.training_f1 is not None:
        print(f"Training F1 (weighted): {stats.training_f1:.3f}")
    if stats.training_hierarchical_loss is not None:
        print(f"Training hierarchical loss (M={args.hierarchical_base:.2f}): {stats.training_hierarchical_loss:.3f}")
    print(f"Non-zero parameters: {stats.num_nonzero_params:,} / {stats.num_total_params:,} ({sparsity_pct:.1f}% sparse)")
    print(f"L1 regularization (C={args.C}, alpha={stats.l1_alpha:.4f})")


if __name__ == "__main__":
    main()
