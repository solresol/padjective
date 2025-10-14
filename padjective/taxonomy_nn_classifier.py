"""Train neural network classifiers to predict product taxonomy from tags.

This module provides utilities for training neural network models using scikit-learn's
MLPClassifier to predict taxonomy IDs from product tags.
"""

from __future__ import annotations

import argparse
import html
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db, tag_features
else:
    from . import db, tag_features


@dataclass(slots=True)
class TrainingStats:
    """Metadata about the trained neural network classifier."""

    samples: int
    taxonomies: int
    unique_tags: int
    hidden_layers: tuple[int, ...]
    training_accuracy: float
    cross_validation_folds: int | None = None
    cross_validation_mean_accuracy: float | None = None
    cross_validation_std_accuracy: float | None = None


def load_training_data(
    conn,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 2,
    min_samples_per_taxonomy: int = 5,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], pd.DataFrame]:
    """Load product tags and taxonomy labels for training.

    Args:
        conn: Database connection
        product_table: Qualified product table name
        min_tag_count: Minimum occurrences for a tag to be included
        min_samples_per_taxonomy: Minimum samples per taxonomy to include

    Returns:
        tuple: (features, encoded_labels, feature_names, metadata)

    Note:
        ``metadata`` retains the original ``taxonomy_id`` values and includes a
        ``taxonomy_index`` column that maps each product to its encoded label.
    """
    features, metadata, feature_names = tag_features.extract_tag_features(
        conn,
        product_table=product_table,
        include_taxonomy=True,
        min_tag_count=min_tag_count,
    )

    # Filter out products without taxonomy
    has_taxonomy = metadata["taxonomy_id"].notna()
    taxonomy_mask = has_taxonomy.to_numpy(dtype=bool, copy=False)
    features = features[taxonomy_mask]
    metadata = metadata.loc[has_taxonomy].copy().reset_index(drop=True)

    if len(metadata) == 0:
        raise ValueError("No products with taxonomy classifications found")

    # Filter taxonomies by minimum sample count
    taxonomy_counts = metadata["taxonomy_id"].value_counts()
    valid_taxonomies = taxonomy_counts[taxonomy_counts >= min_samples_per_taxonomy].index
    mask = metadata["taxonomy_id"].isin(valid_taxonomies)
    taxonomy_filter = mask.to_numpy(dtype=bool, copy=False)
    features = features[taxonomy_filter]
    metadata = metadata.loc[mask].copy().reset_index(drop=True)

    if len(metadata) == 0:
        raise ValueError(
            f"No taxonomies with at least {min_samples_per_taxonomy} samples found"
        )

    # Encode taxonomy labels as integers for compatibility with scikit-learn.
    # Some taxonomy identifiers are strings/UUIDs which cause downstream
    # validation (e.g. ``np.isnan`` checks inside ``MLPClassifier``) to fail when
    # cross-validating.  Factorizing provides a dense integer representation
    # while preserving the original taxonomy values in ``metadata``.
    encoded_labels, _ = pd.factorize(metadata["taxonomy_id"], sort=True)
    metadata["taxonomy_index"] = encoded_labels
    labels = encoded_labels.astype(np.int32, copy=False)

    return features, labels, feature_names, metadata


def train_nn_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    max_iter: int = 200,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
) -> tuple[MLPClassifier, TrainingStats]:
    """Train a neural network classifier.

    Args:
        features: Sparse feature matrix (n_samples x n_features)
        labels: Target labels (n_samples,)
        hidden_layer_sizes: Tuple specifying hidden layer architecture
        max_iter: Maximum training iterations
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of data to use for validation (if early_stopping=True)

    Returns:
        tuple: (trained_model, training_stats)
    """
    if len(labels) == 0:
        raise ValueError("Cannot train on empty dataset")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 taxonomies to train")

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=10,
        verbose=False,
    )

    model.fit(features, labels)
    accuracy = float(model.score(features, labels))

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=len(unique_labels),
        unique_tags=features.shape[1],
        hidden_layers=hidden_layer_sizes,
        training_accuracy=accuracy,
    )

    return model, stats


def cross_validate_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    n_folds: int = 5,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    max_iter: int = 200,
) -> list[float]:
    """Evaluate neural network using stratified k-fold cross-validation.

    Args:
        features: Sparse feature matrix
        labels: Target labels
        n_folds: Number of cross-validation folds
        hidden_layer_sizes: Hidden layer architecture
        max_iter: Maximum iterations

    Returns:
        List of accuracy scores for each fold
    """
    if len(labels) == 0:
        return []

    # Determine maximum possible folds
    unique, counts = np.unique(labels, return_counts=True)
    max_possible_folds = int(counts.min())
    n_splits = min(n_folds, max_possible_folds)

    if n_splits < 2:
        return []

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        features,
        labels,
        cv=cv,
        scoring="accuracy",
        n_jobs=1,  # Neural networks don't parallelize well with n_jobs
    )

    return [float(score) for score in scores]


def save_model_to_database(
    database_path: Path,
    model: MLPClassifier,
    stats: TrainingStats,
    cv_scores: list[float] | None = None,
) -> int:
    """Persist model metadata to SQLite.

    Note: We save metadata and statistics, but not the full model weights
    (which can be very large). Use pickle/joblib to save the model separately.

    Args:
        database_path: Path to SQLite database
        model: Trained model
        stats: Training statistics
        cv_scores: Cross-validation scores

    Returns:
        Model ID in the database
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)

    cv_scores_list = [float(s) for s in (cv_scores or [])]
    cv_folds = len(cv_scores_list) if cv_scores_list else None
    cv_mean = float(np.mean(cv_scores_list)) if cv_scores_list else None
    cv_std = float(np.std(cv_scores_list, ddof=0)) if cv_scores_list else None

    timestamp = datetime.now(timezone.utc).isoformat()
    hidden_layers_str = ",".join(str(x) for x in stats.hidden_layers)

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Create tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_nn_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                samples INTEGER NOT NULL,
                taxonomies INTEGER NOT NULL,
                unique_tags INTEGER NOT NULL,
                hidden_layers TEXT NOT NULL,
                training_accuracy REAL NOT NULL,
                cv_folds INTEGER,
                cv_mean_accuracy REAL,
                cv_std_accuracy REAL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_nn_cv_scores (
                model_id INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES taxonomy_nn_models(id) ON DELETE CASCADE
            )
            """
        )

        # Insert model metadata
        cursor = conn.execute(
            """
            INSERT INTO taxonomy_nn_models (
                trained_at, samples, taxonomies, unique_tags, hidden_layers,
                training_accuracy, cv_folds, cv_mean_accuracy, cv_std_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                stats.samples,
                stats.taxonomies,
                stats.unique_tags,
                hidden_layers_str,
                stats.training_accuracy,
                cv_folds,
                cv_mean,
                cv_std,
            ),
        )
        model_id = int(cursor.lastrowid)

        # Insert CV scores
        if cv_scores_list:
            conn.executemany(
                """
                INSERT INTO taxonomy_nn_cv_scores (model_id, fold, accuracy)
                VALUES (?, ?, ?)
                """,
                [(model_id, i + 1, float(score)) for i, score in enumerate(cv_scores_list)],
            )

        conn.commit()

    return model_id


def render_report_html(
    stats: TrainingStats,
    output_path: Path,
) -> None:
    """Render HTML report summarizing neural network training.

    Args:
        stats: Training statistics
        output_path: Path to save HTML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intro = (
        "<p>This report summarizes a neural network classifier trained "
        "to predict product taxonomies from tags using scikit-learn's MLPClassifier.</p>"
    )

    metadata_items = [
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>",
        f"<li><strong>Taxonomies:</strong> {stats.taxonomies:,}</li>",
        f"<li><strong>Input features (tags):</strong> {stats.unique_tags:,}</li>",
        f"<li><strong>Hidden layers:</strong> {stats.hidden_layers}</li>",
        f"<li><strong>Training accuracy:</strong> {stats.training_accuracy:.3f}</li>",
    ]

    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        if stats.cross_validation_folds:
            cv_text += f" ({stats.cross_validation_folds} folds)"
        metadata_items.append(f"<li><strong>Cross-validated accuracy:</strong> {cv_text}</li>")

    metadata = "<ul class=\"stats\">" + "".join(metadata_items) + "</ul>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Taxonomy Neural Network Classifier</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin-top: 0; }}
    .stats {{ list-style: none; padding: 0; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
    .stats li {{ background: white; padding: 0.75rem 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); }}
  </style>
</head>
<body>
  <h1>Taxonomy Neural Network Classifier</h1>
  {intro}
  {metadata}
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    """Command-line interface for training neural network taxonomy classifiers."""
    parser = argparse.ArgumentParser(
        description="Train neural network to predict product taxonomy from tags"
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
        "--model-database",
        type=Path,
        default=Path("data/taxonomy_nn_classifier.sqlite"),
        help="SQLite database for model metadata storage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/taxonomy_nn_classifier"),
        help="Directory for HTML reports",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=2,
        help="Minimum tag occurrences to include",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum samples per taxonomy",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="100",
        help="Comma-separated hidden layer sizes (e.g., '100,50')",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum training iterations",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation (faster for testing)",
    )

    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_layer_sizes = tuple(int(x.strip()) for x in args.hidden_layers.split(","))

    # Load data
    conn = db.get_connection(args.dsn)
    features, labels, feature_names, metadata = load_training_data(
        conn,
        product_table=args.product_table,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
    )
    conn.close()

    print(f"Loaded {len(labels)} products with {len(feature_names)} tags")
    print(f"Architecture: {len(feature_names)} -> {hidden_layer_sizes} -> {len(np.unique(labels))}")

    # Cross-validate (optional)
    cv_scores = None
    if not args.skip_cv:
        print("\nRunning cross-validation...")
        cv_scores = cross_validate_classifier(
            features,
            labels,
            n_folds=args.cv_folds,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.max_iter,
        )

    # Train
    print("\nTraining final model...")
    model, stats = train_nn_classifier(
        features,
        labels,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
    )

    if cv_scores:
        stats.cross_validation_folds = len(cv_scores)
        stats.cross_validation_mean_accuracy = float(np.mean(cv_scores))
        stats.cross_validation_std_accuracy = float(np.std(cv_scores, ddof=0))

    # Save
    save_model_to_database(
        args.model_database,
        model,
        stats,
        cv_scores,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "nn_report.html"
    render_report_html(stats, html_path)

    print(f"\nTraining complete!")
    print(f"  Samples: {stats.samples:,}")
    print(f"  Taxonomies: {stats.taxonomies:,}")
    print(f"  Training accuracy: {stats.training_accuracy:.3f}")
    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        print(f"  Cross-validated accuracy: {cv_text} ({stats.cross_validation_folds} folds)")
    print(f"\nMetadata saved to {args.model_database}")
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
