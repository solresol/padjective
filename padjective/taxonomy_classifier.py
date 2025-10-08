"""Train classifiers to predict product taxonomy from tags.

This module provides utilities for training logistic regression and neural network
models that predict taxonomy IDs from product tags. The models are evaluated using
stratified cross-validation and results are stored in SQLite.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
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
    """Metadata about the trained classifier."""

    samples: int
    taxonomies: int
    unique_tags: int
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
        tuple: (features, labels, feature_names, metadata)
    """
    features, metadata, feature_names = tag_features.extract_tag_features(
        conn,
        product_table=product_table,
        include_taxonomy=True,
        min_tag_count=min_tag_count,
    )

    # Filter out products without taxonomy
    has_taxonomy = metadata["taxonomy_id"].notna()
    features = features[has_taxonomy]
    metadata = metadata[has_taxonomy].copy().reset_index(drop=True)

    if len(metadata) == 0:
        raise ValueError("No products with taxonomy classifications found")

    # Filter taxonomies by minimum sample count
    taxonomy_counts = metadata["taxonomy_id"].value_counts()
    valid_taxonomies = taxonomy_counts[taxonomy_counts >= min_samples_per_taxonomy].index
    mask = metadata["taxonomy_id"].isin(valid_taxonomies)
    features = features[mask]
    metadata = metadata[mask].copy().reset_index(drop=True)

    if len(metadata) == 0:
        raise ValueError(
            f"No taxonomies with at least {min_samples_per_taxonomy} samples found"
        )

    labels = metadata["taxonomy_id"].to_numpy()
    return features, labels, feature_names, metadata


def train_logistic_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    max_iter: int = 1000,
) -> tuple[LogisticRegression, TrainingStats]:
    """Train a logistic regression classifier.

    Args:
        features: Sparse feature matrix (n_samples x n_features)
        labels: Target labels (n_samples,)
        max_iter: Maximum iterations for solver

    Returns:
        tuple: (trained_model, training_stats)
    """
    if len(labels) == 0:
        raise ValueError("Cannot train on empty dataset")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 taxonomies to train")

    model = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(features, labels)
    accuracy = float(model.score(features, labels))

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=len(unique_labels),
        unique_tags=features.shape[1],
        training_accuracy=accuracy,
    )

    return model, stats


def cross_validate_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    n_folds: int = 5,
    max_iter: int = 1000,
) -> list[float]:
    """Evaluate classifier using stratified k-fold cross-validation.

    Args:
        features: Sparse feature matrix
        labels: Target labels
        n_folds: Number of cross-validation folds
        max_iter: Maximum iterations for solver

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

    model = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(
        model,
        features,
        labels,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    return [float(score) for score in scores]


def compute_tag_coefficients(
    feature_names: list[str],
    classes: np.ndarray,
    coef_matrix: np.ndarray,
) -> pd.DataFrame:
    """Create a summary table of coefficient magnitudes per tag.

    Args:
        feature_names: List of tag names
        classes: Array of class labels
        coef_matrix: Coefficient matrix (n_classes x n_features)

    Returns:
        DataFrame with tag statistics
    """
    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)

    abs_coef = np.abs(coef_matrix)
    max_indices = abs_coef.argmax(axis=0)
    max_values = abs_coef[max_indices, range(abs_coef.shape[1])]
    sum_values = abs_coef.sum(axis=0)

    rows = []
    for idx, tag in enumerate(feature_names):
        class_index = int(max_indices[idx])
        weight = coef_matrix[class_index, idx]
        rows.append({
            "tag": tag,
            "top_taxonomy": str(classes[class_index]),
            "top_weight": float(weight),
            "max_abs_coef": float(max_values[idx]),
            "sum_abs_coef": float(sum_values[idx]),
        })

    summary = pd.DataFrame(rows)
    summary.sort_values(
        ["max_abs_coef", "sum_abs_coef"],
        ascending=[False, False],
        inplace=True,
    )
    summary.reset_index(drop=True, inplace=True)
    return summary


def save_model_to_database(
    database_path: Path,
    model: LogisticRegression,
    stats: TrainingStats,
    feature_names: list[str],
    summary: pd.DataFrame,
    cv_scores: list[float] | None = None,
) -> int:
    """Persist model weights and metadata to SQLite.

    Args:
        database_path: Path to SQLite database
        model: Trained model
        stats: Training statistics
        feature_names: List of feature (tag) names
        summary: Coefficient summary DataFrame
        cv_scores: Cross-validation scores

    Returns:
        Model ID in the database
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)

    classes = model.classes_
    coef_matrix = model.coef_
    intercepts = model.intercept_

    # Handle binary case
    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)
    if len(classes) == 2 and coef_matrix.shape[0] == 1:
        coef_row = coef_matrix[0]
        intercept_value = float(intercepts[0])
        coef_matrix = np.vstack([-coef_row, coef_row])
        intercepts = np.array([-intercept_value, intercept_value])

    cv_scores_list = [float(s) for s in (cv_scores or [])]
    cv_folds = len(cv_scores_list) if cv_scores_list else None
    cv_mean = float(np.mean(cv_scores_list)) if cv_scores_list else None
    cv_std = float(np.std(cv_scores_list, ddof=0)) if cv_scores_list else None

    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # Create tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_classifier_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                samples INTEGER NOT NULL,
                taxonomies INTEGER NOT NULL,
                unique_tags INTEGER NOT NULL,
                training_accuracy REAL NOT NULL,
                cv_folds INTEGER,
                cv_mean_accuracy REAL,
                cv_std_accuracy REAL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_classifier_cv_scores (
                model_id INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES taxonomy_classifier_models(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_classifier_coefficients (
                model_id INTEGER NOT NULL,
                taxonomy_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                weight REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES taxonomy_classifier_models(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_classifier_intercepts (
                model_id INTEGER NOT NULL,
                taxonomy_id TEXT NOT NULL,
                intercept REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES taxonomy_classifier_models(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_classifier_tag_summary (
                model_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                top_taxonomy TEXT NOT NULL,
                top_weight REAL NOT NULL,
                max_abs_coef REAL NOT NULL,
                sum_abs_coef REAL NOT NULL,
                PRIMARY KEY (model_id, tag),
                FOREIGN KEY(model_id) REFERENCES taxonomy_classifier_models(id) ON DELETE CASCADE
            )
            """
        )

        # Insert model metadata
        cursor = conn.execute(
            """
            INSERT INTO taxonomy_classifier_models (
                trained_at, samples, taxonomies, unique_tags,
                training_accuracy, cv_folds, cv_mean_accuracy, cv_std_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                stats.samples,
                stats.taxonomies,
                stats.unique_tags,
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
                INSERT INTO taxonomy_classifier_cv_scores (model_id, fold, accuracy)
                VALUES (?, ?, ?)
                """,
                [(model_id, i + 1, float(score)) for i, score in enumerate(cv_scores_list)],
            )

        # Insert coefficients
        coefficient_rows = []
        for class_idx, taxonomy_id in enumerate(classes):
            for feature_idx, tag in enumerate(feature_names):
                coefficient_rows.append((
                    model_id,
                    str(taxonomy_id),
                    str(tag),
                    float(coef_matrix[class_idx, feature_idx]),
                ))

        if coefficient_rows:
            conn.executemany(
                """
                INSERT INTO taxonomy_classifier_coefficients (model_id, taxonomy_id, tag, weight)
                VALUES (?, ?, ?, ?)
                """,
                coefficient_rows,
            )

        # Insert intercepts
        intercept_rows = [
            (model_id, str(taxonomy_id), float(intercept))
            for taxonomy_id, intercept in zip(classes, intercepts)
        ]
        if intercept_rows:
            conn.executemany(
                """
                INSERT INTO taxonomy_classifier_intercepts (model_id, taxonomy_id, intercept)
                VALUES (?, ?, ?)
                """,
                intercept_rows,
            )

        # Insert tag summary
        if not summary.empty:
            conn.executemany(
                """
                INSERT INTO taxonomy_classifier_tag_summary (
                    model_id, tag, top_taxonomy, top_weight, max_abs_coef, sum_abs_coef
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        model_id,
                        str(row["tag"]),
                        str(row["top_taxonomy"]),
                        float(row["top_weight"]),
                        float(row["max_abs_coef"]),
                        float(row["sum_abs_coef"]),
                    )
                    for row in summary.to_dict(orient="records")
                ],
            )

        conn.commit()

    return model_id


def render_coefficients_html(
    summary: pd.DataFrame,
    stats: TrainingStats,
    output_path: Path,
    top_n: int = 50,
) -> None:
    """Render HTML report summarizing tag coefficients.

    Args:
        summary: Coefficient summary DataFrame
        stats: Training statistics
        output_path: Path to save HTML file
        top_n: Number of top tags to display
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intro = (
        "<p>This report summarizes a multinomial logistic regression model trained "
        "to predict product taxonomies from tags. The tables below list the tags "
        "with the strongest coefficients.</p>"
    )

    metadata_items = [
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>",
        f"<li><strong>Taxonomies:</strong> {stats.taxonomies:,}</li>",
        f"<li><strong>Unique tags:</strong> {stats.unique_tags:,}</li>",
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

    def render_table(title: str, data: pd.DataFrame) -> str:
        if data.empty:
            return f"<section><h2>{html.escape(title)}</h2><p>No data available.</p></section>"

        rows = []
        for idx, row in enumerate(data.head(top_n).to_dict(orient="records"), start=1):
            rows.append(
                f"<tr>"
                f"<td>{idx}</td>"
                f"<td>{html.escape(row['tag'])}</td>"
                f"<td>{html.escape(row['top_taxonomy'])}</td>"
                f"<td>{row['top_weight']:.4f}</td>"
                f"<td>{row['max_abs_coef']:.4f}</td>"
                f"<td>{row['sum_abs_coef']:.4f}</td>"
                f"</tr>"
            )

        return f"""
<section>
  <h2>{html.escape(title)}</h2>
  <table class="coeff-table">
    <thead>
      <tr>
        <th>Rank</th><th>Tag</th><th>Top Taxonomy</th>
        <th>Weight</th><th>Max |coef|</th><th>Sum |coef|</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</section>"""

    max_table = render_table(
        "Tags ranked by maximum absolute coefficient",
        summary.sort_values(["max_abs_coef", "sum_abs_coef"], ascending=[False, False]),
    )

    sum_table = render_table(
        "Tags ranked by sum of absolute coefficients",
        summary.sort_values(["sum_abs_coef", "max_abs_coef"], ascending=[False, False]),
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Taxonomy Tag Coefficients</title>
  <style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
    h1 {{ margin-top: 0; }}
    .stats {{ list-style: none; padding: 0; display: flex; gap: 1.5rem; flex-wrap: wrap; }}
    .stats li {{ background: white; padding: 0.75rem 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); }}
    section {{ margin-top: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); border-radius: 0.75rem; overflow: hidden; }}
    th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #0b6ce3; color: white; font-weight: 600; }}
    tr:nth-child(even) td {{ background: #f1f5f9; }}
  </style>
</head>
<body>
  <h1>Taxonomy Tag Coefficients</h1>
  {intro}
  {metadata}
  {max_table}
  {sum_table}
</body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")


def main() -> None:
    """Command-line interface for training taxonomy classifiers."""
    parser = argparse.ArgumentParser(
        description="Train logistic regression to predict product taxonomy from tags"
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
        default=Path("data/taxonomy_classifier.sqlite"),
        help="SQLite database for model storage",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/taxonomy_classifier"),
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
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum solver iterations",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top tags to display in HTML",
    )

    args = parser.parse_args()

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

    # Cross-validate
    cv_scores = cross_validate_classifier(
        features,
        labels,
        n_folds=args.cv_folds,
        max_iter=args.max_iter,
    )

    # Train
    model, stats = train_logistic_classifier(
        features,
        labels,
        max_iter=args.max_iter,
    )

    if cv_scores:
        stats.cross_validation_folds = len(cv_scores)
        stats.cross_validation_mean_accuracy = float(np.mean(cv_scores))
        stats.cross_validation_std_accuracy = float(np.std(cv_scores, ddof=0))

    # Compute summary
    summary = compute_tag_coefficients(feature_names, model.classes_, model.coef_)

    # Save
    save_model_to_database(
        args.model_database,
        model,
        stats,
        feature_names,
        summary,
        cv_scores,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "tag_coefficients.html"
    render_coefficients_html(summary, stats, html_path, top_n=args.top_n)

    print(f"\nTrained on {stats.samples:,} samples covering {stats.taxonomies:,} taxonomies")
    print(f"Training accuracy: {stats.training_accuracy:.3f}")
    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        print(f"Cross-validated accuracy: {cv_text} ({stats.cross_validation_folds} folds)")
    print(f"\nModel saved to {args.model_database}")
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
