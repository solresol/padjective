"""Train classifiers to predict product taxonomy from tags.

This module provides utilities for training logistic regression and neural network
models that predict taxonomy IDs from product tags. The models are evaluated using
stratified cross-validation and results are stored in Postgres.
"""

from __future__ import annotations

import argparse
import html
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from psycopg import sql
from scipy import sparse
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
        # Get prime base
        cur.execute(
            sql.SQL("SELECT prime_base FROM padjective.umllr_fold_metrics WHERE cv_fold = %s"),
            (cv_fold,)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"No umllr fold metrics found for cv_fold={cv_fold}")
        prime_base = int(row[0])

        # Get encodings
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
    """Calculate p-adic loss for predictions.

    Args:
        y_true: True taxonomy IDs
        y_pred: Predicted taxonomy IDs
        encodings: Mapping from taxonomy_id to p-adic encoding
        prime_base: Prime base for p-adic distance

    Returns:
        tuple: (total_loss, per_sample_losses)
    """
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
    training_f1: float | None = None
    training_hierarchical_loss: float | None = None
    cross_validation_folds: int | None = None
    cross_validation_mean_accuracy: float | None = None
    cross_validation_std_accuracy: float | None = None
    cross_validation_mean_f1: float | None = None
    cross_validation_std_f1: float | None = None
    cross_validation_mean_hierarchical_loss: float | None = None
    cross_validation_std_hierarchical_loss: float | None = None


@dataclass(slots=True)
class CrossValidationResults:
    """Container for per-fold evaluation metrics."""

    accuracy: list[float]
    f1_weighted: list[float]
    hierarchical_loss: list[float]

    @property
    def folds(self) -> int:
        return len(self.accuracy)


def select_top_tags(
    features: sparse.csr_matrix,
    feature_names: list[str],
    max_tags: int,
) -> tuple[sparse.csr_matrix, list[str]]:
    """Select the top N most common tags and filter the feature matrix.

    Args:
        features: Sparse feature matrix (n_samples x n_features)
        feature_names: List of tag names
        max_tags: Maximum number of tags to keep

    Returns:
        tuple: (filtered_features, filtered_feature_names)
    """
    if max_tags >= len(feature_names):
        return features, feature_names

    # Count occurrences of each tag (column sums)
    tag_counts = np.array(features.sum(axis=0)).flatten()

    # Get indices of top N most common tags
    top_indices = np.argsort(tag_counts)[::-1][:max_tags]
    top_indices = np.sort(top_indices)  # Keep in original order

    # Filter features and feature names
    filtered_features = features[:, top_indices]
    filtered_feature_names = [feature_names[i] for i in top_indices]

    return filtered_features, filtered_feature_names


def load_training_data(
    conn,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 2,
    min_samples_per_taxonomy: int = 5,
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    list[str],
    pd.DataFrame,
    data_access.ProductDataset,
]:
    """Load product tags and taxonomy labels for training.

    Args:
        conn: Database connection
        product_table: Qualified product table name
        min_tag_count: Minimum occurrences for a tag to be included
        min_samples_per_taxonomy: Minimum samples per taxonomy to include

    Returns:
        tuple: (features, labels, feature_names, metadata)
    """
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

    if "taxonomy_path" not in metadata.columns:
        raise ValueError("taxonomy_path column is required in metadata")

    labels = metadata["taxonomy_id"].to_numpy()
    return dataset.features, labels, dataset.feature_names, metadata, dataset



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
    taxonomy_paths: Mapping[Any, Sequence[str]],
    *,
    n_folds: int = 5,
    max_iter: int = 1000,
    hierarchical_base: float = 1.1,
) -> CrossValidationResults:
    """Evaluate classifier using stratified k-fold cross-validation."""

    if len(labels) == 0:
        return CrossValidationResults([], [], [])

    # Determine maximum possible folds
    unique, counts = np.unique(labels, return_counts=True)
    max_possible_folds = int(counts.min())
    n_splits = min(n_folds, max_possible_folds)

    if n_splits < 2:
        return CrossValidationResults([], [], [])

    model = LogisticRegression(
        max_iter=max_iter,
        solver="lbfgs",
        class_weight="balanced",
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }

    def _hierarchical_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return hierarchical_loss_score(
            y_true,
            y_pred,
            taxonomy_paths,
            base=hierarchical_base,
        )

    scoring["hierarchical"] = make_scorer(_hierarchical_score)

    results = cross_validate(
        model,
        features,
        labels,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
    )

    return CrossValidationResults(
        accuracy=[float(score) for score in results.get("test_accuracy", [])],
        f1_weighted=[float(score) for score in results.get("test_f1_weighted", [])],
        hierarchical_loss=[float(score) for score in results.get("test_hierarchical", [])],
    )


def _expand_binary_coefficients(
    classes: np.ndarray,
    coef_matrix: np.ndarray,
    intercepts: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return coefficient and intercept arrays expanded for binary classifiers."""

    matrix = coef_matrix
    expanded_intercepts = intercepts

    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    if len(classes) == 2 and matrix.shape[0] == 1:
        coef_row = matrix[0]
        matrix = np.vstack([-coef_row, coef_row])
        if intercepts is not None:
            intercept_value = float(intercepts[0])
            expanded_intercepts = np.array([-intercept_value, intercept_value])

    return classes, matrix, expanded_intercepts


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
    _, expanded_coef, _ = _expand_binary_coefficients(classes, coef_matrix)

    abs_coef = np.abs(expanded_coef)
    max_indices = abs_coef.argmax(axis=0)
    max_values = abs_coef[max_indices, range(abs_coef.shape[1])]
    sum_values = abs_coef.sum(axis=0)

    rows = []
    for idx, tag in enumerate(feature_names):
        class_index = int(max_indices[idx])
        weight = expanded_coef[class_index, idx]
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


def compute_taxonomy_top_tags(
    feature_names: list[str],
    classes: np.ndarray,
    coef_matrix: np.ndarray,
    *,
    top_k: int = 20,
) -> pd.DataFrame:
    """Return the highest-weight tags for each taxonomy class."""

    _, expanded_coef, _ = _expand_binary_coefficients(classes, coef_matrix)

    rows: list[dict[str, Any]] = []
    for class_index, taxonomy_id in enumerate(classes):
        weights = expanded_coef[class_index]
        if top_k <= 0:
            sorted_indices: Sequence[int] = []
        else:
            sorted_indices = np.argsort(weights)[::-1][:top_k]
        for rank, feature_idx in enumerate(sorted_indices, start=1):
            rows.append(
                {
                    "taxonomy_id": str(taxonomy_id),
                    "tag": feature_names[feature_idx],
                    "weight": float(weights[feature_idx]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)


def save_model_to_database(
    conn,
    schema: str,
    model: LogisticRegression,
    stats: TrainingStats,
    summary: pd.DataFrame,
    class_distribution: pd.DataFrame,
    top_tags: pd.DataFrame,
    *,
    taxonomy_paths: Mapping[str, str] | None = None,
    cv_results: CrossValidationResults | None = None,
) -> int:
    """Persist model weights and metadata to Postgres."""

    classes = model.classes_
    coef_matrix = model.coef_
    intercepts = model.intercept_
    _, expanded_coef, expanded_intercepts = _expand_binary_coefficients(
        classes, coef_matrix, intercepts
    )

    cv_accuracy = [float(s) for s in (cv_results.accuracy if cv_results else [])]
    cv_f1 = [float(s) for s in (cv_results.f1_weighted if cv_results else [])]
    cv_hier = [float(s) for s in (cv_results.hierarchical_loss if cv_results else [])]

    cv_folds = len(cv_accuracy) if cv_accuracy else None
    cv_mean = float(np.mean(cv_accuracy)) if cv_accuracy else None
    cv_std = float(np.std(cv_accuracy, ddof=0)) if cv_accuracy else None
    cv_mean_f1 = float(np.mean(cv_f1)) if cv_f1 else None
    cv_std_f1 = float(np.std(cv_f1, ddof=0)) if cv_f1 else None
    cv_mean_hier = float(np.mean(cv_hier)) if cv_hier else None
    cv_std_hier = float(np.std(cv_hier, ddof=0)) if cv_hier else None

    trained_at = datetime.now(timezone.utc)
    taxonomy_path_lookup = taxonomy_paths or {}

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_lr_models (
                    trained_at, samples, taxonomies, unique_tags,
                    training_accuracy, training_f1, training_hierarchical_loss,
                    cv_folds, cv_mean_accuracy, cv_std_accuracy,
                    cv_mean_f1, cv_std_f1, cv_mean_hierarchical_loss, cv_std_hierarchical_loss
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """
            ).format(schema=sql.Identifier(schema)),
            (
                trained_at,
                stats.samples,
                stats.taxonomies,
                stats.unique_tags,
                stats.training_accuracy,
                float(stats.training_f1) if stats.training_f1 is not None else None,
                float(stats.training_hierarchical_loss)
                if stats.training_hierarchical_loss is not None
                else None,
                cv_folds,
                cv_mean,
                cv_std,
                cv_mean_f1,
                cv_std_f1,
                cv_mean_hier,
                cv_std_hier,
            ),
        )
        model_id = int(cur.fetchone()[0])

    if cv_accuracy:
        rows = [
            (
                model_id,
                i + 1,
                cv_accuracy[i],
                cv_f1[i] if i < len(cv_f1) else None,
                cv_hier[i] if i < len(cv_hier) else None,
            )
            for i in range(len(cv_accuracy))
        ]
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_lr_cv_scores (
                        model_id, fold, accuracy, f1_weighted, hierarchical_loss
                    ) VALUES (%s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                rows,
            )

    if expanded_intercepts is not None:
        intercept_rows = [
            (
                model_id,
                str(taxonomy_id),
                taxonomy_path_lookup.get(str(taxonomy_id)),
                float(intercept),
            )
            for taxonomy_id, intercept in zip(classes, expanded_intercepts)
        ]
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_lr_intercepts (
                        model_id, taxonomy_id, taxonomy_path, intercept
                    ) VALUES (%s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                intercept_rows,
            )

    summary_records = summary.to_dict("records")
    if summary_records:
        summary_rows = [
            (
                model_id,
                row["tag"],
                row["top_taxonomy"],
                row.get("top_taxonomy_path"),
                float(row["top_weight"]),
                float(row["max_abs_coef"]),
                float(row["sum_abs_coef"]),
            )
            for row in summary_records
        ]
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_lr_tag_summary (
                        model_id, tag, top_taxonomy_id, top_taxonomy_path,
                        top_weight, max_abs_weight, sum_abs_weight
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                summary_rows,
            )

    class_records = class_distribution.to_dict("records")
    if class_records:
        distribution_rows = [
            (
                model_id,
                row["taxonomy_id"],
                row.get("taxonomy_path"),
                int(row["sample_count"]),
                float(row["sample_fraction"]),
            )
            for row in class_records
        ]
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_lr_class_distribution (
                        model_id, taxonomy_id, taxonomy_path, sample_count, sample_fraction
                    ) VALUES (%s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                distribution_rows,
            )

    top_tag_records = top_tags.to_dict("records")
    if top_tag_records:
        top_tag_rows = [
            (
                model_id,
                row["taxonomy_id"],
                row.get("taxonomy_path"),
                row["tag"],
                float(row["weight"]),
                int(row["rank"]),
            )
            for row in top_tag_records
        ]
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_lr_top_tags (
                        model_id, taxonomy_id, taxonomy_path, tag, weight, rank
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                top_tag_rows,
            )

    conn.commit()
    return model_id


def render_coefficients_html(
    summary: pd.DataFrame,
    stats: TrainingStats,
    output_path: Path,
    *,
    top_n: int = 50,
    hierarchical_base: float = 1.1,
) -> None:
    """Render HTML report summarizing tag coefficients.

    Args:
        summary: Coefficient summary DataFrame
        stats: Training statistics
        output_path: Path to save HTML file
        top_n: Number of top tags to display
        hierarchical_base: Base used for hierarchical loss metric
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

    if stats.training_f1 is not None:
        metadata_items.append(
            f"<li><strong>Training F1 (weighted):</strong> {stats.training_f1:.3f}</li>"
        )
    if stats.training_hierarchical_loss is not None:
        metadata_items.append(
            "<li><strong>Training hierarchical loss (M={:.2f}):</strong> {:.3f}</li>".format(
                hierarchical_base,
                stats.training_hierarchical_loss,
            )
        )

    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        if stats.cross_validation_folds:
            cv_text += f" ({stats.cross_validation_folds} folds)"
        metadata_items.append(f"<li><strong>Cross-validated accuracy:</strong> {cv_text}</li>")

    if stats.cross_validation_mean_f1 is not None:
        cv_text = f"{stats.cross_validation_mean_f1:.3f}"
        if stats.cross_validation_std_f1 is not None:
            cv_text += f" ± {stats.cross_validation_std_f1:.3f}"
        metadata_items.append(
            f"<li><strong>Cross-validated F1 (weighted):</strong> {cv_text}</li>"
        )

    if stats.cross_validation_mean_hierarchical_loss is not None:
        cv_text = f"{stats.cross_validation_mean_hierarchical_loss:.3f}"
        if stats.cross_validation_std_hierarchical_loss is not None:
            cv_text += f" ± {stats.cross_validation_std_hierarchical_loss:.3f}"
        metadata_items.append(
            "<li><strong>Cross-validated hierarchical loss (M={:.2f}):</strong> {}</li>".format(
                hierarchical_base,
                cv_text,
            )
        )

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
        "--results-schema",
        default="padjective",
        help="Schema where classifier results are stored",
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
        "--max-tags",
        type=int,
        default=None,
        help="Maximum number of most common tags to use (for fair comparison with umllr)",
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
        "--fold",
        type=int,
        default=None,
        help="Train on specific CV fold (0-4). Uses umllr fold assignments. If not specified, runs full cross-validation.",
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
    parser.add_argument(
        "--top-tags-per-taxonomy",
        type=int,
        default=25,
        help="Number of highest-weight tags to store per taxonomy",
    )
    parser.add_argument(
        "--hierarchical-base",
        type=float,
        default=1.1,
        help="Base M used for hierarchical loss (loss = M^{-T})",
    )

    args = parser.parse_args()

    # Load data
    data_conn = db.get_connection(args.dsn)
    try:
        features, labels, feature_names, metadata, dataset = load_training_data(
            data_conn,
            product_table=args.product_table,
            min_tag_count=args.min_tag_count,
            min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        )
    finally:
        data_conn.close()

    print(f"Loaded {len(labels)} products with {len(feature_names)} tags")

    # Apply max-tags constraint if specified
    if args.max_tags is not None and args.max_tags < len(feature_names):
        features, feature_names = select_top_tags(features, feature_names, args.max_tags)
        print(f"Selected top {len(feature_names)} most common tags (--max-tags={args.max_tags})")

    taxonomy_paths = build_taxonomy_path_map(metadata)
    ensure_taxonomy_paths_cover_labels(np.unique(labels), taxonomy_paths)

    # Handle fold-based training if --fold is specified
    if args.fold is not None:
        # Filter data by cv_fold
        cv_fold_data = metadata["cv_fold"].to_numpy()
        has_fold = ~pd.isna(cv_fold_data)

        if not has_fold.any():
            raise ValueError("No cv_fold data found. Run umllr first to generate fold assignments.")

        # Only use products with fold assignments
        features = features[has_fold]
        labels = labels[has_fold]
        metadata = metadata[has_fold].reset_index(drop=True)
        cv_fold_data = cv_fold_data[has_fold]

        # Split train/test
        test_mask = cv_fold_data == args.fold
        train_mask = ~test_mask

        X_train, X_test = features[train_mask], features[test_mask]
        y_train, y_test = labels[train_mask], labels[test_mask]

        print(f"\nFold {args.fold}: Training on {len(y_train)} products, testing on {len(y_test)} products")

        # Train model
        model, stats = train_logistic_classifier(
            X_train,
            y_train,
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

        print(f"\nFold {args.fold} Results:")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  Test F1 (weighted): {test_f1:.4f}")
        print(f"  Test hierarchical loss (M={args.hierarchical_base:.2f}): {test_hierarchical:.4f}")
        print(f"  P-adic loss (total): {padic_loss:.4f}")
        print(f"  P-adic loss (mean): {padic_loss_mean:.6f}")
        print(f"  Prime base: {prime_base}")

        # Save fold results to database
        save_conn = db.get_connection(args.dsn)
        try:
            with save_conn.cursor() as cur:
                # Delete existing results for this fold
                cur.execute(
                    sql.SQL("DELETE FROM {schema}.taxonomy_lr_fold_results WHERE cv_fold = %s").format(
                        schema=sql.Identifier(args.results_schema)
                    ),
                    (args.fold,)
                )

                # Insert new results
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.taxonomy_lr_fold_results
                        (cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                         padic_loss_total, padic_loss_mean, prime_base,
                         num_train_samples, num_test_samples)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                    ).format(schema=sql.Identifier(args.results_schema)),
                    (args.fold, test_accuracy, test_f1, test_hierarchical,
                     padic_loss, padic_loss_mean, prime_base,
                     len(y_train), len(y_test))
                )
            save_conn.commit()
            print(f"\nResults saved to {args.results_schema}.taxonomy_lr_fold_results")
        finally:
            save_conn.close()

        # Skip the rest of the normal workflow
        return

    # Cross-validate
    cv_results = cross_validate_classifier(
        features,
        labels,
        taxonomy_paths,
        n_folds=args.cv_folds,
        max_iter=args.max_iter,
        hierarchical_base=args.hierarchical_base,
    )

    # Train
    model, stats = train_logistic_classifier(
        features,
        labels,
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

    if cv_results.folds:
        stats.cross_validation_folds = cv_results.folds
        stats.cross_validation_mean_accuracy = float(np.mean(cv_results.accuracy))
        stats.cross_validation_std_accuracy = float(np.std(cv_results.accuracy, ddof=0))
        stats.cross_validation_mean_f1 = float(np.mean(cv_results.f1_weighted))
        stats.cross_validation_std_f1 = float(np.std(cv_results.f1_weighted, ddof=0))
        stats.cross_validation_mean_hierarchical_loss = float(
            np.mean(cv_results.hierarchical_loss)
        )
        stats.cross_validation_std_hierarchical_loss = float(
            np.std(cv_results.hierarchical_loss, ddof=0)
        )

    # Compute summary artefacts
    summary = compute_tag_coefficients(feature_names, model.classes_, model.coef_)
    taxonomy_path_lookup = {
        str(taxonomy_id): " / ".join(path)
        for taxonomy_id, path in taxonomy_paths.items()
    }
    summary["top_taxonomy_path"] = summary["top_taxonomy"].map(
        lambda taxonomy_id: taxonomy_path_lookup.get(str(taxonomy_id))
    )

    class_counts = metadata["taxonomy_id"].value_counts()
    total_samples = float(len(metadata)) if len(metadata) else 1.0
    class_distribution = pd.DataFrame(
        [
            {
                "taxonomy_id": str(taxonomy_id),
                "taxonomy_path": taxonomy_path_lookup.get(str(taxonomy_id)),
                "sample_count": int(count),
                "sample_fraction": float(count) / total_samples,
            }
            for taxonomy_id, count in class_counts.items()
        ]
    )
    if not class_distribution.empty:
        class_distribution.sort_values(
            ["sample_fraction", "sample_count"], ascending=[False, False], inplace=True
        )
        class_distribution.reset_index(drop=True, inplace=True)

    top_tags = compute_taxonomy_top_tags(
        feature_names,
        model.classes_,
        model.coef_,
        top_k=max(0, args.top_tags_per_taxonomy),
    )
    if not top_tags.empty:
        top_tags["taxonomy_path"] = top_tags["taxonomy_id"].map(
            lambda taxonomy_id: taxonomy_path_lookup.get(str(taxonomy_id))
        )

    # Persist to Postgres
    results_conn = db.get_connection(args.dsn)
    try:
        model_id = save_model_to_database(
            results_conn,
            args.results_schema,
            model,
            stats,
            summary,
            class_distribution,
            top_tags,
            taxonomy_paths=taxonomy_path_lookup,
            cv_results=cv_results,
        )
    finally:
        results_conn.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "tag_coefficients.html"
    render_coefficients_html(
        summary,
        stats,
        html_path,
        top_n=args.top_n,
        hierarchical_base=args.hierarchical_base,
    )

    print(f"\nTrained on {stats.samples:,} samples covering {stats.taxonomies:,} taxonomies")
    print(f"Training accuracy: {stats.training_accuracy:.3f}")
    if stats.training_f1 is not None:
        print(f"Training F1 (weighted): {stats.training_f1:.3f}")
    if stats.training_hierarchical_loss is not None:
        print(
            "Training hierarchical loss (M={:.2f}): {:.3f}".format(
                args.hierarchical_base, stats.training_hierarchical_loss
            )
        )
    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        print(f"Cross-validated accuracy: {cv_text} ({stats.cross_validation_folds} folds)")
    if stats.cross_validation_mean_f1 is not None:
        cv_text = f"{stats.cross_validation_mean_f1:.3f}"
        if stats.cross_validation_std_f1 is not None:
            cv_text += f" ± {stats.cross_validation_std_f1:.3f}"
        print(f"Cross-validated F1 (weighted): {cv_text}")
    if stats.cross_validation_mean_hierarchical_loss is not None:
        cv_text = f"{stats.cross_validation_mean_hierarchical_loss:.3f}"
        if stats.cross_validation_std_hierarchical_loss is not None:
            cv_text += f" ± {stats.cross_validation_std_hierarchical_loss:.3f}"
        print(
            f"Cross-validated hierarchical loss (M={args.hierarchical_base:.2f}): {cv_text}"
        )
    print(
        f"\nModel saved to {args.results_schema}.taxonomy_lr_models as ID {model_id}"
    )
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
