"""Train neural network classifiers to predict product taxonomy from tags.

This module provides utilities for training neural network models using PyTorch to
predict taxonomy IDs from product tags.
"""

from __future__ import annotations

import argparse
import html
import sqlite3
import sys
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from typing import Mapping, Sequence

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

try:
    from psycopg2 import sql
except ImportError:
    try:
        from psycopg import sql
    except ImportError:
        sql = None


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
        encodings = {str(row[0]): int(row[1]) for row in cur.fetchall()}

    return encodings, prime_base


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
    """Metadata about the trained neural network classifier."""

    samples: int
    taxonomies: int
    unique_tags: int
    hidden_layers: tuple[int, ...]
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
    """Container for neural network cross-validation metrics."""

    accuracy: list[float]
    f1_weighted: list[float]
    hierarchical_loss: list[float]

    @property
    def folds(self) -> int:
        return len(self.accuracy)


def _build_model(
    input_dim: int, hidden_layer_sizes: tuple[int, ...], output_dim: int
) -> nn.Module:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for size in hidden_layer_sizes:
        layers.append(nn.Linear(previous_dim, size))
        layers.append(nn.ReLU())
        previous_dim = size
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def _to_tensor_dataset(features: sparse.csr_matrix, labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(features.astype(np.float32, copy=False).toarray())
    y = torch.from_numpy(labels.astype(np.int64, copy=False))
    return TensorDataset(x, y)


def _collect_predictions(model: nn.Module, dataset: Dataset) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            predictions = logits.argmax(dim=1)
            all_targets.append(batch_y.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    y_true = np.concatenate(all_targets) if all_targets else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    return y_true, y_pred


def _evaluate_accuracy(model: nn.Module, dataset: Dataset) -> float:
    if len(dataset) == 0:
        return 0.0

    y_true, y_pred = _collect_predictions(model, dataset)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def load_training_data(
    conn,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 5,
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
        tuple: (features, encoded_labels, feature_names, metadata)

    Note:
        ``metadata`` retains the original ``taxonomy_id`` values and includes a
        ``taxonomy_index`` column that maps each product to its encoded label.
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

    # Encode taxonomy labels as integers for compatibility with PyTorch loss
    # functions. Some taxonomy identifiers are strings/UUIDs which can cause
    # downstream validation to fail when numeric operations are expected.
    # Factorizing provides a dense integer representation while preserving the
    # original taxonomy values in ``metadata``.
    encoded_labels, _ = pd.factorize(metadata["taxonomy_id"], sort=True)
    metadata["taxonomy_index"] = encoded_labels
    labels = encoded_labels.astype(np.int32, copy=False)

    return dataset.features, labels, dataset.feature_names, metadata, dataset


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


def train_nn_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    max_iter: int = 200,
    early_stopping: bool = True,
    validation_fraction: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    patience: int = 10,
    num_classes: int | None = None,
) -> tuple[nn.Module, TrainingStats]:
    """Train a neural network classifier using PyTorch.

    Args:
        num_classes: Total number of output classes. If None, inferred from unique labels in training data.
                     Should be set to the total number of classes in the full dataset when doing CV splits.
    """

    if len(labels) == 0:
        raise ValueError("Cannot train on empty dataset")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 taxonomies to train")

    # Use provided num_classes or infer from training data
    output_dim = num_classes if num_classes is not None else len(unique_labels)
    if output_dim < len(unique_labels):
        raise ValueError(f"num_classes ({output_dim}) must be >= unique labels in training data ({len(unique_labels)})")

    dataset = _to_tensor_dataset(features, labels)

    if early_stopping and 0 < validation_fraction < 1:
        val_size = max(1, int(len(dataset) * validation_fraction))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Validation fraction too large for dataset size")
        generator = torch.Generator().manual_seed(42)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset = dataset
        val_dataset = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(features.shape[1], hidden_layer_sizes, output_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = (
        DataLoader(val_dataset, batch_size=batch_size) if val_dataset is not None else None
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(max_iter):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    logits = model(batch_x)
                    loss = criterion(logits, batch_y)
                    val_loss += loss.item()
                    val_batches += 1
            average_val_loss = val_loss / max(val_batches, 1)

            if average_val_loss < best_val_loss - 1e-4:
                best_val_loss = average_val_loss
                best_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping and epochs_without_improvement >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    accuracy = _evaluate_accuracy(model, dataset)

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=output_dim,
        unique_tags=features.shape[1],
        hidden_layers=hidden_layer_sizes,
        training_accuracy=accuracy,
    )

    return model, stats


def cross_validate_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    taxonomy_paths: Mapping[int, Sequence[str]],
    *,
    n_folds: int = 5,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    max_iter: int = 200,
    hierarchical_base: float = 1.1,
) -> CrossValidationResults:
    """Evaluate neural network using stratified k-fold cross-validation."""

    if len(labels) == 0:
        return CrossValidationResults([], [], [])

    unique, counts = np.unique(labels, return_counts=True)
    max_possible_folds = int(counts.min())
    n_splits = min(n_folds, max_possible_folds)

    if n_splits < 2:
        return CrossValidationResults([], [], [])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies: list[float] = []
    f1_scores: list[float] = []
    hierarchical_scores: list[float] = []
    placeholder = np.zeros((len(labels), 1), dtype=np.float32)

    # Get total number of classes from the full dataset
    num_classes = len(unique)

    for train_index, test_index in cv.split(placeholder, labels):
        train_features = features[train_index]
        train_labels = labels[train_index]
        test_dataset = _to_tensor_dataset(features[test_index], labels[test_index])

        model, _ = train_nn_classifier(
            train_features,
            train_labels,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            early_stopping=True,
            num_classes=num_classes,
        )

        y_true, y_pred = _collect_predictions(model, test_dataset)
        if y_true.size == 0:
            continue

        accuracies.append(float(np.mean(y_true == y_pred)))
        f1_scores.append(float(f1_score(y_true, y_pred, average="weighted")))
        hierarchical_scores.append(
            float(
                hierarchical_loss_score(
                    y_true,
                    y_pred,
                    taxonomy_paths,
                    base=hierarchical_base,
                )
            )
        )

    return CrossValidationResults(accuracies, f1_scores, hierarchical_scores)


def save_model_to_database(
    database_path: Path,
    model: nn.Module,
    stats: TrainingStats,
    cv_results: CrossValidationResults | None = None,
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
                training_f1 REAL,
                training_hierarchical_loss REAL,
                cv_folds INTEGER,
                cv_mean_accuracy REAL,
                cv_std_accuracy REAL,
                cv_mean_f1 REAL,
                cv_std_f1 REAL,
                cv_mean_hierarchical_loss REAL,
                cv_std_hierarchical_loss REAL
            )
            """
        )

        info_models = conn.execute("PRAGMA table_info(taxonomy_nn_models)").fetchall()
        existing_columns = {row[1] for row in info_models}
        if "training_f1" not in existing_columns:
            conn.execute("ALTER TABLE taxonomy_nn_models ADD COLUMN training_f1 REAL")
        if "training_hierarchical_loss" not in existing_columns:
            conn.execute(
                "ALTER TABLE taxonomy_nn_models ADD COLUMN training_hierarchical_loss REAL"
            )
        if "cv_mean_f1" not in existing_columns:
            conn.execute("ALTER TABLE taxonomy_nn_models ADD COLUMN cv_mean_f1 REAL")
        if "cv_std_f1" not in existing_columns:
            conn.execute("ALTER TABLE taxonomy_nn_models ADD COLUMN cv_std_f1 REAL")
        if "cv_mean_hierarchical_loss" not in existing_columns:
            conn.execute(
                "ALTER TABLE taxonomy_nn_models ADD COLUMN cv_mean_hierarchical_loss REAL"
            )
        if "cv_std_hierarchical_loss" not in existing_columns:
            conn.execute(
                "ALTER TABLE taxonomy_nn_models ADD COLUMN cv_std_hierarchical_loss REAL"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS taxonomy_nn_cv_scores (
                model_id INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                accuracy REAL,
                f1_weighted REAL,
                hierarchical_loss REAL,
                FOREIGN KEY(model_id) REFERENCES taxonomy_nn_models(id) ON DELETE CASCADE
            )
            """
        )

        info_scores = conn.execute("PRAGMA table_info(taxonomy_nn_cv_scores)").fetchall()
        existing_score_columns = {row[1] for row in info_scores}
        if "f1_weighted" not in existing_score_columns:
            conn.execute("ALTER TABLE taxonomy_nn_cv_scores ADD COLUMN f1_weighted REAL")
        if "hierarchical_loss" not in existing_score_columns:
            conn.execute(
                "ALTER TABLE taxonomy_nn_cv_scores ADD COLUMN hierarchical_loss REAL"
            )

        # Insert model metadata
        cursor = conn.execute(
            """
            INSERT INTO taxonomy_nn_models (
                trained_at, samples, taxonomies, unique_tags, hidden_layers,
                training_accuracy, training_f1, training_hierarchical_loss,
                cv_folds, cv_mean_accuracy, cv_std_accuracy,
                cv_mean_f1, cv_std_f1, cv_mean_hierarchical_loss, cv_std_hierarchical_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                stats.samples,
                stats.taxonomies,
                stats.unique_tags,
                hidden_layers_str,
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
        model_id = int(cursor.lastrowid)

        # Insert CV scores
        if cv_accuracy:
            conn.executemany(
                """
                INSERT INTO taxonomy_nn_cv_scores (
                    model_id, fold, accuracy, f1_weighted, hierarchical_loss
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        model_id,
                        i + 1,
                        cv_accuracy[i],
                        cv_f1[i] if i < len(cv_f1) else None,
                        cv_hier[i] if i < len(cv_hier) else None,
                    )
                    for i in range(len(cv_accuracy))
                ],
            )

        conn.commit()

    return model_id


def render_report_html(
    stats: TrainingStats,
    output_path: Path,
    *,
    hierarchical_base: float = 1.1,
) -> None:
    """Render HTML report summarizing neural network training.

    Args:
        stats: Training statistics
        output_path: Path to save HTML file
        hierarchical_base: Base used for hierarchical loss metric
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    intro = (
        "<p>This report summarizes a neural network classifier trained "
        "to predict product taxonomies from tags using PyTorch.</p>"
    )

    metadata_items = [
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>",
        f"<li><strong>Taxonomies:</strong> {stats.taxonomies:,}</li>",
        f"<li><strong>Input features (tags):</strong> {stats.unique_tags:,}</li>",
        f"<li><strong>Hidden layer sizes:</strong> {stats.hidden_layers}</li>",
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
    parser.add_argument(
        "--hierarchical-base",
        type=float,
        default=1.1,
        help="Base M used for hierarchical loss (loss = M^{-T})",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=None,
        help="Maximum number of most common tags to use (for fair comparison with umllr)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Train on specific CV fold (0-4). Uses umllr fold assignments. If not specified, runs full cross-validation.",
    )

    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_layer_sizes = tuple(int(x.strip()) for x in args.hidden_layers.split(","))

    # Load data
    conn = db.get_connection(args.dsn)
    features, labels, feature_names, metadata, dataset = load_training_data(
        conn,
        product_table=args.product_table,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
    )
    conn.close()

    # Apply max-tags constraint if specified
    if args.max_tags is not None and args.max_tags < len(feature_names):
        features, feature_names = select_top_tags(features, feature_names, args.max_tags)
        print(f"Selected top {len(feature_names)} most common tags (--max-tags={args.max_tags})")

    num_total_classes = len(np.unique(labels))
    print(f"Loaded {len(labels)} products with {len(feature_names)} tags")
    print(f"Architecture: {len(feature_names)} -> {hidden_layer_sizes} -> {num_total_classes}")

    taxonomy_paths = build_taxonomy_path_map(metadata, id_column="taxonomy_index")
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

        # Get total number of classes from the full dataset (not just training data)
        num_classes = len(np.unique(labels))

        # Train model
        model, stats = train_nn_classifier(
            X_train,
            y_train,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.max_iter,
            num_classes=num_classes,
        )

        # Predict on test set
        test_dataset = _to_tensor_dataset(X_test, y_test)
        y_true, y_pred = _collect_predictions(model, test_dataset)

        # Calculate metrics
        test_accuracy = float(np.mean(y_pred == y_true))
        test_f1 = float(f1_score(y_true, y_pred, average="weighted"))
        test_hierarchical = float(hierarchical_loss_score(
            y_true,
            y_pred,
            taxonomy_paths,
            base=args.hierarchical_base,
        ))

        # Calculate p-adic loss
        # Need to map predictions back to taxonomy_id
        original_taxonomy_ids = metadata.iloc[test_mask.nonzero()[0]]["taxonomy_id"].tolist()

        padic_conn = db.get_connection(args.dsn)
        try:
            encodings, prime_base = _load_padic_encodings(padic_conn, args.fold)

            # Map encoded labels back to taxonomy IDs
            taxonomy_id_by_index = metadata.drop_duplicates(subset=["taxonomy_index"]).set_index("taxonomy_index")["taxonomy_id"].to_dict()
            y_true_taxonomy = np.array([taxonomy_id_by_index.get(idx, "") for idx in y_true])
            y_pred_taxonomy = np.array([taxonomy_id_by_index.get(idx, "") for idx in y_pred])

            padic_loss, _ = calculate_padic_loss(y_true_taxonomy, y_pred_taxonomy, encodings, prime_base)
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
                # Override default tablespace temporarily to avoid permission issues
                cur.execute("SET LOCAL default_tablespace = ''")

                # Create table if it doesn't exist
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS padjective.taxonomy_nn_fold_results (
                        cv_fold INTEGER PRIMARY KEY,
                        test_accuracy REAL NOT NULL,
                        test_f1 REAL NOT NULL,
                        test_hierarchical_loss REAL NOT NULL,
                        padic_loss_total REAL NOT NULL,
                        padic_loss_mean REAL NOT NULL,
                        prime_base INTEGER NOT NULL,
                        num_train_samples INTEGER NOT NULL,
                        num_test_samples INTEGER NOT NULL,
                        hidden_layers TEXT NOT NULL,
                        max_tags INTEGER
                    )
                    """
                )

                # Delete existing results for this fold
                cur.execute(
                    "DELETE FROM padjective.taxonomy_nn_fold_results WHERE cv_fold = %s",
                    (args.fold,)
                )

                # Insert new results
                hidden_layers_str = ",".join(str(x) for x in hidden_layer_sizes)
                cur.execute(
                    """
                    INSERT INTO padjective.taxonomy_nn_fold_results
                    (cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                     padic_loss_total, padic_loss_mean, prime_base,
                     num_train_samples, num_test_samples, hidden_layers, max_tags)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (args.fold, test_accuracy, test_f1, test_hierarchical,
                     padic_loss, padic_loss_mean, prime_base,
                     len(y_train), len(y_test), hidden_layers_str, args.max_tags)
                )

                # Save individual predictions
                test_metadata = metadata[test_mask].reset_index(drop=True)
                cur.execute(
                    "DELETE FROM padjective.taxonomy_nn_predictions WHERE cv_fold = %s",
                    (args.fold,)
                )

                for idx in range(len(y_test)):
                    product_id = int(test_metadata.iloc[idx]["product_id"])
                    true_tax_id = y_true_taxonomy[idx]
                    pred_tax_id = y_pred_taxonomy[idx]

                    # Calculate individual p-adic loss
                    true_encoding = encodings.get(true_tax_id, 0)
                    pred_encoding = encodings.get(pred_tax_id, 0)
                    ind_loss = _padic_distance(true_encoding, pred_encoding, prime_base)

                    cur.execute(
                        """
                        INSERT INTO padjective.taxonomy_nn_predictions
                        (cv_fold, product_id, true_taxonomy_id, predicted_taxonomy_id, loss)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (args.fold, product_id, true_tax_id, pred_tax_id, ind_loss)
                    )

                # Save first-layer weights for feature importance analysis
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS padjective.taxonomy_nn_input_weights (
                        cv_fold INTEGER NOT NULL,
                        tag TEXT NOT NULL,
                        hidden_unit INTEGER NOT NULL,
                        weight DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (cv_fold, tag, hidden_unit)
                    )
                    """
                )

                # Delete existing weights for this fold
                cur.execute(
                    "DELETE FROM padjective.taxonomy_nn_input_weights WHERE cv_fold = %s",
                    (args.fold,)
                )

                # Extract first-layer weights (input -> hidden)
                # model[0] is the first Linear layer with shape (hidden_units, n_features)
                first_layer_weights = model[0].weight.data.cpu().numpy()

                # Save each weight
                for hidden_idx in range(first_layer_weights.shape[0]):
                    for feature_idx in range(first_layer_weights.shape[1]):
                        tag = feature_names[feature_idx]
                        weight = float(first_layer_weights[hidden_idx, feature_idx])

                        cur.execute(
                            """
                            INSERT INTO padjective.taxonomy_nn_input_weights
                            (cv_fold, tag, hidden_unit, weight)
                            VALUES (%s, %s, %s, %s)
                            """,
                            (args.fold, tag, hidden_idx, weight)
                        )

            save_conn.commit()
            print(f"\nResults saved to padjective.taxonomy_nn_fold_results and padjective.taxonomy_nn_input_weights")
        finally:
            save_conn.close()

        # Skip the rest of the normal workflow
        return

    # Cross-validate (optional)
    cv_results: CrossValidationResults | None = None
    if not args.skip_cv:
        print("\nRunning cross-validation...")
        cv_results = cross_validate_classifier(
            features,
            labels,
            taxonomy_paths,
            n_folds=args.cv_folds,
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=args.max_iter,
            hierarchical_base=args.hierarchical_base,
        )

    # Train
    print("\nTraining final model...")
    model, stats = train_nn_classifier(
        features,
        labels,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=args.max_iter,
    )

    full_dataset = _to_tensor_dataset(features, labels)
    train_true, train_pred = _collect_predictions(model, full_dataset)
    if train_true.size:
        stats.training_accuracy = float(np.mean(train_true == train_pred))
        stats.training_f1 = float(f1_score(train_true, train_pred, average="weighted"))
        stats.training_hierarchical_loss = float(
            hierarchical_loss_score(
                train_true,
                train_pred,
                taxonomy_paths,
                base=args.hierarchical_base,
            )
        )

    if cv_results and cv_results.folds:
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

    # Save
    save_model_to_database(
        args.model_database,
        model,
        stats,
        cv_results,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.output_dir / "nn_report.html"
    render_report_html(
        stats,
        html_path,
        hierarchical_base=args.hierarchical_base,
    )

    print(f"\nTraining complete!")
    print(f"  Samples: {stats.samples:,}")
    print(f"  Taxonomies: {stats.taxonomies:,}")
    print(f"  Training accuracy: {stats.training_accuracy:.3f}")
    if stats.training_f1 is not None:
        print(f"  Training F1 (weighted): {stats.training_f1:.3f}")
    if stats.training_hierarchical_loss is not None:
        print(
            "  Training hierarchical loss (M={:.2f}): {:.3f}".format(
                args.hierarchical_base, stats.training_hierarchical_loss
            )
        )
    if stats.cross_validation_mean_accuracy is not None:
        cv_text = f"{stats.cross_validation_mean_accuracy:.3f}"
        if stats.cross_validation_std_accuracy is not None:
            cv_text += f" ± {stats.cross_validation_std_accuracy:.3f}"
        print(f"  Cross-validated accuracy: {cv_text} ({stats.cross_validation_folds} folds)")
    if stats.cross_validation_mean_f1 is not None:
        cv_text = f"{stats.cross_validation_mean_f1:.3f}"
        if stats.cross_validation_std_f1 is not None:
            cv_text += f" ± {stats.cross_validation_std_f1:.3f}"
        print(f"  Cross-validated F1 (weighted): {cv_text}")
    if stats.cross_validation_mean_hierarchical_loss is not None:
        cv_text = f"{stats.cross_validation_mean_hierarchical_loss:.3f}"
        if stats.cross_validation_std_hierarchical_loss is not None:
            cv_text += f" ± {stats.cross_validation_std_hierarchical_loss:.3f}"
        print(
            f"  Cross-validated hierarchical loss (M={args.hierarchical_base:.2f}): {cv_text}"
        )
    print(f"\nMetadata saved to {args.model_database}")
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
