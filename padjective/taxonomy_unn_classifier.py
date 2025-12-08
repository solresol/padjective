"""Train unconstrained neural network classifier with L1 regularization and weight pruning.

This module provides an unconstrained neural network model that uses ALL available tags
(like ULR) but with neural network flexibility. L1 regularization is applied during
training, and post-training weight pruning is used to achieve sparsity. The number of
non-zero parameters is tracked as a key metric.

Architecture: input_tags -> 256 hidden (ReLU) -> num_classes
Hyperparameters determined through experimentation:
- L1 lambda: 0.0001
- Pruning threshold: 0.01
- Learning rate: 0.001
- Batch size: 256
- Early stopping patience: 15
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import f1_score

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db
    from padjective.metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
    )
else:
    from . import data_access, db
    from .metrics import (
        build_taxonomy_path_map,
        ensure_taxonomy_paths_cover_labels,
        hierarchical_loss_score,
    )


# Hyperparameters (determined through experimentation)
HIDDEN_SIZE = 256
L1_LAMBDA = 0.0001
PRUNING_THRESHOLD = 0.01
LEARNING_RATE = 0.001
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15


def _load_padic_encodings(conn, cv_fold: int) -> tuple[dict[str, int], int]:
    """Load p-adic encodings for a specific CV fold."""
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
    """Calculate p-adic valuation of n."""
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


def load_training_data(
    conn,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], pd.DataFrame, data_access.ProductDataset]:
    """Load product tags and taxonomy labels for training (ALL tags)."""
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

    labels = metadata["taxonomy_id"].to_numpy()
    return dataset.features, labels, dataset.feature_names, metadata, dataset


class L1RegularizedNN(nn.Module):
    """Neural network with L1 regularization on weights."""

    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_size)
        self.output = nn.Linear(hidden_size, output_dim)
        self.hidden_size = hidden_size

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        return self.output(x)

    def l1_loss(self) -> torch.Tensor:
        """Calculate L1 norm of all weights."""
        l1 = torch.sum(torch.abs(self.hidden.weight))
        l1 += torch.sum(torch.abs(self.output.weight))
        return l1

    def apply_pruning(self, threshold: float = PRUNING_THRESHOLD) -> None:
        """Zero out weights below threshold."""
        with torch.no_grad():
            mask = torch.abs(self.hidden.weight) > threshold
            self.hidden.weight.data *= mask.float()
            bias_mask = torch.abs(self.hidden.bias) > threshold
            self.hidden.bias.data *= bias_mask.float()

            mask = torch.abs(self.output.weight) > threshold
            self.output.weight.data *= mask.float()
            bias_mask = torch.abs(self.output.bias) > threshold
            self.output.bias.data *= bias_mask.float()

    def count_nonzero_params(self) -> tuple[int, int]:
        """Count non-zero parameters."""
        nonzero = 0
        total = 0

        for param in [self.hidden.weight, self.hidden.bias, self.output.weight, self.output.bias]:
            data = param.data.cpu().numpy()
            nonzero += np.count_nonzero(data)
            total += data.size

        return nonzero, total


@dataclass(slots=True)
class TrainingStats:
    """Metadata about the trained classifier."""
    samples: int
    taxonomies: int
    unique_tags: int
    hidden_size: int
    training_accuracy: float
    num_nonzero_params: int = 0
    num_total_params: int = 0
    l1_lambda: float = L1_LAMBDA
    pruning_threshold: float = PRUNING_THRESHOLD
    training_f1: float | None = None
    training_hierarchical_loss: float | None = None


def train_unn_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    label_encoder: dict[str, int],
    *,
    hidden_size: int = HIDDEN_SIZE,
    l1_lambda: float = L1_LAMBDA,
    pruning_threshold: float = PRUNING_THRESHOLD,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
) -> tuple[L1RegularizedNN, TrainingStats]:
    """Train an L1-regularized neural network with weight pruning.

    Args:
        features: Sparse feature matrix (n_samples x n_features)
        labels: Target labels as taxonomy IDs
        label_encoder: Mapping from taxonomy_id to integer index
        hidden_size: Number of hidden units
        l1_lambda: L1 regularization strength
        pruning_threshold: Weights below this are zeroed after training
        learning_rate: Adam learning rate
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience

    Returns:
        tuple: (trained_model, training_stats)
    """
    if len(labels) == 0:
        raise ValueError("Cannot train on empty dataset")

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 taxonomies to train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X = torch.from_numpy(features.toarray().astype(np.float32))
    y = torch.from_numpy(np.array([label_encoder[l] for l in labels]).astype(np.int64))

    dataset = TensorDataset(X, y)

    # Split for validation
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    # Build model
    input_dim = features.shape[1]
    output_dim = len(label_encoder)
    model = L1RegularizedNN(input_dim, hidden_size, output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(max_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss = criterion(outputs, batch_y)
            l1_loss = l1_lambda * model.l1_loss()
            loss = ce_loss + l1_loss
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                ce_loss = criterion(outputs, batch_y)
                l1_loss = l1_lambda * model.l1_loss()
                val_loss += (ce_loss + l1_loss).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model and apply pruning
    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    model.apply_pruning(pruning_threshold)

    # Calculate training accuracy on full dataset
    model.eval()
    X_device = X.to(device)
    with torch.no_grad():
        outputs = model(X_device)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    accuracy = float(np.mean(predictions == y.numpy()))
    nonzero, total = model.count_nonzero_params()

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=len(unique_labels),
        unique_tags=features.shape[1],
        hidden_size=hidden_size,
        training_accuracy=accuracy,
        num_nonzero_params=nonzero,
        num_total_params=total,
        l1_lambda=l1_lambda,
        pruning_threshold=pruning_threshold,
    )

    return model, stats


def create_tables(conn, schema: str = "padjective") -> None:
    """Create UNN results tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace = ''")

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.taxonomy_unn_fold_results (
                    cv_fold INTEGER PRIMARY KEY,
                    test_accuracy REAL NOT NULL,
                    test_f1 REAL NOT NULL,
                    test_hierarchical_loss REAL NOT NULL,
                    padic_loss_total REAL NOT NULL,
                    padic_loss_mean REAL NOT NULL,
                    prime_base INTEGER NOT NULL,
                    num_train_samples INTEGER NOT NULL,
                    num_test_samples INTEGER NOT NULL,
                    hidden_size INTEGER NOT NULL,
                    num_nonzero_params INTEGER NOT NULL,
                    num_total_params INTEGER NOT NULL,
                    l1_lambda REAL NOT NULL,
                    pruning_threshold REAL NOT NULL,
                    trained_at TIMESTAMPTZ DEFAULT now()
                )
                """
            ).format(schema=sql.Identifier(schema))
        )

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.taxonomy_unn_predictions (
                    cv_fold INTEGER NOT NULL,
                    product_id INTEGER NOT NULL,
                    true_taxonomy_id TEXT NOT NULL,
                    predicted_taxonomy_id TEXT NOT NULL,
                    loss REAL NOT NULL,
                    PRIMARY KEY (cv_fold, product_id)
                )
                """
            ).format(schema=sql.Identifier(schema))
        )

    conn.commit()


def main() -> None:
    """Command-line interface for training UNN classifier."""
    parser = argparse.ArgumentParser(
        description="Train unconstrained neural network to predict product taxonomy from tags"
    )
    parser.add_argument("--dsn", help="Postgres DSN")
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
        required=True,
        help="Train on specific CV fold (0-4). Uses umllr fold assignments.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=HIDDEN_SIZE,
        help=f"Hidden layer size (default: {HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--l1-lambda",
        type=float,
        default=L1_LAMBDA,
        help=f"L1 regularization strength (default: {L1_LAMBDA})",
    )
    parser.add_argument(
        "--pruning-threshold",
        type=float,
        default=PRUNING_THRESHOLD,
        help=f"Weight pruning threshold (default: {PRUNING_THRESHOLD})",
    )
    parser.add_argument(
        "--hierarchical-base",
        type=float,
        default=1.1,
        help="Base M used for hierarchical loss",
    )

    args = parser.parse_args()

    conn = db.get_connection(args.dsn)

    print(f"Training unconstrained neural network classifier for fold {args.fold}...")

    # Load data
    features, labels, feature_names, metadata, dataset = load_training_data(
        conn,
        product_table=args.product_table,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
    )
    print(f"Loaded {len(labels)} products with {len(feature_names)} tags (using ALL tags)")

    # Load p-adic encodings
    encodings, prime_base = _load_padic_encodings(conn, args.fold)

    # Build taxonomy path map
    taxonomy_paths = build_taxonomy_path_map(metadata)
    ensure_taxonomy_paths_cover_labels(labels, taxonomy_paths)

    # Get fold assignments from umllr
    with conn.cursor() as cur:
        cur.execute("SELECT product_id, cv_fold FROM padjective.umllr_predictions")
        fold_assignments = {row[0]: row[1] for row in cur.fetchall()}

    # Split by fold
    test_mask = metadata["product_id"].apply(lambda pid: fold_assignments.get(pid) == args.fold)
    train_mask = ~test_mask

    train_features = features[train_mask.values]
    train_labels = labels[train_mask.values]
    test_features = features[test_mask.values]
    test_labels = labels[test_mask.values]

    print(f"\nFold {args.fold}: Training on {train_features.shape[0]} products, testing on {test_features.shape[0]} products")

    # Create label encoder (using all labels to ensure consistent encoding)
    unique_labels = np.unique(labels)
    label_encoder = {l: i for i, l in enumerate(unique_labels)}
    label_decoder = {i: l for l, i in label_encoder.items()}

    # Train model
    model, stats = train_unn_classifier(
        train_features,
        train_labels,
        label_encoder,
        hidden_size=args.hidden_size,
        l1_lambda=args.l1_lambda,
        pruning_threshold=args.pruning_threshold,
    )

    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    X_test = torch.from_numpy(test_features.toarray().astype(np.float32)).to(device)
    y_test_encoded = np.array([label_encoder[l] for l in test_labels])

    with torch.no_grad():
        outputs = model(X_test)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    y_pred_labels = np.array([label_decoder[p] for p in predictions])

    # Calculate metrics
    test_accuracy = float(np.mean(predictions == y_test_encoded))
    test_f1 = float(f1_score(y_test_encoded, predictions, average="weighted"))
    test_hierarchical = hierarchical_loss_score(
        test_labels.tolist(), y_pred_labels.tolist(), taxonomy_paths, base=args.hierarchical_base
    )

    # P-adic loss
    padic_total, padic_losses = calculate_padic_loss(test_labels, y_pred_labels, encodings, prime_base)
    padic_mean = padic_total / len(test_labels)

    print(f"\nFold {args.fold} Results:")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Test F1 (weighted): {test_f1:.4f}")
    print(f"  Test hierarchical loss (M={args.hierarchical_base:.2f}): {test_hierarchical:.4f}")
    print(f"  P-adic loss (total): {padic_total:.4f}")
    print(f"  P-adic loss (mean): {padic_mean:.6f}")
    print(f"  Prime base: {prime_base}")
    print(f"  Non-zero parameters: {stats.num_nonzero_params:,} / {stats.num_total_params:,} ({(1 - stats.num_nonzero_params/stats.num_total_params)*100:.1f}% sparse)")
    print(f"  L1 lambda: {args.l1_lambda}, Pruning threshold: {args.pruning_threshold}")

    # Save results to database
    create_tables(conn, args.results_schema)

    with conn.cursor() as cur:
        # Delete existing results for this fold
        cur.execute(
            sql.SQL("DELETE FROM {schema}.taxonomy_unn_fold_results WHERE cv_fold = %s").format(
                schema=sql.Identifier(args.results_schema)
            ),
            (args.fold,)
        )

        # Insert fold results
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.taxonomy_unn_fold_results
                (cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                 padic_loss_total, padic_loss_mean, prime_base,
                 num_train_samples, num_test_samples, hidden_size,
                 num_nonzero_params, num_total_params, l1_lambda, pruning_threshold)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            ).format(schema=sql.Identifier(args.results_schema)),
            (
                args.fold, test_accuracy, test_f1, test_hierarchical,
                padic_total, padic_mean, prime_base,
                len(train_labels), len(test_labels), args.hidden_size,
                stats.num_nonzero_params, stats.num_total_params,
                args.l1_lambda, args.pruning_threshold
            )
        )

        # Save predictions
        cur.execute(
            sql.SQL("DELETE FROM {schema}.taxonomy_unn_predictions WHERE cv_fold = %s").format(
                schema=sql.Identifier(args.results_schema)
            ),
            (args.fold,)
        )

        test_metadata = metadata[test_mask].reset_index(drop=True)
        for idx in range(len(test_labels)):
            product_id = int(test_metadata.iloc[idx]["product_id"])
            true_tax_id = test_labels[idx]
            pred_tax_id = y_pred_labels[idx]
            loss = padic_losses[idx]

            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_unn_predictions
                    (cv_fold, product_id, true_taxonomy_id, predicted_taxonomy_id, loss)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(args.results_schema)),
                (args.fold, product_id, true_tax_id, pred_tax_id, loss)
            )

    conn.commit()
    print(f"\nResults saved to {args.results_schema}.taxonomy_unn_fold_results")

    conn.close()


if __name__ == "__main__":
    main()
