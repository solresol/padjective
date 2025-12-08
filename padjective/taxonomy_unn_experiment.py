"""Experiment with unconstrained neural network architectures and L1 regularization.

This script tests various hyperparameter combinations for an unconstrained neural network
that uses ALL tags (like ULR) but with neural network flexibility. L1 regularization
is applied to encourage sparsity.

Run this to determine the best architecture before creating the production classifier.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from psycopg import sql
from scipy import sparse
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

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

from sklearn.metrics import f1_score


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

    def __init__(self, input_dim: int, hidden_layers: tuple[int, ...], output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()

        prev_dim = input_dim
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_dim, hidden_size))
            prev_dim = hidden_size
        self.layers.append(nn.Linear(prev_dim, output_dim))

        self.hidden_layers = hidden_layers

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def l1_loss(self) -> torch.Tensor:
        """Calculate L1 norm of all weights."""
        l1 = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            l1 += torch.sum(torch.abs(layer.weight))
        return l1

    def count_nonzero_params(self, threshold: float = 1e-6) -> tuple[int, int]:
        """Count non-zero parameters (weights with abs value > threshold)."""
        nonzero = 0
        total = 0
        for layer in self.layers:
            weight = layer.weight.data.cpu().numpy()
            bias = layer.bias.data.cpu().numpy()
            nonzero += np.count_nonzero(np.abs(weight) > threshold)
            nonzero += np.count_nonzero(np.abs(bias) > threshold)
            total += weight.size + bias.size
        return nonzero, total


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    hidden_layers: tuple[int, ...]
    l1_lambda: float
    learning_rate: float = 0.001
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config: ExperimentConfig
    test_accuracy: float
    test_f1: float
    test_hierarchical_loss: float
    padic_loss_mean: float
    num_nonzero_params: int
    num_total_params: int
    sparsity_pct: float
    epochs_trained: int
    final_train_loss: float


def train_and_evaluate(
    config: ExperimentConfig,
    train_features: sparse.csr_matrix,
    train_labels: np.ndarray,
    test_features: sparse.csr_matrix,
    test_labels: np.ndarray,
    label_encoder: dict[str, int],
    label_decoder: dict[int, str],
    encodings: dict[str, int],
    prime_base: int,
    taxonomy_paths: dict[str, str],
    hierarchical_base: float = 1.1,
    verbose: bool = True,
) -> ExperimentResult:
    """Train a model with given config and evaluate on test set."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert to tensors
    X_train = torch.from_numpy(train_features.toarray().astype(np.float32))
    y_train = torch.from_numpy(np.array([label_encoder[l] for l in train_labels]).astype(np.int64))
    X_test = torch.from_numpy(test_features.toarray().astype(np.float32))
    y_test_encoded = np.array([label_encoder[l] for l in test_labels])

    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)

    # Split for validation
    val_size = max(1, int(len(train_dataset) * 0.1))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    # Build model
    input_dim = train_features.shape[1]
    output_dim = len(label_encoder)
    model = L1RegularizedNN(input_dim, config.hidden_layers, output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    final_train_loss = 0.0

    for epoch in range(config.max_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            ce_loss = criterion(outputs, batch_y)
            l1_loss = config.l1_lambda * model.l1_loss()
            loss = ce_loss + l1_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        final_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                ce_loss = criterion(outputs, batch_y)
                l1_loss = config.l1_lambda * model.l1_loss()
                val_loss += (ce_loss + l1_loss).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch + 1}")
                break

    epochs_trained = epoch + 1

    # Load best model
    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    # Evaluate on test set
    model.eval()
    X_test_device = X_test.to(device)
    with torch.no_grad():
        outputs = model(X_test_device)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    # Decode predictions back to taxonomy IDs
    y_pred_labels = np.array([label_decoder[p] for p in predictions])

    # Calculate metrics
    test_accuracy = float(np.mean(predictions == y_test_encoded))
    test_f1 = float(f1_score(y_test_encoded, predictions, average="weighted"))

    # Hierarchical loss
    test_hierarchical = hierarchical_loss_score(
        test_labels.tolist(), y_pred_labels.tolist(), taxonomy_paths, base=hierarchical_base
    )

    # P-adic loss
    padic_total, _ = calculate_padic_loss(test_labels, y_pred_labels, encodings, prime_base)
    padic_mean = padic_total / len(test_labels)

    # Sparsity
    nonzero, total = model.count_nonzero_params()
    sparsity_pct = (1 - nonzero / total) * 100 if total > 0 else 0

    return ExperimentResult(
        config=config,
        test_accuracy=test_accuracy,
        test_f1=test_f1,
        test_hierarchical_loss=test_hierarchical,
        padic_loss_mean=padic_mean,
        num_nonzero_params=nonzero,
        num_total_params=total,
        sparsity_pct=sparsity_pct,
        epochs_trained=epochs_trained,
        final_train_loss=final_train_loss,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment with unconstrained neural network architectures"
    )
    parser.add_argument("--dsn", help="Postgres DSN")
    parser.add_argument("--fold", type=int, default=0, help="CV fold to use (0-4)")
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: fewer configs, fewer epochs"
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)

    print("Loading training data (ALL tags)...")
    features, labels, feature_names, metadata, dataset = load_training_data(conn)
    print(f"Loaded {len(labels)} products with {len(feature_names)} tags")

    # Load p-adic encodings
    encodings, prime_base = _load_padic_encodings(conn, args.fold)
    print(f"Prime base: {prime_base}")

    # Build taxonomy path map for hierarchical loss
    taxonomy_paths = build_taxonomy_path_map(metadata)
    ensure_taxonomy_paths_cover_labels(labels, taxonomy_paths)

    # Get fold assignments from umllr
    with conn.cursor() as cur:
        cur.execute(
            "SELECT product_id, cv_fold FROM padjective.umllr_predictions"
        )
        fold_assignments = {row[0]: row[1] for row in cur.fetchall()}

    # Split by fold
    test_mask = metadata["product_id"].apply(lambda pid: fold_assignments.get(pid) == args.fold)
    train_mask = ~test_mask

    train_features = features[train_mask.values]
    train_labels = labels[train_mask.values]
    test_features = features[test_mask.values]
    test_labels = labels[test_mask.values]

    print(f"Fold {args.fold}: {train_features.shape[0]} train, {test_features.shape[0]} test")

    # Create label encoder/decoder
    unique_labels = np.unique(labels)
    label_encoder = {l: i for i, l in enumerate(unique_labels)}
    label_decoder = {i: l for l, i in label_encoder.items()}

    # Define experiment configurations
    input_dim = features.shape[1]  # 1487 tags
    output_dim = len(unique_labels)  # 192 classes

    if args.quick:
        # Quick mode for testing
        architectures = [
            (256,),
            (512,),
        ]
        l1_lambdas = [0.1, 0.01, 0.001, 0.0001]
        max_epochs = 50
    else:
        # Full experiment grid
        # Include much stronger L1 to achieve sparsity
        architectures = [
            # Single hidden layer
            (256,),
            (512,),
            (1024,),
            # Two hidden layers
            (512, 256),
            (256, 128),
        ]
        # Much wider range of L1 - NN needs stronger regularization than sklearn
        l1_lambdas = [1.0, 0.1, 0.01, 0.001, 0.0001]
        max_epochs = 200

    configs = []
    for arch in architectures:
        for l1 in l1_lambdas:
            configs.append(ExperimentConfig(
                hidden_layers=arch,
                l1_lambda=l1,
                max_epochs=max_epochs,
            ))

    print(f"\nRunning {len(configs)} experiments...\n")
    print("=" * 100)

    results = []
    for i, config in enumerate(configs):
        arch_str = " -> ".join(map(str, [input_dim] + list(config.hidden_layers) + [output_dim]))
        print(f"\n[{i+1}/{len(configs)}] Architecture: {arch_str}, L1 lambda: {config.l1_lambda}")

        result = train_and_evaluate(
            config=config,
            train_features=train_features,
            train_labels=train_labels,
            test_features=test_features,
            test_labels=test_labels,
            label_encoder=label_encoder,
            label_decoder=label_decoder,
            encodings=encodings,
            prime_base=prime_base,
            taxonomy_paths=taxonomy_paths,
        )
        results.append(result)

        print(f"    Accuracy: {result.test_accuracy:.4f}, F1: {result.test_f1:.4f}")
        print(f"    P-adic loss: {result.padic_loss_mean:.4f}, Hierarchical: {result.test_hierarchical_loss:.4f}")
        print(f"    Non-zero params: {result.num_nonzero_params:,} / {result.num_total_params:,} ({result.sparsity_pct:.1f}% sparse)")
        print(f"    Epochs: {result.epochs_trained}")

    # Summary table
    print("\n" + "=" * 100)
    print("\nRESULTS SUMMARY (sorted by p-adic loss):\n")

    # Sort by p-adic loss
    results.sort(key=lambda r: r.padic_loss_mean)

    print(f"{'Architecture':<25} {'L1 Lambda':<12} {'P-adic':<10} {'Accuracy':<10} {'F1':<10} {'Non-zero':<15} {'Sparsity':<10}")
    print("-" * 100)

    for r in results:
        arch_str = "->".join(map(str, r.config.hidden_layers))
        print(f"{arch_str:<25} {r.config.l1_lambda:<12.5f} {r.padic_loss_mean:<10.4f} {r.test_accuracy:<10.4f} {r.test_f1:<10.4f} {r.num_nonzero_params:<15,} {r.sparsity_pct:<10.1f}%")

    # Best by different metrics
    print("\n" + "=" * 100)
    print("\nBEST CONFIGURATIONS:")

    best_padic = min(results, key=lambda r: r.padic_loss_mean)
    print(f"\nBest p-adic loss: {best_padic.config.hidden_layers} with L1={best_padic.config.l1_lambda}")
    print(f"  -> P-adic: {best_padic.padic_loss_mean:.4f}, Accuracy: {best_padic.test_accuracy:.4f}, Sparsity: {best_padic.sparsity_pct:.1f}%")

    best_accuracy = max(results, key=lambda r: r.test_accuracy)
    print(f"\nBest accuracy: {best_accuracy.config.hidden_layers} with L1={best_accuracy.config.l1_lambda}")
    print(f"  -> P-adic: {best_accuracy.padic_loss_mean:.4f}, Accuracy: {best_accuracy.test_accuracy:.4f}, Sparsity: {best_accuracy.sparsity_pct:.1f}%")

    best_sparse = max(results, key=lambda r: r.sparsity_pct)
    print(f"\nMost sparse: {best_sparse.config.hidden_layers} with L1={best_sparse.config.l1_lambda}")
    print(f"  -> P-adic: {best_sparse.padic_loss_mean:.4f}, Accuracy: {best_sparse.test_accuracy:.4f}, Sparsity: {best_sparse.sparsity_pct:.1f}%")

    # Pareto-optimal (good p-adic loss AND good sparsity)
    print("\n" + "=" * 100)
    print("\nRECOMMENDATION:")

    # Find configurations with good balance
    # Filter to those with sparsity > 50%
    sparse_results = [r for r in results if r.sparsity_pct > 50]
    if sparse_results:
        best_sparse_padic = min(sparse_results, key=lambda r: r.padic_loss_mean)
        print(f"\nBest p-adic among sparse models (>50% sparsity):")
        print(f"  Architecture: {best_sparse_padic.config.hidden_layers}")
        print(f"  L1 lambda: {best_sparse_padic.config.l1_lambda}")
        print(f"  P-adic loss: {best_sparse_padic.padic_loss_mean:.4f}")
        print(f"  Accuracy: {best_sparse_padic.test_accuracy:.4f}")
        print(f"  Sparsity: {best_sparse_padic.sparsity_pct:.1f}%")
        print(f"  Non-zero params: {best_sparse_padic.num_nonzero_params:,} / {best_sparse_padic.num_total_params:,}")
    else:
        print("No models achieved >50% sparsity with standard L1.")
        print(f"Best overall: {best_padic.config.hidden_layers} with L1={best_padic.config.l1_lambda}")

    conn.close()


def run_pruning_experiment():
    """Test different pruning thresholds on the best model configuration."""
    parser = argparse.ArgumentParser(description="Test weight pruning thresholds")
    parser.add_argument("--dsn", help="Postgres DSN")
    parser.add_argument("--fold", type=int, default=0, help="CV fold to use (0-4)")
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)

    print("Loading training data (ALL tags)...")
    features, labels, feature_names, metadata, dataset = load_training_data(conn)
    print(f"Loaded {len(labels)} products with {len(feature_names)} tags")

    # Load p-adic encodings
    encodings, prime_base = _load_padic_encodings(conn, args.fold)

    # Build taxonomy path map
    taxonomy_paths = build_taxonomy_path_map(metadata)
    ensure_taxonomy_paths_cover_labels(labels, taxonomy_paths)

    # Get fold assignments
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

    print(f"Fold {args.fold}: {train_features.shape[0]} train, {test_features.shape[0]} test")

    # Create label encoder/decoder
    unique_labels = np.unique(labels)
    label_encoder = {l: i for i, l in enumerate(unique_labels)}
    label_decoder = {i: l for l, i in label_encoder.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train best model (256 hidden, L1=0.0001)
    print("\nTraining best model configuration: 256 hidden, L1=0.0001...")

    X_train = torch.from_numpy(train_features.toarray().astype(np.float32))
    y_train = torch.from_numpy(np.array([label_encoder[l] for l in train_labels]).astype(np.int64))
    X_test = torch.from_numpy(test_features.toarray().astype(np.float32))
    y_test_encoded = np.array([label_encoder[l] for l in test_labels])

    train_dataset = TensorDataset(X_train, y_train)
    val_size = max(1, int(len(train_dataset) * 0.1))
    train_size = len(train_dataset) - val_size
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_subset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=256, shuffle=False)

    input_dim = train_features.shape[1]
    output_dim = len(label_encoder)
    model = L1RegularizedNN(input_dim, (256,), output_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    l1_lambda = 0.0001

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(200):
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
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_model_state:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    print("\nTesting different pruning thresholds...")
    print("=" * 100)

    # Test various pruning thresholds
    thresholds = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.5]

    print(f"\n{'Threshold':<12} {'P-adic':<10} {'Accuracy':<10} {'F1':<10} {'Non-zero':<15} {'Sparsity':<10}")
    print("-" * 80)

    for threshold in thresholds:
        # Create a copy of model and apply pruning
        pruned_model = L1RegularizedNN(input_dim, (256,), output_dim).to(device)
        pruned_model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()})

        # Apply pruning (set weights below threshold to zero)
        with torch.no_grad():
            for layer in pruned_model.layers:
                mask = torch.abs(layer.weight) > threshold
                layer.weight.data *= mask.float()
                # Also prune biases
                bias_mask = torch.abs(layer.bias) > threshold
                layer.bias.data *= bias_mask.float()

        # Evaluate
        pruned_model.eval()
        X_test_device = X_test.to(device)
        with torch.no_grad():
            outputs = pruned_model(X_test_device)
            predictions = outputs.argmax(dim=1).cpu().numpy()

        y_pred_labels = np.array([label_decoder[p] for p in predictions])
        test_accuracy = float(np.mean(predictions == y_test_encoded))
        test_f1 = float(f1_score(y_test_encoded, predictions, average="weighted"))
        padic_total, _ = calculate_padic_loss(test_labels, y_pred_labels, encodings, prime_base)
        padic_mean = padic_total / len(test_labels)

        nonzero, total = pruned_model.count_nonzero_params(threshold=0)  # Count actual zeros now
        sparsity = (1 - nonzero / total) * 100

        print(f"{threshold:<12.0e} {padic_mean:<10.4f} {test_accuracy:<10.4f} {test_f1:<10.4f} {nonzero:<15,} {sparsity:<10.1f}%")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--pruning":
        sys.argv.pop(1)  # Remove --pruning flag
        run_pruning_experiment()
    else:
        main()
