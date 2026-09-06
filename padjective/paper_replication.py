"""Run frozen, anonymised paper evidence without PostgreSQL or project imports.

The release builder copies this file beside benchmark_runtime.py and a generated
digitwise_runtime.py. Product ingestion remains a PostgreSQL-only operation;
this module is solely an offline replication consumer of published snapshots.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

try:
    from . import benchmark_runtime as runtime
except ImportError:  # Standalone submission release.
    import benchmark_runtime as runtime


def environment_differences(root):
    manifest = json.loads((root / "manifest.json").read_text())
    differences = []
    if platform.python_version() != manifest["python_version"]:
        differences.append(f"Python {platform.python_version()} != {manifest['python_version']}")
    for name, expected in manifest["versions"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "missing"
        if actual != expected:
            differences.append(f"{name} {actual} != {expected}")
    return differences


def neural_rows(tables, widths, folds):
    features, labels = runtime._build_sparse_matrix(
        tables.products, tables.tags, tables.product_tags
    )
    fold_ids = tables.products["cv_fold"].to_numpy(dtype=int)
    encoded = {
        str(tid): runtime.encode_path(runtime.parse_taxonomy_digits(str(path)), 71)
        for tid, path in tables.products[["taxonomy_id", "taxonomy_path"]]
        .drop_duplicates().itertuples(index=False, name=None)
    }
    rows = []
    for width in widths:
        for fold in folds:
            train, test = fold_ids != fold, fold_ids == fold
            model = runtime.MLPClassifier(
                hidden_layer_sizes=(width,), activation="relu", alpha=1e-4,
                batch_size=256, max_iter=10000, random_state=42,
            )
            model.fit(features[train], labels[train])
            predictions = model.predict(features[test])
            losses = [runtime.p_adic_distance(encoded[str(a)], encoded[str(b)], 71)
                      for a, b in zip(labels[test], predictions, strict=True)]
            rows.append({
                "hidden_units": width, "cv_fold": fold,
                "mean_loss": float(np.mean(losses)),
                "exact_accuracy": float(np.mean(labels[test] == predictions)),
                "num_params": runtime._dense_model_params(model),
                "iterations_used": int(model.n_iter_),
            })
            print(json.dumps(rows[-1]), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("models", "neural", "digitwise"), required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--widths", default="4,8,12,24,48,2000")
    parser.add_argument("--caps", default="32,64,128,256,512,768,1024,1536,2048,2971")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-environment-drift", action="store_true",
                        help="Exploratory only: permit versions different from the frozen run")
    args = parser.parse_args()
    folds = [int(v) for v in args.folds.split(",")]
    if not set(folds) <= set(range(5)):
        parser.error("folds must be drawn from 0,1,2,3,4")
    if args.output.exists():
        parser.error("output already exists; use a new run filename")
    differences = environment_differences(args.root)
    if differences and not args.allow_environment_drift:
        parser.error("Not the frozen paper environment: " + "; ".join(differences)
                     + ". Follow README.md, or explicitly allow an exploratory run.")
    if differences:
        print("EXPLORATORY ENVIRONMENT: " + "; ".join(differences), flush=True)
    if args.suite == "digitwise":
        import digitwise_runtime as digitwise
        with gzip.open(args.root / "digitwise-products.jsonl.gz", "rt") as handle:
            records = [runtime.ProductRecord(**json.loads(line)) for line in handle]
        output = []
        for cap in [int(v) for v in args.caps.split(",")]:
            for fold in folds:
                result = digitwise.run_fold(
                    fold, records, p=71, precision=7, max_tags=cap,
                    feature_selection="frequency_independent", trials=96,
                    seed=0, acceptance_threshold=0.9,
                )
                row = {
                    "cap": cap, "fold": fold, "mean_loss": result.mean_loss,
                    "selected_tags": len(result.selected_tags),
                    "first_digit_agreement": result.fit.diagnostics[0].validation_inlier_fraction,
                    "accepted_prefix_digits": result.fit.accepted_prefix_digits,
                    "diagnostics": [asdict(d) for d in result.fit.diagnostics],
                }
                output.append(row)
                print(json.dumps(row), flush=True)
    else:
        tables = runtime.load_snapshot_tables(args.root / "paper", snapshot_label="paper")
        if args.suite == "models":
            runtime.DEFAULT_UNN_HIDDEN = 2000
            runtime.DEFAULT_UNN_MAX_ITER = 10000
            output = runtime.build_snapshot_benchmark_bundle(tables)
        else:
            output = neural_rows(tables, [int(v) for v in args.widths.split(",")], folds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
