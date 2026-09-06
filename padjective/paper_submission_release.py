"""Export a frozen paper replication package directly from Shopify PostgreSQL."""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import shutil
import subprocess

from . import benchmark_runtime, data_access, db, taxonomy_mihara_comparison, umllr
from .product_taxonomy_bench_export import export_snapshot
from .product_taxonomy_bench_notebook import write_notebook


REPLICATION_README = """# Frozen paper replication release: 6 September 2026

This directory contains anonymised archived matrices and standalone numerical
code. It neither connects to PostgreSQL nor requires the private catalogue.
Operational ingestion remains PostgreSQL-only; these are research exports.

## Run

Use Python 3.11.11 and the pinned `requirements.txt` (the source `uv.lock` is also
included). From this directory:

```sh
uv venv --python 3.11.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python paper_replication.py --suite models --output models.json
.venv/bin/python paper_replication.py --suite neural --output neural.json
.venv/bin/python paper_replication.py --suite digitwise --output digitwise.json
```

The models suite runs all five folds, all primary models, separate budget
ablations and tag-order ablations. The primary neural width is 2,000 (selected
after the sweep), with max_iter=10,000, seed=42, alpha=0.0001 and batch_size=256.
The neural suite sweeps widths 4, 8, 12, 24, 48 and 2,000 on those same folds.
Random tag-order seeds are 7, 13, 23, 37 and 101; their reported SD is over all
25 fold-by-seed losses. Other SDs in that comparison are over five fold losses.

The stochastic continuation starts from the greedy fit, uses p=71, 2,000
iterations, seed=42+fold, temperature 1 times 0.9995 per step (stopping below 0.001),
and keeps the best iterate. A uniform tag is perturbed: with probability 1/2
add a uniform signed power of p (exponent 0 through 5), otherwise add a
uniform integer from -1,000 through 1,000. Uphill acceptance is exp(-delta/T).
It optimises raw training-score loss; the zero-score reporting default is
fitted after optimisation. No Mahler transform is used. This is an adaptation,
not a reproduction of Zubarev's full published algorithm.

## Two different matrices

`paper/`: 6,693 products, 2,542 tag columns and 363 classes, with archived
fold IDs. This is the main benchmark and neural-sweep population.

`digitwise-products.jsonl.gz`: 6,527 products, 2,747 tags and 308 classes,
exported separately from the archived snapshot's raw-tag rematerialisation.
The loader's preselection vocabulary contains 2,971 tags. Anonymous tag IDs
preserve lexical order and each row's original tag sequence; row IDs reveal
neither merchant nor product identity. No product titles or source URLs appear.
This is the bounded digitwise diagnostic's population, not a replacement for
the main matrix. Its loss should only be compared with the greedy baseline
computed on this same population (mean 0.249694).

The digitwise suite uses p=71, precision=7, 96 trials, seed=0 with the source
fold/digit offsets, and frequency-independent feature selection. Caps are
32, 64, 128, 256, 512, 768, 1,024, 1,536, 2,048 and 2,971. The 90% diagnostic
threshold was chosen for this experiment and is not prescribed by Mihara.
Forced continuation after a failed threshold is diagnostic only. The full
sweep can take many hours. `--caps 32 --folds 0` is a smoke test, not a full run.

`manifest.json` records source commit, Python/package versions, counts and
checksums. `stochastic-fold-results.csv` records the corrected five-fold run.
Paper figure inputs and generators are versioned in the companion repository
https://github.com/solresol/papers/tree/codex/paper-benchmark-rerun/padjective/padic-journal
and identified by the manuscript revision. Those CSVs reproduce plots; the
standalone runners here reproduce model fits. Floating reduction order and
platform libraries can affect last-bit agreement and neural optimisation.

## Use and attribution

The parent dataset is marked `license: other`; this release does not grant a
new licence to the source catalogue or change existing rights. Contact Greg
Baker through the dataset repository for reuse terms beyond those explicitly
provided there. Cite the dataset's frozen `paper-submission-2026-09-06` revision
and the accompanying paper. This release contains anonymised research data,
not the original merchants' catalogue text.
"""


def render_digitwise_runtime() -> str:
    """Extract unchanged pure definitions, excluding all database operations."""
    source = inspect.getsource(taxonomy_mihara_comparison)
    lines = source.splitlines()
    parts = [
        '"""Bounded Mihara-inspired diagnostic; 90% is our chosen cutoff."""',
        "from __future__ import annotations",
        "from collections import Counter, defaultdict",
        "from dataclasses import dataclass",
        "import math, random",
        "from typing import Any, Mapping, Sequence",
        "import numpy as np",
        "from benchmark_runtime import (ProductRecord, summarize_encoded_predictions, "
        "p_adic_distance as _p_adic_distance)",
    ]
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "_ensure_storage":
            break
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign)):
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            parts.append("\n".join(lines[start - 1:node.end_lineno]))
    return "\n\n".join(parts) + "\n"


def anonymise_diagnostic(records):
    tags = sorted({tag for record in records for tag in record.tags})
    mapping = {tag: f"tag{i:06d}" for i, tag in enumerate(tags, 1)}
    return [
        {
            "product_id": i, "product_key": f"row{i:06d}",
            "tags": [mapping[tag] for tag in record.tags],
            "encoded_path": record.encoded_path, "cv_fold": record.cv_fold,
            "taxonomy_id": record.taxonomy_id, "taxonomy_depth": record.taxonomy_depth,
            "title_tag_positions": [],
        }
        for i, record in enumerate(records)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists():
        parser.error("output directory already exists; use a fresh destination")
    root.mkdir(parents=True)
    conn = db.get_connection(args.dsn)
    try:
        snapshot_id, _ = data_access._resolve_snapshot_id(conn, schema="padjective", snapshot_ref="paper")
        metadata = export_snapshot(
            conn, schema="padjective", snapshot_ref="paper", snapshot_id=snapshot_id,
            out_dir=root / "paper", formats=("jsonl",), gzip_jsonl=True, rows_per_shard=10000,
        )
        records, p, _, taxonomies, dataset = umllr._load_products(
            conn, "cantbuymelove.product", None, snapshot_ref="paper",
            min_tag_count=5, min_samples_per_taxonomy=5,
        )
        anonymous = anonymise_diagnostic(records)
        counts = (len(anonymous), len(taxonomies), len({tag for r in anonymous for tag in r["tags"]}))
        if counts != (6527, 308, 2747) or p != 71:
            raise ValueError(f"Diagnostic matrix changed: {counts}, p={p}")
        with gzip.open(root / "digitwise-products.jsonl.gz", "wt") as handle:
            for row in anonymous:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM padjective.paper_revision_zubarev WHERE snapshot_ref = %s ORDER BY cv_fold", ("paper",))
            with (root / "stochastic-fold-results.csv").open("w") as handle:
                writer = csv.writer(handle)
                writer.writerow([c.name for c in cur.description])
                writer.writerows(cur.fetchall())
    finally:
        conn.close()
    package = Path(__file__).resolve().parent
    for name in ("benchmark_runtime.py", "paper_replication.py"):
        shutil.copy2(package / name, root / name)
    (root / "digitwise_runtime.py").write_text(render_digitwise_runtime())
    write_notebook(root / "notebooks/product_taxonomy_bench.ipynb")
    shutil.copy2(package.parent / "uv.lock", root / "uv.lock")
    distributions = (
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib",
        "threadpoolctl", "contourpy", "cycler", "fonttools", "kiwisolver",
        "packaging", "pillow", "pyparsing", "python-dateutil", "pytz", "six", "tzdata",
    )
    versions = {name: importlib.metadata.version(name) for name in distributions}
    (root / "requirements.txt").write_text("\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n")
    (root / "README.md").write_text(REPLICATION_README)
    (root / ".python-version").write_text(platform.python_version() + "\n")
    manifest = {
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "paper_snapshot_id": str(snapshot_id),
        "paper_counts": [metadata.product_count, metadata.tag_count, metadata.taxonomy_count],
        "digitwise_counts": {"products": counts[0], "taxonomies": counts[1], "tags": counts[2]},
        "digitwise_preselection_tags": len(dataset.feature_names),
        "python_version": platform.python_version(),
        "versions": versions, "stochastic_objective": "raw_score_then_fit_reporting_default",
        "digitwise_threshold_origin": "experiment-defined, not prescribed by Mihara",
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(root.rglob("*")) if path.is_file()},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(root), "counts": manifest["digitwise_counts"]}))


if __name__ == "__main__":
    main()
