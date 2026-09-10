"""Build an additive frozen follow-up release; verify public data against Postgres."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request


REFERENCE_COMMIT = "7cbb35030dcb2175d130e44a42442a1efb1c983b"
REFERENCE_TAG_COMMIT = "6a496bd7f1969629a41f547371d15d3375aa86c1"
REFERENCE_MANIFEST_SHA256 = "31d48bc37447f8fd24d30fa4b5527b120c32854aa95e987a2892d89688e4d3b9"
SNAPSHOT = "244ddbe3-0a2c-4c04-9436-0a0253108a09"
DIGEST = "039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222"
HELPERS = {
    "paper_published_methods.py": ["training_feature_order", "exact_rank_certificate", "inclusion_certificate"],
    "paper_validate_published.py": ["exact_scores", "compare_scores", "reference_predictions"],
    "paper_validate_randomised.py": ["fingerprint", "distance_matrix", "direct_medoid", "direct_linear", "direct_certificate"],
    "paper_validate_ensemble_scaling.py": ["nested_consensus"],
}
EVIDENCE = {
    "docs/ensemble-scaling/results.json": "ensemble-results.json",
    "docs/ensemble-scaling/analysis.json": "ensemble-analysis.json",
    "docs/ensemble-scaling/metric-audit.json": "ensemble-metric-audit.json",
    "docs/ensemble-scaling/results.md": "ensemble-results.md",
    "docs/ensemble-scaling-protocol.md": "ensemble-scaling-protocol.md",
    "docs/randomised-methods-protocol.md": "randomised-methods-protocol.md",
    "docs/published-methods-results.json": "published-results.json",
    "docs/published-methods-capacity-bounds.json": "published-capacity-bounds.json",
    "docs/published-methods-results.md": "published-results.md",
    "docs/published-methods-comparison.md": "published-methods-protocol.md",
}

README = """# Frozen paper replication release: 10 September 2026

This additive release accompanies *Benchmarking p-adic Linear Regression on
Sparse Product Data: Sparse Greedy Models and Randomised Linear Ensembles*.
It adds the 8 September published-method comparison and the 10 September
243-member bank. It does not replace the immutable 6 September release.

## Reproduce the newer fits without the private database

Download this complete directory, not just its README. With `uv` installed:

```sh
uv venv --python 3.11.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python followup_replication.py --suite published --workers 3 --output-dir rerun-published
.venv/bin/python followup_replication.py --suite ensembles --workers 4 --output-dir rerun-ensembles
```

The runners require only local, anonymised files. There is no PostgreSQL,
Shopify, network, credential or production-service dependency. The publisher
verified this public matrix against the archived Shopify PostgreSQL matrix:
all 6,693 rows, 2,542 ordered tag columns, 363 paths and five stored folds.
Eleven rows have stale scalar `tag_count` metadata (smaller than their actual
feature list); neither numerical loader uses it. The unchanged relation/list
is authoritative, and the parity check compares every actual matrix entry.
Operational ingestion remains PostgreSQL-only; this is an offline research
consumer of an already-public export, not a CSV-based production pipeline.

Use at most four workers on a shared host. Each worker uses one numerical
thread for these newer experiments. Published Zubarev runs are bounded at
300 sampling seconds (construction/scoring outside that budget); Mihara uses
180 fitting seconds and one million draws, preceded by a 120-second rank
audit. Coordinate fits use 100 sweeps and 300 fitting seconds, and must finish
a complete no-improvement sweep. Allow tens of minutes and several GB RAM
for the complete experiment grids. Resource-limited runs on slower machines
can fail reference verification; they are not silently scored as completed fits.

For a small smoke test, use `--suite ensembles --folds 0 --members 1` and a
separate output directory. Partial runs are labelled `full_grid: false`.
Reusing an output directory resumes verified case files only when its release
manifest and run settings match. Keep generated outputs outside the release.
The new runner checks every file checksum and Python/package versions before
running. It compares exact coefficient/prediction fingerprints and independently
recomputed scores, not just rounded means. Completed coordinate fits are also
independently re-certified. It reconstructs all 945 ensemble rows, including
20 alternative fixed rosters. Wall-clock times and interrupted Mihara draw/
restart counts are not expected to match across machines. An incomplete
Mihara fit has no predictive score; a time limit is not an algorithmic failure
proof. Full-rank/inclusion impossibility certificates are distinguished from
those externally interrupted attempts.

## Coverage and provenance

- `algorithms/`: unchanged pure numerical modules plus explicitly extracted
  pure audit helpers. `manifest.json` records source hashes and exact helper
  names; no fit was translated or redesigned for this archive.
- `evidence/`: the original aggregate results, fingerprints and dated protocols.
  The 65 published-method cases are 45 Zubarev schedules (three degrees, three
  independent random seeds, five folds) and 20 Mihara cases. The 1,215
  coordinate fits form 243 models per fold; the offline check refits even the
  45 models reused in the original follow-up. Nine nested ensemble sizes are
  retained; 135 was best on the primary observed curve, while 243 was the
  predeclared maximum, not a held-out-selected best checkpoint.
- `reference/`: a byte-preserved copy of the 6 September replication directory
  at commit `7cbb35030dcb2175d130e44a42442a1efb1c983b`. The earlier dated tag
  itself points to `6a496bd7f1969629a41f547371d15d3375aa86c1`; it is not moved.
  Its original manifest is verified before reuse. Run its separate
  `paper_replication.py` for the classical, sparse-greedy, neural-width and
  historical diagnostic suites. Follow its 12-thread environment for the
  width-2,000 reference result (loss 0.0755591194); the one-thread neural
  sensitivity is not a replacement for that result.
- `data-parity.json`: exact archived PostgreSQL/public-matrix parity evidence.
- `validation/`: fresh offline reproduction summaries, added only after checks
  pass. Per-product predictions and private database credentials are not
  published. The local rerunner generates its own per-case predictions.
- `paper-evidence/`, when present: the frozen manuscript, quantitative inputs,
  generators and build instructions. These reproduce the figures independently
  of the expensive fitting stage. The manifest identifies the paper commit.

The older protocols document decisions as of their dates; later manuscript
choices (such as the selected ensemble endpoints) do not retroactively rewrite
those records. This release freezes the existing scientific results and adds
reproduction checks; it does not select new seeds or tune against held-out folds.
Matched-budget feature-dropping ablations remain a separate question and are
not observations in the intended active-support cross-model regression.

## Correction to historical documentation

The preserved 6 September README/manifest incorrectly said that the 90%
threshold was not prescribed by Mihara. The published Algorithm 2 does include
a strict, rank-dependent 9/10 inclusion condition. The older bounded digitwise
diagnostic did not implement the complete published procedure, notably its
restart/acceptance semantics; its forced continuation is diagnostic only.
The September 8 implementation in `algorithms/published_mihara.py` implements
Algorithms 1--8 and keeps external resource limits separate from acceptance.
This correction does not change the frozen historical files or their hashes.
Similarly, the old greedy-start stochastic continuation is a separate
adaptation, not Zubarev's published Mahler-polynomial/Gibbs method. The newer
comparator starts independently from uniformly random coefficients and uses
the published finite-quotient transition distribution. It is not seeded with
the sparse greedy solution.

## Rights

The parent dataset remains `license: other`. This release does not grant new
rights to the source catalogue or broaden existing reuse terms. No raw tag
strings, product titles, merchant identifiers or source product URLs are added.
Public numeric taxonomy paths and the existing anonymous tag/product IDs are
unchanged. Contact Greg Baker through the dataset repository for reuse terms.
"""


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_helpers(source_dir):
    text = "from __future__ import annotations\nfrom collections import Counter, defaultdict\nfrom fractions import Fraction\nimport hashlib\nimport json\nimport time\nimport numpy as np\nfrom .published_zubarev import MahlerDesign\n\n"
    for filename, names in HELPERS.items():
        source = (source_dir / filename).read_text()
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        for name in names:
            text += ast.get_source_segment(source, functions[name]) + "\n\n"
    return text


def build_algorithms(source_dir, root):
    target = root / "algorithms"
    target.mkdir()
    (target / "__init__.py").write_text('"""Pure frozen numerical algorithms; no database dependency."""\n')
    for name in ("published_mihara.py", "published_zubarev.py", "randomised_linear.py"):
        shutil.copy2(source_dir / name, target / name)
    (target / "helpers.py").write_text(extract_helpers(source_dir))
    shutil.copy2(source_dir / "followup_replication.py", root / "followup_replication.py")


def download_reference(root):
    base = f"https://huggingface.co/datasets/gregb/product-taxonomy-bench/resolve/{REFERENCE_COMMIT}/submission/2026-09-06/"
    directory = root / "reference"
    directory.mkdir()
    content = urllib.request.urlopen(base + "manifest.json", timeout=60).read()
    assert hashlib.sha256(content).hexdigest() == REFERENCE_MANIFEST_SHA256
    (directory / "manifest.json").write_bytes(content)
    manifest = json.loads(content)

    def download(item):
        name, digest = item
        path = directory / name
        assert path.resolve().is_relative_to(directory.resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
        content = urllib.request.urlopen(base + name, timeout=90).read()
        assert hashlib.sha256(content).hexdigest() == digest, name
        path.write_bytes(content)

    with ThreadPoolExecutor(max_workers=6) as executor:
        list(executor.map(download, manifest["sha256"].items()))
    return manifest


def main():
    from . import db
    from .paper_published_methods import _snapshot_digest
    from .paper_revision_experiments import _load_paper_dataset
    import numpy as np
    from psycopg.types.json import Jsonb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    repository = source.parent
    reference = download_reference(root)
    build_algorithms(source, root)
    sys.path.insert(0, str(root))
    runtime = importlib.import_module("followup_replication")
    matrix, targets, folds, names, digest = runtime.load_matrix(root)
    with db.get_connection() as conn:
        dataset = _load_paper_dataset(conn, snapshot_ref=SNAPSHOT, schema="padjective")
        assert digest == _snapshot_digest(dataset) == DIGEST
        assert names == dataset.feature_names and matrix.shape == dataset.features.shape == (6693, 2542)
        assert (matrix != dataset.features).nnz == 0
        assert np.array_equal(targets, [r.encoded_path for r in dataset.records])
        assert np.array_equal(folds, [r.cv_fold for r in dataset.records])
        assert len(set(targets)) == 363
        public_rows = [row for path in (root / "reference/paper").glob("products-*.jsonl.gz")
                       for row in runtime.read_jsonl(path)]
        stale_counts = sum(row["tag_count"] != len(row["tag_features"]) for row in public_rows)
        assert stale_counts == 11
        parity = dict(status="passed", snapshot_id=SNAPSHOT, snapshot_digest=digest,
                      rows=len(targets), ordered_features=len(names), paths=len(set(targets)),
                      folds={str(f): int(np.sum(folds == f)) for f in range(5)},
                      nonzero_entries=int(matrix.nnz), differing_matrix_entries=0,
                      stale_tag_count_metadata_rows=stale_counts,
                      metadata_scope="tag_count is not used by either numerical loader; actual feature lists are unchanged",
                      row_order_equal=True, feature_order_equal=True, labels_equal=True, folds_equal=True,
                      reference_commit=REFERENCE_COMMIT, reference_manifest_sha256=REFERENCE_MANIFEST_SHA256)
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_replication_release_checks (
                release_name TEXT NOT NULL, check_name TEXT NOT NULL, checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                evidence JSONB NOT NULL, PRIMARY KEY(release_name,check_name)
                ) TABLESPACE pg_default""")
            cur.execute("INSERT INTO padjective.paper_replication_release_checks (release_name,check_name,evidence) VALUES (%s,%s,%s)",
                        ("paper-submission-2026-09-10", "data-parity", Jsonb(parity)))
    (root / "data-parity.json").write_text(json.dumps(parity, indent=2)+"\n")
    (root / "evidence").mkdir()
    for original, filename in EVIDENCE.items():
        shutil.copy2(repository / original, root / "evidence" / filename)
    for filename in ("requirements.txt", ".python-version"):
        shutil.copy2(root / "reference" / filename, root / filename)
    (root / "README.md").write_text(README)
    source_files = sorted(set(HELPERS) | {"published_mihara.py", "published_zubarev.py", "randomised_linear.py",
                                        "followup_replication.py", "paper_followup_release.py"})
    manifest = dict(release="paper-submission-2026-09-10", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip(),
        snapshot_id=SNAPSHOT, snapshot_digest=DIGEST, python_version=platform.python_version(),
        versions=reference["versions"], numerical_threads=1, reference_numerical_threads=12,
        reference_commit=REFERENCE_COMMIT, reference_tag_commit=REFERENCE_TAG_COMMIT,
        source_sha256={name: checksum(source/name) for name in source_files},
        extracted_helpers=HELPERS, validation_status="pending",
        sha256={str(path.relative_to(root)): checksum(path) for path in sorted(root.rglob("*"))
                if path.is_file() and "__pycache__" not in path.parts})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(dict(event="release_built", directory=str(root), parity=parity)), flush=True)


if __name__ == "__main__":
    main()
