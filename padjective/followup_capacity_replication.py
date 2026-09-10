"""Recompute the separate post-hoc representation bound from the public matrix.

This is an oracle bound using held-out labels after fitting, not a trained
predictor or a signal for selecting degrees, coefficients or stopping points.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics


def prefix_bound(features, targets):
    counts = defaultdict(Counter)
    for row, target in zip(features, targets, strict=True):
        counts[tuple(map(int, row))][int(target) % 71] += 1
    correct = sum(max(group.values()) for group in counts.values())
    n = len(targets)
    return dict(n=n, groups=len(counts), maximum_root_correct=correct,
                root_error_floor=float(Fraction(n-correct, n)))


def verify_capacity(root):
    from followup_replication import load_matrix, training_feature_order

    matrix, targets, folds, names, digest = load_matrix(root)
    matrix = matrix.toarray()
    path = root / "evidence/published-capacity-bounds.json"
    original = json.loads(path.read_text())
    assert digest == original["snapshot_digest"]
    expected = {(r["fold"], r["degree"]): r for r in original["bounds"]}
    assert len(expected) == len(original["bounds"]) == 15
    rows = []
    for fold in range(5):
        order = training_feature_order(matrix[folds != fold], names)
        for positions in (1, 2, 3):
            degree = 71**positions-1
            values = matrix[folds == fold][:, order[:positions]]
            row = dict(fold=fold, degree=degree, first_tag_positions=positions,
                       **prefix_bound(values, targets[folds == fold]))
            assert row == expected[fold, degree]
            rows.append(row)
    means = {str(degree): statistics.fmean(r["root_error_floor"] for r in rows if r["degree"] == degree)
             for degree in (70, 5040, 357910)}
    assert means == original["mean_root_error_floor_by_degree"]
    return dict(status="passed", snapshot_digest=digest, checked_bounds=len(rows),
                scope=original["scope"], bounds=rows, mean_root_error_floor_by_degree=means,
                reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    from followup_replication import verify_release

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify_release(args.root)
    assert not args.output.exists(), "Use a new output path"
    report = verify_capacity(args.root)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(status=report["status"], checked_bounds=report["checked_bounds"])), flush=True)


if __name__ == "__main__":
    main()
