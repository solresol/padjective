"""Offline replication of the September 8/10 paper experiments.

Distributed at the release root alongside the database-free algorithms package.
Operational product ingestion remains PostgreSQL-only. This consumer reads the
already-public frozen research export; it never contacts a database or network.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy import sparse
from threadpoolctl import threadpool_limits

from algorithms.helpers import (
    compare_scores, direct_certificate, direct_linear, exact_rank_certificate,
    exact_scores, fingerprint, inclusion_certificate, nested_consensus,
    reference_predictions, training_feature_order,
)
from algorithms.published_mihara import (
    ResourceBudget, fit_published_mihara, predict_published_mihara,
)
from algorithms.published_zubarev import GibbsProblem, MahlerDesign, fit_published_zubarev
from algorithms.randomised_linear import (
    apply_default, consensus_predictions, fit_coordinates, fit_default, linear_predictions,
)


def read_jsonl(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def load_matrix(root):
    """Match the PostgreSQL loader's row, tag-rank and within-row ordering."""
    directory = Path(root) / "reference/paper"
    tags = sorted(read_jsonl(directory / "tags.jsonl.gz"), key=lambda r: r["tag_rank"])
    names = tuple(r["tag_id"] for r in tags)
    assert len(names) == len(set(names)) and len({r["tag_rank"] for r in tags}) == len(tags)
    products = sorted([row for path in sorted(directory.glob("products-*.jsonl.gz"))
                       for row in read_jsonl(path)], key=lambda r: r["product_id_hash"])
    assert len({r["product_id_hash"] for r in products}) == len(products)
    index = {name: i for i, name in enumerate(names)}
    rows, columns, targets, folds, ordered = [], [], [], [], []
    for i, row in enumerate(products):
        features = sorted(item["tag_id"] for item in row["tag_features"])
        assert len(features) == len(set(features)) == row["tag_count"]
        path = [int(d) for d in row["taxonomy_path"].split(".")]
        assert all(0 < d < 71 for d in path) and len(path) <= 7
        target = sum(d * 71**j for j, d in enumerate(path))
        targets.append(target)
        folds.append(row["cv_fold"])
        ordered.append([row["product_id_hash"], row["cv_fold"], target, features])
        for feature in features:
            rows.append(i)
            columns.append(index[feature])
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.int64), (rows, columns)),
                               shape=(len(products), len(names)))
    digest = hashlib.sha256()
    for row in ordered:
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return matrix, np.asarray(targets, dtype=np.int64), np.asarray(folds), names, digest.hexdigest()


def verify_release(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release checksum mismatch: {name}")
    if platform.python_version() != manifest["python_version"]:
        raise ValueError("Use the pinned Python version for a reference replication")
    for name, expected in manifest["versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError(f"Dependency mismatch: {name}; install requirements.txt")
    return manifest


def initialise(root):
    global DATA, ROOT
    ROOT = Path(root)
    DATA = load_matrix(ROOT)


def linear_case(reference):
    started = time.monotonic()
    matrix, y, folds, _, _ = DATA
    fold, seed = reference["fold"], reference["seed"]
    train, test = folds != fold, folds == fold
    x_train, x_test = matrix[train], matrix[test]
    with threadpool_limits(limits=1):
        fit = fit_coordinates(x_train, y[train], p=71, precision=7, seed=seed,
                              max_sweeps=100, seconds=300)
        assert fit.status == reference["status"] == "coordinate_optimum", fit.status
        weights = fit.coefficients.tolist()
        assert fingerprint(weights) == reference["coefficient_sha256"], (fold, seed, "coefficients")
        train_raw = direct_linear(x_train, weights, 71**7)
        test_raw = direct_linear(x_test, weights, 71**7)
        assert train_raw == linear_predictions(x_train, weights, 71, 7).tolist()
        assert test_raw == linear_predictions(x_test, weights, 71, 7).tolist()
        default = fit_default(y[train], train_raw, p=71, precision=7)
        train_pred, test_pred = apply_default(train_raw, default), apply_default(test_raw, default)
        assert fingerprint(test_pred.tolist()) == reference["prediction_sha256"]
        for target, prediction, part in ((y[train], train_raw, "training_raw"),
                (y[test], test_raw, "held_out_raw"), (y[train], train_pred, "training"),
                (y[test], test_pred, "held_out")):
            compare_scores(exact_scores(target.tolist(), list(map(int, prediction)), 71, 7), reference["metrics"][part])
        certified = direct_certificate(x_train, y[train], weights, train_raw, 71, 7)
        assert certified == reference["certified_coordinates"]
        active = float(np.asarray(x_test @ (fit.coefficients != 0).astype(np.int64)).mean())
        assert abs(active-reference["metrics"]["mean_nonzero_terms_consulted"]) < 2e-12
        assert int(np.count_nonzero(fit.coefficients)) == reference["metrics"]["stored_nonzero_coefficients"]
        fallback = float(np.mean(np.asarray(test_raw) == 0))
        assert abs(fallback-reference["metrics"]["mean_fallback_uses"]) < 2e-12
    return dict(kind="linear", fold=fold, seed=seed, status=fit.status,
                coefficient_sha256=fingerprint(weights), prediction_sha256=fingerprint(test_pred.tolist()),
                certified_coordinates=certified, elapsed_seconds=time.monotonic()-started,
                predictions=test_pred.tolist())


def published_case(reference):
    started = time.monotonic()
    matrix, y, folds, names, _ = DATA
    fold = reference["fold"]
    train, test = folds != fold, folds == fold
    dense = matrix.toarray()
    order = training_feature_order(dense[train], names)
    if reference["max_tags"]:
        order = order[:reference["max_tags"]]
    x_train, x_test = dense[train][:, order], dense[test][:, order]
    result = dict(kind=reference["method"], fold=fold, run_id=reference["run_id"])
    with threadpool_limits(limits=1):
        if reference["method"] == "zubarev":
            designs = [MahlerDesign.build(x, p=71, precision=7, degree=reference["degree"])
                       for x in (x_train, x_test)]
            problem = GibbsProblem.create(designs[0], y[train])
            fit = fit_published_zubarev(problem, seed=reference["seed"], initialisation="random",
                                       betas=(0, 4, 16, 64, 256), draws_per_beta=64,
                                       proposals_per_beta=200000, seconds=300)
            assert fit.status == reference["status"] == "schedule_complete", fit.status
            assert fit.accepted_transitions == reference["accepted_transitions"] == 320
            assert fingerprint(fit.coefficients.tolist()) == reference["coefficient_sha256"]
            for design, target, part in zip(designs, (y[train], y[test]), ("training", "held_out")):
                predictions = reference_predictions(design, fit.coefficients.tolist())
                assert predictions == design.predict(fit.coefficients).tolist()
                compare_scores(exact_scores(target.tolist(), predictions, 71, 7), reference["metrics"][part])
            assert fingerprint(predictions) == reference["prediction_sha256"]
            result.update(coefficient_sha256=fingerprint(fit.coefficients.tolist()),
                          prediction_sha256=fingerprint(predictions), status=fit.status)
        else:
            rank = exact_rank_certificate(x_train, 71, seconds=120)
            for key, value in reference["rank_certificate"].items():
                if key != "elapsed_seconds":
                    assert rank[key] == value, (key, rank[key], value)
            if reference["rank_reduce"]:
                selected = rank["independent_feature_indices"]
                assert len(selected) == reference["selected_features"]
                x_train, x_test = x_train[:, selected], x_test[:, selected]
            inclusion = inclusion_certificate(x_train, y[train], 71)
            assert inclusion == reference["inclusion_certificate"]
            if rank["status"] == "rank_obstruction" and not reference["rank_reduce"]:
                status = "rank_obstruction"
            elif inclusion["status"] == "inclusion_obstruction":
                status = "inclusion_obstruction"
            else:
                fit = fit_published_mihara(x_train, y[train], p=71, precision=7, rep=3,
                                          seed=reference["seed"],
                                          budget=ResourceBudget(max_draws=1000000, seconds=180))
                status = fit.status
                result["fit"] = {k: v for k, v in asdict(fit).items() if k != "coefficients"}
                # Wall-clock-limited draw/restart counts are not reproducibility targets.
                assert fit.completed_digits == reference["fit"]["completed_digits"]
                if fit.completed:
                    predictions = predict_published_mihara(fit, x_test, p=71)
                    compare_scores(exact_scores(y[test].tolist(), predictions, 71, 7), reference["metrics"]["held_out"])
            assert status == reference["status"], status
            result.update(status=status, rank_certificate={k: v for k, v in rank.items()
                          if k != "independent_feature_indices"}, inclusion_certificate=inclusion)
    result["elapsed_seconds"] = time.monotonic()-started
    return result


def verify_ensembles(root, cases, selected_folds, count):
    matrix, y, folds, _, _ = DATA
    expected = json.loads((root / "evidence/ensemble-results.json").read_text())
    reference = {(r["fold"], r["roster"], r["members"]): r for r in expected["ensembles"]}
    sizes = [n for n in expected["sizes"] if n <= count]
    members = {(r["fold"], r["seed"]): r for r in cases}
    checks = []
    for fold in selected_folds:
        bank = np.array([members[fold, base+fold]["predictions"]
                         for base in expected["seed_bases"][:count]], dtype=np.int64).T
        options = np.unique(y[folds != fold])
        orders = [np.arange(count)]
        if count == 243:
            orders += [np.random.default_rng(20260910+r*100+fold).permutation(count) for r in range(1, 21)]
        for roster, size, predictions in nested_consensus(bank, options, sizes, orders):
            row = reference[fold, roster, size]
            assert fingerprint(options.tolist()) == row["candidate_sha256"]
            assert fingerprint(predictions.tolist()) == row["prediction_sha256"]
            compare_scores(exact_scores(y[folds == fold].tolist(), predictions.tolist(), 71, 7), row["metrics"])
            if roster == 0:
                direct, work = consensus_predictions(bank[:, :size], options, p=71, precision=7)
                assert np.array_equal(predictions, direct) and work == row["consensus_work"]
            checks.append(dict(fold=fold, roster=roster, members=size, prediction_sha256=row["prediction_sha256"]))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--suite", choices=("ensembles", "published"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--members", type=int, default=243, help="Ensemble smoke-test prefix; full replication uses 243")
    args = parser.parse_args()
    if not 1 <= args.members <= 243 or not args.folds or len(set(args.folds)) != len(args.folds) or not set(args.folds) <= set(range(5)):
        parser.error("Unique folds 0..4 and 1..243 members required")
    root = args.root.resolve()
    manifest = verify_release(root)
    initialise(root)
    assert DATA[-1] == manifest["snapshot_digest"]
    assert DATA[0].shape == (6693, 2542) and len(np.unique(DATA[1])) == 363
    if args.suite == "ensembles":
        evidence = json.loads((root / "evidence/ensemble-results.json").read_text())
        bases = set(evidence["seed_bases"][:args.members])
        records = [r for r in evidence["models"] if r["fold"] in args.folds and r["seed"]-r["fold"] in bases]
        assert len(records) == args.members*len(args.folds)
        function = linear_case
    else:
        evidence = json.loads((root / "evidence/published-results.json").read_text())
        records = [r for r in evidence["runs"] if r["fold"] in args.folds]
        assert len(records) == 13*len(args.folds)
        function = published_case
    context = dict(suite=args.suite, folds=args.folds, members=args.members,
                   manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest())
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "context.json"
    if context_path.exists():
        assert json.loads(context_path.read_text()) == context, "Resume context mismatch"
    else:
        assert not list(output.iterdir()), "Use an empty output directory"
        context_path.write_text(json.dumps(context, indent=2)+"\n")
    cases, pending = [], []
    for row in records:
        path = output / (row["run_id"] + ".json")
        if path.exists():
            result = json.loads(path.read_text())
            if function is linear_case:
                assert fingerprint(result["predictions"]) == row["prediction_sha256"]
            for key in ("status", "coefficient_sha256", "prediction_sha256"):
                if key in row:
                    assert result[key] == row[key]
            cases.append(result)
        else:
            pending.append((row, path))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialise, initargs=(str(root),)) as executor:
        futures = {executor.submit(function, row): path for row, path in pending}
        for future in as_completed(futures):
            result = future.result()
            futures[future].write_text(json.dumps(result, indent=2)+"\n")
            cases.append(result)
            print(json.dumps(dict(event="case_verified", completed=len(cases), expected=len(records),
                                  kind=result["kind"], fold=result["fold"])), flush=True)
    ensembles = verify_ensembles(root, cases, args.folds, args.members) if function is linear_case else []
    report = dict(context=context, status="passed", full_grid=set(args.folds)==set(range(5))
                  and (args.suite=="published" or args.members==243), cases=len(cases),
                  ensemble_rows=len(ensembles), ensemble_checks=ensembles,
                  results=[{k: v for k, v in row.items() if k != "predictions"} for row in cases])
    (output / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(event="validation_complete", status="passed", cases=len(cases),
                         ensembles=len(ensembles), full_grid=report["full_grid"])), flush=True)


if __name__ == "__main__":
    main()
