"""Validate stored comparator evidence and export aggregate-only research rows."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from psycopg.rows import dict_row

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_revision_experiments import _load_paper_dataset
from .published_zubarev import MahlerDesign


def exact_scores(targets: list[int], predictions: list[int], p: int, precision: int) -> dict:
    """Independent integer-numerator metric, with no iterative float summation."""
    assert len(targets) == len(predictions) and targets
    q = p**precision
    units = correct = roots = 0
    for target, predicted in zip(targets, predictions, strict=True):
        assert 0 <= target < q and 0 <= predicted < q
        delta = abs(target - predicted)
        correct += delta == 0
        roots += delta % p == 0
        if not delta:
            continue
        exponent = 0
        while delta % p == 0:
            exponent += 1
            delta //= p
        units += p**(precision-exponent)
    n = len(targets)
    return dict(n=n, mean_padic_loss=float(Fraction(units, n*q)),
                exact_accuracy=float(Fraction(correct, n)), first_digit_accuracy=float(Fraction(roots, n)))


def reference_predictions(design: MahlerDesign, coefficients: list[int]) -> list[int]:
    """Python-integer sums, independent of the fitting code's radix/block dot."""
    weights = np.array(coefficients, dtype=object)
    values = []
    for row in design.basis:
        active = np.flatnonzero(row)
        value = sum(int(row[i]) * int(weights[i]) for i in active) % design.modulus
        values.append(value)
    return [values[int(i)] for i in design.row_groups]


def compare_scores(expected: dict, actual: dict) -> None:
    assert expected.keys() == actual.keys()
    for key in expected:
        assert abs(expected[key] - actual[key]) < 2e-12, (key, expected[key], actual[key])


def configuration_key(row: dict) -> tuple:
    c = row["configuration"]
    return (row["method"], row["cv_fold"], c["max_tags"], c["rank_reduce"],
            c["degree"] if row["method"] == "zubarev" else 0, c["seed"])


def grouped_root_error_floor(features: np.ndarray, targets: np.ndarray, p: int) -> dict:
    groups = defaultdict(Counter)
    for values, target in zip(features, targets, strict=True):
        groups[tuple(int(v) for v in values)][int(target) % p] += 1
    correct = sum(max(counts.values()) for counts in groups.values())
    return dict(n=len(targets), groups=len(groups), maximum_root_correct=correct,
                root_error_floor=float(Fraction(len(targets)-correct, len(targets))))


def capacity_bounds(rows: list[dict], dataset) -> dict:
    """Post-hoc oracle bound, NOT a fitted model or a held-out tuning signal.

    K=p^r-1 cannot distinguish binary inputs with the same first r coordinates
    at the root. Allow an oracle to choose the best root value separately for
    every such group of held-out rows: even it cannot beat the reported bound.
    """
    digest = _snapshot_digest(dataset)
    matrix = dataset.features.toarray().astype(np.int64)
    assert np.all((matrix == 0) | (matrix == 1))
    names = {name: i for i, name in enumerate(dataset.feature_names)}
    folds = np.array([row.cv_fold for row in dataset.records])
    targets = np.array([row.encoded_path for row in dataset.records], dtype=np.int64)
    selected_rows = {}
    for row in rows:
        if row["method"] != "zubarev":
            continue
        key = (row["cv_fold"], row["configuration"]["degree"])
        assert row["evidence"]["snapshot_digest"] == digest
        if key in selected_rows:
            assert selected_rows[key]["evidence"]["feature_order"] == row["evidence"]["feature_order"]
        selected_rows[key] = row
    assert set(selected_rows) == {(fold, degree) for fold in range(5) for degree in (70, 5040, 357910)}
    output = []
    for (fold, degree), row in sorted(selected_rows.items()):
        positions = {70: 1, 5040: 2, 357910: 3}[degree]
        assert degree + 1 == 71**positions
        selected = [names[name] for name in row["evidence"]["feature_order"][:positions]]
        test = folds == fold
        bound = grouped_root_error_floor(matrix[test][:, selected], targets[test], 71)
        output.append(dict(fold=fold, degree=degree, first_tag_positions=positions, **bound))
    return dict(snapshot_id=PAPER_SNAPSHOT, snapshot_digest=digest,
        scope="Post-hoc held-out-label oracle bound; not used for training, selection, or stopping",
        validation=dict(status="passed", method="Exact modal counts within identical binary prefix groups"),
        source_commits=sorted({row["configuration"]["source_commit"] for row in rows}),
        bounds=output,
        mean_root_error_floor_by_degree={degree: statistics.fmean(row["root_error_floor"] for row in output if row["degree"] == degree)
                                         for degree in (70, 5040, 357910)})


def validate(rows: list[dict], dataset) -> dict:
    """Require the whole predeclared grid, not a favourable subset of runs."""
    expected = set()
    for fold in range(5):
        for degree in (70, 5040, 357910):
            for base in (42, 1729, 20260907):
                expected.add(("zubarev", fold, 0, False, degree, base+fold))
        for cap in (0, 32, 128):
            expected.add(("mihara", fold, cap, False, 0, 42+fold))
        expected.add(("mihara", fold, 0, True, 0, 42+fold))
    assert len(rows) == 65, f"Expected 65 finished runs, got {len(rows)}"
    assert {configuration_key(row) for row in rows} == expected
    assert len({row["run_id"] for row in rows}) == 65
    digest = _snapshot_digest(dataset)
    matrix = dataset.features.toarray().astype(np.int64)
    features = {name: i for i, name in enumerate(dataset.feature_names)}
    folds = np.array([row.cv_fold for row in dataset.records])
    targets = np.array([row.encoded_path for row in dataset.records], dtype=np.int64)
    checked_predictions = checked_training = 0
    exports = []
    cache_key = None
    designs = None
    for row in sorted(rows, key=lambda row: (row["method"], row["configuration"]["degree"], row["cv_fold"], row["configuration"]["seed"])):
        c, e, m = row["configuration"], row["evidence"], row["metrics"]
        assert str(row["snapshot_id"]) == PAPER_SNAPSHOT and not row["pilot"]
        assert row["status"] not in {"running", "failed"} and row["finished_at"] is not None
        assert e["snapshot_digest"] == digest
        assert c["precision"] == 7 and e["p"] == dataset.prime_base == 71
        train = folds != row["cv_fold"]
        assert e["n_train"] == int(train.sum()) and e["n_test"] == int((~train).sum())
        export = dict(run_id=str(row["run_id"]), method=row["method"], fold=row["cv_fold"],
            status=row["status"], max_tags=c["max_tags"], rank_reduce=c["rank_reduce"],
            seed=c["seed"], degree=c["degree"] if row["method"] == "zubarev" else None,
            n_train=e["n_train"], n_test=e["n_test"], elapsed_seconds=e["elapsed_seconds"], metrics=m)
        if row["method"] == "zubarev":
            assert c["initialisation"] == "random" and c["betas"] == [0, 4, 16, 64, 256]
            assert c["draws_per_beta"] == 64 and c["proposals_per_beta"] == 200000 and c["seconds"] == 300
            assert e["fit"]["initialisation"] == "random"
            fit = e["fit"]
            assert fit["accepted_transitions"] == sum(stage["accepted"] for stage in fit["history"])
            if row["status"] == "schedule_complete":
                assert [stage["beta"] for stage in fit["history"]] == c["betas"]
                assert all(stage["accepted"] == 64 and stage["status"] == "complete" for stage in fit["history"])
                assert fit["accepted_transitions"] == 320
            assert abs(fit["best_loss"] - m["training"]["mean_padic_loss"]) < 2e-12
            key = (row["cv_fold"], c["degree"], tuple(e["feature_order"]))
            if key != cache_key:
                selected = [features[name] for name in e["feature_order"]]
                designs = tuple(MahlerDesign.build(matrix[mask][:, selected], p=71, precision=7,
                                                   degree=c["degree"]) for mask in (train, ~train))
                cache_key = key
            coefficients = fit["coefficients"]
            assert len(coefficients) == c["degree"] + 1 and all(0 <= value < 71**7 for value in coefficients)
            training_predictions = reference_predictions(designs[0], coefficients)
            predictions = reference_predictions(designs[1], coefficients)
            assert predictions == e["predictions"], row["run_id"]
            compare_scores(exact_scores(targets[train].tolist(), training_predictions, 71, 7), m["training"])
            compare_scores(exact_scores(targets[~train].tolist(), predictions, 71, 7), m["held_out"])
            checked_training += len(training_predictions)
            checked_predictions += len(predictions)
            export.update(accepted_transitions=fit["accepted_transitions"], proposals=fit["proposals"],
                full_evaluations=fit["full_evaluations"], root_loss_lower_bound=e["root_loss_lower_bound"],
                design_bytes=e["design"]["bytes"], design_seconds=e["design"]["seconds"],
                fit_seconds=fit["elapsed_seconds"], sampler=e["sampler"],
                coefficient_sha256=hashlib.sha256(json.dumps(coefficients, separators=(",", ":")).encode()).hexdigest(),
                prediction_sha256=hashlib.sha256(json.dumps(predictions, separators=(",", ":")).encode()).hexdigest())
        else:
            assert c["rep"] == 3 and c["seconds"] == 180 and c["max_draws"] == 1000000
            rank = {k: v for k, v in e["rank_certificate"].items() if k != "independent_feature_indices"}
            export.update(rank_certificate=rank, inclusion_certificate=e["inclusion_certificate"])
            if c["rank_reduce"]:
                export["selected_features"] = e["independent_column_preprocessing"]["n_features"]
            if "fit" in e:
                export["fit"] = {k: v for k, v in e["fit"].items() if k != "coefficients"}
            if row["status"] != "precision_complete":
                assert not m and "predictions" not in e
            else:
                # Do not assume no Mihara run can complete: audit one if it does.
                selected_names = e.get("independent_column_preprocessing", {}).get("feature_order", e["feature_order"])
                selected = [features[name] for name in selected_names]
                coefficients = e["fit"]["coefficients"]
                predictions = [(sum(int(v)*int(w) for v, w in zip(values, coefficients[:-1], strict=True))
                                + coefficients[-1]) % (71**7) for values in matrix[~train][:, selected]]
                assert predictions == e["predictions"]
                compare_scores(exact_scores(targets[~train].tolist(), predictions, 71, 7), m["held_out"])
                checked_predictions += len(predictions)
        exports.append(export)
        print(json.dumps(dict(event="validated_run", run_id=str(row["run_id"]), method=row["method"],
                              fold=row["cv_fold"], degree=export["degree"])), flush=True)
    summaries = []
    for degree in (70, 5040, 357910):
        group = [r for r in exports if r["method"] == "zubarev" and r["degree"] == degree]
        by_fold = defaultdict(list)
        by_seed = defaultdict(list)
        for r in group:
            loss = r["metrics"]["held_out"]["mean_padic_loss"]
            by_fold[r["fold"]].append(loss)
            by_seed[r["seed"]-r["fold"]].append(loss)
        fold_means = [statistics.fmean(by_fold[fold]) for fold in range(5)]
        summaries.append(dict(degree=degree, runs=len(group), status_counts=dict(Counter(r["status"] for r in group)),
            mean_padic_loss=statistics.fmean(fold_means), fold_sd_after_seed_average=statistics.stdev(fold_means),
            pooled_padic_loss=sum(r["metrics"]["held_out"]["mean_padic_loss"] * r["n_test"] for r in group) / sum(r["n_test"] for r in group),
            fold_means=fold_means, seed_five_fold_means={seed: statistics.fmean(losses) for seed, losses in by_seed.items()},
            mean_exact_accuracy=statistics.fmean(r["metrics"]["held_out"]["exact_accuracy"] for r in group),
            mean_first_digit_accuracy=statistics.fmean(r["metrics"]["held_out"]["first_digit_accuracy"] for r in group),
            mean_stored_nonzero_coefficients=statistics.fmean(r["metrics"]["stored_nonzero_coefficients"] for r in group),
            mean_nonzero_terms_consulted=statistics.fmean(r["metrics"]["mean_nonzero_terms_consulted"] for r in group),
            mean_elapsed_seconds=statistics.fmean(r["elapsed_seconds"] for r in group),
            min_elapsed_seconds=min(r["elapsed_seconds"] for r in group), max_elapsed_seconds=max(r["elapsed_seconds"] for r in group)))
    return dict(snapshot_id=PAPER_SNAPSHOT, snapshot_digest=digest, n_products=len(dataset.records),
        n_features=len(dataset.feature_names), p=71, precision=7, run_count=len(rows),
        source_commits=sorted({r["configuration"]["source_commit"] for r in rows}),
        validation=dict(status="passed", independently_reconstructed_held_out_predictions=checked_predictions,
            independently_reconstructed_training_predictions=checked_training,
            metric_tolerance=2e-12, polynomial_arithmetic="Python-integer dot products; separately tested modular Mahler basis"),
        zubarev_summaries=summaries, runs=exports)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-only", action="store_true", help="Export a separately labelled post-hoc oracle bound")
    args = parser.parse_args()
    with db.get_connection() as conn:
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""SELECT * FROM padjective.paper_published_method_runs
                WHERE NOT pilot AND configuration->>'source_commit'=%s ORDER BY started_at""", (args.source_commit,))
            rows = cur.fetchall()
    result = capacity_bounds(rows, dataset) if args.capacity_only else validate(rows, dataset)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(event="validation_complete", output=str(args.output),
        sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(), **result["validation"])), flush=True)


if __name__ == "__main__":
    main()
