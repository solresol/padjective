"""Independently check randomised fits, then form the fixed consensus grid.

Private predictions stay in Postgres. Only aggregate evidence is exported.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_batch import jobs
from .paper_randomised_methods import SEED_BASES, SNAPSHOT_DIGEST, supported_order
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import compare_scores, exact_scores, reference_predictions
from .published_zubarev import MahlerDesign
from .randomised_linear import consensus_predictions


def fingerprint(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def distance_matrix(left, right, p, precision):
    """Direct pairwise integer distances, not the fitter's prefix-count trick."""
    a, b = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    delta = np.abs(a[:, None] - b[None, :])
    costs = np.full(delta.shape, p**precision, dtype=np.int64)
    costs[delta == 0] = 0
    for depth in range(1, precision):
        costs[(delta != 0) & (delta % p**depth == 0)] = p**(precision-depth)
    return costs


def direct_medoid(values, p, precision):
    unique, counts = np.unique(values, return_counts=True)
    costs = distance_matrix(unique, unique, p, precision) @ counts
    index = int(np.argmin(costs))
    return int(unique[index]), int(costs[index])


def direct_linear(features, coefficients, modulus):
    x = features.tocsr()
    return [sum(int(coefficients[j]) for j in x.indices[x.indptr[i]:x.indptr[i+1]]) % modulus
            for i in range(x.shape[0])]


def direct_certificate(features, targets, coefficients, predictions, p, precision):
    x, q = features.tocsc(), p**precision
    residual = (np.asarray(targets) - predictions) % q
    examined = 0
    for j in range(x.shape[1]):
        ids = x.indices[x.indptr[j]:x.indptr[j+1]]
        if not len(ids):
            continue
        adjusted = (residual[ids] + coefficients[j]) % q
        _, minimum = direct_medoid(adjusted, p, precision)
        before = int(distance_matrix(residual[ids], [0], p, precision).sum())
        assert minimum >= before, (j, minimum, before)
        examined += 1
    return examined


def direct_consensus(predictions, candidates, p, precision):
    options = np.unique(candidates)
    values, inverse = np.unique(predictions, return_inverse=True)
    indices = inverse.reshape(np.asarray(predictions).shape)
    lookup = distance_matrix(values, options, p, precision)
    return np.array([options[int(np.argmin(lookup[row].sum(axis=0)))] for row in indices])


def validate(rows, dataset, source_commit):
    expected = {name: dict(zip(args[::2], args[1::2])) for name, args in jobs()}
    assert len(rows) == len(expected) == 110
    assert {r["job_key"] for r in rows} == set(expected)
    assert len({str(r["run_id"]) for r in rows}) == 110
    assert len({str(r["batch_id"]) for r in rows}) == 1
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST
    assert dataset.prime_base == 71
    folds = np.array([r.cv_fold for r in dataset.records])
    targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    exports, members = [], {}
    training_checked = held_out_checked = certified_coordinates = 0
    for row in sorted(rows, key=lambda r: r["job_key"]):
        c, e, metrics = row["configuration"], row["evidence"], row["metrics"]
        assert c["source_commit"] == source_commit
        assert not c["pilot"] and not row["pilot"]
        assert str(row["snapshot_id"]) == PAPER_SNAPSHOT and c["precision"] == 7
        assert row["method"] == c["method"] and row["cv_fold"] == c["fold"]
        assert str(row["batch_id"]) == c["batch_id"] and c["job_key"] == row["job_key"]
        for option, value in expected[row["job_key"]].items():
            assert str(c[option[2:].replace("-", "_")]) == value or float(c[option[2:].replace("-", "_")]) == float(value)
        assert row["finished_at"] is not None and row["status"] not in {"failed", "running"}
        assert e["snapshot_digest"] == SNAPSHOT_DIGEST
        assert e["feature_names"] == list(dataset.feature_names)
        p, precision, q = 71, 7, 71**7
        train = folds != row["cv_fold"]
        x_train, x_test = dataset.features[train], dataset.features[~train]
        y_train, y_test = targets[train], targets[~train]
        assert e["n_train"] == len(y_train) and e["n_test"] == len(y_test)
        assert e["p"] == p and e["precision"] == precision and e["n_features"] == x_train.shape[1]
        is_linear = row["method"].startswith("linear")
        if is_linear:
            assert row["status"] == "coordinate_optimum", "Incomplete linear grid: do not form survivor ensembles"
            history = e["fit"]["history"]
            assert history[-1]["complete"] and history[-1]["changes"] == 0
            assert all(h["complete"] for h in history)
            assert all(a["training_loss_units"] >= b["training_loss_units"] for a, b in zip(history, history[1:]))
            assert set(e["models"]) == {"one_pass", "coordinate"}
            designs = None
        else:
            assert row["status"] == "schedule_complete", "Incomplete polynomial grid: do not form survivor ensembles"
            fit = e["fit"]
            assert fit["initialisation"] == "random"
            assert fit["accepted_transitions"] == 320
            assert [h["beta"] for h in fit["history"]] == [0, 4, 16, 64, 256]
            assert all(h["accepted"] == 64 and h["status"] == "complete" for h in fit["history"])
            assert fit["proposals"] == sum(h["proposals"] for h in fit["history"])
            order = supported_order(x_train, dataset.feature_names, c["seed"])
            assert order.tolist() == e["feature_order"]
            designs = [MahlerDesign.build(x[:, order].toarray(), p=p, precision=precision, degree=c["degree"])
                       for x in (x_train, x_test)]
            assert set(e["models"]) == {"polynomial"}
        for label, model in e["models"].items():
            score = metrics[label]
            weights = model["coefficients"]
            assert model["completed"] and all(0 <= w < q for w in weights)
            assert len(weights) == (x_train.shape[1] if is_linear else c["degree"]+1)
            assert score["stored_nonzero_coefficients"] == sum(w != 0 for w in weights)
            if is_linear:
                train_raw = direct_linear(x_train, weights, q)
                test_raw = direct_linear(x_test, weights, q)
                assert train_raw == model["training_raw_predictions"]
                assert test_raw == model["held_out_raw_predictions"]
                zero = np.asarray(train_raw) == 0
                if np.any(zero):
                    default, _ = direct_medoid(y_train[zero], p, precision)
                else:
                    counts = Counter(y_train.tolist())
                    default = min(counts, key=lambda v: (-counts[v], v))
                assert default == model["default"]
                train_pred = [v if v else default for v in train_raw]
                test_pred = [v if v else default for v in test_raw]
                for actual, predicted, part in ((y_train, train_raw, "training_raw"), (y_test, test_raw, "held_out_raw")):
                    compare_scores(exact_scores(actual.tolist(), predicted, p, precision), score[part])
                raw_units = int(distance_matrix((y_train - train_raw) % q, [0], p, precision).sum())
                assert raw_units == history[0 if label == "one_pass" else -1]["training_loss_units"]
                if label == "coordinate":
                    assert raw_units == e["fit"]["final_units"]
                    certified_coordinates += direct_certificate(x_train, y_train, weights, train_raw, p, precision)
                cost = float(np.asarray(x_test @ np.array([w != 0 for w in weights], dtype=np.int64)).mean())
                fallback = sum(v == 0 for v in test_raw) / len(test_raw)
                fit_seconds = history[0]["elapsed_seconds"] if label == "one_pass" else e["fit"]["elapsed_seconds"]
            else:
                train_pred, test_pred = [reference_predictions(d, weights) for d in designs]
                assert abs(e["fit"]["best_loss"] - score["training"]["mean_padic_loss"]) < 2e-12
                used = np.count_nonzero(designs[1].basis[:, np.asarray(weights) != 0], axis=1)
                cost, fallback = float(used[designs[1].row_groups].mean()), 0.0
                fit_seconds = e["fit"]["elapsed_seconds"]
            assert train_pred == model["training_predictions"]
            assert test_pred == model["predictions"]
            assert abs(cost - score["mean_nonzero_terms_consulted"]) < 1e-12
            assert abs(fallback - score["mean_fallback_uses"]) < 1e-12
            for actual, predicted, part in ((y_train, train_pred, "training"), (y_test, test_pred, "held_out")):
                compare_scores(exact_scores(actual.tolist(), predicted, p, precision), score[part])
            training_checked += len(train_pred)
            held_out_checked += len(test_pred)
            family = f"{row['method']}/{label}" if is_linear else f"zubarev_random/k{c['degree']}"
            export = dict(run_id=str(row["run_id"]), job_key=row["job_key"], family=family,
                fold=row["cv_fold"], seed=c["seed"], status=row["status"], metrics=score,
                fit_seconds=fit_seconds, job_elapsed_seconds=e["elapsed_seconds"],
                coefficient_sha256=fingerprint(weights), prediction_sha256=fingerprint(test_pred))
            if is_linear:
                export["sweeps"] = 1 if label == "one_pass" else len(history)
            else:
                export.update(design_bytes=e["design"]["bytes"], design_seconds=e["design"]["seconds"],
                    root_visible_training_coverage=float(np.asarray(x_train[:, order[:3 if c["degree"] == 357910 else 4]].sum(axis=1) > 0).mean()))
            exports.append(export)
            members[(family, row["cv_fold"], c["seed"])] = (export, model)
        print(json.dumps(dict(event="validated_run", job_key=row["job_key"])), flush=True)
    return exports, members, dict(status="passed", runs=110, models=len(exports),
        reconstructed_training_predictions=training_checked, reconstructed_held_out_predictions=held_out_checked,
        independently_certified_coordinates=certified_coordinates)


def form_ensembles(members, dataset):
    folds = np.array([r.cv_fold for r in dataset.records])
    y = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    families = ("linear_random/one_pass", "linear_random/coordinate", "zubarev_random/k357910", "zubarev_random/k357911")
    exports, private = [], []
    for family in families:
        for fold in range(5):
            sizes = (1, 3) if family.endswith("357911") else (1, 3, 9)
            candidates = np.unique(y[folds != fold])
            for size in sizes:
                selected = [members[(family, fold, base+fold)] for base in SEED_BASES[:size]]
                values = np.array([model["predictions"] for _, model in selected], dtype=np.int64).T
                started = time.monotonic()
                predicted, work = consensus_predictions(values, candidates, p=71, precision=7)
                elapsed = time.monotonic()-started
                reference = direct_consensus(values, candidates, 71, 7)
                assert np.array_equal(predicted, reference)
                row = dict(family=family, fold=fold, members=size,
                    member_run_ids=[r["run_id"] for r, _ in selected],
                    seeds=[r["seed"] for r, _ in selected],
                    metrics=exact_scores(y[folds == fold].tolist(), predicted.tolist(), 71, 7),
                    mean_member_terms_consulted=sum(r["metrics"]["mean_nonzero_terms_consulted"] for r, _ in selected),
                    mean_member_fallback_uses=sum(r["metrics"]["mean_fallback_uses"] for r, _ in selected),
                    stored_member_nonzero_coefficients=sum(r["metrics"]["stored_nonzero_coefficients"] for r, _ in selected),
                    total_member_fit_seconds=sum(r["fit_seconds"] for r, _ in selected),
                    consensus_seconds=elapsed, consensus_work=work,
                    candidate_sha256=fingerprint(candidates.tolist()), prediction_sha256=fingerprint(predicted.tolist()),
                    validation="all predictions match exhaustive candidate scoring")
                exports.append(row)
                private.append(dict(**row, predictions=predicted.tolist(), candidates=candidates.tolist()))
    assert len(exports) == 55
    return exports, private


def summarise(exports, ensembles):
    output = []
    groups = defaultdict(list)
    for row in exports:
        groups[(row["family"], 0)].append(row)
        # M=1 uses the first seed, not the mean of all seed runs. Keep its raw
        # counterpart visible so projection and seed choice are not conflated.
        if not row["family"].startswith("linear_association") and row["seed"]-row["fold"] == SEED_BASES[0]:
            groups[(row["family"] + "/first1", 0)].append(row)
        # Matched three-seed control for the boundary-degree sensitivity.
        if row["family"] == "zubarev_random/k357910" and row["seed"]-row["fold"] in SEED_BASES[:3]:
            groups[(row["family"] + "/first3", 0)].append(row)
    for row in ensembles:
        groups[(row["family"], row["members"])].append(row)
    for (family, size), group in sorted(groups.items()):
        folds = defaultdict(list)
        for row in group:
            folds[row["fold"]].append(row["metrics"] if size else row["metrics"]["held_out"])
        fold_scores = {str(fold): {key: statistics.fmean(r[key] for r in records)
            for key in ("mean_padic_loss", "first_digit_accuracy", "exact_accuracy")} for fold, records in sorted(folds.items())}
        output.append(dict(family=family, members=size, observations=len(group), fold_scores=fold_scores,
            mean={key: statistics.fmean(r[key] for r in fold_scores.values()) for key in next(iter(fold_scores.values()))},
            mean_terms_consulted=statistics.fmean(r["mean_member_terms_consulted"] if size else r["metrics"]["mean_nonzero_terms_consulted"] for r in group),
            mean_stored_nonzero_coefficients=statistics.fmean(r["stored_member_nonzero_coefficients"] if size else r["metrics"]["stored_nonzero_coefficients"] for r in group),
            mean_fit_seconds=statistics.fmean(r["total_member_fit_seconds"] if size else r["fit_seconds"] for r in group)))
    paired = []
    for method in ("linear_random", "linear_association"):
        first = {(r["fold"], r["seed"]): r for r in exports if r["family"] == f"{method}/one_pass"}
        final = [r for r in exports if r["family"] == f"{method}/coordinate"]
        differences = [r["metrics"]["held_out"]["mean_padic_loss"] - first[(r["fold"], r["seed"])]["metrics"]["held_out"]["mean_padic_loss"] for r in final]
        paired.append(dict(method=method, pairs=len(differences), improved=sum(v < 0 for v in differences),
            tied=sum(v == 0 for v in differences), worsened=sum(v > 0 for v in differences),
            mean_final_minus_first_loss=statistics.fmean(differences),
            min_sweeps=min(r["sweeps"] for r in final), max_sweeps=max(r["sweeps"] for r in final)))
    decoder = []
    for row in output:
        if row["members"] != 1:
            continue
        raw = next(r for r in output if r["family"] == row["family"] + "/first1")
        decoder.append(dict(family=row["family"], raw_seed42_loss=raw["mean"]["mean_padic_loss"],
            projected_seed42_loss=row["mean"]["mean_padic_loss"],
            projected_minus_raw_loss=row["mean"]["mean_padic_loss"]-raw["mean"]["mean_padic_loss"]))
    return output, paired, decoder


def persist(conn, batch_id, report, private):
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_randomised_ensemble_runs (
            batch_id UUID NOT NULL, family TEXT NOT NULL, cv_fold INTEGER NOT NULL,
            members INTEGER NOT NULL, evidence JSONB NOT NULL,
            PRIMARY KEY(batch_id, family, cv_fold, members)
        ) TABLESPACE pg_default""")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_randomised_validations (
            batch_id UUID PRIMARY KEY, validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            report JSONB NOT NULL
        ) TABLESPACE pg_default""")
        for row in private:
            cur.execute("""INSERT INTO padjective.paper_randomised_ensemble_runs
                (batch_id,family,cv_fold,members,evidence) VALUES (%s,%s,%s,%s,%s)""",
                (batch_id, row["family"], row["fold"], row["members"], Jsonb(row)))
        cur.execute("INSERT INTO padjective.paper_randomised_validations (batch_id,report) VALUES (%s,%s)", (batch_id, Jsonb(report)))
    conn.commit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    conn = db.get_connection()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM padjective.paper_randomised_method_runs WHERE batch_id=%s ORDER BY job_key", (args.batch_id,))
        rows = cur.fetchall()
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    exports, members, validation = validate(rows, dataset, args.source_commit)
    ensembles, private = form_ensembles(members, dataset)
    summaries, paired, decoder = summarise(exports, ensembles)
    validation.update(ensemble_rows=len(ensembles),
        reconstructed_consensus_predictions=sum(row["metrics"]["n"] for row in ensembles),
        elapsed_seconds=time.monotonic()-started)
    report = dict(batch_id=args.batch_id, snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
        source_commit=args.source_commit,
        validator_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        validation=validation, run_statuses=dict(Counter(row["status"] for row in rows)),
        worker_job_seconds=sum(row["evidence"]["elapsed_seconds"] for row in rows),
        first_started_at=min(row["started_at"] for row in rows).isoformat(),
        last_finished_at=max(row["finished_at"] for row in rows).isoformat(),
        summaries=summaries, paired=paired, decoder_controls=decoder, models=exports, ensembles=ensembles)
    persist(conn, args.batch_id, report, private)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    conn.close()
    print(json.dumps(dict(event="validation_complete", validation=validation, summaries=summaries, paired=paired)), flush=True)


if __name__ == "__main__":
    main()
