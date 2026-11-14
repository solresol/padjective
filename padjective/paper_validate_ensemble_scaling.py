"""Independently validate the extended bank and its fixed nested ensembles."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .paper_ensemble_scaling_batch import BANK_SEEDS, BASE_BATCH, BASE_COMMIT, SIZES
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import compare_scores, exact_scores
from .paper_validate_randomised import (
    direct_certificate, direct_linear, direct_medoid, distance_matrix, fingerprint,
)
from .randomised_linear import consensus_predictions


def nested_consensus(predictions, candidates, sizes, orders, p=71, precision=7):
    """Exact cumulative candidate costs, with no products x members x paths array.

    Orders must be permutations of the complete bank. Output is a generator so
    callers can persist one roster at a time. Smallest numeric candidate wins ties.
    """
    bank = np.asarray(predictions, dtype=np.int64)
    assert bank.ndim == 2 and bank.shape[0] and bank.shape[1]
    assert len(sizes) == len(set(sizes)) and all(1 <= n <= bank.shape[1] for n in sizes)
    options = np.unique(candidates)
    q = p**precision
    assert len(options) and bank.shape[1]*q <= np.iinfo(np.int64).max
    assert np.all((bank >= 0) & (bank < q)) and np.all((options >= 0) & (options < q))
    values, inverse = np.unique(bank, return_inverse=True)
    indices = inverse.reshape(bank.shape)
    lookup = distance_matrix(values, options, p, precision)
    for roster, order in enumerate(orders):
        assert sorted(order) == list(range(bank.shape[1]))
        costs = np.zeros((bank.shape[0], len(options)), dtype=np.int64)
        for size, member in enumerate(order, start=1):
            costs += lookup[indices[:, member]]
            if size in sizes:
                yield roster, size, options[np.argmin(costs, axis=1)]


def validate_member(row, context, source_commit, batch_id):
    """Reconstruct all final predictions and certify every supported coordinate."""
    fold = row["cv_fold"]
    x_train, x_test, y_train, y_test, names = context[fold]
    c, e, metrics = row["configuration"], row["evidence"], row["metrics"]["coordinate"]
    base = c["seed"]-fold
    reused = base in BANK_SEEDS[:9]
    assert base in BANK_SEEDS and fold in range(5)
    assert str(row["batch_id"]) == (BASE_BATCH if reused else batch_id)
    assert c["source_commit"] == (BASE_COMMIT if reused else source_commit)
    assert row["job_key"] == c["job_key"] == f"linear_random-f{fold}-s{base}"
    assert c["batch_id"] == str(row["batch_id"]) and c["fold"] == fold
    assert row["method"] == c["method"] == "linear_random"
    assert not row["pilot"] and not c["pilot"]
    assert str(row["snapshot_id"]) == PAPER_SNAPSHOT and c["precision"] == 7
    assert c["max_sweeps"] == 100 and c["seconds"] == 300
    assert row["status"] == "coordinate_optimum" and row["finished_at"] is not None
    assert e["snapshot_digest"] == SNAPSHOT_DIGEST and e["feature_names"] == names
    assert e["n_train"] == len(y_train) and e["n_test"] == len(y_test)
    assert e["n_features"] == x_train.shape[1] and e["p"] == 71 and e["precision"] == 7
    history = e["fit"]["history"]
    assert history[-1]["complete"] and history[-1]["changes"] == 0
    assert all(h["complete"] for h in history)
    assert all(a["training_loss_units"] >= b["training_loss_units"] for a, b in zip(history, history[1:]))
    model = e["models"]["coordinate"]
    weights = model["coefficients"]
    q = 71**7
    assert model["completed"] and len(weights) == x_train.shape[1]
    assert all(0 <= w < q for w in weights)
    train_raw = direct_linear(x_train, weights, q)
    test_raw = direct_linear(x_test, weights, q)
    assert train_raw == model["training_raw_predictions"]
    assert test_raw == model["held_out_raw_predictions"]
    zero = np.asarray(train_raw) == 0
    if np.any(zero):
        default, _ = direct_medoid(y_train[zero], 71, 7)
    else:
        counts = Counter(y_train.tolist())
        default = min(counts, key=lambda v: (-counts[v], v))
    assert default == model["default"]
    train_pred = [v if v else default for v in train_raw]
    test_pred = [v if v else default for v in test_raw]
    assert train_pred == model["training_predictions"] and test_pred == model["predictions"]
    for actual, predicted, part in ((y_train, train_raw, "training_raw"),
            (y_test, test_raw, "held_out_raw"), (y_train, train_pred, "training"),
            (y_test, test_pred, "held_out")):
        compare_scores(exact_scores(actual.tolist(), predicted, 71, 7), metrics[part])
    units = int(distance_matrix((y_train-train_raw) % q, [0], 71, 7).sum())
    assert units == e["fit"]["final_units"] == history[-1]["training_loss_units"]
    certified = direct_certificate(x_train, y_train, weights, train_raw, 71, 7)
    active = float(np.asarray(x_test @ np.array([w != 0 for w in weights], dtype=np.int64)).mean())
    fallback = sum(v == 0 for v in test_raw) / len(test_raw)
    assert abs(active-metrics["mean_nonzero_terms_consulted"]) < 1e-12
    assert abs(fallback-metrics["mean_fallback_uses"]) < 1e-12
    assert sum(w != 0 for w in weights) == metrics["stored_nonzero_coefficients"]
    export = dict(run_id=str(row["run_id"]), origin_batch=str(row["batch_id"]),
        fold=fold, seed=c["seed"], reused=reused, status=row["status"], metrics=metrics,
        fit_seconds=e["fit"]["elapsed_seconds"], sweeps=len(history),
        certified_coordinates=certified, coefficient_sha256=fingerprint(weights),
        prediction_sha256=fingerprint(test_pred))
    return export, test_pred


def ensure_storage(conn):
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_ensemble_scaling_checks (
            batch_id UUID NOT NULL, run_id UUID NOT NULL, validator_commit TEXT NOT NULL,
            checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), evidence JSONB NOT NULL,
            PRIMARY KEY(batch_id,run_id)) TABLESPACE pg_default""")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_ensemble_scaling_results (
            batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, roster INTEGER NOT NULL,
            members INTEGER NOT NULL, evidence JSONB NOT NULL,
            PRIMARY KEY(batch_id,cv_fold,roster,members)) TABLESPACE pg_default""")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_ensemble_scaling_reports (
            batch_id UUID PRIMARY KEY, validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            report JSONB NOT NULL) TABLESPACE pg_default""")
    conn.commit()


def validate_bank(conn, dataset, batch_id, source_commit, validator_commit, wait_seconds):
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST and dataset.prime_base == 71
    folds = np.array([r.cv_fold for r in dataset.records])
    y = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    context = {f: (dataset.features[folds != f], dataset.features[folds == f],
        y[folds != f], y[folds == f], list(dataset.feature_names)) for f in range(5)}
    expected = {(f, base+f) for f in range(5) for base in BANK_SEEDS}
    members = {}
    deadline = time.monotonic()+wait_seconds
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT report FROM padjective.paper_randomised_validations WHERE batch_id=%s", (BASE_BATCH,))
        original = cur.fetchone()
    assert original and original["report"]["validation"]["status"] == "passed"
    while len(members) < len(expected):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("""SELECT run_id,status,cv_fold,configuration->>'seed' AS seed
                FROM padjective.paper_randomised_method_runs
                WHERE batch_id=ANY(%s::uuid[]) AND method='linear_random' AND NOT pilot""", ([BASE_BATCH, batch_id],))
            inventory = cur.fetchall()
        conn.commit()
        keys = [(r["cv_fold"], int(r["seed"])) for r in inventory]
        assert len(keys) == len(set(keys)) and set(keys) <= expected
        assert all(r["status"] in {"running", "coordinate_optimum"} for r in inventory)
        available = [r for r in inventory if r["status"] == "coordinate_optimum"
                     and (r["cv_fold"], int(r["seed"])) not in members]
        for record in available:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("SELECT * FROM padjective.paper_randomised_method_runs WHERE run_id=%s", (record["run_id"],))
                row = cur.fetchone()
                cur.execute("""SELECT validator_commit,evidence FROM padjective.paper_ensemble_scaling_checks
                    WHERE batch_id=%s AND run_id=%s""", (batch_id, record["run_id"]))
                saved = cur.fetchone()
            if saved:
                assert saved["validator_commit"] == validator_commit
                export, predictions = saved["evidence"], row["evidence"]["models"]["coordinate"]["predictions"]
                assert export["prediction_sha256"] == fingerprint(predictions)
                assert export["coefficient_sha256"] == fingerprint(row["evidence"]["models"]["coordinate"]["coefficients"])
            else:
                export, predictions = validate_member(row, context, source_commit, batch_id)
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO padjective.paper_ensemble_scaling_checks
                        (batch_id,run_id,validator_commit,evidence) VALUES (%s,%s,%s,%s)""",
                        (batch_id, row["run_id"], validator_commit, Jsonb(export)))
            conn.commit()
            members[(row["cv_fold"], row["configuration"]["seed"])] = export, predictions
            if len(members) % 25 == 0 or len(members) == len(expected):
                print(json.dumps(dict(event="validated_members", count=len(members), expected=len(expected))), flush=True)
        if not available:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Only {len(members)}/{len(expected)} members available")
            time.sleep(min(15, max(0, deadline-time.monotonic())))
    assert set(members) == expected
    return members, context


def form_ensembles(conn, batch_id, members, context):
    output = []
    for fold in range(5):
        _, _, y_train, y_test, _ = context[fold]
        options = np.unique(y_train)
        records = [members[(fold, base+fold)][0] for base in BANK_SEEDS]
        bank = np.array([members[(fold, base+fold)][1] for base in BANK_SEEDS], dtype=np.int64).T
        orders = [np.arange(len(BANK_SEEDS))] + [np.random.default_rng(20260910+r*100+fold).permutation(len(BANK_SEEDS))
                                               for r in range(1, 21)]
        final = None
        for roster, size, predicted in nested_consensus(bank, options, SIZES, orders):
            selected = [records[i] for i in orders[roster][:size]]
            work = dict(members=size, candidate_paths=len(options),
                candidate_prefix_evaluations_per_prediction=7*len(options),
                member_prefix_observations_per_prediction=7*size)
            elapsed = None
            if roster == 0:
                started = time.monotonic()
                reference, prefix_work = consensus_predictions(bank[:, :size], options, p=71, precision=7)
                elapsed = time.monotonic()-started
                assert np.array_equal(reference, predicted) and work == prefix_work
                if size <= 9:
                    with conn.cursor(row_factory=dict_row) as cur:
                        cur.execute("""SELECT evidence FROM padjective.paper_randomised_ensemble_runs
                            WHERE batch_id=%s AND family='linear_random/coordinate' AND cv_fold=%s AND members=%s""",
                            (BASE_BATCH, fold, size))
                        old = cur.fetchone()
                    assert old and old["evidence"]["predictions"] == predicted.tolist()
            if size == len(BANK_SEEDS):
                if final is None:
                    final = predicted
                assert np.array_equal(final, predicted)
            active = sum(r["metrics"]["mean_nonzero_terms_consulted"] for r in selected)
            fallback = sum(r["metrics"]["mean_fallback_uses"] for r in selected)
            export = dict(fold=fold, roster=roster, members=size,
                metrics=exact_scores(y_test.tolist(), predicted.tolist(), 71, 7),
                mean_member_terms_consulted=active, mean_member_fallback_uses=fallback,
                broader_scoring_proxy=active+fallback+7*len(options)+7*size,
                stored_member_nonzero_coefficients=sum(r["metrics"]["stored_nonzero_coefficients"] for r in selected),
                total_member_fit_seconds=sum(r["fit_seconds"] for r in selected),
                consensus_seconds=elapsed, consensus_work=work,
                member_run_ids=[r["run_id"] for r in selected],
                candidate_sha256=fingerprint(options.tolist()), prediction_sha256=fingerprint(predicted.tolist()))
            private = dict(**export, predictions=predicted.tolist(), candidates=options.tolist())
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO padjective.paper_ensemble_scaling_results
                    (batch_id,cv_fold,roster,members,evidence) VALUES (%s,%s,%s,%s,%s)
                    ON CONFLICT (batch_id,cv_fold,roster,members) DO NOTHING""",
                    (batch_id, fold, roster, size, Jsonb(private)))
                cur.execute("""SELECT evidence->>'prediction_sha256' FROM padjective.paper_ensemble_scaling_results
                    WHERE batch_id=%s AND cv_fold=%s AND roster=%s AND members=%s""", (batch_id, fold, roster, size))
                assert cur.fetchone()[0] == export["prediction_sha256"]
            conn.commit()
            # Full membership is reconstructible from the protocol and is kept in Postgres.
            export.pop("member_run_ids")
            output.append(export)
        print(json.dumps(dict(event="ensemble_fold_complete", fold=fold, rows=21*len(SIZES))), flush=True)
    assert len(output) == 5*21*len(SIZES)
    return output


def summarise(ensembles):
    output = []
    for roster in range(21):
        for size in SIZES:
            rows = [r for r in ensembles if r["roster"] == roster and r["members"] == size]
            assert len(rows) == 5 and {r["fold"] for r in rows} == set(range(5))
            output.append(dict(roster=roster, members=size,
                mean={k: statistics.fmean(r["metrics"][k] for r in rows)
                      for k in ("mean_padic_loss", "exact_accuracy", "first_digit_accuracy")},
                fold_loss_sd=statistics.stdev(r["metrics"]["mean_padic_loss"] for r in rows),
                **{k: statistics.fmean(r[k] for r in rows) for k in (
                    "mean_member_terms_consulted", "mean_member_fallback_uses", "broader_scoring_proxy",
                    "stored_member_nonzero_coefficients", "total_member_fit_seconds")}))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=7200)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh aggregate output path"
    validator_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.monotonic()
    with db.get_connection() as conn:
        ensure_storage(conn)
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
        members, context = validate_bank(conn, dataset, args.batch_id, args.source_commit, validator_commit, args.wait_seconds)
        ensembles = form_ensembles(conn, args.batch_id, members, context)
        models = [v[0] for _, v in sorted(members.items())]
        report = dict(batch_id=args.batch_id, base_batch=BASE_BATCH, snapshot_id=PAPER_SNAPSHOT,
            snapshot_digest=SNAPSHOT_DIGEST, source_commit=args.source_commit, validator_commit=validator_commit,
            seed_bases=BANK_SEEDS, sizes=SIZES, roster_count=21,
            validation=dict(status="passed", members=len(models), new_members=sum(not r["reused"] for r in models),
                reconstructed_training_predictions=sum(r["metrics"]["training"]["n"] for r in models),
                reconstructed_held_out_predictions=sum(r["metrics"]["held_out"]["n"] for r in models),
                independently_certified_coordinates=sum(r["certified_coordinates"] for r in models),
                primary_consensus_predictions_checked=sum(r["metrics"]["n"] for r in ensembles if r["roster"] == 0),
                ensemble_rows=len(ensembles), elapsed_including_wait_seconds=time.monotonic()-started),
            mean_input_active_features=statistics.fmean(float(x_test.getnnz(axis=1).mean())
                for _, x_test, _, _, _ in context.values()),
            feature_count=len(dataset.feature_names), models=models, ensembles=ensembles, summaries=summarise(ensembles))
        with conn.cursor() as cur:
            cur.execute("INSERT INTO padjective.paper_ensemble_scaling_reports (batch_id,report) VALUES (%s,%s)",
                        (args.batch_id, Jsonb(report)))
        conn.commit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(json.dumps(dict(event="validation_complete", validation=report["validation"],
        primary=[r for r in report["summaries"] if r["roster"] == 0])), flush=True)


if __name__ == "__main__":
    main()
