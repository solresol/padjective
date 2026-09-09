"""Isolated Postgres-backed randomised linear/polynomial comparison jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
import platform
import subprocess
import time
import uuid

import numpy as np
from psycopg.types.json import Jsonb

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import exact_scores
from .published_zubarev import GibbsProblem, MahlerDesign, fit_published_zubarev
from .randomised_linear import (
    apply_default, coordinate_certificate, fit_coordinates, fit_default,
    linear_predictions,
)

SNAPSHOT_DIGEST = "039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222"
SEED_BASES = (42, 1729, 20260907, 104729, 130363, 155921, 181081, 205019, 230003)
METHODS = ("linear_random", "linear_association", "zubarev_random")


def supported_order(features, names, seed: int) -> np.ndarray:
    """Separate RNG stream for polynomial input permutation; no label access."""
    counts = np.asarray(features.sum(axis=0)).reshape(-1)
    present = sorted((i for i, c in enumerate(counts) if c), key=lambda i: names[i])
    absent = sorted((i for i, c in enumerate(counts) if not c), key=lambda i: names[i])
    chosen = np.random.default_rng(seed + 1_000_000).permutation(present).tolist()
    return np.array(chosen + absent, dtype=np.int64)


def association_order(features, targets, names) -> list[int]:
    matrix = features.tocsc()
    scored = []
    for j in range(matrix.shape[1]):
        ids = matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]
        if len(ids):
            counts = Counter(int(targets[i]) for i in ids)
            scored.append((-max(counts.values())/len(ids), -len(ids), names[j], j))
    return [r[-1] for r in sorted(scored)]


def ensure_storage(conn):
    with conn.cursor() as cur:
        # Serialize first-use DDL even when independent job controllers start
        # together. IF NOT EXISTS alone does not prevent pg_type races.
        cur.execute("SELECT pg_advisory_xact_lock(71357911)")
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_randomised_method_runs (
            run_id UUID PRIMARY KEY, batch_id UUID NOT NULL, job_key TEXT NOT NULL,
            snapshot_id UUID NOT NULL, method TEXT NOT NULL, cv_fold INTEGER NOT NULL,
            pilot BOOLEAN NOT NULL, configuration JSONB NOT NULL, status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), finished_at TIMESTAMPTZ,
            metrics JSONB, evidence JSONB, UNIQUE (batch_id, job_key)
        ) TABLESPACE pg_default""")
    conn.commit()


def model_evidence(weights, x_train, x_test, y_train, y_test, *, p, precision,
                   pilot, completed) -> tuple[dict, dict]:
    train_raw = linear_predictions(x_train, weights, p, precision)
    default = fit_default(y_train, train_raw, p=p, precision=precision)
    train_pred = apply_default(train_raw, default)
    record = dict(coefficients=weights.tolist(), default=default,
                  training_raw_predictions=train_raw.tolist(),
                  training_predictions=train_pred.tolist(), completed=completed)
    metrics = dict(training_raw=exact_scores(y_train.tolist(), train_raw.tolist(), p, precision),
                   training=exact_scores(y_train.tolist(), train_pred.tolist(), p, precision),
                   stored_nonzero_coefficients=int(np.count_nonzero(weights)))
    if not pilot:
        test_raw = linear_predictions(x_test, weights, p, precision)
        test_pred = apply_default(test_raw, default)
        record.update(held_out_raw_predictions=test_raw.tolist(), predictions=test_pred.tolist())
        metrics.update(held_out_raw=exact_scores(y_test.tolist(), test_raw.tolist(), p, precision),
                       held_out=exact_scores(y_test.tolist(), test_pred.tolist(), p, precision),
                       mean_nonzero_terms_consulted=float(
                           np.asarray(x_test @ (weights != 0)).mean()),
                       mean_fallback_uses=float(np.mean(test_raw == 0)))
    return record, metrics


def run(args):
    conn = db.get_connection()
    ensure_storage(conn)
    run_id = uuid.uuid4()
    config = vars(args).copy()
    config.update(python=platform.python_version(), numpy=np.__version__,
                  hostname=platform.node(), process_id=os.getpid(),
                  source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO padjective.paper_randomised_method_runs
            (run_id,batch_id,job_key,snapshot_id,method,cv_fold,pilot,configuration,status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'running')""",
            (run_id, args.batch_id, args.job_key, PAPER_SNAPSHOT, args.method,
             args.fold, args.pilot, Jsonb(config)))
    conn.commit()
    started = time.monotonic()
    status, metrics, evidence = "failed", {}, {}
    emit = lambda event, **kw: print(json.dumps(dict(event=event, run_id=str(run_id), **kw)), flush=True)
    emit("started", method=args.method, fold=args.fold, seed=args.seed)
    try:
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
        digest = _snapshot_digest(dataset)
        if digest != SNAPSHOT_DIGEST or dataset.prime_base != 71:
            raise ValueError("Frozen snapshot drift")
        folds = np.array([r.cv_fold for r in dataset.records])
        train = folds != args.fold
        if not np.any(~train) or args.fold not in range(5):
            raise ValueError("Unknown or empty fold")
        y = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
        x_train = dataset.features[train].astype(np.int64).tocsr()
        x_test = dataset.features[~train].astype(np.int64).tocsr()
        y_train, y_test = y[train], y[~train]
        p, precision = 71, args.precision
        evidence.update(snapshot_digest=digest, n_train=len(y_train), n_test=len(y_test),
                        n_features=x_train.shape[1], p=p, precision=precision,
                        feature_names=list(dataset.feature_names), models={})
        if args.method.startswith("linear"):
            first = (association_order(x_train, y_train, dataset.feature_names)
                     if args.method == "linear_association" else None)
            fit = fit_coordinates(x_train, y_train, p=p, precision=precision, seed=args.seed,
                                  max_sweeps=args.max_sweeps, seconds=args.seconds,
                                  first_order=first, callback=lambda h: emit("sweep", **h))
            status = fit.status
            evidence["fit"] = {k: v for k, v in asdict(fit).items()
                               if k not in ("coefficients", "first_pass_coefficients", "residuals")}
            pairs = [("coordinate", fit.coefficients, status == "coordinate_optimum")]
            if fit.first_pass_coefficients is not None:
                pairs.insert(0, ("one_pass", fit.first_pass_coefficients, True))
            for label, weights, completed in pairs:
                record, score = model_evidence(weights, x_train, x_test, y_train, y_test,
                    p=p, precision=precision, pilot=args.pilot, completed=completed)
                if label == "coordinate" and completed:
                    record["coordinate_certificate"] = coordinate_certificate(
                        x_train, y_train, weights, p=p, precision=precision)
                    if record["coordinate_certificate"]["status"] != "coordinate_optimum":
                        raise AssertionError("False convergence")
                evidence["models"][label], metrics[label] = record, score
        else:
            order = supported_order(x_train, dataset.feature_names, args.seed)
            evidence["feature_order"] = order.tolist()
            a, b = x_train[:, order].toarray(), x_test[:, order].toarray()
            before = time.monotonic()
            design = MahlerDesign.build(a, p=p, precision=precision, degree=args.degree)
            evidence["design"] = dict(bytes=int(design.basis.nbytes),
                seconds=time.monotonic()-before, distinct_inputs=len(design.basis),
                input_digits=design.input_digits)
            problem = GibbsProblem.create(design, y_train)
            emit("design_ready", **evidence["design"])
            fit = fit_published_zubarev(problem, seed=args.seed, initialisation="random",
                betas=(0, 4, 16, 64, 256), draws_per_beta=args.draws_per_beta,
                proposals_per_beta=200000, seconds=args.seconds)
            status = fit.status
            evidence["fit"] = {k: v for k, v in asdict(fit).items() if k != "coefficients"}
            record = dict(coefficients=fit.coefficients.tolist(), completed=status == "schedule_complete",
                          training_predictions=design.predict(fit.coefficients).tolist())
            scores = dict(training=exact_scores(y_train.tolist(), record["training_predictions"], p, precision),
                          stored_nonzero_coefficients=int(np.count_nonzero(fit.coefficients)))
            if not args.pilot:
                test_design = MahlerDesign.build(b, p=p, precision=precision, degree=args.degree)
                record["predictions"] = test_design.predict(fit.coefficients).tolist()
                consulted = np.count_nonzero(test_design.basis[:, fit.coefficients != 0], axis=1)
                scores.update(held_out=exact_scores(y_test.tolist(), record["predictions"], p, precision),
                    mean_nonzero_terms_consulted=float(consulted[test_design.row_groups].mean()),
                    mean_fallback_uses=0.0)
            evidence["models"]["polynomial"], metrics["polynomial"] = record, scores
        evidence["elapsed_seconds"] = time.monotonic()-started
    except Exception as exc:
        status = "failed"
        conn.rollback()
        evidence["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        with conn.cursor() as cur:
            cur.execute("""UPDATE padjective.paper_randomised_method_runs
                SET status=%s,finished_at=NOW(),metrics=%s,evidence=%s WHERE run_id=%s""",
                (status, Jsonb(metrics), Jsonb(evidence), run_id))
        conn.commit()
        conn.close()
        # Pilot logs contain fitting information only.
        emit("finished", status=status, metrics=metrics, elapsed_seconds=time.monotonic()-started)
    return str(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-id", default=str(uuid.uuid4()))
    parser.add_argument("--job-key", default="single")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--precision", type=int, default=7)
    parser.add_argument("--degree", type=int, default=357910)
    parser.add_argument("--draws-per-beta", type=int, default=64)
    parser.add_argument("--max-sweeps", type=int, default=100)
    parser.add_argument("--seconds", type=float, default=300)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
