"""Postgres-backed isolated runs of the published Mihara/Zubarev comparators."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import time
import uuid

import numpy as np
from psycopg.types.json import Jsonb

from . import db
from .paper_revision_experiments import _load_paper_dataset
from .published_mihara import ResourceBudget, fit_published_mihara, predict_published_mihara
from .published_zubarev import GibbsProblem, MahlerDesign, fit_published_zubarev


PAPER_SNAPSHOT = "244ddbe3-0a2c-4c04-9436-0a0253108a09"


def training_feature_order(matrix: np.ndarray, names: tuple[str, ...]) -> list[int]:
    counts = np.count_nonzero(matrix, axis=0)
    return sorted(range(len(names)), key=lambda index: (-int(counts[index]), names[index]))


def exact_rank_certificate(matrix: np.ndarray, p: int, *, seconds: float = 120) -> dict:
    """Sparse exact forward elimination of [1|X], separate from the fit.

    A completed rank smaller than D+1 certifies that Algorithm 6 cannot return
    its required full-rank affine system. It is not a sampled rejection.
    """
    started = time.monotonic()
    bases: dict[int, dict[int, int]] = {}
    columns = matrix.shape[1] + 1
    examined = 0
    for dense in matrix:
        if time.monotonic() - started >= seconds:
            return dict(status="rank_audit_time_limit", rank_lower_bound=len(bases),
                        columns=columns, examined=examined, elapsed_seconds=time.monotonic()-started)
        row = {int(i) + 1: int(dense[i]) % p for i in np.flatnonzero(dense % p)}
        row[0] = 1  # Retain the intercept in any explicit independent-column run.
        while row:
            pivot = min(row)
            if pivot not in bases:
                inverse = pow(row[pivot], -1, p)
                bases[pivot] = {i: value * inverse % p for i, value in row.items()}
                break
            factor = row[pivot]
            for column, value in bases[pivot].items():
                updated = (row.get(column, 0) - factor * value) % p
                if updated:
                    row[column] = updated
                else:
                    row.pop(column, None)
            if time.monotonic() - started >= seconds:
                return dict(status="rank_audit_time_limit", rank_lower_bound=len(bases),
                            columns=columns, examined=examined, elapsed_seconds=time.monotonic()-started)
        examined += 1
        if len(bases) == columns:
            break
    return dict(status="full_rank" if len(bases) == columns else "rank_obstruction",
                rank=len(bases), columns=columns, examined=examined,
                independent_feature_indices=[i - 1 for i in sorted(bases) if i],
                elapsed_seconds=time.monotonic()-started)


def inclusion_certificate(matrix: np.ndarray, targets: np.ndarray, p: int) -> dict:
    """Duplicate-input upper bound for Algorithm 2 at full affine rank.

    Even an arbitrary deterministic predictor cannot exceed the sum of the
    modal first-digit counts within identical input vectors. This is an upper
    bound, not a fitted classifier or a claim that the affine bound is attained.
    """
    groups = defaultdict(Counter)
    for row, target in zip(matrix, targets, strict=True):
        groups[np.asarray(row % p, dtype=np.int64).tobytes()][int(target) % p] += 1
    maximum = sum(max(counts.values()) for counts in groups.values())
    n, rank = len(matrix), matrix.shape[1] + 1
    impossible = n <= rank or 10 * (maximum - rank) <= 9 * (n - rank)
    return dict(status="inclusion_obstruction" if impossible else "not_ruled_out",
                identical_input_groups=len(groups), maximum_agreement_count=maximum,
                n=n, full_affine_rank=rank, agreement_upper_bound=maximum/n)


def independent_scores(actual: list[int], predicted: list[int], p: int) -> dict:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Scoring requires aligned nonempty observations")
    loss = 0.0
    exact = leading = 0
    for target, prediction in zip(actual, predicted, strict=True):
        difference = abs(int(target) - int(prediction))
        if difference == 0:
            exact += 1
            leading += 1
            continue
        leading += int(difference % p == 0)
        distance = 1.0
        while difference % p == 0:
            distance /= p
            difference //= p
        loss += distance
    return dict(n=len(actual), mean_padic_loss=loss/len(actual),
                exact_accuracy=exact/len(actual), first_digit_accuracy=leading/len(actual))


def _ensure_storage(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_published_method_runs (
            run_id UUID PRIMARY KEY,
            snapshot_id UUID NOT NULL,
            method TEXT NOT NULL,
            cv_fold INTEGER NOT NULL,
            pilot BOOLEAN NOT NULL,
            configuration JSONB NOT NULL,
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            metrics JSONB,
            evidence JSONB
        ) TABLESPACE pg_default""")
    conn.commit()


def _snapshot_digest(dataset) -> str:
    digest = hashlib.sha256()
    for row in dataset.records:
        digest.update(json.dumps([row.product_key, row.cv_fold, row.encoded_path, row.tags],
                                 separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def run(args) -> str:
    conn = db.get_connection()
    _ensure_storage(conn)
    run_id = uuid.uuid4()
    config = vars(args).copy()
    config.update(python=platform.python_version(), numpy=np.__version__, hostname=platform.node(),
                  process_id=os.getpid(), source_commit=subprocess.check_output(
                      ["git", "rev-parse", "HEAD"], text=True).strip())
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO padjective.paper_published_method_runs
            (run_id,snapshot_id,method,cv_fold,pilot,configuration,status)
            VALUES (%s,%s,%s,%s,%s,%s,'running')""",
                    (run_id, args.snapshot, args.method, args.fold, args.pilot, Jsonb(config)))
    conn.commit()
    print(json.dumps(dict(event="started", run_id=str(run_id), configuration=config)), flush=True)
    started = time.monotonic()
    evidence: dict = {}
    metrics: dict = {}
    status = "failed"
    try:
        dataset = _load_paper_dataset(conn, snapshot_ref=args.snapshot, schema="padjective")
        folds = np.array([record.cv_fold for record in dataset.records])
        if args.fold not in folds:
            raise ValueError("Unknown fold")
        train = folds != args.fold
        test = ~train
        matrix = dataset.features.toarray().astype(np.int64)
        order = training_feature_order(matrix[train], dataset.feature_names)
        if args.max_tags:
            order = order[:args.max_tags]
        x_train, x_test = matrix[train][:, order], matrix[test][:, order]
        y = np.array([record.encoded_path for record in dataset.records], dtype=np.int64)
        p = dataset.prime_base
        evidence.update(snapshot_digest=_snapshot_digest(dataset), n_snapshot=len(y),
                        n_train=int(train.sum()), n_test=int(test.sum()),
                        n_features=len(order), feature_order=[dataset.feature_names[i] for i in order],
                        p=p, precision=args.precision)
        if args.method == "mihara":
            rank = exact_rank_certificate(x_train, p, seconds=args.rank_seconds)
            evidence["rank_certificate"] = rank
            print(json.dumps(dict(event="rank_audit", run_id=str(run_id), **rank)), flush=True)
            if args.rank_reduce and "independent_feature_indices" in rank:
                selected = rank["independent_feature_indices"]
                x_train, x_test = x_train[:, selected], x_test[:, selected]
                evidence["independent_column_preprocessing"] = dict(
                    n_features=len(selected), feature_order=[dataset.feature_names[order[i]] for i in selected],
                    scope="Modulo-p training column span; no claim of equivalence on unseen rows or higher digits")
            inclusion = inclusion_certificate(x_train, y[train], p)
            evidence["inclusion_certificate"] = inclusion
            print(json.dumps(dict(event="inclusion_audit", run_id=str(run_id), **inclusion)), flush=True)
            if rank["status"] == "rank_obstruction" and not args.rank_reduce:
                status = "rank_obstruction"
            elif inclusion["status"] == "inclusion_obstruction":
                status = "inclusion_obstruction"
            else:
                fit = fit_published_mihara(x_train, y[train], p=p, precision=args.precision,
                    rep=args.rep, seed=args.seed, budget=ResourceBudget(
                        max_draws=args.max_draws, seconds=args.seconds))
                evidence["fit"] = asdict(fit)
                status = fit.status
                if fit.completed:
                    metrics["training"] = independent_scores(y[train].tolist(), predict_published_mihara(fit, x_train, p=p), p)
                    if not args.pilot:
                        predictions = predict_published_mihara(fit, x_test, p=p)
                        metrics["held_out"] = independent_scores(y[test].tolist(), predictions, p)
                        evidence["predictions"] = predictions
        else:
            before_design = time.monotonic()
            design = MahlerDesign.build(x_train, p=p, precision=args.precision, degree=args.degree)
            evidence["design"] = dict(seconds=time.monotonic()-before_design,
                bytes=int(design.basis.nbytes), distinct_inputs=len(design.basis),
                root_active_columns=design.root_columns.tolist(), input_digits=design.input_digits)
            problem = GibbsProblem.create(design, y[train])
            evidence["root_loss_lower_bound"] = problem.root_lower_bound
            evidence["sampler"] = problem.sampler
            print(json.dumps(dict(event="design_ready", run_id=str(run_id), **evidence["design"],
                                  root_loss_lower_bound=problem.root_lower_bound)), flush=True)
            fit = fit_published_zubarev(problem, seed=args.seed, initialisation=args.initialisation,
                betas=args.betas, draws_per_beta=args.draws_per_beta,
                proposals_per_beta=args.proposals_per_beta, seconds=args.seconds)
            evidence["fit"] = {**asdict(fit), "coefficients": fit.coefficients.tolist()}
            status = fit.status
            metrics["training"] = independent_scores(y[train].tolist(), design.predict(fit.coefficients).tolist(), p)
            if not args.pilot:
                test_design = MahlerDesign.build(x_test, p=p, precision=args.precision, degree=args.degree)
                predictions = test_design.predict(fit.coefficients).tolist()
                metrics["held_out"] = independent_scores(y[test].tolist(), predictions, p)
                evidence["predictions"] = predictions
                # These are polynomial terms consulted, not tag coefficients.
                consulted = np.count_nonzero(test_design.basis[:, fit.coefficients != 0], axis=1)
                metrics["mean_nonzero_terms_consulted"] = float(consulted[test_design.row_groups].mean())
                metrics["stored_nonzero_coefficients"] = int(np.count_nonzero(fit.coefficients))
        evidence["elapsed_seconds"] = time.monotonic() - started
    except Exception as exc:
        status = "failed"
        conn.rollback()
        evidence["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        with conn.cursor() as cur:
            cur.execute("""UPDATE padjective.paper_published_method_runs
                SET status=%s,finished_at=NOW(),metrics=%s,evidence=%s WHERE run_id=%s""",
                        (status, Jsonb(metrics), Jsonb(evidence), run_id))
        conn.commit()
        conn.close()
        print(json.dumps(dict(event="finished", run_id=str(run_id), status=status, metrics=metrics,
                              elapsed_seconds=time.monotonic()-started)), flush=True)
    return str(run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mihara", "zubarev"), required=True)
    parser.add_argument("--snapshot", default=PAPER_SNAPSHOT)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--pilot", action="store_true", help="Never score the outer held-out fold")
    parser.add_argument("--max-tags", type=int, default=0, help="0 uses every archived feature")
    parser.add_argument("--rank-reduce", action="store_true", help="Explicit Mihara independent-column preprocessing experiment")
    parser.add_argument("--precision", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--rank-seconds", type=float, default=120)
    parser.add_argument("--max-draws", type=int, default=1_000_000)
    parser.add_argument("--rep", type=int, default=3)
    parser.add_argument("--degree", type=int, default=5040)
    parser.add_argument("--initialisation", choices=("zeros", "random"), default="zeros")
    parser.add_argument("--betas", type=float, nargs="+", default=[0, 4, 16, 64, 256])
    parser.add_argument("--draws-per-beta", type=int, default=64)
    parser.add_argument("--proposals-per-beta", type=int, default=200000)
    args = parser.parse_args()
    if args.max_tags < 0:
        parser.error("--max-tags must be nonnegative")
    if args.rank_reduce and args.method != "mihara":
        parser.error("--rank-reduce applies only to Mihara")
    run(args)


if __name__ == "__main__":
    main()
