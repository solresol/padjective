"""Frozen random-subspace study: isolated, resumable Postgres component bank."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .paper_audit_ensemble_scaling import prefix_count_scores
from .paper_ensemble_scaling_batch import BANK_SEEDS, SIZES, fitting_fingerprints
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import compare_scores, exact_scores
from .paper_validate_randomised import direct_certificate, direct_linear, fingerprint
from .randomised_linear import apply_default, fit_coordinates, fit_default, linear_predictions
from .subspace_voting import FRACTIONS, RULES, feature_mask, path_digits

BASE_ENSEMBLE_BATCH = '059d7eb8-14b5-4d5d-857f-134a2098fd89'
PROTOCOL = Path('docs/subspace-voting-protocol.md')
CONTEXT = {}


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def ensure_storage(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(71357912)')
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_subspace_batches (
            batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            manifest JSONB NOT NULL, status TEXT NOT NULL
        ) TABLESPACE pg_default''')
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_subspace_runs (
            batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, member_index INTEGER NOT NULL,
            fraction DOUBLE PRECISION NOT NULL, started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ, status TEXT NOT NULL, evidence JSONB,
            PRIMARY KEY(batch_id,cv_fold,member_index,fraction)
        ) TABLESPACE pg_default''')
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_subspace_results (
            batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, fraction DOUBLE PRECISION NOT NULL,
            roster INTEGER NOT NULL, members INTEGER NOT NULL, rule TEXT NOT NULL,
            evidence JSONB NOT NULL,
            PRIMARY KEY(batch_id,cv_fold,fraction,roster,members,rule)
        ) TABLESPACE pg_default''')
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_subspace_reports (
            batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            report JSONB NOT NULL
        ) TABLESPACE pg_default''')
    conn.commit()


def load_context():
    with db.get_connection() as conn:
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema='padjective')
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST and dataset.prime_base == 71
    assert dataset.features.shape == (6693, 2542)
    assert np.all(dataset.features.data == 1)
    folds = np.array([r.cv_fold for r in dataset.records])
    targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    for candidate in np.unique(targets):
        path_digits(candidate)
    return {f: dict(x_train=dataset.features[folds != f].astype(np.int64).tocsr(),
        x_test=dataset.features[folds == f].astype(np.int64).tocsr(),
        y_train=targets[folds != f], y_test=targets[folds == f],
        feature_names=list(dataset.feature_names)) for f in range(5)}


def initialise_worker():
    global CONTEXT
    CONTEXT = load_context()


def jobs():
    # Keep the schedule fixed and interleave fractions/folds, not score-driven.
    return [(fold, index, fraction) for index in range(len(BANK_SEEDS))
            for fraction in FRACTIONS[:-1] for fold in range(5)]


def fit_one(batch_id, fold, index, fraction):
    started = time.monotonic()
    base = BANK_SEEDS[index]
    with db.get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''INSERT INTO padjective.paper_subspace_runs
                (batch_id,cv_fold,member_index,fraction,status) VALUES (%s,%s,%s,%s,'running')''',
                (batch_id, fold, index, fraction))
        conn.commit()
        evidence = dict(seed_base=base, fit_seed=base+fold)
        status = 'failed'
        try:
            c = CONTEXT[fold]
            eligible = np.flatnonzero(c['x_train'].getnnz(axis=0))
            mask = feature_mask(eligible, fraction, fold, base)
            x_train, x_test = c['x_train'][:, mask], c['x_test'][:, mask]
            fit = fit_coordinates(x_train, c['y_train'], p=71, precision=7,
                                  seed=base+fold, max_sweeps=100, seconds=300)
            status = fit.status
            evidence.update(mask=mask.tolist(), eligible_features=len(eligible),
                selected_features=len(mask), mask_sha256=fingerprint(mask.tolist()),
                fit={k: v for k, v in asdict(fit).items()
                     if k not in ('coefficients', 'first_pass_coefficients', 'residuals')})
            if status != 'coordinate_optimum':
                raise RuntimeError('Incomplete fit; do not select a survivor-only bank')
            weights = fit.coefficients
            train_raw = direct_linear(x_train, weights, 71**7)
            test_raw = direct_linear(x_test, weights, 71**7)
            assert train_raw == linear_predictions(x_train, weights, 71, 7).tolist()
            assert test_raw == linear_predictions(x_test, weights, 71, 7).tolist()
            certified = direct_certificate(x_train, c['y_train'], weights, train_raw, 71, 7)
            assert certified == len(mask)
            default = fit_default(c['y_train'], np.asarray(train_raw), p=71, precision=7)
            train_pred = apply_default(np.asarray(train_raw), default).tolist()
            test_pred = apply_default(np.asarray(test_raw), default).tolist()
            metrics = {}
            for name, y, pred in (('training_raw', c['y_train'], train_raw),
                    ('training', c['y_train'], train_pred), ('held_out_raw', c['y_test'], test_raw),
                    ('held_out', c['y_test'], test_pred)):
                metrics[name] = exact_scores(y.tolist(), pred, 71, 7)
                compare_scores(metrics[name], prefix_count_scores(y, pred))
            evidence.update(coefficients=weights.tolist(), coefficient_sha256=fingerprint(weights.tolist()),
                default=int(default), training_raw_predictions=train_raw, training_predictions=train_pred,
                held_out_raw_predictions=test_raw, predictions=test_pred,
                prediction_sha256=fingerprint(test_pred), certified_coordinates=certified,
                metrics=metrics, stored_nonzero_coefficients=int(np.count_nonzero(weights)),
                mean_active_terms=float(np.asarray(x_test @ (weights != 0).astype(np.int64)).mean()),
                mean_fallback_uses=float(np.mean(np.asarray(test_raw) == 0)))
        except Exception as exc:
            status = 'failed'
            evidence['error'] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            evidence['job_seconds'] = time.monotonic()-started
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute('''UPDATE padjective.paper_subspace_runs SET status=%s,
                    finished_at=NOW(),evidence=%s WHERE batch_id=%s AND cv_fold=%s
                    AND member_index=%s AND fraction=%s''',
                    (status, Jsonb(evidence), batch_id, fold, index, fraction))
            conn.commit()
    return dict(fold=fold, member_index=index, fraction=fraction, status=status,
                job_seconds=time.monotonic()-started)


def manifest(workers):
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), protocol_text=PROTOCOL.read_text(),
        snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST, base_batch=BASE_ENSEMBLE_BATCH,
        fitting_fingerprints=fitting_fingerprints(), fractions=FRACTIONS, sizes=SIZES, rules=RULES,
        seed_bases=BANK_SEEDS, new_fits=len(jobs()), reused_fits=1215, workers=workers,
        python=platform.python_version(), numpy=np.__version__, hostname=platform.node())


def run(args):
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise ValueError(f'Start with {key}=1 before importing NumPy')
    specification = manifest(args.workers)
    batch_id = args.batch_id or str(uuid.uuid4())
    with db.get_connection() as conn:
        ensure_storage(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('SELECT manifest,status FROM padjective.paper_subspace_batches WHERE batch_id=%s', (batch_id,))
            prior = cur.fetchone()
            if prior:
                assert prior['manifest'] == json.loads(json.dumps(specification)), 'Source/protocol drift on resume'
                assert prior['status'] in ('fitting', 'fits_complete')
            else:
                cur.execute('''INSERT INTO padjective.paper_subspace_batches (batch_id,manifest,status)
                    VALUES (%s,%s,'fitting')''', (batch_id, Jsonb(specification)))
            cur.execute('''SELECT cv_fold,member_index,fraction,status FROM padjective.paper_subspace_runs
                WHERE batch_id=%s''', (batch_id,))
            saved = cur.fetchall()
        conn.commit()
    assert all(r['status'] == 'coordinate_optimum' for r in saved), 'Investigate unfinished jobs before resuming'
    completed = {(r['cv_fold'], r['member_index'], r['fraction']) for r in saved}
    assert completed <= set(jobs())
    pending = [job for job in jobs() if job not in completed]
    emit('subspace_batch_started', batch_id=batch_id, pending=len(pending), already_complete=len(completed),
         source_commit=specification['source_commit'], protocol_sha256=specification['protocol_sha256'])
    count = len(completed)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialise_worker) as pool:
        futures = [pool.submit(fit_one, batch_id, *job) for job in pending]
        for future in as_completed(futures):
            result = future.result()
            count += 1
            if count % 25 == 0 or count == len(jobs()):
                emit('subspace_fit_progress', completed=count, expected=len(jobs()), **result)
    with db.get_connection() as conn:
        row = conn.execute('''SELECT count(*) FROM padjective.paper_subspace_runs
            WHERE batch_id=%s AND status='coordinate_optimum' ''', (batch_id,)).fetchone()
        assert row[0] == len(jobs())
        conn.execute("UPDATE padjective.paper_subspace_batches SET status='fits_complete' WHERE batch_id=%s", (batch_id,))
    emit('subspace_fits_complete', batch_id=batch_id, completed=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--batch-id', help='Resume only an identical, cleanly interrupted protocol')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
