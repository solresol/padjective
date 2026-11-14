"""Frozen 10/25/75/90 percent reference sweep, using the audited fitting engine."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

import numpy as np
from psycopg.types.json import Jsonb

from . import db
from .half_reference_experiment import CONTROL_BATCH, evaluate_trajectory, fit_trajectory
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .residual_ensemble_experiment import TRAJECTORIES, emit, ensure_storage

PERCENTS = (10, 25, 75, 90)
HALF_BATCH = 'bb57fea7-2e63-441a-ab03-16e853671090'
PROTOCOL = Path('docs/reference-fraction-sweep-protocol.md')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def controls(conn):
    states = conn.execute('SELECT batch_id,status FROM padjective.residual_ensemble_batches '
        'WHERE batch_id=ANY(%s::uuid[])', ([HALF_BATCH, CONTROL_BATCH],)).fetchall()
    assert len(states) == 2 and all(s == 'validated' for _, s in states)
    report = conn.execute('SELECT report FROM padjective.residual_ensemble_reports WHERE batch_id=%s',
                          (HALF_BATCH,)).fetchone()[0]
    assert report['validation']['status'] == 'passed' and len(report['rows']) == 6075
    assert report['manifest']['snapshot_digest'] == SNAPSHOT_DIGEST
    rows = conn.execute('SELECT cv_fold,roster,method,members,evidence FROM padjective.residual_ensemble_scores '
        'WHERE batch_id=ANY(%s::uuid[])', ([HALF_BATCH, CONTROL_BATCH],)).fetchall()
    mapping = {(fold, roster, method, size): e for fold, roster, method, size, e in rows}
    assert len(mapping) == len(rows) == 6075
    assert all(mapping[(r['fold'], r['roster'], r['method'], r['members'])] == r for r in report['rows'])
    return report, mapping


def ensure_sweep_storage(conn):
    ensure_storage(conn)
    conn.execute('SELECT pg_advisory_xact_lock(71357919)')
    conn.execute("SET LOCAL default_tablespace='pg_default'")
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.reference_fraction_sweeps (
        sweep_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        fits_finished_at TIMESTAMPTZ, validated_at TIMESTAMPTZ,
        manifest JSONB NOT NULL, batches JSONB NOT NULL, status TEXT NOT NULL,
        report JSONB, error TEXT) TABLESPACE pg_default''')
    conn.commit()


def specification(control_report, workers):
    files = ('padjective/half_reference_ensemble.py', 'padjective/half_reference_experiment.py',
        'padjective/reference_fraction_sweep.py', 'padjective/randomised_linear.py',
        'padjective/residual_ensemble.py', 'padjective/residual_ensemble_experiment.py', str(PROTOCOL))
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    return json.loads(json.dumps(dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        protocol_text=PROTOCOL.read_text(), control_batch=HALF_BATCH, base_control_batch=CONTROL_BATCH,
        control_report_sha256=canonical_digest(control_report), snapshot_digest=SNAPSHOT_DIGEST,
        percents=PERCENTS, trajectories=TRAJECTORIES, workers=workers,
        python=platform.python_version(), numpy=np.__version__, hostname=platform.node())))


def full_reference_prefix(percent):
    return 99//(100-percent)+1


def jobs(batches):
    return [(batches[str(p)], p, fold, roster) for roster in TRAJECTORIES
            for fold in range(5) for p in PERCENTS]


def fit(args):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '1', f'Set {name}=1 before importing NumPy'
    with db.get_connection() as conn:
        ensure_sweep_storage(conn)
        control_report, _ = controls(conn)
        spec = specification(control_report, args.workers)
        previous = conn.execute('SELECT manifest,batches,status FROM padjective.reference_fraction_sweeps '
            'WHERE sweep_id=%s', (args.sweep_id,)).fetchone()
        if previous:
            assert previous[0] == spec and previous[2] in ('fitting', 'failed', 'fits_complete')
            batches = previous[1]
            conn.execute("UPDATE padjective.reference_fraction_sweeps SET status='fitting',error=NULL WHERE sweep_id=%s",
                         (args.sweep_id,))
        else:
            batches = {str(p): str(uuid.uuid4()) for p in PERCENTS}
            conn.execute('''INSERT INTO padjective.reference_fraction_sweeps
                (sweep_id,manifest,batches,status) VALUES (%s,%s,%s,'fitting')''',
                (args.sweep_id, Jsonb(spec), Jsonb(batches)))
            for p in PERCENTS:
                child = dict(spec, sweep_id=args.sweep_id, reference_percent=p)
                conn.execute('''INSERT INTO padjective.residual_ensemble_batches (batch_id,manifest,status)
                    VALUES (%s,%s,'fitting')''', (batches[str(p)], Jsonb(child)))
                for fold in range(5):
                    for roster in TRAJECTORIES:
                        conn.execute('''INSERT INTO padjective.residual_ensemble_trajectories
                            (batch_id,cv_fold,roster,status) VALUES (%s,%s,%s,'fitting')''',
                            (batches[str(p)], fold, roster))
        saved = conn.execute('SELECT batch_id,cv_fold,roster,status FROM '
            'padjective.residual_ensemble_trajectories WHERE batch_id=ANY(%s::uuid[])',
            (list(batches.values()),)).fetchall()
        complete = {(str(b), f, r) for b, f, r, status in saved if status in ('complete', 'reference_perfect')}
        assert len(saved) == 60
        pending = [(b, p, f, r) for b, p, f, r in jobs(batches) if (b, f, r) not in complete]
    emit('sweep_started', sweep_id=args.sweep_id, batches=batches,
         pending_trajectories=len(pending), workers=args.workers)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_trajectory, b, f, r) for b, _, f, r in pending]
            for future in as_completed(futures):
                emit('trajectory_finished', **future.result())
    except BaseException as exc:
        with db.get_connection() as conn:
            conn.execute("UPDATE padjective.reference_fraction_sweeps SET status='failed',error=%s WHERE sweep_id=%s",
                         (str(exc), args.sweep_id))
        raise
    with db.get_connection() as conn:
        done = conn.execute("SELECT count(*) FROM padjective.residual_ensemble_trajectories WHERE "
            "batch_id=ANY(%s::uuid[]) AND status IN ('complete','reference_perfect')",
            (list(batches.values()),)).fetchone()[0]
        assert done == 60
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='fits_complete',finished_at=NOW() "
                     'WHERE batch_id=ANY(%s::uuid[])', (list(batches.values()),))
        conn.execute("UPDATE padjective.reference_fraction_sweeps SET status='fits_complete',fits_finished_at=NOW() "
                     'WHERE sweep_id=%s', (args.sweep_id,))
    emit('all_fits_complete', sweep_id=args.sweep_id, trajectories=60)


def audit_one(batch, percent, fold, roster):
    return dict(batch_id=batch, percent=percent, **evaluate_trajectory(batch, fold, roster))


def evaluate(args):
    with db.get_connection() as conn:
        spec, batches, status = conn.execute('SELECT manifest,batches,status FROM '
            'padjective.reference_fraction_sweeps WHERE sweep_id=%s', (args.sweep_id,)).fetchone()
        control_report, control_rows = controls(conn)
        assert status in ('fits_complete', 'validated') and spec == specification(control_report, args.workers)
        children = conn.execute('SELECT batch_id,manifest,status FROM padjective.residual_ensemble_batches '
            'WHERE batch_id=ANY(%s::uuid[])', (list(batches.values()),)).fetchall()
        assert len(children) == 4
        for batch, child_spec, child_status in children:
            p = next(p for p in PERCENTS if batches[str(p)] == str(batch))
            assert child_spec == dict(spec, sweep_id=args.sweep_id, reference_percent=p)
            assert child_status in ('fits_complete', 'validated')
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_one, *job) for job in jobs(batches)]
        for future in as_completed(futures):
            results.append(future.result())
    child_reports = {}
    for p in PERCENTS:
        parts = [r for r in results if r['percent'] == p]
        rows = [row for r in parts for row in r['rows']]
        assert len(parts) == 15 and len(rows) == 2025
        for row in rows:
            if row['members'] <= full_reference_prefix(p):
                expected = control_rows[(row['fold'], row['roster'], 'error_only', row['members'])]
                assert row['prediction_sha256'] == expected['prediction_sha256']
        child_reports[p] = dict(batch_id=batches[str(p)],
            manifest=dict(spec, sweep_id=args.sweep_id, reference_percent=p),
            validation=dict(status='passed', trajectories=15,
                certified_coordinates=sum(r['certified_coordinates'] for r in parts),
                reference_checkpoint_checks=sum(r['reference_checkpoint_checks'] for r in parts),
                identical_full_reference_prefix=full_reference_prefix(p)), rows=rows)
    new_rows = [row for child in child_reports.values() for row in child['rows']]
    assert len(new_rows) == 8100
    report = dict(sweep_id=args.sweep_id, manifest=spec, batches=batches,
        validation=dict(status='passed', trajectories=60, new_score_rows=8100,
            certified_coordinates=sum(r['certified_coordinates'] for r in results),
            reference_checkpoint_checks=sum(r['reference_checkpoint_checks'] for r in results),
            control_rows_verified=len(control_rows)),
        rows=sorted(new_rows+control_report['rows'],
                    key=lambda r: (r['roster'], r['method'], r['members'], r['fold'])))
    assert len(report['rows']) == 14175
    with db.get_connection() as conn:
        current_control, _ = controls(conn)
        assert canonical_digest(current_control) == spec['control_report_sha256']
        for p, child in child_reports.items():
            conn.execute('''INSERT INTO padjective.residual_ensemble_reports (batch_id,report) VALUES (%s,%s)
                ON CONFLICT (batch_id) DO UPDATE SET report=EXCLUDED.report''',
                (batches[str(p)], Jsonb(child)))
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='validated' WHERE batch_id=ANY(%s::uuid[])",
                     (list(batches.values()),))
        conn.execute("UPDATE padjective.reference_fraction_sweeps SET status='validated',validated_at=NOW(),report=%s "
                     'WHERE sweep_id=%s', (Jsonb(report), args.sweep_id))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    emit('sweep_audit_complete', sweep_id=args.sweep_id, **report['validation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit', 'evaluate'))
    parser.add_argument('--sweep-id', default=str(uuid.uuid4()))
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=6)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    (fit if args.command == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
