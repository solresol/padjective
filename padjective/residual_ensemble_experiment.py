"""Postgres-only fitting and independent replay of error-only training ensembles."""
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
from .paper_ensemble_scaling_batch import BANK_SEEDS, SIZES
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import compare_scores, exact_scores
from .paper_validate_randomised import direct_certificate, direct_linear, direct_medoid, fingerprint
from .randomised_linear import apply_default, consensus_predictions, loss_units
from .residual_ensemble import CumulativeConsensus, error_rows, fit_error_member

BASE_BATCH = '059d7eb8-14b5-4d5d-857f-134a2098fd89'
TRAJECTORIES = {0: 243, 9: 81, 18: 81}
PROTOCOL = Path('docs/residual-ensemble-protocol.md')


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def order_for(offset):
    return BANK_SEEDS[offset:] + BANK_SEEDS[:offset]


def ensure_storage(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT pg_advisory_xact_lock(71357918)')
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        for statement in (
            '''CREATE TABLE IF NOT EXISTS padjective.residual_ensemble_batches (
                batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                finished_at TIMESTAMPTZ, manifest JSONB NOT NULL, status TEXT NOT NULL,
                error TEXT) TABLESPACE pg_default''',
            '''CREATE TABLE IF NOT EXISTS padjective.residual_ensemble_trajectories (
                batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, roster INTEGER NOT NULL,
                status TEXT NOT NULL, members INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(batch_id,cv_fold,roster)) TABLESPACE pg_default''',
            '''CREATE TABLE IF NOT EXISTS padjective.residual_ensemble_members (
                batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, roster INTEGER NOT NULL,
                member_index INTEGER NOT NULL, evidence JSONB NOT NULL,
                PRIMARY KEY(batch_id,cv_fold,roster,member_index)) TABLESPACE pg_default''',
            '''CREATE TABLE IF NOT EXISTS padjective.residual_ensemble_scores (
                batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, roster INTEGER NOT NULL,
                method TEXT NOT NULL, members INTEGER NOT NULL, evidence JSONB NOT NULL,
                PRIMARY KEY(batch_id,cv_fold,roster,method,members)) TABLESPACE pg_default''',
            '''CREATE TABLE IF NOT EXISTS padjective.residual_ensemble_reports (
                batch_id UUID PRIMARY KEY, report JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()) TABLESPACE pg_default''',
        ):
            cur.execute(statement)
    conn.commit()


def load_fold(conn, fold):
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema='padjective')
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST
    assert dataset.prime_base == 71 and dataset.features.shape == (6693, 2542)
    folds = np.array([r.cv_fold for r in dataset.records])
    targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    return dict(x_train=dataset.features[folds != fold].astype(np.int64).tocsr(),
        x_test=dataset.features[folds == fold].astype(np.int64).tocsr(),
        y_train=targets[folds != fold], y_test=targets[folds == fold])


def load_baseline(conn, fold):
    report = conn.execute('SELECT report FROM padjective.paper_ensemble_scaling_reports '
                          'WHERE batch_id=%s', (BASE_BATCH,)).fetchone()[0]
    assert report['validation']['status'] == 'passed'
    assert report['snapshot_digest'] == SNAPSHOT_DIGEST
    entries = {row['seed']: row for row in report['models'] if row['fold'] == fold}
    assert set(entries) == {s+fold for s in BANK_SEEDS}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('''SELECT run_id, status, configuration, evidence->'models'->'coordinate' AS model
            FROM padjective.paper_randomised_method_runs WHERE run_id=ANY(%s::uuid[])''',
            ([r['run_id'] for r in entries.values()],))
        rows = cur.fetchall()
    assert len(rows) == 243
    output = {}
    for row in rows:
        seed, model = row['configuration']['seed'], row['model']
        assert row['status'] == 'coordinate_optimum'
        assert fingerprint(model['coefficients']) == entries[seed]['coefficient_sha256']
        assert fingerprint(model['predictions']) == entries[seed]['prediction_sha256']
        output[seed] = dict(model=model, run_id=str(row['run_id']))
    return output


def fit_trajectory(batch_id, fold, roster):
    limit = TRAJECTORIES[roster]
    with db.get_connection() as conn:
        context = load_fold(conn, fold)
        # This fitting path deliberately never reads context['y_test'].
        x_train, y_train, x_test = (context[k] for k in ('x_train', 'y_train', 'x_test'))
        candidates = np.unique(y_train)
        state = CumulativeConsensus(len(y_train), candidates, max_members=limit)
        previous = None
        saved = conn.execute('''SELECT member_index,evidence FROM padjective.residual_ensemble_members
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s ORDER BY member_index''',
            (batch_id, fold, roster)).fetchall()
        assert [r[0] for r in saved] == list(range(len(saved)))
        for _, evidence in saved:
            assert error_rows(y_train, previous).tolist() == evidence['selected_rows']
            previous = state.add(evidence['training_predictions'])
        first_baseline = None
        if not saved:
            first_baseline = load_baseline(conn, fold)[order_for(roster)[0]+fold]['model']
        total = len(saved)
        status = 'complete'
        for index in range(len(saved), limit):
            started = time.monotonic()
            member = fit_error_member(x_train, y_train, x_test, previous_predictions=previous,
                                      seed=order_for(roster)[index]+fold)
            if member is None:
                status = 'training_perfect'
                break
            if index == 0:
                assert member.coefficients.tolist() == first_baseline['coefficients']
                assert member.default == first_baseline['default']
                assert member.predictions.tolist() == first_baseline['predictions']
                assert member.training_predictions.tolist() == first_baseline['training_predictions']
            updated = state.add(member.training_predictions)
            fit = {k: v for k, v in asdict(member.fit).items()
                   if k not in ('coefficients', 'first_pass_coefficients', 'residuals')}
            evidence = dict(seed=order_for(roster)[index]+fold,
                selected_rows=member.selected_rows.tolist(),
                selected_rows_sha256=fingerprint(member.selected_rows.tolist()),
                coefficients=member.coefficients.tolist(), default=member.default,
                training_predictions=member.training_predictions.tolist(),
                predictions=member.predictions.tolist(), fit=fit,
                coefficient_sha256=fingerprint(member.coefficients.tolist()),
                prediction_sha256=fingerprint(member.predictions.tolist()),
                training_prediction_sha256=fingerprint(member.training_predictions.tolist()),
                selected_rows_count=len(member.selected_rows),
                full_training_raw_loss_units=int(loss_units(y_train-member.training_raw, 71, 7).sum()),
                consensus_training_errors=int(np.count_nonzero(updated != y_train)),
                forgotten_correct_rows=0 if previous is None else int(np.count_nonzero(
                    (previous == y_train) & (updated != y_train))),
                job_seconds=time.monotonic()-started)
            conn.execute('''INSERT INTO padjective.residual_ensemble_members
                (batch_id,cv_fold,roster,member_index,evidence) VALUES (%s,%s,%s,%s,%s)''',
                (batch_id, fold, roster, index, Jsonb(evidence)))
            total = index+1
            conn.execute('''UPDATE padjective.residual_ensemble_trajectories SET members=%s
                WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (total, batch_id, fold, roster))
            conn.commit()
            previous = updated
            if total == 1 or total % 9 == 0 or total == limit:
                emit('fit_progress', fold=fold, roster=roster, members=total, limit=limit,
                     training_rows=evidence['selected_rows_count'],
                     training_errors=evidence['consensus_training_errors'])
        conn.execute('''UPDATE padjective.residual_ensemble_trajectories SET status=%s,members=%s
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (status, total, batch_id, fold, roster))
    return dict(fold=fold, roster=roster, members=total, status=status)


def source_manifest():
    files = ('padjective/randomised_linear.py', 'padjective/residual_ensemble.py',
             'padjective/residual_ensemble_experiment.py', str(PROTOCOL))
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    return dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files},
        protocol_text=PROTOCOL.read_text(), snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
        baseline_batch=BASE_BATCH, trajectories=TRAJECTORIES, seed_bases=BANK_SEEDS,
        python=platform.python_version(), numpy=np.__version__, hostname=platform.node())


def run(args):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '1', f'Set {name}=1 before starting Python'
    specification = json.loads(json.dumps(source_manifest()))
    with db.get_connection() as conn:
        ensure_storage(conn)
        prior = conn.execute('SELECT manifest,status FROM padjective.residual_ensemble_batches '
                             'WHERE batch_id=%s', (args.batch_id,)).fetchone()
        if prior:
            assert prior[0] == specification, 'Source/protocol drift on resume'
            assert prior[1] in ('fitting', 'fits_complete', 'failed')
            conn.execute("UPDATE padjective.residual_ensemble_batches SET status='fitting',error=NULL "
                         'WHERE batch_id=%s', (args.batch_id,))
        else:
            conn.execute('''INSERT INTO padjective.residual_ensemble_batches (batch_id,manifest,status)
                VALUES (%s,%s,'fitting')''', (args.batch_id, Jsonb(specification)))
            for roster in TRAJECTORIES:
                for fold in range(5):
                    conn.execute('''INSERT INTO padjective.residual_ensemble_trajectories
                        (batch_id,cv_fold,roster,status) VALUES (%s,%s,%s,'fitting')''',
                        (args.batch_id, fold, roster))
        pending = conn.execute("SELECT cv_fold,roster FROM padjective.residual_ensemble_trajectories "
            "WHERE batch_id=%s AND status='fitting' ORDER BY roster,cv_fold", (args.batch_id,)).fetchall()
    emit('batch_started', batch_id=args.batch_id, pending_trajectories=len(pending), workers=args.workers)
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_trajectory, args.batch_id, f, r) for f, r in pending]
            for future in as_completed(futures):
                emit('trajectory_finished', **future.result())
    except BaseException as exc:
        with db.get_connection() as conn:
            conn.execute("UPDATE padjective.residual_ensemble_batches SET status='failed',error=%s "
                         'WHERE batch_id=%s', (str(exc), args.batch_id))
        raise
    with db.get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM padjective.residual_ensemble_trajectories "
            "WHERE batch_id=%s AND status IN ('complete','training_perfect')", (args.batch_id,)).fetchone()[0] == 15
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='fits_complete',finished_at=NOW() "
                     'WHERE batch_id=%s', (args.batch_id,))
    emit('fits_complete', batch_id=args.batch_id)


def audited_member(context, model, selected, *, certify):
    """Independent Python integer sums/defaults and optional full coordinate audit."""
    weights = model['coefficients']
    train_raw = np.array(direct_linear(context['x_train'], weights, 71**7))
    test_raw = np.array(direct_linear(context['x_test'], weights, 71**7))
    subset_raw, subset_y = train_raw[selected], context['y_train'][selected]
    if np.any(subset_raw == 0):
        default = direct_medoid(subset_y[subset_raw == 0], 71, 7)[0]
    else:
        unique, counts = np.unique(subset_y, return_counts=True)
        default = int(unique[np.argmax(counts)])
    assert default == model['default']
    train_pred, test_pred = apply_default(train_raw, default), apply_default(test_raw, default)
    assert train_pred.tolist() == model['training_predictions']
    assert test_pred.tolist() == model['predictions']
    examined = 0
    if certify:
        examined = direct_certificate(context['x_train'][selected], subset_y, weights,
                                      subset_raw, 71, 7)
        assert int(loss_units(subset_y-subset_raw, 71, 7).sum()) == model['fit']['final_units']
        assert model['fit']['history'][-1]['complete'] and model['fit']['history'][-1]['changes'] == 0
        assert fingerprint(weights) == model['coefficient_sha256']
        assert fingerprint(test_pred.tolist()) == model['prediction_sha256']
        assert fingerprint(train_pred.tolist()) == model['training_prediction_sha256']
        assert fingerprint(selected.tolist()) == model['selected_rows_sha256']
        assert int(loss_units(context['y_train']-train_raw, 71, 7).sum()) == model['full_training_raw_loss_units']
    return train_pred, test_pred, examined


def evaluate_trajectory(batch_id, fold, roster):
    with db.get_connection() as conn:
        context, baseline = load_fold(conn, fold), load_baseline(conn, fold)
        rows = conn.execute('''SELECT member_index,evidence FROM padjective.residual_ensemble_members
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s ORDER BY member_index''',
            (batch_id, fold, roster)).fetchall()
        trajectory = conn.execute('''SELECT status,members FROM padjective.residual_ensemble_trajectories
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (batch_id, fold, roster)).fetchone()
        assert trajectory[0] in ('complete', 'training_perfect')
        assert len(rows) == trajectory[1] and [r[0] for r in rows] == list(range(len(rows)))
        if trajectory[0] == 'complete':
            assert len(rows) == TRAJECTORIES[roster]
        candidates = np.unique(context['y_train'])
        exported, certified = [], 0
        for method in ('error_only', 'independent'):
            train_state = CumulativeConsensus(len(context['y_train']), candidates)
            test_state = CumulativeConsensus(len(context['y_test']), candidates)
            previous = None
            test_bank, train_bank = [], []
            stored = active = 0
            for index in range(TRAJECTORIES[roster]):
                selected = error_rows(context['y_train'], previous) if method == 'error_only' else np.arange(len(context['y_train']))
                if method == 'error_only' and index >= len(rows):
                    assert not len(selected) and trajectory[0] == 'training_perfect'
                    train_vote, test_vote = previous, test_state.predict()
                    forgotten = 0
                else:
                    model = rows[index][1] if method == 'error_only' else baseline[order_for(roster)[index]+fold]['model']
                    if method == 'error_only':
                        assert selected.tolist() == model['selected_rows']
                        assert model['seed'] == order_for(roster)[index]+fold
                    train, test, count = audited_member(context, model, selected, certify=method == 'error_only')
                    certified += count
                    train_vote, test_vote = train_state.add(train), test_state.add(test)
                    forgotten = 0 if previous is None else int(np.count_nonzero(
                        (previous == context['y_train']) & (train_vote != context['y_train'])))
                    test_bank.append(test)
                    train_bank.append(train)
                    weights = np.asarray(model['coefficients'])
                    stored += int(np.count_nonzero(weights))
                    active += float(np.asarray(context['x_test'] @ (weights != 0).astype(np.int64)).mean())
                    if method == 'error_only':
                        assert forgotten == model['forgotten_correct_rows']
                        assert int(np.count_nonzero(train_vote != context['y_train'])) == model['consensus_training_errors']
                if index+1 in SIZES:
                    for bank, vote in ((train_bank, train_vote), (test_bank, test_vote)):
                        prefix, _ = consensus_predictions(np.column_stack(bank), candidates, p=71, precision=7)
                        assert np.array_equal(prefix, vote)
                evidence = dict(fold=fold, roster=roster, method=method, members=index+1,
                    fitted_members=train_state.members, training_rows=len(selected),
                    training=prefix_count_scores(context['y_train'], train_vote),
                    held_out=prefix_count_scores(context['y_test'], test_vote),
                    forgotten_correct_rows=forgotten, stored_nonzero_coefficients=stored,
                    mean_active_terms=active, prediction_sha256=fingerprint(test_vote.tolist()))
                for part, y, pred in (('training', context['y_train'], train_vote),
                                      ('held_out', context['y_test'], test_vote)):
                    compare_scores(exact_scores(y.tolist(), pred.tolist(), 71, 7), evidence[part])
                exported.append(evidence)
                conn.execute('''INSERT INTO padjective.residual_ensemble_scores
                    (batch_id,cv_fold,roster,method,members,evidence) VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (batch_id,cv_fold,roster,method,members) DO UPDATE SET evidence=EXCLUDED.evidence''',
                    (batch_id, fold, roster, method, index+1, Jsonb(evidence)))
                previous = train_vote
            conn.commit()
        # Paired first-member predictions must be identical under both paths.
        assert exported[0]['prediction_sha256'] == exported[TRAJECTORIES[roster]]['prediction_sha256']
    emit('trajectory_audited', fold=fold, roster=roster, certified_coordinates=certified)
    return dict(rows=exported, certified_coordinates=certified)


def evaluate(args):
    with db.get_connection() as conn:
        row = conn.execute('SELECT manifest,status FROM padjective.residual_ensemble_batches '
                           'WHERE batch_id=%s', (args.batch_id,)).fetchone()
        assert row and row[1] in ('fits_complete', 'validated')
        assert row[0] == json.loads(json.dumps(source_manifest())), 'Source drift during audit'
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_trajectory, args.batch_id, fold, roster)
                   for roster in TRAJECTORIES for fold in range(5)]
        for future in as_completed(futures):
            results.append(future.result())
    report = dict(batch_id=args.batch_id, manifest=row[0],
        validation=dict(status='passed', trajectories=15,
            certified_coordinates=sum(r['certified_coordinates'] for r in results),
            audit='All masks, integer predictions, defaults, subset coordinate optima, consensus checkpoints, and two integer metrics'),
        rows=sorted([r for result in results for r in result['rows']],
                    key=lambda r: (r['roster'], r['method'], r['members'], r['fold'])))
    assert len(report['rows']) == 2*5*sum(TRAJECTORIES.values())
    with db.get_connection() as conn:
        conn.execute('''INSERT INTO padjective.residual_ensemble_reports (batch_id,report) VALUES (%s,%s)
            ON CONFLICT (batch_id) DO UPDATE SET report=EXCLUDED.report''', (args.batch_id, Jsonb(report)))
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='validated' WHERE batch_id=%s",
                     (args.batch_id,))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    emit('audit_complete', batch_id=args.batch_id, **report['validation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('fit', 'evaluate'))
    parser.add_argument('--batch-id', default=str(uuid.uuid4()))
    parser.add_argument('--workers', type=int, choices=range(1, 4), default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    (run if args.command == 'fit' else evaluate)(args)


if __name__ == '__main__':
    main()
