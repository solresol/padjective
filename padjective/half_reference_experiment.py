"""Frozen Postgres experiment: train against a fresh random half-reference vote."""
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
from psycopg.types.json import Jsonb

from . import db
from .half_reference_ensemble import SAMPLING_SEED, reference_consensus, reference_members
from .paper_audit_ensemble_scaling import prefix_count_scores
from .paper_ensemble_scaling_batch import SIZES
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_validate_published import compare_scores, exact_scores
from .paper_validate_randomised import fingerprint
from .randomised_linear import consensus_predictions, loss_units
from .residual_ensemble import CumulativeConsensus, error_rows, fit_error_member
from .residual_ensemble_experiment import (
    TRAJECTORIES, audited_member, emit, ensure_storage, load_baseline, load_fold, order_for,
)

CONTROL_BATCH = '06577b0c-a433-42bd-b8cd-a865838c11ad'
PROTOCOL = Path('docs/half-reference-ensemble-protocol.md')


def reference(bank, candidates, fold, roster):
    selected = reference_members(len(bank), fold=fold, roster=roster)
    if not len(selected):
        return selected, None
    return selected, reference_consensus(np.column_stack([bank[i] for i in selected]), candidates)


def manifest():
    paths = ('padjective/half_reference_ensemble.py', 'padjective/half_reference_experiment.py',
        'padjective/residual_ensemble.py', 'padjective/residual_ensemble_experiment.py',
        'padjective/randomised_linear.py', str(PROTOCOL))
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True)
    return json.loads(json.dumps(dict(
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        files_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
        protocol_text=PROTOCOL.read_text(), control_batch=CONTROL_BATCH,
        algorithm='half_reference', sampling_seed=SAMPLING_SEED, trajectories=TRAJECTORIES,
        snapshot_digest=SNAPSHOT_DIGEST, python=platform.python_version(),
        numpy=np.__version__, hostname=platform.node())))


def fit_trajectory(batch_id, fold, roster):
    limit = TRAJECTORIES[roster]
    with db.get_connection() as conn:
        c = load_fold(conn, fold)
        # Held-out labels are never used in this fitting path.
        x, y, test = (c[k] for k in ('x_train', 'y_train', 'x_test'))
        candidates = np.unique(y)
        state = CumulativeConsensus(len(y), candidates, max_members=limit)
        bank = []
        saved = conn.execute('''SELECT member_index,evidence FROM padjective.residual_ensemble_members
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s ORDER BY member_index''',
            (batch_id, fold, roster)).fetchall()
        assert [r[0] for r in saved] == list(range(len(saved)))
        for _, e in saved:
            ids, vote = reference(bank, candidates, fold, roster)
            assert ids.tolist() == e['reference_members']
            assert error_rows(y, vote).tolist() == e['selected_rows']
            bank.append(np.array(e['training_predictions'], dtype=np.int64))
            state.add(bank[-1])
        baseline = None if saved else load_baseline(conn, fold)[order_for(roster)[0]+fold]['model']
        status = 'complete'
        for index in range(len(saved), limit):
            started = time.monotonic()
            ids, vote = reference(bank, candidates, fold, roster)
            previous_full = state.predict() if bank else None
            member = fit_error_member(x, y, test, previous_predictions=vote,
                                      seed=order_for(roster)[index]+fold)
            if member is None:
                status = 'reference_perfect'
                break
            if index == 0:
                assert member.coefficients.tolist() == baseline['coefficients']
                assert member.default == baseline['default']
                assert member.predictions.tolist() == baseline['predictions']
                assert member.training_predictions.tolist() == baseline['training_predictions']
            updated = state.add(member.training_predictions)
            bank.append(member.training_predictions)
            e = dict(seed=order_for(roster)[index]+fold,
                reference_members=ids.tolist(),
                reference_prediction_sha256=None if vote is None else fingerprint(vote.tolist()),
                selected_rows=member.selected_rows.tolist(),
                selected_rows_sha256=fingerprint(member.selected_rows.tolist()),
                coefficients=member.coefficients.tolist(), default=member.default,
                training_predictions=member.training_predictions.tolist(),
                predictions=member.predictions.tolist(),
                coefficient_sha256=fingerprint(member.coefficients.tolist()),
                prediction_sha256=fingerprint(member.predictions.tolist()),
                training_prediction_sha256=fingerprint(member.training_predictions.tolist()),
                fit={k: v for k, v in asdict(member.fit).items()
                     if k not in ('coefficients', 'first_pass_coefficients', 'residuals')},
                full_training_raw_loss_units=int(loss_units(y-member.training_raw, 71, 7).sum()),
                selected_rows_count=len(member.selected_rows),
                consensus_training_errors=int(np.count_nonzero(updated != y)),
                forgotten_correct_rows=0 if previous_full is None else int(np.count_nonzero(
                    (previous_full == y) & (updated != y))), job_seconds=time.monotonic()-started)
            conn.execute('''INSERT INTO padjective.residual_ensemble_members
                (batch_id,cv_fold,roster,member_index,evidence) VALUES (%s,%s,%s,%s,%s)''',
                (batch_id, fold, roster, index, Jsonb(e)))
            conn.execute('''UPDATE padjective.residual_ensemble_trajectories SET members=%s
                WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (len(bank), batch_id, fold, roster))
            conn.commit()
            if len(bank) == 1 or len(bank) % 9 == 0 or len(bank) == limit:
                emit('fit_progress', fold=fold, roster=roster, members=len(bank), limit=limit,
                     reference_members=len(ids), training_rows=len(member.selected_rows),
                     training_errors=e['consensus_training_errors'])
        conn.execute('''UPDATE padjective.residual_ensemble_trajectories SET status=%s,members=%s
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (status, len(bank), batch_id, fold, roster))
    return dict(fold=fold, roster=roster, members=len(bank), status=status)


def run(args):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '1', f'Set {name}=1 before starting Python'
    specification = manifest()
    with db.get_connection() as conn:
        ensure_storage(conn)
        control = conn.execute('SELECT status FROM padjective.residual_ensemble_batches WHERE batch_id=%s',
                               (CONTROL_BATCH,)).fetchone()
        assert control and control[0] == 'validated'
        prior = conn.execute('SELECT manifest,status FROM padjective.residual_ensemble_batches WHERE batch_id=%s',
                             (args.batch_id,)).fetchone()
        if prior:
            assert prior[0] == specification and prior[1] in ('fitting', 'failed', 'fits_complete')
            conn.execute("UPDATE padjective.residual_ensemble_batches SET status='fitting',error=NULL WHERE batch_id=%s",
                         (args.batch_id,))
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
    emit('batch_started', batch_id=args.batch_id, workers=args.workers, pending_trajectories=len(pending))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fit_trajectory, args.batch_id, fold, roster) for fold, roster in pending]
            for future in as_completed(futures):
                emit('trajectory_finished', **future.result())
    except BaseException as exc:
        with db.get_connection() as conn:
            conn.execute("UPDATE padjective.residual_ensemble_batches SET status='failed',error=%s WHERE batch_id=%s",
                         (str(exc), args.batch_id))
        raise
    with db.get_connection() as conn:
        assert conn.execute("SELECT count(*) FROM padjective.residual_ensemble_trajectories WHERE batch_id=%s "
            "AND status IN ('complete','reference_perfect')", (args.batch_id,)).fetchone()[0] == 15
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='fits_complete',finished_at=NOW() WHERE batch_id=%s",
                     (args.batch_id,))
    emit('fits_complete', batch_id=args.batch_id)


def evaluate_trajectory(batch_id, fold, roster):
    with db.get_connection() as conn:
        c = load_fold(conn, fold)
        y, candidates = c['y_train'], np.unique(c['y_train'])
        records = conn.execute('''SELECT member_index,evidence FROM padjective.residual_ensemble_members
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s ORDER BY member_index''',
            (batch_id, fold, roster)).fetchall()
        status, count = conn.execute('''SELECT status,members FROM padjective.residual_ensemble_trajectories
            WHERE batch_id=%s AND cv_fold=%s AND roster=%s''', (batch_id, fold, roster)).fetchone()
        assert status in ('complete', 'reference_perfect')
        assert len(records) == count and [r[0] for r in records] == list(range(count))
        assert count == TRAJECTORIES[roster] or status == 'reference_perfect'
        states = [CumulativeConsensus(len(c[k]), candidates) for k in ('y_train', 'y_test')]
        banks = [[], []]
        exported, certified, ref_checks = [], 0, 0
        active = stored = 0
        previous_full = None
        for index in range(TRAJECTORIES[roster]):
            if index < count:
                ids, vote = reference(banks[0], candidates, fold, roster)
                selected = error_rows(y, vote)
                model = records[index][1]
                assert ids.tolist() == model['reference_members']
                assert selected.tolist() == model['selected_rows']
                assert (None if vote is None else fingerprint(vote.tolist())) == model['reference_prediction_sha256']
                assert model['seed'] == order_for(roster)[index]+fold
                train, test, checked = audited_member(c, model, selected, certify=True)
                certified += checked
                if index+1 in SIZES and vote is not None:
                    prefix, _ = consensus_predictions(np.column_stack([banks[0][i] for i in ids]),
                                                       candidates, p=71, precision=7)
                    assert np.array_equal(prefix, vote)
                    ref_checks += 1
                votes = [state.add(pred) for state, pred in zip(states, (train, test), strict=True)]
                for bank, pred in zip(banks, (train, test), strict=True):
                    bank.append(pred)
                weights = np.array(model['coefficients'])
                stored += int(np.count_nonzero(weights))
                active += float(np.asarray(c['x_test'] @ (weights != 0).astype(np.int64)).mean())
                forgotten = 0 if previous_full is None else int(np.count_nonzero(
                    (previous_full == y) & (votes[0] != y)))
                assert forgotten == model['forgotten_correct_rows']
                assert int(np.count_nonzero(votes[0] != y)) == model['consensus_training_errors']
            else:
                if index == count:
                    ids, vote = reference(banks[0], candidates, fold, roster)
                    assert status == 'reference_perfect' and not len(error_rows(y, vote))
                selected, forgotten = [], 0
                votes = [s.predict() for s in states]
            if index+1 in SIZES:
                for bank, vote in zip(banks, votes, strict=True):
                    prefix, _ = consensus_predictions(np.column_stack(bank), candidates, p=71, precision=7)
                    assert np.array_equal(prefix, vote)
            evidence = dict(fold=fold, roster=roster, method='half_reference', members=index+1,
                fitted_members=len(banks[0]), reference_members_count=len(ids), training_rows=len(selected),
                training=prefix_count_scores(y, votes[0]), held_out=prefix_count_scores(c['y_test'], votes[1]),
                forgotten_correct_rows=forgotten, stored_nonzero_coefficients=stored,
                mean_active_terms=active, prediction_sha256=fingerprint(votes[1].tolist()))
            for part, actual, pred in (('training', y, votes[0]), ('held_out', c['y_test'], votes[1])):
                compare_scores(exact_scores(actual.tolist(), pred.tolist(), 71, 7), evidence[part])
            conn.execute('''INSERT INTO padjective.residual_ensemble_scores
                (batch_id,cv_fold,roster,method,members,evidence) VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (batch_id,cv_fold,roster,method,members) DO UPDATE SET evidence=EXCLUDED.evidence''',
                (batch_id, fold, roster, 'half_reference', index+1, Jsonb(evidence)))
            exported.append(evidence)
            previous_full = votes[0]
        conn.commit()
    emit('trajectory_audited', fold=fold, roster=roster, certified_coordinates=certified,
         reference_checkpoint_checks=ref_checks)
    return dict(rows=exported, certified_coordinates=certified, reference_checkpoint_checks=ref_checks)


def evaluate(args):
    with db.get_connection() as conn:
        specification, status = conn.execute('SELECT manifest,status FROM padjective.residual_ensemble_batches '
            'WHERE batch_id=%s', (args.batch_id,)).fetchone()
        assert status in ('fits_complete', 'validated') and specification == manifest()
        control = conn.execute('SELECT report FROM padjective.residual_ensemble_reports WHERE batch_id=%s',
                               (CONTROL_BATCH,)).fetchone()[0]
        assert control['validation']['status'] == 'passed'
        assert control['manifest']['snapshot_digest'] == SNAPSHOT_DIGEST
        assert len(control['rows']) == 4050
        persisted = conn.execute('SELECT cv_fold,roster,method,members,evidence FROM '
            'padjective.residual_ensemble_scores WHERE batch_id=%s', (CONTROL_BATCH,)).fetchall()
        mapping = {(f, r, method, m): e for f, r, method, m, e in persisted}
        assert len(mapping) == 4050
        assert all(mapping[(r['fold'], r['roster'], r['method'], r['members'])] == r for r in control['rows'])
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_trajectory, args.batch_id, fold, roster)
                   for roster in TRAJECTORIES for fold in range(5)]
        for future in as_completed(futures):
            results.append(future.result())
    new_rows = [row for result in results for row in result['rows']]
    assert len(new_rows) == 2025
    for row in new_rows:
        if row['members'] <= 2:
            method = 'independent' if row['members'] == 1 else 'error_only'
            assert row['prediction_sha256'] == mapping[(row['fold'], row['roster'], method, row['members'])]['prediction_sha256']
    report = dict(batch_id=args.batch_id, manifest=specification,
        validation=dict(status='passed', trajectories=15,
            certified_coordinates=sum(r['certified_coordinates'] for r in results),
            reference_checkpoint_checks=sum(r['reference_checkpoint_checks'] for r in results),
            control_rows_verified=4050),
        rows=sorted(new_rows+control['rows'], key=lambda r: (r['roster'], r['method'], r['members'], r['fold'])))
    with db.get_connection() as conn:
        conn.execute('''INSERT INTO padjective.residual_ensemble_reports (batch_id,report) VALUES (%s,%s)
            ON CONFLICT (batch_id) DO UPDATE SET report=EXCLUDED.report''', (args.batch_id, Jsonb(report)))
        conn.execute("UPDATE padjective.residual_ensemble_batches SET status='validated' WHERE batch_id=%s", (args.batch_id,))
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
