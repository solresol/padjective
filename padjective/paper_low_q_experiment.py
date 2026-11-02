"""Retrain low-Q regressors on the fixed PostgreSQL paper snapshot.

Private predictions/coefficients stay in PostgreSQL. Export aggregate scores
only. The reference metric keeps its base-71 encoding; the depth weight varies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np
from psycopg.types.json import Jsonb

from . import benchmark_runtime as runtime, db, low_q_linear as low
from .paper_ensemble_scaling_batch import BANK_SEEDS
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset, q_weighted_distance
from .randomised_linear import apply_default, linear_predictions
from .subspace_voting import feature_mask


EXPERIMENT = "low-q-2026-11-02-v2"
SOURCE_BATCH = "773bf2ab-d2b1-4c32-bfde-15d501782fa0"
Q_VALUES = {"sqrt2": math.sqrt(2), "2": 2., "3": 3., "5": 5., "10": 10., "71": 71.}
ORDERS = ("taxonomy_association", "random", "frequency", "battle_elo", "mean_title_position")
RANDOM_SEEDS = (7, 13, 23, 37, 101)
_WORKER_EXECUTE = None


def _execute_job(job):
    # Linux fork preserves the read-only dataset and the local job closure.
    # Each process opens its own PostgreSQL connections.
    return _WORKER_EXECUTE(job)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def score(actual, predicted):
    histogram, exact = {}, 0
    for a, b in zip(actual, predicted, strict=True):
        delta = abs(int(a) - int(b))
        if not delta:
            exact += 1
            continue
        depth = 0
        while delta % 71 == 0:
            delta //= 71
            depth += 1
        histogram[depth] = histogram.get(depth, 0) + 1
    n = len(actual)
    assert n and exact + sum(histogram.values()) == n
    losses = {key: math.fsum(count * q ** (-depth) for depth, count in histogram.items()) / n
              for key, q in Q_VALUES.items()}
    # Independent row-by-row implementation checks the requested low-Q score.
    direct = math.fsum(q_weighted_distance(int(a), int(b), prime_base=71, q=math.sqrt(2))
                       for a, b in zip(actual, predicted, strict=True)) / n
    assert abs(direct - losses["sqrt2"]) < 1e-12
    return dict(n=n, exact_correct=exact, first_disagreement_counts=histogram,
                exact_accuracy=exact / n, root_accuracy=1 - histogram.get(0, 0) / n,
                losses=losses, prediction_sha256=fingerprint([int(v) for v in predicted]))


def ensure_storage(conn):
    db.ensure_table(conn, "padjective", "paper_low_q_models", [
        "experiment_key TEXT NOT NULL", "q_label TEXT NOT NULL", "family TEXT NOT NULL",
        "cv_fold INTEGER NOT NULL", "fraction DOUBLE PRECISION NOT NULL", "member_index INTEGER NOT NULL",
        "status TEXT NOT NULL", "evidence JSONB NOT NULL", "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "PRIMARY KEY(experiment_key,q_label,family,cv_fold,fraction,member_index)"])
    db.ensure_table(conn, "padjective", "paper_low_q_reports", [
        "experiment_key TEXT PRIMARY KEY", "report JSONB NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()"])


def fit_greedy(dataset, context, *, kind, strategy, seed, battles):
    if kind == '71':
        original = runtime.umllr_run_fold(context['fold'], dataset.records, battles, 71,
                                          tag_order_strategy=strategy, tag_order_seed=seed)
        predictions = [r.predicted_value for r in original.predictions]
        return dict(status='one_pass_complete',
                    coefficients=[dict(tag=c.tag, coefficient=c.coefficient) for c in original.coefficients],
                    default=original.default_prediction, predictions=predictions,
                    scores=score(context['y_test'], predictions), tag_order_seed=seed,
                    reference_implementation='Original historical floating-point greedy fitter')
    train = context['training_records']
    order = runtime._tag_order(train, battles, context['fold'], strategy=strategy, seed=seed)
    matrix = context['x_train'].tocsc()
    residuals = context['y_train'].copy()
    coefficients = np.zeros(matrix.shape[1], dtype=np.int64)
    columns = {tag: i for i, tag in enumerate(dataset.feature_names)}
    for tag in order:
        col = columns[tag]
        rows = matrix.indices[matrix.indptr[col]:matrix.indptr[col + 1]]
        value, _ = low.best_choice(residuals[rows], kind=kind)
        coefficients[col] = value
        residuals[rows] -= value
    train_raw = np.asarray(context['x_train'] @ coefficients).reshape(-1)
    test_raw = np.asarray(context['x_test'] @ coefficients).reshape(-1)
    zero = train_raw == 0
    if np.any(zero):
        default = low.best_choice(context['y_train'][zero], kind=kind,
                                  candidates=context['y_train'])[0]
    else:
        default = runtime._select_default_prediction([], context['y_train'].tolist(), 71)
    predictions = apply_default(test_raw, default)
    return dict(status='one_pass_complete', coefficients=coefficients.tolist(),
                default=default, predictions=predictions.tolist(),
                scores=score(context['y_test'], predictions), tag_order_seed=seed)


def run(args):
    source_hash = hashlib.sha256(Path(low.__file__).read_bytes()).hexdigest()
    with db.get_connection() as conn:
        ensure_storage(conn)
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
        assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST and dataset.features.shape == (6693, 2542)
        reference = conn.execute("SELECT report FROM padjective.paper_subspace_reports WHERE batch_id=%s",
                                 (SOURCE_BATCH,)).fetchone()[0]
        assert reference['validation']['status'] == 'passed'
        order_report = conn.execute("SELECT report FROM padjective.paper_order_significance "
                                    "ORDER BY created_at DESC LIMIT 1").fetchone()[0]
        order_controls = {(r['tag_order_strategy'], r['tag_order_seed'], f['cv_fold']): f
                          for r in order_report['runs'] for f in r['folds']}
        rosters = {}
        for fold in range(5):
            e = conn.execute("SELECT evidence FROM padjective.paper_subspace_results WHERE batch_id=%s "
                             "AND cv_fold=%s AND fraction=1 AND roster=0 AND members=243 AND rule='raw_valid_medoid'",
                             (SOURCE_BATCH, fold)).fetchone()[0]
            rosters[fold] = e['member_indices']
            assert len(rosters[fold]) == 243 and set(rosters[fold]) == set(range(243))
    folds = np.array([r.cv_fold for r in dataset.records])
    y = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    context = {}
    for f in range(5):
        training = folds != f
        context[f] = dict(fold=f, x_train=dataset.features[training].astype(np.int64).tocsr(),
                          x_test=dataset.features[~training].astype(np.int64).tocsr(),
                          y_train=y[training], y_test=y[~training],
                          training_records=[r for r in dataset.records if r.cv_fold != f])
    battles = runtime._derive_battles(dataset.records)
    jobs = []
    for kind in ('71', 'sqrt2', '2'):
        for strategy in ORDERS:
            for seed in (RANDOM_SEEDS if strategy == 'random' else (None,)):
                jobs += [(kind, 'greedy_' + strategy, f, 1., -1 if seed is None else seed) for f in range(5)]
    # The primary roster is preserved: a 15-member pilot is a prefix of the
    # archived primary 243-member roster, not a newly selected seed bank.
    for position in range(args.members):
        for kind in ('sqrt2', '2'):
            for fraction in (1., .75):
                jobs += [(kind, 'coordinate', f, fraction, rosters[f][position]) for f in range(5)]

    def execute(job):
        kind, family, fold, fraction, index = job
        key = (args.experiment_key, *job)
        with db.get_connection() as conn:
            existing = conn.execute("SELECT status,evidence FROM padjective.paper_low_q_models WHERE "
                "experiment_key=%s AND q_label=%s AND family=%s AND cv_fold=%s AND fraction=%s AND member_index=%s", key).fetchone()
            if existing and existing[0] in ('one_pass_complete', 'coordinate_optimum'):
                assert existing[1]['algorithm_sha256'] == source_hash
                return job, existing[1]
        started = time.monotonic()
        c = context[fold]
        if family.startswith('greedy_'):
            result = fit_greedy(dataset, c, kind=kind, strategy=family[7:],
                                seed=None if index == -1 else index, battles=battles)
            if kind == '71':
                control = order_controls[(family[7:], None if index == -1 else index, fold)]
                assert abs(result['scores']['losses']['71'] - control['padic_loss_mean']) < 1e-10
                assert abs(result['scores']['exact_accuracy'] - control['exact_accuracy']) < 1e-12
        else:
            base = BANK_SEEDS[index]
            eligible = np.flatnonzero(c['x_train'].getnnz(axis=0))
            # All-feature fits retain the original design column indices.
            mask = np.arange(len(dataset.feature_names)) if fraction == 1. else feature_mask(eligible, fraction, fold, base)
            a, b = c['x_train'][:, mask], c['x_test'][:, mask]
            fit = low.fit_coordinates(a, c['y_train'], kind=kind, seed=base + fold,
                                      seconds=300, max_sweeps=100)
            assert fit['status'] == 'coordinate_optimum', ('Incomplete fit', job, fit['status'])
            check = low.certificate(a, c['y_train'], fit['coefficients'], kind=kind)
            assert check['status'] == 'coordinate_optimum'
            train_raw = linear_predictions(a, fit['coefficients'], 71, 7)
            test_raw = linear_predictions(b, fit['coefficients'], 71, 7)
            default = low.fit_default(c['y_train'], train_raw, kind=kind)
            predictions = apply_default(test_raw, default)
            result = dict(status=fit['status'], coefficients=fit['coefficients'].tolist(),
                          default=default, predictions=predictions.tolist(),
                          history=fit['history'], certificate=check, seed=base + fold,
                          mask_sha256=fingerprint(mask.tolist()), scores=score(c['y_test'], predictions))
        result.update(algorithm_sha256=source_hash, snapshot_digest=SNAPSHOT_DIGEST,
                      elapsed_seconds=time.monotonic() - started)
        with db.get_connection() as conn:
            conn.execute("INSERT INTO padjective.paper_low_q_models "
                "(experiment_key,q_label,family,cv_fold,fraction,member_index,status,evidence) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT "
                "(experiment_key,q_label,family,cv_fold,fraction,member_index) DO UPDATE "
                "SET status=EXCLUDED.status,evidence=EXCLUDED.evidence,updated_at=NOW()",
                (*key, result['status'], Jsonb(result)))
        return job, result

    records = {}
    global _WORKER_EXECUTE
    _WORKER_EXECUTE = execute
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('fork')) as pool:
        futures = [pool.submit(_execute_job, job) for job in jobs]
        try:
            for number, future in enumerate(as_completed(futures), 1):
                job, evidence = future.result()
                records[job] = evidence
                if number % 20 == 0 or number == len(jobs):
                    print(json.dumps(dict(event='fit_progress', completed=number, total=len(jobs))), flush=True)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    assert len(records) == len(jobs)
    report = evaluate(args, dataset, context, records, rosters, source_hash)
    with db.get_connection() as conn:
        conn.execute("INSERT INTO padjective.paper_low_q_reports (experiment_key,report) VALUES (%s,%s) "
                     "ON CONFLICT(experiment_key) DO UPDATE SET report=EXCLUDED.report,updated_at=NOW()",
                     (args.experiment_key, Jsonb(report)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(event='complete', output=str(args.output), configurations=len(report['series']))), flush=True)


def evaluate(args, dataset, context, records, rosters, source_hash):
    series = []

    def add(label, train_q, scores):
        assert set(scores) == set(range(5))
        item = dict(configuration=label, training_q=train_q, folds=[scores[f] for f in range(5)])
        item['mean_losses'] = {k: mean(scores[f]['losses'][k] for f in range(5)) for k in Q_VALUES}
        item['fold_sd'] = {k: stdev(scores[f]['losses'][k] for f in range(5)) for k in Q_VALUES}
        item['exact_accuracy'] = mean(scores[f]['exact_accuracy'] for f in range(5))
        item['root_accuracy'] = mean(scores[f]['root_accuracy'] for f in range(5))
        series.append(item)

    for kind in ('71', 'sqrt2', '2'):
        for strategy in ORDERS:
            rows = {}
            for f in range(5):
                parts = [records[(kind, 'greedy_' + strategy, f, 1., seed)]['scores']
                         for seed in (RANDOM_SEEDS if strategy == 'random' else (-1,))]
                rows[f] = dict(n=len(context[f]['y_test']),
                    losses={k: mean(p['losses'][k] for p in parts) for k in Q_VALUES},
                    exact_accuracy=mean(p['exact_accuracy'] for p in parts),
                    root_accuracy=mean(p['root_accuracy'] for p in parts))
            add(strategy, kind, rows)
    for kind in ('sqrt2', '2'):
        for fraction in (1., .75):
            for size in (1, 15, 243):
                if size > args.members:
                    continue
                rows = {}
                for f in range(5):
                    bank = np.array([records[(kind, 'coordinate', f, fraction, j)]['predictions']
                                     for j in rosters[f][:size]], dtype=np.int64).T
                    pred = low.consensus_predictions(bank, context[f]['y_train'], kind=kind)
                    rows[f] = score(context[f]['y_test'], pred)
                add(f'ensemble_{fraction:g}_{size}', kind, rows)
    with db.get_connection() as conn:
        for fraction in (1., .75):
            for size in (1, 15, 243):
                rows = {}
                for f in range(5):
                    old = conn.execute("SELECT evidence FROM padjective.paper_subspace_results WHERE batch_id=%s "
                        "AND cv_fold=%s AND fraction=%s AND roster=0 AND members=%s AND rule='raw_valid_medoid'",
                        (SOURCE_BATCH, f, fraction, size)).fetchone()[0]
                    rows[f] = score(context[f]['y_test'], old['predictions'])
                    assert rows[f]['prediction_sha256'] == old['prediction_sha256']
                    assert abs(rows[f]['losses']['71'] - old['metrics']['mean_padic_loss']) < 1e-12
                add(f'ensemble_{fraction:g}_{size}', '71', rows)
        for strategy in ORDERS:
            expected = conn.execute("SELECT avg(mean_loss) FROM padjective.paper_revision_q_sensitivity "
                "WHERE snapshot_ref='paper' AND q=71 AND tag_order_strategy=%s", (strategy,)).fetchone()[0]
            actual = next(s for s in series if s['training_q'] == '71' and s['configuration'] == strategy)
            assert abs(actual['mean_losses']['71'] - expected) < 1e-10, (strategy, actual['mean_losses']['71'], expected)
    rankings = {}
    for eval_q in Q_VALUES:
        comparable = [s for s in series if not s['configuration'].startswith('ensemble_')
                      or int(s['configuration'].rsplit('_', 1)[1]) <= args.members]
        rankings[eval_q] = [dict(configuration=s['configuration'], training_q=s['training_q'],
                                loss=s['mean_losses'][eval_q])
                            for s in sorted(comparable, key=lambda s: s['mean_losses'][eval_q])]
    return dict(experiment_key=args.experiment_key, created_at=datetime.now(timezone.utc).isoformat(),
                snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST, products=6693, prime=71,
                precision=7, members_retrained=args.members, q_values=Q_VALUES,
                workers=args.workers, parallelism='forked single-threaded worker processes',
                algorithm_sha256=source_hash, classical_baselines_included=False,
                reference_greedy_implementation='Original historical fitter, fold loss and exact accuracy verified',
                interpretation='Exploratory reused-fold comparison; fixed original fits rescored and low-Q fits retrained. '
                               'Ensemble consensus uses the training Q. No hyperparameters selected from these scores.',
                series=series, rankings=rankings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--members', type=int, choices=(15, 243), default=15)
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=4)
    parser.add_argument('--experiment-key', default=EXPERIMENT)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
