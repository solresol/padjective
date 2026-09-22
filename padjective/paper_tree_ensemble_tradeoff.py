"""Isolated Postgres experiment comparing pruned trees and linear ensembles."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from . import db
from .paper_audit_ensemble_scaling import prefix_count_scores
from .paper_ensemble_scaling_batch import SIZES
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_validate_published import exact_scores, compare_scores
from .paper_validate_randomised import fingerprint

SUBSPACE_BATCH = '773bf2ab-d2b1-4c32-bfde-15d501782fa0'
PROTOCOL = Path('docs/tree-ensemble-tradeoff-protocol.md')
DEPTHS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, None)
LEAVES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
ALPHAS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1)
FOREST_DEPTHS = (2, 4, 8, 16, 32, 64, 128, None)
FOREST_SIZES = (1, 3, 9, 27, 81)
SEEDS = (42, 1729, 20260922)
CONTEXT = {}


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def tree_configs():
    for weight in ('balanced', None):
        for sweep, values, argument in (('depth', DEPTHS, 'max_depth'),
                ('leaves', LEAVES, 'max_leaf_nodes'), ('alpha', ALPHAS, 'ccp_alpha')):
            for value in values:
                yield dict(family='tree', class_weight=weight, seed=42,
                           sweep=sweep, **{argument: value})


def forest_configs():
    for weight in ('balanced', None):
        for depth in FOREST_DEPTHS:
            for seed in SEEDS:
                yield dict(family='forest', class_weight=weight, seed=seed, max_depth=depth)


def config_key(config):
    return json.dumps(config, sort_keys=True, separators=(',', ':'))


def load_context():
    with db.get_connection() as conn:
        data = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema='padjective')
    assert _snapshot_digest(data) == SNAPSHOT_DIGEST
    assert data.features.shape == (6693, 2542) and data.prime_base == 71
    folds = np.array([r.cv_fold for r in data.records])
    targets = np.array([r.encoded_path for r in data.records], dtype=np.int64)
    return {f: dict(x_train=data.features[folds != f], x_test=data.features[folds == f],
        labels_train=data.labels[folds != f], y_train=targets[folds != f],
        y_test=targets[folds == f], codes=data.encoded_by_taxonomy,
        row_sha256=fingerprint([r.product_key for r in data.records if r.cv_fold == f]))
        for f in range(5)}


def initialise():
    global CONTEXT
    CONTEXT = load_context()


def ensure_storage(conn):
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace='pg_default'")
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_tree_tradeoff_batches (
            batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            manifest JSONB NOT NULL, status TEXT NOT NULL, report JSONB
        ) TABLESPACE pg_default''')
        cur.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_tree_tradeoff_runs (
            batch_id UUID NOT NULL, cv_fold INTEGER NOT NULL, config_key TEXT NOT NULL,
            configuration JSONB NOT NULL, evidence JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY(batch_id,cv_fold,config_key)
        ) TABLESPACE pg_default''')
    conn.commit()


def score(y, pred):
    pred = np.asarray(pred, dtype=np.int64)
    result = exact_scores(y.tolist(), pred.tolist(), 71, 7)
    compare_scores(result, prefix_count_scores(y, pred))
    return result


def tree_counts(model, x, *, probabilities=False):
    """Count logical inference state, and independently validate path lengths."""
    t = model.tree_
    leaf = t.children_left == -1
    internal, leaves = int(np.count_nonzero(~leaf)), int(np.count_nonzero(leaf))
    paths = np.asarray(model.decision_path(x).sum(axis=1)).ravel().astype(int)-1
    reached = model.apply(x)
    for i in np.unique(np.linspace(0, x.shape[0]-1, min(12, x.shape[0]), dtype=int)):
        row = x[i].toarray().ravel()
        node, steps = 0, 0
        while t.children_left[node] != -1:
            node = t.children_left[node] if row[t.feature[node]] <= t.threshold[node] else t.children_right[node]
            steps += 1
        assert node == reached[i] and steps == paths[i]
    leaf_nnz = np.count_nonzero(t.value[:, 0, :], axis=1)
    nonzero_probabilities = int(leaf_nnz[leaf].sum())
    slots = 4*internal + (2*nonzero_probabilities+1 if probabilities else leaves)
    h = hashlib.sha256()
    for values in (t.children_left, t.children_right, t.feature, t.threshold, t.value):
        h.update(np.asarray(values).tobytes())
    return dict(internal_nodes=internal, leaves=leaves,
        leaf_nonzero_probabilities=nonzero_probabilities, stored_slots=slots,
        max_depth=int(t.max_depth), model_sha256=h.hexdigest()), paths, leaf_nnz[reached]


def cost_summary(values):
    return dict(mean=float(np.mean(values)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=int(np.max(values)))


def encode(labels, codes):
    return np.array([codes[str(v)] for v in labels], dtype=np.int64)


def save_result(conn, batch, fold, config, evidence):
    key = config_key(config)
    conn.execute('''INSERT INTO padjective.paper_tree_tradeoff_runs
        (batch_id,cv_fold,config_key,configuration,evidence) VALUES (%s,%s,%s,%s,%s)''',
        (batch, fold, key, Jsonb(config), Jsonb(evidence)))
    conn.commit()


def fit_job(batch, fold, config):
    c = CONTEXT[fold]
    args = {k: v for k, v in config.items() if k not in ('family', 'seed', 'sweep')}
    args['random_state'] = config['seed']
    started = time.monotonic()
    with db.get_connection() as conn:
        if config['family'] == 'tree':
            model = DecisionTreeClassifier(**args).fit(c['x_train'], c['labels_train'])
            fitted = time.monotonic()-started
            pred = encode(model.predict(c['x_test']), c['codes'])
            train = encode(model.predict(c['x_train']), c['codes'])
            counts, paths, _ = tree_counts(model, c['x_test'])
            evidence = dict(metrics=score(c['y_test'], pred), training=score(c['y_train'], train),
                predictions=pred.tolist(), prediction_sha256=fingerprint(pred.tolist()),
                row_sha256=c['row_sha256'], active=cost_summary(paths),
                broader_work=float(np.mean(paths)+1), fit_seconds=fitted, counts=counts)
            save_result(conn, batch, fold, config, evidence)
        else:
            # One fixed bank, prefix evaluation: all class columns share model.classes_.
            model = RandomForestClassifier(n_estimators=max(FOREST_SIZES),
                max_features='sqrt', bootstrap=True, n_jobs=1, **args).fit(c['x_train'], c['labels_train'])
            fitted = time.monotonic()-started
            full_estimators = list(model.estimators_)
            test_sum = np.zeros((len(c['y_test']), len(model.classes_)))
            train_sum = np.zeros((len(c['y_train']), len(model.classes_)))
            paths = np.zeros(len(c['y_test']), dtype=np.int64)
            contributions = np.zeros(len(c['y_test']), dtype=np.int64)
            counts = dict(internal_nodes=0, leaves=0, leaf_nonzero_probabilities=0, stored_slots=0)
            hashes, depths = [], []
            for n, estimator in enumerate(full_estimators, 1):
                test_sum += estimator.predict_proba(c['x_test'])
                train_sum += estimator.predict_proba(c['x_train'])
                stats, one_path, one_contrib = tree_counts(estimator, c['x_test'], probabilities=True)
                paths += one_path
                contributions += one_contrib
                for k in counts:
                    counts[k] += stats[k]
                hashes.append(stats['model_sha256'])
                depths.append(stats['max_depth'])
                if n not in FOREST_SIZES:
                    continue
                pred = encode(model.classes_[np.argmax(test_sum, axis=1)], c['codes'])
                train = encode(model.classes_[np.argmax(train_sum, axis=1)], c['codes'])
                model.estimators_ = full_estimators[:n]
                expected = encode(model.predict(c['x_test']), c['codes'])
                assert np.array_equal(pred, expected)
                np.testing.assert_allclose(model.predict_proba(c['x_test']), test_sum/n, rtol=0, atol=2e-14)
                evidence = dict(metrics=score(c['y_test'], pred), training=score(c['y_train'], train),
                    predictions=pred.tolist(), prediction_sha256=fingerprint(pred.tolist()),
                    row_sha256=c['row_sha256'], active=cost_summary(paths),
                    broader_work=float(np.mean(paths+contributions)+len(model.classes_)-1),
                    full_bank_fit_seconds=fitted, counts=dict(counts, mean_depth=float(np.mean(depths)),
                        max_depth=max(depths), member_hashes=list(hashes)))
                save_result(conn, batch, fold, dict(config, members=n), evidence)
    return dict(fold=fold, configuration=config, job_seconds=time.monotonic()-started)


def reuse_linear(conn, batch, context):
    source = conn.execute('SELECT report FROM padjective.paper_subspace_reports WHERE batch_id=%s',
                          (SUBSPACE_BATCH,)).fetchone()[0]
    assert source['validation']['status'] == 'passed'
    assert source['manifest']['snapshot_digest'] == SNAPSHOT_DIGEST
    components = source['components']
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('''SELECT cv_fold,fraction,members,evidence FROM padjective.paper_subspace_results
            WHERE batch_id=%s AND fraction IN (0.75,1.0) AND roster=0
            AND rule='raw_valid_medoid' ORDER BY fraction,cv_fold,members''', (SUBSPACE_BATCH,))
        rows = cur.fetchall()
    assert len(rows) == 90
    for row in rows:
        fold, fraction, size, old = row['cv_fold'], row['fraction'], row['members'], row['evidence']
        c = context[fold]
        assert size in SIZES and fingerprint(old['predictions']) == old['prediction_sha256']
        compare_scores(score(c['y_test'], old['predictions']), old['metrics'])
        candidates = np.unique(c['y_train']).tolist()
        assert fingerprint(candidates) == old['candidate_sha256']
        assert all(v in candidates for v in old['predictions'])
        members = [m for m in components if m['fold']==fold and m['fraction']==fraction and m['member_index']<size]
        assert len(members)==size
        nonzero = sum(m['stored_nonzero_coefficients'] for m in members)
        assert nonzero == old['stored_nonzero_coefficients']
        active = sum(m['mean_active_terms'] for m in members)
        assert abs(active-old['mean_active_terms'])<1e-10
        config = dict(family='padic', fraction=fraction, members=size, seed='fixed_primary_roster')
        evidence = dict(metrics=old['metrics'], predictions=old['predictions'],
            prediction_sha256=old['prediction_sha256'], row_sha256=c['row_sha256'],
            active=dict(mean=active), source_batch=SUBSPACE_BATCH,
            # Default uses are retained as a bound here; their omission changes <m units.
            broader_work_lower=active+7*size+7*len(candidates),
            broader_work_upper=active+8*size+7*len(candidates),
            counts=dict(stored_slots=2*nonzero+size+len(candidates),
                        nonzero_coefficients=nonzero, candidate_paths=len(candidates)))
        save_result(conn, batch, fold, config, evidence)
    return len(rows)


def run(args):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        assert os.environ.get(name) == '1', f'Set {name}=1 before starting Python'
    assert 1 <= args.workers <= 2
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
    specification = dict(snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), protocol_text=PROTOCOL.read_text(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__,
        hostname=platform.node(), workers=args.workers, source_subspace_batch=SUBSPACE_BATCH,
        trees=list(tree_configs()), forests=list(forest_configs()), forest_sizes=FOREST_SIZES)
    batch = args.batch_id or str(uuid.uuid4())
    context = load_context()
    with db.get_connection() as conn:
        ensure_storage(conn)
        conn.execute('''INSERT INTO padjective.paper_tree_tradeoff_batches
            (batch_id,manifest,status) VALUES (%s,%s,'running')''', (batch, Jsonb(specification)))
        conn.commit()
        reused = reuse_linear(conn, batch, context)
    jobs = [(fold, config) for config in [*tree_configs(), *forest_configs()] for fold in range(5)]
    emit('started', batch_id=batch, jobs=len(jobs), reused_rows=reused)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialise) as pool:
        futures = [pool.submit(fit_job,batch,fold,config) for fold,config in jobs]
        for done, future in enumerate(as_completed(futures),1):
            result = future.result()
            if done%10 == 0 or done==len(jobs):
                emit('progress', done=done, total=len(jobs), **result)
    with db.get_connection() as conn:
        export_report(conn, batch, args.output)
    emit('complete', batch_id=batch, output=str(args.output))


def export_report(conn, batch, output):
    """Read back every private prediction; only export aggregate evidence."""
    context = load_context()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute('SELECT manifest FROM padjective.paper_tree_tradeoff_batches WHERE batch_id=%s',(batch,))
        manifest = cur.fetchone()['manifest']
        cur.execute('''SELECT cv_fold,configuration,evidence FROM padjective.paper_tree_tradeoff_runs
            WHERE batch_id=%s ORDER BY config_key,cv_fold''',(batch,))
        rows=cur.fetchall()
    expected = 90 + 5*len(list(tree_configs())) + 5*len(list(forest_configs()))*len(FOREST_SIZES)
    assert len(rows)==expected
    keys=set()
    output_rows=[]
    for row in rows:
        f,c,e=row['cv_fold'],row['configuration'],row['evidence']
        assert (f,config_key(c)) not in keys
        keys.add((f,config_key(c)))
        assert e['row_sha256']==context[f]['row_sha256']
        assert fingerprint(e['predictions'])==e['prediction_sha256']
        compare_scores(score(context[f]['y_test'],e['predictions']),e['metrics'])
        assert e['metrics']['n']==len(context[f]['y_test'])
        output_rows.append(dict(fold=f,config=c,**{k:v for k,v in e.items() if k!='predictions'}))
    report=dict(batch_id=batch,manifest=manifest,rows=output_rows,
        validation=dict(status='passed',rows=len(rows),
            prediction_scores_rechecked=sum(r['metrics']['n'] for r in output_rows)))
    conn.execute("UPDATE padjective.paper_tree_tradeoff_batches SET status='complete',report=%s WHERE batch_id=%s",
                 (Jsonb(report),batch))
    conn.commit()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:
        json.dump(report,f,indent=2)
        f.write('\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--batch-id')
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.export_only:
        assert args.batch_id
        with db.get_connection() as conn:
            export_report(conn,args.batch_id,args.output)
    else:
        run(args)


if __name__=='__main__':
    main()
