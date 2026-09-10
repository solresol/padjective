"""Immutable, resumable live-catalogue comparison; no production model writes."""
from __future__ import annotations

import argparse
from collections import Counter
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
from scipy import sparse

from . import data_access, db, live_exact, tagbattle
from .paper_ensemble_scaling_batch import BANK_SEEDS
from .paper_published_methods import PAPER_SNAPSHOT
from .product_hash import canonicalize_product_url, hash_product_url
from .subspace_voting import corrected_test, feature_mask

P, E = 83, 8
FRACTIONS = (.75, 1.)
PROTOCOL = Path('docs/live-subspace-protocol.md')
MAX_ROWS = 100000
CONTEXT = {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def emit(event, **data):
    print(json.dumps(dict(event=event, **data), default=str), flush=True)


def encode(path):
    digits = [int(v) for v in path.split('.')]
    if not digits or len(digits) > E or any(not 0 < v < P for v in digits):
        raise ValueError('Live taxonomy outgrew the frozen p=83, E=8 encoding')
    return sum(v*P**i for i, v in enumerate(digits))


def store_fold(domain):
    return int(hashlib.sha256(('live-subspace-v1:'+domain.strip().lower()).encode()).hexdigest(), 16) % 5


def ensure_storage(conn):
    conn.execute('SELECT pg_advisory_xact_lock(71357913)')
    conn.execute("SET LOCAL default_tablespace='pg_default'")
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_cohorts (
        cohort_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        content_sha256 TEXT NOT NULL UNIQUE, manifest JSONB NOT NULL
    ) TABLESPACE pg_default''')
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_products (
        cohort_id UUID NOT NULL, product_key TEXT NOT NULL, evidence JSONB NOT NULL,
        PRIMARY KEY(cohort_id,product_key)
    ) TABLESPACE pg_default''')
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_observations (
        observation_id UUID PRIMARY KEY, observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        cohort_id UUID NOT NULL, evidence JSONB NOT NULL
    ) TABLESPACE pg_default''')
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_batches (
        batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        cohort_id UUID NOT NULL, status TEXT NOT NULL, manifest JSONB NOT NULL
    ) TABLESPACE pg_default''')
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_models (
        batch_id UUID NOT NULL, fold INTEGER NOT NULL, fraction DOUBLE PRECISION NOT NULL,
        member INTEGER NOT NULL, finished_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        evidence JSONB NOT NULL, PRIMARY KEY(batch_id,fold,fraction,member)
    ) TABLESPACE pg_default''')
    conn.execute('''CREATE TABLE IF NOT EXISTS padjective.live_subspace_reports (
        batch_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        report JSONB NOT NULL, predictions JSONB NOT NULL
    ) TABLESPACE pg_default''')
    conn.commit()


SOURCE_SQL = '''SELECT p.id,p.myshopify_domain,p.product_handle,p.product_url,
    p.created_at,p.updated_at,p.product_title IS NOT NULL AS has_title,
    pd.product_handle IS NOT NULL AS has_details,
    pd.product_detail->'product'->>'tags' AS tags,
    pt.taxonomy_id,pt.taxonomy_source,pt.classified_at,r.numeric_path
    FROM cantbuymelove.product p
    LEFT JOIN public.product_details pd USING(myshopify_domain,run_name,product_handle)
    LEFT JOIN cantbuymelove.product_taxonomy pt
      ON pt.product_id=p.id AND pt.taxonomy_source='gold_llm'
    LEFT JOIN padjective.taxonomy_path_reconciliation r ON r.taxonomy_id=pt.taxonomy_id
    ORDER BY p.updated_at,p.created_at,p.id'''


def prepare_rows(source, paper_keys):
    """Latest entity/URL deduplication; all exclusions remain in the ledger."""
    counts = Counter()
    latest = {}
    overlaps = {}
    entity_labels = {}
    for row in source:
        counts['source_rows'] += 1
        if counts['source_rows'] > MAX_ROWS:
            raise ValueError('Live source exceeded bounded-fitting ceiling; no truncation allowed')
        domain = (row['myshopify_domain'] or '').strip().lower()
        handle = (row['product_handle'] or '').strip()
        if not domain or not handle:
            counts['missing_entity_key'] += 1
            continue
        key = digest([domain, handle])
        canonical = canonicalize_product_url(row['product_url'], myshopify_domain=domain, product_handle=handle)
        url_hash = hash_product_url(canonical) if canonical else None
        overlaps[key] = overlaps.get(key, False) or url_hash in paper_keys
        entity_labels.setdefault(key, set()).add(row['taxonomy_id'])
        if key in latest:
            counts['duplicate_entity_rows'] += 1
        latest[key] = (row, domain, url_hash)
    counts['entities_with_conflicting_historical_labels'] = sum(len(v) > 1 for v in entity_labels.values())
    products = {}
    url_owner = {}
    for key, (row, domain, url_hash) in sorted(latest.items(), key=lambda kv: (
            kv[1][0]['updated_at'], kv[1][0]['created_at'], kv[1][0]['id'])):
        if not row['has_details']:
            counts['missing_details'] += 1
            continue
        if row['taxonomy_source'] != 'gold_llm' or not row['taxonomy_id']:
            counts['missing_reference_label'] += 1
            continue
        if not row['numeric_path']:
            counts['unresolved_reference_path'] += 1
            continue
        if not url_hash:
            counts['missing_canonical_url'] += 1
            continue
        target = encode(row['numeric_path'])
        tags = sorted(tagbattle.filter_nested_tags(data_access.parse_tags(row['tags'])))
        overlap = overlaps[key]
        if url_hash in url_owner:
            old_key = url_owner[url_hash]
            old = products.pop(old_key)
            overlap = overlap or old['paper_overlap']
            counts['duplicate_canonical_urls'] += 1
            counts['duplicate_url_label_conflicts'] += old['taxonomy_id'] != row['taxonomy_id']
        url_owner[url_hash] = key
        evidence = dict(product_key=key, store_hash=digest(domain), fold=store_fold(domain),
            paper_overlap=overlap, taxonomy_id=row['taxonomy_id'], path=row['numeric_path'],
            target=target, tags=tags, feature_sha256=digest(tags),
            source_product_id=row['id'], source_updated_at=str(row['updated_at']),
            classified_at=str(row['classified_at']), label_source='gold_llm')
        evidence['content_sha256'] = digest({k: evidence[k] for k in
            ('product_key', 'store_hash', 'fold', 'paper_overlap', 'taxonomy_id', 'path', 'tags')})
        products[key] = evidence
    rows = [products[k] for k in sorted(products)]
    counts.update(dict(retained_products=len(rows), stores=len({r['store_hash'] for r in rows}),
        paper_overlap=sum(r['paper_overlap'] for r in rows),
        outside_paper=sum(not r['paper_overlap'] for r in rows),
        empty_tags=sum(not r['tags'] for r in rows),
        taxonomies=len({r['taxonomy_id'] for r in rows})))
    return rows, dict(counts)


def capture():
    with db.get_connection() as conn:
        ensure_storage(conn)
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ')
        conn.execute("SET LOCAL statement_timeout='180s'")
        conn.execute("SET LOCAL max_parallel_workers_per_gather=1")
        n = conn.execute('SELECT count(*) FROM cantbuymelove.product').fetchone()[0]
        if n > MAX_ROWS:
            raise ValueError(f'{n} source rows exceed the {MAX_ROWS} fitting ceiling')
        paper = {r[0] for r in conn.execute('''SELECT product_id_hash FROM
            padjective.product_taxonomy_bench_products WHERE snapshot_id=%s''', (PAPER_SNAPSHOT,))}
        if len(paper) != 6693:
            raise ValueError('Unexpected immutable paper membership')
        with conn.cursor(name='live_subspace_source', row_factory=dict_row) as cur:
            cur.itersize = 1000
            cur.execute(SOURCE_SQL)
            rows, coverage = prepare_rows(cur, paper)
        if coverage['source_rows'] != n or not rows:
            raise ValueError('Source join changed grain or produced empty cohort')
        content = digest([r['content_sha256'] for r in rows])
        prior = conn.execute('''SELECT cohort_id FROM padjective.live_subspace_cohorts
            WHERE content_sha256=%s''', (content,)).fetchone()
        cohort = str(prior[0]) if prior else str(uuid.uuid4())
        manifest = dict(coverage=coverage, source_query=SOURCE_SQL,
            captured_at=str(conn.execute('SELECT transaction_timestamp()').fetchone()[0]),
            prime=P, precision=E, paper_snapshot=PAPER_SNAPSHOT,
            content_sha256=content, fold_sizes=[sum(r['fold'] == f for r in rows) for f in range(5)])
        if not all(manifest['fold_sizes']):
            raise ValueError('Every store-held-out fold must be nonempty')
        if not prior:
            conn.execute('''INSERT INTO padjective.live_subspace_cohorts
                (cohort_id,content_sha256,manifest) VALUES (%s,%s,%s)''', (cohort, content, Jsonb(manifest)))
            with conn.cursor() as cur:
                cur.executemany('''INSERT INTO padjective.live_subspace_products
                    (cohort_id,product_key,evidence) VALUES (%s,%s,%s)''',
                    [(cohort, r['product_key'], Jsonb(r)) for r in rows])
        conn.execute('''INSERT INTO padjective.live_subspace_observations
            (observation_id,cohort_id,evidence) VALUES (%s,%s,%s)''',
            (str(uuid.uuid4()), cohort, Jsonb(manifest)))
    emit('live_cohort_captured', cohort_id=cohort, reused=bool(prior), **manifest)
    return cohort


def load_rows(conn, cohort):
    saved = conn.execute('''SELECT content_sha256 FROM padjective.live_subspace_cohorts
        WHERE cohort_id=%s''', (cohort,)).fetchone()
    if not saved:
        raise ValueError('Unknown cohort')
    rows = [r[0] for r in conn.execute('''SELECT evidence FROM padjective.live_subspace_products
        WHERE cohort_id=%s ORDER BY product_key''', (cohort,))]
    if digest([r['content_sha256'] for r in rows]) != saved[0]:
        raise ValueError('Cohort digest mismatch')
    for row in rows:
        actual = digest({k: row[k] for k in
            ('product_key', 'store_hash', 'fold', 'paper_overlap', 'taxonomy_id', 'path', 'tags')})
        if row['content_sha256'] != actual or encode(row['path']) != row['target']:
            raise ValueError('Corrupt cohort row')
    return rows


def context(rows, fold):
    train = [r for r in rows if r['fold'] != fold]
    test = [r for r in rows if r['fold'] == fold]
    counts = Counter(tag for row in train for tag in row['tags'])
    vocabulary = sorted(tag for tag, count in counts.items() if count >= 5)
    if not vocabulary or not train or not test:
        raise ValueError('Nonempty vocabulary and folds required')
    index = {v: j for j, v in enumerate(vocabulary)}

    def matrix(records):
        rr, cc = [], []
        for i, row in enumerate(records):
            for tag in row['tags']:
                if tag in index:
                    rr.append(i)
                    cc.append(index[tag])
        return sparse.csr_matrix((np.ones(len(rr), dtype=np.int64), (rr, cc)),
                                 shape=(len(records), len(vocabulary)))

    x_train, x_test = matrix(train), matrix(test)
    return dict(train=train, test=test, x_train=x_train, x_test=x_test,
                y_train=np.array([r['target'] for r in train]),
                y_test=np.array([r['target'] for r in test]), vocabulary=vocabulary)


def initialise(cohort):
    global CONTEXT
    with db.get_connection() as conn:
        rows = load_rows(conn, cohort)
    CONTEXT = {f: context(rows, f) for f in range(5)}


def fit_one(batch, fold, fraction, member):
    started = time.monotonic()
    c = CONTEXT[fold]
    seed = BANK_SEEDS[member]
    mask = feature_mask(np.arange(len(c['vocabulary'])), fraction, fold, seed)
    x_train, x_test = c['x_train'][:, mask], c['x_test'][:, mask]
    fitted = live_exact.fit(x_train, c['y_train'], p=P, precision=E, seed=seed+fold)
    if fitted['status'] != 'coordinate_optimum':
        raise RuntimeError(f'Incomplete member {fold}/{fraction}/{member}: {fitted["status"]}')
    weights = fitted['coefficients']
    train_raw = live_exact.certificate(x_train, c['y_train'], weights, P, E)
    if fitted['history'][-1]['loss_units'] != live_exact.total(live_exact.units(c['y_train']-train_raw, P, E)):
        raise AssertionError('Saved fitting loss disagrees with reconstructed model')
    raw = live_exact.direct_predict(x_test, weights, P**E)
    if not np.array_equal(raw, live_exact.predict(x_test, weights, P, E)):
        raise AssertionError('Independent held-out predictions disagree')
    default = live_exact.default_value(c['y_train'], train_raw, P, E)
    pred = np.where(raw == 0, default, raw)
    evidence = dict(fit=fitted, mask=mask.tolist(), mask_sha256=digest(mask.tolist()),
        coefficient_sha256=digest(weights), default=int(default), seed_base=seed,
        fit_seed=seed+fold, vocabulary_sha256=digest(c['vocabulary']),
        test_keys_sha256=digest([r['product_key'] for r in c['test']]),
        predictions=pred.tolist(), prediction_sha256=digest(pred.tolist()),
        certified_coordinates=len(mask), eligible_features=len(c['vocabulary']),
        selected_features=len(mask), nonzero_coefficients=int(np.count_nonzero(weights)),
        job_seconds=time.monotonic()-started)
    with db.get_connection() as conn:
        conn.execute('''INSERT INTO padjective.live_subspace_models
            (batch_id,fold,fraction,member,evidence) VALUES (%s,%s,%s,%s,%s)''',
            (batch, fold, fraction, member, Jsonb(evidence)))
    return dict(fold=fold, fraction=fraction, member=member, seconds=evidence['job_seconds'])


def specification(members):
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise ValueError('Clean tracked source required')
    return dict(members=members, prime=P, precision=E, seed_bases=list(BANK_SEEDS[:members]),
        fractions=list(FRACTIONS), store_folds=5, max_sweeps=100, fitting_seconds=600,
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path('padjective/live_subspace.py'), Path('padjective/live_exact.py'), Path('padjective/subspace_voting.py')]},
        python=platform.python_version(), numpy=np.__version__)


def run(cohort, members, workers, batch_id=None):
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise ValueError(f'Set {name}=1 before importing numerical libraries')
    spec = specification(members)
    with db.get_connection() as conn:
        ensure_storage(conn)
        # Session lock is held by this connection throughout fitting/evaluation.
        if not conn.execute('SELECT pg_try_advisory_lock(71357914)').fetchone()[0]:
            raise RuntimeError('Another live-subspace run is active; do not duplicate it')
        rows = load_rows(conn, cohort)
        if batch_id:
            prior = conn.execute('''SELECT cohort_id,manifest,status FROM padjective.live_subspace_batches
                WHERE batch_id=%s''', (batch_id,)).fetchone()
            if not prior or str(prior[0]) != cohort or prior[1] != spec:
                raise ValueError('Resume needs identical source, cohort and protocol')
            if prior[2] == 'validated':
                emit('live_batch_already_validated', batch_id=batch_id)
                return batch_id
        else:
            # Reuse identical frozen data/design; do not silently revise older code runs.
            prior = conn.execute('''SELECT batch_id,status FROM padjective.live_subspace_batches
                WHERE cohort_id=%s AND manifest=%s ORDER BY created_at DESC LIMIT 1''',
                (cohort, Jsonb(spec))).fetchone()
            if prior:
                batch_id = str(prior[0])
                if prior[1] == 'validated':
                    emit('live_batch_already_validated', batch_id=batch_id)
                    return batch_id
            else:
                batch_id = str(uuid.uuid4())
                conn.execute('''INSERT INTO padjective.live_subspace_batches
                    (batch_id,cohort_id,status,manifest) VALUES (%s,%s,'fitting',%s)''',
                    (batch_id, cohort, Jsonb(spec)))
        saved = {(r[0], r[1], r[2]) for r in conn.execute('''SELECT fold,fraction,member
            FROM padjective.live_subspace_models WHERE batch_id=%s''', (batch_id,))}
        jobs = [(f, fraction, m) for m in range(members) for f in range(5) for fraction in FRACTIONS]
        if not saved <= set(jobs):
            raise ValueError('Unexpected saved model key')
        conn.execute("UPDATE padjective.live_subspace_batches SET status='fitting' WHERE batch_id=%s", (batch_id,))
        conn.commit()
        emit('live_batch_started', batch_id=batch_id, cohort_id=cohort, saved=len(saved),
             expected=len(jobs), workers=workers, members=members)
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialise, initargs=(cohort,)) as pool:
                pending = [pool.submit(fit_one, batch_id, *job) for job in jobs if job not in saved]
                completed = len(saved)
                try:
                    for future in as_completed(pending):
                        result = future.result()
                        completed += 1
                        if completed % 10 == 0 or completed == len(jobs):
                            emit('live_fit_progress', batch_id=batch_id, completed=completed, expected=len(jobs), **result)
                except BaseException:
                    for future in pending:
                        future.cancel()
                    raise
            conn.execute("UPDATE padjective.live_subspace_batches SET status='evaluating' WHERE batch_id=%s", (batch_id,))
            conn.commit()
            evaluate(conn, batch_id, rows, spec)
        except BaseException:
            conn.rollback()
            conn.execute("UPDATE padjective.live_subspace_batches SET status='failed' WHERE batch_id=%s", (batch_id,))
            conn.commit()
            raise
    return batch_id


def compare_trend(previous_rows, current_rows, previous_predictions, current_predictions):
    old = {r['product_key']: r for r in previous_rows}
    new = {r['product_key']: r for r in current_rows}
    common = old.keys() & new.keys()
    same = sorted(k for k in common if old[k]['content_sha256'] == new[k]['content_sha256'])
    result = dict(unchanged_common=len(same), changed_common=len(common)-len(same),
        new_products=len(new.keys()-old.keys()), removed_products=len(old.keys()-new.keys()),
        interpretation='Rolling refits: changes combine retraining and cohort composition, not frozen-model drift')
    if same:
        for fraction in FRACTIONS:
            key = str(fraction)
            y = [new[k]['target'] for k in same]
            result[key] = dict(previous=live_exact.scores(y, [previous_predictions[k][key] for k in same], P, E),
                current=live_exact.scores(y, [current_predictions[k][key] for k in same], P, E))
    return result


def evaluate(conn, batch, rows, spec):
    reports, predictions, memberships = [], {}, {}
    for fold in range(5):
        c = context(rows, fold)
        candidates = np.unique(c['y_train'])
        feature_sets = {r['feature_sha256'] for r in c['train']}
        groups = dict(all=np.ones(len(c['test']), dtype=bool),
            paper_overlap=np.array([r['paper_overlap'] for r in c['test']]),
            outside_paper=np.array([not r['paper_overlap'] for r in c['test']]),
            zero_features=c['x_test'].getnnz(axis=1) == 0,
            has_features=c['x_test'].getnnz(axis=1) > 0,
            unseen_target=~np.isin(c['y_test'], candidates),
            unseen_feature_set=np.array([r['feature_sha256'] not in feature_sets for r in c['test']]))
        for i, row in enumerate(c['test']):
            memberships[row['product_key']] = {name: bool(keep[i]) for name, keep in groups.items()}
        for fraction in FRACTIONS:
            models = list(conn.execute('''SELECT member,evidence FROM padjective.live_subspace_models
                WHERE batch_id=%s AND fold=%s AND fraction=%s ORDER BY member''', (batch, fold, fraction)))
            if [r[0] for r in models] != list(range(spec['members'])):
                raise ValueError('Incomplete bank; no survivor ensemble')
            bank = []
            for member, evidence in models:
                mask = feature_mask(np.arange(len(c['vocabulary'])), fraction, fold, BANK_SEEDS[member])
                weights = evidence['fit']['coefficients']
                if (evidence['fit']['status'] != 'coordinate_optimum'
                        or evidence['fit_seed'] != BANK_SEEDS[member]+fold
                        or evidence['seed_base'] != BANK_SEEDS[member]
                        or evidence['certified_coordinates'] != len(mask)
                        or evidence['mask'] != mask.tolist()
                        or evidence['mask_sha256'] != digest(mask.tolist())
                        or evidence['vocabulary_sha256'] != digest(c['vocabulary'])
                        or evidence['coefficient_sha256'] != digest(weights)
                        or evidence['test_keys_sha256'] != digest([r['product_key'] for r in c['test']])):
                    raise ValueError('Saved model provenance mismatch')
                training_raw = live_exact.direct_predict(c['x_train'][:, mask], weights, P**E)
                if live_exact.default_value(c['y_train'], training_raw, P, E) != evidence['default']:
                    raise ValueError('Fitting-only default mismatch')
                raw = live_exact.direct_predict(c['x_test'][:, mask], weights, P**E)
                pred = np.where(raw == 0, evidence['default'], raw)
                if pred.tolist() != evidence['predictions'] or digest(pred.tolist()) != evidence['prediction_sha256']:
                    raise ValueError('Reconstructed predictions disagree')
                bank.append(pred)
            bank = np.array(bank).T
            pred = live_exact.consensus(bank, candidates, P, E)
            # Independent direct distances for deterministic coverage across this fold.
            for i in np.unique(np.linspace(0, len(pred)-1, 8, dtype=int)):
                costs = [sum(int(v) for v in live_exact.units(bank[i]-candidate, P, E)) for candidate in candidates]
                if pred[i] != candidates[np.argmin(costs)]:
                    raise AssertionError('Independent consensus check failed')
            for row, value in zip(c['test'], pred):
                predictions.setdefault(row['product_key'], {})[str(fraction)] = int(value)
            coverage = dict(eligible_features=len(c['vocabulary']),
                candidate_paths=len(candidates), zero_feature_products=int(np.sum(c['x_test'].getnnz(axis=1) == 0)),
                unseen_target_products=int(np.sum(~np.isin(c['y_test'], candidates))),
                feature_sets_seen_in_training=sum(r['feature_sha256'] in feature_sets for r in c['test']),
                total_tag_occurrences=sum(len(r['tags']) for r in c['test']),
                eligible_tag_occurrences=int(c['x_test'].nnz),
                union_selected_features=len({j for _, m in models for j in m['mask']}),
                stored_nonzero_coefficients=sum(m[1]['nonzero_coefficients'] for m in models),
                component_job_seconds=sum(m[1]['job_seconds'] for m in models))
            for name, keep in groups.items():
                if np.any(keep):
                    reports.append(dict(fold=fold, fraction=fraction, cohort=name,
                        metrics=live_exact.scores(c['y_test'][keep], pred[keep], P, E), fold_coverage=coverage))
        emit('live_fold_evaluated', batch_id=batch, fold=fold, n=len(c['test']))
    comparisons = {}
    for group in ('all', 'paper_overlap', 'outside_paper'):
        selected = {(r['fold'], r['fraction']): r['metrics'] for r in reports if r['cohort'] == group}
        if len(selected) == 10:
            delta = [selected[f, .75]['mean_padic_loss']-selected[f, 1.]['mean_padic_loss'] for f in range(5)]
            # Training overlap is the whole fitting cohort, not only the subgroup.
            sizes = [sum(r['fold'] == f for r in rows) for f in range(5)]
            comparisons[group] = corrected_test(delta, sizes)
            comparisons[group]['mean_fold_loss'] = {str(fr): float(np.mean([
                selected[f, fr]['mean_padic_loss'] for f in range(5)])) for fr in FRACTIONS}
    pooled = {}
    for group in groups:
        selected = [r for r in rows if memberships[r['product_key']][group]]
        if selected:
            pooled[group] = {str(fr): live_exact.scores([r['target'] for r in selected],
                [predictions[r['product_key']][str(fr)] for r in selected], P, E) for fr in FRACTIONS}
    previous = conn.execute('''SELECT b.cohort_id,r.predictions FROM padjective.live_subspace_batches b
        JOIN padjective.live_subspace_reports r USING(batch_id)
        WHERE b.batch_id<>%s AND b.status='validated' AND (b.manifest->>'members')::int=%s
        ORDER BY b.created_at DESC LIMIT 1''', (batch, spec['members'])).fetchone()
    trend = compare_trend(load_rows(conn, previous[0]), rows, previous[1], predictions) if previous else None
    report = dict(batch_id=batch, manifest=spec, fold_results=reports, comparisons=comparisons,
        pooled=pooled, trend=trend, predictions_sha256=digest(predictions),
        validated_products=len(predictions), independent_consensus_checks=80,
        status='engineering_smoke_only' if spec['members'] != 243 else 'validated_rolling_cv',
        caveat='Gold LLM reference agreement; repeated observational CV, not fresh independent test evidence')
    conn.execute('''INSERT INTO padjective.live_subspace_reports(batch_id,report,predictions)
        VALUES (%s,%s,%s)''', (batch, Jsonb(report), Jsonb(predictions)))
    check = conn.execute('''SELECT report,predictions FROM padjective.live_subspace_reports
        WHERE batch_id=%s''', (batch,)).fetchone()
    if check[0] != json.loads(json.dumps(report)) or check[1] != predictions:
        raise ValueError('Postgres report readback mismatch')
    conn.execute("UPDATE padjective.live_subspace_batches SET status='validated' WHERE batch_id=%s", (batch,))
    conn.commit()
    emit('live_batch_validated', batch_id=batch, products=len(predictions), members=spec['members'],
         comparisons=comparisons, pooled=pooled, trend=trend)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('capture', 'run', 'cycle', 'status'))
    parser.add_argument('--cohort')
    parser.add_argument('--batch-id')
    parser.add_argument('--members', type=int, choices=(3, 243), default=243)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    if args.command == 'status':
        with db.get_connection() as conn:
            for row in conn.execute('''SELECT b.batch_id,b.cohort_id,b.status,b.created_at,
                    b.manifest->>'members',count(m.member) FROM padjective.live_subspace_batches b
                    LEFT JOIN padjective.live_subspace_models m USING(batch_id)
                    GROUP BY b.batch_id ORDER BY b.created_at DESC LIMIT 5'''):
                emit('live_batch_status', batch_id=row[0], cohort_id=row[1], status=row[2],
                     created_at=row[3], members=row[4], completed_models=row[5])
        return
    cohort = capture() if args.command in ('capture', 'cycle') else args.cohort
    if args.command != 'capture':
        if not cohort:
            parser.error('--cohort required for run')
        run(cohort, args.members, args.workers, args.batch_id)


if __name__ == '__main__':
    main()
