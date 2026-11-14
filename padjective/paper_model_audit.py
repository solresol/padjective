"""Audit saved paper models and September follow-up directly in Shopify Postgres.

No models are refitted. Product rows and coefficients stay in the remote process;
only aggregates are emitted and appended to padjective.paper_manuscript_audits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime, timezone
from statistics import mean, variance
import uuid

import numpy as np
from psycopg.types.json import Jsonb
from scipy.stats import t

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset

ENSEMBLE_BATCH = '059d7eb8-14b5-4d5d-857f-134a2098fd89'
SUBSPACE_BATCH = '773bf2ab-d2b1-4c32-bfde-15d501782fa0'
TREE_BATCH = '95b8a6bf-28b9-4a17-9ee6-58759d149ff5'


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def member_check(features, coefficients, default, saved_raw, saved_predictions):
    """Compare integer and quotient definitions without revealing row values."""
    q = 71**7
    weights = np.asarray(coefficients, dtype=np.int64)
    totals = np.asarray(features @ weights).reshape(-1)
    residues = totals % q
    predictions = np.where(residues == 0, default, residues)
    assert residues.tolist() == saved_raw, 'Stored raw scores differ from quotient calculation'
    assert predictions.tolist() == saved_predictions, 'Stored default-adjusted predictions differ'
    return dict(rows=len(totals), wrapped_integer_sums=int(np.count_nonzero(totals != residues)),
        modular_zero_with_nonzero_integer_sum=int(np.count_nonzero((residues == 0) & (totals != 0))),
        integer_and_quotient_reported_outputs_differ=int(np.count_nonzero(
            np.where(totals == 0, default, totals) != predictions)))


def audit_members(conn):
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema='padjective')
    assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST
    folds = np.asarray([r.cv_fold for r in dataset.records])
    context = {f: {part: dataset.features[folds == f if part == 'held_out' else folds != f]
                  for part in ('training', 'held_out')} for f in range(5)}
    bank = conn.execute('SELECT report FROM padjective.paper_ensemble_scaling_reports WHERE batch_id=%s',
                        (ENSEMBLE_BATCH,)).fetchone()[0]
    assert bank['validation']['status'] == 'passed' and len(bank['models']) == 1215
    counts = {str(fr): {part: Counter() for part in ('training', 'held_out')} for fr in (.75, 1.)}
    checked = Counter()
    for meta in bank['models']:
        e = conn.execute("SELECT evidence->'models'->'coordinate' FROM padjective.paper_randomised_method_runs WHERE run_id=%s",
                         (meta['run_id'],)).fetchone()[0]
        assert fingerprint(e['coefficients']) == meta['coefficient_sha256']
        assert fingerprint(e['predictions']) == meta['prediction_sha256']
        for part, raw, pred in (('training', 'training_raw_predictions', 'training_predictions'),
                               ('held_out', 'held_out_raw_predictions', 'predictions')):
            counts['1.0'][part].update(member_check(context[meta['fold']][part], e['coefficients'],
                e['default'], e[raw], e[pred]))
        checked['1.0'] += 1
        if checked['1.0'] % 300 == 0:
            print(f"Checked {checked['1.0']} all-feature members", file=sys.stderr, flush=True)
    with conn.cursor(name='paper_mask_model_audit') as cur:
        cur.execute('SELECT cv_fold,status,evidence FROM padjective.paper_subspace_runs WHERE batch_id=%s AND fraction=%s ORDER BY cv_fold,member_index',
                    (SUBSPACE_BATCH, .75))
        for fold, status, e in cur:
            assert status == 'coordinate_optimum' and fingerprint(e['coefficients']) == e['coefficient_sha256']
            assert fingerprint(e['predictions']) == e['prediction_sha256']
            for part, raw, pred in (('training', 'training_raw_predictions', 'training_predictions'),
                                   ('held_out', 'held_out_raw_predictions', 'predictions')):
                counts['0.75'][part].update(member_check(context[fold][part][:, e['mask']], e['coefficients'],
                    e['default'], e[raw], e[pred]))
            checked['0.75'] += 1
            if checked['0.75'] % 300 == 0:
                print(f"Checked {checked['0.75']} masked members", file=sys.stderr, flush=True)
    assert dict(checked) == {'1.0': 1215, '0.75': 1215}
    return dict(status='passed', snapshot_digest=SNAPSHOT_DIGEST, checked_models=dict(checked),
                counts={fr: {part: dict(c) for part, c in parts.items()} for fr, parts in counts.items()},
                stored_quotient_predictions_matched=True,
                note='Integer sums are a counterfactual interpretation of the manuscript, not another fitted experiment.')


def audit_live(conn):
    output = []
    source = conn.execute('''SELECT b.batch_id,b.cohort_id,b.status,c.manifest,r.report,r.predictions
        FROM padjective.live_subspace_batches b JOIN padjective.live_subspace_cohorts c USING(cohort_id)
        JOIN padjective.live_subspace_reports r USING(batch_id)
        WHERE b.status='validated' ORDER BY b.created_at''').fetchall()
    for batch, cohort_id, status, cohort, report, predictions in source:
        if report['manifest']['members'] != 243:
            continue
        assert report['status'] == 'validated_rolling_cv'
        assert fingerprint(predictions) == report['predictions_sha256']
        grid = conn.execute('''SELECT fold,fraction,count(*),count(*) FILTER (
            WHERE evidence->'fit'->>'status'='coordinate_optimum')
            FROM padjective.live_subspace_models WHERE batch_id=%s GROUP BY fold,fraction''', (batch,)).fetchall()
        assert len(grid) == 10 and all(n == opt == 243 for _, _, n, opt in grid)
        rows = [r[0] for r in conn.execute('SELECT evidence FROM padjective.live_subspace_products WHERE cohort_id=%s',
                                          (cohort_id,))]
        assert len(rows) == len(predictions) == report['validated_products']
        totals = {(f, fr, group): Counter() for f in range(5) for fr in ('0.75', '1.0')
                  for group in ('all', 'outside_paper')}
        p = report['manifest']['prime']; scale = p**(report['manifest']['precision']-1)
        for row in rows:
            assert row['label_source'] == 'gold_llm'
            for fr in ('0.75', '1.0'):
                delta = abs(int(row['target']) - int(predictions[row['product_key']][fr]))
                units = scale if delta else 0
                while delta and delta % p == 0:
                    delta //= p; units //= p
                values = dict(n=1, loss_units=units)
                totals[row['fold'], fr, 'all'].update(values)
                if not row['paper_overlap']:
                    totals[row['fold'], fr, 'outside_paper'].update(values)
        comparisons = {}
        for group in ('all', 'outside_paper'):
            loss = {fr: [totals[f, fr, group]['loss_units'] / (scale*totals[f, fr, group]['n'])
                         for f in range(5)] for fr in ('0.75', '1.0')}
            differences = [a-b for a, b in zip(loss['0.75'], loss['1.0'])]
            ratio = mean(n/(len(rows)-n) for n in cohort['fold_sizes'])
            se = ((.2+ratio)*variance(differences))**.5
            difference = mean(differences)
            ci = [difference-t.ppf(.975,4)*se, difference+t.ppf(.975,4)*se]
            pv = float(2*t.sf(abs(difference/se), 4)) if se else float(difference == 0)
            old = report['comparisons'][group]
            assert abs(difference-old['mean_difference']) < 1e-12
            assert abs(pv-old['p_value']) < 1e-10
            assert np.allclose(ci, old['ci95'], atol=1e-12, rtol=0)
            comparisons[group] = dict(mean_loss={fr: mean(v) for fr,v in loss.items()},
                fold_loss=loss, difference=difference, ci95=ci, p_value=pv)
        output.append(dict(batch_id=str(batch), captured_at=cohort['captured_at'], products=len(rows),
            outside_paper=cohort['coverage']['outside_paper'], prime=p, precision=report['manifest']['precision'],
            fold_sizes=cohort['fold_sizes'], coverage=cohort['coverage'],
            source_commit=report['manifest']['source_commit'],
            checked_models=sum(g[2] for g in grid), comparisons=comparisons))
    assert output
    # Each capture is a repeated look at strongly overlapping data, not a new independent trial.
    for group in ('all', 'outside_paper'):
        order = sorted(range(len(output)), key=lambda i: output[i]['comparisons'][group]['p_value'])
        previous = 0.
        for rank, index in enumerate(order):
            c = output[index]['comparisons'][group]
            previous = min(1., max(previous, (len(output)-rank)*c['p_value']))
            c['holm_over_captures'] = previous
    return dict(status='passed', runs=output, metrics_recomputed_from_private_saved_predictions=True,
                note='Store-grouped refits; overlapping captures and paper identities prevent independent confirmation.')


def main():
    with db.get_connection() as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY')
        report = dict(schema_version=1, checked_at=datetime.now(timezone.utc).isoformat(),
                      source_sha256=globals().get('_source_sha256') or
                      hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      members=audit_members(conn), live=audit_live(conn))
        tree = conn.execute('SELECT report FROM padjective.paper_tree_tradeoff_batches WHERE batch_id=%s',
                            (TREE_BATCH,)).fetchone()[0]
        report['tree'] = dict(batch_id=TREE_BATCH, validation=tree['validation'],
                             manifest=tree['manifest'], rows=tree['rows'])
    audit_id = str(uuid.uuid4())
    report['audit_id'] = audit_id
    with db.get_connection() as conn:
        conn.execute("SET LOCAL default_tablespace='pg_default'")
        conn.execute('''CREATE TABLE IF NOT EXISTS padjective.paper_manuscript_audits (
            audit_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), report JSONB NOT NULL
            ) TABLESPACE pg_default''')
        conn.execute('INSERT INTO padjective.paper_manuscript_audits(audit_id,report) VALUES(%s,%s)',
                     (audit_id, Jsonb(report)))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
