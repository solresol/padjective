"""Paired inference-only ablation of the frozen ensemble bank; no refitting.

Run from a Padjective checkout, with its package on PYTHONPATH. Read all model
and product data directly from Shopify Postgres. Export aggregate evidence only.
The optional isolated persistence table never updates original experiment rows.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import combinations_with_replacement
import json
from pathlib import Path
import statistics
import subprocess
import time
import uuid

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from padjective import db
from padjective.paper_audit_ensemble_scaling import prefix_count_scores
from padjective.paper_ensemble_scaling_batch import BANK_SEEDS, SIZES
from padjective.paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from padjective.paper_randomised_methods import SNAPSHOT_DIGEST
from padjective.paper_revision_experiments import _load_paper_dataset
from padjective.paper_validate_published import compare_scores, exact_scores
from padjective.paper_validate_randomised import direct_medoid, distance_matrix, fingerprint
from padjective.randomised_linear import consensus_predictions, loss_units, medoid

BATCH = "059d7eb8-14b5-4d5d-857f-134a2098fd89"
P, E = 71, 7
Q = P**E


def synthetic_checks():
    checked = 0
    # All unrestricted minimizers, not just our tie-selected one, are observed.
    for count in range(1, 5):
        for sample in combinations_with_replacement(range(9), count):
            options = np.arange(9)
            costs = distance_matrix(options, sample, 3, 2).sum(axis=1)
            winners = options[costs == costs.min()]
            assert set(winners).issubset(sample)
            assert medoid(sample, 3, 2) == direct_medoid(sample, 3, 2)
            assert medoid(sample, 3, 2)[0] == int(winners.min())
            checked += 1
    assert consensus_predictions([[72, 143, 2]], [1, 2], p=P, precision=E)[0].tolist() == [1]
    assert medoid([72, 143, 2], P, E)[0] == 72
    return checked


def score(y, prediction):
    answer = exact_scores(y.tolist(), prediction.tolist(), P, E)
    compare_scores(prefix_count_scores(y, prediction, P, E), answer)
    return answer


def analyse():
    started = time.monotonic()
    synthetic = synthetic_checks()
    with db.get_connection() as conn:
        conn.read_only = True
        conn.isolation_level = __import__('psycopg').IsolationLevel.REPEATABLE_READ
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT report,validated_at FROM padjective.paper_ensemble_scaling_reports WHERE batch_id=%s", (BATCH,))
            source = cur.fetchone()
        aggregate = source['report']
        assert aggregate['validation']['status'] == 'passed'
        assert len(aggregate['models']) == 1215 and len(aggregate['ensembles']) == 945
        dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema='padjective')
        assert _snapshot_digest(dataset) == SNAPSHOT_DIGEST == aggregate['snapshot_digest']
        assert dataset.prime_base == P and dataset.features.shape == (6693, 2542)
        folds = np.array([r.cv_fold for r in dataset.records])
        targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
        assert len(np.unique(targets)) == 363
        rows, private, checked_members, member_invalid = [], [], 0, 0
        for fold in range(5):
            y = targets[folds == fold]
            candidates = np.unique(targets[folds != fold])
            metadata = [m for m in aggregate['models'] if m['fold'] == fold]
            assert len(metadata) == 243
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute("""SELECT run_id,cv_fold,status,configuration->>'seed' AS seed,
                    evidence->'models'->'coordinate'->'predictions' AS predictions
                    FROM padjective.paper_randomised_method_runs WHERE run_id=ANY(%s::uuid[])""",
                    ([m['run_id'] for m in metadata],))
                saved = {str(r['run_id']): r for r in cur.fetchall()}
                cur.execute("""SELECT members,evidence FROM padjective.paper_ensemble_scaling_results
                    WHERE batch_id=%s AND cv_fold=%s AND roster=0 ORDER BY members""", (BATCH, fold))
                reference = {r['members']:r['evidence'] for r in cur.fetchall()}
            assert set(reference) == set(SIZES)
            ordered, ordered_ids = {}, {}
            for meta in metadata:
                raw = saved[meta['run_id']]
                assert raw['status'] == 'coordinate_optimum' and raw['cv_fold'] == fold
                assert fingerprint(raw['predictions']) == meta['prediction_sha256']
                assert len(raw['predictions']) == len(y)
                ordered[int(raw['seed'])-fold] = raw['predictions']
                ordered_ids[int(raw['seed'])-fold] = meta['run_id']
                checked_members += 1
            assert set(ordered) == set(BANK_SEEDS)
            bank = np.array([ordered[b] for b in BANK_SEEDS], dtype=np.int64).T
            member_invalid += int(np.count_nonzero(~np.isin(bank, candidates)))
            for size in SIZES:
                member = bank[:, :size]
                constrained, _ = consensus_predictions(member, candidates, p=P, precision=E)
                ref = reference[size]
                assert constrained.tolist() == ref['predictions']
                meta = next(r for r in aggregate['ensembles'] if r['fold']==fold and r['roster']==0 and r['members']==size)
                assert fingerprint(constrained.tolist()) == meta['prediction_sha256']
                predicted, optimum_cost = [], []
                for values in member:
                    winner, cost = medoid(values, P, E)
                    assert (winner, cost) == direct_medoid(values, P, E)
                    predicted.append(winner)
                    optimum_cost.append(cost)
                unrestricted = np.array(predicted, dtype=np.int64)
                assert np.all(np.any(member == unrestricted[:, None], axis=1))
                valid = np.isin(unrestricted, candidates)
                assert np.all(constrained[valid] == unrestricted[valid])
                constrained_cost = loss_units(member-constrained[:, None], P, E).sum(axis=1)
                unrestricted_cost = np.array(optimum_cost, dtype=np.int64)
                assert np.all(constrained_cost >= unrestricted_cost)
                baseline = score(y, constrained)
                compare_scores(baseline, meta['metrics'])
                comparison = score(y, unrestricted)
                restricted_loss = loss_units(y-constrained, P, E)
                free_loss = loss_units(y-unrestricted, P, E)
                delta = free_loss-restricted_loss
                changed = constrained != unrestricted
                assert np.array_equal(changed, ~valid)
                root_changed = constrained % P != unrestricted % P
                assert np.all(~root_changed | changed)
                result = dict(fold=fold, members=size, n=len(y), candidate_paths=len(candidates),
                    restricted=baseline, unrestricted=comparison,
                    loss_delta_unrestricted_minus_restricted=comparison['mean_padic_loss']-baseline['mean_padic_loss'],
                    delta_loss_units=int(delta.sum()),
                    changed_predictions=int(changed.sum()), root_changed=int(root_changed.sum()),
                    unrestricted_outside_training_paths=int((~valid).sum()),
                    unrestricted_outside_entire_snapshot_paths=int((~np.isin(unrestricted, np.unique(targets))).sum()),
                    restriction_better=int((delta>0).sum()), restriction_worse=int((delta<0).sum()),
                    changed_same_loss=int((changed & (delta==0)).sum()),
                    restriction_better_units=int(delta[delta>0].sum()),
                    restriction_worse_units=int(delta[delta<0].sum()),
                    root_changed_delta_units=int(delta[root_changed].sum()),
                    same_root_delta_units=int(delta[~root_changed].sum()),
                    consensus_objective_equal=int((constrained_cost==unrestricted_cost).sum()),
                    consensus_objective_strictly_worse_restricted=int((constrained_cost>unrestricted_cost).sum()),
                    restricted_sha256=fingerprint(constrained.tolist()),
                    unrestricted_sha256=fingerprint(predicted),
                    candidate_sha256=fingerprint(candidates.tolist()),
                    member_run_ids=[ordered_ids[b] for b in BANK_SEEDS[:size]])
                assert result['restriction_better']+result['restriction_worse']+result['changed_same_loss']==result['changed_predictions']
                rows.append(result)
                private.append(dict(fold=fold, members=size, unrestricted_predictions=predicted))
            print(json.dumps(dict(event='fold_complete', fold=fold, checked_sizes=list(SIZES))), flush=True)
    summaries = []
    for size in SIZES:
        selected = [r for r in rows if r['members']==size]
        total = sum(r['n'] for r in selected)
        assert total == 6693
        summary = dict(members=size, n=total)
        for kind in ('restricted', 'unrestricted'):
            summary[kind] = {key:statistics.fmean(r[kind][key] for r in selected)
                             for key in ('mean_padic_loss','exact_accuracy','first_digit_accuracy')}
        summary['loss_delta'] = summary['unrestricted']['mean_padic_loss']-summary['restricted']['mean_padic_loss']
        summary['relative_loss_increase'] = summary['loss_delta']/summary['restricted']['mean_padic_loss']
        summary['fold_loss_delta_range'] = [min(r['loss_delta_unrestricted_minus_restricted'] for r in selected),
                                             max(r['loss_delta_unrestricted_minus_restricted'] for r in selected)]
        for key in ('changed_predictions','root_changed','unrestricted_outside_training_paths',
                    'unrestricted_outside_entire_snapshot_paths','restriction_better','restriction_worse',
                    'changed_same_loss','delta_loss_units','restriction_better_units','restriction_worse_units',
                    'root_changed_delta_units','same_root_delta_units','consensus_objective_equal',
                    'consensus_objective_strictly_worse_restricted'):
            summary[key] = sum(r[key] for r in selected)
        summary['changed_fraction'] = summary['changed_predictions']/total
        summary['pooled_loss_delta'] = float(Fraction(summary['delta_loss_units'], total*Q))
        summaries.append(summary)
        print(json.dumps(dict(event='summary', **summary)), flush=True)
    report = dict(analysis='valid_path_restriction_ablation', analysis_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(), batch_id=BATCH,
        snapshot_id=PAPER_SNAPSHOT, snapshot_digest=SNAPSHOT_DIGEST,
        source_report_sha256=fingerprint(aggregate), source_report_validated_at=source['validated_at'].isoformat(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metric='mean p-adic distance; lower is better; equal-weight mean of five fold means',
        design='Paired inference-only ablation; original nested primary roster; no fits, defaults, folds or members changed',
        unrestricted_rule='Smallest-code medoid of default-adjusted member predictions; exact global ultrametric minimum',
        restricted_rule='Smallest-code minimum-total-distance path observed in the fitting fold',
        sizes=list(SIZES), p=P, precision=E, rows=rows, summaries=summaries,
        validation=dict(status='passed', synthetic_cases=synthetic, checked_members=checked_members,
            restricted_consensuses_matched=len(rows), independently_checked_unrestricted_decisions=sum(r['n'] for r in rows),
            held_out_member_outputs=sum(r['n'] for r in rows if r['members']==243)*243,
            member_outputs_outside_training_paths=member_invalid,
            independent_methods='Prefix medoid and exhaustive direct-distance medoid agree on every decision; integer and prefix metrics agree',
            elapsed_seconds=time.monotonic()-started),
        limitations=['Exploratory paired analysis on existing five folds, not an independent test set.',
            'No equivalence threshold was specified; differences reported directly, not a significance/equivalence claim.',
            'Outside training paths does not necessarily mean invalid under the full external Shopify taxonomy.',
            'Primary roster only; all-member size 243 is invariant to roster order.',
            'This is not the distinct procedure of projecting each member first, then taking its medoid.'])
    return report, private


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--persist', action='store_true')
    args=parser.parse_args()
    assert not args.output.exists()
    report, private = analyse()
    if args.persist:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET LOCAL default_tablespace='pg_default'")
                cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_inference_ablations (
                    analysis_id UUID PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    batch_id UUID NOT NULL, report JSONB NOT NULL, private_evidence JSONB NOT NULL
                    ) TABLESPACE pg_default""")
                cur.execute("INSERT INTO padjective.paper_inference_ablations (analysis_id,batch_id,report,private_evidence) VALUES (%s,%s,%s,%s)",
                            (report['analysis_id'],BATCH,Jsonb(report),Jsonb(private)))
            conn.commit()
            with conn.cursor() as cur:
                cur.execute("SELECT report,private_evidence FROM padjective.paper_inference_ablations WHERE analysis_id=%s", (report['analysis_id'],))
                saved=cur.fetchone()
                assert saved == (report,private)
                cur.execute("""SELECT c.relname, COALESCE(t.spcname,dt.spcname) FROM pg_class c
                    JOIN pg_namespace n ON n.oid=c.relnamespace
                    JOIN pg_database d ON d.datname=current_database()
                    JOIN pg_tablespace dt ON dt.oid=d.dattablespace
                    LEFT JOIN pg_tablespace t ON t.oid=c.reltablespace
                    WHERE n.nspname='padjective' AND c.relname IN ('paper_inference_ablations','paper_inference_ablations_pkey')""")
                spaces=cur.fetchall()
                assert len(spaces)==2 and all(space=='pg_default' for _,space in spaces)
        print(json.dumps(dict(event='persisted_and_read_back', analysis_id=report['analysis_id'], tablespaces=spaces)), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(dict(event='complete', output=str(args.output), validation=report['validation'])), flush=True)


if __name__=='__main__':
    main()
