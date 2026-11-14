"""Validate the complete subspace bank, evaluate all fixed rules, and analyse."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import db
from .paper_audit_ensemble_scaling import prefix_count_scores
from .paper_ensemble_scaling_batch import BANK_SEEDS, SIZES, fitting_fingerprints
from .paper_subspace_analysis import analyse
from .paper_subspace_experiment import BASE_ENSEMBLE_BATCH, PROTOCOL, emit, ensure_storage, load_context
from .paper_validate_published import compare_scores, exact_scores
from .paper_validate_randomised import direct_linear, direct_medoid, fingerprint
from .randomised_linear import consensus_predictions
from .subspace_voting import FRACTIONS, RULES, aggregate_rosters, feature_mask, path_digits, project_codes


def reference_survivor(voters):
    """Separate tuple-tree implementation, including grouped continue votes."""
    paths=[path_digits(v) for v in voters]
    def mode(values):
        counts=Counter(values)
        return min(counts,key=lambda v:(-counts[v],v))
    prefix=(mode(v[0] for v in paths),)
    paths=[v for v in paths if v[:1]==prefix]
    while True:
        stops=sum(v==prefix for v in paths)
        if stops>=len(paths)-stops:
            return sum(d*71**i for i,d in enumerate(prefix))
        paths=[v for v in paths if len(v)>len(prefix)]
        prefix=prefix+(mode(v[len(prefix)] for v in paths),)
        paths=[v for v in paths if v[:len(prefix)]==prefix]


def verify_sample(bank,candidates,requests,roster,size,predictions):
    order=next(order for r,order,_ in requests if r==roster)
    rows=np.unique(np.linspace(0,len(bank)-1,8,dtype=int))
    voters=bank[rows][:,order[:size]]
    projected=project_codes(voters,candidates)
    for key,values in (('raw_valid_medoid',voters),('projected_medoid',projected)):
        reference,_=consensus_predictions(values,candidates,p=71,precision=7)
        assert np.array_equal(reference,predictions[key][rows])
    for i,v in enumerate(projected):
        counts=Counter(v.tolist())
        assert min(counts,key=lambda c:(-counts[c],c))==predictions['projected_plurality'][rows[i]]
        assert reference_survivor(v)==predictions['projected_survivor'][rows[i]]
    for i,v in enumerate(voters):
        counts=Counter(v.tolist())
        raw_mode=min(counts,key=lambda c:(-counts[c],c))
        reference,_=consensus_predictions([[raw_mode]],candidates,p=71,precision=7)
        assert reference[0]==predictions['raw_plurality_project'][rows[i]]


def check_model(e,c,mask):
    """Read-back reconstruction independent of the saved fitting predictions."""
    weights=e['coefficients']
    assert len(weights)==len(mask) and all(0<=w<71**7 for w in weights)
    assert fingerprint(weights)==e['coefficient_sha256']
    assert fingerprint(e['predictions'])==e['prediction_sha256']
    for part,field in (('x_train','training_raw_predictions'),('x_test','held_out_raw_predictions')):
        assert direct_linear(c[part][:,mask],weights,71**7)==e[field]
    raw=np.asarray(e['training_raw_predictions'])
    if np.any(raw==0):
        default,_=direct_medoid(c['y_train'][raw==0],71,7)
    else:
        counts=Counter(c['y_train'].tolist())
        default=min(counts,key=lambda v:(-counts[v],v))
    assert default==e['default']
    assert [v if v else default for v in e['held_out_raw_predictions']]==e['predictions']
    assert [v if v else default for v in e['training_raw_predictions']]==e['training_predictions']
    for name,target,pred in (('training_raw',c['y_train'],e['training_raw_predictions']),
            ('training',c['y_train'],e['training_predictions']),
            ('held_out_raw',c['y_test'],e['held_out_raw_predictions']),
            ('held_out',c['y_test'],e['predictions'])):
        compare_scores(prefix_count_scores(target,pred),e['metrics'][name])
    nonzero=int(np.count_nonzero(weights))
    active=float(np.asarray(c['x_test'][:,mask]@np.asarray([w!=0 for w in weights],dtype=np.int64)).mean())
    assert nonzero==e['stored_nonzero_coefficients']
    assert abs(active-e['mean_active_terms'])<1e-12


def get_bank(conn,batch_id,fraction,fold,c,baseline):
    eligible=np.flatnonzero(c['x_train'].getnnz(axis=0))
    members=[]
    if fraction==1.:
        with conn.cursor(row_factory=dict_row) as cur:
            ids=[m['run_id'] for m in baseline['models'] if m['fold']==fold]
            cur.execute('''SELECT run_id,configuration,status,evidence,metrics
                FROM padjective.paper_randomised_method_runs WHERE run_id=ANY(%s::uuid[])''',(ids,))
            source={r['configuration']['seed']:r for r in cur.fetchall()}
        meta={m['seed']:m for m in baseline['models'] if m['fold']==fold}
        assert len(source)==len(meta)==243
        for base in BANK_SEEDS:
            row=source[base+fold]
            old=row['evidence']['models']['coordinate']
            m=meta[base+fold]
            assert row['status']=='coordinate_optimum' and old['completed']
            assert fingerprint(old['predictions'])==m['prediction_sha256']
            assert fingerprint(old['coefficients'])==m['coefficient_sha256']
            scores=row['metrics']['coordinate']
            e=dict(old,coefficient_sha256=m['coefficient_sha256'],prediction_sha256=m['prediction_sha256'],
                metrics={key:scores[key] for key in ('training_raw','training','held_out_raw','held_out')},
                stored_nonzero_coefficients=scores['stored_nonzero_coefficients'],
                mean_active_terms=scores['mean_nonzero_terms_consulted'])
            check_model(e,c,np.arange(c['x_train'].shape[1]))
            members.append(dict(e,mask=eligible.tolist(),selected_features=len(eligible),
                                run_id=str(row['run_id']),reused=True))
    else:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('''SELECT member_index,status,evidence FROM padjective.paper_subspace_runs
                WHERE batch_id=%s AND cv_fold=%s AND fraction=%s ORDER BY member_index''',
                (batch_id,fold,fraction))
            source=cur.fetchall()
        assert [r['member_index'] for r in source]==list(range(243))
        for index,row in enumerate(source):
            e=row['evidence']
            assert row['status']=='coordinate_optimum'
            mask=feature_mask(eligible,fraction,fold,BANK_SEEDS[index])
            assert mask.tolist()==e['mask'] and fingerprint(mask.tolist())==e['mask_sha256']
            assert e['fit_seed']==BANK_SEEDS[index]+fold and e['seed_base']==BANK_SEEDS[index]
            assert e['certified_coordinates']==e['selected_features']==len(mask)
            assert e['eligible_features']==len(eligible)
            h=e['fit']['history']
            assert h[-1]['complete'] and h[-1]['changes']==0 and all(s['complete'] for s in h)
            assert all(a['training_loss_units']>=b['training_loss_units'] for a,b in zip(h,h[1:]))
            check_model(e,c,mask)
            members.append(dict(e,reused=False))
    assert len(members)==243
    bank=np.array([m['predictions'] for m in members],dtype=np.int64).T
    return bank,members,len(eligible)


def evaluate(args):
    output=args.output
    assert not output.exists(), 'Do not overwrite a previous aggregate export'
    evaluator_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
    fingerprints=fitting_fingerprints()
    context=load_context()
    with db.get_connection() as conn:
        ensure_storage(conn)
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('SELECT manifest,status FROM padjective.paper_subspace_batches WHERE batch_id=%s',(args.batch_id,))
            batch=cur.fetchone()
            assert batch and batch['status']=='fits_complete'
            assert batch['manifest']['protocol_sha256']==hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
            assert batch['manifest']['fitting_fingerprints']==fingerprints
            cur.execute('SELECT count(*) AS n FROM padjective.paper_subspace_runs WHERE batch_id=%s AND status=%s',
                        (args.batch_id,'coordinate_optimum'))
            assert cur.fetchone()['n']==4860
            cur.execute('SELECT report FROM padjective.paper_ensemble_scaling_reports WHERE batch_id=%s',(BASE_ENSEMBLE_BATCH,))
            baseline=cur.fetchone()['report']
            assert baseline['validation']['status']=='passed' and len(baseline['models'])==1215
        conn.commit()
        results=[]
        component_summaries=[]
        for fraction in FRACTIONS:
            for fold,c in context.items():
                bank,members,eligible=get_bank(conn,args.batch_id,fraction,fold,c,baseline)
                candidates=np.unique(c['y_train'])
                requests=[(0,np.arange(243),SIZES)]+[(r,np.random.default_rng(20260911+100*r+fold).permutation(243),(9,81))
                                                       for r in range(1,21)]
                # Independent prefix-based nearest-path checks for 8 complete voter rows.
                sample=bank[np.unique(np.linspace(0,len(bank)-1,8,dtype=int))]
                projection,_=consensus_predictions(sample.reshape(-1,1),candidates,p=71,precision=7)
                assert np.array_equal(projection.reshape(sample.shape),project_codes(sample,candidates))
                for i,m in enumerate(members):
                    component_summaries.append(dict(fraction=fraction,fold=fold,member_index=i,
                        seed_base=BANK_SEEDS[i],selected_features=m['selected_features'],
                        stored_nonzero_coefficients=m['stored_nonzero_coefficients'],
                        mean_active_terms=m['mean_active_terms'],prediction_sha256=m['prediction_sha256'],
                        coefficient_sha256=m['coefficient_sha256'],reused=m['reused']))
                for roster,size,predictions in aggregate_rosters(bank,candidates,requests):
                    verify_sample(bank,candidates,requests,roster,size,predictions)
                    order=requests[roster][1][:size]
                    selected=[members[i] for i in order]
                    costs=dict(selected_features_total=sum(m['selected_features'] for m in selected),
                        feature_union_count=len(set(j for m in selected for j in m['mask'])),
                        eligible_features=eligible,
                        stored_nonzero_coefficients=sum(m['stored_nonzero_coefficients'] for m in selected),
                        mean_active_terms=sum(m['mean_active_terms'] for m in selected))
                    for rule,pred in predictions.items():
                        metrics=exact_scores(c['y_test'].tolist(),pred.tolist(),71,7)
                        compare_scores(metrics,prefix_count_scores(c['y_test'],pred))
                        evidence=dict(fraction=fraction,fold=fold,roster=roster,members=size,rule=rule,
                            metrics=metrics,prediction_sha256=fingerprint(pred.tolist()),
                            candidate_sha256=fingerprint(candidates.tolist()),candidate_paths=len(candidates),
                            mean_depth=float(np.mean([len(path_digits(v)) for v in pred])),
                            exact_correct=int(np.count_nonzero(c['y_test']==pred)),
                            root_correct=int(np.count_nonzero(c['y_test']%71==pred%71)),**costs)
                        assert evidence['exact_correct']==round(metrics['exact_accuracy']*metrics['n'])
                        assert evidence['root_correct']==round(metrics['first_digit_accuracy']*metrics['n'])
                        if fraction==1. and roster==0 and rule==RULES[0]:
                            original=next(r for r in baseline['ensembles'] if r['fold']==fold and r['roster']==0 and r['members']==size)
                            assert original['prediction_sha256']==evidence['prediction_sha256']
                            compare_scores(original['metrics'],metrics)
                        key=(args.batch_id,fold,fraction,roster,size,rule)
                        private=dict(evidence,predictions=pred.tolist(),member_indices=order.tolist())
                        with conn.cursor(row_factory=dict_row) as cur:
                            cur.execute('''SELECT evidence FROM padjective.paper_subspace_results WHERE
                                batch_id=%s AND cv_fold=%s AND fraction=%s AND roster=%s AND members=%s AND rule=%s''',key)
                            saved=cur.fetchone()
                            if saved:
                                assert saved['evidence']==private,'A resumed evaluation must reproduce exactly'
                            else:
                                cur.execute('''INSERT INTO padjective.paper_subspace_results
                                    (batch_id,cv_fold,fraction,roster,members,rule,evidence)
                                    VALUES (%s,%s,%s,%s,%s,%s,%s)''',(*key,Jsonb(private)))
                        results.append(evidence)
                    conn.commit()
                emit('subspace_bank_evaluated',fraction=fraction,fold=fold,complete_banks=len(component_summaries)//243)
        assert len(results)==6125 and len(component_summaries)==6075
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('''SELECT min(started_at) AS first_started,max(finished_at) AS last_finished,
                sum((evidence->>'job_seconds')::float) AS summed_job_seconds,
                sum((evidence->'fit'->>'elapsed_seconds')::float) AS summed_fitting_seconds
                FROM padjective.paper_subspace_runs WHERE batch_id=%s''',(args.batch_id,))
            execution=dict(cur.fetchone())
        execution['fitting_wall_seconds']=(execution['last_finished']-execution['first_started']).total_seconds()
        for key in ('first_started','last_finished'):
            execution[key]=execution[key].isoformat()
        report=dict(batch_id=args.batch_id,manifest=batch['manifest'],evaluator_commit=evaluator_commit,
                    execution=execution,
                    validation=dict(status='passed',component_readbacks=6075,new_coordinate_certificates=4860,
                        original_prediction_sets_matched=45,evaluation_rows=6125,
                        independent_aggregation_sample_rows_per_set=8,all_metrics_double_checked=True),
                    components=component_summaries,fold_results=results,analysis=analyse(results))
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute('SELECT evidence FROM padjective.paper_subspace_results WHERE batch_id=%s',(args.batch_id,))
            readback=cur.fetchall()
            assert len(readback)==len(results)
            for row in readback:
                e=row['evidence']
                assert fingerprint(e['predictions'])==e['prediction_sha256']
                compare_scores(prefix_count_scores(context[e['fold']]['y_test'],e['predictions']),e['metrics'])
            cur.execute('INSERT INTO padjective.paper_subspace_reports (batch_id,report) VALUES (%s,%s)',
                        (args.batch_id,Jsonb(report)))
            cur.execute("UPDATE padjective.paper_subspace_batches SET status='validated' WHERE batch_id=%s",(args.batch_id,))
        conn.commit()
        with conn.cursor() as cur:
            cur.execute('SELECT report FROM padjective.paper_subspace_reports WHERE batch_id=%s',(args.batch_id,))
            assert cur.fetchone()[0]==report
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as out:
        json.dump(report,out,indent=2)
        out.write('\n')
    emit('subspace_evaluation_complete',batch_id=args.batch_id,output=str(output),
         sha256=hashlib.sha256(output.read_bytes()).hexdigest(),validation=report['validation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-id',required=True)
    parser.add_argument('--output',type=Path,required=True)
    evaluate(parser.parse_args())


if __name__=='__main__':
    main()
