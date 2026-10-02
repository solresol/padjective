"""Read-only audit of the frozen coarse arm, its motions and existing scores."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import platform
import time

import numpy as np

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_salazar_audit import scores
from .paper_salazar_coarse import ROOT,VARIANT
from .paper_salazar_replay import load,original_fingerprint
from .paper_salazar_replay_audit import point_units,quantiles,raw_slopes
from .paper_salazar_single import fingerprint,independent_predictions


def audit():
    assert platform.node()=="raksasa"
    started=time.monotonic()
    result=json.loads((ROOT/"results.json").read_text())
    assert result["status"]=="completed"
    conn,matrix,targets,initial,_,provenance=load()
    assert provenance==result["provenance"]
    with conn.cursor() as cur:
        cur.execute("SELECT status,configuration,evidence,metrics FROM padjective.paper_salazar_single_runs WHERE run_id=%s",(result["run_id"],))
        status,config,evidence,stored_metrics=cur.fetchone()
    assert status=="completed" and config["initial_log_radius"]==-1
    assert config["selection"]==result["configuration"]["selection"]
    record=evidence["models"][VARIANT]
    assert stored_metrics[VARIANT]==result["metrics"]
    assert record["optimization_status"]==result["fit"]["status"]
    assert record["completed"]==(result["fit"]["status"]=="update_limit")
    assert fingerprint(record["coefficients"])==result["endpoint_coefficient_sha256"]
    for name,digest in result["trace_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    counts=Counter()
    vertices=Counter()
    digit_depths=Counter()
    times,ratios=[],[]
    previous_centers=initial.tolist()
    previous_logs=[-1.0]*len(initial)
    shallowest=np.full(len(initial),-1.0)
    last_units=point_units(matrix,targets,previous_centers)
    point_states=1
    first_center_change=None
    first_first_digit_change=None
    clipped_limiters=Counter()
    for number,line in enumerate((ROOT/"trace.jsonl").open(),1):
        event=json.loads(line)
        assert event["update"]==number
        assert len(event["batch_indices"])==32 and len(set(event["batch_indices"]))==32
        assert all(0<=v<71**7 for v in event["centers_after"])
        assert all(-7<=v<=0 for v in event["logs_after"])
        present=set(np.unique(matrix[event["batch_indices"]].indices).tolist())
        assert present==set(event["batch_supported_columns"])
        selected=set(event["selected"])
        assert len(selected)==len(event["selected"])==len(event["groups"])
        assert all(len(selected.intersection(group))==1 for group in event["groups"])
        interesting={c["column"]:c for c in event["interesting_coordinates"]}
        motions={m["column"]:m for m in event["motions"]}
        bounds=[]
        step=Counter()
        step["selected"]=len(selected)
        step["batch_absent_selected"]=len(selected-present)
        step["negative_feasible"]=sum(c["negative_feasible"] for c in interesting.values())
        step["history_suppressed"]=sum(c["negative_feasible"] and c["maximum_speed"]==0 for c in interesting.values())
        step["zero_speed"]=len(selected)-len(motions)
        for j,m in motions.items():
            c=interesting[j]
            assert m["direction"]==c["direction"] and m["speed"]==c["maximum_speed"]>0
            assert m["requested_distance"]==event["requested_time"]*m["speed"]
            assert m["applied_distance"]==event["applied_time"]*m["speed"]
            bounds.append(m["boundary_distance"]/m["speed"])
            hit=event["applied_time"]==bounds[-1]
            assert hit==m["boundary_hit"]
            if hit:
                assert event["logs_after"][j]==m["boundary_log"]
                clipped_limiters[j]+=1
            else:
                radius=71**previous_logs[j]+(m["applied_distance"] if m["direction"]==0 else -m["applied_distance"])
                assert event["logs_after"][j]==math.log(radius,71)
            center=previous_centers[j]
            depth=c["vertex_depth"]
            child=m["direction"]>0 and depth is not None and depth<7
            if child:
                power=71**depth
                center=(center+(m["direction"]-1-center//power%71)*power)%71**7
            assert event["centers_after"][j]==center
            changed=center!=previous_centers[j]
            radius_changed=event["logs_after"][j]!=previous_logs[j]
            assert changed==m["center_changed"] and radius_changed==m["log_changed"]
            assert m["numerical_noop"]==(not changed and not radius_changed)
            if changed:
                digit_depths[depth]+=1
                if first_center_change is None:
                    first_center_change=number
                if center%71!=previous_centers[j]%71 and first_first_digit_change is None:
                    first_first_digit_change=number
            step["proposals"]+=1
            step["strict_radius_changes"]+=int(radius_changed)
            step["center_changes"]+=int(changed)
            step["numerical_noops"]+=int(not changed and not radius_changed)
            step["batch_absent_motions"]+=int(j not in present)
            step["zero_current_slope_motions"]+=int(c["chosen_used_slope"]==0)
            step["positive_current_slope_motions"]+=int(c["chosen_used_slope"]>0)
            step["negative_current_slope_motions"]+=int(c["chosen_used_slope"]<0)
            step["child_proposals"]+=int(child)
            step["boundary_hits"]+=int(hit)
        for j in selected-set(motions):
            assert event["centers_after"][j]==previous_centers[j] and event["logs_after"][j]==previous_logs[j]
        for j in set(range(len(initial)))-selected:
            assert event["centers_after"][j]==previous_centers[j] and event["logs_after"][j]==previous_logs[j]
        assert event["applied_time"]==min([event["requested_time"]]+bounds)
        assert event["clipping_coordinates"]==[j for j,m in motions.items() if m["boundary_hit"]]
        step["clipped_updates"]=int(bool(motions) and event["applied_time"]<event["requested_time"])
        step["tied_updates"]=int(bool(event["tie_rows"]))
        step["tie_rows"]=len(event["tie_rows"])
        for key,value in step.items():
            assert value==event["counts"].get(key,0),(number,key)
        counts.update(event["counts"])
        vertices.update({int(k):v for k,v in event["selected_vertex_depth_counts"].items()})
        if event["centers_after"]!=previous_centers:
            last_units=point_units(matrix,targets,event["centers_after"])
            point_states+=1
        assert math.isclose(event["point_loss"],float(Fraction(last_units,len(targets)*71**7)),rel_tol=0,abs_tol=1e-16)
        times.append(event["applied_time"])
        ratios.append(event["applied_time"]/event["requested_time"])
        shallowest=np.maximum(shallowest,event["logs_after"])
        previous_centers,previous_logs=event["centers_after"],event["logs_after"]
        if number%100==0:
            print(json.dumps(dict(event="audit_progress",updates=number,elapsed_seconds=time.monotonic()-started)),flush=True)
    assert number==result["fit"]["updates"]==result["completed_updates"]
    assert dict(counts)==result["diagnostic_counts"]
    assert counts["proposals"]==result["fit"]["moves"]
    assert counts["center_changes"]==result["fit"]["digit_changes"]
    assert counts["selected"]==result["fit"]["coordinate_updates"]
    assert previous_centers==record["coefficients"] and previous_logs==record["log_radii"]
    assert fingerprint(previous_centers)==result["endpoint_coefficient_sha256"]
    verified_vectors=0
    detailed_updates=[]
    for line in (ROOT/"detailed.jsonl").open():
        event=json.loads(line)
        detailed_updates.append(event["update"])
        rows=[]
        for i in event["batch_indices"]:
            cols=matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]].tolist()
            pred=sum(event["centers_before"][j] for j in cols)%71**7
            rows.append((cols,int(targets[i]),pred))
        present=set(j for cols,_,_ in rows for j in cols)
        for c in event["coordinates"]:
            if c["column"] in present:
                assert raw_slopes(rows,event["centers_before"],event["logs_before"],c["column"],c["vertex_depth"])==c["raw_slopes"]
                verified_vectors+=1
            else:
                assert all(v==0 for v in c["raw_slopes"])
    # Independent endpoint train/held-out scores, default, support and hashes from frozen PG evidence.
    dataset=_load_paper_dataset(conn,snapshot_ref=PAPER_SNAPSHOT,schema="padjective")
    assert _snapshot_digest(dataset)==SNAPSHOT_DIGEST
    train=np.array([r.cv_fold!=0 for r in dataset.records])
    metric=result["metrics"]
    for mask,label,raw_key,pred_key in [(train,"training","training_raw_predictions","training_predictions"),
        (~train,"held_out","held_out_raw_predictions","predictions")]:
        rows=[r for r,take in zip(dataset.records,mask,strict=True) if take]
        x=dataset.features[mask].tocsr()
        raw=independent_predictions(x,previous_centers,71**7)
        predicted=[v if v else record["default"] for v in raw]
        assert raw==record[raw_key] and predicted==record[pred_key]
        for suffix,values in [("_raw",raw),("",predicted)]:
            assert scores([r.encoded_path for r in rows],values,[r.taxonomy_depth for r in rows])==metric[label+suffix]
    test=dataset.features[~train].tocsr()
    active=[sum(previous_centers[j]!=0 for j in test.indices[test.indptr[i]:test.indptr[i+1]]) for i in range(test.shape[0])]
    assert int(np.count_nonzero(previous_centers))==metric["stored_nonzero_coefficients"]
    assert float(Fraction(sum(active),len(active)))==metric["prediction_active_support"]["mean"]
    assert metric["prediction_active_support"]["median"]==float(np.median(active))
    assert metric["prediction_active_support"]["p95"]==float(np.quantile(active,.95))
    assert metric["prediction_active_support"]["maximum"]==max(active)
    assert fingerprint(record["held_out_raw_predictions"])==metric["raw_prediction_sha256"]
    assert fingerprint(record["predictions"])==metric["prediction_sha256"]
    assert original_fingerprint(conn)==provenance["original_row_fingerprint"]
    conn.rollback()
    conn.close()
    output=dict(status="passed",checked_updates=number,counts=dict(counts),
        independently_scored_point_states=point_states,detailed_updates=detailed_updates,
        independently_verified_directional_vectors=verified_vectors,
        first_coefficient_change_update=first_center_change,first_first_digit_change_update=first_first_digit_change,
        selected_vertex_depth_counts=dict(vertices),digit_change_depth_counts=dict(digit_depths),
        maximum_log_radius_ever=float(shallowest.max()),coordinates_ever_at_root=int(np.count_nonzero(shallowest>=0)),
        applied_time_distribution=quantiles(times),applied_to_requested_time_ratio=quantiles(ratios),
        limiting_coordinate_counts=dict(clipped_limiters),
        endpoint_metric_sets_verified=4,endpoint_predictions_reconstructed=6693,
        original_row_unchanged=True,elapsed_seconds=time.monotonic()-started,
        scope="Stored trajectory and frozen PG endpoint audit; no fitting or model selection")
    (ROOT/"audit.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output))


if __name__=="__main__":
    audit()
