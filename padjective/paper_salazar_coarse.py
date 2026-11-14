"""One fixed warm Adam radius-71^-1 hypothesis test, on raksasa only."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
from psycopg.types.json import Jsonb

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_salazar_replay import load, original_fingerprint, ORIGINAL_RUN
from .paper_salazar_single import fingerprint, full_metrics, encode_logs
from .randomised_linear import linear_predictions
from .salazar_affine import fit_tree

ROOT=Path("artifacts/salazar-coarse-adam-20261002")
VARIANT="warm_adam_radius_71_minus_1"
CONFIG=dict(p=71,precision=7,fold=0,seed=42,batch_size=32,optimizer="adam",
    initial_log_radius=-1,kappa=1.0,beta=0.9,beta2=0.999,epsilon=1e-8,
    updates=1000,seconds=120,outer_seconds=180,
    selection="Fixed budget endpoint, including time-limited endpoint; no checkpoint, rate or seed selection",
    held_out_status="Exploratory follow-up of a previously inspected fold, not an independent confirmatory test",
    extraction="Retained canonical centers modulo 71^7, with existing train-fitted zero-score default",
    diagnostic_protocol="Compact states and motion checks every update; full vectors at 1,2,3,10,every100 and the fixed endpoint",
    comparison="Preserved exact baseline and original radius-71^-7 warm Adam; no refitting or replacing them")


def run():
    assert platform.node()=="raksasa"
    if ROOT.exists():
        raise RuntimeError("Coarse artifacts already exist; inspect completed/partial state without repeating fitting")
    ROOT.mkdir()
    started=time.monotonic()
    run_id=str(uuid.uuid4())
    result=dict(run_id=run_id,status="loading",configuration=CONFIG,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
            for n in ["padjective/salazar_affine.py","padjective/paper_salazar_coarse.py"]},
        completed_updates=0)
    def local_save():
        (ROOT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    local_save()
    ro=writer=None
    evidence={"models":{},"fit":{}}
    metrics={}
    def save(status):
        result["status"]=status
        local_save()
        with writer.cursor() as cur:
            cur.execute("""UPDATE padjective.paper_salazar_single_runs
                SET status=%s,evidence=%s,metrics=%s WHERE run_id=%s""",
                (status,Jsonb(evidence),Jsonb(metrics),run_id))
            assert cur.rowcount==1
        writer.commit()
    try:
        ro,x_train,y_train,initial,_,provenance=load()
        result["provenance"]=provenance
        result["loading_seconds"]=time.monotonic()-started
        (ROOT/"private-initial.json").write_text(json.dumps(dict(coefficients=initial.tolist(),initial_log_radius=-1))+"\n")
        writer=db.get_connection()
        with writer.cursor() as cur:
            cur.execute("SELECT COALESCE(t.spcname,'pg_default') FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid LEFT JOIN pg_tablespace t ON c.reltablespace=t.oid WHERE n.nspname='padjective' AND c.relname='paper_salazar_single_runs'")
            assert cur.fetchone()[0]=="pg_default"
            cur.execute("""INSERT INTO padjective.paper_salazar_single_runs
                (run_id,snapshot_id,configuration,status) VALUES (%s,%s,%s,'running')""",
                (run_id,PAPER_SNAPSHOT,Jsonb(dict(CONFIG,variant=VARIANT,provenance=provenance,
                    source_commit=result["source_commit"],source_sha256=result["source_sha256"]))))
        writer.commit()
        evidence.update(snapshot_digest=SNAPSHOT_DIGEST,provenance=provenance)
        save("running")
        print(json.dumps(dict(event="started",run_id=run_id,configuration=CONFIG)),flush=True)
        callback_seconds=0.0
        shallowest=np.full(len(initial),-1.0)
        changed_columns=set()
        digit_depths=Counter()
        counts=Counter()
        applied_times=[]
        last_event=None
        with (ROOT/"trace.jsonl").open("w") as compact, (ROOT/"detailed.jsonl").open("w") as detailed:
            def diagnostic(event):
                nonlocal callback_seconds,last_event,shallowest
                before=time.monotonic()
                last_event=event
                shallowest=np.maximum(shallowest,event["logs_after"])
                present=set(np.unique(x_train[event["batch_indices"]].indices).tolist())
                coordinates={c["column"]:c for c in event["coordinates"]}
                motions={m["column"]:m for m in event["motions"]}
                useful=[]
                per=Counter()
                vertices=Counter()
                for j,c in coordinates.items():
                    available=[c["up_allowed"]]+[c["down_allowed"]]*(len(c["speeds"])-1)
                    negative=any(a and s<0 for a,s in zip(available,c["used_slopes"],strict=True))
                    speed=max(c["speeds"])
                    per["selected"]+=1
                    per["batch_absent_selected"]+=int(j not in present)
                    per["negative_feasible"]+=int(negative)
                    per["history_suppressed"]+=int(negative and speed==0)
                    per["zero_speed"]+=int(speed==0)
                    per["negative_child_history_guards"]+=sum(int(available[k] and c["used_slopes"][k]<0 and c["moment"][k] is None) for k in range(1,len(c["speeds"])))
                    if c["vertex_depth"] is not None:
                        vertices[c["vertex_depth"]]+=1
                    if speed or negative:
                        m=motions.get(j)
                        direction=m["direction"] if m else None
                        useful.append(dict(column=j,clock=c["clock"],vertex_depth=c["vertex_depth"],
                            maximum_speed=speed,negative_feasible=negative,
                            direction=direction,chosen_raw_slope=c["raw_slopes"][direction] if m else None,
                            chosen_used_slope=c["used_slopes"][direction] if m else None))
                for j,m in motions.items():
                    c=coordinates[j]
                    per["proposals"]+=1
                    per["strict_radius_changes"]+=int(m["log_changed"])
                    per["center_changes"]+=int(m["center_changed"])
                    per["numerical_noops"]+=int(m["numerical_noop"])
                    per["batch_absent_motions"]+=int(j not in present)
                    per["zero_current_slope_motions"]+=int(c["used_slopes"][m["direction"]]==0)
                    per["positive_current_slope_motions"]+=int(c["used_slopes"][m["direction"]]>0)
                    per["negative_current_slope_motions"]+=int(c["used_slopes"][m["direction"]]<0)
                    child=m["direction"]>0 and c["vertex_depth"] is not None and c["vertex_depth"]<7
                    per["child_proposals"]+=int(child)
                    if m["center_changed"]:
                        changed_columns.add(j)
                        digit_depths[c["vertex_depth"]]+=1
                per["clipped_updates"]+=int(event["clipped"])
                per["boundary_hits"]+=sum(m["boundary_hit"] for m in event["motions"])
                per["tied_updates"]+=int(bool(event["tie_rows"]))
                per["tie_rows"]+=len(event["tie_rows"])
                counts.update(per)
                applied_times.append(event["applied_time"])
                record={k:event[k] for k in ["update","batch_indices","groups","selected","tie_rows",
                    "centers_after","logs_after","motions","requested_time","applied_time","clipped",
                    "clipping_coordinates","point_loss","direct_loss"]}
                record.update(batch_supported_columns=sorted(present),interesting_coordinates=useful,
                    counts=dict(per),selected_vertex_depth_counts=dict(vertices))
                compact.write(json.dumps(record,separators=(",",":"))+"\n")
                compact.flush()
                if event["update"] in {1,2,3,10} or event["update"]%100==0:
                    detailed.write(json.dumps(event,separators=(",",":"))+"\n")
                    detailed.flush()
                result["completed_updates"]=event["update"]
                result["last_training_point_loss"]=event["point_loss"]
                if event["update"]==1 or event["update"]%100==0:
                    local_save()
                    print(json.dumps(dict(event="progress",update=event["update"],point_loss=event["point_loss"],
                        direct_loss=event["direct_loss"],maximum_log_radius=max(event["logs_after"]),
                        center_changes=counts["center_changes"],elapsed_seconds=time.monotonic()-started)),flush=True)
                callback_seconds+=time.monotonic()-before
            def checkpoint(item,centers,logs):
                evidence["checkpoint"]=dict(item,coefficients=centers.tolist(),log_radii=encode_logs(logs))
                save("running")
            fit=fit_tree(x_train,y_train,initial,p=71,precision=7,seed=42,optimizer="adam",
                initial_log_radius=-1,updates=1000,seconds=120,batch_size=32,kappa=1,
                beta=.9,beta2=.999,epsilon=1e-8,checkpoint=checkpoint,diagnostic_callback=diagnostic)
            if last_event and last_event["update"] not in {1,2,3,10} and last_event["update"]%100:
                detailed.write(json.dumps(last_event,separators=(",",":"))+"\n")
        assert original_fingerprint(ro)==provenance["original_row_fingerprint"]
        fit_record={k:v for k,v in asdict(fit).items() if k not in {"centers","log_radii"}}
        evidence["fit"][VARIANT]=fit_record
        evidence["models"][VARIANT]=dict(coefficients=fit.centers.tolist(),log_radii=encode_logs(fit.log_radii),
            optimization_status=fit.status,completed=fit.status=="update_limit")
        result.update(fit=fit_record,endpoint_coefficient_sha256=fingerprint(fit.centers.tolist()),
            diagnostic_callback_seconds=callback_seconds,
            diagnostic_counts=dict(counts),digit_change_depth_counts=dict(digit_depths),
            unique_changed_columns=len(changed_columns),net_changed_columns=int(np.count_nonzero(fit.centers!=initial)),
            net_changed_first_digits=int(np.count_nonzero(fit.centers%71!=initial%71)),
            maximum_log_radius_ever=float(shallowest.max()),coordinates_ever_at_root=int(np.count_nonzero(shallowest>=0)),
            total_common_applied_time=sum(applied_times))
        (ROOT/"private-endpoint.json").write_text(json.dumps(evidence["models"][VARIANT])+"\n")
        save("endpoint_frozen")
        print(json.dumps(dict(event="endpoint_frozen",run_id=run_id,updates=fit.updates,status=fit.status,
            coefficient_sha256=result["endpoint_coefficient_sha256"])),flush=True)
        # First new held-out predictions and original score access happen only after the frozen endpoint.
        dataset=_load_paper_dataset(ro,snapshot_ref=PAPER_SNAPSHOT,schema="padjective")
        assert _snapshot_digest(dataset)==SNAPSHOT_DIGEST
        train=np.array([r.cv_fold!=0 for r in dataset.records])
        y=np.array([r.encoded_path for r in dataset.records],dtype=np.int64)
        depths=np.array([r.taxonomy_depth for r in dataset.records])
        test=dataset.features[~train].astype(np.int64).tocsr()
        before=time.monotonic()
        private,score=full_metrics(fit.centers,x_train,test,y_train,y[~train],depths[train],depths[~train])
        result["endpoint_scoring_and_audit_seconds"]=time.monotonic()-before
        private.update(coefficients=fit.centers.tolist(),log_radii=encode_logs(fit.log_radii),
            optimization_status=fit.status,completed=fit.status=="update_limit")
        timings=[]
        for _ in range(5):
            before=time.perf_counter()
            linear_predictions(test,fit.centers,71,7)
            timings.append(time.perf_counter()-before)
        score.update(fit_seconds=fit.elapsed_seconds,total_training_seconds=fit.elapsed_seconds+4.082228513900191,
            median_raw_prediction_seconds=float(np.median(timings)),sampled_training_examples=fit.updates*32,
            training_example_equivalents=fit.updates*32/len(y_train))
        evidence["models"][VARIANT]=private
        metrics[VARIANT]=score
        with ro.cursor() as cur:
            cur.execute("SELECT metrics FROM padjective.paper_salazar_single_runs WHERE run_id=%s",(ORIGINAL_RUN,))
            original=cur.fetchone()[0]
        result["comparison_metrics"]={"exact_coordinate":original["exact_coordinate"],
            "warm_adam_radius_71_minus_7":original["warm_adam"],VARIANT:score}
        result["metrics"]=score
        result["original_row_unchanged"]=original_fingerprint(ro)==provenance["original_row_fingerprint"]
        assert result["original_row_unchanged"]
        result["trace_sha256"]={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ["trace.jsonl","detailed.jsonl"]}
        result["trace_bytes"]={n:(ROOT/n).stat().st_size for n in result["trace_sha256"]}
        result["elapsed_seconds"]=time.monotonic()-started
        save("completed")
        with writer.cursor() as cur:
            cur.execute("UPDATE padjective.paper_salazar_single_runs SET finished_at=NOW() WHERE run_id=%s",(run_id,))
        writer.commit()
        print(json.dumps(dict(event="completed",run_id=run_id,fit=fit_record,metrics=score,
            diagnostic_counts=dict(counts),elapsed_seconds=result["elapsed_seconds"])),flush=True)
    except Exception as exc:
        result.update(status="failed",error=dict(type=type(exc).__name__,message=str(exc)))
        local_save()
        if writer is not None:
            writer.rollback()
            evidence["error"]=result["error"]
            save("failed")
        raise
    finally:
        if ro is not None:
            ro.rollback()
            ro.close()
        if writer is not None:
            writer.close()


if __name__=="__main__":
    run()
