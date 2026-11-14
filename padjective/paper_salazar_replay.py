"""Final bounded warm-Adam diagnostic replay; all database access is read-only."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_salazar_single import fingerprint
from .salazar_affine import fit_tree

ORIGINAL_RUN="b5bff126-fb4a-492d-969b-b960ccc2220f"
ROOT=Path("artifacts/salazar-warm-adam-replay-20261002")
CONFIG=dict(p=71,precision=7,fold=0,seed=42,batch_size=32,optimizer="adam",
    initial_log_radius=-7,kappa=1.0,beta=0.9,beta2=0.999,epsilon=1e-8,
    updates=100,seconds=20,overall_seconds=30,
    scope="Training-only diagnostic replay of the original frozen warm Adam member")


def original_fingerprint(conn):
    with conn.cursor() as cur:
        cur.execute("""SELECT md5(configuration::text || metrics::text || evidence::text),status
            FROM padjective.paper_salazar_single_runs WHERE run_id=%s""",(ORIGINAL_RUN,))
        return list(cur.fetchone())


def load():
    conn=db.get_connection()
    with conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET LOCAL statement_timeout = '5s'")
        cur.execute("""SELECT status,configuration,snapshot_id,
            evidence->'snapshot_digest',evidence->'n_train',evidence->'n_features',
            evidence->'n_supported',evidence->'training_nnz',
            evidence->'feature_names_sha256',evidence->'mask_sha256',
            evidence->'models'->'exact_coordinate'->'coefficients',
            evidence->'fit'->'warm_adam'->'history',
            metrics->'exact_coordinate'->'coefficient_sha256'
            FROM padjective.paper_salazar_single_runs WHERE run_id=%s""",(ORIGINAL_RUN,))
        row=cur.fetchone()
    assert row is not None and row[0]=="completed"
    configuration=row[1]
    for key in ["p","precision","fold","seed","batch_size","kappa","beta","beta2","epsilon"]:
        assert configuration[key]==CONFIG[key]
    assert configuration["warm_initial_log_radius"]==CONFIG["initial_log_radius"]
    assert str(row[2])==PAPER_SNAPSHOT and row[3]==SNAPSHOT_DIGEST
    original_stamp=original_fingerprint(conn)
    dataset=_load_paper_dataset(conn,snapshot_ref=PAPER_SNAPSHOT,schema="padjective")
    assert _snapshot_digest(dataset)==SNAPSHOT_DIGEST and dataset.prime_base==71
    train=np.array([record.cv_fold!=0 for record in dataset.records])
    matrix=dataset.features[train].astype(np.int64).tocsr()
    targets=np.array([record.encoded_path for record,take in zip(dataset.records,train,strict=True) if take],dtype=np.int64)
    mask=np.asarray(matrix.sum(axis=0)).reshape(-1)>0
    weights=np.array(row[10],dtype=np.int64)
    assert [matrix.shape[0],matrix.shape[1],int(mask.sum()),matrix.nnz]==list(row[4:8])
    assert fingerprint(list(dataset.feature_names))==row[8]
    assert fingerprint(mask.tolist())==row[9]
    assert fingerprint(weights.tolist())==row[12]
    assert weights.shape==(2542,) and not np.any(weights[~mask])
    provenance=dict(original_run_id=ORIGINAL_RUN,snapshot_id=PAPER_SNAPSHOT,
        snapshot_digest=SNAPSHOT_DIGEST,n_train=matrix.shape[0],n_features=matrix.shape[1],
        n_supported=int(mask.sum()),training_nnz=matrix.nnz,
        feature_names_sha256=row[8],mask_sha256=row[9],initial_coefficient_sha256=row[12],
        original_row_fingerprint=original_stamp,original_configuration=configuration)
    return conn,matrix,targets,weights,row[11],provenance


def run():
    assert platform.node()=="raksasa","Authorized replay host is raksasa"
    started=time.monotonic()
    if ROOT.exists():
        raise RuntimeError("Replay artifacts already exist; preserve completed/partial progress without rerunning")
    ROOT.mkdir()
    result=dict(status="loading",configuration=CONFIG,hostname=platform.node(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ["padjective/salazar_affine.py","padjective/paper_salazar_replay.py"]},
        completed_updates=0,checkpoint_comparisons=[])
    def save():
        (ROOT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    save()
    conn=None
    try:
        conn,matrix,targets,initial,original_history,provenance=load()
        result.update(provenance=provenance,status="running",loading_seconds=time.monotonic()-started)
        reference={h["update"]:h for h in original_history if h["update"]<=100}
        (ROOT/"private-initial.json").write_text(json.dumps(dict(coefficients=initial.tolist(),
            original_checkpoints=reference),indent=2)+"\n")
        save()
        print(json.dumps(dict(event="data_verified",**{k:v for k,v in provenance.items() if k!="original_configuration"})),flush=True)
        with (ROOT/"trace.jsonl").open("w") as trace:
            def diagnostic(event):
                assert time.monotonic()-started<CONFIG["overall_seconds"],"Replay overall cap"
                present=np.unique(matrix[event["batch_indices"]].indices)
                event["batch_supported_columns"]=present.tolist()
                trace.write(json.dumps(event,separators=(",",":"))+"\n")
                trace.flush()
                result["completed_updates"]=event["update"]
                result["last_training_point_loss"]=event["point_loss"]
                result["last_maximum_log_radius"]=max(event["logs_after"])
                if event["update"]%10==0 or event["update"]==1:
                    save()
                    print(json.dumps(dict(event="diagnostic_progress",update=event["update"],
                        point_loss=event["point_loss"],maximum_log_radius=max(event["logs_after"]),
                        elapsed_seconds=time.monotonic()-started)),flush=True)
            def checkpoint(item,centers,logs):
                if item["update"] in reference:
                    expected=reference[item["update"]]
                    fields=[key for key in expected if key!="elapsed_seconds"]
                    mismatches=[key for key in fields if item[key]!=expected[key]]
                    result["checkpoint_comparisons"].append(dict(update=item["update"],
                        matched_fields=fields,mismatches=mismatches,status="passed" if not mismatches else "failed"))
                    save()
                    assert not mismatches, f"Original checkpoint mismatch: {mismatches}"
            fit=fit_tree(matrix,targets,initial,p=71,precision=7,seed=42,optimizer="adam",
                initial_log_radius=-7,updates=100,seconds=min(20,30-(time.monotonic()-started)),
                batch_size=32,kappa=1,beta=0.9,beta2=0.999,epsilon=1e-8,
                checkpoint=checkpoint,diagnostic_callback=diagnostic)
        after=original_fingerprint(conn)
        assert after==provenance["original_row_fingerprint"]
        result.update(status="completed",fit={k:v for k,v in asdict(fit).items() if k not in {"centers","log_radii"}},
            endpoint_coefficient_sha256=fingerprint(fit.centers.tolist()),
            coefficients_unchanged=bool(np.array_equal(initial,fit.centers)),
            original_row_unchanged=True,elapsed_seconds=time.monotonic()-started,
            trace_bytes=(ROOT/"trace.jsonl").stat().st_size,
            trace_sha256=hashlib.sha256((ROOT/"trace.jsonl").read_bytes()).hexdigest())
        save()
        print(json.dumps(dict(event="finished",**result)),flush=True)
    except Exception as exc:
        result.update(status="failed",error=dict(type=type(exc).__name__,message=str(exc)),
            elapsed_seconds=time.monotonic()-started)
        save()
        raise
    finally:
        if conn is not None:
            conn.rollback()
            conn.close()


if __name__=="__main__":
    run()
