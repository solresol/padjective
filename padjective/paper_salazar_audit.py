"""Independent Python-integer audit of persisted single-member evidence."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform

import numpy as np

from . import db
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_salazar_single import fingerprint


def scores(targets, predictions, depths):
    units = correct = root = prefix = 0
    for gold,pred,depth in zip(targets,predictions,depths,strict=True):
        delta = abs(int(gold)-int(pred))
        if delta == 0:
            correct += 1
            root += 1
            prefix += int(depth)
            continue
        valuation = 0
        while delta % 71 == 0:
            delta //= 71
            valuation += 1
        units += 71**(7-valuation)
        root += valuation > 0
        prefix += min(valuation,int(depth))
    n=len(targets)
    return dict(n=n,mean_padic_loss=float(Fraction(units,n*71**7)),
        exact_accuracy=float(Fraction(correct,n)),first_digit_accuracy=float(Fraction(root,n)),
        mean_shared_prefix_depth=float(Fraction(prefix,n)))


def audit(run_id, output):
    assert platform.node() == "raksasa"
    conn=db.get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT status,configuration,metrics,evidence FROM padjective.paper_salazar_single_runs WHERE run_id=%s",(run_id,))
        status,config,metrics,evidence=cur.fetchone()
    assert status == "completed"
    dataset=_load_paper_dataset(conn,snapshot_ref=PAPER_SNAPSHOT,schema="padjective")
    assert _snapshot_digest(dataset) == evidence["snapshot_digest"] == SNAPSHOT_DIGEST
    train=np.array([row.cv_fold!=config["fold"] for row in dataset.records])
    weights_seen=[]
    checked=0
    for variant,record in evidence["models"].items():
        weights=record["coefficients"]
        assert len(weights)==2542 and all(0<=w<71**7 for w in weights)
        assert fingerprint(weights)==metrics[variant]["coefficient_sha256"]
        fit_status=evidence["fit"][variant]["status"]
        assert record["optimization_status"]==fit_status
        assert record["completed"]==(fit_status in {"coordinate_optimum","update_limit"})
        for mask,label,raw_key,pred_key in [(train,"training","training_raw_predictions","training_predictions"),
                                          (~train,"held_out","held_out_raw_predictions","predictions")]:
            matrix=dataset.features[mask].tocsr()
            rows=[row for row,take in zip(dataset.records,mask,strict=True) if take]
            raw=[sum(int(weights[j]) for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]])%71**7
                 for i in range(matrix.shape[0])]
            predicted=[v if v else record["default"] for v in raw]
            assert raw==record[raw_key] and predicted==record[pred_key]
            gold=[r.encoded_path for r in rows]
            depths=[r.taxonomy_depth for r in rows]
            for key,values in [(label+"_raw",raw),(label,predicted)]:
                independently=scores(gold,values,depths)
                assert independently==metrics[variant][key]
            checked+=len(rows)
        test_matrix=dataset.features[~train].tocsr()
        active=[sum(weights[j]!=0 for j in test_matrix.indices[
            test_matrix.indptr[i]:test_matrix.indptr[i+1]])
            for i in range(test_matrix.shape[0])]
        assert sum(w!=0 for w in weights)==metrics[variant]["stored_nonzero_coefficients"]
        assert float(Fraction(sum(active),len(active)))==metrics[variant]["mean_nonzero_terms_consulted"]
        weights_seen.append((variant,weights))
    models=dict(weights_seen)
    assert models["warm_momentum"]==models["warm_adam"]==models["exact_coordinate"]
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(t.spcname,'pg_default') FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid LEFT JOIN pg_tablespace t ON c.reltablespace=t.oid WHERE n.nspname='padjective' AND c.relname='paper_salazar_single_runs'")
        tablespace=cur.fetchone()[0]
    assert tablespace == "pg_default"
    conn.close()
    result=dict(run_id=run_id,status="passed",snapshot_digest=SNAPSHOT_DIGEST,
        independently_reconstructed_predictions=checked,metric_sets=16,
        all_warm_coefficients_identical=True,tablespace=tablespace,
        method="Python integer sums and Fraction metric numerators from stored coefficients and frozen Postgres rows")
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--output",type=Path,default=Path("artifacts/salazar-single-audit.json"))
    args=parser.parse_args()
    audit(args.run_id,args.output)
