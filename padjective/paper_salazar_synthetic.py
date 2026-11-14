"""Bounded, fully logged five-row validation; no product data are accessed."""
from fractions import Fraction
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy import sparse

from .salazar_affine import fit_tree
from .randomised_linear import coordinate_certificate

X=sparse.csr_matrix([[1,0],[0,1],[1,1],[1,1],[1,1]],dtype=np.int64)
Y=np.array([1,26,0,0,0],dtype=np.int64)
CONFIG=dict(p=3,precision=3,initial_centers=[0,0],initial_log_radius=-3,
    optimizers=["gd","momentum","adam"],kappa=1,beta=0.9,beta2=0.999,epsilon=1e-8,
    full_batch_updates=100,controlled_updates=400,controlled_singleton_updates=200,
    stochastic_optimizers=["momentum","adam"],seeds=[42,1729],batch_sizes=[1,2],
    stochastic_updates=1000,seconds_per_trial=5,overall_seconds=60,
    extraction="retained canonical centers modulo 27",scope="five synthetic rows only")


def units(weights, p=3, precision=3):
    q=p**precision
    targets=[1,q-1,0,0,0]
    total=0
    for i,gold in enumerate(targets):
        predicted=sum(int(weights[j]) for j in X.indices[X.indptr[i]:X.indptr[i+1]])%q
        delta=(predicted-gold)%q
        if delta:
            value=q
            while delta%p==0:
                delta//=p
                value//=p
            total+=value
    return total


def analytic_certificate(p, precision):
    """A universal valuation argument, independently checked by exact medoids.

    With the other coefficient zero, the five-row unnormalized cost is
    |w-s|_p + 1 + 3|w|_p, s=+1 or -1. If p divides w, |w-s|=1,
    so cost is 2+3|w|, with equality only at residue w=0. Otherwise
    |w|=1 and cost is 4+|w-s| >=4. This covers every residue without
    enumerating q or q^2. The singleton equations force (1,q-1), which
    also cancels on the three joint rows and has the lower bound loss 0.
    """
    q=p**precision
    certificate=coordinate_certificate(X,np.array([1,q-1,0,0,0]),[0,0],p=p,precision=precision)
    assert certificate["status"]=="coordinate_optimum"
    assert units([0,0],p,precision)==2*q
    assert units([1,q-1],p,precision)==0
    # Bounded probes of every valuation stratum and both singleton signs.
    probes=sorted({0,1,q-1}|{s*p**depth%q for s in [1,-1] for depth in range(1,precision)})
    for w in probes:
        for column in range(2):
            weights=[0,0]
            weights[column]=w
            assert units(weights,p,precision)>=2*q
            assert (units(weights,p,precision)==2*q)==(w==0)
    return dict(status="passed",method="Universal valuation-case proof plus exact residual-medoid coordinate certificate",
        modulus=q,denominator=5*q,initial_units=2*q,initial_loss=0.4,
        coordinate_minimizers=[[0],[0]],global_minimum_units=0,joint_minimizers=[[1,q-1]],
        proof="For each coordinate, C(w)=|w-s|_p+1+3|w|_p. If p divides w, C(w)=2+3|w|_p, equal to 2 only at w=0. For a unit, C(w)=4+|w-s|_p>=4. Singleton rows force the unique zero-loss pair (1,-1) modulo q.",
        exact_medoid_certificate=certificate,bounded_probe_candidates=probes,
        bounded_coordinate_probes_checked=2*len(probes),joint_candidates_checked=1,
        exhaustive_enumeration=False)


def certify(p=3, precision=3):
    if (p,precision)!=(3,3):
        return analytic_certificate(p,precision)
    before=units([0,0])
    coordinate_costs=[[units([candidate,0]) for candidate in range(27)],
                      [units([0,candidate]) for candidate in range(27)]]
    joint=[(units(pair),list(pair)) for pair in itertools.product(range(27),repeat=2)]
    optimum=min(cost for cost,_ in joint)
    minimizers=[pair for cost,pair in joint if cost==optimum]
    assert before==54 and all(min(costs)==before for costs in coordinate_costs)
    assert optimum==0 and minimizers==[[1,26]]
    return dict(status="passed",initial_units=before,initial_loss=float(Fraction(before,135)),
        denominator=135,coordinate_candidates_checked=54,joint_candidates_checked=729,
        coordinate_cost_units=coordinate_costs,global_minimum_units=optimum,
        joint_minimizers=minimizers,method="Independent Python integer exhaustive finite-ring scoring")


def trial(key,*,optimizer,seed,batch_size,updates,schedule,root,deadline,p=3,precision=3):
    q=p**precision
    score=lambda weights:units(weights,p,precision)
    filename=root/(key+".jsonl")
    summary=dict(optimizer=optimizer,seed=seed,batch_size=batch_size,
        trace_file=filename.name,requested_updates=updates,initial_units=2*q,
        best_units=2*q,first_improvement_update=None,first_zero_update=None,
        negative_raw_slope_selections=0,zero_used_slope_selections=0,
        clipped_updates=0,boundary_hits=0,numerical_noops=0,strict_radius_changes=0,
        changed_center_motions=0,tied_updates=0,minimum_applied_positive_time=None,
        controlled_switch_units=None,diagnostic_rows=0)
    started=time.monotonic()
    with filename.open("w") as trace:
        def log(event):
            assert time.monotonic()<deadline,"Synthetic overall time cap"
            cost=score(event["centers_after"])
            event["independent_point_units"]=cost
            assert event["point_loss"]==float(Fraction(cost,5*q))
            assert event["centers_before"]==[0,0] if event["update"]==1 else True
            if cost<2*q and summary["first_improvement_update"] is None:
                summary["first_improvement_update"]=event["update"]
            if cost==0 and summary["first_zero_update"] is None:
                summary["first_zero_update"]=event["update"]
            summary["best_units"]=min(summary["best_units"],cost)
            summary["negative_raw_slope_selections"]+=sum(min(c["raw_slopes"])<0 for c in event["coordinates"])
            summary["zero_used_slope_selections"]+=sum(all(v==0 for v in c["used_slopes"]) for c in event["coordinates"])
            summary["clipped_updates"]+=int(event["clipped"])
            summary["boundary_hits"]+=sum(int(m["boundary_hit"]) for m in event["motions"])
            summary["numerical_noops"]+=int(event["numerical_noops"])
            summary["strict_radius_changes"]+=int(event["strict_radius_changes"])
            summary["changed_center_motions"]+=sum(int(m["center_changed"]) for m in event["motions"])
            summary["tied_updates"]+=int(bool(event["tie_rows"]))
            if event["motions"] and event["applied_time"]>0:
                previous=summary["minimum_applied_positive_time"]
                summary["minimum_applied_positive_time"]=min(previous or event["applied_time"],event["applied_time"])
            if event["update"]==200:
                summary["controlled_switch_units"]=cost
            summary["diagnostic_rows"]+=1
            trace.write(json.dumps(event,separators=(",",":"))+"\n")
        fit=fit_tree(X,np.array([1,q-1,0,0,0]),[0,0],p=p,precision=precision,seed=seed,optimizer=optimizer,
            initial_log_radius=-precision,updates=updates,seconds=min(5,deadline-time.monotonic()),
            batch_size=batch_size,kappa=1,batch_schedule=schedule,diagnostic_callback=log)
    summary.update(status=fit.status,updates=fit.updates,endpoint_centers=fit.centers.tolist(),
        endpoint_log_radii=fit.log_radii.tolist(),endpoint_units=score(fit.centers),
        endpoint_loss=float(Fraction(score(fit.centers),5*q)),moves=fit.moves,
        coordinate_updates=fit.coordinate_updates,digit_changes=fit.digit_changes,
        elapsed_seconds=time.monotonic()-started,
        trace_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),trace_bytes=filename.stat().st_size)
    summary["escaped_at_endpoint"]=summary["endpoint_units"]<2*q
    print(json.dumps(dict(event="trial_finished",key=key,**summary)),flush=True)
    return summary


def run(p=3, precision=3):
    assert platform.node()=="raksasa","Authorized synthetic host is raksasa"
    assert (p,precision) in {(3,3),(71,7)},"Only the two authorized fixed profiles"
    started=time.monotonic()
    deadline=started+60
    root=Path("artifacts/salazar-synthetic-20261002" if p==3 else "artifacts/salazar-synthetic-p71-20261002")
    root.mkdir(exist_ok=True)
    # Do not overwrite prior traces on a repeated invocation.
    if (root/"results.json").exists() or list(root.glob("*.jsonl")):
        raise RuntimeError("Synthetic artifacts already exist; preserve them before an explicitly authorized rerun")
    config=dict(CONFIG,p=p,precision=precision,initial_log_radius=-precision,
        extraction=f"retained canonical centers modulo {p**precision}")
    result=dict(configuration=config,hostname=platform.node(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
            for n in ["padjective/salazar_affine.py","padjective/paper_salazar_synthetic.py"]},
        certification=certify(p,precision),trials={})
    print(json.dumps(dict(event="certified",**result["certification"])),flush=True)
    full=lambda _:np.arange(5,dtype=np.int64)
    controlled=lambda step:np.array([0,1],dtype=np.int64) if step<=200 else np.arange(5,dtype=np.int64)
    def add(key,**kw):
        result["trials"][key]=trial(key,root=root,deadline=deadline,p=p,precision=precision,**kw)
        (root/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    for optimizer in CONFIG["optimizers"]:
        add("full_"+optimizer,optimizer=optimizer,seed=42,batch_size=5,updates=100,schedule=full)
    for optimizer in CONFIG["optimizers"]:
        add("controlled_"+optimizer,optimizer=optimizer,seed=42,batch_size=2,updates=400,schedule=controlled)
    for optimizer,seed,size in itertools.product(CONFIG["stochastic_optimizers"],CONFIG["seeds"],CONFIG["batch_sizes"]):
        add(f"stochastic_{optimizer}_{seed}_b{size}",optimizer=optimizer,seed=seed,
            batch_size=size,updates=1000,schedule=None)
    result.update(status="completed",elapsed_seconds=time.monotonic()-started)
    (root/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(event="finished",status="completed",elapsed_seconds=result["elapsed_seconds"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["p3","p71"],default="p3")
    args=parser.parse_args()
    run(*( (3,3) if args.profile=="p3" else (71,7) ))
