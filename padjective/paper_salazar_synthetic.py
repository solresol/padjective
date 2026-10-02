"""Bounded, fully logged five-row validation; no product data are accessed."""
from fractions import Fraction
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

X=sparse.csr_matrix([[1,0],[0,1],[1,1],[1,1],[1,1]],dtype=np.int64)
Y=np.array([1,26,0,0,0],dtype=np.int64)
CONFIG=dict(p=3,precision=3,initial_centers=[0,0],initial_log_radius=-3,
    optimizers=["gd","momentum","adam"],kappa=1,beta=0.9,beta2=0.999,epsilon=1e-8,
    full_batch_updates=100,controlled_updates=400,controlled_singleton_updates=200,
    stochastic_optimizers=["momentum","adam"],seeds=[42,1729],batch_sizes=[1,2],
    stochastic_updates=1000,seconds_per_trial=5,overall_seconds=60,
    extraction="retained canonical centers modulo 27",scope="five synthetic rows only")


def units(weights):
    total=0
    for i,gold in enumerate(Y):
        predicted=sum(int(weights[j]) for j in X.indices[X.indptr[i]:X.indptr[i+1]])%27
        delta=(predicted-int(gold))%27
        if delta:
            value=27
            while delta%3==0:
                delta//=3
                value//=3
            total+=value
    return total


def certify():
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


def trial(key,*,optimizer,seed,batch_size,updates,schedule,root,deadline):
    filename=root/(key+".jsonl")
    summary=dict(optimizer=optimizer,seed=seed,batch_size=batch_size,
        trace_file=filename.name,requested_updates=updates,initial_units=54,
        best_units=54,first_improvement_update=None,first_zero_update=None,
        negative_raw_slope_selections=0,zero_used_slope_selections=0,
        clipped_updates=0,boundary_hits=0,numerical_noops=0,strict_radius_changes=0,
        changed_center_motions=0,tied_updates=0,minimum_applied_positive_time=None,
        controlled_switch_units=None,diagnostic_rows=0)
    started=time.monotonic()
    with filename.open("w") as trace:
        def log(event):
            assert time.monotonic()<deadline,"Synthetic overall time cap"
            cost=units(event["centers_after"])
            event["independent_point_units"]=cost
            assert event["point_loss"]==float(Fraction(cost,135))
            assert event["centers_before"]==[0,0] if event["update"]==1 else True
            if cost<54 and summary["first_improvement_update"] is None:
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
        fit=fit_tree(X,Y,[0,0],p=3,precision=3,seed=seed,optimizer=optimizer,
            initial_log_radius=-3,updates=updates,seconds=min(5,deadline-time.monotonic()),
            batch_size=batch_size,kappa=1,batch_schedule=schedule,diagnostic_callback=log)
    summary.update(status=fit.status,updates=fit.updates,endpoint_centers=fit.centers.tolist(),
        endpoint_log_radii=fit.log_radii.tolist(),endpoint_units=units(fit.centers),
        endpoint_loss=float(Fraction(units(fit.centers),135)),moves=fit.moves,
        coordinate_updates=fit.coordinate_updates,digit_changes=fit.digit_changes,
        elapsed_seconds=time.monotonic()-started,
        trace_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),trace_bytes=filename.stat().st_size)
    summary["escaped_at_endpoint"]=summary["endpoint_units"]<54
    print(json.dumps(dict(event="trial_finished",key=key,**summary)),flush=True)
    return summary


def run():
    assert platform.node()=="raksasa","Authorized synthetic host is raksasa"
    started=time.monotonic()
    deadline=started+60
    root=Path("artifacts/salazar-synthetic-20261002")
    root.mkdir(exist_ok=True)
    # Do not overwrite prior traces on a repeated invocation.
    if (root/"results.json").exists() or list(root.glob("*.jsonl")):
        raise RuntimeError("Synthetic artifacts already exist; preserve them before an explicitly authorized rerun")
    result=dict(configuration=CONFIG,hostname=platform.node(),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        source_sha256={n:hashlib.sha256(Path(n).read_bytes()).hexdigest()
            for n in ["padjective/salazar_affine.py","padjective/paper_salazar_synthetic.py"]},
        certification=certify(),trials={})
    print(json.dumps(dict(event="certified",**result["certification"])),flush=True)
    full=lambda _:np.arange(5,dtype=np.int64)
    controlled=lambda step:np.array([0,1],dtype=np.int64) if step<=200 else np.arange(5,dtype=np.int64)
    def add(key,**kw):
        result["trials"][key]=trial(key,root=root,deadline=deadline,**kw)
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
    run()
