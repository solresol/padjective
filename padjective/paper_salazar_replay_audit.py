"""Independent integer point scoring and stored-motion audit; no optimization."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import platform

import numpy as np

from .paper_salazar_replay import ROOT, load, original_fingerprint
from .paper_salazar_single import fingerprint

P,E,Q=71,7,71**7


def norm(delta):
    delta=int(delta)%Q
    if not delta:
        return 0.0
    exponent=0
    while delta%P==0:
        delta//=P
        exponent+=1
    return P**(-exponent)


def point_units(matrix,targets,weights):
    total=0
    for i,target in enumerate(targets):
        predicted=sum(int(weights[j]) for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]])%Q
        delta=(int(target)-predicted)%Q
        value=Q if delta else 0
        while delta and delta%P==0:
            delta//=P
            value//=P
        total+=value
    return total


def quantiles(values):
    return dict(minimum=float(min(values)),median=float(np.median(values)),
        p95=float(np.quantile(values,.95)),maximum=float(max(values))) if values else None


def raw_slopes(rows,weights,logs,column,depth):
    """Derive germs by explicit child residues, independently of sparse exception accumulation."""
    radius=P**logs[column]
    children=P if depth is not None and depth<E else 1
    slopes=[0.0]*(1+children)
    for cols,target,predicted in rows:
        if column not in cols:
            continue
        other=max([P**logs[j] for j in cols if j!=column],default=0.0)
        error=norm(target-predicted)
        if radius>=other:
            slopes[0]+=.5 if error<=radius else -.5
        if radius<=other:
            continue
        if children==1:
            slopes[1]+=-.5 if error<radius else .5
        else:
            power=P**depth
            old_digit=weights[column]//power%P
            for digit in range(P):
                child=weights[column]+(digit-old_digit)*power
                child_error=norm(target-predicted+weights[column]-child)
                slopes[1+digit]+=-.5 if child_error<radius else .5
    return [value/len(rows) for value in slopes]


def audit():
    assert platform.node()=="raksasa"
    result=json.loads((ROOT/"results.json").read_text())
    assert result["status"]=="completed"
    trace=ROOT/"trace.jsonl"
    assert hashlib.sha256(trace.read_bytes()).hexdigest()==result["trace_sha256"]
    conn,matrix,targets,initial,reference,provenance=load()
    assert provenance==result["provenance"]
    private=json.loads((ROOT/"private-initial.json").read_text())
    assert private["coefficients"]==initial.tolist()
    baseline_units=point_units(matrix,targets,initial)
    baseline_loss=float(Fraction(baseline_units,len(targets)*Q))
    counts=Counter()
    time_values,ratios,distances=[],[],[]
    limiting_columns=Counter()
    first_limiting_step={}
    shallowest=[-7.0]*matrix.shape[1]
    previous_centers,previous_logs=initial.tolist(),[-7.0]*matrix.shape[1]
    minimum_reached_vertex=Counter()
    verified_slopes=0
    with trace.open() as stream:
        for number,line in enumerate(stream,1):
            event=json.loads(line)
            assert event["update"]==number
            assert event["centers_before"]==previous_centers and event["logs_before"]==previous_logs
            assert len(event["batch_indices"])==32 and len(set(event["batch_indices"]))==32
            assert event["centers_after"]==initial.tolist()
            assert math.isclose(event["point_loss"],baseline_loss,rel_tol=0,abs_tol=1e-16)
            assert all(-E<=value<=0 for value in event["logs_after"])
            rows=[]
            for i in event["batch_indices"]:
                cols=matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]].tolist()
                pred=sum(previous_centers[j] for j in cols)%Q
                rows.append((cols,int(targets[i]),pred))
            present=set(j for cols,_,_ in rows for j in cols)
            assert sorted(present)==event["batch_supported_columns"]
            counts["tied_updates"]+=int(bool(event["tie_rows"]))
            counts["attaining_tie_rows"]+=len(event["tie_rows"])
            counts["coupled_groups"]+=sum(len(group)>1 for group in event["groups"])
            assert len(event["groups"])==len(event["selected"])
            assert all(sum(j in group for j in event["selected"])==1 for group in event["groups"])
            coordinates={c["column"]:c for c in event["coordinates"]}
            motions={m["column"]:m for m in event["motions"]}
            assert set(coordinates)==set(event["selected"])
            for j,c in coordinates.items():
                counts["coordinate_selections"]+=1
                supported=j in present
                counts["batch_supported_selections"]+=int(supported)
                counts["batch_absent_selections"]+=int(not supported)
                available=[c["up_allowed"]]+[c["down_allowed"]]*(len(c["speeds"])-1)
                feasible_negative=any(a and s<0 for a,s in zip(available,c["used_slopes"],strict=True))
                zero=max(c["speeds"])==0
                counts["negative_feasible_slope_selections"]+=int(feasible_negative)
                counts["negative_feasible_slopes_suppressed"]+=int(feasible_negative and zero)
                counts["zero_speed_selections"]+=int(zero)
                counts["terminal_selected_coordinates"]+=int(c["vertex_depth"]==E)
                if c["vertex_depth"] is not None:
                    minimum_reached_vertex[c["vertex_depth"]]+=1
                counts["negative_children_blocked_by_history"]+=sum(
                    int(available[k] and c["used_slopes"][k]<0 and c["moment"][k] is None)
                    for k in range(1,len(c["speeds"])))
                assert (j in motions)==(not zero)
                if supported:
                    independently=raw_slopes(rows,previous_centers,previous_logs,j,c["vertex_depth"])
                    assert independently==c["raw_slopes"]
                    verified_slopes+=1
                else:
                    assert all(value==0 for value in c["raw_slopes"])
            # A feasible joint retained-center downward germ may be hidden by
            # attaining ties. This is only a relaxation derivative, not a point escape.
            for group in event["groups"]:
                if len(group)<=1 or previous_logs[group[0]]<=-E:
                    continue
                radius=P**previous_logs[group[0]]
                assert all(previous_logs[j]==previous_logs[group[0]] for j in group)
                members=set(group)
                joint=0.0
                individual={j:0.0 for j in group}
                for cols,target,predicted in rows:
                    active=[j for j in cols if previous_logs[j]==max(previous_logs[k] for k in cols)]
                    if not active or not set(active).issubset(members):
                        continue
                    slope=-.5 if norm(target-predicted)<radius else .5
                    joint+=slope
                    if len(active)==1:
                        individual[active[0]]+=slope
                counts["feasible_coupled_downward_germs_checked"]+=1
                counts["negative_joint_downward_germs"]+=int(joint<0)
                counts["negative_joint_descent_hidden_from_all_individual_down_germs"]+=int(joint<0 and min(individual.values())>=0)
            bounds=[]
            for j,m in motions.items():
                c=coordinates[j]
                direction=m["direction"]
                assert m["speed"]==max(c["speeds"])==c["speeds"][direction]
                bound=m["boundary_distance"]/m["speed"]
                bounds.append(bound)
                hit=event["applied_time"]==bound
                assert hit==m["boundary_hit"]
                assert m["requested_distance"]==event["requested_time"]*m["speed"]
                assert m["applied_distance"]==event["applied_time"]*m["speed"]
                if hit:
                    assert event["logs_after"][j]==m["boundary_log"]
                else:
                    radius=P**previous_logs[j]+(m["applied_distance"] if direction==0 else -m["applied_distance"])
                    assert event["logs_after"][j]==math.log(radius,P)
                changed=previous_logs[j]!=event["logs_after"][j]
                assert changed==m["log_changed"] and not m["center_changed"]
                assert m["numerical_noop"]==(not changed)
                counts["motion_proposals"]+=1
                counts["strict_radius_changes"]+=int(changed)
                counts["numerical_noops"]+=int(not changed)
                counts["boundary_hits"]+=int(hit)
                counts["motions_with_batch_absent_column"]+=int(j not in present)
                counts["motions_against_current_minibatch_slope"]+=int(c["used_slopes"][direction]>=0)
                counts["child_proposals"]+=int(direction>0 and c["vertex_depth"] is not None and c["vertex_depth"]<E)
                if hit:
                    limiting_columns[j]+=1
                    first_limiting_step.setdefault(j,number)
                distances.append(m["applied_distance"])
            assert event["applied_time"]==min([event["requested_time"]]+bounds)
            assert event["clipping_coordinates"]==[j for j,m in motions.items() if m["boundary_hit"]]
            assert event["clipped"]==(bool(motions) and event["applied_time"]<event["requested_time"])
            counts["clipped_updates"]+=int(event["clipped"])
            time_values.append(event["applied_time"])
            ratios.append(event["applied_time"]/event["requested_time"])
            shallowest=[max(a,b) for a,b in zip(shallowest,event["logs_after"],strict=True)]
            previous_centers,previous_logs=event["centers_after"],event["logs_after"]
    assert number==result["completed_updates"]==result["fit"]["updates"]
    assert counts["motion_proposals"]==result["fit"]["moves"]
    assert counts["coordinate_selections"]==result["fit"]["coordinate_updates"]
    assert result["fit"]["digit_changes"]==0 and result["coefficients_unchanged"]
    assert fingerprint(previous_centers)==result["endpoint_coefficient_sha256"]
    assert original_fingerprint(conn)==provenance["original_row_fingerprint"]
    conn.rollback()
    conn.close()
    out=dict(status="passed",checked_updates=number,counts=dict(counts),
        independently_verified_supported_slopes=verified_slopes,
        exact_training_loss_units=baseline_units,training_raw_point_loss=baseline_loss,
        coefficient_changes=0,original_row_unchanged=True,
        checkpoint_comparisons=result["checkpoint_comparisons"],
        selected_vertex_depth_counts=dict(sorted(minimum_reached_vertex.items())),
        maximum_log_radius_ever=max(shallowest),
        coordinates_ever_above_floor=sum(value>-7 for value in shallowest),
        coordinates_ever_at_or_above_depth_6=sum(value>=-6 for value in shallowest),
        coordinates_ever_at_or_above_depth_5=sum(value>=-5 for value in shallowest),
        applied_time_distribution=quantiles(time_values),applied_to_requested_time_ratio=quantiles(ratios),
        positive_motion_distance_distribution=quantiles([value for value in distances if value>0]),
        limiting_coordinate_counts=dict(limiting_columns),
        limiting_coordinate_first_updates=first_limiting_step,
        scope="Read-only training trace audit; no held-out predictions or fitting",
        interpretation="Counts identify mechanisms in this prefix. No counterfactual ablation establishes a dominant cause of the original full-run result.")
    (ROOT/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__":
    audit()
