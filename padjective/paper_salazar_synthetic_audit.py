"""Read-only audit of stored synthetic trajectory states and mechanisms."""
import hashlib
import argparse
import json
import math
from pathlib import Path
import platform

from .paper_salazar_synthetic import units


def audit(root=Path("artifacts/salazar-synthetic-20261002")):
    assert platform.node()=="raksasa"
    results=json.loads((root/"results.json").read_text())
    assert results["status"]=="completed"
    p,precision=results["configuration"]["p"],results["configuration"]["precision"]
    q=p**precision
    score=lambda weights:units(weights,p,precision)
    out=dict(status="passed",checked_updates=0,checked_motion_proposals=0,
             p=p,precision=precision,trials={},scope="Stored five-row synthetic traces only; no fitting")
    for key,summary in results["trials"].items():
        path=root/summary["trace_file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest()==summary["trace_sha256"]
        previous_centers,previous_logs=[0,0],[-float(precision),-float(precision)]
        count=0
        d=dict(zero_speed_selections=0,negative_feasible_slopes_suppressed=0,
            negative_child_slopes_blocked_by_vertex_history=0,
            motions_against_current_minibatch_slope=0,
            proposals=0,strict_radius_changes=0,center_changes=0,numerical_noops=0,
            boundary_hits=0,clipped_updates=0,post_switch_nonzero_losses=0,
            maximum_log_radius_seen=-float(precision),root_vertex_selections=0,
            tied_updates=0,coupled_selected_updates=0,root_child_proposals=0,
            root_changed_center_proposals=0,child_proposals=0,
            positive_applied_time_min=None,positive_speed_min=None)
        for line in path.read_text().splitlines():
            event=json.loads(line)
            count+=1
            assert event["update"]==count
            assert event["centers_before"]==previous_centers and event["logs_before"]==previous_logs
            assert score(event["centers_after"])==event["independent_point_units"]
            assert all(0<=v<q for v in event["centers_after"])
            assert all(-precision<=v<=0 for v in event["logs_after"])
            d["tied_updates"]+=int(bool(event["tie_rows"]))
            d["coupled_selected_updates"]+=int(any(len(group)>1 for group in event["groups"]))
            assert len(event["selected"])==len(event["groups"])
            assert all(sum(j in group for j in event["selected"])==1 for group in event["groups"])
            d["maximum_log_radius_seen"]=max(d["maximum_log_radius_seen"],*event["logs_after"])
            ratios=[]
            motions={m["column"]:m for m in event["motions"]}
            coordinates={c["column"]:c for c in event["coordinates"]}
            assert set(coordinates)==set(event["selected"])
            for j,c in coordinates.items():
                assert c["clock"]>=1
                d["root_vertex_selections"]+=int(c["vertex_depth"]==0)
                available=[c["up_allowed"]]+[c["down_allowed"]]*(len(c["speeds"])-1)
                negative=any(allowed and slope<0 for allowed,slope in zip(available,c["used_slopes"],strict=True))
                no_speed=max(c["speeds"])==0
                d["zero_speed_selections"]+=int(no_speed)
                d["negative_feasible_slopes_suppressed"]+=int(negative and no_speed)
                d["negative_child_slopes_blocked_by_vertex_history"]+=sum(
                    int(available[k] and c["used_slopes"][k]<0 and c["moment"][k] is None)
                    for k in range(1,len(c["speeds"])))
                assert (j in motions)==(not no_speed)
            for j,m in motions.items():
                d["proposals"]+=1
                c=coordinates[j]
                direction=m["direction"]
                child=direction>0 and c["vertex_depth"] is not None and c["vertex_depth"]<precision
                d["child_proposals"]+=int(child)
                d["root_child_proposals"]+=int(child and c["vertex_depth"]==0)
                d["root_changed_center_proposals"]+=int(child and c["vertex_depth"]==0 and m["center_changed"])
                d["positive_speed_min"]=min(d["positive_speed_min"] or m["speed"],m["speed"])
                if event["applied_time"]>0:
                    d["positive_applied_time_min"]=min(d["positive_applied_time_min"] or event["applied_time"],event["applied_time"])
                assert m["speed"]==max(c["speeds"])==c["speeds"][direction]
                assert m["requested_distance"]==event["requested_time"]*m["speed"]
                assert m["applied_distance"]==event["applied_time"]*m["speed"]
                ratios.append(m["boundary_distance"]/m["speed"])
                assert event["applied_time"]<=ratios[-1]
                d["motions_against_current_minibatch_slope"]+=int(c["used_slopes"][direction]>=0)
                hit=event["applied_time"]==ratios[-1]
                assert hit==m["boundary_hit"]
                before_log=event["logs_before"][j]
                if hit:
                    assert event["logs_after"][j]==m["boundary_log"]
                else:
                    r=p**before_log+(m["applied_distance"] if direction==0 else -m["applied_distance"])
                    assert event["logs_after"][j]==math.log(r,p)
                log_change=event["logs_before"][j]!=event["logs_after"][j]
                center_change=event["centers_before"][j]!=event["centers_after"][j]
                assert log_change==m["log_changed"] and center_change==m["center_changed"]
                assert m["numerical_noop"]==(not log_change and not center_change)
                d["strict_radius_changes"]+=int(log_change)
                d["center_changes"]+=int(center_change)
                d["numerical_noops"]+=int(m["numerical_noop"])
                d["boundary_hits"]+=int(hit)
            assert event["applied_time"]==min([event["requested_time"]]+ratios)
            assert event["clipping_coordinates"]==[j for j,m in motions.items() if m["boundary_hit"]]
            assert event["clipped"]==(bool(motions) and event["applied_time"]<event["requested_time"])
            d["clipped_updates"]+=int(event["clipped"])
            if key.startswith("controlled") and count>200:
                d["post_switch_nonzero_losses"]+=int(event["independent_point_units"]!=0)
            previous_centers,previous_logs=event["centers_after"],event["logs_after"]
        assert count==summary["updates"]==summary["diagnostic_rows"]
        assert previous_centers==summary["endpoint_centers"] and previous_logs==summary["endpoint_log_radii"]
        assert score(previous_centers)==summary["endpoint_units"]
        for audit_key,summary_key in [("strict_radius_changes","strict_radius_changes"),
            ("center_changes","digit_changes"),("numerical_noops","numerical_noops"),
            ("boundary_hits","boundary_hits"),("clipped_updates","clipped_updates"),("proposals","moves")]:
            assert d[audit_key]==summary[summary_key]
        if key.startswith("full"):
            assert d["proposals"]==0 and previous_centers==[0,0]
        if summary["endpoint_units"]==0:
            assert previous_centers==[1,q-1]
        if key.startswith("controlled") and summary["controlled_switch_units"]==0:
            assert d["post_switch_nonzero_losses"]==0
        out["checked_updates"]+=count
        out["checked_motion_proposals"]+=d["proposals"]
        out["trials"][key]=d
    assert out["checked_updates"]==sum(t["updates"] for t in results["trials"].values())<=9500
    (root/"audit.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(out))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=["p3","p71"],default="p3")
    args=parser.parse_args()
    audit(Path("artifacts/salazar-synthetic-20261002" if args.profile=="p3" else "artifacts/salazar-synthetic-p71-20261002"))
