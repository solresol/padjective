"""Compare the saved first 100 updates; no data query, fitting or selection."""
from itertools import islice
import json
import platform
from pathlib import Path

from .paper_salazar_coarse import ROOT
from .paper_salazar_replay_audit import quantiles


def compare():
    assert platform.node()=="raksasa"
    fine=Path("artifacts/salazar-warm-adam-replay-20261002/trace.jsonl")
    coarse=ROOT/"trace.jsonl"
    profile=[dict(proposals=0,center_changes=0,numerical_noops=0,clipped_updates=0,
        maximum_log_radius=-7.0 if i==0 else -1.0) for i in range(2)]
    ratios=[[],[]]
    with fine.open() as a,coarse.open() as b:
        for number,lines in enumerate(zip(a,islice(b,100),strict=True),1):
            events=[json.loads(line) for line in lines]
            assert events[0]["batch_indices"]==events[1]["batch_indices"]
            assert all(event["update"]==number for event in events)
            for i,event in enumerate(events):
                profile[i]["proposals"]+=len(event["motions"])
                profile[i]["center_changes"]+=sum(m["center_changed"] for m in event["motions"])
                profile[i]["numerical_noops"]+=sum(m["numerical_noop"] for m in event["motions"])
                profile[i]["clipped_updates"]+=int(event["clipped"])
                profile[i]["maximum_log_radius"]=max(profile[i]["maximum_log_radius"],*event["logs_after"])
                profile[i]["endpoint_point_loss"]=event["point_loss"]
                ratios[i].append(event["applied_time"]/event["requested_time"])
    assert number==100
    for i in range(2):
        profile[i]["applied_to_requested_time_ratio"]=quantiles(ratios[i])
    result=dict(status="passed",matched_batch_updates=100,profiles={"fine":profile[0],"coarse":profile[1]},
        scope="Matched first-100-update trace windows; diagnostic comparison only, no checkpoint selection")
    (ROOT/"prefix-comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    compare()
