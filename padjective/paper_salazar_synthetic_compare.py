"""Read-only comparison of the two fixed synthetic rosters and saved traces."""
import json
from pathlib import Path
import platform


def compare():
    assert platform.node()=="raksasa"
    roots=[Path("artifacts/salazar-synthetic-20261002"),
           Path("artifacts/salazar-synthetic-p71-20261002")]
    runs=[json.loads((root/"results.json").read_text()) for root in roots]
    assert all(run["status"]=="completed" for run in runs)
    variable={"p","precision","initial_log_radius","extraction"}
    fixed=[{k:v for k,v in run["configuration"].items() if k not in variable} for run in runs]
    assert fixed[0]==fixed[1]
    adapter_hashes=[run["source_sha256"]["padjective/salazar_affine.py"] for run in runs]
    assert adapter_hashes[0]==adapter_hashes[1]
    assert runs[0]["trials"].keys()==runs[1]["trials"].keys()
    output=dict(status="passed",identical_optimizer_sha256=adapter_hashes[0],
        identical_fixed_configuration=fixed[0],matched_batch_updates=0,trials={},
        scope="Stored synthetic trajectories only; no fitting, product data or selection")
    for key in runs[0]["trials"]:
        summaries=[run["trials"][key] for run in runs]
        traces=[[json.loads(line) for line in (root/summary["trace_file"]).read_text().splitlines()]
                for root,summary in zip(roots,summaries,strict=True)]
        assert len(traces[0])==len(traces[1])==summaries[0]["requested_updates"]
        assert all(a["batch_indices"]==b["batch_indices"] for a,b in zip(*traces,strict=True))
        output["matched_batch_updates"]+=len(traces[0])
        profiles=[]
        for run,summary,trace in zip(runs,summaries,traces,strict=True):
            profile=dict(p=run["configuration"]["p"],precision=run["configuration"]["precision"],
                first_improvement_update=summary["first_improvement_update"],
                first_zero_update=summary["first_zero_update"],endpoint_loss=summary["endpoint_loss"],
                best_loss=summary["best_units"]/(5*run["configuration"]["p"]**run["configuration"]["precision"]),
                endpoint_centers=summary["endpoint_centers"],
                maximum_log_radius_seen=max(v for e in trace for v in e["logs_after"]),
                proposals=summary["moves"],digit_changes=summary["digit_changes"],
                tied_updates=summary["tied_updates"],clipped_updates=summary["clipped_updates"],
                numerical_noops=summary["numerical_noops"],root_selections=0,
                root_child_proposals=0,root_changed_center_proposals=0,
                initial_raw_slopes=[c["raw_slopes"] for c in trace[0]["coordinates"]],
                initial_used_slopes=[c["used_slopes"] for c in trace[0]["coordinates"]])
            for e in trace:
                coordinates={c["column"]:c for c in e["coordinates"]}
                profile["root_selections"]+=sum(c["vertex_depth"]==0 for c in coordinates.values())
                for m in e["motions"]:
                    at_root=coordinates[m["column"]]["vertex_depth"]==0 and m["direction"]>0
                    profile["root_child_proposals"]+=int(at_root)
                    profile["root_changed_center_proposals"]+=int(at_root and m["center_changed"])
            profiles.append(profile)
        output["trials"][key]=profiles
    assert output["matched_batch_updates"]==9500
    (roots[1]/"comparison.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output))


if __name__=="__main__":
    compare()
