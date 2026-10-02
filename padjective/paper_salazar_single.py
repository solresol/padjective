"""Frozen fold-0/seed-42 affine continuation feasibility experiment on raksasa."""
from __future__ import annotations

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
from .metrics import shared_prefix_depth_encoded
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST, model_evidence
from .paper_revision_experiments import _load_paper_dataset
from .randomised_linear import coordinate_certificate, fit_coordinates, linear_predictions
from .salazar_affine import fit_tree


CONFIG = dict(fold=0, seed=42, p=71, precision=7,
              exact_max_sweeps=100, exact_seconds=300,
              updates=1000, seconds_per_tree_fit=120, batch_size=32,
              kappa=1.0, beta=0.9, beta2=0.999, epsilon=1e-8,
              warm_initial_log_radius=-7, cold_initial_log_radius=0,
              variants=["exact_coordinate", "warm_momentum", "warm_adam", "cold_adam"],
              extraction="retained center modulo p^7; regression Section 5",
              mask="all training-supported features; unsupported coefficients frozen zero",
              selection="fixed endpoint, no held-out or validation tuning",
              grouping="conservative attaining-term connected components, no cancellation splitting",
              constraints="Z_p disks with radius in [p^-7,1]; no fractional coefficients",
              official_source="53ffd9e79890e76b82ef2dc3c7230984053f9801")


def fingerprint(values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def encode_logs(logs):
    return [float(v) if np.isfinite(v) else None for v in logs]


def independent_predictions(matrix, weights, modulus):
    return [sum(int(weights[j]) for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]) % modulus
            for i in range(matrix.shape[0])]


def full_metrics(weights, x_train, x_test, y_train, y_test, depths_train, depths_test):
    evidence, metrics = model_evidence(weights, x_train, x_test, y_train, y_test,
                                      p=71, precision=7, pilot=False, completed=True)
    for label, matrix, y, depths in [("training", x_train, y_train, depths_train),
                                    ("held_out", x_test, y_test, depths_test)]:
        raw = independent_predictions(matrix, weights, 71**7)
        assert raw == evidence["training_raw_predictions" if label=="training" else "held_out_raw_predictions"]
        for suffix, pred in [("_raw", raw), ("", evidence["training_predictions" if label=="training" else "predictions"])]:
            metrics[label+suffix]["mean_shared_prefix_depth"] = float(np.mean([
                shared_prefix_depth_encoded(int(a),int(b),71,true_depth=int(d))
                for a,b,d in zip(y,pred,depths,strict=True)]))
    active = np.asarray(x_test @ (weights != 0)).reshape(-1)
    metrics["prediction_active_support"] = dict(mean=float(active.mean()),
        median=float(np.median(active)), p95=float(np.quantile(active,0.95)), maximum=int(active.max()))
    metrics["coefficient_sha256"] = fingerprint(weights.tolist())
    metrics["raw_prediction_sha256"] = fingerprint(evidence["held_out_raw_predictions"])
    metrics["prediction_sha256"] = fingerprint(evidence["predictions"])
    # One-member valid-path projection is deliberately excluded from this
    # raw/default paired experiment; there is no added decoder or ensemble.
    return evidence, metrics


def run(output=Path("artifacts/salazar-single-results.json")):
    if platform.node() != "raksasa":
        raise RuntimeError("This experiment is authorized on raksasa only")
    run_id = str(uuid.uuid4())
    config = dict(CONFIG, hostname=platform.node(), python=platform.python_version(),
        numpy=np.__version__, source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    config["source_sha256"] = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in ["padjective/salazar_affine.py", "padjective/paper_salazar_single.py"]}
    conn = db.get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(71357912)")
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute("""CREATE TABLE IF NOT EXISTS padjective.paper_salazar_single_runs (
            run_id UUID PRIMARY KEY, snapshot_id UUID NOT NULL,
            configuration JSONB NOT NULL, status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), finished_at TIMESTAMPTZ,
            metrics JSONB, evidence JSONB
        ) TABLESPACE pg_default""")
        cur.execute("""INSERT INTO padjective.paper_salazar_single_runs
            (run_id,snapshot_id,configuration,status) VALUES (%s,%s,%s,'running')""",
            (run_id,PAPER_SNAPSHOT,Jsonb(config)))
    conn.commit()
    evidence, metrics = {"models":{}, "fit":{}}, {}
    started = time.monotonic()
    def emit(event, **values):
        print(json.dumps(dict(event=event,run_id=run_id,**values)),flush=True)
    def save(status="running"):
        with conn.cursor() as cur:
            cur.execute("""UPDATE padjective.paper_salazar_single_runs SET
                status=%s, metrics=%s,evidence=%s WHERE run_id=%s""",
                (status,Jsonb(metrics),Jsonb(evidence),run_id))
        conn.commit()
    emit("started", configuration=config)
    try:
        dataset = _load_paper_dataset(conn,snapshot_ref=PAPER_SNAPSHOT,schema="padjective")
        digest = _snapshot_digest(dataset)
        assert digest == SNAPSHOT_DIGEST and dataset.prime_base == 71
        folds = np.array([r.cv_fold for r in dataset.records])
        train = folds != 0
        x_train = dataset.features[train].astype(np.int64).tocsr()
        x_test = dataset.features[~train].astype(np.int64).tocsr()
        y = np.array([r.encoded_path for r in dataset.records],dtype=np.int64)
        depths = np.array([r.taxonomy_depth for r in dataset.records])
        y_train, y_test = y[train],y[~train]
        mask = np.asarray(x_train.sum(axis=0)).reshape(-1)>0
        evidence.update(snapshot_digest=digest,n_train=int(train.sum()),n_test=int((~train).sum()),
            n_features=x_train.shape[1],n_supported=int(mask.sum()), training_nnz=int(x_train.nnz),
            feature_names=list(dataset.feature_names),
            feature_names_sha256=fingerprint(list(dataset.feature_names)), mask_sha256=fingerprint(mask.tolist()))
        emit("data_verified", **{k:v for k,v in evidence.items() if k not in ("models","fit","feature_names")})
        fit = fit_coordinates(x_train,y_train,p=71,precision=7,seed=42,max_sweeps=100,seconds=300,
            callback=lambda h: emit("exact_sweep",**h))
        certificate = coordinate_certificate(x_train,y_train,fit.coefficients,p=71,precision=7)
        assert fit.status == certificate["status"] == "coordinate_optimum"
        evidence["coordinate_certificate"] = certificate
        evidence["fit"]["exact_coordinate"] = {k:v for k,v in asdict(fit).items()
            if k not in ("coefficients","first_pass_coefficients","residuals")}
        evidence["models"]["exact_coordinate"] = {"coefficients":fit.coefficients.tolist()}
        # Verify same-seed historical member without selecting by its scores.
        with conn.cursor() as cur:
            cur.execute("""SELECT run_id,evidence->'models'->'coordinate'->'coefficients'
                FROM padjective.paper_randomised_method_runs WHERE method='linear_random'
                AND cv_fold=0 AND NOT pilot AND configuration->>'seed'='42'
                AND status='coordinate_optimum' ORDER BY started_at LIMIT 1""")
            previous = cur.fetchone()
        if previous:
            evidence["historical_member"] = dict(run_id=str(previous[0]),
                coefficients_identical=previous[1] == fit.coefficients.tolist())
            assert evidence["historical_member"]["coefficients_identical"]
        save()
        fits = {"exact_coordinate":fit.coefficients}
        for variant in CONFIG["variants"][1:]:
            warm = variant.startswith("warm")
            optimizer = variant.split("_")[1]
            initial = fit.coefficients if warm else np.zeros_like(fit.coefficients)
            def checkpoint(item, centers, logs):
                evidence["checkpoint"] = dict(variant=variant,**item,
                    coefficients=centers.tolist(),log_radii=encode_logs(logs))
                save()
                emit("tree_checkpoint",variant=variant,**item)
            tree = fit_tree(x_train,y_train,initial,p=71,precision=7,seed=42,
                optimizer=optimizer,initial_log_radius=-7 if warm else 0,
                updates=1000,seconds=120,batch_size=32,kappa=1,
                checkpoint=checkpoint)
            fits[variant] = tree.centers
            evidence["models"][variant] = dict(coefficients=tree.centers.tolist(),log_radii=encode_logs(tree.log_radii))
            evidence["fit"][variant] = {k:v for k,v in asdict(tree).items() if k not in ("centers","log_radii")}
            save()
            emit("tree_finished",variant=variant,**{k:v for k,v in evidence["fit"][variant].items() if k!="history"})
        # Freeze all endpoints before held-out prediction. No rate search,
        # restart, seed search, early stopping or best-checkpoint selection.
        emit("training_finished_held_out_evaluation_begins")
        for variant,weights in fits.items():
            before = time.monotonic()
            private, score = full_metrics(weights,x_train,x_test,y_train,y_test,depths[train],depths[~train])
            # Independent audit timing is not inference latency.
            score["prediction_and_audit_seconds"] = time.monotonic()-before
            times=[]
            for _ in range(5):
                before=time.perf_counter()
                linear_predictions(x_test,weights,71,7)
                times.append(time.perf_counter()-before)
            score["median_raw_prediction_seconds"] = float(np.median(times))
            score["fit_seconds"] = evidence["fit"][variant]["elapsed_seconds"]
            score["total_training_seconds"] = score["fit_seconds"]+(fit.elapsed_seconds if variant.startswith("warm") else 0)
            if variant != "exact_coordinate":
                score["sampled_training_examples"] = evidence["fit"][variant]["updates"]*32
                score["training_example_equivalents"] = score["sampled_training_examples"]/len(y_train)
            metrics[variant] = score
            evidence["models"][variant].update(private)
        evidence["elapsed_seconds"] = time.monotonic()-started
        save("completed")
        with conn.cursor() as cur:
            cur.execute("UPDATE padjective.paper_salazar_single_runs SET finished_at=NOW() WHERE run_id=%s",(run_id,))
        conn.commit()
        result = dict(run_id=run_id,status="completed",configuration=config,
            snapshot_id=PAPER_SNAPSHOT,snapshot_digest=digest,
            n_train=len(y_train),n_test=len(y_test),n_features=x_train.shape[1],n_supported=int(mask.sum()),
            historical_member=evidence.get("historical_member"), coordinate_certificate=certificate,
            metrics=metrics,fit=evidence["fit"],elapsed_seconds=evidence["elapsed_seconds"])
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+"\n")
        emit("finished",status="completed",metrics=metrics)
        return result
    except Exception as exc:
        conn.rollback()
        evidence["error"] = dict(type=type(exc).__name__,message=str(exc))
        # Preserve already-saved checkpoints if the new checkpoint itself
        # cannot be serialized. Error bookkeeping must not repeat that failure.
        evidence.pop("checkpoint",None)
        save("failed")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
