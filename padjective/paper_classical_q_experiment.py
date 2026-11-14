"""Recover paper classical predictions from PostgreSQL and rescore depth weights.

Fits retain their original classification objectives. Private predictions stay
in PostgreSQL; only fold aggregates and provenance are exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from statistics import mean, stdev
import time
import warnings

import numpy as np
from psycopg.types.json import Jsonb
from threadpoolctl import threadpool_info, threadpool_limits

from . import benchmark_runtime as runtime, db
from .paper_low_q_experiment import Q_VALUES, fingerprint, score
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset


EXPERIMENT = "classical-q-2026-11-02"
MODELS = ("dummy", "ulr", "dt", "levelwise", "unn")


def make_estimator(key):
    if key == "dummy":
        return runtime.DummyClassifier(strategy="most_frequent")
    if key == "ulr":
        return runtime.LogisticRegression(
            penalty="l1", solver="saga", C=1.0,
            max_iter=runtime.DEFAULT_ULR_MAX_ITER, n_jobs=-1, random_state=42,
        )
    if key == "dt":
        return runtime.DecisionTreeClassifier(class_weight="balanced", random_state=42)
    if key == "unn":
        # The main paper uses the largest tested width, rather than the runtime
        # module's older width-12 default. See the archived neural fold extract.
        return runtime.MLPClassifier(
            hidden_layer_sizes=(2000,), activation="relu", alpha=1e-4,
            batch_size=256, max_iter=10_000, random_state=42,
        )
    raise ValueError(f"No flat classifier for {key!r}")


def reference_check(scores, reference):
    checks = {
        "q71_loss_difference": scores["losses"]["71"] - reference["padic_loss_mean"],
        "exact_accuracy_difference": scores["exact_accuracy"] - reference["exact_accuracy"],
    }
    if "num_test_samples" in reference:
        checks["test_size_difference"] = scores["n"] - reference["num_test_samples"]
    checks["matches"] = all(abs(value) < 1e-12 for value in checks.values())
    return checks


def environment(reference):
    expected = reference["environment"]
    actual = dict(
        python_version=platform.python_version(),
        versions={k: importlib.metadata.version(k) for k in expected["versions"]},
        numerical_libraries=[{k: v for k, v in item.items() if k != "filepath"}
                             for item in threadpool_info()],
    )
    if actual["python_version"] != expected["python_version"]:
        raise ValueError("Python differs from the frozen paper environment")
    if actual["versions"] != expected["versions"]:
        raise ValueError("Package versions differ from the frozen paper environment")
    for lib in expected["numerical_libraries"]:
        if not any(all(a.get(k) == v for k, v in lib.items())
                   for a in actual["numerical_libraries"]):
            raise ValueError(f"Numerical library differs from the frozen environment: {lib}")
    return actual


def ensure_storage(conn):
    db.ensure_table(conn, "padjective", "paper_classical_q_runs", [
        "experiment_key TEXT NOT NULL", "model_key TEXT NOT NULL", "cv_fold INTEGER NOT NULL",
        "status TEXT NOT NULL", "evidence JSONB NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        "PRIMARY KEY (experiment_key, model_key, cv_fold)",
    ])
    db.ensure_table(conn, "padjective", "paper_low_q_reports", [
        "experiment_key TEXT PRIMARY KEY", "report JSONB NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ])
    conn.commit()


def fit_predict(dataset, paths, key, fold):
    fold_ids = np.asarray([r.cv_fold for r in dataset.records])
    train, test = fold_ids != fold, fold_ids == fold
    x_train, x_test = dataset.features[train], dataset.features[test]
    y_train, y_test = dataset.labels[train], dataset.labels[test]
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if key == "levelwise":
            models = runtime.train_levelwise_models(x_train, y_train, paths)
            predictions = [runtime.predict_levelwise_taxonomy(x_test[i], models)[0]
                           for i in range(x_test.shape[0])]
            details = dict(classifiers=sum(m.classifier is not None for m in models.values()),
                           settings=dict(max_iter=1000, solver="lbfgs", class_weight="balanced"))
        else:
            model = make_estimator(key)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test).tolist()
            details = dict(settings=model.get_params(),
                           iterations=np.asarray(getattr(model, "n_iter_", [])).tolist())
    encoded = dataset.encoded_by_taxonomy
    actual_values = [encoded[str(v)] for v in y_test]
    predicted_values = [encoded[str(v)] for v in predictions]
    return dict(
        scores=score(actual_values, predicted_values),
        # These arrays are deliberately excluded from the exported report.
        product_keys=[r.product_key for r, held_out in zip(dataset.records, test) if held_out],
        actual_taxonomy_ids=y_test.tolist(), predicted_taxonomy_ids=predictions,
        predicted_values=predicted_values, num_train=len(y_train),
        wall_seconds=time.monotonic() - start,
        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
        **details,
    )


def collect(conn, reference, provenance, output):
    with conn.cursor() as cur:
        cur.execute("""SELECT model_key, cv_fold, status, evidence
                       FROM padjective.paper_classical_q_runs
                       WHERE experiment_key=%s ORDER BY model_key, cv_fold""", (EXPERIMENT,))
        rows = cur.fetchall()
    if {(m, f) for m, f, status, _ in rows if status == "complete"} != {
        (m, f) for m in MODELS for f in range(5)
    }:
        raise ValueError("All 25 reference-matched fits are required before collection")
    series = []
    for key in MODELS:
        evidence = [e for m, _, _, e in rows if m == key]
        if any(e["provenance"] != provenance for e in evidence):
            raise ValueError("Cached fit provenance differs")
        folds = [dict(cv_fold=f, **e["scores"], reference_check=e["reference_check"],
                      wall_seconds=e["wall_seconds"], warnings=e["warnings"])
                 for m, f, _, e in rows if m == key]
        series.append(dict(
            configuration=key, training_q="classification_objective_unchanged", folds=folds,
            mean_losses={q: mean(f["losses"][q] for f in folds) for q in Q_VALUES},
            fold_sd={q: stdev(f["losses"][q] for f in folds) for q in Q_VALUES},
            exact_accuracy=mean(f["exact_accuracy"] for f in folds),
            root_accuracy=mean(f["root_accuracy"] for f in folds),
        ))
    report = dict(experiment_key=EXPERIMENT, created_at=datetime.now(timezone.utc).isoformat(),
                  snapshot_id=str(PAPER_SNAPSHOT), snapshot_digest=SNAPSHOT_DIGEST,
                  products=sum(f["n"] for f in series[0]["folds"]), prime=71,
                  q_values=Q_VALUES, provenance=provenance,
                  protocol="Original classification objectives; refits recover matching predictions; "
                           "equal-weight fold means and sample fold SDs; Q changes only evaluation",
                  reference_sha256=reference["source_sha256"], series=series)
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
                       VALUES (%s,%s) ON CONFLICT(experiment_key)
                       DO UPDATE SET report=EXCLUDED.report,updated_at=NOW()""",
                    (EXPERIMENT, Jsonb(report)))
    conn.commit()
    output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--collect", type=Path)
    args = parser.parse_args()
    keys, folds = args.models.split(","), [int(f) for f in args.folds.split(",")]
    if not set(keys) <= set(MODELS) or not set(folds) <= set(range(5)):
        parser.error("Unknown models or folds")
    reference = json.loads(args.reference.read_text())
    threadpool_limits(limits=reference["environment"]["numerical_threads"])
    provenance = dict(source_commit=args.source_commit, snapshot_digest=SNAPSHOT_DIGEST,
                      reference_sha256=fingerprint(reference), environment=environment(reference),
                      implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      runtime_sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest())
    with db.get_connection() as conn:
        ensure_storage(conn)
        if args.collect:
            collect(conn, reference, provenance, args.collect)
            return
        dataset = _load_paper_dataset(conn, snapshot_ref=str(PAPER_SNAPSHOT), schema="padjective")
        if _snapshot_digest(dataset) != SNAPSHOT_DIGEST or dataset.features.shape != (6693, 2542):
            raise ValueError("Frozen paper snapshot differs")
        if dataset.features.dtype != np.float32 or dataset.prime_base != 71:
            raise ValueError("Paper matrix dtype or encoding differs")
        with conn.cursor() as cur:
            cur.execute("""SELECT DISTINCT taxonomy_id,taxonomy_path
                           FROM padjective.product_taxonomy_bench_products
                           WHERE snapshot_id=%s""", (PAPER_SNAPSHOT,))
            paths = {str(k): runtime.parse_taxonomy_path(v) for k, v in cur.fetchall()}
        conn.commit()
        for key in keys:
            for fold in folds:
                with conn.cursor() as cur:
                    cur.execute("""SELECT status,evidence FROM padjective.paper_classical_q_runs
                                   WHERE experiment_key=%s AND model_key=%s AND cv_fold=%s""",
                                (EXPERIMENT, key, fold))
                    cached = cur.fetchone()
                conn.commit()
                if cached and cached[0] == "complete":
                    if cached[1]["provenance"] != provenance:
                        raise ValueError("Cached fit provenance differs")
                    print(json.dumps(dict(model=key, fold=fold, status="cached")), flush=True)
                    continue
                print(json.dumps(dict(model=key, fold=fold, status="fitting")), flush=True)
                evidence = fit_predict(dataset, paths, key, fold)
                ref = next(r for r in reference["references"][key] if r["cv_fold"] == fold)
                evidence["reference_check"] = reference_check(evidence["scores"], ref)
                evidence["provenance"] = provenance
                status = "complete" if evidence["reference_check"]["matches"] else "reference_mismatch"
                with conn.cursor() as cur:
                    cur.execute("""INSERT INTO padjective.paper_classical_q_runs
                                   (experiment_key,model_key,cv_fold,status,evidence)
                                   VALUES (%s,%s,%s,%s,%s)
                                   ON CONFLICT(experiment_key,model_key,cv_fold)
                                   DO UPDATE SET status=EXCLUDED.status,evidence=EXCLUDED.evidence,
                                                 updated_at=NOW()""",
                                (EXPERIMENT, key, fold, status, Jsonb(evidence)))
                conn.commit()
                print(json.dumps(dict(model=key, fold=fold, status=status,
                                      seconds=evidence["wall_seconds"], scores=evidence["scores"],
                                      reference_check=evidence["reference_check"])), flush=True)
                if status != "complete":
                    raise ValueError("Recovered predictions fail the archived Q=71/accuracy checks")


if __name__ == "__main__":
    main()
