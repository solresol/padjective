"""Compare the paper's headline learners on the frozen September store folds."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
from statistics import mean, stdev
import time
import warnings

import numpy as np
from psycopg.types.json import Jsonb
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits

from . import benchmark_runtime as runtime, db, live_exact, live_subspace as live
from .paper_classical_q_experiment import make_estimator
from .paper_tree_ensemble_tradeoff import SEEDS

EXPERIMENT = "paper-store-held-out-headline-2026-11-08"
BATCH = "729fd099-6e5d-4e5d-85e2-907d5aff3f7b"
FOLD_SIZES = (5802, 6151, 6832, 6580, 6473)
FOREST_SIZES = (9, 81)
FOREST_JOBS = {f"forest-{seed}": seed for seed in SEEDS}
FITTED = ("unn", "ulr", "greedy", "levelwise", "dt", "dummy")
JOBS = [(f, key) for f in range(5) for key in (*FITTED[:2], *FOREST_JOBS, *FITTED[2:])]
_rows = None
_threads = None


def score(actual, predictions):
    disagreements = Counter()
    exact = 0
    for target, prediction in zip(actual, predictions, strict=True):
        delta = abs(int(target) - int(prediction))
        if delta == 0:
            exact += 1
            continue
        depth = 0
        while delta % live.P == 0:
            depth += 1
            delta //= live.P
        disagreements[depth] += 1
    n = len(actual)
    return dict(n=n, loss_q2=sum(count * 2.0**(-d) for d, count in disagreements.items()) / n,
                root_accuracy=1 - disagreements[0] / n, exact_accuracy=exact / n,
                first_disagreement_counts={str(d): count for d, count in sorted(disagreements.items())},
                exact_correct=exact, prediction_sha256=live.digest(predictions))


def forest_predictions(x_train, y_train, x_test, seed, sizes=FOREST_SIZES):
    """Stream the same sklearn tree seeds and probability sums as a full bank."""
    random = np.random.RandomState(seed)
    model = RandomForestClassifier(n_estimators=1, max_features="sqrt", bootstrap=True,
                                   n_jobs=1, random_state=random)
    labels = np.unique(y_train)
    probability_sum = np.zeros((x_test.shape[0], len(labels)))
    result = {}
    for member in range(1, max(sizes) + 1):
        model.fit(x_train, y_train)
        if not np.array_equal(model.classes_, labels):
            raise ValueError("Forest class columns differ across trees")
        probability_sum += model.predict_proba(x_test)
        if member in sizes:
            result[member] = labels[probability_sum.argmax(axis=1)]
    return result


def initialise(rows, threads):
    global _rows, _threads
    _rows, _threads = rows, threads


def fit_job(job):
    fold, key = job
    live.emit("fit_started", fold=fold, model=key)
    started = time.monotonic()
    c = live.context(_rows, fold)
    train, test = c["train"], c["test"]
    if key == "greedy":
        tags = set(c["vocabulary"])
        records = [runtime.ProductRecord(i, r["product_key"], [t for t in r["tags"] if t in tags],
                    r["target"], r["fold"], r["taxonomy_id"], len(r["path"].split(".")), ())
                   for i, r in enumerate(train + test)]
        fitted = runtime.umllr_run_fold(fold, records, [], live.P, tag_order_strategy="taxonomy_association")
        predictions = {key: [int(p.predicted_value) for p in fitted.predictions]}
        warnings_seen = []
        iterations = []
    else:
        x_train, x_test = (c[k].astype(np.float32) for k in ("x_train", "x_test"))
        y_train = np.array([r["taxonomy_id"] for r in train])
        codes = {r["taxonomy_id"]: r["target"] for r in train}
        iterations = []
        with threadpool_limits(limits=_threads), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if key in FOREST_JOBS:
                labels = {f"forest-{size}": values for size, values in
                          forest_predictions(x_train, y_train, x_test, FOREST_JOBS[key]).items()}
            elif key == "levelwise":
                paths = {r["taxonomy_id"]: tuple(r["path"].split(".")) for r in train}
                fitted = runtime.train_levelwise_models(x_train, y_train, paths)
                labels = {key: [runtime.predict_levelwise_taxonomy(x_test[i], fitted)[0]
                                for i in range(len(test))]}
            else:
                fitted = make_estimator(key)
                if key == "ulr":
                    fitted.set_params(n_jobs=1)
                fitted.fit(x_train, y_train)
                labels = {key: fitted.predict(x_test)}
                iterations = np.asarray(getattr(fitted, "n_iter_", [])).tolist()
        predictions = {name: [codes[label] for label in values] for name, values in labels.items()}
        warnings_seen = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    return dict(test_sha256=live.digest([r["product_key"] for r in test]),
                vocabulary_sha256=live.digest(c["vocabulary"]), eligible_tags=len(c["vocabulary"]),
                no_usable_tags=int(np.count_nonzero(c["x_test"].getnnz(axis=1) == 0)),
                predictions=predictions, scores={k: score(c["y_test"], v) for k, v in predictions.items()},
                elapsed_seconds=time.monotonic() - started, iterations=iterations, warnings=warnings_seen)


def load_archive(conn):
    cohort, meta, archive, predictions = conn.execute("""
        SELECT b.cohort_id,c.manifest,r.report,r.predictions
        FROM padjective.live_subspace_batches b
        JOIN padjective.live_subspace_cohorts c USING (cohort_id)
        JOIN padjective.live_subspace_reports r USING (batch_id)
        WHERE b.batch_id=%s AND b.status='validated'
    """, (BATCH,)).fetchone()
    rows = live.load_rows(conn, cohort)
    sizes = Counter(r["fold"] for r in rows)
    stores = {(r["store_hash"], r["fold"]) for r in rows}
    ids = {(r["taxonomy_id"], r["path"]) for r in rows}
    if (tuple(sizes[f] for f in range(5)) != FOLD_SIZES or len(rows) != sum(FOLD_SIZES)
            or len(stores) != len({store for store, _ in stores})
            or len(ids) != len({category for category, _ in ids})
            or any(r["label_source"] != "gold_llm" for r in rows)):
        raise ValueError("The store folds or reference labels differ from the frozen capture")
    if (archive["status"] != "validated_rolling_cv" or archive["manifest"]["members"] != 243
            or live.digest(predictions) != archive["predictions_sha256"]
            or set(predictions) != {r["product_key"] for r in rows}):
        raise ValueError("The archived ensemble predictions differ from the validated batch")
    return rows, dict(cohort_id=str(cohort), cohort_sha256=meta["content_sha256"],
                      captured_at=meta["captured_at"], coverage=meta["coverage"],
                      archive_prediction_sha256=archive["predictions_sha256"]), predictions


def collect(rows, predictions, saved, manifest):
    models = {k: [] for k in (*FITTED, "forest-9", "forest-81", "ensemble-75", "ensemble-all")}
    for fold, n in enumerate(FOLD_SIZES):
        c = live.context(rows, fold)
        test_hash = live.digest([r["product_key"] for r in c["test"]])
        vocab_hash = live.digest(c["vocabulary"])
        for key in (*FITTED, *FOREST_JOBS):
            item = saved[fold, key]
            if item["test_sha256"] != test_hash or item["vocabulary_sha256"] != vocab_hash:
                raise ValueError("Saved fit has different test rows or vocabulary")
            for label, predicted in item["predictions"].items():
                if score(c["y_test"], predicted) != item["scores"][label]:
                    raise ValueError("Saved predictions fail the score replay")
        for key in FITTED:
            evidence = saved[fold, key]
            models[key].append(dict(fold=fold, **evidence["scores"][key], iterations=evidence["iterations"],
                                    warnings=evidence["warnings"]))
        for size in FOREST_SIZES:
            key = f"forest-{size}"
            seeds = {str(seed): saved[fold, name]["scores"][key] for name, seed in FOREST_JOBS.items()}
            models[key].append(dict(fold=fold, n=n, seeds=seeds, **{
                metric: mean(s[metric] for s in seeds.values())
                for metric in ("loss_q2", "root_accuracy", "exact_accuracy")}))
        for fraction, key in (("0.75", "ensemble-75"), ("1.0", "ensemble-all")):
            pred = [predictions[r["product_key"]][fraction] for r in c["test"]]
            if not set(pred).issubset(set(map(int, c["y_train"]))):
                raise ValueError("An archived ensemble predicted a path outside the fitting labels")
            models[key].append(dict(fold=fold, **score(c["y_test"], pred)))
    result = [dict(model=key, folds=folds, loss_q2=mean(r["loss_q2"] for r in folds),
                   loss_q2_sd=stdev(r["loss_q2"] for r in folds),
                   root_accuracy=mean(r["root_accuracy"] for r in folds),
                   exact_accuracy=mean(r["exact_accuracy"] for r in folds))
              for key, folds in models.items()]
    return dict(manifest=manifest, status="validated", models=sorted(result, key=lambda r: r["loss_q2"]))


def run(source_commit, workers, threads):
    with db.get_connection() as conn:
        rows, provenance, predictions = load_archive(conn)
    manifest = dict(experiment_key=EXPERIMENT, archive_batch=BATCH, source_commit=source_commit,
                    fold_sizes=list(FOLD_SIZES), valuation_prime=live.P, evaluation_q=2,
                    native_fitting_q=live.P, numerical_threads=threads, forest_seeds=list(SEEDS),
                    python=platform.python_version(), packages={name: importlib.metadata.version(name)
                    for name in ("numpy", "scipy", "scikit-learn")}, **provenance)
    fingerprint = live.digest(manifest)
    with db.get_connection() as conn:
        if not conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()[0]:
            raise RuntimeError("This comparison is already running")
        db.ensure_table(conn, "padjective", "paper_store_held_out_runs", [
            "experiment_key TEXT NOT NULL", "fold INTEGER NOT NULL", "model TEXT NOT NULL",
            "evidence JSONB NOT NULL", "PRIMARY KEY (experiment_key, fold, model)"])
        cached = conn.execute("""SELECT fold,model,evidence FROM padjective.paper_store_held_out_runs
            WHERE experiment_key=%s""", (EXPERIMENT,)).fetchall()
        if any(item["manifest_sha256"] != fingerprint for _, _, item in cached):
            raise ValueError("Cannot combine fits from different manifests")
        saved = {(f, key): item for f, key, item in cached}
        conn.commit()
        pending = [job for job in JOBS if job not in saved]
        live.emit("comparison_started", jobs=len(JOBS), pending=len(pending), manifest=manifest)
        if pending:
            failed = []
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialise, initargs=(rows, threads)) as executor:
                futures = {executor.submit(fit_job, job): job for job in pending}
                for future in as_completed(futures):
                    fold, key = futures[future]
                    try:
                        item = future.result()
                    except Exception as error:
                        failed.append((fold, key))
                        live.emit("fit_failed", fold=fold, model=key, error=repr(error))
                        continue
                    item["manifest_sha256"] = fingerprint
                    conn.execute("""INSERT INTO padjective.paper_store_held_out_runs
                        (experiment_key,fold,model,evidence) VALUES (%s,%s,%s,%s)""",
                        (EXPERIMENT, fold, key, Jsonb(item)))
                    conn.commit()
                    saved[fold, key] = item
                    live.emit("fit_saved", fold=fold, model=key, completed=len(saved),
                              total=len(JOBS), seconds=item["elapsed_seconds"], iterations=item["iterations"])
            if failed:
                raise RuntimeError(f"Store-held-out jobs failed: {failed}")
        report = collect(rows, predictions, saved, manifest)
        conn.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
            VALUES (%s,%s) ON CONFLICT (experiment_key) DO NOTHING""", (EXPERIMENT, Jsonb(report)))
        readback = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                                (EXPERIMENT,)).fetchone()[0]
        if readback != report:
            raise ValueError("Aggregate report failed Postgres readback")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--threads", default=2, type=int)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("Workers and numerical threads must be positive")
    report = run(args.source_commit, args.workers, args.threads)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    live.emit("comparison_complete", output=args.output, models=len(report["models"]))


if __name__ == "__main__":
    main()
