"""Refit the paper learners after replacing product folds with store folds."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
from statistics import mean, stdev

import numpy as np
from psycopg.types.json import Jsonb
from threadpoolctl import threadpool_limits

from . import benchmark_runtime as runtime, db, live_subspace as live
from . import paper_store_held_out as headline
from .paper_classical_q_experiment import MODELS as CLASSICAL, fit_predict
from .paper_ensemble_scaling_batch import BANK_SEEDS
from .paper_published_methods import PAPER_SNAPSHOT, _snapshot_digest
from .paper_randomised_methods import SNAPSHOT_DIGEST
from .paper_revision_experiments import _load_paper_dataset
from .paper_store_held_out_greedy import ORDERS
from .paper_validate_randomised import direct_certificate, direct_consensus, direct_linear
from .product_hash import canonicalize_product_url, hash_product_url
from .randomised_linear import apply_default, consensus_predictions, fit_coordinates, fit_default
from .subspace_voting import feature_mask

EXPERIMENT = "paper-matched-store-folds-2026-11-09"
FOLDS = range(5)
PRIME, PRECISION = 71, 7
FRACTIONS = (.75, 1.)
SIZES = (1, 9, 15, 243)
SCORE = partial(headline.score, prime=PRIME)
_dataset = None
_paths = None
_contexts = None
_threads = None


def load_data(conn):
    dataset = _load_paper_dataset(conn, snapshot_ref=PAPER_SNAPSHOT, schema="padjective")
    if (_snapshot_digest(dataset) != SNAPSHOT_DIGEST or dataset.features.shape != (6693, 2542)
            or dataset.prime_base != PRIME):
        raise ValueError("The frozen paper snapshot does not match")
    paths = {str(label): runtime.parse_taxonomy_path(path) for label, path in conn.execute("""
        SELECT DISTINCT taxonomy_id,taxonomy_path FROM padjective.product_taxonomy_bench_products
        WHERE snapshot_id=%s
    """, (PAPER_SNAPSHOT,))}
    keys = {r.product_key for r in dataset.records}
    stores = defaultdict(set)
    for url, domain, handle in conn.execute("""
        SELECT product_url,myshopify_domain,product_handle FROM cantbuymelove.product
    """):
        canonical = canonicalize_product_url(url, myshopify_domain=domain, product_handle=handle)
        if canonical and domain and (key := hash_product_url(canonical)) in keys:
            stores[key].add(domain.strip().lower())
    if stores.keys() != keys or any(len(v) != 1 for v in stores.values()):
        raise ValueError("A frozen benchmark URL has missing or conflicting store membership")
    assignment = [(r.product_key, next(iter(stores[r.product_key]))) for r in dataset.records]
    records = [replace(r, cv_fold=live.store_fold(domain))
               for r, (_, domain) in zip(dataset.records, assignment, strict=True)]
    provenance = dict(snapshot_id=PAPER_SNAPSHOT, product_fold_digest=SNAPSHOT_DIGEST,
        store_fold_digest=_snapshot_digest(replace(dataset, records=records)),
        store_assignment_sha256=live.digest(assignment), stores=len({domain for _, domain in assignment}),
        taxonomy_sha256=live.digest(paths), feature_order_sha256=live.digest(dataset.feature_names),
        title_positions_sha256=live.digest([r.title_tag_positions for r in dataset.records]),
        features=dataset.features.shape[1], categories=len(paths))
    return replace(dataset, records=records), paths, provenance


def contexts(dataset):
    folds = np.array([r.cv_fold for r in dataset.records])
    targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)
    return {f: dict(train=folds != f, test=folds == f,
        x_train=dataset.features[folds != f].astype(np.int64),
        x_test=dataset.features[folds == f].astype(np.int64),
        y_train=targets[folds != f], y_test=targets[folds == f]) for f in FOLDS}


def initialise(dataset, paths, threads):
    global _dataset, _paths, _contexts, _threads
    _dataset, _paths, _contexts, _threads = dataset, paths, contexts(dataset), threads


def ensemble_job(fold, fraction, member, c):
    seed = BANK_SEEDS[member]
    eligible = np.flatnonzero(c["x_train"].getnnz(axis=0))
    mask = feature_mask(eligible, fraction, fold, seed)
    train, test = c["x_train"][:, mask], c["x_test"][:, mask]
    fit = fit_coordinates(train, c["y_train"], p=PRIME, precision=PRECISION,
                          seed=seed+fold, max_sweeps=100, seconds=300)
    if fit.status != "coordinate_optimum":
        raise ValueError(f"Incomplete ensemble member: {fit.status}")
    weights = fit.coefficients.tolist()
    train_raw = direct_linear(train, weights, PRIME**PRECISION)
    if direct_certificate(train, c["y_train"], weights, train_raw, PRIME, PRECISION) != len(mask):
        raise ValueError("Incomplete independent coordinate certificate")
    default = fit_default(c["y_train"], train_raw, p=PRIME, precision=PRECISION)
    predicted = apply_default(direct_linear(test, weights, PRIME**PRECISION), default).tolist()
    return dict(fraction=fraction, member=member, seed=seed + fold, sweeps=len(fit.history),
                mask=mask.tolist(), coefficients=weights, default=int(default), predictions=predicted,
                prediction_sha256=live.digest(predicted))


def fit_job(job):
    fold, key, args = job
    c = _contexts[fold]
    if key == "ensemble":
        return ensemble_job(fold, *args, c)
    if key == "greedy-orders":
        records = _dataset.records
        battles = runtime._derive_battles([r for r in records if r.cv_fold != fold])
        fits = []
        for order, seed in ORDERS:
            fit = runtime.umllr_run_fold(fold, records, battles, PRIME,
                                         tag_order_strategy=order, tag_order_seed=seed)
            pred = [p.predicted_value for p in fit.predictions]
            fits.append(dict(model=order, seed=seed, predictions=pred, scores=SCORE(c["y_test"], pred)))
        return dict(fits=fits)
    with threadpool_limits(limits=_threads):
        if key in headline.FOREST_JOBS:
            train, test = c["train"], c["test"]
            values = headline.forest_predictions(_dataset.features[train], _dataset.labels[train],
                                                 _dataset.features[test], headline.FOREST_JOBS[key])
            predictions = {f"forest-{n}": [_dataset.encoded_by_taxonomy[str(y)] for y in pred]
                           for n, pred in values.items()}
            return dict(fits=[dict(model=name, seed=headline.FOREST_JOBS[key], predictions=pred,
                                   scores=SCORE(c["y_test"], pred)) for name, pred in predictions.items()])
        fit = fit_predict(_dataset, _paths, key, fold)
    pred = fit["predicted_values"]
    scores = SCORE(c["y_test"], pred)
    if abs(scores["loss_q2"] - fit["scores"]["losses"]["2"]) > 1e-12:
        raise ValueError("Classical scoring implementations disagree")
    return dict(fits=[dict(model=key, seed=None, predictions=pred, scores=scores,
                           iterations=fit.get("iterations", []), warnings=fit["warnings"])])


def name(job):
    _, key, args = job
    return "-".join([key, *map(str, args)])


def collect(dataset, saved, manifest):
    models = defaultdict(list)
    for fold, c in contexts(dataset).items():
        grouped = defaultdict(list)
        for key in (*CLASSICAL, *headline.FOREST_JOBS, "greedy-orders"):
            for fit in saved[fold, key]["fits"]:
                if SCORE(c["y_test"], fit["predictions"]) != fit["scores"]:
                    raise ValueError("Saved prediction scores disagree")
                grouped[fit["model"]].append({k: v for k, v in fit.items() if k != "predictions"})
        for fraction in FRACTIONS:
            eligible = np.flatnonzero(c["x_train"].getnnz(axis=0))
            members = [saved[fold, name((fold, "ensemble", (fraction, i)))] for i in range(len(BANK_SEEDS))]
            for i, member in enumerate(members):
                mask = member["mask"]
                seed = BANK_SEEDS[i]
                if (member["seed"] != seed + fold or member["member"] != i or member["fraction"] != fraction
                        or mask != feature_mask(eligible, fraction, fold, seed).tolist()):
                    raise ValueError("Saved ensemble member differs from the roster")
                reconstructed = apply_default(direct_linear(c["x_test"][:, mask], member["coefficients"],
                                               PRIME**PRECISION), member["default"]).tolist()
                if reconstructed != member["predictions"] or live.digest(reconstructed) != member["prediction_sha256"]:
                    raise ValueError("Ensemble predictions fail independent replay")
            bank = np.asarray([m["predictions"] for m in members]).T
            for size in SIZES:
                prefix = bank[:, :size]
                pred, _ = consensus_predictions(prefix, c["y_train"], p=PRIME, precision=PRECISION)
                if not np.array_equal(pred, direct_consensus(prefix, c["y_train"], PRIME, PRECISION)):
                    raise ValueError("Ensemble consensus fails independent replay")
                pred = pred.tolist()
                key = f"ensemble-{fraction:g}-{size}"
                grouped[key].append(dict(model=key, seed=None, scores=SCORE(c["y_test"], pred)))
        for key, fits in grouped.items():
            models[key].append(dict(fold=fold, fits=fits, **{
                metric: mean(r["scores"][metric] for r in fits)
                for metric in ("loss_q2", "root_accuracy", "exact_accuracy")}))
    return dict(status="validated", manifest=manifest,
        models=sorted([dict(model=key, folds=rows, loss_q2=mean(r["loss_q2"] for r in rows),
            loss_q2_sd=stdev(r["loss_q2"] for r in rows), exact_accuracy=mean(r["exact_accuracy"] for r in rows),
            root_accuracy=mean(r["root_accuracy"] for r in rows)) for key, rows in models.items()],
            key=lambda r: r["loss_q2"]))


def run(source_commit, workers, threads):
    with db.get_connection() as conn:
        conn.read_only = True
        dataset, paths, provenance = load_data(conn)
    folds = contexts(dataset)
    manifest = dict(experiment_key=EXPERIMENT, source_commit=source_commit, evaluation_q=2,
        valuation_prime=PRIME, native_fitting_q=PRIME, precision=PRECISION, threads=threads,
        fold_sizes=[len(c["y_test"]) for c in folds.values()],
        unseen_target_products=[int(np.count_nonzero(~np.isin(c["y_test"], c["y_train"]))) for c in folds.values()],
        seed_bases=list(BANK_SEEDS), forest_seeds=list(headline.FOREST_JOBS.values()),
        python_version=platform.python_version(),
        versions={key:version(key) for key in ("numpy", "scipy", "scikit-learn")}, **provenance)
    jobs = [(f, key, ()) for key in (*reversed(CLASSICAL), *headline.FOREST_JOBS, "greedy-orders") for f in FOLDS]
    jobs.extend((f, "ensemble", (fraction, member)) for member in range(len(BANK_SEEDS))
                for fraction in FRACTIONS for f in FOLDS)
    fingerprint = live.digest(manifest)
    with db.get_connection() as conn:
        if not conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()[0]:
            raise RuntimeError("The matched store comparison is already running")
        saved = {(fold, key): e for fold, key, e in conn.execute("""
            SELECT fold,model,evidence FROM padjective.paper_store_held_out_runs WHERE experiment_key=%s
        """, (EXPERIMENT,))}
        if any(e["manifest_sha256"] != fingerprint for e in saved.values()):
            raise ValueError("Cannot combine fits from different manifests")
        conn.commit()
        pending = [j for j in jobs if (j[0], name(j)) not in saved]
        live.emit("matched_store_started", jobs=len(jobs), pending=len(pending),
                  **{k: v for k, v in manifest.items() if k != "seed_bases"})
        failed = []
        if pending:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialise, initargs=(dataset, paths, threads)) as pool:
                futures = {pool.submit(fit_job, job): job for job in pending}
                for future in as_completed(futures):
                    job = futures[future]
                    fold, key = job[0], name(job)
                    try:
                        item = future.result()
                    except Exception as error:
                        failed.append((fold, key))
                        live.emit("matched_store_fit_failed", fold=fold, model=key, error=repr(error))
                        continue
                    item["manifest_sha256"] = fingerprint
                    conn.execute("""INSERT INTO padjective.paper_store_held_out_runs
                        (experiment_key,fold,model,evidence) VALUES (%s,%s,%s,%s)""",
                        (EXPERIMENT,fold,key,Jsonb(item)))
                    conn.commit()
                    saved[fold, key] = item
                    if job[1] != "ensemble" or len(saved) % 100 == 0 or len(saved) == len(jobs):
                        live.emit("matched_store_fit_saved", fold=fold, model=key, complete=len(saved), total=len(jobs))
        if failed:
            raise RuntimeError(f"Matched store jobs failed: {failed}")
        report = collect(dataset, saved, manifest)
        conn.execute("""INSERT INTO padjective.paper_low_q_reports(experiment_key,report)
            VALUES (%s,%s) ON CONFLICT (experiment_key) DO NOTHING""", (EXPERIMENT, Jsonb(report)))
        stored = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                              (EXPERIMENT,)).fetchone()[0]
        if stored != report:
            raise ValueError("Aggregate report failed Postgres readback")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("Workers and threads must be positive")
    report = run(args.source_commit, args.workers, args.threads)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    live.emit("matched_store_complete", models=len(report["models"]), output=args.output)


if __name__ == "__main__":
    main()
