"""Measure individual voters and all prefixes on the matched paper store folds."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
import json
import multiprocessing
from pathlib import Path
import platform
from statistics import mean, stdev

import numpy as np
from psycopg.types.json import Jsonb

from . import db, paper_matched_store_folds as matched
from .live_subspace import digest, emit
from .paper_store_held_out import FOREST_JOBS, forest_probabilities
from .paper_validate_randomised import direct_linear, distance_matrix
from .randomised_linear import apply_default, consensus_predictions
from .subspace_voting import feature_mask, project_codes

EXPERIMENT = "paper-matched-store-ensembles-2026-11-09"
NEW_FRACTIONS = (.25, .50)
FRACTIONS = (*NEW_FRACTIONS, *matched.FRACTIONS)
N_MEMBERS = len(matched.BANK_SEEDS)
N_TREES = 81
METRICS = ("loss_q2", "root_accuracy", "exact_accuracy")


def fit_job(job):
    fold, key, _ = job
    if key == "ensemble":
        return matched.fit_job(job)
    dataset = matched._dataset
    c = matched._contexts[fold]
    train, test = c["train"], c["test"]
    seed = FOREST_JOBS[key]
    trees = forest_probabilities(dataset.features[train], dataset.labels[train],
                                dataset.features[test], seed, N_TREES)
    prefixes, members = [], []
    total = None
    for labels, probabilities in trees:
        if total is None:
            codes = np.array([dataset.encoded_by_taxonomy[str(label)] for label in labels])
            total = np.zeros_like(probabilities)
        total += probabilities
        members.append(matched.SCORE(c["y_test"], codes[probabilities.argmax(axis=1)].tolist()))
        prefixes.append(matched.SCORE(c["y_test"], codes[total.argmax(axis=1)].tolist()))
    return dict(seed=seed, prefixes=prefixes, members=members)


def read_fits(conn, experiment):
    return {(fold, key): item for fold, key, item in conn.execute("""
        SELECT fold,model,evidence FROM padjective.paper_store_held_out_runs
        WHERE experiment_key=%s
    """, (experiment,))}


def fit_missing(conn, saved, manifest, dataset, paths, workers):
    jobs = [(f, key, ()) for key in FOREST_JOBS for f in matched.FOLDS]
    jobs += [(f, "ensemble", (fraction, member)) for member in range(N_MEMBERS)
             for fraction in NEW_FRACTIONS for f in matched.FOLDS]
    pending = [job for job in jobs if (job[0], matched.name(job)) not in saved]
    emit("store_ensemble_sweep_started", total=len(jobs), pending=len(pending))
    if not pending:
        return
    fingerprint = digest(manifest)
    failures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=matched.initialise, initargs=(dataset, paths, 1)) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            fold, key = job[0], matched.name(job)
            try:
                item = future.result()
            except Exception as error:
                failures.append((fold, key))
                emit("store_ensemble_sweep_fit_failed", fold=fold, model=key, error=repr(error))
                continue
            item["manifest_sha256"] = fingerprint
            conn.execute("""
                INSERT INTO padjective.paper_store_held_out_runs(experiment_key,fold,model,evidence)
                VALUES (%s,%s,%s,%s)
            """, (EXPERIMENT, fold, key, Jsonb(item)))
            conn.commit()
            saved[fold, key] = item
            if job[1] != "ensemble" or len(saved) % 100 == 0 or len(saved) == len(jobs):
                emit("store_ensemble_sweep_fit_saved", fold=fold, model=key, complete=len(saved), total=len(jobs))
    if failures:
        raise RuntimeError(f"Incomplete store ensemble sweep fits: {failures}")


def bank_from_fits(saved, fold, fraction, c):
    eligible = np.flatnonzero(c["x_train"].getnnz(axis=0))
    predictions = []
    for index, seed in enumerate(matched.BANK_SEEDS):
        item = saved[fold, matched.name((fold, "ensemble", (fraction, index)))]
        mask = feature_mask(eligible, fraction, fold, seed).tolist()
        if (item["mask"], item["fraction"], item["member"], item["seed"]) != (mask, fraction, index, seed + fold):
            raise ValueError("Stored member does not match the requested bank")
        raw = direct_linear(c["x_test"][:, mask], item["coefficients"], matched.PRIME**matched.PRECISION)
        pred = apply_default(raw, item["default"]).tolist()
        if pred != item["predictions"] or digest(pred) != item["prediction_sha256"]:
            raise ValueError("Stored member failed independent prediction replay")
        predictions.append(pred)
    return np.asarray(predictions, dtype=np.int64).T


def raw_prefixes(bank, candidates):
    """Accumulate the native distance to each candidate as each voter is added."""
    values, indices = np.unique(bank, return_inverse=True)
    indices = indices.reshape(bank.shape)
    lookup = distance_matrix(values, candidates, matched.PRIME, matched.PRECISION)
    costs = np.zeros((len(bank), len(candidates)), dtype=np.int64)
    for column in indices.T:
        costs += lookup[column]
        yield candidates[costs.argmin(axis=1)]


def scores(c, predictions):
    return matched.SCORE(c["y_test"], predictions.tolist())


def score_bank(bank, c, old_prefixes):
    candidates = np.unique(c["y_train"])
    sample = np.unique(np.linspace(0, len(bank) - 1, 5, dtype=int))
    prefixes = []
    for size, pred in enumerate(raw_prefixes(bank, candidates), start=1):
        check, _ = consensus_predictions(bank[sample, :size], candidates,
                                         p=matched.PRIME, precision=matched.PRECISION)
        if not np.array_equal(check, pred[sample]):
            raise ValueError("Incremental prefix failed independent consensus check")
        score = scores(c, pred)
        if size in old_prefixes and score != old_prefixes[size]:
            raise ValueError("Prefix differs from the prior matched result")
        prefixes.append(score)
    projected = project_codes(bank, candidates, matched.PRIME, matched.PRECISION)
    return prefixes, [scores(c, pred) for pred in bank.T], [scores(c, pred) for pred in projected.T]


def summarise(fold_scores):
    loss = [r["loss_q2"] for r in fold_scores]
    return dict(**{metric: mean(r[metric] for r in fold_scores) for metric in METRICS},
                loss_q2_sd=stdev(loss), fold_loss_q2=loss)


def expected_prior_loss(c):
    """Expected loss for a random label drawn from the fold's training frequencies."""
    train, counts = np.unique(c["y_train"], return_counts=True)
    test, test_counts = np.unique(c["y_test"], return_counts=True)
    native = distance_matrix(test, train, matched.PRIME, matched.PRECISION)
    loss = np.zeros_like(native, dtype=float)
    for depth in range(matched.PRECISION):
        loss[native == matched.PRIME**(matched.PRECISION - depth)] = 2.0**(-depth)
    return float((test_counts / test_counts.sum()) @ loss @ (counts / counts.sum()))


def member_summary(members, constant, prior):
    values = [row["loss_q2"] for row in members]
    return dict(count=len(values), mean_loss_q2=mean(values), median_loss_q2=float(np.median(values)),
                min_loss_q2=min(values), max_loss_q2=max(values),
                beat_constant=sum(loss < constant for loss in values),
                beat_random_training_prior=sum(loss < prior for loss in values))


def collect(dataset, saved, old_fits, old_report, manifest):
    contexts = matched.contexts(dataset)
    old_prefixes = {r["model"]: [f["fits"][0]["scores"] for f in r["folds"]]
                    for r in old_report["models"] if r["model"].startswith("ensemble-")}
    baseline = summarise([old_fits[f, "dummy"]["fits"][0]["scores"] for f in matched.FOLDS])
    prior_folds = [expected_prior_loss(c) for c in contexts.values()]
    constant, prior = baseline["loss_q2"], mean(prior_folds)
    fractions = []
    for fraction in FRACTIONS:
        evidence = saved if fraction in NEW_FRACTIONS else old_fits
        results = []
        for fold, c in contexts.items():
            bank = bank_from_fits(evidence, fold, fraction, c)
            control = {size: old_prefixes[f"ensemble-{fraction:g}-{size}"][fold] for size in matched.SIZES
                       if fraction in matched.FRACTIONS}
            results.append(score_bank(bank, c, control))
            emit("store_ensemble_sweep_bank_scored", fold=fold, fraction=fraction)
        prefixes, raw, projected = [list(map(summarise, zip(*rows, strict=True)))
                                    for rows in zip(*results, strict=True)]
        fractions.append(dict(fraction=fraction, prefixes=prefixes,
            members=[dict(seed=seed, raw=a, projected=b)
                     for seed, a, b in zip(matched.BANK_SEEDS, raw, projected, strict=True)],
            raw_member_summary=member_summary(raw, constant, prior),
            projected_member_summary=member_summary(projected, constant, prior)))

    forest_members = []
    for key, seed in FOREST_JOBS.items():
        for fold in matched.FOLDS:
            for old in old_fits[fold, key]["fits"]:
                size = int(old["model"].split("-")[-1])
                if saved[fold, key]["prefixes"][size - 1] != old["scores"]:
                    raise ValueError("Forest prefix differs from the prior matched result")
        forest_members.extend(dict(seed=seed, tree=i + 1,
                                   **summarise([saved[f, key]["members"][i] for f in matched.FOLDS]))
                              for i in range(N_TREES))
    forest_prefixes = []
    for i in range(N_TREES):
        by_fold = [{metric: mean(saved[f, key]["prefixes"][i][metric] for key in FOREST_JOBS)
                    for metric in METRICS} for f in matched.FOLDS]
        forest_prefixes.append(summarise(by_fold))
    return dict(status="validated", manifest=manifest, constant=baseline,
                random_training_prior=dict(loss_q2=prior, fold_loss_q2=prior_folds),
                balanced_tree=summarise([old_fits[f, "dt"]["fits"][0]["scores"] for f in matched.FOLDS]),
                fractions=fractions, forest=dict(prefixes=forest_prefixes, members=forest_members,
                    member_summary=member_summary(forest_members, constant, prior)))


def run(source_commit, workers):
    with db.get_connection() as conn:
        conn.read_only = True
        dataset, paths, provenance = matched.load_data(conn)
        old_report = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                                  (matched.EXPERIMENT,)).fetchone()[0]
        old_fits = read_fits(conn, matched.EXPERIMENT)
    parent = old_report["manifest"]
    if (old_report["status"] != "validated" or any(parent[k] != v for k, v in provenance.items())
            or parent["seed_bases"] != list(matched.BANK_SEEDS)
            or parent["forest_seeds"] != list(FOREST_JOBS.values())
            or parent["python_version"] != platform.python_version()
            or parent["versions"] != {package: version(package) for package in parent["versions"]}
            or any(fit["manifest_sha256"] != digest(parent) for fit in old_fits.values())):
        raise ValueError("Existing store-fold fits use a different protocol")
    manifest = dict(experiment_key=EXPERIMENT, source_commit=source_commit, parent_manifest_sha256=digest(parent),
                    parent_experiment=matched.EXPERIMENT, fractions=list(FRACTIONS),
                    p_adic_prefix_sizes=list(range(1, N_MEMBERS + 1)),
                    forest_prefix_sizes=list(range(1, N_TREES + 1)),
                    valuation_prime=matched.PRIME, precision=matched.PRECISION, evaluation_q=2,
                    native_fitting_q=parent["native_fitting_q"], fold_sizes=parent["fold_sizes"],
                    seed_bases=parent["seed_bases"], forest_seeds=parent["forest_seeds"], **provenance)
    with db.get_connection() as conn:
        locked, = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()
        if not locked:
            raise RuntimeError("The matched store ensemble sweep is already running")
        saved = read_fits(conn, EXPERIMENT)
        if any(fit["manifest_sha256"] != digest(manifest) for fit in saved.values()):
            raise ValueError("Cannot resume from a different sweep manifest")
        conn.commit()
        fit_missing(conn, saved, manifest, dataset, paths, workers)
        report = collect(dataset, saved, old_fits, old_report, manifest)
        conn.execute("""
            INSERT INTO padjective.paper_low_q_reports(experiment_key,report) VALUES (%s,%s)
            ON CONFLICT (experiment_key) DO NOTHING
        """, (EXPERIMENT, Jsonb(report)))
        stored, = conn.execute("SELECT report FROM padjective.paper_low_q_reports WHERE experiment_key=%s",
                               (EXPERIMENT,)).fetchone()
        if report != stored:
            raise ValueError("Sweep report failed Postgres readback")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    report = run(args.source_commit, args.workers)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    emit("store_ensemble_sweep_complete", output=args.output)


if __name__ == "__main__":
    main()
