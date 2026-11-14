"""Compare member sums and lowest-valuation selection under identical consensus."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

import numpy as np
from psycopg.types.json import Jsonb

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_ensembles as sweep
from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as oracle
from .live_subspace import digest, emit
from .paper_low_q_experiment import SOURCE_BATCH
from .paper_subspace_evaluate import get_bank
from .paper_subspace_experiment import BASE_ENSEMBLE_BATCH
from .paper_valuation_selector import select
from .subspace_voting import feature_mask

EXPERIMENT = "paper-valuation-ensembles-2026-11-09"
P, E = matched.PRIME, matched.PRECISION
MODULUS = P**E
FRACTIONS = (.75, 1., .50, .25)
N_MEMBERS = len(matched.BANK_SEEDS)
_datasets = _contexts = _controls = _store_hashes = None


def initialise(datasets, controls, store_hashes):
    global _datasets, _contexts, _controls, _store_hashes
    _datasets = datasets
    _contexts = {split: matched.contexts(dataset) for split, dataset in datasets.items()}
    _controls, _store_hashes = controls, store_hashes


def select_rows(matrix, weights, default):
    """Apply the selector to each sparse row, including the existing zero-term default."""
    weights = np.asarray(weights, dtype=np.int64)
    nonzero = weights != 0
    active = matrix[:, nonzero].tocsr()
    values = weights[nonzero].tolist()
    output = np.full(active.shape[0], default, dtype=np.int64)
    for i, (start, stop) in enumerate(zip(active.indptr[:-1], active.indptr[1:], strict=True)):
        if stop == start + 1:
            output[i] = values[active.indices[start]]
        elif stop > start:
            terms = [values[j] for j in active.indices[start:stop]]
            output[i] = select(terms, P)[0]
    return output


def repeated_bank(split, fold, fraction, context):
    with db.get_connection() as conn:
        conn.read_only = True
        if split == "product":
            report, = conn.execute(
                "SELECT report FROM padjective.paper_ensemble_scaling_reports WHERE batch_id=%s",
                (BASE_ENSEMBLE_BATCH,)).fetchone()
            bank, members, _ = get_bank(conn, SOURCE_BATCH, fraction, fold, context, report)
        else:
            experiment = sweep.EXPERIMENT if fraction in sweep.NEW_FRACTIONS else matched.EXPERIMENT
            saved = {(fold, name): item for name, item in conn.execute("""
                SELECT model,evidence FROM padjective.paper_store_held_out_runs
                WHERE experiment_key=%s AND fold=%s AND model LIKE %s
            """, (experiment, fold, f"ensemble-{fraction}-%"))}
            if len(saved) != N_MEMBERS or any(
                    item["manifest_sha256"] != _store_hashes[experiment] for item in saved.values()):
                raise ValueError("Store members differ from the archived fitting manifest")
            bank = sweep.bank_from_fits(saved, fold, fraction, context)
            members = [saved[fold, matched.name((fold, "ensemble", (fraction, i)))]
                       for i in range(N_MEMBERS)]
    selected = []
    test = context["x_test"]
    for item in members:
        mask = item["mask"]
        if split == "product" and fraction == 1.:
            # The original unmasked coefficient array includes every column.
            mask = np.arange(test.shape[1])
        selected.append(select_rows(test[:, mask], item["coefficients"], item["default"]))
    return bank, np.asarray(selected).T


def one_pass_bank(dataset, fold, fraction, context):
    supported = np.flatnonzero(context["x_train"].getnnz(axis=0))
    columns = {tag: j for j, tag in enumerate(dataset.feature_names)}
    original, selected = [], []
    for seed in matched.BANK_SEEDS:
        mask = feature_mask(supported, fraction, fold, seed)
        tags = {dataset.feature_names[j] for j in mask}
        records = [replace(r, tags=[tag for tag in r.tags if tag in tags]) for r in dataset.records]
        fit = runtime.umllr_run_fold(fold, records, [], P, tag_order_strategy=oracle.ORDER)
        weights = np.zeros(context["x_test"].shape[1], dtype=np.int64)
        for term in fit.coefficients:
            weights[columns[term.tag]] = term.coefficient
        raw = np.asarray(context["x_test"] @ weights).ravel()
        reconstructed = np.where(raw == 0, fit.default_prediction, raw)
        if reconstructed.tolist() != [p.predicted_value for p in fit.predictions]:
            raise ValueError("Masked one-pass sum failed independent reconstruction")
        original.append(reconstructed)
        selected.append(select_rows(context["x_test"], weights, fit.default_prediction))
    original, selected = np.asarray(original).T, np.asarray(selected).T
    # A common interval shorter than the modulus precludes aliasing with valid paths.
    lo = min(original.min(), selected.min(), context["y_train"].min(), context["y_test"].min())
    hi = max(original.max(), selected.max(), context["y_train"].max(), context["y_test"].max())
    if int(hi) - int(lo) >= MODULUS:
        raise ValueError("One-pass integer outputs exceed the consensus precision")
    return original % MODULUS, selected % MODULUS


def fit_job(job):
    split, family, fraction, fold = job
    context = _contexts[split][fold]
    if family == "repeated":
        original, selected = repeated_bank(split, fold, fraction, context)
    else:
        original, selected = one_pass_bank(_datasets[split], fold, fraction, context)
    variants = {}
    for key, bank in (("sum", original), ("selector", selected)):
        prefixes, raw, projected = sweep.score_bank(bank, context, {})
        variants[key] = dict(prefixes=prefixes, raw_members=raw, projected_members=projected)
    old = _controls.get((split, fraction, fold), {}) if family == "repeated" else {}
    for size, loss in old.items():
        if abs(variants["sum"]["prefixes"][size - 1]["loss_q2"] - loss) > 1e-12:
            raise ValueError(f"Archived sum consensus did not reproduce at {size} members")
    return dict(variants=variants, changed_member_predictions=int(np.count_nonzero(original != selected)))


def archived_controls(conn, provenance):
    product = oracle.archived_report(conn, oracle.PRODUCT_REFERENCE)
    store = oracle.archived_report(conn, sweep.EXPERIMENT)
    headline = oracle.archived_report(conn, matched.EXPERIMENT)
    if product["snapshot_digest"] != provenance["product_fold_digest"] or any(
            headline["manifest"][k] != v or store["manifest"][k] != v for k, v in provenance.items()):
        raise ValueError("Archived ensembles use different snapshots or store folds")
    controls = {(split, fraction, fold): {} for split in ("product", "store")
                for fraction in FRACTIONS for fold in matched.FOLDS}
    for row in product["subspace"]:
        if row["roster"] == 0 and row["rule"] == "raw_valid_medoid" and row["fraction"] in FRACTIONS:
            controls["product", row["fraction"], row["fold"]][row["members"]] = row["scores"]["losses"]["2"]
    for bank in store["fractions"]:
        for size, row in enumerate(bank["prefixes"], start=1):
            for fold, value in enumerate(row["fold_loss_q2"]):
                controls["store", bank["fraction"], fold][size] = value
    if any(len(scores) != (9 if split == "product" else N_MEMBERS)
           for (split, _, _), scores in controls.items()):
        raise ValueError("Incomplete sum control scores")
    hashes = {matched.EXPERIMENT: digest(headline["manifest"]), sweep.EXPERIMENT: digest(store["manifest"])}
    return controls, hashes


def job_key(job):
    split, family, fraction, _ = job
    return f"{split}-{family}-{fraction:g}"


def fit_missing(conn, jobs, saved, manifest, datasets, controls, hashes, workers):
    pending = [job for job in jobs if (job[3], job_key(job)) not in saved]
    if not pending:
        return
    fingerprint = digest(manifest)
    failures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise, initargs=(datasets, controls, hashes)) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            fold, key = job[3], job_key(job)
            try:
                item = future.result()
            except Exception as error:
                failures.append((fold, key))
                emit("valuation_ensemble_failed", fold=fold, model=key, error=repr(error))
                continue
            item["manifest_sha256"] = fingerprint
            conn.execute("""
                INSERT INTO padjective.paper_store_held_out_runs(experiment_key,fold,model,evidence)
                VALUES (%s,%s,%s,%s)
            """, (EXPERIMENT, fold, key, Jsonb(item)))
            conn.commit()
            saved[fold, key] = item
            emit("valuation_ensemble_bank_saved", fold=fold, model=key, complete=len(saved), total=len(jobs))
    if failures:
        raise RuntimeError(f"Valuation ensemble banks failed: {failures}")


def collect(saved, jobs):
    reports = []
    for split, family, fraction, fold in jobs:
        if fold != 0:
            continue
        key = job_key((split, family, fraction, fold))
        rows = [saved[f, key] for f in matched.FOLDS]
        variants = {}
        for kind in ("sum", "selector"):
            values = [r["variants"][kind] for r in rows]
            variants[kind] = {
                group: [sweep.summarise(scores) for scores in zip(*(r[group] for r in values), strict=True)]
                for group in ("prefixes", "raw_members", "projected_members")
            }
        reports.append(dict(split=split, family=family, fraction=fraction, variants=variants,
                            changed_member_predictions=sum(r["changed_member_predictions"] for r in rows)))
    return reports


def run(source_commit, workers):
    provenance, inputs = oracle.load_splits()
    datasets = {split: data for split, (data, _, _) in inputs.items()}
    jobs = [(split, "repeated", fraction, fold) for fraction in FRACTIONS
            for fold in matched.FOLDS for split in datasets]
    jobs += [(split, "one-pass", .75, fold) for fold in matched.FOLDS for split in datasets]
    with db.get_connection() as conn:
        locked, = conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()
        if not locked:
            raise RuntimeError("The valuation ensemble audit is already running")
        controls, hashes = archived_controls(conn, provenance)
        manifest = dict(source_commit=source_commit, valuation_prime=P, precision=E, evaluation_q=2,
                        fitting_q=P, consensus_q=P, seed_bases=list(matched.BANK_SEEDS),
                        subspace_batch=SOURCE_BATCH, base_ensemble_batch=BASE_ENSEMBLE_BATCH,
                        store_manifests=hashes, **provenance)
        saved = sweep.read_fits(conn, EXPERIMENT)
        if any(item["manifest_sha256"] != digest(manifest) for item in saved.values()):
            raise ValueError("Cached banks were computed with a different manifest")
        conn.commit()
        emit("valuation_ensemble_started", total=len(jobs), already_saved=len(saved))
        fit_missing(conn, jobs, saved, manifest, datasets, controls, hashes, workers)
        return oracle.save_report(dict(experiment_key=EXPERIMENT, status="validated", manifest=manifest,
            protocol="Compare the fitted sum with lowest-valuation medoid selection within each member. "
                     "Both ensembles choose the training path minimising total native p-adic distance. "
                     "Score every prefix of the fixed roster with equal fold weights.",
            banks=collect(saved, jobs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    args.output.write_text(json.dumps(run(args.source_commit, args.workers), indent=2) + "\n")


if __name__ == "__main__":
    main()
