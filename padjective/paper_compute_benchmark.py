"""Time the published model recipes one fold at a time on an idle Linux host."""
from __future__ import annotations

import argparse
from dataclasses import replace
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
from statistics import mean, median
import subprocess
import sys
import time
import warnings

import numpy as np
from psycopg.types.json import Jsonb
from threadpoolctl import threadpool_info, threadpool_limits

from . import benchmark_runtime as runtime, db
from . import paper_classical_q_experiment as classical
from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as oracle
from . import paper_valuation_ensembles as valuation_ensembles
from .live_subspace import digest, emit
from .randomised_linear import consensus_predictions, fit_coordinates, fit_default, linear_predictions
from .subspace_voting import feature_mask

EXPERIMENT = "paper-compute-2026-11-09"
MODELS = ("one-pass", "medoid-15", "ulr", "repeated-243", "unn")
P, E = matched.PRIME, matched.PRECISION
SAMPLE_SECONDS = 2
REPEATS = 3
IDLE_SAMPLE_SECONDS = 10
MAX_BACKGROUND_BUSY = .08
MAX_IOWAIT = .03
MIN_WORKER_SHARE = .90


def cpu_counters():
    return {
        fields[0]: [int(x) for x in fields[1:9]]
        for line in Path("/proc/stat").read_text().splitlines()
        if (fields := line.split()) and fields[0].startswith("cpu")
    }


def cpu_window(before, after, cpus):
    elapsed = [b - a for a, b in zip(before, after, strict=True)]
    total = sum(elapsed)
    return dict(busy=(total - elapsed[3] - elapsed[4]) / total, iowait=elapsed[4] / total,
                cpus=cpus)


def wait_for_idle(cpu, siblings):
    idle = 0
    while idle < 2:
        before = cpu_counters()
        time.sleep(IDLE_SAMPLE_SECONDS)
        after = cpu_counters()
        host = cpu_window(before["cpu"], after["cpu"], len(after) - 1)
        core = [cpu_window(before[f"cpu{i}"], after[f"cpu{i}"], 1) for i in siblings]
        if (host["busy"] <= MAX_BACKGROUND_BUSY and host["iowait"] <= MAX_IOWAIT
                and all(row["busy"] <= .05 for row in core)):
            idle += 1
        else:
            idle = 0
            emit("compute_waiting_for_idle", cpu=cpu, host=host, core=core)
    return dict(host=host, core=core, sample_seconds=IDLE_SAMPLE_SECONDS, consecutive_windows=idle)


def peak_rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def fit_greedy(training, feature_names, fold):
    terms, default = runtime.fit_one_pass(fold, training, (), P, tag_order_strategy=oracle.ORDER)
    weights = {term.tag: term.coefficient for term in terms}
    return np.array([weights.get(tag, 0) for tag in feature_names], dtype=np.int64), default


def fit_model(key, fold, dataset, x_train, y_train, training, train_mask):
    if key in ("ulr", "unn"):
        estimator = classical.make_estimator(key)
        estimator.fit(x_train, dataset.labels[train_mask])
        codes = dataset.encoded_by_taxonomy

        def predict(x):
            return np.array([codes[str(label)] for label in estimator.predict(x)])

        return predict, dict(iterations=np.asarray(estimator.n_iter_).tolist())

    if key == "one-pass":
        weights, default = fit_greedy(training, dataset.feature_names, fold)

        def predict(x):
            raw = x @ weights
            return np.where(raw == 0, default, raw)

        return predict, dict(members=1)

    candidates = np.unique(y_train)
    eligible = np.flatnonzero(x_train.getnnz(axis=0))
    if key == "medoid-15":
        members = []
        for seed in matched.BANK_SEEDS[:15]:
            mask = feature_mask(eligible, .75, fold, seed)
            tags = {dataset.feature_names[j] for j in mask}
            masked = [replace(row, tags=[tag for tag in row.tags if tag in tags]) for row in training]
            members.append(fit_greedy(masked, dataset.feature_names, fold))

        def predict(x):
            bank = np.column_stack([valuation_ensembles.select_rows(x, w, b) for w, b in members])
            return consensus_predictions(bank % P**E, candidates, p=P, precision=E)[0]

        return predict, dict(members=len(members), feature_fraction=.75)

    weights = np.zeros((x_train.shape[1], len(matched.BANK_SEEDS)), dtype=np.int64)
    defaults = np.empty(weights.shape[1], dtype=np.int64)
    for j, seed in enumerate(matched.BANK_SEEDS):
        mask = feature_mask(eligible, .75, fold, seed)
        masked = x_train[:, mask]
        fit = fit_coordinates(masked, y_train, p=P, precision=E, seed=seed + fold,
                              max_sweeps=100, seconds=300)
        if fit.status != "coordinate_optimum":
            raise ValueError(f"Ensemble member {j} did not converge: {fit.status}")
        weights[mask, j] = fit.coefficients
        raw = linear_predictions(masked, fit.coefficients, P, E)
        defaults[j] = fit_default(y_train, raw, p=P, precision=E)

    def predict(x):
        raw = (x @ weights) % P**E
        bank = np.where(raw == 0, defaults, raw)
        return consensus_predictions(bank, candidates, p=P, precision=E)[0]

    return predict, dict(members=weights.shape[1], feature_fraction=.75)


def reference_scores(conn, key, fold):
    if key in ("ulr", "unn"):
        row, = conn.execute("""
            SELECT evidence FROM padjective.paper_classical_q_runs
            WHERE experiment_key=%s AND model_key=%s AND cv_fold=%s AND status='complete'
        """, (classical.EXPERIMENT, key, fold)).fetchone()
        return row["scores"]
    if key == "medoid-15":
        row, = conn.execute("""
            SELECT evidence FROM padjective.paper_store_held_out_runs
            WHERE experiment_key=%s AND model='product-one-pass-0.75' AND fold=%s
        """, (valuation_ensembles.EXPERIMENT, fold)).fetchone()
        return row["variants"]["selector"]["prefixes"][14]
    archived = oracle.archived_report(conn, oracle.PRODUCT_REFERENCE)
    if key == "one-pass":
        return next(r["scores"] for r in archived["greedy"] if (r["fold"], r["strategy"]) == (fold, oracle.ORDER))
    return next(r["scores"] for r in archived["subspace"] if
                (r["fold"], r["fraction"], r["roster"], r["members"], r["rule"]) ==
                (fold, .75, 0, 243, "raw_valid_medoid"))


def worker(key, fold, cpu):
    os.sched_setaffinity(0, {cpu})
    with db.get_connection() as conn:
        conn.read_only = True
        dataset = matched._load_paper_dataset(conn, snapshot_ref=matched.PAPER_SNAPSHOT, schema="padjective")
    if matched._snapshot_digest(dataset) != matched.SNAPSHOT_DIGEST or dataset.prime_base != P:
        raise ValueError("The frozen benchmark differs")
    train_mask = np.array([r.cv_fold != fold for r in dataset.records])
    training = [r for r in dataset.records if r.cv_fold != fold]
    x_train, x_test = dataset.features[train_mask], dataset.features[~train_mask]
    if key not in ("ulr", "unn"):
        x_train, x_test = x_train.astype(np.int64), x_test.astype(np.int64)
    targets = np.array([r.encoded_path for r in dataset.records], dtype=np.int64)

    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pools = [{k: v for k, v in p.items() if k != "filepath"} for p in threadpool_info()]
        if any(p["num_threads"] != 1 for p in pools):
            raise ValueError("A numerical library ignored the single-thread budget")
        host_start = cpu_counters()
        t0, cpu0 = time.perf_counter(), time.process_time()
        predict, details = fit_model(key, fold, dataset, x_train, targets[train_mask], training, train_mask)
        fit_seconds, fit_cpu = time.perf_counter() - t0, time.process_time() - cpu0
        fit_peak = peak_rss_mib()
        fit_host = cpu_counters()
        prediction = predict(x_test)
        measurements = []
        for _ in range(REPEATS):
            start, cpu_start = time.perf_counter(), time.process_time()
            batches = 0
            while time.perf_counter() - start < SAMPLE_SECONDS:
                last = predict(x_test)
                batches += 1
            seconds, cpu_seconds = time.perf_counter() - start, time.process_time() - cpu_start
            if not np.array_equal(prediction, last):
                raise ValueError("Predictions changed during timing")
            measurements.append(dict(seconds=seconds, cpu_seconds=cpu_seconds, batches=batches,
                                     predictions_per_second=batches * len(prediction) / seconds))
        peak = peak_rss_mib()
        host_end = cpu_counters()

    scores = matched.SCORE(targets[~train_mask], prediction.tolist())
    with db.get_connection() as conn:
        conn.read_only = True
        reference = reference_scores(conn, key, fold)
    loss = reference["losses"]["2"] if "losses" in reference else reference["loss_q2"]
    exact_match = reference["prediction_sha256"] == scores["prediction_sha256"]
    if key not in ("ulr", "unn") and not exact_match:
        raise ValueError(f"Predictions differ from the paper reference for {key}, fold {fold}")
    return dict(model=key, fold=fold, train_products=len(training), test_products=len(prediction),
                fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu, fit_peak_rss_mib=fit_peak, peak_rss_mib=peak,
                predictions_per_second=median(r["predictions_per_second"] for r in measurements),
                timing_samples=measurements, scores=scores, reference_loss_q2=loss,
                reference_prediction_match=exact_match, numerical_libraries=pools,
                fit_host=cpu_window(host_start["cpu"], fit_host["cpu"], len(host_start) - 1),
                predict_host=cpu_window(fit_host["cpu"], host_end["cpu"], len(host_start) - 1),
                warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
                **details)


def collect(saved):
    rows = []
    for key in MODELS:
        folds = [saved[f, key] for f in matched.FOLDS]
        rows.append(dict(model=key, folds=folds,
                         mean_fit_seconds=mean(r["fit_seconds"] for r in folds),
                         median_predictions_per_second=median(r["predictions_per_second"] for r in folds),
                         max_peak_rss_mib=max(r["peak_rss_mib"] for r in folds),
                         max_fit_peak_rss_mib=max(r["fit_peak_rss_mib"] for r in folds)))
    return rows


def competing_load(item):
    timing = item["timing_samples"]
    ratios = {
        "fit": item["fit_cpu_seconds"] / item["fit_seconds"],
        "predict": sum(r["cpu_seconds"] for r in timing) / sum(r["seconds"] for r in timing),
    }
    return {
        phase: dict(cpu_share=ratio, background_busy=max(0., host["busy"] - ratio / host["cpus"]),
                    iowait=host["iowait"])
        for phase, ratio in ratios.items() for host in [item[f"{phase}_host"]]
    }


def run(source_commit, cpu):
    sibling_spec = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text().strip()
    siblings = sorted({i for section in sibling_spec.split(",") for lo, _, hi in [section.partition("-")]
                       for i in range(int(lo), int(hi or lo) + 1)})
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("The requested CPU is outside the available affinity mask")
    cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                     if line.startswith("model name"))
    manifest = dict(source_commit=source_commit, machine=platform.node(), cpu_model=cpu_model,
        cpu=cpu, core_siblings=siblings, system=platform.platform(), python=platform.python_version(),
        packages={name: version(name) for name in ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
        snapshot_id=matched.PAPER_SNAPSHOT, snapshot_digest=matched.SNAPSHOT_DIGEST, numerical_threads=1,
        evaluation_q=2, fitting_q=P, consensus_q=P, sample_seconds=SAMPLE_SECONDS, repeats=REPEATS,
        idle_gate=dict(sample_seconds=IDLE_SAMPLE_SECONDS, max_background_busy=MAX_BACKGROUND_BUSY,
                       max_iowait=MAX_IOWAIT, min_worker_cpu_share=MIN_WORKER_SHARE))
    fingerprint = digest(manifest)
    wait_for_idle(cpu, siblings)
    with db.get_connection() as conn:
        if not conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (EXPERIMENT,)).fetchone()[0]:
            raise RuntimeError("The compute benchmark is already running")
        saved = {(fold, key): item for fold, key, item in conn.execute("""
            SELECT fold,model,evidence FROM padjective.paper_store_held_out_runs WHERE experiment_key=%s
        """, (EXPERIMENT,))}
        if any(r["manifest_sha256"] != fingerprint for r in saved.values()):
            raise ValueError("Cannot resume timings from a different setup")
        conn.commit()
        emit("compute_benchmark_started", complete=len(saved), total=len(MODELS)*len(matched.FOLDS), manifest=manifest)
        for fold in matched.FOLDS:
            for key in MODELS:
                if (fold, key) in saved:
                    continue
                while True:
                    idle = wait_for_idle(cpu, siblings)
                    emit("compute_fit_started", fold=fold, model=key, idle=idle)
                    command = [sys.executable, "-m", "padjective.paper_compute_benchmark",
                               "--worker", key, "--fold", str(fold), "--cpu", str(cpu)]
                    item = json.loads(subprocess.check_output(command, text=True))
                    load = competing_load(item)
                    if all(r["background_busy"] <= MAX_BACKGROUND_BUSY and r["iowait"] <= MAX_IOWAIT
                           and r["cpu_share"] >= MIN_WORKER_SHARE for r in load.values()):
                        break
                    emit("compute_discarded_busy_measurement", fold=fold, model=key, load=load)
                item.update(manifest_sha256=fingerprint, idle_before_fit=idle, timed_competing_load=load)
                conn.execute("""
                    INSERT INTO padjective.paper_store_held_out_runs(experiment_key,fold,model,evidence)
                    VALUES (%s,%s,%s,%s)
                """, (EXPERIMENT, fold, key, Jsonb(item)))
                conn.commit()
                saved[fold, key] = item
                emit("compute_fit_saved", fold=fold, model=key, complete=len(saved),
                     seconds=item["fit_seconds"], predictions_per_second=item["predictions_per_second"],
                     peak_rss_mib=item["peak_rss_mib"], reference_match=item["reference_prediction_match"])
        return oracle.save_report(dict(
            experiment_key=EXPERIMENT, status="validated", manifest=manifest, models=collect(saved),
            protocol="One process and one pinned logical CPU per model and fold. Fit wall time includes "
                     "masking and default fitting. Timing starts from resident sparse features, excludes "
                     "database I/O and scoring, and includes fresh predictions, defaults and consensus. "
                     "Three repeated full-fold batch timings follow one warmup. Peak RSS is the process "
                     "high-water mark through fit and prediction, including the interpreter and input data."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--worker", choices=MODELS)
    parser.add_argument("--fold", type=int, choices=matched.FOLDS)
    args = parser.parse_args()
    if args.worker:
        if args.fold is None:
            parser.error("--worker requires --fold")
        print(json.dumps(worker(args.worker, args.fold, args.cpu)))
    else:
        if args.source_commit is None or args.output is None:
            parser.error("--source-commit and --output are required")
        args.output.write_text(json.dumps(run(args.source_commit, args.cpu), indent=2) + "\n")


if __name__ == "__main__":
    main()
