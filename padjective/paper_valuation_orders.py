"""Apply lowest-valuation inference to every frozen one-pass tag order."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_folds as matched
from . import paper_sum_oracle as oracle
from . import paper_valuation_selector as selector
from .live_subspace import digest
from .paper_store_held_out_greedy import ORDERS

EXPERIMENT = "paper-valuation-orders-2026-11-09"
METRICS = ("loss_q2", "root_accuracy", "exact_accuracy")


def controls(conn, provenance):
    product = oracle.archived_report(conn, oracle.PRODUCT_REFERENCE)
    store = oracle.archived_report(conn, matched.EXPERIMENT)
    previous = oracle.archived_report(conn, selector.EXPERIMENT)
    manifest = store["manifest"]
    if product["snapshot_digest"] != provenance["product_fold_digest"] or any(
        manifest[k] != v or previous[k] != v for k, v in provenance.items()
    ):
        raise ValueError("The tag-order controls use different data")
    product_scores = {(r["fold"], r["strategy"], r["seed"]): r["scores"] for r in product["greedy"]}
    store_scores = {}
    for fold, item in conn.execute("""
        SELECT fold,evidence FROM padjective.paper_store_held_out_runs
        WHERE experiment_key=%s AND model='greedy-orders'
    """, (matched.EXPERIMENT,)):
        if item["manifest_sha256"] != digest(manifest):
            raise ValueError("The store fits have a different manifest")
        for row in item["fits"]:
            store_scores[fold, row["model"], row["seed"]] = row["scores"]
    expected = {(fold, order, seed) for fold in matched.FOLDS for order, seed in ORDERS}
    if product_scores.keys() != expected or store_scores.keys() != expected:
        raise ValueError("The original order/seed/fold grid is incomplete")
    return dict(product=product_scores, store=store_scores), previous


def summarise(fits):
    folds = []
    for fold in matched.FOLDS:
        seeds = [r for r in fits if r["fold"] == fold]
        scores = {variant: {metric: mean(r[variant][metric] for r in seeds) for metric in METRICS}
                  for variant in ("original", "selected")}
        folds.append(dict(fold=fold, seeds=seeds, **scores))
    return dict(
        **{variant: {**{metric: mean(r[variant][metric] for r in folds) for metric in METRICS},
                     "loss_q2_sd": stdev(r[variant]["loss_q2"] for r in folds)}
           for variant in ("original", "selected")},
        folds=folds,
    )


def run(source_commit):
    provenance, inputs = oracle.load_splits()
    with db.get_connection() as conn:
        conn.read_only = True
        references, previous = controls(conn, provenance)
    splits = {}
    for name, (dataset, _, scoring) in inputs.items():
        by_order = {order: [] for order, _ in ORDERS}
        for fold in matched.FOLDS:
            battles = runtime._derive_battles([r for r in dataset.records if r.cv_fold != fold])
            for order, seed in ORDERS:
                row = selector.fold_report(dataset, fold, references[name][fold, order, seed], scoring,
                                           strategy=order, seed=seed, battles=battles)
                if order == oracle.ORDER and row != previous["splits"][name]["folds"][fold]:
                    raise ValueError("The original taxonomy-association selector did not reproduce")
                by_order[order].append(dict(seed=seed, **row))
        splits[name] = {order: summarise(fits) for order, fits in by_order.items()}
    return oracle.save_report(dict(
        experiment_key=EXPERIMENT, status="validated", source_commit=source_commit,
        valuation_prime=matched.PRIME, fitting_q=matched.PRIME, evaluation_q=2, medoid_q=matched.PRIME,
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT,
                        selector=selector.EXPERIMENT),
        protocol="Reuse each original one-pass fit and default. Select the native p-adic medoid of the "
                 "nonzero coefficients with lowest valuation. Break ties by signed integer order. "
                 "Average random seeds within each fold, then weight all five folds equally.",
        **provenance, splits=splits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.source_commit), indent=2) + "\n")


if __name__ == "__main__":
    main()
