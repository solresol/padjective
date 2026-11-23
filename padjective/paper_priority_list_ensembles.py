"""Ensemble priority lists under the valid-path consensus of the one-pass bank.

Members reuse the archived bank's seeds, feature masks and consensus, so the
masked scheme differs from the archived one-pass ensembles only in how each
member is fitted and how it predicts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from statistics import mean

import numpy as np

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_ensembles as sweep
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_sum_oracle as oracle
from . import paper_valuation_ensembles as valuation_ensembles
from .paper_valuation_selector import select
from .q_loss_rescore import load_predictions
from .randomised_linear import consensus_predictions
from .subspace_voting import corrected_test, feature_mask, holm

EXPERIMENT = "paper-priority-list-ensembles-2026-11-23"
P, MODULUS = matched.PRIME, matched.PRIME**matched.PRECISION
FIT_Q = 71
SIZES = (1, 3, 9, 15, 27, 45, 81, 135, 243)
TEST_SIZES = (15, 243)
MASKS = {"mask_0.75": .75, "mask_0.5": .5}
SCHEMES = (*MASKS, "random_order")
CLASSICAL = dict(product=("ulr", "unn", "dt"), store=("ulr", "unn", "forest-81", "dt"))
TESTED_CLASSICAL = ("ulr", "unn")
_datasets = _contexts = _controls = None


def initialise(datasets, controls):
    global _datasets, _contexts, _controls
    _datasets, _controls = datasets, controls
    _contexts = {split: matched.contexts(dataset) for split, dataset in datasets.items()}


def member_order(scheme, fold, seed, static_order, supported, feature_names):
    """A member's tag order: the static order within its feature mask, or a full shuffle.

    Taxonomy association ranks each tag from its own products alone, so the
    order fitted on masked records is the full order restricted to the mask.
    """
    if scheme == "random_order":
        return runtime._random_order(static_order, seed + fold)
    kept = {feature_names[j] for j in feature_mask(supported, MASKS[scheme], fold, seed)}
    return [tag for tag in static_order if tag in kept]


def member_predictions(training, testing, order, pooled, default):
    """One member's held-out predictions for each way of filling its list."""
    rank, coefficients = plist.fill(training, order, pooled, prime=P, q=FIT_Q)
    return {variant: plist.predict_priority(testing, rank, values, default)[0]
            for variant, values in coefficients.items()}


def control_loss(dataset, fold, supported):
    """Refit the first archived one-pass member and score its minimum-valuation medoid."""
    kept = {dataset.feature_names[j] for j in feature_mask(supported, .75, fold, matched.BANK_SEEDS[0])}
    records = [replace(r, tags=[tag for tag in r.tags if tag in kept]) for r in dataset.records]
    fit = runtime.umllr_run_fold(fold, records, [], P, tag_order_strategy=oracle.ORDER)
    coefficients = {c.tag: c.coefficient for c in fit.coefficients if c.coefficient}
    held_out = [r for r in records if r.cv_fold == fold]
    selected = []
    for record in held_out:
        active = [coefficients[tag] for tag in record.tags if tag in coefficients]
        selected.append((select(active, P)[0] if active else fit.default_prediction) % MODULUS)
    return matched.SCORE([r.encoded_path for r in held_out], selected)["loss_q2"]


def score_prefixes(bank, context):
    """Consensus scores for every prefix of a products-by-members bank."""
    candidates = np.unique(context["y_train"])
    sample = np.unique(np.linspace(0, len(bank) - 1, 5, dtype=int))
    prefixes = []
    for size, predictions in enumerate(sweep.raw_prefixes(bank, candidates), start=1):
        if size in SIZES:
            check, _ = consensus_predictions(bank[sample, :size], candidates,
                                             p=P, precision=matched.PRECISION)
            if not np.array_equal(check, predictions[sample]):
                raise ValueError("Incremental prefix failed independent consensus check")
        prefixes.append(sweep.scores(context, predictions))
    return prefixes


def fold_job(job):
    split, fold = job
    dataset, context = _datasets[split], _contexts[split][fold]
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    targets = [r.encoded_path for r in testing]
    if targets != context["y_test"].tolist():
        raise ValueError("Held-out rows differ from the fold context")
    supported = np.flatnonzero(context["x_train"].getnnz(axis=0))
    if abs(control_loss(dataset, fold, supported) - _controls[split][fold]) > 1e-12:
        raise ValueError(f"The archived one-pass member did not reproduce on {split} fold {fold}")

    static = runtime._tag_order(training, [], fold, strategy=oracle.ORDER)
    pooled, default = plist.fit_pooled(training, prime=P, q=FIT_Q)
    single = {variant: matched.SCORE(targets, predictions) for variant, predictions in
              member_predictions(training, testing, static, pooled, default).items()}
    banks = {scheme: {variant: [] for variant in plist.VARIANTS} for scheme in SCHEMES}
    for seed in matched.BANK_SEEDS:
        for scheme in SCHEMES:
            order = member_order(scheme, fold, seed, static, supported, dataset.feature_names)
            for variant, predictions in member_predictions(
                    training, testing, order, pooled, default).items():
                banks[scheme][variant].append(predictions)
    schemes = {}
    for scheme, variants in banks.items():
        schemes[scheme] = {}
        for variant, members in variants.items():
            bank = np.asarray(members, dtype=np.int64).T
            prefixes = score_prefixes(bank, context)
            schemes[scheme][variant] = dict(
                prefix_loss_q2=[row["loss_q2"] for row in prefixes],
                sizes={size: prefixes[size - 1] for size in SIZES},
                member_loss_q2=[sweep.scores(context, column)["loss_q2"] for column in bank.T])
    return dict(split=split, fold=fold, n=len(testing), single=single, schemes=schemes)


def archived_references(conn, datasets):
    """Per-fold losses of the archived one-pass ensembles and classical models."""
    one_pass, controls = {}, {}
    for bank in oracle.archived_report(conn, valuation_ensembles.EXPERIMENT)["banks"]:
        if bank["family"] != "one-pass" or bank["fraction"] != .75:
            continue
        controls[bank["split"]] = bank["variants"]["selector"]["raw_members"][0]["fold_loss_q2"]
        one_pass[bank["split"]] = {
            name: {size: bank["variants"][key]["prefixes"][size - 1] for size in SIZES}
            for name, key in (("sum", "sum"), ("medoid", "selector"))}
    classical = dict(product={}, store={})
    for key in CLASSICAL["product"]:
        classical["product"][key] = [
            matched.SCORE(*load_predictions(conn, datasets["product"], key, fold))["loss_q2"]
            for fold in matched.FOLDS]
    stored = {m["model"]: m for m in oracle.archived_report(conn, matched.EXPERIMENT)["models"]}
    for key in CLASSICAL["store"]:
        classical["store"][key] = [fold["loss_q2"] for fold in stored[key]["folds"]]
    if one_pass.keys() != datasets.keys():
        raise ValueError("The archived one-pass ensemble bank is incomplete")
    return one_pass, controls, classical


def family(pairs, sizes):
    """Corrected paired tests of fold losses, Holm-adjusted within the family."""
    tests = {name: corrected_test([x - y for x, y in zip(a, b, strict=True)], sizes)
             for name, (a, b) in pairs.items()}
    for name, adjusted in zip(tests, holm([t["p_value"] for t in tests.values()]), strict=True):
        tests[name]["holm_p_value"] = adjusted
    return tests


def collect(rows, one_pass, classical):
    rows = sorted(rows, key=lambda row: row["fold"])
    sizes = [row["n"] for row in rows]
    single = {variant: sweep.summarise([row["single"][variant] for row in rows])
              for variant in plist.VARIANTS}
    schemes, tests = {}, {}
    for scheme in SCHEMES:
        schemes[scheme] = {}
        for variant in plist.VARIANTS:
            folds = [row["schemes"][scheme][variant] for row in rows]
            members = [mean(values) for values in zip(*(f["member_loss_q2"] for f in folds), strict=True)]
            schemes[scheme][variant] = dict(
                prefix_loss_q2=[mean(values) for values in
                                zip(*(f["prefix_loss_q2"] for f in folds), strict=True)],
                sizes={str(size): sweep.summarise([f["sizes"][size] for f in folds]) for size in SIZES},
                members=dict(mean_loss_q2=mean(members), min_loss_q2=min(members),
                             max_loss_q2=max(members)))
        losses = {(variant, size): schemes[scheme][variant]["sizes"][str(size)]["fold_loss_q2"]
                  for variant in plist.VARIANTS for size in TEST_SIZES}
        tests[scheme] = dict(
            minus_one_pass_medoid=family({
                f"{variant}@{size}": (value, one_pass["medoid"][size]["fold_loss_q2"])
                for (variant, size), value in losses.items()}, sizes),
            minus_classical=family({
                f"{variant}@243-{model}": (losses[variant, 243], classical[model])
                for variant in plist.VARIANTS for model in TESTED_CLASSICAL}, sizes),
            minus_single_list=family({
                variant: (losses[variant, 243], single[variant]["fold_loss_q2"])
                for variant in plist.VARIANTS}, sizes))
    return dict(
        fold_sizes=sizes, single_list=single,
        classical={model: dict(loss_q2=mean(values), fold_loss_q2=values)
                   for model, values in classical.items()},
        one_pass_ensemble={kind: {str(size): row for size, row in by_size.items()}
                           for kind, by_size in one_pass.items()},
        schemes=schemes, tests=tests)


def run(source_commit, workers, *, archive=False):
    provenance, inputs = oracle.load_splits()
    datasets = {split: data for split, (data, _, _) in inputs.items()}
    with db.get_connection() as conn:
        conn.read_only = True
        one_pass, controls, classical = archived_references(conn, datasets)
    jobs = [(split, fold) for split in datasets for fold in matched.FOLDS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise, initargs=(datasets, controls)) as pool:
        rows = list(pool.map(fold_job, jobs))
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=P, precision=matched.PRECISION, evaluation_q=2, fitting_q=FIT_Q,
        consensus_q=P, members=len(matched.BANK_SEEDS), sizes=list(SIZES),
        seed_bases=list(matched.BANK_SEEDS),
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT,
                        one_pass_ensembles=valuation_ensembles.EXPERIMENT),
        protocol="Each member is a priority list. Masked members keep the archived one-pass "
                 "bank's feature mask for their seed and order the kept tags by taxonomy "
                 "association; random-order members keep every tag and shuffle them with the "
                 "member seed plus the fold. Every member is filled in three ways: cell medoids, "
                 "cell medoids backfilled with pooled values, and pooled values alone. The ensemble "
                 "predicts the fitting-fold path with least total native 71-adic distance to the "
                 "member predictions, smallest code on ties, for every prefix of the fixed roster. "
                 "The first archived one-pass member is refitted on every fold and must reproduce "
                 "its archived minimum-valuation score. Folds have equal weight.",
        **provenance,
        splits={split: collect([r for r in rows if r["split"] == split], one_pass[split],
                               classical[split]) for split in datasets})
    return oracle.save_report(report) if archive else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--archive", action="store_true",
                        help="also insert the report into padjective.paper_low_q_reports")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    report = run(args.source_commit, args.workers, archive=args.archive)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
