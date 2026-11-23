"""Compare each one-pass tag order with the same rule reapplied to the products left."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

from . import benchmark_runtime as runtime
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_sum_oracle as oracle

EXPERIMENT = "paper-priority-list-dynamic-2026-11-23"
RULES = ("taxonomy_association", "battle_elo", "mean_title_position", "frequency")
FIT_Q = 71
TESTED = ("cell", "cell_backfill")
EXTRA = ("stored", "train_loss_q2", "heads_changed")
_datasets = None


def initialise(datasets):
    global _datasets
    _datasets = datasets


def static_rule(strategy, fold):
    """The one-pass ordering rule as a function of the records it may look at."""
    def rule(records):
        battles = runtime._derive_battles(records) if strategy == "battle_elo" else []
        return runtime._tag_order(records, battles, fold, strategy=strategy)
    return rule


def heads(records, order):
    rank = {tag: position for position, tag in enumerate(order)}
    return [plist.head(record.tags, rank, rank) for record in records]


def fold_job(job):
    split, fold = job
    dataset = _datasets[split]
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    pooled, default = plist.fit_pooled(training, prime=matched.PRIME, q=FIT_Q)
    rows = {}
    for strategy in RULES:
        rule = static_rule(strategy, fold)
        static = rule(training)
        dynamic = plist.dynamic_order(training, rule, static)
        if static[0] != dynamic[0] or sorted(static) != sorted(dynamic):
            raise ValueError(f"Dynamic {strategy} order is not a reordering that starts like the static one")
        static_heads = heads(testing, static)
        for kind, order in (("static", static), ("dynamic", dynamic)):
            changed = sum(a != b for a, b in zip(static_heads, heads(testing, order), strict=True))
            scores = plist.list_rows(training, testing, order, pooled, default,
                                     prime=matched.PRIME, q=FIT_Q)
            rows[strategy, kind] = {variant: dict(fold=fold, heads_changed=changed / len(testing), **row)
                                    for variant, row in scores.items()}
    return dict(split=split, fold=fold, n=len(testing), rows=rows)


def collect(results):
    results = sorted(results, key=lambda r: r["fold"])
    sizes = [r["n"] for r in results]
    orders = {strategy: {kind: {variant: plist.summarise(
        [r["rows"][strategy, kind][variant] for r in results], EXTRA)
        for variant in plist.VARIANTS} for kind in ("static", "dynamic")} for strategy in RULES}
    keys = [(strategy, variant) for strategy in RULES for variant in TESTED]
    tests = plist.paired(
        {f"{s}:{v}": orders[s]["dynamic"][v] for s, v in keys},
        {f"{s}:{v}": orders[s]["static"][v] for s, v in keys}, sizes)
    return dict(fold_sizes=sizes, orders=orders, dynamic_minus_static=tests)


def run(source_commit, workers, *, archive=False):
    provenance, inputs = oracle.load_splits()
    datasets = {split: data for split, (data, _, _) in inputs.items()}
    jobs = [(split, fold) for split in datasets for fold in matched.FOLDS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise, initargs=(datasets,)) as pool:
        results = list(pool.map(fold_job, jobs))
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=FIT_Q, rules=list(RULES),
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT),
        protocol="The static order applies a one-pass ordering rule once to the fitting fold. The "
                 "dynamic order applies the same rule to the fitting products that have none of "
                 "the tags placed so far, places the first tag it returns and repeats; tags that "
                 "head nothing follow in static order. Battle Elo is recomputed from the battles of "
                 "the remaining products at every step. Both orders are then fitted as priority "
                 "lists with cell medoids, backfilled cell medoids and pooled values, exactly as in "
                 "the priority-list experiment. Folds have equal weight.",
        **provenance,
        splits={split: collect([r for r in results if r["split"] == split]) for split in datasets})
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
