"""Decide the taxonomy path one level at a time, with a priority list at each node.

A plain priority list lets one tag fix the whole path. Here the root has its
own list for the top-level digit, ordered by how pure each tag is at that
level. Each child has a list for the next digit, fitted on the products under
it, and so on down to a chosen depth. Below that depth a plain list predicts
the rest of the path. Depth zero is the plain list.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
from statistics import mean

from . import benchmark_runtime as runtime, db
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_priority_list_graph as graph
from . import paper_sum_oracle as oracle
from .q_loss_rescore import load_predictions
from .subspace_voting import corrected_test, holm

EXPERIMENT = "paper-priority-list-levels-2026-11-24"
FIT_Q = 71
P, FULL = matched.PRIME, matched.PRECISION
REFERENCES = ("list", "spread_list")
VARIANTS = {
    "list": dict(limit=0),
    "levels_1": dict(limit=1),
    "levels_2": dict(limit=2),
    "levels_3": dict(limit=3),
    "levels_1_vote": dict(limit=1, select="vote"),
    "spread_list": dict(limit=0, terminal="spread"),
    "spread_levels_1": dict(limit=1, terminal="spread"),
    "spread_levels_2": dict(limit=2, terminal="spread"),
    "spread_levels_3": dict(limit=3, terminal="spread"),
    "levels_all": dict(limit=FULL),
    "levels_all_vote": dict(limit=FULL, select="vote"),
    "levels_all_consistent": dict(limit=FULL, consistent=True),
    "levels_all_shared_order": dict(limit=FULL, shared=True),
}
EXTRA = ("prefix2_accuracy", "prefix3_accuracy", "train_loss_q2", "nodes")
CLASSICAL = ("ulr", "unn")
SHOWN = ("loss_q2", "root_accuracy", "exact_accuracy")
_datasets = None


def initialise(datasets):
    global _datasets
    _datasets = datasets


def digit(code, level):
    return code // P ** level % P


def spread_order(records):
    """Tags by the mean distance of their labels to their medoid, tightest first."""
    labels = defaultdict(Counter)
    for record in records:
        for tag in set(record.tags):
            labels[tag][record.encoded_path] += 1
    counts = {tag: sum(seen.values()) for tag, seen in labels.items()}
    scale = float(FIT_Q) ** plist.DEPTH_CAP
    score = {tag: plist.medoid_cost(seen, prime=P, q=FIT_Q)[1] / scale / counts[tag]
             for tag, seen in labels.items()}
    return graph.centrality_order(score, counts, descending=False)


TERMINAL_ORDERS = dict(association=runtime._taxonomy_association_order, spread=spread_order)


def choose(node, tags):
    """The next digit at a split node, and whether a tag decided it."""
    values = node["values"]
    if node["select"] == "vote":
        votes = Counter(values[tag] for tag in set(tags) if tag in values)
        if not votes:
            return node["default"], False
        most = max(votes.values())
        return min(value for value, count in votes.items() if count == most), True
    tag = plist.head(tags, node["rank"], values)
    return (node["default"], False) if tag is None else (values[tag], True)


def fit_node(records, level=0, *, limit, terminal="association", select="priority",
             consistent=False, shared_order=None):
    """Fit the model for the products under one taxonomy node.

    A node whose products share one label predicts it. Above ``limit`` a node
    splits: it fits a backfilled priority list for the next base-71 digit,
    with tags ordered by the share of their products on their most common
    digit, and each digit gets a child fitted on the products that have it.
    A vote node instead gives each tag its most common digit and takes the
    plurality. With ``consistent`` a child is fitted only on the products
    this node sends to it correctly. From ``limit`` down a node is a plain
    backfilled list for the whole remaining path.
    """
    labels = {record.encoded_path for record in records}
    if len(labels) == 1:
        return dict(kind="leaf", value=next(iter(labels)))
    if level >= limit:
        pooled, default = plist.fit_pooled(records, prime=P, q=FIT_Q)
        rank, filled = plist.fill(records, TERMINAL_ORDERS[terminal](records), pooled, prime=P, q=FIT_Q)
        return dict(kind="list", rank=rank, values=filled["cell_backfill"], default=default)

    digits = [digit(record.encoded_path, level) for record in records]
    local = [replace(record, encoded_path=value, taxonomy_id=str(value))
             for record, value in zip(records, digits, strict=True)]
    pooled, default = plist.fit_pooled(local, prime=P, q=FIT_Q)
    order = shared_order or runtime._taxonomy_association_order(local)
    rank, filled = plist.fill(local, order, pooled, prime=P, q=FIT_Q)
    node = dict(kind="split", select=select, rank=rank, default=default,
                values=filled["pooled_priority" if select == "vote" else "cell_backfill"])
    sent = [choose(node, record.tags)[0] for record in records] if consistent else digits
    node["children"] = {}
    for value in sorted(set(digits)):
        under = [record for record, own in zip(records, digits) if own == value]
        kept = [record for record, own, got in zip(records, digits, sent) if own == value == got]
        node["children"][value] = fit_node(
            kept or under, level + 1, limit=limit, terminal=terminal, select=select,
            consistent=consistent, shared_order=shared_order)
    return node


def predict(node, tags):
    """The predicted code, and whether a tag decided every step."""
    decided = True
    while node["kind"] == "split":
        value, by_tag = choose(node, tags)
        decided &= by_tag
        node = node["children"][value]
    if node["kind"] == "leaf":
        return node["value"], decided
    tag = plist.head(tags, node["rank"], node["values"])
    return (node["default"], False) if tag is None else (node["values"][tag], decided)


def size(node):
    """Number of split and list nodes."""
    if node["kind"] == "leaf":
        return 0
    return 1 + sum(size(child) for child in node.get("children", {}).values())


def prefix_accuracy(scores, depth):
    """Share of products whose first ``depth`` digits are right."""
    wrong = sum(count for level, count in scores["first_disagreement_counts"].items() if int(level) < depth)
    return 1 - wrong / scores["n"]


def variant_row(training, testing, options, shared_order):
    options = dict(options)
    if options.pop("shared", False):
        options["shared_order"] = shared_order
    model = fit_node(training, **options)
    targets = [record.encoded_path for record in testing]
    row = plist.scored(targets, *zip(*(predict(model, record.tags) for record in testing), strict=True))
    fitted = matched.SCORE([record.encoded_path for record in training],
                           [predict(model, record.tags)[0] for record in training])
    row.update(prefix2_accuracy=prefix_accuracy(row, 2), prefix3_accuracy=prefix_accuracy(row, 3),
               train_loss_q2=fitted["loss_q2"], nodes=size(model))
    return row


def fold_job(job):
    split, fold = job
    dataset = _datasets[split]
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    shared_order = runtime._tag_order(training, [], fold, strategy="taxonomy_association")
    return dict(split=split, fold=fold, n=len(testing), rows={
        name: variant_row(training, testing, options, shared_order) for name, options in VARIANTS.items()})


def classical_scores(conn, datasets):
    """Archived per-fold scores of L1 logistic regression and the neural network."""
    stored = {m["model"]: m for m in oracle.archived_report(conn, matched.EXPERIMENT)["models"]}
    return dict(
        product={key: [matched.SCORE(*load_predictions(conn, datasets["product"], key, fold))
                       for fold in matched.FOLDS] for key in CLASSICAL},
        store={key: stored[key]["folds"] for key in CLASSICAL})


def collect(results, classical):
    results = sorted(results, key=lambda r: r["fold"])
    sizes = [r["n"] for r in results]
    fits = {name: plist.summarise([dict(fold=r["fold"], **r["rows"][name]) for r in results], EXTRA)
            for name in results[0]["rows"]}

    def against(reference, names):
        return plist.paired({name: fits[name] for name in names},
                            {name: fits[reference] for name in names}, sizes)

    versus = {f"{name}-{model}": corrected_test(
        [mine["loss_q2"] - theirs["loss_q2"] for mine, theirs in zip(fits[name]["folds"], folds, strict=True)],
        sizes) for model, folds in classical.items() for name in fits}
    for key, adjusted in zip(versus, holm([t["p_value"] for t in versus.values()]), strict=True):
        versus[key]["holm_p_value"] = adjusted
    return dict(
        fold_sizes=sizes, fits=fits,
        classical={model: dict(
            **{key: mean(fold[key] for fold in folds) for key in SHOWN if key in folds[0]},
            fold_loss_q2=[fold["loss_q2"] for fold in folds]) for model, folds in classical.items()},
        tests=dict(
            minus_list=against("list", [name for name in fits if name != "list"]),
            minus_spread_list=against("spread_list", [name for name in fits if name not in REFERENCES]),
            minus_classical=versus))


def run(source_commit, workers, *, archive=False):
    provenance, inputs = oracle.load_splits()
    datasets = {split: data for split, (data, _, _) in inputs.items()}
    with db.get_connection() as conn:
        conn.read_only = True
        classical = classical_scores(conn, datasets)
    jobs = [(split, fold) for split in datasets for fold in matched.FOLDS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise, initargs=(datasets,)) as pool:
        results = list(pool.map(fold_job, jobs))
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=P, evaluation_q=2, fitting_q=FIT_Q,
        variants={name: {key: value for key, value in options.items()} for name, options in VARIANTS.items()},
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT),
        protocol="The model is a tree of priority lists that follows the taxonomy. A node whose "
                 "fitting products share one label predicts it. A split node fits a backfilled "
                 "priority list for the next base-71 digit of the label on the products under it: "
                 "tags are ordered by the share of their products on their most common digit, with "
                 "ties to the larger tag and then the tag name, each tag takes the most common digit "
                 "of the products it heads, and a product with no fitted tag takes the node's most "
                 "common digit. Each digit has a child fitted on the products that have it. Nodes "
                 "at the depth limit and below fit a plain backfilled priority list at weight 71 "
                 "for the whole remaining path, ordered by taxonomy association or by taxonomy "
                 "spread on the products under the node. Limit zero is the plain list. The vote "
                 "variants give each tag its most common digit and take the plurality, with ties "
                 "to the smaller digit. The consistent variant fits each child only on the products "
                 "its parent sends to it correctly. The shared-order variant uses the "
                 "taxonomy-association order of the whole fitting fold at every node. The archived "
                 "L1 logistic regression and neural network are compared on the same folds. Folds "
                 "have equal weight.",
        **provenance,
        splits={split: collect([r for r in results if r["split"] == split], classical[split])
                for split in datasets})
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
