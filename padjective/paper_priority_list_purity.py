"""Order a priority list by how tightly each tag's labels sit in the taxonomy.

Taxonomy association ranks a tag by the share of its products on its most
common path, so every other path counts as equally wrong. Here a tag's noise
is the mean distance of its labels to their medoid under the tree distance
at weight ``q``, which is the loss of predicting that medoid for the tag's
own products. As ``q`` falls to one every distance tends to one and the
noise becomes one minus the taxonomy association. A large ``q`` counts only
the top of the tree. The least noisy tag decides first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from math import fsum, log
import multiprocessing
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr

from . import benchmark_runtime as runtime
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_priority_list_graph as graph
from . import paper_priority_list_levels as levels
from . import paper_sum_oracle as oracle

EXPERIMENT = "paper-priority-list-purity-2026-11-27"
P, DEPTH = matched.PRIME, matched.PRECISION
REFERENCE = "association"
CELL_Q = (71, 2)
SPREAD_Q = (1.25, 1.5, 2, 3, 4, 8, 16, 71)
SHRUNK_Q = PAIRWISE_Q = (2, 71)
BALL_SHARES = (1.0, 0.9, 0.75)
TESTED = "cell_backfill"
EXTRA = ("stored", "train_loss_q2")
_datasets = None


def initialise(datasets):
    global _datasets
    _datasets = datasets


def residues(counts, *, prime=P, depth=DEPTH):
    """Counts of a multiset of codes by their first ``k`` digits, for ``k`` from zero to ``depth``."""
    if not counts or min(counts) < 0 or max(counts) >= prime ** depth:
        raise ValueError("Codes must be nonnegative and shorter than the depth")
    result = []
    for k in range(depth + 1):
        level = Counter()
        for code, n in counts.items():
            level[code % prime ** k] += n
        result.append(level)
    return result


def spread(counts, q, *, prime=P, depth=DEPTH):
    """Mean distance from a multiset of codes to its medoid, where codes that first differ at digit ``k`` are ``q**-k`` apart."""
    by_depth = residues(counts, prime=prime, depth=depth)

    def cost(code):
        return fsum((shallow[code % prime ** k] - deep[code % prime ** (k + 1)]) * q ** -k
                    for k, (shallow, deep) in enumerate(zip(by_depth, by_depth[1:])))

    return min(cost(code) for code in counts) / sum(counts.values())


def exact_match_noise(counts):
    """Share of codes that differ from the most common one: the spread under the discrete metric."""
    return 1 - max(counts.values()) / sum(counts.values())


def pairwise(counts, q, *, prime=P, depth=DEPTH):
    """Mean distance between two different members of a multiset of codes; zero for a single member."""
    n = sum(counts.values())
    if n < 2:
        return 0.0
    agreeing = [sum(v * v for v in level.values()) for level in residues(counts, prime=prime, depth=depth)]
    return fsum((shallow - deep) * q ** -k
                for k, (shallow, deep) in enumerate(zip(agreeing, agreeing[1:]))) / (n * (n - 1))


def ball_depth(counts, share, *, prime=P, depth=DEPTH):
    """Number of leading digits on which at least ``share`` of the codes agree."""
    n = sum(counts.values())
    return max(k for k, level in enumerate(residues(counts, prime=prime, depth=depth))
               if max(level.values()) >= share * n - 1e-9)


def entropy(counts):
    """Shannon entropy of the exact codes, in nats."""
    n = sum(counts.values())
    return -fsum(v / n * log(v / n) for v in counts.values())


def tag_labels(records):
    """The multiset of label codes under every tag."""
    labels = defaultdict(Counter)
    for record in records:
        for tag in set(record.tags):
            labels[tag][record.encoded_path] += 1
    return labels


def noise_scores(records, *, prime=P, depth=DEPTH):
    """Every noise score for every tag; lower is purer."""
    labels = tag_labels(records)
    sizes = {tag: sum(seen.values()) for tag, seen in labels.items()}
    everything = Counter(record.encoded_path for record in records)

    def per_tag(score):
        return {tag: score(seen) for tag, seen in labels.items()}

    scores = dict(exact_match=per_tag(exact_match_noise), entropy=per_tag(entropy))
    for q in SPREAD_Q:
        scores[f"spread_q{q:g}"] = per_tag(lambda seen: spread(seen, q, prime=prime, depth=depth))
    for q in SHRUNK_Q:
        prior = spread(everything, q, prime=prime, depth=depth)
        plain = scores[f"spread_q{q:g}"]
        scores[f"spread_q{q:g}_shrunk"] = {
            tag: (sizes[tag] * plain[tag] + prior) / (sizes[tag] + 1) for tag in labels}
    for q in PAIRWISE_Q:
        scores[f"pairwise_q{q:g}"] = per_tag(lambda seen: pairwise(seen, q, prime=prime, depth=depth))
    # The depth has few values, so ties within a depth go to the smaller spread, which is below one.
    native = scores[f"spread_q{prime:g}"] if prime in SPREAD_Q else per_tag(
        lambda seen: spread(seen, prime, prime=prime, depth=depth))
    for share in BALL_SHARES:
        scores[f"ball_{share:g}"] = {
            tag: native[tag] - ball_depth(seen, share, prime=prime, depth=depth)
            for tag, seen in labels.items()}
    return scores, sizes


def purity_orders(records, *, prime=P, depth=DEPTH):
    """Tags from least to most noisy under each score, with ties to the larger tag and then the name."""
    scores, sizes = noise_scores(records, prime=prime, depth=depth)
    return {name: graph.centrality_order(score, sizes, descending=False) for name, score in scores.items()}


def fold_job(job):
    split, fold = job
    dataset = _datasets[split]
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    reference = runtime._tag_order(training, [], fold, strategy="taxonomy_association")
    orders = purity_orders(training)
    checks = dict(exact_match_is_association=orders.pop("exact_match") == reference,
                  spread_q71_is_taxonomy_spread=orders["spread_q71"] == levels.spread_order(training))
    orders = {REFERENCE: reference, **orders}
    position = {tag: i for i, tag in enumerate(reference)}
    correlation = {name: float(spearmanr([position[tag] for tag in order], range(len(order))).statistic)
                   for name, order in orders.items()}
    rows = {}
    for q in CELL_Q:
        pooled, default = plist.fit_pooled(training, prime=P, q=q)
        for name, order in orders.items():
            fitted = plist.list_rows(training, testing, order, pooled, default, prime=P, q=q)
            for row in fitted.values():
                row["prefix2_accuracy"] = levels.prefix_accuracy(row, 2)
            rows[f"{name}|cells_q{q}"] = fitted
    return dict(split=split, fold=fold, n=len(testing), rows=rows, checks=checks, correlation=correlation)


def collect(results):
    results = sorted(results, key=lambda r: r["fold"])
    sizes = [r["n"] for r in results]
    fits = {key: {variant: plist.summarise(
        [dict(fold=r["fold"], **r["rows"][key][variant]) for r in results], (*EXTRA, "prefix2_accuracy"))
        for variant in plist.VARIANTS} for key in results[0]["rows"]}
    compared = [key for key in fits if not key.startswith(f"{REFERENCE}|")]
    tests = plist.paired(
        {key: fits[key][TESTED] for key in compared},
        {key: fits[f"{REFERENCE}|{key.split('|')[1]}"][TESTED] for key in compared}, sizes)
    for key, test in tests.items():
        differences = [a["loss_q2"] - b["loss_q2"] for a, b in zip(
            fits[key][TESTED]["folds"], fits[f"{REFERENCE}|{key.split('|')[1]}"][TESTED]["folds"], strict=True)]
        test["folds_lower"] = sum(d < 0 for d in differences)
    return dict(
        fold_sizes=sizes, fits=fits, tests_minus_association=tests,
        checks={key: all(r["checks"][key] for r in results) for key in results[0]["checks"]},
        spearman_with_association={
            key: mean(r["correlation"][key] for r in results) for key in results[0]["correlation"]})


def run(source_commit, workers, *, archive=False):
    provenance, inputs = oracle.load_splits()
    datasets = {split: data for split, (data, _, _) in inputs.items()}
    jobs = [(split, fold) for split in datasets for fold in matched.FOLDS]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialise, initargs=(datasets,)) as pool:
        results = list(pool.map(fold_job, jobs))
    report = dict(
        experiment_key=EXPERIMENT, status="exploratory", source_commit=source_commit,
        valuation_prime=P, evaluation_q=2, cell_fitting_q=list(CELL_Q), spread_q=list(SPREAD_Q),
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT),
        protocol="Each score gives every tag a noise from the labels of its fitting products, and "
                 "tags are sorted from least to most noisy with ties to the larger tag and then the "
                 "tag name. Spread is the mean distance of a tag's labels to their medoid when "
                 "labels that first differ at digit k are q to the power minus k apart. The shrunk "
                 "spread adds one pseudo-product at the spread of all fitting labels. Pairwise is "
                 "the mean distance between two different products of the tag. Ball is the number "
                 "of leading digits on which the stated share of a tag's labels agree, longest "
                 "first, with ties to the smaller spread at weight 71. Entropy is the Shannon "
                 "entropy of the exact labels. Each order is fitted as a priority list with cell "
                 "medoids, backfilled cell medoids and pooled values, with cells fitted at weight "
                 "71 and at weight 2. Taxonomy association is refitted as the reference and must "
                 "equal the order by spread under the discrete metric. Folds have equal weight.",
        **provenance,
        splits={split: collect([r for r in results if r["split"] == split]) for split in datasets})
    return oracle.save_report(report) if archive else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--archive", action="store_true",
                        help="also insert the report into padjective.paper_low_q_reports")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("Workers must be positive")
    report = run(args.source_commit, args.workers, archive=args.archive)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
