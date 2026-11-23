"""Order tags by graph centrality and fit the orders as priority lists.

Three graphs are built from the fitting fold only. The co-occurrence graph
joins tags that share a product and uses no labels. The tag-path graph joins
each tag to the taxonomy paths of its products. The battle graph points from
the losing tag of each title battle to the winner.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
from statistics import mean

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh, svds
from scipy.stats import spearmanr

from . import benchmark_runtime as runtime
from . import paper_matched_store_folds as matched
from . import paper_priority_list as plist
from . import paper_sum_oracle as oracle

EXPERIMENT = "paper-priority-list-graph-2026-11-23"
FIT_Q = 71
REFERENCE = "taxonomy_association"
TESTED = ("cell", "cell_backfill")
EXTRA = ("stored", "train_loss_q2")
_datasets = None


def initialise(datasets):
    global _datasets
    _datasets = datasets


def cooccurrence_graph(training):
    """Tags joined when a fitting product carries both, weighted by the number that do."""
    graph = nx.Graph()
    weights = Counter()
    for record in training:
        tags = sorted(set(record.tags))
        graph.add_nodes_from(tags)
        weights.update((a, b) for i, a in enumerate(tags) for b in tags[i + 1:])
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in weights.items())
    return graph


def tag_path_graph(training):
    """Bipartite graph of tags and taxonomy paths, weighted by the products they share."""
    graph = nx.Graph()
    weights = Counter((("tag", tag), ("path", record.encoded_path))
                      for record in training for tag in set(record.tags))
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in weights.items())
    return graph


def battle_graph(training):
    """Directed graph from the loser of each title battle to its winner."""
    graph = nx.DiGraph()
    graph.add_nodes_from({tag for record in training for tag in record.tags})
    weights = Counter((b.loser_tag, b.winner_tag) for b in runtime._derive_battles(training))
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in weights.items())
    return graph


def leading_vector(matrix, *, singular=False):
    """Absolute leading eigenvector, or left singular vector, from a fixed start."""
    if singular:
        left, _, _ = svds(matrix, k=1, v0=np.ones(min(matrix.shape)))
        return np.abs(left[:, 0])
    _, vectors = eigsh(matrix, k=1, which="LA", v0=np.ones(matrix.shape[0]))
    return np.abs(vectors[:, 0])


def centralities(training):
    """Centrality of every fitting-fold tag in each graph."""
    co = cooccurrence_graph(training)
    tags = sorted(co)
    scores = dict(
        cooccurrence_degree=dict(co.degree()),
        cooccurrence_strength=dict(co.degree(weight="weight")),
        cooccurrence_pagerank=nx.pagerank(co, weight="weight"),
        cooccurrence_betweenness=nx.betweenness_centrality(co),
        cooccurrence_eigenvector=dict(zip(tags, leading_vector(
            nx.to_scipy_sparse_array(co, nodelist=tags, weight="weight", dtype=float)), strict=True)))

    paths = tag_path_graph(training)
    codes = sorted(node for node in paths if node[0] == "path")
    incidence = nx.bipartite.biadjacency_matrix(
        paths, [("tag", tag) for tag in tags], codes, weight="weight", dtype=float)
    between = nx.betweenness_centrality(paths)
    rank = nx.pagerank(paths, weight="weight")
    scores.update(
        path_degree={tag: paths.degree(("tag", tag)) for tag in tags},
        path_betweenness={tag: between["tag", tag] for tag in tags},
        path_pagerank={tag: rank["tag", tag] for tag in tags},
        path_eigenvector=dict(zip(tags, leading_vector(incidence, singular=True), strict=True)))

    scores["battle_pagerank"] = nx.pagerank(battle_graph(training), weight="weight")
    counts = Counter(tag for record in training for tag in set(record.tags))
    scores["frequency"] = dict(counts)
    facts = dict(
        tags=len(tags), products=len(training),
        tags_per_product=mean(len(set(r.tags)) for r in training),
        cooccurrence_edges=co.number_of_edges(),
        cooccurrence_components=nx.number_connected_components(co),
        largest_component=len(max(nx.connected_components(co), key=len)),
        tag_path_edges=paths.number_of_edges(),
        single_path_tags=sum(paths.degree(("tag", tag)) == 1 for tag in tags))
    return scores, counts, facts


def centrality_order(score, counts, *, descending):
    """Sort tags by centrality, then larger tags, then name. Scores are rounded to ten digits."""
    sign = -1 if descending else 1
    return sorted(counts, key=lambda tag: (sign * float(f"{score[tag]:.10g}"), -counts[tag], tag))


def fold_job(job):
    split, fold = job
    dataset = _datasets[split]
    training = [r for r in dataset.records if r.cv_fold != fold]
    testing = [r for r in dataset.records if r.cv_fold == fold]
    pooled, default = plist.fit_pooled(training, prime=matched.PRIME, q=FIT_Q)

    def fit(order):
        return plist.list_rows(training, testing, order, pooled, default, prime=matched.PRIME, q=FIT_Q)

    reference = runtime._tag_order(training, [], fold, strategy=REFERENCE)
    rows = {REFERENCE: [fit(reference)]}
    rows["random"] = [fit(runtime._tag_order(training, [], fold, strategy="random", seed=seed))
                      for seed in runtime.DEFAULT_RANDOM_SEEDS]
    scores, counts, facts = centralities(training)
    position = {tag: i for i, tag in enumerate(reference)}
    correlation = {}
    for name, score in scores.items():
        correlation[name] = float(spearmanr([score[tag] for tag in reference],
                                            [position[tag] for tag in reference]).statistic)
        for direction in ("ascending", "descending"):
            order = centrality_order(score, counts, descending=direction == "descending")
            rows[f"{name}:{direction}"] = [fit(order)]
    return dict(split=split, fold=fold, n=len(testing), rows=rows, facts=facts, correlation=correlation)


def collect(results):
    results = sorted(results, key=lambda r: r["fold"])
    sizes = [r["n"] for r in results]
    orders = {}
    for name in results[0]["rows"]:
        orders[name] = {variant: plist.summarise(
            [dict(fold=r["fold"], **fit[variant]) for r in results for fit in r["rows"][name]], EXTRA)
            for variant in plist.VARIANTS}
    graph = [name for name in orders if ":" in name]
    tests = {}
    for baseline in (REFERENCE, "random"):
        keys = [(name, variant) for name in graph for variant in TESTED]
        tests[f"minus_{baseline}"] = plist.paired(
            {f"{n}|{v}": orders[n][v] for n, v in keys},
            {f"{n}|{v}": orders[baseline][v] for n, v in keys}, sizes)
    return dict(
        fold_sizes=sizes, orders=orders, tests=tests,
        facts={key: mean(r["facts"][key] for r in results) for key in results[0]["facts"]},
        spearman_with_reference_position={
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
        valuation_prime=matched.PRIME, evaluation_q=2, fitting_q=FIT_Q,
        references=dict(product=oracle.PRODUCT_REFERENCE, store=matched.EXPERIMENT),
        protocol="Three graphs are built from each fitting fold: tags joined by shared products, "
                 "tags joined to the taxonomy paths of their products, and title battles directed "
                 "from loser to winner. Degree, strength, PageRank with damping 0.85, leading "
                 "eigenvector and unweighted betweenness give a score for every tag. Tags are "
                 "sorted by each score in both directions, with ties to the larger tag and then the "
                 "tag name, and each order is fitted as a priority list with cell medoids, "
                 "backfilled cell medoids and pooled values. Taxonomy association and five random "
                 "orders are refitted as references. Folds have equal weight.",
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
