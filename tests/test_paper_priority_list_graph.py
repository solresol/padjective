import random

import networkx as nx
import numpy as np
import pytest

from padjective import paper_priority_list_graph as graph
from padjective.benchmark_runtime import ProductRecord

P = 71


def record(index, tags, target, positions=()):
    return ProductRecord(index, f"p{index}", list(tags), target, 1, str(target), 1, tuple(positions))


def toy(seed, n=80):
    rng = random.Random(seed)
    tags = [f"t{k:02d}" for k in range(14)]
    rows = []
    for i in range(n):
        chosen = rng.sample(tags, rng.randint(1, 4))
        positions = [(tag, 0, rng.randint(0, 30)) for tag in rng.sample(chosen, rng.randint(0, len(chosen)))]
        rows.append(record(i, chosen, rng.choice([1, 2, 1 + P, 2 + P]), positions))
    return rows


def test_graphs_count_shared_products_paths_and_battles():
    training = [record(0, ["a", "b"], 1, [("a", 0, 1), ("b", 0, 5)]),
                record(1, ["a", "b", "c"], 2, [("b", 0, 2), ("a", 0, 9)]),
                record(2, ["c"], 2), record(3, ["d"], 1)]
    co = graph.cooccurrence_graph(training)
    assert co["a"]["b"]["weight"] == 2 and co["a"]["c"]["weight"] == 1
    assert sorted(co) == ["a", "b", "c", "d"] and co.degree("d") == 0
    paths = graph.tag_path_graph(training)
    assert paths["tag", "c"]["path", 2]["weight"] == 2
    assert paths.degree(("tag", "a")) == 2 and paths.degree(("tag", "d")) == 1
    battles = graph.battle_graph(training)
    # The tag further right wins: b beats a in the first title, a beats b in the second.
    assert battles["a"]["b"]["weight"] == 1 and battles["b"]["a"]["weight"] == 1
    assert sorted(battles) == ["a", "b", "c", "d"]


def test_a_hub_tag_is_most_central_in_the_cooccurrence_graph():
    training = [record(i, ["hub", f"leaf{i}"], 1 + i % 3) for i in range(8)]
    scores, counts, facts, _ = graph.centralities(training)
    for name in ("cooccurrence_degree", "cooccurrence_strength", "cooccurrence_pagerank",
                 "cooccurrence_betweenness", "cooccurrence_eigenvector", "path_degree",
                 "path_betweenness", "frequency"):
        assert max(scores[name], key=scores[name].get) == "hub", name
        assert graph.centrality_order(scores[name], counts, descending=True)[0] == "hub"
        assert graph.centrality_order(scores[name], counts, descending=False)[-1] == "hub"
    assert facts["tags"] == 9 and facts["cooccurrence_components"] == 1
    assert facts["single_path_tags"] == 8


def test_orders_break_ties_by_size_then_name():
    counts = {"a": 2, "b": 5, "c": 5, "d": 1}
    score = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 1.0}
    assert graph.centrality_order(score, counts, descending=False) == ["b", "c", "a", "d"]
    assert graph.centrality_order(score, counts, descending=True) == ["d", "b", "c", "a"]


@pytest.mark.parametrize("seed", range(3))
def test_every_tag_gets_a_finite_score_and_scores_match_networkx(seed):
    training = toy(seed)
    scores, counts, _, _ = graph.centralities(training)
    tags = sorted(counts)
    for name, score in scores.items():
        assert sorted(score) == tags, name
        assert all(np.isfinite(score[tag]) for tag in tags), name
    co = graph.cooccurrence_graph(training)
    if nx.is_connected(co):
        reference = nx.eigenvector_centrality_numpy(co, weight="weight")
        ours = scores["cooccurrence_eigenvector"]
        scale = sum(reference.values()) / sum(ours.values())
        assert all(abs(ours[tag] * scale - reference[tag]) < 1e-8 for tag in tags)
    again, _, _, _ = graph.centralities(training)
    assert all(again[name] == pytest.approx(scores[name]) for name in scores)


def test_completeness_scores_separate_a_bridge_from_a_clique_member():
    # Two triangles of specific tags, joined only through "bridge".
    training = [record(0, ["a1", "a2", "a3", "bridge"], 1), record(1, ["a1", "a2", "a3"], 1),
                record(2, ["b1", "b2", "b3", "bridge"], 2), record(3, ["b1", "b2", "b3"], 2)]
    scores, counts, facts, placed = graph.centralities(training)
    clustering = scores["cooccurrence_clustering"]
    assert clustering["a1"] == 1 and clustering["bridge"] == pytest.approx(6 / 15)
    assert scores["path_shared_prefix"]["a1"] == 7 and scores["path_shared_prefix"]["bridge"] == 0
    assert scores["path_dispersion"]["a1"] == 0 and scores["path_dispersion"]["bridge"] == pytest.approx(0.5)
    # Every tag but the bridge is pure, and peeling them leaves nothing for the bridge to head.
    assert "bridge" not in placed and len(placed) == 6
    assert facts["kernel_products"] == 0 and facts["peeled_products"] == 4


def test_shared_prefix_counts_agreeing_leading_digits():
    assert graph.shared_prefix({1 + 2 * P + 3 * P * P, 1 + 2 * P + 5 * P * P}) == 2
    assert graph.shared_prefix({1, 2}) == 0
    assert graph.shared_prefix({1 + P}) == 7
