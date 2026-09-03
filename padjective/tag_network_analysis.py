"""Analyse circular structure in a frozen product-taxonomy snapshot.

The primary graph is bipartite: products connect to their retained tags.  A
three-product circular example is therefore a six-cycle.  The secondary graph
is the directed tag-battle graph induced by title order; its weak components
are the independent ranking problems and its strongly connected components
locate contradictory/cyclic ordering evidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence
import uuid

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import data_access, db


DEFAULT_SCHEMA = "padjective"
DEFAULT_SNAPSHOT_REF = "paper"
DEFAULT_DRAWS = 100_000
DEFAULT_SEED = 42
DEFAULT_HUB_CAPS = (1000, 500, 250, 100, 50, 25)
ANALYSIS_VERSION = 1


@dataclass(frozen=True)
class SnapshotMetadata:
    snapshot_id: str
    snapshot_name: str
    created_at: str
    as_of: str | None
    product_count: int
    tag_count: int
    taxonomy_count: int
    min_tag_count: int
    min_samples_per_taxonomy: int
    code_version: str | None
    note: str | None


@dataclass(frozen=True)
class SnapshotGraphData:
    metadata: SnapshotMetadata
    product_tags: dict[str, tuple[str, ...]]
    title_positions: dict[str, tuple[tuple[str, int, int], ...]]


def load_snapshot_graph(
    conn,
    *,
    schema: str = DEFAULT_SCHEMA,
    snapshot_ref: str = DEFAULT_SNAPSHOT_REF,
) -> SnapshotGraphData:
    """Load the frozen incidence and title-position records from Postgres."""

    snapshot_id, _ = data_access._resolve_snapshot_id(
        conn,
        schema=schema,
        snapshot_ref=snapshot_ref,
    )
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    snapshot_id::text AS snapshot_id,
                    snapshot_name,
                    created_at,
                    as_of,
                    product_count,
                    tag_count,
                    taxonomy_count,
                    min_tag_count,
                    min_samples_per_taxonomy,
                    code_version,
                    note
                FROM {schema}.product_taxonomy_bench_snapshots
                WHERE snapshot_id = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Snapshot metadata missing for {snapshot_ref!r}")
        metadata = SnapshotMetadata(
            snapshot_id=str(row["snapshot_id"]),
            snapshot_name=str(row["snapshot_name"]),
            created_at=row["created_at"].isoformat(),
            as_of=row["as_of"].isoformat() if row["as_of"] is not None else None,
            product_count=int(row["product_count"]),
            tag_count=int(row["tag_count"]),
            taxonomy_count=int(row["taxonomy_count"]),
            min_tag_count=int(row["min_tag_count"]),
            min_samples_per_taxonomy=int(row["min_samples_per_taxonomy"]),
            code_version=str(row["code_version"]) if row["code_version"] else None,
            note=str(row["note"]) if row["note"] else None,
        )

        cur.execute(
            sql.SQL(
                """
                SELECT product_id_hash, tag_id
                FROM {schema}.product_taxonomy_bench_product_tags
                WHERE snapshot_id = %s
                ORDER BY product_id_hash, tag_id
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        tags_by_product: dict[str, list[str]] = defaultdict(list)
        for incidence in cur:
            tags_by_product[str(incidence["product_id_hash"])].append(
                str(incidence["tag_id"])
            )

        cur.execute(
            sql.SQL(
                """
                SELECT product_id_hash, tag_id, title_part, title_position
                FROM {schema}.product_taxonomy_bench_title_tags
                WHERE snapshot_id = %s
                ORDER BY product_id_hash, title_part, title_position, tag_id
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        positions_by_product: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        for position in cur:
            positions_by_product[str(position["product_id_hash"])].append(
                (
                    str(position["tag_id"]),
                    int(position["title_part"]),
                    int(position["title_position"]),
                )
            )

    product_tags = {
        product: tuple(sorted(set(tags))) for product, tags in tags_by_product.items()
    }
    if len(product_tags) != metadata.product_count:
        raise ValueError(
            "Snapshot product count disagrees with its incidence table: "
            f"metadata={metadata.product_count}, observed={len(product_tags)}"
        )
    observed_tags = {tag for tags in product_tags.values() for tag in tags}
    if len(observed_tags) != metadata.tag_count:
        raise ValueError(
            "Snapshot tag count disagrees with its incidence table: "
            f"metadata={metadata.tag_count}, observed={len(observed_tags)}"
        )

    return SnapshotGraphData(
        metadata=metadata,
        product_tags=product_tags,
        title_positions={
            product: tuple(positions_by_product.get(product, ()))
            for product in product_tags
        },
    )


def _percentile(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return int(ordered[index])


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p95": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "median": _percentile(values, 0.5),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _connected_components(adjacency: Mapping[str, set[str]]) -> list[set[str]]:
    visited: set[str] = set()
    components: list[set[str]] = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        visited.add(start)
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _two_core(adjacency: Mapping[str, set[str]]) -> set[str]:
    degree = {node: len(neighbours) for node, neighbours in adjacency.items()}
    queue = deque(node for node, value in degree.items() if value < 2)
    removed: set[str] = set()
    while queue:
        node = queue.popleft()
        if node in removed:
            continue
        removed.add(node)
        for neighbour in adjacency[node]:
            if neighbour in removed:
                continue
            degree[neighbour] -= 1
            if degree[neighbour] == 1:
                queue.append(neighbour)
    return set(adjacency) - removed


def _component_record(component: set[str], adjacency: Mapping[str, set[str]]) -> dict[str, int]:
    product_count = sum(node.startswith("p:") for node in component)
    tag_count = sum(node.startswith("t:") for node in component)
    edge_count = sum(len(adjacency[node] & component) for node in component) // 2
    return {
        "products": product_count,
        "tags": tag_count,
        "nodes": len(component),
        "edges": edge_count,
        "cycle_rank": edge_count - len(component) + 1,
    }


def _component_bins(component_products: Sequence[int]) -> list[dict[str, int | str]]:
    bins = (
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-25", 11, 25),
        ("26-100", 26, 100),
        ("101-500", 101, 500),
        ("501+", 501, math.inf),
    )
    return [
        {
            "size": label,
            "components": sum(lower <= value <= upper for value in component_products),
            "products": sum(
                value for value in component_products if lower <= value <= upper
            ),
        }
        for label, lower, upper in bins
    ]


def _build_bipartite_graph(
    product_tags: Mapping[str, Sequence[str]],
    *,
    maximum_tag_degree: int | None = None,
) -> tuple[dict[str, set[str]], set[str]]:
    tag_degrees = Counter(tag for tags in product_tags.values() for tag in tags)
    excluded_tags = {
        tag
        for tag, degree in tag_degrees.items()
        if maximum_tag_degree is not None and degree > maximum_tag_degree
    }
    adjacency: dict[str, set[str]] = {
        f"p:{product}": set() for product in product_tags
    }
    for product, tags in product_tags.items():
        product_node = f"p:{product}"
        for tag in tags:
            if tag in excluded_tags:
                continue
            tag_node = f"t:{tag}"
            adjacency.setdefault(tag_node, set()).add(product_node)
            adjacency[product_node].add(tag_node)
    return adjacency, excluded_tags


def analyse_bipartite_structure(
    product_tags: Mapping[str, Sequence[str]],
    *,
    maximum_tag_degree: int | None = None,
) -> dict[str, object]:
    """Measure components, independent cycles, and the cyclic two-core."""

    adjacency, excluded_tags = _build_bipartite_graph(
        product_tags,
        maximum_tag_degree=maximum_tag_degree,
    )
    components = _connected_components(adjacency)
    records = [_component_record(component, adjacency) for component in components]
    records.sort(key=lambda row: (row["products"], row["tags"], row["edges"]), reverse=True)
    edge_count = sum(len(neighbours) for neighbours in adjacency.values()) // 2
    node_count = len(adjacency)
    cycle_rank = edge_count - node_count + len(components)
    core_nodes = _two_core(adjacency)
    core_edges = sum(len(adjacency[node] & core_nodes) for node in core_nodes) // 2
    core_components = (
        _connected_components(
            {node: adjacency[node] & core_nodes for node in core_nodes}
        )
        if core_nodes
        else []
    )
    total_products = len(product_tags)
    retained_tags = sum(node.startswith("t:") for node in adjacency)
    isolated_products = sum(
        node.startswith("p:") and not neighbours
        for node, neighbours in adjacency.items()
    )
    product_component_sizes = [record["products"] for record in records]
    tag_component_sizes = [record["tags"] for record in records]
    nontrivial_records = [record for record in records if record["edges"] > 0]
    largest = records[0] if records else {
        "products": 0,
        "tags": 0,
        "nodes": 0,
        "edges": 0,
        "cycle_rank": 0,
    }

    return {
        "maximum_tag_degree": maximum_tag_degree,
        "excluded_hub_tags": len(excluded_tags),
        "excluded_tag_ids": sorted(excluded_tags),
        "products": total_products,
        "retained_tags": retained_tags,
        "edges": edge_count,
        "nodes": node_count,
        "components": len(components),
        "nontrivial_components": len(nontrivial_records),
        "isolated_products": isolated_products,
        "largest_component_product_fraction": (
            largest["products"] / total_products if total_products else 0.0
        ),
        "largest_component_tag_fraction": (
            largest["tags"] / retained_tags if retained_tags else 0.0
        ),
        "largest_component": largest,
        "products_outside_largest_component": total_products - largest["products"],
        "tags_outside_largest_component": retained_tags - largest["tags"],
        "cycle_rank": cycle_rank,
        "cycle_rank_per_edge": cycle_rank / edge_count if edge_count else 0.0,
        "two_core": {
            "nodes": len(core_nodes),
            "products": sum(node.startswith("p:") for node in core_nodes),
            "tags": sum(node.startswith("t:") for node in core_nodes),
            "edges": core_edges,
            "components": len(core_components),
            "node_fraction": len(core_nodes) / node_count if node_count else 0.0,
            "edge_fraction": core_edges / edge_count if edge_count else 0.0,
        },
        "component_product_size": _distribution(product_component_sizes),
        "component_tag_size": _distribution(tag_component_sizes),
        "component_bins": _component_bins(product_component_sizes),
        "components_with_at_most_10_products": sum(
            record["products"] <= 10 for record in records
        ),
        "products_in_components_with_at_most_10_products": sum(
            record["products"] for record in records if record["products"] <= 10
        ),
        "components_with_at_most_100_products": sum(
            record["products"] <= 100 for record in records
        ),
        "products_in_components_with_at_most_100_products": sum(
            record["products"] for record in records if record["products"] <= 100
        ),
        "components_with_at_most_10_tags": sum(
            record["tags"] <= 10 for record in records
        ),
        "products_in_components_with_at_most_10_tags": sum(
            record["products"] for record in records if record["tags"] <= 10
        ),
        "top_components": records[:100],
    }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    spread = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def sample_three_product_cycles(
    product_tags: Mapping[str, Sequence[str]],
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    maximum_examples: int = 5,
) -> dict[str, object]:
    """Estimate closure of connected three-product chains and retain witnesses."""

    if draws <= 0:
        raise ValueError("draws must be positive")
    tag_to_products: dict[str, list[str]] = defaultdict(list)
    tag_sets = {product: set(tags) for product, tags in product_tags.items()}
    for product, tags in tag_sets.items():
        for tag in tags:
            tag_to_products[tag].append(product)
    eligible = [
        product
        for product, tags in tag_sets.items()
        if sum(len(tag_to_products[tag]) >= 2 for tag in tags) >= 2
    ]
    if not eligible:
        raise ValueError("No product has two recurrent tags")

    rng = random.Random(seed)
    valid_chains = 0
    closed_chains = 0
    examples: list[dict[str, object]] = []
    example_keys: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for _ in range(draws):
        middle = rng.choice(eligible)
        eligible_tags = [
            tag for tag in tag_sets[middle] if len(tag_to_products[tag]) >= 2
        ]
        left_tag, right_tag = rng.sample(sorted(eligible_tags), 2)
        left = rng.choice(
            [product for product in tag_to_products[left_tag] if product != middle]
        )
        right_candidates = [
            product
            for product in tag_to_products[right_tag]
            if product not in {middle, left}
        ]
        if not right_candidates:
            continue
        right = rng.choice(right_candidates)
        valid_chains += 1
        closing_tags = sorted(
            (tag_sets[left] & tag_sets[right]) - {left_tag, right_tag}
        )
        if not closing_tags:
            continue
        closed_chains += 1
        closing_tag = closing_tags[0]
        example_key = (
            tuple(sorted((left, middle, right))),
            tuple(sorted((left_tag, right_tag, closing_tag))),
        )
        if len(examples) >= maximum_examples or example_key in example_keys:
            continue
        example_keys.add(example_key)
        examples.append(
            {
                "products": [left, middle, right],
                "tags": [left_tag, right_tag, closing_tag],
                "incidence": {
                    left: [left_tag, closing_tag],
                    middle: [left_tag, right_tag],
                    right: [right_tag, closing_tag],
                },
            }
        )

    interval = _wilson_interval(closed_chains, valid_chains)
    return {
        "draws": draws,
        "valid_chains": valid_chains,
        "closed_chains": closed_chains,
        "closure_rate": closed_chains / valid_chains if valid_chains else 0.0,
        "closure_rate_95ci": list(interval),
        "eligible_middle_products": len(eligible),
        "sampling_definition": (
            "Choose a product with two recurrent tags, follow each tag to a "
            "different endpoint product, then test whether the endpoints share "
            "a third distinct tag."
        ),
        "examples": examples,
    }


def _strongly_connected_components(
    nodes: Iterable[str],
    outgoing: Mapping[str, set[str]],
    incoming: Mapping[str, set[str]],
) -> list[set[str]]:
    """Return SCCs using iterative Kosaraju passes."""

    ordered_nodes = sorted(set(nodes))
    visited: set[str] = set()
    finish_order: list[str] = []
    for start in ordered_nodes:
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[str, bool]] = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                finish_order.append(node)
                continue
            stack.append((node, True))
            for neighbour in sorted(outgoing.get(node, ()), reverse=True):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append((neighbour, False))

    visited.clear()
    components: list[set[str]] = []
    for start in reversed(finish_order):
        if start in visited:
            continue
        visited.add(start)
        component: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            component.add(node)
            for neighbour in incoming.get(node, ()):
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def _choose_distinct_supporters(
    first: Sequence[str], second: Sequence[str], third: Sequence[str]
) -> tuple[str, str, str] | None:
    for left in first:
        for middle in second:
            if middle == left:
                continue
            for right in third:
                if right not in {left, middle}:
                    return left, middle, right
    return None


def analyse_battle_graph(
    all_tags: Iterable[str],
    title_positions: Mapping[str, Sequence[tuple[str, int, int]]],
    *,
    maximum_examples: int = 5,
) -> dict[str, object]:
    """Analyse direction, weak components, SCCs, and exact directed triangles."""

    edge_supporters: dict[tuple[str, str], set[str]] = defaultdict(set)
    battle_occurrences = 0
    for product, positions in title_positions.items():
        by_part: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for tag, part, position in positions:
            by_part[int(part)].append((int(position), tag))
        for ordered in by_part.values():
            ordered.sort()
            for left_index in range(len(ordered)):
                for right_index in range(left_index + 1, len(ordered)):
                    loser = ordered[left_index][1]
                    winner = ordered[right_index][1]
                    if winner == loser:
                        continue
                    battle_occurrences += 1
                    edge_supporters[(winner, loser)].add(product)

    nodes = set(all_tags)
    outgoing: dict[str, set[str]] = {tag: set() for tag in nodes}
    incoming: dict[str, set[str]] = {tag: set() for tag in nodes}
    undirected: dict[str, set[str]] = {tag: set() for tag in nodes}
    for winner, loser in edge_supporters:
        outgoing[winner].add(loser)
        incoming[loser].add(winner)
        undirected[winner].add(loser)
        undirected[loser].add(winner)

    weak_components = _connected_components(undirected)
    weak_components.sort(key=len, reverse=True)
    sccs = _strongly_connected_components(nodes, outgoing, incoming)
    sccs.sort(key=len, reverse=True)
    cyclic_sccs = [component for component in sccs if len(component) > 1]
    reciprocal_pairs = [
        (winner, loser)
        for winner, loser in edge_supporters
        if winner < loser and (loser, winner) in edge_supporters
    ]

    directed_triangles = 0
    triangle_examples: list[dict[str, object]] = []
    for first in sorted(nodes):
        for second in sorted(outgoing[first]):
            for third in sorted(outgoing[second] & incoming[first]):
                if len({first, second, third}) != 3 or first != min(first, second, third):
                    continue
                directed_triangles += 1
                if len(triangle_examples) >= maximum_examples:
                    continue
                supporter_lists = (
                    sorted(edge_supporters[(first, second)]),
                    sorted(edge_supporters[(second, third)]),
                    sorted(edge_supporters[(third, first)]),
                )
                supporters = _choose_distinct_supporters(*supporter_lists)
                if supporters is None:
                    continue
                triangle_examples.append(
                    {
                        "tags": [first, second, third],
                        "direction": [
                            f"{first} > {second}",
                            f"{second} > {third}",
                            f"{third} > {first}",
                        ],
                        "products": list(supporters),
                        "edge_support_counts": [len(values) for values in supporter_lists],
                    }
                )

    unique_undirected_edges = sum(len(neighbours) for neighbours in undirected.values()) // 2
    weak_sizes = [len(component) for component in weak_components]
    scc_sizes = [len(component) for component in sccs]
    active_tags = sum(bool(neighbours) for neighbours in undirected.values())
    nontrivial_weak_components = [
        component for component in weak_components if len(component) > 1
    ]
    largest_weak = weak_sizes[0] if weak_sizes else 0
    largest_scc = scc_sizes[0] if scc_sizes else 0
    return {
        "tags": len(nodes),
        "active_tags": active_tags,
        "inactive_tags": len(nodes) - active_tags,
        "battle_occurrences": battle_occurrences,
        "unique_directed_edges": len(edge_supporters),
        "unique_undirected_edges": unique_undirected_edges,
        "weak_components": len(weak_components),
        "nontrivial_weak_components": len(nontrivial_weak_components),
        "weak_component_size": _distribution(weak_sizes),
        "largest_weak_component_tag_fraction": largest_weak / len(nodes) if nodes else 0.0,
        "largest_weak_component_active_tag_fraction": (
            largest_weak / active_tags if active_tags else 0.0
        ),
        "weak_components_with_at_most_10_tags": sum(size <= 10 for size in weak_sizes),
        "tags_in_weak_components_with_at_most_10_tags": sum(
            size for size in weak_sizes if size <= 10
        ),
        "largest_weak_component_permutation_log10": (
            math.lgamma(largest_weak + 1) / math.log(10) if largest_weak else 0.0
        ),
        "underlying_cycle_rank": (
            unique_undirected_edges - len(nodes) + len(weak_components)
        ),
        "strong_components": len(sccs),
        "strong_component_size": _distribution(scc_sizes),
        "cyclic_strong_components": len(cyclic_sccs),
        "tags_in_cyclic_strong_components": sum(len(component) for component in cyclic_sccs),
        "cyclic_strong_component_tag_fraction_of_active": (
            sum(len(component) for component in cyclic_sccs) / active_tags
            if active_tags
            else 0.0
        ),
        "largest_strong_component_tag_fraction": largest_scc / len(nodes) if nodes else 0.0,
        "reciprocal_tag_pairs": len(reciprocal_pairs),
        "directed_three_tag_cycles": directed_triangles,
        "directed_three_tag_cycle_examples": triangle_examples,
        "top_weak_components": [
            {"tags": len(component), "tag_ids": sorted(component)[:25]}
            for component in weak_components[:100]
        ],
        "top_strong_components": [
            {"tags": len(component), "tag_ids": sorted(component)[:25]}
            for component in sccs[:100]
        ],
    }


def tag_expansion(
    product_tags: Mapping[str, Sequence[str]],
    start_tag: str,
    *,
    maximum_depth: int = 6,
) -> dict[str, object]:
    """Return bounded tag -> product -> tag breadth-first growth."""

    adjacency, _ = _build_bipartite_graph(product_tags)
    start = f"t:{start_tag}"
    if start not in adjacency:
        raise ValueError(f"Unknown tag: {start_tag!r}")
    visited = {start}
    frontier = {start}
    levels: list[dict[str, int]] = []
    for depth in range(1, maximum_depth + 1):
        following = {
            neighbour
            for node in frontier
            for neighbour in adjacency[node]
            if neighbour not in visited
        }
        visited.update(following)
        levels.append(
            {
                "depth": depth,
                "new_products": sum(node.startswith("p:") for node in following),
                "new_tags": sum(node.startswith("t:") for node in following),
                "cumulative_products": sum(node.startswith("p:") for node in visited),
                "cumulative_tags": sum(node.startswith("t:") for node in visited),
            }
        )
        frontier = following
        if not frontier:
            break
    return {"start_tag": start_tag, "levels": levels}


def _representative_expansions(
    product_tags: Mapping[str, Sequence[str]],
    cycle_sample: Mapping[str, object],
) -> list[dict[str, object]]:
    degrees = Counter(tag for tags in product_tags.values() for tag in tags)
    ordered = sorted(degrees, key=lambda tag: (degrees[tag], tag))
    candidates = [ordered[-1], ordered[len(ordered) // 2]] if ordered else []
    examples = cycle_sample.get("examples") or []
    if examples:
        candidates.append(str(examples[0]["tags"][0]))
    result: list[dict[str, object]] = []
    for tag in dict.fromkeys(candidates):
        expansion = tag_expansion(product_tags, tag)
        expansion["degree"] = degrees[tag]
        result.append(expansion)
    return result


def _decision_summary(
    bipartite: Mapping[str, object],
    battle: Mapping[str, object],
    sensitivity: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    largest_product_fraction = float(bipartite["largest_component_product_fraction"])
    largest_battle_fraction = float(battle["largest_weak_component_tag_fraction"])
    max_battle_tags = int(battle["weak_component_size"]["max"])
    permutation_bruteforce_feasible = max_battle_tags <= 10
    best_diagnostic = min(
        sensitivity,
        key=lambda row: float(row["largest_component_product_fraction"]),
    )
    return {
        "exact_component_decomposition_assessment": (
            "strong"
            if largest_product_fraction <= 0.5
            else "partial_only"
            if largest_product_fraction <= 0.8
            else "limited"
        ),
        "exact_battle_permutation_bruteforce_feasible": permutation_bruteforce_feasible,
        "largest_product_component_fraction": largest_product_fraction,
        "largest_battle_component_fraction": largest_battle_fraction,
        "largest_battle_component_tags": max_battle_tags,
        "largest_product_component": bipartite["largest_component"],
        "products_outside_largest_component": bipartite[
            "products_outside_largest_component"
        ],
        "exact_components_outside_largest": int(bipartite["components"]) - 1,
        "inactive_battle_tags": battle["inactive_tags"],
        "best_hub_suppressed_diagnostic": {
            "maximum_tag_degree": best_diagnostic["maximum_tag_degree"],
            "largest_component_product_fraction": best_diagnostic[
                "largest_component_product_fraction"
            ],
            "excluded_hub_tags": best_diagnostic["excluded_hub_tags"],
            "isolated_products": best_diagnostic["isolated_products"],
        },
        "interpretation_rule": (
            "Disconnected full-graph components are exact independent subproblems. "
            "Hub-suppressed components are diagnostic only because removing a tag "
            "also removes a real model constraint. Permutation brute force is labelled "
            "feasible only when the largest battle component has at most 10 tags."
        ),
    }


def analyse_snapshot(
    data: SnapshotGraphData,
    *,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    hub_caps: Sequence[int] = DEFAULT_HUB_CAPS,
) -> dict[str, object]:
    """Run the complete network analysis for one immutable snapshot."""

    bipartite = analyse_bipartite_structure(data.product_tags)
    sensitivity = [
        analyse_bipartite_structure(data.product_tags, maximum_tag_degree=cap)
        for cap in hub_caps
    ]
    cycles = sample_three_product_cycles(data.product_tags, draws=draws, seed=seed)
    all_tags = {tag for tags in data.product_tags.values() for tag in tags}
    battle = analyse_battle_graph(all_tags, data.title_positions)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_ref": DEFAULT_SNAPSHOT_REF,
        "snapshot": asdict(data.metadata),
        "parameters": {"draws": draws, "seed": seed, "hub_caps": list(hub_caps)},
        "definitions": {
            "product_tag_graph": "Undirected bipartite graph with one edge per retained product-tag incidence.",
            "cycle_rank": "E - V + C: the number of independent undirected cycles.",
            "two_core": "Maximal subgraph in which every retained node has degree at least two.",
            "battle_graph": "Directed winner-to-loser tag graph derived from relative title positions.",
        },
        "bipartite": bipartite,
        "hub_sensitivity": sensitivity,
        "three_product_cycles": cycles,
        "tag_expansions": _representative_expansions(data.product_tags, cycles),
        "battle": battle,
        "decision": _decision_summary(bipartite, battle, sensitivity),
    }


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)
    with conn.cursor() as cur:
        cur.execute("SET LOCAL default_tablespace = 'pg_default'")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.tag_network_analysis_runs (
                    run_id UUID PRIMARY KEY,
                    snapshot_ref TEXT NOT NULL,
                    snapshot_id UUID NOT NULL,
                    analysis_version INTEGER NOT NULL,
                    parameters JSONB NOT NULL,
                    result JSONB NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE (snapshot_id, analysis_version)
                ) TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS tag_network_analysis_runs_updated_idx
                ON {schema}.tag_network_analysis_runs (updated_at)
                TABLESPACE pg_default
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def persist_analysis(
    conn,
    result: Mapping[str, object],
    *,
    schema: str = DEFAULT_SCHEMA,
    snapshot_ref: str = DEFAULT_SNAPSHOT_REF,
) -> str:
    _ensure_storage(conn, schema)
    run_id = str(uuid.uuid4())
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.tag_network_analysis_runs (
                    run_id, snapshot_ref, snapshot_id, analysis_version,
                    parameters, result
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_id, analysis_version) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    snapshot_ref = EXCLUDED.snapshot_ref,
                    parameters = EXCLUDED.parameters,
                    result = EXCLUDED.result,
                    updated_at = NOW()
                RETURNING run_id::text
                """
            ).format(schema=sql.Identifier(schema)),
            (
                run_id,
                snapshot_ref,
                result["snapshot"]["snapshot_id"],
                result["analysis_version"],
                Jsonb(result["parameters"]),
                Jsonb(result),
            ),
        )
        stored_run_id = str(cur.fetchone()[0])
    conn.commit()
    return stored_run_id


def _parse_hub_caps(value: str) -> tuple[int, ...]:
    caps = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not caps or any(cap <= 0 for cap in caps):
        raise argparse.ArgumentTypeError("hub caps must be positive comma-separated integers")
    return caps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse circular product-tag and title-battle networks in a frozen snapshot."
    )
    parser.add_argument("--dsn")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--snapshot-ref", default=DEFAULT_SNAPSHOT_REF)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--hub-caps",
        type=_parse_hub_caps,
        default=DEFAULT_HUB_CAPS,
        help="Comma-separated maximum tag degrees for sensitivity analysis.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-persist", action="store_true")
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        data = load_snapshot_graph(
            conn,
            schema=args.schema,
            snapshot_ref=args.snapshot_ref,
        )
        result = analyse_snapshot(
            data,
            draws=args.draws,
            seed=args.seed,
            hub_caps=args.hub_caps,
        )
        result["snapshot_ref"] = args.snapshot_ref
        if not args.no_persist:
            result["run_id"] = persist_analysis(
                conn,
                result,
                schema=args.schema,
                snapshot_ref=args.snapshot_ref,
            )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        conn.close()


if __name__ == "__main__":  # pragma: no cover
    main()
