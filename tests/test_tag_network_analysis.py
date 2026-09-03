from __future__ import annotations

import pytest

from padjective.tag_network_analysis import (
    analyse_battle_graph,
    analyse_bipartite_structure,
    sample_three_product_cycles,
    tag_expansion,
)


def _triangle_products() -> dict[str, tuple[str, ...]]:
    return {
        "product-a": ("tag-a", "tag-c"),
        "product-b": ("tag-a", "tag-b"),
        "product-c": ("tag-b", "tag-c"),
    }


def test_bipartite_triangle_is_one_six_cycle() -> None:
    result = analyse_bipartite_structure(_triangle_products())

    assert result["products"] == 3
    assert result["retained_tags"] == 3
    assert result["edges"] == 6
    assert result["components"] == 1
    assert result["cycle_rank"] == 1
    assert result["cycle_rank_per_edge"] == pytest.approx(1 / 6)
    assert result["two_core"]["nodes"] == 6
    assert result["two_core"]["edge_fraction"] == pytest.approx(1.0)


def test_hub_suppression_is_explicit_and_preserves_isolated_products() -> None:
    product_tags = {
        "p1": ("hub", "a"),
        "p2": ("hub", "b"),
        "p3": ("hub",),
    }

    result = analyse_bipartite_structure(product_tags, maximum_tag_degree=2)

    assert result["excluded_hub_tags"] == 1
    assert result["excluded_tag_ids"] == ["hub"]
    assert result["isolated_products"] == 1
    assert result["largest_component_product_fraction"] == pytest.approx(1 / 3)


def test_cycle_sampler_finds_three_distinct_products_and_tags() -> None:
    result = sample_three_product_cycles(
        _triangle_products(), draws=100, seed=42, maximum_examples=2
    )

    assert result["valid_chains"] == 100
    assert result["closed_chains"] == 100
    assert result["closure_rate"] == pytest.approx(1.0)
    example = result["examples"][0]
    assert len(set(example["products"])) == 3
    assert set(example["tags"]) == {"tag-a", "tag-b", "tag-c"}
    assert all(len(tags) == 2 for tags in example["incidence"].values())


def test_battle_graph_detects_directed_triangle_with_distinct_supporters() -> None:
    positions = {
        # Later title position wins: a > b, b > c, c > a.
        "product-1": (("tag-b", 0, 0), ("tag-a", 0, 5)),
        "product-2": (("tag-c", 0, 0), ("tag-b", 0, 5)),
        "product-3": (("tag-a", 0, 0), ("tag-c", 0, 5)),
    }

    result = analyse_battle_graph({"tag-a", "tag-b", "tag-c"}, positions)

    assert result["unique_directed_edges"] == 3
    assert result["weak_components"] == 1
    assert result["strong_components"] == 1
    assert result["directed_three_tag_cycles"] == 1
    assert len(set(result["directed_three_tag_cycle_examples"][0]["products"])) == 3


def test_tag_expansion_alternates_products_and_tags() -> None:
    result = tag_expansion(_triangle_products(), "tag-a", maximum_depth=3)

    assert result["levels"] == [
        {
            "depth": 1,
            "new_products": 2,
            "new_tags": 0,
            "cumulative_products": 2,
            "cumulative_tags": 1,
        },
        {
            "depth": 2,
            "new_products": 0,
            "new_tags": 2,
            "cumulative_products": 2,
            "cumulative_tags": 3,
        },
        {
            "depth": 3,
            "new_products": 1,
            "new_tags": 0,
            "cumulative_products": 3,
            "cumulative_tags": 3,
        },
    ]
