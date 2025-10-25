"""Tests for taxonomy_classifier module."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from padjective.taxonomy_classifier import (
    compute_tag_coefficients,
    compute_taxonomy_top_tags,
)


def test_compute_tag_coefficients_basic():
    """Test basic coefficient computation."""
    feature_names = ["TAG1", "TAG2", "TAG3"]
    classes = np.array(["tax1", "tax2"])
    coef_matrix = np.array([
        [1.0, -0.5, 0.3],  # tax1
        [-0.8, 1.2, -0.2],  # tax2
    ])

    summary = compute_tag_coefficients(feature_names, classes, coef_matrix)

    assert len(summary) == 3
    assert "tag" in summary.columns
    assert "top_taxonomy" in summary.columns
    assert "top_weight" in summary.columns
    assert "max_abs_coef" in summary.columns
    assert "sum_abs_coef" in summary.columns

    # TAG2 should have highest max_abs_coef (1.2)
    assert summary.iloc[0]["tag"] == "TAG2"
    assert summary.iloc[0]["top_taxonomy"] == "tax2"
    assert summary.iloc[0]["top_weight"] == 1.2


def test_compute_tag_coefficients_binary():
    """Test coefficient computation with binary classifier output."""
    feature_names = ["TAG1", "TAG2"]
    classes = np.array(["tax1", "tax2"])
    coef_matrix = np.array([0.5, -0.3])  # 1D array (binary case)

    summary = compute_tag_coefficients(feature_names, classes, coef_matrix)

    assert len(summary) == 2
    assert summary["tag"].tolist() == ["TAG1", "TAG2"]


def test_compute_tag_coefficients_sorting():
    """Test that results are sorted by max_abs_coef."""
    feature_names = ["TAG1", "TAG2", "TAG3"]
    classes = np.array(["tax1"])
    coef_matrix = np.array([[0.1, 0.9, 0.3]])

    summary = compute_tag_coefficients(feature_names, classes, coef_matrix)

    # Should be sorted by max_abs_coef descending
    assert summary["tag"].tolist() == ["TAG2", "TAG3", "TAG1"]
    assert summary["max_abs_coef"].tolist() == [0.9, 0.3, 0.1]


def test_compute_taxonomy_top_tags() -> None:
    feature_names = ["ALPHA", "BETA", "GAMMA", "DELTA"]
    classes = np.array(["A", "B"])
    coef_matrix = np.array(
        [
            [0.4, -1.2, 0.6, 0.1],
            [-0.5, 0.9, -0.7, 0.2],
        ]
    )

    top_tags = compute_taxonomy_top_tags(feature_names, classes, coef_matrix, top_k=2)

    assert list(top_tags.columns) == [
        "taxonomy_id",
        "tag",
        "weight",
        "rank",
    ]
    class_a = top_tags[top_tags["taxonomy_id"] == "A"]
    assert class_a.iloc[0]["tag"] == "GAMMA"
    assert class_a.iloc[0]["weight"] == pytest.approx(0.6)
    assert class_a.iloc[1]["tag"] == "ALPHA"


def test_compute_taxonomy_top_tags_binary_expansion() -> None:
    feature_names = ["X", "Y"]
    classes = np.array(["NEG", "POS"])
    coef_matrix = np.array([0.8, -0.4])

    top_tags = compute_taxonomy_top_tags(feature_names, classes, coef_matrix, top_k=1)
    assert len(top_tags) == 2
    assert set(top_tags["taxonomy_id"]) == {"NEG", "POS"}
