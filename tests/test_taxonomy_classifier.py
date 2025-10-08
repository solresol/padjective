"""Tests for taxonomy_classifier module."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from padjective.taxonomy_classifier import compute_tag_coefficients


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
