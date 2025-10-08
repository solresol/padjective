"""Tests for tag_features module."""

import pytest
from padjective.tag_features import normalize_tag, parse_tags


def test_normalize_tag():
    """Test tag normalization."""
    assert normalize_tag("  chocolate  ") == "CHOCOLATE"
    assert normalize_tag("Milk Chocolate") == "MILK CHOCOLATE"
    assert normalize_tag("") == ""


def test_parse_tags_empty():
    """Test parsing empty or None tag strings."""
    assert parse_tags(None) == []
    assert parse_tags("") == []
    assert parse_tags("   ") == []


def test_parse_tags_single():
    """Test parsing a single tag."""
    assert parse_tags("chocolate") == ["CHOCOLATE"]
    assert parse_tags("  chocolate  ") == ["CHOCOLATE"]


def test_parse_tags_multiple():
    """Test parsing multiple comma-separated tags."""
    assert parse_tags("chocolate,milk,dark") == ["CHOCOLATE", "MILK", "DARK"]
    assert parse_tags("chocolate, milk, dark") == ["CHOCOLATE", "MILK", "DARK"]
    assert parse_tags("  chocolate  ,  milk  ,  dark  ") == ["CHOCOLATE", "MILK", "DARK"]


def test_parse_tags_with_empty_parts():
    """Test that empty parts are filtered out."""
    assert parse_tags("chocolate,,milk") == ["CHOCOLATE", "MILK"]
    assert parse_tags(",chocolate,milk,") == ["CHOCOLATE", "MILK"]


def test_parse_tags_case_normalization():
    """Test that tags are uppercased."""
    assert parse_tags("Chocolate,Milk Chocolate,DARK") == [
        "CHOCOLATE",
        "MILK CHOCOLATE",
        "DARK",
    ]
