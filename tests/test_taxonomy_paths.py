from __future__ import annotations

import pytest

from padjective.taxonomy_paths import (
    TaxonomyPathRepairError,
    build_reconciliation_rows,
    is_numeric_taxonomy_path,
)


def test_numeric_taxonomy_path_validation_is_strict() -> None:
    assert is_numeric_taxonomy_path("1")
    assert is_numeric_taxonomy_path("13.3.2.9.6")
    assert not is_numeric_taxonomy_path(None)
    assert not is_numeric_taxonomy_path("")
    assert not is_numeric_taxonomy_path("Apparel & Accessories > Clothing")
    assert not is_numeric_taxonomy_path("not-a-path")


def test_reconciliation_prefers_live_then_snapshot_then_inference() -> None:
    current = [
        ("gid://shopify/TaxonomyCategory/aa", "1"),
        ("gid://shopify/TaxonomyCategory/aa-1", "Clothing"),
        ("gid://shopify/TaxonomyCategory/aa-1-13", "Shirts & Tops"),
    ]
    historical = [
        ("gid://shopify/TaxonomyCategory/aa-1", "1.1"),
    ]

    rows, report = build_reconciliation_rows(current, historical)

    by_id = {row.taxonomy_id: row for row in rows}
    assert by_id["gid://shopify/TaxonomyCategory/aa"].numeric_path == "1"
    assert by_id["gid://shopify/TaxonomyCategory/aa"].resolution_source == "current_numeric"
    assert by_id["gid://shopify/TaxonomyCategory/aa-1"].numeric_path == "1.1"
    assert by_id["gid://shopify/TaxonomyCategory/aa-1"].resolution_source == "historical_snapshot"
    assert by_id["gid://shopify/TaxonomyCategory/aa-1-13"].numeric_path == "1.1.13"
    assert by_id["gid://shopify/TaxonomyCategory/aa-1-13"].resolution_source == "taxonomy_id_inference"
    assert report.taxonomy_count == 3
    assert report.resolved_taxonomy_count == 3
    assert not report.unresolved_taxonomy_ids


def test_reconciliation_preserves_existing_source_and_updates_observed_path() -> None:
    current = [
        ("gid://shopify/TaxonomyCategory/fr-10-2", "Food > Snacks"),
    ]
    existing = [
        ("gid://shopify/TaxonomyCategory/fr-10-2", "10.10.2", "historical_snapshot"),
    ]

    rows, report = build_reconciliation_rows(current, existing_rows=existing)

    assert rows[0].numeric_path == "10.10.2"
    assert rows[0].observed_taxonomy_path == "Food > Snacks"
    assert rows[0].resolution_source == "historical_snapshot"
    assert report.historical_snapshot_count == 1


def test_reconciliation_rejects_conflicting_historical_evidence() -> None:
    current = [("gid://shopify/TaxonomyCategory/aa-1", "Clothing")]
    historical = [
        ("gid://shopify/TaxonomyCategory/aa-1", "1.1"),
        ("gid://shopify/TaxonomyCategory/aa-1", "1.2"),
    ]

    with pytest.raises(TaxonomyPathRepairError, match="Conflicting numeric paths"):
        build_reconciliation_rows(current, historical)


def test_reconciliation_rejects_taxonomy_id_mismatch() -> None:
    current = [
        ("gid://shopify/TaxonomyCategory/aa", "1"),
        ("gid://shopify/TaxonomyCategory/aa-2", "1.3"),
    ]

    with pytest.raises(TaxonomyPathRepairError, match="inference"):
        build_reconciliation_rows(current)


def test_reconciliation_reports_unresolved_new_root() -> None:
    current = [("gid://shopify/TaxonomyCategory/zz-1", "Unknown > Child")]

    rows, report = build_reconciliation_rows(current)

    assert rows == []
    assert report.unresolved_taxonomy_ids == (
        "gid://shopify/TaxonomyCategory/zz-1",
    )
