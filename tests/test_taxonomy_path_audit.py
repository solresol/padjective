from __future__ import annotations

from padjective.taxonomy_path_audit import (
    AuditProductRow,
    calculate_eligibility_audit,
)


def _row(
    product_id: int,
    *,
    tags: str | None = "RED, SHIRT",
    taxonomy_id: str | None = "tax-1",
    observed_path: str | None = "Apparel > Shirts",
    resolved_path: str | None = "1.1",
    product_url: str | None = None,
    handle: str | None = None,
) -> AuditProductRow:
    return AuditProductRow(
        product_id=product_id,
        title=f"Product {product_id}",
        has_product_details=True,
        raw_tags=tags,
        taxonomy_id=taxonomy_id,
        observed_taxonomy_path=observed_path,
        resolved_taxonomy_path=resolved_path,
        taxonomy_name="Shirts" if taxonomy_id else None,
        product_url=product_url,
        myshopify_domain="example.myshopify.com",
        product_handle=handle or f"product-{product_id}",
    )


def test_audit_matches_benchmark_thresholds_and_url_deduplication() -> None:
    rows = [
        _row(1, product_url="https://example.com/products/a?variant=1"),
        _row(2, product_url="https://example.com/products/a?variant=2"),
        _row(3, tags="ONE-OFF", handle="c"),
        _row(4, taxonomy_id=None, observed_path=None, resolved_path=None),
        _row(5, observed_path="1.1", handle="e"),
    ]

    audit = calculate_eligibility_audit(
        rows,
        catalogue_products=6,
        min_tag_count=2,
        min_samples_per_taxonomy=2,
    )

    assert audit.catalogue_products == 6
    assert audit.products_with_taxonomy_metadata == 4
    assert audit.products_with_resolved_numeric_path == 4
    assert audit.products_in_taxonomies_meeting_minimum == 4
    assert audit.products_with_frequent_tags == 3
    assert audit.products_with_canonical_url == 3
    assert audit.benchmark_products_after_url_deduplication == 2
    assert audit.raw_numeric_path_products == 1
    assert audit.reconciled_display_path_products == 3
    assert audit.unresolved_taxonomy_path_products == 0
    assert audit.taxonomies_meeting_minimum == 1
    assert audit.benchmark_taxonomies_after_filters == 1
    assert audit.eligible_tags == 2


def test_audit_exposes_unresolved_paths_separately_from_product_errors() -> None:
    rows = [
        _row(1, resolved_path=None),
        _row(2, observed_path="1.1", resolved_path="1.1"),
    ]

    audit = calculate_eligibility_audit(
        rows,
        catalogue_products=2,
        min_tag_count=1,
        min_samples_per_taxonomy=1,
    )

    assert audit.products_with_taxonomy_metadata == 2
    assert audit.products_with_resolved_numeric_path == 1
    assert audit.unresolved_taxonomy_path_products == 1
    assert audit.benchmark_products_after_url_deduplication == 1
