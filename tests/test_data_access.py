from __future__ import annotations

import pytest

from padjective.data_access import (
    SnapshotProductMembership,
    SourceSnapshotRow,
    materialize_snapshot_products,
)
from padjective.product_hash import canonicalize_product_url, hash_product_url


def _hashed_row(
    *,
    product_id: int,
    handle: str,
    title: str,
    raw_tags: str,
) -> SourceSnapshotRow:
    return SourceSnapshotRow(
        product_id=product_id,
        title=title,
        raw_tags=raw_tags,
        product_url=None,
        myshopify_domain="example.myshopify.com",
        product_handle=handle,
    )


def _membership(handle: str, *, cv_fold: int) -> SnapshotProductMembership:
    canonical_url = canonicalize_product_url(
        None,
        myshopify_domain="example.myshopify.com",
        product_handle=handle,
    )
    assert canonical_url is not None
    return SnapshotProductMembership(
        product_id_hash=hash_product_url(canonical_url),
        taxonomy_id=f"tax-{handle}",
        taxonomy_path="1.2.3",
        taxonomy_name=f"Taxonomy {handle}",
        cv_fold=cv_fold,
    )


def test_materialize_snapshot_products_uses_first_matching_source_row() -> None:
    snapshot_rows = [
        _membership("alpha", cv_fold=0),
        _membership("beta", cv_fold=1),
    ]
    source_rows = iter(
        [
            _hashed_row(product_id=10, handle="alpha", title="Alpha One", raw_tags="A, B"),
            _hashed_row(product_id=11, handle="alpha", title="Alpha Duplicate", raw_tags="A, B"),
            _hashed_row(product_id=12, handle="beta", title="Beta", raw_tags="C"),
        ]
    )

    records = materialize_snapshot_products(snapshot_rows, source_rows)

    assert [record.product_id for record in records] == [10, 12]
    assert [record.cv_fold for record in records] == [0, 1]
    assert records[0].taxonomy_id == "tax-alpha"
    assert records[1].tags == ("C",)


def test_materialize_snapshot_products_rejects_missing_snapshot_rows() -> None:
    snapshot_rows = [_membership("gamma", cv_fold=2)]
    source_rows = iter(
        [
            _hashed_row(product_id=20, handle="alpha", title="Alpha", raw_tags="A"),
        ]
    )

    with pytest.raises(ValueError, match="Missing 1 products"):
        materialize_snapshot_products(snapshot_rows, source_rows)
