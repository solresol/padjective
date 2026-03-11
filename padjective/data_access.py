"""Shared data access helpers for training datasets and reporting.

This module centralises the logic for reading product, tag, and taxonomy data
from Postgres so that every consumer (parameter constrained logistic regression,
parameter constrained neural networks, UM-LLR, historical metrics, and the
website) works from the exact same view of the world.  Keeping the data shaping
code in one place avoids subtle discrepancies in sample counts across the project
and makes it easy to expose the raw inputs for reporting purposes.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row
from scipy import sparse

from . import db
from .product_hash import canonicalize_product_url, hash_product_url

# Regex pattern to match taxonomy hierarchical separators
_TAXONOMY_SEPARATOR_RE = re.compile(r"[>/|]")


def is_valid_taxonomy_path(path: Optional[str]) -> bool:
    """Check if a taxonomy path is a valid numeric path.

    Valid taxonomy paths are numeric hierarchical codes like "1.1.4" or "13.3.2.9.6".
    Invalid/defective paths contain hierarchical separators (/, >, |) indicating the
    field was incorrectly populated with taxonomy_name data instead of the numeric path.

    Args:
        path: Taxonomy path string to validate

    Returns:
        True if the path is numeric (no hierarchical separators), False if it
        contains separators that indicate it's actually a taxonomy_name
    """
    if not path:
        return False
    # Valid paths should NOT contain hierarchical separators - they should be numeric like "1.1.4"
    return not bool(_TAXONOMY_SEPARATOR_RE.search(path))


@dataclass(frozen=True)
class ProductRecord:
    """Normalised representation of a Shopify product."""

    product_id: int
    title: str
    tags: Tuple[str, ...]
    raw_tags: Optional[str]
    taxonomy_id: Optional[str]
    taxonomy_path: Optional[str]
    taxonomy_name: Optional[str]
    cv_fold: Optional[int]


@dataclass(frozen=True)
class SnapshotProductMembership:
    """Canonical benchmark-snapshot membership metadata."""

    product_id_hash: str
    taxonomy_id: str
    taxonomy_path: str
    taxonomy_name: str
    cv_fold: Optional[int]


@dataclass(frozen=True)
class SourceSnapshotRow:
    """Source row used to recover raw tags/titles for a benchmark snapshot."""

    product_id: int
    title: str
    raw_tags: Optional[str]
    product_url: Optional[str]
    myshopify_domain: Optional[str]
    product_handle: Optional[str]


@dataclass(frozen=True)
class DiscardedProduct:
    """Product that was excluded from a dataset along with the reason."""

    record: ProductRecord
    reason: str


@dataclass(frozen=True)
class DiscardedTag:
    """Tag that was removed because it was too infrequent."""

    tag: str
    count: int


@dataclass
class ProductDataset:
    """Container describing the derived training dataset."""

    records: List[ProductRecord]
    discarded_products: List[DiscardedProduct]
    discarded_tags: List[DiscardedTag]
    feature_names: List[str]
    features: sparse.csr_matrix
    metadata: pd.DataFrame
    tag_counts: Dict[str, int]

    @property
    def taxonomy_count(self) -> int:
        return int(self.metadata["taxonomy_id"].nunique()) if not self.metadata.empty else 0

    @property
    def product_count(self) -> int:
        return len(self.records)


def normalize_tag(tag: str) -> str:
    """Normalise a tag to uppercase and trim surrounding whitespace."""

    return tag.strip().upper()


def parse_tags(tag_string: Optional[str]) -> List[str]:
    """Split a comma separated tag string into normalised tags."""

    if not tag_string:
        return []
    tags = [normalize_tag(part) for part in tag_string.split(",") if normalize_tag(part)]
    return tags


def _table_has_column(conn, *, schema: str, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
            """,
            (schema, table, column),
        )
        return cur.fetchone() is not None


def _stream_live_products(
    conn,
    *,
    product_table: str,
) -> Iterator[ProductRecord]:
    """Yield product rows directly from the live Shopify catalog tables."""

    product_identifier = db.qualified_identifier(product_table)

    query = sql.SQL(
        """
        SELECT
            p.id,
            p.product_title AS title,
            pd.product_detail->'product'->>'tags' AS tags,
            pt.taxonomy_id,
            t.taxonomy_path,
            t.taxonomy_name,
            up.cv_fold
        FROM {products} AS p
        JOIN public.product_details pd ON (
            p.myshopify_domain = pd.myshopify_domain
            AND p.run_name = pd.run_name
            AND p.product_handle = pd.product_handle
        )
        LEFT JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        LEFT JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
        LEFT JOIN padjective.umllr_predictions up ON up.product_id = p.id
        WHERE p.product_title IS NOT NULL
        ORDER BY p.id
        """
    ).format(products=product_identifier)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        try:
            row_iterator = iter(cur)
        except TypeError:
            row_iterator = iter(cur.fetchall())
        for row in row_iterator:
            product_id = row.get("id")
            if product_id is None:
                continue
            title = row.get("title") or ""
            raw_tags = row.get("tags")
            tags = tuple(parse_tags(raw_tags))
            taxonomy_id = row.get("taxonomy_id")
            taxonomy_path = row.get("taxonomy_path")
            taxonomy_name = row.get("taxonomy_name")
            cv_fold = row.get("cv_fold")
            if cv_fold is not None:
                try:
                    cv_fold = int(cv_fold)
                except (TypeError, ValueError):
                    cv_fold = None
            yield ProductRecord(
                product_id=int(product_id),
                title=str(title),
                tags=tags,
                raw_tags=raw_tags,
                taxonomy_id=str(taxonomy_id) if taxonomy_id is not None else None,
                taxonomy_path=str(taxonomy_path) if taxonomy_path is not None else None,
                taxonomy_name=str(taxonomy_name) if taxonomy_name is not None else None,
                cv_fold=cv_fold,
            )


def _resolve_snapshot_id(conn, *, schema: str, snapshot_ref: str) -> Tuple[str, datetime | None]:
    """Resolve a benchmark snapshot alias/name/UUID to ``(snapshot_id, as_of)``."""

    snapshot_ref = snapshot_ref.strip()
    if not snapshot_ref:
        raise ValueError("snapshot_ref must not be empty")

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT s.snapshot_id::text AS snapshot_id, s.as_of
                FROM {schema}.product_taxonomy_bench_snapshots s
                LEFT JOIN {schema}.product_taxonomy_bench_snapshot_aliases a
                    ON a.snapshot_id = s.snapshot_id
                WHERE a.alias = %s OR s.snapshot_name = %s OR s.snapshot_id::text = %s
                ORDER BY
                    CASE
                        WHEN a.alias = %s THEN 0
                        WHEN s.snapshot_name = %s THEN 1
                        ELSE 2
                    END,
                    s.created_at DESC
                LIMIT 1
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_ref, snapshot_ref, snapshot_ref, snapshot_ref, snapshot_ref),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Unknown snapshot reference: {snapshot_ref!r}")
        return str(row["snapshot_id"]), row["as_of"]


def _load_snapshot_membership(
    conn,
    *,
    schema: str,
    snapshot_ref: str,
) -> tuple[datetime | None, list[SnapshotProductMembership]]:
    snapshot_id, as_of = _resolve_snapshot_id(conn, schema=schema, snapshot_ref=snapshot_ref)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT product_id_hash, taxonomy_id, taxonomy_path, taxonomy_name, cv_fold
                FROM {schema}.product_taxonomy_bench_products
                WHERE snapshot_id = %s
                ORDER BY product_id_hash
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        rows = [
            SnapshotProductMembership(
                product_id_hash=str(row["product_id_hash"]),
                taxonomy_id=str(row["taxonomy_id"]),
                taxonomy_path=str(row["taxonomy_path"]),
                taxonomy_name=str(row["taxonomy_name"]),
                cv_fold=int(row["cv_fold"]) if row["cv_fold"] is not None else None,
            )
            for row in cur.fetchall()
        ]

    return as_of, rows


def _stream_snapshot_source_rows(
    conn,
    *,
    product_table: str,
    as_of: datetime | None,
) -> Iterator[SourceSnapshotRow]:
    product_identifier = db.qualified_identifier(product_table)
    conditions: list[sql.SQL] = [sql.SQL("p.product_title IS NOT NULL")]
    params: list[object] = []

    if as_of is not None and _table_has_column(
        conn, schema="public", table="product_details", column="when_fetched"
    ):
        conditions.append(sql.SQL("pd.when_fetched <= %s"))
        # ``when_fetched`` is stored without time zone on raksasa.
        params.append(as_of.replace(tzinfo=None))

    query = sql.SQL(
        """
        SELECT
            p.id,
            p.product_title AS title,
            pd.product_detail->'product'->>'tags' AS tags,
            p.product_url,
            p.myshopify_domain,
            p.product_handle
        FROM {products} AS p
        JOIN public.product_details pd ON (
            p.myshopify_domain = pd.myshopify_domain
            AND p.run_name = pd.run_name
            AND p.product_handle = pd.product_handle
        )
        WHERE {where_clause}
        ORDER BY p.id
        """
    ).format(
        products=product_identifier,
        where_clause=sql.SQL(" AND ").join(conditions),
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        for row in cur:
            product_id = row.get("id")
            if product_id is None:
                continue
            yield SourceSnapshotRow(
                product_id=int(product_id),
                title=str(row.get("title") or ""),
                raw_tags=row.get("tags"),
                product_url=row.get("product_url"),
                myshopify_domain=row.get("myshopify_domain"),
                product_handle=row.get("product_handle"),
            )


def materialize_snapshot_products(
    snapshot_rows: List[SnapshotProductMembership],
    source_rows: Iterator[SourceSnapshotRow],
) -> List[ProductRecord]:
    """Recover raw-tag source rows for a benchmark snapshot deterministically."""

    membership_by_hash = {row.product_id_hash: row for row in snapshot_rows}
    matched_hashes: set[str] = set()
    records: List[ProductRecord] = []

    for row in source_rows:
        canonical_url = canonicalize_product_url(
            row.product_url,
            myshopify_domain=row.myshopify_domain,
            product_handle=row.product_handle,
        )
        if not canonical_url:
            continue
        product_hash = hash_product_url(canonical_url)
        snapshot_row = membership_by_hash.get(product_hash)
        if snapshot_row is None or product_hash in matched_hashes:
            continue
        matched_hashes.add(product_hash)
        tags = tuple(parse_tags(row.raw_tags))
        records.append(
            ProductRecord(
                product_id=row.product_id,
                title=row.title,
                tags=tags,
                raw_tags=row.raw_tags,
                taxonomy_id=snapshot_row.taxonomy_id,
                taxonomy_path=snapshot_row.taxonomy_path,
                taxonomy_name=snapshot_row.taxonomy_name,
                cv_fold=snapshot_row.cv_fold,
            )
        )

    missing_hashes = sorted(set(membership_by_hash) - matched_hashes)
    if missing_hashes:
        raise ValueError(
            "Could not recover all benchmark snapshot products from source catalog rows. "
            f"Missing {len(missing_hashes)} products."
        )

    return records


def stream_products(
    conn,
    *,
    product_table: str = "cantbuymelove.product",
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
) -> Iterator[ProductRecord]:
    """Yield normalised product rows from Postgres."""

    if snapshot_ref is None:
        yield from _stream_live_products(conn, product_table=product_table)
        return

    as_of, snapshot_rows = _load_snapshot_membership(
        conn,
        schema=snapshot_schema,
        snapshot_ref=snapshot_ref,
    )
    source_rows = _stream_snapshot_source_rows(
        conn,
        product_table=product_table,
        as_of=as_of,
    )
    for record in materialize_snapshot_products(snapshot_rows, source_rows):
        yield record


def build_feature_dataset(
    conn,
    *,
    product_table: str = "cantbuymelove.product",
    require_taxonomy: bool = True,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: Optional[int] = None,
    snapshot_ref: str | None = None,
    snapshot_schema: str = "padjective",
) -> ProductDataset:
    """Build a sparse tag matrix and accompanying metadata from Postgres."""

    all_records = list(
        stream_products(
            conn,
            product_table=product_table,
            snapshot_ref=snapshot_ref,
            snapshot_schema=snapshot_schema,
        )
    )

    # Use all available products
    # Sort by product_id for consistency
    all_records.sort(key=lambda record: record.product_id)

    if require_taxonomy:
        eligible_records = [record for record in all_records if record.taxonomy_id]
    else:
        eligible_records = list(all_records)

    tag_counter: Counter[str] = Counter()
    for record in eligible_records:
        tag_counter.update(record.tags)

    valid_tags = {tag for tag, count in tag_counter.items() if count >= min_tag_count}
    discarded_tags = [DiscardedTag(tag=tag, count=count) for tag, count in tag_counter.items() if count < min_tag_count]

    taxonomy_counts: Counter[str] = Counter()
    if require_taxonomy:
        for record in eligible_records:
            if record.taxonomy_id:
                taxonomy_counts.update([record.taxonomy_id])

    valid_taxonomies: set[str] | None = None
    if require_taxonomy and min_samples_per_taxonomy:
        valid_taxonomies = {tid for tid, count in taxonomy_counts.items() if count >= min_samples_per_taxonomy}

    included: List[ProductRecord] = []
    discarded_products: List[DiscardedProduct] = []

    for record in eligible_records:
        if not record.tags:
            discarded_products.append(DiscardedProduct(record, "no_tags"))
            continue
        if require_taxonomy and not record.taxonomy_id:
            discarded_products.append(DiscardedProduct(record, "missing_taxonomy"))
            continue
        if require_taxonomy and not is_valid_taxonomy_path(record.taxonomy_path):
            discarded_products.append(DiscardedProduct(record, "defective_taxonomy_path"))
            continue
        if require_taxonomy and valid_taxonomies is not None:
            if record.taxonomy_id not in valid_taxonomies:
                discarded_products.append(DiscardedProduct(record, "insufficient_taxonomy_samples"))
                continue
        included.append(record)

    feature_names = sorted(valid_tags)
    tag_index = {tag: idx for idx, tag in enumerate(feature_names)}

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    metadata_rows: List[Dict[str, object]] = []

    for row_idx, record in enumerate(included):
        valid_tag_count = 0
        for tag in record.tags:
            if tag in tag_index:
                rows.append(row_idx)
                cols.append(tag_index[tag])
                data.append(1.0)
                valid_tag_count += 1
        metadata_rows.append(
            {
                "product_id": record.product_id,
                "title": record.title,
                "tags": ", ".join(record.tags),
                "tag_count": len(record.tags),
                "valid_tag_count": valid_tag_count,
                "taxonomy_id": record.taxonomy_id,
                "taxonomy_name": record.taxonomy_name,
                "taxonomy_path": record.taxonomy_path,
                "cv_fold": record.cv_fold,
            }
        )

    feature_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(included), len(feature_names)),
        dtype=float,
    )

    metadata = pd.DataFrame(metadata_rows)

    return ProductDataset(
        records=included,
        discarded_products=discarded_products,
        discarded_tags=discarded_tags,
        feature_names=feature_names,
        features=feature_matrix,
        metadata=metadata,
        tag_counts=dict(tag_counter),
    )
