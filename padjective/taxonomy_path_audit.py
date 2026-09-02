"""Reconcile taxonomy paths and record benchmark eligibility counts.

This command is intended to run before the nightly modelling pipeline.  It
persists a compact stage ledger so changes in the final benchmark population
can be attributed to a specific eligibility rule rather than mistaken for a
change in the size of the source catalogue.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Iterable

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import data_access, db, tagbattle, taxonomy_paths
from .product_hash import canonicalize_product_url, hash_product_url


@dataclass(frozen=True)
class AuditProductRow:
    product_id: int
    title: str | None
    has_product_details: bool
    raw_tags: str | None
    taxonomy_id: str | None
    observed_taxonomy_path: str | None
    resolved_taxonomy_path: str | None
    taxonomy_name: str | None
    product_url: str | None
    myshopify_domain: str | None
    product_handle: str | None


@dataclass(frozen=True)
class EligibilityAudit:
    """Sequential product counts matching benchmark-snapshot eligibility."""

    catalogue_products: int
    products_with_title: int
    products_with_details: int
    products_with_taxonomy_metadata: int
    products_with_resolved_numeric_path: int
    products_with_nonempty_tags: int
    products_in_taxonomies_meeting_minimum: int
    products_with_frequent_tags: int
    products_with_canonical_url: int
    benchmark_products_after_url_deduplication: int
    raw_numeric_path_products: int
    reconciled_display_path_products: int
    unresolved_taxonomy_path_products: int
    taxonomies_meeting_minimum: int
    benchmark_taxonomies_after_filters: int
    eligible_tags: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def calculate_eligibility_audit(
    rows: Iterable[AuditProductRow],
    *,
    catalogue_products: int,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
) -> EligibilityAudit:
    """Calculate the benchmark funnel from source rows in one deterministic pass."""

    materialized = list(rows)
    titled = [row for row in materialized if row.title is not None]
    with_details = [row for row in titled if row.has_product_details]
    with_taxonomy = [
        row
        for row in with_details
        if row.taxonomy_id and row.observed_taxonomy_path and row.taxonomy_name
    ]
    resolved = [
        row
        for row in with_taxonomy
        if taxonomy_paths.is_numeric_taxonomy_path(row.resolved_taxonomy_path)
    ]

    tags_by_product: dict[int, list[str]] = {}
    tagged: list[AuditProductRow] = []
    for row in resolved:
        tags = tagbattle.filter_nested_tags(data_access.parse_tags(row.raw_tags))
        if not tags:
            continue
        tags_by_product[row.product_id] = tags
        tagged.append(row)

    taxonomy_counts = Counter(row.taxonomy_id for row in tagged)
    valid_taxonomies = {
        taxonomy_id
        for taxonomy_id, count in taxonomy_counts.items()
        if taxonomy_id is not None and count >= min_samples_per_taxonomy
    }
    taxonomy_eligible = [
        row for row in tagged if row.taxonomy_id in valid_taxonomies
    ]

    tag_counts: Counter[str] = Counter()
    for row in taxonomy_eligible:
        tag_counts.update(tags_by_product[row.product_id])
    valid_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_count}
    tag_eligible = [
        row
        for row in taxonomy_eligible
        if any(tag in valid_tags for tag in tags_by_product[row.product_id])
    ]

    canonical_products: dict[str, AuditProductRow] = {}
    canonical_url_count = 0
    for row in tag_eligible:
        canonical_url = canonicalize_product_url(
            row.product_url,
            myshopify_domain=row.myshopify_domain,
            product_handle=row.product_handle,
        )
        if canonical_url:
            canonical_url_count += 1
            canonical_products.setdefault(hash_product_url(canonical_url), row)

    raw_numeric = sum(
        taxonomy_paths.is_numeric_taxonomy_path(row.observed_taxonomy_path)
        for row in with_taxonomy
    )
    reconciled_display = sum(
        not taxonomy_paths.is_numeric_taxonomy_path(row.observed_taxonomy_path)
        and taxonomy_paths.is_numeric_taxonomy_path(row.resolved_taxonomy_path)
        for row in with_taxonomy
    )

    return EligibilityAudit(
        catalogue_products=int(catalogue_products),
        products_with_title=len(titled),
        products_with_details=len(with_details),
        products_with_taxonomy_metadata=len(with_taxonomy),
        products_with_resolved_numeric_path=len(resolved),
        products_with_nonempty_tags=len(tagged),
        products_in_taxonomies_meeting_minimum=len(taxonomy_eligible),
        products_with_frequent_tags=len(tag_eligible),
        products_with_canonical_url=canonical_url_count,
        benchmark_products_after_url_deduplication=len(canonical_products),
        raw_numeric_path_products=int(raw_numeric),
        reconciled_display_path_products=int(reconciled_display),
        unresolved_taxonomy_path_products=len(with_taxonomy) - len(resolved),
        taxonomies_meeting_minimum=len(valid_taxonomies),
        benchmark_taxonomies_after_filters=len(
            {
                row.taxonomy_id
                for row in canonical_products.values()
                if row.taxonomy_id is not None
            }
        ),
        eligible_tags=len(valid_tags),
    )


def _stream_audit_rows(
    conn,
    *,
    product_table: str,
    schema: str,
) -> Iterable[AuditProductRow]:
    product_identifier = db.qualified_identifier(product_table)
    query = sql.SQL(
        """
        SELECT
            p.id,
            p.product_title,
            pd.product_handle IS NOT NULL AS has_product_details,
            pd.product_detail->'product'->>'tags' AS tags,
            pt.taxonomy_id,
            t.taxonomy_path AS observed_taxonomy_path,
            {resolved_path} AS resolved_taxonomy_path,
            t.taxonomy_name,
            p.product_url,
            p.myshopify_domain,
            p.product_handle
        FROM {products} p
        LEFT JOIN public.product_details pd ON (
            p.myshopify_domain = pd.myshopify_domain
            AND p.run_name = pd.run_name
            AND p.product_handle = pd.product_handle
        )
        LEFT JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        LEFT JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
        {reconciliation_join}
        ORDER BY p.id
        """
    ).format(
        products=product_identifier,
        resolved_path=taxonomy_paths.taxonomy_path_sql(),
        reconciliation_join=taxonomy_paths.reconciliation_join_sql(schema=schema),
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            yield AuditProductRow(
                product_id=int(row["id"]),
                title=row.get("product_title"),
                has_product_details=bool(row.get("has_product_details")),
                raw_tags=row.get("tags"),
                taxonomy_id=row.get("taxonomy_id"),
                observed_taxonomy_path=row.get("observed_taxonomy_path"),
                resolved_taxonomy_path=row.get("resolved_taxonomy_path"),
                taxonomy_name=row.get("taxonomy_name"),
                product_url=row.get("product_url"),
                myshopify_domain=row.get("myshopify_domain"),
                product_handle=row.get("product_handle"),
            )


def _catalogue_product_count(conn, product_table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {products}").format(
                products=db.qualified_identifier(product_table)
            )
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # pragma: no cover - best effort metadata
        return None


def _ensure_audit_storage(conn, schema: str) -> None:
    db.ensure_table(
        conn,
        schema,
        "taxonomy_path_reconciliation_audits",
        (
            "audit_id UUID PRIMARY KEY",
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            "product_table TEXT NOT NULL",
            "min_tag_count INTEGER NOT NULL",
            "min_samples_per_taxonomy INTEGER NOT NULL",
            "reconciliation_counts JSONB NOT NULL",
            "eligibility_counts JSONB NOT NULL",
            "code_version TEXT",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} ON {schema}."
                "taxonomy_path_reconciliation_audits (created_at) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_taxonomy_path_audits_created_at_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )


def run_audit(
    conn,
    *,
    schema: str = taxonomy_paths.DEFAULT_SCHEMA,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    persist: bool = True,
) -> tuple[taxonomy_paths.ReconciliationReport, EligibilityAudit]:
    """Run reconciliation, calculate the funnel, and optionally persist it."""

    reconciliation = taxonomy_paths.reconcile_taxonomy_paths(conn, schema=schema)
    eligibility = calculate_eligibility_audit(
        _stream_audit_rows(conn, product_table=product_table, schema=schema),
        catalogue_products=_catalogue_product_count(conn, product_table),
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )

    if persist:
        _ensure_audit_storage(conn, schema)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {schema}.taxonomy_path_reconciliation_audits (
                        audit_id,
                        product_table,
                        min_tag_count,
                        min_samples_per_taxonomy,
                        reconciliation_counts,
                        eligibility_counts,
                        code_version
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                (
                    uuid.uuid4(),
                    product_table,
                    min_tag_count,
                    min_samples_per_taxonomy,
                    Jsonb(asdict(reconciliation)),
                    Jsonb(eligibility.as_dict()),
                    _git_head(),
                ),
            )
        conn.commit()

    return reconciliation, eligibility


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair numeric taxonomy paths and audit benchmark eligibility."
    )
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--schema", default=taxonomy_paths.DEFAULT_SCHEMA)
    parser.add_argument("--product-table", default="cantbuymelove.product")
    parser.add_argument("--min-tag-count", type=int, default=5)
    parser.add_argument("--min-samples-per-taxonomy", type=int, default=5)
    parser.add_argument(
        "--no-persist", action="store_true", help="Calculate without inserting an audit row"
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        reconciliation, eligibility = run_audit(
            conn,
            schema=args.schema,
            product_table=args.product_table,
            min_tag_count=args.min_tag_count,
            min_samples_per_taxonomy=args.min_samples_per_taxonomy,
            persist=not args.no_persist,
        )
    finally:
        conn.close()

    print(
        json.dumps(
            {
                "reconciliation": asdict(reconciliation),
                "eligibility": eligibility.as_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
