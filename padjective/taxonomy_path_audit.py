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
from typing import Iterable, Sequence

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from . import data_access, db, tagbattle, taxonomy_paths
from .product_hash import canonicalize_product_url, hash_product_url


AUDIT_BATCH_SIZE = 5000


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


@dataclass(frozen=True)
class _AuditTempTables:
    """Session-local Postgres tables used for bounded-memory aggregation."""

    stage: str
    valid_taxonomies: str
    valid_tags: str
    canonical_products: str

    @classmethod
    def create(cls) -> "_AuditTempTables":
        token = uuid.uuid4().hex[:12]
        prefix = f"padjective_audit_{token}"
        return cls(
            stage=f"{prefix}_stage",
            valid_taxonomies=f"{prefix}_taxonomies",
            valid_tags=f"{prefix}_tags",
            canonical_products=f"{prefix}_products",
        )


def calculate_eligibility_audit(
    rows: Sequence[AuditProductRow],
    *,
    catalogue_products: int,
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
) -> EligibilityAudit:
    """Reference calculation for small, already-materialised row collections.

    Production audits use :func:`_calculate_eligibility_audit_from_postgres`,
    which stages only resolved, tagged products in session-local Postgres tables
    and never retains the source catalogue in application memory.
    """

    titled = [row for row in rows if row.title is not None]
    with_details = [row for row in titled if row.has_product_details]
    with_taxonomy = [
        row
        for row in with_details
        if row.taxonomy_id and row.taxonomy_name
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


def _create_audit_temp_tables(conn, tables: _AuditTempTables) -> None:
    """Create session-local aggregation tables in ``pg_default``."""

    with conn.cursor() as cur:
        cur.execute(
            "SELECT set_config('default_tablespace', %s, true)",
            (db.DEFAULT_TABLESPACE,),
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TEMPORARY TABLE {stage} (
                    product_id BIGINT NOT NULL,
                    taxonomy_id TEXT NOT NULL,
                    tags TEXT[] NOT NULL,
                    product_url TEXT,
                    myshopify_domain TEXT,
                    product_handle TEXT
                ) ON COMMIT DROP TABLESPACE {tablespace}
                """
            ).format(
                stage=sql.Identifier(tables.stage),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TEMPORARY TABLE {valid_taxonomies} (
                    taxonomy_id TEXT NOT NULL
                ) ON COMMIT DROP TABLESPACE {tablespace}
                """
            ).format(
                valid_taxonomies=sql.Identifier(tables.valid_taxonomies),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX {valid_taxonomies_index}
                ON {valid_taxonomies} (taxonomy_id)
                TABLESPACE {tablespace}
                """
            ).format(
                valid_taxonomies_index=sql.Identifier(
                    f"{tables.valid_taxonomies}_id_idx"
                ),
                valid_taxonomies=sql.Identifier(
                    "pg_temp", tables.valid_taxonomies
                ),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TEMPORARY TABLE {valid_tags} (
                    tag TEXT NOT NULL
                ) ON COMMIT DROP TABLESPACE {tablespace}
                """
            ).format(
                valid_tags=sql.Identifier(tables.valid_tags),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX {valid_tags_index}
                ON {valid_tags} (tag)
                TABLESPACE {tablespace}
                """
            ).format(
                valid_tags_index=sql.Identifier(f"{tables.valid_tags}_tag_idx"),
                valid_tags=sql.Identifier("pg_temp", tables.valid_tags),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TEMPORARY TABLE {canonical_products} (
                    product_id_hash TEXT NOT NULL,
                    taxonomy_id TEXT NOT NULL
                ) ON COMMIT DROP TABLESPACE {tablespace}
                """
            ).format(
                canonical_products=sql.Identifier(tables.canonical_products),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL(
                """
                CREATE UNIQUE INDEX {canonical_products_index}
                ON {canonical_products} (product_id_hash)
                TABLESPACE {tablespace}
                """
            ).format(
                canonical_products_index=sql.Identifier(
                    f"{tables.canonical_products}_hash_idx"
                ),
                canonical_products=sql.Identifier(
                    "pg_temp", tables.canonical_products
                ),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )


def _flush_audit_stage_rows(
    conn,
    tables: _AuditTempTables,
    rows: list[tuple[int, str, list[str], str | None, str | None, str | None]],
) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {stage} (
                    product_id,
                    taxonomy_id,
                    tags,
                    product_url,
                    myshopify_domain,
                    product_handle
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """
            ).format(stage=sql.Identifier("pg_temp", tables.stage)),
            rows,
        )
    rows.clear()


def _prepare_audit_aggregates(
    conn,
    tables: _AuditTempTables,
    *,
    min_tag_count: int,
    min_samples_per_taxonomy: int,
) -> tuple[int, int, int, int]:
    """Materialise valid taxonomy/tag IDs and return the four scalar counts."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX {stage_taxonomy_index}
                ON {stage} (taxonomy_id) TABLESPACE {tablespace}
                """
            ).format(
                stage_taxonomy_index=sql.Identifier(
                    f"{tables.stage}_taxonomy_idx"
                ),
                stage=sql.Identifier("pg_temp", tables.stage),
                tablespace=sql.Identifier(db.DEFAULT_TABLESPACE),
            )
        )
        cur.execute(
            sql.SQL("ANALYZE {stage}").format(
                stage=sql.Identifier("pg_temp", tables.stage)
            )
        )
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {valid_taxonomies} (taxonomy_id)
                SELECT taxonomy_id
                FROM {stage}
                GROUP BY taxonomy_id
                HAVING COUNT(*) >= %s
                """
            ).format(
                valid_taxonomies=sql.Identifier(
                    "pg_temp", tables.valid_taxonomies
                ),
                stage=sql.Identifier("pg_temp", tables.stage),
            ),
            (min_samples_per_taxonomy,),
        )
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {valid_tags} (tag)
                SELECT product_tag.tag
                FROM {stage} AS staged
                JOIN {valid_taxonomies} AS valid_taxonomy USING (taxonomy_id)
                CROSS JOIN LATERAL unnest(staged.tags) AS product_tag(tag)
                GROUP BY product_tag.tag
                HAVING COUNT(*) >= %s
                """
            ).format(
                valid_tags=sql.Identifier("pg_temp", tables.valid_tags),
                stage=sql.Identifier("pg_temp", tables.stage),
                valid_taxonomies=sql.Identifier(
                    "pg_temp", tables.valid_taxonomies
                ),
            ),
            (min_tag_count,),
        )
        cur.execute(
            sql.SQL(
                """
                SELECT
                    (SELECT COUNT(*) FROM {stage}),
                    (SELECT COUNT(*)
                     FROM {stage} AS staged
                     JOIN {valid_taxonomies} AS valid_taxonomy
                       USING (taxonomy_id)),
                    (SELECT COUNT(*) FROM {valid_taxonomies}),
                    (SELECT COUNT(*) FROM {valid_tags})
                """
            ).format(
                stage=sql.Identifier("pg_temp", tables.stage),
                valid_taxonomies=sql.Identifier(
                    "pg_temp", tables.valid_taxonomies
                ),
                valid_tags=sql.Identifier("pg_temp", tables.valid_tags),
            )
        )
        counts = cur.fetchone()

    return tuple(int(count) for count in counts)


def _stream_tag_eligible_rows(
    conn,
    tables: _AuditTempTables,
):
    query = sql.SQL(
        """
        SELECT
            staged.product_id,
            staged.taxonomy_id,
            staged.product_url,
            staged.myshopify_domain,
            staged.product_handle
        FROM {stage} AS staged
        JOIN {valid_taxonomies} AS valid_taxonomy USING (taxonomy_id)
        WHERE EXISTS (
            SELECT 1
            FROM unnest(staged.tags) AS product_tag(tag)
            JOIN {valid_tags} AS valid_tag USING (tag)
        )
        ORDER BY staged.product_id
        """
    ).format(
        stage=sql.Identifier("pg_temp", tables.stage),
        valid_taxonomies=sql.Identifier("pg_temp", tables.valid_taxonomies),
        valid_tags=sql.Identifier("pg_temp", tables.valid_tags),
    )

    cursor_name = f"padjective_audit_candidates_{uuid.uuid4().hex}"
    with conn.cursor(name=cursor_name, row_factory=dict_row) as cur:
        cur.itersize = AUDIT_BATCH_SIZE
        cur.execute(query)
        yield from cur


def _flush_canonical_products(
    conn,
    tables: _AuditTempTables,
    rows: list[tuple[str, str]],
) -> None:
    if not rows:
        return
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                """
                INSERT INTO {canonical_products} (product_id_hash, taxonomy_id)
                VALUES (%s, %s)
                ON CONFLICT (product_id_hash) DO NOTHING
                """
            ).format(
                canonical_products=sql.Identifier(
                    "pg_temp", tables.canonical_products
                )
            ),
            rows,
        )
    rows.clear()


def _calculate_eligibility_audit_from_postgres(
    conn,
    *,
    product_table: str,
    schema: str,
    min_tag_count: int,
    min_samples_per_taxonomy: int,
) -> EligibilityAudit:
    """Calculate the funnel with bounded application memory.

    The full source join is consumed through a server-side cursor. Only
    resolved products with nonempty filtered tags are staged in session-local
    Postgres tables; taxonomy/tag aggregation and URL deduplication therefore
    do not retain product-sized Python collections.
    """

    tables = _AuditTempTables.create()
    _create_audit_temp_tables(conn, tables)

    products_with_title = 0
    products_with_details = 0
    products_with_taxonomy_metadata = 0
    products_with_resolved_numeric_path = 0
    raw_numeric_path_products = 0
    reconciled_display_path_products = 0
    unresolved_taxonomy_path_products = 0
    stage_rows: list[
        tuple[int, str, list[str], str | None, str | None, str | None]
    ] = []

    for row in _stream_audit_rows(conn, product_table=product_table, schema=schema):
        if row.title is None:
            continue
        products_with_title += 1
        if not row.has_product_details:
            continue
        products_with_details += 1
        if not row.taxonomy_id or not row.taxonomy_name:
            continue
        products_with_taxonomy_metadata += 1

        observed_is_numeric = taxonomy_paths.is_numeric_taxonomy_path(
            row.observed_taxonomy_path
        )
        resolved_is_numeric = taxonomy_paths.is_numeric_taxonomy_path(
            row.resolved_taxonomy_path
        )
        if observed_is_numeric:
            raw_numeric_path_products += 1
        elif resolved_is_numeric:
            reconciled_display_path_products += 1
        if not resolved_is_numeric:
            unresolved_taxonomy_path_products += 1
            continue

        products_with_resolved_numeric_path += 1
        tags = tagbattle.filter_nested_tags(data_access.parse_tags(row.raw_tags))
        if not tags:
            continue
        stage_rows.append(
            (
                row.product_id,
                row.taxonomy_id,
                tags,
                row.product_url,
                row.myshopify_domain,
                row.product_handle,
            )
        )
        if len(stage_rows) >= AUDIT_BATCH_SIZE:
            _flush_audit_stage_rows(conn, tables, stage_rows)
    _flush_audit_stage_rows(conn, tables, stage_rows)

    (
        products_with_nonempty_tags,
        products_in_taxonomies_meeting_minimum,
        taxonomies_meeting_minimum,
        eligible_tags,
    ) = _prepare_audit_aggregates(
        conn,
        tables,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )

    products_with_frequent_tags = 0
    products_with_canonical_url = 0
    canonical_rows: list[tuple[str, str]] = []
    for row in _stream_tag_eligible_rows(
        conn,
        tables,
    ):
        products_with_frequent_tags += 1
        canonical_url = canonicalize_product_url(
            row.get("product_url"),
            myshopify_domain=row.get("myshopify_domain"),
            product_handle=row.get("product_handle"),
        )
        if not canonical_url:
            continue
        products_with_canonical_url += 1
        canonical_rows.append(
            (hash_product_url(canonical_url), str(row["taxonomy_id"]))
        )
        if len(canonical_rows) >= AUDIT_BATCH_SIZE:
            _flush_canonical_products(conn, tables, canonical_rows)
    _flush_canonical_products(conn, tables, canonical_rows)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(*), COUNT(DISTINCT taxonomy_id)
                FROM {canonical_products}
                """
            ).format(
                canonical_products=sql.Identifier(
                    "pg_temp", tables.canonical_products
                )
            )
        )
        benchmark_products, benchmark_taxonomies = cur.fetchone()

    return EligibilityAudit(
        catalogue_products=_catalogue_product_count(conn, product_table),
        products_with_title=products_with_title,
        products_with_details=products_with_details,
        products_with_taxonomy_metadata=products_with_taxonomy_metadata,
        products_with_resolved_numeric_path=products_with_resolved_numeric_path,
        products_with_nonempty_tags=products_with_nonempty_tags,
        products_in_taxonomies_meeting_minimum=(
            products_in_taxonomies_meeting_minimum
        ),
        products_with_frequent_tags=products_with_frequent_tags,
        products_with_canonical_url=products_with_canonical_url,
        benchmark_products_after_url_deduplication=int(benchmark_products),
        raw_numeric_path_products=raw_numeric_path_products,
        reconciled_display_path_products=reconciled_display_path_products,
        unresolved_taxonomy_path_products=unresolved_taxonomy_path_products,
        taxonomies_meeting_minimum=taxonomies_meeting_minimum,
        benchmark_taxonomies_after_filters=int(benchmark_taxonomies),
        eligible_tags=eligible_tags,
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

    cursor_name = f"padjective_audit_source_{uuid.uuid4().hex}"
    with conn.cursor(name=cursor_name, row_factory=dict_row) as cur:
        cur.itersize = AUDIT_BATCH_SIZE
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
    eligibility = _calculate_eligibility_audit_from_postgres(
        conn,
        product_table=product_table,
        schema=schema,
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
