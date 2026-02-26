"""Build reproducible, anonymized dataset snapshots for product-taxonomy-bench.

This module creates a deterministic, publishable snapshot of Shopify product tag
data (as scraped into the Shopify stores Postgres database) while avoiding
redistributing raw tags, titles, or URLs.

Snapshot outputs are persisted to the ``padjective`` schema in Postgres.

Dataset design (per user request):
- Tags are anonymised to ``tagNNNNNN`` identifiers.
- For each product+tag we record whether the tag appears in the title and, if
  so, the first (title_part, position) where it appears.
- Title overlap positions are also recorded per title part (splitting on
  ``" - "``), matching the "tag battle" logic in the paper.
- The product identifier is a SHA-256 hash of a canonical product URL.
- Taxonomy targets are stored as taxonomy_id/path/name.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db, tagbattle
    from padjective.cv import calculate_cv_folds
else:  # pragma: no cover - imported as a package
    from . import data_access, db, tagbattle
    from .cv import calculate_cv_folds


DEFAULT_SCHEMA = "padjective"
TAG_PREFIX = "tag"
TAG_WIDTH = 6
_UTC_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2})(?::(?P<second>\d{2}))?\s*UTC",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SourceProductRow:
    product_id: int
    title: str
    product_url: Optional[str]
    myshopify_domain: Optional[str]
    product_handle: Optional[str]
    raw_tags: Optional[str]
    taxonomy_id: str
    taxonomy_path: str
    taxonomy_name: str


def build_tag_id_map(
    tags: Iterable[str],
    *,
    prefix: str = TAG_PREFIX,
    width: int = TAG_WIDTH,
) -> Dict[str, str]:
    """Return a deterministic anonymization mapping for tags.

    Tags are sorted lexicographically (after normalisation by callers) and
    assigned sequential identifiers starting at 1.
    """

    ordered = sorted(set(tags))
    return {tag: f"{prefix}{idx + 1:0{width}d}" for idx, tag in enumerate(ordered)}


def canonicalize_product_url(
    product_url: Optional[str],
    *,
    myshopify_domain: Optional[str] = None,
    product_handle: Optional[str] = None,
) -> Optional[str]:
    """Return a canonical URL suitable for stable hashing.

    Prefers the stored ``product_url``; falls back to
    ``https://{myshopify_domain}/products/{product_handle}`` when needed.

    The canonicalisation normalises scheme/host casing and strips query strings,
    fragments, and trailing slashes.
    """

    url = (product_url or "").strip()
    if not url:
        domain = (myshopify_domain or "").strip()
        handle = (product_handle or "").strip().lstrip("/")
        if not domain or not handle:
            return None
        url = f"https://{domain}/products/{handle}"

    url = url.strip()
    if "://" not in url:
        url = "https://" + url.lstrip("/")

    parts = urlsplit(url)
    scheme = (parts.scheme or "https").lower()
    netloc = parts.netloc.lower()
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def hash_product_url(url: str) -> str:
    """Return a SHA-256 hex digest for ``url``."""

    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def title_parts(title: str) -> List[str]:
    """Split the title into parts matching the tag-battle logic."""

    parts = tagbattle.split_title(title or "")
    return parts or [""]


def title_part_positions(title: str, tags: Sequence[str]) -> Dict[int, Dict[str, int]]:
    """Return tag overlap positions for each title part.

    The returned mapping is ``{part_idx: {tag: position}}`` and only includes
    tags that appear in the corresponding title part.
    """

    positions: Dict[int, Dict[str, int]] = {}
    for part_idx, part in enumerate(title_parts(title)):
        found = tagbattle.tag_positions(part, tags)
        if found:
            positions[part_idx] = found
    return positions


def first_title_occurrences(
    part_positions: Mapping[int, Mapping[str, int]],
) -> Dict[str, Tuple[int, int]]:
    """Return earliest (title_part, position) for each tag."""

    first: Dict[str, Tuple[int, int]] = {}
    for part_idx, tag_map in part_positions.items():
        for tag, position in tag_map.items():
            candidate = (int(part_idx), int(position))
            existing = first.get(tag)
            if existing is None or candidate < existing:
                first[tag] = candidate
    return first


def _git_head() -> Optional[str]:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:  # pragma: no cover - best-effort only
        return None


def _table_has_column(conn, schema: str, table: str, column: str) -> bool:
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


def parse_as_of(value: str) -> datetime:
    """Parse an as-of timestamp from either ISO8601 or ``YYYY-MM-DD HH:MM UTC`` strings."""

    text = value.strip()
    if not text:
        raise ValueError("as-of timestamp must not be empty")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        match = _UTC_TIMESTAMP_RE.search(text)
        if not match:
            raise
        second = match.group("second") or "00"
        parsed = datetime.fromisoformat(
            f"{match.group('date')}T{match.group('time')}:{second}+00:00"
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _stream_source_products(
    conn, product_table: str, *, as_of: datetime | None = None
) -> Iterator[SourceProductRow]:
    product_identifier = db.qualified_identifier(product_table)
    conditions: list[sql.SQL] = [sql.SQL("p.product_title IS NOT NULL")]
    params: list[object] = []

    if as_of is not None and _table_has_column(conn, "public", "product_details", "when_fetched"):
        conditions.append(sql.SQL("pd.when_fetched <= %s"))
        params.append(as_of.replace(tzinfo=None))

    where_clause = sql.SQL(" AND ").join(conditions)

    query = sql.SQL(
        """
        SELECT
            p.id,
            p.product_title AS title,
            p.product_url,
            p.myshopify_domain,
            p.product_handle,
            pd.product_detail->'product'->>'tags' AS tags,
            pt.taxonomy_id,
            t.taxonomy_path,
            t.taxonomy_name
        FROM {products} AS p
        JOIN public.product_details pd ON (
            p.myshopify_domain = pd.myshopify_domain
            AND p.run_name = pd.run_name
            AND p.product_handle = pd.product_handle
        )
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
        WHERE {where_clause}
        ORDER BY p.id
        """
    ).format(products=product_identifier, where_clause=where_clause)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, params)
        for row in cur:
            taxonomy_id = row.get("taxonomy_id")
            taxonomy_path = row.get("taxonomy_path")
            taxonomy_name = row.get("taxonomy_name")
            if taxonomy_id is None or taxonomy_path is None or taxonomy_name is None:
                continue
            yield SourceProductRow(
                product_id=int(row.get("id")),
                title=str(row.get("title") or ""),
                product_url=row.get("product_url"),
                myshopify_domain=row.get("myshopify_domain"),
                product_handle=row.get("product_handle"),
                raw_tags=row.get("tags"),
                taxonomy_id=str(taxonomy_id),
                taxonomy_path=str(taxonomy_path),
                taxonomy_name=str(taxonomy_name),
            )


def _ensure_storage(conn, schema: str) -> None:
    db.ensure_schema(conn, schema)

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_snapshots",
        (
            "snapshot_id UUID PRIMARY KEY",
            "snapshot_name TEXT NOT NULL UNIQUE",
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            "as_of TIMESTAMPTZ",
            "product_table TEXT NOT NULL",
            "min_tag_count INTEGER NOT NULL",
            "min_samples_per_taxonomy INTEGER NOT NULL",
            "product_count INTEGER NOT NULL",
            "tag_count INTEGER NOT NULL",
            "taxonomy_count INTEGER NOT NULL",
            "note TEXT",
            "code_version TEXT",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_snapshots (created_at) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_snapshots_created_at_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "ALTER TABLE {schema}.product_taxonomy_bench_snapshots "
                "ADD COLUMN IF NOT EXISTS as_of TIMESTAMPTZ"
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_snapshot_aliases",
        (
            "alias TEXT PRIMARY KEY",
            "snapshot_id UUID NOT NULL",
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
        ),
    )

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_products",
        (
            "snapshot_id UUID NOT NULL",
            "product_id_hash TEXT NOT NULL",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT NOT NULL",
            "taxonomy_name TEXT NOT NULL",
            "cv_fold INTEGER",
            "tag_count INTEGER NOT NULL",
            "title_part_count INTEGER NOT NULL",
            "PRIMARY KEY (snapshot_id, product_id_hash)",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_products (taxonomy_id) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_products_taxonomy_id_idx"),
                schema=sql.Identifier(schema),
            ),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_products (cv_fold) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_products_cv_fold_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_tags",
        (
            "snapshot_id UUID NOT NULL",
            "tag_id TEXT NOT NULL",
            "tag_rank INTEGER NOT NULL",
            "PRIMARY KEY (snapshot_id, tag_id)",
            "UNIQUE (snapshot_id, tag_rank)",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_tags (tag_rank) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_tags_rank_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_product_tags",
        (
            "snapshot_id UUID NOT NULL",
            "product_id_hash TEXT NOT NULL",
            "tag_id TEXT NOT NULL",
            "in_title BOOLEAN NOT NULL",
            "title_part INTEGER",
            "title_position INTEGER",
            "PRIMARY KEY (snapshot_id, product_id_hash, tag_id)",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_product_tags (tag_id) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_product_tags_tag_id_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )

    db.ensure_table(
        conn,
        schema,
        "product_taxonomy_bench_title_tags",
        (
            "snapshot_id UUID NOT NULL",
            "product_id_hash TEXT NOT NULL",
            "title_part INTEGER NOT NULL",
            "tag_id TEXT NOT NULL",
            "title_position INTEGER NOT NULL",
            "PRIMARY KEY (snapshot_id, product_id_hash, title_part, tag_id)",
        ),
        indexes_sql=(
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index} "
                "ON {schema}.product_taxonomy_bench_title_tags (tag_id) TABLESPACE pg_default"
            ).format(
                index=sql.Identifier(f"{schema}_ptb_title_tags_tag_id_idx"),
                schema=sql.Identifier(schema),
            ),
        ),
    )


def _snapshot_name_exists(conn, schema: str, snapshot_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT 1 FROM {schema}.product_taxonomy_bench_snapshots WHERE snapshot_name = %s"
            ).format(schema=sql.Identifier(schema)),
            (snapshot_name,),
        )
        return cur.fetchone() is not None


def _count_taxonomies(
    conn,
    product_table: str,
    *,
    as_of: datetime | None,
) -> Counter[str]:
    taxonomy_counts: Counter[str] = Counter()
    for row in _stream_source_products(conn, product_table, as_of=as_of):
        tags = tagbattle.filter_nested_tags(data_access.parse_tags(row.raw_tags))
        if not tags:
            continue
        if not data_access.is_valid_taxonomy_path(row.taxonomy_path):
            continue
        taxonomy_counts[row.taxonomy_id] += 1
    return taxonomy_counts


def _count_tags(
    conn,
    product_table: str,
    *,
    valid_taxonomies: set[str],
    as_of: datetime | None,
) -> Counter[str]:
    tag_counts: Counter[str] = Counter()
    for row in _stream_source_products(conn, product_table, as_of=as_of):
        if row.taxonomy_id not in valid_taxonomies:
            continue
        if not data_access.is_valid_taxonomy_path(row.taxonomy_path):
            continue
        tags = tagbattle.filter_nested_tags(data_access.parse_tags(row.raw_tags))
        if not tags:
            continue
        tag_counts.update(tags)
    return tag_counts


def create_snapshot(
    *,
    dsn: Optional[str],
    schema: str,
    snapshot_name: str,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 5,
    min_samples_per_taxonomy: int = 5,
    as_of: datetime | None = None,
    alias: Optional[str] = None,
    note: Optional[str] = None,
    batch_size: int = 5000,
) -> uuid.UUID:
    """Create a new anonymised snapshot and persist it in Postgres."""

    conn = db.get_connection(dsn)
    try:
        _ensure_storage(conn, schema)

        if _snapshot_name_exists(conn, schema, snapshot_name):
            raise ValueError(f"Snapshot name already exists: {snapshot_name!r}")

        snapshot_id = uuid.uuid4()
        code_version = _git_head()

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {schema}.product_taxonomy_bench_snapshots (
                        snapshot_id,
                        snapshot_name,
                        product_table,
                        as_of,
                        min_tag_count,
                        min_samples_per_taxonomy,
                        product_count,
                        tag_count,
                        taxonomy_count,
                        note,
                        code_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 0, %s, %s)
                    """
                ).format(schema=sql.Identifier(schema)),
                (
                    snapshot_id,
                    snapshot_name,
                    product_table,
                    as_of,
                    min_tag_count,
                    min_samples_per_taxonomy,
                    note,
                    code_version,
                ),
            )
        conn.commit()

        fold_assignments = calculate_cv_folds(conn, product_table)

        taxonomy_counts = _count_taxonomies(conn, product_table, as_of=as_of)
        valid_taxonomies = {
            taxonomy_id
            for taxonomy_id, count in taxonomy_counts.items()
            if count >= min_samples_per_taxonomy
        }

        tag_counts = _count_tags(
            conn, product_table, valid_taxonomies=valid_taxonomies, as_of=as_of
        )
        valid_tags = {
            tag for tag, count in tag_counts.items() if count >= min_tag_count
        }

        tag_id_map = build_tag_id_map(valid_tags)
        tag_id_for = tag_id_map.get

        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {schema}.product_taxonomy_bench_tags (snapshot_id, tag_id, tag_rank) "
                    "VALUES (%s, %s, %s)"
                ).format(schema=sql.Identifier(schema)),
                [
                    (snapshot_id, tag_id, rank)
                    for rank, tag_id in enumerate(
                        (tag_id_map[tag] for tag in sorted(valid_tags)), start=1
                    )
                ],
            )
        conn.commit()

        product_rows: List[Tuple[uuid.UUID, str, str, str, str, Optional[int], int, int]] = []
        product_tag_rows: List[Tuple[uuid.UUID, str, str, bool, Optional[int], Optional[int]]] = []
        title_tag_rows: List[Tuple[uuid.UUID, str, int, str, int]] = []

        product_count = 0
        taxonomy_ids: set[str] = set()

        def flush(cur) -> None:
            nonlocal product_rows, product_tag_rows, title_tag_rows
            if product_rows:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.product_taxonomy_bench_products (
                            snapshot_id,
                            product_id_hash,
                            taxonomy_id,
                            taxonomy_path,
                            taxonomy_name,
                            cv_fold,
                            tag_count,
                            title_part_count
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """
                    ).format(schema=sql.Identifier(schema)),
                    product_rows,
                )
                product_rows = []

            if product_tag_rows:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.product_taxonomy_bench_product_tags (
                            snapshot_id,
                            product_id_hash,
                            tag_id,
                            in_title,
                            title_part,
                            title_position
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """
                    ).format(schema=sql.Identifier(schema)),
                    product_tag_rows,
                )
                product_tag_rows = []

            if title_tag_rows:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.product_taxonomy_bench_title_tags (
                            snapshot_id,
                            product_id_hash,
                            title_part,
                            tag_id,
                            title_position
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """
                    ).format(schema=sql.Identifier(schema)),
                    title_tag_rows,
                )
                title_tag_rows = []

        with conn.cursor() as cur:
            for row in _stream_source_products(conn, product_table, as_of=as_of):
                if row.taxonomy_id not in valid_taxonomies:
                    continue
                if not data_access.is_valid_taxonomy_path(row.taxonomy_path):
                    continue

                raw_tags = tagbattle.filter_nested_tags(data_access.parse_tags(row.raw_tags))
                if not raw_tags:
                    continue

                filtered_tags = [tag for tag in raw_tags if tag in valid_tags]
                if not filtered_tags:
                    continue

                canonical_url = canonicalize_product_url(
                    row.product_url,
                    myshopify_domain=row.myshopify_domain,
                    product_handle=row.product_handle,
                )
                if not canonical_url:
                    continue

                product_hash = hash_product_url(canonical_url)
                cv_fold = fold_assignments.get(row.product_id)

                parts = title_parts(row.title)
                parts_positions = title_part_positions(row.title, filtered_tags)

                occurrences = first_title_occurrences(parts_positions)

                taxonomy_ids.add(row.taxonomy_id)
                product_rows.append(
                    (
                        snapshot_id,
                        product_hash,
                        row.taxonomy_id,
                        row.taxonomy_path,
                        row.taxonomy_name,
                        cv_fold,
                        len(filtered_tags),
                        len(parts),
                    )
                )

                for tag in filtered_tags:
                    tag_id = tag_id_for(tag)
                    if not tag_id:
                        continue
                    first = occurrences.get(tag)
                    in_title = first is not None
                    if in_title:
                        title_part, title_position = first
                    else:
                        title_part = None
                        title_position = None
                    product_tag_rows.append(
                        (
                            snapshot_id,
                            product_hash,
                            tag_id,
                            in_title,
                            title_part,
                            title_position,
                        )
                    )

                for part_idx, tag_positions in parts_positions.items():
                    for tag, pos in tag_positions.items():
                        tag_id = tag_id_for(tag)
                        if not tag_id:
                            continue
                        title_tag_rows.append(
                            (snapshot_id, product_hash, int(part_idx), tag_id, int(pos))
                        )

                product_count += 1
                if product_count % batch_size == 0:
                    flush(cur)
                    conn.commit()

            flush(cur)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    UPDATE {schema}.product_taxonomy_bench_snapshots
                    SET product_count = %s,
                        tag_count = %s,
                        taxonomy_count = %s
                    WHERE snapshot_id = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (product_count, len(valid_tags), len(taxonomy_ids), snapshot_id),
            )
        conn.commit()

        if alias:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {schema}.product_taxonomy_bench_snapshot_aliases (
                            alias,
                            snapshot_id
                        ) VALUES (%s, %s)
                        ON CONFLICT (alias) DO UPDATE SET
                            snapshot_id = EXCLUDED.snapshot_id,
                            updated_at = now()
                        """
                    ).format(schema=sql.Identifier(schema)),
                    (alias, snapshot_id),
                )
            conn.commit()

        return snapshot_id
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create anonymized, reproducible product-taxonomy-bench snapshots in Postgres."
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Schema to persist snapshot tables (default: padjective)",
    )
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table name",
    )
    parser.add_argument(
        "--snapshot-name",
        required=True,
        help="Unique snapshot name (e.g. paper-2026-02-11 or latest-2026-02-24)",
    )
    parser.add_argument(
        "--alias",
        help="Optional alias to update (e.g. latest or paper)",
    )
    parser.add_argument(
        "--note",
        help="Optional human note stored with the snapshot",
    )
    parser.add_argument(
        "--as-of",
        help=(
            "Optional point-in-time cutoff for the source data. "
            "Accepts ISO8601 (e.g. 2026-02-11T19:15:00Z) or 'YYYY-MM-DD HH:MM UTC'."
        ),
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=5,
        help="Minimum number of products a tag must appear in",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum number of products required per taxonomy label",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Flush insert batches every N products",
    )
    args = parser.parse_args()

    as_of = parse_as_of(args.as_of) if args.as_of else None

    snapshot_id = create_snapshot(
        dsn=args.dsn,
        schema=args.schema,
        snapshot_name=args.snapshot_name,
        product_table=args.product_table,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        as_of=as_of,
        alias=args.alias,
        note=args.note,
        batch_size=args.batch_size,
    )

    print(f"Created snapshot {args.snapshot_name} ({snapshot_id})")


if __name__ == "__main__":
    main()
