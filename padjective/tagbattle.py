"""Utilities for ranking product tags by their position in titles."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:
    from . import db


@dataclass(frozen=True)
class Battle:
    """Simple container representing a pairwise comparison."""

    product_id: int | None
    winner_tag: str
    loser_tag: str


def filter_nested_tags(tags: Iterable[str]) -> List[str]:
    """Remove tags that are substrings of other tags.

    Tags are compared case-insensitively and returned in their original order
    without duplicates.
    """

    unique: List[str] = []
    seen = set()
    for tag in tags:
        tag = tag.strip()
        if not tag or tag.lower() in seen:
            continue
        unique.append(tag)
        seen.add(tag.lower())

    filtered: List[str] = []
    for tag in unique:
        tag_lower = tag.lower()
        if any(
            tag_lower != other.lower() and tag_lower in other.lower()
            for other in unique
        ):
            continue
        filtered.append(tag)
    return filtered


def split_title(title: str) -> List[str]:
    """Split a product title on ``" - "`` if present."""

    return [part.strip() for part in title.split(" - ")]


def tag_positions(title: str, tags: Iterable[str]) -> Dict[str, int]:
    """Return the start index of each tag found in ``title``.

    The search is case-insensitive and only the first occurrence is recorded.
    Tags that are not present are omitted from the result.
    """

    lower_title = title.lower()
    positions: Dict[str, int] = {}
    for tag in tags:
        idx = lower_title.find(tag.lower())
        if idx != -1:
            positions[tag] = idx
    return positions


def build_battles(title: str, tag_string: str) -> List[Tuple[str, str]]:
    """Return ordered tag pairs derived from ``title`` and ``tag_string``."""

    tags = [t.strip() for t in tag_string.split(",") if t.strip()]
    tags = [t.upper() for t in filter_nested_tags(tags)]
    pairs: List[Tuple[str, str]] = []
    for part in split_title(title):
        positions = tag_positions(part, tags)
        if len(positions) < 2:
            continue
        ordered_tags = list(positions.keys())
        for i in range(len(ordered_tags)):
            for j in range(i + 1, len(ordered_tags)):
                t1, t2 = ordered_tags[i], ordered_tags[j]
                if positions[t1] == positions[t2]:
                    continue
                if positions[t1] < positions[t2]:
                    pairs.append((t1, t2))
                else:
                    pairs.append((t2, t1))
    return pairs


def ensure_storage(conn, schema: str) -> None:
    """Ensure the ``padjective`` schema and ``battles`` table exist."""

    db.ensure_schema(conn, schema)
    battle_columns = (
        "product_id BIGINT",
        "winner_tag TEXT NOT NULL",
        "loser_tag TEXT NOT NULL",
        "recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    indexes = (
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index} "
            "ON {schema}.battles (winner_tag) TABLESPACE pg_default"
        ).format(
            index=sql.Identifier(f"{schema}_battles_winner_idx"),
            schema=sql.Identifier(schema),
        ),
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {index} "
            "ON {schema}.battles (loser_tag) TABLESPACE pg_default"
        ).format(
            index=sql.Identifier(f"{schema}_battles_loser_idx"),
            schema=sql.Identifier(schema),
        ),
    )
    db.ensure_table(conn, schema, "battles", battle_columns, indexes_sql=indexes)


def insert_battles(conn, schema: str, battles: Sequence[Battle]) -> None:
    """Persist a batch of ``Battle`` objects to Postgres."""

    if not battles:
        return

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "INSERT INTO {schema}.battles (product_id, winner_tag, loser_tag) "
                "VALUES (%s, %s, %s)"
            ).format(schema=sql.Identifier(schema)),
            [(b.product_id, b.winner_tag, b.loser_tag) for b in battles],
        )
    conn.commit()


def stream_products(conn, taxonomy_table: str, product_view: str):
    """Yield product rows filtered by entries in the taxonomy table."""

    taxonomy_identifier = db.qualified_identifier(taxonomy_table)
    product_identifier = db.qualified_identifier(product_view)
    taxonomy_key_column = sql.Identifier("product_key")
    product_key_column = sql.Identifier("key")
    query = sql.SQL(
        """
        WITH selected_products AS (
            SELECT DISTINCT {taxonomy_key} AS product_key
            FROM {taxonomy}
        )
        SELECT p.id, p.title, p.tags
        FROM {products} AS p
        JOIN selected_products sp ON sp.product_key = p.{product_key}
        WHERE p.title IS NOT NULL
        ORDER BY p.id
        """
    ).format(
        taxonomy_key=taxonomy_key_column,
        taxonomy=taxonomy_identifier,
        products=product_identifier,
        product_key=product_key_column,
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            yield row


def process_database(
    dsn: str | None,
    schema: str,
    taxonomy_table: str,
    product_view: str,
    batch_size: int = 1000,
) -> None:
    """Stream Shopify data and populate the battles table."""

    conn = db.get_connection(dsn)
    ensure_storage(conn, schema)

    buffer: List[Battle] = []
    for row in stream_products(conn, taxonomy_table, product_view):
        title = row.get("title") or ""
        tag_string = row.get("tags") or ""
        product_id = row.get("id")
        for winner, loser in build_battles(title, tag_string):
            buffer.append(Battle(product_id=product_id, winner_tag=winner, loser_tag=loser))
        if len(buffer) >= batch_size:
            insert_battles(conn, schema, buffer)
            buffer.clear()

    if buffer:
        insert_battles(conn, schema, buffer)

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tag ordering within product titles using Postgres data."
    )
    parser.add_argument(
        "--dsn",
        help=(
            "Postgres DSN for the Shopify stores database. If omitted, the script "
            "uses SHOPIFY_DB_DSN or DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Destination schema for derived tables.",
    )
    parser.add_argument(
        "--taxonomy-table",
        default="cantbuymelove.product_taxonomy",
        help="Qualified taxonomy table to select product IDs from.",
    )
    parser.add_argument(
        "--product-view",
        default="cantbuymelove.product",
        help="Qualified product view containing titles and tags.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Number of comparisons to buffer before writing to Postgres.",
    )
    args = parser.parse_args()
    process_database(
        dsn=args.dsn,
        schema=args.schema,
        taxonomy_table=args.taxonomy_table,
        product_view=args.product_view,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
