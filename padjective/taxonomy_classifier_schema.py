"""Verify Postgres objects required for taxonomy classifier results."""

from __future__ import annotations

import argparse

from typing import Iterable

from . import db


EXPECTED_TABLES: tuple[str, ...] = (
    "taxonomy_lr_models",
    "taxonomy_lr_cv_scores",
    "taxonomy_lr_class_distribution",
    "taxonomy_lr_tag_summary",
    "taxonomy_lr_top_tags",
    "taxonomy_lr_intercepts",
)


def _check_schema_exists(conn, schema: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = %s
            """,
            (schema,),
        )
        return cur.fetchone() is not None


def _missing_tables(conn, schema: str, tables: Iterable[str]) -> list[str]:
    missing: list[str] = []
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            if cur.fetchone() is None:
                missing.append(table)
    return missing


def ensure_taxonomy_classifier_schema(conn, schema: str) -> None:
    """Validate presence of taxonomy classifier tables without creating them."""

    if not _check_schema_exists(conn, schema):
        raise RuntimeError(
            "Schema %s does not exist. Ask a database administrator to create it and "
            "run the manual DDL script." % schema
        )

    missing = _missing_tables(conn, schema, EXPECTED_TABLES)
    if missing:
        missing_csv = ", ".join(sorted(missing))
        raise RuntimeError(
            "Missing required tables in schema %s: %s. Run the manual DDL script "
            "before executing this command." % (schema, missing_csv)
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that Postgres tables for taxonomy classifier results already exist",
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema to prepare",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        ensure_taxonomy_classifier_schema(conn, args.schema)
    finally:
        conn.close()

    print(
        f"Verified taxonomy classifier tables exist in schema {args.schema}.",
    )


if __name__ == "__main__":
    main()
