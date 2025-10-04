"""Utilities for interacting with the Shopify Postgres database."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import psycopg
from psycopg import sql


def get_connection(dsn: str | None = None) -> psycopg.Connection:
    """Return a psycopg connection using ``dsn`` or environment defaults.

    The function checks the ``SHOPIFY_DB_DSN`` environment variable first and
    then falls back to ``DATABASE_URL``. A missing DSN results in a
    ``RuntimeError`` with a helpful message so that callers can surface the
    configuration problem early.
    """

    effective_dsn = dsn or os.getenv("SHOPIFY_DB_DSN") or os.getenv("DATABASE_URL")
    if not effective_dsn:
        raise RuntimeError(
            "Provide a Postgres DSN via --dsn, SHOPIFY_DB_DSN, or DATABASE_URL"
        )
    return psycopg.connect(effective_dsn)


def _split_qualified_name(name: str) -> Tuple[str, str]:
    parts = name.split(".")
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            f"Expected a fully qualified identifier in the form schema.table, got {name!r}"
        )
    return parts[0], parts[1]


def qualified_identifier(name: str) -> sql.Identifier:
    """Return a safe SQL identifier for ``schema.table`` strings."""

    schema, table = _split_qualified_name(name)
    return sql.Identifier(schema, table)


def ensure_schema(conn: psycopg.Connection, schema: str) -> None:
    """Create ``schema`` if it does not yet exist."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {schema}").format(
                schema=sql.Identifier(schema)
            )
        )
    conn.commit()


def ensure_table(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    columns_sql: Iterable[str],
    indexes_sql: Iterable[str] | None = None,
) -> None:
    """Ensure a table exists using the provided column and index SQL fragments."""

    column_block = ",\n".join(columns_sql)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.{table} (
                    {columns}
                ) TABLESPACE pg_default
                """
            ).format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(table),
                columns=sql.SQL(column_block),
            )
        )
        if indexes_sql:
            for statement in indexes_sql:
                cur.execute(statement)
    conn.commit()


def truncate_table(conn: psycopg.Connection, schema: str, table: str) -> None:
    """Remove all rows from ``schema.table``."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {schema}.{table}").format(
                schema=sql.Identifier(schema),
                table=sql.Identifier(table),
            )
        )
    conn.commit()
