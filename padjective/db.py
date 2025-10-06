"""Utilities for interacting with the Shopify Postgres database."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import psycopg
from psycopg import sql


def get_connection(dsn: str | None = None) -> psycopg.Connection:
    """Return a psycopg connection using ``dsn`` or environment defaults.

    The function first checks ``dsn`` and the ``SHOPIFY_DB_DSN`` and
    ``DATABASE_URL`` environment variables. If none of those values are
    provided we fall back to ``psycopg``'s default parameter resolution which
    honours the standard ``PG*`` environment variables (``PGHOST``,
    ``PGDATABASE``, etc.). This allows the application to run in environments
    where those variables are already populated without requiring an explicit
    DSN.
    """

    effective_dsn = dsn or os.getenv("SHOPIFY_DB_DSN") or os.getenv("DATABASE_URL")
    if effective_dsn:
        return psycopg.connect(effective_dsn)
    return psycopg.connect()


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
