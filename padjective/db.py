"""Utilities for interacting with the Shopify Postgres database."""

from __future__ import annotations

import os
from typing import Iterable, Tuple

import psycopg
from psycopg import sql


DEFAULT_TABLESPACE = "pg_default"


def get_connection(dsn: str | None = None) -> psycopg.Connection:
    """Return a psycopg connection using ``dsn`` or environment defaults.

    The function first checks ``dsn`` and the ``SHOPIFY_DB_DSN`` and
    ``DATABASE_URL`` environment variables. If none of those values are
    provided we fall back to the ``shopifystores`` database name. This avoids
    unexpected connections when ``PGDATABASE`` defaults to the current user
    name (``psycopg``'s behaviour when no parameters are supplied).
    """

    effective_dsn = dsn or os.getenv("SHOPIFY_DB_DSN") or os.getenv("DATABASE_URL")
    if effective_dsn:
        return psycopg.connect(effective_dsn)
    return psycopg.connect(dbname="shopifystores")


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
    """Ensure ``schema`` exists and is available to the current user."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = %s
            """,
            (schema,),
        )
        if cur.fetchone() is None:
            try:
                cur.execute(
                    sql.SQL(
                        "CREATE SCHEMA IF NOT EXISTS {schema} AUTHORIZATION CURRENT_USER"
                    ).format(schema=sql.Identifier(schema))
                )
            except psycopg.Error as exc:  # pragma: no cover - defensive guard
                conn.rollback()
                raise RuntimeError(
                    "Schema %s is not available. Ask a database administrator to create it "
                    "and grant access before running this command." % schema
                ) from exc
            else:
                conn.commit()


def ensure_table(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    columns_sql: Iterable[str],
    indexes_sql: Iterable[object] | None = None,
    table_tablespace: str = DEFAULT_TABLESPACE,
    index_tablespace: str | None = None,
) -> None:
    """Ensure a table exists using the provided column and index SQL fragments."""

    column_block = ",\n".join(columns_sql)

    if table_tablespace != DEFAULT_TABLESPACE:
        raise ValueError(
            "table_tablespace must be pg_default to comply with deployment requirements"
        )

    if index_tablespace and index_tablespace != DEFAULT_TABLESPACE:
        raise ValueError(
            "index_tablespace must be pg_default to comply with deployment requirements"
        )

    effective_index_tablespace = index_tablespace or table_tablespace

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        table_exists = cur.fetchone() is not None

        if not table_exists:
            tablespace_identifier = sql.Identifier(DEFAULT_TABLESPACE)
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {schema}.{table} (
                        {columns}
                    ) TABLESPACE {tablespace}
                    """
                ).format(
                    schema=sql.Identifier(schema),
                    table=sql.Identifier(table),
                    columns=sql.SQL(column_block),
                    tablespace=tablespace_identifier,
                )
            )
        if indexes_sql:
            for statement in indexes_sql:
                if hasattr(statement, "as_string"):
                    statement_text = statement.as_string(conn)
                else:
                    statement_text = str(statement)

                if effective_index_tablespace and "TABLESPACE" not in statement_text.upper():
                    index_tablespace_sql = sql.Identifier(effective_index_tablespace).as_string(conn)
                    statement_text = f"{statement_text} TABLESPACE {index_tablespace_sql}"

                cur.execute(statement_text)
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
