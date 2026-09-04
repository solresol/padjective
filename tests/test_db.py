import pytest
from psycopg import sql

from padjective import db


class RecordingConnection:
    def __init__(self, table_exists: bool) -> None:
        self.table_exists = table_exists
        self.statements: list[tuple[object, object | None]] = []
        self.commits = 0
        self.fetchone_calls = 0

    def cursor(self) -> "RecordingCursor":
        return RecordingCursor(self)

    def commit(self) -> None:
        self.commits += 1


class RecordingCursor:
    def __init__(self, connection: RecordingConnection) -> None:
        self.connection = connection

    def __enter__(self) -> "RecordingCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement: object, params: object | None = None) -> None:
        self.connection.statements.append((statement, params))

    def fetchone(self) -> tuple[int] | None:
        self.connection.fetchone_calls += 1
        if self.connection.table_exists:
            return (1,)
        return None


def test_ensure_table_rejects_non_default_table_tablespace() -> None:
    with pytest.raises(ValueError, match="table_tablespace must be pg_default"):
        db.ensure_table(
            conn=None,  # type: ignore[arg-type]
            schema="padjective",
            table="example",
            columns_sql=["id INTEGER"],
            table_tablespace="shopifystores_space",
        )


def test_ensure_table_rejects_non_default_index_tablespace() -> None:
    with pytest.raises(ValueError, match="index_tablespace must be pg_default"):
        db.ensure_table(
            conn=None,  # type: ignore[arg-type]
            schema="padjective",
            table="example",
            columns_sql=["id INTEGER"],
            index_tablespace="shopifystores_space",
        )


def test_ensure_table_skips_creation_when_table_exists() -> None:
    conn = RecordingConnection(table_exists=True)

    db.ensure_table(
        conn=conn,
        schema="padjective",
        table="example",
        columns_sql=["id INTEGER"],
    )

    assert conn.fetchone_calls == 1
    assert conn.commits == 1
    assert len(conn.statements) == 1
    statement, params = conn.statements[0]
    assert isinstance(statement, str)
    assert "information_schema.tables" in statement
    assert params == ("padjective", "example")


def test_ensure_table_skips_index_ddl_when_table_exists() -> None:
    conn = RecordingConnection(table_exists=True)

    db.ensure_table(
        conn=conn,
        schema="padjective",
        table="example",
        columns_sql=["id INTEGER"],
        indexes_sql=[
            "CREATE INDEX IF NOT EXISTS example_id_idx "
            "ON padjective.example (id) TABLESPACE pg_default"
        ],
    )

    assert conn.fetchone_calls == 1
    assert conn.commits == 1
    assert len(conn.statements) == 1
    statement, params = conn.statements[0]
    assert isinstance(statement, str)
    assert "information_schema.tables" in statement
    assert params == ("padjective", "example")


def test_ensure_table_creates_when_table_is_missing() -> None:
    conn = RecordingConnection(table_exists=False)

    db.ensure_table(
        conn=conn,
        schema="padjective",
        table="example",
        columns_sql=["id INTEGER"],
    )

    assert conn.fetchone_calls == 1
    assert conn.commits == 1
    assert len(conn.statements) == 3
    set_statement, set_params = conn.statements[1]
    assert isinstance(set_statement, str)
    assert "set_config('default_tablespace'" in set_statement
    assert set_params == ("pg_default",)
    create_statement, create_params = conn.statements[2]
    assert isinstance(create_statement, sql.Composed)
    assert create_params is None


def test_ensure_table_creates_indexes_with_a_missing_table() -> None:
    conn = RecordingConnection(table_exists=False)

    db.ensure_table(
        conn=conn,
        schema="padjective",
        table="example",
        columns_sql=["id INTEGER"],
        indexes_sql=[
            "CREATE INDEX IF NOT EXISTS example_id_idx "
            "ON padjective.example (id) TABLESPACE pg_default"
        ],
    )

    assert conn.fetchone_calls == 1
    assert conn.commits == 1
    assert len(conn.statements) == 4
    index_statement, index_params = conn.statements[3]
    assert index_statement.endswith("TABLESPACE pg_default")
    assert index_params is None
