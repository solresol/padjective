from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from padjective import db


def connection(database="shopifystores", port=5433, major=18, user="padjective"):
    return Mock(info=SimpleNamespace(dbname=database, port=port, server_version=major * 10000, user=user))


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("SHOPIFY_DB_DSN", "DATABASE_URL", "PGPORT", "PGDATABASE", "PGHOST",
                 "PADJECTIVE_EXPECTED_DATABASE", "PADJECTIVE_EXPECTED_PORT",
                 "PADJECTIVE_EXPECTED_SERVER_MAJOR", "PADJECTIVE_EXPECTED_USER"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_explicit_pg18_socket(monkeypatch):
    conn = connection()
    connect = Mock(return_value=conn)
    monkeypatch.setattr(db.psycopg, "connect", connect)
    assert db.get_connection() is conn
    connect.assert_called_once_with(dbname="shopifystores", host="/var/run/postgresql", port="5433")
    conn.cursor.assert_not_called()


@pytest.mark.parametrize("bad", [connection(port=5432, major=16), connection(port=5432),
                                 connection(major=16), connection(database="other")])
def test_wrong_route_closes_before_application_sql(monkeypatch, bad):
    monkeypatch.setattr(db.psycopg, "connect", Mock(return_value=bad))
    monkeypatch.setenv("PGPORT", "5432")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        db.get_connection()
    bad.close.assert_called_once()
    bad.cursor.assert_not_called()


def test_explicit_dsn_has_precedence_but_cannot_evade_guard(monkeypatch):
    connect = Mock(return_value=connection(port=5432, major=16))
    monkeypatch.setattr(db.psycopg, "connect", connect)
    monkeypatch.setenv("SHOPIFY_DB_DSN", "dbname=shopifystores port=5433")
    with pytest.raises(RuntimeError):
        db.get_connection("dbname=shopifystores port=5432")
    connect.assert_called_once_with("dbname=shopifystores port=5432")


def test_expected_role_and_isolated_rehearsal(monkeypatch):
    monkeypatch.setenv("PADJECTIVE_EXPECTED_USER", "padjective")
    with pytest.raises(RuntimeError):
        db.verify_connection(connection(user="postgres"))
    monkeypatch.setenv("PADJECTIVE_EXPECTED_DATABASE", "shopify_derived_verify_example")
    db.verify_connection(connection(database="shopify_derived_verify_example"))
