import pytest

from padjective import db


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
