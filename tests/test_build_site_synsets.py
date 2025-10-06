from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from padjective.build_site import build_site


def _prepare_synset_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE product_synsets (
                product_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                tags TEXT,
                synset_id TEXT,
                synset_name TEXT,
                synset_definition TEXT,
                confidence REAL,
                not_found INTEGER NOT NULL,
                reason TEXT,
                model TEXT NOT NULL,
                raw_response TEXT NOT NULL,
                processed_at TEXT
            )
            """
        )
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        rows = [
            (
                1,
                "Widget",
                "tag1",
                "n07737745",
                "banana",
                "elongated crescent-shaped yellow fruit",
                0.9,
                0,
                "Classic banana gadget",
                "gpt-5-mini",
                "{}",
                (base_time).isoformat(sep=" "),
            ),
            (
                2,
                "Gadget",
                "tag2",
                None,
                None,
                None,
                None,
                1,
                "No clear WordNet match",
                "gpt-5-mini",
                "{}",
                (base_time + timedelta(minutes=10)).isoformat(sep=" "),
            ),
            (
                3,
                "Thingamajig",
                "tag3",
                "n03595614",
                "jersey",
                "a knit fabric",
                0.7,
                0,
                "Apparel related",
                "gpt-5-mini",
                "{}",
                (base_time + timedelta(minutes=20)).isoformat(sep=" "),
            ),
        ]
        conn.executemany(
            """
            INSERT INTO product_synsets (
                product_id, title, tags, synset_id, synset_name, synset_definition,
                confidence, not_found, reason, model, raw_response, processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, _query):
        return self

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)


def test_build_site_includes_synset_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")

    csv_path = tmp_path / "products.csv"
    csv_path.write_text(
        "title,tags\nWidget,tag1\nGadget,tag2\nThingamajig,tag3\n",
        encoding="utf-8",
    )

    synset_db = tmp_path / "synsets.sqlite"
    _prepare_synset_db(synset_db)

    battle_pairs = [("TAG1", "TAG2"), ("TAG1", "TAG3"), ("TAG3", "TAG2")]
    battles_conn = _FakeConnection(battle_pairs)

    output_dir = tmp_path / "site"
    metadata = build_site(
        csv_path,
        output_dir,
        precomputed_database=battles_conn,
        synset_db=synset_db,
    )

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "WordNet synset tagging" in index_html
    assert "Products classified" in index_html
    assert "banana" in index_html
    assert "Recent products without a synset" in index_html
    assert "Biggest losers" in index_html

    synset_page = (output_dir / "synsets" / "n07737745.html").read_text(
        encoding="utf-8"
    )
    assert "Widget" in synset_page
    assert "Classic banana gadget" in synset_page

    assert metadata["synsets"]["processed"] == 3
    assert metadata["synsets"]["not_found"] == 1
    assert any(
        entry["synset_id"] == "n07737745" for entry in metadata["synsets"]["synsets"]
    )
