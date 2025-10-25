from pathlib import Path
from typing import Any

from padjective.build_site import build_site


class _FakeCursor:
    def __init__(self, rows: list[tuple[str, str]]):
        self._rows = rows

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, _query: str, _params: Any = None) -> "_FakeCursor":
        return self

    def fetchall(self) -> list[tuple[str, str]]:
        return list(self._rows)


class _FakeConnection:
    def __init__(self, rows: list[tuple[str, str]]):
        self._rows = rows

    def cursor(self, **_kwargs: Any) -> _FakeCursor:
        return _FakeCursor(self._rows)


def test_build_site_includes_taxonomy_summary(tmp_path: Path, monkeypatch) -> None:
    pairs = [("TAG1", "TAG2"), ("TAG1", "TAG3"), ("TAG3", "TAG2")]
    fake_conn = _FakeConnection(pairs)

    taxonomy_summary = {
        "model_id": 42,
        "trained_at": "2024-03-01T12:00:00",
        "stats": {
            "samples": 1234,
            "taxonomies": 56,
            "unique_tags": 789,
            "training_accuracy": 0.88,
            "training_f1": 0.76,
            "cross_validation": {
                "folds": 5,
                "mean_accuracy": 0.81,
                "std_accuracy": 0.02,
                "mean_f1": 0.73,
            },
        },
        "class_distribution": [
            {
                "taxonomy_id": "123",
                "taxonomy_path": "Home / Decor",
                "sample_count": 120,
                "sample_fraction": 0.12,
            },
            {
                "taxonomy_id": "456",
                "taxonomy_path": "Apparel / Tops",
                "sample_count": 80,
                "sample_fraction": 0.08,
            },
        ],
        "top_tags": [
            {
                "tag": "BLUE",
                "top_taxonomy_id": "456",
                "top_taxonomy_path": "Apparel / Tops",
                "top_weight": 0.34,
                "max_abs_weight": 0.48,
            }
        ],
    }

    monkeypatch.setattr(
        "padjective.build_site._collect_taxonomy_classifier_summary",
        lambda _conn, schema: taxonomy_summary,
    )

    monkeypatch.setattr(
        "padjective.build_site._collect_database_stats",
        lambda _conn, _schema: {"products": 3, "unique_tags": 3},
    )

    output_dir = tmp_path / "site"
    metadata = build_site(
        output_dir,
        precomputed_database=fake_conn,
        battle_schema="padjective",
    )

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Shopify taxonomy classification" in index_html
    assert "Apparel / Tops" in index_html
    assert "BLUE" in index_html

    metadata_json = (output_dir / "metadata.json").read_text(encoding="utf-8")
    assert "taxonomy_classifier" in metadata
    assert metadata["taxonomy_classifier"]["model_id"] == 42
    assert "Home / Decor" in metadata_json
