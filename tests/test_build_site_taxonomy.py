from pathlib import Path
from typing import Any

import pandas as pd

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

    class _FakeDataset:
        def __init__(self) -> None:
            self.product_count = 3
            self.feature_names = ["ALPHA", "BETA"]
            self.taxonomy_count = 2
            self.metadata = pd.DataFrame(
                [
                    {
                        "product_id": 1,
                        "title": "Sample",
                        "taxonomy_name": "Example",
                        "taxonomy_id": "123",
                        "taxonomy_path": "Example",
                        "tags": "ALPHA, BETA",
                        "tag_count": 2,
                        "valid_tag_count": 2,
                        "cv_fold": 0,
                    }
                ]
            )
            self.discarded_products = []
            self.discarded_tags = []

    monkeypatch.setattr(
        "padjective.build_site.data_access.build_feature_dataset",
        lambda *_, **__: _FakeDataset(),
    )

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
    assert "Explore the full dataset" in index_html

    metadata_json = (output_dir / "metadata.json").read_text(encoding="utf-8")
    assert "taxonomy_classifier" in metadata
    assert metadata["taxonomy_classifier"]["model_id"] == 42
    assert "Home / Decor" in metadata_json
    assert "dataset" in metadata


def test_build_site_links_to_umllr_pages(tmp_path: Path, monkeypatch) -> None:
    fake_conn = _FakeConnection([])

    taxonomy_summary = {
        "model_id": 99,
        "trained_at": "2024-03-01T12:00:00",
        "stats": {
            "samples": 100,
            "taxonomies": 10,
            "unique_tags": 50,
            "training_accuracy": 0.9,
        },
        "class_distribution": [],
        "top_tags": [],
    }

    umllr_summary = {
        "metrics": [
            {
                "cv_fold": 1,
                "loss": 0.2,
                "mean_loss": 0.2,
                "prime_base": 3,
                "max_digit": 7,
            }
        ],
        "coefficients": {1: []},
        "predictions": {1: []},
    }

    monkeypatch.setattr(
        "padjective.build_site._collect_taxonomy_classifier_summary",
        lambda _conn, schema: taxonomy_summary,
    )
    monkeypatch.setattr(
        "padjective.build_site._collect_database_stats",
        lambda _conn, _schema: {"products": 0, "unique_tags": 0},
    )
    monkeypatch.setattr(
        "padjective.build_site._load_umllr_results",
        lambda _conn, _schema: umllr_summary,
    )

    output_dir = tmp_path / "site"
    build_site(
        output_dir,
        precomputed_database=fake_conn,
        battle_schema="padjective",
    )

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "href=\"umllr/index.html\"" in index_html


def test_build_site_surfaces_ablation_and_levelwise_results(tmp_path: Path, monkeypatch) -> None:
    fake_conn = _FakeConnection([])

    class _FakeDataset:
        def __init__(self) -> None:
            self.product_count = 2
            self.feature_names = ["ALPHA", "BETA"]
            self.taxonomy_count = 2
            self.metadata = pd.DataFrame(
                [
                    {
                        "product_id": 1,
                        "title": "Sample",
                        "taxonomy_name": "Example",
                        "taxonomy_id": "123",
                        "taxonomy_path": "1.1",
                        "tags": "ALPHA, BETA",
                        "tag_count": 2,
                        "valid_tag_count": 2,
                        "cv_fold": 0,
                    }
                ]
            )
            self.discarded_products = []
            self.discarded_tags = []

    monkeypatch.setattr(
        "padjective.build_site.data_access.build_feature_dataset",
        lambda *_, **__: _FakeDataset(),
    )
    monkeypatch.setattr(
        "padjective.build_site._collect_taxonomy_classifier_summary",
        lambda _conn, schema: None,
    )
    monkeypatch.setattr(
        "padjective.build_site._collect_database_stats",
        lambda _conn, _schema: {"products": 2, "unique_tags": 2},
    )
    monkeypatch.setattr(
        "padjective.build_site._load_umllr_results",
        lambda _conn, _schema: {
            "metrics": [
                {
                    "cv_fold": 1,
                    "loss": 0.2,
                    "mean_loss": 0.2,
                    "prime_base": 71,
                    "max_digit": 10,
                    "accuracy": 0.4,
                    "f1": 0.35,
                    "prefix2_accuracy": 0.7,
                    "mean_scoring_ops": 5.0,
                    "num_nonzero_coefficients": 42,
                }
            ],
            "coefficients": {1: []},
            "predictions": {1: []},
        },
    )
    monkeypatch.setattr(
        "padjective.build_site._load_umllr_order_ablation_results",
        lambda _conn, _schema: {
            "runs": [
                {
                    "run_key": "battle_elo",
                    "tag_order_strategy": "battle_elo",
                    "tag_order_seed": None,
                    "mean_loss": 0.2,
                    "mean_prefix2_accuracy": 0.7,
                    "mean_scoring_ops": 5.0,
                    "folds": [],
                },
                {
                    "run_key": "random_seed_7",
                    "tag_order_strategy": "random",
                    "tag_order_seed": 7,
                    "mean_loss": 0.25,
                    "mean_prefix2_accuracy": 0.6,
                    "mean_scoring_ops": 5.0,
                    "folds": [],
                },
            ],
            "random_summary": {
                "mean_loss": 0.255,
                "loss_std": 0.01,
            },
        },
    )
    monkeypatch.setattr(
        "padjective.build_site._load_taxonomy_levelwise_fold_results",
        lambda _conn, schema="padjective": [
            {
                "cv_fold": 1,
                "test_accuracy": 0.5,
                "test_f1": 0.45,
                "test_hierarchical_loss": 0.25,
                "padic_loss_total": 0.5,
                "padic_loss_mean": 0.25,
                "prime_base": 71,
                "num_train_samples": 10,
                "num_test_samples": 2,
                "num_nodes": 3,
                "num_classifiers": 2,
                "num_nonzero_params": 17,
                "exact_accuracy": 0.5,
                "prefix1_accuracy": 1.0,
                "prefix2_accuracy": 0.5,
                "mean_shared_prefix_depth": 1.5,
                "mean_scoring_ops": 3.0,
            }
        ],
    )

    output_dir = tmp_path / "site"
    metadata = build_site(
        output_dir,
        precomputed_database=fake_conn,
        battle_schema="padjective",
    )

    umllr_html = (output_dir / "umllr" / "index.html").read_text(encoding="utf-8")
    assert "Tag-order ablations" in umllr_html
    assert "battle_elo" in umllr_html
    assert "random" in umllr_html

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Level-wise Logistic Regression" in index_html
    assert "href=\"levelwise_logistic_regression/index.html\"" in index_html

    assert metadata["taxonomy_levelwise"]["overview_page"] == "levelwise_logistic_regression/index.html"
