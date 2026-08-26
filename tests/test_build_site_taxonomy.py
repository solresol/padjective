import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from padjective.build_site import (
    _build_active_params_regression_frame,
    _fit_active_params_regression,
    build_site,
)


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


def _write_benchmark_bundle_fixture(root: Path, *, view: str) -> None:
    view_dir = root / view
    view_dir.mkdir(parents=True, exist_ok=True)
    mean_loss = 0.2946 if view == "paper" else 0.3012
    bundle = {
        "bundle_version": 1,
        "snapshot": {
            "label": view,
            "snapshot_name": f"{view}-2026-02-11T1915Z",
            "as_of": "2026-02-11T19:15:00+00:00" if view == "paper" else None,
            "product_count_filtered": 6527,
            "tag_count_filtered": 2971,
            "taxonomy_count_filtered": 308,
            "prime_base": 71,
        },
        "models": {
            "rows": [
                {
                    "model_key": "dummy",
                    "model_label": "Dummy Baseline",
                    "short_label": "Dummy",
                    "color": "#94a3b8",
                    "marker": "X",
                    "params": 1.0,
                    "mean_padic_loss": 0.5515,
                    "mean_exact_accuracy": 0.10,
                    "mean_prefix2_accuracy": 0.25,
                    "mean_scoring_ops": 1.0,
                    "log10_params": 0.0,
                    "parsimony_score": -0.15,
                },
                {
                    "model_key": "umllr",
                    "model_label": "Importance-Optimised p-adic Linear Regression",
                    "short_label": "Importance-Optimised",
                    "color": "#0b6ce3",
                    "marker": "o",
                    "params": 852.0,
                    "mean_padic_loss": mean_loss,
                    "mean_exact_accuracy": 0.44,
                    "mean_prefix2_accuracy": 0.61,
                    "mean_scoring_ops": 1.42,
                    "log10_params": 2.9304395947667004,
                    "parsimony_score": 0.12,
                },
            ]
        },
        "ablation": {
            "baseline_mean_padic_loss": mean_loss,
            "random_summary": {"mean_loss": mean_loss + 0.01},
            "runs": [
                {
                    "tag_order_strategy": "battle_elo",
                    "tag_order_seed": None,
                    "run_key": "battle_elo",
                },
                {
                    "tag_order_strategy": "taxonomy_association",
                    "tag_order_seed": None,
                    "run_key": "taxonomy_association",
                },
            ],
            "strategy_rows": [
                {
                    "tag_order_strategy": "taxonomy_association",
                    "run_key": "taxonomy_association",
                    "mean_padic_loss": mean_loss - 0.02,
                    "loss_delta_vs_baseline": -0.02,
                    "wins_vs_baseline": 5,
                    "comparisons_vs_baseline": 5,
                    "mean_exact_accuracy": 0.52,
                    "mean_prefix2_accuracy": 0.67,
                    "mean_scoring_ops": 1.35,
                },
                {
                    "tag_order_strategy": "battle_elo",
                    "run_key": "battle_elo",
                    "mean_padic_loss": mean_loss,
                    "loss_delta_vs_baseline": 0.0,
                    "wins_vs_baseline": 0,
                    "comparisons_vs_baseline": 5,
                    "mean_exact_accuracy": 0.44,
                    "mean_prefix2_accuracy": 0.61,
                    "mean_scoring_ops": 1.42,
                },
            ],
        },
        "narrative": {
            "best_ablation_strategy": "taxonomy_association",
            "best_ablation_mean_padic_loss": mean_loss - 0.02,
            "best_ablation_delta_vs_baseline": -0.02,
        },
    }
    (view_dir / "benchmark.json").write_text(json.dumps(bundle) + "\n", encoding="utf-8")


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
        lambda _conn, _schema, snapshot_ref=None: {
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
            "snapshot_ref": snapshot_ref,
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
        ablation_snapshot_ref="paper",
    )

    umllr_html = (output_dir / "umllr" / "index.html").read_text(encoding="utf-8")
    assert "Tag-order ablations" in umllr_html
    assert "battle_elo" in umllr_html
    assert "random" in umllr_html
    assert "fixed <code>paper</code> snapshot" in umllr_html

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Level-wise Logistic Regression" in index_html
    assert "href=\"levelwise_logistic_regression/index.html\"" in index_html

    assert metadata["taxonomy_levelwise"]["overview_page"] == "levelwise_logistic_regression/index.html"


def test_build_site_renders_benchmark_pages(tmp_path: Path, monkeypatch) -> None:
    fake_conn = _FakeConnection([])
    benchmark_root = tmp_path / "reports"
    _write_benchmark_bundle_fixture(benchmark_root, view="latest")
    _write_benchmark_bundle_fixture(benchmark_root, view="paper")

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

    def _fake_generate_outputs(_leaderboard, rankings_html: Path, chart_path: Path, rows: int = 20) -> None:
        rankings_html.write_text("<table></table>", encoding="utf-8")
        chart_path.write_bytes(b"fake")

    monkeypatch.setattr(
        "padjective.build_site.data_access.build_feature_dataset",
        lambda *_, **__: _FakeDataset(),
    )
    monkeypatch.setattr(
        "padjective.build_site.ranking.load_pairs",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "padjective.build_site.ranking.compute_rankings",
        lambda _pairs: pd.DataFrame(columns=["tag", "score", "component"]),
    )
    monkeypatch.setattr(
        "padjective.build_site.display.generate_outputs",
        _fake_generate_outputs,
    )
    monkeypatch.setattr(
        "padjective.build_site._create_comprehensive_dumps",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "padjective.build_site._collect_database_stats",
        lambda _conn, _schema: {"products": 2, "unique_tags": 2},
    )
    monkeypatch.setattr(
        "padjective.build_site._collect_taxonomy_classifier_summary",
        lambda _conn, schema="padjective": None,
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

    output_dir = tmp_path / "site"
    metadata = build_site(
        output_dir,
        precomputed_database=fake_conn,
        battle_schema="padjective",
        benchmark_report_root=benchmark_root,
        benchmark_views=("latest", "paper"),
    )

    assert (output_dir / "benchmark" / "index.html").is_file()
    assert (output_dir / "benchmark" / "latest" / "index.html").is_file()
    assert (output_dir / "benchmark" / "latest" / "ablation.html").is_file()
    assert (output_dir / "benchmark" / "paper" / "index.html").is_file()
    assert (output_dir / "benchmark" / "paper" / "ablation.html").is_file()

    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert "Open benchmark pages" in index_html
    assert "href=\"benchmark/index.html\"" in index_html
    assert "Latest active params / classification" in index_html
    assert "Paper active params / classification" in index_html

    umllr_html = (output_dir / "umllr" / "index.html").read_text(encoding="utf-8")
    assert "paper comparison page" in umllr_html
    assert "../benchmark/paper/ablation.html" in umllr_html

    paper_html = (output_dir / "benchmark" / "paper" / "ablation.html").read_text(encoding="utf-8")
    assert "Fixed benchmark snapshot shared by the Hugging Face notebook" in paper_html
    assert "taxonomy_association" in paper_html
    assert "Ordering methods" in paper_html
    assert "single most common taxonomy" in paper_html
    assert "Avg active params / classification" in paper_html

    assert metadata["benchmark"]["views"]["paper"]["ablation_page"] == "benchmark/paper/ablation.html"
    assert metadata["benchmark"]["views"]["paper"]["umllr_mean_scoring_ops"] == 1.42


def test_build_site_benchmark_only_renders_paper_view(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "reports"
    _write_benchmark_bundle_fixture(benchmark_root, view="paper")

    output_dir = tmp_path / "paper-site"
    metadata = build_site(
        output_dir,
        benchmark_report_root=benchmark_root,
        benchmark_views=("paper",),
        benchmark_only=True,
    )

    assert (output_dir / "index.html").is_file()
    assert (output_dir / "benchmark" / "paper" / "index.html").is_file()
    assert (output_dir / "benchmark" / "paper" / "ablation.html").is_file()

    paper_index = (output_dir / "benchmark" / "paper" / "index.html").read_text(encoding="utf-8")
    assert "Shared benchmark bundle for the site, notebook, and paper." in paper_index
    assert "Small-multiples dashboard generated directly from the comparison table rows below." in paper_index
    assert "Log-log scatter of trained parameters versus mean p-adic loss" in paper_index
    assert "Scatter plot of avg active params versus mean p-adic loss, excluding PCLR and PCNN." in paper_index
    assert "Scatter plot of avg active params versus log10(mean p-adic loss), excluding PCLR and PCNN." in paper_index
    assert "Fitted equation: mean p-adic loss =" in paper_index
    assert "Fitted equation: log10(mean p-adic loss) =" in paper_index
    assert "Regression fitted on log10(active params): R²=" in paper_index
    assert "p=" in paper_index
    assert "Trained params" in paper_index
    assert "Avg active params / classification" in paper_index

    paper_ablation = (output_dir / "benchmark" / "paper" / "ablation.html").read_text(encoding="utf-8")
    assert "taxonomy_association" in paper_ablation
    assert "Taxonomy-peaked tags first" in paper_ablation
    assert "Bar chart generated from the same bundle rows consumed by the notebook." in paper_ablation
    assert metadata["benchmark"]["views"]["paper"]["summary_page"] == "benchmark/paper/index.html"


def test_active_params_regression_excludes_parameter_constrained_models() -> None:
    bundle = {
        "models": {
            "rows": [
                {
                    "model_key": "dummy",
                    "model_label": "Dummy Baseline",
                    "short_label": "Dummy",
                    "mean_scoring_ops": 1.0,
                    "mean_padic_loss": 0.60,
                },
                {
                    "model_key": "umllr",
                    "model_label": "Importance-Optimised p-adic Linear Regression",
                    "short_label": "Importance-Optimised",
                    "mean_scoring_ops": 1.2,
                    "mean_padic_loss": 0.30,
                },
                {
                    "model_key": "pclr",
                    "model_label": "Parameter-constrained Logistic Regression",
                    "short_label": "PCLR",
                    "mean_scoring_ops": 600.0,
                    "mean_padic_loss": 0.70,
                },
                {
                    "model_key": "ulr",
                    "model_label": "Unconstrained Logistic Regression with L1",
                    "short_label": "ULR",
                    "mean_scoring_ops": 300.0,
                    "mean_padic_loss": 0.08,
                },
                {
                    "model_key": "pcnn",
                    "model_label": "Parameter-constrained Neural Network",
                    "short_label": "PCNN",
                    "mean_scoring_ops": 9000.0,
                    "mean_padic_loss": 0.68,
                },
            ]
        }
    }

    frame = _build_active_params_regression_frame(bundle)

    assert set(frame["model_key"]) == {"dummy", "umllr", "ulr"}

    raw_regression = _fit_active_params_regression(frame, log_loss=False)
    log_regression = _fit_active_params_regression(frame, log_loss=True)

    assert raw_regression is not None
    assert log_regression is not None
    assert 0.0 <= raw_regression["r_squared"] <= 1.0
    assert 0.0 <= log_regression["r_squared"] <= 1.0


def test_active_params_regression_requires_distinct_support_values() -> None:
    frame = pd.DataFrame(
        {
            "mean_scoring_ops": [1.0, 1.0],
            "mean_padic_loss": [0.6, 0.3],
            "log10_mean_padic_loss": np.log10([0.6, 0.3]),
        }
    )

    assert _fit_active_params_regression(frame, log_loss=False) is None
    assert _fit_active_params_regression(frame, log_loss=True) is None
