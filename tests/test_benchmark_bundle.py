import json
from pathlib import Path

import pandas as pd

from padjective.benchmark_bundle import (
    load_bundle,
    render_ablation_html,
    render_ablation_strategy_guide_html,
    render_benchmark_numbers_tex,
    render_model_comparison_html,
    write_bundle_outputs,
    write_paper_tex_outputs,
)
from padjective.build_site import build_site
from padjective.benchmark_runtime import (
    BattleRecord,
    ProductRecord,
    _initialize_coefficients_umllr_style,
    build_snapshot_benchmark_bundle,
    load_snapshot_tables,
)


def _write_snapshot_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = {
        "snapshot_id": "00000000-0000-0000-0000-000000000001",
        "snapshot_name": "paper-2026-02-11T1915Z",
        "created_at": "2026-02-11T19:15:00+00:00",
        "as_of": "2026-02-11T19:15:00+00:00",
        "product_table": "cantbuymelove.product",
        "min_tag_count": 1,
        "min_samples_per_taxonomy": 1,
        "product_count": 6,
        "tag_count": 3,
        "taxonomy_count": 2,
        "note": "fixture",
        "code_version": "deadbeef",
    }
    (root / "snapshot.json").write_text(json.dumps(snapshot) + "\n", encoding="utf-8")

    tags = [
        {"tag_id": "tag000001", "tag_rank": 1},
        {"tag_id": "tag000002", "tag_rank": 2},
        {"tag_id": "tag000003", "tag_rank": 3},
    ]
    with (root / "tags.jsonl").open("w", encoding="utf-8") as handle:
        for row in tags:
            handle.write(json.dumps(row) + "\n")
    products = [
        {
            "product_id_hash": "prod-a0",
            "taxonomy_id": "tax-a",
            "taxonomy_path": "1.1",
            "taxonomy_name": "A",
            "cv_fold": 0,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 0},
                {"tag_id": "tag000002", "in_title": True, "title_part": 0, "title_position": 5},
            ],
        },
        {
            "product_id_hash": "prod-a1",
            "taxonomy_id": "tax-a",
            "taxonomy_path": "1.1",
            "taxonomy_name": "A",
            "cv_fold": 1,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 0},
                {"tag_id": "tag000002", "in_title": True, "title_part": 0, "title_position": 5},
            ],
        },
        {
            "product_id_hash": "prod-a2",
            "taxonomy_id": "tax-a",
            "taxonomy_path": "1.1",
            "taxonomy_name": "A",
            "cv_fold": 0,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 1},
                {"tag_id": "tag000002", "in_title": True, "title_part": 0, "title_position": 6},
            ],
        },
        {
            "product_id_hash": "prod-b0",
            "taxonomy_id": "tax-b",
            "taxonomy_path": "2.1",
            "taxonomy_name": "B",
            "cv_fold": 1,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 0},
                {"tag_id": "tag000003", "in_title": True, "title_part": 0, "title_position": 5},
            ],
        },
        {
            "product_id_hash": "prod-b1",
            "taxonomy_id": "tax-b",
            "taxonomy_path": "2.1",
            "taxonomy_name": "B",
            "cv_fold": 0,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 0},
                {"tag_id": "tag000003", "in_title": True, "title_part": 0, "title_position": 5},
            ],
        },
        {
            "product_id_hash": "prod-b2",
            "taxonomy_id": "tax-b",
            "taxonomy_path": "2.1",
            "taxonomy_name": "B",
            "cv_fold": 1,
            "tag_count": 2,
            "title_part_count": 1,
            "tag_features": [
                {"tag_id": "tag000001", "in_title": True, "title_part": 0, "title_position": 1},
                {"tag_id": "tag000003", "in_title": True, "title_part": 0, "title_position": 6},
            ],
        },
    ]
    with (root / "products-00000.jsonl").open("w", encoding="utf-8") as handle:
        for row in products:
            handle.write(json.dumps(row) + "\n")


def test_zubarev_initializer_uses_the_primary_greedy_order(monkeypatch) -> None:
    records = [
        ProductRecord(
            product_id=1,
            product_key="one",
            tags=["less-specific", "more-specific"],
            encoded_path=8,
            cv_fold=0,
            taxonomy_id="tax-a",
            taxonomy_depth=2,
            title_tag_positions=(),
        ),
        ProductRecord(
            product_id=2,
            product_key="two",
            tags=["less-specific"],
            encoded_path=4,
            cv_fold=0,
            taxonomy_id="tax-b",
            taxonomy_depth=2,
            title_tag_positions=(),
        ),
    ]
    observed: dict[str, str] = {}

    def fake_order(training, battles, holdout_fold, *, strategy, seed=None):
        observed["strategy"] = strategy
        return ["more-specific", "less-specific"]

    monkeypatch.setattr("padjective.benchmark_runtime._tag_order", fake_order)

    coefficients = _initialize_coefficients_umllr_style(
        records,
        [BattleRecord("more-specific", "less-specific", 0)],
        1,
        3,
    )

    assert observed["strategy"] == "taxonomy_association"
    assert list(coefficients) == ["more-specific", "less-specific"]


def test_snapshot_bundle_writes_json_csv_html_and_tex(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "paper"
    _write_snapshot_fixture(snapshot_dir)

    tables = load_snapshot_tables(snapshot_dir, snapshot_label="paper")
    bundle = build_snapshot_benchmark_bundle(tables)

    assert bundle["snapshot"]["label"] == "paper"
    assert bundle["snapshot"]["product_count_filtered"] == 6
    assert bundle["ablation"]["baseline_strategy"] == "battle_elo"
    umllr_row = next(row for row in bundle["models"]["rows"] if row["model_key"] == "umllr")
    taxonomy_association_row = next(
        row
        for row in bundle["ablation"]["strategy_rows"]
        if row["tag_order_strategy"] == "taxonomy_association"
    )
    assert umllr_row["mean_padic_loss"] == taxonomy_association_row["mean_padic_loss"]
    assert umllr_row["mean_scoring_ops"] == taxonomy_association_row["mean_scoring_ops"]
    model_keys = {row["model_key"] for row in bundle["models"]["rows"]}
    assert {"dummy", "umllr", "levelwise", "zubarev"} <= model_keys

    out_dir = tmp_path / "reports" / "paper"
    tex_dir = tmp_path / "generated"
    write_bundle_outputs(bundle, out_dir)
    write_paper_tex_outputs(bundle, tex_dir)

    loaded = load_bundle(out_dir / "benchmark.json")
    assert loaded["snapshot"]["snapshot_name"] == "paper-2026-02-11T1915Z"

    model_csv = pd.read_csv(out_dir / "model_comparison.csv")
    assert "model_label" in model_csv.columns
    assert "mean_padic_loss" in model_csv.columns

    ablation_csv = pd.read_csv(out_dir / "umllr_ablation.csv")
    assert "tag_order_strategy" in ablation_csv.columns
    assert "loss_delta_vs_baseline" in ablation_csv.columns

    model_html = render_model_comparison_html(bundle)
    ablation_html = render_ablation_html(bundle)
    strategy_guide_html = render_ablation_strategy_guide_html(bundle)
    assert "benchmark-table" in model_html
    assert "Trained params" in model_html
    assert "Avg active params / classification" in model_html
    assert "battle_elo" in ablation_html
    assert "taxonomy_association" in ablation_html
    assert "Avg active params / classification" in ablation_html
    assert "Taxonomy-peaked tags first" in strategy_guide_html
    assert "single most common taxonomy" in strategy_guide_html

    tex_numbers = render_benchmark_numbers_tex(bundle)
    assert "\\PadBenchBestAblationStrategy" in tex_numbers
    assert "\\PadBenchFilteredProducts" in tex_numbers
    assert (tex_dir / "model_comparison_table.tex").is_file()
    assert (tex_dir / "umllr_ablation_table.tex").is_file()


def test_paper_bundle_html_and_tex_stay_in_parity(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "paper"
    _write_snapshot_fixture(snapshot_dir)

    bundle = build_snapshot_benchmark_bundle(
        load_snapshot_tables(snapshot_dir, snapshot_label="paper")
    )
    reports_root = tmp_path / "reports"
    write_bundle_outputs(bundle, reports_root / "paper")
    tex_dir = tmp_path / "generated"
    write_paper_tex_outputs(bundle, tex_dir)

    site_dir = tmp_path / "site"
    build_site(
        site_dir,
        benchmark_report_root=reports_root,
        benchmark_views=("paper",),
        benchmark_only=True,
    )

    benchmark_json = load_bundle(reports_root / "paper" / "benchmark.json")
    html_text = (site_dir / "benchmark" / "paper" / "index.html").read_text(encoding="utf-8")
    ablation_html_text = (site_dir / "benchmark" / "paper" / "ablation.html").read_text(encoding="utf-8")
    tex_text = (tex_dir / "benchmark_numbers.tex").read_text(encoding="utf-8")
    model_table_tex = (tex_dir / "model_comparison_table.tex").read_text(encoding="utf-8")
    ablation_table_tex = (tex_dir / "umllr_ablation_table.tex").read_text(encoding="utf-8")

    filtered_products = benchmark_json["snapshot"]["product_count_filtered"]
    best_strategy = benchmark_json["narrative"]["best_ablation_strategy"]
    umllr_row = next(
        row for row in benchmark_json["models"]["rows"] if row["model_key"] == "umllr"
    )
    best_ablation = benchmark_json["ablation"]["strategy_rows"][0]

    assert f"{filtered_products:,}" in html_text
    assert f"{umllr_row['mean_padic_loss']:.6f}" in html_text
    assert best_strategy in ablation_html_text
    assert f"{best_ablation['mean_padic_loss']:.6f}" in ablation_html_text

    assert f"\\providecommand{{\\PadBenchFilteredProducts}}{{{filtered_products:,}}}" in tex_text
    assert "Avg active params." in model_table_tex
    assert f"{umllr_row['mean_padic_loss']:.6f}" in model_table_tex
    assert f"{umllr_row['mean_scoring_ops']:.2f}" in model_table_tex
    assert "Unconstrained Neural Network with L2" in model_table_tex
    assert "Unconstrained Neural Network with L1" not in model_table_tex
    assert best_strategy.replace("_", "\\_") in ablation_table_tex
    assert f"{best_ablation['mean_padic_loss']:.6f}" in ablation_table_tex
    assert "Fold SD" in ablation_table_tex
    assert "\\%" in ablation_table_tex


def test_snapshot_bundle_tolerates_missing_title_overlap_fields(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "paper"
    _write_snapshot_fixture(snapshot_dir)

    products_path = snapshot_dir / "products-00000.jsonl"
    rows = [json.loads(line) for line in products_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["tag_features"][1]["in_title"] = False
    rows[0]["tag_features"][1]["title_part"] = None
    rows[0]["tag_features"][1]["title_position"] = None
    products_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    bundle = build_snapshot_benchmark_bundle(
        load_snapshot_tables(snapshot_dir, snapshot_label="paper")
    )

    assert bundle["snapshot"]["product_count_filtered"] == 6
    assert bundle["ablation"]["baseline_strategy"] == "battle_elo"
