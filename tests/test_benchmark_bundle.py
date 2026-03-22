import json
from pathlib import Path

import pandas as pd

from padjective.benchmark_bundle import (
    load_bundle,
    render_ablation_html,
    render_benchmark_numbers_tex,
    render_model_comparison_html,
    write_bundle_outputs,
    write_paper_tex_outputs,
)
from padjective.benchmark_runtime import (
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


def test_snapshot_bundle_writes_json_csv_html_and_tex(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "paper"
    _write_snapshot_fixture(snapshot_dir)

    tables = load_snapshot_tables(snapshot_dir, snapshot_label="paper")
    bundle = build_snapshot_benchmark_bundle(tables)

    assert bundle["snapshot"]["label"] == "paper"
    assert bundle["snapshot"]["product_count_filtered"] == 6
    assert bundle["ablation"]["baseline_strategy"] == "battle_elo"
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
    assert "benchmark-table" in model_html
    assert "battle_elo" in ablation_html
    assert "taxonomy_association" in ablation_html

    tex_numbers = render_benchmark_numbers_tex(bundle)
    assert "\\PadBenchBestAblationStrategy" in tex_numbers
    assert "\\PadBenchFilteredProducts" in tex_numbers
    assert (tex_dir / "model_comparison_table.tex").is_file()
    assert (tex_dir / "umllr_ablation_table.tex").is_file()

