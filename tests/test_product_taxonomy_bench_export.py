import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

import padjective.product_taxonomy_bench_export as export_mod
import padjective.hf_sync as sync_mod
from padjective.product_taxonomy_bench_export import (
    DEFAULT_DATASET_CITATION_BIBTEX,
    SnapshotMetadata,
    export_snapshot,
    load_snapshot_actual_counts,
    render_hf_dataset_card,
    stage_hf_notebook,
)
from padjective.product_taxonomy_bench_notebook import render_notebook
from padjective.release_licensing import stage_release_licenses


def test_render_hf_dataset_card_includes_snapshots() -> None:
    paper = SnapshotMetadata(
        snapshot_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        snapshot_name="paper-2026-02-11",
        created_at=datetime(2026, 2, 11, 19, 15, tzinfo=timezone.utc),
        as_of=datetime(2026, 2, 11, 19, 15, tzinfo=timezone.utc),
        product_table="cantbuymelove.product",
        min_tag_count=5,
        min_samples_per_taxonomy=5,
        product_count=6933,
        tag_count=9845,
        taxonomy_count=407,
        note=None,
        code_version="deadbeef",
    )
    latest = SnapshotMetadata(
        snapshot_id=uuid.UUID("00000000-0000-0000-0000-000000000002"),
        snapshot_name="latest-2026-02-24",
        created_at=datetime(2026, 2, 24, 0, 0, tzinfo=timezone.utc),
        as_of=None,
        product_table="cantbuymelove.product",
        min_tag_count=5,
        min_samples_per_taxonomy=5,
        product_count=7000,
        tag_count=9900,
        taxonomy_count=410,
        note="rolling",
        code_version="cafebabe",
    )

    card = render_hf_dataset_card(
        dataset_id="product-taxonomy-bench",
        pretty_name="Product Taxonomy Bench (Anonymized)",
        paper=paper,
        latest=latest,
    )

    assert "paper-2026-02-11" in card
    assert "latest-2026-02-24" in card
    assert "product-taxonomy-bench" in card
    assert "Product Taxonomy Bench (Anonymized)" in card
    assert "notebooks/product_taxonomy_bench.ipynb" in card
    assert "Open in Colab" in card
    assert "defaults to the fixed `paper` snapshot" in card
    assert "configs:" in card
    assert "- config_name: paper" in card
    assert "default: true" in card
    assert '"paper/products-*.jsonl.gz"' in card
    assert "- config_name: latest" in card
    assert '"latest/products-*.jsonl.gz"' in card
    assert DEFAULT_DATASET_CITATION_BIBTEX in card
    assert "SHIMP (Shopify Hierarchical Inference for Merchant Products)" in card
    assert "title={{SHIMP: Shopify Hierarchical Inference for Merchant Products}}" in card
    assert "doi={10.57967/hf/8413}" in card
    assert "license: cc-by-4.0" in card
    assert "[LICENSE-CODE](./LICENSE-CODE)" in card
    assert "license: other" not in card
    assert "TODO" not in card


def test_stage_hf_notebook_generates_ablation_notebook(tmp_path) -> None:
    notebook_path = stage_hf_notebook(tmp_path)
    notebook_text = notebook_path.read_text(encoding="utf-8")
    assert "UMLLR tag-order ablation" in notebook_text
    assert "Active parameters vs p-adic loss" in notebook_text
    assert "Fitted equation:" in notebook_text
    assert "mean_prefix2_accuracy" in notebook_text
    assert notebook_path.name == "product_taxonomy_bench.ipynb"


def test_release_licences_reach_the_hub_uploader(tmp_path, monkeypatch) -> None:
    notices = stage_release_licenses(tmp_path)
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-4.0\n---\n")
    for name in ("paper", "latest"):
        (tmp_path / name).mkdir()
    uploaded = {}
    folders = []

    class FakeApi:
        def create_repo(self, **kwargs):
            pass

        def upload_file(self, *, path_in_repo, path_or_fileobj, **kwargs):
            uploaded[path_in_repo] = Path(path_or_fileobj).read_bytes()

        def upload_folder(self, *, path_in_repo, **kwargs):
            folders.append(path_in_repo)

    monkeypatch.setattr(sync_mod, "_require_huggingface_hub", lambda: FakeApi)
    sync_mod.upload_export_root(
        tmp_path, sync_mod.UploadOptions(repo_id="test/shimp", replace_folders=())
    )

    assert set(uploaded) == {"README.md", "LICENSE", "LICENSE-DATA", "LICENSE-CODE", "THIRD_PARTY_NOTICES.md"}
    assert all(uploaded[path.name] == path.read_bytes() for path in notices)
    assert b"Attribution 4.0 International" in uploaded["LICENSE-DATA"]
    assert b"Copyright (c) 2026 Gregory D. Baker" in uploaded["LICENSE-CODE"]
    assert b"Copyright (c) Shopify" in uploaded["THIRD_PARTY_NOTICES.md"]
    assert set(folders) == {"paper", "latest"}


def test_render_notebook_includes_embedded_runtime_and_tables() -> None:
    notebook = render_notebook()
    combined = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
    )
    assert "build_snapshot_benchmark_bundle" in combined
    assert "load_snapshot_tables_from_hf" in combined
    assert "ablation_table" in combined
    assert "ACTIVE_PARAMS_EXCLUDED = {\"pclr\", \"pcnn\", \"zubarev\"}" in combined
    assert "log10(mean p-adic loss)" in combined


class _CountsCursor:
    def __init__(self, row: dict[str, int]) -> None:
        self._row = row

    def __enter__(self) -> "_CountsCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, _query, _params=None) -> None:
        return None

    def fetchone(self) -> dict[str, int]:
        return self._row


class _CountsConnection:
    def __init__(self, row: dict[str, int]) -> None:
        self._row = row

    def cursor(self, *_, **__) -> _CountsCursor:
        return _CountsCursor(self._row)


def test_load_snapshot_actual_counts_reads_persisted_tables() -> None:
    conn = _CountsConnection(
        {
            "product_count": 6693,
            "tag_count": 2542,
            "taxonomy_count": 363,
        }
    )

    counts = load_snapshot_actual_counts(
        conn,  # type: ignore[arg-type]
        "padjective",
        uuid.UUID("00000000-0000-0000-0000-000000000001"),
    )

    assert counts == (6693, 2542, 363)


def test_export_snapshot_writes_actual_counts_to_snapshot_json(
    tmp_path, monkeypatch
) -> None:
    stale_meta = SnapshotMetadata(
        snapshot_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        snapshot_name="paper-2026-02-11T1915Z",
        created_at=datetime(2026, 2, 26, 6, 35, 34, tzinfo=timezone.utc),
        as_of=datetime(2026, 2, 11, 19, 15, tzinfo=timezone.utc),
        product_table="cantbuymelove.product",
        min_tag_count=5,
        min_samples_per_taxonomy=5,
        product_count=6718,
        tag_count=2542,
        taxonomy_count=363,
        note="Paper cutoff as_of=2026-02-11T19:15:00+00:00",
        code_version="deadbeef",
    )

    monkeypatch.setattr(
        export_mod, "load_snapshot_metadata", lambda *_args, **_kwargs: stale_meta
    )
    monkeypatch.setattr(
        export_mod,
        "load_snapshot_actual_counts",
        lambda *_args, **_kwargs: (6693, 2542, 363),
    )
    monkeypatch.setattr(export_mod, "export_tags", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        export_mod, "export_products_jsonl", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        export_mod, "export_products_parquet", lambda *_args, **_kwargs: []
    )

    metadata = export_snapshot(
        object(),  # type: ignore[arg-type]
        schema="padjective",
        snapshot_ref="paper",
        snapshot_id=stale_meta.snapshot_id,
        out_dir=tmp_path,
        formats=("jsonl",),
        gzip_jsonl=True,
        rows_per_shard=1000,
    )

    snapshot_payload = json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))

    assert metadata.product_count == 6693
    assert metadata.tag_count == 2542
    assert metadata.taxonomy_count == 363
    assert snapshot_payload["product_count"] == 6693
    assert snapshot_payload["tag_count"] == 2542
    assert snapshot_payload["taxonomy_count"] == 363
