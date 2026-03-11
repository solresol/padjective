from datetime import datetime, timezone
import uuid

from padjective.product_taxonomy_bench_export import SnapshotMetadata, render_hf_dataset_card


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
