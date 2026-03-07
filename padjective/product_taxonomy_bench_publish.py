"""Create and export the `paper` + `latest` product-taxonomy-bench datasets.

This is a convenience wrapper for publication:

- The `paper` snapshot is created with an explicit point-in-time cutoff (as-of)
  parsed from the paper TeX source (or provided explicitly).
- The `latest` snapshot is created from the current database state.
- Both snapshots are exported to JSONL/Parquet plus a Hugging Face dataset card.
"""

from __future__ import annotations

import argparse
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from psycopg import sql

from . import db
from .hf_sync import UploadOptions, upload_export_root
from .product_taxonomy_bench import create_snapshot, parse_as_of
from .product_taxonomy_bench_export import (
    SnapshotMetadata,
    export_snapshot,
    render_hf_dataset_card,
)


_UTC_TIMESTAMP_RE = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})[ T](?P<time>\d{2}:\d{2})(?::(?P<second>\d{2}))?\s*UTC",
    re.IGNORECASE,
)


def parse_latest_utc_timestamp_from_tex(tex: str) -> datetime:
    """Infer the dataset cutoff timestamp from the paper TeX.

    We first look for an explicit "last update timestamp of ... UTC" phrase. If
    that isn't present we fall back to taking the max UTC timestamp anywhere in
    the TeX file.
    """

    explicit = re.search(
        r"last update timestamp of\\s*(\\d{4}-\\d{2}-\\d{2}\\s+\\d{2}:\\d{2}(?::\\d{2})?\\s*UTC)",
        tex,
        flags=re.IGNORECASE,
    )
    if explicit:
        return parse_as_of(explicit.group(1))

    matches = list(_UTC_TIMESTAMP_RE.finditer(tex))
    if not matches:
        raise ValueError("No 'YYYY-MM-DD HH:MM UTC' timestamps found in TeX source")

    parsed: list[datetime] = []
    for match in matches:
        second = match.group("second") or "00"
        parsed.append(
            datetime.fromisoformat(
                f"{match.group('date')}T{match.group('time')}:{second}+00:00"
            ).astimezone(timezone.utc)
        )

    return max(parsed)


def snapshot_name_from_timestamp(prefix: str, when: datetime) -> str:
    """Create a stable snapshot_name with UTC timestamp granularity (minute)."""

    when = when.astimezone(timezone.utc)
    return f"{prefix}-{when.strftime('%Y-%m-%dT%H%MZ')}"


def _get_snapshot_id_by_name(conn, schema: str, snapshot_name: str) -> Optional[uuid.UUID]:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT snapshot_id FROM {schema}.product_taxonomy_bench_snapshots WHERE snapshot_name = %s"
            ).format(schema=sql.Identifier(schema)),
            (snapshot_name,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return uuid.UUID(str(row[0]))


def _upsert_alias(conn, schema: str, alias: str, snapshot_id: uuid.UUID) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.product_taxonomy_bench_snapshot_aliases (alias, snapshot_id)
                VALUES (%s, %s)
                ON CONFLICT (alias) DO UPDATE SET
                    snapshot_id = EXCLUDED.snapshot_id,
                    updated_at = now()
                """
            ).format(schema=sql.Identifier(schema)),
            (alias, snapshot_id),
        )
    conn.commit()


@dataclass(frozen=True)
class PublishResult:
    paper: SnapshotMetadata
    latest: SnapshotMetadata
    first1000: SnapshotMetadata


def publish(
    *,
    dsn: str | None,
    schema: str,
    product_table: str,
    paper_as_of: datetime,
    min_tag_count: int,
    min_samples_per_taxonomy: int,
    out_root: Path,
    formats: Sequence[str],
    gzip_jsonl: bool,
    rows_per_shard: int,
    dataset_id: str,
    pretty_name: str,
    hf_repo_id: str | None = None,
    hf_token: str | None = None,
    hf_replace_folders: Sequence[str] = ("latest", "first1000"),
) -> PublishResult:
    paper_snapshot_name = snapshot_name_from_timestamp("paper", paper_as_of)
    latest_snapshot_name = snapshot_name_from_timestamp(
        "latest", datetime.now(timezone.utc)
    )
    first1000_snapshot_name = snapshot_name_from_timestamp(
        "first1000", datetime.now(timezone.utc)
    )

    try:
        paper_id = create_snapshot(
            dsn=dsn,
            schema=schema,
            snapshot_name=paper_snapshot_name,
            product_table=product_table,
            min_tag_count=min_tag_count,
            min_samples_per_taxonomy=min_samples_per_taxonomy,
            as_of=paper_as_of,
            alias="paper",
            note=f"Paper cutoff as_of={paper_as_of.isoformat()}",
        )
    except ValueError:
        conn = db.get_connection(dsn)
        try:
            paper_id = _get_snapshot_id_by_name(conn, schema, paper_snapshot_name)
            if paper_id is None:
                raise
            _upsert_alias(conn, schema, "paper", paper_id)
        finally:
            conn.close()

    latest_id = create_snapshot(
        dsn=dsn,
        schema=schema,
        snapshot_name=latest_snapshot_name,
        product_table=product_table,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
        as_of=None,
        alias="latest",
        note="Rolling latest snapshot",
    )

    first1000_id = create_snapshot(
        dsn=dsn,
        schema=schema,
        snapshot_name=first1000_snapshot_name,
        product_table=product_table,
        min_tag_count=1,
        min_samples_per_taxonomy=1,
        max_products=1000,
        as_of=None,
        alias="first1000",
        note="First 1000 products (ordered by product id) with taxonomy labels",
    )

    conn = db.get_connection(dsn)
    try:
        paper_meta = export_snapshot(
            conn,
            schema=schema,
            snapshot_ref="paper",
            snapshot_id=paper_id,
            out_dir=out_root / "paper",
            formats=formats,
            gzip_jsonl=gzip_jsonl,
            rows_per_shard=rows_per_shard,
        )
        latest_meta = export_snapshot(
            conn,
            schema=schema,
            snapshot_ref="latest",
            snapshot_id=latest_id,
            out_dir=out_root / "latest",
            formats=formats,
            gzip_jsonl=gzip_jsonl,
            rows_per_shard=rows_per_shard,
        )
        first1000_meta = export_snapshot(
            conn,
            schema=schema,
            snapshot_ref="first1000",
            snapshot_id=first1000_id,
            out_dir=out_root / "first1000",
            formats=formats,
            gzip_jsonl=gzip_jsonl,
            rows_per_shard=rows_per_shard,
        )
    finally:
        conn.close()

    readme = render_hf_dataset_card(
        dataset_id=dataset_id,
        pretty_name=pretty_name,
        paper=paper_meta,
        latest=latest_meta,
        first1000=first1000_meta,
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    if hf_repo_id:
        upload_export_root(
            out_root,
            UploadOptions(
                repo_id=hf_repo_id,
                token=hf_token,
                commit_message=f"Update exports (paper={paper_meta.snapshot_name}, latest={latest_meta.snapshot_name})",
                replace_folders=tuple(hf_replace_folders),
            ),
        )

    return PublishResult(paper=paper_meta, latest=latest_meta, first1000=first1000_meta)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create `paper` and `latest` snapshots and export them for Hugging Face publication."
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing snapshot tables (default: padjective)",
    )
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table name",
    )
    parser.add_argument(
        "--paper-tex",
        type=Path,
        default=Path("../papers/padjective/padjective.tex"),
        help="Path to the paper TeX used to infer the paper cutoff timestamp.",
    )
    parser.add_argument(
        "--paper-as-of",
        help="Override paper cutoff timestamp (ISO8601 or 'YYYY-MM-DD HH:MM UTC').",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=5,
        help="Minimum number of products a tag must appear in",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum number of products required per taxonomy label",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root directory (creates paper/ and latest/ subdirs).",
    )
    parser.add_argument(
        "--formats",
        default="jsonl,parquet",
        help="Comma-separated list: jsonl, parquet (default: jsonl,parquet).",
    )
    parser.add_argument(
        "--no-gzip",
        action="store_true",
        help="Disable gzip compression for JSONL outputs.",
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=200_000,
        help="Maximum products per output shard file.",
    )
    parser.add_argument(
        "--dataset-id",
        default="product-taxonomy-bench",
        help="HF dataset repo id/name used in the dataset card template.",
    )
    parser.add_argument(
        "--pretty-name",
        default="Product Taxonomy Bench (Anonymized)",
        help="HF dataset pretty_name used in the dataset card template.",
    )
    parser.add_argument(
        "--hf-repo-id",
        help="Optional Hugging Face dataset repo id (e.g. username/product-taxonomy-bench). Uploads exports when set.",
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token (otherwise uses huggingface_hub defaults/env).",
    )
    parser.add_argument(
        "--hf-replace-folders",
        default="latest",
        help="Comma-separated list of folders to delete before upload (default: latest).",
    )
    args = parser.parse_args()

    formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    unknown = [f for f in formats if f not in {"jsonl", "parquet"}]
    if unknown:
        raise SystemExit(f"Unknown formats: {', '.join(sorted(set(unknown)))}")

    if args.paper_as_of:
        paper_as_of = parse_as_of(args.paper_as_of)
    else:
        tex_text = args.paper_tex.read_text(encoding="utf-8")
        paper_as_of = parse_latest_utc_timestamp_from_tex(tex_text)

    replace_folders = tuple(
        part.strip().strip("/")
        for part in args.hf_replace_folders.split(",")
        if part.strip().strip("/")
    )

    publish(
        dsn=args.dsn,
        schema=args.schema,
        product_table=args.product_table,
        paper_as_of=paper_as_of,
        min_tag_count=args.min_tag_count,
        min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        out_root=args.out_root,
        formats=formats,
        gzip_jsonl=not args.no_gzip,
        rows_per_shard=args.rows_per_shard,
        dataset_id=args.dataset_id,
        pretty_name=args.pretty_name,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        hf_replace_folders=replace_folders,
    )


if __name__ == "__main__":
    main()
