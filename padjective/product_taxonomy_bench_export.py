"""Export product-taxonomy-bench snapshots for publication (JSONL/Parquet + HF card).

The snapshot source of truth is Postgres (Shopify stores database). Exports are
materialized to the local filesystem for publishing on platforms such as
Hugging Face.
"""

from __future__ import annotations

import argparse
import gzip
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

if __package__ in {None, ""}:
    import sys

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
    from padjective.product_taxonomy_bench_notebook import write_notebook
else:  # pragma: no cover
    from . import db
    from .product_taxonomy_bench_notebook import write_notebook

try:  # Optional dependency; used only for Parquet exports.
    import pyarrow as pa  # type: ignore[import-not-found]
    import pyarrow.parquet as pq  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional
    pa = None
    pq = None


DEFAULT_SCHEMA = "padjective"
DEFAULT_HF_NOTEBOOK_SOURCE = (
    Path(__file__).resolve().parent.parent / "docs" / "product_taxonomy_bench.ipynb"
)
DEFAULT_HF_NOTEBOOK_DEST = Path("notebooks/product_taxonomy_bench.ipynb")


@dataclass(frozen=True)
class SnapshotMetadata:
    snapshot_id: uuid.UUID
    snapshot_name: str
    created_at: datetime
    as_of: Optional[datetime]
    product_table: str
    min_tag_count: int
    min_samples_per_taxonomy: int
    product_count: int
    tag_count: int
    taxonomy_count: int
    note: Optional[str]
    code_version: Optional[str]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def resolve_snapshot_id(
    conn: psycopg.Connection,
    schema: str,
    snapshot_ref: str,
) -> uuid.UUID:
    """Resolve a snapshot reference (alias, name, or UUID) to ``snapshot_id``."""

    snapshot_ref = snapshot_ref.strip()
    if not snapshot_ref:
        raise ValueError("snapshot_ref must not be empty")

    try:
        return uuid.UUID(snapshot_ref)
    except ValueError:
        pass

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT snapshot_id
                FROM {schema}.product_taxonomy_bench_snapshot_aliases
                WHERE alias = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_ref,),
        )
        row = cur.fetchone()
        if row:
            return uuid.UUID(str(row[0]))

        cur.execute(
            sql.SQL(
                """
                SELECT snapshot_id
                FROM {schema}.product_taxonomy_bench_snapshots
                WHERE snapshot_name = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_ref,),
        )
        row = cur.fetchone()
        if row:
            return uuid.UUID(str(row[0]))

    raise ValueError(f"Unknown snapshot reference: {snapshot_ref!r}")


def snapshot_ref_is_alias(
    conn: psycopg.Connection, schema: str, snapshot_ref: str
) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT 1
                FROM {schema}.product_taxonomy_bench_snapshot_aliases
                WHERE alias = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_ref,),
        )
        return cur.fetchone() is not None


def load_snapshot_metadata(
    conn: psycopg.Connection, schema: str, snapshot_id: uuid.UUID
) -> SnapshotMetadata:
    has_as_of = _table_has_column(
        conn, schema=schema, table="product_taxonomy_bench_snapshots", column="as_of"
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    snapshot_id,
                    snapshot_name,
                    created_at,
                    {as_of_column},
                    product_table,
                    min_tag_count,
                    min_samples_per_taxonomy,
                    product_count,
                    tag_count,
                    taxonomy_count,
                    note,
                    code_version
                FROM {schema}.product_taxonomy_bench_snapshots
                WHERE snapshot_id = %s
                """
            ).format(
                schema=sql.Identifier(schema),
                as_of_column=sql.SQL("as_of") if has_as_of else sql.SQL("NULL AS as_of"),
            ),
            (snapshot_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

    return SnapshotMetadata(
        snapshot_id=uuid.UUID(str(row["snapshot_id"])),
        snapshot_name=str(row["snapshot_name"]),
        created_at=row["created_at"],
        as_of=row.get("as_of"),
        product_table=str(row["product_table"]),
        min_tag_count=int(row["min_tag_count"]),
        min_samples_per_taxonomy=int(row["min_samples_per_taxonomy"]),
        product_count=int(row["product_count"]),
        tag_count=int(row["tag_count"]),
        taxonomy_count=int(row["taxonomy_count"]),
        note=row.get("note"),
        code_version=row.get("code_version"),
    )


def _table_has_column(conn: psycopg.Connection, *, schema: str, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s AND column_name = %s
            """,
            (schema, table, column),
        )
        return cur.fetchone() is not None


def stream_products_for_export(
    conn: psycopg.Connection, schema: str, snapshot_id: uuid.UUID
) -> Iterator[dict[str, Any]]:
    """Yield one JSON-serialisable record per product (streaming)."""

    query = sql.SQL(
        """
        SELECT
            p.product_id_hash,
            p.taxonomy_id,
            p.taxonomy_path,
            p.taxonomy_name,
            p.cv_fold,
            p.tag_count,
            p.title_part_count,
            pt.tag_id,
            pt.in_title,
            pt.title_part,
            pt.title_position
        FROM {schema}.product_taxonomy_bench_products p
        JOIN {schema}.product_taxonomy_bench_product_tags pt ON (
            pt.snapshot_id = p.snapshot_id
            AND pt.product_id_hash = p.product_id_hash
        )
        WHERE p.snapshot_id = %s
        ORDER BY p.product_id_hash, pt.tag_id
        """
    ).format(schema=sql.Identifier(schema))

    current_product: dict[str, Any] | None = None
    current_hash: str | None = None
    tag_features: list[dict[str, Any]] = []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (snapshot_id,))
        for row in cur:
            product_hash = str(row["product_id_hash"])
            if current_hash != product_hash:
                if current_product is not None:
                    current_product["tag_features"] = tag_features
                    yield current_product
                current_hash = product_hash
                tag_features = []
                current_product = {
                    "product_id_hash": product_hash,
                    "taxonomy_id": str(row["taxonomy_id"]),
                    "taxonomy_path": str(row["taxonomy_path"]),
                    "taxonomy_name": str(row["taxonomy_name"]),
                    "cv_fold": int(row["cv_fold"]) if row["cv_fold"] is not None else None,
                    "tag_count": int(row["tag_count"]),
                    "title_part_count": int(row["title_part_count"]),
                }

            tag_features.append(
                {
                    "tag_id": str(row["tag_id"]),
                    "in_title": bool(row["in_title"]),
                    "title_part": (
                        int(row["title_part"]) if row["title_part"] is not None else None
                    ),
                    "title_position": (
                        int(row["title_position"])
                        if row["title_position"] is not None
                        else None
                    ),
                }
            )

    if current_product is not None:
        current_product["tag_features"] = tag_features
        yield current_product


def export_tags(
    conn: psycopg.Connection,
    schema: str,
    snapshot_id: uuid.UUID,
    *,
    out_dir: Path,
    formats: Sequence[str],
    gzip_jsonl: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tag_id, tag_rank
                FROM {schema}.product_taxonomy_bench_tags
                WHERE snapshot_id = %s
                ORDER BY tag_rank
                """
            ).format(schema=sql.Identifier(schema)),
            (snapshot_id,),
        )
        rows = list(cur.fetchall())

    if "jsonl" in formats:
        path = out_dir / ("tags.jsonl.gz" if gzip_jsonl else "tags.jsonl")
        opener = gzip.open if gzip_jsonl else open  # type: ignore[assignment]
        with opener(path, "wt", encoding="utf-8") as f:
            for row in rows:
                f.write(_json_dumps({"tag_id": row["tag_id"], "tag_rank": int(row["tag_rank"])}))
                f.write("\n")

    if "parquet" in formats:
        if pa is None or pq is None:
            raise RuntimeError(
                "Parquet export requested but pyarrow is not available. Install it with `uv add pyarrow`."
            )
        table = pa.Table.from_pylist(
            [{"tag_id": row["tag_id"], "tag_rank": int(row["tag_rank"])} for row in rows],
            schema=pa.schema(
                [
                    ("tag_id", pa.string()),
                    ("tag_rank", pa.int32()),
                ]
            ),
        )
        pq.write_table(table, out_dir / "tags.parquet", compression="zstd")


def export_snapshot_metadata(metadata: SnapshotMetadata, *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "snapshot_id": str(metadata.snapshot_id),
        "snapshot_name": metadata.snapshot_name,
        "created_at": metadata.created_at.isoformat(),
        "as_of": metadata.as_of.isoformat() if metadata.as_of else None,
        "product_table": metadata.product_table,
        "min_tag_count": metadata.min_tag_count,
        "min_samples_per_taxonomy": metadata.min_samples_per_taxonomy,
        "product_count": metadata.product_count,
        "tag_count": metadata.tag_count,
        "taxonomy_count": metadata.taxonomy_count,
        "note": metadata.note,
        "code_version": metadata.code_version,
    }
    (out_dir / "snapshot.json").write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def export_products_jsonl(
    conn: psycopg.Connection,
    schema: str,
    snapshot_id: uuid.UUID,
    *,
    out_dir: Path,
    gzip_jsonl: bool,
    rows_per_shard: int,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")

    shard_paths: list[Path] = []
    shard_idx = 0
    row_idx = 0

    def make_path(index: int) -> Path:
        name = f"products-{index:05d}.jsonl"
        if gzip_jsonl:
            name += ".gz"
        return out_dir / name

    opener = gzip.open if gzip_jsonl else open  # type: ignore[assignment]
    current_path = make_path(shard_idx)
    shard_paths.append(current_path)
    out_file = opener(current_path, "wt", encoding="utf-8")

    try:
        for record in stream_products_for_export(conn, schema, snapshot_id):
            out_file.write(_json_dumps(record))
            out_file.write("\n")
            row_idx += 1
            if row_idx % rows_per_shard == 0:
                out_file.close()
                shard_idx += 1
                current_path = make_path(shard_idx)
                shard_paths.append(current_path)
                out_file = opener(current_path, "wt", encoding="utf-8")
    finally:
        out_file.close()

    # If we created an empty final shard, drop it.
    if shard_paths and shard_paths[-1].stat().st_size == 0:
        shard_paths[-1].unlink(missing_ok=True)
        shard_paths.pop()

    return shard_paths


def export_products_parquet(
    conn: psycopg.Connection,
    schema: str,
    snapshot_id: uuid.UUID,
    *,
    out_dir: Path,
    rows_per_shard: int,
    batch_rows: int = 5000,
) -> list[Path]:
    if pa is None or pq is None:
        raise RuntimeError(
            "Parquet export requested but pyarrow is not available. Install it with `uv add pyarrow`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows_per_shard <= 0:
        raise ValueError("rows_per_shard must be positive")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")

    schema_pa = pa.schema(
        [
            ("product_id_hash", pa.string()),
            ("taxonomy_id", pa.string()),
            ("taxonomy_path", pa.string()),
            ("taxonomy_name", pa.string()),
            ("cv_fold", pa.int32()),
            ("tag_count", pa.int32()),
            ("title_part_count", pa.int32()),
            (
                "tag_features",
                pa.list_(
                    pa.struct(
                        [
                            ("tag_id", pa.string()),
                            ("in_title", pa.bool_()),
                            ("title_part", pa.int32()),
                            ("title_position", pa.int32()),
                        ]
                    )
                ),
            ),
        ]
    )

    shard_paths: list[Path] = []
    shard_idx = 0
    record_idx = 0

    def make_path(index: int) -> Path:
        return out_dir / f"products-{index:05d}.parquet"

    current_path = make_path(shard_idx)
    shard_paths.append(current_path)
    writer: pq.ParquetWriter | None = pq.ParquetWriter(
        current_path, schema=schema_pa, compression="zstd"
    )

    buffer: list[dict[str, Any]] = []

    def flush() -> None:
        nonlocal buffer, writer
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=schema_pa)
        assert writer is not None
        writer.write_table(table)
        buffer = []

    try:
        for record in stream_products_for_export(conn, schema, snapshot_id):
            record_idx += 1
            buffer.append(record)
            if len(buffer) >= batch_rows:
                flush()
            if record_idx % rows_per_shard == 0:
                flush()
                assert writer is not None
                writer.close()
                shard_idx += 1
                current_path = make_path(shard_idx)
                shard_paths.append(current_path)
                writer = pq.ParquetWriter(
                    current_path, schema=schema_pa, compression="zstd"
                )
    finally:
        flush()
        if writer is not None:
            writer.close()

    # Drop any empty trailing shard.
    if shard_paths and shard_paths[-1].stat().st_size == 0:
        shard_paths[-1].unlink(missing_ok=True)
        shard_paths.pop()

    return shard_paths


def render_hf_dataset_card(
    *,
    dataset_id: str,
    pretty_name: str,
    paper: SnapshotMetadata,
    latest: SnapshotMetadata,
    first1000: SnapshotMetadata | None = None,
) -> str:
    def _fmt(meta: SnapshotMetadata) -> str:
        as_of = f"; as_of `{meta.as_of.isoformat()}`" if meta.as_of else ""
        return (
            f"- `{meta.snapshot_name}` (created `{meta.created_at.isoformat()}`; "
            f"{meta.product_count:,} products, {meta.tag_count:,} tags, {meta.taxonomy_count:,} taxonomies{as_of})"
        )

    front_matter = "\n".join(
        [
            "---",
            f"pretty_name: {pretty_name}",
            "language:",
            "- en",
            "task_categories:",
            "- text-classification",
            "task_ids:",
            "- multi-class-classification",
            "tags:",
            "- ecommerce",
            "- shopify",
            "- taxonomy",
            "- benchmarking",
            "- p-adic",
            "- ultrametric",
            "license: other",
            "---",
        ]
    )

    config_lines = [
        "## Configurations",
        "",
        "Two configurations are provided:" if first1000 is None else "Three configurations are provided:",
        "",
        "### Paper snapshot",
        _fmt(paper),
        "",
        "### Latest snapshot",
        _fmt(latest),
        "",
    ]
    if first1000 is not None:
        config_lines.extend(
            [
                "### First 1000 snapshot",
                _fmt(first1000),
                "",
            ]
        )

    body = "\n".join(
        [
            "# Dataset Summary",
            "",
            f"`{dataset_id}` is an anonymised benchmark dataset for predicting Shopify Product Taxonomy categories from Shopify product tags.",
            "",
            "This dataset does **not** include raw product titles, raw tags, or product URLs. Tags are anonymised as `tagNNNNNN`.",
            "",
            "# Start Here",
            "",
            "- Read this dataset card for the snapshot layout and field definitions.",
            "- Open the benchmark notebook at [`notebooks/product_taxonomy_bench.ipynb`](./notebooks/product_taxonomy_bench.ipynb). On the notebook page, use the Hub's **Open in Colab** button to run it interactively. The notebook defaults to the fixed `paper` snapshot.",
            "- Use the snapshot folders according to your goal: `paper/` for the canonical point-in-time paper snapshot, `latest/` for the rolling benchmark, and `first1000/` for a tiny sanity-check slice when present.",
            "",
            *config_lines,
            "# Data Fields",
            "",
            "Each record corresponds to one product:",
            "",
            "- `product_id_hash`: SHA-256 hash of a canonicalised product URL",
            "- `taxonomy_id`: Shopify taxonomy GID",
            "- `taxonomy_path`: Numeric hierarchy path (dot-separated) when available",
            "- `taxonomy_name`: Human-readable hierarchy name",
            "- `cv_fold`: 0–4 fold assignment (or null if missing)",
            "- `tag_features`: list of `{tag_id, in_title, title_part, title_position}`",
            "",
            "Tag semantics are not included; `tag_id` values are stable only within a snapshot.",
            "",
            "# Generation",
            "",
            "Products were collected by fetching public Shopify product `.json` endpoints, then joined to the taxonomy label used by the cantbuymelove site. Tags are uppercased and substring-nested tags are filtered before anonymisation. Title overlap positions are computed by case-insensitive substring search and splitting titles on `\" - \"` to match the paper’s tag-battle logic. The paper snapshot is generated with a fixed `as_of` cutoff timestamp.",
            "",
            "# Citation",
            "",
            "Add your paper citation here (BibTeX).",
            "",
            "```bibtex",
            "@article{todo,",
            "  title={TODO},",
            "  author={TODO},",
            "  year={2026}",
            "}",
            "```",
        ]
    )

    return front_matter + "\n\n" + body + "\n"


def stage_hf_notebook(
    out_root: Path,
    notebook_dest: Path = DEFAULT_HF_NOTEBOOK_DEST,
) -> Path:
    """Generate the benchmark notebook into the export root for Hub publication."""

    destination = out_root / notebook_dest
    return write_notebook(destination)


def export_snapshot(
    conn: psycopg.Connection,
    *,
    schema: str,
    snapshot_ref: str,
    snapshot_id: uuid.UUID,
    out_dir: Path,
    formats: Sequence[str],
    gzip_jsonl: bool,
    rows_per_shard: int,
) -> SnapshotMetadata:
    metadata = load_snapshot_metadata(conn, schema, snapshot_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    _clear_snapshot_output_dir(out_dir)
    export_snapshot_metadata(metadata, out_dir=out_dir)
    export_tags(conn, schema, snapshot_id, out_dir=out_dir, formats=formats, gzip_jsonl=gzip_jsonl)

    if "jsonl" in formats:
        export_products_jsonl(
            conn,
            schema,
            snapshot_id,
            out_dir=out_dir,
            gzip_jsonl=gzip_jsonl,
            rows_per_shard=rows_per_shard,
        )
    if "parquet" in formats:
        export_products_parquet(
            conn,
            schema,
            snapshot_id,
            out_dir=out_dir,
            rows_per_shard=rows_per_shard,
        )

    return metadata


def _clear_snapshot_output_dir(out_dir: Path) -> None:
    patterns = (
        "snapshot.json",
        "tags.jsonl",
        "tags.jsonl.gz",
        "tags.parquet",
        "products-*.jsonl",
        "products-*.jsonl.gz",
        "products-*.parquet",
    )
    for pattern in patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export product-taxonomy-bench snapshots to JSONL/Parquet and scaffold a HF dataset card."
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--schema",
        default=DEFAULT_SCHEMA,
        help="Schema containing snapshot tables (default: padjective)",
    )
    parser.add_argument(
        "--snapshot",
        action="append",
        default=[],
        help="Snapshot reference (alias, name, or UUID). Repeatable.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root directory (subdirs are created per snapshot).",
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
        "--write-dataset-card",
        action="store_true",
        help="Write a Hugging Face dataset card README.md at out-root (expects paper+latest snapshots).",
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
    args = parser.parse_args()

    formats = [part.strip().lower() for part in args.formats.split(",") if part.strip()]
    unknown = [f for f in formats if f not in {"jsonl", "parquet"}]
    if unknown:
        raise SystemExit(f"Unknown formats: {', '.join(sorted(set(unknown)))}")
    if "parquet" in formats and (pa is None or pq is None):
        raise SystemExit(
            "Parquet export requested but pyarrow is not available. Install it with `uv add pyarrow` "
            "or run with `--formats jsonl`."
        )

    if not args.snapshot:
        raise SystemExit("Provide at least one --snapshot (e.g. --snapshot paper --snapshot latest)")

    gzip_jsonl = not args.no_gzip

    conn = db.get_connection(args.dsn)
    try:
        exported: dict[str, SnapshotMetadata] = {}

        for snapshot_ref in args.snapshot:
            snapshot_id = resolve_snapshot_id(conn, args.schema, snapshot_ref)
            is_alias = snapshot_ref_is_alias(conn, args.schema, snapshot_ref)
            out_name = snapshot_ref if is_alias else load_snapshot_metadata(conn, args.schema, snapshot_id).snapshot_name
            out_dir = args.out_root / out_name
            meta = export_snapshot(
                conn,
                schema=args.schema,
                snapshot_ref=snapshot_ref,
                snapshot_id=snapshot_id,
                out_dir=out_dir,
                formats=formats,
                gzip_jsonl=gzip_jsonl,
                rows_per_shard=args.rows_per_shard,
            )
            exported[out_name] = meta

        if args.write_dataset_card:
            if "paper" not in exported or "latest" not in exported:
                raise SystemExit(
                    "--write-dataset-card expects exported snapshots to include out dirs named 'paper' and 'latest'. "
                    "Use aliases (e.g. --snapshot paper --snapshot latest)."
                )
            readme = render_hf_dataset_card(
                dataset_id=args.dataset_id,
                pretty_name=args.pretty_name,
                paper=exported["paper"],
                latest=exported["latest"],
            )
            args.out_root.mkdir(parents=True, exist_ok=True)
            (args.out_root / "README.md").write_text(readme, encoding="utf-8")
            stage_hf_notebook(args.out_root)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
