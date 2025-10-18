"""Utilities for extracting product tags as sparse feature matrices.

This module provides functionality to extract tags from products stored in the
Shopify database and convert them into sparse dataframes suitable for machine
learning tasks like taxonomy classification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row
from scipy import sparse

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:
    from . import db


def normalize_tag(tag: str) -> str:
    """Normalize a tag to uppercase and strip whitespace.

    Args:
        tag: Raw tag string

    Returns:
        Normalized tag string
    """
    return tag.strip().upper()


def parse_tags(tag_string: str | None) -> list[str]:
    """Parse a comma-separated tag string into a list of normalized tags.

    Args:
        tag_string: Comma-separated tag string (or None)

    Returns:
        List of normalized, non-empty tags
    """
    if not tag_string:
        return []

    tags = []
    for tag in tag_string.split(","):
        normalized = normalize_tag(tag)
        if normalized:
            tags.append(normalized)
    return tags


def stream_product_tags(
    conn,
    product_table: str = "cantbuymelove.product",
    include_taxonomy: bool = True,
) -> Iterator[dict]:
    """Stream products with their tags and optional taxonomy information.

    Args:
        conn: psycopg connection to the database
        product_table: Qualified name of the product table
        include_taxonomy: Whether to join with taxonomy table

    Yields:
        dict: Product records with id, title, tags, and optionally taxonomy_id
        and taxonomy_path
    """
    product_identifier = db.qualified_identifier(product_table)

    if include_taxonomy:
        query = sql.SQL(
            """
            SELECT
                p.id,
                p.product_title AS title,
                pd.product_detail->'product'->>'tags' AS tags,
                pt.taxonomy_id,
                t.taxonomy_path
            FROM {products} AS p
            JOIN public.product_details pd ON
                p.myshopify_domain = pd.myshopify_domain
                AND p.run_name = pd.run_name
                AND p.product_handle = pd.product_handle
            JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
            JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
            WHERE p.product_title IS NOT NULL
            ORDER BY p.id
            """
        ).format(products=product_identifier)
    else:
        query = sql.SQL(
            """
            SELECT
                p.id,
                p.product_title AS title,
                pd.product_detail->'product'->>'tags' AS tags
            FROM {products} AS p
            JOIN public.product_details pd ON
                p.myshopify_domain = pd.myshopify_domain
                AND p.run_name = pd.run_name
                AND p.product_handle = pd.product_handle
            WHERE p.product_title IS NOT NULL
            ORDER BY p.id
            """
        ).format(products=product_identifier)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            yield row


def extract_tag_features(
    conn,
    product_table: str = "cantbuymelove.product",
    include_taxonomy: bool = True,
    min_tag_count: int = 2,
) -> tuple[sparse.csr_matrix, pd.DataFrame, list[str]]:
    """Extract tag features as a sparse matrix.

    Args:
        conn: psycopg connection to the database
        product_table: Qualified name of the product table
        include_taxonomy: Whether to include taxonomy_id in the metadata
        min_tag_count: Minimum number of products a tag must appear in to be included

    Returns:
        tuple: (sparse_matrix, metadata_df, feature_names)
            - sparse_matrix: scipy CSR sparse matrix (n_products x n_tags)
            - metadata_df: DataFrame with product_id, title, and optionally taxonomy_id
            - feature_names: List of tag names corresponding to matrix columns
    """
    # First pass: collect all products and tags
    products = []
    product_tags_list = []
    tag_counts = {}

    for row in stream_product_tags(conn, product_table, include_taxonomy):
        product_id = row["id"]
        title = row.get("title", "")
        tags_str = row.get("tags", "")
        taxonomy_id = row.get("taxonomy_id")

        tags = parse_tags(tags_str)

        product_record = {
            "product_id": product_id,
            "title": title,
        }
        if include_taxonomy:
            product_record["taxonomy_id"] = taxonomy_id
            product_record["taxonomy_path"] = row.get("taxonomy_path")

        products.append(product_record)
        product_tags_list.append(tags)

        # Count tag occurrences
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Filter tags by minimum count
    valid_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_count}
    feature_names = sorted(valid_tags)
    tag_to_index = {tag: idx for idx, tag in enumerate(feature_names)}

    # Build sparse matrix
    n_products = len(products)
    n_features = len(feature_names)

    rows = []
    cols = []
    data = []

    for product_idx, tags in enumerate(product_tags_list):
        for tag in tags:
            if tag in tag_to_index:
                rows.append(product_idx)
                cols.append(tag_to_index[tag])
                data.append(1.0)

    sparse_matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_products, n_features),
        dtype=float,
    )

    # Create metadata DataFrame
    if include_taxonomy:
        metadata_df = pd.DataFrame(
            products,
            columns=["product_id", "title", "taxonomy_id", "taxonomy_path"],
        )
    else:
        metadata_df = pd.DataFrame(products, columns=["product_id", "title"])

    return sparse_matrix, metadata_df, feature_names


def create_dense_dataframe(
    sparse_matrix: sparse.csr_matrix,
    metadata_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Convert sparse matrix to dense DataFrame (use with caution for large datasets).

    Args:
        sparse_matrix: Sparse feature matrix
        metadata_df: Product metadata
        feature_names: Tag names

    Returns:
        Dense DataFrame with metadata and tag columns
    """
    dense_array = sparse_matrix.toarray()
    tag_df = pd.DataFrame(dense_array, columns=feature_names)
    result = pd.concat([metadata_df.reset_index(drop=True), tag_df], axis=1)
    return result


def main() -> None:
    """Command-line interface for tag feature extraction."""
    parser = argparse.ArgumentParser(
        description="Extract product tags as sparse feature matrix"
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table name",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=2,
        help="Minimum occurrences for a tag to be included",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save sparse matrix (in npz format)",
    )
    parser.add_argument(
        "--output-metadata",
        type=Path,
        help="Optional path to save metadata CSV",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)

    sparse_matrix, metadata_df, feature_names = extract_tag_features(
        conn,
        product_table=args.product_table,
        include_taxonomy=True,
        min_tag_count=args.min_tag_count,
    )

    conn.close()

    print(f"Extracted {sparse_matrix.shape[0]} products with {sparse_matrix.shape[1]} tags")
    print(f"Sparsity: {1 - sparse_matrix.nnz / (sparse_matrix.shape[0] * sparse_matrix.shape[1]):.2%}")

    if args.output:
        sparse.save_npz(args.output, sparse_matrix)
        print(f"Saved sparse matrix to {args.output}")

        # Save feature names
        feature_path = args.output.with_suffix(".features.txt")
        feature_path.write_text("\n".join(feature_names))
        print(f"Saved feature names to {feature_path}")

    if args.output_metadata:
        metadata_df.to_csv(args.output_metadata, index=False)
        print(f"Saved metadata to {args.output_metadata}")


if __name__ == "__main__":
    main()
