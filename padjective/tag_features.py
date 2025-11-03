"""Utilities for extracting product tags as sparse feature matrices.

The heavy lifting is delegated to :mod:`padjective.data_access`, ensuring every
consumer works from the same normalised dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import sparse

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import data_access, db
else:
    from . import data_access, db

normalize_tag = data_access.normalize_tag
parse_tags = data_access.parse_tags


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
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=include_taxonomy,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=None,
    )

    metadata = dataset.metadata.copy()
    if not include_taxonomy:
        metadata = metadata[["product_id", "title", "tags", "tag_count", "valid_tag_count"]]

    return dataset.features, metadata, dataset.feature_names


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
