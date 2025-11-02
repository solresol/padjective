"""Snapshot current model performance metrics for historical tracking.

This script captures the current state of all models (umllr, LR, NN) and stores
a snapshot in the model_performance_history table for trend analysis over time.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:
    from . import db

try:
    from psycopg2 import sql
except ImportError:
    try:
        from psycopg import sql
    except ImportError:
        sql = None


def create_history_table(conn, schema: str = "padjective") -> None:
    """Create the model performance history table if it doesn't exist."""
    with conn.cursor() as cur:
        # Override default tablespace to avoid permission issues
        cur.execute("SET LOCAL default_tablespace = ''")

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {schema}.model_performance_history (
                    snapshot_date DATE NOT NULL,
                    snapshot_time TIMESTAMPTZ NOT NULL DEFAULT now(),
                    num_products INTEGER NOT NULL,
                    num_tags INTEGER NOT NULL,
                    num_taxonomies INTEGER NOT NULL,
                    umllr_mean_padic_loss REAL,
                    lr_mean_padic_loss REAL,
                    nn_mean_padic_loss REAL,
                    umllr_mean_accuracy REAL,
                    lr_mean_accuracy REAL,
                    nn_mean_accuracy REAL,
                    PRIMARY KEY (snapshot_date)
                )
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def get_dataset_stats(conn, product_table: str) -> tuple[int, int, int]:
    """Get current dataset statistics.

    Returns:
        tuple: (num_products, num_tags, num_taxonomies)
    """
    with conn.cursor() as cur:
        # Count products with taxonomy
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(DISTINCT p.id)
                FROM {product_table} AS p
                JOIN public.product_details pd ON
                    p.myshopify_domain = pd.myshopify_domain
                    AND p.run_name = pd.run_name
                    AND p.product_handle = pd.product_handle
                LEFT JOIN public.product_taxonomy pt ON p.id = pt.product_id
                WHERE pt.taxonomy_id IS NOT NULL
                """
            ).format(product_table=sql.Identifier(*product_table.split("."))),
        )
        num_products = cur.fetchone()[0]

        # Count distinct tags (tags that appear at least twice)
        cur.execute(
            sql.SQL(
                """
                WITH product_tags AS (
                    SELECT p.id,
                           UNNEST(string_to_array(pd.product_detail->'product'->>'tags', ',')) AS tag
                    FROM {product_table} AS p
                    JOIN public.product_details pd ON
                        p.myshopify_domain = pd.myshopify_domain
                        AND p.run_name = pd.run_name
                        AND p.product_handle = pd.product_handle
                    LEFT JOIN public.product_taxonomy pt ON p.id = pt.product_id
                    WHERE pt.taxonomy_id IS NOT NULL
                )
                SELECT COUNT(DISTINCT TRIM(tag))
                FROM product_tags
                WHERE TRIM(tag) != ''
                GROUP BY TRIM(tag)
                HAVING COUNT(*) >= 2
                """
            ).format(product_table=sql.Identifier(*product_table.split("."))),
        )
        num_tags = len(cur.fetchall())

        # Count distinct taxonomies
        cur.execute(
            sql.SQL(
                """
                SELECT COUNT(DISTINCT pt.taxonomy_id)
                FROM {product_table} AS p
                LEFT JOIN public.product_taxonomy pt ON p.id = pt.product_id
                WHERE pt.taxonomy_id IS NOT NULL
                """
            ).format(product_table=sql.Identifier(*product_table.split("."))),
        )
        num_taxonomies = cur.fetchone()[0]

    return num_products, num_tags, num_taxonomies


def get_umllr_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None]:
    """Get umllr mean p-adic loss across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy)
    """
    with conn.cursor() as cur:
        # Get mean p-adic loss from predictions
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(loss)
                FROM {schema}.umllr_predictions
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        mean_padic_loss = float(row[0]) if row and row[0] is not None else None

    # umllr doesn't track accuracy, so return None
    return mean_padic_loss, None


def get_lr_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None]:
    """Get LR mean p-adic loss and accuracy across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy)
                FROM {schema}.taxonomy_lr_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0]), float(row[1]) if row[1] is not None else None

    return None, None


def get_nn_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None]:
    """Get NN mean p-adic loss and accuracy across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy)
                FROM {schema}.taxonomy_nn_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0]), float(row[1]) if row[1] is not None else None

    return None, None


def snapshot_metrics(
    conn,
    product_table: str = "cantbuymelove.product",
    schema: str = "padjective",
    snapshot_date: str | None = None,
) -> None:
    """Snapshot current model performance metrics.

    Args:
        conn: Database connection
        product_table: Qualified product table name
        schema: Schema name for results tables
        snapshot_date: Date for snapshot (YYYY-MM-DD), defaults to today
    """
    # Create history table if it doesn't exist
    create_history_table(conn, schema)

    # Get current date for snapshot
    if snapshot_date:
        snap_date = datetime.fromisoformat(snapshot_date).date()
    else:
        snap_date = datetime.now(timezone.utc).date()

    # Gather dataset statistics
    num_products, num_tags, num_taxonomies = get_dataset_stats(conn, product_table)

    # Gather model metrics
    umllr_padic, umllr_acc = get_umllr_metrics(conn, schema)
    lr_padic, lr_acc = get_lr_metrics(conn, schema)
    nn_padic, nn_acc = get_nn_metrics(conn, schema)

    # Insert or update snapshot
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.model_performance_history
                (snapshot_date, num_products, num_tags, num_taxonomies,
                 umllr_mean_padic_loss, lr_mean_padic_loss, nn_mean_padic_loss,
                 umllr_mean_accuracy, lr_mean_accuracy, nn_mean_accuracy)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    snapshot_time = now(),
                    num_products = EXCLUDED.num_products,
                    num_tags = EXCLUDED.num_tags,
                    num_taxonomies = EXCLUDED.num_taxonomies,
                    umllr_mean_padic_loss = EXCLUDED.umllr_mean_padic_loss,
                    lr_mean_padic_loss = EXCLUDED.lr_mean_padic_loss,
                    nn_mean_padic_loss = EXCLUDED.nn_mean_padic_loss,
                    umllr_mean_accuracy = EXCLUDED.umllr_mean_accuracy,
                    lr_mean_accuracy = EXCLUDED.lr_mean_accuracy,
                    nn_mean_accuracy = EXCLUDED.nn_mean_accuracy
                """
            ).format(schema=sql.Identifier(schema)),
            (
                snap_date,
                num_products,
                num_tags,
                num_taxonomies,
                umllr_padic,
                lr_padic,
                nn_padic,
                umllr_acc,
                lr_acc,
                nn_acc,
            ),
        )
    conn.commit()

    print(f"Snapshot saved for {snap_date}")
    print(f"  Products: {num_products}, Tags: {num_tags}, Taxonomies: {num_taxonomies}")
    print(f"  umllr p-adic loss: {umllr_padic:.6f}" if umllr_padic else "  umllr: no data")
    print(f"  LR p-adic loss: {lr_padic:.6f}" if lr_padic else "  LR: no data")
    print(f"  NN p-adic loss: {nn_padic:.6f}" if nn_padic else "  NN: no data")


def main() -> None:
    """Command-line interface for snapshotting model performance metrics."""
    parser = argparse.ArgumentParser(
        description="Snapshot current model performance metrics for historical tracking"
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
        "--schema",
        default="padjective",
        help="Schema name for results tables",
    )
    parser.add_argument(
        "--date",
        help="Snapshot date (YYYY-MM-DD), defaults to today",
    )

    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        snapshot_metrics(
            conn,
            product_table=args.product_table,
            schema=args.schema,
            snapshot_date=args.date,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
