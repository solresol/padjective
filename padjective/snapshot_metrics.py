"""Snapshot current model performance metrics for historical tracking.

This script captures the current state of all models (umllr, PCLR, PCNN) and stores
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
    from padjective import data_access, db
else:
    from . import data_access, db

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
                    dummy_mean_padic_loss REAL,
                    umllr_mean_accuracy REAL,
                    lr_mean_accuracy REAL,
                    nn_mean_accuracy REAL,
                    dummy_mean_accuracy REAL,
                    PRIMARY KEY (snapshot_date)
                )
                """
            ).format(schema=sql.Identifier(schema))
        )

        # Add missing columns if table already exists (migration)
        cur.execute(
            sql.SQL(
                """
                ALTER TABLE {schema}.model_performance_history
                ADD COLUMN IF NOT EXISTS dummy_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS dummy_mean_accuracy REAL,
                ADD COLUMN IF NOT EXISTS ulr_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS ulr_mean_accuracy REAL,
                ADD COLUMN IF NOT EXISTS ulr_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS lr_mean_params REAL,
                ADD COLUMN IF NOT EXISTS nn_mean_input_weights REAL,
                ADD COLUMN IF NOT EXISTS unn_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS unn_mean_accuracy REAL,
                ADD COLUMN IF NOT EXISTS unn_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS dt_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS dt_mean_accuracy REAL,
                ADD COLUMN IF NOT EXISTS dt_mean_tree_depth REAL,
                ADD COLUMN IF NOT EXISTS zubarev_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS zubarev_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS zubarev_zeros_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS zubarev_zeros_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_m1_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_m1_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_m2_mean_padic_loss REAL,
                ADD COLUMN IF NOT EXISTS zubarev_umllr_m2_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS umllr_mean_nonzero_params REAL,
                ADD COLUMN IF NOT EXISTS dt_mean_effective_params REAL
                """
            ).format(schema=sql.Identifier(schema))
        )
    conn.commit()


def get_dataset_stats(
    conn,
    product_table: str,
    *,
    min_tag_count: int = 2,
    min_samples_per_taxonomy: int = 5,
) -> tuple[int, int, int]:
    """Get current dataset statistics.

    Returns:
        tuple: (num_products, num_tags, num_taxonomies)
    """
    dataset = data_access.build_feature_dataset(
        conn,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )

    num_products = dataset.product_count
    num_tags = len(dataset.feature_names)
    num_taxonomies = dataset.taxonomy_count

    return num_products, num_tags, num_taxonomies


def get_umllr_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None]:
    """Get umllr mean p-adic loss and non-zero params across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_nonzero_params)
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

        # Get average non-zero coefficients per fold
        mean_nonzero = None
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(num_coeffs) FROM (
                    SELECT cv_fold, COUNT(*) as num_coeffs
                    FROM {schema}.umllr_tag_coefficients
                    WHERE coefficient != 0
                    GROUP BY cv_fold
                ) sub
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            mean_nonzero = float(row[0])

    # umllr doesn't track accuracy, so return None
    return mean_padic_loss, None, mean_nonzero


def get_lr_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None]:
    """Get PCLR mean p-adic loss, accuracy, and parameter count across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_params)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy)
                FROM {schema}.taxonomy_pclr_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        padic_loss = float(row[0]) if row and row[0] is not None else None
        accuracy = float(row[1]) if row and row[1] is not None else None

        # Get parameter count from coefficients table
        mean_params = None
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(param_count) FROM (
                    SELECT cv_fold, COUNT(*) as param_count
                    FROM {schema}.taxonomy_pclr_coefficients
                    GROUP BY cv_fold
                ) sub
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            mean_params = float(row[0])

    return padic_loss, accuracy, mean_params


def get_nn_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None]:
    """Get PCNN mean p-adic loss, accuracy, and input weight count across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_input_weights)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy)
                FROM {schema}.taxonomy_pcnn_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        padic_loss = float(row[0]) if row and row[0] is not None else None
        accuracy = float(row[1]) if row and row[1] is not None else None

        # Get input weight count from weights table
        mean_input_weights = None
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(weight_count) FROM (
                    SELECT cv_fold, COUNT(*) as weight_count
                    FROM {schema}.taxonomy_pcnn_input_weights
                    GROUP BY cv_fold
                ) sub
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            mean_input_weights = float(row[0])

    return padic_loss, accuracy, mean_input_weights


def get_dummy_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None]:
    """Get dummy baseline mean p-adic loss and accuracy across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(loss), AVG(accuracy)
                FROM {schema}.dummy_fold_metrics
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return float(row[0]), float(row[1]) if row[1] is not None else None

    return None, None


def get_ulr_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None]:
    """Get ULR (Unconstrained Logistic Regression) mean metrics across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_nonzero_params)
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy), AVG(num_nonzero_params)
                FROM {schema}.taxonomy_ulr_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return (
                float(row[0]),
                float(row[1]) if row[1] is not None else None,
                float(row[2]) if row[2] is not None else None,
            )

    return None, None, None


def get_unn_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None]:
    """Get UNN (Unconstrained Neural Network) mean metrics across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_nonzero_params)
    """
    # Check if table exists first
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'taxonomy_unn_fold_results'
            )
            """,
            (schema,)
        )
        if not cur.fetchone()[0]:
            return None, None, None

        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy), AVG(num_nonzero_params)
                FROM {schema}.taxonomy_unn_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return (
                float(row[0]),
                float(row[1]) if row[1] is not None else None,
                float(row[2]) if row[2] is not None else None,
            )

    return None, None, None


def get_dt_metrics(conn, schema: str = "padjective") -> tuple[float | None, float | None, float | None, float | None]:
    """Get DT (Decision Tree) mean metrics across all folds.

    Returns:
        tuple: (mean_padic_loss, mean_accuracy, mean_tree_depth, mean_effective_params)
    """
    # Check if table exists first
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'taxonomy_dt_fold_results'
            )
            """,
            (schema,)
        )
        if not cur.fetchone()[0]:
            return None, None, None, None

        cur.execute(
            sql.SQL(
                """
                SELECT AVG(padic_loss_mean), AVG(test_accuracy), AVG(tree_depth), AVG(effective_params)
                FROM {schema}.taxonomy_dt_fold_results
                """
            ).format(schema=sql.Identifier(schema))
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            return (
                float(row[0]),
                float(row[1]) if row[1] is not None else None,
                float(row[2]) if row[2] is not None else None,
                float(row[3]) if row[3] is not None else None,
            )

    return None, None, None, None


def get_zubarev_metrics(conn, schema: str = "padjective", initialization_method: str = "umllr", mahler_degree: int = 0) -> tuple[float | None, float | None]:
    """Get Zubarev p-adic polynomial regression mean metrics across all folds.

    Args:
        conn: Database connection
        schema: Schema name for results tables
        initialization_method: Initialization method ('umllr' or 'zeros')
        mahler_degree: Mahler polynomial degree (0, 1, or 2)

    Returns:
        tuple: (mean_padic_loss, mean_nonzero_params)
    """
    # Check if table exists first
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = %s AND table_name = 'zubarev_fold_metrics'
            )
            """,
            (schema,)
        )
        if not cur.fetchone()[0]:
            return None, None

        # Get mean p-adic loss from predictions
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(loss)
                FROM {schema}.zubarev_predictions
                WHERE initialization_method = %s AND mahler_degree = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (initialization_method, mahler_degree)
        )
        row = cur.fetchone()
        mean_padic_loss = float(row[0]) if row and row[0] is not None else None

        # Get mean non-zero coefficients per fold
        cur.execute(
            sql.SQL(
                """
                SELECT AVG(num_nonzero) FROM (
                    SELECT cv_fold, COUNT(*) as num_nonzero
                    FROM {schema}.zubarev_tag_coefficients
                    WHERE coefficient != 0 AND initialization_method = %s AND mahler_degree = %s
                    GROUP BY cv_fold
                ) sub
                """
            ).format(schema=sql.Identifier(schema)),
            (initialization_method, mahler_degree)
        )
        row = cur.fetchone()
        mean_nonzero = float(row[0]) if row and row[0] is not None else None

    return mean_padic_loss, mean_nonzero


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
    umllr_padic, umllr_acc, umllr_nonzero = get_umllr_metrics(conn, schema)
    lr_padic, lr_acc, lr_params = get_lr_metrics(conn, schema)
    nn_padic, nn_acc, nn_input_weights = get_nn_metrics(conn, schema)
    dummy_padic, dummy_acc = get_dummy_metrics(conn, schema)
    ulr_padic, ulr_acc, ulr_nonzero = get_ulr_metrics(conn, schema)
    unn_padic, unn_acc, unn_nonzero = get_unn_metrics(conn, schema)
    dt_padic, dt_acc, dt_depth, dt_effective_params = get_dt_metrics(conn, schema)
    zubarev_padic, zubarev_nonzero = get_zubarev_metrics(conn, schema, initialization_method="umllr", mahler_degree=0)
    zubarev_umllr_padic, zubarev_umllr_nonzero = get_zubarev_metrics(conn, schema, initialization_method="umllr", mahler_degree=0)
    zubarev_zeros_padic, zubarev_zeros_nonzero = get_zubarev_metrics(conn, schema, initialization_method="zeros", mahler_degree=0)
    zubarev_umllr_m1_padic, zubarev_umllr_m1_nonzero = get_zubarev_metrics(conn, schema, initialization_method="umllr", mahler_degree=1)
    zubarev_umllr_m2_padic, zubarev_umllr_m2_nonzero = get_zubarev_metrics(conn, schema, initialization_method="umllr", mahler_degree=2)

    # Insert or update snapshot
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                INSERT INTO {schema}.model_performance_history
                (snapshot_date, num_products, num_tags, num_taxonomies,
                 umllr_mean_padic_loss, lr_mean_padic_loss, nn_mean_padic_loss, dummy_mean_padic_loss,
                 umllr_mean_accuracy, lr_mean_accuracy, nn_mean_accuracy, dummy_mean_accuracy,
                 ulr_mean_padic_loss, ulr_mean_accuracy, ulr_mean_nonzero_params,
                 lr_mean_params, nn_mean_input_weights,
                 unn_mean_padic_loss, unn_mean_accuracy, unn_mean_nonzero_params,
                 dt_mean_padic_loss, dt_mean_accuracy, dt_mean_tree_depth,
                 zubarev_mean_padic_loss, zubarev_mean_nonzero_params,
                 zubarev_umllr_mean_padic_loss, zubarev_umllr_mean_nonzero_params,
                 zubarev_zeros_mean_padic_loss, zubarev_zeros_mean_nonzero_params,
                 zubarev_umllr_m1_mean_padic_loss, zubarev_umllr_m1_mean_nonzero_params,
                 zubarev_umllr_m2_mean_padic_loss, zubarev_umllr_m2_mean_nonzero_params,
                 umllr_mean_nonzero_params, dt_mean_effective_params)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (snapshot_date) DO UPDATE SET
                    snapshot_time = now(),
                    num_products = EXCLUDED.num_products,
                    num_tags = EXCLUDED.num_tags,
                    num_taxonomies = EXCLUDED.num_taxonomies,
                    umllr_mean_padic_loss = EXCLUDED.umllr_mean_padic_loss,
                    lr_mean_padic_loss = EXCLUDED.lr_mean_padic_loss,
                    nn_mean_padic_loss = EXCLUDED.nn_mean_padic_loss,
                    dummy_mean_padic_loss = EXCLUDED.dummy_mean_padic_loss,
                    umllr_mean_accuracy = EXCLUDED.umllr_mean_accuracy,
                    lr_mean_accuracy = EXCLUDED.lr_mean_accuracy,
                    nn_mean_accuracy = EXCLUDED.nn_mean_accuracy,
                    dummy_mean_accuracy = EXCLUDED.dummy_mean_accuracy,
                    ulr_mean_padic_loss = EXCLUDED.ulr_mean_padic_loss,
                    ulr_mean_accuracy = EXCLUDED.ulr_mean_accuracy,
                    ulr_mean_nonzero_params = EXCLUDED.ulr_mean_nonzero_params,
                    lr_mean_params = EXCLUDED.lr_mean_params,
                    nn_mean_input_weights = EXCLUDED.nn_mean_input_weights,
                    unn_mean_padic_loss = EXCLUDED.unn_mean_padic_loss,
                    unn_mean_accuracy = EXCLUDED.unn_mean_accuracy,
                    unn_mean_nonzero_params = EXCLUDED.unn_mean_nonzero_params,
                    dt_mean_padic_loss = EXCLUDED.dt_mean_padic_loss,
                    dt_mean_accuracy = EXCLUDED.dt_mean_accuracy,
                    dt_mean_tree_depth = EXCLUDED.dt_mean_tree_depth,
                    zubarev_mean_padic_loss = EXCLUDED.zubarev_mean_padic_loss,
                    zubarev_mean_nonzero_params = EXCLUDED.zubarev_mean_nonzero_params,
                    zubarev_umllr_mean_padic_loss = EXCLUDED.zubarev_umllr_mean_padic_loss,
                    zubarev_umllr_mean_nonzero_params = EXCLUDED.zubarev_umllr_mean_nonzero_params,
                    zubarev_zeros_mean_padic_loss = EXCLUDED.zubarev_zeros_mean_padic_loss,
                    zubarev_zeros_mean_nonzero_params = EXCLUDED.zubarev_zeros_mean_nonzero_params,
                    zubarev_umllr_m1_mean_padic_loss = EXCLUDED.zubarev_umllr_m1_mean_padic_loss,
                    zubarev_umllr_m1_mean_nonzero_params = EXCLUDED.zubarev_umllr_m1_mean_nonzero_params,
                    zubarev_umllr_m2_mean_padic_loss = EXCLUDED.zubarev_umllr_m2_mean_padic_loss,
                    zubarev_umllr_m2_mean_nonzero_params = EXCLUDED.zubarev_umllr_m2_mean_nonzero_params,
                    umllr_mean_nonzero_params = EXCLUDED.umllr_mean_nonzero_params,
                    dt_mean_effective_params = EXCLUDED.dt_mean_effective_params
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
                dummy_padic,
                umllr_acc,
                lr_acc,
                nn_acc,
                dummy_acc,
                ulr_padic,
                ulr_acc,
                ulr_nonzero,
                lr_params,
                nn_input_weights,
                unn_padic,
                unn_acc,
                unn_nonzero,
                dt_padic,
                dt_acc,
                dt_depth,
                zubarev_padic,
                zubarev_nonzero,
                zubarev_umllr_padic,
                zubarev_umllr_nonzero,
                zubarev_zeros_padic,
                zubarev_zeros_nonzero,
                zubarev_umllr_m1_padic,
                zubarev_umllr_m1_nonzero,
                zubarev_umllr_m2_padic,
                zubarev_umllr_m2_nonzero,
                umllr_nonzero,
                dt_effective_params,
            ),
        )
    conn.commit()

    print(f"Snapshot saved for {snap_date}")
    print(f"  Products: {num_products}, Tags: {num_tags}, Taxonomies: {num_taxonomies}")
    print(f"  umllr p-adic loss: {umllr_padic:.6f}" if umllr_padic else "  umllr: no data")
    print(f"  PCLR p-adic loss: {lr_padic:.6f}" if lr_padic else "  PCLR: no data")
    print(f"  PCNN p-adic loss: {nn_padic:.6f}" if nn_padic else "  PCNN: no data")
    print(f"  ULR p-adic loss: {ulr_padic:.6f}" if ulr_padic else "  ULR: no data")
    print(f"  UNN p-adic loss: {unn_padic:.6f}" if unn_padic else "  UNN: no data")
    print(f"  DT p-adic loss: {dt_padic:.6f}" if dt_padic else "  DT: no data")
    print(f"  Zubarev (UMLLR) p-adic loss: {zubarev_umllr_padic:.6f}" if zubarev_umllr_padic else "  Zubarev (UMLLR): no data")
    print(f"  Zubarev (zeros) p-adic loss: {zubarev_zeros_padic:.6f}" if zubarev_zeros_padic else "  Zubarev (zeros): no data")
    print(f"  Zubarev (UMLLR M1) p-adic loss: {zubarev_umllr_m1_padic:.6f}" if zubarev_umllr_m1_padic else "  Zubarev (UMLLR M1): no data")
    print(f"  Zubarev (UMLLR M2) p-adic loss: {zubarev_umllr_m2_padic:.6f}" if zubarev_umllr_m2_padic else "  Zubarev (UMLLR M2): no data")
    print(f"  Dummy p-adic loss: {dummy_padic:.6f}" if dummy_padic else "  Dummy: no data")


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
