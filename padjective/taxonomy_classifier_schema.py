"""Prepare Postgres tables for taxonomy classifier results."""

from __future__ import annotations

import argparse

from psycopg import sql

from . import db


def ensure_taxonomy_classifier_schema(conn, schema: str) -> None:
    """Ensure the logistic taxonomy classifier tables exist."""

    db.ensure_schema(conn, schema)

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_models",
        [
            "id BIGSERIAL PRIMARY KEY",
            "trained_at TIMESTAMPTZ NOT NULL",
            "samples BIGINT NOT NULL",
            "taxonomies INTEGER NOT NULL",
            "unique_tags INTEGER NOT NULL",
            "training_accuracy DOUBLE PRECISION NOT NULL",
            "training_f1 DOUBLE PRECISION",
            "training_hierarchical_loss DOUBLE PRECISION",
            "cv_folds INTEGER",
            "cv_mean_accuracy DOUBLE PRECISION",
            "cv_std_accuracy DOUBLE PRECISION",
            "cv_mean_f1 DOUBLE PRECISION",
            "cv_std_f1 DOUBLE PRECISION",
            "cv_mean_hierarchical_loss DOUBLE PRECISION",
            "cv_std_hierarchical_loss DOUBLE PRECISION",
        ],
    )

    foreign_key = (
        f"FOREIGN KEY (model_id) REFERENCES {schema}.taxonomy_lr_models(id) ON DELETE CASCADE"
    )

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_cv_scores",
        [
            "model_id BIGINT NOT NULL",
            "fold INTEGER NOT NULL",
            "accuracy DOUBLE PRECISION",
            "f1_weighted DOUBLE PRECISION",
            "hierarchical_loss DOUBLE PRECISION",
            foreign_key,
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS taxonomy_lr_cv_scores_model_idx "
                "ON {schema}.taxonomy_lr_cv_scores (model_id)"
            ).format(schema=sql.Identifier(schema)).as_string(conn),
        ],
    )

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_class_distribution",
        [
            "model_id BIGINT NOT NULL",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT",
            "sample_count BIGINT NOT NULL",
            "sample_fraction DOUBLE PRECISION NOT NULL",
            foreign_key,
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS taxonomy_lr_class_distribution_model_idx "
                "ON {schema}.taxonomy_lr_class_distribution (model_id)"
            ).format(schema=sql.Identifier(schema)).as_string(conn),
        ],
    )

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_tag_summary",
        [
            "model_id BIGINT NOT NULL",
            "tag TEXT NOT NULL",
            "top_taxonomy_id TEXT",
            "top_taxonomy_path TEXT",
            "top_weight DOUBLE PRECISION NOT NULL",
            "max_abs_weight DOUBLE PRECISION NOT NULL",
            "sum_abs_weight DOUBLE PRECISION NOT NULL",
            "PRIMARY KEY (model_id, tag)",
            foreign_key,
        ],
    )

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_top_tags",
        [
            "model_id BIGINT NOT NULL",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT",
            "tag TEXT NOT NULL",
            "weight DOUBLE PRECISION NOT NULL",
            "rank INTEGER NOT NULL",
            foreign_key,
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS taxonomy_lr_top_tags_model_idx "
                "ON {schema}.taxonomy_lr_top_tags (model_id)"
            ).format(schema=sql.Identifier(schema)).as_string(conn),
        ],
    )

    db.ensure_table(
        conn,
        schema,
        "taxonomy_lr_intercepts",
        [
            "model_id BIGINT NOT NULL",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT",
            "intercept DOUBLE PRECISION NOT NULL",
            foreign_key,
        ],
        indexes_sql=[
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS taxonomy_lr_intercepts_model_idx "
                "ON {schema}.taxonomy_lr_intercepts (model_id)"
            ).format(schema=sql.Identifier(schema)).as_string(conn),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure Postgres tables exist for taxonomy classifier results",
    )
    parser.add_argument(
        "--dsn",
        help="Postgres DSN (uses SHOPIFY_DB_DSN or DATABASE_URL if omitted)",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema to prepare",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        ensure_taxonomy_classifier_schema(conn, args.schema)
    finally:
        conn.close()

    print(
        f"Prepared taxonomy classifier tables in schema {args.schema}.",
    )


if __name__ == "__main__":
    main()
