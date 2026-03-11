-- Create UMLLR output tables in padjective schema
-- Run this with admin privileges to create tables for the padjective user

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Create umllr_tag_coefficients table
CREATE TABLE IF NOT EXISTS padjective.umllr_tag_coefficients (
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    coefficient NUMERIC NOT NULL,
    sequence INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, tag)
);

-- Create index for umllr_tag_coefficients
CREATE INDEX IF NOT EXISTS padjective_umllr_coeff_sequence_idx
    ON padjective.umllr_tag_coefficients (sequence);

-- Create umllr_fold_metrics table
CREATE TABLE IF NOT EXISTS padjective.umllr_fold_metrics (
    cv_fold INTEGER PRIMARY KEY,
    loss DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    max_digit INTEGER NOT NULL,
    default_prediction NUMERIC,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE padjective.umllr_fold_metrics
    ADD COLUMN IF NOT EXISTS exact_accuracy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS prefix1_accuracy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS prefix2_accuracy DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mean_shared_prefix_depth DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS mean_scoring_ops DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS tag_order_strategy TEXT DEFAULT 'battle_elo',
    ADD COLUMN IF NOT EXISTS tag_order_seed INTEGER;

-- Create umllr_predictions table
CREATE TABLE IF NOT EXISTS padjective.umllr_predictions (
    cv_fold INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    true_value NUMERIC NOT NULL,
    predicted_value NUMERIC NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, product_id)
);

-- Create index for umllr_predictions
CREATE INDEX IF NOT EXISTS padjective_umllr_predictions_fold_idx
    ON padjective.umllr_predictions (cv_fold);

-- Create umllr_taxonomy_encodings table
CREATE TABLE IF NOT EXISTS padjective.umllr_taxonomy_encodings (
    cv_fold INTEGER NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT NOT NULL,
    encoded_value NUMERIC NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, taxonomy_id)
);

-- Create index for umllr_taxonomy_encodings
CREATE INDEX IF NOT EXISTS padjective_umllr_taxonomy_encodings_fold_idx
    ON padjective.umllr_taxonomy_encodings (cv_fold);

-- Create dummy_fold_metrics table for baseline classifier
CREATE TABLE IF NOT EXISTS padjective.dummy_fold_metrics (
    cv_fold INTEGER PRIMARY KEY,
    loss DOUBLE PRECISION NOT NULL,
    accuracy DOUBLE PRECISION NOT NULL,
    most_common_value NUMERIC NOT NULL,
    most_common_taxonomy_id TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create dummy_predictions table
CREATE TABLE IF NOT EXISTS padjective.dummy_predictions (
    cv_fold INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    true_value NUMERIC NOT NULL,
    predicted_value NUMERIC NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, product_id)
);

-- Create index for dummy_predictions
CREATE INDEX IF NOT EXISTS padjective_dummy_predictions_fold_idx
    ON padjective.dummy_predictions (cv_fold);

-- Create umllr_coefficient_candidates table for debugging coefficient selection
CREATE TABLE IF NOT EXISTS padjective.umllr_coefficient_candidates (
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    candidate_value BIGINT NOT NULL,
    total_loss DOUBLE PRECISION NOT NULL,
    product_count INTEGER NOT NULL,
    was_selected BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (cv_fold, tag, candidate_value)
);

-- Create index for umllr_coefficient_candidates
CREATE INDEX IF NOT EXISTS padjective_umllr_coeff_candidates_fold_tag_idx
    ON padjective.umllr_coefficient_candidates (cv_fold, tag);

-- Create umllr_tag_products table for tracking residuals per product per tag
CREATE TABLE IF NOT EXISTS padjective.umllr_tag_products (
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    product_id BIGINT NOT NULL,
    residual_before BIGINT NOT NULL,
    residual_after BIGINT NOT NULL,
    PRIMARY KEY (cv_fold, tag, product_id)
);

-- Create index for umllr_tag_products
CREATE INDEX IF NOT EXISTS padjective_umllr_tag_products_fold_tag_idx
    ON padjective.umllr_tag_products (cv_fold, tag);

-- Create umllr_order_ablation_fold_metrics table
CREATE TABLE IF NOT EXISTS padjective.umllr_order_ablation_fold_metrics (
    run_key TEXT NOT NULL,
    tag_order_strategy TEXT NOT NULL,
    tag_order_seed INTEGER,
    cv_fold INTEGER NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    max_digit INTEGER NOT NULL,
    default_prediction NUMERIC,
    exact_accuracy DOUBLE PRECISION,
    prefix1_accuracy DOUBLE PRECISION,
    prefix2_accuracy DOUBLE PRECISION,
    mean_shared_prefix_depth DOUBLE PRECISION,
    mean_scoring_ops DOUBLE PRECISION,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_key, cv_fold)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS padjective_umllr_ablation_strategy_idx
    ON padjective.umllr_order_ablation_fold_metrics (tag_order_strategy, tag_order_seed) TABLESPACE pg_default;

-- Create umllr_order_ablation_predictions table
CREATE TABLE IF NOT EXISTS padjective.umllr_order_ablation_predictions (
    run_key TEXT NOT NULL,
    tag_order_strategy TEXT NOT NULL,
    tag_order_seed INTEGER,
    cv_fold INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    true_value NUMERIC NOT NULL,
    predicted_value NUMERIC NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_key, cv_fold, product_id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS padjective_umllr_ablation_predictions_strategy_idx
    ON padjective.umllr_order_ablation_predictions (tag_order_strategy, tag_order_seed, cv_fold) TABLESPACE pg_default;

-- Grant all privileges on schema and tables to padjective user
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA padjective TO padjective;

-- Set default privileges for future tables in padjective schema
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON TABLES TO padjective;
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON SEQUENCES TO padjective;
