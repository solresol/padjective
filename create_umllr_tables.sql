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

-- Grant all privileges on schema and tables to padjective user
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA padjective TO padjective;

-- Set default privileges for future tables in padjective schema
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON TABLES TO padjective;
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON SEQUENCES TO padjective;
