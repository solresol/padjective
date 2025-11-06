-- Migration script to add default prediction support and dummy classifier tables
-- Run this with admin privileges (e.g., as user gregb)
--
-- Usage: psql -U gregb -f migrate_umllr_dummy_classifier.sql

-- Add default_prediction column to umllr_fold_metrics
ALTER TABLE padjective.umllr_fold_metrics
ADD COLUMN IF NOT EXISTS default_prediction NUMERIC;

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

-- Grant privileges to padjective user
GRANT SELECT, INSERT, UPDATE, DELETE ON
    padjective.dummy_fold_metrics,
    padjective.dummy_predictions
TO padjective;
