-- Create taxonomy unconstrained logistic regression fold results tables
-- Run this with admin privileges to create tables for per-fold metrics

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Create taxonomy_ulr_fold_results table
-- This table tracks L1-regularized logistic regression with ALL tags (unconstrained)
CREATE TABLE IF NOT EXISTS padjective.taxonomy_ulr_fold_results (
    cv_fold INTEGER PRIMARY KEY,
    test_accuracy DOUBLE PRECISION NOT NULL,
    test_f1 DOUBLE PRECISION NOT NULL,
    test_hierarchical_loss DOUBLE PRECISION NOT NULL,
    padic_loss_total DOUBLE PRECISION NOT NULL,
    padic_loss_mean DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    num_train_samples INTEGER NOT NULL,
    num_test_samples INTEGER NOT NULL,
    num_tags INTEGER NOT NULL,
    num_nonzero_params INTEGER NOT NULL,
    num_total_params INTEGER NOT NULL,
    l1_C DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create taxonomy_ulr_predictions table for individual predictions
CREATE TABLE IF NOT EXISTS padjective.taxonomy_ulr_predictions (
    cv_fold INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    true_taxonomy_id TEXT NOT NULL,
    predicted_taxonomy_id TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, product_id)
);

-- Create index on fold for efficient per-fold queries
CREATE INDEX IF NOT EXISTS taxonomy_ulr_predictions_fold_idx
    ON padjective.taxonomy_ulr_predictions(cv_fold);

-- Grant privileges
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_ulr_fold_results TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_ulr_predictions TO padjective;
