-- Create taxonomy parameter constrained logistic regression fold results tables
-- Run this with admin privileges to create tables for per-fold metrics

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Create taxonomy_pclr_fold_results table
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_fold_results (
    cv_fold INTEGER PRIMARY KEY,
    test_accuracy DOUBLE PRECISION NOT NULL,
    test_f1 DOUBLE PRECISION NOT NULL,
    test_hierarchical_loss DOUBLE PRECISION NOT NULL,
    padic_loss_total DOUBLE PRECISION NOT NULL,
    padic_loss_mean DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    num_train_samples INTEGER NOT NULL,
    num_test_samples INTEGER NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Grant privileges
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_pclr_fold_results TO padjective;
