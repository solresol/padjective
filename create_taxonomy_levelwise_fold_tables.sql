-- Create taxonomy level-wise logistic regression fold results tables
-- Run this with admin privileges to create tables for per-fold metrics

CREATE SCHEMA IF NOT EXISTS padjective;

CREATE TABLE IF NOT EXISTS padjective.taxonomy_levelwise_fold_results (
    cv_fold INTEGER PRIMARY KEY,
    test_accuracy DOUBLE PRECISION NOT NULL,
    test_f1 DOUBLE PRECISION NOT NULL,
    test_hierarchical_loss DOUBLE PRECISION NOT NULL,
    padic_loss_total DOUBLE PRECISION NOT NULL,
    padic_loss_mean DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    num_train_samples INTEGER NOT NULL,
    num_test_samples INTEGER NOT NULL,
    num_nodes INTEGER NOT NULL,
    num_classifiers INTEGER NOT NULL,
    exact_accuracy DOUBLE PRECISION NOT NULL,
    prefix1_accuracy DOUBLE PRECISION NOT NULL,
    prefix2_accuracy DOUBLE PRECISION NOT NULL,
    mean_shared_prefix_depth DOUBLE PRECISION NOT NULL,
    mean_scoring_ops DOUBLE PRECISION NOT NULL,
    trained_at TIMESTAMPTZ NOT NULL DEFAULT now()
) TABLESPACE pg_default;

CREATE TABLE IF NOT EXISTS padjective.taxonomy_levelwise_predictions (
    cv_fold INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    true_taxonomy_id TEXT NOT NULL,
    predicted_taxonomy_id TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, product_id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS taxonomy_levelwise_predictions_fold_idx
    ON padjective.taxonomy_levelwise_predictions(cv_fold) TABLESPACE pg_default;

GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_levelwise_fold_results TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_levelwise_predictions TO padjective;
