-- Create taxonomy neural network fold results tables
-- Run this with admin privileges to create tables for per-fold NN metrics

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Create taxonomy_nn_fold_results table
CREATE TABLE IF NOT EXISTS padjective.taxonomy_nn_fold_results (
    cv_fold INTEGER PRIMARY KEY,
    test_accuracy REAL NOT NULL,
    test_f1 REAL NOT NULL,
    test_hierarchical_loss REAL NOT NULL,
    padic_loss_total REAL NOT NULL,
    padic_loss_mean REAL NOT NULL,
    prime_base INTEGER NOT NULL,
    num_train_samples INTEGER NOT NULL,
    num_test_samples INTEGER NOT NULL,
    hidden_layers TEXT NOT NULL,
    max_tags INTEGER
);

-- Grant privileges
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON padjective.taxonomy_nn_fold_results TO padjective;
