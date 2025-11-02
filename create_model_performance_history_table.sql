-- Create model performance history table for tracking trends over time
-- Run this with admin privileges to create table for historical snapshots

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Create model_performance_history table
CREATE TABLE IF NOT EXISTS padjective.model_performance_history (
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
);

-- Grant privileges
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON padjective.model_performance_history TO padjective;
