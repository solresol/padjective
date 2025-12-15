-- Create Zubarev output tables in padjective schema
-- Run this with admin privileges to create tables for the padjective user
--
-- Zubarev's method: p-adic polynomial regression with Mahler basis
-- From arXiv:2503.23488

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS padjective;

-- Use default tablespace if custom one isn't available
SET default_tablespace = 'pg_default';

-- Create zubarev_tag_coefficients table
CREATE TABLE IF NOT EXISTS padjective.zubarev_tag_coefficients (
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    coefficient NUMERIC NOT NULL,
    sequence INTEGER NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, tag)
);

-- Create index for zubarev_tag_coefficients
CREATE INDEX IF NOT EXISTS padjective_zubarev_coeff_sequence_idx
    ON padjective.zubarev_tag_coefficients (sequence);

-- Create zubarev_fold_metrics table
CREATE TABLE IF NOT EXISTS padjective.zubarev_fold_metrics (
    cv_fold INTEGER PRIMARY KEY,
    loss DOUBLE PRECISION NOT NULL,
    prime_base INTEGER NOT NULL,
    max_digit INTEGER NOT NULL,
    default_prediction NUMERIC,
    iterations_used INTEGER NOT NULL,
    final_temperature DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Create zubarev_predictions table
CREATE TABLE IF NOT EXISTS padjective.zubarev_predictions (
    cv_fold INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    true_value NUMERIC NOT NULL,
    predicted_value NUMERIC NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, product_id)
);

-- Create index for zubarev_predictions
CREATE INDEX IF NOT EXISTS padjective_zubarev_predictions_fold_idx
    ON padjective.zubarev_predictions (cv_fold);

-- Create zubarev_mahler_weights table for the polynomial basis weights
-- Stores w_k coefficients for f(s) = sum_k w_k * C(s, k)
CREATE TABLE IF NOT EXISTS padjective.zubarev_mahler_weights (
    cv_fold INTEGER NOT NULL,
    k INTEGER NOT NULL,  -- Polynomial degree (0, 1, 2, ...)
    weight NUMERIC NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (cv_fold, k)
);

-- Grant all privileges on schema and tables to padjective user
GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA padjective TO padjective;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA padjective TO padjective;

-- Set default privileges for future tables in padjective schema
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON TABLES TO padjective;
ALTER DEFAULT PRIVILEGES IN SCHEMA padjective
    GRANT ALL PRIVILEGES ON SEQUENCES TO padjective;
