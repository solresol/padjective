-- Append-only result tables for the bounded Mihara taxonomy comparison.
--
-- The Python runner can create these objects itself.  This script is the
-- administrator-facing equivalent for hosts where the padjective role cannot
-- create schema objects.

CREATE SCHEMA IF NOT EXISTS padjective;
SET default_tablespace = 'pg_default';

CREATE TABLE IF NOT EXISTS padjective.taxonomy_mihara_runs (
    run_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL,
    snapshot_ref TEXT NOT NULL,
    product_table TEXT NOT NULL,
    prime_base INTEGER NOT NULL,
    precision_requested INTEGER NOT NULL,
    max_tags INTEGER NOT NULL,
    feature_selection TEXT NOT NULL,
    trials INTEGER NOT NULL,
    seed BIGINT NOT NULL,
    acceptance_threshold DOUBLE PRECISION NOT NULL,
    product_count INTEGER NOT NULL,
    taxonomy_count INTEGER NOT NULL,
    available_tag_count INTEGER NOT NULL
) TABLESPACE pg_default;

CREATE TABLE IF NOT EXISTS padjective.taxonomy_mihara_fold_results (
    run_id UUID NOT NULL,
    cv_fold INTEGER NOT NULL,
    train_samples INTEGER NOT NULL,
    test_samples INTEGER NOT NULL,
    selected_tag_count INTEGER NOT NULL,
    nonzero_parameters INTEGER NOT NULL,
    digits_fitted INTEGER NOT NULL,
    accepted_prefix_digits INTEGER NOT NULL,
    all_digits_accepted BOOLEAN NOT NULL,
    stop_reason TEXT NOT NULL,
    total_loss DOUBLE PRECISION NOT NULL,
    mean_loss DOUBLE PRECISION NOT NULL,
    exact_accuracy DOUBLE PRECISION NOT NULL,
    prefix1_accuracy DOUBLE PRECISION NOT NULL,
    prefix2_accuracy DOUBLE PRECISION NOT NULL,
    mean_shared_prefix_depth DOUBLE PRECISION NOT NULL,
    mean_scoring_ops DOUBLE PRECISION NOT NULL,
    baseline_mean_loss DOUBLE PRECISION,
    loss_delta_vs_umllr DOUBLE PRECISION,
    digit_diagnostics JSONB NOT NULL,
    PRIMARY KEY (run_id, cv_fold)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS padjective_taxonomy_mihara_fold_idx
    ON padjective.taxonomy_mihara_fold_results (cv_fold)
    TABLESPACE pg_default;

CREATE TABLE IF NOT EXISTS padjective.taxonomy_mihara_coefficients (
    run_id UUID NOT NULL,
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    coefficient NUMERIC NOT NULL,
    sequence INTEGER NOT NULL,
    is_intercept BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, cv_fold, tag)
) TABLESPACE pg_default;

CREATE TABLE IF NOT EXISTS padjective.taxonomy_mihara_predictions (
    run_id UUID NOT NULL,
    cv_fold INTEGER NOT NULL,
    product_id BIGINT NOT NULL,
    true_value NUMERIC NOT NULL,
    predicted_value NUMERIC NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, cv_fold, product_id)
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS padjective_taxonomy_mihara_prediction_product_idx
    ON padjective.taxonomy_mihara_predictions (product_id)
    TABLESPACE pg_default;

GRANT USAGE ON SCHEMA padjective TO padjective;
GRANT SELECT, INSERT, UPDATE, DELETE ON
    padjective.taxonomy_mihara_runs,
    padjective.taxonomy_mihara_fold_results,
    padjective.taxonomy_mihara_coefficients,
    padjective.taxonomy_mihara_predictions
TO padjective;
