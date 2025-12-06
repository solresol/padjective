-- Execute these statements as a superuser or a role with privileges to create
-- schemas, tables, indexes, and foreign keys.

-- 1. Ensure the padjective schema exists and is owned by the current user.
CREATE SCHEMA IF NOT EXISTS padjective AUTHORIZATION CURRENT_USER;

-- 2. Create the taxonomy_pclr_models table to record trained parameter constrained logistic regression models.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_models (
    id BIGSERIAL PRIMARY KEY,
    trained_at TIMESTAMPTZ NOT NULL,
    samples BIGINT NOT NULL,
    taxonomies INTEGER NOT NULL,
    unique_tags INTEGER NOT NULL,
    training_accuracy DOUBLE PRECISION NOT NULL,
    training_f1 DOUBLE PRECISION,
    training_hierarchical_loss DOUBLE PRECISION,
    cv_folds INTEGER,
    cv_mean_accuracy DOUBLE PRECISION,
    cv_std_accuracy DOUBLE PRECISION,
    cv_mean_f1 DOUBLE PRECISION,
    cv_std_f1 DOUBLE PRECISION,
    cv_mean_hierarchical_loss DOUBLE PRECISION,
    cv_std_hierarchical_loss DOUBLE PRECISION
);

-- 3. Create the taxonomy_pclr_cv_scores table to store per-fold cross-validation metrics.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_cv_scores (
    model_id BIGINT NOT NULL,
    fold INTEGER NOT NULL,
    accuracy DOUBLE PRECISION,
    f1_weighted DOUBLE PRECISION,
    hierarchical_loss DOUBLE PRECISION,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_pclr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_cv_scores_model_idx
    ON padjective.taxonomy_pclr_cv_scores (model_id);

-- 4. Create the taxonomy_pclr_class_distribution table for class balance diagnostics.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_class_distribution (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    sample_count BIGINT NOT NULL,
    sample_fraction DOUBLE PRECISION NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_pclr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_class_distribution_model_idx
    ON padjective.taxonomy_pclr_class_distribution (model_id);

-- 5. Create the taxonomy_pclr_tag_summary table summarising tag influence.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_tag_summary (
    model_id BIGINT NOT NULL,
    tag TEXT NOT NULL,
    top_taxonomy_id TEXT,
    top_taxonomy_path TEXT,
    top_weight DOUBLE PRECISION NOT NULL,
    max_abs_weight DOUBLE PRECISION NOT NULL,
    sum_abs_weight DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model_id, tag),
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_pclr_models(id) ON DELETE CASCADE
);

-- 6. Create the taxonomy_pclr_top_tags table that records highest-weighted tags per taxonomy.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_top_tags (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    tag TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_pclr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_top_tags_model_idx
    ON padjective.taxonomy_pclr_top_tags (model_id);

-- 7. Create the taxonomy_pclr_intercepts table storing per-class intercepts.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_intercepts (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    intercept DOUBLE PRECISION NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_pclr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_intercepts_model_idx
    ON padjective.taxonomy_pclr_intercepts (model_id);

-- 8. Create the taxonomy_pclr_predictions table for individual test predictions per fold.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_predictions (
    cv_fold INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    true_taxonomy_id TEXT NOT NULL,
    predicted_taxonomy_id TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, product_id)
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_predictions_fold_idx
    ON padjective.taxonomy_pclr_predictions (cv_fold);

-- 9. Create the taxonomy_pclr_coefficients table storing tag coefficients per taxonomy per fold.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pclr_coefficients (
    cv_fold INTEGER NOT NULL,
    taxonomy_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    coefficient DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, taxonomy_id, tag)
);

CREATE INDEX IF NOT EXISTS taxonomy_pclr_coefficients_fold_idx
    ON padjective.taxonomy_pclr_coefficients (cv_fold);

-- 10. Create the taxonomy_pcnn_predictions table for individual test predictions per fold.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pcnn_predictions (
    cv_fold INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    true_taxonomy_id TEXT NOT NULL,
    predicted_taxonomy_id TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, product_id)
);

CREATE INDEX IF NOT EXISTS taxonomy_pcnn_predictions_fold_idx
    ON padjective.taxonomy_pcnn_predictions (cv_fold);

-- 11. Create the taxonomy_pcnn_input_weights table storing first-layer weights per fold.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_pcnn_input_weights (
    cv_fold INTEGER NOT NULL,
    tag TEXT NOT NULL,
    hidden_unit INTEGER NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, tag, hidden_unit)
);

CREATE INDEX IF NOT EXISTS taxonomy_pcnn_input_weights_fold_idx
    ON padjective.taxonomy_pcnn_input_weights (cv_fold);

CREATE INDEX IF NOT EXISTS taxonomy_pcnn_input_weights_tag_idx
    ON padjective.taxonomy_pcnn_input_weights (tag);

-- 12. Create the taxonomy_ulr_fold_results table for unconstrained L1-regularized LR.
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

-- 13. Create the taxonomy_ulr_predictions table for individual test predictions per fold.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_ulr_predictions (
    cv_fold INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    true_taxonomy_id TEXT NOT NULL,
    predicted_taxonomy_id TEXT NOT NULL,
    loss DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (cv_fold, product_id)
);

CREATE INDEX IF NOT EXISTS taxonomy_ulr_predictions_fold_idx
    ON padjective.taxonomy_ulr_predictions (cv_fold);

-- 14. Grant padjective role access to the schema, tables, and sequences.
GRANT USAGE ON SCHEMA padjective TO padjective;

GRANT SELECT, INSERT, UPDATE, DELETE ON
    padjective.taxonomy_pclr_models,
    padjective.taxonomy_pclr_cv_scores,
    padjective.taxonomy_pclr_class_distribution,
    padjective.taxonomy_pclr_tag_summary,
    padjective.taxonomy_pclr_top_tags,
    padjective.taxonomy_pclr_intercepts,
    padjective.taxonomy_pclr_predictions,
    padjective.taxonomy_pclr_coefficients,
    padjective.taxonomy_pcnn_predictions,
    padjective.taxonomy_pcnn_input_weights,
    padjective.taxonomy_ulr_fold_results,
    padjective.taxonomy_ulr_predictions
TO padjective;

GRANT USAGE, SELECT ON SEQUENCE padjective.taxonomy_pclr_models_id_seq TO padjective;
