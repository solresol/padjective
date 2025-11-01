-- Execute these statements as a superuser or a role with privileges to create
-- schemas, tables, indexes, and foreign keys.

-- 1. Ensure the padjective schema exists and is owned by the current user.
CREATE SCHEMA IF NOT EXISTS padjective AUTHORIZATION CURRENT_USER;

-- 2. Create the taxonomy_lr_models table to record trained logistic regression models.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_models (
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

-- 3. Create the taxonomy_lr_cv_scores table to store per-fold cross-validation metrics.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_cv_scores (
    model_id BIGINT NOT NULL,
    fold INTEGER NOT NULL,
    accuracy DOUBLE PRECISION,
    f1_weighted DOUBLE PRECISION,
    hierarchical_loss DOUBLE PRECISION,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_lr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_lr_cv_scores_model_idx
    ON padjective.taxonomy_lr_cv_scores (model_id);

-- 4. Create the taxonomy_lr_class_distribution table for class balance diagnostics.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_class_distribution (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    sample_count BIGINT NOT NULL,
    sample_fraction DOUBLE PRECISION NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_lr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_lr_class_distribution_model_idx
    ON padjective.taxonomy_lr_class_distribution (model_id);

-- 5. Create the taxonomy_lr_tag_summary table summarising tag influence.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_tag_summary (
    model_id BIGINT NOT NULL,
    tag TEXT NOT NULL,
    top_taxonomy_id TEXT,
    top_taxonomy_path TEXT,
    top_weight DOUBLE PRECISION NOT NULL,
    max_abs_weight DOUBLE PRECISION NOT NULL,
    sum_abs_weight DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (model_id, tag),
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_lr_models(id) ON DELETE CASCADE
);

-- 6. Create the taxonomy_lr_top_tags table that records highest-weighted tags per taxonomy.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_top_tags (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    tag TEXT NOT NULL,
    weight DOUBLE PRECISION NOT NULL,
    rank INTEGER NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_lr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_lr_top_tags_model_idx
    ON padjective.taxonomy_lr_top_tags (model_id);

-- 7. Create the taxonomy_lr_intercepts table storing per-class intercepts.
CREATE TABLE IF NOT EXISTS padjective.taxonomy_lr_intercepts (
    model_id BIGINT NOT NULL,
    taxonomy_id TEXT NOT NULL,
    taxonomy_path TEXT,
    intercept DOUBLE PRECISION NOT NULL,
    FOREIGN KEY (model_id) REFERENCES padjective.taxonomy_lr_models(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS taxonomy_lr_intercepts_model_idx
    ON padjective.taxonomy_lr_intercepts (model_id);

-- 8. Grant padjective role access to the schema, tables, and sequences.
GRANT USAGE ON SCHEMA padjective TO padjective;

GRANT SELECT, INSERT, UPDATE, DELETE ON
    padjective.taxonomy_lr_models,
    padjective.taxonomy_lr_cv_scores,
    padjective.taxonomy_lr_class_distribution,
    padjective.taxonomy_lr_tag_summary,
    padjective.taxonomy_lr_top_tags,
    padjective.taxonomy_lr_intercepts
TO padjective;

GRANT USAGE, SELECT ON SEQUENCE padjective.taxonomy_lr_models_id_seq TO padjective;
