-- Migration: Rename logistic regression tables to parameter constrained logistic regression
-- Run this with admin privileges (or as table owner) to rename taxonomy_lr_* tables to taxonomy_pclr_*
--
-- NOTE: Some tables may be owned by different users. Tables owned by 'padjective' role
-- can be renamed by that role, but tables owned by 'gregb' require admin access.
-- The following tables need admin access to rename:
--   - taxonomy_lr_models (owned by gregb)
--   - taxonomy_lr_cv_scores (owned by gregb)
--   - taxonomy_lr_class_distribution (owned by gregb)
--   - taxonomy_lr_tag_summary (owned by gregb)
--   - taxonomy_lr_top_tags (owned by gregb)
--   - taxonomy_lr_intercepts (owned by gregb)

-- Rename main tables
ALTER TABLE IF EXISTS padjective.taxonomy_lr_models RENAME TO taxonomy_pclr_models;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_cv_scores RENAME TO taxonomy_pclr_cv_scores;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_class_distribution RENAME TO taxonomy_pclr_class_distribution;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_tag_summary RENAME TO taxonomy_pclr_tag_summary;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_top_tags RENAME TO taxonomy_pclr_top_tags;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_intercepts RENAME TO taxonomy_pclr_intercepts;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_predictions RENAME TO taxonomy_pclr_predictions;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_coefficients RENAME TO taxonomy_pclr_coefficients;
ALTER TABLE IF EXISTS padjective.taxonomy_lr_fold_results RENAME TO taxonomy_pclr_fold_results;

-- Rename indexes
ALTER INDEX IF EXISTS padjective.taxonomy_lr_cv_scores_model_idx RENAME TO taxonomy_pclr_cv_scores_model_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_lr_class_distribution_model_idx RENAME TO taxonomy_pclr_class_distribution_model_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_lr_top_tags_model_idx RENAME TO taxonomy_pclr_top_tags_model_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_lr_intercepts_model_idx RENAME TO taxonomy_pclr_intercepts_model_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_lr_predictions_fold_idx RENAME TO taxonomy_pclr_predictions_fold_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_lr_coefficients_fold_idx RENAME TO taxonomy_pclr_coefficients_fold_idx;

-- Rename sequence
ALTER SEQUENCE IF EXISTS padjective.taxonomy_lr_models_id_seq RENAME TO taxonomy_pclr_models_id_seq;

SELECT 'Migration complete: taxonomy_lr_* tables renamed to taxonomy_pclr_*' AS status;
