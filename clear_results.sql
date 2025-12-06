-- Clear all model results to restart from today
-- This preserves core data (products, taxonomy) but clears all computed results

-- UMLLR results
TRUNCATE TABLE padjective.umllr_fold_metrics CASCADE;
TRUNCATE TABLE padjective.umllr_tag_coefficients CASCADE;
TRUNCATE TABLE padjective.umllr_predictions CASCADE;
TRUNCATE TABLE padjective.umllr_taxonomy_encodings CASCADE;

-- Dummy classifier results
TRUNCATE TABLE padjective.dummy_fold_metrics CASCADE;
TRUNCATE TABLE padjective.dummy_predictions CASCADE;

-- Taxonomy parameter constrained logistic regression results
TRUNCATE TABLE padjective.taxonomy_pclr_models CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_cv_scores CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_class_distribution CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_tag_summary CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_top_tags CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_intercepts CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_predictions CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_coefficients CASCADE;
TRUNCATE TABLE padjective.taxonomy_pclr_fold_results CASCADE;

-- Parameter constrained neural network results
TRUNCATE TABLE padjective.taxonomy_pcnn_fold_results CASCADE;
TRUNCATE TABLE padjective.taxonomy_pcnn_predictions CASCADE;

-- Tag battles
TRUNCATE TABLE padjective.battles CASCADE;

-- Tag rankings (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'padjective'
               AND table_name = 'tag_rankings') THEN
        TRUNCATE TABLE padjective.tag_rankings CASCADE;
    END IF;
END $$;

-- Historical snapshots (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'padjective'
               AND table_name = 'historical_metrics') THEN
        TRUNCATE TABLE padjective.historical_metrics CASCADE;
    END IF;
END $$;

-- UMLLR predictions tracking (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'padjective'
               AND table_name = 'umllr_predictions') THEN
        TRUNCATE TABLE padjective.umllr_predictions CASCADE;
    END IF;
END $$;

SELECT 'All model results have been cleared. Ready to restart from today.' AS status;
