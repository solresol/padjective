-- Migration: Rename neural network tables to parameter constrained neural network
-- Run this with admin privileges (or as table owner) to rename taxonomy_nn_* tables to taxonomy_pcnn_*

-- Rename main tables
ALTER TABLE IF EXISTS padjective.taxonomy_nn_fold_results RENAME TO taxonomy_pcnn_fold_results;
ALTER TABLE IF EXISTS padjective.taxonomy_nn_predictions RENAME TO taxonomy_pcnn_predictions;
ALTER TABLE IF EXISTS padjective.taxonomy_nn_input_weights RENAME TO taxonomy_pcnn_input_weights;

-- Rename indexes
ALTER INDEX IF EXISTS padjective.taxonomy_nn_fold_results_pkey RENAME TO taxonomy_pcnn_fold_results_pkey;
ALTER INDEX IF EXISTS padjective.taxonomy_nn_predictions_pkey RENAME TO taxonomy_pcnn_predictions_pkey;
ALTER INDEX IF EXISTS padjective.taxonomy_nn_predictions_fold_idx RENAME TO taxonomy_pcnn_predictions_fold_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_nn_input_weights_pkey RENAME TO taxonomy_pcnn_input_weights_pkey;
ALTER INDEX IF EXISTS padjective.taxonomy_nn_input_weights_fold_idx RENAME TO taxonomy_pcnn_input_weights_fold_idx;
ALTER INDEX IF EXISTS padjective.taxonomy_nn_input_weights_tag_idx RENAME TO taxonomy_pcnn_input_weights_tag_idx;

SELECT 'Migration complete: taxonomy_nn_* tables renamed to taxonomy_pcnn_*' AS status;
