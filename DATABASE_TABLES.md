# Padjective Database Tables

This document lists all database tables in the `padjective` schema and how they are created.

## Table Creation Methods

All tables are created automatically by the Python programs when they run, using `CREATE TABLE IF NOT EXISTS`. However, SQL scripts are provided for:
- Documentation
- Manual admin setup
- Explicit permission grants

## Tables and Their Creation

### 1. Tag Battle Tables

**SQL Script:** `create_umllr_tables.sql` (or via `padjective/sql/tagbattles.sql`)

- `tag_battles` - Records of tag pair comparisons from product titles

### 2. umllr (Universal Machine Learning Linear Regression) Tables

**SQL Script:** `create_umllr_tables.sql`  
**Created by:** `padjective/umllr.py`

- `umllr_tag_coefficients` - P-adic coefficients assigned to each tag per fold
- `umllr_fold_metrics` - Per-fold loss plus exact/prefix accuracy, shared-prefix depth, scoring ops, and tag-order metadata
- `umllr_predictions` - Individual product predictions with p-adic loss
- `umllr_taxonomy_encodings` - P-adic encodings of taxonomy paths per fold
- `umllr_order_ablation_fold_metrics` - Per-fold results for tag-order ablation runs, tagged by snapshot (`live`, `paper`, etc.)
- `umllr_order_ablation_predictions` - Product-level predictions for tag-order ablation runs, tagged by snapshot (`live`, `paper`, etc.)

### 3. Parameter Constrained Logistic Regression Classifier Tables

**SQL Script:** `create_taxonomy_pclr_fold_tables.sql`
**Created by:** `padjective/taxonomy_classifier.py`

- `taxonomy_pclr_fold_results` - Per-fold test metrics (accuracy, F1, hierarchical loss, p-adic loss)
- `taxonomy_pclr_models` - Full training metadata and cross-validation results

**Schema Creation:** `padjective/taxonomy_classifier_schema.py`

### 4. Parameter Constrained Neural Network Classifier Tables

**SQL Script:** `create_taxonomy_pcnn_fold_tables.sql`
**Created by:** `padjective/taxonomy_pcnn_classifier.py`

- `taxonomy_pcnn_fold_results` - Per-fold test metrics (accuracy, F1, hierarchical loss, p-adic loss)
- `taxonomy_pcnn_models` - Full training metadata (stored in SQLite at `data/taxonomy_pcnn_classifier.sqlite`)
- `taxonomy_pcnn_cv_scores` - Cross-validation scores (stored in SQLite)

### 5. Unconstrained Logistic Regression (ULR) Tables

**SQL Script:** `create_taxonomy_ulr_fold_tables.sql`
**Created by:** `padjective/taxonomy_ulr_classifier.py`

- `taxonomy_ulr_fold_results` - Per-fold test metrics including non-zero parameter counts
- `taxonomy_ulr_predictions` - Individual product predictions with p-adic loss

Unlike PCLR and PCNN, ULR uses ALL available tags with L1 regularization to achieve sparsity. The `num_nonzero_params` column tracks how many parameters remain non-zero after L1 shrinkage.

### 6. Level-wise Logistic Regression Tables

**SQL Script:** `create_taxonomy_levelwise_fold_tables.sql`
**Created by:** `padjective/taxonomy_levelwise_classifier.py`

- `taxonomy_levelwise_fold_results` - Per-fold hierarchy-aware metrics, p-adic loss, non-zero parameter counts, and scoring ops
- `taxonomy_levelwise_predictions` - Individual product predictions from the top-down level-wise baseline

This baseline trains one logistic decision per internal taxonomy node and always emits a valid taxonomy path from the training-fold tree.

### 7. Mihara Digitwise Taxonomy Comparison Tables

**SQL Script:** `create_taxonomy_mihara_tables.sql`
**Created by:** `padjective/taxonomy_mihara_comparison.py`

- `taxonomy_mihara_runs` - Append-only experiment configuration and corpus metadata
- `taxonomy_mihara_fold_results` - Held-out metrics, matched UMLLR deltas, and per-digit acceptance diagnostics
- `taxonomy_mihara_coefficients` - Fold-local tag coefficients and affine intercepts
- `taxonomy_mihara_predictions` - Held-out raw p-adic predictions and losses

### 8. Historical Performance Tracking Tables

**SQL Script:** `create_model_performance_history_table.sql`
**Created by:** `padjective/snapshot_metrics.py`

- `model_performance_history` - Daily snapshots of all model metrics and dataset statistics
  - Includes ULR metrics: `ulr_mean_padic_loss`, `ulr_mean_accuracy`, `ulr_mean_nonzero_params`

## Running the Scripts

To manually create all tables with proper permissions:

```bash
# As admin/superuser
psql -f create_umllr_tables.sql
psql -f create_taxonomy_pclr_fold_tables.sql
psql -f create_taxonomy_pcnn_fold_tables.sql
psql -f create_taxonomy_ulr_fold_tables.sql
psql -f create_taxonomy_levelwise_fold_tables.sql
psql -f create_taxonomy_mihara_tables.sql
psql -f create_model_performance_history_table.sql
```

## Table Cleanup

The following tables are cleaned (old data deleted) before each cronscript run:

- `umllr_*` tables - Cleaned by `umllr.py` before inserting new fold data
- `taxonomy_pclr_fold_results` - Per-fold cleanup by `taxonomy_classifier.py`
- `taxonomy_pcnn_fold_results` - Per-fold cleanup by `taxonomy_pcnn_classifier.py`
- `taxonomy_ulr_fold_results` - Per-fold cleanup by `taxonomy_ulr_classifier.py`
- `taxonomy_levelwise_*` tables - Per-fold cleanup by `taxonomy_levelwise_classifier.py`

The following tables accumulate data over time (no cleanup):

- `model_performance_history` - Uses `ON CONFLICT DO UPDATE` to update existing dates
- `taxonomy_mihara_*` - Append-only experimental runs keyed by `run_id`

## Verification

To verify all tables exist:

```bash
psql -c "\dt padjective.*"
```

To check current snapshot history:

```bash
psql -c "SELECT * FROM padjective.model_performance_history ORDER BY snapshot_date DESC LIMIT 10"
```
