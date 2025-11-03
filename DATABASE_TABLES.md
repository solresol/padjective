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
- `umllr_fold_metrics` - Loss, prime base, max digit per fold
- `umllr_predictions` - Individual product predictions with p-adic loss
- `umllr_taxonomy_encodings` - P-adic encodings of taxonomy paths per fold

### 3. Logistic Regression Classifier Tables

**SQL Script:** `create_taxonomy_lr_fold_tables.sql`
**Created by:** `padjective/taxonomy_classifier.py`

- `taxonomy_lr_models` - Full training metadata and cross-validation results
- `taxonomy_lr_cv_scores` - Per-fold accuracy, F1, and hierarchical loss metrics
- `taxonomy_lr_class_distribution` - Class balance for the training set
- `taxonomy_lr_tag_summary` - Top-weighted taxonomy assignments per tag
- `taxonomy_lr_top_tags` - Highest-weighted tags for each taxonomy
- `taxonomy_lr_intercepts` - Per-class intercept terms for the logistic model
- `taxonomy_lr_ignored_products` - Products excluded because their taxonomy path lacked a dot
- `taxonomy_lr_excluded_taxonomies` - Taxonomies filtered out for failing the minimum sample count
- `taxonomy_lr_fold_results` - Per-fold test metrics (accuracy, F1, hierarchical loss, p-adic loss)

**Schema Creation:** `padjective/taxonomy_classifier_schema.py`

### 4. Neural Network Classifier Tables

**SQL Script:** `create_taxonomy_nn_fold_tables.sql`  
**Created by:** `padjective/taxonomy_nn_classifier.py`

- `taxonomy_nn_fold_results` - Per-fold test metrics (accuracy, F1, hierarchical loss, p-adic loss)
- `taxonomy_nn_models` - Full training metadata (stored in SQLite at `data/taxonomy_nn_classifier.sqlite`)
- `taxonomy_nn_cv_scores` - Cross-validation scores (stored in SQLite)

### 5. Historical Performance Tracking Tables

**SQL Script:** `create_model_performance_history_table.sql`  
**Created by:** `padjective/snapshot_metrics.py`

- `model_performance_history` - Daily snapshots of all model metrics and dataset statistics

## Running the Scripts

To manually create all tables with proper permissions:

```bash
# As admin/superuser
psql -f create_umllr_tables.sql
psql -f create_taxonomy_lr_fold_tables.sql
psql -f create_taxonomy_nn_fold_tables.sql
psql -f create_model_performance_history_table.sql
```

## Table Cleanup

The following tables are cleaned (old data deleted) before each cronscript run:

- `umllr_*` tables - Cleaned by `umllr.py` before inserting new fold data
- `taxonomy_lr_fold_results` - Per-fold cleanup by `taxonomy_classifier.py`
- `taxonomy_nn_fold_results` - Per-fold cleanup by `taxonomy_nn_classifier.py`

The following table accumulates data over time (no cleanup):

- `model_performance_history` - Uses `ON CONFLICT DO UPDATE` to update existing dates

## Verification

To verify all tables exist:

```bash
psql -c "\dt padjective.*"
```

To check current snapshot history:

```bash
psql -c "SELECT * FROM padjective.model_performance_history ORDER BY snapshot_date DESC LIMIT 10"
```
