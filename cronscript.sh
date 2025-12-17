#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

OUTPUT_DIR=${PADJECTIVE_SITE_DIR:-build/site}
TAXONOMY_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_CLASSIFIER_REPORT_DIR:-build/taxonomy_classifier}
TAXONOMY_PCNN_CLASSIFIER_DB=${PADJECTIVE_TAXONOMY_PCNN_CLASSIFIER_DB:-data/taxonomy_pcnn_classifier.sqlite}
TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR:-build/taxonomy_pcnn_classifier}
TAXONOMY_ULR_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_ULR_CLASSIFIER_REPORT_DIR:-build/unconstrained_logistic_regression}
TAGBATTLE_SCHEMA=${PADJECTIVE_TAGBATTLE_SCHEMA:-padjective}
TAXONOMY_RESULTS_SCHEMA=${PADJECTIVE_TAXONOMY_RESULTS_SCHEMA:-padjective}
TAGBATTLE_PRODUCT_TABLE=${PADJECTIVE_TAGBATTLE_PRODUCT_TABLE:-cantbuymelove.product}
TAGBATTLE_BATCH_SIZE=${PADJECTIVE_TAGBATTLE_BATCH_SIZE:-2000}
SHOPIFY_DSN=${PADJECTIVE_SHOPIFY_DSN:-}

mkdir -p "$TAXONOMY_CLASSIFIER_REPORT_DIR"
mkdir -p "$(dirname "$TAXONOMY_PCNN_CLASSIFIER_DB")"
mkdir -p "$TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR"
mkdir -p "$TAXONOMY_ULR_CLASSIFIER_REPORT_DIR"

if [[ -n "$SHOPIFY_DSN" ]]; then
    TAGBATTLE_DSN_ARGS=(--dsn "$SHOPIFY_DSN")
else
    TAGBATTLE_DSN_ARGS=()
fi

uv run padjective/tagbattle.py \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAGBATTLE_SCHEMA" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --batch-size "$TAGBATTLE_BATCH_SIZE"

uv run -m padjective.umllr \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAGBATTLE_SCHEMA" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE"

# Ensure taxonomy classifier schema exists
uv run -m padjective.taxonomy_classifier_schema \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAXONOMY_RESULTS_SCHEMA"

# Train taxonomy classifiers (once per fold)
# Use --max-tags 32 to get 32 × 199 = 6,368 parameters (close to 6,466 unique tags)
for fold in 0 1 2 3 4; do
    echo "Training taxonomy classifier for fold $fold..."
    uv run -m padjective.taxonomy_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
        --output-dir "$TAXONOMY_CLASSIFIER_REPORT_DIR" \
        --fold "$fold" \
        --max-tags 32
done

# Train full taxonomy classifier (populates taxonomy_pclr_models summary table for index card)
echo "Training full taxonomy classifier..."
uv run -m padjective.taxonomy_classifier \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
    --output-dir "$TAXONOMY_CLASSIFIER_REPORT_DIR" \
    --max-tags 32

# Train parameter constrained neural network classifiers (once per fold)
# Use --max-tags 32 and --hidden-layers 27 for ~6,200 parameters:
#   Input-to-Hidden: 32 × 27 + 27 = 891 parameters
#   Hidden-to-Output: 27 × 199 + 199 = 5,572 parameters
#   Total: 6,463 parameters
for fold in 0 1 2 3 4; do
    echo "Training parameter constrained neural network classifier for fold $fold..."
    uv run -m padjective.taxonomy_pcnn_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --model-database "$TAXONOMY_PCNN_CLASSIFIER_DB" \
        --output-dir "$TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR" \
        --hidden-layers "27" \
        --max-tags 32 \
        --fold "$fold"
done

# Train unconstrained logistic regression classifiers with L1 regularization (once per fold)
# Uses ALL tags with L1 regularization to achieve sparsity
for fold in 0 1 2 3 4; do
    echo "Training unconstrained L1-regularized logistic regression classifier for fold $fold..."
    uv run -m padjective.taxonomy_ulr_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
        --output-dir "$TAXONOMY_ULR_CLASSIFIER_REPORT_DIR" \
        --fold "$fold"
done

# Train unconstrained neural network classifiers with L1 regularization and weight pruning (once per fold)
# Uses ALL tags with L1 regularization during training and post-training pruning for sparsity
for fold in 0 1 2 3 4; do
    echo "Training unconstrained neural network classifier for fold $fold..."
    uv run -m padjective.taxonomy_unn_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
        --fold "$fold"
done

# Train decision tree classifiers (once per fold)
# Uses ALL tags - tree depth provides natural complexity control
for fold in 0 1 2 3 4; do
    echo "Training decision tree classifier for fold $fold..."
    uv run -m padjective.taxonomy_dt_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
        --fold "$fold"
done

# Train Zubarev p-adic polynomial regression (stochastic alternative to UMLLR)
# Uses Mahler basis with simulated annealing optimization
# Train with UMLLR initialization
echo "Training Zubarev p-adic polynomial regression (UMLLR initialization)..."
uv run -m padjective.zubarev \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAGBATTLE_SCHEMA" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --max-iterations 10000 \
    --initialization-method umllr

# Train with zeros initialization
echo "Training Zubarev p-adic polynomial regression (zeros initialization)..."
uv run -m padjective.zubarev \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAGBATTLE_SCHEMA" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --max-iterations 50000 \
    --initialization-method zeros

# Snapshot current metrics for historical tracking
uv run -m padjective.snapshot_metrics \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --schema "$TAGBATTLE_SCHEMA"

uv run -m padjective.build_site --output "$OUTPUT_DIR" "${TAGBATTLE_DSN_ARGS[@]}" --schema "$TAGBATTLE_SCHEMA"

REMOTE_SITE="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/padjective.symmachus.org/htdocs/"
rsync -az --delete "$OUTPUT_DIR/" "$REMOTE_SITE"

# Sync taxonomy classifier reports
if [ -d "$TAXONOMY_CLASSIFIER_REPORT_DIR" ]; then
    rsync -az "$TAXONOMY_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/taxonomy_classifier/"
fi

if [ -d "$TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR" ]; then
    rsync -az "$TAXONOMY_PCNN_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/parameter_constrained_neural_network/"
fi

if [ -d "$TAXONOMY_ULR_CLASSIFIER_REPORT_DIR" ]; then
    rsync -az "$TAXONOMY_ULR_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/unconstrained_logistic_regression/"
fi

# Sync data dumps
DUMP_SOURCE="$OUTPUT_DIR/datadumps/"
if [ -d "$DUMP_SOURCE" ]; then
    REMOTE_DUMPS="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"
    rsync -az "$DUMP_SOURCE" "$REMOTE_DUMPS"
fi
