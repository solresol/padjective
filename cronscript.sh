#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

OUTPUT_DIR=${PADJECTIVE_SITE_DIR:-build/site}
TAXONOMY_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_CLASSIFIER_REPORT_DIR:-build/taxonomy_classifier}
TAXONOMY_NN_CLASSIFIER_DB=${PADJECTIVE_TAXONOMY_NN_CLASSIFIER_DB:-data/taxonomy_nn_classifier.sqlite}
TAXONOMY_NN_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_NN_CLASSIFIER_REPORT_DIR:-build/taxonomy_nn_classifier}
TAGBATTLE_SCHEMA=${PADJECTIVE_TAGBATTLE_SCHEMA:-padjective}
TAXONOMY_RESULTS_SCHEMA=${PADJECTIVE_TAXONOMY_RESULTS_SCHEMA:-padjective}
TAGBATTLE_PRODUCT_TABLE=${PADJECTIVE_TAGBATTLE_PRODUCT_TABLE:-cantbuymelove.product}
TAGBATTLE_BATCH_SIZE=${PADJECTIVE_TAGBATTLE_BATCH_SIZE:-2000}
SHOPIFY_DSN=${PADJECTIVE_SHOPIFY_DSN:-}

mkdir -p "$TAXONOMY_CLASSIFIER_REPORT_DIR"
mkdir -p "$(dirname "$TAXONOMY_NN_CLASSIFIER_DB")"
mkdir -p "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR"

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
# Use --max-tags 109 for fair comparison with umllr (~10,000 parameters)
for fold in 0 1 2 3 4; do
    echo "Training taxonomy classifier for fold $fold..."
    uv run -m padjective.taxonomy_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
        --output-dir "$TAXONOMY_CLASSIFIER_REPORT_DIR" \
        --fold "$fold" \
        --max-tags 109
done

# Train full taxonomy classifier (populates taxonomy_pclr_models summary table for index card)
echo "Training full taxonomy classifier..."
uv run -m padjective.taxonomy_classifier \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --results-schema "$TAXONOMY_RESULTS_SCHEMA" \
    --output-dir "$TAXONOMY_CLASSIFIER_REPORT_DIR" \
    --max-tags 109

# Train neural network classifiers (once per fold)
# Use --max-tags 109 and --hidden-layers 49 for fair comparison with umllr (~10,000 parameters)
for fold in 0 1 2 3 4; do
    echo "Training neural network classifier for fold $fold..."
    uv run -m padjective.taxonomy_nn_classifier \
        "${TAGBATTLE_DSN_ARGS[@]}" \
        --product-table "$TAGBATTLE_PRODUCT_TABLE" \
        --model-database "$TAXONOMY_NN_CLASSIFIER_DB" \
        --output-dir "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR" \
        --hidden-layers "49" \
        --max-tags 109 \
        --fold "$fold"
done

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

if [ -d "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR" ]; then
    rsync -az "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/taxonomy_nn_classifier/"
fi

# Sync data dumps
DUMP_SOURCE="$OUTPUT_DIR/datadumps/"
if [ -d "$DUMP_SOURCE" ]; then
    REMOTE_DUMPS="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"
    rsync -az "$DUMP_SOURCE" "$REMOTE_DUMPS"
fi
