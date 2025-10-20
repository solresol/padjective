#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

CSV_PATH=${PADJECTIVE_PRODUCTS_CSV:-products_point_one_percent_sample.csv}
OUTPUT_DIR=${PADJECTIVE_SITE_DIR:-build/site}
TASKS_DB=${PADJECTIVE_TASKS_DB:-data/holdout_tasks.sqlite}
TOTAL_TASKS=${PADJECTIVE_TOTAL_HOLDOUT_TASKS:-5000}
TASK_BATCH=${PADJECTIVE_TASK_BATCH:-250}
SYNSETS_DB=${PADJECTIVE_SYNSETS_DB:-data/product_synsets.sqlite}
CLASSIFIER_DB=${PADJECTIVE_CLASSIFIER_DB:-data/synset_classifier.sqlite}
CLASSIFIER_REPORT_DIR=${PADJECTIVE_CLASSIFIER_REPORT_DIR:-build/synset_classifier}
TAXONOMY_CLASSIFIER_DB=${PADJECTIVE_TAXONOMY_CLASSIFIER_DB:-data/taxonomy_classifier.sqlite}
TAXONOMY_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_CLASSIFIER_REPORT_DIR:-build/taxonomy_classifier}
TAXONOMY_NN_CLASSIFIER_DB=${PADJECTIVE_TAXONOMY_NN_CLASSIFIER_DB:-data/taxonomy_nn_classifier.sqlite}
TAXONOMY_NN_CLASSIFIER_REPORT_DIR=${PADJECTIVE_TAXONOMY_NN_CLASSIFIER_REPORT_DIR:-build/taxonomy_nn_classifier}
TAGBATTLE_SCHEMA=${PADJECTIVE_TAGBATTLE_SCHEMA:-padjective}
TAGBATTLE_PRODUCT_TABLE=${PADJECTIVE_TAGBATTLE_PRODUCT_TABLE:-cantbuymelove.product}
TAGBATTLE_BATCH_SIZE=${PADJECTIVE_TAGBATTLE_BATCH_SIZE:-2000}
SHOPIFY_DSN=${PADJECTIVE_SHOPIFY_DSN:-}

mkdir -p "$(dirname "$TASKS_DB")"
mkdir -p "$(dirname "$SYNSETS_DB")"
mkdir -p "$(dirname "$CLASSIFIER_DB")"
mkdir -p "$CLASSIFIER_REPORT_DIR"
mkdir -p "$(dirname "$TAXONOMY_CLASSIFIER_DB")"
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

# Train taxonomy classifiers
uv run padjective/taxonomy_classifier.py \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --model-database "$TAXONOMY_CLASSIFIER_DB" \
    --output-dir "$TAXONOMY_CLASSIFIER_REPORT_DIR"

uv run padjective/taxonomy_nn_classifier.py \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --model-database "$TAXONOMY_NN_CLASSIFIER_DB" \
    --output-dir "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR" \
    --hidden-layers "100,50"

uv run -m padjective.experiments --tasks-db "$TASKS_DB" init --total "$TOTAL_TASKS"
uv run -m padjective.experiments --tasks-db "$TASKS_DB" run "${TAGBATTLE_DSN_ARGS[@]}" --schema "$TAGBATTLE_SCHEMA" --take "$TASK_BATCH"
uv run -m padjective.build_site --csv "$CSV_PATH" --output "$OUTPUT_DIR" "${TAGBATTLE_DSN_ARGS[@]}" --schema "$TAGBATTLE_SCHEMA" --tasks-db "$TASKS_DB" --synset-db "$SYNSETS_DB"

# Legacy synset classifier (deprecated but keep for now)
if [ -f "$SYNSETS_DB" ]; then
    uv run padjective/synset_classifier.py --database "$SYNSETS_DB" --model-database "$CLASSIFIER_DB" --output-dir "$CLASSIFIER_REPORT_DIR" || true
fi

REMOTE_SITE="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/padjective.symmachus.org/htdocs/"
rsync -avz --delete "$OUTPUT_DIR/" "$REMOTE_SITE"

# Sync taxonomy classifier reports
if [ -d "$TAXONOMY_CLASSIFIER_REPORT_DIR" ]; then
    rsync -avz "$TAXONOMY_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/taxonomy_classifier/"
fi

if [ -d "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR" ]; then
    rsync -avz "$TAXONOMY_NN_CLASSIFIER_REPORT_DIR/" "$REMOTE_SITE/taxonomy_nn_classifier/"
fi

# Sync data dumps
DUMP_SOURCE="$OUTPUT_DIR/datadumps/"
if [ -d "$DUMP_SOURCE" ]; then
    REMOTE_DUMPS="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"
    rsync -avz "$DUMP_SOURCE" "$REMOTE_DUMPS"
fi
