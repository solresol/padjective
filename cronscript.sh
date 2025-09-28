#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

CSV_PATH=${PADJECTIVE_PRODUCTS_CSV:-products_point_one_percent_sample.csv}
OUTPUT_DIR=${PADJECTIVE_SITE_DIR:-build/site}
TASKS_DB=${PADJECTIVE_TASKS_DB:-data/holdout_tasks.sqlite}
TOTAL_TASKS=${PADJECTIVE_TOTAL_HOLDOUT_TASKS:-5000}
TASK_BATCH=${PADJECTIVE_TASK_BATCH:-250}
BATTLES_DB=${PADJECTIVE_BATTLES_DB:-build/battles.sqlite}
SYNSETS_DB=${PADJECTIVE_SYNSETS_DB:-data/product_synsets.sqlite}
SYNSETS_BATCH=${PADJECTIVE_SYNSETS_BATCH:-1000}

mkdir -p "$(dirname "$TASKS_DB")"
mkdir -p "$(dirname "$BATTLES_DB")"
mkdir -p "$(dirname "$SYNSETS_DB")"

uv run -m padjective.product_synsets --csv "$CSV_PATH" --database "$SYNSETS_DB" --batch "$SYNSETS_BATCH"
uv run padjective/tagbattle.py --csv "$CSV_PATH" --database "$BATTLES_DB"
uv run -m padjective.experiments init --tasks-db "$TASKS_DB" --total "$TOTAL_TASKS"
uv run -m padjective.experiments run --tasks-db "$TASKS_DB" --database "$BATTLES_DB" --take "$TASK_BATCH"
uv run -m padjective.build_site --csv "$CSV_PATH" --output "$OUTPUT_DIR" --precomputed-database "$BATTLES_DB" --tasks-db "$TASKS_DB" --synset-db "$SYNSETS_DB"

REMOTE_SITE="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/padjective.symmachus.org/htdocs/"
rsync -avz --delete "$OUTPUT_DIR/" "$REMOTE_SITE"

DUMP_SOURCE="$OUTPUT_DIR/datadumps/"
if [ -d "$DUMP_SOURCE" ]; then
    REMOTE_DUMPS="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"
    rsync -avz "$DUMP_SOURCE" "$REMOTE_DUMPS"
fi
