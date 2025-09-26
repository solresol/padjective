#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

CSV_PATH=${PADJECTIVE_PRODUCTS_CSV:-products_point_one_percent_sample.csv}
OUTPUT_DIR=${PADJECTIVE_SITE_DIR:-build/site}

uv run -m padjective.build_site --csv "$CSV_PATH" --output "$OUTPUT_DIR"

REMOTE_SITE="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/padjective.symmachus.org/htdocs/"
rsync -avz --delete "$OUTPUT_DIR/" "$REMOTE_SITE"

DUMP_SOURCE="$OUTPUT_DIR/datadumps/"
if [ -d "$DUMP_SOURCE" ]; then
    REMOTE_DUMPS="padjective@merah.cassia.ifost.org.au:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"
    rsync -avz "$DUMP_SOURCE" "$REMOTE_DUMPS"
fi
