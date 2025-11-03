#!/usr/bin/env bash
# Dump the Postgres tables required to run padjective's end-to-end workflows
# and sync them to the shared datadumps host.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DUMP_DIR=${PADJECTIVE_DUMP_DIR:-"$SCRIPT_DIR/datadumps"}
REMOTE_TARGET=${PADJECTIVE_REMOTE_DUMP_TARGET:-"merah:/var/www/vhosts/datadumps.ifost.org.au/htdocs/padjective/"}
PG_DUMP_BIN=${PG_DUMP_BIN:-pg_dump}
PSQL_BIN=${PSQL_BIN:-psql}
RSYNC_BIN=${RSYNC_BIN:-rsync}

REQUIRED_TABLES=(
  "cantbuymelove.product"
  "cantbuymelove.product_taxonomy"
  "cantbuymelove.taxonomy"
  "padjective.battles"
  "padjective.tag_rankings"
  "padjective.umllr_tag_coefficients"
  "padjective.umllr_fold_metrics"
  "padjective.umllr_predictions"
  "padjective.umllr_taxonomy_encodings"
  "padjective.taxonomy_lr_models"
  "padjective.taxonomy_lr_cv_scores"
  "padjective.taxonomy_lr_intercepts"
  "padjective.taxonomy_lr_tag_summary"
  "padjective.taxonomy_lr_class_distribution"
  "padjective.taxonomy_lr_top_tags"
  "padjective.taxonomy_lr_ignored_products"
  "padjective.taxonomy_lr_excluded_taxonomies"
  "padjective.taxonomy_lr_fold_results"
  "padjective.taxonomy_nn_fold_results"
  "padjective.model_performance_history"
)

if ! command -v "$PG_DUMP_BIN" >/dev/null 2>&1; then
  echo "Error: pg_dump not found (looked for '$PG_DUMP_BIN')." >&2
  exit 1
fi

if ! command -v "$PSQL_BIN" >/dev/null 2>&1; then
  echo "Error: psql not found (looked for '$PSQL_BIN')." >&2
  exit 1
fi

if ! command -v "$RSYNC_BIN" >/dev/null 2>&1; then
  echo "Error: rsync not found (looked for '$RSYNC_BIN')." >&2
  exit 1
fi

mkdir -p "$DUMP_DIR"

PG_CONN_ARGS=()
if [[ -n "${SHOPIFY_DB_DSN:-}" ]]; then
  PG_CONN_ARGS+=("--dbname=${SHOPIFY_DB_DSN}")
fi

for table in "${REQUIRED_TABLES[@]}"; do
  safe_name=${table//./_}
  output_path="$DUMP_DIR/${safe_name}.sql"
  echo "Dumping $table → $output_path"
  "$PG_DUMP_BIN" \
    "${PG_CONN_ARGS[@]}" \
    --no-owner \
    --no-privileges \
    --data-only \
    --column-inserts \
    --table="${table}" \
    > "$output_path"
  gzip -f "$output_path"
done

dump_filtered_product_details() {
  local output_path="$DUMP_DIR/public_product_details_for_cantbuymelove_product.csv"
  local copy_sql

  if [[ -n "${PADJECTIVE_PRODUCT_DETAILS_COPY_SQL:-}" ]]; then
    copy_sql=${PADJECTIVE_PRODUCT_DETAILS_COPY_SQL}
  else
    read -r -d '' copy_sql <<'SQL'
\copy (
    SELECT pd.*
    FROM public.product_details AS pd
    JOIN (
        SELECT DISTINCT myshopify_domain, run_name, product_handle
        FROM cantbuymelove.product
    ) AS required_products
    USING (myshopify_domain, run_name, product_handle)
) TO STDOUT WITH (FORMAT csv, HEADER true)
SQL
  fi

  echo "Exporting filtered public.product_details rows → ${output_path}"
  printf '%s\n' "$copy_sql" | "$PSQL_BIN" "${PG_CONN_ARGS[@]}" >"$output_path"
  gzip -f "$output_path"
}

dump_filtered_product_details

echo "Syncing dumps to $REMOTE_TARGET"
"$RSYNC_BIN" -avz --delete "$DUMP_DIR/" "$REMOTE_TARGET"
