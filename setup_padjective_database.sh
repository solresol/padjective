#!/usr/bin/env bash
# Install PostgreSQL, download padjective data dumps, and restore them locally.
set -euo pipefail

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  echo "This script must be run as root (sudo)." >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_URL=${PADJECTIVE_DUMP_BASE_URL:-"https://datadumps.ifost.org.au/padjective/"}
WORK_DIR=${PADJECTIVE_DUMP_WORKDIR:-"${SCRIPT_DIR}/.padjective_dumps"}
DB_NAME=${PADJECTIVE_DATABASE_NAME:-padjective}
DB_USER=${PADJECTIVE_DATABASE_USER:-padjective}
DB_PASSWORD=${PADJECTIVE_DATABASE_PASSWORD:-padjective}
PSQL_BIN=${PSQL_BIN:-psql}
CURL_BIN=${CURL_BIN:-curl}
APT_GET=${APT_GET_BIN:-apt-get}
SYSTEMCTL=${SYSTEMCTL_BIN:-systemctl}
PG_SUPERUSER=${PG_SUPERUSER:-postgres}

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
  "padjective.taxonomy_lr_fold_results"
  "padjective.taxonomy_nn_fold_results"
  "padjective.model_performance_history"
)

PRODUCT_DETAILS_FILE="public_product_details_for_cantbuymelove_product.csv.gz"

ensure_dependencies() {
  export DEBIAN_FRONTEND=noninteractive
  ${APT_GET} update
  ${APT_GET} install -y postgresql postgresql-client python3 ${CURL_BIN} ca-certificates gzip
}

start_postgres() {
  if command -v pg_lsclusters >/dev/null 2>&1; then
    while read -r version cluster rest; do
      ${SYSTEMCTL} start "postgresql@${version}-${cluster}" >/dev/null 2>&1 || true
    done < <(pg_lsclusters --no-header 2>/dev/null || true)
  fi
  ${SYSTEMCTL} restart postgresql >/dev/null 2>&1 || true
}

sudo_psql() {
  sudo -u "${PG_SUPERUSER}" ${PSQL_BIN} -v ON_ERROR_STOP=1 "$@"
}

ensure_role_and_database() {
  sudo_psql <<SQL
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', '${DB_USER}', '${DB_PASSWORD}');
  ELSE
    EXECUTE format('ALTER ROLE %I LOGIN PASSWORD %L', '${DB_USER}', '${DB_PASSWORD}');
  END IF;
END
$$;
SQL

  sudo_psql <<SQL
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = '${DB_NAME}') THEN
    EXECUTE format('CREATE DATABASE %I OWNER %I', '${DB_NAME}', '${DB_USER}');
  END IF;
END
$$;
SQL
}

run_schema_scripts() {
  local sql_file
  for sql_file in \
    "${SCRIPT_DIR}/padjective/sql/tagbattles.sql" \
    "${SCRIPT_DIR}/create_umllr_tables.sql" \
    "${SCRIPT_DIR}/create_taxonomy_lr_fold_tables.sql" \
    "${SCRIPT_DIR}/create_taxonomy_nn_fold_tables.sql" \
    "${SCRIPT_DIR}/create_model_performance_history_table.sql"; do
    if [[ -f "${sql_file}" ]]; then
      sudo_psql -d "${DB_NAME}" -f "${sql_file}" >/dev/null
    fi
  done
}

fetch_file() {
  local remote_name="$1"
  local output_path="$2"
  if [[ -f "${output_path}" ]]; then
    echo "Using cached ${remote_name}" >&2
    return
  fi
  echo "Downloading ${remote_name}" >&2
  ${CURL_BIN} -fSL "${BASE_URL}${remote_name}" -o "${output_path}"
}

parse_sql_columns() {
  local sql_file="$1"
  python3 - "$sql_file" <<'PY'
import json
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')
match = re.search(r"INSERT\s+INTO\s+([\w\."]+)\s*\(([^)]+)\)\s*VALUES", text, re.IGNORECASE | re.DOTALL)
if not match:
    print("[]")
    sys.exit(0)
columns = [c.strip().strip('"') for c in match.group(2).split(',')]
print(json.dumps(columns))
PY
}

ensure_table_for_sql() {
  local qualified_table="$1"
  local sql_file="$2"
  local columns_json
  columns_json=$(parse_sql_columns "${sql_file}")
  if [[ "${columns_json}" == "[]" ]]; then
    return
  fi

  local ensure_sql
  ensure_sql=$(COLUMNS_JSON="${columns_json}" python3 - "$qualified_table" <<'PY'
import json
import os
import sys

qualified = sys.argv[1].replace('"', '')
columns = json.loads(os.environ['COLUMNS_JSON'])
if '.' in qualified:
    schema, table = qualified.split('.', 1)
else:
    schema, table = 'public', qualified

TYPE_OVERRIDES = {
    'id': 'BIGINT',
    'product_id': 'BIGINT',
    'taxonomy_id': 'TEXT',
    'taxonomy_path': 'TEXT',
    'taxonomy_name': 'TEXT',
    'winner_tag': 'TEXT',
    'loser_tag': 'TEXT',
    'tag': 'TEXT',
    'component': 'INTEGER',
    'score': 'DOUBLE PRECISION',
    'updated_at': 'TIMESTAMPTZ',
    'recorded_at': 'TIMESTAMPTZ',
    'classified_at': 'TIMESTAMPTZ',
    'trained_at': 'TIMESTAMPTZ',
    'snapshot_date': 'DATE',
    'snapshot_time': 'TIMESTAMPTZ',
    'num_products': 'INTEGER',
    'num_tags': 'INTEGER',
    'num_taxonomies': 'INTEGER',
    'umllr_mean_padic_loss': 'REAL',
    'lr_mean_padic_loss': 'REAL',
    'nn_mean_padic_loss': 'REAL',
    'umllr_mean_accuracy': 'REAL',
    'lr_mean_accuracy': 'REAL',
    'nn_mean_accuracy': 'REAL',
    'prompt_tokens': 'INTEGER',
    'completion_tokens': 'INTEGER',
    'total_tokens': 'INTEGER',
    'raw_output': 'JSONB',
    'product_detail': 'JSONB',
    'loss': 'DOUBLE PRECISION',
    'prime_base': 'INTEGER',
    'prime_power': 'INTEGER',
    'max_digit': 'INTEGER',
    'num_train_samples': 'INTEGER',
    'num_test_samples': 'INTEGER',
    'cv_fold': 'INTEGER',
    'encoded_value': 'NUMERIC',
    'product_key': 'TEXT',
    'store_domain': 'TEXT',
    'product_title': 'TEXT',
    'product_url': 'TEXT',
    'product_handle': 'TEXT',
    'myshopify_domain': 'TEXT',
    'run_name': 'TEXT',
    'sequence': 'INTEGER',
    'tag_count': 'INTEGER',
    'hidden_layers': 'TEXT',
    'test_accuracy': 'DOUBLE PRECISION',
    'test_f1': 'DOUBLE PRECISION',
    'test_hierarchical_loss': 'DOUBLE PRECISION',
    'padic_loss_total': 'DOUBLE PRECISION',
    'padic_loss_mean': 'DOUBLE PRECISION',
}

def column_type(name: str) -> str:
    key = name.lower()
    if key in TYPE_OVERRIDES:
        return TYPE_OVERRIDES[key]
    if key.endswith('_id'):
        return 'BIGINT'
    if key.endswith('_at'):
        return 'TIMESTAMPTZ'
    if key.startswith('num_') or key.endswith('_count'):
        return 'INTEGER'
    if 'loss' in key or 'accuracy' in key or key.endswith('_score'):
        return 'DOUBLE PRECISION'
    if 'tokens' in key:
        return 'INTEGER'
    if 'json' in key:
        return 'JSONB'
    return 'TEXT'

create_columns = ', '.join(f'"{col}" {column_type(col)}' for col in columns)
statements = [
    f'CREATE SCHEMA IF NOT EXISTS "{schema}";',
    f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" ({create_columns});'
]
for col in columns:
    statements.append(
        f'ALTER TABLE "{schema}"."{table}" ADD COLUMN IF NOT EXISTS "{col}" {column_type(col)};'
    )
print('\n'.join(statements))
PY
  )

  sudo_psql -d "${DB_NAME}" <<SQL
${ensure_sql}
SQL
}

ensure_table_for_csv() {
  local qualified_table="$1"
  local csv_file="$2"

  local columns_json
  columns_json=$(python3 - "$csv_file" <<'PY'
import csv
import json
import sys

with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    reader = csv.reader(handle)
    header = next(reader)
columns = [col.strip() for col in header]
print(json.dumps(columns))
PY
  )

  local ensure_sql
  ensure_sql=$(COLUMNS_JSON="${columns_json}" python3 - "$qualified_table" <<'PY'
import json
import os
import sys

qualified = sys.argv[1].replace('"', '')
columns = json.loads(os.environ['COLUMNS_JSON'])
if '.' in qualified:
    schema, table = qualified.split('.', 1)
else:
    schema, table = 'public', qualified

def column_type(name: str) -> str:
    lower = name.lower()
    if lower == 'product_detail':
        return 'JSONB'
    if lower in {'product_id', 'id'}:
        return 'BIGINT'
    if lower in {'prompt_tokens', 'completion_tokens', 'total_tokens'}:
        return 'INTEGER'
    if lower.endswith('_at'):
        return 'TIMESTAMPTZ'
    if lower.startswith('num_') or lower.endswith('_count'):
        return 'INTEGER'
    return 'TEXT'

create_cols = ', '.join(f'"{col}" {column_type(col)}' for col in columns)
statements = [
    f'CREATE SCHEMA IF NOT EXISTS "{schema}";',
    f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" ({create_cols});'
]
for col in columns:
    statements.append(
        f'ALTER TABLE "{schema}"."{table}" ADD COLUMN IF NOT EXISTS "{col}" {column_type(col)};'
    )
print('\n'.join(statements))
PY
  )

  sudo_psql -d "${DB_NAME}" <<SQL
${ensure_sql}
SQL

  local copy_columns
  copy_columns=$(COLUMNS_JSON="${columns_json}" python3 - <<'PY'
import json
import os

cols = json.loads(os.environ['COLUMNS_JSON'])
print(', '.join(f'"{c}"' for c in cols))
PY
  )

  sudo_psql -d "${DB_NAME}" <<SQL
TRUNCATE TABLE ${qualified_table};
\copy ${qualified_table} (${copy_columns}) FROM '${csv_file}' WITH (FORMAT csv, HEADER true);
SQL
}

restore_sql_table() {
  local qualified_table="$1"
  local gz_path="$2"
  local safe_name
  safe_name=$(basename "${gz_path}" .gz)
  local sql_path="${WORK_DIR}/${safe_name}"
  gunzip -c "${gz_path}" >"${sql_path}"
  ensure_table_for_sql "${qualified_table}" "${sql_path}"
  sudo_psql -d "${DB_NAME}" -c "TRUNCATE TABLE ${qualified_table};" >/dev/null
  sudo_psql -d "${DB_NAME}" -f "${sql_path}" >/dev/null
}

main() {
  mkdir -p "${WORK_DIR}"
  ensure_dependencies
  start_postgres
  ensure_role_and_database
  run_schema_scripts

  local table
  for table in "${REQUIRED_TABLES[@]}"; do
    local safe_name="${table//./_}.sql.gz"
    local gz_path="${WORK_DIR}/${safe_name}"
    fetch_file "${safe_name}" "${gz_path}"
    restore_sql_table "${table}" "${gz_path}"
  done

  local product_details_path="${WORK_DIR}/${PRODUCT_DETAILS_FILE}"
  fetch_file "${PRODUCT_DETAILS_FILE}" "${product_details_path}"
  local csv_path="${WORK_DIR}/public_product_details_for_cantbuymelove_product.csv"
  gunzip -c "${product_details_path}" >"${csv_path}"
  ensure_table_for_csv "public.product_details" "${csv_path}"

  echo "Padjective database restored into '${DB_NAME}'."
}

main "$@"
