#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"

git pull -q

HF_OUTPUT_ROOT=${PADJECTIVE_HF_OUT_ROOT:-build/hf/product-taxonomy-bench}
HF_REPO_ID=${PADJECTIVE_HF_REPO_ID:-gregb/product-taxonomy-bench}
PAPER_TEX=${PADJECTIVE_PAPER_TEX:-../papers/padjective/sigir-ecom/padjective-ecom.tex}
PAPER_GENERATED_DIR=${PADJECTIVE_PAPER_GENERATED_DIR:-../papers/padjective/sigir-ecom/generated}
TAGBATTLE_SCHEMA=${PADJECTIVE_TAGBATTLE_SCHEMA:-padjective}
TAGBATTLE_PRODUCT_TABLE=${PADJECTIVE_TAGBATTLE_PRODUCT_TABLE:-cantbuymelove.product}
SHOPIFY_DSN=${PADJECTIVE_SHOPIFY_DSN:-}

mkdir -p "$HF_OUTPUT_ROOT"
mkdir -p "$PAPER_GENERATED_DIR"

if [[ -n "$SHOPIFY_DSN" ]]; then
    TAGBATTLE_DSN_ARGS=(--dsn "$SHOPIFY_DSN")
else
    TAGBATTLE_DSN_ARGS=()
fi

uv run -m padjective.product_taxonomy_bench_publish \
    "${TAGBATTLE_DSN_ARGS[@]}" \
    --schema "$TAGBATTLE_SCHEMA" \
    --product-table "$TAGBATTLE_PRODUCT_TABLE" \
    --paper-tex "$PAPER_TEX" \
    --out-root "$HF_OUTPUT_ROOT" \
    --hf-repo-id "$HF_REPO_ID" \
    --paper-generated-dir "$PAPER_GENERATED_DIR"
