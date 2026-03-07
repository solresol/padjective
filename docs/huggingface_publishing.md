# Automating Hugging Face publishing

This repo can generate two dataset exports:

- `paper/`: a fixed point-in-time snapshot (cut off at the timestamp recorded in the paper, or an explicit `--paper-as-of`)
- `latest/`: a rolling snapshot from the current database state

The exports are written to a local folder and can optionally be uploaded to a
Hugging Face *dataset* repo.

The publish/export step also stages the benchmark notebook to:

- `notebooks/product_taxonomy_bench.ipynb`

so the dataset card can link directly to a rendered notebook on the Hub.

## One-shot publish (export + upload)

Prereqs on the machine doing the publish:

- access to the Shopify Postgres database (`SHOPIFY_DB_DSN` or `DATABASE_URL`)
- Hugging Face access token (recommended: `HF_TOKEN`)
- `huggingface_hub` installed (this repo treats it as optional)

Example:

```bash
export SHOPIFY_DB_DSN='postgresql://…'
export HF_TOKEN='hf_…'

uv run -m padjective.product_taxonomy_bench_publish \
  --out-root /data/hf/product-taxonomy-bench \
  --hf-repo-id yourname/product-taxonomy-bench
```

Notes:

- For a strictly stable paper snapshot, pin it explicitly:
  `--paper-as-of "2026-02-11 19:15 UTC"` (or whatever the paper’s dataset cutoff
  is) so future edits to the paper TeX can’t move the cutoff.
- If you don’t have `pyarrow` installed, either install it or run with
  `--formats jsonl`.

## Separate steps (export, then upload)

Export:

```bash
uv run -m padjective.product_taxonomy_bench_publish --out-root /data/hf/product-taxonomy-bench
```

Upload:

```bash
uv run -m padjective.hf_sync \
  --repo-id yourname/product-taxonomy-bench \
  --out-root /data/hf/product-taxonomy-bench
```

## Cron example (daily at 03:00)

Add something like this on the publishing box (edit paths/env as needed):

```cron
0 3 * * * cd /path/to/padjective && SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' uv run -m padjective.product_taxonomy_bench_publish --paper-as-of "2026-02-11 19:15 UTC" --out-root /data/hf/product-taxonomy-bench --hf-repo-id yourname/product-taxonomy-bench
```
