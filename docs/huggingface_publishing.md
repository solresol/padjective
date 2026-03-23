# Automating Hugging Face publishing

This repo can generate two dataset exports:

- `paper/`: a fixed point-in-time snapshot (cut off at the timestamp recorded in the paper, or an explicit `--paper-as-of`)
- `latest/`: a rolling snapshot from the current database state

The exports are written to a local folder and can optionally be uploaded to a
Hugging Face *dataset* repo.

The publish/export step also stages the benchmark notebook to:

- `notebooks/product_taxonomy_bench.ipynb`

and writes shared benchmark artifacts to:

- `reports/paper/benchmark.json`
- `reports/paper/model_comparison.csv`
- `reports/paper/umllr_ablation.csv`
- `reports/latest/...`

It also refreshes the generated TeX includes used by the SIGIR eCom paper under
`../papers/padjective/sigir-ecom/generated/`.

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
  --paper-tex ../papers/padjective/sigir-ecom/padjective-ecom.tex \
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
uv run -m padjective.product_taxonomy_bench_publish \
  --out-root /data/hf/product-taxonomy-bench \
  --paper-tex ../papers/padjective/sigir-ecom/padjective-ecom.tex
```

Upload:

```bash
uv run -m padjective.hf_sync \
  --repo-id yourname/product-taxonomy-bench \
  --out-root /data/hf/product-taxonomy-bench
```

## Weekly publish cadence

The intended schedule is weekly, after the nightly site job:

- Monday at 06:30 Australia/Sydney time

The helper script `weekly_hf_publish.sh` wraps the default publish command:

```bash
SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```

## Cron example (Monday 06:30 Australia/Sydney time)

Add something like this on the publishing box (edit paths/env as needed):

```cron
30 6 * * 1 cd /path/to/padjective && SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```
