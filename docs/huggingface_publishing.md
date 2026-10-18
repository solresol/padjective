# Publishing SHIMP to Hugging Face

**SHIMP** stands for **Shopify Hierarchy Inference from Merchant Products**.
The default publishing target in `weekly_hf_publish.sh` is `gregb/shimp`.
The existing dataset at
[`gregb/product-taxonomy-bench`](https://huggingface.co/datasets/gregb/product-taxonomy-bench)
is the legacy target; this code change does not rename, move, delete, or upload
either Hub repository. Publishing to `gregb/shimp` is a separate live step on
the publishing machine with database access and a suitable Hugging Face token.

This repo can generate three dataset exports:

- `paper/`: a fixed point-in-time snapshot (cut off at the timestamp recorded in the paper, or an explicit `--paper-as-of`)
- `latest/`: a rolling snapshot from the current database state
- `first1000/`: the first 1,000 labelled products for a small sanity check

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
  --out-root /data/hf/shimp \
  --paper-tex ../papers/padjective/sigir-ecom/padjective-ecom.tex \
  --hf-repo-id gregb/shimp
```

Notes:

- Omitting `--hf-repo-id` keeps the Python publisher export-only. Generated
  cards and notebooks default to `gregb/shimp`; with `--hf-repo-id`, the card
  and notebook use that target. `--dataset-id` and `--pretty-name` remain
  available to override card metadata explicitly.
- For a strictly stable paper snapshot, pin it explicitly:
  `--paper-as-of "2026-02-11 19:15 UTC"` (or whatever the paper’s dataset cutoff
  is) so future edits to the paper TeX can’t move the cutoff.
- If you don’t have `pyarrow` installed, either install it or run with
  `--formats jsonl`.

## Separate steps (export, then upload)

Export:

```bash
uv run -m padjective.product_taxonomy_bench_publish \
  --out-root /data/hf/shimp \
  --paper-tex ../papers/padjective/sigir-ecom/padjective-ecom.tex
```

Upload:

```bash
uv run -m padjective.hf_sync \
  --repo-id gregb/shimp \
  --out-root /data/hf/shimp
```

## Weekly publish cadence

The intended schedule is weekly, after the nightly site job:

- Monday at 06:30 Australia/Sydney time

The helper script `weekly_hf_publish.sh` wraps the default publish command:

```bash
SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```

`PADJECTIVE_HF_REPO_ID` still overrides the weekly upload target, including
`PADJECTIVE_HF_REPO_ID=gregb/product-taxonomy-bench` for the legacy repository.
The local output default remains `build/hf/product-taxonomy-bench` for existing
jobs; set `PADJECTIVE_HF_OUT_ROOT` to choose another directory.

Python module names, `product-taxonomy-bench-*` commands, notebook paths,
Postgres table names, and `PRODUCT_TAXONOMY_BENCH_*` notebook environment
variables remain compatible. For example, set
`PRODUCT_TAXONOMY_BENCH_DATASET_ID=gregb/product-taxonomy-bench` to read the
existing release before SHIMP is published. Snapshot cutoffs, anonymisation,
fields, and cross-validation folds are unchanged.

## Cron example (Monday 06:30 Australia/Sydney time)

Add something like this on the publishing box (edit paths/env as needed):

```cron
30 6 * * 1 cd /path/to/padjective && SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```
