# Publishing SHIMP to Hugging Face

SHIMP (Shopify Hierarchical Inference for Merchant Products) is hosted at
`gregb/product-taxonomy-bench` under DOI [10.57967/hf/8413](https://doi.org/10.57967/hf/8413).
Both publishing commands use SHIMP as the default display name.

The dataset card declares `license: cc-by-4.0`. Both commands stage `LICENSE`,
`LICENSE-DATA`, `LICENSE-CODE` and `THIRD_PARTY_NOTICES.md`. The viewer selects
only `products-*.jsonl` and `products-*.jsonl.gz` within each snapshot.

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
  --hf-repo-id gregb/product-taxonomy-bench
```

Notes:

- Omitting `--hf-repo-id` keeps the Python publisher export-only. Generated
  cards and notebooks default to `gregb/product-taxonomy-bench`; with `--hf-repo-id`, the card
  and notebook use that target. `--dataset-id` and `--pretty-name` remain
  available to override card metadata explicitly. When `--dataset-id` is a
  short display label, the repository link, citation, and notebook still use
  `--hf-repo-id` (or `gregb/product-taxonomy-bench` for a local export). A full `owner/name`
  passed to `--dataset-id` also selects the notebook and citation target when
  `--hf-repo-id` is omitted.
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
  --repo-id gregb/product-taxonomy-bench \
  --out-root /data/hf/shimp
```

## Weekly publish cadence

The intended schedule is weekly, after the nightly site job:

- Monday at 06:30 Australia/Sydney time

The helper script `weekly_hf_publish.sh` wraps the default publish command:

```bash
SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```

`PADJECTIVE_HF_REPO_ID` overrides the weekly upload target.
The local output default remains `build/hf/product-taxonomy-bench` for existing
jobs; set `PADJECTIVE_HF_OUT_ROOT` to choose another directory.

Python module names, `product-taxonomy-bench-*` commands, notebook paths,
Postgres table names, and `PRODUCT_TAXONOMY_BENCH_*` notebook environment
variables remain compatible. Snapshot cutoffs, anonymisation,
fields, and cross-validation folds are unchanged.

## Cron example (Monday 06:30 Australia/Sydney time)

Add something like this on the publishing box (edit paths/env as needed):

```cron
30 6 * * 1 cd /path/to/padjective && SHOPIFY_DB_DSN='postgresql://…' HF_TOKEN='hf_…' ./weekly_hf_publish.sh
```
