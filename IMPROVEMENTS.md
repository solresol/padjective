# IMPROVEMENTS.md

*Analysis date: 2026-07-11*

padjective computes p-adic adjective embeddings and product-taxonomy classification over the Shopify products Postgres database, publishing daily benchmark artifacts (site, Hugging Face dataset, paper tables) via `cronscript.sh`. The repo is in healthy shape: clean working tree, active commit history (recent work on active-parameter benchmark tables/plots), uv-managed dependencies with a lockfile, and a broad pytest suite (`uv run -m pytest -q`). The main issues are structural (one 10k-line module), stale artifacts from the abandoned CSV workflow, and drift between the three overlapping READMEs.

## Bugs & Fixes

- **Stale hardcoded paper timestamp.** `cronscript.sh` defaults `PADJECTIVE_PAPER_AS_OF` to `2026-02-11 19:15 UTC`. Five months later every regenerated paper table silently claims February data unless the env var is set on the cron box. Default it to `$(date -u '+%Y-%m-%d %H:%M UTC')` instead.
- **Cron script depends on a sibling repo path.** `PADJECTIVE_PAPER_GENERATED_DIR` defaults to `../papers/padjective/sigir-ecom/generated`; if the papers checkout is missing the `mkdir -p` will succeed but the outputs land in an orphaned directory nobody publishes. Add an existence check on `../papers` (or fail loudly) before writing.
- **`git pull -q` at the top of `cronscript.sh`** will silently abort the whole nightly run (`set -euo pipefail`) if the box ever has a dirty tree or network blip. Consider `git pull -q || echo "pull failed, running with current checkout"` so a transient GitHub outage doesn't skip a night of results.

## Improvements

- **Split `padjective/build_site.py` (10,618 lines).** It is 38% of the codebase in one file. Break it into a `build_site/` package (page rendering, chart generation, taxonomy pages, benchmark pages). This is the single highest-leverage refactor for maintainability, and `tests/test_build_site_taxonomy.py` gives you a safety net.
- **`benchmark_runtime.py` (1,903 lines) and `umllr.py` (1,639 lines)** are next in line for the same treatment; at minimum extract shared table/TeX formatting (the recent commits `1179729`, `87e151a`, `6ccf7dd` all patch label/escaping logic that appears to be duplicated between table and plot code).
- **Consolidate the tag-battle → ranking → display pipeline entry points.** `cronscript.sh` invokes many `uv run padjective/foo.py` steps with a large env-var surface. A single `padjective/pipeline.py --stage ...` (or a Makefile) would make local reproduction of a failed cron stage much easier.

## Testing

- Coverage is good breadth-wise (22 test files), but there is no test for `cronscript.sh` orchestration or `build_site.py` beyond the taxonomy pages. Add a smoke test that runs the site build against a tiny fixture DB (the `setup_padjective_database.sh` + `cantbuymelove-schema.sql` pair used in `test_datadumps.py` could seed it).
- `hf_sync.py`, `benchmark_runtime.py`, `product_taxonomy_bench_notebook.py`, and `snapshot_metrics.py` have no matching `tests/test_*.py`. The HF publish path (`weekly_hf_publish.sh`) is externally visible — a dry-run test would catch card-metadata regressions like the one fixed in `718a067`.

## Documentation

- **Three overlapping READMEs.** `README.md` (18.8k), `README2.md` (2.7k, an older subset), and `AGENTS.md` all describe the project. Delete or fold `README2.md` into `README.md`; its content is fully superseded.
- **AGENTS.md contradicts itself with CLAUDE.md's sample-CSV instructions.** AGENTS.md says "We no longer read from CSV files or SQLite databases", yet the top of AGENTS.md still describes `products_point_one_percent_sample.csv` as the working data. Rewrite the opening to match the Postgres-only workflow.
- `daily_results_site_ideas.md` is an untriaged idea backlog (hero overview, depth timelines, component explorer, search & compare). Either promote items into a TODO with priorities or note which are implemented in `build_site.py` — right now it's unclear what's done.
- `DATABASE_TABLES.md` exists — good — but the many loose `create_*.sql` / `migrate_*.sql` files in the repo root partly duplicate `migrations/`. Document (or enforce) which is authoritative.

## Housekeeping

- **Remove `products_point_one_percent_sample.csv` (5.7 MB) from the repo.** The project explicitly no longer reads CSVs; the file bloats every clone. Delete it (and `extract_products_from_shopifyscrape*.sql` if they're equally dead), or move to a data release. Consider `git filter-repo` if clone size matters.
- Move the ~14 root-level `*.sql` files into `padjective/sql/` or `migrations/` so the root directory shows structure at a glance.
- `.DS_Store` is present in the working tree; confirm it's gitignored (add `**/.DS_Store` if not).
- Dependencies: `uv.lock` was last touched 2026-02-11 — run `uv lock --upgrade` and re-run tests; five months of torch/sklearn fixes have shipped since.

## Security

- No committed credentials found: DSNs come from `PADJECTIVE_SHOPIFY_DSN` env var, which is the right pattern. Keep it that way; ensure the cron box's env file is outside the repo.

## Quick Wins

1. Fix the `PADJECTIVE_PAPER_AS_OF` default in `cronscript.sh` (one line).
2. Delete `README2.md` and the sample CSV.
3. Add `.DS_Store` to `.gitignore` if missing.
4. `uv lock --upgrade` + test run.
