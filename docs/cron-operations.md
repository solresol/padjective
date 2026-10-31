# Nightly cron output and operational evidence

## Execution and output policy

The Raksasa service account `padjective` runs
`/home/padjective/padjective/cronscript.sh` at **05:30 Australia/Sydney daily**.
Greg's own crontab has only an obsolete Padjective comment. The dedicated
service checkout tracks `main`; the research checkout and pinned experiment
worktrees are separate. Do not move execution to another computer.

`cronscript.sh` enters the standard-library `padjective/cron_output.py`
supervisor before Git/uv startup, then runs the original pipeline once. The
supervisor changes output handling only:

- Routine stdout goes to a fresh private log under
  `/home/padjective/logs/padjective-nightly/`. `PADJECTIVE_CRON_LOG_DIR`
  overrides this directory for a test or manual invocation.
- Stderr is retained in the same log and forwarded to cron immediately.
  Python warnings, traceback/SQL exceptions, Git/uv/SSH/rsync errors and
  sparse-taxonomy fold-fallback warnings remain visible.
- A nonzero exit keeps its exit code and adds the log path and a bounded
  diagnostic tail (at most the final 20 lines from the last 4 KiB).
  Stdout-only explanations are therefore available on failure too.
- Each log has an adjacent private JSON receipt: start time, supervisor PID,
  finish time, child exit code and `running`, `succeeded`, `failed` or
  `logging_failed` status. A successful receipt means command exit 0, not an
  independent scientific or publication validation. An interrupted supervisor
  can leave a `running` receipt; check the recorded PID and cron journal.
- Log/receipt setup failure stops before starting the workload and alerts on
  stderr. Detected completion-write failures also alert and make the wrapper
  fail. No credentials, environment dump or extra command-line logging is added.
- Directories newly created for logs are mode 0700; logs and receipts are 0600.
  The child's working directory, arguments, umask, training parameters,
  ordering and publication destinations are retained. No lock, rerun, new
  experiment, database migration, schedule change or automatic deletion is added.

Four best-effort site-generation exceptions historically printed `Warning:`
on stdout. They now print on stderr, preserving alerts for failed SQL dumps,
prediction-detail pages and tag-debug pages even if the site build returns 0.

The `PADJECTIVE_CRON_LOGGED` environment marker prevents supervisor recursion;
leave it unset when invoking the normal shell entry point.

## What is in PostgreSQL

Read-only checks on **31 October 2026** confirmed the real service account
uses local **PG16/5432, database `shopifystores`, role `padjective`**. A
`shopifystores` copy also exists on PG18/5433; its presence is not evidence of
a cutover. Routing and credentials were not changed.

The nightly fold metrics, predictions and model outputs are persisted in
`padjective.umllr_*`, `taxonomy_pclr_*`, `taxonomy_pcnn_*`,
`taxonomy_ulr_*`, `taxonomy_unn_*`, `taxonomy_dt_*` and `zubarev_*`.
`model_performance_history` preserves a daily metric/dataset snapshot.
`taxonomy_path_reconciliation_audits` persists reconciliation and eligibility
counts, including unresolved path counts/IDs. Benchmark snapshot membership
and export inputs are in `product_taxonomy_bench_*`.

Today's evidence includes PCLR model **393**, five rows for each classifier's
fold results, the daily snapshot at **11:10:03 Sydney**, and a reconciliation
audit at **05:30:52 Sydney** with **0 unresolved path products** and **[]
unresolved taxonomy IDs**. The daily snapshot records **24,511 products**;
PCLR's earlier fit used **24,415**. These are live reads, not one fixed cohort.

ULR folds 0–2 recorded 7,927 loaded tags; folds 3–4 recorded 7,942. Their stored
train/test totals stayed 24,415 after filtering out missing fold assignments.
UNN's stored train/test totals became 24,511: unlike ULR/DT, its train mask
includes rows without a held-out fold assignment. DT filtered back to 24,415.
This explains the observed count pattern without establishing full input
identity or a fair matched-cohort comparison. No model or source-data behaviour
was changed as part of reducing cron output.

Stored fold assignments do not explicitly identify whether their generator
selected StratifiedKFold or deterministic KFold. Do not silence that warning
unless the split method is first recorded reliably. The current code calls
the shared helper with its existing warning settings; this change adds no
warning filter and does not claim a warning was emitted today.

The PCNN SQLite default is a legacy **full-model** metadata path. Nightly uses
only `--fold` PCNN runs, which persist in PostgreSQL and return before the
SQLite-saving branch. No `data/taxonomy_pcnn_classifier.sqlite` was present in
the service checkout at inspection. It is not an active nightly data gap.

## What still needs operational tracking outside those tables

- End-to-end Git/uv setup, complete stdout/stderr, failures and interruption,
  plus rsync/publication completion, have no equivalent nightly database run
  ledger. The private logs/receipts preserve this evidence. Database metrics
  may be saved before later site generation or upload fails.
- The separate **03:00 Sydney on the first of each month** Hugging Face
  publisher already appends both streams to
  `/home/padjective/logs/product-taxonomy-bench-publish.log`. Its upload result
  is operational state, not proof supplied by a model table. Last inspected
  log modification was **1 October 03:00 Sydney**; no error-marked lines were
  found. Its existing redirect means failures are log-only, so inspect that
  log when checking monthly publication. That wrapper, lock, schedule and
  credential handling were left unchanged; no extra upload was invoked.
- PG16's existing collation-version mismatch (database 2.39, OS 2.43) appeared
  in psql probes. This belongs to the separate database maintenance/migration
  work; it was neither fixed nor suppressed here.

The 31 October daily cron journal shows launch at **05:30:01 Sydney** and
session close at **13:27:51**. The local site index was updated at **12:39:26**.
A cron session close and local file timestamp alone do not prove remote
publication success. Inspection found no active Padjective, experiment or
publication process and no other client session on this database.

All inspected database experiment status columns were terminal: 11 validated
live-subspace batches, 6 validated residual batches, a validated reference
fraction sweep, a complete tree batch and completed/limited historical method
runs. One Salazar failure is historical (2 October); it is stored alongside
two completed runs, not a currently running job requiring separate tracking.

## Verification without training or publication

Run `uv run -m pytest -q tests/test_cron_output.py` for tiny subprocess tests.
They cover quiet success; retained warning/traceback output; original failure
codes; stdout-only errors; missing executables; log/receipt failures; private
file modes; and the real shell entry point with **all Git, uv and rsync calls
replaced by stubs**, including abort on the first failed Git call. Export
exception tests use fake dependencies and no database.

The full repository suite passed **155 tests, 6 skipped**, with 12 existing
small-batch sklearn warnings. The focused cron/site/CV selection passed 23.
No expensive nightly or research workload was run to test this change.

After deployment, compare the installed source hashes and unchanged service
crontab; a service-account smoke should invoke the supervisor with a tiny
command in a separate temporary log directory, never the actual pipeline.
The next ordinary scheduled run provides the first real nightly receipt.
