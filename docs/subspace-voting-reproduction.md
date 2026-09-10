# Reproducing the feature-subset and voting study

The [fixed protocol](subspace-voting-protocol.md) defines the experiment; the
[theory notes](subspace-voting-theory.md) distinguish expressivity from fitting
and generalisation. The experiment is isolated from the production taxonomy
pipeline and from the paper's frozen 10 September release.

## Inputs and environment

Run on raksasa, reading the frozen snapshot from the Shopify `shopifystores`
Postgres database with the existing `padjective.db` connection configuration.
No CSV or SQLite path is supported. The dataset digest must match the protocol.
The previously validated 1,215-member all-feature bank and its report must
already exist in Postgres. Original model and ensemble tables are read-only.

Use `uv sync --frozen`. The executed fits use Python 3.11.11 and NumPy 2.2.5.
Set BLAS/OpenMP thread limits before launching Python. Four workers are the
maximum allowed by the runner. New fits, their private predictions and all
evaluation evidence are stored in the four `paper_subspace_*` tables, with
explicit `pg_default` tablespaces for tables and indexes.

## Fit, validate and export

The training source was frozen at `2ca78551e5a7a2f3ecae4fb29929a2b6274398f0`.
Run from a clean dedicated checkout; do not alter a running worktree. The runner
checks that the original numerical fitting files remain byte-identical to their
validated baseline. Its first output records a new batch UUID and the protocol
hash before any new fit is made.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run -m padjective.paper_subspace_experiment --workers 4
```

The active study's batch is `773bf2ab-d2b1-4c32-bfde-15d501782fa0`. Once every
one of the 4,860 new fits has finished with an independent coordinate-optimum
certificate, evaluate from a separate clean checkout containing the evaluator:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run -m padjective.paper_subspace_evaluate \
  --batch-id 773bf2ab-d2b1-4c32-bfde-15d501782fa0 \
  --output build/subspace-20260911/results.json
```

Evaluation refuses a partial bank. It reconstructs every component's fitting
and held-out predictions, verifies masks/seeds/hashes/defaults, checks all
metrics two ways, compares all 45 original primary predictions with the
reference bank, and checks fixed product samples using a separate aggregation
implementation. Every resulting private prediction set is read back from
Postgres and rescored. The export contains aggregate evidence only.

The 6,125 evaluation rows comprise 1,125 primary-grid fold rows and 5,000 roster
sensitivity rows. Only five paired fold values enter each statistical test.
The report records both training and evaluator commits. Re-evaluation can
resume exact partial evaluation rows, but existing aggregate exports are never
overwritten. Investigate failed fits rather than selecting a surviving subset.

```sh
uv run -m padjective.paper_subspace_figures \
  --results build/subspace-20260911/results.json \
  --output-directory build/subspace-20260911
uv run -m pytest -q
```

The figure exporter follows [the figure contract](subspace-figure-contract.md).
Inspect the PDF/PNG outputs before sharing. None of these commands changes the
main manuscript, its original active-support regression, or the frozen release.
