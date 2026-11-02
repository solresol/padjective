# Validation and reproduction

All 25 recovered model/fold results reproduce the archived Q=71 loss and exact
accuracy within 1e-12. The independent validator checks 33,465 held-out predictions
against frozen PostgreSQL product IDs, labels and fold order, recomputes all six
Q scores and their fold means/sample SDs, and verifies the exported classical
report equals its stored PostgreSQL report. See `validation.json`.

The neural iteration counts also reproduce exactly: folds 0–4 use
62, 135, 103, 89 and 109 iterations. The reference Python, package versions,
OpenBLAS implementations and 12-thread settings match. The prediction table
and its primary-key index resolve to `pg_default`. Every recovery worker has
exited; no fitting job remains active.

`uv run -m pytest -q`: 347 passed, six skipped, with the 12 existing batch-size
warnings. Tests include rejection of mismatched populations and snapshots,
separation of fixed and retrained comparisons, the width-2,000 neural reference,
and independent low-Q score examples. The complete ranking figure was rendered
and visually checked after correcting a clipped final axis label.

Recovery fits used source commit `240e2de7819485ea9c36745300d5468d54c6cc01`.
Collection and the independent PostgreSQL validator used source
`c16c22d9471ee0372ba440c6eb2c871f91dfbf92`; the fitting implementation and runtime
files are identical in those two commits. The source hashes and full reference
environment are retained in `results.json` and `reference.json`. Execution used
isolated source archives on raksasa, preserving its unrelated checkout and jobs.
The original native-model report remains unchanged.

The combined tables, paired comparisons and figure can be regenerated from the
public aggregate files, without PostgreSQL or product data:

```sh
uv run python -m padjective.paper_classical_q_results \
  --native docs/low-q-sensitivity-2026-11-02/results.json \
  --classical docs/low-q-classical-2026-11-02/results.json \
  --output docs/low-q-classical-2026-11-02
```

Recovering predictions requires the frozen reference environment and PostgreSQL
snapshot, as described in `protocol.md`. `validate_postgres.py` runs with the
archived source root on `PYTHONPATH`. Private product-level predictions remain
in PostgreSQL; only aggregate evidence is exported here. These reused folds
provide a descriptive sensitivity comparison. Point-estimate rank changes do
not establish statistical significance.
