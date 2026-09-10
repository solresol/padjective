# Live subspace experiment: operation and provenance

The experiment is isolated from the nightly production models. Its design is
fixed in [the protocol](live-subspace-protocol.md). Do not pull new source into
a fitting worktree or change the protocol in response to intermediate scores.

## First input capture

At 06:37 AEST on 11 September 2026, the repeatable-read capture retained
31,038 products from 1,989 stores and 2,954 reference categories. This includes
all 6,693 paper-snapshot identities and 24,345 other products. Of the original
31,204 source rows, 76 were older copies of a store/handle and 90 had no usable
reference classification. Fourteen duplicated entities had conflicting
historical labels; the latest row was retained. There were 7,551 empty tag
sets, which remain in evaluation.

The source labels are all `gold_llm`. These results measure agreement with that
LLM reference. They do not measure accuracy on the approximately 49.9-million-
record unlabelled scraper archive.

- Cohort: `28c3ec83-5bd4-414b-b207-d95ae98af6ac`
- Content digest: `aa232cca42a7ede32ef767b283af2055088d09ba9e361182d529b8aaca63825d`
- Store-held-out fold sizes: 5,700 / 6,078 / 6,686 / 6,266 / 6,308.
- The first engineering smoke batch was
  `3be31bb3-ba4b-4916-a8a0-ba6eff76d8e7`, at source `296b6ea`.
  All 30 component fits and all 31,038 paired predictions passed validation.
  Its three-member scores are not the 243-member experiment result.
- All six new tables and their seven indexes were checked in PostgreSQL:
  each resolves to `pg_default`.

## Commands on raksasa

Dedicated checkout: `/home/gregb/tmp/padjective-live-subspace-20260911`.
The existing `/home/gregb/devel/padjective` production checkout is not changed.
Use the existing Shopify Postgres connection; never copy credentials or
product records to local files.

```sh
cd /home/gregb/tmp/padjective-live-subspace-20260911
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Capture input only, with a dated observation even if content is unchanged.
/usr/local/bin/uv run -m padjective.live_subspace capture

# Fit/validate the frozen first cohort (2,430 members: 2 variants x 5 folds x 243).
/usr/local/bin/uv run -m padjective.live_subspace run \
  --cohort 28c3ec83-5bd4-414b-b207-d95ae98af6ac --members 243 --workers 4

# Subsequent daily observation: capture, skip identical completed data, otherwise fit.
/usr/local/bin/uv run -m padjective.live_subspace cycle --members 243 --workers 4

# Compact progress, without private predictions or product information.
/usr/local/bin/uv run -m padjective.live_subspace status
```

Run long fits in a durable background process with a fresh dated log under
`experiment-logs/`; do not leave the job dependent on an interactive SSH
connection. Never run two fitting processes: a PostgreSQL advisory lock rejects
duplicates. The runner resumes an identical unfinished batch on the same
cohort, source and protocol. It saves each certified member separately. Do not
delete a failed batch or construct an ensemble from only completed members.

Treat `validated` on `live_subspace_batches` together with an exact 2,430
member count and a read-back `live_subspace_reports` row as completion.
`report.status` must be `validated_rolling_cv`; an engineering smoke report
does not qualify. The report includes exact aggregate counts, primary paired
fold comparisons, overlap/non-overlap and zero-feature subgroups, model
digests and unchanged-common-product trend comparisons. Private paired
predictions remain in the same Postgres database.

The first successful full comparison is a baseline, not yet a time trend.
Later observations separate unchanged common products, changed products,
arrivals and removals. Daily retraining is not a fixed-model prospective test;
do not present repeated daily p-values as independent confirmation.

Stop and report a failed member, missing label provenance, encoding overflow,
unexpected source growth beyond 100,000 rows or a dirty/drifted fitting checkout.
Do not broaden the source query to the entire unlabelled archive, initiate new
LLM labelling, overwrite production model tables or insert these variants into
the paper's original log-log regression.
