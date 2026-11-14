# Manuscript checks, 1 October 2026

The audit is recorded in Shopify Postgres as
`padjective.paper_manuscript_audits`, audit
`7cac5a6f-cb7b-4426-8a0a-4daf72b86e3d`. The table explicitly uses
`pg_default`. `results.json` exports aggregates from that record. Product rows,
coefficients, member masks and product-level predictions remain in Postgres.

## Integer scores and finite-precision scores

The one-pass model uses an integer sum. Repeated-coordinate fits reduce the
sum modulo `71**7` before applying their zero-score default. For example,
coefficients `71**7-1` and `1` sum to a nonzero integer but to zero modulo
`71**7`; the repeated-coordinate predictor returns its fitted default.

The audit reconstructed both training and held-out raw and default-adjusted
predictions for 1,215 all-feature members and 1,215 75% members. Every saved
prediction matched the modular implementation. In 29,515 of 3,252,798
held-out member evaluations the integer sum was nonzero but its residue was
zero. These are repeated evaluations of the benchmark products across members,
not distinct products. The manuscript needs to state the modular definition;
this check gives no reason to change the fitted models or their scores.

## Active coefficient use and trees

The September 22 tree report includes both fractions at all nine ensemble
sizes, including 1, 3, 9 and 15 members. Its 1,690 fold/seed rows are identical
to the freshly retrieved Postgres report. The original report validated
2,262,234 prediction scores; this audit checked its aggregate identity and
recomputed its budget selections, without refitting or rechecking every tree
prediction again.

Best observed unweighted losses within each mean active-count ceiling:

| Ceiling | All-feature ensemble | 75% ensemble | Tree | Forest |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.196485 | 0.198949 | 0.525845 | 0.532483 |
| 30 | 0.159282 | 0.147371 | 0.434789 | 0.493108 |
| 100 | 0.150952 | 0.136581 | 0.298135 | 0.315573 |
| 300 | 0.144178 | 0.134655 | 0.090718 | 0.120642 |

The active metric counts member coefficients actually consulted or tree branch
decisions, addressing the member calculations needed for simulatability.
Stored slots measure a different resource. The active count excludes defaults
and consensus, and tree traversal excludes its leaf output; it is not a count
of every step in an entire prediction. The capacity frontiers are descriptive
selections on reused folds.

## Full labelled-catalogue follow-up

Nine validated 243-member comparisons exist, captured September 11–18, with
31,038–31,838 products. All 2,430 components per capture have completed
coordinate-optimum status. The audit independently recomputed equal-fold
losses, paired uncertainty and tests from saved private predictions, for all
products and those outside the paper snapshot.

Every capture favours the 75% variant in mean loss. The latest losses are
0.626980 versus 0.634456. Two overall pointwise tests have `p<0.05`; none
survives Holm correction over the nine captures (minimum adjusted
`p=0.269694`). None of the outside-paper subgroup tests survives its separate
nine-capture correction either.

These evaluations hold out stores, retain empty tags and rare paths, and
use the larger taxonomy with prime 83 and precision 8. They check the feature
fraction's observed direction and catalogue coverage. They do not repeat the
classical comparison, and strongly overlapping captures are not independent
confirmation datasets. No new models were fitted by this audit. An exact
September 2 cutoff was not evaluated: these are the dated completed captures
available in Postgres.

## Reproduction

Run `uv run -m padjective.paper_model_audit` in a configured checkout with
Shopify Postgres access. It emits only an aggregate JSON report and appends
that report to Postgres. For this audit the script was streamed to the existing
dedicated September checkout on raksasa; its SHA-256 is recorded in the
report. The existing model source revision is recorded separately.

Save the aggregate stdout, then compact it with:

```sh
uv run -m padjective.paper_model_audit_export aggregate-report.json docs/manuscript-audit/results.json
```

The compact export retains the original Postgres record's canonical JSON
digest and the tree rows' digest. The manuscript stores an identical copy at
`padjective/padic-journal/data/manuscript-audit-results.json`; its table
generator checks the file checksum, completed counts, fold arithmetic,
intervals and nine-comparison Holm corrections before rendering. The earlier
five-capture export remains historical evidence with its original cutoff.
