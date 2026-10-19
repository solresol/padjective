# Distinct valuations versus distinct occupied powers — 19 October 2026

The two conditions select different predictions. In this new saved-fit analysis,
50 multi-active predictions have distinct valuations; only 10 have disjoint
canonical seven-digit supports. Valuation separation is associated with lower
loss even when higher digit supports overlap. The disjoint subgroup has still
smaller wrong-only loss, but its exact-hit rate is 40% versus 37.5% for the 40
valuation-separated, overlapping cases. Ten cases across four taxonomies do
not establish a broad accuracy advantage for the stronger condition.

This is **new exploratory evidence from saved September checkpoint fits**.
It is not a reconstruction or replacement of the published 102-versus-1,139
strict-refinement table. No manuscript, original result or fitted model was
modified. Exact hits mean agreement with recorded reference codes, not newly
adjudicated product truth. No causal claim is made.

## Primary comparison

All groups below use the same 1,128 held-out products with at least two
nonzero active contributions, and exactly the same predictions and outcomes.
Disjoint support is a subset of valuation separation at the primary depth.
The exclusive partition is 10 disjoint, 40 valuation-separated with overlap,
and 1,078 repeated-valuation cases.

| Group (multi-active only) | n | Exact hits | Exact-hit rate | Mean loss | Loss among incorrect | Taxonomies |
| --- | --- | --- | --- | --- | --- | --- |
| Distinct valuations (includes disjoint) | 50 | 19/50 | 38% | 0.0614683 | 0.0991424 | 31 |
| Repeated valuations | 1078 | 101/1078 | 9.3692% | 0.865756 | 0.955256 | 207 |
| Disjoint occupied powers, depth 7 | 10 | 4/10 | 40% | 6.03502e-05 | 0.000100584 | 4 |
| Overlapping occupied powers, depth 7 | 1118 | 116/1118 | 10.3757% | 0.837529 | 0.934489 | 209 |
| Distinct valuations, overlapping powers | 40 | 15/40 | 37.5% | 0.0768203 | 0.122912 | 30 |

Every valuation-separated and every disjoint case has exactly two active
contributions. For a same-count descriptive comparison, the other
760 two-contribution cases have 2 exact hits
(0.2632%), mean loss 0.983043,
and wrong-only mean loss 0.985637.
This controls the addition count alone, not taxonomy composition or difficulty.

Forty of the 50 valuation-separated cases have overlapping digit supports;
20 produce carries. None of the 10 disjoint cases produces a carry.
The 30 multi-contribution zero residues all have nonzero integer sums and
use the fitted fallback after reduction modulo `71**7`. None is
valuation-separated. Defaults are not counted as tag contributions.

## Zero and one contribution cases

Both predicates hold vacuously for zero or one contribution. These groups are
kept out of the primary comparison because their outcomes differ sharply.

| Active contributions | n | Exact hits | Exact-hit rate | Mean loss | Loss among incorrect | Taxonomies |
| --- | --- | --- | --- | --- | --- | --- |
| Zero | 1053 | 58/1053 | 5.50807% | 0.493014 | 0.521753 | 245 |
| One | 4512 | 3185/4512 | 70.5895% | 0.0589947 | 0.200591 | 353 |
| Two or more | 1128 | 120/1128 | 10.6383% | 0.830105 | 0.928927 | 209 |
| All | 6693 | 3363/6693 | 50.2465% | 0.257237 | 0.517023 | 363 |

The pooled all-product loss weights products. Equal weighting of the five
fold means is a different aggregation; fold rows are preserved below and in
the JSON. Correct predictions contribute zero to loss, and wrong-only means
exclude them. A dash denotes an empty group or no incorrect cases.

## Fold-specific joint groups

| Fold | Joint group | n | Exact hits | Mean loss | Loss among incorrect |
| --- | --- | --- | --- | --- | --- |
| 0 | Disjoint | 1 | 1 | 0 | — |
| 0 | Valuation-only | 14 | 5 | 0.00308916 | 0.00480535 |
| 0 | Repeated valuations | 212 | 18 | 0.891643 | 0.974373 |
| 1 | Disjoint | 2 | 0 | 0.000198373 | 0.000198373 |
| 1 | Valuation-only | 9 | 3 | 0.112764 | 0.169146 |
| 1 | Repeated valuations | 223 | 17 | 0.888148 | 0.961441 |
| 2 | Disjoint | 5 | 1 | 4.13511e-05 | 5.16888e-05 |
| 2 | Valuation-only | 9 | 4 | 0.00156557 | 0.00281803 |
| 2 | Repeated valuations | 201 | 22 | 0.851099 | 0.955703 |
| 3 | Disjoint | 1 | 1 | 0 | — |
| 3 | Valuation-only | 4 | 0 | 0.250149 | 0.250149 |
| 3 | Repeated valuations | 212 | 17 | 0.849528 | 0.92359 |
| 4 | Disjoint | 1 | 1 | 0 | — |
| 4 | Valuation-only | 4 | 3 | 0.25 | 1 |
| 4 | Repeated valuations | 230 | 27 | 0.847951 | 0.960733 |

The source folds have overlapping training sets, so these are not five
independent replications. In particular, disjoint exact-hit rates vary from
zero to one with only one to five disjoint observations per fold.

## Taxonomy composition

The full JSON preserves every group's metrics and counts for all 363
reference taxonomies. The four represented in the disjoint subgroup are:

| Reference taxonomy | Disjoint n | Disjoint exact | Disjoint wrong-only loss | Valuation-only n | Valuation-only exact |
| --- | --- | --- | --- | --- | --- |
| Dresses (aa-1-4) | 3 | 0 | 2.79399e-06 | 3 | 3 |
| Chairs (fr-7) | 1 | 0 | 0.000198373 | 0 | 0 |
| Household Appliances (hg-9) | 2 | 0 | 0.000198373 | 1 | 0 |
| Cosmetic & Toiletry Bags (lb-3) | 4 | 4 | — | 1 | 0 |

All four disjoint exact hits are Cosmetic & Toiletry Bags; the six disjoint
errors occur in the other three taxonomies. All six errors preserve the full
reference path and add deeper digits: the three Dresses errors have error
valuation 3 (reference depth 3); the Chairs and two Household Appliances
errors have error valuation 2 (reference depth 2). The latter three outputs
are other valid frozen label codes, while the three Dresses outputs are not
among the frozen labels. This supplemental read-only audit used the private
row evidence identified below. The very small losses therefore also reflect
reference-label granularity, not just abstract arithmetic structure.

The groups have different taxonomy coverage (4 for disjoint support, 30 for
valuation-only overlap, 207 for repeated valuations). These summaries do not
adjust for taxonomy or feature quality. Neither the observed loss difference
nor the 2.5-percentage-point exact-hit difference between the two separated
subgroups establishes that imposing disjoint support would improve a model.

## Depth sensitivity

Depth 7 matches the saved quotient and maximum reference-path depth. The
prediction, target, loss and valuation predicate remain unchanged throughout;
only the support predicate is truncated. “Fully visible” excludes rows with
any nonzero contribution vanishing under that truncation.

| Depth | Multi rows with hidden contributions | Disjoint n | Disjoint exact | Disjoint mean loss | Fully visible disjoint n |
| --- | --- | --- | --- | --- | --- |
| 1 | 113 | 53 | 20 | 0.0579964 | 0 |
| 2 | 95 | 47 | 19 | 0.0222435 | 4 |
| 3 | 27 | 24 | 8 | 0.00180259 | 7 |
| 4 | 4 | 14 | 5 | 0.00104915 | 10 |
| 5 | 0 | 10 | 4 | 6.03502e-05 | 10 |
| 6 | 0 | 10 | 4 | 6.03502e-05 | 10 |
| 7 | 0 | 10 | 4 | 6.03502e-05 | 10 |

The primary disjoint membership is identical at depths 5, 6 and 7. All stored
coefficients are canonical nonnegative residues below `71**7`, so inspecting
additional digits of those representatives changes nothing. This does not
claim disjoint infinite supports for other signed lifts. The shallower rows
show why invisible contributions must not silently be treated as evidence of
separation. Signed digit extraction uses reduction modulo `71**D`, never
digits of `abs(coefficient)`.

## Reproduction and provenance

See [the full protocol](../refinement-comparison-protocol.md) and
[the analysis code](../../padjective/paper_refinement_comparison.py).
The runner reads the frozen feature relation used by the saved fits, including
the original pre-anonymisation deduplication and nested-tag filtering. It
does not reparse current raw product tags.

- Snapshot: `244ddbe3-0a2c-4c04-9436-0a0253108a09`; 6,693 products, 2,542 features, 363 labels.
- Ordered snapshot digest: `039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
- Saved-fit batch: `a2b0744f-175e-4899-a3d5-e6a8478a8d73`; model key `linear_association/one_pass`.
- Fitter source: `289f5498c498c2c56eb13e31f29fa9958e049df1`.
- Model: first complete association-ordered sweep of the exact finite-quotient
  coordinate fitter, not the published floating-point greedy model or the
  repeated-sweep endpoint. The protocol was chosen before inspecting outcomes.
- Analysis code commit: `759b3d9fb1c958be21a1e99ee7064d13ffb8a782`.
- Analysis source SHA-256: `d7723f3f2f7606af0a0c9c7aa84291a9e4c7f6cd1b1cdb91db60a882fa8ca41f`.
- Aggregate JSON: [results-2026-10-19.json](results-2026-10-19.json), SHA-256
  `2bf1df1e2870323fd5d1fc3c64f3256dee8dc8dcb0c7879b23d983e61f91eeb3`.
- Private PostgreSQL analysis ID: `291dae98-735f-4ad1-aede-395deecbd2d7`, in
  `padjective.paper_refinement_comparisons` (`pg_default`).
- Private row/contribution evidence SHA-256:
  `3dcd60e2f1337c7330c389ccf526cad835c149f45ec5bfc1b42c86fdf439a7a2`. Product keys, individual predictions
  and contribution lists remain in PostgreSQL; the checked-in JSON is aggregate.

All five saved coefficient vectors, feature orders and snapshot digests were
checked. Integer reconstruction matched all 26,772 saved training predictions
and all 6,693 held-out predictions. The training-only default was independently
recomputed in every fold, and independently scored held-out metrics matched
the archived metrics. Every product occurs in exactly one held-out fold.
Per-model coefficient, prediction, configuration and evidence hashes are in
the aggregate JSON. The read-only database audit plus summary computation
took 2.657 seconds after imports; no model was trained.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync -m padjective.paper_refinement_comparison \
  --persist --output /tmp/refinement-comparison-new-run.json
```

The runner refuses existing output paths. Reruns append a new isolated
analysis record; original evidence is never updated. Timestamps, analysis IDs
and elapsed times vary; the numerical results and frozen input/model digests
must agree. The source code and protocol explain zero handling, finite
precision, canonical signs and the two mathematical predicates.

A second, read-only [replay auditor](../../scripts/audit_refinement_comparison.py)
rechecked all 6,693 private contribution rows, all five original model hashes,
fallbacks and every published fold/taxonomy/depth aggregate. It also calculated
the disjoint error-depth diagnostic above. Its complete aggregate record is
[audit-2026-10-19.json](audit-2026-10-19.json); status: **passed**.

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync python scripts/audit_refinement_comparison.py \
  --report docs/refinement-comparison/results-2026-10-19.json \
  --output /tmp/refinement-replay-new-check.json
```

This check requires the original private database but makes no database writes.
Its JSON binds the code hash, primary public-report hash and private evidence
hash; no product-level values are exported.

Validation: **290 tests passed, 6 skipped** under the existing Python 3.11
environment, including 10 new tests of counterexamples, carry propagation,
negative canonical digits, truncation, empty groups, exact/wrong-only scoring,
exhaustive small-ring implications and saved-prediction/default tampering.
The suite reported its existing 12 small-sample neural batch-size warnings.

## Interpretation and next evidence

The result supports an association between valuation separation and both
exact reference agreement and closer errors in this saved-fit population.
It also establishes that most valuation-separated cases are not disjoint in
their higher digit support. Disjoint supports select a much smaller subgroup
with low error severity; its incremental exact-hit advantage over overlapping
valuation-separated cases is not established by these ten concentrated rows.

A confirmatory comparison would freeze a larger independent cohort, retain
both predicates and wrong-only loss in advance, and compare within taxonomy
and active-count strata. Testing an intervention that forces either condition
requires separately fitted models and a held-out evaluation. No such training
or manuscript revision was performed here.
