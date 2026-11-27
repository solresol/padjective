# Frozen-2 confirmation of the priority-list results

**Draft, 27 November 2026. Not yet frozen.** Nothing here binds until the
decisions in the last section are settled and this file is committed at the
tagged run commit.

## Purpose

The priority lists and the level-by-level lists were designed after looking at
held-out scores on the paper snapshot
(`244ddbe3-0a2c-4c04-9436-0a0253108a09`). The manuscript says so, and says
they need an independent test. This run is that test.

It fits a fixed list of models once, on products and stores that took no part
in designing them, and tests a fixed list of comparisons. Frozen-2 can confirm
each claim once. Anything tried on it afterwards is exploratory.

## What these data have already been used for

- **Summing and classical models:** the one-pass sum and the classical
  learners are refitted nightly on the growing catalogue, and their scores
  have been seen.
- **Ensembles of sums:** the 75% and all-feature ensembles were compared on
  three store-grouped live cohorts in September
  ([results](subspace-voting-results.md)).
- **Selecting models:** every `paper_priority_list*.py` run and the
  minimum-valuation runs read only the paper snapshot. Confirm before the
  freeze that no ad hoc run has fitted one on other products.
- **Settings:** the classical hyperparameters, the 75% feature fraction and
  the tag order were chosen on the paper snapshot. They are frozen here.
- **Planning counts:** on 27 November the pool was counted from labels, tags
  and store membership ([counts](frozen2-planning-counts-2026-11-27.json)).
  No model was fitted. The counts informed the population, the reading rule
  for family B and the cutoff.

The run is therefore a clean test of the selecting models and of their
comparisons with everything else. The absolute scores of the summing and
classical models on later data are not new information.

## Population

Use the paper's snapshot builder, `product_taxonomy_bench.create_snapshot`,
with the paper's settings: nested-tag filtering, paths with at least five
products, then tags on at least five of those products, and a product kept if
it has a retained path and at least one retained tag. As in the paper, both
counts use the whole snapshot pool, including products later held out.

Three restrictions are new:

- **Reference labels only.** Keep a product only if its label has
  `taxonomy_source = 'gold_llm'`. The builder's source query does not filter
  on this today; `live_subspace.py` does. All 38,385 labels in the source
  table had this source on 27 November, so the filter is a guard.
- **No paper products.** Drop every product whose hash is in the paper
  snapshot.
- **No paper stores.** Drop every product from a store that has a product in
  the paper snapshot, using the store membership of
  `paper_matched_store_folds.store_membership`.

The eligibility counts are taken after these restrictions.

**Cutoff.** The cutoff is on label time: keep a product only if its label has
`classified_at` at or before the cutoff. The builder's `--as-of` option cannot
do this. It filters on `product_details.when_fetched`, and every product was
fetched by 20 January 2025, before the first label was written in October
2025. The pool grows because products are labelled, not because they are
fetched. A product that is relabelled after the cutoff drops out.

If the pool has fewer than 6,693 products after filtering, the third
restriction is dropped and the pool is every non-paper product. The
new-store products are then reported as a subgroup. This rule depends only on
counts and is applied before any fit. At the planning counts below the pool
has 13,368 products, so the rule does not apply.

Identity is the canonical product URL. The same goods listed by two stores
count as two products, as they do in the paper.

## Planning counts, 27 November 2026

These are not the preflight. They were taken read-only on Raksasa by applying
the builder's eligibility rules inside each pool. With no exclusions the same
code reproduces the nightly snapshot `latest-2026-11-27T0028Z` exactly
(23,314 products). No label was newer than 27 November 00:00 UTC.

| | Paper snapshot | New stores only | Every non-paper product |
|---|---:|---:|---:|
| Products | 6,693 | 13,368 | 16,314 |
| Stores | 395 | 790 | 988 |
| Paths | 363 | 625 | 773 |
| Tags | 2,542 | 3,783 | 4,435 |
| Prime | 71 | 79 | 83 |
| Precision | 7 | 8 | 8 |
| Paths shared with the paper snapshot | | 222 | 254 |
| Products on paths the paper snapshot lacks | | 5,301 | 6,537 |
| Store-holdout fold sizes | | 2,399 to 2,835 | 3,020 to 3,643 |
| Held-out products on unseen paths, store holdouts | | 10% to 18% | 9% to 13% |
| Largest store | | 239 | 239 |
| Share of products in the ten largest stores | | 15% | 12% |

- **Frozen-2 is a larger task, not more of the same task.** Two in five of
  its products are on paths the paper snapshot lacks, because more paths
  reach five products in a larger pool. Absolute losses will not be
  comparable with the paper snapshot's. The comparisons are.
- **Almost all of the pool was labelled after the paper snapshot was built**
  on 26 February 2026. In the new-store pool, 208 products were labelled
  before March.
- **Labelling adds 1,400 to 2,000 eligible non-paper products a month.**
  From March to October, 1,359 to 1,851 a month were from new stores. In
  November to the 26th, 274 of 1,633 were.
- **Stores are small.** The median store has three products in either pool.

## Splits

- **Product holdouts:** the five folds that the builder records for each
  product (`cv.calculate_cv_folds`, seed 42). The builder assigns them over
  every labelled product in the source table, stratified by path, so the
  folds inside the pool are only roughly equal in size.
- **Store holdouts:** five folds from `live_subspace.store_fold`, the SHA-256
  of the normalised myshopify domain.

Feature ordering, fitting, defaults and consensus candidates use four folds.
The fifth is scored. Held-out products whose reference path is absent from
the fitting folds stay in the denominator.

## Encoding and loss

The prime is the smallest prime above the largest sibling index in the
frozen-2 reference paths, and the precision is the depth of the deepest path.
Both are fixed by the preflight. At the planning counts they are 79 and 8 for
the new-store pool, against 71 and 7 in the paper. The primary loss is mean
$d_{p,2}$. Mean
$d_{p,p}$, root accuracy and exact accuracy are secondary. Lists are fitted
at weight $p$, as in the paper.

Each fold gets a fold mean, and the five fold means are weighted equally.

## Models

Nine primary rows, the same as Table 1 of the manuscript. No setting is tuned
on frozen-2.

| Row | Definition | Paper snapshot, products / stores |
|---|---|---|
| Constant | medoid of all fitting labels | 0.6931 / 0.6940 |
| One-pass sum | taxonomy-association order, `paper_sum_oracle.term_rows` | 0.3283 / 0.7128 |
| Minimum-valuation medoid | same coefficients, `paper_valuation_selector.select` | 0.2340 / 0.6594 |
| 243 sums, 75% features | `paper_matched_store_folds.ensemble_job`, `BANK_SEEDS`, 100 sweeps, 300 s, valid-path consensus | 0.2142 / 0.6736 |
| Backfilled priority list | taxonomy-association order, `paper_priority_list.list_rows`, variant `cell_backfill` | 0.1516 / 0.6420 |
| Level-by-level lists | split at every level, `paper_priority_list_levels` variant `levels_all` | 0.1417 / 0.6241 |
| L1 logistic regression | `make_estimator("ulr")`: saga, C = 1, 200 iterations, seed 42 | 0.1535 / 0.6708 |
| Neural network | `make_estimator("unn")`: width 2,000, alpha 1e-4, batch 256, seed 42 | 0.1418 / 0.6539 |
| Random forest, 81 trees | `paper_store_held_out.forest_predictions`, the same seeds averaged within folds | 0.1405 / 0.6753 |

Secondary rows, reported without tests: the plain and pooled priority lists
(by-products of `list_rows`), and L1 logistic regression run to convergence,
because the paper's fit stops at its 200-iteration cap.

For every row, also record the mean number of coefficients, decisions or
weights consulted per prediction, the number of stored coefficients, and the
share of held-out products that fall to the default. The level-by-level model
has no such measurements on the paper snapshot.

## Comparisons

All tests are two-sided on the five paired fold differences in $d_{p,2}$,
with `subspace_voting.corrected_test` and `holm`, as in the paper.

**A. Selecting against summing, product holdouts.** Holm over four tests.
A comparison is confirmed if the mean difference is negative and the adjusted
$p$ is below 0.05.

| | Difference | Paper snapshot |
|---|---|---:|
| A1 | backfilled list − one-pass sum | −0.177 |
| A2 | backfilled list − minimum-valuation medoid | −0.082 |
| A3 | backfilled list − 243 sums, 75% features | −0.063 |
| A4 | level-by-level − backfilled list | −0.010 |

**B. Against the classical models, product holdouts.** The claim is that the
selecting model is not worse than the classical model by more than a margin
of 0.01. Report each mean difference with its two-sided 95% interval, and
read it as:

- **lower** if the two-sided interval lies below zero;
- **not worse** if the one-sided 95% upper bound is below +0.01;
- **higher** if the two-sided interval lies above zero;
- **unresolved** otherwise.

A difference can be both higher and not worse, and is then reported as both.
The three readings are not adjusted for multiplicity, and the manuscript says
so.

| | Difference | Paper snapshot |
|---|---|---:|
| B1 | backfilled list − L1 logistic regression | −0.002 |
| B2 | level-by-level − neural network | −0.000 |
| B3 | level-by-level − random forest | +0.001 |

The first draft read B as "level" when the whole two-sided interval lay
within ±0.01. The planning counts show that this pool cannot give that
reading. On the paper snapshot the standard errors of such differences were
about 0.005 to 0.0065, back-calculated from the reported $p$ values. If the
standard error falls as the square root of the number of products, the
new-store pool gives about 0.0035 to 0.0046. With four degrees of freedom:

- the two-sided half-width is 0.010 to 0.013, which leaves almost no room
  inside ±0.01 even when the mean difference is zero;
- the one-sided upper bound is the mean difference plus 0.0075 to 0.0098, so
  "not worse" can be reached when the mean difference is near zero or
  negative;
- an adjusted one-sided bound over three readings would add about 0.011 to
  0.015 and could not be reached.

The scaling is rough, because frozen-2 has more paths than the paper snapshot
and its fold variance may differ. For B2 and B3 the reading is likely to be
close either way.

**C. Against the classical models, store holdouts.** Holm over two tests,
confirmed as in A.

| | Difference | Paper snapshot |
|---|---|---:|
| C1 | level-by-level − neural network | −0.030 |
| C2 | level-by-level − L1 logistic regression | −0.047 |

Everything else is descriptive.

## Procedure

1. **Write and check the code on the paper snapshot only.** The runner must
   reproduce the eighteen paper-snapshot values in the model table before it
   is pointed at anything else.
2. **Freeze.** Settle the decisions below, remove the draft notice, tag the
   commit.
3. **Build and preflight.** Build the snapshot from the tagged commit. Record
   and commit, before any fit:
   - products, stores, paths and tags;
   - prime and precision;
   - fold sizes for both splits;
   - the share of held-out products whose path is unseen in fitting, for both
     splits;
   - labels by `taxonomy_source`, and the latest label time;
   - the snapshot and fold digests.
4. **Stop before fitting** if any label is not `gold_llm` or is newer than
   the cutoff, or if any product hash or (under the third restriction) store
   domain is shared with the paper snapshot.
5. **Run once** on Raksasa from the tagged commit. No smoke test may score a
   frozen-2 product.
6. **Failures.** A crash before any frozen-2 score has been written may be
   fixed and rerun, with the fix logged below. Once a score exists, that run
   is the run. If an ensemble member does not reach a coordinate optimum
   within its caps, the ensemble row is reported as incomplete and A3 leaves
   its family. The caps are not changed.
7. **Archive** the report with status `confirmatory`, and export aggregates
   only. Product rows and predictions stay on Raksasa.

## Reporting

The manuscript reports the frozen-2 values beside the paper-snapshot values
for all nine rows, with every comparison above, whatever the outcome.

- A claim keeps its place in the abstract and conclusion only if its
  comparison is confirmed, or for B reads "not worse" or "lower".
- A "not worse" reading supports "within 0.01 of", not "level with". The
  abstract's wording follows the reading.
- A claim that is not confirmed is reported as not confirmed.
- The two snapshots have different paths, primes and precisions. The
  manuscript does not compare a model's absolute loss across them.
- The "exploratory" wording is removed only for confirmed comparisons.

## Code needed before the freeze

- Builder options to exclude a snapshot's product hashes and store domains,
  the `gold_llm` filter, and a cutoff on `classified_at`.
- A runner that takes a snapshot id, prime and precision, and calls the
  existing fitting functions. The current runs are tied to the paper snapshot
  through `matched.PRIME`, `matched.PAPER_SNAPSHOT` and `oracle.load_splits`,
  and they check archived paper scores that frozen-2 does not have.
- Counters for stored coefficients and coefficients consulted in the list and
  level-by-level models.
- The preflight command.

## Decisions to confirm

1. **Population.** New stores only: 13,368 products at the planning counts.
   The alternative is every non-paper product: 16,314 products, but 2,583 of
   them are from stores whose tagging habits shaped the designs. The larger
   pool would lower the one-sided bound in family B by about a tenth.
2. **Depth of the level-by-level model.** Split at every level, as in the
   manuscript's abstract and Table 1. On the paper snapshot the depths differ
   by about 0.001, so this choice is not informed by those scores.
3. **Reading rule for family B.** "Not worse" by a one-sided bound at a
   margin of 0.01, unadjusted. The margin is the size of the differences
   among the leading models on the paper snapshot. The alternative is the
   first draft's two-sided "level" rule, which this pool cannot satisfy.
4. **Cutoff.** Labels with `classified_at` at or before 27 November 2026,
   00:00 UTC. This gives the planning counts above. A later cutoff adds few
   new-store products at November's rate.

## Deviations

None yet.
