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
  on this today; `live_subspace.py` does.
- **No paper products.** Drop every product whose hash is in the paper
  snapshot.
- **No paper stores.** Drop every product from a store that has a product in
  the paper snapshot, using the store membership of
  `paper_matched_store_folds.store_membership`.

The eligibility counts are taken after these restrictions. Fix one `--as-of`
cutoff for the source data before the build.

If the pool has fewer than 6,693 products after filtering, the third
restriction is dropped and the pool is every non-paper product. The
new-store products are then reported as a subgroup. This rule depends only on
counts and is applied before any fit.

Identity is the canonical product URL. The same goods listed by two stores
count as two products, as they do in the paper.

## Splits

- **Product holdouts:** the five folds that the builder records for each
  product (`cv.calculate_cv_folds`, seed 42).
- **Store holdouts:** five folds from `live_subspace.store_fold`, the SHA-256
  of the normalised myshopify domain.

Feature ordering, fitting, defaults and consensus candidates use four folds.
The fifth is scored. Held-out products whose reference path is absent from
the fitting folds stay in the denominator.

## Encoding and loss

The prime is the smallest prime above the largest sibling index in the
frozen-2 reference paths, and the precision is the depth of the deepest path.
Both are fixed by the preflight. The primary loss is mean $d_{p,2}$. Mean
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

**B. Against the classical models, product holdouts.** These are estimates,
not tests. Report each mean difference with its 95% interval and read it
against a margin of 0.01:

- **level** if the whole interval lies within ±0.01;
- **lower** or **higher** if the interval excludes zero and is not within
  the margin;
- **unresolved** otherwise.

| | Difference | Paper snapshot |
|---|---|---:|
| B1 | backfilled list − L1 logistic regression | −0.002 |
| B2 | level-by-level − neural network | −0.000 |
| B3 | level-by-level − random forest | +0.001 |

On the paper snapshot the standard errors of such differences were about
0.005 to 0.0065, back-calculated from the reported $p$ values, which gives a
half-width of 0.014 to 0.018 with four degrees of freedom. If the standard
error falls as the square root of the number of products, the pool needs
roughly two to three times the paper's 6,693 products before "level" is a
possible reading. With fewer, the honest result for B is "unresolved".

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
   - labels by `taxonomy_source`;
   - the snapshot and fold digests.
4. **Stop before fitting** if any label is not `gold_llm`, or if any product
   hash or (under the third restriction) store domain is shared with the
   paper snapshot.
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
  comparison is confirmed, or for B reads "level" or "lower".
- A claim that is not confirmed is reported as not confirmed.
- The "exploratory" wording is removed only for confirmed comparisons.

## Code needed before the freeze

- Builder options to exclude a snapshot's product hashes and store domains,
  and the `gold_llm` filter.
- A runner that takes a snapshot id, prime and precision, and calls the
  existing fitting functions. The current runs are tied to the paper snapshot
  through `matched.PRIME`, `matched.PAPER_SNAPSHOT` and `oracle.load_splits`,
  and they check archived paper scores that frozen-2 does not have.
- Counters for stored coefficients and coefficients consulted in the list and
  level-by-level models.
- The preflight command.

## Decisions to confirm

1. **Population.** New stores only, with the fallback above. The alternative
   is every non-paper product from the start: more data, but products from
   stores whose tagging habits shaped the designs.
2. **Depth of the level-by-level model.** Split at every level, as in the
   manuscript's abstract and Table 1. On the paper snapshot the depths differ
   by about 0.001, so this choice is not informed by those scores.
3. **Margin for family B.** 0.01, the size of the differences among the
   leading models on the paper snapshot.
4. **Cutoff.** The `--as-of` timestamp for the source data.

## Deviations

None yet.
