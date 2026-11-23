# Priority-list fits, 23 November 2026

A priority list predicts with one coefficient: that of the highest-priority
tag present. For a fixed tag order, each tag takes the exact medoid of the
labels of the fitting products it heads, which is the training optimum for
that order. This replaces the sum at both fitting and prediction.

On product holdouts the priority list lowers mean $d_{71,2}$ loss for all
five one-pass orders, from 0.234 to 0.158 for taxonomy association, and
backfilling tags that head nothing takes it to 0.152. A learned order
lowers the fitting loss but not the held-out loss. On store holdouts the
exact fit is not distinguishable from the existing minimum-valuation
medoid, and the order-free pooled values score lowest.

Report: `docs/priority-list-2026-11-23.json`, experiment
`paper-priority-list-2026-11-23-v2`, source commit `038b3d6`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The first version of the report, without the backfilled and greedy rows, is
at commit `658bbdd`; every number it held is unchanged.

## Models

- **Sum** and **Medoid** are the frozen one-pass fits with the original sum
  and with the minimum-valuation medoid. The run reproduces their archived
  scores (`paper-valuation-orders-2026-11-09`) before comparing.
- **Priority list** fits each tag on the products it heads and predicts the
  coefficient of the highest-priority tag that has one.
- **Backfilled** keeps those coefficients and gives every tag that heads
  nothing the medoid of all its fitting products.
- **Pooled, priority** gives every tag the medoid of all its fitting
  products and selects by the same order.
- **Pooled vote** uses the same pooled values with no order: the native
  71-adic medoid of the active values.

Products with no usable tag get the medoid of all fitting labels. As a
constant predictor that default scores 0.6931 on product holdouts and
0.6940 on store holdouts.

The **greedy** orders are learned from the fitting fold. Each step takes
the tag whose labels, on the products not yet headed, have the lowest mean
distance to their medoid. The mean is shrunk towards the constant
predictor's by $m$ pseudo-products, and ties go to the larger tag. Tags
that head nothing follow in taxonomy-association order.

## Mean loss, fitting weight 71

Product holdouts:

| Order | Sum | Medoid | Priority list | Backfilled | Pooled, priority |
|---|---:|---:|---:|---:|---:|
| Taxonomy association | 0.3283 | 0.2340 | 0.1576 | 0.1516 | 0.1657 |
| Battle Elo | 0.3912 | 0.3033 | 0.2031 | 0.1954 | 0.2107 |
| Random | 0.3830 | 0.2977 | 0.2141 | 0.2072 | 0.2195 |
| Mean title position | 0.4044 | 0.3305 | 0.2508 | 0.2414 | 0.2533 |
| Frequency | 0.3871 | 0.3510 | 0.2759 | 0.2680 | 0.2778 |
| Greedy, $m=0$ | | | 0.1566 | 0.1521 | 0.1729 |
| Greedy, $m=1$ | | | 0.1781 | 0.1693 | 0.1832 |
| Greedy, $m=4$ | | | 0.1879 | 0.1782 | 0.1875 |

Pooled vote: 0.1856.

Store holdouts:

| Order | Sum | Medoid | Priority list | Backfilled | Pooled, priority |
|---|---:|---:|---:|---:|---:|
| Taxonomy association | 0.7128 | 0.6594 | 0.6517 | 0.6420 | 0.6259 |
| Battle Elo | 0.7479 | 0.7021 | 0.6573 | 0.6486 | 0.6357 |
| Random | 0.7505 | 0.7036 | 0.6723 | 0.6592 | 0.6464 |
| Mean title position | 0.7979 | 0.7159 | 0.6648 | 0.6495 | 0.6417 |
| Frequency | 0.7941 | 0.7314 | 0.6826 | 0.6658 | 0.6528 |
| Greedy, $m=0$ | | | 0.6615 | 0.6566 | 0.6327 |
| Greedy, $m=1$ | | | 0.6597 | 0.6527 | 0.6269 |
| Greedy, $m=4$ | | | 0.6616 | 0.6516 | 0.6266 |

Pooled vote: 0.6260.

For the five one-pass orders, fitting at weight 2 changes no entry by more
than 0.006. The greedy orders depend on the weight, because it sets the
purity measure:

| Greedy order, fitting weight 2 | Product: list | Product: backfilled | Store: list | Store: backfilled | Store: pooled |
|---|---:|---:|---:|---:|---:|
| $m=0$ | 0.1570 | 0.1525 | 0.6645 | 0.6620 | 0.6375 |
| $m=1$ | 0.1523 | 0.1458 | 0.6505 | 0.6430 | 0.6205 |
| $m=4$ | 0.1610 | 0.1528 | 0.6635 | 0.6525 | 0.6261 |

The backfilled taxonomy-association list on product holdouts has root
accuracy 0.919, exact accuracy 0.657 and $L_{71}$ 0.082. The list before
backfilling stores a mean of 1,011 coefficients per fold.

## Fitting loss against held-out loss

On product folds at fitting weight 71, the taxonomy-association list has
fitting loss 0.096 and held-out loss 0.158. The unsmoothed greedy order
lowers the fitting loss to 0.075 and leaves the held-out loss at 0.157.
Lower fitting loss from a better order did not carry over to held-out
products here.

## Paired tests

Fold differences in $d_{71,2}$ loss with the Nadeau--Bengio correction,
Holm adjusted within each family and split.

| Comparison | Product holdouts | Store holdouts |
|---|---|---|
| Priority list minus medoid, five orders | -0.075 to -0.100; adjusted p at most 0.001 for every order | -0.008 to -0.051; no order below 0.05 (taxonomy association 0.65) |
| Pooled, priority minus medoid, five orders | -0.068 to -0.093; adjusted p at most 0.001 | -0.034 to -0.079; below 0.05 for random, title position and frequency; taxonomy association 0.195 |
| Priority list minus pooled, priority, five orders | -0.002 to -0.008; not significant | +0.022 to +0.030; pooled is lower on every fold for four orders; adjusted p 0.047 to 0.24 |
| Backfilled minus priority list, eight orders | -0.005 to -0.010; lower on every fold for every order; adjusted p 0.019 to 0.048 (0.014 to 0.064 at fitting weight 2) | -0.005 to -0.017; adjusted p at least 0.34 |
| Backfilled minus pooled, priority, eight orders | -0.009 to -0.021; adjusted p at least 0.27 | +0.008 to +0.026; adjusted p at least 0.22 |
| Greedy minus taxonomy association, list and backfilled | $m=0$: within 0.001. $m=1$ and $m=4$ at weight 71: worse by 0.018 to 0.030, adjusted p 0.036 to 0.050. At weight 2 no difference is significant; $m=1$ backfilled is 0.006 lower on all five folds, unadjusted p 0.17 | -0.001 to +0.020; none significant |

## Where the product-holdout gain comes from

Most of it is coverage. With taxonomy association, the one-pass sum fit
leaves 15.2% of held-out products without a nonzero coefficient, and those
products score 0.648 with the default. The priority list leaves 1.5%, and
backfilling leaves none. On the products each model does cover, the medoid
rule scores 0.160 and the priority list 0.150. The two covered sets differ,
so this is a decomposition of each score and not a matched comparison.

On store holdouts the corresponding coverage is 38.9% for the one-pass fit,
57.5% for the priority list and 62.3% for the backfilled and pooled lists.
Backfilling therefore reaches the same products as the pooled list and
still scores higher, so the store gap between them is in the coefficient
values and not in coverage.

## Limits

- These comparisons reuse the same five folds as every earlier analysis.
  The backfilled and greedy variants were added after the first results on
  those folds were known. A second frozen snapshot from the live data would
  give an untouched check.
- The lowest product-holdout entry, 0.1458, is the best of sixteen
  order-and-weight combinations and is not significantly below the
  backfilled taxonomy-association list.
- The models were not tested against the classical learners. The
  manuscript's product and store means for those are 0.1535 and 0.6708 for
  L1 logistic regression and 0.1418 and 0.6539 for the wide network.
- No local search over orders, shrinkage between cell and pooled values, or
  ensemble of lists was run.

## Reproduce

```bash
uv run -m padjective.paper_priority_list --source-commit 038b3d6 --output docs/priority-list-2026-11-23.json
```

The run reads the frozen snapshot and archived references from PostgreSQL
and takes about 40 seconds on Raksasa. Add `--archive` to insert the report
into `padjective.paper_low_q_reports`.
