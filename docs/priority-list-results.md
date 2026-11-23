# Priority-list fits, 23 November 2026

A priority list predicts with one coefficient: that of the highest-priority
tag present. For a fixed tag order, each tag takes the exact medoid of the
labels of the fitting products it heads, which is the training optimum for
that order. This replaces the sum at both fitting and prediction.

On product holdouts the priority list lowers mean $d_{71,2}$ loss for all
five one-pass orders, from 0.234 to 0.158 for taxonomy association. On store
holdouts the exact fit is not distinguishable from the existing
minimum-valuation medoid, and the order-free pooled values score lowest.

Report: `docs/priority-list-2026-11-23.json`, source commit `5042d01`,
status `exploratory`. It was not inserted into
`padjective.paper_low_q_reports`.

## Models

- **Sum** and **Medoid** are the frozen one-pass fits with the original sum
  and with the minimum-valuation medoid. The run reproduces their archived
  scores (`paper-valuation-orders-2026-11-09`) before comparing.
- **Priority list** fits each tag on the products it heads and predicts the
  coefficient of the highest-priority tag that has one.
- **Pooled, priority** gives each tag the medoid of all its fitting products
  and selects by the same order.
- **Pooled vote** uses the same pooled values with no order: the native
  71-adic medoid of the active values.

Products with no usable tag get the medoid of all fitting labels. As a
constant predictor that default scores 0.6931 on product holdouts and
0.6940 on store holdouts.

## Mean loss, fitting weight 71

Product holdouts:

| Order | Sum | Medoid | Priority list | Pooled, priority |
|---|---:|---:|---:|---:|
| Taxonomy association | 0.3283 | 0.2340 | 0.1576 | 0.1657 |
| Battle Elo | 0.3912 | 0.3033 | 0.2031 | 0.2107 |
| Random | 0.3830 | 0.2977 | 0.2141 | 0.2195 |
| Mean title position | 0.4044 | 0.3305 | 0.2508 | 0.2533 |
| Frequency | 0.3871 | 0.3510 | 0.2759 | 0.2778 |

Pooled vote: 0.1856.

Store holdouts:

| Order | Sum | Medoid | Priority list | Pooled, priority |
|---|---:|---:|---:|---:|
| Taxonomy association | 0.7128 | 0.6594 | 0.6517 | 0.6259 |
| Battle Elo | 0.7479 | 0.7021 | 0.6573 | 0.6357 |
| Random | 0.7505 | 0.7036 | 0.6723 | 0.6464 |
| Mean title position | 0.7979 | 0.7159 | 0.6648 | 0.6417 |
| Frequency | 0.7941 | 0.7314 | 0.6826 | 0.6528 |

Pooled vote: 0.6260.

Fitting at weight 2 changes no entry by more than 0.004.

For the taxonomy-association priority list on product holdouts: root
accuracy 0.913, exact accuracy 0.651, $L_{71}$ 0.088, fitting loss 0.096,
and a mean of 1,011 stored coefficients per fold.

## Paired tests

Fold differences in $d_{71,2}$ loss with the Nadeau--Bengio correction, Holm
adjusted across the five orders within each split.

| Comparison | Product holdouts | Store holdouts |
|---|---|---|
| Priority list minus medoid | -0.075 to -0.100; adjusted p at most 0.001 for every order | -0.008 to -0.051; no order below 0.05 (taxonomy association 0.65) |
| Pooled, priority minus medoid | -0.068 to -0.093; adjusted p at most 0.001 | -0.034 to -0.079; below 0.05 for random, title position and frequency; taxonomy association 0.195 |
| Priority list minus pooled, priority | -0.002 to -0.008; not significant | +0.022 to +0.030; pooled is lower on every fold for four orders; adjusted p 0.047 to 0.24 |

## Where the product-holdout gain comes from

Most of it is coverage. With taxonomy association, the one-pass sum fit
leaves 15.2% of held-out products without a nonzero coefficient, and those
products score 0.648 with the default. The priority list leaves 1.5%.
On the products each model does cover, the medoid rule scores 0.160 and the
priority list 0.150. The two covered sets differ, so this is a
decomposition of each score and not a matched comparison.

On store holdouts the corresponding coverage is 38.9% for the one-pass fit,
57.5% for the priority list and 62.3% for the pooled values.

## Limits

- These comparisons reuse the same five folds as every earlier analysis and
  were run after the minimum-valuation results were known.
- The models were not tested against the classical learners. The
  manuscript's product and store means for those are 0.1535 and 0.6708 for
  L1 logistic regression and 0.1418 and 0.6539 for the wide network.
- Tags that head no fitting product have no coefficient in the priority
  list and are skipped at prediction. No backfilled or learned-order variant
  was run.

## Reproduce

```bash
uv run -m padjective.paper_priority_list --source-commit 5042d01 --output docs/priority-list-2026-11-23.json
```

The run reads the frozen snapshot and archived references from PostgreSQL
and takes about 30 seconds on Raksasa. Add `--archive` to insert the report
into `padjective.paper_low_q_reports`.
