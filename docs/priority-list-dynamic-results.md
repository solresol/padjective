# Dynamic tag orders for priority lists, 23 November 2026

A dynamic order reapplies a one-pass ordering rule to the fitting products
that have none of the tags placed so far, places the first tag the rule
returns, and repeats. Doing this for taxonomy association, Battle Elo, mean
title position and frequency changes which tag heads 11% to 18% of held-out
products on product holdouts, but no order improves significantly. Taxonomy
association and Battle Elo get slightly worse, title position does not
move, and frequency gets slightly better.

Report: `docs/priority-list-dynamic-2026-11-23.json`, experiment
`paper-priority-list-dynamic-2026-11-23`, source commit `2604bc8`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The priority-list models are described in `docs/priority-list-results.md`.

## Orders

The static order applies the rule once to the whole fitting fold, as the
one-pass fits do. The dynamic order starts with the same first tag. After
each placement the rule sees only the products that tag did not claim, so a
tag is ranked on the products it could still head. Battle Elo is recomputed
from the title battles of the remaining products at every step. Tags that
head nothing follow in static order.

Both orders are fitted as priority lists at weight 71 in the three ways used
before: cell medoids (**List**), backfilled cell medoids (**Backfilled**)
and pooled values (**Pooled**). The static rows reproduce
`docs/priority-list-2026-11-23.json` exactly.

## Mean loss

Product holdouts:

| Rule | Order | List | Backfilled | Pooled | Fitting loss | Heads changed |
|---|---|---:|---:|---:|---:|---:|
| Taxonomy association | static | 0.1576 | 0.1516 | 0.1657 | 0.096 | |
| Taxonomy association | dynamic | 0.1620 | 0.1573 | 0.1753 | 0.082 | 17.8% |
| Battle Elo | static | 0.2031 | 0.1954 | 0.2107 | 0.157 | |
| Battle Elo | dynamic | 0.2143 | 0.2072 | 0.2208 | 0.166 | 11.3% |
| Mean title position | static | 0.2508 | 0.2414 | 0.2533 | 0.206 | |
| Mean title position | dynamic | 0.2517 | 0.2421 | 0.2445 | 0.206 | 11.5% |
| Frequency | static | 0.2759 | 0.2680 | 0.2778 | 0.233 | |
| Frequency | dynamic | 0.2736 | 0.2623 | 0.2648 | 0.232 | 11.6% |

Store holdouts:

| Rule | Order | List | Backfilled | Pooled | Fitting loss | Heads changed |
|---|---|---:|---:|---:|---:|---:|
| Taxonomy association | static | 0.6517 | 0.6420 | 0.6259 | 0.097 | |
| Taxonomy association | dynamic | 0.6671 | 0.6646 | 0.6373 | 0.084 | 13.0% |
| Battle Elo | static | 0.6573 | 0.6486 | 0.6357 | 0.154 | |
| Battle Elo | dynamic | 0.6585 | 0.6513 | 0.6352 | 0.162 | 10.4% |
| Mean title position | static | 0.6648 | 0.6495 | 0.6417 | 0.199 | |
| Mean title position | dynamic | 0.6624 | 0.6520 | 0.6380 | 0.199 | 6.0% |
| Frequency | static | 0.6826 | 0.6658 | 0.6528 | 0.223 | |
| Frequency | dynamic | 0.6695 | 0.6559 | 0.6436 | 0.227 | 7.6% |

"Heads changed" is the share of held-out products whose highest-priority
tag differs from the static order's.

## Paired tests

Dynamic minus static fold losses for the list and backfilled fits, with the
Nadeau--Bengio correction and Holm adjustment across the eight comparisons
in each split.

| Rule | Product holdouts | Store holdouts |
|---|---|---|
| Taxonomy association | +0.004 and +0.006; static lower on four folds; unadjusted p 0.35 and 0.22 | +0.015 and +0.023; static lower on four and five folds; unadjusted p 0.18 and 0.12 |
| Battle Elo | +0.011 and +0.012; static lower on every fold; unadjusted p 0.039 and 0.048, adjusted 0.31 and 0.33 | +0.001 and +0.003; unadjusted p at least 0.78 |
| Mean title position | +0.001 and +0.001; unadjusted p at least 0.79 | -0.003 and +0.002; unadjusted p at least 0.60 |
| Frequency | -0.002 and -0.006; dynamic lower on four and five folds; unadjusted p 0.48 and 0.11 | -0.013 and -0.010; dynamic lower on five and four folds; unadjusted p 0.32 and 0.36 |

No adjusted p-value is below 0.31.

## Reading

Dynamic taxonomy association fits the fitting fold better (0.082 against
0.096) and predicts held-out products slightly worse, as the greedy purity
order did. A tag ranked only on its leftover products can rise on the
strength of a few of them, which is a noisier basis than all of its
products.

The earlier greedy order in `docs/priority-list-results.md` is also dynamic.
It differs from dynamic taxonomy association in measuring purity by
hierarchical distance to the medoid where taxonomy association uses the
share of the most common exact category. It scored 0.1566 and 0.1521 on
product holdouts, against 0.1620 and 0.1573 here.

## Limits

- These comparisons reuse the same five folds as every earlier analysis.
- Random orders have no dynamic counterpart and were not included.
- Fits use weight 71 only.

## Reproduce

```bash
uv run -m padjective.paper_priority_list_dynamic --source-commit 2604bc8 --workers 10 --output docs/priority-list-dynamic-2026-11-23.json
```

The run takes about 30 seconds with ten workers on Raksasa. Add `--archive`
to insert the report into `padjective.paper_low_q_reports`.
