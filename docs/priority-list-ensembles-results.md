# Priority-list ensembles, 23 November 2026

Ensembling adds little to a priority list. With the archived one-pass
bank's seeds, 75% feature masks and valid-path consensus, 243 backfilled
lists score 0.1486 on product holdouts against 0.1516 for one list using
every feature, and that difference is not significant. The same ensembles
are 0.04 to 0.06 below the archived one-pass medoid ensemble, on every
fold.

Against the classical models, a single backfilled list is
indistinguishable from L1 logistic regression on product holdouts. The
neural network scores lower than every list model there, though not
significantly after correction. On store holdouts every list model lies
between 0.626 and 0.644, numerically below the network's 0.654, and no
difference is significant.

Report: `docs/priority-list-ensembles-2026-11-23.json`, experiment
`paper-priority-list-ensembles-2026-11-23`, source commit `2a4d03a`,
status `exploratory`. It was not inserted into
`padjective.paper_low_q_reports`. The single-list models are described in
`docs/priority-list-results.md`.

## Members and consensus

Every member is a priority list fitted at weight 71 and filled in three
ways: cell medoids (**List**), cell medoids backfilled with pooled values
(**Backfilled**), and pooled values alone (**Pooled**).

- **75% masks** and **50% masks**: member $s$ keeps the feature mask the
  archived one-pass bank drew for its seed and orders the kept tags by
  taxonomy association.
- **Random orders**: every member keeps all tags and shuffles them with its
  seed plus the fold.

The ensemble predicts the fitting-fold path with the least total native
71-adic distance to the member predictions, exactly as the archived
ensembles do. Sizes are prefixes of the fixed 243-member roster.

Checks made by the run: the first archived one-pass member is refitted on
every fold and reproduces its archived minimum-valuation score; the
single-list scores equal those in `docs/priority-list-2026-11-23.json`;
and the incremental consensus is compared with a direct one at each
reported size.

## Mean loss by ensemble size

Product holdouts:

| Members | Size | List | Backfilled | Pooled |
|---|---:|---:|---:|---:|
| 75% masks | 1 | 0.2048 | 0.1989 | 0.2141 |
| 75% masks | 15 | 0.1546 | 0.1503 | 0.1672 |
| 75% masks | 243 | 0.1535 | 0.1486 | 0.1657 |
| 50% masks | 15 | 0.2222 | 0.2189 | 0.2363 |
| 50% masks | 243 | 0.2153 | 0.2118 | 0.2317 |
| Random orders | 1 | 0.2190 | 0.2107 | 0.2207 |
| Random orders | 15 | 0.1672 | 0.1608 | 0.1959 |
| Random orders | 243 | 0.1603 | 0.1548 | 0.1903 |
| One list, all features | 1 | 0.1576 | 0.1516 | 0.1657 |

Archived one-pass ensemble with 75% masks: sum 0.2824 at 15 members and
0.2853 at 243; medoid 0.2089 and 0.2069. L1 logistic regression 0.1535,
neural network 0.1418, decision tree 0.1859.

Store holdouts:

| Members | Size | List | Backfilled | Pooled |
|---|---:|---:|---:|---:|
| 75% masks | 1 | 0.6456 | 0.6486 | 0.6352 |
| 75% masks | 15 | 0.6395 | 0.6361 | 0.6237 |
| 75% masks | 243 | 0.6438 | 0.6355 | 0.6259 |
| 50% masks | 15 | 0.6422 | 0.6413 | 0.6349 |
| 50% masks | 243 | 0.6371 | 0.6349 | 0.6322 |
| Random orders | 1 | 0.6687 | 0.6502 | 0.6362 |
| Random orders | 15 | 0.6444 | 0.6346 | 0.6374 |
| Random orders | 243 | 0.6407 | 0.6274 | 0.6307 |
| One list, all features | 1 | 0.6517 | 0.6420 | 0.6259 |

Archived one-pass ensemble with 75% masks: sum 0.6842 and 0.6913; medoid
0.6478 and 0.6465. L1 logistic regression 0.6708, neural network 0.6539,
81-tree forest 0.6753, decision tree 0.7423.

The size-1 rows are the first roster member. Across all 243 members with
75% masks the mean individual loss is 0.206, 0.200 and 0.213 on product
holdouts and 0.659, 0.653 and 0.640 on store holdouts.

## What ensembling changes

- Masked members are individually worse than one list on all features
  (0.200 against 0.152, backfilled, product holdouts). The consensus
  recovers that loss and a little more.
- With pooled values and 75% masks, the 243-member ensemble has the same
  loss as the single pooled list on every fold of both splits.
- 50% masks cost about 0.06 on product holdouts.
- Random-order members average 0.204 to 0.217 on product holdouts. Their
  consensus reaches 0.155 backfilled, which matches the single
  taxonomy-association list and does not beat it.

## Paired tests

Fold differences in $d_{71,2}$ loss with the Nadeau--Bengio correction,
Holm adjusted within each family. With five folds, differences of about
0.01 are not resolved.

Product holdouts:

| Comparison | Difference | Result |
|---|---|---|
| 75%-mask list ensembles minus archived medoid ensemble, 15 and 243 members | -0.041 to -0.059 | lower on every fold; adjusted p at most 0.003 |
| Random-order ensembles minus archived medoid ensemble | -0.013 to -0.052 | lower on every fold; adjusted p at most 0.020 |
| 243 members, 75% masks, minus one list | list -0.004, backfilled -0.003, pooled 0.000 | unadjusted p 0.052 and 0.19 |
| One list minus L1 logistic regression | list +0.004, backfilled -0.002, pooled +0.012 | unadjusted p 0.59, 0.77, 0.12 |
| One list minus neural network | list +0.016, backfilled +0.010, pooled +0.024 | network lower on every fold; unadjusted p 0.045, 0.106, 0.017; adjusted 0.23, 0.42, 0.10 |
| 243 backfilled, 75% masks, minus L1 and minus network | -0.005 and +0.007 | unadjusted p 0.57 and 0.35 |

Store holdouts: no comparison is significant. The 75%-mask ensembles at
243 members are 0.003 to 0.021 below the archived medoid ensemble
(unadjusted p at least 0.39), 0.027 to 0.045 below L1 logistic regression
(at least 0.14) and 0.010 to 0.028 below the network (at least 0.21). The
single pooled list is below L1 on all five folds (-0.045, unadjusted p
0.14) and below the network on four (-0.028, 0.21).

## Limits

- These comparisons reuse the same five folds as every earlier analysis,
  and the list variants were chosen after earlier results on those folds.
- Sizes are nested prefixes of one roster, so the rows for different sizes
  are not independent.
- Members were fitted at weight 71 only. No bootstrap, learned-order or
  mixed-scheme ensemble was run.
- On product holdouts the forest was not tested, because its fold
  predictions are not in the archived classical runs.

## Reproduce

```bash
uv run -m padjective.paper_priority_list_ensembles --source-commit 2a4d03a --workers 10 --output docs/priority-list-ensembles-2026-11-23.json
```

The run takes about 75 seconds with ten workers on Raksasa. Add
`--archive` to insert the report into `padjective.paper_low_q_reports`.
