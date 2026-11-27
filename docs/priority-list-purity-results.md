# Ordering a priority list by tree-distance noise, 27 November 2026

Taxonomy association ranks a tag by the share of its products on its most
common path. That is the spread of the tag's labels under the discrete
metric, where every wrong path is equally wrong. This run measures the same
spread with the tree distance at a range of weights and puts the least noisy
tag first. On product holdouts the backfilled list falls from 0.1516 to
0.1427 to 0.1432 at weights 2 to 4, lower on every fold, which is within
0.001 of the level-by-level lists and of the neural network. No comparison
survives adjustment over the 32 in this run. On store holdouts the change is
about -0.006 and not significant.

Report: `docs/priority-list-purity-2026-11-27.json`, experiment
`paper-priority-list-purity-2026-11-27`, source commit `f490cef`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The priority-list models are described in `docs/priority-list-results.md`.

## Scores

Each score gives a tag a noise from the labels of its fitting products. Tags
are sorted from least to most noisy, with ties to the larger tag and then
the tag name. Cells are fitted at weight 71, as before.

- **Spread at weight $q$** is the mean distance of a tag's labels to their
  medoid, where labels that first differ at digit $k$ are $q^{-k}$ apart.
  It is the loss of predicting the medoid for the tag's own products.
  - As $q$ falls to one, every distance tends to one and the spread tends
    to one minus the taxonomy association. The order by spread under the
    discrete metric equals the taxonomy-association order on every fold of
    both splits.
  - At weight 71 it is the taxonomy spread of
    `docs/priority-list-graph-results.md`, and reproduces it exactly.
- **Shrunk spread** adds one pseudo-product at the spread of all fitting
  labels.
- **Pairwise** is the mean distance between two different products of the
  tag.
- **Ball** is the number of leading digits on which a stated share of the
  tag's labels agree, longest first, with ties to the smaller spread at
  weight 71. With a share of one it gives the same order as spread at
  weight 71 on every fold.
- **Entropy** is the Shannon entropy of the exact labels. Like taxonomy
  association it ignores the tree.

## Product holdouts

Mean $d_{71,2}$ loss of the backfilled list over five folds. The change is
against taxonomy association, with the unadjusted p-value. "Correlation" is
the Spearman correlation with position in the taxonomy-association order.

| Order | Loss | Change (p) | Folds lower | Top level | Two levels | Exact | Correlation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Taxonomy association | 0.1516 | | | 0.919 | 0.854 | 0.657 | 1.00 |
| Entropy | 0.1497 | -0.0019 (0.42) | 4 | 0.921 | 0.857 | 0.658 | 0.98 |
| Spread, weight 1.25 | 0.1468 | -0.0047 (0.21) | 4 | 0.925 | 0.859 | 0.656 | 0.98 |
| Spread, weight 1.5 | 0.1437 | -0.0079 (0.036) | 5 | 0.927 | 0.864 | 0.658 | 0.96 |
| Spread, weight 2 | 0.1432 | -0.0084 (0.012) | 5 | 0.929 | 0.865 | 0.655 | 0.92 |
| Spread, weight 3 | 0.1429 | -0.0086 (0.038) | 5 | 0.930 | 0.865 | 0.653 | 0.89 |
| Spread, weight 4 | 0.1427 | -0.0088 (0.028) | 5 | 0.931 | 0.866 | 0.652 | 0.88 |
| Spread, weight 8 | 0.1448 | -0.0068 (0.033) | 5 | 0.930 | 0.864 | 0.648 | 0.87 |
| Spread, weight 16 | 0.1455 | -0.0061 (0.010) | 5 | 0.929 | 0.864 | 0.646 | 0.87 |
| Spread, weight 71 | 0.1455 | -0.0060 (0.017) | 5 | 0.929 | 0.863 | 0.646 | 0.87 |
| Shrunk spread, weight 2 | 0.1451 | -0.0064 (0.065) | 5 | 0.929 | 0.864 | 0.643 | 0.82 |
| Shrunk spread, weight 71 | 0.1670 | +0.0155 (0.005) | 0 | 0.928 | 0.844 | 0.572 | 0.31 |
| Pairwise, weight 2 | 0.1429 | -0.0087 (0.045) | 5 | 0.930 | 0.866 | 0.652 | 0.90 |
| Pairwise, weight 71 | 0.1457 | -0.0059 (0.023) | 5 | 0.929 | 0.863 | 0.645 | 0.86 |
| Ball, 90% of labels | 0.1445 | -0.0071 (0.030) | 5 | 0.929 | 0.863 | 0.651 | 0.88 |
| Ball, 75% of labels | 0.1448 | -0.0068 (0.054) | 5 | 0.928 | 0.862 | 0.652 | 0.93 |

- **Every tree-aware order is below taxonomy association**, except the
  shrunk spread at weight 71. From weight 1.5 up, each is lower on all five
  folds.
- **The loss is lowest at weights 2 to 4**, not at the native weight 71.
  Weight 2 is the weight of the evaluation loss.
- **The gain is at the top of the tree.** Top-level accuracy rises from
  0.919 to about 0.930. Exact accuracy holds at weights up to 4 and falls
  to 0.646 at weights 16 and 71.
- **Ignoring the tree gains little.** Entropy is 0.002 below taxonomy
  association.
- **Shrinking hurts.** At weight 71 one pseudo-product moves most
  single-product tags behind the large tags.
- **Against the other models.** The list ordered by spread at weight 2 is
  0.0014 above the level-by-level lists (95% interval -0.0025 to +0.0054)
  and 0.0014 above the neural network (-0.0103 to +0.0131). It is 0.0104
  below L1 logistic regression (-0.0249 to +0.0042), lower on four folds.
- The fitting loss falls from 0.096 to 0.091 at weight 2.
- Fitting the cells at weight 2 changes no entry by more than 0.001.
- The family has 32 comparisons: sixteen orders, with cells fitted at weight
  71 and at weight 2. The smallest Holm-adjusted p among the orders that
  lower the loss is 0.15.

## Store holdouts

| Order | Loss | Change (p) | Folds lower | Top level | Two levels | Exact |
|---|---:|---:|---:|---:|---:|---:|
| Taxonomy association | 0.6420 | | | 0.465 | 0.324 | 0.151 |
| Entropy | 0.6436 | +0.0016 (0.73) | 3 | 0.464 | 0.320 | 0.149 |
| Spread, weight 1.25 | 0.6376 | -0.0044 (0.32) | 4 | 0.470 | 0.329 | 0.151 |
| Spread, weight 1.5 | 0.6358 | -0.0062 (0.26) | 4 | 0.472 | 0.331 | 0.152 |
| Spread, weight 2 | 0.6362 | -0.0058 (0.40) | 3 | 0.472 | 0.330 | 0.149 |
| Spread, weight 3 | 0.6357 | -0.0063 (0.38) | 3 | 0.474 | 0.330 | 0.148 |
| Spread, weight 4 | 0.6360 | -0.0060 (0.40) | 3 | 0.475 | 0.328 | 0.146 |
| Spread, weight 8 | 0.6371 | -0.0049 (0.43) | 3 | 0.477 | 0.328 | 0.139 |
| Spread, weight 16 | 0.6362 | -0.0058 (0.34) | 3 | 0.477 | 0.332 | 0.138 |
| Spread, weight 71 | 0.6353 | -0.0067 (0.29) | 3 | 0.478 | 0.332 | 0.138 |
| Shrunk spread, weight 2 | 0.6321 | -0.0099 (0.44) | 3 | 0.484 | 0.330 | 0.143 |
| Shrunk spread, weight 71 | 0.6423 | +0.0003 (0.98) | 3 | 0.487 | 0.309 | 0.116 |
| Pairwise, weight 2 | 0.6350 | -0.0070 (0.28) | 5 | 0.474 | 0.333 | 0.148 |
| Pairwise, weight 71 | 0.6348 | -0.0072 (0.24) | 3 | 0.479 | 0.333 | 0.138 |
| Ball, 90% of labels | 0.6357 | -0.0063 (0.32) | 4 | 0.475 | 0.328 | 0.147 |
| Ball, 75% of labels | 0.6347 | -0.0073 (0.24) | 4 | 0.475 | 0.332 | 0.148 |

- The spread orders are 0.004 to 0.007 below taxonomy association. No
  comparison is significant, before or after adjustment.
- The list ordered by spread at weight 2 is 0.012 above the level-by-level
  lists and 0.018 below the neural network. Neither difference is
  significant (unadjusted p 0.21 and 0.31).

## Caveats

- The folds have been reused for every priority-list experiment, and these
  orders were chosen after seeing that taxonomy spread at weight 71 scored
  below taxonomy association. The adjustment covers only this run.
- The weights from 1.5 to 71 give correlated orders, so the five-fold
  agreement across them is not independent evidence.
- The neural network, L1 logistic regression and the level-by-level lists
  are the archived fits.
