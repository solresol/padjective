# Priority lists fitted level by level, 24 November 2026

A plain priority list lets one tag decide the whole taxonomy path. This run
lets the top level be decided by its own list, ordered by how pure each tag
is at the top level, and then decides the rest of the path among the
products under that top-level category. On product holdouts the loss falls
from 0.1516 to between 0.1415 and 0.1431, lower on every fold for every
such model, with Holm-adjusted p between 0.016 and 0.043. That is level
with the neural network at 0.1418. On store holdouts the loss falls from
0.6420 to about 0.624, which is not significant.

Report: `docs/priority-list-levels-2026-11-24.json`, experiment
`paper-priority-list-levels-2026-11-24`, source commit `e78f903`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The priority-list models are described in `docs/priority-list-results.md`.

## Model

The model is a tree of priority lists that follows the taxonomy.

- **Split node.** It fits a backfilled priority list for the next base-71
  digit of the label, on the fitting products under the node. Tags are
  ordered by the share of their products on their most common digit, with
  ties to the larger tag and then the tag name. Each tag takes the most
  common digit of the products it heads. A product with no fitted tag takes
  the node's most common digit. Each digit has a child node fitted on the
  products that have it.
- **List node.** From a chosen depth down, a node fits a plain backfilled
  priority list at weight 71 for the whole remaining path, on the products
  under it, ordered by taxonomy association or by taxonomy spread.
- **Leaf.** A node whose fitting products share one label predicts it.

Depth zero is the plain list; the two plain rows reproduce
`docs/priority-list-graph-2026-11-23.json` exactly. For a fixed order a
split node's values minimise the number of wrong digits at that node, by
the same argument as for cell medoids. The fit is level by level and is not
shown to be jointly optimal.

Variants: a vote at split nodes (each tag votes for its most common digit),
children fitted only on the products their parent sends to them correctly,
and one taxonomy-association order shared by every node.

## Product holdouts

Mean d71,2 loss over five folds. The change is against the plain
taxonomy-association list, with the Holm-adjusted p-value over the twelve
comparisons in brackets. "Top level" and "Two levels" are the shares of
products with the first one and two levels right. Nodes counts split and
list nodes.

| Model | Loss | Change (adjusted p) | Top level | Two levels | Exact | Fitting loss | Nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain list (taxonomy association) | 0.1516 |  | 0.919 | 0.854 | 0.657 | 0.096 | 1 |
| Top level, then plain lists | 0.1431 | -0.0084 (0.043) | 0.930 | 0.863 | 0.659 | 0.085 | 21 |
| Two levels, then plain lists | 0.1426 | -0.0089 (0.023) | 0.930 | 0.869 | 0.655 | 0.083 | 60 |
| Three levels, then plain lists | 0.1416 | -0.0100 (0.024) | 0.930 | 0.869 | 0.657 | 0.081 | 107 |
| Every level | 0.1417 | -0.0098 (0.024) | 0.930 | 0.869 | 0.655 | 0.081 | 125 |
| Every level, consistent children | 0.1418 | -0.0097 (0.017) | 0.930 | 0.869 | 0.656 | 0.081 | 122 |
| Every level, one shared order | 0.1490 | -0.0026 (0.051) | 0.921 | 0.858 | 0.658 | 0.097 | 125 |
| Top-level vote, then plain lists | 0.1714 | +0.0198 (0.051) | 0.900 | 0.835 | 0.635 | 0.134 | 21 |
| Vote at every level | 0.1984 | +0.0469 (0.016) | 0.900 | 0.809 | 0.541 | 0.177 | 125 |
| Plain list (taxonomy spread) | 0.1455 | -0.0060 (0.051) | 0.929 | 0.863 | 0.646 | 0.093 | 1 |
| Top level, then spread lists | 0.1419 | -0.0096 (0.016) | 0.930 | 0.868 | 0.656 | 0.085 | 21 |
| Two levels, then spread lists | 0.1430 | -0.0085 (0.017) | 0.930 | 0.869 | 0.650 | 0.083 | 60 |
| Three levels, then spread lists | 0.1415 | -0.0100 (0.024) | 0.930 | 0.869 | 0.657 | 0.081 | 107 |
| L1 logistic regression | 0.1535 | | 0.915 | | 0.659 | | |
| Neural network | 0.1418 | | 0.925 | | 0.669 | | |

- **Every level-by-level list with its own order at each node beats the
  plain list.** The change is -0.0100 to -0.0084 across the eight such
  models, lower on every fold for every one.
- **The gain is at the top level.** Top-level accuracy rises from 0.919 to
  0.930, and exact accuracy stays between 0.650 and 0.659 against 0.657. Going
  from one level to three changes the fold losses by -0.0051 to +0.0014,
  which was not tested.
- **The level-specific order does the work.** With one shared order at every
  node the change is only -0.0026.
- **Level with the neural network.** The difference from the network is
  -0.0002 to +0.0014 (unadjusted p at least 0.80). The network has
  lower top-level accuracy (0.925) and higher exact accuracy (0.669). The
  difference from L1 logistic regression is -0.0120 to -0.0104, with
  unadjusted p between 0.12 and 0.17.
- **Against the taxonomy-spread list** the change is -0.0040 to -0.0024.
  Only the two-level model survives adjustment there (adjusted p 0.010).
- **Votes are worse** than the plain list at 0.1714 and 0.1984.
- **Fitting children on consistent products changes nothing**: 0.1418 against
  0.1417.
- The fitting loss falls from 0.096 to about 0.081. Unlike the learned and
  peeled orders, this lower fitting loss carries over to held-out products.

## Store holdouts

| Model | Loss | Change (adjusted p) | Top level | Two levels | Exact | Fitting loss | Nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain list (taxonomy association) | 0.6420 |  | 0.465 | 0.324 | 0.151 | 0.097 | 1 |
| Top level, then plain lists | 0.6264 | -0.0156 (1.000) | 0.483 | 0.342 | 0.157 | 0.085 | 21 |
| Two levels, then plain lists | 0.6264 | -0.0156 (1.000) | 0.483 | 0.349 | 0.150 | 0.083 | 57 |
| Three levels, then plain lists | 0.6241 | -0.0179 (1.000) | 0.483 | 0.349 | 0.156 | 0.082 | 101 |
| Every level | 0.6241 | -0.0179 (1.000) | 0.483 | 0.349 | 0.156 | 0.082 | 118 |
| Every level, consistent children | 0.6265 | -0.0155 (1.000) | 0.483 | 0.344 | 0.151 | 0.081 | 112 |
| Every level, one shared order | 0.6358 | -0.0062 (1.000) | 0.475 | 0.328 | 0.150 | 0.097 | 118 |
| Top-level vote, then plain lists | 0.6273 | -0.0147 (1.000) | 0.486 | 0.337 | 0.151 | 0.133 | 21 |
| Vote at every level | 0.6377 | -0.0043 (1.000) | 0.486 | 0.334 | 0.108 | 0.175 | 118 |
| Plain list (taxonomy spread) | 0.6353 | -0.0067 (1.000) | 0.478 | 0.332 | 0.138 | 0.093 | 1 |
| Top level, then spread lists | 0.6245 | -0.0175 (1.000) | 0.483 | 0.346 | 0.157 | 0.085 | 21 |
| Two levels, then spread lists | 0.6246 | -0.0174 (1.000) | 0.483 | 0.349 | 0.154 | 0.084 | 57 |
| Three levels, then spread lists | 0.6241 | -0.0179 (1.000) | 0.483 | 0.349 | 0.156 | 0.082 | 101 |
| L1 logistic regression | 0.6708 | | 0.433 | | 0.127 | | |
| Neural network | 0.6539 | | 0.451 | | 0.137 | | |

- The level-by-level lists are -0.0179 to -0.0155 against the plain list,
  with unadjusted p between 0.14 and 0.27. No comparison is significant.
  All but the consistent-children model are lower on every fold, by 0.001
  to 0.032.
- Top-level accuracy rises from 0.465 to 0.483.
- They are -0.0299 to -0.0275 against the neural network, also not
  significant (unadjusted p at least 0.19).
- The pooled priority list of `docs/priority-list-results.md` has 0.6259 on
  these folds, so this is not a clear gain over what was already there.

## Caveats

- The folds have been reused for every priority-list experiment, and these
  variants were chosen after seeing earlier results on them. The adjustment
  covers only the twelve comparisons of this run.
- The level-by-level models differ from each other by about 0.001, so this
  run does not say how many levels to use.
- The neural network and L1 logistic regression are the archived fits. They
  were not retuned.
