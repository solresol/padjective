# Graph-centrality tag orders for priority lists, 23 November 2026

Tag centrality measures how generic a tag is, and generic tags should
decide last. Sorting tags from least to most central beats the reverse for
ten of eleven scores on product holdouts. An order that uses no labels, the
number of distinct tags a tag co-occurs with, reaches 0.164 on product
holdouts as a backfilled priority list, against 0.207 for random orders and
0.152 for taxonomy association. On the graph that joins tags to taxonomy
paths, the number of distinct paths and betweenness match taxonomy
association. PageRank and the leading eigenvector are weaker than degree
and betweenness on both graphs.

Report: `docs/priority-list-graph-2026-11-23.json`, experiment
`paper-priority-list-graph-2026-11-23-v2`, source commit `51f3a65`, status
`exploratory`. It was not inserted into `padjective.paper_low_q_reports`.
The priority-list models are described in `docs/priority-list-results.md`.

## Graphs

All three graphs are built from the fitting fold only.

- **Co-occurrence graph.** Tags are joined when a product carries both,
  weighted by the number of such products. It uses no labels.
- **Tag-path graph.** Each tag is joined to the taxonomy paths of its
  products, weighted by product count.
- **Battle graph.** Each title battle is an edge from the losing tag to the
  winner. It uses no labels.

Mean over the five product fitting folds: 2542 tags,
5354 products and 5.1 tags per product. The
co-occurrence graph has 51,401 edges and 98 connected
components, the largest holding 1,897 tags. The tag-path graph has
7,926 edges, and 1,000 tags attach to a single path.

Scores are degree, weighted degree, PageRank with damping 0.85, the leading
eigenvector (leading left singular vector on the tag-path graph) and
unweighted betweenness. Each score orders the tags in both directions, with
ties to the larger tag and then the tag name. Each order is fitted at weight
71 as a priority list in the three usual ways. The taxonomy-association and
random rows reproduce `docs/priority-list-2026-11-23.json` exactly.

## Mean loss

Product holdouts:

| Score | Uses labels | Least central first: list | Backfilled | Pooled | Most central first: backfilled |
|---|---|---:|---:|---:|---:|
| Co-occurrence: distinct neighbours | no | 0.1714 | 0.1644 | 0.1812 | 0.2705 |
| Co-occurrence: betweenness | no | 0.1708 | 0.1651 | 0.1801 | 0.2750 |
| Co-occurrence: leading eigenvector | no | 0.1771 | 0.1696 | 0.1898 | 0.2577 |
| Co-occurrence: weighted degree | no | 0.1773 | 0.1705 | 0.1841 | 0.2591 |
| Co-occurrence: PageRank | no | 0.1780 | 0.1713 | 0.1850 | 0.2648 |
| Product-tag graph: degree (frequency) | no | 0.1782 | 0.1731 | 0.1856 | 0.2680 |
| Tag-path: distinct paths | yes | 0.1556 | 0.1492 | 0.1636 | 0.2792 |
| Tag-path: betweenness | yes | 0.1544 | 0.1473 | 0.1646 | 0.2771 |
| Tag-path: leading singular vector | yes | 0.1688 | 0.1637 | 0.1835 | 0.2629 |
| Tag-path: PageRank | yes | 0.1786 | 0.1737 | 0.1844 | 0.2729 |
| Battle graph: PageRank | no | 0.2536 | 0.2443 | 0.2531 | 0.2367 |
| Taxonomy association (reference) | yes | 0.1576 | 0.1516 | 0.1657 | |
| Random, five seeds (reference) | no | 0.2141 | 0.2072 | 0.2195 | |

Store holdouts:

| Score | Uses labels | Least central first: list | Backfilled | Pooled | Most central first: backfilled |
|---|---|---:|---:|---:|---:|
| Co-occurrence: distinct neighbours | no | 0.6638 | 0.6538 | 0.6380 | 0.6633 |
| Co-occurrence: betweenness | no | 0.6605 | 0.6478 | 0.6348 | 0.6659 |
| Co-occurrence: leading eigenvector | no | 0.6683 | 0.6693 | 0.6547 | 0.6523 |
| Co-occurrence: weighted degree | no | 0.6667 | 0.6592 | 0.6401 | 0.6624 |
| Co-occurrence: PageRank | no | 0.6695 | 0.6582 | 0.6394 | 0.6595 |
| Product-tag graph: degree (frequency) | no | 0.6661 | 0.6566 | 0.6412 | 0.6658 |
| Tag-path: distinct paths | yes | 0.6485 | 0.6444 | 0.6260 | 0.6688 |
| Tag-path: betweenness | yes | 0.6517 | 0.6483 | 0.6290 | 0.6679 |
| Tag-path: leading singular vector | yes | 0.6676 | 0.6606 | 0.6483 | 0.6633 |
| Tag-path: PageRank | yes | 0.6711 | 0.6613 | 0.6452 | 0.6620 |
| Battle graph: PageRank | no | 0.6734 | 0.6592 | 0.6464 | 0.6494 |
| Taxonomy association (reference) | yes | 0.6517 | 0.6420 | 0.6259 | |
| Random, five seeds (reference) | no | 0.6723 | 0.6592 | 0.6464 | |

## Reading

- **Direction.** Least central first is better than most central first for
  ten of the eleven scores on product holdouts and nine on store holdouts,
  comparing backfilled lists. Most central first is the existing frequency
  order, which was already the worst one-pass order.
- **Label-free orders.** On product holdouts every co-occurrence score and
  frequency, least central first, beats random orders on all five folds
  (adjusted p at most 0.012 for the backfilled lists). The best,
  distinct neighbours, is 0.043 below random and 0.013 above taxonomy
  association (taxonomy association lower on every fold, unadjusted p
  0.053).
- **Tag-path graph.** Distinct paths and betweenness, least central first,
  are 0.002 and 0.004 below taxonomy association on product holdouts,
  backfilled, and neither difference is significant (unadjusted p 0.49 and
  0.32). Their Spearman correlations with position in the
  taxonomy-association order are 0.91 and 0.87, so they are close to the
  same order.
- **Different orders, similar loss.** Distinct neighbours has a Spearman
  correlation of -0.04 with position in the taxonomy-association order and
  is only 0.013 behind it. The two agree on putting generic tags last and
  disagree about the order among specific tags.
- **Battle graph.** PageRank on the battle graph is worse than random
  orders. Only tags that appear in titles have battles, and the remaining
  tags tie and fall back to larger-tag-first, which is the frequency order.
  This row is therefore dominated by the tie rule and is not a fair test of
  the battle graph.
- **Store holdouts.** No least-central-first order differs significantly
  from taxonomy association. Distinct paths matches it, and as a plain list
  is 0.024 below random orders on every fold (adjusted p 0.022). The
  label-free orders fall between the two references and are not
  significantly different from either.

## Further scores and pure-tag peeling

A second batch added five scores and one order built from an exact
reduction. The rows of the first batch are unchanged.

- **Neighbourhood completeness** is the local clustering coefficient: the
  share of pairs of a tag's neighbours in the co-occurrence graph that are
  themselves joined. The graph's density is 0.016 and the mean
  completeness is 0.73.
- **Core number** is the largest $k$ for which the tag survives in the
  $k$-core of the co-occurrence graph.
- **Taxonomy spread** is the mean $d_{71,71}$ distance from a tag's
  labels to their medoid. It is taxonomy association with the hierarchical
  distance in place of exact category match.
- **Shared path prefix** is the number of leading digits on which all of a
  tag's labels agree.

Product holdouts:

| Order | Uses labels | List | Backfilled | Pooled | Reverse direction: backfilled |
|---|---|---:|---:|---:|---:|
| Co-occurrence: neighbourhood completeness, most complete first | no | 0.1755 | 0.1693 | 0.1874 | 0.2679 |
| Co-occurrence: core number, shallowest first | no | 0.1778 | 0.1711 | 0.1979 | 0.2673 |
| Co-occurrence: mean neighbour degree, highest first | no | 0.2003 | 0.1944 | 0.2230 | 0.2170 |
| Taxonomy spread of a tag's paths, tightest first | yes | 0.1519 | 0.1455 | 0.1606 | 0.2835 |
| Shared path prefix, longest first | yes | 0.1651 | 0.1569 | 0.1729 | 0.2790 |
| Pure tags peeled first, then taxonomy association | yes | 0.1604 | 0.1557 | 0.1697 | |
| Taxonomy association (reference) | yes | 0.1576 | 0.1516 | 0.1657 | |
| Random, five seeds (reference) | no | 0.2141 | 0.2072 | 0.2195 | |

Store holdouts:

| Order | Uses labels | List | Backfilled | Pooled | Reverse direction: backfilled |
|---|---|---:|---:|---:|---:|
| Co-occurrence: neighbourhood completeness, most complete first | no | 0.6698 | 0.6523 | 0.6392 | 0.6559 |
| Co-occurrence: core number, shallowest first | no | 0.6670 | 0.6559 | 0.6409 | 0.6638 |
| Co-occurrence: mean neighbour degree, highest first | no | 0.6633 | 0.6530 | 0.6444 | 0.6476 |
| Taxonomy spread of a tag's paths, tightest first | yes | 0.6445 | 0.6353 | 0.6249 | 0.6781 |
| Shared path prefix, longest first | yes | 0.6481 | 0.6422 | 0.6269 | 0.6733 |
| Pure tags peeled first, then taxonomy association | yes | 0.6585 | 0.6513 | 0.6305 | |
| Taxonomy association (reference) | yes | 0.6517 | 0.6420 | 0.6259 | |
| Random, five seeds (reference) | no | 0.6723 | 0.6592 | 0.6464 | |

Neighbourhood completeness orders tags usefully without labels: most
complete first is 0.038 below random orders on every product fold
(adjusted p 0.005, backfilled). It is 0.018 above taxonomy association and
no better than the simpler count of distinct neighbours.

Taxonomy spread is the lowest static order tried. On product holdouts it
is below taxonomy association on every fold, by 0.006 for both the plain
and the backfilled list (unadjusted p 0.077 and 0.017). It gains at the
top of the tree: backfilled root accuracy is 0.929 against 0.919, while
exact accuracy is 0.646 against 0.657. These are two of
66 comparisons with taxonomy association in this report, and neither
survives Holm adjustment, so the result needs an untouched check. On store
holdouts it is 0.007 lower and not significant (unadjusted p 0.29).

### Peeling pure tags

A tag is pure when all of its products share one label. Moving a pure tag
to the front of any order cannot raise the fitting loss: the tag then
heads all of its products exactly, and every other tag heads a subset of
what it headed before. Repeating this on the products that remain gives a
prefix of the order that is optimal for fitting, and leaves a kernel on
which the choice of order matters.

On the product fitting folds, peeling places a mean of 1,306 tags and
settles 2,336 of 5,354 products with zero fitting loss. The
kernel has 3,018 products and 1,150 tags in 63
co-occurrence components, the largest with 955 tags. Exact search
over the kernel is therefore still out of reach.

Peeling first and then following taxonomy association lowers the fitting
loss from 0.096 to 0.093 and does not lower the held-out loss (0.1557
against 0.1516 backfilled on product holdouts).

## Limits

- These comparisons reuse the same five folds as every earlier analysis.
- Betweenness ignores edge weights.
- No combination of scores, and no graph built from unlabelled products
  outside the benchmark, was tried.

## Reproduce

```bash
uv run -m padjective.paper_priority_list_graph --source-commit 51f3a65 --workers 10 --output docs/priority-list-graph-2026-11-23.json
```

The run takes about 100 seconds with ten workers on Raksasa. Add `--archive`
to insert the report into `padjective.paper_low_q_reports`.
