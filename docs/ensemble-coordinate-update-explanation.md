# Meaning of the repeated-coordinate/ensemble sentence

The sentence means that the ensemble experiment already includes a
single-model control trained with repeated updates. Combining more models
improves on that control, so the improvement within that experiment cannot
be explained just by replacing one training pass with several.

Source sentence: [journal results overview, lines 124–127](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/results-overview.tex#L124).
No manuscript wording has been changed.

## Three different counts

- **One pass:** choose an order for the tag features, fit each coefficient
  once and leave it fixed. The published best ordering is taxonomy association.
- **Repeated coordinate updates:** fit one coefficient vector, revisit its
  coordinates in fresh random orders, and accept only strict improvements
  of its training objective. A full sweep with no accepted change ends the fit.
  This is a coordinate optimum, not a guarantee of a global optimum.
- **Ensemble members:** separately fitted coefficient vectors, here made
  diverse by independent random feature-update orders. The number of members
  is not the number of passes through one model.

The repeated-coordinate member uses the finite quotient modulo `71**7`,
exact integer objective comparisons and a training-only fitted default.
The published one-pass model uses signed integer arithmetic and floating-point
loss comparisons. The sentence's one-member control is a converged
random-order repeated fit with seed base 42 (actual seed base plus fold), not
the average or best of every single-member fit.

## Scores and what they measure

The reported losses below are equal-weight means over the same five held-out
folds of the frozen 6,693-product snapshot. Smaller is better; zero is exact
agreement. A first-digit mistake has loss 1, while a mistake first appearing
at the next digit has loss `1/71`. Mean p-adic loss is not an error percentage.

| Model | Mean p-adic loss | Exact-path accuracy | Active member terms |
| --- | --- | --- | --- |
| Best one-pass taxonomy-association fit, raw signed predictor | 0.263237 | 49.82% | 1.11 |
| One repeated-coordinate member, projected | 0.277064 | 47.26% | 1.36 |
| 15 repeated-coordinate members, consensus | 0.159282 | 55.81% | 20.08 |
| 243 repeated-coordinate members, consensus | 0.144596 | 57.52% | 324.76 |

The 0.277064 control is slightly worse than the best one-pass result, but
that is not a paired experiment isolating the effect of extra passes: the
ordering, arithmetic, coefficient-selection/tie protocol and projection
differ. It does not establish that revisiting coefficients always hurts.

The nested ensemble-size series is the cleaner comparison: every member
uses the same repeated-coordinate fitting recipe, and every size uses the
same valid-training-path consensus rule. Sizes 1, 15 and 243 use nested
prefixes of a predeclared seed roster, with no held-out member selection.
The size-1 control is the first model in that roster.

Sources: [results overview](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/results-overview.tex#L20),
[ensemble-size table](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/ensemble_size_table.tex),
and [fixed-bank protocol](https://github.com/solresol/padjective/blob/c488032d5cd295b211c0d442b78f4b287ada3ff6/docs/ensemble-scaling-protocol.md).

## Valid-path projection and consensus

An arithmetic member output need not equal a taxonomy code. The candidate
set is the valid codes observed in that fold's **training data**, not the
held-out answer or necessarily the entire external Shopify taxonomy.

For one member, projection chooses a candidate minimizing its p-adic distance
to the member output. For several members, consensus chooses a candidate
minimizing the sum of its distances to every member output. Members have equal
weight, and ties choose the smallest canonical integer code. This is not an
arithmetic average of the codes. With one member the consensus formula reduces
exactly to nearest-valid-training-path projection.

The paired decoder ablation holds the member predictions, fitting, defaults,
folds and seed roster fixed, and removes only that candidate restriction:

| Members | Restricted to training paths | Unrestricted medoid of member outputs |
| --- | --- | --- |
| 1 | 0.277064 | 0.277381 |
| 15 | 0.159282 | 0.174505 |
| 243 | 0.144596 | 0.152410 |

Thus a large combination gain remains without the path restriction, while
the restriction contributes an additional improvement. These are paired,
inference-only observations on the frozen data, not universal guarantees.
The one-member projection changes 1,231 predictions; 26 improve in loss,
none worsen and 1,205 change code without changing loss.

Sources: [member fitting and consensus definition](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/method-details.tex#L334),
[decoder ablation](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/results-details.tex#L110),
and [saved decoder evidence](https://github.com/solresol/papers/blob/48678e75285e74fb00ae8b9cf0e493e434f19b43/padjective/padic-journal/data/valid-path-ablation-results.json).

## What the comparison supports

The common dataset, folds, member-fitting recipe and candidate-path decision
rule support the narrow conclusion: **the observed gain within the ensemble
size series is not merely the switch to repeated updates**, because the
one-member control already makes that switch.

This is not a fixed-computation comparison. More members require more fitted
models, more stored coefficients and more work at prediction time. The active
coefficient counter also excludes defaults and candidate-path search; the
paper reports a separate broader scoring proxy. The result does not isolate
the benefit of combining predictions from the benefit of spending more
computation, nor show that no better single fit could achieve the same loss.

## Proposed clearer wording

> The one-member control already uses repeated coordinate updates and
> valid-path projection. On the same five folds, increasing the ensemble
> from one to 15 or 243 members lowers mean loss from 0.277064 to 0.159282 or
> 0.144596. Thus the improvement within this series is not solely a consequence
> of revisiting coefficients; it accompanies combining more models and using
> more computation.
