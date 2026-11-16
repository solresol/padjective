# Independent frozen one-pass replay, 16 November 2026

This audit read the frozen PostgreSQL snapshot in a repeatable-read, read-only
transaction. No models, predictions or database rows were changed. Numerical
source: `b98d1871934010bd03a5126f52dae02a48604b74`. The snapshot digest is
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.

## Finding

All 6,693 archived taxonomy-association predictions reproduce from the frozen
rows. Auditing all 12,709 fitted coordinates with exact rational distances
finds no strictly suboptimal coordinate choice. However, 19 choices are tied
exactly but floating-point accumulation selects a coefficient other than the
smallest signed integer. This differs from the ideal tie rule in the paper.
All five zero-score defaults agree with their exact medoid and tie rule.

A separate fitter reconstructs taxonomy-association order with exact rational
association scores, performs its own residual updates and default selection,
and predicts without calling the production fitting loop. Its coefficients,
defaults and predictions agree with the exact-selector replay of that loop.
The exact fit changes 17 coefficients and 48 held-out predictions (3, 28, 8,
2 and 7 across the folds). Equal-fold mean losses are:

| Evaluation | Archived floating-point fit | Exact smallest-integer ties |
|---|---:|---:|
| q=2 | 0.32828541482446744 | 0.3278732132121567 |
| q=71 | 0.26323667562627723 | 0.26322821713378897 |

This is a numerical sensitivity result, not permission to replace historical
results. Existing coefficients, tables and predictions remain unchanged.
The manuscript should distinguish ideal exact coordinate minimisation from
its historical floating-point implementation. Repeated-coordinate fits use
exact integer loss comparisons and are a different implementation.

## Reproduction

Run with the existing Shopify connection and dependencies:

```sh
uv run python reference/independent_exact_replay.py --output /tmp/exact-replay.json
```

The script requires the fixed snapshot and archived taxonomy-association
prediction bank. It refuses a changed snapshot digest, reads within a
read-only transaction, then calculates in memory. Its output contains aggregate
audit evidence and numerical coefficient discrepancies, not product records.
The saved output is `independent-replay-2026-11-16.json`.

## Other checks and limits

An independent Fraction/brute-force check of the repeated-coordinate routines
passed 800 residual medoids, 800 constrained ensemble consensuses and 300 small
fits whose every possible coordinate replacement was checked. The headline
one-pass q=2 values and minimum-valuation-medoid losses were independently
reaggregated from fold evidence. The latter agree at 0.2339640595344194 for
products and 0.6593714056338599 for stores.

This audit does not freshly refit every ensemble, classical model or comparator.
It does not establish bitwise reproducibility on all numerical environments.
The primary q=2 weight was chosen after sensitivity analysis, as already recorded
in `docs/q2-primary-2026-11-05/README.md`; it is an exploratory reporting choice.
