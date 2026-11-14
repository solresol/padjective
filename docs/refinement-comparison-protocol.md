# Valuations and occupied powers: new saved-fit analysis

This analysis is separate from the published strict-refinement table. It does
not edit that table, any manuscript claim, or the original model evidence.

## Population and fitted model

Read the frozen PostgreSQL snapshot
`244ddbe3-0a2c-4c04-9436-0a0253108a09`: 6,693 products, 2,542 binary features,
363 taxonomy labels, five archived folds. Its ordered-record digest must equal
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.

Use the five saved `linear_association/one_pass` checkpoints in
`padjective.paper_randomised_method_runs`, batch
`a2b0744f-175e-4899-a3d5-e6a8478a8d73`, source commit
`289f5498c498c2c56eb13e31f29fa9958e049df1` (September 2026).
These are the **first complete association-ordered sweeps of the exact
finite-quotient coordinate fitter**, with canonical coefficients modulo
`71**7`. They are not the published floating-point one-pass greedy fit, nor
the later repeated-sweep endpoints. The analysis must fail on missing folds,
snapshot/order drift, noncanonical coefficients, or reconstruction differences.
No fitting, coefficient selection, or tuning is performed.

The frozen feature relation is authoritative. Snapshot construction applies
`parse_tags` and `filter_nested_tags` before anonymisation
(`product_taxonomy_bench.py`); the loader deduplicates feature IDs. Verify the
loaded matrix equals these exact per-product binary feature rows. Do not
reconstruct features from mutable raw product tags or substring-filter
anonymised IDs. Unknown/absent features contribute zero, as in the fitted model.

Independently reconstruct every saved training and held-out raw score by
integer addition modulo `71**7`. Recompute each fallback from its training
zero-residue rows with the original exact medoid/tie rule, then compare all
default-adjusted predictions and held-out metrics with the saved records.
Each product must occur in exactly one held-out fold.

## Two predicates

For one product, collect its nonzero contributions `z_j = w_j*x_j`, counted
once per active binary feature.

1. **Valuation separation:** the integers `v_71(z_j)` are pairwise distinct.
2. **Disjoint occupied powers at depth D:** write the canonical expansion of
   each `z_j mod 71**D`; its nonzero-digit index set must be disjoint from the
   corresponding sets of every other contribution.

The primary depth is **D=7**, matching the fitted quotient and maximum taxonomy
depth. All saved coefficients are canonical residues in `[0,71**7)`, so this
examines all digits of their stored representatives and no nonzero active
contribution is invisible. Disjoint support implies valuation separation here;
the reverse need not hold. This is not a claim about disjoint infinite supports
of arbitrary lifts in `Z_71`.

Negative integers have infinite canonical p-adic expansions. A signed lift must
be reduced modulo `71**D` before digit extraction; `abs(z_j)` is incorrect for
support extraction (although it is valid for valuation). The archived model
already specifies canonical residues, so its digit representation is fixed.

Report support sensitivity at depths **1 through 7**, holding predictions,
outcomes and the full-coefficient valuation predicate fixed. Shallow truncation
can make entire nonzero contributions invisible and disjointness vacuous.
Report their count and an additional comparison restricted to rows where
every active contribution remains visible. The selected canonical coefficients
have no digits above position 6, so depths above 7 give identical support
classifications and need no separate computation.

Zero coefficients are omitted. Both predicates hold vacuously for zero or one
active contribution, but the primary comparisons exclude these rows and
report them separately. A default is used whenever the raw residue is zero,
including cancellation/wraparound, and is not counted as a tag contribution.
Report empty-support zeros separately from multi-contribution modular zeros.

## Outcomes and stratification

On the identical held-out predictions, report sample size, exact-hit count and
rate, integer loss numerator with denominator `n*71**7`, mean p-adic loss, and
mean loss conditional on an incorrect prediction. Empty conditional groups
have `null`, not zero, loss. Correct cases contribute zero to loss.

Keep zero-active, one-active and multi-active cases separate. Within
multi-active cases, report each predicate and its complement, and the joint
partition: disjoint support, valuation-separated but overlapping support,
and non-separated valuations. Preserve aggregates by fold, taxonomy label and
active contribution count, including group sizes and represented taxonomies.
Pooled summaries weight products; fold-specific results are retained.

This is an exploratory post-hoc association. Group membership may track
taxonomy difficulty, sparsity or tag quality. Neither predicate was imposed
during fitting. No causal improvement, independent fold replication, or
out-of-sample intervention effect is claimed. Inspect both exact accuracy and
wrong-only loss before attributing an overall loss difference to closer errors.

## Execution, provenance and persistence

From a checkout with the locked Python 3.11 environment and Shopify Postgres:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync -m padjective.paper_refinement_comparison \
  --persist --output docs/refinement-comparison/results-2026-10-19.json
```

Choose a new output filename for reruns; the runner refuses overwrites. It
performs only bounded reads in a repeatable-read transaction, then optionally
appends to the dedicated `padjective.paper_refinement_comparisons` table in
`pg_default`, with a one-second lock timeout and a 20-second statement timeout.
It never writes source fit, production prediction or snapshot tables. No
parallel workers, training jobs, migrations or operational job controls are
started or changed.

The public JSON contains only aggregates plus source snapshot/model/code
hashes. Product keys, individual outcomes and active contribution lists remain
in the dedicated private PostgreSQL evidence record. Its digest is included in
the public report. Source model, coefficient, prediction and configuration
hashes bind the result to the exact saved fits; original fit records remain
untouched. Tests cover overlap/carry counterexamples, signs, truncation,
degenerate cases, exhaustive small-ring implications, wrong-only scoring,
and rejection of modified predictions/defaults.

After persistence, `scripts/audit_refinement_comparison.py` can read the private
rows back and independently recheck their sums, defaults, classifications and
all exported aggregates against the public report. It checks original model
hashes and reports reference-depth diagnostics for disjoint-support errors.
This replay performs no PostgreSQL writes:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync python scripts/audit_refinement_comparison.py \
  --report docs/refinement-comparison/results-2026-10-19.json \
  --output /tmp/refinement-replay-new-check.json
```

## Mathematical counterexamples

At base 2, contributions `3=1+2` and `2` have distinct valuations 0 and 1 but
share occupied power 1. Their sum is `5=1+4`, with a carry. Expanding the first
contribution into digit terms gives `1,2,2`, which fails valuation separation.
At base 3, `4=1+3` plus `3` overlaps without a carry; `4` plus `6=2*3` overlaps
and carries. Thus neither overlap nor carries are ruled out by distinct
valuations. The manuscript's base-71 worked coefficients likewise overlap at
power 2 (digits 20 and 53) and carry.

Disjoint support does not imply correctness: at base 3 contributions `1,3`
predict 4 and have maximal p-adic loss against target 2, whereas non-separated
contributions `1,1` predict that target exactly. With sorted valuations `r_j`,
adding the next term preserves digits strictly below `r_j`; overlapping higher
digits may change. This algebraic statement alone is not a theorem about loss.
