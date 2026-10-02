# Bounded single-member tree continuation, 2 October 2026

Authorized host: raksasa, existing SSH from udara. Isolated worktree:
`/home/gregb/experiments/padjective-salazar-single-20261002`, branch
`codex/salazar-single-20261002`, base `6e0dbffa481437c86ef45a540126a1644e70a41e`.
No pushes, manuscript edits, production model updates, ensembles, services or
package installations. One process, nice 15, all numerical threads 1.

Sources checked: [Salazar et al., v1](https://arxiv.org/html/2609.25501v1)
and [official library](https://github.com/google-deepmind/padic-ml), pinned at
`53ffd9e79890e76b82ef2dc3c7230984053f9801`. The library README identifies ongoing
refactoring; this release has array operations but no optimizer. This is an
independent adapter, not an official-library replication.

## Frozen budget and endpoints

Use original Postgres snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`, digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Fold 0 is held out; all other stored folds train. Seed 42. All 2,542 available
binary tags remain aligned; freeze unsupported training features at coefficient
zero and radius zero. No intercept or feature selection is added. Prime 71,
precision seven, canonical coefficients modulo 71^7 throughout.

1. Refit the existing exact randomised linear member from zero, with the
   historical seed/order; limit 100 sweeps/300 fitting seconds. Audit every
   coordinate at the endpoint and compare coefficients with the historical
   seed-42 member if present. If not certified, stop.
2. Lift the SAME fitted coefficients to radius 71^-7. Continue separately
   with Momentum and Adam, each at 1,000 updates/120 seconds, batch size 32,
   seed 42, kappa 1 (reference radius 1), beta 0.9, beta2 0.999, epsilon 1e-8.
3. Secondary control: Adam from centers zero and radius 1, same mask,
   precision, batches, seed, update/time ceilings and rate.

Freeze all endpoints before generating new held-out predictions. Select no
best checkpoint, learning rate or seed. Checkpoints every 100 updates contain
training scores only. Use the retained regression representatives (paper
Section 5) modulo 71^7 for final predictions; do not use the classification
extraction in G.3. Retrain the existing zero-score default on training rows for
each endpoint. Report raw and default-adjusted scores separately, never tree
training loss as final prediction loss. No valid-path projection is added.

Hard outer execution limit: 900 seconds for the experiment process; each
optimizer checks its 120-second ceiling between updates. Loading, certificates,
checkpointing and final audit are timed separately or explicitly included.
If a method times out, report its actual updates/work; never claim equal work
solely from the same ceilings. Include incremental and warm-fit-inclusive times,
sampled example counts, coordinate updates and coefficient changes. The exact
control is certified stationary for its coordinate objective; further exact
sweeps cannot strictly improve it. This is an exploratory comparison with
matched capacity and starts, not a matched-operation efficiency benchmark.

## Mathematical contract and limitations

For binary features, the affine output center is the sum of present
coefficients, and its radius is the maximum of their radii (Eq. 7). Direct loss
is max(p-adic residual norm, output radius) minus half the output radius (Eq. 9).
At a tied maximum, raising one coordinate changes the output radius; lowering
one leaves it fixed. A strict maximum has the Appendix F.1 half-unit loss slopes;
at a vertex at most one child points toward each target. Child changes replace
the appropriate base-71 digit and retain higher digits. Tests enumerate every
child at several primes and compare analytic slopes with finite differences.

Groups are connected components of tied attaining terms, selected using G.1's
awaiting-turn queue. Cancellation-based group splitting is omitted, so groups
can be coarser and yield a different schedule. Momentum retains upward/downward
histories and vertex-specific weighted excess/child derivatives using coordinate
clocks; Adam uses one second moment per coordinate and counter bias corrections.
All selected coordinates use a common time clipped at the FIRST reached vertex,
as in Eq. 12. Unselected histories remain fixed. Radius logs use float64 and
clipped vertices are exactly integer logs.

The disk interval is [71^-7,1]. Descent beyond seven digits and ascent outside
Z_p are disabled. Unavailable terminal downward derivatives enter histories as
zero; this is an explicit finite-capacity convention, not an equation claimed
to come from the paper. Within the interval, modulo reduction does not change
the direct disk loss: residual differences by multiples of 71^7 have norm at
most the minimum positive disk radius. At radius zero, evaluate the existing
finite-residue task instead of claiming exact Q_p regression equivalence.

Tied radii can produce zero coordinate gradients despite useful joint descent.
Grouping prevents incorrect simultaneous independent slopes; it does not solve
the full joint tangent optimization. Momentum may increase training loss. The
paper supplies no general convergence guarantee. One fold and seed, fixed rate,
conservative groups, and finite capacity limit conclusions substantially.

## Evidence placement

Read data directly from Shopify Postgres. Write private coefficients, radii,
checkpoints and row-aligned predictions to new table
`padjective.paper_salazar_single_runs` with explicit `pg_default` tablespace.
Git contains only adapter, runner, tests, protocol and aggregate output with
fingerprints. Reconstruct every train/test raw prediction using Python integer
sums independently of the sparse fitting arithmetic. Report taxonomy loss,
exact accuracy, gold-depth-capped shared prefix, stored nonzero coefficients,
prediction-wise nonzero support and fallback use. Parent review is required
before any broader run.
