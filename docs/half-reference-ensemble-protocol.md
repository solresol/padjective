# Error-only training against a random half of previous members

Fixed on 20 October 2026 before scoring this variant. Keep the manuscript and
all previous experiment rows unchanged. Use the same Shopify Postgres snapshot,
five folds, p=71, precision 7 and repeated-coordinate fitting recipe as the
[full-reference experiment](residual-ensemble-protocol.md). Data and private
evidence remain in Postgres; all new objects explicitly use `pg_default`.

Before fitting member t>1, sample ceil((t-1)/2) of its previous members uniformly
without replacement. Draw a fresh sample at each step, shared by every training
row. The sampling RNG is separate from coordinate fitting and uses NumPy
SeedSequence([20261020, fold, roster, previous_count]). Fit only rows whose
whole taxonomy path that sampled consensus gets wrong. Rows can re-enter.
Defaults use only those selected rows. Keep the original training-fold
candidate codebook, equal-weight p-adic consensus and smallest-code tie rule.

The first member trains normally on all rows. Every fitted member votes on
every product for final predictions. Neither sampling nor training masks use
held-out labels. Preserve the prior seed orders: primary roster 0 through 243
members, cyclic offsets 9 and 18 through 81 each. These sensitivity checks vary
both starting order and the subset-sampling stream on the same five folds.
Evaluate every prefix; no score-driven extensions, weights or early stopping.
If a sampled reference has no training errors, stop that trajectory, record
`reference_perfect`, and retain the full ensemble's current predictor. This
does not assert that the full ensemble is perfect. Reject incomplete fits.

Use the validated prior batch `06577b0c-a433-42bd-b8cd-a865838c11ad` for the
matched independent and all-previous-error controls at every size. Verify its
saved report/status and row identity. Refit the first new member and require
exact coefficient/default/prediction equality with its original bank member.

Store new evidence under a separate batch in `padjective.residual_ensemble_*`:
reference member indices and prediction hashes, selected row indices, fitted
coefficients, defaults, prediction vectors and histories. Replay every sample
and mask. Independently reconstruct integer predictions/defaults and certify
all supported coordinates on the actual subsets. Check reference and full
consensus against the existing prefix decoder at predeclared sizes; test the
new vectorised decoder against both existing integer implementations.
Separate fitting from scoring. Run at most three single-threaded workers on
raksasa after resource preflight. Require all 15 trajectories before export.

Report equal-fold mean held-out p-adic loss, exact-path and root accuracy,
per-fold curves, training errors and rows selected, forgotten correct training
rows, stored coefficients and active coefficient consultations. Compare all
three methods at matched sizes, show every prefix, and examine whether the
even/odd oscillation persists. Report observed best sizes as exploratory
selection, not a validated stopping rule. Finite late-size behaviour cannot
prove an asymptotic floor. Export aggregate-only JSON, a separate report and
static figures; do not overwrite previous reports or manuscript assets.
