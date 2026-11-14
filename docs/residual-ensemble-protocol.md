# Error-only training ensembles, 20 October 2026

This exploratory experiment does not modify the manuscript. Fix this protocol
before inspecting the new held-out scores. Reuse the Shopify Postgres snapshot
`244ddbe3-0a2c-4c04-9436-0a0253108a09`, SHA-256
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`:
6,693 products, 2,542 binary features, 363 paths, five frozen folds. Results
remain in the `padjective` schema, with explicit `pg_default` tablespaces.

The first member uses every training row. Before each subsequent fit, decode
the preceding members' equal-weight minimum-total-p-adic-distance consensus
over the valid paths in the original training fold. Retain precisely the
training rows whose **whole path** is wrong. A row can re-enter when a later
vote becomes incorrect. This is a freshly computed mask, not permanent removal,
not a union of individual members' successes, and not numeric residual addition.
Fit a new zero-initialised repeated-coordinate model on the retained rows,
with its zero-score default fitted on those rows alone. Keep the original
feature columns and original training-fold candidate codebook. Use p=71,
precision 7, strict integer improvements, 100 sweeps and 300 seconds per fit.
Reject a member unless it completes a no-improvement sweep. If there are zero
remaining errors, stop fitting that trajectory and retain its final predictor.

For prediction, **every fitted member votes on every product**. True held-out
labels cannot identify which products need a specialist. No oracle gating,
validation-label masks, member weights, seed selection or score-based stopping.
The question tested is adaptive training with the established inference rule.

Use the existing 243-member bank's fixed seed bases, actual seed = base + fold.
Primary order is the original order, through 243 members. Two additional
trajectories use cyclic rotations starting at bank indices 9 and 18, through
81 members each. These are starting-order sensitivity checks on the same folds,
not independent datasets. Evaluate every prefix; highlight the existing sizes
1,3,9,15,27,45,81,135,243. The limit is fixed independently of held-out scores.
At most three single-threaded worker processes run on raksasa after preflight.

Use the validated independent bank
`059d7eb8-14b5-4d5d-857f-134a2098fd89` as a matched baseline: same seed order,
member budget, full training rows, defaults and consensus. Reconstruct its
predictions from saved coefficients and recheck fingerprints. The first new
fit in each trajectory must reproduce its matched baseline exactly.

Persist coefficients, masks, member predictions and fit histories privately.
Training and evaluation are separate commands; held-out labels are not passed
to the fitting algorithm. Replay every training mask, independently reconstruct
integer predictions and defaults, certify supported coordinates on each actual
training subset, and compare cumulative distance consensus with the prefix
implementation at the highlighted sizes. Independently recompute metrics with
integer distances and prefix histograms. Require all trajectories before export.

Report equal-fold mean held-out p-adic loss (primary), exact-path accuracy and
root accuracy, individual fold curves, training error counts, the number of
rows used for each fit, forgotten correct rows, stored nonzero coefficients,
and active coefficient consultations. Also report a common full-training
objective to expose changes caused by fitting only errors. Show primary curves
through 243 and all three starting-order curves through 81; do not average an
unbalanced roster at larger sizes. Describe late-size changes and observed
best prefixes; a held-out-selected best size is exploratory, not a validated
stopping rule or a proof of asymptotic saturation. Export aggregate-only JSON,
a separate Markdown report and static figures. No product rows leave Postgres.
