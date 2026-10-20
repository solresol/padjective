# Reference committee fraction sweep

Fixed 20 October 2026 before viewing held-out scores for the four new rates.
Keep the manuscript, earlier experiment rows, original reports and figures
unchanged. Reuse the same frozen Shopify Postgres snapshot, five folds,
features, p=71, precision 7, strict repeated-coordinate fits and full-ensemble
inference as the [50% experiment](half-reference-ensemble-protocol.md).

New reference fractions are **10%, 25%, 75%, 90%**. At member t>1 sample
ceil(percent*(t-1)/100) previous members uniformly without replacement, shared
across all training rows. Fit only whole-path errors under that sample's
valid-training-path consensus. Draw a fresh committee for each member.
Use the unchanged NumPy SeedSequence([20261020, fold, roster, previous_count])
and rng.choice(previous_count, sample_count, replace=False). Only sample_count
changes with fraction. Its 50% sequence must exactly reproduce the previous
experiment. Sampling remains separate from coordinate fitting and label-free.
Defaults use only selected rows. Every fitted member votes on every product.

Use the identical primary seed order through 243 members and offsets 9 and 18
through 81 each, all five folds: **2,025 fits per fraction, 8,100 new fits**.
Evaluate every prefix. No label-driven seed/rate selection, stopping or grid
extension. A perfect sampled reference stops its trajectory with the existing
`reference_perfect` convention; retain the current full-ensemble predictor.
Reject incomplete fits. Use at most six single-threaded workers on raksasa;
preflight found 12 logical CPUs, 11 GiB available RAM and 113 GiB disk space.
Interleave fractions and folds; finish all fitting before any new scoring.

Reuse the validated 50% report, batch
`bb57fea7-2e63-441a-ab03-16e853671090`, and its 100%-reference and independent
controls from `06577b0c-a433-42bd-b8cd-a865838c11ad`. Freeze their identity,
digest and all 6,075 evidence rows before fitting. Verify DB readback at audit.
Refit each first member against its bank coefficient/default/prediction
fingerprints. Before a fraction first excludes a previous member, its entire
prefix must reproduce the 100%-reference control: through size 2 for 10/25%,
4 for 75%, 10 for 90%. Keep all existing rows untouched.

Persist a master sweep and four separate child batches, including private
coefficients, row masks, reference member indices, defaults, predictions,
histories and hashes in the `padjective` schema. Explicitly create all objects
in `pg_default`. Only aggregate scores and fingerprints may be exported.
Replay all samples and masks; independently reconstruct integer predictions,
defaults and supported-coordinate certificates on the actual subsets. Compare
reference and full votes with the established prefix decoder at the existing
checkpoints and all metrics under both integer implementations. Require all
60 new trajectories, the full prefix grids and unchanged controls before a
validated master report. Resume only matching source/protocol/manifests.

Report equal-fold mean held-out p-adic loss (primary), exact and root accuracy,
fold curves, training errors, selected training rows, forgotten correct rows,
stored coefficients and active consultations. Show fraction curves through
243 for the primary order, all three orders through 81, and comparisons at
matched member counts against 50%, 100% and independent fits. Show all rates
without choosing a winning rate from the held-out scores. Display the existing
sizes 1,3,9,15,27,45,81,135,243, with observed minima explicitly marked post hoc.

For patterns, describe late-size behaviour over 81–135 and 135–243, the
even/odd full-ensemble loss gap over 135–243, and variation across the three
orders through 81. Also compare losses conditional on the actual reference
committee's odd/even size; the ceiling rule produces different parity cycles
at different fractions. These descriptive checks are correlated observations,
not independent statistical trials or a proof of an asymptotic floor. Best
prefix/rate selections remain exploratory on repeatedly reused folds. Write a
separate Markdown report, aggregate JSON and static research figures; inspect
the figures before delivery. No manuscript changes or model deployment.
