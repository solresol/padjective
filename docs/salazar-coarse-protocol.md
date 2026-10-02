# Single coarse-lift hypothesis test, 2 October 2026

Authorized continuation of the bounded single-member study. One warm Adam
member on raksasa, initial radius 71^-1 instead of 71^-7. Reuse saved fitted
centers, original snapshot/fold, feature order, support mask, seed 42, batch
32, prime 71, precision 7, kappa 1, beta .9, beta2 .999, epsilon 1e-8 and
the unchanged bounded optimizer. No exact refit, grid, ensemble or extra seed.

Start at zero histories. Radius 71^-1 preserves the first fitted digit in
each initial coefficient disk; the retained canonical centers keep the
original point predictor before movement. Fix one endpoint at the first of
1,000 completed updates or the 120-second fitting ceiling. Do not select a
checkpoint by training or held-out loss. The optimizer checks time between
updates, so one final update may cross the ceiling slightly. Outer process
timeout 180 seconds includes loading, fitting, freezing and scoring.

Diagnostics count toward fitting time. Log compact states/motion checks and
aggregate slope/history/clipping/vertex/digit/tie effects every update; save
full derivative vectors at updates 1,2,3,10,every100 and the fixed endpoint.
This avoids serializing all child vectors every update. Record callback time,
actual updates and sampled rows. The same ceilings do not guarantee the same
work or diagnostic overhead as the original fine-lift run. No efficiency or
convergence claim follows from a time-limited endpoint.

Use one nice-15 process and one numerical thread. Verify resource/load and
artifact absence first. Refuse to overwrite a completed or partial run. Save
progress locally and coefficients/radii to a new UUID row in the existing
padjective.paper_salazar_single_runs table (verified pg_default). Leave all
original rows and manuscript work untouched.

Freeze and persist the endpoint before generating any new held-out prediction
or reading original held-out scores for comparison. Extract canonical centers
and use the existing training-fitted zero-score default. Then compute paired
raw/default taxonomy loss, exact/root accuracy, gold-depth-capped shared
prefix, stored nonzero coefficients, prediction-wise active support, fallback
use and runtime. Audit independently from frozen Postgres coefficients and
rows. Preserve the exact baseline and both warm initializations in the report.

The held-out fold was previously inspected while designing this hypothesis;
the comparison is exploratory and not an independent confirmatory test.
There is one predetermined coarse initialization and no adaptive held-out
tuning or winner selection. Initial relaxation and future tie groups change
with the radius, so even an improved endpoint cannot isolate clipping alone
as the earlier failure's cause. Paper/adapter differences remain applicable.

Stop after this comparison and report a clear bounded conclusion. No further
experiment is automatically authorized. Private traces/coefficients stay on
raksasa/Postgres; Git and the same private Library report/patch contain code,
protocol and aggregate evidence only. No pushes or publication.
