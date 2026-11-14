# Larger linear ensembles: follow-up protocol, 10 September 2026

This follow-up is motivated by the already-observed 1/3/9-member results.
The additional sizes, seed roster, stopping limits and analysis below are fixed
before looking at new held-out scores. It remains an exploratory extension on
the same five folds, not a new independent test dataset.

## Scope and fixed grid

Use the frozen 6,693-product, 2,542-tag, 363-path paper snapshot directly from
Shopify Postgres: `244ddbe3-0a2c-4c04-9436-0a0253108a09`, digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Do not change the paper, the six-model roster, production models, old results,
or immutable release. Do not add parameter-budget-matched ablations to the
cross-model regression.

Study only the random-order, exact repeated-coordinate linear model that
formed the previous nine-member ensemble. Coefficients start at zero; use
precision 7, p=71, 100-sweep / 300-fitting-second limits, exact medoid updates,
strict-improvement acceptance and training-only zero-score defaults.
Every fit must finish a complete no-improvement sweep. No incomplete-member
or survivor-only ensembles are permitted.

The bank contains 243 seed bases per fold. Retain the original nine, then
append Python 3.11 `random.Random(20260910).sample(range(100000000,2000000000),234)`.
Actual seed is base plus fold. The first nine fits per fold are reused from
validated batch `a2b0744f-175e-4899-a3d5-e6a8478a8d73` (source `289f549`).
The numerical fitting files must be byte-identical before reuse is accepted.
This requires 1,170 new fits, plus the 45 reused fits. Maximum size 243 is a
resource-bounded extension, not a held-out-selected stopping point.

Primary ensemble sizes: **1, 3, 9, 15, 27, 45, 81, 135, 243**. Use nested
prefixes of the fixed bank, with the same valid-training-path minimum-total-
p-adic-distance consensus and smaller-code tie rule as before. Reproduce the
old size-1/3/9 predictions exactly. Do not reselect seeds, weights or members
using held-out labels. Run at most four single-threaded workers on raksasa.
Preflight found 12 logical CPUs, 37 GiB available RAM and 366 GB free disk.
Previous fitting-only pilot timings establish the unchanged resource budget.

## Variation, trend and extrapolation

Report all five primary fold curves and their equal-fold means. Also evaluate
20 fixed alternative bank permutations, using NumPy seeds `20260910 + r*100 +
fold`, r=1..20. These show sensitivity to which members are included, conditional
on the fitted bank. They are not new independent training samples. At size 243
all permutations contain the same members, so their zero spread must not be
reported as zero uncertainty about generalisation or fresh model banks.

Fit two descriptive scaling models to mean loss versus ensemble size:
`L(m)=a*m^(-b)` and `L(m)=c+a*m^(-b)` with c>=0, a>=0, b>0. Retain their
residuals, fitted floors and leave-one-fold-out sensitivity. Fit through size
135 and assess the size-243 forecast before fitting the full curve. This is a
functional-form check along the size axis, not an independent product test.
Compare early-size extrapolation with the extended evidence; do not force a
zero asymptote or select the curve that produces an attractive crossing.
Use unweighted nonlinear least squares on loss (equal weight per size),
separately for the primary roster and the mean of the 20 alternative rosters.
Leave-one-fold-out ranges are sensitivity ranges, not confidence intervals;
keep the comparator target fixed at its full five-fold mean for these ranges.

The target is the best measured comparator on these folds, NN-2000 loss
0.0755591194166452; L1 logistic loss 0.08583859907596855 is a second reference.
These are not a claim of field-wide state of the art. Report a crossing only
as a conditional ensemble-size estimate. If the fitted floor is above target,
report no crossing under that model. Report instability and weak fit rather
than a precise forecast. Reaching a target on a fitted curve does not establish
that an actual larger ensemble will reach it.

## Log-log relationship

Reconstruct the original six primary points with NN-2000 and its reference
thread result. The existing active-support line stays frozen. Its nominal OLS
statistics describe six configurations on one dataset, not a universal law.

Overlay ensemble points using (a) summed member nonzero-coefficient
consultations, matching the original greedy counter's convention, and (b) a
separate broader scoring counter adding default uses, seven candidate-prefix
checks per available training path and seven member-prefix observations per
member. The latter adds heterogeneous operations; it is a sensitivity proxy,
not a measured runtime or an equivalent parameter unit. Also report global
stored nonzero coefficient counts separately from per-prediction support.

For each size, report the frozen line's predicted loss, observed/predicted
ratio and vertical log residual. Refit the original six plus one ensemble
point at a time, especially the predeclared largest size; report nominal slope,
R-squared and p-values for both counters. Also expose the mechanical all-size
refit but flag its p-value as unsuitable for independent-observation inference:
nested ensembles and repeated sizes are correlated points from one family.
Do not claim significance is rescued or destroyed by treating them as
independent replications.

## Validation, persistence and chart contract

Reconstruct all final model predictions with Python-integer coefficient sums,
recompute defaults/losses independently, and re-audit every supported coordinate.
Verify every primary consensus output against the existing prefix algorithm;
the faster cumulative direct-distance evaluator is a computational reuse for
the bank-permutation analysis, not a change in the ensemble decision rule.
Test cumulative evaluation, tie handling, order invariance, fitted curves and
crossing/no-crossing cases on synthetic examples. Require the complete grid.

Persist new fits, private predictions, ensemble evidence and aggregate reports
in separate batch rows/tables in `padjective`, with explicit `pg_default`.
Export aggregate metrics, model identifiers and fingerprints only.

Visuals: (1) ensemble-size/loss curve with all nine ordered sizes, five-fold
context, roster sensitivity and the two comparator levels; (2) the original
six-point log-log active-support scatter with frozen line and labelled
ensemble trajectory, using separate panels for the two counting conventions.
Retain fold counts, accuracy, stored coefficients and scoring counters in the
supporting data. Use blue ensemble marks, neutral references, dashed forecast
segments and explicit log-axis labels. Inline chart widgets are not available
in this tool session, so export reproducible static research figures and
inspect them before delivery. Keep extrapolation distinguishable from measured
sizes. No manuscript figures are overwritten.
