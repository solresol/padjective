# Randomised representation, coordinate search and ensemble study

Authorised 9 September 2026. This is a new, separately identified experiment;
the published-method study, manuscript, production models and frozen release
remain unchanged while these hypotheses are tested.

## Questions and controls

1. Does randomising the input order improve Zubarev's finite polynomial fits?
   Within each fit the order remains fixed and the published coefficient
   transition sampler is unchanged. This is independent representation search,
   not a claim that the original transition law changes input coordinates.
   Primary degree 357,910; a paired boundary check at degree 357,911 introduces
   the fourth root-visible coordinate without the next complete expansion.
2. Does exact coordinate minimisation with repeated random sweeps improve over
   a one-pass fit? Save the first pass and final model from the same trajectory.
   A training-taxonomy-association first pass is a separate control; subsequent
   sweeps are random. A full sweep with no strictly improving update certifies
   a coordinatewise optimum. Time/sweep limits do not certify convergence.
3. Do fixed small ensembles help? Use valid-path minimum-total-p-adic-distance
   consensus. Include a one-member decoder control: projecting to a valid path
   must not be mistaken for an ensemble benefit. Do not average encoded labels.

## Fixed data and arithmetic

Read the original 6,693-product paper snapshot and five stored folds directly
from Shopify Postgres. Snapshot 244ddbe3-0a2c-4c04-9436-0a0253108a09;
digest 039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222.
The 2,542 available tags and snapshot eligibility rules are not refitted.
All fitting, feature ordering, defaults, termination and ensemble choices use
only the fitting folds. Consensus candidate paths are the distinct training
labels, sorted numerically, with smaller codes breaking ties. This does not
use held-out labels to define the decoder's support.

New linear fits use canonical residues modulo 71^7 and exact integer loss
units, not floating-point approximate ties. Coefficients start at zero.
At each coordinate undo its current contribution, compute an exact residual
medoid, and accept only a strict reduction of the raw fitting objective.
Keep the current coefficient on ties. Empty-support tags remain zero.
Each new sweep independently permutes every training-supported feature.
Final zero-score defaults are fitted separately to training labels by the
existing medoid/mode rule; raw and default-adjusted metrics are both retained.
Consequently the paired one-pass control, not the older floating-point/raw-
integer greedy run, isolates revisiting coordinates. Training monotonicity
does not imply monotonicity after default substitution or on held-out data.

Zubarev orders randomly permute training-supported tags in a separate random
stream, with zero-support tags appended. Every fit starts with fresh random
coefficients; no greedy seeding or transfer. Degree, full order, seed, accepted
transition count and completion status are retained. Polynomial predictions
remain raw canonical residues, with no fitted default; only the separately
labelled consensus experiments project them onto valid paths.

## Planned grid and training-only pilots

Seed bases, in fixed ensemble membership order:
42, 1729, 20260907, 104729, 130363, 155921, 181081, 205019, 230003.
Actual seed is base plus fold. Ensemble sizes 1, 3, 9 use prefixes of this
roster, never best-performing seeds. The boundary-degree sensitivity uses
only the first three seeds, with sizes 1 and 3 and a matched three-seed
primary-degree comparison.

Training-only fold-0 pilots establish budgets before any new held-out scores:
random and association-initialised linear trajectories, plus random-order
polynomial fits at degrees 357,910 and 357,911. Pilot resource settings and
observed completion are recorded below before the complete grid is dispatched.
The default proposed full polynomial schedule is beta 0/4/16/64/256,
64 accepted transitions per stage, 200,000 proposals per stage and 300 sampler
seconds, matching the preceding study. Construction and prediction are timed
separately. No timeout may be relabelled as a completed schedule.

Full grid: five folds times nine random linear trajectories and one
association-initialised control, plus five folds times nine primary polynomial
fits and three boundary fits: 110 jobs. Linear trajectories retain both their
one-pass checkpoint and their endpoint. Ensemble formation requires every
predeclared member and its appropriate completion status; no survivor-only
ensembles or held-out seed selection.

Use at most three single-threaded workers on raksasa after checking load,
memory and disk. Persist run configurations, model evidence and predictions
in a new padjective table with explicit pg_default tablespace. Export only
aggregate metrics, run identifiers and fingerprints to Git. Retain private
row-aligned predictions, coefficients and feature orders in Postgres.

## Validation and reporting

Check medoids against exhaustive small finite rings; verify update monotonicity,
tie retention, full-sweep stopping and joint-change local minima. Independently
reconstruct predictions and metric numerators. For converged linear fits,
audit every coordinate again against the final state. Check consensus against
exhaustive candidate scoring, including invalid member outputs and ties.
Require the complete predeclared run grid and immutable snapshot digest.

Report all seed/fold results, fold means after seed averaging, fixed-membership
ensembles, paired first-pass changes, root/exact accuracy, prediction cost and
worker wall time. Member support and consensus work are reported separately;
an ensemble does not inherit one member's cost or remain a single linear model.
Seeds are repeated optimisation runs, not independent population samples.
The old greedy/classical results remain references, not rerun replacements.
