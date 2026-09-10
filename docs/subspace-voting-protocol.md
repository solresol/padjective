# Random feature subsets and ensemble voting: 11 September 2026

This protocol is fixed before fitting or inspecting any new subset-model
held-out scores. The earlier all-feature ensemble and decoder-ablation results
are already known. This is an exploratory extension on an extensively reused
benchmark, not a newly reserved test set.

## Fixed inputs and training

Read the frozen 6,693-product, 2,542-feature, five-fold input directly from
Shopify Postgres: snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`, digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Keep all observations; do not bootstrap rows. Eligible features are those with
nonzero support in the fitting fold, not those selected using held-out labels.

Feature fractions: 1/8, 1/4, 1/2, 3/4, and 1. For each fold/member, independently
permute eligible column indices with NumPy
`default_rng(SeedSequence([20260911, fold, seed_base]))`. Select the first
`ceil(fraction * eligible_count)` entries and sort their column indices before
fitting. The four masks for a member are nested; different members may overlap.
Fit seeds remain `seed_base + fold`, separate from feature-mask randomness.

Use the existing 243 `BANK_SEEDS` and primary nested sizes
1, 3, 9, 15, 27, 45, 81, 135, 243. Reuse the original 1,215 validated all-feature
fits from ensemble batch `059d7eb8-14b5-4d5d-857f-134a2098fd89`; require their
prediction hashes and fitting-file identity. Fit all 4,860 new subset members
from zero with unchanged p=71, precision=7, exact medoid coordinate updates,
strict improvements, 100-sweep and 300-fitting-second caps. Accept only complete
no-improvement sweeps. A separate direct-distance coordinate certificate and
Python-integer prediction reconstruction must pass for every new member.
Fit the zero-output default on the fitting fold only. No survivor-only grids.

Use at most four single-threaded workers on raksasa. Preflight: 38 GiB available
RAM, 366 GB free disk, load about 1.5. Persist independent batch/model/evaluation
rows in the padjective schema with explicit pg_default for tables and indexes.
Original production models, benchmark rows, manuscript results and frozen
release remain unchanged by this side experiment.

## Five aggregation rules

All votes have equal weight. All exact ties choose the smallest canonical code,
except the explicit stop-versus-branch tie, which chooses stop.

1. `raw_valid_medoid`: current minimum-total-p-adic-distance consensus over
   valid paths observed in the fitting fold, applied to default-adjusted raw
   component outputs. This is the reference.
2. `projected_medoid`: first project each component output to its nearest
   fitting-fold path (smaller-code ties), then take their p-adic medoid. This
   isolates the effect of moving the projection before aggregation.
3. `projected_plurality`: use the most frequent whole projected path. This is
   plurality, not a requirement that the winner exceed half the votes.
4. `projected_survivor`: first choose the most frequent root among projected
   paths. Retain only its voters. At each subsequent level, compare the count
   stopping at the current prefix with the total count continuing. Stop on
   ties. Otherwise discard stopping voters, choose the most frequent next
   branch (smallest digit ties), retain only that branch's voters, and repeat.
   Reported paths always belong to at least one surviving member; no unsupported
   internal taxonomy nodes are invented.
5. `raw_plurality_project`: take the most frequent default-adjusted raw code,
   then project that code to the nearest fitting-fold path. This checks a
   different placement of the validity decoder for whole-output voting.

Thus all five rules return fitting-fold paths, and all coincide for one member.
The three projected rules share identical voters. Differences against the
current raw-consensus rule can include both aggregation and projection-order
effects; do not attribute all of that difference to voting alone.

## Metrics, hypotheses and statistics

Primary outcome: equal-weight mean of five fold mean p-adic losses (lower is
better). Also retain exact accuracy, first-digit accuracy, prediction hashes,
depth, coverage of the feature union, actual selected-feature counts, stored
nonzero coefficients and mean active nonzero member terms. Fold-specific
outcome counts and complete grids must be available, not just the best cell.

Twelve primary two-sided contrasts, with one Holm family at alpha=.05:

- Four feature fractions versus all features at M=243 under raw_valid_medoid.
- Four alternative aggregators versus raw_valid_medoid, all features, M=243.
- Four subset-size-by-ensemble-size interactions: for each feature fraction,
  `(subset M243 - all-feature M243) - (subset M1 - all-feature M1)` under
  raw_valid_medoid. Negative means ensembling helps the subset model relatively
  more than it helps the all-feature model.

Use five paired fold differences. Approximate overlap-corrected t standard error:
`sqrt((1/5 + mean(n_test/n_train)) * sample_variance(differences))`, four degrees
of freedom; report signed mean difference, two-sided p, pointwise 95% interval,
Holm-adjusted p, and Bonferroni simultaneous intervals for the 12 primary
contrasts. This Nadeau-Bengio-style correction is a sensitivity-based approximate
test for this single five-fold partition, not an exact calibration guarantee.
Do not treat products, seeds, nested sizes or alternative rosters as independent
training-sample replications. A non-significant result is not equivalence.

For secondary screening, compare each of the 216 non-reference fraction/rule/size
cells against the all-feature raw_valid_medoid at the same size, with a separate
Holm correction over all 216. Add paired contrasts of each projected voting rule
against projected_medoid at each fraction for M=243 (10 contrasts; separate
Holm family), to distinguish voting from member projection. Report all results.
Primary and secondary findings must remain labelled separately.

As roster sensitivity, evaluate 20 fixed bank permutations at sizes 9 and 81
only, using seeds `20260911 + 100*roster + fold`, roster 1..20. These describe
member-selection variability conditional on the fitted bank and are not extra
degrees of freedom for statistical tests. At M243 the roster is invariant.

No configuration is promoted into the main paper or its active-support
regression automatically. In particular, feature-dropping ablations remain a
separate experiment, not additional observations on the original scaling line.

## Theory and sources

Investigate feature-union invariance, common-kernel invariance, finite-precision
class-size bounds, strict-majority agreement, cross-subset interactions and
counterexamples distinguishing the three voting rules. State whether each result
concerns representability, the fitted optimiser, or generalisation. Elementary
derivations are not to be described as novel without a separate literature check.

Methodological sources checked on 11 September 2026:

- Tin Kam Ho (1998), *The Random Subspace Method for Constructing Decision
  Forests*, IEEE TPAMI 20(8), 832–844, DOI 10.1109/34.709601. This motivates the
  general mask construction, not a claim that the present linear members are trees.
- Nadeau and Bengio (1999), *Inference for the Generalization Error*:
  https://papers.nips.cc/paper/1661-inference-for-the-generalization-error.pdf
- Official corrected-CV comparison implementation and limitations:
  https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_stats.html
