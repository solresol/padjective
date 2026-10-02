# Synthetic validation protocol, 2 October 2026

Authorized after review of the initial product experiment. Run only on raksasa,
one nice-15 process, numerical thread limits one. No product rows, ensemble,
rate grid or new seed search. Preserve original product results and traces.

Finite domain: p=3, precision 3, canonical coefficients modulo 27. Five binary
rows: (1,0), (0,1), (1,1), (1,1), (1,1); targets 1,26,0,0,0. Start centers 0,0,
radii 3^-3 and zero optimizer histories. Exhaust all 27 choices of either
coordinate and all 27^2 joint states with independent Python integer scoring.
Confirm start loss 2/5, coordinate optimality and joint solution (1,26) of zero
finite-residue loss before fitting.

Frozen grid, kappa 1, beta 0.9, beta2 0.999, epsilon 1e-8:

* Full-batch GD/Momentum/Adam: seed 42, 100 updates, all five rows each update.
  This is an expected stationary control, not an escape requirement.
* Controlled GD/Momentum/Adam: seed 42, 400 updates. Updates 1–200 use exactly
  rows 0 and 1; updates 201–400 use all five rows. This deliberately curated
  gradient stream checks an escape path; it is not unbiased training evidence.
* Uniform minibatch Momentum/Adam: seeds 42 and 1729, batch sizes 1 and 2,
  1,000 updates each. Report all eight trials, including failures, with no
  endpoint selection or deadline extension.

Each fit checks a five-second wall-clock ceiling between updates. All 14 fits,
certification and trace writing share a 60-second overall ceiling; an external
75-second timeout also includes startup. Diagnostic overhead is included.
Stop on a failed mathematical assertion. A time-limited fit retains its actual
update count. A repeated invocation refuses to overwrite prior traces.

Log every update: raw versus boundary-adjusted directional slopes, moments and
speeds, attaining ties and coupled groups, selected coordinates and counters,
before/after centers and radii, requested/shared applied time, limiting
coordinates, boundary hits, actual radius changes and numerical no-ops. Score
all five rows independently on every logged endpoint. Retain endpoint and best
trajectory point loss separately; a transient improvement is not a selected
model. Preserve the original proposal counter for compatibility, and report
strict state changes separately.

This toy is a whole-coordinate optimum, unlike the paper's single-digit
adverse example. The bounded binary adapter's precision, radius caps and
boundary histories are unchanged. Its full-batch stationary outcome would
follow nonnegative available directions, not establish an implementation bug
or a failure of the paper. Controlled/stochastic escape must be observed rather
than assumed.

Code: `padjective.paper_salazar_synthetic`; traces under
`artifacts/salazar-synthetic-20261002`; aggregate certification/results exported
to `docs/salazar-synthetic-results.json`. No private product predictions are
included in those synthetic artifacts.
