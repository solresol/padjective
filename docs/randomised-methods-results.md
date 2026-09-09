# Randomised-method results, 9 September 2026

Repeated coordinate updates and small linear ensembles both helped on the
paper dataset. The nine-member coordinate ensemble achieved mean held-out
p-adic loss **0.161197**, compared with **0.213885** for the strongest single
linear control in this experiment. It had lower loss in every fold.
Uniformly randomising Zubarev's input order did not produce a competitive
polynomial model at either tested degree.

All 110 predeclared fits and 55 fold-by-ensemble-size evaluations are complete
and independently checked. Assessment: **share with caveats**. These are
controlled results on one frozen dataset, not population-level guarantees.
The manuscript, existing benchmark rows, production models and frozen release
were not changed. The existing six-model active-support analysis remains
separate; none of these models has been inserted into it.

## Repeated coefficient updates

All coefficients start at zero. A random sweep visits every supported tag,
including tags visited in earlier sweeps. Each update finds the exact best
residual medoid, accepting a strict decrease in the raw training objective.
A complete no-improvement sweep terminates fitting. A separate control uses
training-taxonomy association order on the first sweep, then random sweeps.

| Linear method | Fits | Mean held-out loss | Root accuracy | Exact-path accuracy |
|---|---:|---:|---:|---:|
| Random order, one pass | 45 | 0.306163 | 69.49% | 44.32% |
| Random order, repeated sweeps | 45 | 0.271032 | 73.00% | 47.06% |
| Association order, one pass | 5 | 0.257369 | 74.36% | 50.23% |
| Association first pass, then random sweeps | 5 | 0.213885 | 78.71% | 53.49% |

Every repeated fit improved held-out loss over its own first-pass checkpoint:
45/45 random-order pairs and 5/5 association-initialised pairs. Random-order
fits converged after 4–7 sweeps; association-initialised fits after 5–6.
Independent final-state checks examined 127,090 supported coordinates across
the 50 fits and found no improving single-coordinate change.

This is a coordinatewise optimum, not a global-optimality claim. Tests include
a case where no single change improves the model but a joint change does.
Training monotonicity concerns the raw objective; the fitted zero-score
default is applied afterwards. Both raw and default-adjusted results are saved.

These new fits use exact integer loss comparisons and canonical residues
modulo 71^7. The paired first-pass controls isolate revisiting coordinates.
The historical greedy reference (0.263237) uses its original implementation;
it was neither replaced nor rerun, and differences from it cannot all be
attributed to revisiting coordinates.

## Ensembling

Members are the first 1, 3 or 9 predeclared seeds, with no held-out selection.
The output is the training-supported taxonomy path with minimum total p-adic
distance to the member predictions. Encoded labels are not averaged.

| Random-order linear members | One-pass ensemble loss | Repeated-coordinate ensemble loss |
|---:|---:|---:|
| 1, valid-path projection only | 0.316073 | 0.277064 |
| 3 | 0.219468 | 0.196485 |
| 9 | 0.177023 | 0.161197 |

The one-member rows use seed base 42, not the mean of all nine seeds. For that
same coordinate model, loss before projection was 0.277381: projection alone
accounts for only 0.000317 of improvement. Combining models is responsible
for the much larger reduction. The machine-readable report retains matched
one-member controls for every ensemble family.

The nine-member coordinate ensemble has 83.99% root accuracy and 55.50%
exact-path accuracy. Its loss is 24.63% lower than the association-initialised
single-model control. Both three- and nine-member coordinate ensembles beat
their corresponding one-pass ensembles in all five folds.
The three-member coordinate ensemble has 52.45% exact-path accuracy, slightly
below the single association-initialised model's 53.49%; nine members improve
both the distance loss and exact-path accuracy.

There is extra prediction work. The nine-member coordinate ensemble consults
12.02 nonzero member terms per product on average, versus 1.07 for the single
association-initialised model. The consensus implementation additionally
checks 359–363 candidate paths through seven prefix depths: 2,513–2,541
candidate-prefix evaluations and 63 member-prefix observations per product.
These counts are different operations and should not be added as if they
were interchangeable. Consensus alone took 0.22–0.24 seconds per held-out
fold on this host. Nine member fits took a summed 25.54 fitting seconds per
fold, versus 2.89 seconds for the single association-initialised fit, excluding
data loading and final scoring. The ensemble is not one linear model.

## Randomised Zubarev representations

Input order is randomly chosen once per fit, using training-supported tags;
the coefficient sampler within each fit is unchanged. This tests independent
random representations, not root-order changes inside a single published
transition chain. Every fit starts with fresh random coefficients, without
greedy seeding or coefficient transfer.

| Representation | Seed bases | Raw model loss | Three-member consensus | Nine-member consensus |
|---|---:|---:|---:|---:|
| Degree 357,910 | 9 | 0.561501 | 0.560455 | 0.560460 |
| Degree 357,911 | 3 | 0.562024 | 0.560939 | Not run |

The matched first-three-seed raw degree-357,910 mean is 0.561935. Thus the
one-term boundary extension gave no mean improvement in this paired
sensitivity check. It does expose the fourth input coordinate at the root;
it does not supply all fourth-coordinate interactions or the next complete
polynomial expansion. A unit test checks that distinction.

The preceding frequency-ordered degree-357,910 reference was 0.564095;
the existing constant baseline is 0.560442. Those runs remain historical
references, not new controls fitted here. The random-order polynomial models
and ensembles remain close to the constant baseline. All raw polynomial
models have zero exact-path accuracy.

There is an identifiable representation limitation: across the primary fits,
only 0.51% of training products on average have any of the three root-visible
random tags. Almost all inputs therefore share the all-zero root signature.
This descriptive training-feature check did not select the input orders.
The result concerns uniform random ordering and these truncations; it does
not rule out better encodings or other stochastic representation searches.

## Evidence and scope

The [predeclared protocol](randomised-methods-protocol.md) fixes the original
6,693 products, 2,542 available tags and five stored folds from Shopify
Postgres. Means weight folds equally after averaging the specified seeds
within each fold. The same nine seed bases are used for the random linear
and primary polynomial fits; the boundary degree uses the first three.
Seeds are repeated optimisation runs, not independent population samples.
All choices and resource limits were frozen after training-only pilots.

All 50 linear fits converged; all 60 polynomial fits completed 320 accepted
transitions on the fixed five-stage schedule. There were no full-grid
failures, timeouts, survivor-only ensembles or replacement seeds. Fitting
ran from 18:34:05 to 18:51:14 AEST on raksasa with three single-threaded
workers; total summed job time was 3,102 seconds. Validation took another
411 seconds. These are shared-host observations, not portable time bounds.

Independent reconstruction checked 856,704 training and 214,176 held-out
model predictions. Direct exhaustive candidate scoring agreed with all
73,623 ensemble predictions. Integer-numerator metrics matched the saved
scores. Exact medoid, tie, convergence, overflow, representation-boundary
and consensus tests passed. The final local suite was **213 passed,
6 skipped**; the skips require a separate test database, not production data.

Configurations, coefficients, row-aligned predictions and ensemble evidence
remain in the new `padjective.paper_randomised_method_runs`,
`padjective.paper_randomised_ensemble_runs` and
`padjective.paper_randomised_validations` tables, all verified in `pg_default`.
The [aggregate-only result record](randomised-methods-results.json) contains
every seed/fold metric, configuration identifiers, costs and fingerprints.

- Batch: `a2b0744f-175e-4899-a3d5-e6a8478a8d73`.
- Snapshot: `244ddbe3-0a2c-4c04-9436-0a0253108a09`.
- Snapshot digest: `039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
- Fitting source: `289f5498c498c2c56eb13e31f29fa9958e049df1`.
- Validator source: `917da9f0be034a870f847c7f27399b8546239c8c`.

The results support adding repeated-coordinate fitting and its small
ensembles as separately identified paper comparisons. Keep their prediction
costs visible. The randomised polynomial result is a useful negative
sensitivity check, not evidence against every implementation of Zubarev's
method. Paper wording and table changes remain for the next discussion.
