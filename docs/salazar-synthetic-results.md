# Synthetic validation update, 2 October 2026

The adapter **can escape a certified whole-coordinate optimum** on the
five-row finite-ring toy. All three controlled schedules reached the unique
zero-loss solution, and two of eight predeclared unbiased stochastic trials
did too. The other six did not improve within their fixed budgets. This closes
the missing positive-control gap without establishing a general escape
guarantee or changing the earlier Padjective product-data result.

## Certification and controlled path

For p=3, precision 3, the five binary rows are (1,0), (0,1), and three copies of
(1,1); targets are (1,26,0,0,0). Coefficients (0,0) have raw point loss 2/5.
Independent Python integer scoring exhausted all **54 single-coordinate
candidates** and **729 joint states**. Neither coordinate can improve alone;
(1,26) is the unique joint minimizer, with zero loss modulo 27. The existing
exact coordinate fitter and certificate also verify the starting optimum.

| Controlled optimizer | First strict point improvement | First zero loss | Endpoint |
|---|---:|---:|---|
| GD | Update 8 | Update 14 | (1,26), loss 0 |
| Momentum | Update 20 | Update 33 | (1,26), loss 0 |
| Adam | Update 7 | Update 12 | (1,26), loss 0 |

Each schedule used exactly the two missed singleton rows for updates 1–200,
then all five rows for updates 201–400. Every post-switch endpoint stayed at
zero loss. This is a deliberately curated escape-path check, not an unbiased
learning result. It verifies growth, branch choice, refinement, history and
clipping from the certified starting state. The two stochastic escapes below
use the adapter's unchanged uniform minibatch sampler.

## Complete fixed stochastic roster

Each trial used 1,000 updates, zero histories, initial radii 3^-3, kappa 1,
beta 0.9, beta2 0.999 and epsilon 1e-8. There was no rate or seed search and no
best-checkpoint extraction. All endpoints are reported.

| Optimizer | Seed | Batch | First zero-loss update | Endpoint loss |
|---|---:|---:|---:|---:|
| Momentum | 42 | 1 | Not reached | 0.4 |
| Momentum | 42 | 2 | Not reached | 0.4 |
| Momentum | 1729 | 1 | 321 | 0 |
| Momentum | 1729 | 2 | Not reached | 0.4 |
| Adam | 42 | 1 | Not reached | 0.4 |
| Adam | 42 | 2 | Not reached | 0.4 |
| Adam | 1729 | 1 | 40 | 0 |
| Adam | 1729 | 2 | Not reached | 0.4 |

Both successful endpoints are (1,26). Every nonescape endpoint is (0,0), and
none of those six even transiently achieved a lower point loss. Seeds are two
predeclared optimization trajectories, not population evidence for an escape
probability. The eight outcomes do not estimate comparative optimizer quality.

## Stationarity and diagnosis

The three full-batch controls stayed at (0,0), with loss 0.4 over 100 updates.
At that lifted start, a coordinate's raw upward slope is +1/5 and raw downward
slope +1/10. At the lower capacity floor the adapter disables downward motion
and inserts zero into its downward history. The selected available directions
are nonnegative, so zero-history GD/Momentum/Adam have zero speed. This is
expected stationarity under the adapter's constraints and coordinate selection,
not a failed escape control or evidence against the paper.

The useful tied-downward-germ test now also verifies a distinct geometry case:
for F=c1+c2, target 0, equal open-edge radii and centers 0, each individual
downward derivative vanishes, whereas moving both radii down has derivative
-1/2. That can improve the relaxation without changing point coefficients.
One-coordinate-per-group selection does not solve joint tangent optimization.

All 14 traces log raw and boundary-adjusted slopes, moments, speeds, attaining
ties, coupled groups, selected coordinates, requested/common applied times,
limiting boundaries, center/radius changes, full-objective point loss and
numerical no-ops. A read-only audit verified **9,500 updates and 1,309 motion
proposals** against persisted before/after states, exact point scores and
shared clipping. All 1,309 proposals produced strict radius changes, and no
floating-point no-op occurred in this toy. This does not exclude numerical
problems at p=71 or much higher dimension.

Observed nonescape mechanisms:

* For seed 1729, batch 2, both Momentum and Adam had **198 selections with a
  negative feasible minibatch slope but zero speed after history averaging**.
  Neither made any proposal; clipping and floating-point motion were not the
  cause of their nonmovement.
* Momentum seed 42/batch 1 grew to log radius -0.2241 but never reached the
  root vertex; batch 2 reached only -2. Adam seed 42/batch 2 reached -0.9529.
  Their traces show repeated history suppression and clipped radial motion,
  with no coefficient-digit changes. These are observed histories under the
  declared budget, not a proof that a larger budget would escape.
* Adam seed 42/batch 1 did reach the root, with eight root-coordinate selections,
  but retained every digit. Three negative current child slopes were blocked
  by the vertex-history guard. Merely reaching a coarse vertex is not enough
  to select the jointly useful branches.

No implementation bug was found in certification, the controlled/stochastic
positive controls, derivative tests, or the persisted-state audit. This is a
bounded conclusion about tested cases, not proof that the adapter is bug-free.
The full-batch nonescape has a mathematical explanation; stochastic nonescapes
show momentum/history and coupling mechanisms. The causal contribution of
capacity restrictions or conservative grouping was not isolated by ablation.
The cause of the original p=71 negative result remains unresolved.

## Preserved method differences and next step

The adapter remains a binary-input, no-intercept, finite-residue implementation
with fixed absolute precision, radii in [p^-E,1], zero substituted for unavailable
terminal downward histories, constrained Adam second moments, and conservative
attaining-tie groups without cancellation splitting. Its centers cannot carry
negative valuations. The paper uses exact signed Z[1/p] representatives with
significant-digit capacity, a general tree space and additional affine grouping
tests. Its reported adverse start is a single-digit local minimum, whereas this
toy is whole-coordinate optimal. Positive controls do not remove those departures
or make this an official-library replication.

Recommended next action, **not executed**: repeat only this five-row toy at
p=71 and seven digits, replacing target 26 by 71^7-1. Use the same controlled
schedule, fixed seeds/batches, caps and full diagnostics. Certify coordinate
optimality with exact medoid/certificate reasoning and the known zero-loss joint
solution; exhaustive enumeration of (71^7)^2 states is inappropriate. This
isolates prime/precision coarsening from the 2,542-feature product problem
before considering any additional product-data run or ensemble.

## Reproduction and verification

Execution host: raksasa; source commit
`44dc8b3ee4da7b3c281ab0e1f0f38adcec4901f5`. The entire 14-trial run took
**5.03 seconds**, within five seconds per fit, a 60-second overall internal cap
and a 75-second external timeout. It used one nice-15 process and one numerical
thread. No product data were accessed and no original database rows changed.

* [Frozen protocol](salazar-synthetic-protocol.md)
* [Certification and all results](salazar-synthetic-results.json)
* [Trace audit and mechanisms](salazar-synthetic-audit.json)
* Full JSONL traces: `artifacts/salazar-synthetic-20261002` in the isolated
  raksasa checkout; about 8.2 MiB, with per-file hashes in the results export.

The suite now includes controlled escape from the certified whole-coordinate
optimum under GD/Momentum/Adam and the two observed fixed-seed stochastic
positive controls. It also verifies useful tied descent, terminal raw versus
adjusted slopes, batch-schedule validation and instrumentation that leaves
seeded trajectories unchanged. **304 passed, 6 skipped**; 12 pre-existing
sklearn batch-size warnings. The original 290-test run and product results
remain preserved separately.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 75s nice -n 15 uv run --no-sync -m padjective.paper_salazar_synthetic
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync -m padjective.paper_salazar_synthetic_audit
```

The runner refuses to overwrite existing traces; an explicitly authorized
reproduction must preserve and relocate the prior artifact directory first.
No broader runs, ensembles, pushes or manuscript changes were performed.
