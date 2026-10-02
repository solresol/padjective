# p=71, seven-digit positive control, 2 October 2026

The unchanged bounded adapter **escapes the certified whole-coordinate
optimum at the product model's prime and precision**. All three curated
controls reached zero point loss. Three of eight predeclared stochastic
trials reached the same unique solution, compared with two at p=3. This
supports the tested escape mechanics at p=71; it does not explain or improve
the earlier 2,542-feature product result, establish general convergence, or
make the adapter equivalent to the paper's method.

## Valid bounded certificate

Let q=71^7=9,095,120,158,391. The unchanged five binary rows are (1,0),
(0,1), and three copies of (1,1), with targets (1,q-1,0,0,0). Initial
coefficients (0,0) have point loss 2/5. With the other coefficient fixed at
zero, either coordinate w has unnormalized cost

`C(w) = |w-s|_71 + 1 + 3|w|_71`, for `s=1` or `s=-1`.

If 71 divides w, `|w-s|_71=1`, so `C(w)=2+3|w|_71`; this equals 2 only
for the zero residue. If w is a unit, `C(w)=4+|w-s|_71 >=4`. Thus every
nonzero replacement of either coordinate is strictly worse. This proves a
strict whole-coordinate optimum for the entire finite ring without any
large enumeration. The singleton equations force the unique zero-loss
joint solution (1,q-1), which also cancels in the three joint rows. Since
point loss is nonnegative, that is a global minimizer.

The existing exact residual-medoid certificate independently examined both
coordinates and reported `coordinate_optimum`. Independent Python integer
scoring checked both initial and joint states, plus bounded probes for each
valuation stratum and both signs. Initial units are 18,190,240,316,782 with
denominator 45,475,600,791,955. The analytic certificate was also checked
against the preserved p=3 exhaustive certificate. No q or q^2 enumeration
was attempted.

## Matched controlled paths

All starts used centers (0,0), radius 71^-7 and zero histories. Settings,
mask, seeds, batches, kappa 1, beta 0.9, beta2 0.999 and epsilon 1e-8 match
the low-prime protocol. Exactly singleton rows 0 and 1 were supplied for
updates 1–200, then all five rows for 201–400.

| Optimizer | p=3 first improvement / zero | p=71 first improvement / zero | p=71 endpoint |
|---|---:|---:|---|
| GD | 8 / 14 | 11 / 21 | (1,q-1), loss 0 |
| Momentum | 20 / 33 | 19 / 34 | (1,q-1), loss 0 |
| Adam | 7 / 12 | 12 / 22 | (1,q-1), loss 0 |

Every post-switch endpoint stayed at zero loss. Each p=71 control crossed
the root into both required branches and made eight retained-center changes;
the p=3 controls made four. Those counts describe changes to representatives,
not numbers of differing predictions. The negative coefficient q-1 needs
seven nonzero base-71 digits, whereas the positive coefficient 1 needs one.
The schedules deliberately curate an escape path; they are not unbiased
training evidence.

## Complete stochastic roster

Uniform minibatches use the same sampler and fixed seeds, 1,000 updates each.
No seed search, tuning, extension, best-checkpoint extraction or selection
was performed. All successful endpoints are (1,q-1), loss zero; the other
five endpoints remain (0,0), loss 0.4, without even transient improvement.

| Optimizer | Seed | Batch | p=3 first zero update | p=71 first improvement / zero | p=71 endpoint loss |
|---|---:|---:|---:|---:|---:|
| Momentum | 42 | 1 | Not reached | Not reached | 0.4 |
| Momentum | 42 | 2 | Not reached | Not reached | 0.4 |
| Momentum | 1729 | 1 | 321 | 305 / 330 | 0 |
| Momentum | 1729 | 2 | Not reached | Not reached | 0.4 |
| Adam | 42 | 1 | Not reached | 402 / 419 | 0 |
| Adam | 42 | 2 | Not reached | Not reached | 0.4 |
| Adam | 1729 | 1 | 40 | 54 / 71 | 0 |
| Adam | 1729 | 2 | Not reached | Not reached | 0.4 |

A read-only comparison verified the optimizer source hash is identical and
all **9,500 update minibatches match exactly** between profiles. Differences
in radii and group/branch choices can still alter later coordinate histories.
The prime and precision changed together, so this comparison does not isolate
their individual effects. Two fixed seeds do not estimate escape probabilities
or comparative optimizer performance. The extra Adam escape does not show
that the larger prime improves optimization in general.

## Slopes, history, vertices, ties and clipping

The three full-batch controls stayed stationary for all 100 updates. As at
p=3, their initial raw up/down slopes are +0.2/+0.1. The lower-radius cap
disables downward motion and substitutes zero in its history, leaving no
negative available direction. Three joint rows tie the two radii and couple
them into one component, with one coordinate selected per update. This
stationarity is expected under the declared adapter conventions.

All p=71 traces record raw and boundary-adjusted slopes, moments, speeds,
coordinate clocks, vertex depth, root/child choices, ties/groups, requested
and common applied times, limiting boundaries and actual state changes.
The persisted-state audit passed **9,500 updates and 1,156 motion proposals**.
All 1,156 proposals strictly changed a radius; none was a numerical no-op.
The smallest positive applied time was about 1.60e-16 in controlled GD and
still produced a strict radius change. This tested case does not exclude
floating-point problems when thousands of coordinates share a clipping time.

Observed nonescape mechanisms:

* Both seed-1729/batch-2 trials again had 198 negative feasible minibatch-slope
  selections suppressed to zero speed by their accumulated histories. All
  1,099 coordinate selections had zero speed. They made no motion proposal,
  so neither clipping nor floating-point no-ops caused their nonmovement.
  The two-coordinate tie appeared in 901 of 1,000 updates.
* Seed-42/batch-2 Momentum and Adam each made 27 proposals, with 18 clipped
  updates, and reached at most log radius -3. Both had 224 negative feasible
  slope selections suppressed by history and never reached the root. At p=3
  the corresponding maximum logs were -2 and -0.9529. Different numerical
  log depths across primes should not be interpreted as equal physical radii.
* Seed-42/batch-1 Momentum reached the root in this profile, with 10 root
  selections and two root-child proposals. Both retained the old digit;
  there were no coefficient changes. One current negative child slope was
  blocked by the vertex-history guard. In total, 324 negative feasible slope
  selections were suppressed and 140 proposals followed stored momentum
  against the current minibatch slope. Its p=3 counterpart never reached
  the root. Root access alone therefore does not ensure a useful branch.
* Seed-42/batch-1 Adam changed both root digits and ultimately reached zero;
  its p=3 counterpart reached the root but kept its digits. The other two
  stochastic escapes also changed both root digits and completed refinement.
  These are observations of the fixed histories, not isolated causal effects
  of prime, precision, grouping or capacity.

No bug was found in this bounded control's certificates, mathematical tests,
positive paths or stored-motion audit. That conclusion is limited to the
tested cases. The original product-data negative result remains unresolved.

## One proposed product diagnostic — not launched

Replay **only the first 100 updates of the original warm Adam member**, with
a 20-second fitting cap and 30-second overall cap, from the frozen snapshot
and exact-fitted coefficients. Keep p=71, precision 7, seed 42, batch 32,
radius 71^-7, feature mask, all optimizer settings and tie grouping unchanged.
Use training rows only; read the stored snapshot rather than creating a new
one. Write isolated traces without changing the original model or result rows.

Instrument the same raw/used slopes, history suppression, selected children,
vertex depths and ties as this control. In particular, summarize which
coordinate sets the common clipping time, the ratio of applied/requested
time, actual radius changes/no-ops, and the shallowest vertex reached by each
supported coordinate. Compare the replay's checkpoint losses and maximum
radii with the original first/100th checkpoints to check reproducibility.

This single replay most directly distinguishes suppressed feasible gradients,
shared-clipping throttling, and failure to reach useful coefficient-changing
vertices. A dominant cause cannot be asserted from the old proposal counter.
The proposed replay is a diagnostic, not an ablation or an improvement claim;
it may leave the 1,000-update outcome unexplained. No product run was started
under this authorization.

## Reproduction, preserved evidence and limitations

Execution host raksasa; source commit
`7dcea60b7d0fcc9074a0149c58dbada96b94facf`. The complete fixed roster took
**5.57 seconds**, with all trials finishing their update budgets (slowest fit
0.639 seconds), inside five seconds per fit, 60 seconds overall and the
75-second outer timeout. One nice-15 process and one numerical thread were
used. Preflight found about 39 GiB available, existing unrelated compute and
no backup process. No product data, other jobs or original results were modified.

* [Frozen protocol](salazar-synthetic-p71-protocol.md)
* [Analytic certificate and all trial results](salazar-synthetic-p71-results.json)
* [Persisted-state audit](salazar-synthetic-p71-audit.json)
* [Matched-profile comparison](salazar-synthetic-p71-comparison.json)
* [Full tests](salazar-synthetic-p71-tests.txt): **313 passed, 6 skipped**, with
  12 existing sklearn batch-size warnings. Regressions retain all three
  curated p=71 escapes and all three observed stochastic p=71 escapes.
* Full JSONL traces remain under `artifacts/salazar-synthetic-p71-20261002`
  in the isolated raksasa checkout, with per-file hashes in the results export.
  A local evidence copy preserves both profiles. Older result/audit files
  and report versions remain preserved.

The optimizer hash remains
`9601f31fe1fc492b5d28e211c2c54bed0a54c127a9755c26915cdc039c4019ff`.
Only the control runner, certificate, read-only audit/comparison and tests
were extended. The method still restricts absolute precision, canonical
finite residues and radii to [p^-E,1], substitutes terminal downward history,
and conservatively groups attaining ties without the paper's cancellation
tests. Positive paths do not establish equivalence to the paper's significant-
digit, signed Z[1/p] method. The released official library still supplies no
optimizer used here. Prior verified source review remains in the report.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 75s nice -n 15 uv run --no-sync -m padjective.paper_salazar_synthetic --profile p71
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 30s nice -n 15 uv run --no-sync -m padjective.paper_salazar_synthetic_audit --profile p71
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 15s nice -n 15 uv run --no-sync -m padjective.paper_salazar_synthetic_compare
```

The runner refuses to overwrite existing traces. Any authorized reproduction
must preserve the prior artifact directory first. No broader fitting, ensemble,
push, publication, installation or manuscript change was performed. No runtime,
capacity or authorization blocker remains for this completed control.
