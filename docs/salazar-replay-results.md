# Final bounded Adam diagnostic, 2 October 2026

The original warm Adam trajectory is reproducible. A training-only replay on
raksasa completed all **100 updates in 14.99 fitting seconds**, within the
20-second cap. Update 1 and 100 match the original counters, training point
and direct losses, groups and radius extrema exactly. Coefficients remained
unchanged at every logged update. The original stored result row was read-only
and its configuration, metrics, evidence and status fingerprint stayed intact.

The replay establishes real radial motion with extremely short common steps
and little coarsening of the coefficient disks. It rules out floating-point
no-ops and universal zero feasible slopes as explanations of this prefix.
**Shared clipping from very small disks is a plausible bottleneck, not an
established cause of the full 1,000-update negative result.** No ablation was
run, and no product-data improvement was obtained.

## Matched data, member and checks

Original run: `b5bff126-fb4a-492d-969b-b960ccc2220f`; snapshot
`244ddbe3-0a2c-4c04-9436-0a0253108a09`; verified digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Reuse its saved exact-fitted member rather than refitting it. The same 5,393
training rows, 2,542 supported features, 26,863 nonzeros, feature order and
training mask were verified by fingerprints. The snapshot loader reads the
frozen snapshot for its digest; only training rows enter fitting and scoring.
No held-out prediction was generated in this diagnostic.

Settings remain p=71, precision 7, seed 42, batch 32, Adam, initial radius
71^-7 and zero histories, kappa 1, beta 0.9, beta2 0.999 and epsilon 1e-8.
The optimizer and sampler are unchanged. Source review against the original
run confirms only optional instrumentation/scheduled-batch hooks were added;
the replay uses the original uniform sampler. The current adapter hash remains
`9601f31fe1fc492b5d28e211c2c54bed0a54c127a9755c26915cdc039c4019ff`.

An independent audit reconstructed the exact training point score using
Python integer sums: units **12,807,779,023,827,546**, raw mean loss
**0.26111689009390665**. It derived every supported selected-coordinate
directional-slope vector by explicit child residues, independently of the
adapter's sparse exceptional-digit accumulation: **11,172 vectors matched**.
It also checked all stored common times, boundaries, before/after radii,
proposal speeds and actual state changes. The coefficient fingerprint remains
`2580c3aaaa7d8232a18f271dbd7ad5c9b3f08db79c75964f4fd157b0fd844098`.

## Observed mechanisms

| Quantity over the 100-update prefix | Result |
|---|---:|
| Coordinate selections | 249,935 |
| Motion proposals / strict radius changes | 20,920 / 20,920 |
| Numerical no-ops / coefficient changes | 0 / 0 |
| Clipped updates / boundary hits | 100 / 106 |
| Features ever above the terminal floor | 337 of 2,542 |
| Features reaching radius 71^-6 / 71^-5 | 14 / 0 |
| Shallowest log radius reached at any time | -5.8341095998 |
| Selected vertices | 229,312 at depth 7; 52 at depth 6 |
| Child proposals | 4, all retaining the old coefficient digit |
| Negative feasible slope selections / suppressed to zero speed | 1,979 / 8 |
| Negative child slopes blocked by the vertex-history guard | 1 |

Every update stopped at the first encountered vertex. The applied/requested
common-time ratio had median **4.1153e-15**, maximum **9.5432e-12** and
minimum **1.7914e-19**. Total common applied time was **2.6675e-11**; this
is the tree optimizer's step-time parameter, not wall time. Median applied
motion distance was **1.0566e-16**. Even those tiny moves changed stored radii.
No coordinate reached depth 5, much less the shallower vertices needed to
alter lower-order coefficient digits. At update 100 the largest log radius
was -5.8403283322; the distinction from the shallowest transient radius above
matters. Relaxed loss moved from 0.26111689009390754 at update 1 to
0.2611168900937926 at update 100, about 1.15e-13. Point predictions did not
change, so this relaxation change is not a classification improvement.

Sparse minibatches explain many zero instantaneous derivatives. Of 249,935
selected coordinates, **238,763 were absent from their current minibatch**.
Stored histories can still move them: **18,235 of 20,920 motions** involved
an absent feature. Across all proposals, 18,949 chosen directions had zero
current slope, 149 had a strictly positive current slope, and 1,822 had a
negative current slope. Zero-gradient motion is therefore distinguished
from strictly uphill motion. This behavior is allowed by the implemented
momentum convention; it is not itself an implementation bug. Only 8 negative
feasible selections were stopped by history, so history suppression alone
does not describe the observed radial progress.

Attaining ties occurred in all updates: 822 tied rows and 761 coupled-group
instances. At update 38, a two-coordinate group at log radius -6.3636910738
had a feasible retained-center joint downward slope **-1/64**, while each
individual retained-center downward slope was zero. This is a verified
instance of hidden joint relaxation descent in the real prefix. It does not
change point coefficients and does not establish that changing grouping would
improve product predictions. Only one feasible nonterminal coupled retained-
center downward germ appeared under this audit's criterion; unavailable
terminal downward germs and joint branch changes are outside that count.

## Scientific conclusion and one next experiment

The adapter can escape the certified synthetic coordinate optimum at p=71
and seven digits, yet this exact-fitted product member's observed prefix
spends its radial progress close to the terminal precision floor. Its
unchanged point score is an actual unchanged-coefficient result, not merely
a misleading relaxed score or an extraction error. Tiny shared steps and
sparse-minibatch carried histories are measured mechanisms. Whether a
coarser initial lift would produce useful coefficient branches and held-out
improvement remains an untested hypothesis. The trace does not isolate
clipping, initial scale, conservative grouping, capacity or optimizer history
as a dominant cause of the full-run result.

Recommend exactly one predefined continuation, **not launched**: the same
saved member and Adam settings, but start every supported coefficient at
radius **71^-1** instead of 71^-7. This preserves the fitted first base-71
digit in each initial disk and avoids starting six radius levels below it.
Keep centers, data, mask, seed 42, batch 32, precision 7, kappa and all
optimizer conventions fixed; use the original ceiling of 1,000 updates/120
fitting seconds with diagnostics. No radius grid or best-checkpoint choice.
Freeze that single endpoint before any new held-out scoring, then compare
actual coefficient changes and raw/default taxonomy loss, exact accuracy,
prefix depth, support and runtime against the preserved original baseline.

This tests the initial-scale hypothesis while retaining the optimizer. The
initial relaxed objective and subsequent tie groups will differ, so success
would not establish that clipping alone caused the earlier negative result.
The hypothesis is weaker than a convergence guarantee or a claim of faithful
replication of the paper. The existing finite-residue, radius-bound and
conservative-grouping departures remain fully applicable.

## Preservation and reproduction

Execution commit `c6fc45250f3fc1b94de3f5ce61beda9df5c11752`; fitting elapsed
14.9913 seconds, total process work 15.3496 seconds, snapshot loading 0.3423
seconds. A 30-second external timeout covered startup/loading/fitting. One
nice-15 process and one numerical thread were used. There was one fitting
replay only. The initial read-only audit reached its separate 30-second limit
because its group-selection verifier scanned all coordinate pairs; an
equivalent set-based check completed the audit in 4.23 seconds. A final
read-only pass separating zero/positive slope motion took 4.32 seconds.
Neither audit repeated training. The timed-out log and first successful audit
remain preserved alongside the final one.

* [Frozen diagnostic protocol](salazar-replay-protocol.md)
* [Replay results and checkpoint comparisons](salazar-replay-results.json)
* [Independent trace audit](salazar-replay-audit.json)
* [Tests](salazar-replay-tests.txt): **313 passed, 6 skipped**, 12 existing
  sklearn batch-size warnings. No optimizer implementation was changed.
* Private trace: `artifacts/salazar-warm-adam-replay-20261002/trace.jsonl`
  in the isolated raksasa checkout, 77,631,646 bytes, SHA-256
  `e0e3e15787bf2e76e7bc8f11dbea7e906439eeb52e9caa2366c47f91e875fe8b`.
  Private initial coefficients and full product traces stay on raksasa.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 30s nice -n 15 uv run --no-sync -m padjective.paper_salazar_replay
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 30s nice -n 15 uv run --no-sync -m padjective.paper_salazar_replay_audit
```

The runner refuses to overwrite completed or partial artifacts. Do not
repeat the fitting replay as part of an audit. All original product scores,
synthetic results, manuscript changes and report history are preserved.
No further model fitting, ensemble, search, push or publication was launched.
The final bounded diagnostic has no remaining runtime or authorization blocker.
