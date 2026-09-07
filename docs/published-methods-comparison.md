# Published p-adic method comparison (7 September 2026)

## Scope and status

Implementing and validating, before full product-data runs. The author approved
Zubarev's polynomial model (not a requirement to retain the greedy linear
representation), independent starts, and Mihara's published procedure.
Existing September results and the frozen release are retained as historical
experiments, not overwritten or relabelled as these new algorithms.

## Sources and fidelity contract

- Zubarev: arXiv:2503.23488v2, equations (5)--(6), (10)--(12), (16)--(17).
  Source archive SHA-256:
  `c440b8bf60a45c6db3d74964e41aad353e40cae44440f4c0e74c0eef274808da`.
- Mihara: arXiv:2604.13137v2, Algorithms 1--8, including the corrected
  fitting-point subtraction in Algorithm 2 and the copy/update rule in
  Algorithm 5. Source archive SHA-256:
  `0f9f3980155467bc57f5c2dfe9c8490b2655977a76d753075d4137fdf046fb34`.
- Original archives are outside Git in
  `/Users/gregb/Documents/padjective-published-methods-20260907.oaM8zU/`.

Important correction to the previous review: Algorithm 2 explicitly compares
`(c-L)/(N-L)` with `(9/10)*p^(-(D+1-L))`. At full affine rank `L=D+1`, this
is a 90% test. The earlier claim that 90% was absent from the published method
was wrong. The old code's principal differences are its full-system consensus
sampling, missing intermediate inclusion tests, bounded candidate selection,
and forced continuation, not simply the numerical value 90%.

Mihara's row-space membership calculation may replace explicit construction of
the equivalent family of affine equations in Algorithm 2. Unit tests must
compare these formulations. Random draws remain with replacement; rejected
dependent/inconsistent draws and restart semantics follow the pseudocode.
External time/draw limits interrupt execution, returning an unfinished status
and partial diagnostic state, never an invented completed fit. A separately
identified rank certificate may prove a requested full-rank step impossible.
No feature removal is silently used to repair such a case.

Zubarev's model is `sum(k=0..K) w[k] binom(h(x), k)` with the published digit
interleaving `h`. Arithmetic is reduced modulo `p^E`, with enough extra input
digits to evaluate the binomial basis correctly. Polynomial degree, coefficient
precision, input order and computational budget are explicit implementation
choices. Binary product inputs are not silently treated as a weighted tag sum.

The transition density in (17) is proportional to
`exp(-beta*(L(w+xi)-L(w)))` relative to Haar measure. On a finite quotient,
translation invariance cancels the old-state factor: the next state has mass
proportional to `exp(-beta*L(v))`. Exact rejection sampling from uniform
coefficient residues implements this finite-precision distribution; a local
Metropolis step is not substituted and called the same kernel. A certified
loss lower bound may tighten the rejection envelope without changing it.
The chosen optimiser returns the least training-loss *accepted state* visited,
including its independent initial state, and reports transition completion
and proposal counts separately. Rejected proposals cannot be used as results.

A contiguous Mahler truncation has a resolution limit: modulo `p`, degrees
through `K` can distinguish only the first `1+floor(log_p K)` interleaved digits
(for `K>=1`). This must be tested and reported; increasing CPU alone does not
turn a modest degree into a model that sees all tag coordinates at the root.

## Validation gates

1. Exact modular elimination, affine membership, and all loop/stop edge cases.
2. Mihara recovery on independently generated dense affine data with no noise
   and with sparse digitwise noise; reproducible seeds, exact known coefficients.
3. Zubarev interleaving and modular binomial basis against direct integer
   calculations; finite transition frequencies against enumerated tiny spaces.
4. Independent metric calculation and fold-local feature ordering checks.
5. A small instrumented pilot to establish runtime/memory, then record the
   final full-run budget before examining outer-fold predictive results.

## Data and preservation

Read the archived `paper` snapshot directly from Shopify Postgres:
`244ddbe3-0a2c-4c04-9436-0a0253108a09`. Reuse its five stored folds.
Store run configuration, status, per-fold results and provenance in new
`padjective` tables with explicit `pg_default` tablespace. Do not mutate
production model tables, snapshot products or existing experiment rows.
Full-feature and any explicitly named lower-dimensional experiments must be
reported separately. No held-out labels determine input order, model degree,
starting point, stopping rule or selected restart.

The primary cross-model roster and matched-budget appendix stay separate.
Any new cross-model figure requires an explicit decision about which completed
configuration it represents; unsuccessful fits do not get fabricated losses.

## Execution record

- raksasa checked: 12 logical CPUs, about 39 GiB available RAM, 98 GiB disk free;
  one long-running training process occupies one CPU. Initial concurrency will
  leave capacity for existing services and use single-threaded numerical workers.
- Local branch: `codex/published-padic-methods`.
- Full-run settings and result validation: pending the implementation/pilot gates.
- Pure algorithm gates pass: 20 tests cover both published implementations,
  including noisy affine recovery, direct-integer Mahler evaluation, overflow
  checks and a 5,000-draw comparison with an enumerated transition distribution.
- Pilot settings (no outer-fold scoring): fold 0, fixed seven-digit precision;
  Mihara full features and raw frequency prefixes 32/128, `rep=3`, at most
  100,000 random-row draws or 30 seconds after the rank audit; Zubarev degrees
  70/5,040/357,910, independent zero start, beta schedule 0/4/16/64/256, up to
  eight accepted transitions and 20,000 proposals per stage, 30-second sampler
  budget. The larger degrees are complete truncations at successive base-71
  digit boundaries, not sparse replacements for the polynomial model.

### Training-only pilot findings and implementation refinements

The initial six fold-0 pilots are preserved in the new run table, source
`d31a2c2`. All-feature Mihara has rank 1,695 of 2,543; the 128-tag prefix has
rank 115 of 129. The 32-tag prefix has full rank but completed no digit in
100,000 draws (29,139 restarts). Identical-input first-digit upper bounds are
53.4582% at 32 tags and 64.1758% at 128 tags. These bounds certify that neither
prefix can satisfy Algorithm 2's full-rank test, regardless of runtime.

To give Mihara a separate identifiable-input test, explicitly select a training
column basis over F_71, retaining the intercept, and run the unmodified recovery
procedure on those columns. This preserves the modulo-p training column span;
it is not silently substituted for the full-feature run, and does not claim
equivalence on held-out inputs or higher digits. The selected columns are saved.

All three initial Zubarev pilots reached proposal limits. Improve the exact
sampler, not its target law: when the root evaluation matrix is surjective,
Haar measure induces independent uniform root predictions. Sample each group's
prediction with probability proportional to exp(beta * matching_count/N),
sample coefficients uniformly conditional on those predictions (including the
nullspace), and reject with probability 1-exp(-beta*(L-L_root)). This is the
same finite-quotient Gibbs distribution. The remaining loss is in [0,1/p].
Fallback to whole-vector Haar rejection if the root map is not surjective.
Tiny-space frequency tests cover both samplers, higher coefficient digits and
nontrivial nullspaces. No rejected candidate is selected as a fitted result.

Repeat training-only fold-0 pilots with the improved sampler at all three
degrees and with the explicitly rank-reduced full-feature Mihara input
(100,000 draws/30 seconds). Set full-run budgets after these pilots.
