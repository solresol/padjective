# Published-comparator run record, 8 September 2026

## Outcome

The new implementations and all 65 predeclared comparison runs are complete.
All 45 Zubarev fits completed their sampling schedules. The 20 Mihara cases
gave ten rank obstructions, five inclusion obstructions, and five unfinished
fits at the external time limit. No predictive loss is assigned to an
unfinished Mihara fit.

Overall assessment: **share with caveats**. The computations are verified;
the observations concern the specified finite representations and budgets,
not the authors' methods under every possible encoding or sampling regime.
The manuscript has deliberately not been rewritten in this pass.

## Zubarev: independent starts and the published representation

These fits use the digit-interleaving map, a contiguous Mahler basis and the
finite-precision version of the transition law in
[Zubarev, arXiv:2503.23488v2](https://arxiv.org/html/2503.23488v2).
They do not reuse greedy coefficients or replace the model with a linear tag sum.
The normalised transition law is independent of the preceding coefficient
vector, so a greedy seed is not required. The implementation and exact
root-conditioned rejection sampler are explained in the
[source-fidelity and protocol record](published-methods-comparison.md).

Results use the original 6,693 products, 2,542 tag coordinates and five stored
folds, with three seed bases per degree. Each fit visits 320 accepted states
on the fixed beta schedule, then returns the least-training-loss visited state.
The primary degree was fixed as the largest piloted degree before examining
new held-out results; the other degrees are sensitivity runs.

| Degree | Tag positions visible at the root | Mean held-out loss | Fold SD after averaging seeds | Mean elapsed seconds per fit |
|---:|---:|---:|---:|---:|
| 70 | 1 | 0.561393 | 0.011385 | 0.69 |
| 5,040 | 2 | 0.561903 | 0.010770 | 1.74 |
| 357,910 (primary) | 3 | 0.564095 | 0.010846 | 99.77 |

Means give equal weight to the five folds, after averaging the three seeds
within a fold. Pooled product-weighted means and individual seed means are
also retained in the machine-readable results. The seeds are repeated
optimiser runs, not 15 independent samples from the population. Timings are
observed worker wall times on a shared host, not portable complexity estimates.

For context, the existing fixed-paper greedy reference is 0.263237 and the
constant baseline is 0.560442. These reference values were not rerun or
overwritten here. All new polynomial fits have zero exact-path accuracy;
mean first-digit accuracy is approximately 44%. Predictions are raw canonical
residues, without an added nearest-taxonomy decoder or a fitted default.
Completed sampling schedules do not establish convergence to a global optimum.

### The main limitation is identifiable, not just a conjecture about runtime

For K < p^r, Lucas' congruence means that the first output digit sees only
the first r digits of the interleaved input. With binary tags these are the
first r tag coordinates. At K=357,910, the full seven-digit evaluation can
depend on nine tag positions, but the root still depends on only three.
Input order was training-frequency order with anonymised tag-ID tie-breaking.

After the runs, a separate diagnostic allowed an oracle to choose the best
root prediction independently within each identical held-out prefix group.
Even this oracle has mean root error **0.557751**, at all three tested degrees.
Every root error contributes one unit of p-adic loss, so this is a lower bound
on the held-out loss of every polynomial in the tested representation,
irrespective of its optimiser or coefficients. The primary observed loss is
only 0.006345 above this bound. More coefficient search within this
representation cannot approach the existing greedy result.

This bound deliberately uses held-out labels: it is a post-hoc capacity
certificate, not a predictor, fitted baseline, validation-selection criterion,
or source of training information. It did not change the run protocol.
It does not apply to another input order, encoding, or a larger/noncontiguous
basis. In particular, it does not contradict Zubarev's approximation theorem.
With the present direct binary encoding, root dependence on the last of the
2,542 tag coordinates would require contiguous degree at least 71^2541.

## Mihara: published recovery versus input obstructions

The implementation follows the staged affine inclusion, copy/update,
restart and digit-recursion procedures in
[Mihara, arXiv:2604.13137v2](https://arxiv.org/html/2604.13137v2).
Independent synthetic tests recover known coefficients with digitwise noise,
including at the paper's actual p=71 and seven-digit precision.

Important correction to the earlier manuscript explanation: Algorithm 2
explicitly uses `(c-L)/(N-L) > (9/10)*p^(-(D+1-L))`. At full affine rank this
is a greater-than-90% test after subtracting the fitting points. The old
adaptation's deviations were its full-system consensus sampling, missing
intermediate inclusion tests and restart procedure, and forced continuation;
the numerical 90% factor was not absent from Mihara's paper.

| Input experiment | Five-fold result |
|---|---|
| All 2,542 raw tags plus intercept | Exact rank 1,695–1,711 rather than 2,543; required full-rank return is impossible. |
| First 128 training-frequency tags | Exact affine rank 115–124 rather than 129; also rank obstructed. |
| First 32 training-frequency tags | Full affine rank 33, but even an arbitrary predictor of these inputs can agree with only 53.46–53.99% of training root labels. The full-rank inclusion test is impossible. |
| Explicit full-input independent-column preprocessing | Retains 1,694–1,710 tags plus the intercept. All five runs reach the 180-second limit with no completed digit. |

The preprocessing experiment preserves the modulo-71 training column span.
It is identified separately; it does not claim equivalence on unseen products
or higher coefficient digits. Each run retained the published procedure with
rep=3 and an external limit of one million random-row draws or 180 seconds.
All five stopped on time before reaching an inclusion test. That is an
unfinished search, not proof that no satisfactory affine model exists.
The product inputs are sparse, correlated binary vectors; these results are
not a test of the paper's guarantees for its random affine-graph sampling regime.
No forced continuation was used to manufacture a predictive competitor.

## Validation and reproducibility

- Full local suite: **186 passed, 6 skipped**; the skips are existing tests.
  Thirty comparator/audit tests also passed on raksasa before the final
  additional oracle-bound unit test was added locally.
- Exact modular elimination and affine membership were checked against
  enumerated equations. Noisy synthetic recovery checks both small primes and
  the full paper precision. Mahler arithmetic was checked against integer
  binomial coefficients, including overflow-sensitive cases.
- Both rejection samplers were checked against enumerated finite-state
  distributions, including higher digits and nontrivial coefficient nullspaces.
- All **60,237 held-out** and **240,948 training** polynomial predictions were
  reconstructed with Python-integer coefficient sums, independently of the
  fitting code's optimised dot product. All saved predictions matched exactly.
  Independently accumulated integer-numerator losses matched within 2e-12.
- The audit requires the complete 65-run configuration grid. It verifies
  stored folds, row counts, snapshot digest, seeds, budgets, completed sampling
  stages, and absence of invented metrics for interrupted Mihara runs.
- Run and model evidence is stored in
  `padjective.paper_published_method_runs` in Shopify Postgres on raksasa.
  The table and primary-key index were verified in `pg_default`. Existing
  snapshot, model and old experimental tables are not modified by these runners.
- Run source: `7757ee2bf92fc4879933a786b7932ebe108e734d`;
  prediction-audit source: `71787a4`; capacity-audit source: `6bc4c99`.
  Branch: `codex/published-padic-methods`.
- Snapshot: `244ddbe3-0a2c-4c04-9436-0a0253108a09`;
  input digest: `039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
- Batch: `a36900e0-b0d3-4c46-bd0a-e3d33d5322bf`.
  Runs occupied 23:53:05 on 7 September through 00:06:50 on 8 September 2026,
  Australia/Sydney, with three single-threaded workers. Full private evidence
  and logs remain in the isolated raksasa experiment checkout.

The [validated aggregate/run export](published-methods-results.json) contains
every run UUID, settings needed to identify its configuration, fold metrics,
resource diagnostics and model/prediction fingerprints, without product-level
predictions or coefficient arrays. SHA-256:
`ad157e8a15f361068763fcc879dec69bddaa398f3a51571b6ac1d1ae9f21cd2c`.
The [separate capacity-bound export](published-methods-capacity-bounds.json)
has SHA-256
`ad907ae1b50c84528f019040e52cbcde2591b6ac479f11ca7db496c72adf799a`.

Re-run the predeclared batch with `python -m padjective.paper_published_batch`.
Audit its stored source revision with `python -m padjective.paper_validate_published
--source-commit COMMIT --output RESULTS.json`; use `--capacity-only` for the
separate post-hoc certificate. A repeated batch must use a separately recorded
run selection/revision rather than being silently merged into this 65-run grid.

No manuscript, cross-model plot, neural-width choice, matched-budget appendix,
PDF, reMarkable document or frozen Hugging Face release was changed in this
pass. The old comparison sections need revision in light of these results;
the present manuscript should not be represented as updated or submission-ready.
