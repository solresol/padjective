# Published final greedy fit: signed valuations and occupied powers

**The published strict-refinement table reproduces, and it tests distinct
valuations.** Repeating only the five published greedy fits recovered all
reported fold counts, three-decimal loss means and displayed Mann–Whitney
p-values. The same predictions give 102 valuation-separated cases and only
15 cases with pairwise-disjoint infinite canonical digit supports.

The exact-hit rates are essentially the same for disjoint supports (4/15,
26.67%) and valuation separation with overlap (23/87, 26.44%). Their wrong-only
losses are 0.09228 and 0.11078. The smaller group is too small and concentrated
to establish an incremental accuracy advantage. This is an exploratory
association with recorded labels, not a causal claim or independent product
truth adjudication.

This is a **separate regenerated published-greedy analysis**, not the earlier
[September checkpoint-fit comparison](https://github.com/solresol/padjective/blob/6a97469bc0af84c67e0ecedc8c154904f83cce88/docs/refinement-comparison/README.md).
The latter used a different exact finite-quotient fitter and must not be
substituted for this paper-fit result. Both remain preserved. No manuscript,
old model/result row, migration control or database table was changed here.

## Matched predictions and the two predicates

The same frozen 6,693 products, 2,542 anonymised features, 363 reference labels
and five archived folds are used for both predicates. Frozen features already
apply deduplication and nested-tag filtering before anonymisation. We use the
published runtime's exact record construction and feature ordering, never
mutable raw tags or substring filtering of anonymous IDs.

The published predictor adds its **actual signed integer coefficients**. Its
training-only fitted default is substituted exactly when that integer sum is
zero. Neither the predictions nor the loss are reduced modulo `71**7`.
Each active nonzero `w_j*x_j` is counted once per binary feature.

- Valuation separation means their `v_71` values are pairwise distinct.
- Infinite-support disjointness means their full canonical `Z_71` digit
  supports are pairwise disjoint, including infinite tails of negative terms.

Primary comparisons exclude zero/one-contribution cases. Disjoint supports
imply distinct valuations; the converse fails. The joint partition is
15 disjoint, 87 valuation-separated with overlap, and 1,139 repeated-valuation
cases, all on the same 1,241 multi-active predictions.

| Condition, multi-active only | n | Exact hits | Exact-hit rate | Mean loss | Wrong-only mean loss | Taxonomies |
| --- | --- | --- | --- | --- | --- | --- |
| Distinct valuations (includes disjoint) | 102 | 27 | 26.47% | 0.07945869 | 0.1080638 | 65 |
| Repeated valuations | 1139 | 127 | 11.15% | 0.8396339 | 0.945003 | 207 |
| Disjoint infinite digit supports | 15 | 4 | 26.67% | 0.0676725 | 0.09228069 | 8 |
| Distinct valuations, overlapping supports | 87 | 23 | 26.44% | 0.08149079 | 0.1107765 | 59 |

All disjoint cases have two active contributions. The valuation-separated
group has 101 two-contribution cases and one three-contribution case (the
latter is incorrect with loss 1). Restricting to two contributions gives
27/101 exact for valuation separation; the 773 repeated-valuation cases have
3 exact hits and mean loss 0.9781905. This controls active count only; it does
not control taxonomy difficulty, merchant dependence or feature quality.

## Reproducing the paper table

| Fold | Strict n | Strict mean loss | Non-strict n | Non-strict mean loss | Two-sided U-test p |
| --- | --- | --- | --- | --- | --- |
| 0 | 32 | 0.09513922 | 220 | 0.850193 | 8.780984e-17 |
| 1 | 18 | 0.1119378 | 234 | 0.8806471 | 1.954761e-15 |
| 2 | 26 | 0.001122331 | 209 | 0.8138708 | 1.366158e-15 |
| 3 | 12 | 0.08343276 | 234 | 0.8380333 | 1.960846e-09 |
| 4 | 14 | 0.143934 | 242 | 0.814175 | 2.903736e-08 |
| All | 102 | 0.07945869 | 1139 | 0.8396339 | 4.477616e-61 |

Every displayed number in the original strict-refinement table matches after
its stated rounding. The pooled U-test is 4.47761566286e-61,
which rounds to `4.5e-61`; the pooled exact-hit chi-squared p-value is
1.42843911311e-05, matching the SIGIR
version's `1.4e-5`. These are post-hoc, product-level descriptive tests, not
independent fold replication or a causal argument.

The paper's overall equal-fold loss is reproduced as
`0.26323667562627723` versus archived
`0.26323667562627684` (agreement within `1e-12`). The all-product pooled loss
below is slightly different because it weights products rather than folds.
The archived fold metrics are checked individually, not only by their mean.

Original paper weight checkpoints were not located. This result regenerates
the checksum-pinned executable protocol and matches its archived scalar
results and strict-refinement table; it is not a bitwise comparison against
unlocated original coefficient/prediction vectors. The five newly captured
vectors and predictions are now preserved and replayable.

## Zero/one-active cases and cancellation

| Active contributions | n | Exact hits | Exact-hit rate | Mean loss | Wrong-only mean loss |
| --- | --- | --- | --- | --- | --- |
| Zero | 1016 | 58 | 5.709% | 0.5018456 | 0.5322287 |
| One | 4436 | 3123 | 70.4% | 0.06473882 | 0.2187216 |
| Two or more | 1241 | 154 | 12.41% | 0.7771537 | 0.8872565 |
| All | 6693 | 3335 | 49.83% | 0.2631861 | 0.5245695 |

Both predicates are vacuous for zero or one contribution; these cases remain
separate. Among 1,060 default uses, 1,016 have no active coefficient and 44
are exact cancellation of nonzero signed contributions. None of the latter
is valuation-separated. The fallback is not counted as a tag contribution.
Conditional loss is computed only among incorrect predictions; an empty
conditional group is `null`, shown as a dash, rather than zero.

## Infinite negative expansions and finite-depth sensitivity

Negative integers have an eventual all-70 tail in base 71. The exact test
does not enumerate that infinite tail:

1. Two negative contributions necessarily overlap eventually, so fail.
2. Positive contributions have finite supports, checked against one another.
3. With one negative contribution, inspect its canonical digits only at the
   finitely many occupied positive positions. Python floor division/modulo
   gives those digits without replacing the number by its absolute value.

For example, in base 2, actual signed contributions `-3,-2` have distinct
valuations but overlapping infinite tails. Their residues `1,2 mod 4` have
disjoint two-digit supports. Thus disjoint infinite support is not invariant
under choosing different lifts of the same finite residues.

Sensitivity truncates canonical digits of each actual signed contribution
modulo `71**D` while holding predictions, loss and the full valuation predicate
fixed. Entire contributions can vanish at shallow depths:

| Depth | Disjoint n | Hidden-contribution rows | Finite-only false positives versus infinite support |
| --- | --- | --- | --- |
| 1 | 106 | 188 | 91 |
| 2 | 100 | 171 | 85 |
| 3 | 56 | 62 | 41 |
| 4 | 29 | 17 | 14 |
| 5 | 15 | 0 | 0 |
| 6 | 15 | 0 | 0 |
| 7 | 15 | 0 | 0 |
| 8 | 15 | 0 | 0 |
| 10 | 15 | 0 | 0 |
| 14 | 15 | 0 | 0 |

For these fitted contributions, depth 5 onward already gives the same
membership as the exact infinite-support test. Depth 8
is a **proved sufficient bound**: one more than the maximum digit length of
any fitted coefficient's absolute value. Above that bound all positives have
ended and all negative tails have begun, so agreement is no longer merely an
empirical cutoff check. Depths 8, 10 and 14 were explicitly checked too.

There are 68, 63, 66, 59 and 72 negative fitted coefficients across folds
0 through 4. Eight of the 15 disjoint predictions contain one negative
contribution. Reinterpreting these same coefficients as canonical nonnegative
residues modulo `71**7` changes zero support classifications on this dataset,
although it can change them in general as the counterexample shows.
The integer raw score differs from its quotient residue for 169 predictions;
no nonzero integer raw score becomes zero modulo `71**7` in these fits.
This arithmetic counterfactual does not replace the signed predictor and
does not identify the cause of differences from the earlier checkpoint fit.

A supplemental replay makes the sign restriction explicit. Among the
disjoint cases, the seven with only positive contributions have four exact
hits, while the eight with a negative contribution have none. This is
structural: under disjoint infinite supports, a negative term's eventual
all-70 tail cannot be changed by finitely supported positive terms. The
integer prediction remains negative and cannot equal a positive taxonomy
code. Overlap is necessary to cancel that tail when a negative correction
participates in an exact positive prediction. This is another reason the
stronger property is not a generic correctness certificate.

The fixed-default quotient counterfactual changes 169 numerical predictions
but changes zero exact-hit indicators, zero losses and zero fallback triggers
on these held-out rows. The maximum nonzero error valuation is 4, below the
quotient depth 7. These checks are preserved separately in
`2026-10-19/signed-details.json`, generated by
`scripts/explain_final_greedy_signs.py`; they do not refit the quotient model
or explain away differences from its earlier saved checkpoint coefficients.

## Fold-specific joint groups

| Fold | Joint group | n | Exact hits | Mean loss | Wrong-only mean loss |
| --- | --- | --- | --- | --- | --- |
| 0 | Disjoint | 2 | 1 | 1.396995e-06 | 2.793991e-06 |
| 0 | Valuation-only overlap | 30 | 7 | 0.1014817 | 0.1323675 |
| 0 | Repeated valuations | 220 | 27 | 0.850193 | 0.9691319 |
| 1 | Disjoint | 3 | 0 | 0.004827085 | 0.004827085 |
| 1 | Valuation-only overlap | 15 | 3 | 0.13336 | 0.1667 |
| 1 | Repeated valuations | 234 | 18 | 0.8806471 | 0.9540343 |
| 2 | Disjoint | 7 | 1 | 8.621457e-05 | 0.0001005837 |
| 2 | Valuation-only overlap | 19 | 7 | 0.001504058 | 0.002381425 |
| 2 | Repeated valuations | 209 | 29 | 0.8138708 | 0.9449944 |
| 3 | Disjoint | 1 | 1 | 0 | — |
| 3 | Valuation-only overlap | 11 | 1 | 0.09101756 | 0.1001193 |
| 3 | Repeated valuations | 234 | 23 | 0.8380333 | 0.9293829 |
| 4 | Disjoint | 2 | 1 | 0.5 | 1 |
| 4 | Valuation-only overlap | 12 | 5 | 0.0845897 | 0.1450109 |
| 4 | Repeated valuations | 242 | 30 | 0.814175 | 0.9293885 |

## Taxonomy composition and uncertainty

All 363 taxonomies have full group counts and metrics in `summary.json`.
The eight represented by disjoint-support cases are:

| Reference taxonomy | Disjoint n | Disjoint exact | Disjoint wrong-only loss | Overlap-only n | Overlap-only exact |
| --- | --- | --- | --- | --- | --- |
| Short Trousers (aa-1-14-4) | 1 | 0 | 2.793991e-06 | 0 | 0 |
| Dresses (aa-1-4) | 3 | 0 | 2.793991e-06 | 3 | 3 |
| Beanies (aa-2-17-2) | 1 | 0 | 0.0001983733 | 0 | 0 |
| Bundles (bu) | 1 | 0 | 0.0001983733 | 0 | 0 |
| Chairs (fr-7) | 1 | 0 | 0.0001983733 | 0 | 0 |
| Eye Creams (hb-3-2-9-19) | 1 | 0 | 1 | 1 | 0 |
| Household Appliances (hg-9) | 2 | 0 | 0.0001983733 | 0 | 0 |
| Cosmetic & Toiletry Bags (lb-3) | 5 | 4 | 0.01408451 | 0 | 0 |

All four disjoint exact hits are Cosmetic & Toiletry Bags. One disjoint
Eye Creams prediction has the maximal loss 1; disjointness does not prevent
a wrong coarse prediction. The remaining errors are closer, but excluding
that case after observing it would be an unjustified change of population.

For rough scale only, binomial Wilson intervals are:

| Group | Illustrative 95% Wilson interval |
| --- | --- |
| multi_disjoint_infinite | 10.9%–52.0% |
| multi_not_valuation_separated | 9.5%–13.1% |
| multi_valuation_only | 18.3%–36.6% |
| multi_valuation_separated | 18.9%–35.8% |

These intervals and the unadjusted disjoint-versus-overlap Fisher p-value
`1.0` assume independent rows.
They do not account for merchant/taxonomy clustering or overlapping training
sets and are not confirmatory inference. In particular, the tiny difference
between 4/15 and 23/87 does not establish superiority or equivalence.

## Provenance, saved models and reproduction

- Snapshot UUID: `244ddbe3-0a2c-4c04-9436-0a0253108a09`.
- Canonical ordered-record digest: `039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
- Published reference source: `6d146b8a2146ab55152d96bc12bac69b0f15c1ea`.
- Frozen runtime SHA-256: `4618605fcdf466a840c19f9de783bd003cf6aad996ea6d4de46dd7070f595aa1`.
- New runner commit: `c488032d5cd295b211c0d442b78f4b287ada3ff6`.
- Runner SHA-256: `441f4c49000ca5c0da18a15c824ce599fa357e9cda3a5ed00b86e7a6ca0d20bc`.
- Summary SHA-256: `c7004b1b980cf16bf3af402be62c33fe45dd7005ba74c63f4d82c3bba91ddf77`.
- Dependency versions and all input/model hashes are in
  [summary.json](2026-10-19/summary.json); replay and paper-table checks are
  in [audit.json](2026-10-19/audit.json).

The five `fold-*.json.gz` files contain regenerated anonymised coefficient
tables, training-only defaults and predictions in frozen held-out row order.
Their checksums bind the readout to runnable saved fits. Inputs are already
published anonymised matrices; no raw product titles, raw tag text or private
catalogue export is included. The new analysis makes **no database connection
or database write**.

The five fits and analysis completed in 13.05
seconds on one CPU, with 198.0 MiB peak
resident memory and a 1 GiB address-space ceiling. The two-minute cap was not
reached; the larger allowance was unnecessary. Replay verifies all new saved
predictions/defaults and every fold/taxonomy/depth aggregate without refitting.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync -m padjective.paper_final_greedy_refinement \
  --release-root build/paper-submission-2026-09-10/reference \
  --output-dir /tmp/final-greedy-new-run

PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  uv run --no-sync python scripts/audit_final_greedy_refinement.py \
  --release-root build/paper-submission-2026-09-10/reference \
  --run-dir docs/refinement-final-greedy/2026-10-19 \
  --output /tmp/final-greedy-new-audit.json
```

Existing output paths are refused. The runner calls only the selected five
`taxonomy_association` greedy fits, not the full multi-model/neural suite.
The same runner also offers `--replay-source` to reconstruct a saved run
without fitting. It enforces a single CPU, a 1 GiB ceiling and a 120-second
wall cap; the validated run needs no relaxation.

Validation: **295 tests passed, 6 skipped**, including 15 refinement tests
covering exact infinite tails, finite-lift counterexamples, exhaustive small
signed tuples, zero handling, wrong-only loss and nonzero errors beyond
`71**7`. The full suite ran on one CPU. Its existing 12 small-sample neural
batch-size warnings remain. The manuscript and published table are unchanged.
