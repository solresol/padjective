# Fixed coarse-lift comparison, 2 October 2026

Starting the same fitted member at radius **71^-1** changed the trajectory
and allowed **15 coefficient-digit changes**, but produced **no clear overall
held-out quality improvement**. Adjusted held-out loss decreased by only
3.11315e-5, while exact accuracy lost five of 1,300 predictions and mean
shared-prefix depth fell. The run hit its 120-second fitting ceiling after
270 updates; the preserved fine-lift run completed 1,000 updates. No winner
was selected and no model was replaced.

This single fixed test supports initial radius as a constraint on the observed
trajectory. It does not establish clipping alone as the cause of the original
negative result or demonstrate a generally superior optimizer. The same
held-out fold had already been inspected when forming this hypothesis, so
the reported held-out comparison is **exploratory, not confirmatory**.

## Baseline and both warm initializations

All use the same 5,393 training rows and 1,300 held-out rows, 2,542 supported
binary features, feature order/mask, p=71 and absolute precision 7. Both warm
arms reuse the exact-fitted seed-42 centers, with seed 42, batch 32, Adam,
kappa 1, beta .9, beta2 .999, epsilon 1e-8 and zero histories. The single
prescribed change is initial log radius -1 instead of -7. No exact refit,
extra seed, radius grid, ensemble or adaptive held-out tuning was performed.

| Existing extracted metric | Exact coordinate baseline | Fine warm Adam, 71^-7 | Coarse warm Adam, 71^-1 |
|---|---:|---:|---:|
| Held-out adjusted taxonomy loss | 0.3018845384 | 0.3018845384 | 0.3018534069 |
| Held-out adjusted exact accuracy | 45.6923% | 45.6923% | 45.3077% |
| Held-out adjusted root accuracy | 69.9231% | 69.9231% | 69.9231% |
| Held-out adjusted mean prefix depth | 2.1184615385 | 2.1184615385 | 2.1138461538 |
| Held-out raw taxonomy loss | 0.3561931858 | 0.3561931858 | 0.3561620543 |
| Held-out raw exact accuracy | 44.8462% | 44.8462% | 44.4615% |
| Held-out raw mean prefix depth | 2.0061538462 | 2.0061538462 | 2.0015384615 |
| Training adjusted taxonomy loss | 0.2137366292 | 0.2137366292 | 0.2139910937 |
| Training raw taxonomy loss | 0.2611168901 | 0.2611168901 | 0.2611859296 |
| Training adjusted exact accuracy | 57.6859% | 57.6859% | 56.7402% |
| Stored nonzero coefficients | 748 | 748 | 751 |
| Held-out mean active support | 1.4123076923 | 1.4123076923 | 1.4192307692 |
| Held-out support median / p95 / max | 1 / 4 / 10 | 1 / 4 / 10 | 1 / 4 / 10 |
| Held-out fallback fraction | 0.1238461538 | 0.1238461538 | 0.1230769231 |

Exact adjusted held-out predictions dropped from 594 to 589; mean prefix
depth lost six total shared digits across the 1,300 rows. Both raw and adjusted
loss deltas are -0.0000311315039, while their exact/prefix changes are worse.
The coarse arm also worsened both training point losses. These are mixed
objective outcomes, not evidence for a robust gain. Existing center extraction
and the train-fitted zero-score default are used throughout; relaxed training
loss is not reported as prediction performance. No valid-path projection was
added. Original scores remain preserved in their original database row.

## Endpoint and budget discipline

Each warm arm had the original ceiling of 1,000 updates/120 fitting seconds.
The new endpoint was frozen and persisted at the first budget limit, before
any new held-out prediction or access to original held-out scores for the
comparison. Its time-limited endpoint was retained even though an earlier
training checkpoint could have looked better; there was no best-checkpoint
extraction. It is **not converged** and its model metadata marks the update
schedule incomplete. Completed run status means execution/scoring completed.

| Work / time | Original fine warm Adam | Fixed coarse warm Adam |
|---|---:|---:|
| Actual updates | 1,000 | 270 |
| Optimization status | update_limit | time_limit |
| Fitting seconds | 73.4128 | 120.3818 |
| With original exact-fit cost included | 77.4951 | 124.4640 |
| Coordinate selections | 2,518,908 | 673,750 |
| Sampled training examples | 32,000 | 8,640 |
| Training-example equivalents | 5.9336 | 1.6021 |
| Median raw held-out prediction seconds | 0.0002856073 | 0.0002862466 |

The optimizer checks time between updates; one completed update crossed the
120-second limit by 0.382 seconds. Total coarse process work was 121.6335
seconds inside the 180-second outer timeout. One nice-15 process and one
numerical thread were used on raksasa; initial load was about 1.23 with 39 GiB
available and no backup contention observed.

**Matched ceilings are not matched work or instrumentation overhead.** The
coarse diagnostic callback consumed 18.2307 seconds (15.1% of fitting time).
Diagnostic construction and full-training scores inside the optimizer were
not separately timed, so this is not total instrumentation overhead. The
original fine run did not have these per-update diagnostics. Coarser vertex
states can also require larger child-vector operations. This test cannot
attribute the throughput difference to any one factor or claim an efficiency
advantage. No deadline extension or repeat was used to equalize update counts.

## Clipping, coarsening and digit diagnostics

A read-only comparison verified **identical minibatches for the first 100
updates** of the earlier fine replay and this coarse arm. The optimizer hash
is unchanged. Comparing that equal-length prefix avoids using different trace
windows to describe the initial-scale effect:

| First 100 updates | Fine lift | Coarse lift |
|---|---:|---:|
| Median applied/requested common-time ratio | 4.1153e-15 | 3.3548e-6 |
| Maximum log radius reached | -5.8341095998 | -0.7229158673 |
| Motion proposals | 20,920 | 23,766 |
| Coefficient changes | 0 | 9 |
| Numerical no-ops | 0 | 0 |
| Clipped updates | 100 | 100 |
| Endpoint training point loss | 0.2611168901 | 0.2611726886 |

The median common step ratio was about 8.2e8 times larger in the coarse prefix,
but every update still clipped. Initial relaxation and subsequent tie groups
also changed, so this is not a clipping-only ablation. The coarse start
preserves each fitted first digit in its initial disk. No coefficient ever
reached the root vertex and no retained first digit changed during this run.

Over all 270 updates, the audit verified:

* 100,771 proposals, 100,602 strict radius changes, and **169 numerical
  no-ops** (about 0.17% of proposals). These occurred after the first 100
  updates. Their existence is a measured float64 artifact; no counterfactual
  quantifies its effect on the endpoint.
* **15 distinct coefficients changed**, first at update 4: 14 changes of
  digit index 1 and one of index 2 (indices begin with 0). All 15 remained
  changed at the endpoint. First-digit changes were zero.
* 302 child proposals, 270 clipped updates, 284 boundary hits, maximum log
  radius -0.7229158673 and zero coordinates reaching log radius 0.
* Applied/requested time ratio over the entire 270-update trace: median
  8.2760e-7, p95 9.0418e-5, maximum 0.0198081. Total common applied time
  was 0.0590732, which is a tree step-time parameter rather than wall time.
* 646,400 of 673,750 selections lacked current-minibatch feature support;
  91,248 motions involved such features. Chosen current slopes were zero
  for 94,926 motions, positive for 302 and negative for 5,543. These
  categories distinguish carried history from strictly uphill movement.
* 5,865 negative feasible-slope selections, only 20 suppressed to zero
  speed by history; no negative-child vertex-history guards. Ties occurred
  in all updates, with 2,372 attaining-tie rows.

The larger start enabled coefficient exploration without reaching root-level
branches. It neither eliminated clipping nor produced a clear quality win.
The tiny observed held-out loss decrease should not be used to declare this
coarse initialization a winner over the preserved model. This conclusion is
bounded to the fixed member, budget and exploratory fold; convergence,
generalization and a dominant cause of the fine-lift result remain unproven.

## Audit, provenance and preserved artifacts

New PG run: `be559584-51c3-45b2-a5a4-a05f10164683` in the existing
`padjective.paper_salazar_single_runs` table, verified `pg_default`. Original
run `b5bff126-fb4a-492d-969b-b960ccc2220f` stayed unchanged. Snapshot digest,
feature order, support mask and initial coefficient hash matched the original.
The fixed endpoint coefficient hash is
`7fb656527f3de76012c4238a4ffb45c954e73a5642981246a7fcec343232702d`.
Execution commit `1b4684391545878eefc5b9eebd2de0ea4360a607`; adapter hash
`9601f31fe1fc492b5d28e211c2c54bed0a54c127a9755c26915cdc039c4019ff`.

The independent read-only audit passed **all 270 motion/state records**, 14
independently scored training coefficient states, 490 explicit child-residue
direction vectors at the seven frozen detailed checkpoints, and all four
endpoint raw/default training/held-out metric sets. It reconstructed 6,693
endpoint predictions from Python integer sums and verified support and hashes
against frozen PG evidence. Audit time was 4.01 seconds. Tests: **313 passed,
6 skipped**, with 12 existing sklearn batch-size warnings.

* [Frozen protocol](salazar-coarse-protocol.md)
* [Both warm endpoints and preserved baseline metrics](salazar-coarse-results.json)
* [Independent endpoint and motion audit](salazar-coarse-audit.json)
* [Matched first-100-update traces](salazar-coarse-prefix.json)
* [Full tests](salazar-coarse-tests.txt)
* Private host artifacts: `artifacts/salazar-coarse-adam-20261002`; compact
  trace 61,899,318 bytes and detailed trace 24,102,546 bytes, with hashes in
  the results export. Coefficients and product-level predictions stay in PG
  and private host artifacts, outside Git and Library.

Paper/adapter differences are unchanged: canonical finite residues, fixed
absolute precision, radii in [71^-7,1], terminal-history substitution, float64
edge radii and conservative attaining-tie groups without cancellation splits.
This is not an official-library replication or a proof of equivalence to the
paper's signed significant-digit tree method. The measured numerical no-ops
reinforce the need to keep that distinction explicit.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 180s nice -n 15 uv run --no-sync -m padjective.paper_salazar_coarse
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 60s nice -n 15 uv run --no-sync -m padjective.paper_salazar_coarse_audit
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  timeout 30s nice -n 15 uv run --no-sync -m padjective.paper_salazar_coarse_prefix
```

The runner refuses existing completed/partial artifacts. Exactly one coarse
fitting arm was run; resumed messages triggered artifact checks, not repeated
training. All prior results, report history and manuscript work are preserved.
The comparison is complete and stopped. No additional experiment, ensemble,
search, push, publication or deployment was performed. There is no pending
compute or remaining blocker for this delivered result.
