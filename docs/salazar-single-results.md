# Single-member continuation result, 2 October 2026

The bounded warm-continuation experiment found no improvement. Both Momentum
and Adam retained every coefficient of the existing exact randomised member;
their raw and default-adjusted predictions are identical to the baseline.
The uninformed Adam control learned digits but performed substantially worse.
No ensemble or wider search was launched.

| Model | Held-out loss | Exact accuracy | Shared prefix | Nonzero coefficients | Mean active support | Fit seconds | Updates/status |
|---|---:|---:|---:|---:|---:|---:|---|
| Exact coordinate | 0.3018845384 | 45.6923% | 2.11846 | 748 | 1.41231 | 4.08 | Six sweeps; coordinate optimum |
| Warm Momentum | 0.3018845384 | 45.6923% | 2.11846 | 748 | 1.41231 | 53.33 additional | 1,000; update limit |
| Warm Adam | 0.3018845384 | 45.6923% | 2.11846 | 748 | 1.41231 | 73.41 additional | 1,000; update limit |
| Uninformed Adam | 0.5505501368 | 3.4615% | 0.60231 | 652 | 1.72385 | 120.12 | 874; time limit |

These scores include the existing zero-score default, fitted separately on
training rows. Raw scores for the exact/warm models are loss 0.3561931858,
accuracy 44.8462%, shared prefix 2.00615. The uninformed raw scores are loss
0.6063111865, accuracy 2.6154%, shared prefix 0.49462. Prefix depth is capped at
each gold taxonomy's depth. No valid-path decoder was added.

The fitted warm models use the same 748 stored nonzero coefficients, with
prediction-wise active support median 1, 95th percentile 4, maximum 10. Default
substitution occurs on 12.3846% of test predictions. The uninformed model stores
fewer coefficients but consults more nonzero terms: median 1, 95th percentile 5,
maximum 11; fallback rate 17%. Sparse raw prediction over 1,300 rows takes about
0.00028 seconds; this micro-timing excludes the independent audit and default
substitution and supports no broad inference-speed claim.

## Data, environment and budgets

Executed on `raksasa` through existing SSH from `udara.cassia.ifost.org.au`.
Raksasa has 12 logical CPUs, 61 GiB RAM with about 39 GiB available, and 328 GiB
free disk at initial inspection; initial load was 1.39. One unrelated root
training process was active; no backup process was observed. This experiment
used one nice-15 process, numerical thread limits of one, and about 180 MiB RSS.
Load remained about 2.2 while it ran. Existing Python 3.11.11, NumPy 2.2.5,
SciPy 1.15.3, psycopg, pytest and uv were used. No package installation occurred.

Frozen Postgres snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`, verified digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`:
5,393 training rows and 1,300 held-out fold-0 rows, seed 42, p=71, seven digits,
2,542 aligned binary features, all supported in training. The exact baseline
is coefficient-identical to historical member
`dbd36954-ba9a-455b-8b49-ed3cc095c728`. A separate pass certified all 2,542
coordinates, with no improving coordinate.

The [protocol](salazar-single-protocol.md) froze all configurations before new
held-out scoring. Exact fitting had ceilings of 100 sweeps/300 seconds; each
tree fit had ceilings of 1,000 updates/120 seconds and batch 32. Warm runs each
sampled 32,000 training examples; the secondary cold run sampled 27,968. All
endpoints were frozen before test prediction. Checkpoint scoring, certificates
and warm fitting are additional work, so sampled-example counts are not total
comparable arithmetic work. Warm total fitting times were 57.42 and 77.50
seconds including the shared exact fit. Complete successful process time,
including data loading, certificates, checkpoints and scoring, was 252.16
seconds. Common ceilings produced different work, not an equal-operation
efficiency benchmark.

## What this establishes

The adapter can move tree states and learn coefficient digits from an
uninformed state, but neither tested warm continuation changed a digit within
the declared budget. Their final largest parameter radii were approximately
71^-5.85 and 71^-5.84; most remained much smaller. Raw training point loss stayed
0.26111689009390665. Direct tree loss changed only around 1e-13 on the warm
trajectories; that is not a prediction improvement and is too small to support
a meaningful relaxed-loss gain claim.

Tied maxima and shared first-vertex clipping are plausible obstacles, but this
run does not identify the cause conclusively. A larger radius or altered
schedule could change outcomes; neither was tuned or tried. This is one seed,
one fold, one fixed rate, and a bounded independent implementation. It does not
disprove the general method, establish convergence, or warrant an ensemble.

The [paper](https://arxiv.org/html/2609.25501v1) and
[official source](https://github.com/google-deepmind/padic-ml) were inspected,
with library commit `53ffd9e79890e76b82ef2dc3c7230984053f9801` pinned. Its released
source has array operations but no optimizer yet. Our affine derivative tests
enumerate children and check exact residue arithmetic, attaining ties,
representative invariance, momentum history weighting and common clipping.
Conservative groups omit cancellation splitting; seven-digit capacity, radius
bounds and zero histories for unavailable terminal descent differ from an
unbounded tree optimizer. This is not claimed to be an official-library or
unconstrained-method replication.

## Evidence and reproduction

Successful run: `b5bff126-fb4a-492d-969b-b960ccc2220f`, execution code commit
`fe6a089f94296e94cc973a86e975fd8145728c44` on branch
`codex/salazar-single-20261002`, worktree
`/home/gregb/experiments/padjective-salazar-single-20261002`.
The worktree base is `6e0dbffa481437c86ef45a540126a1644e70a41e`; the canonical
raksasa checkout and the local reference checkout were preserved.

Private centers, radii, checkpoints and row-aligned predictions remain in
`shopifystores.padjective.paper_salazar_single_runs`, explicitly using
`pg_default`. Aggregate export: [JSON](salazar-single-results.json).
The [readback audit](salazar-single-audit.json) reconstructed all 26,772 stored
train/test predictions with Python integer sums and recomputed 16 metric sets
using exact numerators. Coefficient and prediction hashes matched. Final full
suite: **290 passed, 6 skipped**, with 12 pre-existing sklearn batch-size
warnings; mathematical/fitter subset: 28 passed.

First attempt `fcded55e-8911-4e05-bb26-36a8fe4492f5` failed before held-out
scoring because a NumPy counter could not be serialized. Its database row and
log are retained as failed. The counter was fixed and the complete experiment
repeated unchanged. Later endpoint bookkeeping was clarified so the secondary
time-limited model is explicitly incomplete; this changed no coefficients,
predictions, scores or fit-status records.

From the isolated raksasa checkout, reproduce with the existing environment:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  timeout 900s nice -n 15 uv run --no-sync -m padjective.paper_salazar_single
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync -m padjective.paper_salazar_audit b5bff126-fb4a-492d-969b-b960ccc2220f
```

Reproduction creates a new private run row. The cold endpoint is time-limited,
so its exact number of updates depends on host performance. Warm fixed-update
trajectories are seeded. No pushes, publication, production model writes or
manuscript changes occurred. There are no remaining access blockers. Stop here
for parent review before broader experiments.
