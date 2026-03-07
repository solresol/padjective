# Parsimony baseline derivation

## Why parsimony matters

The point of this project is not just to find the lowest-loss model in an
absolute sense. It is to understand how much predictive structure we can get
from relatively small models, especially the p-adic ones.

We therefore care about parsimony because it lets us ask:

- how much loss do we achieve for a given model size?
- when two models have similar loss, which one gets there with fewer
  parameters?
- do the p-adic methods stay competitive even when they are much smaller than
  unconstrained baselines?

Parsimony is the metric that turns that research goal into something we can
track on the site and in the benchmark notebook.

## Goal

We want a parsimony score that still rewards small models, but does not drift too
much as the benchmark dataset gets harder over time.

## Where the original score came from

The original scoring system was derived from a log-log linear regression on the
model-complexity versus loss data. In the original fit, we regressed:

```text
log10(loss) = slope*log10(params) + intercept
```

and obtained a fitted line close to:

```text
log10(loss) = -0.1022*log10(params) - 0.2088
```

with strong fit statistics on that original chart (`R² = 0.9114`, `p = 0.0115`,
`n = 5`).

For presentation and scoring, that was rounded to the cleaner baseline:

```text
log10(loss) = -0.1*log10(params) - 0.2
```

The old score was:

```text
score_old = -log10(loss) - 0.1*log10(params) - 0.2
```

That score penalised model size, but it did not account for the fact that later
snapshots cover more taxonomies and are therefore intrinsically harder.

## Why we changed it

The fitted-line origin of the old score was reasonable, but once we started
looking across historical dataset snapshots we found a practical problem:
parsimony scores were not stable as the dataset got larger and more complex.

In particular, as the benchmark grew to include more taxonomies, the same model
families tended to look a little worse even when their underlying
size/performance tradeoff had not changed in the way we actually cared about.

So the problem was not that the original regression was “wrong”. The problem
was that a fixed size-only baseline did not travel well across snapshots of
different difficulty.

## Calibration target

We calibrated the score against the historical table
`padjective.model_performance_history` on `raksasa`, using the published
snapshots up to **2026-03-06**.

For this calibration we ignored the Dummy baseline and optimised for the learned
models only:

- Decision Tree
- ULR
- UNN
- Importance-Optimised p-adic Linear Regression
- PCLR
- PCNN
- Zubarev

The Dummy baseline is still shown on the site and in the notebook, but it is
treated as a reference point rather than something we try to keep stable.

Our stability objective was:

1. compute the parsimony score for every historical snapshot of each model
2. compute the within-model standard deviation of that score across snapshots
3. average those standard deviations across models

Lower is better.

## What we varied

We kept the size term at `-0.1*log10(params)`, and added a taxonomy-difficulty
adjustment:

```text
score = -log10(loss) - 0.1*log10(params) - 0.2 + gamma*log10(taxonomies / K)
```

Two facts matter here:

- `gamma` controls how strongly we compensate for dataset difficulty
- `K` only shifts the score by a constant, so it changes centering but not
  stability

That second point means `/500` and `/1000` give exactly the same standard
deviations. The denominator is a readability choice.

## Historical sweep

Measured mean within-model standard deviation across the learned models:

| Formula term | Mean std. dev. | Improvement vs old score |
| --- | ---: | ---: |
| old score (`gamma = 0.00`) | 0.032809 | — |
| `+ 0.10*log10(taxonomies / K)` | 0.025777 | 21.4% |
| `+ 0.15*log10(taxonomies / K)` | 0.022795 | 30.5% |
| `+ 0.20*log10(taxonomies / K)` | 0.020454 | 37.7% |
| `+ 0.25*log10(taxonomies / K)` | 0.019052 | 41.9% |
| `+ 0.30*log10(taxonomies / K)` | 0.018765 | 42.8% |
| `+ 0.35*log10(taxonomies / K)` | 0.019510 | 40.5% |

The `0.30` coefficient is the best round-number choice from this sweep. It is
also close to the most stable region rather than being a brittle one-off optimum.

## Chosen score

We therefore use:

```text
score = -log10(loss) - 0.1*log10(params) + 0.3*log10(taxonomies / 1000)
```

Equivalently, the baseline line on the chart is:

```text
log10(loss) = -0.1*log10(params) + 0.3*log10(taxonomies / 1000)
```

This keeps the clean `0.1` and `0.3` coefficients in the final score, while the
old `0.2` term is now treated purely as a centering choice rather than part of
the metric itself.

After choosing the `0.3` taxonomy coefficient, we removed the old `-0.2`
intercept from the final score. That intercept only re-centres the score: it
does not change any historical standard deviations or model ordering within a
snapshot. Dropping it simply shifts scores upward by `0.2`, which makes the
current parsimony tables mostly positive and easier to read.

## Why `/1000`

The denominator does not affect stability, only the numerical centering of the
score. We chose `/1000` because it reads better than `/500` and is easier to
explain.

At the latest historical snapshot on **2026-03-06**:

- products: `8,515`
- tags: `10,942`
- taxonomies: `467`

the taxonomy adjustment is:

- `0.3*log10(467 / 500) = -0.008896`
- `0.3*log10(467 / 1000) = -0.099205`

So `/1000` simply shifts the taxonomy adjustment by about `0.09` relative to
`/500`; it does not make the score more or less stable.

## Stability at the chosen coefficient

Per-model historical standard deviations with the chosen score:

| Model | Std. dev. |
| --- | ---: |
| Importance-Optimised | 0.007095 |
| Zubarev | 0.009308 |
| Decision Tree | 0.014574 |
| PCNN | 0.015305 |
| ULR | 0.023385 |
| PCLR | 0.026442 |
| UNN | 0.035249 |

These values are not identical across models, but they are materially tighter
than under the old score.

## Interpretation

The resulting score has a clear meaning:

- better performance lowers `loss`, which raises the score
- larger models increase `params`, which lowers the score
- harder snapshots increase `taxonomies`, which raise the baseline and stop the
  score from drifting downward just because the task got broader

That is the version now used on the site and in the benchmark notebook.
