# Classical models scored with the same low Q

Five fixed paper folds, 6,693 products. Lower loss is better. Classical training objectives are unchanged. Their predictions were recovered with the archived settings and environment, checking each fold against its original Q=71 loss and exact accuracy.

## Main comparison

Low-Q columns below use the Q-specific retrained native models from the preceding study; classical fits stay unchanged. These are fold means ± sample fold SDs. Exact accuracy refers to the sqrt(2) comparison.

| Model | Q=71 | Q=√2 | Q=2 | Exact accuracy |
|---|---:|---:|---:|---:|
| Neural network (width 2,000) | 0.076 ± 0.003 | 0.201 ± 0.007 | 0.142 ± 0.005 | 66.9% |
| L1 logistic regression | 0.086 ± 0.006 | 0.213 ± 0.008 | 0.154 ± 0.007 | 65.9% |
| Decision tree | 0.110 ± 0.007 | 0.251 ± 0.008 | 0.186 ± 0.007 | 60.9% |
| Level-wise logistic regression | 0.120 ± 0.010 | 0.289 ± 0.006 | 0.210 ± 0.006 | 53.8% |
| 243 members: 75% features | 0.135 ± 0.010 | 0.274 ± 0.010 | 0.207 ± 0.010 | 58.1% |
| 243 members: all features | 0.145 ± 0.012 | 0.274 ± 0.012 | 0.210 ± 0.011 | 58.3% |
| Greedy: taxonomy association | 0.263 ± 0.005 | 0.376 ± 0.016 | 0.326 ± 0.012 | 50.5% |
| Dummy control | 0.560 ± 0.011 | 0.781 ± 0.009 | 0.693 ± 0.010 | 5.3% |

## Findings

The leading three models retain their order in all five comparisons: Neural network (width 2,000), L1 logistic regression, Decision tree.

Across all 16 configurations, level-wise logistic regression moves from rank 4 at Q=71 to 6 under fixed-fit sqrt(2) rescoring, and to 7 when native models are also retrained. At Q=2 its ranks are 4 (fixed) and 5 (retrained).

The paired table below distinguishes mean-order changes from fold-wise consistency. These reused folds support a sensitivity comparison, rather than independent confirmation.

## Complete rankings

Point-estimate ranks are descriptive, without a significance claim. The same folds have already been used for other experiments. Fixed-fit rows isolate rescoring; retrained rows additionally change native fitting and consensus.

### original

| Rank | Model | Mean loss | Fold SD | Exact accuracy |
|---:|---|---:|---:|---:|
| 1 | Neural network (width 2,000) | 0.075559 | 0.003156 | 66.87% |
| 2 | L1 logistic regression | 0.085839 | 0.006492 | 65.87% |
| 3 | Decision tree | 0.110314 | 0.006804 | 60.93% |
| 4 | Level-wise logistic regression | 0.120450 | 0.009740 | 53.81% |
| 5 | 243 members: 75% features | 0.134655 | 0.010275 | 57.39% |
| 6 | 243 members: all features | 0.144596 | 0.012223 | 57.52% |
| 7 | 15 members: 75% features | 0.147371 | 0.008424 | 56.06% |
| 8 | 15 members: all features | 0.159282 | 0.011216 | 55.81% |
| 9 | Greedy: taxonomy association | 0.263237 | 0.005044 | 49.82% |
| 10 | 1 member: all features | 0.277064 | 0.023198 | 47.26% |
| 11 | 1 member: 75% features | 0.288919 | 0.019455 | 44.42% |
| 12 | Greedy: random order | 0.312856 | 0.009310 | 43.73% |
| 13 | Greedy: frequency | 0.319671 | 0.019151 | 42.88% |
| 14 | Greedy: Battle Elo | 0.324925 | 0.018659 | 43.63% |
| 15 | Greedy: title position | 0.342006 | 0.020268 | 42.28% |
| 16 | Dummy control | 0.560442 | 0.011364 | 5.32% |

### fixed_sqrt2

| Rank | Model | Mean loss | Fold SD | Exact accuracy |
|---:|---|---:|---:|---:|
| 1 | Neural network (width 2,000) | 0.200727 | 0.007372 | 66.87% |
| 2 | L1 logistic regression | 0.212540 | 0.007993 | 65.87% |
| 3 | Decision tree | 0.250614 | 0.007730 | 60.93% |
| 4 | 243 members: 75% features | 0.281243 | 0.012357 | 57.39% |
| 5 | 243 members: all features | 0.285396 | 0.014869 | 57.52% |
| 6 | Level-wise logistic regression | 0.289143 | 0.005739 | 53.81% |
| 7 | 15 members: 75% features | 0.295097 | 0.010469 | 56.06% |
| 8 | 15 members: all features | 0.302371 | 0.012695 | 55.81% |
| 9 | Greedy: taxonomy association | 0.382977 | 0.010095 | 49.82% |
| 10 | 1 member: all features | 0.404623 | 0.024448 | 47.26% |
| 11 | 1 member: 75% features | 0.427129 | 0.018607 | 44.42% |
| 12 | Greedy: random order | 0.440463 | 0.012108 | 43.73% |
| 13 | Greedy: frequency | 0.445082 | 0.017528 | 42.88% |
| 14 | Greedy: Battle Elo | 0.446094 | 0.019245 | 43.63% |
| 15 | Greedy: title position | 0.458507 | 0.016685 | 42.28% |
| 16 | Dummy control | 0.780684 | 0.009041 | 5.32% |

### retrained_sqrt2

| Rank | Model | Mean loss | Fold SD | Exact accuracy |
|---:|---|---:|---:|---:|
| 1 | Neural network (width 2,000) | 0.200727 | 0.007372 | 66.87% |
| 2 | L1 logistic regression | 0.212540 | 0.007993 | 65.87% |
| 3 | Decision tree | 0.250614 | 0.007730 | 60.93% |
| 4 | 243 members: 75% features | 0.273704 | 0.010255 | 58.08% |
| 5 | 243 members: all features | 0.274346 | 0.012223 | 58.35% |
| 6 | 15 members: 75% features | 0.287091 | 0.008620 | 56.49% |
| 7 | Level-wise logistic regression | 0.289143 | 0.005739 | 53.81% |
| 8 | 15 members: all features | 0.290001 | 0.011413 | 56.77% |
| 9 | Greedy: taxonomy association | 0.375888 | 0.015864 | 50.54% |
| 10 | 1 member: all features | 0.403131 | 0.023662 | 47.49% |
| 11 | 1 member: 75% features | 0.421615 | 0.017258 | 44.93% |
| 12 | Greedy: Battle Elo | 0.427848 | 0.026036 | 45.41% |
| 13 | Greedy: random order | 0.437159 | 0.013157 | 44.06% |
| 14 | Greedy: title position | 0.454572 | 0.012287 | 42.90% |
| 15 | Greedy: frequency | 0.454918 | 0.012610 | 42.63% |
| 16 | Dummy control | 0.780684 | 0.009041 | 5.32% |

### fixed_q2

| Rank | Model | Mean loss | Fold SD | Exact accuracy |
|---:|---|---:|---:|---:|
| 1 | Neural network (width 2,000) | 0.141756 | 0.005499 | 66.87% |
| 2 | L1 logistic regression | 0.153535 | 0.007110 | 65.87% |
| 3 | Decision tree | 0.185878 | 0.007431 | 60.93% |
| 4 | Level-wise logistic regression | 0.209851 | 0.006202 | 53.81% |
| 5 | 243 members: 75% features | 0.214187 | 0.010697 | 57.39% |
| 6 | 243 members: all features | 0.220990 | 0.013074 | 57.52% |
| 7 | 15 members: 75% features | 0.227829 | 0.009009 | 56.06% |
| 8 | 15 members: all features | 0.237247 | 0.011759 | 55.81% |
| 9 | Greedy: taxonomy association | 0.328285 | 0.008215 | 49.82% |
| 10 | 1 member: all features | 0.347028 | 0.024728 | 47.26% |
| 11 | 1 member: 75% features | 0.365902 | 0.019989 | 44.42% |
| 12 | Greedy: random order | 0.382990 | 0.010662 | 43.73% |
| 13 | Greedy: frequency | 0.387075 | 0.018314 | 42.88% |
| 14 | Greedy: Battle Elo | 0.391201 | 0.019064 | 43.63% |
| 15 | Greedy: title position | 0.404434 | 0.017701 | 42.28% |
| 16 | Dummy control | 0.693115 | 0.009766 | 5.32% |

### retrained_q2

| Rank | Model | Mean loss | Fold SD | Exact accuracy |
|---:|---|---:|---:|---:|
| 1 | Neural network (width 2,000) | 0.141756 | 0.005499 | 66.87% |
| 2 | L1 logistic regression | 0.153535 | 0.007110 | 65.87% |
| 3 | Decision tree | 0.185878 | 0.007431 | 60.93% |
| 4 | 243 members: 75% features | 0.207292 | 0.010234 | 57.82% |
| 5 | Level-wise logistic regression | 0.209851 | 0.006202 | 53.81% |
| 6 | 243 members: all features | 0.210147 | 0.011290 | 58.09% |
| 7 | 15 members: 75% features | 0.222530 | 0.006429 | 56.34% |
| 8 | 15 members: all features | 0.229473 | 0.009860 | 56.20% |
| 9 | Greedy: taxonomy association | 0.326333 | 0.011865 | 49.99% |
| 10 | 1 member: all features | 0.345974 | 0.025747 | 47.42% |
| 11 | 1 member: 75% features | 0.359982 | 0.018777 | 44.79% |
| 12 | Greedy: Battle Elo | 0.379656 | 0.022910 | 44.69% |
| 13 | Greedy: random order | 0.382178 | 0.011061 | 43.79% |
| 14 | Greedy: frequency | 0.392152 | 0.019219 | 42.46% |
| 15 | Greedy: title position | 0.405541 | 0.016065 | 42.28% |
| 16 | Dummy control | 0.693115 | 0.009766 | 5.32% |

## Paired comparison with level-wise logistic regression

Differences are native ensemble minus level-wise loss on the same fold. Negative means favour the ensemble. No significance test or independent test set is implied.

| Comparison | Ensemble | Mean difference | Paired fold SD | Ensemble fold wins |
|---|---|---:|---:|---:|
| fixed_sqrt2 | 243 members: 75% features | -0.007900 | 0.008211 | 4/5 |
| fixed_sqrt2 | 243 members: all features | -0.003747 | 0.009765 | 3/5 |
| retrained_sqrt2 | 243 members: 75% features | -0.015439 | 0.008434 | 5/5 |
| retrained_sqrt2 | 243 members: all features | -0.014797 | 0.007729 | 5/5 |
| fixed_q2 | 243 members: 75% features | 0.004336 | 0.006719 | 1/5 |
| fixed_q2 | 243 members: all features | 0.011138 | 0.007848 | 0/5 |
| retrained_q2 | 243 members: 75% features | -0.002560 | 0.007167 | 4/5 |
| retrained_q2 | 243 members: all features | 0.000296 | 0.007817 | 3/5 |

## Evidence

- `results.json`: classical fold scores, histograms, prediction hashes and provenance; no product rows.
- `rankings.json`: all five comparison arms with paired fold losses.
- `reference.json`: frozen environment and archived fold aggregates, with source SHA-256 hashes.
- `protocol.md`: fixed training and evaluation settings.
- `validation.json`: independent PostgreSQL reconstruction and export checks.
- `validation-notes.md`: test, environment, storage and figure checks.
- `../low-q-sensitivity-2026-11-02/results.json`: unchanged native comparison.

Private recovered predictions are in `padjective.paper_classical_q_runs`; the aggregate classical report is in `padjective.paper_low_q_reports`. No manuscript text was changed.
