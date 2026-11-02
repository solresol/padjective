# Fixed-prediction Q sensitivity

Both axes are logarithmic: Q versus the equal-weight mean held-out d_71,Q loss. The taxonomy encoding and valuation remain base 71.

The older appendix already reports Q = 2, 3, 5, 10 and 71 for greedy and ensemble models. The matched 2 November results add sqrt(2) and the dummy/classical models on all six Q values. No new training or rescoring run was needed for this figure.

The curves use the original Q=71 native fits and unchanged classical objectives throughout. The separate native retraining experiments at sqrt(2) and 2 are not mixed into these curves. Markers identify scored values; connecting segments guide the eye rather than assert a power law. Sample fold SDs are reported in the table below.

The figure uses black lines with distinct markers and dash patterns. Experimental context is in the separate caption, without a subtitle or explanatory footer inside the figure.

## Caption

Mean held-out $d_{71,Q}$ loss versus $Q$ on logarithmic axes. Fixed predictions across five frozen paper folds (6,693 products). Encoding base stays 71. Only the scoring weight $Q$ changes. Markers show the calculated $Q$ values. Lines connect scored points; no power law is fitted

`caption.md` contains the caption text; `caption.tex` provides the LaTeX caption command.

## Scores

| Model | Q=√2 | Q=2 | Q=3 | Q=5 | Q=10 | Q=71 |
|---|---:|---:|---:|---:|---:|---:|
| Dummy control | 0.781 ± 0.009 | 0.693 ± 0.010 | 0.638 ± 0.010 | 0.601 ± 0.011 | 0.578 ± 0.011 | 0.560 ± 0.011 |
| p-adic greedy | 0.383 ± 0.010 | 0.328 ± 0.008 | 0.298 ± 0.007 | 0.280 ± 0.006 | 0.270 ± 0.006 | 0.263 ± 0.005 |
| p-adic ensemble: 243, 75% features | 0.281 ± 0.012 | 0.214 ± 0.011 | 0.177 ± 0.010 | 0.155 ± 0.010 | 0.143 ± 0.010 | 0.135 ± 0.010 |
| p-adic ensemble: 243, all features | 0.285 ± 0.015 | 0.221 ± 0.013 | 0.185 ± 0.012 | 0.164 ± 0.012 | 0.152 ± 0.012 | 0.145 ± 0.012 |
| Neural network, width 2,000 | 0.201 ± 0.007 | 0.142 ± 0.005 | 0.110 ± 0.004 | 0.092 ± 0.004 | 0.082 ± 0.003 | 0.076 ± 0.003 |
| L1 logistic regression | 0.213 ± 0.008 | 0.154 ± 0.007 | 0.121 ± 0.007 | 0.102 ± 0.007 | 0.092 ± 0.007 | 0.086 ± 0.006 |
| Decision tree | 0.251 ± 0.008 | 0.186 ± 0.007 | 0.150 ± 0.007 | 0.129 ± 0.007 | 0.118 ± 0.007 | 0.110 ± 0.007 |
| Level-wise logistic regression | 0.289 ± 0.006 | 0.210 ± 0.006 | 0.166 ± 0.007 | 0.142 ± 0.008 | 0.129 ± 0.009 | 0.120 ± 0.010 |

## Evidence and reproduction

`data.json` contains all 48 plotted points, their five fold scores, log10 coordinates, SD interval endpoints and source hashes. Source aggregate files:

- `../low-q-sensitivity-2026-11-02/results.json` (native fits).
- `../low-q-classical-2026-11-02/results.json` (dummy and classical fits).

All curves use the same 6,693 products and fold sizes 1300, 1337, 1348, 1372, 1336. No product-level records or predictions are included here. The source experiments and their prediction checks are documented in their own directories.

`validation.json` records independent checks of all 48 plotted points and 15 overlapping score cells from the older appendix, the source/PostgreSQL checks, unit tests, and visual checks of the PNG and grayscale-rendered PDF. The plotted data and validation are also stored in `padjective.paper_low_q_reports` under `q-loss-curves-2026-11-02` and `q-loss-curves-2026-11-02-validation`.

```sh
uv run python -m padjective.q_loss_curves \
  --native docs/low-q-sensitivity-2026-11-02/results.json \
  --classical docs/low-q-classical-2026-11-02/results.json \
  --output docs/q-loss-curves-2026-11-02
```
