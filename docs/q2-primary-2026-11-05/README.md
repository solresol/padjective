# Primary q=2 rescoring

This study rescores the original frozen paper predictions at q=2
and q=71. The encoding prime, fitted coefficients, defaults, candidate
sets, consensus rules and folds are unchanged. No models were refitted.

The source is the Shopify PostgreSQL database. The 6,693-product
snapshot digest and all original held-out q=71 metrics reproduce.
Each new score is computed from an error-depth histogram and
checked against an independent row-by-row distance calculation.

| Saved prediction family | Sets |
|---|---:|
| Feature fraction, ensemble size, aggregation and roster | 6,125 |
| Tree and forest configurations | 1,690 |
| Published polynomial runs | 45 |
| Greedy tag orders and seeds | 45 |
| Unrestricted inference controls | 45 |
| Completed single-fold Berkovich pilots | 5 |
| Total | 7,955 |

These sets contain 10,648,370 predictions across the same benchmark
products. The five additional greedy projection checks derive
predictions from the original decoder; they are not new fits.
Mihara's unfinished runs have no held-out predictor to rescore.

Archived prediction hashes are verified where available. The
polynomial exports did not retain prior prediction hashes; their
frozen run IDs, snapshot digest and all archived held-out metrics
are checked. Individual predictions and product records remain in
PostgreSQL. Only counts, scores and provenance are exported.

With the Shopify connection configured, run from this checkout:

```sh
uv run python -m padjective.paper_q2_primary \
  --reference-dir ../papers/padjective/padic-journal/data \
  --output docs/q2-primary-2026-11-05/results.json
uv run -m pytest -q
```

The aggregate report is persisted under
`paper-q2-primary-2026-11-05` in `padjective.paper_low_q_reports`.
The portable paper generator independently replays the original
four-test tag-order family, twelve-test subset/aggregation family,
216-test secondary screen and ten-test projection family at q=2,
with q=71 retained alongside it. Forest and random-order seeds
are averaged within fold before equal-fold means and sample SDs.
The reporting weight was chosen after the sensitivity results;
these are exploratory replays of reused folds.
