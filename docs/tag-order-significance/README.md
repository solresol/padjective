# Tag-order significance, 21 October 2026

- Input: Shopify Postgres snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`.
- Shape: 6,693 products, 2,542 features; five frozen folds.
- Replay: four deterministic orders and five random seeds; all published mean losses reproduced.
- Test: four exploratory, two-sided, approximate Nadeau--Bengio corrected t contrasts, four degrees of freedom, one Holm family.
- Randomisation: five seeds averaged within each fold before paired testing.
- Persistence: `padjective.paper_order_significance`, analysis key `tag-orders-2026-10-21`, `pg_default` table and primary-key index.
- Export: aggregate fold metrics and contrasts only.
- Excluded source: older `umllr_order_ablation_*` rows labelled `paper` have 6,527 observations and different losses.

```sh
uv run -m padjective.paper_order_significance --output /tmp/order-significance.json --analysis-key NEW_UNIQUE_ANALYSIS_KEY
uv run -m pytest -q
```

The JSON records the exact analysis and runtime source SHA-256 values.
