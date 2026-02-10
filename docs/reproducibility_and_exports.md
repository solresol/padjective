# Dataset exports and reproducibility notes

## What is currently exported

The production cron pipeline publishes SQL dumps from `build/site/datadumps/` to
`datadumps.ifost.org.au`.

The exported tables currently include:

- `cantbuymelove.taxonomy`
- `cantbuymelove.product`
- `cantbuymelove.product_taxonomy`
- multiple `padjective.*` result/model tables (`battles`, `umllr_*`,
  `taxonomy_pclr_*`, `taxonomy_pcnn_*`, etc.)
- `public.product_details`, but filtered to products that exist in
  `cantbuymelove.product`

## Implication

At present, the export path is broader than “only products tagged into the taxonomy”.
Because `cantbuymelove.product` is dumped in full, untagged products may be
included in published SQL dumps.

## Reproducibility barriers to be aware of

1. **Data snapshot drift**: training and site generation read directly from a live
   Postgres source, so reruns against a later database state can produce
   different results unless a snapshot boundary is fixed.
2. **Mixed persistence targets**: most outputs persist in Postgres (`padjective`
   schema), but parameter-constrained neural network artifacts are also written
   to a local SQLite file (`data/taxonomy_pcnn_classifier.sqlite`) by default.
3. **Public export scope**: current dump generation exports broad source tables,
   not only the taxonomy-tagged subset.

## Recommended direction

To match the policy “export only taxonomy-tagged products plus processing
results”, restrict dumps to:

- products that join to `cantbuymelove.product_taxonomy`
- derived rows required for replay in `padjective.*`
- optionally filtered `public.product_details` for the same tagged product set

and persist all model outputs in Postgres where feasible.
