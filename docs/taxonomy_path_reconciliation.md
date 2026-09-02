# Taxonomy path reconciliation

Padjective requires Shopify's dotted numeric taxonomy hierarchy keys, such as
`1.1.13.8`, for stable grouping and cross-validation. The shared
`cantbuymelove.taxonomy.taxonomy_path` column can also contain human-readable
display paths. Those are useful to the Can’t Buy Me Love website, but they are
not interchangeable with the numeric key.

The nightly pipeline now reconstructs the numeric key in
`padjective.taxonomy_path_reconciliation` before running any model. Evidence is
used in this order:

1. a numeric path still present in the live taxonomy table;
2. a numeric path preserved in an immutable benchmark snapshot;
3. a path reconstructed from the Shopify taxonomy ID after its root mapping has
   been established by direct evidence.

Conflicting or incomplete evidence stops the pipeline. It is not converted into
a smaller dataset.

Each run also writes a stage ledger to
`padjective.taxonomy_path_reconciliation_audits`. It distinguishes the total
catalogue from the successively filtered benchmark population: title, product
details, complete taxonomy metadata, resolved numeric path, non-empty tags,
taxonomy-frequency threshold, tag-frequency threshold, canonical URL, and URL
deduplication.

## Historical discontinuity

Counts produced after human display paths began replacing numeric paths, but
before this reconciliation was introduced, are not estimates of catalogue size.
They are legacy benchmark counts with an increasing fraction of taxonomy rows
excluded. Those snapshots are immutable and should not be overwritten. The
first reconciled audit is the start of a comparable replacement series; exact
counterfactual totals cannot be reconstructed from already-filtered snapshots.
