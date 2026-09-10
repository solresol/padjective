# Rolling 75% versus all-feature ensembles

This experiment tests whether the paper's promising 75% result survives on the
current labelled catalogue. It does not replace production models or alter the
paper snapshot. The experiment is exploratory: the 75% choice was made after
examining the paper results, and some live products overlap that snapshot.

## Population and reference labels

Read Shopify Postgres directly. The source is `cantbuymelove.product`, joined
by its unique store/run/handle key to `public.product_details`. Reference labels
come only from `cantbuymelove.product_taxonomy.taxonomy_source = 'gold_llm'`.
These are LLM reference classifications, not human-verified truth. Never score
against Padjective's own predictions. Resolve numeric paths using the existing
reconciliation table; missing references and unresolved paths are counted.

The 11 September preflight found 31,204 labelled source records, versus about
49.9 million scraper records in the previous day's archive audit. The latter
includes repeated crawls. Without reference labels, most of that archive cannot
supply accuracy measurements. This first experiment covers the full labelled
catalogue, not all unlabelled scraper records.

Keep the latest source row for each store/handle, then canonical URL; report
duplicate removals and conflicting labels. Any of an entity's canonical URLs
matching the immutable paper snapshot makes it an overlap product. This is
identity overlap, not proof of independent content. Retain raw normalised tag
sets privately in Postgres; never export product titles, URLs or per-row tags.
Keep rare categories, empty tag sets and products whose tags are unseen in
training. Their defaults and out-of-vocabulary rates belong in the result.

The source query runs in a repeatable-read transaction through a server cursor.
The in-memory fitting cohort has a hard 100,000-source-row preflight ceiling;
exceeding it stops the run, not truncates it. The 50-million-record archive is
never materialised by this command. Scaling beyond that ceiling needs a separate
resource/design review.

## Frozen design

Both variants use 243 components and the existing fixed `BANK_SEEDS`, with
`fit_seed = seed_base + fold`. Three-member runs are engineering smoke tests,
not evidence about the final ensemble. Do not select members by test score.

Assign stores, not individual products, to five stable folds using SHA-256 of
the normalised myshopify domain. All products from a store stay together; the
assignment survives new arrivals. Cross-store duplicate content can remain.
This is a stricter, different split from the paper, so absolute scores are not
directly comparable. Every scored product is held out from the model that
predicts it. The same folds, observations and seeds are used for both variants.

Normalise and substring-filter tags as in the paper. Vocabulary eligibility is
at least five fitting products per tag, calculated separately in each fitting
fold without consulting test features or labels. Every component sees either
all eligible features or a random 75% (ceiling), using the existing deterministic
mask construction. Keep rows with zero eligible features; fit the zero-output
default on fitting labels only. Candidate output paths are fitting-label paths
only. Unseen test paths stay in the denominator and are reported separately.

Freeze p=83 and precision E=8 for this series, covering the current maximum
branch 80 and depth eight. Fail on an unrepresentable new path rather than
silently truncate or change the metric. The integer loss scale is p**(E-1):
this removes a common factor from every distance and leaves all minimisers
unchanged. Use Python-integer reduction when required. Unit tests compare the
new implementation with the historical fitter and direct distances; the
historical files remain unchanged.

Fit from zero to a complete strict no-improvement sweep, with 100 sweeps and
600 fitting seconds per component. Every fit needs an independent direct-distance
coordinate certificate, Python-integer prediction reconstruction and a saved
model/prediction digest. Any incomplete member prevents ensemble publication.
Use at most four single-threaded processes on raksasa. Raw, default-adjusted
component predictions are aggregated by the current valid-path p-adic medoid.
No new voting rules or size sweep are selected in this series.

## Evidence and longitudinal interpretation

Persist immutable input cohorts, provenance, model records, per-product paired
predictions and aggregate reports in separate `padjective.live_subspace_*`
tables and indexes, all in `pg_default`. A repeated identical cohort reuses the
completed comparison; dated observations still record that the source was
checked. A changed cohort gets new models and its own report. No old result is
overwritten to look current.

Report overall and paper-overlap/non-overlap performance, exact and root
accuracy, p-adic loss, vocabulary coverage, unseen target paths, zero-feature
rows, stored coefficients, fit times and five fold differences. The primary
comparison is 75% minus all-feature mean fold loss at M=243. The approximate
overlap-corrected five-fold t interval is descriptive evidence, not an exact
test or a licence for repeated daily significance claims. Do not treat
components, products or successive daily runs as independent training
replications. Overlap/non-overlap subgroup contrasts are secondary.

For trends, compare paired errors on unchanged common products separately from
new arrivals, changed rows and removed products. A rolling-CV improvement may
reflect retraining as well as new data; it is not fixed-model temporal
generalisation. Store immutable model versions so a later frozen-model transfer
experiment remains possible. Do not label pre-existing non-paper products as
future arrivals. These models and their reduced-feature configurations remain
outside the original active-support/log-log regression.

Run a daily check after the usual nightly pipeline. Skip refitting when the
cohort is unchanged. Do not change algorithms, thresholds, seeds, ensemble size
or deployment status in response to interim scores. Report completion, coverage
changes and failures; unchanged healthy state needs no notification.
