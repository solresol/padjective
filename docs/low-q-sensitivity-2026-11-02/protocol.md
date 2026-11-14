# Low-Q ranking experiment, 2 November 2026

Question: does reducing the hierarchy's depth weight change the ordering of
the greedy and ensemble configurations in the earlier Q-sensitivity study,
and does retraining with that objective change the answer?

Read the fixed 6,693-product, 2,542-feature input from Shopify PostgreSQL,
snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`, digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Keep its five folds, base-71 encoding, feature order and labels. Original
production and manuscript experiment tables are not modified. New private
fits and aggregate reports use `padjective.paper_low_q_models` and
`padjective.paper_low_q_reports`, with explicit pg_default storage.

Primary depth weight: Q=sqrt(2). Also train with Q=2, which makes two errors
first occurring at the second level equal one top-level error. Under
Q=sqrt(2), that equivalence applies to errors first occurring at level three.
Rescore all outputs at sqrt(2), 2, 3, 5, 10 and 71. These are evaluation weights;
the taxonomy encoding prime remains 71.

Reconstruct the five original one-pass orders with their original rules,
including random seeds 7, 13, 23, 37 and 101. Check the Q=71 control's fold
losses and exact accuracies against the October stored report before trusting
the low-Q comparisons. Use the original historical floating-point greedy
fitter for the controls: a preflight with integer medoids matched Q=71 loss
but changed exact matches through deep-level numerical ties. That preflight
was stopped before any low-Q ranking was inspected; its one completed row
is isolated under experiment v1. The corrected comparison uses v2.
Refit those orders under sqrt(2) and 2, preserving
their tag orders and training-only default policy. Their residuals retain
the original unrestricted-integer convention.

For repeated-coordinate regressors retain precision 7, zero initialisation,
original BANK_SEEDS plus fold, strict improvements, 100-sweep and 300-second
fit limits. Require a complete no-improvement sweep and a separate coordinate
certificate. Use the original full-feature design, and the original 75%
training-supported masks with SeedSequence([20260911, fold, seed_base]).
The primary valid-training-path medoid consensus uses the same Q as training.
Preserve every fold's original primary roster, including its nested 1- and
15-member prefixes. Start with those prefixes; the runner also supports the
full 243-member endpoints. A full bank needs 4,860 low-Q component fits.
No incomplete or survivor-only banks are scored.

The sqrt(2) optimiser uses the Pell convergent 131836323/93222358 to represent
even- and odd-depth penalties with integer costs. A checked sample/precision
bound proves this surrogate preserves the sign of every possible objective
difference at the current dataset size. Unit checks compare against direct
real-valued distance search and reproduce the original Q=71 optimiser and
consensus. Evaluation uses actual sqrt(2), independently checked per row.

Score the archived original ensembles directly from their private saved
predictions, requiring their hashes and Q=71 scores to reproduce. Keep frozen
1-, 15- and 243-member reference scores even when the new run uses 15 members.
Compute equal-weight fold means and sample fold SDs; average random seeds
within each fold. Include exact and top-level accuracy. Export aggregates
only; coefficients and product-level predictions remain in PostgreSQL.

The comparison is descriptive on repeatedly used folds. It does not select
new settings from held-out results or supply a new test population. The old
classical prediction tables use different populations, so they are not pooled
into this comparison. The primary question matches the earlier greedy/ensemble
Q table. Training-Q and scoring-Q are separate fields in every result.

Initially use four single-threaded workers on raksasa. The author subsequently
authorised use of the full machine for roughly the next hour; extend to both
243-member endpoints with 12 single-threaded worker processes, reusing the
completed validated fits. This changes the execution schedule, not the folds,
seeds, masks, objectives or model roster. Source and protocol are versioned
before the new ranking is inspected. The manuscript is not edited by this
experiment.

A performance-only revision stops medoid prefix scans once all observed
candidates are distinguished and skips zero-contribution loss levels. Both
the integer objectives and tie choices match the frozen first implementation
in 1,800 candidate searches and 900 loss-vector checks. Completed fits retain
their original implementation hash and are reused; the final report lists
the actual implementation hashes represented in the bank.
