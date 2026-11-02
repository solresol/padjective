# Classical baselines under the same low-Q scores

User request: add the paper's Euclidean/classical models to the low-Q ranking
comparison. Preserve their original classification training objectives; Q changes
evaluation only. The comparison is descriptive on the existing paper folds.

Use PostgreSQL snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09` (6,693 products,
2,542 features, 363 labels), snapshot digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`.
Keep the float32 sparse binary matrix, row order, feature order, five frozen folds,
and base-71 taxonomy encoding. Score Q = sqrt(2), 2, 3, 5, 10, 71. Report equal-weight
fold means, sample fold SDs, and exact/root accuracy.

The generic stored classical prediction tables use different populations and
cannot supply these predictions. Recover them by fitting the archived settings:

- Dummy: most frequent class.
- L1 logistic regression: saga, C=1, max_iter=200, random_state=42, n_jobs=-1.
- Decision tree: class_weight=balanced, random_state=42.
- Level-wise logistic regression: original runtime implementation, lbfgs,
  class_weight=balanced, max_iter=1000, original majority-class fallbacks.
- Neural network: one width-2,000 ReLU layer, alpha=0.0001, batch_size=256,
  max_iter=10000, random_state=42; other scikit-learn defaults unchanged.

Use the archived Python/package/BLAS versions and 12 numerical threads. Check
every fold's Q=71 loss and exact accuracy against the archived aggregates before
accepting its results. These checks establish numerical reproduction of those
published metrics, rather than proof that every historical predicted label is
identical (the old package preserved aggregates, not per-row predictions).
Record warning messages and iteration counts. Original iteration caps remain.

Store private predictions, ordered product keys, scores, settings and provenance
in `padjective.paper_classical_q_runs` with `pg_default` tablespace. Export
aggregates only, to a new report separate from the completed native low-Q run.
Keep the existing native report unchanged. Compare the classical scores both
with fixed native Q=71 fits and with the separately labelled low-Q retrained fits.
Do not update manuscript claims automatically.
