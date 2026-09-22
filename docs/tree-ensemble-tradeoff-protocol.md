# Trees and p-adic ensembles: complexity–accuracy experiment

Fixed on 22 September 2026 before fitting new classical models. This is an
exploratory comparison on the paper's existing folds. Do not change the paper,
its active-support regression, the production models, or earlier experiment rows.

## Data and models

Read Shopify Postgres snapshot `244ddbe3-0a2c-4c04-9436-0a0253108a09`, digest
`039175941ed2a2ec888bc6c7314c3c5f8afac1201ff604413dedb55641706222`:
6,693 products, 2,542 binary tags, 363 taxonomy paths, five frozen folds.
Preserve feature and row order. Each model fits four folds and scores the fifth.
Use p=71 and precision 7. Report equal-fold mean p-adic loss (primary), root
accuracy and exact-path accuracy, training scores, and individual fold results.

Reuse the validated all-feature and 75%-feature p-adic banks in subspace batch
`773bf2ab-d2b1-4c32-bfde-15d501782fa0` (all-feature source batch
`059d7eb8-14b5-4d5d-857f-134a2098fd89`). Use the fixed primary roster, existing
valid-training-path medoid rule, and sizes 1,3,9,15,27,45,81,135,243. The 75%
variant was promising in prior experiments, so its inclusion is informed by
earlier results, not independent confirmation. Recheck prediction hashes,
candidate sets, fold denominators and scores against the Postgres snapshot.

Decision trees: sklearn Gini classifier, seed 42, all available features.
Sweep maximum depth 1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256,384,512,
unlimited; maximum leaf count 2,4,8,16,32,64,128,256,512,1024; and
cost-complexity alpha 0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,
0.001,0.003,0.01,0.03,0.1. These are separate sweeps, not a Cartesian grid.
Run both the paper's balanced class weights and ordinary unweighted fitting.
The unweighted control matters because the primary evaluation is unweighted.

Random forests: Gini, bootstrap samples, sqrt features per split, maximum
depth 2,4,8,16,32,64,128,unlimited, and nested prefixes of 1,3,9,27,81 trees.
Run both balanced and unweighted fitting, seeds 42,1729,20260922 (unchanged
across folds). Average scores over seeds within each fold, then over folds;
these are repeated fits, not one larger ensemble or independent datasets.
Use sklearn's mean leaf-probability aggregation and argmax classifier.
Do not choose weights, seeds, pruning levels or aggregation using test labels.
The full grid is fixed; no score-driven extensions or early stopping.

## Complexity counters

The paper's 258.35 and 324.76 are active work per prediction, not total stored
parameters. Keep the member-only counter: mean split decisions summed across
trees, or mean nonzero coefficient consultations summed across linear members.
Report distributions over products. A coefficient consultation and a branch
decision are different operations. This counter excludes defaults, p-adic
consensus, forest probability aggregation, and input encoding; it cannot
establish equal latency, energy, or human effort.

Also compare a declared compact inference representation in logical scalar
slots, rather than the incompatible legacy tree parameter proxy:

- Tree: 4 slots per internal node (feature, threshold, two children), 1 per leaf
  (predicted class). Probability vectors can be discarded for hard prediction.
- Forest: 4 per internal node, 2 per nonzero leaf probability (class index and
  probability), and 1 root reference per tree. All leaf distributions are kept
  because sklearn averages probabilities, including impure leaves.
- P-adic ensemble: 2 per nonzero coefficient (feature index and value), 1 default
  per member, 1 code per candidate path. Member boundaries are free, as are
  common feature dictionaries and codebooks for all families.

These slots are a transparent representation proxy, not bytes, information
bits, degrees of freedom, or a minimal encoding. Record native counts too.
For sensitivity, retain broader scoring counts: p-adic member work plus default
uses plus seven prefix observations per member and seven per candidate;
forest branch decisions plus nonzero leaf-probability contributions plus final
class comparisons; tree decisions plus its leaf lookup. These heterogeneous
counts are bookkeeping sensitivity only, not a runtime benchmark.

## Analysis and validation

Plot every measured configuration and descriptive lower-loss frontiers versus
active work and stored slots; show root/exact accuracy and fold variation.
Separate balanced and unweighted classical fits. A frontier is the observed
best score among tried configurations below a cost ceiling, selected on these
same held-out folds. Label it exploratory and potentially optimistic; do not
report a selected configuration's ordinary confidence interval as valid
post-selection inference. No interpolation-based crossing claim. State the
measured configurations on either side of any crossing, and distinguish
beating a shallow tree from beating an unconstrained one. Do not assume
unconstrained trees necessarily win or that observed saturation is a theorem.

Verify the unpruned balanced seed-42 reference against the paper. Independently
score integer p-adic distances and traverse a sample of fitted trees. For
forests reconstruct the probability averages; check every nested prefix against
sklearn prediction. Persist predictions, native counts, metrics, source and
model fingerprints, protocol, package versions, fold identity and final report
to new tables under `padjective`, explicitly in `pg_default`. Export aggregate
JSON, a separate Markdown write-up, and PNG/SVG/PDF figures only. Product-level
data stays in Postgres. Run `uv run -m pytest -q`. Use at most two single-threaded
worker processes on raksasa after checking available resources.

Implementation references: [sklearn random forests](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
and [cost-complexity pruning](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).
