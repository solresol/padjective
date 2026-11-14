# Prime and precision control protocol, 2 October 2026

Only repeat the existing five-row synthetic control on raksasa at p=71 and
precision 7. This is the sole new fitting scope; no product data, ensembles,
rate search, extra seeds or changed optimizer settings. Preserve the p=3
traces, product results, manuscript changes and private Library identities.

Rows remain (1,0), (0,1), and three copies of (1,1). Put q=71^7 and targets
(1,q-1,0,0,0). Initial centers (0,0), log radii -7 and zero histories. For
either coordinate w while the other is zero, the unnormalized loss is
|w-s|_71 + 1 + 3|w|_71 for s=1 or -1. If 71 divides w, this equals
2+3|w|_71, equal to 2 only at the zero residue. For a unit it is
4+|w-s|_71 >=4. Thus (0,0) is a strict whole-coordinate optimum with mean
loss 2/5. The singleton equations force the unique joint zero-loss pair
(1,q-1), which cancels in the joint rows. This proof covers every residue.
Check it against the existing exact residual-medoid certificate and bounded
valuation-stratum probes; never enumerate q^2 or even all q coefficients.

Match the frozen p=3 roster: kappa 1, beta 0.9, beta2 0.999, epsilon 1e-8,
feature mask both coordinates, retained canonical-center extraction.

* Full-batch GD/Momentum/Adam: seed 42, 100 updates, all five rows.
* Controlled GD/Momentum/Adam: seed 42, 400 updates. Exactly singleton rows
  0 and 1 for updates 1–200, then all five rows for updates 201–400.
  This curated path validates mechanics; it is not unbiased learning evidence.
* Uniform minibatch Momentum/Adam: seeds 42 and 1729, batches 1 and 2,
  1,000 updates each. Report all eight endpoints and transient minima.

Exactly 14 fits with five seconds per fit, 60 seconds overall including
certification and traces, and an outer 75-second timeout including startup.
Use one nice-15 process and one numerical thread. Record actual update counts
if a deadline is reached; do not extend or restart an incomplete run silently.
Write each trial checkpoint immediately. Refuse to overwrite existing traces.

Use the same optimizer, batch sampler, tie grouping, boundary history
convention and diagnostics. Log raw/used slopes, moments, speeds, coordinate
clocks, child choices, root/other vertices, groups, attaining ties, common
clipping, before/after states and numerical no-ops. Independently score every
point endpoint with Python integer arithmetic and audit all stored motions.
Compare p=71 outcomes and mechanisms against the preserved p=3 roster.
No optimizer modification or tuning based on the results is authorized here.

The two profiles change both prime and precision. They cannot separate the
effect of those two factors; both retain the same finite-residue adapter
departures from the paper. An escape would validate a path at the product
prime/capacity, not guarantee product-data improvement or paper equivalence.

Run `padjective.paper_salazar_synthetic --profile p71`, then the read-only
`padjective.paper_salazar_synthetic_audit --profile p71`. Artifact directory:
`artifacts/salazar-synthetic-p71-20261002`. Return one proposed small diagnostic
for the unchanged product result if the control works; do not launch it.
