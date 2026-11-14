# Adapter limits and positive-control proposal, 2 October 2026

This is a review of saved evidence. No further training, synthetic experiments,
rate searches or ensembles were run. The negative result applies to the
independent bounded adapter and its declared experiment, not to Salazar's
general method.

## Exact departures and scope

1. **Representation and capacity.** Centers are canonical residues in
   Z/71^7Z, with no fractional coefficients, and every supported parameter disk
   has radius between 71^-7 and 1. Salazar represents signed exact Z[1/p]
   centers with a finite *significant-digit* capacity and allows the tree's
   upward directions. The paper's reported p=3 arithmetic has 19 significant
   digits and its adverse start is lifted to radius 3^-8. It also stops
   descending at capacity: finite precision itself is not an invention here.
   Our fixed absolute seven-digit Z_p restriction, upper-radius cap, and
   shallower available coarsening/refinement are different and were chosen to
   match the existing Padjective model.
2. **Boundary histories.** At the lower radius cap, this adapter substitutes
   zero for the unavailable downward derivatives when updating momentum
   histories. Adam's minimum-derivative second moment also excludes unavailable
   directions. At radius 1 it disables upward motion. These are explicit
   constrained-boundary conventions, not equations asserted to come from the
   paper. They can affect later history even after a coordinate leaves a cap.
3. **Grouping.** The adapter joins all attaining-term ties over the minibatch
   by connected components. It omits the paper's affine cancellation-based
   splitting of contributions that can be separated or cancel. Coarser groups
   mean fewer independent selected coordinates and can change queueing,
   coordinate counters, moments and clipping. It does use one awaiting-turn
   coordinate per group, preserves unselected histories, and applies a common
   time clipped at the first encountered vertex.
4. **Model and evaluation.** This is binary-input, single-output, no-intercept
   linear regression, matching the existing member. It does not implement the
   paper's general affine input scalings, classifiers or nonlinear models.
   Within the positive-radius interval, modulo reduction does not change the
   affine direct disk loss, because multiples of 71^7 have norm at most its
   smallest radius. At radius zero, scores are those of the finite-residue
   Padjective task, not a claim of exact ordinary Q_p equality. Retained-center
   extraction follows the paper's regression convention; the different G.3
   classification extraction was not used.
5. **Protocol.** The warm start is an exact *whole-coordinate* optimum of
   Padjective's objective, which is a stronger/different condition than the
   paper's engineered *single-digit* local minimum. All optimizer histories
   start at zero. We tested one fold and seed, fixed kappa=1, 1,000 updates,
   and radius 71^-7. The paper explores a rate grid and different low-prime
   synthetic data. This is not a reproduction of its adverse-start experiment.
6. **Implementation validation.** The released official source contains array
   operations but no optimizer at the inspected commit. Analytic germ,
   representative, momentum-history and clipping tests validate individual
   pieces of this independent implementation; they are not end-to-end proof
   of an implementation equivalent to the paper's optimizer.

## What moved and what the logs can establish

Warm Momentum recorded 379,691 applied motion proposals and 2,517,834
coordinate updates; warm Adam recorded 368,677 and 2,518,908. The proposal
counter is not a count of distinct changed states: very small floating-point
steps may leave a stored log radius unchanged. However, database readback
confirms real net radius changes in **407/2,542** Momentum coordinates and
**402/2,542** Adam coordinates. Both medians remain at log radius -7. Their
largest final radii have base-71 logs -5.8461876369 and -5.8402966335.

Both first updates moved six coordinates upward to radius 71^-6 from zero
momentum histories. Therefore some selected initial directional slopes were
negative; this was not an all-zero-gradient run. Neither warm trajectory
changed any coefficient digits at any recorded update, and the endpoint
coefficient arrays and predictions match the baseline exactly. This rules out
a mere final rounding/retained-center extraction artefact as the explanation
for the unchanged point model. Other arbitrary representatives of a larger
disk could have different scores, but were not learned or selected here.

Initial grouping produced 2,261 groups from 2,542 coordinates, demonstrating
attaining ties. By the final minibatch there were 2,508/2,510 groups. Those
counts include coordinates absent from the minibatch, so they are not counts
of informative or zero gradients. The unit test shows a tied maximum can hide
a single-coordinate downward radius derivative while joint motion changes the
radius. Its particular target makes the joint downward loss change positive;
it is a coupling test, not a useful joint-descent or escape demonstration.

The logs contain aggregate groups, motion proposals and radius ranges, not
per-coordinate slopes, group membership, requested step sizes, common clipping
times, boundary hits, or floating-point no-op counts. They cannot distinguish
tied-gradient stalls, clipping/numerical stagnation, and insufficient budget.
The recorded radius maxima never approach the coarser digit scales, making
insufficient coarsening plausible. No causal diagnosis or budget sufficiency
claim follows. Warm point loss stayed fixed while direct loss changed around
1e-13, too little to support a meaningful relaxed improvement claim.

## Existing positive controls

There is **no tested escape from a certified whole-coordinate optimum** in the
adapter suite. The one-parameter tests learn target 5 from an uninformed state
under GD, Momentum and Adam, with correct digit choices and preserved masks.
That initial coefficient is not coordinate-optimal. Reproducibility tests and
the real cold run demonstrate digit learning, not escape. The pre-existing
exact-fitter suite has a two-coordinate local-optimum example with a better
joint solution, but it does not run a tree optimizer on it.

## Small synthetic plan for review only

No part of this plan has been executed. Keep the initial review target below
five seconds of one-thread raksasa runtime per optimizer, no external data,
no rate tuning, and no ensemble.

1. **Useful tied descent geometry.** For p=3, F(c)=c1+c2, c=(0,0), target 0,
   and equal open-edge radii r=3^(-1/2), analytically check that each coordinate's
   downward derivative is zero, while lowering both radii at unit rate has
   loss derivative -1/2. Also check the growing-radius and target-outside-disk
   cases. This supplements the current coupling test with genuinely improving
   joint motion; it does not imply that one-coordinate-per-group descent finds
   that motion.
2. **Certified joint-improvement candidate.** Use p=3, precision 3, binary
   rows (1,0), (0,1), and three copies of (1,1); targets (1,26,0,0,0).
   Coefficients (0,0) have mean point loss 2/5 and are coordinatewise optimal:
   any nonzero change sacrifices three correctly fitted joint rows to help at
   most one missed singleton. Exhaust all 27 values of each coordinate and all
   27^2 joint states. The joint solution (1,26) has zero finite-residue loss.
   Lift (0,0) to radius 3^-3. Confirm the coordinate certificate independently.
3. **Separate expected mechanisms.** With a full batch at the lifted start,
   a coordinate's upward slope is +1/5 and its mathematical downward slope
   +1/10 (the adapter disables descent at its lower cap); thus
   zero-history full-batch GD/Momentum/Adam cannot begin descending. Treat this
   as an expected stationary control, not a failed positive control. A useful
   joint point solution need not imply a descending local tree germ.
4. **Controlled escape-path check.** Before stochastic testing, use an
   explicitly predeclared minibatch schedule containing the two singleton rows
   during coarsening/refinement, then the full five-row batch. On those
   singleton rows the initial upward slopes are negative and branch choices
   point toward (1,26). Check every vertex, representative change, momentum
   history, common clipping time and final full-objective loss. This deliberately
   curated gradient stream validates ability to traverse an escape path; it is
   not an unbiased training or performance result. It would require a small
   explicit batch-schedule hook, which has not been implemented.
5. **Predeclared stochastic attempt.** Only after that path check, evaluate
   uniform minibatches of size 1 and 2, fixed seeds 42 and 1729, fixed kappa=1,
   and at most 1,000 updates for Momentum/Adam. Report every endpoint, including
   nonescapes; do not search for a lucky seed or silently lengthen the budget.
   Escape is a hypothesis, not a guaranteed property of this candidate toy.

For these checks, log raw directional slopes, active/coupled group members,
selected coordinates, requested speeds, common clipping times, limiting
coordinate/boundary, strict radius-change counts, digit changes and exact
full-objective loss. If the controlled path or its arithmetic fails, repair the
adapter before treating the Padjective result as evidence about optimization.
Reproducing the paper's actual adverse-start task would instead require a
general affine adapter for its nonbinary p-adic inputs and a separate approved
protocol. Neither extension is authorized by this review plan alone.
