# PAPER_PLAN.md — the paper this should become

*Written 2026-07-12, after a multi-agent review of the padjective repo, the rejected
SIGIR-eCom draft, the thesis (Chapter 11), all accumulated reviews, a literature sweep,
three competing paper designs scored by two independent critics, and — crucially —
two empirical gating checks run against the live database on raksasa (see
"Verified facts" below and [reference/paper_plan_gating_checks.py](reference/paper_plan_gating_checks.py)).*

## Verdict on the current draft

The SIGIR-eCom framing ("we lose 3× on our own loss, but the model is tiny and
auditable") is honest but structurally weak: the headline is a concession, the two
parameter-constrained baselines score *worse than the dummy* (0.734 / 0.673 vs 0.560,
an instant reviewer target), and the paper's most remarkable numbers — the
strict-refinement effect (held-out loss **0.079 vs 0.840**, Mann-Whitney p = 4.5e-61)
and the 1.11-active-params operating point no other model family occupies — are buried
as "diagnostics". The rejection was predictable from the framing, not the substance.

## Verified facts (2026-07-12, live tables, taxonomy_association, p = 71)

1. **The README's "coefficients collapse onto pure powers of p" claim is false for the
   taxonomy model and must be retired.** Of 5,455 nonzero fitted coefficients across 5
   folds, **exactly 0 (0.00%)** are pure powers of 71. The modal coefficient has 4
   nonzero base-71 digits — coefficients are code-like corrections, not powers.
   (The claim originated with the never-run adjective-ordering model in gptslop.tex;
   the paper should state, in one honest paragraph, that it does not hold here. Note
   that only 1.9% of coefficients are themselves current taxonomy codes and ~16% are
   differences of two current codes — this does **not** contradict the contact theorem,
   which predicts each coefficient equals a *residual at fit time*, a telescoped
   quantity; verifying that exactly requires replaying the fit, which is worth doing.)

2. **The linguistics angle is dead as a headline.** The gating scatter both critics
   demanded: Spearman correlation between a tag's coefficient valuation and its mean
   relative title position is ρ ≈ −0.05 to −0.14 per fold, pooled ρ = −0.055,
   permutation p = 0.33 (n = 321 tags). Valuation vs battle win-rate: ρ ≈ −0.10.
   There is no signal. Combined with the existing ablation (battle_elo loses to
   *random* ordering, 21/25; mean_title_position is dead last), the
   adjective-ordering/"royal order" story should be reduced to one honest ablation
   paragraph and never be the frame. Do not build the linguistics paper.

3. **The valuation histogram explains why strict refinement is rare and why the killer
   experiment below has headroom.** 97.4% of nonzero coefficients have valuation 0
   (5,315/5,455; 135 at v=1, 5 at v=2), and 73% of all coefficients are zero. The
   fitted model is mostly a "first informative tag carries a full taxonomy code,
   other tags abstain" lookup; genuine multi-tag composition at distinct depths — the
   regime where the model is 10× better — happens for only ~8% of held-out products.
   Nothing in the current fitter *asks* for valuation-disjointness.

4. **The window is real and open.** Martins, "Learning with the p-adics"
   (arXiv:2512.22692, Dec 2025) is self-described "exploratory theoretical work"
   — building blocks and open problems. Abstract-level check confirms: no flatness
   result, no characterisation of minimisers, no strict refinement, no real-data
   benchmark. v-PuNNs (arXiv:2508.01010) is an architecture paper on clean ontologies.
   Nobody owns "what does optimising a non-Archimedean loss actually *do*, on real
   data". Both critics estimate roughly a one-year window before the Martins line
   occupies it. (Week-1 task: read the full paper, not just the abstract.)

## The recommended paper

**Working title:** *Learning under ultrametric losses is selection, not estimation:
a characterisation of p-adic regression, with strict refinement as a provable and
enforceable mechanism.*

**Core claim (four-part characterisation + one intervention):**
1. *Flatness* — ultrametric losses are locally constant under any Lipschitz
   parameterisation; gradients carry no signal (already proven in padjective.tex).
2. *Selection* — every minimiser of every additive loss φ(|r|_p) passes through n+1
   data points (published contact theorem + the unification.tex extension), so
   optimisation is combinatorial selection over a finite, data-determined lattice;
   the greedy fitter's "coefficient = some current residual" rule is a corollary,
   which is why coefficients look like taxonomy codes.
3. *Collapse regimes* — for q > m the objective is exactly lexicographic in the
   residual-valuation histogram; for p > p₀ it degenerates to contact counting
   (both proven in the thesis). p = 71 deliberately sits between the regimes —
   currently unstated anywhere, and it explains the design.
4. *Strict refinement (the new theorem)* — if a prediction's active coefficients have
   pairwise distinct valuations, the ultrametric inequality holds with *equality*, the
   prediction decomposes level-by-level down the hierarchy, and coarse-level
   cancellation is impossible. Two lines from the strong triangle inequality; a draft
   proposition already exists in the thesis chapter, misplaced under
   "Operational History". This converts the p = 4.5e-61 empirical effect into a
   theorem with a per-depth loss formula — and it is a *certificate no Euclidean
   linear model can emit*.

**The killer experiment (makes it a paper to be proud of):** valuation-disjoint greedy.
Modify the greedy pass so a tag's candidate coefficients are restricted to (or
penalised toward) valuations not already used by co-occurring accepted tags. Run on
the frozen paper snapshot, 5-fold CV, alongside the existing 9 models. Measure:
(a) does strict coverage rise materially from ~8%, and (b) does mean p-adic loss move
from 0.263 toward the 0.086–0.120 Euclidean band? Include a matched-difficulty control
(strict vs non-strict products at equal tag-count/frequency strata) to separate
mechanism from selection. Fact 3 above says the headroom is enormous — the current
fitter produces valuation collisions almost everywhere.

**The insurance policy (run alongside, not instead):** the risk–coverage comparison.
Strict-refinement certificate gating of the p-adic model vs softmax/margin-gated L1
logistic regression and level-wise LR at matched coverage, on the frozen snapshot.
This is the one comparison a skeptical reviewer will demand ("a certificate nobody
compared to confidence gating is decoration"). It also makes the paper two-way robust:

- If the intervention works → headline is **theorem → mechanism → improvement**
  (TMLR, stretch ICML/NeurIPS).
- If it fails but the certificate sits on the risk–coverage frontier → headline is
  **"a free deterministic trust certificate flat models can't give you"**, with a
  certificate-gated model→LLM cascade cost table ($/million products against the
  45M-product corpus) as the applied payoff. Still TMLR-viable.

Both outcomes strengthen thesis Chapter 11; every failure mode still yields a
defensible chapter.

## Supporting work to import (cheap, kills known reviewer objections)

- **Replace the two worse-than-dummy constrained baselines** with matched-budget
  interpretable competitors: L1 logistic swept down to ~1–2 active features per
  prediction, depth-limited trees, one Rudin-style integer scoring system. This
  simultaneously fixes the strawman objection and densifies the active-params–loss
  frontier from 7–9 points to dozens (the current R² = 0.794 regression on n≈7 after
  exclusions is fragile and will be attacked).
- **One hierarchy-aware trained baseline** (hierarchical cross-entropy on the existing
  tag features) as the minimum answer to "you only beat flat baselines you built".
- **Leave-store-out split** (store_domain exists for all products, 1,532 stores; days
  of work) — answers the store-idiom leakage critique of product-level folds.
- **Valid-path projection row** (0.26324 → 0.26279) in the main table — neutralises
  the "15.6% of predictions aren't valid codes" attack.
- **Per-depth (per-p-power) error histograms** for every model — already computed per
  fold; shows *where* in the tree each model fails and makes the 0.086-vs-0.263 gap
  interpretable (root misses dominate: one root-level miss costs 1.0, a leaf miss ~1e-8).
- **Statistics pass** — per-fold CIs, paired tests, Fisher's exact where expected
  counts < 5 (open thesis review items D21/D22); rerun the December-2025-vintage
  Zubarev and strict-refinement tables on the frozen snapshot (todo item 12).
- **Learning-curve figure** from model_performance_history (corpus grew 6.7k → 25.5k
  products) — nearly free, addresses the small-data critique.

## Hygiene queue (do regardless, mostly one-liners)

- Retire the pure-power claim in README.md (cite the measured 0.00%).
- Strip AI-draft artifacts: the "Yann LeCun / not an amateur effort" sentence; the
  hand-fitted parsimony-score constants (−0.1/0.3) or label them as descriptive only.
- The worked example de-anonymises product 2219 of the paper's own anonymised
  benchmark (real title + tags) — replace with a synthetic or consented example.
- Reconcile the corpus count (paper says 55M, FUTURE_PLANS 49.9M, live ~45.8M) to one
  defined number.
- Give the HF dataset a real license (currently `license: other` with no terms) and a
  provenance section (BuiltWith 2020 seed, scrape dates, regulated-category note,
  de-anonymisation risk statement).
- Fix `PADJECTIVE_PAPER_AS_OF` hardcoded default in cronscript.sh; fix the 404'd
  GitHub link on the live site.
- Import the thesis's coefficient non-identifiability caveat into the auditability
  section (audited coefficients are one optimum among many — the thesis already has
  the right language).
- Position the p-adic loss explicitly as Dekel-2004/Bertinetto-2020 tree-induced
  error in algebraic packaging; cite Martins 2512.22692 and v-PuNNs prominently and
  claim only: the ring structure, the selection characterisation, the certificate.

## What NOT to do (in this paper)

- No linguistics/adjective-ordering headline (killed by the gating check above).
- No gold audit of 2,500 products, no hyperbolic/HXE-suite/CRM bake-off, no external
  dataset replication, no LLM baselines — that is the natural *follow-up* benchmark
  paper (NeurIPS D&B / ECIR) if the certificate result lands. A ~500-product
  stratified extension of the first1000 audit is an acceptable cheap substitute if
  reviewers demand label-noise quantification.
- No eigenvarieties material beyond one future-work sentence.
- Approximation guarantee for the greedy: strictly timeboxed (≤3 weeks), ships as a
  remark whichever way it goes.

## Venue and timeline

- **Primary: TMLR** — tolerates characterisation papers with nuanced empirical
  components, no accuracy bar, rolling deadline.
- Fallback for the theorem package: *p-Adic Numbers, Ultrametric Analysis and
  Applications* (already published there; the padic-journal REVTeX variant exists).
- The planned journal resubmission of the applied paper continues as the companion.
- Rough schedule (solo, ~3 months): week 1 = Martins full-paper diff + replay-the-fit
  coefficient check + retire README claim; weeks 2–6 = strict-refinement theorem
  writing + valuation-disjoint greedy + matched-difficulty control + risk–coverage
  figure; in parallel weeks 2–8 = supporting work above; weeks 9–12 = writing.

## Why this is the right bet

It is the only design where (i) the mathematics the thesis already proved becomes
load-bearing rather than decorative, (ii) the paper's strongest existing number
(p = 4.5e-61) becomes the centrepiece instead of a footnote, (iii) both outcomes of
the main experiment are publishable, (iv) the work fits one researcher in one
quarter using infrastructure that already runs nightly, and (v) it lands inside an
open, time-limited window (post-Martins, pre-everyone-else) that the empirical
bake-off and linguistics designs would miss.
