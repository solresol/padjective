# Side-experiment figure contract

Surface: PDF figure exports for the scholarly paper/research notes requested
in this thread, with PNG previews for visual QA. These are journal-context
figures, not an OpenAI-branded report or an HTML/dashboard artifact. Do not
insert them into the submitted-paper candidate automatically.

Source: the validated Postgres-backed `paper_subspace_evaluate` JSON export.
Retain the complete aggregate source, including fold sizes, metrics, model
costs, masks' counts, hashes and statistical families. Never plot a partial grid.

1. **Ensemble-size comparisons.** Question: how does subset fraction interact
   with ensemble size, and how do the five aggregators differ using all features?
   Family: line, two side-by-side panels with a common linear loss axis and
   logarithmic member-count axis. There are nine measured sizes per series,
   five series per panel, and five paired folds behind each point. Static
   renderer: Matplotlib PDF, with a 180-dpi PNG preview. The title is descriptive;
   supported interpretation will be added to the research notes after validation.
   No extrapolation, smoothing, or independent-replication error bands.
2. **Primary comparisons.** Question: which prespecified changes are supported
   after the uncertainty correction? Family: dot-and-interval, three vertically
   stacked groups (subset, aggregator, endpoint interaction), each with four
   contrasts. Plot corrected mean differences and the 95% Bonferroni simultaneous
   intervals for all 12 primary contrasts. Mark zero, and label negative as
   favouring the named alternative. Retain Holm p values in the accompanying
   exact table; do not imply that a simultaneous interval and Holm test must
   give identical accept/reject decisions.

Palette: hard two-root cap, blue (`#3569a8`, with deliberate lighter shades)
and orange (`#d18a3c`), plus charcoal/grey/white. Different marker shapes and line
styles distinguish series without colour. No red/green significance encoding.
Common font: DejaVu Sans. Axis anchors remain visible; light horizontal grids.

Footprint: size comparison 12 by 5 inches; primary contrasts 10 by 9 inches.
Titles, legends, axis labels, metric direction, n=6,693/five folds, exploratory
status and source-batch provenance must fit the actual exported PDF/PNG. Inspect
both previews at their final size; render the PDF too if preview and PDF layouts
diverge. Exact score lookup belongs in the accompanying tables, not on 90 points.

Output names: `subspace-size-comparisons.pdf/.png` and
`subspace-primary-contrasts.pdf/.png`, alongside the aggregate JSON and notes.
