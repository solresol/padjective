# Handwritten corrections captured on 10 September 2026

Status: **Captured and visually transcribed; manuscript edits have not been
applied in this capture pass.** The table records 32 correction groups. A group
can contain several connected pen marks. Unfinished wording, blank quantities
and uncertain arrow destinations are retained below.

## Source and verification

- reMarkable document: `padjective-padic-journal-ensembles-2026-09-10`.
- Document ID: `f602eed6-89ba-42b8-8284-957310681aa6`.
- Latest document modification recorded in the desktop app's local metadata:
  **10 September 2026, 18:48:27 AEST**. This was the newest of the three
  Padjective documents in that library at capture time.
- Exported through the desktop app on 10 September 2026 and reopened as a
  readable **22-page PDF** with the expected title and red handwriting.
- Native pen-stroke files exist for **pages 1, 2 and 3 only**. All three rendered
  pages were visually inspected. The PDF export preserves the clean base's
  printed text on all 22 pages (comparison ignores whitespace).
- This is a new annotated document, distinct from the 6 September copy used
  in the [previous correction pass](../../papers/padjective/padic-journal/annotation-review-2026-09-07.md).
- Capture directory, outside Git:
  `/Users/gregb/Documents/padjective-annotations-20260910/`.
- Annotated export: `padjective-20260910-annotated.pdf`.
- Annotated SHA-256:
  `cd35353530592fcb34af96b791516e459cabc47cf32522d559c234e627279f4e`.
- Clean base: `base.pdf`; SHA-256:
  `d0789e3720aa7dfe352a441fc2e16843c90617619cc795beead6721127c039d6`.
- The same directory preserves `native-source/`, `provenance.json`,
  `source-at-capture.tex`, and the rendered pages under `extraction/pages/`.

Source anchors below refer to
`/Users/gregb/Documents/devel/papers/padjective/padic-journal/padjective-padic.tex`
at Papers commit `09cc434a98c18ed849d5ced1b9f5cce46e3ac5de`. Line numbers are
navigation aids for that revision, not permanent identifiers. The preserved
source includes the later release-link changes; it is not asserted to be the
exact source revision from which the reMarkable base PDF was built.

## Page 1: abstract and introduction

| ID | Source anchor | Handwriting or marked change | Interpretation/status |
| --- | --- | --- | --- |
| P1-01 | L54, “We introduce sparse greedy” | Insert “a family of” before “sparse greedy”. | Clear insertion. |
| P1-02 | L56, after the first abstract sentence ending “merchant-supplied tags.” | “The sparse models have a deterministic training time and create very small, easy-to-simulate models.” | Clear added sentence; connector points after the first sentence, not into the title. Preserve this claim for the edit pass. |
| P1-03 | L59–64, numerical comparison in the abstract | “the best classical comparator's loss of 0.075559” is circled, with a long connector back toward the first greedy-model result. “not convergence to” is struck. | Move/recast the comparator comparison earlier. The exact destination and joining words need interpretation. The long connector crosses the ensemble-method sentence; do not treat that crossing alone as a deletion of the method. |
| P1-04 | L62–64, “The primary roster's lowest mean loss…” | Strike “The primary roster's lowest mean loss is 0.144178 at 135 members;”; replace “the size study” with “A size study of larger ensembles”. | Clear deletion and replacement. Connect with P1-03 when reconstructing the abstract. |
| P1-05 | L69–71, “The greedy model permits…” | Strike the sentence “The greedy model permits short hand-checkable predictions; ensembling improves accuracy at substantially greater cost.” | Clear deletion. |
| P1-06 | L71–72, final abstract sentence | Replace “An exploratory” with “We observe an unexpected”; strike “in the expanded comparison, but depends on counting conventions.” | Clear direction. The unstruck word “persists” needs a grammatical decision when joining the revised sentence; do not claim a fully dictated replacement. |
| P1-07 | L95, “An e-commerce catalogue contains…” | Replace the subject with “The database supporting a Shopify store”. | Clear replacement; handwritten subject is in the left margin. |
| P1-08 | L99–106, Can't Buy Me Love and missing taxonomy labels | “Most products do not have reliable merchant-supplied taxonomy labels.” is circled. A loop/arrow points toward the later “but Can't Buy Me Love…” clause. A horizontal mark also crosses the earlier aggregation-site description. | Preserve as a restructuring instruction. Exact sentence order and the extent of any deletion are uncertain; check the page before applying. |
| P1-09 | L106, “supply them” | Replace “them” with “the missing labels”. | Clear replacement. |
| P1-10 | L112, “we do not claim…” | Strike “we do not claim to have measured human audit times.” | Clear deletion; repair the preceding semicolon when applying. |
| P1-11 | L99–104, catalogue/aggregation-site description | “Which is based on a scrape of — Shopify sites, sampling — products in total.” | Both quantities are blank in the handwriting. The connector rises into the catalogue-description passage; exact attachment needs checking. Do not substitute the 6,693-product experimental sample or an unverified live total. |

## Page 2: introduction and dataset opening

| ID | Source anchor | Handwriting or marked change | Interpretation/status |
| --- | --- | --- | --- |
| P2-01 | L145, “tags, and the fit…” | Strike “and” before “the fit”. | Clear deletion; punctuation may need adjustment. |
| P2-02 | L145–147, linear prediction description | “the fit for an ensemble is…” | Incomplete addition. The author has requested an ensemble explanation but has not dictated its completion. Keep separate from the existing linear-model formula. |
| P2-03 | L169, “a particular p-adic linear regression problem” | Strike “linear”. | Clear deletion. |
| P2-04 | L170, “the polynomial comparator need not itself be linear in the original tags.” | Strike that clause and add “together with some proposed new algorithms that have interesting properties.” | Clear wording. Join it to the preceding empirical-paper description when applying. |
| P2-05 | L168–170, sentence beginning “This is primarily…” | “Make this the first sentence of the paper.” | Structural move. The note appears to refer to the revised empirical-paper sentence in P2-03/P2-04. Confirm its destination as the opening of the introduction, rather than assume a change to the abstract. |
| P2-06 | L176–177, contribution (i) | Strike “linear” from “p-adic linear regression”. | Clear deletion. |
| P2-07 | L182–183, contribution (iv), “and loss in the data set” | Handwritten “p-adic” below the paragraph, with a small insertion mark. | Appears to qualify “loss”. The mark lies between this line and the preceding “classifiers” line; preserve the image reference rather than adopt the automated extractor's competing “classifiers” placement. |
| P2-08 | L188–190, dataset section opening | Circle “As foreshadowed in Section 1,” and move it with an arrow to after the sentence ending “three distinct benchmark configurations.” | Clear relocation; adapt punctuation at its new position. |
| P2-09 | L188, “the public release” | Insert “dataset” between “public” and “release”. | Clear insertion. |
| P2-10 | L196, “filters, it contains” | Replace “it” with “the paper snapshot”. | Clear replacement. |
| P2-11 | L203, “The rolling benchmark” | Handwritten typewriter-format `latest` points at “rolling”. | Replace the adjective with the configuration name, rendered as `\texttt{latest}`. |

## Page 3: dataset and evaluation details

| ID | Source anchor | Handwriting or marked change | Interpretation/status |
| --- | --- | --- | --- |
| P3-01 | L214, “The numerical reference labels” | Strike “numerical reference”; write “ground truth” in quotation marks. | Clear replacement, including the quotation marks. This is a requested wording change, not new evidence that the labels were independently verified. |
| P3-02 | L214, “the reconciled” | Strike “reconciled”. | Clear deletion. |
| P3-03 | L215, “taxowalk catalogue” | Replace “catalogue” with “output”. | Clear replacement. |
| P3-04 | L215–218, adjudication and accuracy qualifications | Strike the two sentences beginning “The experiment does not independently adjudicate all 6,693 labels…” and “Reported accuracy therefore measures agreement…”. | Clear requested deletion of both sentences. Record the instruction without inferring any change in how the labels were produced or checked. |
| P3-05 | L224–225, “The reported product total…” | Strike “The reported product total is therefore the number of eligible products, not the number of records with errors.” | Clear deletion. |
| P3-06 | L227–228, collection route | Strike “through unauthenticated product endpoints.” | Clear deletion of the phrase; the following sentence about available responses remains unmarked. |
| P3-07 | L232, split unit | Replace “The split unit is the product, not the merchant.” with “Cross validation is done by product (not by merchant)”. | Clear replacement. Hyphenation and a terminal full stop can be regularised during application. |
| P3-08 | L233, seed | Add parentheses around “with seed 42”. | Clear punctuation change. |
| P3-09 | L235, stratification fallback | Strike “and otherwise falls back to unstratified folds.” | Clear requested deletion. Preserve the instruction without treating it as evidence that the implementation changed. |
| P3-10 | L236–237, archived assignments | Strike “The archived fold assignments, rather than regeneration from the filtered release, define the reported experiment.” | Clear deletion. The final fold sizes that follow remain unmarked. |

## Items to resolve when applying

1. **Scrape scope (P1-11):** obtain the intended store and product counts from
   the appropriate source/date; neither number is present in the handwriting.
2. **Ensemble description (P2-02):** complete the deliberately unfinished
   explanation using the actual ensemble prediction rule.
3. **Arrow destinations (P1-03, P1-08, P2-05):** settle the comparator placement,
   the catalogue paragraph order, and the destination of the new opening.
4. **Abstract grammar (P1-06):** decide how to join the new “We observe…” opening
   with the retained association wording.
5. **Small insertion mark (P2-07):** verify that “p-adic” qualifies “loss”; the
   handwriting and automated extraction disagree about its placement.

These are capture notes, not an instruction to rerun experiments, change
database records, update a public release, or replace the annotated document.

## Validation

- The skill's automated extraction timed out on page 1 and returned incomplete
  context/confidence fields on pages 2 and 3. Its raw results are retained in
  `extraction/`; they are not the checked correction list. This visually checked
  transcription also corrects its readings of the `latest` target, the dataset
  insertion, the retained `taxowalk` name and the seed-42 parentheses.
- All three pages with native handwriting inspected at readable resolution,
  including enlarged crops for the abstract and detailed notes.
- Printed text in the full annotated export matches the full clean base after
  whitespace normalisation.
- Original PDF, clean PDF, source snapshot and native document files preserved
  outside Git with SHA-256 hashes in `provenance.json`.
- Required Padjective unit tests: `uv run -m pytest -q` — **237 passed,
  6 skipped**. Twelve existing small-sample neural-network batch-size warnings.
