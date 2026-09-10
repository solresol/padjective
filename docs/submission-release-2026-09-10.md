# Frozen replication release: 10 September 2026

Published and independently read back:
[paper-submission-2026-09-10](https://huggingface.co/datasets/gregb/product-taxonomy-bench/tree/paper-submission-2026-09-10/submission/2026-09-10).
This completes the later-experiment packaging item from the manuscript review.
No reported scientific values changed.

## Identity and preservation

- Content commit: `edf77476974fc3b26e9c7a74178124a1d6b1986b`.
- Annotated tag object: `ab1c03a38c43435aa012e97577abae5f2fe4deb5`.
- Manifest SHA-256:
  `ed0792dd8556b19a51248cad07f85a49eac2e3002030ce2cd4e26922bc753b06`.
- Frozen manuscript source: Papers `a57c244c58c5feb89e0980fc9cfd2b804e312798`.
- Initial replication package source: Padjective
  `41f9f185db87aa630fb6f97cb00016dd28fad15c`; sealing source
  `a7984e9951786f7d5af0d128dab39a6256a3b072`.
- The 6 September tag object remains
  `6a496bd7f1969629a41f547371d15d3375aa86c1`, resolving to unchanged content
  commit `7cbb35030dcb2175d130e44a42442a1efb1c983b`.

All 110 new files were downloaded anonymously by immutable commit and checked
against local hashes. All 52 pre-existing non-attribute files retain their
content identifiers. Hugging Face appended one exact new-file LFS rule to
`.gitattributes` for the new canonical PDF; every earlier rule is unchanged.
The publisher rejects rewrites, global/wildcard additions and rules for old
paths. No old release was overwritten, deleted or retagged.

The Hub refs endpoint exposes annotated tag-object IDs; `repo_info` or Git's
peeled `^{}` reference resolves their actual content commits. The manifest's
historically named `reference_tag_commit` field records the raw tag-object ID;
`reference_commit` is the content revision used for every reference download.
The [publication receipt](submission-release-2026-09-10-receipt.json) records
both identities and the single scoped storage-rule addition.

## Fresh offline checks

The clean Python 3.11.11 environment contained only the 18 pinned numerical
dependencies and no `psycopg` driver. It read the public anonymised export,
not the private catalogue. Jobs ran on raksasa with four one-thread workers
for the ensemble bank, then three one-thread workers for the comparators.

| Check | Verified outcome |
|---|---|
| Public input versus archived Postgres input | 6,693 rows, 2,542 ordered columns, 363 paths, all five folds and all 34,042 nonzero entries match |
| Coordinate fits | All 1,215 freshly fitted coefficient and prediction hashes match; 3,088,287 supported coordinates independently re-certified |
| Ensembles | All 945 fold/size/roster predictions and metrics match; primary consensus also checked by the separate prefix evaluator |
| Zubarev | All 45 coefficient/prediction fingerprints match, with 320 accepted transitions per completed schedule |
| Mihara | Ten rank obstructions, five inclusion obstructions and five unfinished 180-second searches reproduce; no unfinished fit is scored |
| Post-hoc capacity certificates | All 15 bounds and degree means recomputed exactly from public rows |

Mihara's wall-clock-limited draw/restart counts and all timings are not
bitwise targets across hardware. The original 6 September package is included
byte-for-byte; its 12-thread reference environment also passed its environment
check. The old classical/neural/digitwise suites were not refitted again in
this release pass; their earlier validation records are retained, including
the width-2,000 reference loss rather than the one-thread sensitivity result.

Eleven archived rows have stale scalar `tag_count` metadata, smaller than
their actual feature lists. Neither loader uses that count. The actual
feature relation/list is authoritative and matches entry-for-entry; neither
the data nor the original metrics were changed to accommodate the metadata.

`validation/input-manifest.json` preserves the pre-sealing manifest against
which the expensive runs executed. Its hash is
`2ca91c6e8dd85b1759e25bc876c0989a2373d5929a8f74c6434f55248491857b`.
The final manifest adds the resulting validation reports and manuscript,
avoiding a self-referential checksum. Numerical source and input hashes are
unchanged between the tested and sealed packages.

Five evidence records (data parity, ensemble replication, published-method
replication, capacity replication and publication) are stored in
`padjective.paper_replication_release_checks`. Both that table and its
primary-key index were verified in `pg_default`; the export builder now sets
the local default explicitly for implicit constraint indexes too. No original
experiment or production-model rows were modified.

## Paper and code validation

The isolated packaged manuscript builds its main PDF and historical supplement;
all 18 manuscript-evidence tests pass. The main PDF remains 22 pages, and the
first 20 pages' extracted text is unchanged. Updated pages 21–22 were rendered
and visually checked; no overfull boxes or unresolved references/citations
were introduced. Existing journal-font and bibliography warnings remain.
The availability section links the new archive. Canonical PDF SHA-256:
`af2c9cde880e5e9d76bb870caac7f9b7008a8bbbbdc8623a81cbdd3fadd81283`.

The repository suite passes 237 tests with six existing skips and 12 existing
neural batch-size warnings. Release-specific tests cover exact source/AST
preservation, feature ordering, duplicate rejection, checksum enforcement,
standalone fitting/consensus, complete-grid gates, capacity arithmetic and
the narrowly scoped storage-rule exception. The documented release-only
download command was tested anonymously; no rolling dataset download is needed.

This pass updates the stable local PDF and review copy. It does not claim a
new reMarkable import or journal submission. Existing annotated documents and
the earlier reMarkable delivery are untouched.
