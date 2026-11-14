# Journal figure formats, checked 22 September 2026

The journal is *p-Adic Numbers, Ultrametric Analysis and Applications*.
Its [specific author instructions](https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/1636992_Guidelines%20for%20Authors%20of%20p-Adic%20Numbers%2C%20Ultrametric%20Analysis%20and%20Applications.pdf)
state: “Figures must be in EPS format and with Courier font.” The downloaded
PDF was read directly with `pdftotext`; it remains accessible although the
web reader could not parse its application/octet-stream response.

The publisher's current [illustration guidance](https://www.pleiades.online/en/authors/guidlines/prepare-electonic-version/images/)
also accepts EPS, requires embedded fonts and lines at least 0.5 pt wide,
asks for 8–8.5 cm or 17–17.5 cm artwork, and excludes background grids and
shading. Line styles and markers should remain distinct in monochrome.
The new journal figures use the two-column width, separate line/marker
styles and no grid. Their source canvas is 17.5 cm; the tight content box
is scaled to the available manuscript width on inclusion.

The past submission workflow supports the EPS recollection. In the local
`sudoku-padic-regression` repository, commit `92d1fc3` (13 September 2026),
“Add reproducible publisher and EPS submission packages”, added
`scripts/build_eps_package.py`. It converts vector PDF originals with
`pdftops -eps -level3 -rasterize never` and validates the EPS files through
Ghostscript. This establishes the earlier packaging workflow, not the
delivery or acceptance of a publisher email.

All capacity-comparison figures now have EPS, PDF, SVG and PNG versions.
Matplotlib writes the first three as vector artwork. Courier is used and
embedded in local EPS/PDF exports. Portable builds can use Nimbus Mono PS,
the URW Courier equivalent, from `fonts-urw-base35`; font selection fails
explicitly if none of those fonts is available. PNGs remain convenient
previews. Selected points identify member or tree counts. Original log–log
views are retained, with linear-loss alternatives and a separate small-ensemble
view. The new paper section uses two paired figures: active work under both
loss scales, then stored state and broader scoring work under linear loss.

The existing main-paper figures were already EPS; their generators now use
Courier typography too. Graphviz's taxonomy illustration is exported through
vector PDF and then EPS so its fonts are embedded. Existing data and the
eight-model regression remain unchanged.
