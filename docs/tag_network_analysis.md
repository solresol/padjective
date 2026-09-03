# Frozen paper snapshot network analysis

`tag-network-analysis` quantifies the circular structure raised in the
handwritten paper review. It always reads the authoritative Postgres snapshot
tables; the default input is the frozen `paper` alias.

The analysis separates two graphs:

- The product-tag incidence graph is bipartite. A three-product example in
  which every product shares a different tag with each of the other products
  is a six-cycle. The global cycle rank, `E - V + C`, counts independent
  undirected cycles. The 2-core measures the size of the cyclic backbone.
- The title-battle graph points from a later (winning) tag to an earlier
  (losing) tag. Weak components are independent ranking subproblems. Strongly
  connected components, reciprocal pairs, and directed three-tag cycles
  quantify contradictory ordering evidence.

The full graph is the only valid basis for exact decomposition. The report also
removes tags above several frequency thresholds as a sensitivity analysis.
Those hub-suppressed networks can reveal latent modules, but they are not exact
independent subproblems because removing a hub also removes a real constraint.

Run on the database host:

```sh
uv run tag-network-analysis \
  --snapshot-ref paper \
  --output build/tag-network-analysis/paper.json
```

The command validates the snapshot metadata against its incidence table,
stores the complete JSON result in
`padjective.tag_network_analysis_runs` (created explicitly in `pg_default`),
and optionally writes the same result to a local JSON file. A fixed seed and
the configured number of draws make the chain-conditioned six-cycle estimate
reproducible.

The brute-force indicator is deliberately conservative: arbitrary permutation
search is marked feasible only when the largest weak tag-battle component has
at most ten tags. Disconnected components can be fitted or ranked separately;
hub-suppressed components require an approximate gating or ensembling design
and subsequent held-out evaluation.
