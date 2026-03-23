# padjective

Calculate p-adic adjective embeddings and product taxonomy classification

## Overview

This project learns adjective ordering rules and product taxonomy classification using p-adic techniques. It combines theoretical research into p-adic embeddings with practical applications in e-commerce product categorization.

The pipeline analyzes product tags from e-commerce databases to determine:
1. **Tag ordering preferences** - which tags typically appear first in product titles
2. **Product taxonomy prediction** - classification of products based on their tags

## Theoretical Motivation

Across multiple projects we have noticed that when we train models with p-adic losses the learned coefficients almost always collapse onto pure powers of the prime \(p\). Even when starting from arbitrary integer weights the optimiser quickly drives them toward \(p^{b_n}\) with coefficient one. Mixed terms of the form \(a_n p^{b_n}\) with \(a_n \neq 1\) are rare in practice. We suspect there is a tropicalisation argument lurking here that would explain the collapse analytically.

### Embedding Strategies Under Test

1. **Byte/character encodings.** Interpret the UTF-8 (or ASCII) byte sequence of a word directly as a p-adic expansion. Words that are 2-adically close therefore share suffixes, which aligns with how Indo-European inflectional endings behave. The hope is that grammatical shifts correspond to simple linear operations in this space.

2. **Lexical hierarchy encodings.** Place each word inside a lightly pruned WordNet-like tree and turn the branch decisions into digits of the p-adic number. We and others have published variants of this approach. The embeddings themselves span a larger set of coefficients, yet downstream supervised learners still favour pure powers of \(p\).

3. **Sequential encodings for adjective order.** Focus on sequences of adjectives that precede a noun. Each adjective receives a p-adic integer that is a single power of \(p\), encoding where the adjective tends to appear relative to its neighbours. This representation is agnostic to meaning but accurately predicts which adjective should come first or next—essentially mirroring the behaviour required of an autoregressive language model.

### What Works So Far

With these encodings in place we fit linear models using p-adic losses. Compared to brute-force p-adic linear regression the supervised optimisation converges much faster and achieves strong accuracy on held-out adjective sequences. The workflow demonstrates that we can efficiently learn ordering preferences without leaning on semantic information.

## Practical Applications

### Tag Hierarchy and Ordering

For any given pair of tags, which one is more likely to appear first in a product title? We identify "equivalent depth" tags where they generally appear at the same depth, and ultimately assign an integer depth to every tag.

### Product Taxonomy Classification

We predict product taxonomy from tags using machine learning approaches, including parameter constrained logistic regression and neural networks. This allows automatic categorization of products based on their tag combinations.

## Database Schema

The project uses the simplified `cantbuymelove` schema:
- `cantbuymelove.product` - Product table with integer primary key (`id`)
- `cantbuymelove.product_taxonomy` - Links products to taxonomies via `product_id`
- `cantbuymelove.taxonomy` - Taxonomy definitions with `taxonomy_id`, `taxonomy_name`, and `taxonomy_path`

Products are joined to `public.product_details` to extract tags from the JSONB `product_detail` field.

### Taxonomy Table Structure

The `cantbuymelove.taxonomy` table provides the complete mapping between taxonomy identifiers and human-readable category names:

```sql
Table: cantbuymelove.taxonomy
Columns:
  taxonomy_id   TEXT PRIMARY KEY  -- Shopify GID (e.g., "gid://shopify/TaxonomyCategory/aa-1-10-2-1")
  taxonomy_name TEXT UNIQUE       -- Full hierarchical name (e.g., "Apparel & Accessories > Clothing > Outerwear > Coats & Jackets > Bolero Jackets")
  taxonomy_path TEXT              -- Either a taxonomy_code OR hierarchical text (see below)
```

#### Taxonomy Data Types

The `taxonomy_path` column contains one of two formats:

1. **taxonomy_code** (numeric hierarchical code): Dot-separated numeric codes representing the hierarchical position
   - Example: `"1.1.10.2.1"` corresponds to "Apparel & Accessories > Clothing > Outerwear > Coats & Jackets > Bolero Jackets"
   - Each number represents a level in the hierarchy
   - Used for p-adic encoding (see below)
   - The taxonomy_id suffix often mirrors this code (e.g., `aa-1-10-2-1`)

2. **Hierarchical text path**: Same format as `taxonomy_name`, using `>` separators
   - Example: `"Media > Sheet Music"`
   - Duplicates the information in taxonomy_name for some categories

Both formats are valid. The numeric taxonomy_code is essential for p-adic distance calculations and compact hierarchical representation.

**Example queries:**

```sql
-- Get the human-readable name for a taxonomy ID
SELECT taxonomy_name FROM cantbuymelove.taxonomy
WHERE taxonomy_id = 'gid://shopify/TaxonomyCategory/aa-1-10-2-1';
-- Returns: "Apparel & Accessories > Clothing > Outerwear > Coats & Jackets > Bolero Jackets"

-- Get taxonomy_code (numeric) for a category
SELECT taxonomy_id, taxonomy_name, taxonomy_path FROM cantbuymelove.taxonomy
WHERE taxonomy_path ~ '^[0-9.]+$' AND taxonomy_name LIKE '%Outerwear%'
ORDER BY taxonomy_path;
-- Shows categories with numeric taxonomy_code

-- Find all taxonomies in a category
SELECT taxonomy_id, taxonomy_name FROM cantbuymelove.taxonomy
WHERE taxonomy_name LIKE 'Apparel & Accessories > Clothing > Outerwear%'
ORDER BY taxonomy_path;
```

The table contains **393 total categories**, of which **390** appear in the current product dataset.

### P-adic Encoding Tables (umllr)

The umllr (Universal Machine Learning Linear Regression) module stores p-adic encodings of taxonomy paths for cross-validation. These tables enable calculating p-adic loss across different models.

#### `padjective.umllr_fold_metrics`

Stores the prime base (p) and metadata for each CV fold:

```sql
Table: padjective.umllr_fold_metrics
Columns:
  cv_fold    INTEGER PRIMARY KEY  -- Cross-validation fold number (0-4)
  loss       DOUBLE PRECISION     -- Total p-adic loss for this fold
  prime_base INTEGER              -- Prime p used for encoding (e.g., 83)
  max_digit  INTEGER              -- Largest digit in any taxonomy path (e.g., 80)
  updated_at TIMESTAMPTZ          -- Last update timestamp
```

The **prime base** is computed as the smallest prime greater than `max_digit`. All folds use the same prime since they encode the same taxonomy paths.

**Example query:**

```sql
SELECT cv_fold, prime_base, loss FROM padjective.umllr_fold_metrics;
-- cv_fold | prime_base | loss
-- --------|------------|--------
--    0    |     83     | 191.11
--    1    |     83     | 194.09
--    ...
```

#### `padjective.umllr_taxonomy_encodings`

Maps each taxonomy ID to its p-adic integer encoding for each CV fold:

```sql
Table: padjective.umllr_taxonomy_encodings
Columns:
  cv_fold       INTEGER              -- Cross-validation fold number
  taxonomy_id   TEXT                 -- Shopify taxonomy GID
  taxonomy_path TEXT                 -- taxonomy_code (e.g., "1.1.10.2.17")
  encoded_value NUMERIC              -- P-adic integer encoding (base p)
  updated_at    TIMESTAMPTZ          -- Last update timestamp
  PRIMARY KEY (cv_fold, taxonomy_id)
```

The **encoded_value** represents the taxonomy_code as a p-adic integer in base `prime_base`. For example, with taxonomy_code "1.1.10.2.17" and base 83:
```
encoded_value = 1×83⁰ + 1×83¹ + 10×83² + 2×83³ + 17×83⁴ = 808,004,005
```

**Example queries:**

```sql
-- Get the p-adic encoding for a specific taxonomy
SELECT cv_fold, encoded_value
FROM padjective.umllr_taxonomy_encodings
WHERE taxonomy_id = 'gid://shopify/TaxonomyCategory/aa-1-10-2-17';

-- Join with taxonomy names to see readable labels
SELECT
  e.cv_fold,
  t.taxonomy_name,
  e.taxonomy_path,
  e.encoded_value
FROM padjective.umllr_taxonomy_encodings e
JOIN cantbuymelove.taxonomy t ON e.taxonomy_id = t.taxonomy_id
WHERE e.cv_fold = 0 AND e.encoded_value != 0
ORDER BY e.encoded_value DESC
LIMIT 10;
```

**Using p-adic encodings for model evaluation:**

Any model that predicts `taxonomy_id` can calculate p-adic loss by:
1. Looking up the predicted taxonomy's `encoded_value` from `umllr_taxonomy_encodings`
2. Looking up the true taxonomy's `encoded_value`
3. Computing p-adic distance: `distance = prime_base^(-v)` where `v` is the p-adic valuation of `|predicted - true|`

This enables comparing taxonomy classifiers (logistic regression, neural networks) using the same p-adic metric as umllr.

### UMLLR Tag-Order Ablation

The cleanest paper ablation in this repo is to hold the greedy UMLLR regressor,
taxonomy encodings, and fold assignments fixed while changing only the tag
ordering heuristic. The trainer already supports:

- `battle_elo`: fold-local Elo ranking derived from the tag battles
- `frequency`: most frequent training tags first
- `mean_title_position`: average recorded title position in the training products
- `taxonomy_association`: tags that are most concentrated in a single taxonomy first
- `random`: seeded random control ordering

Run the ablation against a fixed benchmark snapshot when you want a stable
paper table:

```bash
uv run -m padjective.umllr --snapshot-ref paper --tag-order-strategy all --ablation-only
uv run -m padjective.umllr_ablation_report --snapshot-ref paper --format markdown
```

The first command persists all ablation runs into
`padjective.umllr_order_ablation_fold_metrics` and
`padjective.umllr_order_ablation_predictions` without overwriting the live
`umllr_*` tables used by the rolling website. The second command reads the
snapshot-tagged Postgres rows and emits a paper-ready summary with fold means,
fold standard deviations, and paired deltas versus the `battle_elo`
ordering. Pass `--format latex` if you want a LaTeX table instead of Markdown.

### Shared Benchmark Bundles

The benchmark notebook, dedicated benchmark HTML pages, and generated TeX
includes are all driven by the same `benchmark.json` bundle schema.

Build the rolling live bundle from Postgres:

```bash
uv run -m padjective.benchmark_bundle --out-dir build/benchmark_reports/latest
```

Build the fixed paper bundle from an exported snapshot:

```bash
uv run -m padjective.product_taxonomy_bench_export --snapshot paper --out-root build/product_taxonomy_bench --formats jsonl --no-gzip
uv run -m padjective.benchmark_bundle \
  --snapshot-dir build/product_taxonomy_bench/paper \
  --out-dir build/benchmark_reports/paper \
  --paper-tex-dir ../papers/padjective/sigir-ecom/generated
```

Render only the paper benchmark pages, using the same bundle the notebook and
paper consume:

```bash
uv run -m padjective.build_site \
  --output build/paper-benchmark-site \
  --benchmark-report-root build/benchmark_reports \
  --benchmark-views paper \
  --benchmark-only
```

## Method & Implementation

### tagbattle.py

(Named tagbattle in honour of kittenwar, a very addictive website.)

For each product retrieved from the Shopify stores Postgres database:
 - For each tag, we look to see if the tag is a subset of another tag in that same product line. e.g. if the tags are "chocolate,milk chocolate" then we ignore chocolate
   and only work with milk chocolate
 - If the title has a " - " in it (a dash surrounded by whitespace), then we pretend that we have two separate titles. e.g. If we have "Easter bunny - milk chocolate"
   then we don't say that milk comes after Easter. We just don't know the relationship between Easter and milk from this example
 - For each title:
   - We determine where the tag appears in the title, i.e. which character in the title is the start of the tag using a case-insensitive search. For many tags the answer
     will be "nowhere"
   - For each pair of tags which are somewhere in the title, record which one came first. Pretend that it's a competition, and record which tag won and which tag lost into
     the ``padjective.battles`` table (created on the ``pg_default`` tablespace)

The recorded battle outcome is directional: the tag that starts later in the
title is always stored as the ``winner_tag`` and the earlier one as the
``loser_tag``. If two tags start at the same character we skip the pair so that
we never arbitrarily credit the "leftmost" tag with a win.

### ranking.py

Use the choix library and the Postgres ``padjective.battles`` table from tagbattle.py to produce ranking tables for each tag and persist them in ``padjective.tag_rankings``.

### display.py

Creates text and HTML and images from the results of ranking.py. By default the
script prints the top ten tags to stdout. Use ``--rows`` to control how many
rows are printed (``0`` prints them all).

### tag_features.py

Utilities for extracting product tags as sparse feature matrices suitable for machine learning. This module:
- Streams products from the database
- Parses and normalizes tags
- Creates sparse matrices (product × tag) for efficient memory usage
- Optionally includes taxonomy labels for supervised learning

### taxonomy_classifier.py

Trains a parameter constrained logistic regression model to predict `taxonomy_id` from product tags. Features:
- Stratified cross-validation for evaluation
- Coefficient analysis to identify influential tags
- Postgres storage for model weights and metadata
- HTML reports visualizing tag importance

### taxonomy_pcnn_classifier.py

Trains a parameter constrained neural network (MLPClassifier) to predict `taxonomy_id` from product tags. Features:
- Configurable hidden layer architecture with parameter constraints for fair model comparison
- Early stopping to prevent overfitting
- Cross-validation evaluation
- Metadata storage and HTML reporting

### taxonomy_ulr_classifier.py

Trains an unconstrained logistic regression model with L1 (Lasso) regularization to predict `taxonomy_id` from product tags. Features:
- Uses ALL available tags (no max-tags constraint)
- L1 regularization to achieve sparsity and automatic feature selection
- Tracks the number of non-zero parameters after training
- Cross-validation evaluation with p-adic loss metrics

## Part-of-Speech Considerations

The pipeline currently works with whatever tags appear in the database and
does not attempt to decide whether a tag is an adjective, noun, or another
part of speech. All tags are normalised to uppercase and compared solely by
their character spans within each product title.

## Running the Pipeline

The project uses [uv](https://github.com/astral-sh/uv) for package management.
After installing ``uv`` you can run the whole analysis pipeline with the
defaults provided in the repository:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tag battle analysis (requires SHOPIFY_DB_DSN or DATABASE_URL)
uv run padjective/tagbattle.py
uv run padjective/ranking.py
uv run padjective/display.py

# Extract tag features as sparse matrix
uv run padjective/tag_features.py --output data/tags.npz --output-metadata data/products.csv

# Train taxonomy classifiers
uv run padjective/taxonomy_classifier.py \
    --results-schema padjective \
    --output-dir build/taxonomy_classifier

uv run padjective/taxonomy_pcnn_classifier.py \
    --model-database data/taxonomy_pcnn_classifier.sqlite \
    --output-dir build/taxonomy_pcnn_classifier \
    --hidden-layers "100,50"
```

This sequence:
1. Populates ``padjective.battles`` and ``padjective.tag_rankings`` in Postgres
2. Renders ``tag_rankings.html`` and ``tag_rankings.png`` locally
3. Extracts tag features to numpy sparse format
4. Trains both parameter constrained logistic regression and parameter constrained neural network models to predict taxonomy
5. Generates HTML reports visualizing model performance and tag coefficients

## Model Output

The taxonomy classifiers produce:

### Parameter Constrained Logistic Regression
* **Postgres schema** (`padjective.taxonomy_pclr_*` tables) containing:
  - Model metadata (samples, accuracy, F1, hierarchical loss, CV scores)
  - Class distribution snapshots and per-tag weight summaries
  - Per-taxonomy top-weighted tags and intercepts
* **HTML report** (`build/taxonomy_classifier/tag_coefficients.html`) visualizing:
  - Tags with largest absolute coefficients
  - Tags with largest summed coefficients across all taxonomies
  - Model performance metrics
* Ensure the tables exist by running
  `uv run padjective/taxonomy_classifier_schema.py --schema padjective`

### Parameter Constrained Neural Network
* **SQLite database** (`data/taxonomy_pcnn_classifier.sqlite`) containing:
  - Model metadata (architecture, accuracy, CV scores)
* **HTML report** (`build/taxonomy_pcnn_classifier/pcnn_report.html`) summarizing:
  - Network architecture
  - Training and cross-validation performance

### Complement Naive Bayes (removed)
The Complement Naive Bayes pipeline has been superseded by the parameter
constrained logistic regression workflow above. The old `padjective.taxonomy_nb_*`
tables and `padjective/taxonomy_nb_classifier.py` have been retired.

## Experiments & Evaluation

### Hold-out Experiments

To monitor how well the inferred rankings predict unseen tag orderings, use the
``padjective.experiments`` module. It manages a queue of randomised hold-out
tasks and stores the outcomes in ``holdout_tasks.sqlite`` by default.

```bash
# create (or extend) a task queue of 5,000 random splits
uv run -m padjective.experiments init --total 5000 --test-fraction 0.2

# execute up to 250 pending tasks against Postgres battle data
uv run -m padjective.experiments run \
    --dsn "$SHOPIFY_DB_DSN" \
    --tasks-db holdout_tasks.sqlite \
    --take 250

# show overall progress and mean accuracy
uv run -m padjective.experiments status --tasks-db holdout_tasks.sqlite
```

### Open Research Questions

- Can we formalise why p-adic losses prefer pure powers via tropical geometry or another analytic argument?
- How does training time compare against conventional (real-valued) optimisation baselines for the same tasks?
- Which qualitative examples best illustrate the ordering predictions to a reader who is unfamiliar with p-adic methods?

Answering those questions should make the experiment more compelling while keeping the focus on adjective ordering.

## Deprecated Features

### WordNet Synset Classification

The original synset classification workflow (`product_synsets.py` and `synset_classifier.py`)
has been replaced by the taxonomy-based approach. The codebase now uses the
`taxonomy_path` from the `cantbuymelove.taxonomy` table instead of WordNet synsets.
