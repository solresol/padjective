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

We predict product taxonomy from tags using machine learning approaches, including logistic regression and neural networks. This allows automatic categorization of products based on their tag combinations.

## Database Schema

The project uses the simplified `cantbuymelove` schema:
- `cantbuymelove.product` - Product table with integer primary key (`id`)
- `cantbuymelove.product_taxonomy` - Links products to taxonomies via `product_id`
- `cantbuymelove.taxonomy` - Taxonomy definitions with `taxonomy_id`, `taxonomy_name`, and `taxonomy_path`

Products are joined to `public.product_details` to extract tags from the JSONB `product_detail` field.

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

Trains a multinomial logistic regression model to predict `taxonomy_id` from product tags. Features:
- Stratified cross-validation for evaluation
- Coefficient analysis to identify influential tags
- SQLite storage for model weights and metadata
- HTML reports visualizing tag importance

### taxonomy_nn_classifier.py

Trains a neural network (MLPClassifier) to predict `taxonomy_id` from product tags. Features:
- Configurable hidden layer architecture
- Early stopping to prevent overfitting
- Cross-validation evaluation
- Metadata storage and HTML reporting

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

uv run padjective/taxonomy_nn_classifier.py \
    --model-database data/taxonomy_nn_classifier.sqlite \
    --output-dir build/taxonomy_nn_classifier \
    --hidden-layers "100,50"
```

This sequence:
1. Populates ``padjective.battles`` and ``padjective.tag_rankings`` in Postgres
2. Renders ``tag_rankings.html`` and ``tag_rankings.png`` locally
3. Extracts tag features to numpy sparse format
4. Trains both logistic regression and neural network models to predict taxonomy
5. Generates HTML reports visualizing model performance and tag coefficients

## Model Output

The taxonomy classifiers produce:

### Logistic Regression
* **Postgres schema** (`padjective.taxonomy_lr_*` tables) containing:
  - Model metadata (samples, accuracy, F1, hierarchical loss, CV scores)
  - Class distribution snapshots and per-tag weight summaries
  - Per-taxonomy top-weighted tags and intercepts
* **HTML report** (`build/taxonomy_classifier/tag_coefficients.html`) visualizing:
  - Tags with largest absolute coefficients
  - Tags with largest summed coefficients across all taxonomies
  - Model performance metrics
* Ensure the tables exist by running
  `uv run padjective/taxonomy_classifier_schema.py --schema padjective`

### Neural Network
* **SQLite database** (`data/taxonomy_nn_classifier.sqlite`) containing:
  - Model metadata (architecture, accuracy, CV scores)
* **HTML report** (`build/taxonomy_nn_classifier/nn_report.html`) summarizing:
  - Network architecture
  - Training and cross-validation performance

### Complement Naive Bayes (removed)
The Complement Naive Bayes pipeline has been superseded by the logistic
regression workflow above. The old `padjective.taxonomy_nb_*` tables and
`padjective/taxonomy_nb_classifier.py` have been retired.

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
