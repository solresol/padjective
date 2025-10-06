# padjective

Calculate p-adic adjective embeddings

# Purpose

We want to get a hierarchy of tags: for any given pair of tags, which one is more likely to appear first in a product title? Then we want to identify "equivalent depth"
tags where they generally appear at the same depth, and ultimately assign an integer depth to every tag.

# Method

## tagbattle.py

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

## ranking.py
 
Use the choix library and the Postgres ``padjective.battles`` table from tagbattle.py to produce ranking tables for each tag and persist them in ``padjective.tag_rankings``.

## display.py

Creates text and HTML and images from the results of ranking.py. By default the
script prints the top ten tags to stdout. Use ``--rows`` to control how many
rows are printed (``0`` prints them all).

## Part-of-speech considerations

The pipeline currently works with whatever tags appear in the CSV input and
does not attempt to decide whether a tag is an adjective, noun, or another
part of speech. All tags are normalised to uppercase and compared solely by
their character spans within each product title. If you need linguistic
annotations—such as mapping the head noun of a title to a WordNet synset—you'll
need to run an additional pass outside this repository. That could be a custom
script, a traditional NLP library, or an LLM that labels each product before
feeding the results into the existing ranking workflow.

## Running the pipeline

The project uses [uv](https://github.com/astral-sh/uv) for package management.
After installing ``uv`` you can run the whole analysis pipeline with the
defaults provided in the repository:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# process the Shopify data (requires SHOPIFY_DB_DSN or DATABASE_URL)
uv run padjective/tagbattle.py --taxonomy-table cantbuymelove.product_taxonony --product-view cantbuymelove.product
uv run padjective/ranking.py
uv run padjective/display.py
```

This sequence populates ``padjective.battles`` and ``padjective.tag_rankings``
inside the Shopify stores Postgres database and renders ``tag_rankings.html``
and ``tag_rankings.png`` locally.

## Training a synset classifier

Once ``padjective.product_synsets`` has populated ``product_synsets.sqlite`` you
can train a simple logistic regression model that predicts WordNet synsets from
tags only. The script now evaluates the classifier with stratified
cross-validation, stores the learned weights in SQLite, and produces a static
HTML report that highlights which tags carry the largest coefficients across all
synset classes.

> **Project note**
>
> We're currently blocked on the synset identification phase. Each attempt to
> label products with ``padjective/product_synsets.py`` inserts a literal "no can
> do" value into the database instead of the expected synset, so the downstream
> classifier training cannot proceed. It may be worth extracting this script into
> a standalone repository so we can iterate on the problem independently of the
> rest of the pipeline.

```bash
uv run padjective/synset_classifier.py \
    --database data/product_synsets.sqlite \
    --model-database data/synset_classifier.sqlite \
    --output-dir build/synset_classifier
```

The output directory contains:

* ``tag_coefficients.html`` – a standalone webpage visualising the tags with the
  largest weights (by both maximum and summed absolute coefficient values).

The ``data/synset_classifier.sqlite`` database captures the trained model
metadata, cross-validation scores, per-tag synset weights, and the HTML report's
summary data in a structured format for downstream use.

## Hold-out experiments

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

    
