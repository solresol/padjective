# padjective

Calculate p-adic adjective embeddings

# Purpose

We want to get a hierarchy of tags: for any given pair of tags, which one is more likely to appear first in a product title? Then we want to identify "equivalent depth"
tags where they generally appear at the same depth, and ultimately assign an integer depth to every tag.

# Method

## tagbattle.py

(Named tagbattle in honour of kittenwar, a very addictive website.)

For each product:
 - For each tag, we look to see if the tag is a subset of another tag in that same product line. e.g. if the tags are "chocolate,milk chocolate" then we ignore chocolate
   and only work with milk chocolate
 - If the title has a " - " in it (a dash surrounded by whitespace), then we pretend that we have two separate titles. e.g. If we have "Easter bunny - milk chocolate" 
   then we don't say that milk comes after Easter. We just don't know the relationship between Easter and milk from this example
 - For each title:
   - We determine where the tag appears in the title, i.e. which character in the title is the start of the tag using a case-insensitive search. For many tags the answer
     will be "nowhere"
   - For each pair of tags which are somewhere in the title, record which one came first. Pretend that it's a competition, and record which tag won and which tag lost into
     a sqlite database table

## ranking.py
 
Use the choix library and the sqlite database from tagbattle.py to produce ranking tables for each tag.

## display.py

Creates text and HTML and images from the results of ranking.py. By default the
script prints the top ten tags to stdout. Use ``--rows`` to control how many
rows are printed (``0`` prints them all).

## Running the pipeline

The project uses [uv](https://github.com/astral-sh/uv) for package management.
After installing ``uv`` you can run the whole analysis pipeline with the
defaults provided in the repository:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# process the sample data
uv run padjective/tagbattle.py
uv run padjective/ranking.py
uv run padjective/display.py
```

This will create ``battles.sqlite`` from ``products_point_one_percent_sample.csv``,
produce ``tag_rankings.csv`` and render ``tag_rankings.html`` and
``tag_rankings.png``.

    
