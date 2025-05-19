You have a sample data file to work with: `products_point_one_percent_sample.csv`. This is about 5MB, and has 50,000 products in it. It has two columns: title and tags. The tags
are comma-separated.

The real goal is to work with a full data set (products.csv) which is 1000 times bigger. This isn't in git and will get run on a dedicated computational box.

---

If CLAUDE.md is present, read it.

Run the unit tests with `uv run -m pytest -q`.
