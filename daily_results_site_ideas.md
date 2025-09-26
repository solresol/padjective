# Daily Tag Hierarchy Website Ideas

The experiment already produces rich ranking artifacts (CSV, HTML, PNG) for the sampled Shopify catalog. A daily website that publishes fresh results as the full dataset is processed can turn those artifacts into an evolving story about tag relationships. Below are ideas for structuring that experience.

## Hero Overview
- **Daily headline numbers:** Surface the count of titles processed, distinct tags seen, and the number of tag "battles" recorded for the most recent ingest.
- **Depth snapshot:** Show the current leaderboard of top tags by inferred depth with badges for notable movers compared to the prior day.
- **Pipeline health indicator:** Include a simple status light or log excerpt summarizing whether tagbattle, ranking, and display stages ran successfully overnight.

## Interactive Exploration
- **Depth timeline charts:** Plot how individual tags move up or down in rank over time, with tooltips linking back to representative product titles.
- **Component explorer:** Offer an interactive graph view (e.g., force-directed) for connected components discovered by `ranking.py`, highlighting which tags battle frequently.
- **Search & compare:** Provide a search box to pull up a tag's history, see its typical co-occurring tags, and compare it against another tag's trend.

## Daily Digest & Narrative
- **Automated changelog:** Generate a narrative summary (e.g., "Tag X gained 3 spots, entering the top 20") based on significant Elo shifts.
- **Anomaly callouts:** Flag tags with unusually large score swings or newly discovered relationships, potentially linking to the product titles that caused the change.
- **Depth distribution histograms:** Visualize how many tags fall into each inferred depth bucket to detect structural shifts in the hierarchy.

## Methodology & Transparency
- **Pipeline explanation:** Reuse concise descriptions of `tagbattle.py`, `ranking.py`, and `display.py` so visitors understand how ranks are produced.
- **Data freshness banner:** Note the timestamp of the dataset snapshot and clarify the sampling strategy versus the full corpus.
- **Reproducibility corner:** Link to the Git repository, document commands (e.g., `uv run padjective/tagbattle.py`), and describe how to rerun analyses locally.

## Engagement & Sharing
- **Download center:** Offer the latest CSV/HTML/PNG artifacts for researchers to reuse, along with archived historical bundles.
- **Newsletter signup:** Allow visitors to subscribe to a weekly recap summarizing the biggest shifts in tag hierarchy.
- **API endpoint teaser:** If feasible, expose a lightweight JSON feed of the current rankings so others can build derivative analyses.

These components combine daily operational transparency with deep exploratory tools, helping stakeholders watch the Shopify tag hierarchy evolve in near real time.
