-- Tag battles schema and table
-- This table stores pairwise comparisons of tags based on their position in product titles.

CREATE SCHEMA IF NOT EXISTS padjective;

CREATE TABLE IF NOT EXISTS padjective.battles (
    product_id BIGINT,
    winner_tag TEXT NOT NULL,
    loser_tag TEXT NOT NULL,
    cv_fold INTEGER,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS padjective_battles_winner_idx
    ON padjective.battles (winner_tag);

CREATE INDEX IF NOT EXISTS padjective_battles_loser_idx
    ON padjective.battles (loser_tag);
