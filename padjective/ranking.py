import argparse
import sqlite3
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import choix


def load_pairs(db_path: Path) -> List[Tuple[str, str]]:
    """Load winner/loser pairs from the database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT winner_tag, loser_tag FROM battles")
    pairs = cur.fetchall()
    conn.close()
    return pairs


def compute_rankings(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Return a DataFrame of tags ranked by skill."""
    tags = sorted({t for pair in pairs for t in pair})
    tag_to_id = {tag: i for i, tag in enumerate(tags)}
    data = [(tag_to_id[w], tag_to_id[l]) for w, l in pairs]
    # Bradley-Terry using iterative Luce spectral ranking
    scores = choix.ilsr_pairwise(len(tags), data)
    df = pd.DataFrame({"tag": tags, "score": scores})
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def save_rankings(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tag rankings from battle data.")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("battles.sqlite"),
        help="SQLite database produced by tagbattle.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tag_rankings.csv"),
        help="CSV file to write tag rankings to",
    )
    args = parser.parse_args()
    pairs = load_pairs(args.database)
    df = compute_rankings(pairs)
    save_rankings(df, args.output)


if __name__ == "__main__":
    main()
