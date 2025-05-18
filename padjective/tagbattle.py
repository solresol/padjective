"""Utilities for ranking product tags by their position in titles."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Iterable, List, Dict


def filter_nested_tags(tags: Iterable[str]) -> List[str]:
    """Remove tags that are substrings of other tags.

    Tags are compared case-insensitively and returned in their original order
    without duplicates.
    """
    unique: List[str] = []
    seen = set()
    for tag in tags:
        tag = tag.strip()
        if not tag or tag.lower() in seen:
            continue
        unique.append(tag)
        seen.add(tag.lower())

    filtered: List[str] = []
    for tag in unique:
        tag_lower = tag.lower()
        if any(
            tag_lower != other.lower() and tag_lower in other.lower()
            for other in unique
        ):
            continue
        filtered.append(tag)
    return filtered


def split_title(title: str) -> List[str]:
    """Split a product title on ``" - "`` if present."""
    return [part.strip() for part in title.split(" - ")]


def tag_positions(title: str, tags: Iterable[str]) -> Dict[str, int]:
    """Return the start index of each tag found in ``title``.

    The search is case-insensitive and only the first occurrence is recorded.
    Tags that are not present are omitted from the result.
    """
    lower_title = title.lower()
    positions: Dict[str, int] = {}
    for tag in tags:
        idx = lower_title.find(tag.lower())
        if idx != -1:
            positions[tag] = idx
    return positions


def record_battles(positions: Dict[str, int], cursor: sqlite3.Cursor) -> None:
    """Record pairwise ordering of tags into the ``battles`` table."""
    tags = list(positions.keys())
    for i in range(len(tags)):
        for j in range(i + 1, len(tags)):
            t1, t2 = tags[i], tags[j]
            if positions[t1] == positions[t2]:
                continue
            if positions[t1] < positions[t2]:
                winner, loser = t1, t2
            else:
                winner, loser = t2, t1
            cursor.execute(
                "INSERT INTO battles (winner_tag, loser_tag) VALUES (?, ?)",
                (winner, loser),
            )


def process_product(title: str, tag_string: str, cursor: sqlite3.Cursor) -> None:
    """Process a single product title and tag list."""
    tags = [t.strip() for t in tag_string.split(",") if t.strip()]
    tags = filter_nested_tags(tags)
    for part in split_title(title):
        positions = tag_positions(part, tags)
        if len(positions) < 2:
            continue
        record_battles(positions, cursor)


def process_csv(csv_path: Path, sqlite_path: Path) -> None:
    """Process ``csv_path`` and store results in ``sqlite_path``."""
    conn = sqlite3.connect(str(sqlite_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS battles (winner_tag TEXT, loser_tag TEXT)"
    )
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            process_product(row["title"], row.get("tags", ""), cur)
    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tag ordering within product titles."
    )
    parser.add_argument(
        "csv",
        type=Path,
        help="CSV file containing 'title' and 'tags' columns",
    )
    parser.add_argument(
        "database", type=Path, help="SQLite database to store pair results"
    )
    args = parser.parse_args()
    process_csv(args.csv, args.database)


if __name__ == "__main__":
    main()
