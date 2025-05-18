import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

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


def _connected_components(graph: Dict[str, List[str]]) -> List[List[str]]:
    """Return connected components of an undirected graph."""
    components: List[List[str]] = []
    visited: set[str] = set()
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp: List[str] = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for neighbor in graph[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(comp)
    return components


def _safe_ilsr(num_tags: int, data: List[Tuple[int, int]]) -> List[float]:
    """Return ILSR scores, falling back to win ratios if necessary."""
    if num_tags == 1:
        return [0.0]
    try:
        return list(choix.ilsr_pairwise(num_tags, data))
    except (ValueError, RuntimeError):
        wins = [0] * num_tags
        games = [0] * num_tags
        for w, l in data:
            wins[w] += 1
            games[w] += 1
            games[l] += 1
        return [wins[i] / games[i] if games[i] else 0.0 for i in range(num_tags)]


def compute_rankings(pairs: List[Tuple[str, str]]) -> pd.DataFrame:
    """Return a DataFrame of tags ranked by skill.

    Handles disjoint sets of tags by ranking each connected component
    separately.
    """
    if not pairs:
        return pd.DataFrame(columns=["tag", "component", "score"])

    tags = sorted({t for pair in pairs for t in pair})

    graph: Dict[str, List[str]] = {t: [] for t in tags}
    for w, l in pairs:
        graph[w].append(l)
        graph[l].append(w)

    components = _connected_components(graph)

    records = []
    for comp_id, comp_tags in enumerate(components):
        tag_to_id = {tag: i for i, tag in enumerate(comp_tags)}
        comp_pairs = [
            (tag_to_id[w], tag_to_id[l])
            for w, l in pairs
            if w in tag_to_id and l in tag_to_id
        ]
        scores = _safe_ilsr(len(comp_tags), comp_pairs)
        for tag, score in zip(comp_tags, scores):
            records.append({"tag": tag, "component": comp_id, "score": float(score)})

    df = pd.DataFrame(records)
    return df.sort_values(["component", "score"], ascending=[True, False]).reset_index(drop=True)


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
