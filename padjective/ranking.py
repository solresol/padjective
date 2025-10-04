import argparse
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from psycopg import sql

from . import db


def _elo_scores(num_tags: int, data: List[Tuple[int, int]], k: float = 32.0) -> List[float]:
    """Return Elo ratings for ``num_tags`` competitors."""

    ratings = [0.0] * num_tags
    for winner, loser in data:
        r_w = ratings[winner]
        r_l = ratings[loser]
        expected_w = 1 / (1 + 10 ** ((r_l - r_w) / 400))
        expected_l = 1 - expected_w
        ratings[winner] = r_w + k * (1 - expected_w)
        ratings[loser] = r_l + k * (0 - expected_l)
    return ratings


def load_pairs(conn, schema: str) -> List[Tuple[str, str]]:
    """Load winner/loser pairs from Postgres."""

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT winner_tag, loser_tag FROM {schema}.battles").format(
                schema=sql.Identifier(schema)
            )
        )
        rows = cur.fetchall()
    return [(winner, loser) for winner, loser in rows]


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
    """Return ratings using a memory-friendly Elo algorithm."""
    if num_tags == 1:
        return [0.0]
    return _elo_scores(num_tags, data)


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


def ensure_output_table(conn, schema: str, table: str) -> None:
    columns = (
        "tag TEXT PRIMARY KEY",
        "component INTEGER NOT NULL",
        "score DOUBLE PRECISION NOT NULL",
        "updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    )
    indexes: Sequence[sql.SQL] = ()
    db.ensure_table(conn, schema, table, columns, indexes)


def save_rankings(conn, schema: str, table: str, df: pd.DataFrame) -> None:
    ensure_output_table(conn, schema, table)
    db.truncate_table(conn, schema, table)
    records = list(df[["tag", "component", "score"]].itertuples(index=False, name=None))
    if not records:
        return

    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "INSERT INTO {schema}.{table} (tag, component, score) VALUES (%s, %s, %s)"
            ).format(schema=sql.Identifier(schema), table=sql.Identifier(table)),
            records,
        )
    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tag rankings from Postgres battle data.")
    parser.add_argument(
        "--dsn",
        help="Postgres DSN. Defaults to SHOPIFY_DB_DSN or DATABASE_URL if unset.",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing battle inputs and the output table.",
    )
    parser.add_argument(
        "--output-table",
        default="tag_rankings",
        help="Name of the table (within schema) to store rankings in.",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    pairs = load_pairs(conn, args.schema)
    df = compute_rankings(pairs)
    save_rankings(conn, args.schema, args.output_table, df)
    conn.close()


if __name__ == "__main__":
    main()
