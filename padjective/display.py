import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from psycopg import sql

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:
    from . import db


def generate_outputs(
    df: pd.DataFrame, html_path: Path, image_path: Path, rows: int = 10
) -> None:
    df_sorted = df.sort_values("score", ascending=False)

    # Print selected rows to stdout
    if rows <= 0:
        to_print = df_sorted
    else:
        to_print = df_sorted.head(rows)
    print(to_print.to_string(index=False))

    # Save HTML table
    df_sorted.to_html(html_path, index=False)

    # Plot top 20 tags
    top = df_sorted.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(top["tag"], top["score"], color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Score")
    plt.ylabel("Tag")
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()


def load_rankings(conn, schema: str, table: str) -> pd.DataFrame:
    query = sql.SQL(
        "SELECT tag, component, score FROM {schema}.{table} ORDER BY component, score DESC"
    ).format(schema=sql.Identifier(schema), table=sql.Identifier(table))
    query_text = query.as_string(conn)
    return pd.read_sql_query(query_text, conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display tag ranking results from Postgres.")
    parser.add_argument(
        "--dsn",
        help="Postgres DSN. Defaults to SHOPIFY_DB_DSN or DATABASE_URL if unset.",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing the rankings table.",
    )
    parser.add_argument(
        "--table",
        default="tag_rankings",
        help="Table name holding the rankings data.",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=Path("tag_rankings.html"),
        help="HTML file to write table to",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("tag_rankings.png"),
        help="Image file for plot",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of rows to print to stdout (0 for all)",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    df = load_rankings(conn, args.schema, args.table)
    generate_outputs(df, args.html, args.image, args.rows)
    conn.close()


if __name__ == "__main__":
    main()
