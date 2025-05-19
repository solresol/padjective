import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def generate_outputs(
    csv_path: Path, html_path: Path, image_path: Path, rows: int = 10
) -> None:
    df = pd.read_csv(csv_path)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Display tag ranking results.")
    parser.add_argument(
        "--rankings",
        type=Path,
        default=Path("tag_rankings.csv"),
        help="CSV file produced by ranking.py",
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
    generate_outputs(args.rankings, args.html, args.image, args.rows)


if __name__ == "__main__":
    main()
