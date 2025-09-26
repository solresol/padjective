"""Build a static website showcasing the latest tag ranking results."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from . import display, ranking, tagbattle


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _collect_tag_stats(csv_path: Path) -> Dict[str, int]:
    total_products = 0
    unique_tags: set[str] = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_products += 1
            tags = row.get("tags", "")
            for tag in tags.split(","):
                tag = tag.strip()
                if tag:
                    unique_tags.add(tag.upper())
    return {"products": total_products, "unique_tags": len(unique_tags)}


def _count_battles(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM battles")
        (count,) = cursor.fetchone()
        return int(count)
    finally:
        conn.close()


def _write_sql_dump(db_path: Path, dump_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        with dump_path.open("w", encoding="utf-8") as dump_file:
            for line in conn.iterdump():
                dump_file.write(f"{line}\n")
    finally:
        conn.close()


def _build_index_html(
    output_dir: Path,
    stats: Dict[str, int],
    leaderboard: pd.DataFrame,
    chart_path: Path,
    artifact_links: Dict[str, Path],
    source_csv: Path,
) -> None:
    top_table = leaderboard.head(20).to_html(index=False, classes="leaderboard")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    downloads_list_items = "\n".join(
        f'<li><a href="{path.relative_to(output_dir).as_posix()}">{label}</a></li>'
        for label, path in artifact_links.items()
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Padjective Tag Hierarchy</title>
  <link rel=\"stylesheet\" href=\"assets/styles.css\" />
</head>
<body>
  <header class=\"hero\">
    <h1>Padjective Tag Hierarchy</h1>
    <p class=\"tagline\">Daily insights into how Shopify product tags outrank one another.</p>
    <p class=\"timestamp\">Last updated {generated}</p>
  </header>

  <section class=\"metrics\">
    <div class=\"metric\">
      <span class=\"value\">{stats['products']:,}</span>
      <span class=\"label\">Products analysed</span>
    </div>
    <div class=\"metric\">
      <span class=\"value\">{stats['unique_tags']:,}</span>
      <span class=\"label\">Distinct tags observed</span>
    </div>
    <div class=\"metric\">
      <span class=\"value\">{stats['battles']:,}</span>
      <span class=\"label\">Tag battles recorded</span>
    </div>
  </section>

  <section class=\"leaderboard-section\">
    <div class=\"leaderboard-text\">
      <h2>Leaderboard</h2>
      <p>The Elo-inspired model favours tags that consistently appear earlier in product titles when paired with others. Here are the current top contenders.</p>
    </div>
    <div class=\"leaderboard-table\">
      {top_table}
    </div>
    <figure class=\"chart\">
      <img src=\"assets/{chart_path.name}\" alt=\"Top tags bar chart\" />
      <figcaption>Top 20 tags by inferred depth.</figcaption>
    </figure>
  </section>

  <section class=\"methodology\">
    <h2>How the rankings work</h2>
    <ol>
      <li><strong>Battle generation</strong> &mdash; <code>tagbattle.py</code> scans each product title, comparing the order of every pair of tags.</li>
      <li><strong>Elo-style scoring</strong> &mdash; <code>ranking.py</code> treats each ordering as a battle, rewarding tags that appear closer to the start.</li>
      <li><strong>Visualisation</strong> &mdash; <code>display.py</code> turns the rankings into shareable tables and charts.</li>
    </ol>
    <p>Tags are grouped by connected component so isolated tag families get their own podium.</p>
  </section>

  <section class=\"downloads\">
    <h2>Download the data</h2>
    <ul>
      {downloads_list_items}
    </ul>
    <p>Historical SQL dumps are synchronised to <a href=\"https://datadumps.ifost.org.au/padjective/\">datadumps.ifost.org.au</a>.</p>
  </section>

  <footer>
    <p>Built from <code>{source_csv.name}</code>. Source available on <a href=\"https://github.com/IFost-Sydney-Uni/padjective\">GitHub</a>.</p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html, encoding="utf-8")


def build_site(csv_path: Path, output_dir: Path) -> Dict[str, Any]:
    csv_path = csv_path.resolve()
    _ensure_clean_directory(output_dir)

    assets_dir = output_dir / "assets"
    downloads_dir = output_dir / "downloads"
    datadumps_dir = output_dir / "datadumps"
    for path in (assets_dir, downloads_dir, datadumps_dir):
        path.mkdir(parents=True, exist_ok=True)

    db_path = downloads_dir / "battles.sqlite"
    if db_path.exists():
        db_path.unlink()

    tagbattle.process_csv(csv_path, db_path)

    pairs = ranking.load_pairs(db_path)
    leaderboard = ranking.compute_rankings(pairs)

    rankings_csv = downloads_dir / "tag_rankings.csv"
    ranking.save_rankings(leaderboard, rankings_csv)

    rankings_html = downloads_dir / "tag_rankings_table.html"
    chart_path = assets_dir / "top_tags.png"
    display.generate_outputs(rankings_csv, rankings_html, chart_path, rows=20)

    stats = _collect_tag_stats(csv_path)
    stats["battles"] = _count_battles(db_path)
    stats["components"] = int(leaderboard["component"].nunique()) if not leaderboard.empty else 0

    dump_path = datadumps_dir / "battles.sql"
    _write_sql_dump(db_path, dump_path)

    stylesheet = assets_dir / "styles.css"
    stylesheet.write_text(
        """body {font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #222; background: #f7f7fb;}
header.hero {background: linear-gradient(135deg, #0b6ce3, #66c4ff); color: white; padding: 3rem 1.5rem; text-align: center;}
header.hero h1 {margin-bottom: 0.5rem; font-size: 2.5rem;}
header.hero .tagline {margin: 0 auto 1rem; max-width: 50rem; font-size: 1.1rem;}
header.hero .timestamp {margin: 0; font-style: italic; opacity: 0.85;}
section {padding: 2rem 1.5rem; max-width: 70rem; margin: 0 auto;}
.metrics {display: flex; flex-wrap: wrap; gap: 1rem; justify-content: center;}
.metric {background: white; border-radius: 0.75rem; padding: 1.5rem; flex: 1 1 12rem; text-align: center; box-shadow: 0 12px 30px rgba(11, 108, 227, 0.1);}
.metric .value {display: block; font-size: 2rem; font-weight: 700; color: #0b6ce3;}
.metric .label {font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.08em; color: #555;}
.leaderboard-section {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);}
.leaderboard-section .leaderboard-text {padding-bottom: 1rem;}
.leaderboard-table {overflow-x: auto; padding: 0 1rem 2rem;}
table.leaderboard {border-collapse: collapse; width: 100%; background: white;}
table.leaderboard th, table.leaderboard td {padding: 0.75rem 1rem; border-bottom: 1px solid #e5e7eb; text-align: left;}
table.leaderboard thead {background: #f1f5f9;}
table.leaderboard tbody tr:nth-child(even) {background: #f8fafc;}
.chart {text-align: center; padding: 0 1rem 2rem;}
.chart img {max-width: 100%; height: auto; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);}
.methodology {background: #eef2ff; border-radius: 1rem;}
.methodology ol {line-height: 1.7;}
.downloads ul {list-style: none; padding: 0;}
.downloads li {margin: 0.5rem 0;}
.downloads a {color: #0b6ce3; text-decoration: none; font-weight: 600;}
.downloads a:hover {text-decoration: underline;}
footer {text-align: center; padding: 2rem 1.5rem 3rem; color: #6b7280;}
@media (max-width: 700px) {.metrics {flex-direction: column;} header.hero {padding: 2.5rem 1rem;} header.hero h1 {font-size: 2rem;}}
""",
        encoding="utf-8",
    )

    artifact_links: Dict[str, Path] = {
        "Tag rankings (CSV)": rankings_csv,
        "Tag rankings table (HTML)": rankings_html,
        "Tag battles database": db_path,
        "SQL dump of battles": dump_path,
        "Top tags chart": chart_path,
    }

    _build_index_html(output_dir, stats, leaderboard, chart_path, artifact_links, csv_path)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "stats": stats,
        "artifacts": {label: str(path.relative_to(output_dir)) for label, path in artifact_links.items()},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Padjective website")
    parser.add_argument("--csv", type=Path, required=True, help="Path to the products CSV file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/site"),
        help="Directory where the static site should be written",
    )
    args = parser.parse_args()

    build_site(args.csv, args.output)


if __name__ == "__main__":
    main()
