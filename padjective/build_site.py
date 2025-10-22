"""Build a static website showcasing tag rankings and taxonomy progress."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from psycopg.rows import dict_row

from . import db, display, experiments, ranking, tagbattle


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


def _count_battles(pairs: Sequence[Tuple[str, str]]) -> int:
    return len(pairs)


def _write_sql_dump(pairs: Sequence[Tuple[str, str]], dump_path: Path, schema: str) -> None:
    with dump_path.open("w", encoding="utf-8") as dump_file:
        dump_file.write("BEGIN;\n")
        for winner, loser in pairs:
            safe_winner = winner.replace("'", "''")
            safe_loser = loser.replace("'", "''")
            dump_file.write(
                f"INSERT INTO {schema}.battles (winner_tag, loser_tag) VALUES ('{safe_winner}', '{safe_loser}');\n"
            )
        dump_file.write("COMMIT;\n")




def _build_index_html(
    output_dir: Path,
    stats: Dict[str, int],
    leaderboard: pd.DataFrame,
    chart_path: Path,
    artifact_links: Dict[str, Path],
    experiments_summary: Optional[Dict[str, Any]] = None,
    taxonomy_summary: Optional[Dict[str, Any]] = None,
) -> None:
    top_table = leaderboard.head(20).to_html(index=False, classes="leaderboard")
    bottom_table = (
        leaderboard.sort_values("score", ascending=True)
        .head(20)
        .to_html(index=False, classes=["leaderboard", "leaderboard-bottom-table"])
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    downloads_list_items = "\n".join(
        f'<li><a href="{path.relative_to(output_dir).as_posix()}">{label}</a></li>'
        for label, path in artifact_links.items()
    )

    experiments_block = ""
    if experiments_summary and experiments_summary.get("total"):
        counts = experiments_summary.get("counts", {})
        completed = experiments_summary.get("completed_tasks", 0)
        total = experiments_summary.get("total", 0)
        mean_accuracy = experiments_summary.get("mean_accuracy")
        mean_coverage = experiments_summary.get("mean_coverage")
        recent_rows = experiments_summary.get("recent", [])

        mean_accuracy_text = (
            f"{mean_accuracy * 100:.2f}%" if mean_accuracy is not None else "n/a"
        )
        mean_coverage_text = (
            f"{mean_coverage * 100:.2f}%" if mean_coverage is not None else "n/a"
        )
        test_fraction = experiments_summary.get("test_fraction") or 0.0

        recent_items_list = []
        for row in recent_rows:
            accuracy_cell = (
                f"<td>{row['accuracy'] * 100:.2f}%</td>"
                if row.get("accuracy") is not None
                else "<td>n/a</td>"
            )
            coverage_cell = (
                f"<td>{row['coverage'] * 100:.2f}%</td>"
                if row.get("coverage") is not None
                else "<td>n/a</td>"
            )
            recent_items_list.append(
                "<tr>"
                f"<td>{row['id']}</td>"
                f"<td>{row['seed']}</td>"
                f"<td>{row['evaluated_pairs'] or 0}</td>"
                f"{accuracy_cell}"
                f"{coverage_cell}"
                f"<td>{row['completed_at'] or ''}</td>"
                "</tr>"
            )
        recent_items = "\n".join(recent_items_list)
        if not recent_items:
            recent_items = '<tr><td colspan="6">No completed evaluations yet.</td></tr>'

        experiments_block = f"""
  <section class="experiments">
    <h2>Hold-out experiments</h2>
    <p>We randomly reserve {test_fraction:.0%} of recorded tag battles and check whether the rankings predict the correct ordering.</p>
    <div class="experiments-metrics">
      <div class="metric">
        <span class="value">{completed:,} / {total:,}</span>
        <span class="label">Tasks completed</span>
      </div>
      <div class="metric">
        <span class="value">{counts.get('pending', 0):,}</span>
        <span class="label">Pending tasks</span>
      </div>
      <div class="metric">
        <span class="value">{counts.get('running', 0):,}</span>
        <span class="label">Running tasks</span>
      </div>
      <div class="metric">
        <span class="value">{counts.get('error', 0):,}</span>
        <span class="label">Errors</span>
      </div>
    </div>
    <p class="experiments-accuracy">Average accuracy across completed tasks: {mean_accuracy_text} (coverage {mean_coverage_text}).</p>
    <table class="experiments-table">
      <thead>
        <tr>
          <th>Task</th>
          <th>Seed</th>
          <th>Evaluated battles</th>
          <th>Accuracy</th>
          <th>Coverage</th>
          <th>Completed</th>
        </tr>
      </thead>
      <tbody>
        {recent_items}
      </tbody>
    </table>
  </section>
"""

    taxonomy_section = ""
    if taxonomy_summary:
        stats_block = taxonomy_summary.get("stats", {})
        top_taxonomies = taxonomy_summary.get("taxonomy_priors", [])[:10]
        top_tags_rows = taxonomy_summary.get("top_tags", [])[:15]
        trained_at = taxonomy_summary.get("trained_at")

        summary_items = []
        samples = stats_block.get("samples")
        if samples is not None:
            summary_items.append(f"<li><strong>Samples:</strong> {samples:,}</li>")
        taxonomies_count = stats_block.get("taxonomies")
        if taxonomies_count is not None:
            summary_items.append(
                f"<li><strong>Taxonomy classes:</strong> {taxonomies_count:,}</li>"
            )
        accuracy = stats_block.get("training_accuracy")
        if accuracy is not None:
            summary_items.append(
                f"<li><strong>Training accuracy:</strong> {accuracy:.3f}</li>"
            )
        cv_info = stats_block.get("cross_validation") or {}
        if cv_info:
            mean = cv_info.get("mean_accuracy")
            std = cv_info.get("std_accuracy")
            folds = cv_info.get("folds")
            if mean is not None and folds:
                summary_items.append(
                    "<li><strong>Cross-validated accuracy:</strong> "
                    f"{mean:.3f}"
                    + (f" ± {std:.3f}" if std is not None else "")
                    + f" ({folds} folds)</li>"
                )
        if trained_at:
            summary_items.append(
                f"<li><strong>Trained:</strong> {html.escape(trained_at)}</li>"
            )

        summary_list = '<ul class="taxonomy-stats">' + "".join(summary_items) + "</ul>"

        taxonomy_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.get('taxonomy_id') or '')}</td>"
            f"<td>{html.escape(row.get('taxonomy_path') or 'Unknown')}</td>"
            f"<td>{row.get('prior', 0.0) * 100:.2f}%</td>"
            "</tr>"
            for row in top_taxonomies
        )
        taxonomy_body = taxonomy_rows or '<tr><td colspan="3">No taxonomy data</td></tr>'
        taxonomy_table = (
            '<table class="taxonomy-table">'
            '<thead><tr><th>Taxonomy ID</th><th>Path</th><th>Estimated share</th></tr></thead>'
            f"<tbody>{taxonomy_body}</tbody>"
            "</table>"
        )

        tag_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.get('tag') or '')}</td>"
            f"<td>{html.escape(row.get('top_taxonomy_id') or '')}</td>"
            f"<td>{html.escape(row.get('top_taxonomy_path') or 'Unknown')}</td>"
            f"<td>{row.get('probability', 0.0) * 100:.2f}%</td>"
            f"<td>{row.get('margin', 0.0):.3f}</td>"
            "</tr>"
            for row in top_tags_rows
        )
        tag_body = tag_rows or '<tr><td colspan="5">No tag signals available</td></tr>'
        tag_table = (
            '<table class="tag-taxonomy-table">'
            '<thead><tr><th>Tag</th><th>Taxonomy ID</th><th>Path</th><th>Association</th><th>Margin</th></tr></thead>'
            f"<tbody>{tag_body}</tbody>"
            "</table>"
        )

        taxonomy_section = f"""
  <section class="taxonomy-classifier">
    <h2>Shopify taxonomy classification</h2>
    <p>We train a Complement Naive Bayes model on Shopify tags to predict taxonomy IDs.</p>
    {summary_list}
    <div class="taxonomy-layout">
      <div class="taxonomy-card">
        <h3>Most common taxonomies</h3>
        {taxonomy_table}
      </div>
      <div class="taxonomy-card">
        <h3>Strongest tag signals</h3>
        {tag_table}
      </div>
    </div>
  </section>
"""

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Padjective Tag Hierarchy</title>
  <link rel="stylesheet" href="assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Padjective Tag Hierarchy</h1>
    <p class="tagline">Daily insights into how Shopify product tags outrank one another.</p>
    <p class="timestamp">Last updated {generated}</p>
  </header>

  <section class="metrics">
    <div class="metric">
      <span class="value">{stats['products']:,}</span>
      <span class="label">Products analysed</span>
    </div>
    <div class="metric">
      <span class="value">{stats['unique_tags']:,}</span>
      <span class="label">Distinct tags observed</span>
    </div>
    <div class="metric">
      <span class="value">{stats['battles']:,}</span>
      <span class="label">Tag battles recorded</span>
    </div>
  </section>

  <section class="leaderboard-section">
    <div class="leaderboard-text">
      <h2>Leaderboard</h2>
      <p>The Elo-inspired model favours tags that consistently appear earlier in product titles when paired with others. Here are the current top contenders.</p>
    </div>
    <div class="leaderboard-table">
      {top_table}
    </div>
    <figure class="chart">
      <img src="assets/{chart_path.name}" alt="Top tags bar chart" />
      <figcaption>Top 20 tags by inferred depth.</figcaption>
    </figure>
    <div class="leaderboard-bottom">
      <h3>Biggest losers</h3>
      <p>Tags that our model predicts are most likely to be pushed to the end of product titles.</p>
      <div class="leaderboard-table">
        {bottom_table}
      </div>
    </div>
  </section>

  {taxonomy_section}

  <section class="methodology">
    <h2>How the rankings work</h2>
    <ol>
      <li><strong>Battle generation</strong> &mdash; <code>tagbattle.py</code> scans each product title, comparing the order of every pair of tags.</li>
      <li><strong>Elo-style scoring</strong> &mdash; <code>ranking.py</code> treats each ordering as a battle, rewarding tags that appear closer to the start.</li>
      <li><strong>Visualisation</strong> &mdash; <code>display.py</code> turns the rankings into shareable tables and charts.</li>
    </ol>
    <p>Tags are grouped by connected component so isolated tag families get their own podium.</p>
  </section>

  <section class="downloads">
    <h2>Download the data</h2>
    <ul>
      {downloads_list_items}
    </ul>
    <p>Historical SQL dumps are synchronised to <a href="https://datadumps.ifost.org.au/padjective/">datadumps.ifost.org.au</a>.</p>
  </section>

  {experiments_block}

  <footer>
    <p>Rankings sourced from the Shopify Postgres battle records. Source available on <a href="https://github.com/IFost-Sydney-Uni/padjective">GitHub</a>.</p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_document, encoding="utf-8")


def _collect_taxonomy_nb_summary(conn) -> Optional[Dict[str, Any]]:
    """Fetch the latest ComplementNB taxonomy classifier summary from Postgres."""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                id,
                trained_at,
                samples,
                taxonomies,
                unique_tags,
                training_accuracy,
                cv_folds,
                cv_mean_accuracy,
                cv_std_accuracy
            FROM padjective.taxonomy_nb_models
            ORDER BY trained_at DESC, id DESC
            LIMIT 1
            """
        )
        model_row = cur.fetchone()

    if not model_row:
        return None

    model_id = model_row["id"]
    trained_at = model_row["trained_at"]
    stats: Dict[str, Any] = {
        "samples": model_row["samples"],
        "taxonomies": model_row["taxonomies"],
        "unique_tags": model_row["unique_tags"],
        "training_accuracy": model_row["training_accuracy"],
    }
    if model_row.get("cv_folds"):
        stats["cross_validation"] = {
            "folds": model_row["cv_folds"],
            "mean_accuracy": model_row["cv_mean_accuracy"],
            "std_accuracy": model_row["cv_std_accuracy"],
        }

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT taxonomy_id, taxonomy_path, prior
            FROM padjective.taxonomy_nb_priors
            WHERE model_id = %s
            ORDER BY prior DESC
            """,
            (model_id,),
        )
        taxonomy_priors = [
            {
                "taxonomy_id": row["taxonomy_id"],
                "taxonomy_path": row["taxonomy_path"],
                "prior": float(row["prior"]),
            }
            for row in cur.fetchall()
        ]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT tag, top_taxonomy_id, top_taxonomy_path, probability, margin
            FROM padjective.taxonomy_nb_tag_summary
            WHERE model_id = %s
            ORDER BY margin DESC, probability DESC
            LIMIT 200
            """,
            (model_id,),
        )
        top_tags = [
            {
                "tag": row["tag"],
                "top_taxonomy_id": row["top_taxonomy_id"],
                "top_taxonomy_path": row["top_taxonomy_path"],
                "probability": float(row["probability"]),
                "margin": float(row["margin"]),
            }
            for row in cur.fetchall()
        ]

    return {
        "model_id": model_id,
        "trained_at": trained_at.isoformat(timespec="seconds") if trained_at else None,
        "stats": stats,
        "taxonomy_priors": taxonomy_priors,
        "top_tags": top_tags,
    }
def build_site(
    csv_path: Path,
    output_dir: Path,
    *,
    precomputed_database: Optional[Any] = None,
    battle_schema: str = "padjective",
    tasks_db: Optional[Path] = None,
) -> Dict[str, Any]:
    csv_path = csv_path.resolve()
    _ensure_clean_directory(output_dir)

    assets_dir = output_dir / "assets"
    downloads_dir = output_dir / "downloads"
    datadumps_dir = output_dir / "datadumps"
    for path in (assets_dir, downloads_dir, datadumps_dir):
        path.mkdir(parents=True, exist_ok=True)

    if precomputed_database is None:
        raise ValueError("A Postgres connection is required to build the site")

    pairs = ranking.load_pairs(precomputed_database, battle_schema)
    leaderboard = ranking.compute_rankings(pairs)

    rankings_html = downloads_dir / "tag_rankings_table.html"
    chart_path = assets_dir / "top_tags.png"
    display.generate_outputs(leaderboard, rankings_html, chart_path, rows=20)

    stats = _collect_tag_stats(csv_path)
    stats["battles"] = _count_battles(pairs)
    stats["components"] = int(leaderboard["component"].nunique()) if not leaderboard.empty else 0

    dump_path = datadumps_dir / "battles.sql"
    _write_sql_dump(pairs, dump_path, battle_schema)

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
.leaderboard-bottom {padding: 0 1rem 2.5rem;}
.leaderboard-bottom h3 {margin: 0 0 0.5rem; font-size: 1.35rem;}
.leaderboard-bottom p {margin: 0 0 1rem; color: #475569;}
.leaderboard-bottom-table tbody tr:nth-child(even) {background: #f1f5f9;}
.chart {text-align: center; padding: 0 1rem 2rem;}
.chart img {max-width: 100%; height: auto; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);}
.taxonomy-classifier {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); margin-top: 2rem; padding: 2rem 1.5rem;}
.taxonomy-classifier h2 {margin-top: 0;}
.taxonomy-stats {list-style: none; padding: 0; margin: 1rem 0 2rem; display: flex; flex-wrap: wrap; gap: 1rem;}
.taxonomy-stats li {background: #f1f5f9; border-radius: 0.75rem; padding: 0.75rem 1rem; box-shadow: inset 0 0 0 1px #dbeafe;}
.taxonomy-layout {display: flex; flex-wrap: wrap; gap: 1.5rem;}
.taxonomy-card {flex: 1 1 22rem; background: #f8fafc; border-radius: 0.9rem; box-shadow: inset 0 0 0 1px #e2e8f0; padding: 1rem;}
.taxonomy-card h3 {margin-top: 0;}
table.taxonomy-table, table.tag-taxonomy-table {width: 100%; border-collapse: collapse; background: white; border-radius: 0.75rem; overflow: hidden;}
table.taxonomy-table th, table.taxonomy-table td, table.tag-taxonomy-table th, table.tag-taxonomy-table td {padding: 0.75rem 1rem; border-bottom: 1px solid #e2e8f0; text-align: left;}
table.taxonomy-table thead, table.tag-taxonomy-table thead {background: #0b6ce3; color: white;}
table.taxonomy-table tbody tr:nth-child(even), table.tag-taxonomy-table tbody tr:nth-child(even) {background: #f8fafc;}
.methodology {background: #eef2ff; border-radius: 1rem;}
.methodology ol {line-height: 1.7;}
.experiments {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); margin-top: 2rem; padding-bottom: 2rem;}
.experiments h2 {margin-top: 0; padding: 2rem 1.5rem 0;}
.experiments p {padding: 0 1.5rem;}
.experiments-metrics {display: flex; flex-wrap: wrap; gap: 1rem; padding: 1rem 1.5rem;}
.experiments-accuracy {font-style: italic; color: #334155;}
.experiments-table {width: calc(100% - 3rem); margin: 1rem 1.5rem; border-collapse: collapse;}
.experiments-table th, .experiments-table td {border-bottom: 1px solid #e5e7eb; padding: 0.75rem 1rem; text-align: left;}
.experiments-table thead {background: #f1f5f9;}
.downloads ul {list-style: none; padding: 0;}
.downloads li {margin: 0.5rem 0;}
.downloads a {color: #0b6ce3; text-decoration: none; font-weight: 600;}
.downloads a:hover {text-decoration: underline;}
footer {text-align: center; padding: 2rem 1.5rem 3rem; color: #6b7280;}
@media (max-width: 900px) {.taxonomy-layout {flex-direction: column;}}
"""
    )

    artifact_links: Dict[str, Path] = {
        "Tag rankings table (HTML)": rankings_html,
        "SQL dump of battles": dump_path,
        "Top tags chart": chart_path,
    }

    experiments_summary: Optional[Dict[str, Any]] = None
    if tasks_db is not None and tasks_db.exists():
        experiments_summary = experiments.task_status(tasks_db)

    taxonomy_summary = _collect_taxonomy_nb_summary(precomputed_database)

    _build_index_html(
        output_dir,
        stats,
        leaderboard,
        chart_path,
        artifact_links,
        experiments_summary,
        taxonomy_summary,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "artifacts": {label: str(path.relative_to(output_dir)) for label, path in artifact_links.items()},
        "experiments": experiments_summary,
        "taxonomy_classifier": taxonomy_summary,
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
    parser.add_argument(
        "--dsn",
        help="Postgres DSN for reading battles. Uses SHOPIFY_DB_DSN or DATABASE_URL if unset.",
    )
    parser.add_argument(
        "--schema",
        default="padjective",
        help="Schema containing battles and output tables.",
    )
    parser.add_argument(
        "--tasks-db",
        type=Path,
        default=None,
        help="Optional experiments task database for progress reporting",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        build_site(
            args.csv,
            args.output,
            precomputed_database=conn,
            battle_schema=args.schema,
            tasks_db=args.tasks_db,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
