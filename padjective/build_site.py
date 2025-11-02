"""Build a static website showcasing tag rankings and taxonomy progress."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row

from . import db, display, ranking, tagbattle


def _ensure_clean_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _collect_database_stats(conn, schema: str) -> Dict[str, int]:
    """Return aggregate statistics derived from Postgres battles data."""

    stats: Dict[str, int] = {"products": 0, "unique_tags": 0}

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT COUNT(DISTINCT product_id) FROM {schema}.battles "
                "WHERE product_id IS NOT NULL"
            ).format(schema=sql.Identifier(schema))
        )
        product_row = cur.fetchone()
        if product_row and product_row[0] is not None:
            stats["products"] = int(product_row[0])

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT COUNT(DISTINCT tag) FROM ("
                " SELECT winner_tag AS tag FROM {schema}.battles"
                " UNION ALL"
                " SELECT loser_tag AS tag FROM {schema}.battles"
                ") AS combined_tags"
            ).format(schema=sql.Identifier(schema))
        )
        tag_row = cur.fetchone()
        if tag_row and tag_row[0] is not None:
            stats["unique_tags"] = int(tag_row[0])

    return stats


def _count_battles(pairs: Sequence[Tuple[str, str]]) -> int:
    return len(pairs)


def _table_exists(conn, schema: str, table: str) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
                """,
                (schema, table),
            )
            fetchone = getattr(cur, "fetchone", None)
            if fetchone is None:
                return False
            return fetchone() is not None
    except Exception:
        return False


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


def _load_umllr_results(conn, schema: str) -> Optional[Dict[str, Any]]:
    required_tables = (
        "umllr_fold_metrics",
        "umllr_tag_coefficients",
        "umllr_predictions",
    )
    if not all(_table_exists(conn, schema, table) for table in required_tables):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, loss, prime_base, max_digit
                FROM {schema}.umllr_fold_metrics
                ORDER BY cv_fold
                """
            ).format(schema=sql.Identifier(schema))
        )
        metrics_rows = cur.fetchall()

    if not metrics_rows:
        return None

    metrics = [
        {
            "cv_fold": int(row["cv_fold"]),
            "loss": float(row["loss"]),
            "prime_base": int(row["prime_base"]),
            "max_digit": int(row["max_digit"]),
        }
        for row in metrics_rows
    ]

    coefficients: Dict[int, list[Dict[str, Any]]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, tag, coefficient, sequence
                FROM {schema}.umllr_tag_coefficients
                ORDER BY cv_fold, sequence, tag
                """
            ).format(schema=sql.Identifier(schema))
        )
        for row in cur:
            fold = int(row["cv_fold"])
            coefficients.setdefault(fold, []).append(
                {
                    "tag": row["tag"],
                    "coefficient": int(row["coefficient"]),
                    "sequence": int(row["sequence"]),
                }
            )

    predictions: Dict[int, list[Dict[str, Any]]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, product_id, true_value, predicted_value, loss
                FROM {schema}.umllr_predictions
                ORDER BY cv_fold, product_id
                """
            ).format(schema=sql.Identifier(schema))
        )
        for row in cur:
            fold = int(row["cv_fold"])
            predictions.setdefault(fold, []).append(
                {
                    "product_id": int(row["product_id"]),
                    "true_value": int(row["true_value"]),
                    "predicted_value": int(row["predicted_value"]),
                    "loss": float(row["loss"]),
                }
            )

    return {
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
    }


def _write_umllr_pages(output_dir: Path, summary: Dict[str, Any]) -> Dict[int, Path]:
    pages: Dict[int, Path] = {}
    metrics = summary.get("metrics", [])
    if not metrics:
        return pages

    umllr_dir = output_dir / "umllr"
    umllr_dir.mkdir(parents=True, exist_ok=True)

    coefficients = summary.get("coefficients", {})
    predictions = summary.get("predictions", {})

    for metric in metrics:
        fold = metric["cv_fold"]
        coeff_rows = coefficients.get(fold, [])
        prediction_rows = predictions.get(fold, [])

        coeff_table_rows = "\n".join(
            f"<tr><td>{html.escape(row['tag'])}</td><td>{row['coefficient']}</td><td>{row['sequence']}</td></tr>"
            for row in coeff_rows
        )
        if not coeff_table_rows:
            coeff_table_rows = '<tr><td colspan="3">No coefficients recorded for this fold.</td></tr>'

        prediction_table_rows = "\n".join(
            "<tr>"
            f"<td>{row['product_id']}</td>"
            f"<td>{row['true_value']}</td>"
            f"<td>{row['predicted_value']}</td>"
            f"<td>{row['loss']:.6f}</td>"
            "</tr>"
            for row in prediction_rows
        )
        if not prediction_table_rows:
            prediction_table_rows = '<tr><td colspan="4">No test predictions available for this fold.</td></tr>'

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>umllr fold {fold} results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>umllr fold {fold}</h1>
    <p><a href="../index.html">Back to index</a></p>
    <p><strong>Total p-adic loss:</strong> {metric['loss']:.6f} &middot; <strong>Prime base:</strong> {metric['prime_base']} &middot; <strong>Max digit:</strong> {metric['max_digit']}</p>
    <h2>Tag coefficients</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Tag</th><th>Coefficient</th><th>Order</th></tr>
      </thead>
      <tbody>
        {coeff_table_rows}
      </tbody>
    </table>
    <h2>Test predictions</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Product ID</th><th>Ground truth</th><th>Prediction</th><th>p-adic loss</th></tr>
      </thead>
      <tbody>
        {prediction_table_rows}
      </tbody>
    </table>
  </section>
</body>
</html>
"""

        page_path = umllr_dir / f"fold_{fold}.html"
        page_path.write_text(page_contents, encoding="utf-8")
        pages[fold] = page_path

    return pages




def _build_index_html(
    output_dir: Path,
    stats: Dict[str, int],
    leaderboard: pd.DataFrame,
    chart_path: Path,
    artifact_links: Dict[str, Path],
    taxonomy_summary: Optional[Dict[str, Any]] = None,
    umllr_summary: Optional[Dict[str, Any]] = None,
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


    taxonomy_section = ""
    if taxonomy_summary:
        stats_block = taxonomy_summary.get("stats", {})
        class_distribution = taxonomy_summary.get("class_distribution", [])[:10]
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
                f"<li><strong>Training accuracy:</strong> {accuracy * 100:.2f}%</li>"
            )
        training_f1 = stats_block.get("training_f1")
        if training_f1 is not None:
            summary_items.append(
                f"<li><strong>Training F1 (weighted):</strong> {training_f1 * 100:.2f}%</li>"
            )
        training_hier = stats_block.get("training_hierarchical_loss")
        if training_hier is not None:
            summary_items.append(
                f"<li><strong>Training hierarchical loss:</strong> {training_hier:.3f}</li>"
            )
        cv_info = stats_block.get("cross_validation") or {}
        if cv_info:
            mean = cv_info.get("mean_accuracy")
            std = cv_info.get("std_accuracy")
            folds = cv_info.get("folds")
            if mean is not None and folds:
                summary_items.append(
                    "<li><strong>Cross-validated accuracy:</strong> "
                    f"{mean * 100:.2f}%"
                    + (f" ± {std * 100:.2f}%" if std is not None else "")
                    + f" ({folds} folds)</li>"
                )
            mean_f1 = cv_info.get("mean_f1")
            if mean_f1 is not None:
                std_f1 = cv_info.get("std_f1")
                summary_items.append(
                    "<li><strong>Cross-validated F1 (weighted):</strong> "
                    f"{mean_f1 * 100:.2f}%"
                    + (f" ± {std_f1 * 100:.2f}%" if std_f1 is not None else "")
                    + "</li>"
                )
            mean_hier = cv_info.get("mean_hierarchical_loss")
            if mean_hier is not None:
                std_hier = cv_info.get("std_hierarchical_loss")
                summary_items.append(
                    "<li><strong>Cross-validated hierarchical loss:</strong> "
                    f"{mean_hier:.3f}"
                    + (f" ± {std_hier:.3f}" if std_hier is not None else "")
                    + "</li>"
                )
        if trained_at:
            summary_items.append(
                f"<li><strong>Trained:</strong> {html.escape(trained_at)}</li>"
            )

        summary_list = '<ul class="taxonomy-stats">' + "".join(summary_items) + "</ul>"

        distribution_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.get('taxonomy_id') or '')}</td>"
            f"<td>{html.escape(row.get('taxonomy_path') or 'Unknown')}</td>"
            f"<td>{row.get('sample_count', 0):,}</td>"
            f"<td>{row.get('sample_fraction', 0.0) * 100:.2f}%</td>"
            "</tr>"
            for row in class_distribution
        )
        distribution_body = (
            distribution_rows or '<tr><td colspan="4">No taxonomy data</td></tr>'
        )
        taxonomy_table = (
            '<table class="taxonomy-table">'
            '<thead><tr><th>Taxonomy ID</th><th>Path</th><th>Samples</th><th>Share</th></tr></thead>'
            f"<tbody>{distribution_body}</tbody>"
            "</table>"
        )

        tag_rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(row.get('tag') or '')}</td>"
            f"<td>{html.escape(row.get('top_taxonomy_id') or '')}</td>"
            f"<td>{html.escape(row.get('top_taxonomy_path') or 'Unknown')}</td>"
            f"<td>{row.get('top_weight', 0.0):.4f}</td>"
            f"<td>{row.get('max_abs_weight', 0.0):.4f}</td>"
            "</tr>"
            for row in top_tags_rows
        )
        tag_body = tag_rows or '<tr><td colspan="5">No tag signals available</td></tr>'
        tag_table = (
            '<table class="tag-taxonomy-table">'
            '<thead><tr><th>Tag</th><th>Taxonomy ID</th><th>Path</th><th>Weight</th><th>Max |weight|</th></tr></thead>'
            f"<tbody>{tag_body}</tbody>"
            "</table>"
        )

        taxonomy_section = f"""
  <section class="taxonomy-classifier">
    <h2>Shopify taxonomy classification</h2>
    <p>We train a logistic regression model on Shopify tags to predict taxonomy IDs.</p>
    {summary_list}
    <div class="taxonomy-grid">
      <section>
        <h3>Largest taxonomy classes</h3>
        {taxonomy_table}
      </section>
      <section>
        <h3>Tags with strongest signal</h3>
        {tag_table}
      </section>
    </div>
  </section>
"""

    umllr_block = ""
    if umllr_summary and umllr_summary.get("metrics"):
        rows: list[str] = []
        page_lookup = umllr_summary.get("pages", {})
        for metric in umllr_summary.get("metrics", []):
            fold = metric["cv_fold"]
            link = page_lookup.get(fold)
            if link:
                details = f'<a href="{link}">View fold {fold}</a>'
            else:
                details = f"Fold {fold}"
            rows.append(
                "<tr>"
                f"<td>{fold}</td>"
                f"<td>{metric['loss']:.6f}</td>"
                f"<td>{metric.get('prime_base')}</td>"
                f"<td>{metric.get('max_digit')}</td>"
                f"<td>{details}</td>"
                "</tr>"
            )
        table_body = "\n".join(rows) or '<tr><td colspan="5">No cross-validation folds recorded.</td></tr>'
        umllr_block = f"""
  <section class="umllr">
    <h2>umllr cross-validation</h2>
    <p>The umllr trainer assigns p-adic coefficients to tags and evaluates them on held-out products.</p>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Total loss</th><th>Prime base</th><th>Max digit</th><th>Details</th></tr>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
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
      <p>The Elo-inspired model favours tags that consistently appear later in product titles when paired with others. Here are the current top contenders.</p>
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

  {umllr_block}
  {taxonomy_section}

  <section class="methodology">
    <h2>How the rankings work</h2>
    <ol>
      <li><strong>Battle generation</strong> &mdash; <code>tagbattle.py</code> scans each product title, comparing the order of every pair of tags.</li>
      <li><strong>Elo-style scoring</strong> &mdash; <code>ranking.py</code> treats each ordering as a battle, rewarding tags that appear later in the title (rightmost position wins).</li>
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

  <footer>
    <p>Rankings sourced from the Shopify Postgres battle records. Source available on <a href="https://github.com/IFost-Sydney-Uni/padjective">GitHub</a>.</p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_document, encoding="utf-8")


def _collect_taxonomy_classifier_summary(
    conn, schema: str = "padjective"
) -> Optional[Dict[str, Any]]:
    """Fetch the latest logistic taxonomy classifier summary from Postgres."""

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    id,
                    trained_at,
                    samples,
                    taxonomies,
                    unique_tags,
                    training_accuracy,
                    training_f1,
                    training_hierarchical_loss,
                    cv_folds,
                    cv_mean_accuracy,
                    cv_std_accuracy,
                    cv_mean_f1,
                    cv_std_f1,
                    cv_mean_hierarchical_loss,
                    cv_std_hierarchical_loss
                FROM {schema}.taxonomy_lr_models
                ORDER BY trained_at DESC, id DESC
                LIMIT 1
                """
            ).format(schema=sql.Identifier(schema))
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
    if model_row.get("training_f1") is not None:
        stats["training_f1"] = float(model_row["training_f1"])
    if model_row.get("training_hierarchical_loss") is not None:
        stats["training_hierarchical_loss"] = float(
            model_row["training_hierarchical_loss"]
        )
    if model_row.get("cv_folds"):
        stats["cross_validation"] = {
            "folds": model_row["cv_folds"],
            "mean_accuracy": model_row["cv_mean_accuracy"],
            "std_accuracy": model_row["cv_std_accuracy"],
            "mean_f1": model_row.get("cv_mean_f1"),
            "std_f1": model_row.get("cv_std_f1"),
            "mean_hierarchical_loss": model_row.get("cv_mean_hierarchical_loss"),
            "std_hierarchical_loss": model_row.get("cv_std_hierarchical_loss"),
        }

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT taxonomy_id, taxonomy_path, sample_count, sample_fraction
                FROM {schema}.taxonomy_lr_class_distribution
                WHERE model_id = %s
                ORDER BY sample_fraction DESC, sample_count DESC, taxonomy_id
                """
            ).format(schema=sql.Identifier(schema)),
            (model_id,),
        )
        class_distribution = [
            {
                "taxonomy_id": row["taxonomy_id"],
                "taxonomy_path": row["taxonomy_path"],
                "sample_count": int(row["sample_count"]),
                "sample_fraction": float(row["sample_fraction"]),
            }
            for row in cur.fetchall()
        ]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tag, top_taxonomy_id, top_taxonomy_path,
                       top_weight, max_abs_weight, sum_abs_weight
                FROM {schema}.taxonomy_lr_tag_summary
                WHERE model_id = %s
                ORDER BY max_abs_weight DESC, sum_abs_weight DESC, tag
                LIMIT 200
                """
            ).format(schema=sql.Identifier(schema)),
            (model_id,),
        )
        tag_summary = [
            {
                "tag": row["tag"],
                "top_taxonomy_id": row["top_taxonomy_id"],
                "top_taxonomy_path": row["top_taxonomy_path"],
                "top_weight": float(row["top_weight"]),
                "max_abs_weight": float(row["max_abs_weight"]),
                "sum_abs_weight": float(row["sum_abs_weight"]),
            }
            for row in cur.fetchall()
        ]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT taxonomy_id, taxonomy_path, tag, weight, rank
                FROM {schema}.taxonomy_lr_top_tags
                WHERE model_id = %s
                ORDER BY taxonomy_id, rank
                """
            ).format(schema=sql.Identifier(schema)),
            (model_id,),
        )
        taxonomy_top_tags = [
            {
                "taxonomy_id": row["taxonomy_id"],
                "taxonomy_path": row["taxonomy_path"],
                "tag": row["tag"],
                "weight": float(row["weight"]),
                "rank": int(row["rank"]),
            }
            for row in cur.fetchall()
        ]

    return {
        "model_id": model_id,
        "trained_at": trained_at.isoformat(timespec="seconds") if trained_at else None,
        "stats": stats,
        "class_distribution": class_distribution,
        "top_tags": tag_summary,
        "taxonomy_top_tags": taxonomy_top_tags,
    }
def build_site(
    output_dir: Path,
    *,
    precomputed_database: Optional[Any] = None,
    battle_schema: str = "padjective",
) -> Dict[str, Any]:
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

    stats = _collect_database_stats(precomputed_database, battle_schema)
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
.umllr {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); margin-top: 2rem; padding: 2rem 1.5rem;}
.umllr h2 {margin-top: 0;}
.umllr-summary {width: 100%; border-collapse: collapse; margin-top: 1rem;}
.umllr-summary th, .umllr-summary td {border-bottom: 1px solid #e2e8f0; padding: 0.75rem 1rem; text-align: left;}
.umllr-summary thead {background: #f8fafc;}
.umllr-fold {max-width: 70rem; margin: 0 auto; padding: 2rem 1.5rem;}
.umllr-fold h1 {margin-top: 0;}
.umllr-table {width: 100%; border-collapse: collapse; margin-bottom: 2rem; background: white;}
.umllr-table th, .umllr-table td {border-bottom: 1px solid #e2e8f0; padding: 0.75rem 1rem; text-align: left;}
.umllr-table thead {background: #f8fafc;}
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

    taxonomy_summary = _collect_taxonomy_classifier_summary(
        precomputed_database, schema=battle_schema
    )

    umllr_summary = _load_umllr_results(precomputed_database, battle_schema)
    if umllr_summary:
        pages = _write_umllr_pages(output_dir, umllr_summary)
        umllr_summary["pages"] = {
            fold: path.relative_to(output_dir).as_posix()
            for fold, path in pages.items()
        }

    _build_index_html(
        output_dir,
        stats,
        leaderboard,
        chart_path,
        artifact_links,
        taxonomy_summary,
        umllr_summary,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "artifacts": {label: str(path.relative_to(output_dir)) for label, path in artifact_links.items()},
        "taxonomy_classifier": taxonomy_summary,
        "umllr": umllr_summary,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the Padjective website from Postgres data"
    )
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
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        build_site(
            args.output,
            precomputed_database=conn,
            battle_schema=args.schema,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
