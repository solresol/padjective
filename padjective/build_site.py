"""Build a static website showcasing tag rankings and synset progress."""

from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from . import display, experiments, ranking, tagbattle


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
    experiments_summary: Optional[Dict[str, Any]] = None,
    synset_summary: Optional[Dict[str, Any]] = None,
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
            recent_items = "<tr><td colspan=\"6\">No completed evaluations yet.</td></tr>"

        experiments_block = f"""
  <section class=\"experiments\">
    <h2>Hold-out experiments</h2>
    <p>We randomly reserve {test_fraction:.0%} of recorded tag battles and check whether the rankings predict the correct ordering.</p>
    <div class=\"experiments-metrics\">
      <div class=\"metric\">
        <span class=\"value\">{completed:,} / {total:,}</span>
        <span class=\"label\">Tasks completed</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{counts.get('pending', 0):,}</span>
        <span class=\"label\">Pending tasks</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{counts.get('running', 0):,}</span>
        <span class=\"label\">Running tasks</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{counts.get('error', 0):,}</span>
        <span class=\"label\">Errors</span>
      </div>
    </div>
    <p class=\"experiments-accuracy\">Average accuracy across completed tasks: {mean_accuracy_text} (coverage {mean_coverage_text}).</p>
    <table class=\"experiments-table\">
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

    synset_section = ""
    if synset_summary:
        processed = synset_summary.get("processed", 0)
        remaining = synset_summary.get("remaining", 0)
        not_found = synset_summary.get("not_found", 0)
        progress_pct = synset_summary.get("progress_pct")
        rate_per_day = synset_summary.get("rate_per_day")
        rate_text = "n/a"
        if rate_per_day is not None and rate_per_day > 0:
            rate_text = f"{rate_per_day:,.1f} products/day"
        eta_text = synset_summary.get("eta_text") or "n/a"
        completion_text = synset_summary.get("estimated_completion_text") or ""
        completion_fragment = f" {html.escape(completion_text)}" if completion_text else ""
        last_processed_text = synset_summary.get("last_processed_text") or ""
        last_processed_fragment = f" {html.escape(last_processed_text)}" if last_processed_text else ""

        top_synsets = synset_summary.get("synsets", [])[:10]
        top_rows: Iterable[str] = []
        if top_synsets:
            top_rows = [
                "<tr>"
                f"<td><a href=\"synsets/{html.escape(s['synset_id'])}.html\">{html.escape(s.get('synset_name') or s['synset_id'])}</a></td>"
                f"<td>{html.escape(s['synset_id'])}</td>"
                f"<td>{s['product_count']:,}</td>"
                f"<td>{s['share'] * 100:.1f}%</td>"
                "</tr>"
                for s in top_synsets
            ]
        top_table = "<table class=\"synset-table\"><thead><tr><th>Synset</th><th>ID</th><th>Products</th><th>Share</th></tr></thead><tbody>"
        if top_synsets:
            top_table += "\n".join(top_rows)
        else:
            top_table += "<tr><td colspan=\"4\">No synsets processed yet.</td></tr>"
        top_table += "</tbody></table>"

        progress_text = f"{progress_pct:.1f}%" if progress_pct is not None else "n/a"
        not_found_examples = synset_summary.get("not_found_examples") or []
        not_found_block = ""
        if not_found_examples:
            items = "\n".join(
                f"<li>{html.escape(title)}</li>" for title in not_found_examples if title
            )
            if items:
                not_found_block = (
                    "<div class=\"synset-not-found\">"
                    "<h4>Recent products without a synset</h4>"
                    f"<ul>{items}</ul>"
                    "</div>"
                )

        usage_summary = synset_summary.get("usage") or {}
        usage_daily = usage_summary.get("daily") or []
        tokens_last_24h = int(usage_summary.get("tokens_last_24h") or 0)
        tokens_per_day = float(usage_summary.get("tokens_per_day") or 0.0)
        tokens_per_second = float(usage_summary.get("tokens_per_second") or 0.0)
        daily_quota = float(usage_summary.get("daily_quota") or 5_000_000)
        usage_rate_text = (
            f"{tokens_per_day:,.0f} tokens/day ({tokens_per_second:,.1f} tokens/s)"
            if tokens_per_day
            else "n/a"
        )
        quota_text = f"{daily_quota:,.0f} tokens/day"
        usage_rows = []
        for row in usage_daily:
            usage_rows.append(
                "<tr>"
                f"<td>{html.escape(row['date'])}</td>"
                f"<td>{row['total_tokens']:,}</td>"
                f"<td>{row['input_tokens']:,}</td>"
                f"<td>{row['output_tokens']:,}</td>"
                f"<td>{row['calls']:,}</td>"
                "</tr>"
            )
        usage_table = ""
        if usage_rows:
            usage_table = (
                "<table class=\"synset-usage-table\">"
                "<thead><tr><th>Date</th><th>Total tokens</th><th>Input</th><th>Output</th><th>LLM calls</th></tr></thead>"
                f"<tbody>{''.join(usage_rows)}</tbody>"
                "</table>"
            )
        usage_section = (
            "<div class=\"synset-usage\">"
            "<h3>Token usage</h3>"
            f"<p>Last 24h: {tokens_last_24h:,} tokens. Rolling rate: {usage_rate_text}. "
            f"Quota: {quota_text}.</p>"
            f"{usage_table}"
            "</div>"
            if (tokens_last_24h or usage_rows)
            else ""
        )

        synset_section = f"""
  <section class=\"synset-progress\">
    <div class=\"synset-header\">
      <h2>WordNet synset tagging</h2>
      <p>We ask GPT models to map each product to the closest WordNet synset.</p>
    </div>
    <div class=\"metrics synset-metrics\">
      <div class=\"metric\">
        <span class=\"value\">{processed:,}</span>
        <span class=\"label\">Products classified</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{remaining:,}</span>
        <span class=\"label\">Products remaining</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{not_found:,}</span>
        <span class=\"label\">No WordNet match</span>
      </div>
      <div class=\"metric\">
        <span class=\"value\">{progress_text}</span>
        <span class=\"label\">Coverage so far</span>
      </div>
    </div>
    <p class=\"synset-eta\">
      Processing rate: {rate_text}. Remaining work estimated completion: {eta_text}.{completion_fragment}{last_processed_fragment}
    </p>
    <div class=\"synset-top\">
      <h3>Most common synsets</h3>
      {top_table}
      <p><a href=\"synsets/index.html\">Explore all processed synsets &rarr;</a></p>
    </div>
    {not_found_block}
    {usage_section}
  </section>
"""

    html_document = f"""<!DOCTYPE html>
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
    <div class=\"leaderboard-bottom\">
      <h3>Biggest losers</h3>
      <p>Tags that our model predicts are most likely to be pushed to the end of product titles.</p>
      <div class=\"leaderboard-table\">
        {bottom_table}
      </div>
    </div>
  </section>

  {synset_section}

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

  {experiments_block}

  <footer>
    <p>Built from <code>{source_csv.name}</code>. Source available on <a href=\"https://github.com/IFost-Sydney-Uni/padjective\">GitHub</a>.</p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_document, encoding="utf-8")


def _parse_sqlite_timestamp(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace(" ", "T"))
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    return None


def _format_duration(hours: float) -> str:
    if hours <= 0:
        return "soon"
    total_seconds = int(hours * 3600)
    days, remainder = divmod(total_seconds, 86400)
    hrs, remainder = divmod(remainder, 3600)
    minutes = remainder // 60
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hrs:
        parts.append(f"{hrs} hour{'s' if hrs != 1 else ''}")
    if minutes and not days:
        parts.append(f"{minutes} min")
    return " ".join(parts) or "<1 min"


def _collect_synset_progress(
    db_path: Path, total_products: int
) -> Optional[Dict[str, Any]]:
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        summary_row = conn.execute(
            """
            SELECT
                COUNT(*) AS processed,
                SUM(CASE WHEN not_found = 1 THEN 1 ELSE 0 END) AS not_found,
                MIN(processed_at) AS first_processed,
                MAX(processed_at) AS last_processed
            FROM product_synsets
            """
        ).fetchone()
        processed = int(summary_row["processed"] or 0)
        if processed == 0:
            return {
                "processed": 0,
                "remaining": max(total_products, 0),
                "not_found": 0,
                "progress_pct": 0.0 if total_products else None,
                "synsets": [],
                "eta_text": "n/a",
            }

        not_found = int(summary_row["not_found"] or 0)
        remaining = max(total_products - processed, 0)
        progress_pct = (processed / total_products * 100.0) if total_products else None

        last_processed = _parse_sqlite_timestamp(summary_row["last_processed"])
        last_processed_text = (
            f"Last processed at {last_processed.isoformat(timespec='seconds')}"
            if last_processed
            else ""
        )

        recent_rows = conn.execute(
            """
            SELECT processed_at FROM product_synsets
            WHERE processed_at IS NOT NULL
            ORDER BY processed_at DESC
            LIMIT 500
            """
        ).fetchall()
        recent_times = [
            _parse_sqlite_timestamp(row[0])
            for row in reversed(recent_rows)
            if row[0] is not None
        ]
        rate_per_hour: Optional[float] = None
        eta_text = None
        estimated_completion_text = None
        if len(recent_times) >= 2:
            elapsed = (recent_times[-1] - recent_times[0]).total_seconds() / 3600
            if elapsed > 0:
                window_processed = len(recent_times)
                rate_per_hour = window_processed / elapsed
                if remaining > 0:
                    eta_hours = remaining / rate_per_hour
                    eta_text = _format_duration(eta_hours)
                    if last_processed:
                        projected = last_processed + timedelta(hours=eta_hours)
                        estimated_completion_text = (
                            f"Projected completion around {projected.isoformat(timespec='minutes')}"
                        )
                else:
                    eta_text = "complete"

        if eta_text is None:
            eta_text = "n/a"

        synset_rows = conn.execute(
            """
            SELECT synset_id, synset_name, synset_definition, COUNT(*) AS product_count
            FROM product_synsets
            WHERE not_found = 0 AND synset_id IS NOT NULL
            GROUP BY synset_id, synset_name, synset_definition
            ORDER BY product_count DESC, synset_id
            """
        ).fetchall()

        synsets: list[Dict[str, Any]] = []
        for row in synset_rows:
            synset_id = row["synset_id"]
            if not synset_id:
                continue
            products = conn.execute(
                """
                SELECT product_id, title, tags, confidence, reason, processed_at
                FROM product_synsets
                WHERE synset_id = ? AND not_found = 0
                ORDER BY COALESCE(confidence, 0) DESC, title COLLATE NOCASE
                """,
                (synset_id,),
            ).fetchall()
            product_records = [
                {
                    "product_id": product["product_id"],
                    "title": product["title"],
                    "tags": product["tags"],
                    "confidence": product["confidence"],
                    "reason": product["reason"],
                    "processed_at": product["processed_at"],
                }
                for product in products
            ]
            share = (row["product_count"] or 0) / processed if processed else 0.0
            synsets.append(
                {
                    "synset_id": synset_id,
                    "synset_name": row["synset_name"],
                    "synset_definition": row["synset_definition"],
                    "product_count": int(row["product_count"] or 0),
                    "share": share,
                    "products": product_records,
                }
            )

        not_found_examples = conn.execute(
            """
            SELECT title FROM product_synsets
            WHERE not_found = 1 AND title IS NOT NULL AND title != ''
            ORDER BY processed_at DESC
            LIMIT 5
            """
        ).fetchall()

        usage_table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='synset_usage'"
        ).fetchone()
        daily_usage: list[Dict[str, Any]] = []
        tokens_last_24h = 0
        tokens_per_second = 0.0
        tokens_per_day = 0.0
        daily_quota = 5_000_000
        if usage_table_exists:
            usage_rows = conn.execute(
                """
                SELECT
                    substr(recorded_at, 1, 10) AS usage_day,
                    SUM(COALESCE(total_tokens, 0)) AS total_tokens,
                    SUM(COALESCE(input_tokens, 0)) AS input_tokens,
                    SUM(COALESCE(output_tokens, 0)) AS output_tokens,
                    COUNT(*) AS calls
                FROM synset_usage
                GROUP BY usage_day
                ORDER BY usage_day DESC
                LIMIT 14
                """
            ).fetchall()

            daily_usage = [
                {
                    "date": row["usage_day"],
                    "total_tokens": int(row["total_tokens"] or 0),
                    "input_tokens": int(row["input_tokens"] or 0),
                    "output_tokens": int(row["output_tokens"] or 0),
                    "calls": int(row["calls"] or 0),
                }
                for row in usage_rows
                if row["usage_day"]
            ]

            now = datetime.now(timezone.utc)
            window_start = now - timedelta(days=1)
            usage_window = conn.execute(
                """
                SELECT
                    SUM(COALESCE(total_tokens, 0)) AS total_tokens,
                    MIN(recorded_at) AS first_record
                FROM synset_usage
                WHERE recorded_at >= ?
                """,
                (window_start.isoformat(),),
            ).fetchone()

            tokens_last_24h = int((usage_window["total_tokens"] or 0))
            earliest_usage = _parse_sqlite_timestamp(usage_window["first_record"])
            if earliest_usage is None:
                elapsed_seconds = 24 * 3600
            else:
                elapsed_seconds = max((now - earliest_usage).total_seconds(), 1.0)
            tokens_per_second = (
                tokens_last_24h / elapsed_seconds if tokens_last_24h else 0.0
            )
            tokens_per_day = tokens_per_second * 86400

        return {
            "processed": processed,
            "remaining": remaining,
            "not_found": not_found,
            "progress_pct": progress_pct,
            "rate_per_hour": rate_per_hour,
            "rate_per_day": rate_per_hour * 24 if rate_per_hour else None,
            "eta_text": eta_text,
            "estimated_completion_text": estimated_completion_text,
            "last_processed_text": last_processed_text,
            "synsets": synsets,
            "not_found_examples": [row[0] for row in not_found_examples],
            "usage": {
                "daily": daily_usage,
                "tokens_last_24h": tokens_last_24h,
                "tokens_per_second": tokens_per_second,
                "tokens_per_day": tokens_per_day,
                "daily_quota": daily_quota,
            },
        }
    finally:
        conn.close()


def _build_synset_pages(output_dir: Path, synset_summary: Dict[str, Any]) -> None:
    synsets = synset_summary.get("synsets") or []
    if not synsets:
        return

    synsets_dir = output_dir / "synsets"
    synsets_dir.mkdir(parents=True, exist_ok=True)

    def _product_rows(products: Iterable[Dict[str, Any]]) -> str:
        rows = []
        for product in products:
            confidence = product.get("confidence")
            if confidence is None:
                confidence_text = "n/a"
            else:
                confidence_text = f"{confidence:.2f}"
            reason = html.escape(product.get("reason") or "")
            if reason:
                reason_html = f"<details><summary>Why?</summary><p>{reason}</p></details>"
            else:
                reason_html = ""
            rows.append(
                "<tr>"
                f"<td>{product['product_id']}</td>"
                f"<td>{html.escape(product.get('title') or '')}</td>"
                f"<td>{html.escape(product.get('tags') or '')}</td>"
                f"<td>{confidence_text}</td>"
                f"<td>{reason_html}</td>"
                "</tr>"
            )
        return "\n".join(rows) if rows else "<tr><td colspan=\"5\">No products recorded.</td></tr>"

    index_rows = []
    for synset in synsets:
        synset_id = synset["synset_id"]
        name = synset.get("synset_name") or synset_id
        definition = html.escape(synset.get("synset_definition") or "")
        product_count = synset["product_count"]
        share_pct = synset["share"] * 100 if synset["share"] is not None else 0.0

        index_rows.append(
            "<tr>"
            f"<td><a href=\"{synset_id}.html\">{html.escape(name)}</a></td>"
            f"<td>{html.escape(synset_id)}</td>"
            f"<td>{product_count:,}</td>"
            f"<td>{share_pct:.1f}%</td>"
            f"<td>{definition}</td>"
            "</tr>"
        )

        products_html = _product_rows(synset.get("products", []))
        synset_page = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{html.escape(name)} — WordNet synset</title>
  <link rel=\"stylesheet\" href=\"../assets/styles.css\" />
</head>
<body class=\"synset-page\">
  <header class=\"synset-hero\">
    <p><a href=\"../index.html\">&larr; Back to overview</a></p>
    <h1>{html.escape(name)}</h1>
    <p class=\"synset-id\">{html.escape(synset_id)}</p>
    <p class=\"synset-definition\">{definition or 'Definition unavailable.'}</p>
    <p class=\"synset-count\">{product_count:,} product{'s' if product_count != 1 else ''} mapped here.</p>
  </header>
  <main class=\"synset-content\">
    <table class=\"synset-products\">
      <thead>
        <tr>
          <th>ID</th>
          <th>Title</th>
          <th>Tags</th>
          <th>Confidence</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {products_html}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
        (synsets_dir / f"{synset_id}.html").write_text(synset_page, encoding="utf-8")

    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>WordNet synsets overview</title>
  <link rel=\"stylesheet\" href=\"../assets/styles.css\" />
</head>
<body class=\"synset-page\">
  <header class=\"synset-hero\">
    <p><a href=\"../index.html\">&larr; Back to overview</a></p>
    <h1>All processed synsets</h1>
    <p class=\"synset-definition\">Browse every WordNet concept currently linked to Shopify products.</p>
  </header>
  <main class=\"synset-content\">
    <table class=\"synset-products\">
      <thead>
        <tr>
          <th>Synset</th>
          <th>ID</th>
          <th>Products</th>
          <th>Share</th>
          <th>Definition</th>
        </tr>
      </thead>
      <tbody>
        {''.join(index_rows)}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
    (synsets_dir / "index.html").write_text(index_html, encoding="utf-8")


def build_site(
    csv_path: Path,
    output_dir: Path,
    *,
    precomputed_database: Optional[Path] = None,
    tasks_db: Optional[Path] = None,
    synset_db: Optional[Path] = None,
) -> Dict[str, Any]:
    csv_path = csv_path.resolve()
    _ensure_clean_directory(output_dir)

    assets_dir = output_dir / "assets"
    downloads_dir = output_dir / "downloads"
    datadumps_dir = output_dir / "datadumps"
    for path in (assets_dir, downloads_dir, datadumps_dir):
        path.mkdir(parents=True, exist_ok=True)

    db_path = downloads_dir / "battles.sqlite"

    if precomputed_database is not None:
        source = precomputed_database.resolve()
        destination = db_path.resolve()
        if source != destination:
            if db_path.exists():
                db_path.unlink()
            shutil.copyfile(source, db_path)
        else:
            # The precomputed database already matches the expected output path.
            if not db_path.exists():
                shutil.copyfile(source, db_path)
    else:
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
.leaderboard-bottom {padding: 0 1rem 2.5rem;}
.leaderboard-bottom h3 {margin: 0 0 0.5rem; font-size: 1.35rem;}
.leaderboard-bottom p {margin: 0 0 1rem; color: #475569;}
.leaderboard-bottom-table tbody tr:nth-child(even) {background: #f1f5f9;}
.chart {text-align: center; padding: 0 1rem 2rem;}
.chart img {max-width: 100%; height: auto; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.1);}
.synset-progress {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); margin-top: 2rem; padding-bottom: 2rem;}
.synset-progress .synset-header {padding: 2rem 1.5rem 0;}
.synset-progress .synset-header p {margin: 0.5rem 0 0; color: #475569;}
.synset-progress .synset-top {padding: 0 1.5rem 1.5rem;}
.synset-progress .synset-top h3 {margin-bottom: 0.5rem;}
.synset-progress .synset-top a {color: #0b6ce3; font-weight: 600; text-decoration: none;}
.synset-progress .synset-top a:hover {text-decoration: underline;}
.synset-eta {padding: 1rem 1.5rem 0; text-align: center; font-style: italic; color: #475569;}
.synset-not-found {padding: 0 1.5rem 1.5rem;}
.synset-not-found h4 {margin-bottom: 0.5rem;}
.synset-not-found ul {margin: 0; padding-left: 1.25rem; color: #475569;}
.synset-usage {padding: 0 1.5rem 1.5rem;}
.synset-usage h3 {margin-bottom: 0.5rem;}
.synset-usage p {margin: 0 0 0.75rem; color: #475569;}
.synset-usage-table {width: 100%; border-collapse: collapse; background: white;}
.synset-usage-table th, .synset-usage-table td {padding: 0.6rem 0.75rem; border-bottom: 1px solid #e5e7eb; text-align: left;}
.synset-usage-table thead {background: #f8fafc;}
.synset-usage-table tbody tr:nth-child(even) {background: #f8fafc;}
.synset-table {width: 100%; border-collapse: collapse; background: white;}
.synset-table th, .synset-table td {padding: 0.75rem 1rem; border-bottom: 1px solid #e5e7eb; text-align: left;}
.synset-table thead {background: #f8fafc;}
.synset-table tbody tr:nth-child(even) {background: #f8fafc;}
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
body.synset-page {font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; margin: 0; color: #1f2937; background: #f7f7fb;}
.synset-hero {background: linear-gradient(135deg, #6366f1, #8b5cf6); color: white; padding: 2.5rem 1.5rem; text-align: center;}
.synset-hero a {color: white; opacity: 0.85; text-decoration: none;}
.synset-hero a:hover {opacity: 1; text-decoration: underline;}
.synset-hero h1 {margin: 0.5rem 0 0; font-size: 2.25rem;}
.synset-hero .synset-id {font-family: 'Fira Code', 'SFMono-Regular', Consolas, monospace; margin: 0.5rem 0; opacity: 0.85;}
.synset-hero .synset-count {margin-top: 0.5rem; font-weight: 600;}
.synset-content {max-width: 70rem; margin: 0 auto; padding: 2rem 1.5rem 3rem;}
table.synset-products {width: 100%; border-collapse: collapse; background: white; border-radius: 1rem; overflow: hidden; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);}
table.synset-products th, table.synset-products td {padding: 0.75rem 1rem; border-bottom: 1px solid #e5e7eb; text-align: left; vertical-align: top;}
table.synset-products thead {background: #f1f5f9;}
table.synset-products tbody tr:nth-child(even) {background: #f8fafc;}
.synset-products details {color: #334155;}
@media (max-width: 700px) {.metrics {flex-direction: column;} header.hero {padding: 2.5rem 1rem;} header.hero h1 {font-size: 2rem;} .synset-content {padding: 1.5rem 1rem 2rem;}}
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

    experiments_summary: Optional[Dict[str, Any]] = None
    if tasks_db is not None and tasks_db.exists():
        experiments_summary = experiments.task_status(tasks_db)

    synset_summary: Optional[Dict[str, Any]] = None
    if synset_db is not None:
        synset_summary = _collect_synset_progress(synset_db, stats.get("products", 0))
        if synset_summary:
            _build_synset_pages(output_dir, synset_summary)

    _build_index_html(
        output_dir,
        stats,
        leaderboard,
        chart_path,
        artifact_links,
        csv_path,
        experiments_summary,
        synset_summary,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_path),
        "stats": stats,
        "artifacts": {label: str(path.relative_to(output_dir)) for label, path in artifact_links.items()},
        "experiments": experiments_summary,
        "synsets": synset_summary,
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
        "--precomputed-database",
        type=Path,
        default=None,
        help="Optional precomputed battles SQLite database",
    )
    parser.add_argument(
        "--tasks-db",
        type=Path,
        default=None,
        help="Optional experiments task database for progress reporting",
    )
    parser.add_argument(
        "--synset-db",
        type=Path,
        default=None,
        help="Optional SQLite database with product synset classifications",
    )
    args = parser.parse_args()

    build_site(
        args.csv,
        args.output,
        precomputed_database=args.precomputed_database,
        tasks_db=args.tasks_db,
        synset_db=args.synset_db,
    )


if __name__ == "__main__":
    main()
