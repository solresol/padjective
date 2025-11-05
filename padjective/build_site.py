"""Build a static website showcasing tag rankings and taxonomy progress."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row

from . import data_access, db, display, ranking, tagbattle


def _format_padic_expansion(value: int, base: int) -> tuple[str, str]:
    """Return both taxonomy_path and base-expansion for ``value``.

    Returns:
        tuple: (taxonomy_path, expansion) where:
            - taxonomy_path: least-significant-first dotted notation (e.g., "1.1.10.2.8")
            - expansion: most-significant-first with algebraic expression
    """

    if base <= 1:
        return (f"{value}", f"{value} ({value})")

    sign = "-" if value < 0 else ""
    n = abs(value)
    digits: list[int] = []
    while n:
        digits.append(n % base)
        n //= base
    if not digits:
        digits = [0]

    # taxonomy_path: least-significant-first (natural p-adic order)
    taxonomy_path = ".".join(str(d) for d in digits)
    if sign:
        taxonomy_path = f"-{taxonomy_path}"

    # Base expansion: most-significant-first for readability
    digits_msd = list(reversed(digits))
    digits_str = ".".join(str(d) for d in digits_msd)

    terms: list[str] = []
    for idx, digit in enumerate(digits_msd):
        if digit == 0:
            continue
        power = len(digits_msd) - idx - 1
        if power == 0:
            term = str(digit)
        elif digit == 1:
            term = str(base ** power)
        elif power == 1:
            term = f"{digit}*{base}"
        else:
            term = f"{digit}*{base}^{power}"
        terms.append(term)

    expression = " + ".join(terms) if terms else "0"
    if sign:
        digits_str = f"-{digits_str}"
        if expression != "0":
            expression = f"-({expression})"

    expansion = f"{digits_str} ({expression})"
    return (taxonomy_path, expansion)


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


def _build_tag_rank_lookup(leaderboard: pd.DataFrame) -> Dict[str, int]:
    """Return a lookup of tagbattle rankings keyed by uppercase tag."""

    if leaderboard is None or leaderboard.empty:
        return {}

    sorted_board = (
        leaderboard.sort_values("score", ascending=False, kind="mergesort")
        .drop_duplicates(subset="tag", keep="first")
        .reset_index(drop=True)
    )

    return {str(row["tag"]).upper(): int(idx) + 1 for idx, row in sorted_board.iterrows()}


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


def _write_defective_taxonomy_page(output_dir: Path, dataset: data_access.ProductDataset) -> Path:
    """Create a dedicated page showing products with defective taxonomy labels."""
    page_path = output_dir / "defective_taxonomy.html"

    defective_rows: list[dict[str, Any]] = []
    for discarded in dataset.discarded_products:
        if discarded.reason == "defective_taxonomy_path":
            record = discarded.record
            defective_rows.append(
                {
                    "product_id": record.product_id,
                    "title": record.title,
                    "taxonomy_id": record.taxonomy_id or "",
                    "taxonomy_path": record.taxonomy_path or "",
                    "taxonomy_name": record.taxonomy_name or "",
                    "tags": ", ".join(record.tags),
                }
            )

    if defective_rows:
        defective_df = pd.DataFrame(defective_rows)
        defective_table = defective_df.to_html(index=False, classes=["dataset-table", "defective-table"])
        count_msg = f"<p>{len(defective_rows)} products have defective taxonomy paths containing hierarchical separators when they should be numeric codes.</p>"
    else:
        defective_table = "<p>No products with defective taxonomy paths found.</p>"
        count_msg = "<p>All products have valid numeric taxonomy paths.</p>"

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Defective taxonomy labels</title>
  <link rel="stylesheet" href="assets/styles.css" />
</head>
<body class="dataset-page">
  <header class="hero">
    <h1>Products with defective taxonomy labels</h1>
    <p class="tagline">Products excluded due to invalid taxonomy paths</p>
  </header>

  <section>
    <p><a href="index.html">← Back to index</a></p>
    <h2>Defective taxonomy paths</h2>
    {count_msg}
    <p>These products have taxonomy_path fields incorrectly populated with hierarchical names
    (like "Apparel &amp; Accessories / Clothing / T-Shirts") instead of the proper numeric
    codes (like "1.1.4"). The taxonomy_path field should contain only numeric hierarchical codes,
    while the human-readable hierarchical names belong in the taxonomy_name field. These products
    are excluded from model training to ensure data quality and consistency.</p>
    <div class="table-wrapper">
      {defective_table}
    </div>
  </section>

  <footer>
    <p><a href="index.html">← Back to index</a> | <a href="dataset.html">View full dataset details</a></p>
  </footer>
</body>
</html>"""

    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_dataset_page(output_dir: Path, dataset: data_access.ProductDataset) -> Path:
    page_path = output_dir / "dataset.html"

    included_df = dataset.metadata.copy()
    included_columns = [
        "product_id",
        "title",
        "taxonomy_name",
        "taxonomy_id",
        "taxonomy_path",
        "tags",
        "tag_count",
        "valid_tag_count",
        "cv_fold",
    ]
    included_df = included_df.reindex(columns=included_columns).fillna("")
    included_table = included_df.to_html(index=False, classes=["dataset-table", "included-table"])

    discarded_rows: list[dict[str, Any]] = []
    for discarded in dataset.discarded_products:
        record = discarded.record
        discarded_rows.append(
            {
                "product_id": record.product_id,
                "title": record.title,
                "taxonomy_name": record.taxonomy_name or "",
                "taxonomy_id": record.taxonomy_id or "",
                "taxonomy_path": record.taxonomy_path or "",
                "tags": ", ".join(record.tags),
                "reason": discarded.reason,
            }
        )
    if discarded_rows:
        discarded_df = pd.DataFrame(discarded_rows)
        discarded_table = discarded_df.to_html(index=False, classes=["dataset-table", "discarded-table"])
    else:
        discarded_table = "<p>No products were discarded by the current preprocessing rules.</p>"

    discarded_tags_rows = [
        {"tag": entry.tag, "count": entry.count}
        for entry in sorted(dataset.discarded_tags, key=lambda e: e.count, reverse=True)
    ]
    if discarded_tags_rows:
        discarded_tags_df = pd.DataFrame(discarded_tags_rows)
        discarded_tags_table = discarded_tags_df.to_html(index=False, classes=["dataset-table", "discarded-tags-table"])
    else:
        discarded_tags_table = "<p>All tags met the minimum frequency requirement.</p>"

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Dataset coverage</title>
  <link rel="stylesheet" href="assets/styles.css" />
</head>
<body class="dataset-page">
  <header class="hero">
    <h1>Training dataset</h1>
    <p class="tagline">Every product, tag, and taxonomy considered by the models</p>
  </header>

  <section>
    <p><a href="index.html">← Back to index</a></p>
    <h2>Products used for training</h2>
    <p>The table lists every product that survived preprocessing along with its taxonomy details.</p>
    <div class="table-wrapper">
      {included_table}
    </div>
  </section>

  <section>
    <h2>Discarded products</h2>
    <p>Products can be dropped when they lack taxonomy information or when their taxonomy has fewer than the minimum required samples.</p>
    <div class="table-wrapper">
      {discarded_table}
    </div>
  </section>

  <section>
    <h2>Discarded tags</h2>
    <p>Tags appearing in fewer than the minimum number of products are removed to keep the feature matrix manageable.</p>
    <div class="table-wrapper">
      {discarded_tags_table}
    </div>
  </section>

  <footer>
    <p><a href="index.html">← Back to index</a> | <a href="defective_taxonomy.html">View defective taxonomy labels</a></p>
  </footer>
</body>
</html>"""

    page_path.write_text(page_html, encoding="utf-8")
    return page_path


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

    # Calculate mean loss per fold from predictions
    for metric in metrics:
        fold = metric["cv_fold"]
        fold_predictions = predictions.get(fold, [])
        if fold_predictions:
            mean_loss = sum(p["loss"] for p in fold_predictions) / len(fold_predictions)
            metric["mean_loss"] = mean_loss
        else:
            metric["mean_loss"] = 0.0

    # Load taxonomy name mappings for display
    taxonomy_names: Dict[str, str] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT taxonomy_path, taxonomy_name
            FROM cantbuymelove.taxonomy
            WHERE taxonomy_path IS NOT NULL
              AND taxonomy_name IS NOT NULL
              AND taxonomy_path !~ '[>/|]'
            """
        )
        for row in cur:
            path = row.get("taxonomy_path")
            name = row.get("taxonomy_name")
            if path and name:
                taxonomy_names[str(path)] = str(name)

    return {
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
        "taxonomy_names": taxonomy_names,
    }


def _load_prediction_details(conn, fold: int, schema: str) -> Dict[int, Dict[str, Any]]:
    """Load detailed prediction data for a specific fold.

    Returns a dict mapping product_id to prediction details including:
    - ground_truth: {taxonomy_id, taxonomy_name, taxonomy_path}
    - predictions: {umllr: {...}, lr: {...}, nn: {...}}
    - tags: [list of tags]
    - umllr_coefficients: {tag: coefficient}
    - lr_coefficients: {tag: {taxonomy_id: coefficient}}
    """
    details: Dict[int, Dict[str, Any]] = {}

    # Load taxonomy name mappings
    taxonomy_info: Dict[str, Dict[str, str]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT taxonomy_id, taxonomy_name, taxonomy_path
            FROM cantbuymelove.taxonomy
            WHERE taxonomy_path !~ '[>/|]'
            """
        )
        for row in cur:
            tid = row["taxonomy_id"]
            taxonomy_info[tid] = {
                "taxonomy_id": tid,
                "taxonomy_name": row["taxonomy_name"] or "",
                "taxonomy_path": row["taxonomy_path"] or "",
            }

    # Load umllr predictions and coefficients
    umllr_predictions: Dict[int, Dict[str, Any]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT product_id, true_value, predicted_value, loss
                FROM {schema}.umllr_predictions
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            pid = row["product_id"]
            umllr_predictions[pid] = {
                "true_value": row["true_value"],
                "predicted_value": row["predicted_value"],
                "loss": row["loss"],
            }

    # Load umllr coefficients for this fold
    umllr_tag_coeffs: Dict[str, int] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tag, coefficient
                FROM {schema}.umllr_tag_coefficients
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            umllr_tag_coeffs[row["tag"]] = row["coefficient"]

    # Load LR predictions
    lr_predictions: Dict[int, Dict[str, Any]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT product_id, true_taxonomy_id, predicted_taxonomy_id, loss
                FROM {schema}.taxonomy_lr_predictions
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            pid = row["product_id"]
            lr_predictions[pid] = {
                "true_taxonomy_id": row["true_taxonomy_id"],
                "predicted_taxonomy_id": row["predicted_taxonomy_id"],
                "loss": row["loss"],
            }

    # Load LR coefficients for this fold
    lr_coeffs: Dict[str, Dict[str, float]] = {}  # tag -> {taxonomy_id: coef}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT taxonomy_id, tag, coefficient
                FROM {schema}.taxonomy_lr_coefficients
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            tag = row["tag"]
            tid = row["taxonomy_id"]
            coef = row["coefficient"]
            if tag not in lr_coeffs:
                lr_coeffs[tag] = {}
            lr_coeffs[tag][tid] = coef

    # Load NN predictions
    nn_predictions: Dict[int, Dict[str, Any]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT product_id, true_taxonomy_id, predicted_taxonomy_id, loss
                FROM {schema}.taxonomy_nn_predictions
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            pid = row["product_id"]
            nn_predictions[pid] = {
                "true_taxonomy_id": row["true_taxonomy_id"],
                "predicted_taxonomy_id": row["predicted_taxonomy_id"],
                "loss": row["loss"],
            }

    # Load product tags and ground truth
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT
                p.id AS product_id,
                p.product_title AS title,
                pd.product_detail->'product'->>'tags' AS tags,
                pt.taxonomy_id,
                up.cv_fold
            FROM cantbuymelove.product p
            JOIN public.product_details pd ON (
                p.myshopify_domain = pd.myshopify_domain
                AND p.run_name = pd.run_name
                AND p.product_handle = pd.product_handle
            )
            LEFT JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
            LEFT JOIN padjective.umllr_predictions up ON up.product_id = p.id
            WHERE up.cv_fold = %s
            """,
            (fold,)
        )
        for row in cur:
            pid = row["product_id"]
            taxonomy_id = row["taxonomy_id"]
            tags_str = row["tags"] or ""
            tags = [t.strip().upper() for t in tags_str.split(",") if t.strip()]

            details[pid] = {
                "product_id": pid,
                "title": row["title"] or "",
                "tags": tags,
                "ground_truth": taxonomy_info.get(taxonomy_id, {}),
                "predictions": {
                    "umllr": umllr_predictions.get(pid, {}),
                    "lr": lr_predictions.get(pid, {}),
                    "nn": nn_predictions.get(pid, {}),
                },
                "umllr_coefficients": {tag: umllr_tag_coeffs.get(tag, 0) for tag in tags},
                "lr_coefficients": {tag: lr_coeffs.get(tag, {}) for tag in tags},
            }

    return details


def _write_zero_coefficients_page(
    page_path: Path,
    fold: int,
    zero_coeff_tags: list[str],
    tag_rankings: Dict[str, int],
) -> None:
    """Write a separate page for zero coefficient tags."""
    zero_rows: list[str] = []
    for tag in zero_coeff_tags:
        rank_value = tag_rankings.get(tag.upper())
        rank_label = str(rank_value) if rank_value is not None else "unranked"
        zero_rows.append(
            "<tr>"
            f"<td>{html.escape(tag)}</td>"
            f"<td>{rank_label}</td>"
            "</tr>"
        )

    zero_table_rows = "\n".join(zero_rows)

    page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>umllr fold {fold} - Zero coefficients</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>umllr fold {fold} - Zero coefficients</h1>
    <p><a href="fold_{fold}.html">Back to fold {fold}</a> | <a href="../index.html">Back to index</a></p>
    <p>Tags with zero coefficients ({len(zero_coeff_tags)} total)</p>
    <table class="umllr-table">
      <thead>
        <tr><th>Tag</th><th>Tag Battle Ranking</th></tr>
      </thead>
      <tbody>
        {zero_table_rows}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
    page_path.write_text(page_contents, encoding="utf-8")


def _write_umllr_pages(output_dir: Path, summary: Dict[str, Any]) -> Dict[int, Path]:
    pages: Dict[int, Path] = {}
    metrics = summary.get("metrics", [])
    if not metrics:
        return pages

    umllr_dir = output_dir / "umllr"
    umllr_dir.mkdir(parents=True, exist_ok=True)

    coefficients = summary.get("coefficients", {})
    predictions = summary.get("predictions", {})
    tag_rankings: Dict[str, int] = summary.get("tag_rankings", {})
    taxonomy_names: Dict[str, str] = summary.get("taxonomy_names", {})

    for metric in metrics:
        fold = metric["cv_fold"]
        coeff_rows = coefficients.get(fold, [])
        prediction_rows = predictions.get(fold, [])
        prime_base = metric.get("prime_base", 0)

        non_zero_rows: list[str] = []
        zero_coeff_tags: list[str] = []
        for row in coeff_rows:
            coefficient = row["coefficient"]
            tag = row["tag"]
            if coefficient == 0:
                zero_coeff_tags.append(tag)
                continue
            taxonomy_path, expansion = _format_padic_expansion(coefficient, prime_base)
            # Reverse taxonomy_path to match database format for lookup
            # taxonomy_path is "1.1.10.2.8" (least-significant-first)
            # database has "8.2.10.1.1" (most-significant-first)
            reversed_path = ".".join(reversed(taxonomy_path.split(".")))
            taxonomy_name = taxonomy_names.get(reversed_path, "")
            non_zero_rows.append(
                "<tr>"
                f"<td>{html.escape(tag)}</td>"
                f"<td>{coefficient}</td>"
                f"<td>{html.escape(taxonomy_path)}</td>"
                f"<td>{html.escape(expansion)}</td>"
                f"<td>{html.escape(taxonomy_name)}</td>"
                f"<td>{row['sequence']}</td>"
                "</tr>"
            )

        coeff_table_rows = "\n".join(non_zero_rows)
        if not coeff_table_rows:
            coeff_table_rows = (
                '<tr><td colspan="6">No non-zero coefficients recorded for this fold.</td></tr>'
            )

        # Create separate page for zero coefficients if any exist
        zero_coeff_link = ""
        if zero_coeff_tags:
            zero_page_path = umllr_dir / f"fold_{fold}_zero_coefficients.html"
            _write_zero_coefficients_page(
                zero_page_path,
                fold,
                zero_coeff_tags,
                tag_rankings
            )
            zero_coeff_link = (
                f"\n    <p class=\"zero-coefficients\"><strong>Zero coefficients:</strong> "
                f"{len(zero_coeff_tags)} tags &middot; "
                f'<a href="fold_{fold}_zero_coefficients.html">View zero coefficient tags</a></p>'
            )

        expansion_header = (
            f"Base-{prime_base} expansion" if prime_base and prime_base > 1 else "Base expansion"
        )

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

        mean_loss = metric.get("mean_loss", 0)
        num_predictions = len(prediction_rows)

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
    <p><strong>P-adic loss (mean):</strong> {mean_loss:.6f} &middot; <strong>Test samples:</strong> {num_predictions:,} &middot; <strong>Prime base:</strong> {metric['prime_base']} &middot; <strong>Max digit:</strong> {metric['max_digit']}</p>
    <h2>Tag coefficients</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Tag</th><th>Coefficient</th><th>taxonomy_path</th><th>{expansion_header}</th><th>Taxonomy Name</th><th>Tag Battle Ranking</th></tr>
      </thead>
      <tbody>
        {coeff_table_rows}
      </tbody>
    </table>
{zero_coeff_link}
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




def _build_trends_section(trends_chart_path: Optional[Path], output_dir: Path) -> str:
    """Build HTML section for historical trends chart."""
    if not trends_chart_path:
        return ""

    chart_rel_path = trends_chart_path.relative_to(output_dir).as_posix()
    return f"""
  <section style="max-width: 70rem; margin: 2rem auto; background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); padding: 2rem 1.5rem;">
    <h2 style="margin-top: 0;">Historical Performance Trends</h2>
    <p style="color: #64748b; margin-bottom: 1.5rem;">
      Tracking model performance and dataset growth over time. Lower p-adic loss indicates better predictions.
    </p>
    <figure class="chart">
      <img src="{chart_rel_path}" alt="Historical model performance trends" />
    </figure>
  </section>"""


def _build_index_html(
    output_dir: Path,
    stats: Dict[str, int],
    dataset_stats: Dict[str, int],
    dataset_page: Optional[Path],
    defective_taxonomy_page: Optional[Path],
    elo_page: Path,
    taxonomy_page: Optional[Path],
    umllr_page: Optional[Path],
    taxonomy_nn_page: Optional[Path],
    taxonomy_summary: Optional[Dict[str, Any]] = None,
    umllr_summary: Optional[Dict[str, Any]] = None,
    taxonomy_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_nn_fold_results: Optional[list[Dict[str, Any]]] = None,
    trends_chart_path: Optional[Path] = None,
) -> None:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    discard_note = ""
    discarded_products = dataset_stats.get("discarded_products")
    if discarded_products:
        discard_note = (
            f"<span class=\"discard-note\">{discarded_products:,} products were discarded due to "
            "missing or sparse taxonomy labels.</span>"
        )

    dataset_link = ""
    if dataset_page:
        dataset_link = (
            f'<a href="{dataset_page.relative_to(output_dir).as_posix()}">Explore the full dataset →</a>'
        )
    if defective_taxonomy_page:
        if dataset_link:
            dataset_link += " | "
        dataset_link += (
            f'<a href="{defective_taxonomy_page.relative_to(output_dir).as_posix()}">View defective taxonomy labels →</a>'
        )

    # Build model cards
    taxonomy_card = ""
    if taxonomy_summary and taxonomy_page:
        # Prefer showing p-adic loss from fold results if available
        if taxonomy_fold_results:
            avg_padic_loss = sum(fold["padic_loss_mean"] for fold in taxonomy_fold_results) / len(taxonomy_fold_results)
            metric_display = f"{avg_padic_loss:.4f}"
            metric_label = "Avg p-adic loss"
        else:
            stats_block = taxonomy_summary.get("stats", {})
            cv_info = stats_block.get("cross_validation") or {}
            cv_accuracy = cv_info.get("mean_accuracy")
            if cv_accuracy is not None:
                metric_display = f"{cv_accuracy * 100:.1f}%"
            else:
                metric_display = "—"
            metric_label = "CV accuracy"

        taxonomy_card = f"""
  <div class="model-card">
    <h3>Taxonomy Classifier</h3>
    <p>Logistic regression model predicting Shopify taxonomy from tags</p>
    <div class="card-metric">
      <span class="value">{metric_display}</span>
      <span class="label">{metric_label}</span>
    </div>
    <a href="{taxonomy_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>
  </div>"""

    umllr_card = ""
    if umllr_summary and umllr_page and umllr_summary.get("metrics"):
        metrics = umllr_summary.get("metrics", [])
        # Use mean_loss (per-prediction average) not total loss
        avg_loss = sum(m["mean_loss"] for m in metrics) / len(metrics) if metrics else 0
        umllr_card = f"""
  <div class="model-card">
    <h3>umllr P-adic Regression</h3>
    <p>P-adic coefficients assigned to tags to predict taxonomy</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>
    <a href="{umllr_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>
  </div>"""

    nn_card = ""
    taxonomy_nn_link = ""
    if taxonomy_nn_page:
        taxonomy_nn_link = (
            f'<a href="{taxonomy_nn_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>'
        )

    if taxonomy_nn_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_nn_fold_results) / len(taxonomy_nn_fold_results)
        nn_card = f"""
  <div class="model-card">
    <h3>Neural Network Classifier</h3>
    <p>PyTorch neural network predicting taxonomy from tags</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>
    {taxonomy_nn_link or '<span class="card-link disabled">No report available</span>'}
  </div>"""

    # Combine model cards
    all_cards: list[str] = []
    if umllr_card:
        all_cards.append(umllr_card)
    if nn_card:
        all_cards.append(nn_card)
    if taxonomy_card:
        all_cards.append(taxonomy_card)
    elo_card = f"""
  <div class="model-card">
    <h3>ELO-Inspired Rankings</h3>
    <p>Battle-tested tag hierarchy from product title positions</p>
    <div class="card-metric">
      <span class="value">{stats.get('battles', 0):,}</span>
      <span class="label">Tag battles</span>
    </div>
    <a href="{elo_page.relative_to(output_dir).as_posix()}" class="card-link">View rankings →</a>
  </div>"""
    all_cards.append(elo_card)
    models_grid = "\n".join(all_cards)

    taxonomy_overview_html = ""
    if taxonomy_summary:
        class_distribution = taxonomy_summary.get("class_distribution", [])[:5]
        top_tags = taxonomy_summary.get("top_tags", [])[:5]

        class_rows = "\n".join(
            f"<tr><td>{html.escape(row.get('taxonomy_path') or 'Unknown')}</td><td>{row.get('sample_count', 0):,}</td><td>{row.get('sample_fraction', 0.0) * 100:.1f}%</td></tr>"
            for row in class_distribution
        )
        if not class_rows:
            class_rows = '<tr><td colspan="3">No taxonomy class data available</td></tr>'

        tag_rows = "\n".join(
            f"<tr><td>{html.escape(row.get('tag') or '')}</td><td>{html.escape(row.get('top_taxonomy_path') or 'Unknown')}</td><td>{row.get('top_weight', 0.0):.3f}</td></tr>"
            for row in top_tags
        )
        if not tag_rows:
            tag_rows = '<tr><td colspan="3">No tag signal data available</td></tr>'

        taxonomy_overview_html = f"""
  <section class="taxonomy-classifier">
    <h2>Taxonomy classifier highlights</h2>
    <div class="taxonomy-layout">
      <div class="taxonomy-card">
        <h3>Largest taxonomy classes</h3>
        <table class="taxonomy-table">
          <thead><tr><th>Path</th><th>Samples</th><th>Share</th></tr></thead>
          <tbody>{class_rows}</tbody>
        </table>
      </div>
      <div class="taxonomy-card">
        <h3>Tags with strongest signals</h3>
        <table class="taxonomy-table">
          <thead><tr><th>Tag</th><th>Top taxonomy</th><th>Weight</th></tr></thead>
          <tbody>{tag_rows}</tbody>
        </table>
      </div>
    </div>
  </section>"""

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Padjective Tag Hierarchy</title>
  <link rel="stylesheet" href="assets/styles.css" />
  <style>
.model-cards {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(20rem, 1fr));
  gap: 1.5rem;
  padding: 2rem 1.5rem;
  max-width: 70rem;
  margin: 0 auto;
}}
.model-card {{
  background: white;
  border-radius: 1rem;
  padding: 2rem;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
  display: flex;
  flex-direction: column;
}}
.model-card h3 {{
  margin: 0 0 0.5rem;
  font-size: 1.5rem;
  color: #0b6ce3;
}}
.model-card p {{
  margin: 0 0 1.5rem;
  color: #64748b;
  flex-grow: 1;
}}
.model-card .card-metric {{
  text-align: center;
  padding: 1rem;
  background: #f8fafc;
  border-radius: 0.75rem;
  margin-bottom: 1.5rem;
}}
.model-card .card-metric .value {{
  display: block;
  font-size: 2rem;
  font-weight: 700;
  color: #0b6ce3;
}}
.model-card .card-metric .label {{
  display: block;
  font-size: 0.9rem;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 0.25rem;
}}
.model-card .card-link {{
  display: inline-block;
  color: #0b6ce3;
  text-decoration: none;
  font-weight: 600;
  padding: 0.75rem 1.5rem;
  background: #eff6ff;
  border-radius: 0.5rem;
  text-align: center;
  transition: background 0.2s;
}}
.model-card .card-link.disabled {{
  color: #94a3b8;
  background: #e2e8f0;
  cursor: default;
  pointer-events: none;
}}
.model-card .card-link:hover {{
  background: #dbeafe;
}}
.hero .data-note {{
  margin-top: 0.75rem;
  color: #475569;
  font-size: 0.95rem;
}}
.hero .data-note a {{
  color: #0b6ce3;
  text-decoration: none;
  font-weight: 600;
}}
.hero .data-note a:hover {{
  text-decoration: underline;
}}
  </style>
</head>
<body>
  <header class="hero">
    <h1>Padjective Tag Hierarchy</h1>
    <p class="tagline">Machine learning insights into Shopify product tag organization</p>
    <p class="data-note">Data sourced from <a href="https://cantbuymelove.industrial-linguistics.com/">cantbuymelove.industrial-linguistics.com</a> powering Shopify taxonomy classification and filtered to taxonomies with at least five products.</p>
    <p class="timestamp">Last updated {generated}</p>
  </header>

  <section class="metrics">
    <div class="metric">
      <span class="value">{dataset_stats.get('products', 0):,}</span>
      <span class="label">Products used</span>
    </div>
    <div class="metric">
      <span class="value">{dataset_stats.get('taxonomies', 0):,}</span>
      <span class="label">Taxonomies covered</span>
    </div>
    <div class="metric">
      <span class="value">{dataset_stats.get('unique_tags', 0):,}</span>
      <span class="label">Distinct tags</span>
    </div>
    <div class="metric">
      <span class="value">{stats.get('battles', 0):,}</span>
      <span class="label">Tag battles</span>
    </div>
  </section>

  <section class="dataset-overview">
    <h2>Dataset coverage</h2>
    <p>
      Training data spans <strong>{dataset_stats.get('products', 0):,}</strong> products across
      <strong>{dataset_stats.get('taxonomies', 0):,}</strong> taxonomies and
      <strong>{dataset_stats.get('unique_tags', 0):,}</strong> tags.
      {discard_note}
      {dataset_link}
    </p>
  </section>

  <div class="model-cards">
{models_grid}
  </div>

  {taxonomy_overview_html}

  {_build_trends_section(trends_chart_path, output_dir)}

  <footer>
    <p>Source available on <a href="https://github.com/IFost-Sydney-Uni/padjective">GitHub</a></p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_document, encoding="utf-8")


def _write_elo_rankings_page(
    output_dir: Path,
    leaderboard: pd.DataFrame,
    chart_path: Path,
    stats: Dict[str, int],
) -> Path:
    """Write a dedicated page for ELO rankings."""
    elo_dir = output_dir / "elo"
    elo_dir.mkdir(parents=True, exist_ok=True)

    top_table = leaderboard.head(20).to_html(index=False, classes="leaderboard")
    bottom_table = (
        leaderboard.sort_values("score", ascending=True)
        .head(20)
        .to_html(index=False, classes=["leaderboard", "leaderboard-bottom-table"])
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ELO-Inspired Tag Rankings</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>ELO-Inspired Tag Rankings</h1>
    <p class="tagline">Battle-tested tag hierarchy from product titles</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>How it works</h2>
    <p>Tags that appear <strong>later (rightmost position)</strong> in product titles consistently win battles against tags that appear earlier. We use an ELO-style ranking system to score each tag based on its positional dominance.</p>

    <div class="metrics">
      <div class="metric">
        <span class="value">{stats.get('battles', 0):,}</span>
        <span class="label">Tag battles recorded</span>
      </div>
      <div class="metric">
        <span class="value">{stats.get('components', 0):,}</span>
        <span class="label">Connected components</span>
      </div>
    </div>

    <h2>Top contenders</h2>
    <div class="leaderboard-table">
      {top_table}
    </div>

    <figure class="chart">
      <img src="../assets/{chart_path.name}" alt="Top tags bar chart" />
      <figcaption>Top 20 tags by inferred depth.</figcaption>
    </figure>

    <h2>Biggest losers</h2>
    <p>Tags that our model predicts are most likely to be pushed to the end of product titles.</p>
    <div class="leaderboard-table">
      {bottom_table}
    </div>
  </section>

  <footer>
    <p><a href="../index.html">← Back to index</a></p>
  </footer>
</body>
</html>"""

    page_path = elo_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_taxonomy_lr_fold_pages(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
) -> Dict[int, Path]:
    """Write individual pages for each taxonomy classifier fold."""
    pages: Dict[int, Path] = {}
    if not fold_results:
        return pages

    tax_dir = output_dir / "taxonomy_classifier"
    tax_dir.mkdir(parents=True, exist_ok=True)

    for fold_data in fold_results:
        fold = fold_data["cv_fold"]

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Taxonomy Classifier Fold {fold} Results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>Taxonomy Classifier Fold {fold}</h1>
    <p><a href="index.html">Back to taxonomy classifier overview</a> &middot; <a href="../index.html">Back to main index</a></p>

    <h2>Fold metrics</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>Test accuracy</td><td>{fold_data['test_accuracy'] * 100:.2f}%</td></tr>
        <tr><td>Test F1 score</td><td>{fold_data['test_f1']:.4f}</td></tr>
        <tr><td>Hierarchical loss</td><td>{fold_data['test_hierarchical_loss']:.6f}</td></tr>
        <tr><td>P-adic loss (total)</td><td>{fold_data['padic_loss_total']:.6f}</td></tr>
        <tr><td>P-adic loss (mean)</td><td>{fold_data['padic_loss_mean']:.6f}</td></tr>
        <tr><td>Prime base</td><td>{fold_data['prime_base']}</td></tr>
        <tr><td>Training samples</td><td>{fold_data['num_train_samples']:,}</td></tr>
        <tr><td>Test samples</td><td>{fold_data['num_test_samples']:,}</td></tr>
        <tr><td>Trained at</td><td>{html.escape(fold_data['trained_at'] or 'Unknown')}</td></tr>
      </tbody>
    </table>

    <h2>About p-adic loss</h2>
    <p>P-adic loss measures the distance between predicted and true taxonomy using p-adic metric (base {fold_data['prime_base']}). Lower values indicate closer predictions in the taxonomy hierarchy. This metric is shared with the umllr model for comparison.</p>
  </section>
</body>
</html>
"""

        page_path = tax_dir / f"fold_{fold}.html"
        page_path.write_text(page_contents, encoding="utf-8")
        pages[fold] = page_path

    return pages


def _write_umllr_overview_page(
    output_dir: Path,
    umllr_summary: Dict[str, Any],
) -> Path:
    """Write a main overview page for umllr p-adic regression results."""
    umllr_dir = output_dir / "umllr"
    umllr_dir.mkdir(parents=True, exist_ok=True)

    metrics = umllr_summary.get("metrics", [])
    page_lookup = umllr_summary.get("pages", {})

    if metrics:
        # Use mean_loss (per-prediction average) not total loss
        avg_loss = sum(m.get("mean_loss", 0) for m in metrics) / len(metrics)
        prime_base = metrics[0].get("prime_base", 0)
        max_digit = metrics[0].get("max_digit", 0)
    else:
        avg_loss = 0
        prime_base = 0
        max_digit = 0

    fold_rows = []
    for metric in metrics:
        fold = metric["cv_fold"]
        link = page_lookup.get(fold)
        if link:
            link_text = f'<a href="{Path(link).name}">View details →</a>'
        else:
            link_text = "—"
        mean_loss = metric.get("mean_loss", 0)
        fold_rows.append(
            f"<tr>"
            f"<td>{fold}</td>"
            f"<td>{mean_loss:.6f}</td>"
            f"<td>{link_text}</td>"
            f"</tr>"
        )

    table_body = "\n".join(fold_rows) or '<tr><td colspan="3">No cross-validation folds recorded.</td></tr>'

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>umllr P-adic Tag Regression</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>umllr P-adic Tag Regression</h1>
    <p class="tagline">Using p-adic coefficients to predict taxonomy from tags</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Overview</h2>
    <p>The umllr (Universal Machine Learning Linear Regression) model assigns p-adic integer coefficients to product tags and uses them to predict taxonomy encodings. Each taxonomy path is encoded as a p-adic integer (base {prime_base}), and tags are fitted to minimize p-adic distance on training data.</p>

    <div class="metrics">
      <div class="metric">
        <span class="value">{len(metrics)}</span>
        <span class="label">CV folds</span>
      </div>
      <div class="metric">
        <span class="value">{avg_loss:.4f}</span>
        <span class="label">Average p-adic loss</span>
      </div>
      <div class="metric">
        <span class="value">{prime_base}</span>
        <span class="label">Prime base</span>
      </div>
    </div>

    <h2>Cross-validation results</h2>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>P-adic loss (mean)</th><th>Details</th></tr>
      </thead>
      <tbody>
        {table_body}
      </tbody>
    </table>
  </section>

  <footer>
    <p><a href="../index.html">← Back to index</a></p>
  </footer>
</body>
</html>"""

    page_path = umllr_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_taxonomy_classifier_page(
    output_dir: Path,
    taxonomy_summary: Dict[str, Any],
    fold_results: Optional[list[Dict[str, Any]]] = None,
    fold_pages: Optional[Dict[int, Path]] = None,
) -> Path:
    """Write a dedicated page for taxonomy classifier results."""
    tax_dir = output_dir / "taxonomy_classifier"
    tax_dir.mkdir(parents=True, exist_ok=True)

    stats_block = taxonomy_summary.get("stats", {})
    class_distribution = taxonomy_summary.get("class_distribution", [])[:20]
    top_tags_rows = taxonomy_summary.get("top_tags", [])[:30]
    trained_at = taxonomy_summary.get("trained_at", "Unknown")

    # Build summary metrics
    summary_metrics = []
    if (samples := stats_block.get("samples")) is not None:
        summary_metrics.append(f'<div class="metric"><span class="value">{samples:,}</span><span class="label">Training samples</span></div>')
    if (taxonomies := stats_block.get("taxonomies")) is not None:
        summary_metrics.append(f'<div class="metric"><span class="value">{taxonomies:,}</span><span class="label">Taxonomy classes</span></div>')
    if (accuracy := stats_block.get("training_accuracy")) is not None:
        summary_metrics.append(f'<div class="metric"><span class="value">{accuracy * 100:.1f}%</span><span class="label">Training accuracy</span></div>')

    cv_info = stats_block.get("cross_validation") or {}
    if (mean_acc := cv_info.get("mean_accuracy")) is not None:
        std_acc = cv_info.get("std_accuracy", 0)
        summary_metrics.append(f'<div class="metric"><span class="value">{mean_acc * 100:.1f}% ± {std_acc * 100:.1f}%</span><span class="label">CV accuracy</span></div>')

    metrics_html = "\n".join(summary_metrics)

    # Build fold results table
    fold_results_html = ""
    if fold_results and fold_pages:
        avg_padic_loss = sum(fold["padic_loss_mean"] for fold in fold_results) / len(fold_results)
        fold_rows = []
        for fold_data in fold_results:
            fold = fold_data["cv_fold"]
            link = fold_pages.get(fold)
            if link:
                link_text = f'<a href="{Path(link).name}">View details →</a>'
            else:
                link_text = "—"
            fold_rows.append(
                f"<tr>"
                f"<td>{fold}</td>"
                f"<td>{fold_data['test_accuracy'] * 100:.2f}%</td>"
                f"<td>{fold_data['test_f1']:.4f}</td>"
                f"<td>{fold_data['padic_loss_mean']:.6f}</td>"
                f"<td>{link_text}</td>"
                f"</tr>"
            )
        fold_table_body = "\n".join(fold_rows)
        fold_results_html = f"""
    <h2>Cross-validation fold results</h2>
    <p>Average p-adic loss across all folds: <strong>{avg_padic_loss:.6f}</strong></p>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Test Accuracy</th><th>F1 Score</th><th>P-adic Loss (mean)</th><th>Details</th></tr>
      </thead>
      <tbody>
        {fold_table_body}
      </tbody>
    </table>
"""

    # Build class distribution table
    distribution_rows = "\n".join(
        f"<tr>"
        f"<td>{html.escape(row.get('taxonomy_id') or '')}</td>"
        f"<td>{html.escape(row.get('taxonomy_path') or 'Unknown')}</td>"
        f"<td>{row.get('sample_count', 0):,}</td>"
        f"<td>{row.get('sample_fraction', 0.0) * 100:.2f}%</td>"
        f"</tr>"
        for row in class_distribution
    )
    distribution_body = distribution_rows or '<tr><td colspan="4">No taxonomy data</td></tr>'

    # Build tag table
    tag_rows = "\n".join(
        f"<tr>"
        f"<td>{html.escape(row.get('tag') or '')}</td>"
        f"<td>{html.escape(row.get('top_taxonomy_id') or '')}</td>"
        f"<td>{html.escape(row.get('top_taxonomy_path') or 'Unknown')}</td>"
        f"<td>{row.get('top_weight', 0.0):.4f}</td>"
        f"<td>{row.get('max_abs_weight', 0.0):.4f}</td>"
        f"</tr>"
        for row in top_tags_rows
    )
    tag_body = tag_rows or '<tr><td colspan="5">No tag signals available</td></tr>'

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Taxonomy Classifier (Logistic Regression)</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Taxonomy Classifier</h1>
    <p class="tagline">Predicting Shopify taxonomy from product tags using logistic regression</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model summary</h2>
    <p>This logistic regression model predicts taxonomy IDs from product tags. Trained at: {html.escape(str(trained_at))}</p>

    <div class="metrics">
      {metrics_html}
    </div>

    {fold_results_html}

    <h2>Largest taxonomy classes</h2>
    <table class="taxonomy-table">
      <thead><tr><th>Taxonomy ID</th><th>Path</th><th>Samples</th><th>Share</th></tr></thead>
      <tbody>{distribution_body}</tbody>
    </table>

    <h2>Tags with strongest signal</h2>
    <table class="tag-taxonomy-table">
      <thead><tr><th>Tag</th><th>Taxonomy ID</th><th>Path</th><th>Weight</th><th>Max |weight|</th></tr></thead>
      <tbody>{tag_body}</tbody>
    </table>
  </section>

  <footer>
    <p><a href="../index.html">← Back to index</a></p>
  </footer>
</body>
</html>"""

    page_path = tax_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_taxonomy_nn_page(
    output_dir: Path, fold_results: list[Dict[str, Any]]
) -> Path:
    """Write a report page for the taxonomy neural network classifier."""

    nn_dir = output_dir / "taxonomy_nn_classifier"
    nn_dir.mkdir(parents=True, exist_ok=True)

    num_folds = len(fold_results)
    avg_accuracy = sum(row["test_accuracy"] for row in fold_results) / num_folds
    avg_f1 = sum(row["test_f1"] for row in fold_results) / num_folds
    avg_padic_loss = sum(row["padic_loss_mean"] for row in fold_results) / num_folds

    total_train = sum(row["num_train_samples"] for row in fold_results)
    total_test = sum(row["num_test_samples"] for row in fold_results)

    prime_bases = sorted({row["prime_base"] for row in fold_results})
    hidden_layers = sorted({row["hidden_layers"] for row in fold_results})
    max_tags_values = sorted({row["max_tags"] for row in fold_results if row.get("max_tags") is not None})

    hyperparameters_items = [
        f"<li><strong>Prime base(s):</strong> {', '.join(str(base) for base in prime_bases)}</li>",
        "<li><strong>Hidden layers:</strong> {}</li>".format(
            ", ".join(html.escape(layer) for layer in hidden_layers)
        ),
    ]
    if max_tags_values:
        hyperparameters_items.append(
            f"<li><strong>Max tags used:</strong> {', '.join(str(value) for value in max_tags_values)}</li>"
        )

    fold_rows = []
    for row in fold_results:
        max_tags = row.get("max_tags")
        fold_rows.append(
            """
        <tr>
          <td>{cv_fold}</td>
          <td>{accuracy:.2f}%</td>
          <td>{f1:.4f}</td>
          <td>{hierarchical_loss:.6f}</td>
          <td>{padic_loss:.6f}</td>
          <td>{prime_base}</td>
          <td>{train_samples:,}</td>
          <td>{test_samples:,}</td>
          <td>{hidden_layers}</td>
          <td>{max_tags}</td>
        </tr>
        """.strip().format(
                cv_fold=row["cv_fold"],
                accuracy=row["test_accuracy"] * 100,
                f1=row["test_f1"],
                hierarchical_loss=row["test_hierarchical_loss"],
                padic_loss=row["padic_loss_mean"],
                prime_base=row["prime_base"],
                train_samples=row["num_train_samples"],
                test_samples=row["num_test_samples"],
                hidden_layers=html.escape(row["hidden_layers"]),
                max_tags="—" if max_tags is None else max_tags,
            )
        )

    fold_table_body = "\n".join(fold_rows)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Taxonomy Neural Network Classifier</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Taxonomy Neural Network Classifier</h1>
    <p class="tagline">Cross-validated PyTorch model predicting taxonomy IDs from tags</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model overview</h2>
    <div class="metrics">
      <div class="metric">
        <span class="value">{num_folds}</span>
        <span class="label">CV folds</span>
      </div>
      <div class="metric">
        <span class="value">{avg_accuracy * 100:.2f}%</span>
        <span class="label">Mean accuracy</span>
      </div>
      <div class="metric">
        <span class="value">{avg_f1:.4f}</span>
        <span class="label">Mean F1</span>
      </div>
      <div class="metric">
        <span class="value">{avg_padic_loss:.6f}</span>
        <span class="label">Mean p-adic loss</span>
      </div>
    </div>

    <ul class="taxonomy-stats">
      <li><strong>Total train samples:</strong> {total_train:,}</li>
      <li><strong>Total test samples:</strong> {total_test:,}</li>
    </ul>

    <h3>Key hyperparameters</h3>
    <ul class="taxonomy-stats">
      {''.join(hyperparameters_items)}
    </ul>

    <h2>Cross-validation fold results</h2>
    <table class="taxonomy-table">
      <thead>
        <tr>
          <th>Fold</th>
          <th>Accuracy</th>
          <th>F1 (weighted)</th>
          <th>Hierarchical loss</th>
          <th>P-adic loss (mean)</th>
          <th>Prime base</th>
          <th>Train samples</th>
          <th>Test samples</th>
          <th>Hidden layers</th>
          <th>Max tags</th>
        </tr>
      </thead>
      <tbody>
        {fold_table_body}
      </tbody>
    </table>
  </section>

  <footer>
    <p><a href="../index.html">← Back to index</a></p>
  </footer>
</body>
</html>"""

    page_path = nn_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


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
                  AND taxonomy_path !~ '[>/|]'
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
                  AND top_taxonomy_path !~ '[>/|]'
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


def _load_taxonomy_lr_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy logistic regression fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_lr_fold_results"):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, trained_at
                FROM {schema}.taxonomy_lr_fold_results
                ORDER BY cv_fold
                """
            ).format(schema=sql.Identifier(schema))
        )
        results = []
        for row in cur:
            results.append({
                "cv_fold": int(row["cv_fold"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_f1": float(row["test_f1"]),
                "test_hierarchical_loss": float(row["test_hierarchical_loss"]),
                "padic_loss_total": float(row["padic_loss_total"]),
                "padic_loss_mean": float(row["padic_loss_mean"]),
                "prime_base": int(row["prime_base"]),
                "num_train_samples": int(row["num_train_samples"]),
                "num_test_samples": int(row["num_test_samples"]),
                "trained_at": row["trained_at"].isoformat(timespec="seconds") if row["trained_at"] else None,
            })

    return results if results else None


def _load_taxonomy_nn_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy neural network fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_nn_fold_results"):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, hidden_layers, max_tags
                FROM {schema}.taxonomy_nn_fold_results
                ORDER BY cv_fold
                """
            ).format(schema=sql.Identifier(schema))
        )
        results = []
        for row in cur:
            results.append({
                "cv_fold": int(row["cv_fold"]),
                "test_accuracy": float(row["test_accuracy"]),
                "test_f1": float(row["test_f1"]),
                "test_hierarchical_loss": float(row["test_hierarchical_loss"]),
                "padic_loss_total": float(row["padic_loss_total"]),
                "padic_loss_mean": float(row["padic_loss_mean"]),
                "prime_base": int(row["prime_base"]),
                "num_train_samples": int(row["num_train_samples"]),
                "num_test_samples": int(row["num_test_samples"]),
                "hidden_layers": str(row["hidden_layers"]),
                "max_tags": int(row["max_tags"]) if row["max_tags"] is not None else None,
            })

    return results if results else None


def _generate_historical_trends_chart(conn, output_path: Path, schema: str = "padjective") -> Optional[Path]:
    """Generate a chart showing historical model performance trends.

    Returns:
        Path to generated chart, or None if no historical data exists
    """
    if not _table_exists(conn, schema, "model_performance_history"):
        return None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT snapshot_date, num_products, num_tags, num_taxonomies,
                       umllr_mean_padic_loss, lr_mean_padic_loss, nn_mean_padic_loss
                FROM {schema}.model_performance_history
                ORDER BY snapshot_date
                """
            ).format(schema=sql.Identifier(schema))
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        # Need at least 2 data points to show a trend
        return None

    # Convert to lists for plotting
    dates = [row[0] for row in rows]
    num_products = [row[1] for row in rows]
    num_tags = [row[2] for row in rows]
    num_taxonomies = [row[3] for row in rows]
    umllr_loss = [row[4] if row[4] is not None else None for row in rows]
    lr_loss = [row[5] if row[5] is not None else None for row in rows]
    nn_loss = [row[6] if row[6] is not None else None for row in rows]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot p-adic loss trends
    if any(umllr_loss):
        ax1.plot(dates, umllr_loss, 'o-', label='umllr', color='#0b6ce3', linewidth=2, markersize=6)
    if any(lr_loss):
        ax1.plot(dates, lr_loss, 's-', label='Logistic Regression', color='#10b981', linewidth=2, markersize=6)
    if any(nn_loss):
        ax1.plot(dates, nn_loss, '^-', label='Neural Network', color='#f59e0b', linewidth=2, markersize=6)

    ax1.set_ylabel('P-adic Loss (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=0)

    # Plot dataset growth
    ax2.plot(dates, num_products, 'o-', label='Products', color='#6366f1', linewidth=2, markersize=6)
    ax2.plot(dates, num_taxonomies, 's-', label='Active taxonomies', color='#0ea5e9', linewidth=2, markersize=6)
    if any(value is not None for value in num_tags):
        ax2.plot(dates, num_tags, '^-', label='Distinct tags', color='#f97316', linewidth=2, markersize=6)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dataset Counts', fontsize=12, fontweight='bold')
    ax2.set_title('Dataset Growth', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(bottom=0)
    ax2.legend(loc='best', frameon=True, shadow=True)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def build_site(
    output_dir: Path,
    *,
    precomputed_database: Optional[Any] = None,
    battle_schema: str = "padjective",
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 2,
    min_samples_per_taxonomy: int = 5,
) -> Dict[str, Any]:
    _ensure_clean_directory(output_dir)

    assets_dir = output_dir / "assets"
    downloads_dir = output_dir / "downloads"
    datadumps_dir = output_dir / "datadumps"
    for path in (assets_dir, downloads_dir, datadumps_dir):
        path.mkdir(parents=True, exist_ok=True)

    if precomputed_database is None:
        raise ValueError("A Postgres connection is required to build the site")

    dataset = data_access.build_feature_dataset(
        precomputed_database,
        product_table=product_table,
        require_taxonomy=True,
        min_tag_count=min_tag_count,
        min_samples_per_taxonomy=min_samples_per_taxonomy,
    )
    dataset_stats = {
        "products": dataset.product_count,
        "unique_tags": len(dataset.feature_names),
        "taxonomies": dataset.taxonomy_count,
        "discarded_products": len(dataset.discarded_products),
        "discarded_tags": len(dataset.discarded_tags),
    }

    dataset_page = _write_dataset_page(output_dir, dataset)
    defective_taxonomy_page = _write_defective_taxonomy_page(output_dir, dataset)

    pairs = ranking.load_pairs(precomputed_database, battle_schema)
    leaderboard = ranking.compute_rankings(pairs)
    tag_rank_lookup = _build_tag_rank_lookup(leaderboard)

    rankings_html = downloads_dir / "tag_rankings_table.html"
    chart_path = assets_dir / "top_tags.png"
    display.generate_outputs(leaderboard, rankings_html, chart_path, rows=20)

    stats = _collect_database_stats(precomputed_database, battle_schema)
    stats["battles"] = _count_battles(pairs)
    stats["components"] = int(leaderboard["component"].nunique()) if not leaderboard.empty else 0
    stats["products"] = dataset_stats["products"]
    stats["unique_tags"] = dataset_stats["unique_tags"]
    stats["taxonomies"] = dataset_stats["taxonomies"]

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
.dataset-overview {background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.1); margin: 2rem auto; padding: 1.5rem;}
.dataset-overview h2 {margin-top: 0;}
.dataset-overview p {margin: 0; line-height: 1.6; color: #1f2937;}
.dataset-overview .discard-note {display: block; margin-top: 0.75rem; color: #b91c1c;}
.table-wrapper {overflow-x: auto;}
.dataset-table {width: 100%; border-collapse: collapse; background: white;}
.dataset-table th, .dataset-table td {padding: 0.6rem 0.75rem; border-bottom: 1px solid #e5e7eb; text-align: left;}
.dataset-table thead {background: #f1f5f9;}
.dataset-table tbody tr:nth-child(even) {background: #f8fafc;}
.dataset-page {background: #f7f7fb; color: #1f2937;}
.dataset-page section {max-width: 90rem;}
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

    # Write separate model pages
    elo_page = _write_elo_rankings_page(output_dir, leaderboard, chart_path, stats)

    taxonomy_summary = _collect_taxonomy_classifier_summary(
        precomputed_database, schema=battle_schema
    )
    taxonomy_lr_fold_results = _load_taxonomy_lr_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_page = None
    taxonomy_fold_pages = {}
    if taxonomy_summary:
        if taxonomy_lr_fold_results:
            taxonomy_fold_pages = _write_taxonomy_lr_fold_pages(output_dir, taxonomy_lr_fold_results)
        taxonomy_page = _write_taxonomy_classifier_page(
            output_dir, taxonomy_summary, taxonomy_lr_fold_results, taxonomy_fold_pages
        )

    umllr_summary = _load_umllr_results(precomputed_database, battle_schema)
    umllr_page = None
    if umllr_summary:
        umllr_summary["tag_rankings"] = tag_rank_lookup
        fold_pages = _write_umllr_pages(output_dir, umllr_summary)
        # Add pages to summary before creating overview page so links work
        umllr_summary["pages"] = {
            fold: path.relative_to(output_dir).as_posix()
            for fold, path in fold_pages.items()
        }
        umllr_page = _write_umllr_overview_page(output_dir, umllr_summary)
        umllr_summary["overview_page"] = umllr_page.relative_to(output_dir).as_posix()

    taxonomy_nn_fold_results = _load_taxonomy_nn_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_nn_page = None
    if taxonomy_nn_fold_results:
        taxonomy_nn_page = _write_taxonomy_nn_page(output_dir, taxonomy_nn_fold_results)

    # Generate historical trends chart
    trends_chart_path = _generate_historical_trends_chart(
        precomputed_database,
        assets_dir / "historical_trends.png",
        schema=battle_schema
    )

    _build_index_html(
        output_dir,
        stats,
        dataset_stats,
        dataset_page,
        defective_taxonomy_page,
        elo_page,
        taxonomy_page,
        umllr_page,
        taxonomy_nn_page,
        taxonomy_summary,
        umllr_summary,
        taxonomy_lr_fold_results,
        taxonomy_nn_fold_results,
        trends_chart_path,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "dataset": {
            **dataset_stats,
            "page": dataset_page.relative_to(output_dir).as_posix(),
        },
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
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table to read training data from.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=2,
        help="Minimum occurrences required for a tag to be included in the dataset.",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum products required per taxonomy for inclusion in the dataset.",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        build_site(
            args.output,
            precomputed_database=conn,
            battle_schema=args.schema,
            product_table=args.product_table,
            min_tag_count=args.min_tag_count,
            min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
