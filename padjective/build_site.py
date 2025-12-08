"""Build a static website showcasing tag rankings and taxonomy progress."""

from __future__ import annotations

import argparse
import html
import json
import math
import shutil
import urllib.parse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from psycopg import sql
from psycopg.rows import dict_row
from scipy import stats

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


def _format_long_tag(tag: str, max_length: int = 60) -> str:
    """Format a tag for display, adding soft breaks at punctuation for long tags.

    Args:
        tag: The tag string to format
        max_length: Maximum length before truncating (with ellipsis)

    Returns:
        HTML-escaped tag with soft break opportunities at punctuation
    """
    escaped = html.escape(tag)

    # If short enough, just return escaped version
    if len(tag) <= max_length:
        # Still add soft breaks for medium-length tags with punctuation
        # Use <wbr> (word break opportunity) after common separators
        for sep in [':', ';', ',', '/', '-', '_', '.']:
            escaped = escaped.replace(sep, sep + '<wbr>')
        return escaped

    # For very long tags, truncate and add ellipsis
    # But first, try to find a good break point
    truncated = tag[:max_length]

    # Try to break at a punctuation point near the end
    for sep in [':', ';', ',', '/', '-', '_', '.', ' ']:
        last_sep = truncated.rfind(sep)
        if last_sep > max_length // 2:  # Only break if we keep at least half
            truncated = truncated[:last_sep + 1]
            break

    escaped_truncated = html.escape(truncated)
    # Add soft breaks
    for sep in [':', ';', ',', '/', '-', '_', '.']:
        escaped_truncated = escaped_truncated.replace(sep, sep + '<wbr>')

    return f'<span title="{html.escape(tag)}">{escaped_truncated}…</span>'


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


def _weighted_f1_score(true_labels: Sequence[str], pred_labels: Sequence[str]) -> float | None:
    """Compute weighted F1 score for multiclass predictions."""

    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must be the same length")

    if not true_labels:
        return None

    label_support = Counter(true_labels)
    total = sum(label_support.values())
    if total == 0:
        return None

    correct_counts = Counter()
    predicted_counts = Counter(pred_labels)

    for truth, guess in zip(true_labels, pred_labels):
        if truth == guess:
            correct_counts[truth] += 1

    weighted_sum = 0.0
    for label, support in label_support.items():
        true_positive = correct_counts[label]
        false_positive = predicted_counts[label] - true_positive
        false_negative = support - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        weighted_sum += f1 * support

    return weighted_sum / total


def _padic_breakdown_from_pairs(
    value_pairs: Sequence[Tuple[int, int]],
    prime_base: int,
) -> list[Dict[str, int | float]]:
    """Summarise p-adic agreement by counting shared digits between integers.

    Returns breakdown in descending order: exact match first, then highest exponent to lowest.
    Each entry includes count, cost per mistake, and total contribution to loss.
    """

    if not value_pairs:
        return []

    breakdown_counts: Counter[Any] = Counter()

    for true_value, predicted_value in value_pairs:
        if true_value == predicted_value:
            breakdown_counts["exact"] += 1
            continue

        diff = abs(true_value - predicted_value)
        exponent = 0
        if prime_base > 1 and diff > 0:
            while diff % prime_base == 0:
                diff //= prime_base
                exponent += 1

        breakdown_counts[exponent] += 1

    exponent_keys = [key for key in breakdown_counts.keys() if isinstance(key, int)]
    max_exponent = max(exponent_keys, default=0)

    breakdown: list[Dict[str, int | float]] = []

    # Exact match first (cost = 0)
    exact_count = breakdown_counts.get("exact", 0)
    breakdown.append({
        "label": "Exact match",
        "count": exact_count,
        "cost": 0.0,
        "total_contribution": 0.0,
    })

    # Then descending order: highest exponent to lowest
    for exponent in range(max_exponent, -1, -1):
        count = breakdown_counts.get(exponent, 0)
        cost = prime_base ** (-exponent) if prime_base > 0 else 1.0
        breakdown.append({
            "label": f"p^{exponent}",
            "count": count,
            "cost": cost,
            "total_contribution": count * cost,
        })

    return breakdown


def _load_taxonomy_encoding_lookup(conn, schema: str, fold: int) -> Dict[str, int]:
    """Load taxonomy_id → encoded_value mapping for a specific fold."""

    lookup: Dict[str, int] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT taxonomy_id, encoded_value
                FROM {schema}.umllr_taxonomy_encodings
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,),
        )
        for row in cur:
            taxonomy_id = str(row["taxonomy_id"])
            lookup[taxonomy_id] = int(row["encoded_value"])
    return lookup


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


def _dump_table_to_sql(conn, schema: str, table: str, dump_path: Path) -> None:
    """Export a complete table to SQL INSERT statements."""
    from psycopg.rows import dict_row

    with dump_path.open("w", encoding="utf-8") as f:
        f.write(f"-- Table: {schema}.{table}\n")
        f.write("BEGIN;\n")

        # Get column names
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
                """,
                (schema, table),
            )
            columns = [row[0] for row in cur.fetchall()]

        if not columns:
            f.write("COMMIT;\n")
            return

        # Fetch and write data
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL("SELECT * FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )

            for row in cur:
                values = []
                for col in columns:
                    val = row[col]
                    if val is None:
                        values.append("NULL")
                    elif isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, bool):
                        values.append("TRUE" if val else "FALSE")
                    else:
                        # Escape single quotes
                        escaped = str(val).replace("'", "''")
                        values.append(f"'{escaped}'")

                cols_str = ", ".join(columns)
                vals_str = ", ".join(values)
                f.write(f"INSERT INTO {schema}.{table} ({cols_str}) VALUES ({vals_str});\n")

        f.write("COMMIT;\n")


def _create_comprehensive_dumps(conn, datadumps_dir: Path, schema: str) -> None:
    """Create SQL dumps for all tables needed to recreate the website."""
    from psycopg.rows import dict_row

    # Core data tables from cantbuymelove schema
    core_tables = [
        ("cantbuymelove", "taxonomy"),
        ("cantbuymelove", "product"),
        ("cantbuymelove", "product_taxonomy"),
    ]

    # Results tables from padjective schema
    results_tables = [
        (schema, "battles"),
        (schema, "umllr_fold_metrics"),
        (schema, "umllr_tag_coefficients"),
        (schema, "umllr_predictions"),
        (schema, "umllr_taxonomy_encodings"),
        (schema, "dummy_fold_metrics"),
        (schema, "dummy_predictions"),
        (schema, "taxonomy_pclr_models"),
        (schema, "taxonomy_pclr_class_distribution"),
        (schema, "taxonomy_pclr_tag_summary"),
        (schema, "taxonomy_pclr_top_tags"),
        (schema, "taxonomy_pclr_fold_results"),
        (schema, "taxonomy_pclr_predictions"),
        (schema, "taxonomy_pclr_coefficients"),
        (schema, "taxonomy_pcnn_fold_results"),
        (schema, "taxonomy_pcnn_predictions"),
    ]

    all_tables = core_tables + results_tables

    for table_schema, table_name in all_tables:
        # Check if table exists
        if not _table_exists(conn, table_schema, table_name):
            continue

        dump_path = datadumps_dir / f"{table_schema}_{table_name}.sql"
        try:
            _dump_table_to_sql(conn, table_schema, table_name, dump_path)
            print(f"Exported {table_schema}.{table_name} to {dump_path.name}")
        except Exception as e:
            print(f"Warning: Failed to export {table_schema}.{table_name}: {e}")

    # Special handling for product_details - only dump rows for products in cantbuymelove.product
    if _table_exists(conn, "public", "product_details"):
        dump_path = datadumps_dir / "public_product_details.sql"
        try:
            with dump_path.open("w", encoding="utf-8") as f:
                f.write("-- Table: public.product_details (filtered)\n")
                f.write("BEGIN;\n")

                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(
                        """
                        SELECT pd.*
                        FROM public.product_details pd
                        JOIN cantbuymelove.product p ON
                            pd.myshopify_domain = p.myshopify_domain
                            AND pd.run_name = p.run_name
                            AND pd.product_handle = p.product_handle
                        """
                    )

                    for row in cur:
                        # Escape values
                        domain = row['myshopify_domain'].replace("'", "''")
                        run_name = row['run_name'].replace("'", "''")
                        handle = row['product_handle'].replace("'", "''")
                        # For JSONB, we need to escape it properly
                        detail = str(row['product_detail']).replace("'", "''")

                        f.write(
                            f"INSERT INTO public.product_details (myshopify_domain, run_name, product_handle, product_detail) "
                            f"VALUES ('{domain}', '{run_name}', '{handle}', '{detail}');\n"
                        )

                f.write("COMMIT;\n")
            print(f"Exported public.product_details (filtered) to {dump_path.name}")
        except Exception as e:
            print(f"Warning: Failed to export public.product_details: {e}")


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
        "umllr_taxonomy_encodings",
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

    taxonomy_lookup_by_fold: Dict[int, Dict[int, str]] = {}
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, taxonomy_id, encoded_value
                FROM {schema}.umllr_taxonomy_encodings
                ORDER BY cv_fold, taxonomy_id
                """
            ).format(schema=sql.Identifier(schema))
        )
        for row in cur:
            fold = int(row["cv_fold"])
            encoded_value = int(row["encoded_value"])
            taxonomy_lookup_by_fold.setdefault(fold, {})[encoded_value] = row["taxonomy_id"]

    # Calculate mean loss per fold from predictions and derive accuracy/F1 metrics
    for metric in metrics:
        fold = metric["cv_fold"]
        fold_predictions = predictions.get(fold, [])
        total_predictions = len(fold_predictions)
        metric["total_predictions"] = total_predictions

        # Count non-zero coefficients for this fold
        fold_coefficients = coefficients.get(fold, [])
        num_nonzero = sum(1 for c in fold_coefficients if c.get("coefficient", 0) != 0)
        metric["num_nonzero_coefficients"] = num_nonzero
        metric["num_total_coefficients"] = len(fold_coefficients)

        pair_values = [
            (int(pred["true_value"]), int(pred["predicted_value"]))
            for pred in fold_predictions
        ]

        if fold_predictions:
            mean_loss = sum(p["loss"] for p in fold_predictions) / len(fold_predictions)
            metric["mean_loss"] = mean_loss
            exact_matches = sum(1 for true_value, predicted_value in pair_values if true_value == predicted_value)
            metric["exact_matches"] = exact_matches
            metric["accuracy"] = exact_matches / total_predictions if total_predictions else None
            metric["loss_breakdown"] = _padic_breakdown_from_pairs(pair_values, metric["prime_base"])
        else:
            metric["mean_loss"] = 0.0
            metric["exact_matches"] = 0
            metric["accuracy"] = None
            metric["loss_breakdown"] = []

        lookup = taxonomy_lookup_by_fold.get(fold, {})
        if pair_values and lookup:
            true_labels: list[str] = []
            pred_labels: list[str] = []
            for true_value, predicted_value in pair_values:
                true_label = lookup.get(true_value, f"encoded:{true_value}")
                pred_label = lookup.get(predicted_value, f"encoded:{predicted_value}")
                true_labels.append(str(true_label))
                pred_labels.append(str(pred_label))
            metric["f1"] = _weighted_f1_score(true_labels, pred_labels)
        else:
            metric["f1"] = None

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

    accuracies = [m["accuracy"] for m in metrics if m.get("accuracy") is not None]
    f1_scores = [m["f1"] for m in metrics if m.get("f1") is not None]

    return {
        "metrics": metrics,
        "coefficients": coefficients,
        "predictions": predictions,
        "taxonomy_names": taxonomy_names,
        "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "average_f1": sum(f1_scores) / len(f1_scores) if f1_scores else None,
    }


def _load_dummy_results(conn, schema: str) -> Optional[Dict[str, Any]]:
    """Load dummy classifier results from database."""
    if not _table_exists(conn, schema, "dummy_fold_metrics"):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, loss, accuracy, most_common_value, most_common_taxonomy_id
                FROM {schema}.dummy_fold_metrics
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
            "accuracy": float(row["accuracy"]),
            "most_common_value": int(row["most_common_value"]),
            "most_common_taxonomy_id": str(row["most_common_taxonomy_id"]),
        }
        for row in metrics_rows
    ]

    predictions: Dict[int, list[Dict[str, Any]]] = {}
    if _table_exists(conn, schema, "dummy_predictions"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT cv_fold, product_id, true_value, predicted_value, loss
                    FROM {schema}.dummy_predictions
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

    accuracies = [m["accuracy"] for m in metrics]
    losses = [m["loss"] for m in metrics]

    return {
        "metrics": metrics,
        "predictions": predictions,
        "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "average_loss": sum(losses) / len(losses) if losses else None,
    }


def _write_dummy_fold_pages(
    output_dir: Path,
    summary: Dict[str, Any],
    conn=None,
    schema: str = "padjective"
) -> Dict[int, Path]:
    """Write individual fold pages for dummy classifier."""
    pages: Dict[int, Path] = {}
    metrics = summary.get("metrics", [])
    if not metrics:
        return pages

    dummy_dir = output_dir / "dummy"
    dummy_dir.mkdir(parents=True, exist_ok=True)

    predictions = summary.get("predictions", {})

    # Load taxonomy info for display
    taxonomy_info_by_id: Dict[str, Dict[str, str]] = {}
    if conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT taxonomy_id, taxonomy_name, taxonomy_path
                FROM cantbuymelove.taxonomy
                WHERE taxonomy_path !~ '[>/|]'
                """
            )
            rows = cur.fetchall()
            for row in rows:
                tid = row["taxonomy_id"]
                info = {
                    "taxonomy_id": tid,
                    "taxonomy_name": row["taxonomy_name"] or "",
                    "taxonomy_path": row["taxonomy_path"] or "",
                }
                if tid:
                    taxonomy_info_by_id[tid] = info

    for metric in metrics:
        fold = metric["cv_fold"]
        prediction_rows = predictions.get(fold, [])

        accuracy = metric.get("accuracy", 0)
        accuracy_text = f"{accuracy * 100:.2f}%"
        loss = metric.get("loss", 0)
        num_predictions = len(prediction_rows)
        mean_loss = loss / num_predictions if num_predictions > 0 else 0

        most_common_taxonomy_id = metric.get("most_common_taxonomy_id", "")
        taxonomy_info = taxonomy_info_by_id.get(most_common_taxonomy_id, {})
        taxonomy_name = taxonomy_info.get("taxonomy_name", most_common_taxonomy_id)
        taxonomy_path = taxonomy_info.get("taxonomy_path", "")

        # Build prediction table
        prediction_table_rows_list = []
        for row in prediction_rows:
            prediction_table_rows_list.append(
                "<tr>"
                f"<td>{row['product_id']}</td>"
                f"<td>{row['true_value']}</td>"
                f"<td>{row['predicted_value']}</td>"
                f"<td>{row['loss']:.8f}</td>"
                "</tr>"
            )

        prediction_table_rows = "\n".join(prediction_table_rows_list)
        if not prediction_table_rows:
            prediction_table_rows = '<tr><td colspan="4">No test predictions available for this fold.</td></tr>'

        # Calculate p-adic loss breakdown
        pair_values = [
            (int(row["true_value"]), int(row["predicted_value"]))
            for row in prediction_rows
        ]

        # Get prime base from first metric (same across all folds)
        # We need to look this up from umllr_fold_metrics
        prime_base = 71  # Default, will be updated if available
        if conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    sql.SQL("SELECT prime_base FROM {schema}.umllr_fold_metrics WHERE cv_fold = %s LIMIT 1").format(
                        schema=sql.Identifier(schema)
                    ),
                    (fold,)
                )
                row = cur.fetchone()
                if row:
                    prime_base = int(row["prime_base"])

        breakdown_rows = _padic_breakdown_from_pairs(pair_values, prime_base)

        # Build breakdown HTML
        breakdown_html = ""
        if breakdown_rows:
            breakdown_table_rows = []
            total_predictions = len(prediction_rows)
            for entry in breakdown_rows:
                label = html.escape(str(entry.get("label", "")))
                count = int(entry.get("count", 0))
                percentage = (count / total_predictions * 100) if total_predictions else 0.0
                cost = entry.get("cost", 0.0)
                total_contrib = entry.get("total_contribution", 0.0)
                breakdown_table_rows.append(
                    f"<tr><td>{label}</td><td>{count:,}</td><td>{percentage:.2f}%</td><td>{cost:.6f}</td><td>{total_contrib:.6f}</td></tr>"
                )
            breakdown_body = "\n".join(breakdown_table_rows)
            breakdown_html = f"""
<h2>P-adic loss breakdown</h2>
<table class="umllr-table">
  <thead>
    <tr><th>Agreement</th><th>Count</th><th>Share</th><th>Cost per mistake</th><th>Total contribution</th></tr>
  </thead>
  <tbody>
    {breakdown_body}
  </tbody>
</table>
"""

        page_contents = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Dummy Classifier - Fold {fold}</title>
  <link rel="stylesheet" href="../styles.css">
</head>
<body>
  <header>
    <h1>Dummy Classifier - Fold {fold}</h1>
    <nav>
      <a href="../index.html">← Home</a> |
      <a href="index.html">← Overview</a>
    </nav>
  </header>

  <section class="umllr-fold">
    <h2>Fold {fold} Metrics</h2>
    <table class="umllr-table">
      <tbody>
        <tr><td>Total predictions</td><td>{num_predictions:,}</td></tr>
        <tr><td>Accuracy</td><td>{accuracy_text}</td></tr>
        <tr><td>Total p-adic loss</td><td>{loss:.4f}</td></tr>
        <tr><td>Mean p-adic loss</td><td>{mean_loss:.4f}</td></tr>
        <tr><td>Predicted taxonomy</td><td>{html.escape(taxonomy_name)}</td></tr>
        <tr><td>Predicted taxonomy path</td><td>{html.escape(taxonomy_path)}</td></tr>
        <tr><td>Predicted taxonomy ID</td><td>{html.escape(most_common_taxonomy_id)}</td></tr>
      </tbody>
    </table>

  {breakdown_html}

    <h2>Test predictions</h2>
    <table class="umllr-table">
      <thead>
        <tr>
          <th>Product ID</th>
          <th>True value</th>
          <th>Predicted value</th>
          <th>Loss</th>
        </tr>
      </thead>
      <tbody>
        {prediction_table_rows}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
        page_path = dummy_dir / f"fold_{fold}.html"
        page_path.write_text(page_contents, encoding="utf-8")
        pages[fold] = page_path

    return pages


def _write_dummy_overview_page(
    output_dir: Path,
    summary: Dict[str, Any],
    fold_pages: Dict[int, Path],
) -> Path:
    """Write overview page for dummy classifier."""
    dummy_dir = output_dir / "dummy"
    dummy_dir.mkdir(parents=True, exist_ok=True)

    metrics = summary.get("metrics", [])
    avg_accuracy = summary.get("average_accuracy", 0)
    avg_loss = summary.get("average_loss", 0)

    # Build fold table
    fold_rows = []
    for metric in metrics:
        fold = metric["cv_fold"]
        accuracy = metric.get("accuracy", 0)
        loss = metric.get("loss", 0)
        num_predictions = len(summary.get("predictions", {}).get(fold, []))
        mean_loss = loss / num_predictions if num_predictions > 0 else 0

        fold_link = fold_pages.get(fold)
        if fold_link:
            fold_cell = f'<a href="{fold_link.relative_to(dummy_dir).as_posix()}">Fold {fold}</a>'
        else:
            fold_cell = f"Fold {fold}"

        fold_rows.append(
            f"<tr>"
            f"<td>{fold_cell}</td>"
            f"<td>{num_predictions:,}</td>"
            f"<td>{accuracy * 100:.2f}%</td>"
            f"<td>{mean_loss:.4f}</td>"
            f"</tr>"
        )

    fold_table = "\n".join(fold_rows)

    num_folds = len(metrics)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Dummy Baseline Classifier</title>
  <link rel="stylesheet" href="../styles.css">
</head>
<body>
  <header class="hero">
    <h1>Dummy Baseline Classifier</h1>
    <p class="tagline">Always predicting the most common taxonomy (baseline for comparison)</p>
    <nav>
      <a href="../index.html">← Home</a>
    </nav>
  </header>

  <section>
    <div class="metrics">
      <div class="metric">
        <span class="value">{num_folds}</span>
        <span class="label">CV folds</span>
      </div>
      <div class="metric">
        <span class="value">{avg_loss:.4f}</span>
        <span class="label">Average p-adic loss</span>
      </div>
      <div class="metric">
        <span class="value">{avg_accuracy * 100:.2f}%</span>
        <span class="label">Mean accuracy</span>
      </div>
    </div>

    <h2>Model description</h2>
    <p>
      The dummy classifier is a baseline model that always predicts the most common taxonomy
      class from the training data. This provides a simple benchmark to compare against more
      sophisticated models. Any model that performs worse than this baseline is essentially
      useless.
    </p>
    <p>
      For each fold, the model identifies the most frequent taxonomy in the training set and
      predicts it for every test example, regardless of tags or other features.
    </p>

    <h2>Results by fold</h2>
    <table class="umllr-summary">
      <thead>
        <tr>
          <th>Fold</th>
          <th>Predictions</th>
          <th>Accuracy</th>
          <th>Mean p-adic loss</th>
        </tr>
      </thead>
      <tbody>
        {fold_table}
      </tbody>
    </table>
  </section>
</body>
</html>
"""
    page_path = dummy_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


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
                SELECT
                    up.product_id,
                    up.true_value,
                    up.predicted_value,
                    up.loss,
                    te.taxonomy_id as predicted_taxonomy_id
                FROM {schema}.umllr_predictions up
                LEFT JOIN {schema}.umllr_taxonomy_encodings te ON
                    up.cv_fold = te.cv_fold AND up.predicted_value = te.encoded_value
                WHERE up.cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (fold,)
        )
        for row in cur:
            pid = row["product_id"]
            umllr_predictions[pid] = {
                "true_value": row["true_value"],
                "predicted_value": row["predicted_value"],
                "predicted_taxonomy_id": row["predicted_taxonomy_id"],
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
                FROM {schema}.taxonomy_pclr_predictions
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
                FROM {schema}.taxonomy_pclr_coefficients
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
                FROM {schema}.taxonomy_pcnn_predictions
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

    # Load ULR predictions
    ulr_predictions: Dict[int, Dict[str, Any]] = {}
    if _table_exists(conn, schema, "taxonomy_ulr_predictions"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT product_id, true_taxonomy_id, predicted_taxonomy_id, loss
                    FROM {schema}.taxonomy_ulr_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,)
            )
            for row in cur:
                pid = row["product_id"]
                ulr_predictions[pid] = {
                    "true_taxonomy_id": row["true_taxonomy_id"],
                    "predicted_taxonomy_id": row["predicted_taxonomy_id"],
                    "loss": row["loss"],
                }

    # Load UNN predictions
    unn_predictions: Dict[int, Dict[str, Any]] = {}
    if _table_exists(conn, schema, "taxonomy_unn_predictions"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT product_id, true_taxonomy_id, predicted_taxonomy_id, loss
                    FROM {schema}.taxonomy_unn_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,)
            )
            for row in cur:
                pid = row["product_id"]
                unn_predictions[pid] = {
                    "true_taxonomy_id": row["true_taxonomy_id"],
                    "predicted_taxonomy_id": row["predicted_taxonomy_id"],
                    "loss": row["loss"],
                }

    # Load Dummy predictions
    dummy_predictions: Dict[int, Dict[str, Any]] = {}
    if _table_exists(conn, schema, "dummy_predictions"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT dp.product_id, dp.true_value, dp.predicted_value, dp.loss,
                           te.taxonomy_id as predicted_taxonomy_id
                    FROM {schema}.dummy_predictions dp
                    LEFT JOIN {schema}.umllr_taxonomy_encodings te ON
                        dp.cv_fold = te.cv_fold AND dp.predicted_value = te.encoded_value
                    WHERE dp.cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,)
            )
            for row in cur:
                pid = row["product_id"]
                dummy_predictions[pid] = {
                    "true_value": row["true_value"],
                    "predicted_value": row["predicted_value"],
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
                p.product_url,
                pd.product_detail->'product'->>'tags' AS tags,
                pt.taxonomy_id,
                ps.is_alive,
                ps.http_status_code,
                up.cv_fold
            FROM cantbuymelove.product p
            JOIN public.product_details pd ON (
                p.myshopify_domain = pd.myshopify_domain
                AND p.run_name = pd.run_name
                AND p.product_handle = pd.product_handle
            )
            LEFT JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
            LEFT JOIN cantbuymelove.product_status ps ON ps.product_id = p.id
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

            # Skip products with no tags
            if not tags:
                continue

            details[pid] = {
                "product_id": pid,
                "title": row["title"] or "",
                "product_url": row["product_url"],
                "is_alive": row["is_alive"],
                "http_status_code": row["http_status_code"],
                "tags": tags,
                "ground_truth": taxonomy_info.get(taxonomy_id, {}),
                "predictions": {
                    "umllr": umllr_predictions.get(pid, {}),
                    "lr": lr_predictions.get(pid, {}),
                    "nn": nn_predictions.get(pid, {}),
                    "ulr": ulr_predictions.get(pid, {}),
                    "unn": unn_predictions.get(pid, {}),
                    "dummy": dummy_predictions.get(pid, {}),
                },
                "umllr_coefficients": {tag: umllr_tag_coeffs.get(tag, 0) for tag in tags},
                "lr_coefficients": {tag: lr_coeffs.get(tag, {}) for tag in tags},
            }

    return details


def _write_prediction_detail_page(
    page_path: Path,
    fold: int,
    product_id: int,
    detail: Dict[str, Any],
    taxonomy_info_by_path: Dict[str, Dict[str, str]],
    taxonomy_info_by_id: Dict[str, Dict[str, str]],
    prime_base: int,
) -> None:
    """Write a detailed prediction page for a single product."""

    title = detail.get("title", f"Product {product_id}")
    product_url = detail.get("product_url")
    is_alive = detail.get("is_alive")
    http_status_code = detail.get("http_status_code")
    tags = detail.get("tags", [])
    ground_truth = detail.get("ground_truth", {})
    predictions = detail.get("predictions", {})
    umllr_coeffs = detail.get("umllr_coefficients", {})
    lr_coeffs = detail.get("lr_coefficients", {})

    # Product link section
    product_link_html = ""
    if product_url:
        if is_alive:
            product_link_html = f'<p><strong>Product URL:</strong> <a href="{html.escape(product_url)}" target="_blank">{html.escape(product_url)}</a> ✓ Active</p>'
        else:
            status_text = f"(Status: {http_status_code})" if http_status_code else ""
            product_link_html = f'<p><strong>Product URL:</strong> {html.escape(product_url)} ❌ Not available {status_text}</p>'

    # Ground truth section
    gt_taxonomy_id = ground_truth.get("taxonomy_id", "Unknown")
    gt_taxonomy_name = ground_truth.get("taxonomy_name", "Unknown")
    gt_taxonomy_path = ground_truth.get("taxonomy_path", "Unknown")

    ground_truth_html = f"""
    <div class="detail-section">
      <h2>Ground Truth</h2>
      <table class="detail-table">
        <tr><th>Taxonomy ID</th><td>{html.escape(gt_taxonomy_id)}</td></tr>
        <tr><th>Taxonomy Name</th><td>{html.escape(gt_taxonomy_name)}</td></tr>
        <tr><th>Taxonomy Path</th><td>{html.escape(gt_taxonomy_path)}</td></tr>
      </table>
    </div>
    """

    # Predictions section
    predictions_rows = []

    # umllr prediction
    umllr_pred = predictions.get("umllr", {})
    if umllr_pred:
        pred_tax_id = umllr_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or ""
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>Importance-Optimised p-adic Linear Regression</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{umllr_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    # LR prediction
    lr_pred = predictions.get("lr", {})
    if lr_pred:
        pred_tax_id = lr_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or pred_tax_id
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>PCLR</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{lr_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    # NN prediction
    nn_pred = predictions.get("nn", {})
    if nn_pred:
        pred_tax_id = nn_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or pred_tax_id
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>PCNN</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{nn_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    # ULR prediction
    ulr_pred = predictions.get("ulr", {})
    if ulr_pred:
        pred_tax_id = ulr_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or pred_tax_id
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>ULR</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{ulr_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    # UNN prediction
    unn_pred = predictions.get("unn", {})
    if unn_pred:
        pred_tax_id = unn_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or pred_tax_id
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>UNN</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{unn_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    # Dummy prediction
    dummy_pred = predictions.get("dummy", {})
    if dummy_pred:
        pred_tax_id = dummy_pred.get("predicted_taxonomy_id") or ""
        pred_info = taxonomy_info_by_id.get(pred_tax_id, {}) if pred_tax_id else {}
        pred_tax_path = pred_info.get("taxonomy_path") or pred_tax_id
        pred_tax_name = pred_info.get("taxonomy_name") or ""

        predictions_rows.append(f"""
        <tr>
          <td>Dummy</td>
          <td>{html.escape(pred_tax_path)}</td>
          <td>{html.escape(pred_tax_id)}</td>
          <td>{html.escape(pred_tax_name)}</td>
          <td>{dummy_pred.get("loss", 0.0):.8f}</td>
        </tr>
        """)

    predictions_html = f"""
    <div class="detail-section">
      <h2>Model Predictions</h2>
      <table class="detail-table">
        <thead>
          <tr><th>Model</th><th>Taxonomy Path</th><th>Taxonomy ID</th><th>Taxonomy Name</th><th>P-adic Loss</th></tr>
        </thead>
        <tbody>
          {"".join(predictions_rows)}
        </tbody>
      </table>
    </div>
    """

    # Product tags section
    tags_html = f"""
    <div class="detail-section">
      <h2>Product Tags</h2>
      <p>{", ".join(html.escape(tag) for tag in tags)}</p>
    </div>
    """

    # umllr detailed section
    umllr_tag_rows = []
    for tag in tags:
        coef = umllr_coeffs.get(tag, 0)
        if coef != 0:
            tag_path, tag_expansion = _format_padic_expansion(coef, prime_base)
            # tag_path is already in the correct format (e.g., "1.8.6")
            # Database stores paths in the same format
            tag_info = taxonomy_info_by_path.get(tag_path, {})
            tag_tax_name = tag_info.get("taxonomy_name", "")

            umllr_tag_rows.append(f"""
            <tr>
              <td>{_format_long_tag(tag)}</td>
              <td>{coef}</td>
              <td>{html.escape(tag_path)}</td>
              <td>{html.escape(tag_expansion)}</td>
              <td>{html.escape(tag_tax_name)}</td>
            </tr>
            """)

    umllr_detail_html = ""
    if umllr_tag_rows:
        umllr_detail_html = f"""
        <div class="detail-section">
          <h2>Importance-Optimised p-adic Linear Regression Tag Contributions</h2>
          <table class="detail-table">
            <thead>
              <tr><th>Tag</th><th>Coefficient</th><th>Taxonomy Path</th><th>Expansion</th><th>Taxonomy Name</th></tr>
            </thead>
            <tbody>
              {"".join(umllr_tag_rows)}
            </tbody>
          </table>
        </div>
        """

    # LR detailed section - show coefficients for each tag across all taxonomies
    lr_detail_rows = []
    for tag in tags:
        tag_coeffs = lr_coeffs.get(tag, {})
        if tag_coeffs:
            # Find the taxonomy with the highest coefficient (log odds)
            max_taxonomy = max(tag_coeffs.items(), key=lambda x: x[1])
            max_tax_id = max_taxonomy[0]

            # Build rows for this tag
            for tax_id, coef in sorted(tag_coeffs.items(), key=lambda x: -x[1]):
                tax_info = taxonomy_info_by_id.get(tax_id, {})
                tax_path = tax_info.get("taxonomy_path", tax_id)
                tax_name = tax_info.get("taxonomy_name", "")

                highlight_class = ' class="highlight"' if tax_id == max_tax_id else ''
                lr_detail_rows.append(f"""
                <tr{highlight_class}>
                  <td>{_format_long_tag(tag)}</td>
                  <td>{html.escape(tax_path)}</td>
                  <td>{html.escape(tax_name)}</td>
                  <td>{coef:.6f}</td>
                </tr>
                """)

    lr_detail_html = ""
    if lr_detail_rows:
        lr_detail_html = f"""
        <div class="detail-section">
          <h2>PCLR Coefficients</h2>
          <p>Tag coefficients for each taxonomy class (highest coefficient per tag is highlighted)</p>
          <table class="detail-table">
            <thead>
              <tr><th>Tag</th><th>Taxonomy Path</th><th>Taxonomy Name</th><th>Coefficient</th></tr>
            </thead>
            <tbody>
              {"".join(lr_detail_rows)}
            </tbody>
          </table>
        </div>
        """

    # Complete page
    page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Prediction Detail - Product {product_id} - Fold {fold}</title>
  <link rel="stylesheet" href="../../assets/styles.css" />
  <style>
    .detail-section {{
      margin: 2rem 0;
      background: white;
      padding: 1.5rem;
      border-radius: 0.5rem;
      box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    .detail-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 1rem;
    }}
    .detail-table th, .detail-table td {{
      padding: 0.75rem;
      text-align: left;
      border-bottom: 1px solid #e2e8f0;
    }}
    .detail-table th {{
      background: #f8fafc;
      font-weight: 600;
      color: #475569;
    }}
    .detail-table tr.highlight {{
      background: #fef3c7;
      font-weight: 600;
    }}
    .product-title {{
      font-size: 1.5rem;
      font-weight: 600;
      margin-bottom: 0.5rem;
    }}
  </style>
</head>
<body>
  <section class="umllr-fold">
    <h1>Prediction Detail - Fold {fold}</h1>
    <p><a href="../../umllr/fold_{fold}.html">Back to fold {fold}</a> | <a href="../../index.html">Back to index</a></p>
    <div class="product-title">{html.escape(title)}</div>
    <p><strong>Product ID:</strong> {product_id}</p>
    {product_link_html}

    {ground_truth_html}
    {predictions_html}
    {tags_html}
    {umllr_detail_html}
    {lr_detail_html}
  </section>
</body>
</html>
"""
    page_path.write_text(page_contents, encoding="utf-8")


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
            f"<td>{_format_long_tag(tag)}</td>"
            f"<td>{rank_label}</td>"
            "</tr>"
        )

    zero_table_rows = "\n".join(zero_rows)

    page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Importance-Optimised p-adic Linear Regression fold {fold} - Zero coefficients</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>Importance-Optimised p-adic Linear Regression fold {fold} - Zero coefficients</h1>
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


def _generate_padic_digit_distribution_chart(
    coeff_rows: list[Dict[str, Any]],
    tag_rankings: Dict[str, int],
    prime_base: int,
    output_path: Path
) -> Optional[Path]:
    """Generate a stacked area chart showing cumulative p-adic valuation distributions.

    Args:
        coeff_rows: List of coefficient data with 'tag' and 'coefficient' fields
        tag_rankings: Dict mapping tag names to their battle rankings
        prime_base: The prime base for p-adic expansion
        output_path: Where to save the chart

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not coeff_rows or prime_base < 2:
        return None

    # Build list of (rank, tag, num_zeros)
    # num_zeros is the p-adic valuation (how many times prime divides coefficient)
    tag_data = []
    for row in coeff_rows:
        tag = row["tag"]
        coefficient = row["coefficient"]

        # Get rank from tag_rankings, or use a very high rank if not found
        # Tags without ranks get assigned a very high rank (999999) so they appear at the end
        if tag_rankings:
            rank = tag_rankings.get(tag.upper(), 999999)
        else:
            rank = 999999

        if coefficient == 0:
            # Zero has infinite p-adic valuation
            num_zeros = -1  # Use -1 to represent infinity
        else:
            # Count how many times prime_base divides coefficient
            num_zeros = 0
            temp = abs(coefficient)
            while temp % prime_base == 0:
                num_zeros += 1
                temp //= prime_base
        tag_data.append((rank, tag, num_zeros))

    if len(tag_data) < 2:
        return None

    # Sort by rank
    tag_data.sort(key=lambda x: x[0])

    # Find max number of zeros (excluding infinity)
    max_zeros = max((z for _, _, z in tag_data if z >= 0), default=0)

    # Categories: 0 zeros, 1 zero, 2 zeros, ..., max_zeros, infinity
    categories = list(range(max_zeros + 1)) + [-1]  # -1 represents infinity

    # Compute cumulative proportions at each rank position
    x_positions = []
    zero_proportions = {z: [] for z in categories}

    for i in range(len(tag_data)):
        # Include all tags from rank 0 to i (cumulative)
        included_tags = tag_data[:i+1]

        # Count number of zeros
        zero_counts = {z: 0 for z in categories}
        for _, _, num_zeros in included_tags:
            zero_counts[num_zeros] += 1

        # Calculate proportions
        total = len(included_tags)
        x_positions.append(i)
        for z in categories:
            proportion = zero_counts[z] / total if total > 0 else 0
            zero_proportions[z].append(proportion * 100)  # Convert to percentage

    # Create stacked area chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use different colors
    colors = plt.cm.tab20(range(len(categories)))

    # Stack the areas in order: 0, 1, 2, ..., infinity
    y_stack = []
    current_y = [0] * len(x_positions)

    for idx, z in enumerate(categories):
        next_y = [current_y[j] + zero_proportions[z][j] for j in range(len(x_positions))]
        label = f'{z} leading zeros' if z >= 0 else 'Infinite leading zeros (coeff=0)'
        ax.fill_between(x_positions, current_y, next_y,
                         label=label, alpha=0.7, color=colors[idx])
        current_y = next_y

    ax.set_xlabel('Tag Position (ordered by battle rank)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Proportion (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'P-adic Leading Zeros by Tag Rank ({len(tag_data)} tags)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(x_positions) - 1)  # Ensure full range is shown
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, shadow=True, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _generate_log_nonzero_proportion_chart(
    coeff_rows: list[Dict[str, Any]],
    tag_rankings: Dict[str, int],
    output_path: Path
) -> Optional[Path]:
    """Generate a chart showing log of proportion of non-zero coefficients.

    Args:
        coeff_rows: List of coefficient data with 'tag' and 'coefficient' fields
        tag_rankings: Dict mapping tag names to their battle rankings
        output_path: Where to save the chart

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not coeff_rows:
        return None

    # Build list of (rank, tag, is_nonzero)
    tag_data = []
    for row in coeff_rows:
        tag = row["tag"]
        coefficient = row["coefficient"]

        # Only include tags that have a valid ranking
        if not tag_rankings:
            continue

        rank = tag_rankings.get(tag.upper())
        if rank is None:
            # Skip tags without a valid ranking
            continue

        is_nonzero = (coefficient != 0)
        tag_data.append((rank, tag, is_nonzero))

    if len(tag_data) < 2:
        return None

    # Sort by rank
    tag_data.sort(key=lambda x: x[0])

    # Calculate cumulative proportion of non-zero coefficients
    log_rank_positions = []
    log_proportions = []

    for i in range(len(tag_data)):
        # Include all tags from rank 0 to i (cumulative)
        included_tags = tag_data[:i+1]

        # Count non-zero coefficients
        num_nonzero = sum(1 for _, _, is_nonzero in included_tags if is_nonzero)
        total = len(included_tags)

        proportion = num_nonzero / total if total > 0 else 0

        # Skip positions where proportion is 0 (can't take log)
        if proportion > 0:
            current_rank = tag_data[i][0]
            # Rank must be positive to take a logarithm. If ranks are 0 or negative,
            # fall back to the 1-based index position to maintain monotonic growth.
            if current_rank <= 0:
                current_rank = i + 1
            log_rank_positions.append(math.log(current_rank))
            log_proportions.append(math.log(proportion))

    if len(log_rank_positions) < 2:
        return None

    # Create line chart
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(log_rank_positions, log_proportions, color='#0b6ce3', linewidth=2, alpha=0.8)

    ax.set_xlabel('log(Tag Rank)', fontsize=12, fontweight='bold')
    ax.set_ylabel('log(Proportion of Non-Zero Coefficients)', fontsize=12, fontweight='bold')
    ax.set_title('Log-Log Proportion of Non-Zero Coefficients by Tag Rank', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add horizontal reference line at y=0 (proportion=1, or 100%)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='100% non-zero')
    ax.legend(loc='best', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _generate_rolling_nonzero_chart(
    coeff_rows: list[Dict[str, Any]],
    tag_rankings: Dict[str, int],
    output_path: Path,
    window_size: int = 100
) -> Optional[Path]:
    """Generate a chart showing rolling average of proportion of non-zero coefficients.

    Args:
        coeff_rows: List of coefficient data with 'tag' and 'coefficient' fields
        tag_rankings: Dict mapping tag names to their battle rankings
        output_path: Where to save the chart
        window_size: Size of rolling window (default 100)

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not coeff_rows:
        return None

    # Build list of (rank, tag, is_nonzero)
    # For tags without battle rankings, assign a high rank based on position
    tag_data = []
    max_real_rank = max(tag_rankings.values()) if tag_rankings else 0

    for idx, row in enumerate(coeff_rows):
        tag = row["tag"]
        coefficient = row["coefficient"]

        # Get rank from tag_rankings, or assign based on position
        if tag_rankings:
            rank = tag_rankings.get(tag.upper())
            if rank is None:
                # Assign a high rank for tags without battle rankings
                rank = max_real_rank + 1000 + idx
        else:
            rank = idx

        is_nonzero = (coefficient != 0)
        tag_data.append((rank, tag, is_nonzero))

    if len(tag_data) < window_size:
        return None

    # Sort by rank
    tag_data.sort(key=lambda x: x[0])

    # Calculate rolling average of non-zero proportions
    positions = []
    rolling_proportions = []

    for i in range(window_size - 1, len(tag_data)):
        # Get window of last 'window_size' tags
        window_start = i - window_size + 1
        window = tag_data[window_start:i+1]

        # Count non-zero coefficients in window
        num_nonzero = sum(1 for _, _, is_nonzero in window if is_nonzero)
        proportion = num_nonzero / window_size

        # Use the position in the sorted list (1-based indexing)
        positions.append(i + 1)
        rolling_proportions.append(proportion * 100)  # Convert to percentage

    if len(positions) < 2:
        return None

    # Create line chart
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(positions, rolling_proportions, color='#0b6ce3', linewidth=2, alpha=0.8)

    ax.set_xlabel('Tag Position (sorted by battle rank)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion of Non-Zero Coefficients (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Rolling Average of Non-Zero Coefficients by Tag Rank (window={window_size})',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Add horizontal reference line at 100%
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1, label='100% non-zero')
    ax.legend(loc='best', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _write_tag_debug_page(
    output_path: Path,
    fold: int,
    tag: str,
    conn,
    schema: str,
    prime_base: int,
    taxonomy_info_by_path: Dict[str, Dict[str, str]],
    tag_rankings: Dict[str, int],
) -> None:
    """Write a debug page explaining why a tag coefficient was selected."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get current tag's battle rank
    current_tag_rank = tag_rankings.get(tag, 0)

    # Load candidate coefficients for this tag
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT candidate_value, total_loss, product_count, was_selected
                FROM {schema}.umllr_coefficient_candidates
                WHERE cv_fold = %s AND tag = %s
                ORDER BY total_loss ASC
                """
            ).format(schema=sql.Identifier(schema)),
            (fold, tag)
        )
        candidates = cur.fetchall()

    # Load products with this tag and their residuals
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tp.product_id, tp.residual_before, tp.residual_after,
                       p.product_title
                FROM {schema}.umllr_tag_products tp
                LEFT JOIN cantbuymelove.product p ON tp.product_id = p.id
                WHERE tp.cv_fold = %s AND tp.tag = %s
                ORDER BY tp.residual_before DESC
                """
            ).format(schema=sql.Identifier(schema)),
            (fold, tag)
        )
        products = cur.fetchall()

    # Load all tags for each product, with their coefficients
    product_ids = [p["product_id"] for p in products]
    product_tags: Dict[int, list] = {pid: [] for pid in product_ids}
    if product_ids:
        with conn.cursor(row_factory=dict_row) as cur:
            # Get all tags for these products that have non-zero coefficients in this fold
            # Tags are stored as comma-separated string in product_detail JSONB
            cur.execute(
                sql.SQL(
                    """
                    WITH product_tags AS (
                        SELECT
                            p.id AS product_id,
                            UPPER(TRIM(unnest(string_to_array(
                                pd.product_detail->'product'->>'tags', ','
                            )))) AS tag
                        FROM cantbuymelove.product p
                        JOIN product_details pd ON
                            p.myshopify_domain = pd.myshopify_domain
                            AND p.run_name = pd.run_name
                            AND p.product_handle = pd.product_handle
                        WHERE p.id = ANY(%s)
                    )
                    SELECT pt.product_id, pt.tag, tc.coefficient, tc.sequence
                    FROM product_tags pt
                    JOIN {schema}.umllr_tag_coefficients tc
                        ON pt.tag = tc.tag AND tc.cv_fold = %s
                    WHERE tc.coefficient != 0
                    ORDER BY tc.sequence
                    """
                ).format(schema=sql.Identifier(schema)),
                (product_ids, fold)
            )
            for row in cur.fetchall():
                product_tags[row["product_id"]].append({
                    "tag": row["tag"],
                    "coefficient": row["coefficient"],
                    "sequence": row["sequence"],
                })

    # Get the selected coefficient
    selected_coeff = None
    for c in candidates:
        if c["was_selected"]:
            selected_coeff = c["candidate_value"]
            break

    # Format candidate rows
    candidate_rows = []
    for c in candidates:
        candidate_value = c["candidate_value"]
        taxonomy_path, expansion = _format_padic_expansion(candidate_value, prime_base)
        taxonomy_name = taxonomy_info_by_path.get(taxonomy_path, {}).get("taxonomy_name", "")
        selected_class = ' class="highlight"' if c["was_selected"] else ""
        candidate_rows.append(
            f'<tr{selected_class}>'
            f'<td>{candidate_value}</td>'
            f'<td>{c["total_loss"]:.6f}</td>'
            f'<td>{html.escape(taxonomy_path)}</td>'
            f'<td>{html.escape(expansion)}</td>'
            f'<td>{html.escape(taxonomy_name)}</td>'
            f'<td>{"Yes" if c["was_selected"] else "No"}</td>'
            f'</tr>'
        )

    # Helper to format tag link
    def make_tag_link(t: str) -> str:
        t_url_safe = urllib.parse.quote(t, safe='')
        return f'<a href="../{t_url_safe}/index.html">{html.escape(t)}</a>'

    # Format product rows
    product_rows = []
    for p in products:
        product_id = p["product_id"]
        residual_before = p["residual_before"]
        residual_after = p["residual_after"]
        before_path, before_exp = _format_padic_expansion(residual_before, prime_base)
        after_path, after_exp = _format_padic_expansion(residual_after, prime_base)
        before_tax_name = taxonomy_info_by_path.get(before_path, {}).get("taxonomy_name", "")
        after_tax_name = taxonomy_info_by_path.get(after_path, {}).get("taxonomy_name", "")

        # Split tags into processed (lower sequence) and pending (higher sequence)
        tags_for_product = product_tags.get(product_id, [])
        processed_tags = []
        pending_tags = []
        for t in tags_for_product:
            if t["tag"] == tag:
                continue  # Skip the current tag
            if t["sequence"] < current_tag_rank:
                processed_tags.append(t)
            else:
                pending_tags.append(t)

        # Format as linked lists
        processed_html = ", ".join(make_tag_link(t["tag"]) for t in processed_tags) if processed_tags else "<em>none</em>"
        pending_html = ", ".join(make_tag_link(t["tag"]) for t in pending_tags) if pending_tags else "<em>none</em>"

        # Note: These are training products, not test products, so no prediction page exists
        product_rows.append(
            f'<tr>'
            f'<td>{product_id}</td>'
            f'<td>{html.escape(p["product_title"] or "")}</td>'
            f'<td>{residual_before}</td>'
            f'<td>{html.escape(before_tax_name)}</td>'
            f'<td>{processed_html}</td>'
            f'<td>{residual_after}</td>'
            f'<td>{html.escape(after_tax_name)}</td>'
            f'<td>{pending_html}</td>'
            f'</tr>'
        )

    selected_path, selected_exp = _format_padic_expansion(selected_coeff or 0, prime_base)
    selected_tax_name = taxonomy_info_by_path.get(selected_path, {}).get("taxonomy_name", "")

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Tag Debug: {html.escape(tag)} - Fold {fold}</title>
  <link rel="stylesheet" href="../../../assets/styles.css" />
  <style>
    .highlight {{ background-color: #d4edda !important; }}
    .detail-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
    .detail-table th, .detail-table td {{ padding: 0.5rem; border: 1px solid #dee2e6; text-align: left; }}
    .detail-table thead {{ background: #f8f9fa; }}
    .summary-box {{ background: #e7f3ff; border-radius: 0.5rem; padding: 1rem; margin: 1rem 0; }}
  </style>
</head>
<body>
  <header class="hero">
    <h1>Tag Coefficient Debug: {html.escape(tag)}</h1>
    <p class="tagline">Fold {fold} - Understanding why coefficient {selected_coeff} was selected</p>
  </header>

  <section>
    <p><a href="../../../fold_{fold}.html">← Back to Fold {fold}</a></p>

    <div class="summary-box">
      <h2>Summary</h2>
      <p><strong>Tag:</strong> {html.escape(tag)}</p>
      <p><strong>Selected Coefficient:</strong> {selected_coeff}</p>
      <p><strong>Taxonomy Path:</strong> {html.escape(selected_path)}</p>
      <p><strong>Taxonomy Name:</strong> {html.escape(selected_tax_name)}</p>
      <p><strong>Products with this tag:</strong> {len(products)}</p>
      <p><strong>Unique residual values tried:</strong> {len(candidates)}</p>
    </div>

    <h2>Candidate Coefficients Tried</h2>
    <p>The algorithm tries each unique residual value as a candidate coefficient and selects the one
    with the lowest total p-adic loss. The selected row is highlighted.</p>
    <table class="detail-table">
      <thead>
        <tr>
          <th>Candidate Value</th>
          <th>Total P-adic Loss</th>
          <th>Taxonomy Path</th>
          <th>P-adic Expansion</th>
          <th>Taxonomy Name</th>
          <th>Selected?</th>
        </tr>
      </thead>
      <tbody>
        {"".join(candidate_rows) if candidate_rows else '<tr><td colspan="6">No candidates found</td></tr>'}
      </tbody>
    </table>

    <h2>Products with this Tag</h2>
    <p>These are the training products that have the "{html.escape(tag)}" tag (battle rank {current_tag_rank}).
    The residual is the encoded taxonomy value minus all coefficients from previously processed tags.</p>
    <table class="detail-table">
      <thead>
        <tr>
          <th>Product ID</th>
          <th>Title</th>
          <th>Residual Before</th>
          <th>Taxonomy Before</th>
          <th>Tags Already Processed</th>
          <th>Residual After</th>
          <th>Taxonomy After</th>
          <th>Tags Not Yet Processed</th>
        </tr>
      </thead>
      <tbody>
        {"".join(product_rows) if product_rows else '<tr><td colspan="8">No products found</td></tr>'}
      </tbody>
    </table>
  </section>

  <footer>
    <p><a href="../../../fold_{fold}.html">← Back to Fold {fold}</a></p>
  </footer>
</body>
</html>
"""
    output_path.write_text(page_html)


def _write_umllr_pages(output_dir: Path, summary: Dict[str, Any], conn=None, schema: str = "padjective") -> Dict[int, Path]:
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

    # Load taxonomy info for detail pages (both by path and by id)
    taxonomy_info_by_path: Dict[str, Dict[str, str]] = {}
    taxonomy_info_by_id: Dict[str, Dict[str, str]] = {}
    if conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                """
                SELECT taxonomy_id, taxonomy_name, taxonomy_path
                FROM cantbuymelove.taxonomy
                WHERE taxonomy_path !~ '[>/|]'
                """
            )
            rows = cur.fetchall()
            for row in rows:
                path = row["taxonomy_path"]
                tid = row["taxonomy_id"]
                info = {
                    "taxonomy_id": tid,
                    "taxonomy_name": row["taxonomy_name"] or "",
                    "taxonomy_path": path or "",
                }
                if path:
                    taxonomy_info_by_path[path] = info
                if tid:
                    taxonomy_info_by_id[tid] = info

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
            # taxonomy_path is least-significant-first (e.g., "12.6.2")
            # Database stores paths in the same format: category.subcategory.subsubcategory
            # where least-significant p-adic digit = top-level category
            taxonomy_name = taxonomy_names.get(taxonomy_path, "")
            # Create link to tag debug page
            tag_url_safe = urllib.parse.quote(tag, safe='')
            tag_link = f'<a href="fold_{fold}/tags/{tag_url_safe}/index.html">{_format_long_tag(tag)}</a>'
            non_zero_rows.append(
                "<tr>"
                f"<td>{tag_link}</td>"
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

        # Generate prediction detail pages if conn is available (stored at output_dir level)
        prediction_details = {}
        if conn:
            try:
                prediction_details = _load_prediction_details(conn, fold, schema)

                # Create prediction directory for this fold at output_dir level
                pred_dir = output_dir / f"prediction/{fold}"
                pred_dir.mkdir(parents=True, exist_ok=True)

                # Generate detail pages for each product
                for product_id, detail in prediction_details.items():
                    detail_path = pred_dir / f"{product_id}.html"
                    _write_prediction_detail_page(
                        detail_path,
                        fold,
                        product_id,
                        detail,
                        taxonomy_info_by_path,
                        taxonomy_info_by_id,
                        prime_base
                    )
            except Exception as e:
                print(f"Warning: Could not generate prediction detail pages for fold {fold}: {e}")

            # Generate tag debug pages for each non-zero coefficient tag
            try:
                for row in coeff_rows:
                    if row["coefficient"] != 0:
                        tag = row["tag"]
                        tag_url_safe = urllib.parse.quote(tag, safe='')
                        tag_debug_dir = umllr_dir / f"fold_{fold}" / "tags" / tag_url_safe
                        tag_debug_path = tag_debug_dir / "index.html"
                        _write_tag_debug_page(
                            tag_debug_path,
                            fold,
                            tag,
                            conn,
                            schema,
                            prime_base,
                            taxonomy_info_by_path,
                            tag_rankings,
                        )
            except Exception as e:
                print(f"Warning: Could not generate tag debug pages for fold {fold}: {e}")

        # Build prediction table rows with links to detail pages
        prediction_table_rows_list = []
        for row in prediction_rows:
            product_id = row['product_id']
            if product_id in prediction_details:
                # Link to detail page (relative to umllr directory)
                product_id_cell = f'<a href="../prediction/{fold}/{product_id}.html">{product_id}</a>'
            else:
                product_id_cell = str(product_id)

            prediction_table_rows_list.append(
                "<tr>"
                f"<td>{product_id_cell}</td>"
                f"<td>{row['true_value']}</td>"
                f"<td>{row['predicted_value']}</td>"
                f"<td>{row['loss']:.8f}</td>"
                "</tr>"
            )

        prediction_table_rows = "\n".join(prediction_table_rows_list)
        if not prediction_table_rows:
            prediction_table_rows = '<tr><td colspan="4">No test predictions available for this fold.</td></tr>'

        mean_loss = metric.get("mean_loss", 0)
        num_predictions = len(prediction_rows)
        accuracy_val = metric.get("accuracy")
        accuracy_text = f"{accuracy_val * 100:.2f}%" if accuracy_val is not None else "—"
        f1_val = metric.get("f1")
        f1_text = f"{f1_val:.4f}" if f1_val is not None else "—"
        breakdown_rows = metric.get("loss_breakdown", [])
        total_predictions = metric.get("total_predictions", num_predictions)

        breakdown_html = ""
        if breakdown_rows:
            breakdown_table_rows = []
            for entry in breakdown_rows:
                label = html.escape(str(entry.get("label", "")))
                count = int(entry.get("count", 0))
                percentage = (count / total_predictions * 100) if total_predictions else 0.0
                cost = entry.get("cost", 0.0)
                total_contrib = entry.get("total_contribution", 0.0)
                breakdown_table_rows.append(
                    f"<tr><td>{label}</td><td>{count:,}</td><td>{percentage:.2f}%</td><td>{cost:.6f}</td><td>{total_contrib:.6f}</td></tr>"
                )
            breakdown_body = "\n".join(breakdown_table_rows)
            breakdown_html = f"""
    <h2>P-adic loss breakdown</h2>
    <table class=\"umllr-table\">
      <thead>
        <tr><th>Agreement</th><th>Count</th><th>Share</th><th>Cost per mistake</th><th>Total contribution</th></tr>
      </thead>
      <tbody>
        {breakdown_body}
      </tbody>
    </table>
"""

        # Generate p-adic digit distribution chart
        digit_chart_html = ""
        digit_chart_path = umllr_dir / f"fold_{fold}_digit_distribution.png"
        generated_chart = _generate_padic_digit_distribution_chart(
            coeff_rows, tag_rankings, prime_base, digit_chart_path
        )
        if generated_chart:
            digit_chart_html = f"""
    <h2>P-adic Leading Zeros by Tag Rank</h2>
    <figure class="chart">
      <img src="fold_{fold}_digit_distribution.png" alt="P-adic leading zeros by tag rank" />
      <figcaption>Cumulative distribution of p-adic valuations (leading zeros in the taxonomy path) as tags are included by their battle ranking. Shows how many times the prime base divides each coefficient.</figcaption>
    </figure>
"""

        # Generate log proportion of non-zero coefficients chart
        log_proportion_chart_html = ""
        log_proportion_chart_path = umllr_dir / f"fold_{fold}_log_nonzero_proportion.png"
        generated_log_chart = _generate_log_nonzero_proportion_chart(
            coeff_rows, tag_rankings, log_proportion_chart_path
        )
        if generated_log_chart:
            log_proportion_chart_html = f"""
    <h2>Log Proportion of Non-Zero Coefficients</h2>
    <figure class="chart">
      <img src="fold_{fold}_log_nonzero_proportion.png" alt="Log proportion of non-zero coefficients" />
      <figcaption>Logarithms of the cumulative proportion of tags with non-zero coefficients (excludes infinite p-adic valuation) plotted against the logarithm of their battle ranking. Shows how the proportion of informative tags changes as more tags are included by battle ranking.</figcaption>
    </figure>
"""

        # Generate rolling average of non-zero coefficients chart
        rolling_chart_html = ""
        rolling_chart_path = umllr_dir / f"fold_{fold}_rolling_nonzero.png"
        generated_rolling_chart = _generate_rolling_nonzero_chart(
            coeff_rows, tag_rankings, rolling_chart_path, window_size=10
        )
        if generated_rolling_chart:
            rolling_chart_html = f"""
    <h2>Rolling Average of Non-Zero Coefficients</h2>
    <figure class="chart">
      <img src="fold_{fold}_rolling_nonzero.png" alt="Rolling average of non-zero coefficients" />
      <figcaption>Rolling average (window=10 tags) of the proportion of tags with non-zero coefficients, plotted by battle ranking. Shows how the informativeness of tags changes as we move through the ranking.</figcaption>
    </figure>
"""

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Importance-Optimised p-adic Linear Regression fold {fold} results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>Importance-Optimised p-adic Linear Regression fold {fold}</h1>
    <p><a href="../index.html">Back to index</a></p>
    <p><strong>P-adic loss (mean):</strong> {mean_loss:.8f} &middot; <strong>Test samples:</strong> {num_predictions:,} &middot; <strong>Accuracy:</strong> {accuracy_text} &middot; <strong>F1:</strong> {f1_text} &middot; <strong>Prime base:</strong> {metric['prime_base']} &middot; <strong>Max digit:</strong> {metric['max_digit']}</p>
{breakdown_html}
{digit_chart_html}
{log_proportion_chart_html}
{rolling_chart_html}
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




def _format_regression_stats_html(stats: Optional[Dict[str, Dict[str, float]]], x_label: str) -> str:
    """Format regression statistics as an HTML table."""
    if not stats:
        return ""

    model_names = {
        'umllr': 'Importance-Optimised p-adic LR',
        'lr': 'PCLR',
        'nn': 'PCNN',
        'ulr': 'ULR',
        'dummy': 'Dummy Baseline',
    }
    model_colors = {
        'umllr': '#0b6ce3',
        'lr': '#10b981',
        'nn': '#f59e0b',
        'ulr': '#8b5cf6',
        'dummy': '#94a3b8',
    }

    rows = []
    for key in ['umllr', 'lr', 'nn', 'ulr', 'dummy']:
        if key in stats:
            s = stats[key]
            color = model_colors[key]
            name = model_names[key]
            # Format p-value with scientific notation if very small
            if s['p_value'] < 0.001:
                p_str = f"{s['p_value']:.2e}"
            else:
                p_str = f"{s['p_value']:.4f}"
            rows.append(
                f'<tr><td style="color: {color}; font-weight: bold;">{name}</td>'
                f'<td style="text-align: right;">{s["slope"]:.6f}</td>'
                f'<td style="text-align: right;">{s["intercept"]:.4f}</td>'
                f'<td style="text-align: right;">{s["r_squared"]:.4f}</td>'
                f'<td style="text-align: right;">{p_str}</td></tr>'
            )

    if not rows:
        return ""

    return f"""
    <table style="width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem;">
      <thead>
        <tr style="background: #f8fafc; border-bottom: 2px solid #e2e8f0;">
          <th style="padding: 0.5rem; text-align: left;">Model</th>
          <th style="padding: 0.5rem; text-align: right;">Slope (per {x_label})</th>
          <th style="padding: 0.5rem; text-align: right;">Intercept</th>
          <th style="padding: 0.5rem; text-align: right;">R²</th>
          <th style="padding: 0.5rem; text-align: right;">p-value</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>"""


def _build_trends_section(
    trends_chart_path: Optional[Path],
    perf_vs_products_chart_path: Optional[Path],
    perf_vs_tags_chart_path: Optional[Path],
    params_vs_loss_chart_path: Optional[Path],
    output_dir: Path,
    perf_vs_products_stats: Optional[Dict[str, Dict[str, float]]] = None,
    perf_vs_tags_stats: Optional[Dict[str, Dict[str, float]]] = None,
    params_vs_loss_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Build HTML section for historical trends charts."""
    if not trends_chart_path:
        return ""

    chart_rel_path = trends_chart_path.relative_to(output_dir).as_posix()

    # Build optional charts HTML with regression stats
    perf_vs_products_html = ""
    if perf_vs_products_chart_path:
        products_chart_rel = perf_vs_products_chart_path.relative_to(output_dir).as_posix()
        products_stats_html = _format_regression_stats_html(perf_vs_products_stats, "product")
        perf_vs_products_html = f"""
    <figure class="chart" style="margin-top: 2rem;">
      <img src="{products_chart_rel}" alt="Model performance vs number of products" />
    </figure>
    {products_stats_html}"""

    perf_vs_tags_html = ""
    if perf_vs_tags_chart_path:
        tags_chart_rel = perf_vs_tags_chart_path.relative_to(output_dir).as_posix()
        tags_stats_html = _format_regression_stats_html(perf_vs_tags_stats, "tag")
        perf_vs_tags_html = f"""
    <figure class="chart" style="margin-top: 2rem;">
      <img src="{tags_chart_rel}" alt="Model performance vs number of distinct tags" />
    </figure>
    {tags_stats_html}"""

    params_vs_loss_html = ""
    if params_vs_loss_chart_path:
        params_chart_rel = params_vs_loss_chart_path.relative_to(output_dir).as_posix()

        # Build regression stats table
        params_stats_html = ""
        if params_vs_loss_stats:
            rows = []
            for key, label in [('with_dummy', 'With Dummy'), ('without_dummy', 'Without Dummy')]:
                if key in params_vs_loss_stats:
                    s = params_vs_loss_stats[key]
                    p_val = s['p_value']
                    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else f"{p_val:.2e}"
                    sig = "Yes" if p_val < 0.05 else "No"
                    rows.append(
                        f"<tr><td style=\"text-align: left;\">{label}</td>"
                        f"<td style=\"text-align: right;\">{s['slope']:.4f}</td>"
                        f"<td style=\"text-align: right;\">{s['intercept']:.4f}</td>"
                        f"<td style=\"text-align: right;\">{s['r_squared']:.4f}</td>"
                        f"<td style=\"text-align: right;\">{p_str}</td>"
                        f"<td style=\"text-align: center;\">{sig}</td>"
                        f"<td style=\"text-align: right;\">{int(s['n_points'])}</td></tr>"
                    )
            if rows:
                params_stats_html = f"""
    <div style="margin-top: 1rem; overflow-x: auto;">
      <p style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">
        <strong>Regression: p-adic loss = slope × log₁₀(params) + intercept</strong>
      </p>
      <table style="font-size: 0.85rem; border-collapse: collapse; width: 100%;">
        <thead>
          <tr style="background: #f1f5f9;">
            <th style="padding: 0.5rem; text-align: left; border-bottom: 2px solid #e2e8f0;">Line</th>
            <th style="padding: 0.5rem; text-align: right; border-bottom: 2px solid #e2e8f0;">Slope</th>
            <th style="padding: 0.5rem; text-align: right; border-bottom: 2px solid #e2e8f0;">Intercept</th>
            <th style="padding: 0.5rem; text-align: right; border-bottom: 2px solid #e2e8f0;">R²</th>
            <th style="padding: 0.5rem; text-align: right; border-bottom: 2px solid #e2e8f0;">p-value</th>
            <th style="padding: 0.5rem; text-align: center; border-bottom: 2px solid #e2e8f0;">Significant?</th>
            <th style="padding: 0.5rem; text-align: right; border-bottom: 2px solid #e2e8f0;">n</th>
          </tr>
        </thead>
        <tbody>
          {"".join(rows)}
        </tbody>
      </table>
    </div>"""

        params_vs_loss_html = f"""
    <figure class="chart" style="margin-top: 2rem;">
      <img src="{params_chart_rel}" alt="Model complexity vs performance (parameter count vs p-adic loss)" />
      <figcaption style="text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 0.5rem;">
        Parameter count (log scale) vs p-adic loss. Sparse models use fewer non-zero parameters.
      </figcaption>
    </figure>
    {params_stats_html}"""

    return f"""
  <section style="max-width: 70rem; margin: 2rem auto; background: white; border-radius: 1rem; box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12); padding: 2rem 1.5rem;">
    <h2 style="margin-top: 0;">Historical Performance Trends</h2>
    <p style="color: #64748b; margin-bottom: 1.5rem;">
      Tracking model performance and dataset growth over time. Lower p-adic loss indicates better predictions.
    </p>
    <figure class="chart">
      <img src="{chart_rel_path}" alt="Historical model performance trends" />
    </figure>
    {perf_vs_products_html}
    {perf_vs_tags_html}
    {params_vs_loss_html}
  </section>"""


def _build_index_markdown(
    stats: Dict[str, int],
    dataset_stats: Dict[str, int],
    generated: str,
    taxonomy_summary: Optional[Dict[str, Any]],
    umllr_summary: Optional[Dict[str, Any]],
    dummy_summary: Optional[Dict[str, Any]],
    taxonomy_fold_results: Optional[list[Dict[str, Any]]],
    taxonomy_pcnn_fold_results: Optional[list[Dict[str, Any]]],
    taxonomy_ulr_fold_results: Optional[list[Dict[str, Any]]],
    taxonomy_unn_fold_results: Optional[list[Dict[str, Any]]],
    trends_chart_path: Optional[Path],
    output_dir: Path,
) -> str:
    """Generate markdown version of the index page."""
    lines = [
        "# Padjective Tag Hierarchy",
        "",
        "Machine learning insights into Shopify product tag organization",
        "",
        "Data sourced from [cantbuymelove.industrial-linguistics.com](https://cantbuymelove.industrial-linguistics.com/) powering Shopify taxonomy classification and filtered to taxonomies with at least five products.",
        "",
        f"*Last updated {generated}*",
        "",
        "## Key Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Products used | {dataset_stats.get('products', 0):,} |",
        f"| Taxonomies covered | {dataset_stats.get('taxonomies', 0):,} |",
        f"| Tags used | {dataset_stats.get('unique_tags', 0):,} |",
        f"| Total tags | {dataset_stats.get('total_tags', 0):,} |",
        f"| Tag battles | {stats.get('battles', 0):,} |",
        "",
        "## Dataset Coverage",
        "",
    ]

    discarded_products = dataset_stats.get("discarded_products")
    discard_note = ""
    if discarded_products:
        discard_note = f" *{discarded_products:,} products were discarded due to missing or sparse taxonomy labels.*"

    lines.append(
        f"Training data spans **{dataset_stats.get('products', 0):,}** products across "
        f"**{dataset_stats.get('taxonomies', 0):,}** taxonomies. "
        f"Of **{dataset_stats.get('total_tags', 0):,}** total tags in the dataset, "
        f"**{dataset_stats.get('unique_tags', 0):,}** tags were used "
        f"(tags appearing fewer than 5 times were filtered out).{discard_note}"
    )
    lines.extend([
        "",
        "- [Explore the full dataset](dataset.html)",
        "- [View defective taxonomy labels](defective_taxonomy.html)",
        "",
        "## Models",
        "",
    ])

    # UMLLR model
    if umllr_summary and umllr_summary.get("metrics"):
        metrics = umllr_summary.get("metrics", [])
        avg_loss = sum(m["mean_loss"] for m in metrics) / len(metrics) if metrics else 0
        avg_nonzero_coeffs = sum(m.get("num_nonzero_coefficients", 0) for m in metrics) / len(metrics) if metrics else 0
        lines.extend([
            "### Importance-Optimised p-adic Linear Regression",
            "",
            "P-adic coefficients assigned to tags to predict taxonomy",
            "",
            f"- **Avg p-adic loss:** {avg_loss:.4f}",
        ])
        if avg_nonzero_coeffs > 0:
            lines.append(f"- **Avg non-zero coefficients:** {avg_nonzero_coeffs:,.0f}")
        lines.extend([
            "- [View model](umllr/index.html)",
            "",
        ])

    # Parameter Constrained Neural Network
    if taxonomy_pcnn_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        avg_input_weights = sum(r.get("num_input_weights", 0) for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        lines.extend([
            "### Parameter Constrained Neural Network",
            "",
            "Parameter constrained neural network predicting taxonomy from tags",
            "",
            f"- **Avg p-adic loss:** {avg_loss:.4f}",
        ])
        if avg_input_weights > 0:
            lines.append(f"- **Avg input weights:** {avg_input_weights:,.0f}")
        lines.extend([
            "- [View model](parameter_constrained_neural_network/index.html)",
            "",
        ])

    # Unconstrained Logistic Regression
    if taxonomy_ulr_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        avg_nonzero = sum(r["num_nonzero_params"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        lines.extend([
            "### Unconstrained Logistic Regression",
            "",
            "L1-regularized logistic regression using ALL tags with automatic feature selection",
            "",
            f"- **Avg p-adic loss:** {avg_loss:.4f}",
            f"- **Avg non-zero params:** {avg_nonzero:,.0f}",
            "- [View model](unconstrained_logistic_regression/index.html)",
            "",
        ])

    # Unconstrained Neural Network
    if taxonomy_unn_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        avg_nonzero = sum(r["num_nonzero_params"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        lines.extend([
            "### Unconstrained Neural Network",
            "",
            "L1-regularized neural network using ALL tags with weight pruning",
            "",
            f"- **Avg p-adic loss:** {avg_loss:.4f}",
            f"- **Avg non-zero params:** {avg_nonzero:,.0f}",
            "- [View model](unconstrained_neural_network/index.html)",
            "",
        ])

    # Parameter Constrained Logistic Regression
    if taxonomy_summary and taxonomy_fold_results:
        avg_padic_loss = sum(fold["padic_loss_mean"] for fold in taxonomy_fold_results) / len(taxonomy_fold_results)
        avg_params = sum(fold.get("num_params", 0) for fold in taxonomy_fold_results) / len(taxonomy_fold_results)
        lines.extend([
            "### Parameter Constrained Logistic Regression",
            "",
            "Parameter constrained logistic regression model predicting Shopify taxonomy from tags",
            "",
            f"- **Avg p-adic loss:** {avg_padic_loss:.4f}",
        ])
        if avg_params > 0:
            lines.append(f"- **Avg parameters:** {avg_params:,.0f}")
        lines.extend([
            "- [View model](parameter_constrained_logistic_regression/index.html)",
            "",
        ])

    # Dummy Baseline
    if dummy_summary:
        avg_loss = dummy_summary.get("average_loss")
        loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "—"
        lines.extend([
            "### Dummy Baseline",
            "",
            "Always predicts most common taxonomy (baseline for comparison)",
            "",
            f"- **Avg p-adic loss:** {loss_text}",
            "- **Parameters:** 1",
            "- [View model](dummy/index.html)",
            "",
        ])

    # ELO Rankings
    lines.extend([
        "### ELO-Inspired Rankings",
        "",
        "Battle-tested tag hierarchy from product title positions",
        "",
        f"- **Tag battles:** {stats.get('battles', 0):,}",
        "- [View rankings](elo/index.html)",
        "",
    ])

    # Taxonomy Distribution
    if taxonomy_summary:
        lines.extend([
            "## Taxonomy Distribution",
            "",
            "![Taxonomy class distribution](assets/taxonomy_distribution.png)",
            "",
            "*Distribution of products across the most common taxonomy classes*",
            "",
            "### Top 10 Taxonomy Classes",
            "",
            "| Taxonomy ID | Name | Path | Samples | Share |",
            "|-------------|------|------|---------|-------|",
        ])

        class_distribution = taxonomy_summary.get("class_distribution", [])[:10]
        for row in class_distribution:
            tax_id = row.get("taxonomy_id") or ""
            name = row.get("taxonomy_name") or "Unknown"
            path = row.get("taxonomy_path") or "Unknown"
            samples = row.get("sample_count", 0)
            share = row.get("sample_fraction", 0.0) * 100
            lines.append(f"| {tax_id} | {name} | {path} | {samples:,} | {share:.1f}% |")

        lines.extend([
            "",
            "### Tags with Strongest Signal",
            "",
            "| Tag | Top taxonomy | Weight | Max \\|weight\\| |",
            "|-----|--------------|--------|----------------|",
        ])

        top_tags = taxonomy_summary.get("top_tags", [])[:10]
        for row in top_tags:
            tag = row.get("tag") or ""
            top_tax = row.get("top_taxonomy_path") or "Unknown"
            weight = row.get("top_weight", 0.0)
            max_weight = row.get("max_abs_weight", 0.0)
            lines.append(f"| {tag} | {top_tax} | {weight:.4f} | {max_weight:.4f} |")

        lines.append("")

    # Historical Performance Trends
    if trends_chart_path:
        lines.extend([
            "## Historical Performance Trends",
            "",
            "Tracking model performance and dataset growth over time. Lower p-adic loss indicates better predictions.",
            "",
            "![Historical model performance trends](assets/historical_trends.png)",
            "",
        ])

    # Footer
    lines.extend([
        "---",
        "",
        "Source available on [GitHub](https://github.com/IFost-Sydney-Uni/padjective)",
    ])

    return "\n".join(lines)


def _build_index_html(
    output_dir: Path,
    stats: Dict[str, int],
    dataset_stats: Dict[str, int],
    dataset_page: Optional[Path],
    defective_taxonomy_page: Optional[Path],
    elo_page: Path,
    taxonomy_page: Optional[Path],
    umllr_page: Optional[Path],
    taxonomy_pcnn_page: Optional[Path],
    taxonomy_ulr_page: Optional[Path] = None,
    taxonomy_unn_page: Optional[Path] = None,
    taxonomy_summary: Optional[Dict[str, Any]] = None,
    umllr_summary: Optional[Dict[str, Any]] = None,
    dummy_summary: Optional[Dict[str, Any]] = None,
    taxonomy_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_pcnn_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_ulr_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_unn_fold_results: Optional[list[Dict[str, Any]]] = None,
    trends_chart_path: Optional[Path] = None,
    perf_vs_products_chart_path: Optional[Path] = None,
    perf_vs_tags_chart_path: Optional[Path] = None,
    params_vs_loss_chart_path: Optional[Path] = None,
    perf_vs_products_stats: Optional[Dict[str, Dict[str, float]]] = None,
    perf_vs_tags_stats: Optional[Dict[str, Dict[str, float]]] = None,
    params_vs_loss_stats: Optional[Dict[str, Dict[str, float]]] = None,
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
    dummy_card = ""
    dummy_page = dummy_summary.get("overview_page") if dummy_summary else None
    if dummy_summary and dummy_page:
        avg_loss = dummy_summary.get("average_loss")
        loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "—"
        dummy_card = f"""
  <div class="model-card">
    <h3>Dummy Baseline</h3>
    <p>Always predicts most common taxonomy (baseline for comparison)</p>
    <div class="card-metric">
      <span class="value">{loss_text}</span>
      <span class="label">Avg p-adic loss</span>
    </div>
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">1</span>
      <span class="label">Parameter</span>
    </div>
    <a href="{dummy_page}" class="card-link">View model →</a>
  </div>"""

    taxonomy_card = ""
    if taxonomy_summary and taxonomy_page:
        # Prefer showing p-adic loss from fold results if available
        if taxonomy_fold_results:
            avg_padic_loss = sum(fold["padic_loss_mean"] for fold in taxonomy_fold_results) / len(taxonomy_fold_results)
            metric_display = f"{avg_padic_loss:.4f}"
            metric_label = "Avg p-adic loss"
            avg_params = sum(fold.get("num_params", 0) for fold in taxonomy_fold_results) / len(taxonomy_fold_results)
            params_html = f"""
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">{avg_params:,.0f}</span>
      <span class="label">Avg parameters</span>
    </div>""" if avg_params > 0 else ""
        else:
            stats_block = taxonomy_summary.get("stats", {})
            cv_info = stats_block.get("cross_validation") or {}
            cv_accuracy = cv_info.get("mean_accuracy")
            if cv_accuracy is not None:
                metric_display = f"{cv_accuracy * 100:.1f}%"
            else:
                metric_display = "—"
            metric_label = "CV accuracy"
            params_html = ""

        taxonomy_card = f"""
  <div class="model-card">
    <h3>Parameter Constrained Logistic Regression</h3>
    <p>Logistic regression model predicting Shopify taxonomy from tags</p>
    <div class="card-metric">
      <span class="value">{metric_display}</span>
      <span class="label">{metric_label}</span>
    </div>{params_html}
    <a href="{taxonomy_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>
  </div>"""

    umllr_card = ""
    if umllr_summary and umllr_page and umllr_summary.get("metrics"):
        metrics = umllr_summary.get("metrics", [])
        # Use mean_loss (per-prediction average) not total loss
        avg_loss = sum(m["mean_loss"] for m in metrics) / len(metrics) if metrics else 0
        avg_accuracy = umllr_summary.get("average_accuracy")
        avg_f1 = umllr_summary.get("average_f1")
        accuracy_text = f"{avg_accuracy * 100:.2f}%" if avg_accuracy is not None else "—"
        f1_text = f"{avg_f1:.4f}" if avg_f1 is not None else "—"
        avg_nonzero_coeffs = sum(m.get("num_nonzero_coefficients", 0) for m in metrics) / len(metrics) if metrics else 0
        params_html = f"""
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">{avg_nonzero_coeffs:,.0f}</span>
      <span class="label">Avg non-zero coefficients</span>
    </div>""" if avg_nonzero_coeffs > 0 else ""
        umllr_card = f"""
  <div class="model-card">
    <h3>Importance-Optimised p-adic Linear Regression</h3>
    <p>P-adic coefficients assigned to tags to predict taxonomy</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>{params_html}
    <a href="{umllr_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>
  </div>"""

    pcnn_card = ""
    taxonomy_pcnn_link = ""
    if taxonomy_pcnn_page:
        taxonomy_pcnn_link = (
            f'<a href="{taxonomy_pcnn_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>'
        )

    if taxonomy_pcnn_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        avg_input_weights = sum(r.get("num_input_weights", 0) for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        params_html = f"""
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">{avg_input_weights:,.0f}</span>
      <span class="label">Avg input weights</span>
    </div>""" if avg_input_weights > 0 else ""
        pcnn_card = f"""
  <div class="model-card">
    <h3>Parameter Constrained Neural Network</h3>
    <p>Neural network predicting taxonomy from tags</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>{params_html}
    {taxonomy_pcnn_link or '<span class="card-link disabled">No report available</span>'}
  </div>"""

    ulr_card = ""
    taxonomy_ulr_link = ""
    if taxonomy_ulr_page:
        taxonomy_ulr_link = (
            f'<a href="{taxonomy_ulr_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>'
        )

    if taxonomy_ulr_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        avg_nonzero = sum(r["num_nonzero_params"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        ulr_card = f"""
  <div class="model-card">
    <h3>Unconstrained Logistic Regression</h3>
    <p>L1-regularized model using ALL tags</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">{avg_nonzero:,.0f}</span>
      <span class="label">Non-zero params</span>
    </div>
    {taxonomy_ulr_link or '<span class="card-link disabled">No report available</span>'}
  </div>"""

    unn_card = ""
    taxonomy_unn_link = ""
    if taxonomy_unn_page:
        taxonomy_unn_link = (
            f'<a href="{taxonomy_unn_page.relative_to(output_dir).as_posix()}" class="card-link">View model →</a>'
        )

    if taxonomy_unn_fold_results:
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        avg_nonzero = sum(r["num_nonzero_params"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        unn_card = f"""
  <div class="model-card">
    <h3>Unconstrained Neural Network</h3>
    <p>L1-regularized NN with weight pruning</p>
    <div class="card-metric">
      <span class="value">{avg_loss:.4f}</span>
      <span class="label">Avg p-adic loss</span>
    </div>
    <div class="card-metric" style="margin-top: 0.5rem;">
      <span class="value">{avg_nonzero:,.0f}</span>
      <span class="label">Non-zero params</span>
    </div>
    {taxonomy_unn_link or '<span class="card-link disabled">No report available</span>'}
  </div>"""

    # Combine model cards
    all_cards: list[str] = []
    if umllr_card:
        all_cards.append(umllr_card)
    if pcnn_card:
        all_cards.append(pcnn_card)
    if ulr_card:
        all_cards.append(ulr_card)
    if unn_card:
        all_cards.append(unn_card)
    if taxonomy_card:
        all_cards.append(taxonomy_card)
    if dummy_card:
        all_cards.append(dummy_card)
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
    taxonomy_dist_chart_path = None
    if taxonomy_summary:
        class_distribution = taxonomy_summary.get("class_distribution", [])[:15]

        # Generate taxonomy distribution chart
        if class_distribution:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(parents=True, exist_ok=True)
            chart_path = assets_dir / "taxonomy_distribution.png"
            taxonomy_dist_chart_path = _generate_taxonomy_distribution_chart(
                class_distribution, chart_path, top_n=15
            )

        # Show top 10 in table
        class_distribution_table = class_distribution[:10]
        class_rows = "\n".join(
            f"<tr><td>{html.escape(row.get('taxonomy_id') or '')}</td><td>{html.escape(row.get('taxonomy_name') or 'Unknown')}</td><td>{html.escape(row.get('taxonomy_path') or 'Unknown')}</td><td>{row.get('sample_count', 0):,}</td><td>{row.get('sample_fraction', 0.0) * 100:.1f}%</td></tr>"
            for row in class_distribution_table
        )
        if not class_rows:
            class_rows = '<tr><td colspan="5">No taxonomy class data available</td></tr>'

        top_tags = taxonomy_summary.get("top_tags", [])[:10]
        top_tag_rows = "\n".join(
            f"<tr><td>{_format_long_tag(row.get('tag') or '')}</td><td>{html.escape(row.get('top_taxonomy_path') or 'Unknown')}</td><td>{row.get('top_weight', 0.0):.4f}</td><td>{row.get('max_abs_weight', 0.0):.4f}</td></tr>"
            for row in top_tags
        )
        if not top_tag_rows:
            top_tag_rows = '<tr><td colspan="4">No tag signal data available</td></tr>'

        chart_html = ""
        if taxonomy_dist_chart_path:
            chart_rel_path = taxonomy_dist_chart_path.relative_to(output_dir).as_posix()
            chart_html = f"""
    <figure class="chart">
      <img src="{chart_rel_path}" alt="Taxonomy class distribution" />
      <figcaption>Distribution of products across the most common taxonomy classes</figcaption>
    </figure>"""

        taxonomy_overview_html = f"""
  <section class="taxonomy-classifier">
    <h2>Taxonomy distribution</h2>
    {chart_html}
    <h3>Top 10 taxonomy classes</h3>
    <table class="taxonomy-table">
      <thead><tr><th>Taxonomy ID</th><th>Name</th><th>Path</th><th>Samples</th><th>Share</th></tr></thead>
      <tbody>{class_rows}</tbody>
    </table>
    <h3>Tags with strongest signal</h3>
    <table class="tag-taxonomy-table">
      <thead><tr><th>Tag</th><th>Top taxonomy</th><th>Weight</th><th>Max |weight|</th></tr></thead>
      <tbody>{top_tag_rows}</tbody>
    </table>
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
      <span class="label">Tags used</span>
    </div>
    <div class="metric">
      <span class="value">{dataset_stats.get('total_tags', 0):,}</span>
      <span class="label">Total tags</span>
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
      <strong>{dataset_stats.get('taxonomies', 0):,}</strong> taxonomies.
      Of <strong>{dataset_stats.get('total_tags', 0):,}</strong> total tags in the dataset,
      <strong>{dataset_stats.get('unique_tags', 0):,}</strong> tags were used
      (tags appearing fewer than 5 times were filtered out).
      {discard_note}
      {dataset_link}
    </p>
  </section>

  <div class="model-cards">
{models_grid}
  </div>

  {taxonomy_overview_html}

  {_build_trends_section(trends_chart_path, perf_vs_products_chart_path, perf_vs_tags_chart_path, params_vs_loss_chart_path, output_dir, perf_vs_products_stats, perf_vs_tags_stats, params_vs_loss_stats)}

  <footer>
    <p>Source available on <a href="https://github.com/IFost-Sydney-Uni/padjective">GitHub</a></p>
  </footer>
</body>
</html>
"""

    (output_dir / "index.html").write_text(html_document, encoding="utf-8")

    # Generate markdown version
    markdown_document = _build_index_markdown(
        stats=stats,
        dataset_stats=dataset_stats,
        generated=generated,
        taxonomy_summary=taxonomy_summary,
        umllr_summary=umllr_summary,
        dummy_summary=dummy_summary,
        taxonomy_fold_results=taxonomy_fold_results,
        taxonomy_pcnn_fold_results=taxonomy_pcnn_fold_results,
        taxonomy_ulr_fold_results=taxonomy_ulr_fold_results,
        taxonomy_unn_fold_results=taxonomy_unn_fold_results,
        trends_chart_path=trends_chart_path,
        output_dir=output_dir,
    )
    (output_dir / "index.md").write_text(markdown_document, encoding="utf-8")


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


def _generate_lr_tag_rank_vs_coeff_chart(
    conn,
    cv_fold: int,
    tag_rankings: Dict[str, int],
    output_path: Path,
    schema: str = "padjective"
) -> Optional[Path]:
    """Generate a chart showing tag rank vs maximum coefficient magnitude.

    Args:
        conn: Database connection
        cv_fold: Cross-validation fold number
        tag_rankings: Dict mapping tag names to their battle rankings
        output_path: Where to save the chart
        schema: Database schema name

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not _table_exists(conn, schema, "taxonomy_pclr_coefficients"):
        return None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tag, taxonomy_id, coefficient
                FROM {schema}.taxonomy_pclr_coefficients
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (cv_fold,)
        )
        rows = cur.fetchall()

    if not rows:
        return None

    # Calculate max absolute coefficient for each tag across all taxonomies
    tag_max_coeff = {}
    for tag, taxonomy_id, coeff in rows:
        abs_coeff = abs(coeff)
        if tag not in tag_max_coeff or abs_coeff > tag_max_coeff[tag]:
            tag_max_coeff[tag] = abs_coeff

    # Build list of (rank, max_coeff) for tags with rankings
    data_points = []
    for tag, max_coeff in tag_max_coeff.items():
        rank = tag_rankings.get(tag.upper())
        if rank is not None:
            data_points.append((rank, max_coeff))

    if len(data_points) < 2:
        return None

    # Sort by rank
    data_points.sort(key=lambda x: x[0])
    ranks = [x[0] for x in data_points]
    coeffs = [x[1] for x in data_points]

    # Calculate linear regression
    ranks_array = np.array(ranks)
    coeffs_array = np.array(coeffs)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ranks_array, coeffs_array)
    r_squared = r_value ** 2

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(ranks, coeffs, color='#0b6ce3', alpha=0.6, s=50, label='Data points')

    # Add regression line
    line_x = np.array([min(ranks), max(ranks)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.8,
            label=f'Linear fit: y = {slope:.6f}x + {intercept:.4f}')

    # Add statistics text box
    sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
    if p_value < 0.001:
        p_text = 'p < 0.001'
    else:
        p_text = f'p = {p_value:.4f}'
    stats_text = f'R² = {r_squared:.4f}\n{p_text} {sig_marker}\nn = {len(ranks)}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Tag Battle Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max |Coefficient| across Taxonomies', fontsize=12, fontweight='bold')
    ax.set_title('Tag Rank vs Maximum Absolute Coefficient', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)  # Start y-axis at zero for proper magnitude comparison
    ax.legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _generate_nn_tag_rank_vs_weight_chart(
    conn,
    cv_fold: int,
    tag_rankings: Dict[str, int],
    output_path: Path,
    schema: str = "padjective"
) -> Optional[Path]:
    """Generate a chart showing tag rank vs maximum first-layer weight magnitude.

    Args:
        conn: Database connection
        cv_fold: Cross-validation fold number
        tag_rankings: Dict mapping tag names to their battle rankings
        output_path: Where to save the chart
        schema: Database schema name

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not _table_exists(conn, schema, "taxonomy_pcnn_input_weights"):
        return None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT tag, hidden_unit, weight
                FROM {schema}.taxonomy_pcnn_input_weights
                WHERE cv_fold = %s
                """
            ).format(schema=sql.Identifier(schema)),
            (cv_fold,)
        )
        rows = cur.fetchall()

    if not rows:
        return None

    # Calculate max absolute weight for each tag across all hidden units
    tag_max_weight = {}
    for tag, hidden_unit, weight in rows:
        abs_weight = abs(weight)
        if tag not in tag_max_weight or abs_weight > tag_max_weight[tag]:
            tag_max_weight[tag] = abs_weight

    # Build list of (rank, max_weight) for tags with rankings
    data_points = []
    for tag, max_weight in tag_max_weight.items():
        rank = tag_rankings.get(tag.upper())
        if rank is not None:
            data_points.append((rank, max_weight))

    if len(data_points) < 2:
        return None

    # Sort by rank
    data_points.sort(key=lambda x: x[0])
    ranks = [x[0] for x in data_points]
    weights = [x[1] for x in data_points]

    # Calculate linear regression
    ranks_array = np.array(ranks)
    weights_array = np.array(weights)
    slope, intercept, r_value, p_value, std_err = stats.linregress(ranks_array, weights_array)
    r_squared = r_value ** 2

    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(ranks, weights, color='#0b6ce3', alpha=0.6, s=50, label='Data points')

    # Add regression line
    line_x = np.array([min(ranks), max(ranks)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'r-', linewidth=2, alpha=0.8,
            label=f'Linear fit: y = {slope:.6f}x + {intercept:.4f}')

    # Add statistics text box
    sig_marker = '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
    if p_value < 0.001:
        p_text = 'p < 0.001'
    else:
        p_text = f'p = {p_value:.4f}'
    stats_text = f'R² = {r_squared:.4f}\n{p_text} {sig_marker}\nn = {len(ranks)}'
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Tag Battle Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max |Weight| across Hidden Units', fontsize=12, fontweight='bold')
    ax.set_title('Tag Rank vs Maximum Absolute First-Layer Weight', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)  # Start y-axis at zero for proper magnitude comparison
    ax.legend(loc='upper right', frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def _write_taxonomy_pclr_fold_pages(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
    conn=None,
    tag_rankings: Optional[Dict[str, int]] = None,
    schema: str = "padjective",
) -> Dict[int, Path]:
    """Write individual pages for each taxonomy classifier fold."""
    pages: Dict[int, Path] = {}
    if not fold_results:
        return pages

    tax_dir = output_dir / "parameter_constrained_logistic_regression"
    tax_dir.mkdir(parents=True, exist_ok=True)

    for fold_data in fold_results:
        fold = fold_data["cv_fold"]

        breakdown_rows = fold_data.get("loss_breakdown", [])
        total_predictions = fold_data.get("total_predictions", 0)
        breakdown_html = ""
        if breakdown_rows:
            row_cells: list[str] = []
            for row in breakdown_rows:
                label = html.escape(str(row.get("label", "")))
                count = int(row.get("count", 0))
                percentage = (count / total_predictions * 100) if total_predictions else 0.0
                cost = row.get("cost", 0.0)
                total_contrib = row.get("total_contribution", 0.0)
                row_cells.append(f"<tr><td>{label}</td><td>{count:,}</td><td>{percentage:.2f}%</td><td>{cost:.6f}</td><td>{total_contrib:.6f}</td></tr>")
            breakdown_body = "\n".join(row_cells)
            breakdown_html = f"""
    <h2>P-adic loss breakdown</h2>
    <table class=\"umllr-table\">
      <thead>
        <tr><th>Agreement</th><th>Count</th><th>Share</th><th>Cost per mistake</th><th>Total contribution</th></tr>
      </thead>
      <tbody>
        {breakdown_body}
      </tbody>
    </table>
"""
        else:
            breakdown_html = "<p>No prediction breakdown recorded for this fold.</p>"

        # Generate tag rank vs coefficient chart
        rank_coeff_chart_html = ""
        if conn and tag_rankings:
            rank_coeff_chart_path = tax_dir / f"fold_{fold}_rank_vs_coeff.png"
            generated_chart = _generate_lr_tag_rank_vs_coeff_chart(
                conn, fold, tag_rankings, rank_coeff_chart_path, schema
            )
            if generated_chart:
                rank_coeff_chart_html = f"""
    <h2>Tag Rank vs Coefficient Magnitude</h2>
    <figure class="chart">
      <img src="fold_{fold}_rank_vs_coeff.png" alt="Tag rank vs max coefficient magnitude" />
      <figcaption>Scatter plot showing the relationship between tag battle ranking and maximum absolute coefficient value across all taxonomies. Shows which tags have the strongest influence in the model.</figcaption>
    </figure>
"""

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PCLR Fold {fold} Results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>PCLR Fold {fold}</h1>
    <p><a href="index.html">Back to PCLR overview</a> &middot; <a href="../index.html">Back to main index</a></p>

    <h2>Fold metrics</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>Test accuracy</td><td>{fold_data['test_accuracy'] * 100:.2f}%</td></tr>
        <tr><td>Test F1 score</td><td>{fold_data['test_f1']:.4f}</td></tr>
        <tr><td>Hierarchical loss</td><td>{fold_data['test_hierarchical_loss']:.8f}</td></tr>
        <tr><td>P-adic loss (total)</td><td>{fold_data['padic_loss_total']:.8f}</td></tr>
        <tr><td>P-adic loss (mean)</td><td>{fold_data['padic_loss_mean']:.8f}</td></tr>
        <tr><td>Prime base</td><td>{fold_data['prime_base']}</td></tr>
        <tr><td>Training samples</td><td>{fold_data['num_train_samples']:,}</td></tr>
        <tr><td>Test samples</td><td>{fold_data['num_test_samples']:,}</td></tr>
        <tr><td>Trained at</td><td>{html.escape(fold_data['trained_at'] or 'Unknown')}</td></tr>
      </tbody>
    </table>

    {breakdown_html}

    {rank_coeff_chart_html}

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
    avg_accuracy_value = umllr_summary.get("average_accuracy")
    avg_f1_value = umllr_summary.get("average_f1")

    if metrics:
        # Use mean_loss (per-prediction average) not total loss
        avg_loss = sum(m.get("mean_loss", 0) for m in metrics) / len(metrics)
        prime_base = metrics[0].get("prime_base", 0)
        max_digit = metrics[0].get("max_digit", 0)
    else:
        avg_loss = 0
        prime_base = 0
        max_digit = 0

    avg_accuracy_text = f"{avg_accuracy_value * 100:.2f}%" if avg_accuracy_value is not None else "—"
    avg_f1_text = f"{avg_f1_value:.4f}" if avg_f1_value is not None else "—"

    fold_rows = []
    for metric in metrics:
        fold = metric["cv_fold"]
        link = page_lookup.get(fold)
        if link:
            link_text = f'<a href="{Path(link).name}">View details →</a>'
        else:
            link_text = "—"
        mean_loss = metric.get("mean_loss", 0)
        acc_value = metric.get("accuracy")
        f1_value = metric.get("f1")
        acc_text = f"{acc_value * 100:.2f}%" if acc_value is not None else "—"
        f1_text = f"{f1_value:.4f}" if f1_value is not None else "—"
        fold_rows.append(
            f"<tr>"
            f"<td>{fold}</td>"
            f"<td>{acc_text}</td>"
            f"<td>{f1_text}</td>"
            f"<td>{mean_loss:.8f}</td>"
            f"<td>{link_text}</td>"
            f"</tr>"
        )

    table_body = "\n".join(fold_rows) or '<tr><td colspan="5">No cross-validation folds recorded.</td></tr>'

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
        <span class="value">{avg_accuracy_text}</span>
        <span class="label">Mean accuracy</span>
      </div>
      <div class="metric">
        <span class="value">{avg_f1_text}</span>
        <span class="label">Mean F1</span>
      </div>
      <div class="metric">
        <span class="value">{prime_base}</span>
        <span class="label">Prime base</span>
      </div>
    </div>

    <h2>Cross-validation results</h2>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Accuracy</th><th>F1</th><th>P-adic loss (mean)</th><th>Details</th></tr>
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
    fold_results: list[Dict[str, Any]],
    fold_pages: Dict[int, Path],
) -> Path:
    """Write a dedicated page for taxonomy classifier results.

    Args:
        output_dir: Directory for output files
        taxonomy_summary: Legacy summary data (still used for class distribution and top tags)
        fold_results: Required fold-level results with training statistics
        fold_pages: Required mapping of fold numbers to their detail page paths
    """
    tax_dir = output_dir / "parameter_constrained_logistic_regression"
    tax_dir.mkdir(parents=True, exist_ok=True)

    stats_block = taxonomy_summary.get("stats", {})
    class_distribution = taxonomy_summary.get("class_distribution", [])[:20]
    top_tags_rows = taxonomy_summary.get("top_tags", [])[:30]
    trained_at = taxonomy_summary.get("trained_at", "Unknown")

    # Build summary metrics from fold results
    if not fold_results:
        raise ValueError("fold_results is required for taxonomy classifier page")

    num_folds = len(fold_results)
    total_train = sum(row["num_train_samples"] for row in fold_results)
    total_test = sum(row["num_test_samples"] for row in fold_results)
    avg_accuracy = sum(row["test_accuracy"] for row in fold_results) / num_folds
    avg_f1 = sum(row["test_f1"] for row in fold_results) / num_folds
    avg_padic_loss = sum(row["padic_loss_mean"] for row in fold_results) / num_folds

    summary_metrics = [
        f'<div class="metric"><span class="value">{num_folds}</span><span class="label">CV folds</span></div>',
        f'<div class="metric"><span class="value">{avg_accuracy * 100:.2f}%</span><span class="label">Mean test accuracy</span></div>',
    ]

    # Always show taxonomies if available
    if (taxonomies := stats_block.get("taxonomies")) is not None:
        summary_metrics.append(f'<div class="metric"><span class="value">{taxonomies:,}</span><span class="label">Taxonomy classes</span></div>')

    metrics_html = "\n".join(summary_metrics)

    # Build fold-specific details section
    fold_details_html = f"""
    <h3>Aggregate statistics</h3>
    <ul class="taxonomy-stats">
      <li><strong>Total train samples:</strong> {total_train:,}</li>
      <li><strong>Total test samples:</strong> {total_test:,}</li>
      <li><strong>Mean F1 (weighted):</strong> {avg_f1:.4f}</li>
      <li><strong>Mean p-adic loss:</strong> {avg_padic_loss:.8f}</li>
    </ul>
"""

    # Build fold results table
    if not fold_pages:
        raise ValueError("fold_pages is required for taxonomy classifier page")

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
            f"<td>{fold_data['padic_loss_mean']:.8f}</td>"
            f"<td>{link_text}</td>"
            f"</tr>"
        )
    fold_table_body = "\n".join(fold_rows)
    fold_results_html = f"""
    <h2>Cross-validation fold results</h2>
    <p>Average p-adic loss across all folds: <strong>{avg_padic_loss:.8f}</strong></p>
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
        f"<td>{_format_long_tag(row.get('tag') or '')}</td>"
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
  <title>Parameter Constrained Logistic Regression</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Parameter Constrained Logistic Regression</h1>
    <p class="tagline">Predicting Shopify taxonomy from product tags using parameter constrained logistic regression</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model summary</h2>
    <p>This parameter constrained logistic regression model predicts taxonomy IDs from product tags. Trained at: {html.escape(str(trained_at))}</p>

    <div class="metrics">
      {metrics_html}
    </div>

    {fold_details_html}

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


def _write_taxonomy_pcnn_fold_pages(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
    conn=None,
    tag_rankings: Optional[Dict[str, int]] = None,
    schema: str = "padjective",
) -> Dict[int, Path]:
    """Write individual pages for each parameter constrained neural network fold."""
    pages: Dict[int, Path] = {}
    if not fold_results:
        return pages

    nn_dir = output_dir / "parameter_constrained_neural_network"
    nn_dir.mkdir(parents=True, exist_ok=True)

    for fold_data in fold_results:
        fold = fold_data["cv_fold"]

        breakdown_rows = fold_data.get("loss_breakdown", [])
        total_predictions = fold_data.get("total_predictions", 0)
        breakdown_html = ""
        if breakdown_rows:
            row_cells: list[str] = []
            for row in breakdown_rows:
                label = html.escape(str(row.get("label", "")))
                count = int(row.get("count", 0))
                percentage = (count / total_predictions * 100) if total_predictions else 0.0
                cost = row.get("cost", 0.0)
                total_contrib = row.get("total_contribution", 0.0)
                row_cells.append(f"<tr><td>{label}</td><td>{count:,}</td><td>{percentage:.2f}%</td><td>{cost:.6f}</td><td>{total_contrib:.6f}</td></tr>")
            breakdown_body = "\n".join(row_cells)
            breakdown_html = f"""
    <h2>P-adic loss breakdown</h2>
    <table class=\"umllr-table\">
      <thead>
        <tr><th>Agreement</th><th>Count</th><th>Share</th><th>Cost per mistake</th><th>Total contribution</th></tr>
      </thead>
      <tbody>
        {breakdown_body}
      </tbody>
    </table>
"""
        else:
            breakdown_html = "<p>No prediction breakdown recorded for this fold.</p>"

        # Generate tag rank vs weight chart
        rank_weight_chart_html = ""
        if conn and tag_rankings:
            rank_weight_chart_path = nn_dir / f"fold_{fold}_rank_vs_weight.png"
            generated_chart = _generate_nn_tag_rank_vs_weight_chart(
                conn, fold, tag_rankings, rank_weight_chart_path, schema
            )
            if generated_chart:
                rank_weight_chart_html = f"""
    <h2>Tag Rank vs First-Layer Weight Magnitude</h2>
    <figure class="chart">
      <img src="fold_{fold}_rank_vs_weight.png" alt="Tag rank vs max first-layer weight magnitude" />
      <figcaption>Scatter plot showing the relationship between tag battle ranking and maximum absolute first-layer weight value across all hidden units. Shows which input features contribute most to the parameter constrained neural network's hidden representations.</figcaption>
    </figure>
"""

        max_tags = fold_data.get("max_tags")
        max_tags_row = f"<tr><td>Max tags</td><td>{max_tags if max_tags is not None else '—'}</td></tr>" if max_tags is not None else ""

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PCNN Fold {fold} Results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>PCNN Fold {fold}</h1>
    <p><a href="index.html">Back to PCNN overview</a> &middot; <a href="../index.html">Back to main index</a></p>

    <h2>Fold metrics</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>Test accuracy</td><td>{fold_data['test_accuracy'] * 100:.2f}%</td></tr>
        <tr><td>Test F1 score</td><td>{fold_data['test_f1']:.4f}</td></tr>
        <tr><td>Hierarchical loss</td><td>{fold_data['test_hierarchical_loss']:.8f}</td></tr>
        <tr><td>P-adic loss (total)</td><td>{fold_data['padic_loss_total']:.8f}</td></tr>
        <tr><td>P-adic loss (mean)</td><td>{fold_data['padic_loss_mean']:.8f}</td></tr>
        <tr><td>Prime base</td><td>{fold_data['prime_base']}</td></tr>
        <tr><td>Hidden layer size</td><td>{html.escape(fold_data['hidden_layers'])}</td></tr>
        {max_tags_row}
        <tr><td>Training samples</td><td>{fold_data['num_train_samples']:,}</td></tr>
        <tr><td>Test samples</td><td>{fold_data['num_test_samples']:,}</td></tr>
      </tbody>
    </table>

    {breakdown_html}

    {rank_weight_chart_html}

    <h2>About p-adic loss</h2>
    <p>P-adic loss measures the distance between predicted and true taxonomy using p-adic metric (base {fold_data['prime_base']}). Lower values indicate closer predictions in the taxonomy hierarchy. This metric is shared with the umllr model for comparison.</p>
  </section>
</body>
</html>
"""

        page_path = nn_dir / f"fold_{fold}.html"
        page_path.write_text(page_contents, encoding="utf-8")
        pages[fold] = page_path

    return pages


def _write_taxonomy_pcnn_overview_page(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
    fold_pages: Dict[int, Path],
) -> Path:
    """Write a main overview page for parameter constrained neural network classifier results."""
    nn_dir = output_dir / "parameter_constrained_neural_network"
    nn_dir.mkdir(parents=True, exist_ok=True)

    if not fold_results:
        raise ValueError("fold_results is required for parameter constrained neural network classifier page")

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
        "<li><strong>Hidden layer sizes:</strong> {}</li>".format(
            ", ".join(html.escape(layer) for layer in hidden_layers)
        ),
    ]
    if max_tags_values:
        hyperparameters_items.append(
            f"<li><strong>Max tags used:</strong> {', '.join(str(value) for value in max_tags_values)}</li>"
        )

    fold_rows = []
    for row in fold_results:
        fold = row["cv_fold"]
        link = fold_pages.get(fold)
        if link:
            link_text = f'<a href="{Path(link).name}">View details →</a>'
        else:
            link_text = "—"

        fold_rows.append(
            f"""<tr>
          <td>{fold}</td>
          <td>{row['test_accuracy'] * 100:.2f}%</td>
          <td>{row['test_f1']:.4f}</td>
          <td>{row['padic_loss_mean']:.6f}</td>
          <td>{link_text}</td>
        </tr>"""
        )

    fold_table_body = "\n".join(fold_rows)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Parameter Constrained Neural Network</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Parameter Constrained Neural Network</h1>
    <p class="tagline">Cross-validated PyTorch model predicting taxonomy IDs from tags</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model overview</h2>
    <p>Parameter constrained neural network classifier using PyTorch with configurable hidden layers to predict taxonomy IDs from product tags. Each taxonomy path is encoded as a p-adic integer and the model minimizes p-adic distance.</p>

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
        <span class="value">{avg_padic_loss:.8f}</span>
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

    <h2>Cross-validation results</h2>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Accuracy</th><th>F1</th><th>P-adic loss (mean)</th><th>Details</th></tr>
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


def _write_taxonomy_ulr_fold_pages(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
) -> Dict[int, Path]:
    """Write individual pages for each unconstrained logistic regression fold."""
    pages: Dict[int, Path] = {}
    if not fold_results:
        return pages

    ulr_dir = output_dir / "unconstrained_logistic_regression"
    ulr_dir.mkdir(parents=True, exist_ok=True)

    for fold_data in fold_results:
        fold = fold_data["cv_fold"]

        breakdown_rows = fold_data.get("loss_breakdown", [])
        total_predictions = fold_data.get("total_predictions", 0)
        breakdown_html = ""
        if breakdown_rows:
            row_cells: list[str] = []
            for row in breakdown_rows:
                label = html.escape(str(row.get("label", "")))
                count = int(row.get("count", 0))
                percentage = (count / total_predictions * 100) if total_predictions else 0.0
                cost = row.get("cost", 0.0)
                total_contrib = row.get("total_contribution", 0.0)
                row_cells.append(f"<tr><td>{label}</td><td>{count:,}</td><td>{percentage:.2f}%</td><td>{cost:.6f}</td><td>{total_contrib:.6f}</td></tr>")
            breakdown_body = "\n".join(row_cells)
            breakdown_html = f"""
    <h2>P-adic loss breakdown</h2>
    <table class=\"umllr-table\">
      <thead>
        <tr><th>Agreement</th><th>Count</th><th>Share</th><th>Cost per mistake</th><th>Total contribution</th></tr>
      </thead>
      <tbody>
        {breakdown_body}
      </tbody>
    </table>
"""
        else:
            breakdown_html = "<p>No prediction breakdown recorded for this fold.</p>"

        # Calculate sparsity
        num_nonzero = fold_data.get("num_nonzero_params", 0)
        num_total = fold_data.get("num_total_params", 1)
        sparsity_pct = (1 - num_nonzero / num_total) * 100 if num_total > 0 else 0

        page_contents = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>ULR Fold {fold} Results</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <section class="umllr-fold">
    <h1>ULR Fold {fold}</h1>
    <p><a href="index.html">Back to ULR overview</a> &middot; <a href="../index.html">Back to main index</a></p>

    <h2>Fold metrics</h2>
    <table class="umllr-table">
      <thead>
        <tr><th>Metric</th><th>Value</th></tr>
      </thead>
      <tbody>
        <tr><td>Test accuracy</td><td>{fold_data['test_accuracy'] * 100:.2f}%</td></tr>
        <tr><td>Test F1 score</td><td>{fold_data['test_f1']:.4f}</td></tr>
        <tr><td>Hierarchical loss</td><td>{fold_data['test_hierarchical_loss']:.8f}</td></tr>
        <tr><td>P-adic loss (total)</td><td>{fold_data['padic_loss_total']:.8f}</td></tr>
        <tr><td>P-adic loss (mean)</td><td>{fold_data['padic_loss_mean']:.8f}</td></tr>
        <tr><td>Prime base</td><td>{fold_data['prime_base']}</td></tr>
        <tr><td>Number of tags (input features)</td><td>{fold_data['num_tags']:,}</td></tr>
        <tr><td>Non-zero parameters</td><td>{num_nonzero:,} / {num_total:,} ({sparsity_pct:.1f}% sparse)</td></tr>
        <tr><td>L1 regularization (C)</td><td>{fold_data['l1_c']:.4f}</td></tr>
        <tr><td>Training samples</td><td>{fold_data['num_train_samples']:,}</td></tr>
        <tr><td>Test samples</td><td>{fold_data['num_test_samples']:,}</td></tr>
      </tbody>
    </table>

    {breakdown_html}

    <h2>About L1 regularization</h2>
    <p>L1 (Lasso) regularization promotes sparsity by driving many coefficients to exactly zero. This model uses ALL available tags ({fold_data['num_tags']:,}) but L1 regularization selects which features are actually used. The number of non-zero parameters ({num_nonzero:,}) indicates how many coefficients the model actually uses.</p>
  </section>
</body>
</html>
"""

        page_path = ulr_dir / f"fold_{fold}.html"
        page_path.write_text(page_contents, encoding="utf-8")
        pages[fold] = page_path

    return pages


def _write_taxonomy_ulr_overview_page(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
    fold_pages: Dict[int, Path],
) -> Path:
    """Write a main overview page for unconstrained logistic regression classifier results."""
    ulr_dir = output_dir / "unconstrained_logistic_regression"
    ulr_dir.mkdir(parents=True, exist_ok=True)

    if not fold_results:
        raise ValueError("fold_results is required for ULR classifier page")

    num_folds = len(fold_results)
    avg_accuracy = sum(row["test_accuracy"] for row in fold_results) / num_folds
    avg_f1 = sum(row["test_f1"] for row in fold_results) / num_folds
    avg_padic_loss = sum(row["padic_loss_mean"] for row in fold_results) / num_folds

    total_train = sum(row["num_train_samples"] for row in fold_results)
    total_test = sum(row["num_test_samples"] for row in fold_results)

    # Aggregate parameter stats
    avg_nonzero = sum(row["num_nonzero_params"] for row in fold_results) / num_folds
    avg_total = sum(row["num_total_params"] for row in fold_results) / num_folds
    avg_sparsity = (1 - avg_nonzero / avg_total) * 100 if avg_total > 0 else 0

    num_tags = fold_results[0]["num_tags"]
    l1_c = fold_results[0]["l1_c"]

    fold_rows: list[str] = []
    for row in fold_results:
        fold = row["cv_fold"]
        page_link = ""
        if fold in fold_pages:
            rel_path = fold_pages[fold].relative_to(ulr_dir)
            page_link = f'<a href="{rel_path.as_posix()}">View</a>'
        fold_rows.append(
            f"<tr><td>{fold}</td><td>{row['test_accuracy'] * 100:.2f}%</td>"
            f"<td>{row['test_f1']:.4f}</td><td>{row['padic_loss_mean']:.8f}</td>"
            f"<td>{row['num_nonzero_params']:,}</td><td>{page_link}</td></tr>"
        )

    fold_table_body = "\n".join(fold_rows)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Unconstrained Logistic Regression</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Unconstrained Logistic Regression</h1>
    <p class="tagline">L1-regularized model using ALL tags with automatic feature selection</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model overview</h2>
    <p>Unconstrained logistic regression classifier using L1 (Lasso) regularization to predict taxonomy IDs from product tags. Unlike parameter-constrained models, this classifier uses ALL available tags and relies on L1 regularization to achieve sparsity by driving unimportant coefficients to zero.</p>

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
        <span class="value">{avg_padic_loss:.8f}</span>
        <span class="label">Mean p-adic loss</span>
      </div>
      <div class="metric">
        <span class="value">{avg_nonzero:,.0f}</span>
        <span class="label">Avg non-zero params</span>
      </div>
    </div>

    <ul class="taxonomy-stats">
      <li><strong>Total train samples:</strong> {total_train:,}</li>
      <li><strong>Total test samples:</strong> {total_test:,}</li>
      <li><strong>Number of tags (input features):</strong> {num_tags:,}</li>
      <li><strong>Avg non-zero parameters:</strong> {avg_nonzero:,.0f} / {avg_total:,.0f} ({avg_sparsity:.1f}% sparse)</li>
      <li><strong>L1 regularization (C):</strong> {l1_c:.4f}</li>
    </ul>

    <h2>Cross-validation results</h2>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Accuracy</th><th>F1</th><th>P-adic loss (mean)</th><th>Non-zero params</th><th>Details</th></tr>
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

    page_path = ulr_dir / "index.html"
    page_path.write_text(page_html, encoding="utf-8")
    return page_path


def _write_taxonomy_unn_overview_page(
    output_dir: Path,
    fold_results: list[Dict[str, Any]],
) -> Path:
    """Write a main overview page for unconstrained neural network classifier results."""
    unn_dir = output_dir / "unconstrained_neural_network"
    unn_dir.mkdir(parents=True, exist_ok=True)

    if not fold_results:
        raise ValueError("fold_results is required for UNN classifier page")

    num_folds = len(fold_results)
    avg_accuracy = sum(row["test_accuracy"] for row in fold_results) / num_folds
    avg_f1 = sum(row["test_f1"] for row in fold_results) / num_folds
    avg_padic_loss = sum(row["padic_loss_mean"] for row in fold_results) / num_folds

    total_train = sum(row["num_train_samples"] for row in fold_results)
    total_test = sum(row["num_test_samples"] for row in fold_results)

    # Aggregate parameter stats
    avg_nonzero = sum(row["num_nonzero_params"] for row in fold_results) / num_folds
    avg_total = sum(row["num_total_params"] for row in fold_results) / num_folds
    avg_sparsity = (1 - avg_nonzero / avg_total) * 100 if avg_total > 0 else 0

    hidden_size = fold_results[0]["hidden_size"]
    l1_lambda = fold_results[0]["l1_lambda"]
    pruning_threshold = fold_results[0]["pruning_threshold"]

    fold_rows: list[str] = []
    for row in fold_results:
        fold = row["cv_fold"]
        sparsity = (1 - row["num_nonzero_params"] / row["num_total_params"]) * 100 if row["num_total_params"] > 0 else 0
        fold_rows.append(
            f"<tr><td>{fold}</td><td>{row['test_accuracy'] * 100:.2f}%</td>"
            f"<td>{row['test_f1']:.4f}</td><td>{row['padic_loss_mean']:.6f}</td>"
            f"<td>{row['num_nonzero_params']:,}</td><td>{sparsity:.1f}%</td></tr>"
        )

    fold_table_body = "\n".join(fold_rows)

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Unconstrained Neural Network</title>
  <link rel="stylesheet" href="../assets/styles.css" />
</head>
<body>
  <header class="hero">
    <h1>Unconstrained Neural Network</h1>
    <p class="tagline">Neural network with L1 regularization and weight pruning for sparse predictions</p>
  </header>

  <section>
    <p><a href="../index.html">← Back to index</a></p>

    <h2>Model overview</h2>
    <p>Unconstrained neural network classifier using L1 regularization during training followed by
    post-training weight pruning to achieve sparsity. Unlike parameter-constrained models, this
    classifier uses ALL available tags as input features, relying on the combination of L1
    regularization and pruning to eliminate unimportant connections.</p>

    <h3>Architecture</h3>
    <p>The network uses a single hidden layer with {hidden_size} neurons:</p>
    <ul>
      <li><strong>Input layer:</strong> All tags (one-hot encoded)</li>
      <li><strong>Hidden layer:</strong> {hidden_size} neurons with ReLU activation</li>
      <li><strong>Output layer:</strong> Softmax over all taxonomy classes</li>
    </ul>

    <h3>Training procedure</h3>
    <ol>
      <li>Train with L1 regularization (λ={l1_lambda}) to encourage small weights</li>
      <li>Apply weight pruning (threshold={pruning_threshold}) to zero out small weights</li>
      <li>The pruned model achieves significant sparsity with minimal performance loss</li>
    </ol>

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
      <div class="metric">
        <span class="value">{avg_nonzero:,.0f}</span>
        <span class="label">Avg non-zero params</span>
      </div>
      <div class="metric">
        <span class="value">{avg_sparsity:.1f}%</span>
        <span class="label">Sparsity</span>
      </div>
    </div>

    <ul class="taxonomy-stats">
      <li><strong>Total train samples:</strong> {total_train:,}</li>
      <li><strong>Total test samples:</strong> {total_test:,}</li>
      <li><strong>Hidden layer size:</strong> {hidden_size}</li>
      <li><strong>Avg non-zero parameters:</strong> {avg_nonzero:,.0f} / {avg_total:,.0f} ({avg_sparsity:.1f}% sparse)</li>
      <li><strong>L1 regularization (λ):</strong> {l1_lambda}</li>
      <li><strong>Pruning threshold:</strong> {pruning_threshold}</li>
    </ul>

    <h2>Cross-validation results</h2>
    <table class="umllr-summary">
      <thead>
        <tr><th>Fold</th><th>Accuracy</th><th>F1</th><th>P-adic loss (mean)</th><th>Non-zero params</th><th>Sparsity</th></tr>
      </thead>
      <tbody>
        {fold_table_body}
      </tbody>
    </table>

    <h2>Comparison with other models</h2>
    <p>The unconstrained neural network achieves the best p-adic loss among all models by using
    more parameters (after pruning), while the L1 regularization and pruning ensure that only
    the most important connections are retained. This demonstrates the tradeoff between model
    complexity and prediction accuracy.</p>
  </section>

  <footer>
    <p><a href="../index.html">← Back to index</a></p>
  </footer>
</body>
</html>"""

    page_path = unn_dir / "index.html"
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
                FROM {schema}.taxonomy_pclr_models
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
                SELECT
                    d.taxonomy_id,
                    d.taxonomy_path,
                    t.taxonomy_name,
                    d.sample_count,
                    d.sample_fraction
                FROM {schema}.taxonomy_pclr_class_distribution d
                LEFT JOIN cantbuymelove.taxonomy t ON d.taxonomy_id = t.taxonomy_id
                WHERE d.model_id = %s
                  AND d.taxonomy_path !~ '[>/|]'
                ORDER BY d.sample_fraction DESC, d.sample_count DESC, d.taxonomy_id
                """
            ).format(schema=sql.Identifier(schema)),
            (model_id,),
        )
        class_distribution = [
            {
                "taxonomy_id": row["taxonomy_id"],
                "taxonomy_path": row["taxonomy_path"],
                "taxonomy_name": row.get("taxonomy_name") or "Unknown",
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
                FROM {schema}.taxonomy_pclr_tag_summary
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
                FROM {schema}.taxonomy_pclr_top_tags
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


def _load_taxonomy_pclr_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy parameter constrained logistic regression fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_pclr_fold_results"):
        return None

    # Load coefficient counts per fold (number of non-zero parameters)
    coef_counts: Dict[int, int] = {}
    if _table_exists(conn, schema, "taxonomy_pclr_coefficients"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT cv_fold, COUNT(*) as num_params
                    FROM {schema}.taxonomy_pclr_coefficients
                    GROUP BY cv_fold
                    """
                ).format(schema=sql.Identifier(schema))
            )
            for row in cur:
                coef_counts[int(row["cv_fold"])] = int(row["num_params"])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, trained_at
                FROM {schema}.taxonomy_pclr_fold_results
                ORDER BY cv_fold
                """
            ).format(schema=sql.Identifier(schema))
        )
        results = []
        for row in cur:
            fold = int(row["cv_fold"])
            results.append({
                "cv_fold": fold,
                "test_accuracy": float(row["test_accuracy"]),
                "test_f1": float(row["test_f1"]),
                "test_hierarchical_loss": float(row["test_hierarchical_loss"]),
                "padic_loss_total": float(row["padic_loss_total"]),
                "padic_loss_mean": float(row["padic_loss_mean"]),
                "prime_base": int(row["prime_base"]),
                "num_train_samples": int(row["num_train_samples"]),
                "num_test_samples": int(row["num_test_samples"]),
                "trained_at": row["trained_at"].isoformat(timespec="seconds") if row["trained_at"] else None,
                "num_params": coef_counts.get(fold, 0),
            })

    if not results:
        return None

    encoding_cache: Dict[int, Dict[str, int]] = {}
    for entry in results:
        fold = entry["cv_fold"]
        if fold not in encoding_cache:
            encoding_cache[fold] = _load_taxonomy_encoding_lookup(conn, schema, fold)
        lookup = encoding_cache[fold]
        value_pairs: list[tuple[int, int]] = []
        with conn.cursor(row_factory=dict_row) as pred_cur:
            pred_cur.execute(
                sql.SQL(
                    """
                    SELECT true_taxonomy_id, predicted_taxonomy_id
                    FROM {schema}.taxonomy_pclr_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,),
            )
            for pred_row in pred_cur:
                true_id = pred_row["true_taxonomy_id"]
                pred_id = pred_row["predicted_taxonomy_id"]
                if true_id is None or pred_id is None:
                    continue
                true_value = lookup.get(str(true_id))
                pred_value = lookup.get(str(pred_id))
                if true_value is None or pred_value is None:
                    continue
                value_pairs.append((int(true_value), int(pred_value)))
        entry["loss_breakdown"] = _padic_breakdown_from_pairs(value_pairs, entry["prime_base"]) if value_pairs else []
        entry["total_predictions"] = len(value_pairs)

    return results


def _load_taxonomy_pcnn_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy parameter constrained neural network fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_pcnn_fold_results"):
        return None

    # Load input weight counts per fold (number of parameters in input layer)
    weight_counts: Dict[int, int] = {}
    if _table_exists(conn, schema, "taxonomy_pcnn_input_weights"):
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT cv_fold, COUNT(*) as num_params
                    FROM {schema}.taxonomy_pcnn_input_weights
                    GROUP BY cv_fold
                    """
                ).format(schema=sql.Identifier(schema))
            )
            for row in cur:
                weight_counts[int(row["cv_fold"])] = int(row["num_params"])

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, hidden_layers, max_tags
                FROM {schema}.taxonomy_pcnn_fold_results
                ORDER BY cv_fold
                """
            ).format(schema=sql.Identifier(schema))
        )
        results = []
        for row in cur:
            fold = int(row["cv_fold"])
            results.append({
                "cv_fold": fold,
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
                "num_input_weights": weight_counts.get(fold, 0),
            })

    if not results:
        return None

    encoding_cache: Dict[int, Dict[str, int]] = {}
    for entry in results:
        fold = entry["cv_fold"]
        if fold not in encoding_cache:
            encoding_cache[fold] = _load_taxonomy_encoding_lookup(conn, schema, fold)
        lookup = encoding_cache[fold]
        value_pairs: list[tuple[int, int]] = []
        with conn.cursor(row_factory=dict_row) as pred_cur:
            pred_cur.execute(
                sql.SQL(
                    """
                    SELECT true_taxonomy_id, predicted_taxonomy_id
                    FROM {schema}.taxonomy_pcnn_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,),
            )
            for pred_row in pred_cur:
                true_id = pred_row["true_taxonomy_id"]
                pred_id = pred_row["predicted_taxonomy_id"]
                if true_id is None or pred_id is None:
                    continue
                true_value = lookup.get(str(true_id))
                pred_value = lookup.get(str(pred_id))
                if true_value is None or pred_value is None:
                    continue
                value_pairs.append((int(true_value), int(pred_value)))
        entry["loss_breakdown"] = _padic_breakdown_from_pairs(value_pairs, entry["prime_base"]) if value_pairs else []
        entry["total_predictions"] = len(value_pairs)

    return results


def _load_taxonomy_ulr_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy unconstrained logistic regression (L1) fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_ulr_fold_results"):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, num_tags,
                       num_nonzero_params, num_total_params, l1_c
                FROM {schema}.taxonomy_ulr_fold_results
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
                "num_tags": int(row["num_tags"]),
                "num_nonzero_params": int(row["num_nonzero_params"]),
                "num_total_params": int(row["num_total_params"]),
                "l1_c": float(row["l1_c"]) if row["l1_c"] is not None else 1.0,
            })

    if not results:
        return None

    encoding_cache: Dict[int, Dict[str, int]] = {}
    for entry in results:
        fold = entry["cv_fold"]
        if fold not in encoding_cache:
            encoding_cache[fold] = _load_taxonomy_encoding_lookup(conn, schema, fold)
        lookup = encoding_cache[fold]
        value_pairs: list[tuple[int, int]] = []
        with conn.cursor(row_factory=dict_row) as pred_cur:
            pred_cur.execute(
                sql.SQL(
                    """
                    SELECT true_taxonomy_id, predicted_taxonomy_id
                    FROM {schema}.taxonomy_ulr_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,),
            )
            for pred_row in pred_cur:
                true_id = pred_row["true_taxonomy_id"]
                pred_id = pred_row["predicted_taxonomy_id"]
                if true_id is None or pred_id is None:
                    continue
                true_value = lookup.get(str(true_id))
                pred_value = lookup.get(str(pred_id))
                if true_value is None or pred_value is None:
                    continue
                value_pairs.append((int(true_value), int(pred_value)))
        entry["loss_breakdown"] = _padic_breakdown_from_pairs(value_pairs, entry["prime_base"]) if value_pairs else []
        entry["total_predictions"] = len(value_pairs)

    return results


def _load_taxonomy_unn_fold_results(conn, schema: str = "padjective") -> Optional[list[Dict[str, Any]]]:
    """Load taxonomy unconstrained neural network fold-based results."""
    if not _table_exists(conn, schema, "taxonomy_unn_fold_results"):
        return None

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT cv_fold, test_accuracy, test_f1, test_hierarchical_loss,
                       padic_loss_total, padic_loss_mean, prime_base,
                       num_train_samples, num_test_samples, hidden_size,
                       num_nonzero_params, num_total_params, l1_lambda, pruning_threshold
                FROM {schema}.taxonomy_unn_fold_results
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
                "hidden_size": int(row["hidden_size"]),
                "num_nonzero_params": int(row["num_nonzero_params"]),
                "num_total_params": int(row["num_total_params"]),
                "l1_lambda": float(row["l1_lambda"]) if row["l1_lambda"] is not None else 0.0001,
                "pruning_threshold": float(row["pruning_threshold"]) if row["pruning_threshold"] is not None else 0.01,
            })

    if not results:
        return None

    encoding_cache: Dict[int, Dict[str, int]] = {}
    for entry in results:
        fold = entry["cv_fold"]
        if fold not in encoding_cache:
            encoding_cache[fold] = _load_taxonomy_encoding_lookup(conn, schema, fold)
        lookup = encoding_cache[fold]
        value_pairs: list[tuple[int, int]] = []
        with conn.cursor(row_factory=dict_row) as pred_cur:
            pred_cur.execute(
                sql.SQL(
                    """
                    SELECT true_taxonomy_id, predicted_taxonomy_id
                    FROM {schema}.taxonomy_unn_predictions
                    WHERE cv_fold = %s
                    """
                ).format(schema=sql.Identifier(schema)),
                (fold,),
            )
            for pred_row in pred_cur:
                true_id = pred_row["true_taxonomy_id"]
                pred_id = pred_row["predicted_taxonomy_id"]
                if true_id is None or pred_id is None:
                    continue
                true_value = lookup.get(str(true_id))
                pred_value = lookup.get(str(pred_id))
                if true_value is None or pred_value is None:
                    continue
                value_pairs.append((int(true_value), int(pred_value)))
        entry["loss_breakdown"] = _padic_breakdown_from_pairs(value_pairs, entry["prime_base"]) if value_pairs else []
        entry["total_predictions"] = len(value_pairs)

    return results


def _generate_taxonomy_distribution_chart(class_distribution: list[dict], output_path: Path, top_n: int = 15) -> Optional[Path]:
    """Generate a horizontal bar chart showing taxonomy class distribution.

    Args:
        class_distribution: List of taxonomy distribution dicts with taxonomy_name, sample_count
        output_path: Where to save the chart
        top_n: Number of top taxonomies to show

    Returns:
        Path to generated chart, or None if insufficient data
    """
    if not class_distribution or len(class_distribution) < 2:
        return None

    # Take top N taxonomies
    data = class_distribution[:top_n]

    # Extract labels and values - use taxonomy_name instead of taxonomy_path
    labels = [row.get('taxonomy_name', 'Unknown') for row in data]
    counts = [row['sample_count'] for row in data]

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, max(6, len(data) * 0.4)))

    # Create bars
    bars = ax.barh(range(len(labels)), counts, color='#0b6ce3', alpha=0.8)

    # Customize
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    ax.set_title('Taxonomy Class Distribution', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2,
                f' {count:,}',
                ha='left', va='center', fontsize=9, fontweight='bold')

    # Invert y-axis so largest is on top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


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
                       umllr_mean_padic_loss, lr_mean_padic_loss, nn_mean_padic_loss,
                       dummy_mean_padic_loss, ulr_mean_padic_loss, unn_mean_padic_loss
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
    dummy_loss = [row[7] if row[7] is not None else None for row in rows]
    ulr_loss = [row[8] if row[8] is not None else None for row in rows]
    unn_loss = [row[9] if row[9] is not None else None for row in rows]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot p-adic loss trends
    if any(umllr_loss):
        ax1.plot(dates, umllr_loss, 'o-', label='Importance-Optimised p-adic LR', color='#0b6ce3', linewidth=2, markersize=6)
    if any(lr_loss):
        ax1.plot(dates, lr_loss, 's-', label='PCLR', color='#10b981', linewidth=2, markersize=6)
    if any(nn_loss):
        ax1.plot(dates, nn_loss, '^-', label='PCNN', color='#f59e0b', linewidth=2, markersize=6)
    if any(ulr_loss):
        ax1.plot(dates, ulr_loss, 'D-', label='ULR', color='#8b5cf6', linewidth=2, markersize=6)
    if any(unn_loss):
        ax1.plot(dates, unn_loss, 'p-', label='UNN', color='#ec4899', linewidth=2, markersize=6)
    if any(dummy_loss):
        ax1.plot(dates, dummy_loss, 'x-', label='Dummy Baseline', color='#94a3b8', linewidth=2, markersize=6)

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


def _generate_performance_vs_products_chart(conn, output_path: Path, schema: str = "padjective") -> Tuple[Optional[Path], Optional[Dict[str, Dict[str, float]]]]:
    """Generate a scatter plot showing model performance vs number of products.

    Returns:
        Tuple of (path to generated chart, regression statistics dict) or (None, None) if no data
    """
    if not _table_exists(conn, schema, "model_performance_history"):
        return None, None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT num_products, umllr_mean_padic_loss, lr_mean_padic_loss,
                       nn_mean_padic_loss, dummy_mean_padic_loss, ulr_mean_padic_loss,
                       unn_mean_padic_loss
                FROM {schema}.model_performance_history
                ORDER BY num_products
                """
            ).format(schema=sql.Identifier(schema))
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        return None, None

    num_products = [row[0] for row in rows]
    umllr_loss = [row[1] for row in rows]
    lr_loss = [row[2] for row in rows]
    nn_loss = [row[3] for row in rows]
    dummy_loss = [row[4] for row in rows]
    ulr_loss = [row[5] for row in rows]
    unn_loss = [row[6] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    regression_stats = {}

    # Helper function to plot scatter with regression line and return stats
    def plot_with_regression(x_data, y_data, label, color, marker='o'):
        valid_x = [x for x, y in zip(x_data, y_data) if y is not None]
        valid_y = [y for y in y_data if y is not None]
        if len(valid_x) >= 2:
            ax.scatter(valid_x, valid_y, label=label, color=color, s=60, alpha=0.8, marker=marker)
            # Calculate regression with scipy for p-value
            x_arr = np.array(valid_x)
            y_arr = np.array(valid_y)
            result = stats.linregress(x_arr, y_arr)
            x_line = np.linspace(min(valid_x), max(valid_x), 100)
            y_line = result.slope * x_line + result.intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            return {
                'slope': result.slope,
                'intercept': result.intercept,
                'r_squared': result.rvalue ** 2,
                'p_value': result.pvalue,
                'std_err': result.stderr,
            }
        return None

    if any(v is not None for v in umllr_loss):
        stat = plot_with_regression(num_products, umllr_loss, 'Importance-Optimised p-adic LR', '#0b6ce3', 'o')
        if stat:
            regression_stats['umllr'] = stat

    if any(v is not None for v in lr_loss):
        stat = plot_with_regression(num_products, lr_loss, 'PCLR', '#10b981', 's')
        if stat:
            regression_stats['lr'] = stat

    if any(v is not None for v in nn_loss):
        stat = plot_with_regression(num_products, nn_loss, 'PCNN', '#f59e0b', '^')
        if stat:
            regression_stats['nn'] = stat

    if any(v is not None for v in ulr_loss):
        stat = plot_with_regression(num_products, ulr_loss, 'ULR', '#8b5cf6', 'D')
        if stat:
            regression_stats['ulr'] = stat

    if any(v is not None for v in dummy_loss):
        stat = plot_with_regression(num_products, dummy_loss, 'Dummy Baseline', '#94a3b8', 'x')
        if stat:
            regression_stats['dummy'] = stat

    if any(v is not None for v in unn_loss):
        stat = plot_with_regression(num_products, unn_loss, 'UNN', '#ec4899', 'p')
        if stat:
            regression_stats['unn'] = stat

    ax.set_xlabel('Number of Products', fontsize=12, fontweight='bold')
    ax.set_ylabel('P-adic Loss (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs Dataset Size (Products)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, regression_stats


def _generate_performance_vs_tags_chart(conn, output_path: Path, schema: str = "padjective") -> Tuple[Optional[Path], Optional[Dict[str, Dict[str, float]]]]:
    """Generate a scatter plot showing model performance vs number of distinct tags.

    Returns:
        Tuple of (path to generated chart, regression statistics dict) or (None, None) if no data
    """
    if not _table_exists(conn, schema, "model_performance_history"):
        return None, None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT num_tags, umllr_mean_padic_loss, lr_mean_padic_loss,
                       nn_mean_padic_loss, dummy_mean_padic_loss, ulr_mean_padic_loss,
                       unn_mean_padic_loss
                FROM {schema}.model_performance_history
                ORDER BY num_tags
                """
            ).format(schema=sql.Identifier(schema))
        )
        rows = cur.fetchall()

    if not rows or len(rows) < 2:
        return None, None

    num_tags = [row[0] for row in rows]
    umllr_loss = [row[1] for row in rows]
    lr_loss = [row[2] for row in rows]
    nn_loss = [row[3] for row in rows]
    dummy_loss = [row[4] for row in rows]
    ulr_loss = [row[5] for row in rows]
    unn_loss = [row[6] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    regression_stats = {}

    # Helper function to plot scatter with regression line and return stats
    def plot_with_regression(x_data, y_data, label, color, marker='o'):
        valid_x = [x for x, y in zip(x_data, y_data) if y is not None]
        valid_y = [y for y in y_data if y is not None]
        if len(valid_x) >= 2:
            ax.scatter(valid_x, valid_y, label=label, color=color, s=60, alpha=0.8, marker=marker)
            # Calculate regression with scipy for p-value
            x_arr = np.array(valid_x)
            y_arr = np.array(valid_y)
            result = stats.linregress(x_arr, y_arr)
            x_line = np.linspace(min(valid_x), max(valid_x), 100)
            y_line = result.slope * x_line + result.intercept
            ax.plot(x_line, y_line, color=color, linestyle='--', linewidth=1.5, alpha=0.6)
            return {
                'slope': result.slope,
                'intercept': result.intercept,
                'r_squared': result.rvalue ** 2,
                'p_value': result.pvalue,
                'std_err': result.stderr,
            }
        return None

    if any(v is not None for v in umllr_loss):
        stat = plot_with_regression(num_tags, umllr_loss, 'Importance-Optimised p-adic LR', '#0b6ce3', 'o')
        if stat:
            regression_stats['umllr'] = stat

    if any(v is not None for v in lr_loss):
        stat = plot_with_regression(num_tags, lr_loss, 'PCLR', '#10b981', 's')
        if stat:
            regression_stats['lr'] = stat

    if any(v is not None for v in nn_loss):
        stat = plot_with_regression(num_tags, nn_loss, 'PCNN', '#f59e0b', '^')
        if stat:
            regression_stats['nn'] = stat

    if any(v is not None for v in ulr_loss):
        stat = plot_with_regression(num_tags, ulr_loss, 'ULR', '#8b5cf6', 'D')
        if stat:
            regression_stats['ulr'] = stat

    if any(v is not None for v in dummy_loss):
        stat = plot_with_regression(num_tags, dummy_loss, 'Dummy Baseline', '#94a3b8', 'x')
        if stat:
            regression_stats['dummy'] = stat

    if any(v is not None for v in unn_loss):
        stat = plot_with_regression(num_tags, unn_loss, 'UNN', '#ec4899', 'p')
        if stat:
            regression_stats['unn'] = stat

    ax.set_xlabel('Number of Distinct Tags', fontsize=12, fontweight='bold')
    ax.set_ylabel('P-adic Loss (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance vs Feature Space (Distinct Tags)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, regression_stats


def _generate_params_vs_loss_chart(
    conn,
    output_path: Path,
    schema: str = "padjective",
    taxonomy_pclr_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_pcnn_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_ulr_fold_results: Optional[list[Dict[str, Any]]] = None,
    taxonomy_unn_fold_results: Optional[list[Dict[str, Any]]] = None,
) -> tuple[Optional[Path], Dict[str, Dict[str, float]]]:
    """Generate a scatter plot showing parameter count vs p-adic loss for all models.

    This chart helps visualize the tradeoff between model complexity (parameters)
    and prediction quality (p-adic loss).

    Returns:
        Tuple of (path to generated chart, regression statistics dict)
        Returns (None, {}) if insufficient data
    """
    # Collect data points: (params, loss, label, color, marker)
    data_points = []

    # PCLR - get average params and loss
    if taxonomy_pclr_fold_results:
        avg_params = sum(r.get("num_params", 0) for r in taxonomy_pclr_fold_results) / len(taxonomy_pclr_fold_results)
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_pclr_fold_results) / len(taxonomy_pclr_fold_results)
        if avg_params > 0:
            data_points.append((avg_params, avg_loss, "PCLR", "#10b981", "s"))

    # PCNN - get average input weights and loss
    if taxonomy_pcnn_fold_results:
        avg_params = sum(r.get("num_input_weights", 0) for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_pcnn_fold_results) / len(taxonomy_pcnn_fold_results)
        if avg_params > 0:
            data_points.append((avg_params, avg_loss, "PCNN", "#f59e0b", "^"))

    # ULR - get average non-zero params and loss
    if taxonomy_ulr_fold_results:
        avg_params = sum(r["num_nonzero_params"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_ulr_fold_results) / len(taxonomy_ulr_fold_results)
        data_points.append((avg_params, avg_loss, "ULR", "#8b5cf6", "D"))

    # UNN - get average non-zero params and loss
    if taxonomy_unn_fold_results:
        avg_params = sum(r["num_nonzero_params"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        avg_loss = sum(r["padic_loss_mean"] for r in taxonomy_unn_fold_results) / len(taxonomy_unn_fold_results)
        data_points.append((avg_params, avg_loss, "UNN", "#ec4899", "p"))

    # Get Importance-Optimised p-adic LR (UMLLR) - use actual non-zero coefficients
    if _table_exists(conn, schema, "umllr_fold_metrics"):
        with conn.cursor() as cur:
            # Get average loss
            cur.execute(
                sql.SQL(
                    """
                    SELECT AVG(loss) as avg_loss
                    FROM {schema}.umllr_predictions
                    """
                ).format(schema=sql.Identifier(schema))
            )
            row = cur.fetchone()
            avg_loss = float(row[0]) if row and row[0] is not None else None

            # Get average non-zero coefficients per fold
            cur.execute(
                sql.SQL(
                    """
                    SELECT AVG(num_coeffs) FROM (
                        SELECT cv_fold, COUNT(*) as num_coeffs
                        FROM {schema}.umllr_tag_coefficients
                        WHERE coefficient != 0
                        GROUP BY cv_fold
                    ) sub
                    """
                ).format(schema=sql.Identifier(schema))
            )
            row = cur.fetchone()
            avg_nonzero = float(row[0]) if row and row[0] is not None else None

            if avg_loss is not None and avg_nonzero is not None:
                data_points.append((avg_nonzero, avg_loss, "Importance-Optimised", "#0b6ce3", "o"))

    # Get Dummy baseline (1 parameter - always predicts most common taxonomy)
    if _table_exists(conn, schema, "dummy_fold_metrics"):
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT AVG(loss) as avg_loss
                    FROM {schema}.dummy_fold_metrics
                    """
                ).format(schema=sql.Identifier(schema))
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                data_points.append((1, float(row[0]), "Dummy", "#94a3b8", "X"))

    if len(data_points) < 2:
        return None, {}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    for params, loss, label, color, marker in data_points:
        ax.scatter(params, loss, label=label, color=color, s=150, alpha=0.8, marker=marker, edgecolors='white', linewidths=2)
        # Add label next to point
        ax.annotate(label, (params, loss), textcoords="offset points", xytext=(10, 5),
                   fontsize=10, fontweight='bold', color=color)

    # Compute lines of best fit using log(params) as x
    # Separate points with and without dummy
    all_points = [(params, loss, label) for params, loss, label, _, _ in data_points]
    non_dummy_points = [(params, loss) for params, loss, label in all_points if label != "Dummy"]

    regression_stats: Dict[str, Dict[str, float]] = {}

    # Line of best fit WITH dummy (all points)
    if len(all_points) >= 2:
        log_params_all = np.array([np.log10(p) for p, _, _ in all_points])
        losses_all = np.array([l for _, l, _ in all_points])

        from scipy import stats as scipy_stats
        result_all = scipy_stats.linregress(log_params_all, losses_all)

        regression_stats['with_dummy'] = {
            'slope': result_all.slope,
            'intercept': result_all.intercept,
            'r_squared': result_all.rvalue ** 2,
            'r_value': result_all.rvalue,
            'p_value': result_all.pvalue,
            'std_err': result_all.stderr,
            'n_points': len(all_points),
        }

        # Generate line across the full x range
        x_range = np.linspace(min(log_params_all) - 0.3, max(log_params_all) + 0.3, 100)
        y_fit_all = result_all.slope * x_range + result_all.intercept

        # Convert back from log scale for plotting
        x_range_params = 10 ** x_range
        ax.plot(x_range_params, y_fit_all, '--', color='#94a3b8', linewidth=2, alpha=0.7,
                label=f'With Dummy (R²={result_all.rvalue**2:.3f})')

    # Line of best fit WITHOUT dummy
    if len(non_dummy_points) >= 2:
        log_params_no_dummy = np.array([np.log10(p) for p, _ in non_dummy_points])
        losses_no_dummy = np.array([l for _, l in non_dummy_points])

        result_no_dummy = scipy_stats.linregress(log_params_no_dummy, losses_no_dummy)

        regression_stats['without_dummy'] = {
            'slope': result_no_dummy.slope,
            'intercept': result_no_dummy.intercept,
            'r_squared': result_no_dummy.rvalue ** 2,
            'r_value': result_no_dummy.rvalue,
            'p_value': result_no_dummy.pvalue,
            'std_err': result_no_dummy.stderr,
            'n_points': len(non_dummy_points),
        }

        # Generate line across the non-dummy x range
        x_range_nd = np.linspace(min(log_params_no_dummy) - 0.3, max(log_params_no_dummy) + 0.3, 100)
        y_fit_no_dummy = result_no_dummy.slope * x_range_nd + result_no_dummy.intercept

        # Convert back from log scale for plotting
        x_range_params_nd = 10 ** x_range_nd
        ax.plot(x_range_params_nd, y_fit_no_dummy, '-', color='#ef4444', linewidth=2, alpha=0.7,
                label=f'Without Dummy (R²={result_no_dummy.rvalue**2:.3f})')

    ax.set_xlabel('Number of Parameters (non-zero for sparse models)', fontsize=12, fontweight='bold')
    ax.set_ylabel('P-adic Loss (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity vs Performance', fontsize=14, fontweight='bold', pad=15)
    ax.set_xscale('log')  # Log scale for x-axis since params vary by orders of magnitude
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(bottom=0)

    # Add legend for the regression lines
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path, regression_stats


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
        "total_tags": len(dataset.feature_names) + len(dataset.discarded_tags),
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

    # Create comprehensive database dumps for website recreation
    _create_comprehensive_dumps(precomputed_database, datadumps_dir, battle_schema)

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
.umllr-table {width: 100%; border-collapse: collapse; margin-bottom: 2rem; background: white; table-layout: fixed;}
.umllr-table th, .umllr-table td {border-bottom: 1px solid #e2e8f0; padding: 0.75rem 1rem; text-align: left; overflow-wrap: break-word; word-wrap: break-word;}
.umllr-table thead {background: #f8fafc;}
.umllr-table th:first-child, .umllr-table td:first-child {max-width: 40%; width: 40%; word-break: break-all;}
.tag-cell {max-width: 300px; overflow: hidden; text-overflow: ellipsis; word-break: break-all;}
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
        "SQL dumps (comprehensive)": datadumps_dir,
        "Top tags chart": chart_path,
    }

    # Write separate model pages
    elo_page = _write_elo_rankings_page(output_dir, leaderboard, chart_path, stats)

    taxonomy_summary = _collect_taxonomy_classifier_summary(
        precomputed_database, schema=battle_schema
    )
    taxonomy_pclr_fold_results = _load_taxonomy_pclr_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_page = None
    taxonomy_fold_pages = {}
    if taxonomy_summary:
        if taxonomy_pclr_fold_results:
            taxonomy_fold_pages = _write_taxonomy_pclr_fold_pages(
                output_dir,
                taxonomy_pclr_fold_results,
                conn=precomputed_database,
                tag_rankings=tag_rank_lookup,
                schema=battle_schema
            )
            taxonomy_page = _write_taxonomy_classifier_page(
                output_dir, taxonomy_summary, taxonomy_pclr_fold_results, taxonomy_fold_pages
            )

    umllr_summary = _load_umllr_results(precomputed_database, battle_schema)
    umllr_page = None
    if umllr_summary:
        umllr_summary["tag_rankings"] = tag_rank_lookup
        fold_pages = _write_umllr_pages(output_dir, umllr_summary, conn=precomputed_database, schema=battle_schema)
        # Add pages to summary before creating overview page so links work
        umllr_summary["pages"] = {
            fold: path.relative_to(output_dir).as_posix()
            for fold, path in fold_pages.items()
        }
        umllr_page = _write_umllr_overview_page(output_dir, umllr_summary)
        umllr_summary["overview_page"] = umllr_page.relative_to(output_dir).as_posix()

    dummy_summary = _load_dummy_results(precomputed_database, battle_schema)
    if dummy_summary:
        dummy_fold_pages = _write_dummy_fold_pages(output_dir, dummy_summary, conn=precomputed_database, schema=battle_schema)
        dummy_page = _write_dummy_overview_page(output_dir, dummy_summary, dummy_fold_pages)
        dummy_summary["overview_page"] = dummy_page.relative_to(output_dir).as_posix()

    taxonomy_pcnn_fold_results = _load_taxonomy_pcnn_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_pcnn_page = None
    if taxonomy_pcnn_fold_results:
        taxonomy_pcnn_fold_pages = _write_taxonomy_pcnn_fold_pages(
            output_dir,
            taxonomy_pcnn_fold_results,
            conn=precomputed_database,
            tag_rankings=tag_rank_lookup,
            schema=battle_schema
        )
        taxonomy_pcnn_page = _write_taxonomy_pcnn_overview_page(output_dir, taxonomy_pcnn_fold_results, taxonomy_pcnn_fold_pages)

    taxonomy_ulr_fold_results = _load_taxonomy_ulr_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_ulr_page = None
    if taxonomy_ulr_fold_results:
        taxonomy_ulr_fold_pages = _write_taxonomy_ulr_fold_pages(
            output_dir,
            taxonomy_ulr_fold_results,
        )
        taxonomy_ulr_page = _write_taxonomy_ulr_overview_page(output_dir, taxonomy_ulr_fold_results, taxonomy_ulr_fold_pages)

    taxonomy_unn_fold_results = _load_taxonomy_unn_fold_results(
        precomputed_database, schema=battle_schema
    )
    taxonomy_unn_page = None
    if taxonomy_unn_fold_results:
        taxonomy_unn_page = _write_taxonomy_unn_overview_page(
            output_dir, taxonomy_unn_fold_results
        )

    # Generate historical trends charts
    trends_chart_path = _generate_historical_trends_chart(
        precomputed_database,
        assets_dir / "historical_trends.png",
        schema=battle_schema
    )
    perf_vs_products_chart_path, perf_vs_products_stats = _generate_performance_vs_products_chart(
        precomputed_database,
        assets_dir / "performance_vs_products.png",
        schema=battle_schema
    )
    perf_vs_tags_chart_path, perf_vs_tags_stats = _generate_performance_vs_tags_chart(
        precomputed_database,
        assets_dir / "performance_vs_tags.png",
        schema=battle_schema
    )

    params_vs_loss_chart_path, params_vs_loss_stats = _generate_params_vs_loss_chart(
        precomputed_database,
        assets_dir / "params_vs_loss.png",
        schema=battle_schema,
        taxonomy_pclr_fold_results=taxonomy_pclr_fold_results,
        taxonomy_pcnn_fold_results=taxonomy_pcnn_fold_results,
        taxonomy_ulr_fold_results=taxonomy_ulr_fold_results,
        taxonomy_unn_fold_results=taxonomy_unn_fold_results,
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
        taxonomy_pcnn_page,
        taxonomy_ulr_page,
        taxonomy_unn_page,
        taxonomy_summary,
        umllr_summary,
        dummy_summary,
        taxonomy_pclr_fold_results,
        taxonomy_pcnn_fold_results,
        taxonomy_ulr_fold_results,
        taxonomy_unn_fold_results,
        trends_chart_path,
        perf_vs_products_chart_path,
        perf_vs_tags_chart_path,
        params_vs_loss_chart_path,
        perf_vs_products_stats,
        perf_vs_tags_stats,
        params_vs_loss_stats,
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
