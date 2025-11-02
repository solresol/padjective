"""Shared utilities for cross-validation splits."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Set
import warnings

import numpy as np
from psycopg import sql
from psycopg.rows import dict_row
from sklearn.model_selection import KFold, StratifiedKFold

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from padjective import db
else:  # pragma: no cover - imported as a module
    from . import db

_PRODUCT_TAXONOMY_TABLE = "cantbuymelove.product_taxonomy"


def _get_table_columns(conn, schema: str, table: str) -> Set[str]:
    """Return the set of column names for ``schema.table``."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {row[0] for row in cur.fetchall()}


def _resolve_taxonomy_label_column(conn) -> str | None:
    """Choose the taxonomy column to use for stratification.

    Returns ``taxonomy_id`` when available, falls back to ``taxonomy_path``,
    and ultimately to ``None`` when neither column exists. In that case the
    caller should substitute a deterministic surrogate (such as the product
    identifier) to keep the split logic functional.
    """

    schema, table = _PRODUCT_TAXONOMY_TABLE.split(".", 1)
    columns = _get_table_columns(conn, schema, table)

    if "taxonomy_id" in columns:
        return "taxonomy_id"
    if "taxonomy_path" in columns:
        return "taxonomy_path"

    warnings.warn(
        "cantbuymelove.product_taxonomy does not expose taxonomy_id or taxonomy_path; "
        "falling back to product identifiers for CV fold stratification.",
        RuntimeWarning,
        stacklevel=2,
    )
    return None


def calculate_cv_folds(
    conn,
    product_table: str = "cantbuymelove.product",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[int, int]:
    """Calculate cross-validation fold assignments for products.

    Uses :class:`~sklearn.model_selection.StratifiedKFold` to assign each
    product to a fold. The function prioritises ``taxonomy_id`` when available,
    gracefully downgrades to ``taxonomy_path``, and finally relies on product
    identifiers as a deterministic surrogate should neither taxonomy column be
    present. The same ``random_state`` ensures consistency with taxonomy
    classifier training and other components that rely on deterministic splits.
    """

    product_identifier = db.qualified_identifier(product_table)
    taxonomy_identifier = db.qualified_identifier(_PRODUCT_TAXONOMY_TABLE)
    taxonomy_column = _resolve_taxonomy_label_column(conn)

    taxonomy_expression = (
        sql.SQL("pt.{column}").format(column=sql.Identifier(taxonomy_column))
        if taxonomy_column
        else sql.SQL("p.id")
    )

    query = sql.SQL(
        """
        SELECT
            p.id,
            {taxonomy_expression} AS taxonomy_label
        FROM {products} AS p
        JOIN {taxonomy_table} pt ON pt.product_id = p.id
        WHERE p.product_title IS NOT NULL
        ORDER BY p.id
        """
    ).format(
        products=product_identifier,
        taxonomy_table=taxonomy_identifier,
        taxonomy_expression=taxonomy_expression,
    )

    product_ids = []
    taxonomy_labels = []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            # Exclude products without taxonomy labels
            if row["taxonomy_label"] is None:
                continue
            product_ids.append(row["id"])
            taxonomy_labels.append(row["taxonomy_label"])

    if not product_ids:
        return {}

    product_ids_array = np.array(product_ids)
    taxonomy_labels_array = np.array(taxonomy_labels, dtype=object)

    unique_labels, counts = np.unique(taxonomy_labels_array, return_counts=True)
    min_class_size = counts.min() if counts.size else 0
    can_stratify = taxonomy_column is not None and min_class_size >= n_splits

    if can_stratify:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        if taxonomy_column is not None:
            warnings.warn(
                "Taxonomy distribution too sparse for StratifiedKFold; falling back to "
                "deterministic KFold splits.",
                RuntimeWarning,
                stacklevel=2,
            )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_assignments: Dict[int, int] = {}

    for fold_idx, (_, test_idx) in enumerate(cv.split(product_ids_array, taxonomy_labels_array)):
        for idx in test_idx:
            fold_assignments[int(product_ids_array[idx])] = fold_idx

    return fold_assignments
