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



def calculate_cv_folds(
    conn,
    product_table: str = "cantbuymelove.product",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[int, int]:
    """Calculate cross-validation fold assignments for products.

    Uses :class:`~sklearn.model_selection.StratifiedKFold` to assign each
    product to a fold, stratifying by taxonomy_path. Products without taxonomy
    data are excluded from fold assignment. The same ``random_state`` ensures
    consistency with taxonomy classifier training and other components that
    rely on deterministic splits.
    """

    product_identifier = db.qualified_identifier(product_table)

    taxonomy_column_name = "taxonomy_path"
    column_query = sql.SQL(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        """
    )

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(column_query, ("cantbuymelove", "taxonomy"))
        available_columns: Set[str] = set()
        for row in cur.fetchall():
            if isinstance(row, dict):
                column_name = row.get("column_name")
            else:
                column_name = row[0]
            if column_name:
                available_columns.add(str(column_name))

    if "taxonomy_path" in available_columns:
        taxonomy_column_name = "taxonomy_path"
    elif "taxonomy_label" in available_columns:
        taxonomy_column_name = "taxonomy_label"

    query = sql.SQL(
        """
        SELECT
            p.id,
            t.{taxonomy_column} AS taxonomy_value
        FROM {products} AS p
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
        WHERE p.product_title IS NOT NULL
          AND pt.taxonomy_id IS NOT NULL
        ORDER BY p.id
        """
    ).format(
        products=product_identifier,
        taxonomy_column=sql.Identifier(taxonomy_column_name),
    )

    product_ids = []
    taxonomy_paths = []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            if isinstance(row, dict):
                product_id = row.get("id")
                taxonomy_value = row.get("taxonomy_value")
                if taxonomy_value is None:
                    taxonomy_value = row.get(taxonomy_column_name)
                if taxonomy_value is None:
                    taxonomy_value = row.get("taxonomy_path")
                if taxonomy_value is None:
                    taxonomy_value = row.get("taxonomy_label")
            else:  # pragma: no cover - defensive fallback for tuple rows
                if len(row) >= 2:
                    product_id, taxonomy_value = row[0], row[1]
                else:
                    continue

            if product_id is None or taxonomy_value is None:
                continue

            product_ids.append(product_id)
            taxonomy_paths.append(taxonomy_value)

    if not product_ids:
        return {}

    product_ids_array = np.array(product_ids)
    taxonomy_paths_array = np.array(taxonomy_paths, dtype=object)

    unique_labels, counts = np.unique(taxonomy_paths_array, return_counts=True)
    min_class_size = counts.min() if counts.size else 0
    can_stratify = min_class_size >= n_splits

    if can_stratify:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        warnings.warn(
            "Taxonomy distribution too sparse for StratifiedKFold; falling back to "
            "deterministic KFold splits.",
            RuntimeWarning,
            stacklevel=2,
        )
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_assignments: Dict[int, int] = {}

    for fold_idx, (_, test_idx) in enumerate(cv.split(product_ids_array, taxonomy_paths_array)):
        for idx in test_idx:
            fold_assignments[int(product_ids_array[idx])] = fold_idx

    return fold_assignments
