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

    query = sql.SQL(
        """
        SELECT
            p.id,
            t.taxonomy_path
        FROM {products} AS p
        JOIN cantbuymelove.product_taxonomy pt ON pt.product_id = p.id
        JOIN cantbuymelove.taxonomy t ON t.taxonomy_id = pt.taxonomy_id
        WHERE p.product_title IS NOT NULL
          AND pt.taxonomy_id IS NOT NULL
        ORDER BY p.id
        """
    ).format(products=product_identifier)

    product_ids = []
    taxonomy_paths = []

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query)
        for row in cur:
            product_ids.append(row["id"])
            taxonomy_paths.append(row["taxonomy_path"])

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
