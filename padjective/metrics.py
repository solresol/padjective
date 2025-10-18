"""Shared metric utilities for taxonomy classifiers."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_PATH_SPLIT_RE = re.compile(r"[>/|]+")


def parse_taxonomy_path(path: Any) -> tuple[str, ...]:
    """Normalise a taxonomy path representation into a tuple of segments.

    Args:
        path: Raw taxonomy path value (string, sequence, or ``None``)

    Returns:
        Tuple of hierarchical segments ordered from root to leaf. Empty tuple
        when the path is missing or empty.
    """

    if path is None:
        return ()

    if isinstance(path, str):
        stripped = path.strip()
        if not stripped:
            return ()
        parts = [part.strip() for part in _PATH_SPLIT_RE.split(stripped) if part.strip()]
        if parts:
            return tuple(parts)
        return (stripped,)

    if isinstance(path, Sequence) and not isinstance(path, (bytes, bytearray)):
        parts = [str(part).strip() for part in path if str(part).strip()]
        return tuple(parts)

    text = str(path).strip()
    return (text,) if text else ()


def hierarchical_loss_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    taxonomy_paths: Mapping[Any, Sequence[str]],
    *,
    base: float = 1.1,
) -> float:
    """Compute the mean hierarchical loss for predictions.

    ``taxonomy_paths`` should map taxonomy identifiers to their corresponding
    hierarchy expressed as a tuple of segments from root to leaf. The loss is
    defined as ``base ** (-T)`` where ``T`` is the number of steps from the
    true taxonomy to the lowest common ancestor of the true and predicted
    taxonomies. Missing paths yield a loss of ``0.0`` for that sample.
    """

    if base <= 0:
        raise ValueError("hierarchical loss base must be positive")

    scores: list[float] = []
    for true_label, pred_label in zip(y_true, y_pred):
        true_path = taxonomy_paths.get(true_label)
        pred_path = taxonomy_paths.get(pred_label)
        if not true_path or not pred_path:
            scores.append(0.0)
            continue

        common = 0
        for true_segment, pred_segment in zip(true_path, pred_path):
            if true_segment != pred_segment:
                break
            common += 1

        steps = len(true_path) - common
        scores.append(base ** (-steps))

    if not scores:
        return 0.0

    return float(np.mean(scores))


def ensure_taxonomy_paths_cover_labels(
    labels: Iterable[Any], taxonomy_paths: Mapping[Any, Sequence[str]]
) -> None:
    """Validate that every label has a corresponding taxonomy path."""

    missing = {label for label in labels if label not in taxonomy_paths}
    if missing:
        missing_preview = ", ".join(map(str, sorted(missing)[:5]))
        raise ValueError(
            "Missing taxonomy_path entries for labels: "
            f"{missing_preview}{'…' if len(missing) > 5 else ''}"
        )


def build_taxonomy_path_map(
    metadata: pd.DataFrame,
    *,
    id_column: str = "taxonomy_id",
    path_column: str = "taxonomy_path",
) -> dict[Any, tuple[str, ...]]:
    """Construct a mapping from taxonomy identifier to parsed path segments."""

    if path_column not in metadata.columns:
        raise ValueError(f"metadata is missing required column '{path_column}'")
    if id_column not in metadata.columns:
        raise ValueError(f"metadata is missing required column '{id_column}'")

    taxonomy_info = metadata.loc[
        metadata[path_column].notna(), [id_column, path_column]
    ].drop_duplicates(subset=[id_column])

    mapping: dict[Any, tuple[str, ...]] = {}
    for taxonomy_id, taxonomy_path in taxonomy_info.itertuples(index=False):
        mapping[taxonomy_id] = parse_taxonomy_path(taxonomy_path)

    return mapping
