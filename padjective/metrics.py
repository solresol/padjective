"""Shared metric utilities for taxonomy classifiers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

_PATH_SPLIT_RE = re.compile(r"[>/|]+")
_NUMERIC_TAXONOMY_PATH_RE = re.compile(r"^\d+(?:\.\d+)*$")


@dataclass(frozen=True)
class PredictionSummary:
    """Compact evaluation summary shared across model families."""

    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float | None = None


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
        if _NUMERIC_TAXONOMY_PATH_RE.fullmatch(stripped):
            return tuple(part for part in stripped.split(".") if part)
        parts = [part.strip() for part in _PATH_SPLIT_RE.split(stripped) if part.strip()]
        if parts:
            return tuple(parts)
        return (stripped,)

    if isinstance(path, Sequence) and not isinstance(path, (bytes, bytearray)):
        parts = [str(part).strip() for part in path if str(part).strip()]
        return tuple(parts)

    text = str(path).strip()
    return (text,) if text else ()


def shared_prefix_depth(
    true_path: Sequence[str],
    pred_path: Sequence[str],
) -> int:
    """Return the number of shared root-to-leaf segments."""

    depth = 0
    for true_segment, pred_segment in zip(true_path, pred_path):
        if true_segment != pred_segment:
            break
        depth += 1
    return depth


def _mean_or_zero(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _summarize_shared_depths(
    *,
    exact_matches: Sequence[float],
    shared_depths: Sequence[float],
    scoring_ops: Sequence[float] | None = None,
) -> PredictionSummary:
    prefix1 = [1.0 if depth >= 1 else 0.0 for depth in shared_depths]
    prefix2 = [1.0 if depth >= 2 else 0.0 for depth in shared_depths]
    mean_scoring_ops = _mean_or_zero(scoring_ops) if scoring_ops is not None else None
    return PredictionSummary(
        exact_accuracy=_mean_or_zero(exact_matches),
        prefix1_accuracy=_mean_or_zero(prefix1),
        prefix2_accuracy=_mean_or_zero(prefix2),
        mean_shared_prefix_depth=_mean_or_zero(shared_depths),
        mean_scoring_ops=mean_scoring_ops,
    )


def summarize_taxonomy_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    taxonomy_paths: Mapping[Any, Sequence[str]],
    *,
    scoring_ops: Sequence[float] | None = None,
) -> PredictionSummary:
    """Return exact/prefix-depth metrics for taxonomy label predictions."""

    exact_matches: list[float] = []
    shared_depths: list[float] = []

    for true_label, pred_label in zip(y_true, y_pred):
        true_path = taxonomy_paths.get(true_label) or ()
        pred_path = taxonomy_paths.get(pred_label) or ()
        depth = float(shared_prefix_depth(true_path, pred_path))
        shared_depths.append(depth)
        exact_matches.append(1.0 if tuple(true_path) == tuple(pred_path) and true_path else 0.0)

    return _summarize_shared_depths(
        exact_matches=exact_matches,
        shared_depths=shared_depths,
        scoring_ops=scoring_ops,
    )


def decode_padic_digits(value: int, base: int) -> tuple[int, ...]:
    """Decode ``value`` into root-first base-``p`` digits."""

    if base < 2:
        raise ValueError("base must be at least 2")
    if value == 0:
        return (0,)

    digits: list[int] = []
    remaining = abs(int(value))
    while remaining > 0:
        digits.append(int(remaining % base))
        remaining //= base
    return tuple(digits)


def shared_prefix_depth_encoded(
    true_value: int,
    pred_value: int,
    base: int,
    *,
    true_depth: int,
) -> int:
    """Return shared prefix depth for raw encoded predictions.

    The comparison is capped at ``true_depth`` so a prediction with extra digits
    beyond the gold taxonomy cannot exceed the depth of the gold path.
    """

    if true_depth <= 0:
        return 0
    if true_value == pred_value:
        return true_depth

    diff = abs(int(true_value) - int(pred_value))
    valuation = 0
    while diff and diff % base == 0:
        diff //= base
        valuation += 1
    return min(valuation, true_depth)


def summarize_encoded_predictions(
    true_values: Sequence[int],
    pred_values: Sequence[int],
    *,
    base: int,
    true_depths: Sequence[int],
    scoring_ops: Sequence[float] | None = None,
) -> PredictionSummary:
    """Return exact/prefix-depth metrics for raw p-adic outputs."""

    exact_matches: list[float] = []
    shared_depths: list[float] = []

    for true_value, pred_value, true_depth in zip(true_values, pred_values, true_depths):
        shared_depths.append(
            float(
                shared_prefix_depth_encoded(
                    int(true_value),
                    int(pred_value),
                    base,
                    true_depth=int(true_depth),
                )
            )
        )
        exact_matches.append(1.0 if int(true_value) == int(pred_value) else 0.0)

    return _summarize_shared_depths(
        exact_matches=exact_matches,
        shared_depths=shared_depths,
        scoring_ops=scoring_ops,
    )


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

        steps = len(true_path) - shared_prefix_depth(true_path, pred_path)
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
