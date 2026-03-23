"""Pure-Python benchmark runtime shared by the notebook and report bundles.

This module is intentionally self-contained: it does not depend on Postgres or
other local project modules so the same code can be embedded directly into the
published Hugging Face notebook.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


DEFAULT_PCLR_MAX_TAGS = 32
DEFAULT_PCNN_MAX_TAGS = 32
DEFAULT_PCNN_HIDDEN = 27
DEFAULT_UNN_HIDDEN = 12
DEFAULT_UNN_MAX_ITER = 80
DEFAULT_ULR_MAX_ITER = 200
DEFAULT_ZUBAREV_MAX_ITERATIONS = 2000
DEFAULT_RANDOM_SEEDS: tuple[int, ...] = (7, 13, 23, 37, 101)
DEFAULT_ABLATION_STRATEGIES: tuple[str, ...] = (
    "battle_elo",
    "frequency",
    "mean_title_position",
    "taxonomy_association",
)

_PATH_SPLIT_RE = re.compile(r"[>/|]+")
_SEGMENT_NUMBER_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class PredictionSummary:
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float | None = None


@dataclass(frozen=True)
class SnapshotTables:
    snapshot_metadata: dict[str, Any]
    snapshot_label: str
    products_all: pd.DataFrame
    products: pd.DataFrame
    tags: pd.DataFrame
    product_tags: pd.DataFrame
    title_tags: pd.DataFrame


@dataclass(frozen=True)
class ProductRecord:
    product_id: int
    product_key: str
    tags: List[str]
    encoded_path: int
    cv_fold: int
    taxonomy_id: str
    taxonomy_depth: int
    title_tag_positions: Tuple[Tuple[str, int, int], ...]


@dataclass(frozen=True)
class BattleRecord:
    winner_tag: str
    loser_tag: str
    cv_fold: int | None


@dataclass(frozen=True)
class TagCoefficient:
    tag: str
    coefficient: int
    sequence: int


@dataclass(frozen=True)
class Prediction:
    product_id: int
    true_value: int
    predicted_value: int
    loss: float


@dataclass(frozen=True)
class UMLLRFoldResult:
    cv_fold: int
    coefficients: List[TagCoefficient]
    predictions: List[Prediction]
    loss: float
    default_prediction: int
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float


@dataclass(frozen=True)
class ZubarevFoldResult:
    cv_fold: int
    coefficients: List[TagCoefficient]
    predictions: List[Prediction]
    loss: float
    default_prediction: int
    iterations_used: int
    exact_accuracy: float
    prefix1_accuracy: float
    prefix2_accuracy: float
    mean_shared_prefix_depth: float
    mean_scoring_ops: float
    num_nonzero_params: int


@dataclass(frozen=True)
class NodeModel:
    prefix: tuple[str, ...]
    classifier: LogisticRegression | None
    majority_child: str | None
    majority_leaf_taxonomy_id: str
    children: tuple[str, ...]


def parse_taxonomy_path(path: Any) -> tuple[str, ...]:
    if path is None:
        return ()

    if isinstance(path, str):
        stripped = path.strip()
        if not stripped:
            return ()
        if re.fullmatch(r"\d+(?:\.\d+)*", stripped):
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


def parse_taxonomy_digits(path_value: str | None) -> Tuple[int, ...]:
    if not path_value:
        return ()

    digits: List[int] = []
    for segment in parse_taxonomy_path(path_value):
        segment = segment.strip()
        if not segment:
            continue
        try:
            digits.append(int(segment))
            continue
        except ValueError:
            matches = _SEGMENT_NUMBER_RE.findall(segment)
            if matches:
                digits.extend(int(match) for match in matches)
    return tuple(digits)


def shared_prefix_depth(true_path: Sequence[str], pred_path: Sequence[str]) -> int:
    depth = 0
    for true_segment, pred_segment in zip(true_path, pred_path):
        if true_segment != pred_segment:
            break
        depth += 1
    return depth


def encode_path(digits: Sequence[int], base: int) -> int:
    value = 0
    for power, digit in enumerate(digits):
        value += int(digit) * (base ** power)
    return int(value)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value in {2, 3}:
        return True
    if value % 2 == 0:
        return False
    limit = int(math.isqrt(value)) + 1
    for factor in range(3, limit, 2):
        if value % factor == 0:
            return False
    return True


def next_prime(min_value: int) -> int:
    candidate = max(2, int(min_value) + 1)
    while True:
        if is_prime(candidate):
            return candidate
        candidate += 1


def p_adic_distance(a: int, b: int, base: int) -> float:
    if int(a) == int(b):
        return 0.0
    diff = abs(int(a) - int(b))
    valuation = 0
    while diff and diff % base == 0:
        diff //= base
        valuation += 1
    return float(base ** (-valuation))


def summarize_taxonomy_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    taxonomy_paths: Mapping[Any, Sequence[str]],
    *,
    scoring_ops: Sequence[float] | None = None,
) -> PredictionSummary:
    exact_matches: list[float] = []
    shared_depths: list[float] = []

    for true_label, pred_label in zip(y_true, y_pred):
        true_path = taxonomy_paths.get(true_label) or ()
        pred_path = taxonomy_paths.get(pred_label) or ()
        depth = float(shared_prefix_depth(true_path, pred_path))
        shared_depths.append(depth)
        exact_matches.append(1.0 if tuple(true_path) == tuple(pred_path) and true_path else 0.0)

    prefix1 = [1.0 if depth >= 1 else 0.0 for depth in shared_depths]
    prefix2 = [1.0 if depth >= 2 else 0.0 for depth in shared_depths]
    mean_scoring_ops = float(np.mean(scoring_ops)) if scoring_ops else None
    return PredictionSummary(
        exact_accuracy=float(np.mean(exact_matches)) if exact_matches else 0.0,
        prefix1_accuracy=float(np.mean(prefix1)) if prefix1 else 0.0,
        prefix2_accuracy=float(np.mean(prefix2)) if prefix2 else 0.0,
        mean_shared_prefix_depth=float(np.mean(shared_depths)) if shared_depths else 0.0,
        mean_scoring_ops=mean_scoring_ops,
    )


def shared_prefix_depth_encoded(
    true_value: int,
    pred_value: int,
    base: int,
    *,
    true_depth: int,
) -> int:
    if true_depth <= 0:
        return 0
    if int(true_value) == int(pred_value):
        return int(true_depth)
    diff = abs(int(true_value) - int(pred_value))
    valuation = 0
    while diff and diff % base == 0:
        diff //= base
        valuation += 1
    return min(valuation, int(true_depth))


def summarize_encoded_predictions(
    true_values: Sequence[int],
    pred_values: Sequence[int],
    *,
    base: int,
    true_depths: Sequence[int],
    scoring_ops: Sequence[float] | None = None,
) -> PredictionSummary:
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

    prefix1 = [1.0 if depth >= 1 else 0.0 for depth in shared_depths]
    prefix2 = [1.0 if depth >= 2 else 0.0 for depth in shared_depths]
    mean_scoring_ops = float(np.mean(scoring_ops)) if scoring_ops else None
    return PredictionSummary(
        exact_accuracy=float(np.mean(exact_matches)) if exact_matches else 0.0,
        prefix1_accuracy=float(np.mean(prefix1)) if prefix1 else 0.0,
        prefix2_accuracy=float(np.mean(prefix2)) if prefix2 else 0.0,
        mean_shared_prefix_depth=float(np.mean(shared_depths)) if shared_depths else 0.0,
        mean_scoring_ops=mean_scoring_ops,
    )


def _hf_headers(hf_token: str | None) -> dict[str, str]:
    if hf_token:
        return {"Authorization": f"Bearer {hf_token}"}
    return {}


def _load_json_url(url: str, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def _load_json_url_df(url: str, *, headers: dict[str, str] | None = None) -> pd.DataFrame:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request) as response:
        return pd.read_json(response, lines=True, compression="infer")


def _load_json_path(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl_path(path: Path) -> pd.DataFrame:
    return pd.read_json(path, lines=True, compression="infer")


def _product_frames_from_raw(products_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = products_raw.drop(columns=["tag_features"], errors="ignore").copy()
    if "cv_fold" in products.columns:
        products = products.dropna(subset=["cv_fold"]).copy()
        products["cv_fold"] = products["cv_fold"].astype(int)
    products = products.sort_values("product_id_hash").reset_index(drop=True)

    product_rows: list[dict[str, Any]] = []
    title_rows: list[dict[str, Any]] = []
    for row in products_raw.itertuples(index=False):
        tag_features = getattr(row, "tag_features", None) or []
        product_hash = str(row.product_id_hash)
        for feature in tag_features:
            product_rows.append(
                {
                    "product_id_hash": product_hash,
                    "tag_id": str(feature["tag_id"]),
                    "in_title": bool(feature.get("in_title", False)),
                    "title_part": feature.get("title_part"),
                    "title_position": feature.get("title_position"),
                }
            )
            if (
                feature.get("in_title")
                and pd.notna(feature.get("title_part"))
                and pd.notna(feature.get("title_position"))
            ):
                title_rows.append(
                    {
                        "product_id_hash": product_hash,
                        "tag_id": str(feature["tag_id"]),
                        "title_part": int(feature["title_part"]),
                        "title_position": int(feature["title_position"]),
                    }
                )

    product_tags = pd.DataFrame(
        product_rows,
        columns=["product_id_hash", "tag_id", "in_title", "title_part", "title_position"],
    )
    title_tags = pd.DataFrame(
        title_rows,
        columns=["product_id_hash", "tag_id", "title_part", "title_position"],
    )
    return products, product_tags, title_tags


def load_snapshot_tables(
    snapshot_dir: str | Path,
    *,
    snapshot_label: str | None = None,
    max_products: int | None = None,
) -> SnapshotTables:
    root = Path(snapshot_dir)
    snapshot_path = root / "snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError(str(snapshot_path))

    tag_path = root / "tags.jsonl.gz"
    if not tag_path.is_file():
        tag_path = root / "tags.jsonl"
    if not tag_path.is_file():
        raise FileNotFoundError("tags.jsonl(.gz) not found")

    product_paths = sorted(root.glob("products-*.jsonl*"))
    if not product_paths:
        raise FileNotFoundError("products-*.jsonl(.gz) not found")

    snapshot_metadata = _load_json_path(snapshot_path)
    tags = _load_jsonl_path(tag_path)
    product_frames: list[pd.DataFrame] = []
    total_rows = 0
    for path in product_paths:
        frame = _load_jsonl_path(path)
        if max_products is not None and total_rows + len(frame) > max_products:
            frame = frame.iloc[: max_products - total_rows].copy()
        product_frames.append(frame)
        total_rows += len(frame)
        if max_products is not None and total_rows >= max_products:
            break

    products_raw = (
        pd.concat(product_frames, ignore_index=True)
        if product_frames
        else pd.DataFrame(columns=["product_id_hash", "tag_features"])
    )
    products, product_tags, title_tags = _product_frames_from_raw(products_raw)
    return SnapshotTables(
        snapshot_metadata=snapshot_metadata,
        snapshot_label=snapshot_label or root.name,
        products_all=products_raw,
        products=products,
        tags=tags.sort_values("tag_rank").reset_index(drop=True),
        product_tags=product_tags,
        title_tags=title_tags,
    )


def load_snapshot_tables_from_hf(
    *,
    dataset_id: str,
    revision: str,
    snapshot: str,
    hf_token: str | None = None,
    max_products: int | None = None,
) -> SnapshotTables:
    headers = _hf_headers(hf_token)

    def hf_api_json(path: str) -> dict[str, Any]:
        return _load_json_url(f"https://huggingface.co/api/{path.lstrip('/')}", headers=headers)

    def hf_resolve_url(path: str) -> str:
        dataset_slug = dataset_id
        revision_slug = urllib.parse.quote(revision, safe="")
        path_slug = urllib.parse.quote(path.lstrip("/"), safe="/")
        return f"https://huggingface.co/datasets/{dataset_slug}/resolve/{revision_slug}/{path_slug}"

    snapshot_prefix = snapshot.strip("/") + "/"
    dataset_meta = hf_api_json(f"datasets/{dataset_id}")
    all_paths = [entry.get("rfilename", "") for entry in dataset_meta.get("siblings", [])]
    snapshot_paths = [path for path in all_paths if path.startswith(snapshot_prefix)]
    if not snapshot_paths:
        raise ValueError(
            f"Snapshot folder {snapshot!r} not found in dataset {dataset_id!r}"
        )

    snapshot_json_path = snapshot_prefix + "snapshot.json"
    if snapshot_json_path not in snapshot_paths:
        raise ValueError(f"Missing {snapshot_json_path!r}")

    if snapshot_prefix + "tags.jsonl.gz" in snapshot_paths:
        tags_path = snapshot_prefix + "tags.jsonl.gz"
    elif snapshot_prefix + "tags.jsonl" in snapshot_paths:
        tags_path = snapshot_prefix + "tags.jsonl"
    else:
        raise ValueError(f"Missing tags JSONL file in snapshot {snapshot!r}")

    product_paths = sorted(
        path
        for path in snapshot_paths
        if re.fullmatch(rf"{re.escape(snapshot_prefix)}products-\d+\.jsonl(?:\.gz)?", path)
    )
    if not product_paths:
        raise ValueError(f"Missing products JSONL files in snapshot {snapshot!r}")

    snapshot_metadata = _load_json_url(hf_resolve_url(snapshot_json_path), headers=headers)
    tags = _load_json_url_df(hf_resolve_url(tags_path), headers=headers)
    product_frames: list[pd.DataFrame] = []
    total_rows = 0
    for path in product_paths:
        frame = _load_json_url_df(hf_resolve_url(path), headers=headers)
        if max_products is not None and total_rows + len(frame) > max_products:
            frame = frame.iloc[: max_products - total_rows].copy()
        product_frames.append(frame)
        total_rows += len(frame)
        if max_products is not None and total_rows >= max_products:
            break

    products_raw = (
        pd.concat(product_frames, ignore_index=True)
        if product_frames
        else pd.DataFrame(columns=["product_id_hash", "tag_features"])
    )
    products, product_tags, title_tags = _product_frames_from_raw(products_raw)
    return SnapshotTables(
        snapshot_metadata=snapshot_metadata,
        snapshot_label=snapshot,
        products_all=products_raw,
        products=products,
        tags=tags.sort_values("tag_rank").reset_index(drop=True),
        product_tags=product_tags,
        title_tags=title_tags,
    )


def _build_sparse_matrix(
    products: pd.DataFrame,
    tags: pd.DataFrame,
    product_tags: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    tag_to_col = {tag_id: i for i, tag_id in enumerate(tags["tag_id"].tolist())}
    product_to_row = {pid: i for i, pid in enumerate(products["product_id_hash"].tolist())}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for pid, tag_id in product_tags[["product_id_hash", "tag_id"]].itertuples(index=False, name=None):
        row_idx = product_to_row.get(pid)
        col_idx = tag_to_col.get(tag_id)
        if row_idx is None or col_idx is None:
            continue
        rows.append(int(row_idx))
        cols.append(int(col_idx))
        data.append(1.0)

    matrix = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(products), len(tags)),
        dtype=np.float32,
    )
    labels = products["taxonomy_id"].to_numpy(dtype=object)
    return matrix, labels


def _mean_padic_loss_from_ids(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    taxonomy_encoded: Mapping[Any, int],
    *,
    base: int,
) -> float:
    losses = [
        p_adic_distance(int(taxonomy_encoded[true_label]), int(taxonomy_encoded[pred_label]), base)
        for true_label, pred_label in zip(y_true, y_pred)
    ]
    return float(np.mean(losses)) if losses else 0.0


def _safe_mean(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.mean(numeric))


def _safe_std(values: Sequence[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return float(np.std(np.asarray(numeric, dtype=float), ddof=0))


def _nonzero_linear_scoring_ops(model: LogisticRegression, sample: sparse.spmatrix) -> float:
    csr = sample.tocsr()
    active_indices = csr.indices
    if active_indices.size == 0:
        return float(np.count_nonzero(model.intercept_))
    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return float(np.count_nonzero(coef[:, active_indices]) + np.count_nonzero(model.intercept_))


def _dense_model_scoring_ops(model: MLPClassifier, sample: sparse.spmatrix) -> float:
    csr = sample.tocsr()
    active_indices = csr.indices
    total = 0
    if model.coefs_:
        first = np.asarray(model.coefs_[0])
        if active_indices.size:
            total += int(np.count_nonzero(first[active_indices, :]))
        total += int(np.count_nonzero(model.intercepts_[0]))
        for weight, bias in zip(model.coefs_[1:], model.intercepts_[1:]):
            total += int(np.count_nonzero(weight))
            total += int(np.count_nonzero(bias))
    return float(total)


def _tree_effective_params(model: DecisionTreeClassifier) -> float:
    n_classes = model.n_classes_
    if isinstance(n_classes, np.ndarray):
        n_classes = int(np.max(n_classes))
    else:
        n_classes = int(n_classes)
    return float(model.tree_.node_count * math.log2(max(n_classes, 2)))


def _tree_scoring_ops(model: DecisionTreeClassifier, sample: sparse.spmatrix) -> float:
    return float(max(model.decision_path(sample).nnz - 1, 0))


def _dense_model_params(model: MLPClassifier) -> float:
    return float(sum(arr.size for arr in model.coefs_) + sum(arr.size for arr in model.intercepts_))


def _l1_logistic_nonzero_params(model: LogisticRegression) -> float:
    return float(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))


def _logistic_nonzero_params(model: LogisticRegression) -> float:
    return float(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))


def _evaluate_classifier_cv(
    *,
    model_key: str,
    model_label: str,
    short_label: str,
    color: str,
    marker: str,
    folds: Sequence[int],
    features: sparse.csr_matrix,
    labels: np.ndarray,
    cv_fold_values: np.ndarray,
    taxonomy_paths: Mapping[Any, Sequence[str]],
    taxonomy_encoded: Mapping[Any, int],
    base: int,
    make_model,
    param_counter,
    scoring_ops,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []

    for fold in folds:
        train_mask = cv_fold_values != fold
        test_mask = cv_fold_values == fold
        model = make_model()
        model.fit(features[train_mask], labels[train_mask])
        y_true = labels[test_mask]
        y_pred = model.predict(features[test_mask])
        ops = [
            float(scoring_ops(model, features[test_mask][row_idx]))
            for row_idx in range(features[test_mask].shape[0])
        ]
        summary = summarize_taxonomy_predictions(
            y_true,
            y_pred,
            taxonomy_paths,
            scoring_ops=ops,
        )
        fold_rows.append(
            {
                "cv_fold": int(fold),
                "padic_loss_mean": _mean_padic_loss_from_ids(
                    y_true,
                    y_pred,
                    taxonomy_encoded,
                    base=base,
                ),
                "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
                "exact_accuracy": summary.exact_accuracy,
                "prefix1_accuracy": summary.prefix1_accuracy,
                "prefix2_accuracy": summary.prefix2_accuracy,
                "mean_shared_prefix_depth": summary.mean_shared_prefix_depth,
                "mean_scoring_ops": summary.mean_scoring_ops,
                "num_params": float(param_counter(model)),
                "num_train_samples": int(train_mask.sum()),
                "num_test_samples": int(test_mask.sum()),
            }
        )

    return _aggregate_model_rows(
        model_key=model_key,
        model_label=model_label,
        short_label=short_label,
        color=color,
        marker=marker,
        fold_rows=fold_rows,
    )


def _aggregate_model_rows(
    *,
    model_key: str,
    model_label: str,
    short_label: str,
    color: str,
    marker: str,
    fold_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "model_label": model_label,
        "short_label": short_label,
        "color": color,
        "marker": marker,
        "params": _safe_mean([row.get("num_params") for row in fold_rows]) or 0.0,
        "mean_padic_loss": _safe_mean([row.get("padic_loss_mean") for row in fold_rows]) or 0.0,
        "mean_accuracy": _safe_mean([row.get("accuracy") for row in fold_rows]) or 0.0,
        "mean_exact_accuracy": _safe_mean([row.get("exact_accuracy") for row in fold_rows]) or 0.0,
        "mean_prefix1_accuracy": _safe_mean([row.get("prefix1_accuracy") for row in fold_rows]) or 0.0,
        "mean_prefix2_accuracy": _safe_mean([row.get("prefix2_accuracy") for row in fold_rows]) or 0.0,
        "mean_shared_prefix_depth": _safe_mean([row.get("mean_shared_prefix_depth") for row in fold_rows]) or 0.0,
        "mean_scoring_ops": _safe_mean([row.get("mean_scoring_ops") for row in fold_rows]),
        "loss_std": _safe_std([row.get("padic_loss_mean") for row in fold_rows]) or 0.0,
        "folds": list(fold_rows),
    }


def select_top_tags(
    features: sparse.csr_matrix,
    feature_names: Sequence[str],
    max_tags: int,
) -> tuple[sparse.csr_matrix, list[str]]:
    if max_tags >= len(feature_names):
        return features, list(feature_names)
    tag_counts = np.array(features.sum(axis=0)).flatten()
    top_indices = np.argsort(tag_counts)[::-1][:max_tags]
    top_indices = np.sort(top_indices)
    return features[:, top_indices], [feature_names[idx] for idx in top_indices]


def _build_umllr_records(
    tables: SnapshotTables,
    taxonomy_encoded: Mapping[str, int],
) -> list[ProductRecord]:
    tag_rows_by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables.product_tags.to_dict("records"):
        tag_rows_by_product[str(row["product_id_hash"])].append(row)

    records: list[ProductRecord] = []
    for idx, row in enumerate(
        tables.products[
            ["product_id_hash", "taxonomy_id", "cv_fold", "taxonomy_path"]
        ].itertuples(index=False, name=None)
    ):
        product_hash, taxonomy_id, cv_fold, taxonomy_path = row
        tag_rows = tag_rows_by_product.get(str(product_hash), [])
        title_positions = tuple(
            sorted(
                (
                    (str(tag_row["tag_id"]), int(tag_row["title_part"]), int(tag_row["title_position"]))
                    for tag_row in tag_rows
                    if pd.notna(tag_row.get("title_part"))
                    and pd.notna(tag_row.get("title_position"))
                ),
                key=lambda item: (item[1], item[2], item[0]),
            )
        )
        records.append(
            ProductRecord(
                product_id=int(idx),
                product_key=str(product_hash),
                tags=[str(tag_row["tag_id"]) for tag_row in tag_rows],
                encoded_path=int(taxonomy_encoded[str(taxonomy_id)]),
                cv_fold=int(cv_fold),
                taxonomy_id=str(taxonomy_id),
                taxonomy_depth=len(parse_taxonomy_digits(str(taxonomy_path))),
                title_tag_positions=title_positions,
            )
        )
    return records


def _derive_battles(records: Sequence[ProductRecord]) -> list[BattleRecord]:
    battles: list[BattleRecord] = []
    for record in records:
        by_part: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for tag, part_idx, position in record.title_tag_positions:
            by_part[int(part_idx)].append((int(position), str(tag)))
        for ordered in by_part.values():
            ordered.sort()
            for left_idx in range(len(ordered)):
                for right_idx in range(left_idx + 1, len(ordered)):
                    loser = ordered[left_idx][1]
                    winner = ordered[right_idx][1]
                    if loser == winner:
                        continue
                    battles.append(
                        BattleRecord(
                            winner_tag=winner,
                            loser_tag=loser,
                            cv_fold=int(record.cv_fold),
                        )
                    )
    return battles


def _battle_elo_order(
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    training_tags: Sequence[str],
) -> list[str]:
    ordered_tags = sorted(set(training_tags))
    if not ordered_tags:
        return []

    ratings = {tag: 0.0 for tag in ordered_tags}
    filtered_battles = [
        battle
        for battle in battles
        if battle.cv_fold != holdout_fold
        and battle.winner_tag in ratings
        and battle.loser_tag in ratings
    ]
    for battle in filtered_battles:
        winner_rating = ratings[battle.winner_tag]
        loser_rating = ratings[battle.loser_tag]
        expected_win = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
        expected_loss = 1.0 - expected_win
        ratings[battle.winner_tag] = winner_rating + 32.0 * (1.0 - expected_win)
        ratings[battle.loser_tag] = loser_rating + 32.0 * (0.0 - expected_loss)
    return sorted(ordered_tags, key=lambda tag: (-ratings[tag], tag))


def _frequency_order(training: Sequence[ProductRecord]) -> list[str]:
    counts: Counter[str] = Counter()
    for record in training:
        counts.update(record.tags)
    return [tag for tag, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _mean_title_position_order(training: Sequence[ProductRecord]) -> list[str]:
    scores: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for record in training:
        counts.update(record.tags)
        for tag, part_idx, position in record.title_tag_positions:
            scores[str(tag)].append(float(int(part_idx) * 100000 + int(position)))

    def mean_position(tag: str) -> float:
        values = scores.get(tag)
        if not values:
            return float("-inf")
        return float(sum(values) / len(values))

    return sorted(set(counts), key=lambda tag: (-mean_position(tag), -counts[tag], tag))


def _taxonomy_association_order(training: Sequence[ProductRecord]) -> list[str]:
    tag_totals: Counter[str] = Counter()
    tag_taxonomy_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for record in training:
        for tag in record.tags:
            tag_totals[str(tag)] += 1
            tag_taxonomy_counts[str(tag)][record.taxonomy_id] += 1

    def association_strength(tag: str) -> float:
        total = tag_totals[tag]
        if total <= 0:
            return 0.0
        return max(tag_taxonomy_counts[tag].values(), default=0) / total

    return [
        tag
        for tag in sorted(
            tag_totals,
            key=lambda tag: (-association_strength(tag), -tag_totals[tag], tag),
        )
    ]


def _random_order(training_tags: Sequence[str], seed: int | None) -> list[str]:
    if seed is None:
        raise ValueError("random tag ordering requires an explicit seed")
    ordered_tags = sorted(set(training_tags))
    rng = random.Random(seed)
    rng.shuffle(ordered_tags)
    return ordered_tags


def _tag_order(
    training: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    *,
    strategy: str,
    seed: int | None = None,
) -> list[str]:
    training_tags = sorted({tag for record in training for tag in record.tags})
    if strategy == "battle_elo":
        return _battle_elo_order(battles, holdout_fold, training_tags)
    if strategy == "frequency":
        return _frequency_order(training)
    if strategy == "mean_title_position":
        return _mean_title_position_order(training)
    if strategy == "taxonomy_association":
        return _taxonomy_association_order(training)
    if strategy == "random":
        return _random_order(training_tags, seed)
    raise ValueError(f"Unknown tag order strategy: {strategy}")


def _select_coefficient(values: Sequence[int], base: int) -> int:
    unique_values = sorted(set(int(value) for value in values))
    best_value = unique_values[0]
    best_loss = float("inf")
    for candidate in unique_values:
        total_distance = sum(p_adic_distance(candidate, value, base) for value in values)
        if total_distance < best_loss or (
            math.isclose(total_distance, best_loss) and candidate < best_value
        ):
            best_loss = total_distance
            best_value = candidate
    return int(best_value)


def _select_default_prediction(
    no_tag_values: Sequence[int],
    candidate_values: Sequence[int],
    base: int,
) -> int:
    if not no_tag_values:
        if candidate_values:
            return int(Counter(candidate_values).most_common(1)[0][0])
        return 0
    unique_candidates = sorted(set(int(value) for value in candidate_values)) if candidate_values else [0]
    best_value = unique_candidates[0]
    best_loss = float("inf")
    for candidate in unique_candidates:
        total_loss = sum(p_adic_distance(candidate, value, base) for value in no_tag_values)
        if total_loss < best_loss or (total_loss == best_loss and candidate < best_value):
            best_loss = total_loss
            best_value = candidate
    return int(best_value)


def umllr_run_fold(
    fold: int,
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
    *,
    tag_order_strategy: str,
    tag_order_seed: int | None = None,
) -> UMLLRFoldResult:
    training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]

    product_residuals: Dict[int, int] = {record.product_id: record.encoded_path for record in training}
    tag_to_products: Dict[str, List[int]] = {}
    for record in training:
        for tag in record.tags:
            tag_to_products.setdefault(tag, []).append(record.product_id)

    tag_order = _tag_order(
        training,
        battles,
        fold,
        strategy=tag_order_strategy,
        seed=tag_order_seed,
    )

    coefficients: List[TagCoefficient] = []
    for sequence, tag in enumerate(tag_order):
        product_ids = tag_to_products.get(tag, [])
        values = [product_residuals[pid] for pid in product_ids]
        coefficient = _select_coefficient(values, base) if values else 0
        for pid in product_ids:
            product_residuals[pid] -= coefficient
        coefficients.append(TagCoefficient(tag=tag, coefficient=coefficient, sequence=sequence))

    coefficient_lookup = {entry.tag: entry.coefficient for entry in coefficients}
    no_tag_training_values = [
        record.encoded_path
        for record in training
        if sum(coefficient_lookup.get(tag, 0) for tag in record.tags) == 0
    ]
    all_training_values = [record.encoded_path for record in training]
    default_prediction = _select_default_prediction(no_tag_training_values, all_training_values, base)

    predictions: list[Prediction] = []
    scoring_ops: list[float] = []
    total_loss = 0.0
    for record in testing:
        active_coefficients = [
            coefficient_lookup.get(tag, 0)
            for tag in record.tags
            if coefficient_lookup.get(tag, 0) != 0
        ]
        predicted = int(sum(active_coefficients))
        if predicted == 0:
            predicted = default_prediction
        loss = p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss
        scoring_ops.append(float(len(active_coefficients)))
        predictions.append(
            Prediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )

    summary = summarize_encoded_predictions(
        [record.encoded_path for record in testing],
        [prediction.predicted_value for prediction in predictions],
        base=base,
        true_depths=[record.taxonomy_depth for record in testing],
        scoring_ops=scoring_ops,
    )
    return UMLLRFoldResult(
        cv_fold=int(fold),
        coefficients=coefficients,
        predictions=predictions,
        loss=float(total_loss),
        default_prediction=int(default_prediction),
        exact_accuracy=summary.exact_accuracy,
        prefix1_accuracy=summary.prefix1_accuracy,
        prefix2_accuracy=summary.prefix2_accuracy,
        mean_shared_prefix_depth=summary.mean_shared_prefix_depth,
        mean_scoring_ops=summary.mean_scoring_ops or 0.0,
    )


def _binomial(n: int, k: int) -> int:
    if k < 0:
        return 0
    if k == 0:
        return 1
    if k > abs(n) and n >= 0:
        return 0
    result = 1
    for idx in range(k):
        result = result * (n - idx) // (idx + 1)
    return int(result)


def _mahler_predict(s: int, weights: Sequence[int]) -> int:
    total = 0
    for degree, weight in enumerate(weights):
        total += int(weight) * _binomial(int(s), degree)
    return int(total)


def _compute_zubarev_loss(
    records: Sequence[ProductRecord],
    coefficients: Mapping[str, int],
    mahler_weights: Sequence[int],
    default_prediction: int,
    base: int,
) -> float:
    total_loss = 0.0
    for record in records:
        score = sum(coefficients.get(tag, 0) for tag in record.tags)
        predicted = _mahler_predict(score, mahler_weights) if mahler_weights else score
        if predicted == 0 and not any(coefficients.get(tag, 0) != 0 for tag in record.tags):
            predicted = default_prediction
        total_loss += p_adic_distance(predicted, record.encoded_path, base)
    return float(total_loss)


def _initialize_coefficients_umllr_style(
    training: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    holdout_fold: int,
    base: int,
) -> Dict[str, int]:
    tag_to_products: Dict[str, List[int]] = {}
    for record in training:
        for tag in record.tags:
            tag_to_products.setdefault(tag, []).append(record.product_id)
    tag_order = _battle_elo_order(battles, holdout_fold, list(tag_to_products.keys()))
    product_residuals: Dict[int, int] = {record.product_id: record.encoded_path for record in training}
    coefficients: Dict[str, int] = {}
    for tag in tag_order:
        product_ids = tag_to_products.get(tag, [])
        values = [product_residuals[pid] for pid in product_ids]
        coefficient = _select_coefficient(values, base) if values else 0
        coefficients[tag] = coefficient
        for pid in product_ids:
            product_residuals[pid] -= coefficient
    return coefficients


def _stochastic_optimize(
    training: Sequence[ProductRecord],
    initial_coefficients: Dict[str, int],
    base: int,
    *,
    mahler_degree: int = 0,
    max_iterations: int = DEFAULT_ZUBAREV_MAX_ITERATIONS,
    initial_temperature: float = 1.0,
    cooling_rate: float = 0.9995,
    min_temperature: float = 0.001,
    perturbation_scale: int = 1000,
    seed: int | None = None,
) -> tuple[Dict[str, int], list[int], float, int]:
    if seed is not None:
        random.seed(seed)

    coefficients = dict(initial_coefficients)
    tags = list(coefficients)
    mahler_weights = [0] + [1] + [0] * (mahler_degree - 1) if mahler_degree > 0 else []
    all_values = sorted({record.encoded_path for record in training})
    default_prediction = all_values[0] if all_values else 0

    current_loss = _compute_zubarev_loss(training, coefficients, mahler_weights, default_prediction, base)
    best_coefficients = dict(coefficients)
    best_mahler = list(mahler_weights)
    best_loss = current_loss
    temperature = initial_temperature
    iteration = 0

    while iteration < max_iterations and temperature > min_temperature and tags:
        tag = random.choice(tags)
        old_value = coefficients.get(tag, 0)
        if random.random() < 0.5:
            power = random.randint(0, 5)
            sign = random.choice([-1, 1])
            delta = sign * (base ** power)
        else:
            delta = random.randint(-perturbation_scale, perturbation_scale)

        coefficients[tag] = old_value + delta
        new_loss = _compute_zubarev_loss(training, coefficients, mahler_weights, default_prediction, base)
        accepted = False
        if new_loss < current_loss:
            current_loss = new_loss
            accepted = True
            if new_loss < best_loss:
                best_loss = new_loss
                best_coefficients = dict(coefficients)
                best_mahler = list(mahler_weights)
        elif temperature > 0:
            delta_loss = new_loss - current_loss
            try:
                acceptance_probability = math.exp(-delta_loss / temperature)
            except OverflowError:
                acceptance_probability = 0.0
            if random.random() < acceptance_probability:
                current_loss = new_loss
                accepted = True
        if not accepted:
            coefficients[tag] = old_value
        temperature *= cooling_rate
        iteration += 1

    return best_coefficients, best_mahler, float(best_loss), int(iteration)


def zubarev_run_fold(
    fold: int,
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
    *,
    mahler_degree: int = 0,
    max_iterations: int = DEFAULT_ZUBAREV_MAX_ITERATIONS,
    seed: int | None = None,
    initialization_method: str = "umllr",
) -> ZubarevFoldResult:
    all_training = [record for record in records if record.cv_fold != fold]
    testing = [record for record in records if record.cv_fold == fold]
    fold_seed = seed + fold if seed is not None else None

    if initialization_method == "umllr":
        initial_coefficients = _initialize_coefficients_umllr_style(all_training, battles, fold, base)
    elif initialization_method == "zeros":
        all_tags = {tag for record in all_training for tag in record.tags}
        initial_coefficients = {tag: 0 for tag in all_tags}
    else:
        raise ValueError(f"Unknown initialization_method: {initialization_method}")

    optimized_coefficients, mahler_weights, _, iterations_used = _stochastic_optimize(
        all_training,
        initial_coefficients,
        base,
        mahler_degree=mahler_degree,
        max_iterations=max_iterations,
        seed=fold_seed,
    )

    all_training_values = [record.encoded_path for record in all_training]
    no_tag_training_values = [
        record.encoded_path
        for record in all_training
        if sum(optimized_coefficients.get(tag, 0) for tag in record.tags) == 0
    ]
    default_prediction = _select_default_prediction(no_tag_training_values, all_training_values, base)

    predictions: list[Prediction] = []
    scoring_ops: list[float] = []
    total_loss = 0.0
    nonzero_mahler = int(sum(1 for weight in mahler_weights if weight != 0))
    for record in testing:
        active_coefficients = [
            optimized_coefficients.get(tag, 0)
            for tag in record.tags
            if optimized_coefficients.get(tag, 0) != 0
        ]
        score = int(sum(active_coefficients))
        predicted = _mahler_predict(score, mahler_weights) if mahler_weights else score
        if predicted == 0 and not active_coefficients:
            predicted = default_prediction
        loss = p_adic_distance(predicted, record.encoded_path, base)
        total_loss += loss
        scoring_ops.append(float(len(active_coefficients) + nonzero_mahler))
        predictions.append(
            Prediction(
                product_id=record.product_id,
                true_value=record.encoded_path,
                predicted_value=predicted,
                loss=loss,
            )
        )

    ordered_tags = _battle_elo_order(battles, fold, list(optimized_coefficients.keys()))
    coefficients = [
        TagCoefficient(tag=tag, coefficient=int(optimized_coefficients.get(tag, 0)), sequence=idx)
        for idx, tag in enumerate(ordered_tags)
    ]
    summary = summarize_encoded_predictions(
        [record.encoded_path for record in testing],
        [prediction.predicted_value for prediction in predictions],
        base=base,
        true_depths=[record.taxonomy_depth for record in testing],
        scoring_ops=scoring_ops,
    )
    return ZubarevFoldResult(
        cv_fold=int(fold),
        coefficients=coefficients,
        predictions=predictions,
        loss=float(total_loss),
        default_prediction=int(default_prediction),
        iterations_used=int(iterations_used),
        exact_accuracy=summary.exact_accuracy,
        prefix1_accuracy=summary.prefix1_accuracy,
        prefix2_accuracy=summary.prefix2_accuracy,
        mean_shared_prefix_depth=summary.mean_shared_prefix_depth,
        mean_scoring_ops=summary.mean_scoring_ops or 0.0,
        num_nonzero_params=int(sum(1 for coefficient in coefficients if coefficient.coefficient != 0) + nonzero_mahler),
    )


def train_levelwise_models(
    features: sparse.csr_matrix,
    labels: Sequence[str],
    taxonomy_paths: Mapping[str, Sequence[str]],
    *,
    max_iter: int = 1000,
) -> dict[tuple[str, ...], NodeModel]:
    prefix_indices: dict[tuple[str, ...], list[int]] = defaultdict(list)
    prefix_targets: dict[tuple[str, ...], list[str]] = defaultdict(list)
    prefix_leaf_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    prefix_child_counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

    for idx, label in enumerate(labels):
        label_str = str(label)
        path = tuple(taxonomy_paths[label_str])
        for depth in range(len(path) + 1):
            prefix = path[:depth]
            prefix_leaf_counts[prefix][label_str] += 1
            if depth < len(path):
                prefix_indices[prefix].append(idx)
                prefix_targets[prefix].append(path[depth])
                prefix_child_counts[prefix][path[depth]] += 1

    models: dict[tuple[str, ...], NodeModel] = {}
    for prefix, leaf_counts in prefix_leaf_counts.items():
        majority_leaf = leaf_counts.most_common(1)[0][0]
        child_counts = prefix_child_counts.get(prefix, Counter())
        children = tuple(sorted(child_counts))
        majority_child = child_counts.most_common(1)[0][0] if child_counts else None
        classifier: LogisticRegression | None = None
        if len(children) > 1:
            classifier = LogisticRegression(
                max_iter=max_iter,
                solver="lbfgs",
                class_weight="balanced",
            )
            classifier.fit(features[prefix_indices[prefix]], prefix_targets[prefix])
        models[prefix] = NodeModel(
            prefix=prefix,
            classifier=classifier,
            majority_child=majority_child,
            majority_leaf_taxonomy_id=majority_leaf,
            children=children,
        )
    return models


def _logistic_scoring_ops(model: LogisticRegression, sample: sparse.spmatrix) -> int:
    csr = sample.tocsr()
    active_indices = csr.indices
    if active_indices.size == 0:
        return 0
    coef = model.coef_
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    return int(np.count_nonzero(coef[:, active_indices]))


def predict_levelwise_taxonomy(
    sample: sparse.spmatrix,
    models: Mapping[tuple[str, ...], NodeModel],
) -> tuple[str, int]:
    if () not in models:
        raise ValueError("Level-wise model is missing the root node")

    prefix: tuple[str, ...] = ()
    scoring_ops = 0
    while True:
        node = models[prefix]
        if node.classifier is not None:
            child = str(node.classifier.predict(sample)[0])
            scoring_ops += _logistic_scoring_ops(node.classifier, sample)
        elif node.majority_child is not None:
            child = node.majority_child
        else:
            return node.majority_leaf_taxonomy_id, scoring_ops

        next_prefix = prefix + (child,)
        if next_prefix not in models:
            return node.majority_leaf_taxonomy_id, scoring_ops
        prefix = next_prefix


def _evaluate_levelwise_cv(
    *,
    folds: Sequence[int],
    features: sparse.csr_matrix,
    labels: np.ndarray,
    cv_fold_values: np.ndarray,
    taxonomy_paths: Mapping[str, Sequence[str]],
    taxonomy_encoded: Mapping[str, int],
    base: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = cv_fold_values != fold
        test_mask = cv_fold_values == fold
        X_train = features[train_mask]
        X_test = features[test_mask]
        y_train = labels[train_mask]
        y_test = labels[test_mask]
        models = train_levelwise_models(X_train, y_train, taxonomy_paths)
        predictions: list[str] = []
        scoring_ops: list[float] = []
        for row_idx in range(X_test.shape[0]):
            predicted, ops = predict_levelwise_taxonomy(X_test[row_idx], models)
            predictions.append(predicted)
            scoring_ops.append(float(ops))

        summary = summarize_taxonomy_predictions(
            y_test,
            predictions,
            taxonomy_paths,
            scoring_ops=scoring_ops,
        )
        num_nonzero_params = 0
        for model in models.values():
            if model.classifier is None:
                continue
            num_nonzero_params += int(np.count_nonzero(model.classifier.coef_))
            num_nonzero_params += int(np.count_nonzero(model.classifier.intercept_))

        fold_rows.append(
            {
                "cv_fold": int(fold),
                "padic_loss_mean": _mean_padic_loss_from_ids(
                    y_test,
                    predictions,
                    taxonomy_encoded,
                    base=base,
                ),
                "accuracy": float(np.mean(np.asarray(predictions, dtype=object) == y_test)) if len(y_test) else 0.0,
                "exact_accuracy": summary.exact_accuracy,
                "prefix1_accuracy": summary.prefix1_accuracy,
                "prefix2_accuracy": summary.prefix2_accuracy,
                "mean_shared_prefix_depth": summary.mean_shared_prefix_depth,
                "mean_scoring_ops": summary.mean_scoring_ops,
                "num_params": float(num_nonzero_params),
                "num_train_samples": int(train_mask.sum()),
                "num_test_samples": int(test_mask.sum()),
                "num_nodes": int(len(models)),
                "num_classifiers": int(sum(1 for model in models.values() if model.classifier is not None)),
            }
        )

    return _aggregate_model_rows(
        model_key="levelwise",
        model_label="Level-wise Logistic Regression",
        short_label="Level-wise",
        color="#f97316",
        marker="^",
        fold_rows=fold_rows,
    )


def _evaluate_umllr_strategy(
    *,
    strategy: str,
    seed: int | None,
    folds: Sequence[int],
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        result = umllr_run_fold(
            int(fold),
            records,
            battles,
            base,
            tag_order_strategy=strategy,
            tag_order_seed=seed,
        )
        num_test_samples = sum(1 for record in records if record.cv_fold == fold)
        num_params = float(sum(1 for coefficient in result.coefficients if coefficient.coefficient != 0))
        fold_rows.append(
            {
                "cv_fold": int(fold),
                "padic_loss_mean": float(result.loss / num_test_samples) if num_test_samples else 0.0,
                "accuracy": result.exact_accuracy,
                "exact_accuracy": result.exact_accuracy,
                "prefix1_accuracy": result.prefix1_accuracy,
                "prefix2_accuracy": result.prefix2_accuracy,
                "mean_shared_prefix_depth": result.mean_shared_prefix_depth,
                "mean_scoring_ops": result.mean_scoring_ops,
                "num_params": num_params,
                "num_train_samples": int(sum(1 for record in records if record.cv_fold != fold)),
                "num_test_samples": int(num_test_samples),
            }
        )
    run_key = strategy if seed is None else f"{strategy}_seed_{seed}"
    return {
        "run_key": run_key,
        "tag_order_strategy": strategy,
        "tag_order_seed": seed,
        **_aggregate_model_rows(
            model_key="umllr",
            model_label="Importance-Optimised p-adic Linear Regression",
            short_label="Importance-Optimised",
            color="#0b6ce3",
            marker="o",
            fold_rows=fold_rows,
        ),
    }


def _build_ablation_summary(
    *,
    folds: Sequence[int],
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
    base: int,
) -> dict[str, Any]:
    run_rows = [
        _evaluate_umllr_strategy(
            strategy=strategy,
            seed=None,
            folds=folds,
            records=records,
            battles=battles,
            base=base,
        )
        for strategy in DEFAULT_ABLATION_STRATEGIES
    ]
    for seed in DEFAULT_RANDOM_SEEDS:
        run_rows.append(
            _evaluate_umllr_strategy(
                strategy="random",
                seed=seed,
                folds=folds,
                records=records,
                battles=battles,
                base=base,
            )
        )

    baseline = next(row for row in run_rows if row["tag_order_strategy"] == "battle_elo" and row["tag_order_seed"] is None)
    baseline_by_fold = {row["cv_fold"]: float(row["padic_loss_mean"]) for row in baseline["folds"]}

    strategy_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run_row in run_rows:
        grouped[run_row["tag_order_strategy"]].append(run_row)

    for strategy, rows in grouped.items():
        if strategy == "random":
            fold_rows = [fold_row for row in rows for fold_row in row["folds"]]
            wins = sum(
                1
                for fold_row in fold_rows
                if float(fold_row["padic_loss_mean"]) < baseline_by_fold.get(int(fold_row["cv_fold"]), float("inf"))
            )
            strategy_rows.append(
                {
                    "tag_order_strategy": strategy,
                    "tag_order_seed": None,
                    "run_key": "random",
                    "mean_padic_loss": _safe_mean([row["mean_padic_loss"] for row in rows]) or 0.0,
                    "loss_delta_vs_baseline": (_safe_mean([row["mean_padic_loss"] for row in rows]) or 0.0) - float(baseline["mean_padic_loss"]),
                    "mean_accuracy": _safe_mean([row["mean_accuracy"] for row in rows]) or 0.0,
                    "mean_exact_accuracy": _safe_mean([row["mean_exact_accuracy"] for row in rows]) or 0.0,
                    "mean_prefix1_accuracy": _safe_mean([row["mean_prefix1_accuracy"] for row in rows]) or 0.0,
                    "mean_prefix2_accuracy": _safe_mean([row["mean_prefix2_accuracy"] for row in rows]) or 0.0,
                    "mean_shared_prefix_depth": _safe_mean([row["mean_shared_prefix_depth"] for row in rows]) or 0.0,
                    "mean_scoring_ops": _safe_mean([row["mean_scoring_ops"] for row in rows]),
                    "mean_params": _safe_mean([row["params"] for row in rows]) or 0.0,
                    "wins_vs_baseline": int(wins),
                    "comparisons_vs_baseline": int(len(fold_rows)),
                    "folds": fold_rows,
                }
            )
        else:
            row = rows[0]
            wins = sum(
                1
                for fold_row in row["folds"]
                if float(fold_row["padic_loss_mean"]) < baseline_by_fold.get(int(fold_row["cv_fold"]), float("inf"))
            )
            strategy_rows.append(
                {
                    "tag_order_strategy": strategy,
                    "tag_order_seed": None,
                    "run_key": row["run_key"],
                    "mean_padic_loss": float(row["mean_padic_loss"]),
                    "loss_delta_vs_baseline": float(row["mean_padic_loss"]) - float(baseline["mean_padic_loss"]),
                    "mean_accuracy": float(row["mean_accuracy"]),
                    "mean_exact_accuracy": float(row["mean_exact_accuracy"]),
                    "mean_prefix1_accuracy": float(row["mean_prefix1_accuracy"]),
                    "mean_prefix2_accuracy": float(row["mean_prefix2_accuracy"]),
                    "mean_shared_prefix_depth": float(row["mean_shared_prefix_depth"]),
                    "mean_scoring_ops": row["mean_scoring_ops"],
                    "mean_params": float(row["params"]),
                    "wins_vs_baseline": int(wins),
                    "comparisons_vs_baseline": int(len(row["folds"])),
                    "folds": list(row["folds"]),
                }
            )

    strategy_rows.sort(key=lambda row: row["mean_padic_loss"])
    best_row = strategy_rows[0] if strategy_rows else None
    return {
        "baseline_strategy": "battle_elo",
        "baseline_mean_padic_loss": float(baseline["mean_padic_loss"]),
        "runs": run_rows,
        "strategy_rows": strategy_rows,
        "random_summary": next((row for row in strategy_rows if row["tag_order_strategy"] == "random"), None),
        "best_strategy": best_row["tag_order_strategy"] if best_row else None,
        "best_mean_padic_loss": best_row["mean_padic_loss"] if best_row else None,
        "best_delta_vs_baseline": best_row["loss_delta_vs_baseline"] if best_row else None,
    }


def _build_model_rows(
    tables: SnapshotTables,
    *,
    base: int,
    taxonomy_paths: Mapping[str, Sequence[str]],
    taxonomy_encoded: Mapping[str, int],
    folds: Sequence[int],
    features: sparse.csr_matrix,
    labels: np.ndarray,
    records: Sequence[ProductRecord],
    battles: Sequence[BattleRecord],
) -> list[dict[str, Any]]:
    cv_fold_values = tables.products["cv_fold"].to_numpy(dtype=int)
    feature_names = tables.tags["tag_id"].tolist()

    model_rows = [
        _evaluate_classifier_cv(
            model_key="dummy",
            model_label="Dummy Baseline",
            short_label="Dummy",
            color="#94a3b8",
            marker="X",
            folds=folds,
            features=features,
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: DummyClassifier(strategy="most_frequent"),
            param_counter=lambda _model: 1.0,
            scoring_ops=lambda _model, _sample: 1.0,
        ),
        _evaluate_umllr_strategy(
            strategy="battle_elo",
            seed=None,
            folds=folds,
            records=records,
            battles=battles,
            base=base,
        ),
        _evaluate_classifier_cv(
            model_key="pclr",
            model_label="Parameter-constrained Logistic Regression",
            short_label="PCLR",
            color="#2563eb",
            marker="s",
            folds=folds,
            features=select_top_tags(features, feature_names, DEFAULT_PCLR_MAX_TAGS)[0],
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
            ),
            param_counter=_logistic_nonzero_params,
            scoring_ops=_nonzero_linear_scoring_ops,
        ),
        _evaluate_classifier_cv(
            model_key="ulr",
            model_label="Unconstrained Logistic Regression with L1",
            short_label="ULR",
            color="#8b5cf6",
            marker="D",
            folds=folds,
            features=features,
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0,
                max_iter=DEFAULT_ULR_MAX_ITER,
                n_jobs=-1,
                multi_class="multinomial",
                random_state=42,
            ),
            param_counter=_l1_logistic_nonzero_params,
            scoring_ops=_nonzero_linear_scoring_ops,
        ),
        _evaluate_classifier_cv(
            model_key="pcnn",
            model_label="Parameter-constrained Neural Network",
            short_label="PCNN",
            color="#16a34a",
            marker="P",
            folds=folds,
            features=select_top_tags(features, feature_names, DEFAULT_PCNN_MAX_TAGS)[0],
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: MLPClassifier(
                hidden_layer_sizes=(DEFAULT_PCNN_HIDDEN,),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                max_iter=120,
                random_state=42,
            ),
            param_counter=_dense_model_params,
            scoring_ops=_dense_model_scoring_ops,
        ),
        _evaluate_levelwise_cv(
            folds=folds,
            features=features,
            labels=np.asarray(labels, dtype=object),
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
        ),
        _evaluate_classifier_cv(
            model_key="dt",
            model_label="Decision Tree",
            short_label="Decision Tree",
            color="#14b8a6",
            marker="h",
            folds=folds,
            features=features,
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: DecisionTreeClassifier(class_weight="balanced", random_state=42),
            param_counter=_tree_effective_params,
            scoring_ops=_tree_scoring_ops,
        ),
        _evaluate_classifier_cv(
            model_key="unn",
            model_label="Unconstrained Neural Network with L1",
            short_label="UNN",
            color="#ec4899",
            marker="p",
            folds=folds,
            features=features,
            labels=labels,
            cv_fold_values=cv_fold_values,
            taxonomy_paths=taxonomy_paths,
            taxonomy_encoded=taxonomy_encoded,
            base=base,
            make_model=lambda: MLPClassifier(
                hidden_layer_sizes=(DEFAULT_UNN_HIDDEN,),
                activation="relu",
                alpha=1e-4,
                batch_size=256,
                max_iter=DEFAULT_UNN_MAX_ITER,
                random_state=42,
            ),
            param_counter=_dense_model_params,
            scoring_ops=_dense_model_scoring_ops,
        ),
    ]

    zubarev_fold_rows: list[dict[str, Any]] = []
    for fold in folds:
        result = zubarev_run_fold(
            int(fold),
            records,
            battles,
            base,
            mahler_degree=0,
            max_iterations=DEFAULT_ZUBAREV_MAX_ITERATIONS,
            seed=42,
            initialization_method="umllr",
        )
        num_test_samples = sum(1 for record in records if record.cv_fold == fold)
        zubarev_fold_rows.append(
            {
                "cv_fold": int(fold),
                "padic_loss_mean": float(result.loss / num_test_samples) if num_test_samples else 0.0,
                "accuracy": result.exact_accuracy,
                "exact_accuracy": result.exact_accuracy,
                "prefix1_accuracy": result.prefix1_accuracy,
                "prefix2_accuracy": result.prefix2_accuracy,
                "mean_shared_prefix_depth": result.mean_shared_prefix_depth,
                "mean_scoring_ops": result.mean_scoring_ops,
                "num_params": float(result.num_nonzero_params),
                "num_train_samples": int(sum(1 for record in records if record.cv_fold != fold)),
                "num_test_samples": int(num_test_samples),
                "iterations_used": int(result.iterations_used),
            }
        )
    model_rows.append(
        _aggregate_model_rows(
            model_key="zubarev",
            model_label="Zubarev (UMLLR)",
            short_label="Zubarev",
            color="#7c3aed",
            marker="v",
            fold_rows=zubarev_fold_rows,
        )
    )
    return model_rows


def _add_parsimony_columns(model_rows: list[dict[str, Any]], *, num_taxonomies: int) -> list[dict[str, Any]]:
    slope = -0.1
    intercept = 0.0
    taxonomy_coefficient = 0.3
    taxonomy_reference = 1000.0
    for row in model_rows:
        params = max(float(row["params"]), 1.0)
        loss = max(float(row["mean_padic_loss"]), 1e-12)
        log10_params = math.log10(params)
        log10_loss = math.log10(loss)
        baseline_log10_loss = (
            slope * log10_params
            + intercept
            + taxonomy_coefficient * math.log10(max(float(num_taxonomies), 1.0) / taxonomy_reference)
        )
        row["log10_params"] = log10_params
        row["log10_loss"] = log10_loss
        row["baseline_log10_loss"] = baseline_log10_loss
        row["parsimony_score"] = baseline_log10_loss - log10_loss
    return model_rows


def build_snapshot_benchmark_bundle(
    tables: SnapshotTables,
) -> dict[str, Any]:
    products = tables.products.copy()
    products["cv_fold"] = products["cv_fold"].astype(int)
    folds = sorted(products["cv_fold"].unique().tolist())
    features, labels = _build_sparse_matrix(products, tables.tags, tables.product_tags)

    taxonomy_paths = {
        str(taxonomy_id): parse_taxonomy_path(taxonomy_path)
        for taxonomy_id, taxonomy_path in (
            products[["taxonomy_id", "taxonomy_path"]]
            .drop_duplicates(subset=["taxonomy_id"])
            .itertuples(index=False, name=None)
        )
    }
    taxonomy_digits: dict[str, tuple[int, ...]] = {
        str(taxonomy_id): parse_taxonomy_digits(str(taxonomy_path))
        for taxonomy_id, taxonomy_path in (
            products[["taxonomy_id", "taxonomy_path"]]
            .drop_duplicates(subset=["taxonomy_id"])
            .itertuples(index=False, name=None)
        )
    }
    max_digit = max((max(digits) for digits in taxonomy_digits.values() if digits), default=1)
    base = next_prime(max_digit)
    taxonomy_encoded = {
        taxonomy_id: encode_path(digits, base)
        for taxonomy_id, digits in taxonomy_digits.items()
    }

    records = _build_umllr_records(tables, taxonomy_encoded)
    battles = _derive_battles(records)
    model_rows = _build_model_rows(
        tables,
        base=base,
        taxonomy_paths=taxonomy_paths,
        taxonomy_encoded=taxonomy_encoded,
        folds=folds,
        features=features,
        labels=labels,
        records=records,
        battles=battles,
    )
    model_rows = _add_parsimony_columns(
        model_rows,
        num_taxonomies=int(products["taxonomy_id"].nunique()),
    )
    ablation = _build_ablation_summary(
        folds=folds,
        records=records,
        battles=battles,
        base=base,
    )

    return {
        "bundle_version": 1,
        "source": "snapshot_runtime",
        "snapshot": {
            "label": tables.snapshot_label,
            "snapshot_name": tables.snapshot_metadata.get("snapshot_name") or tables.snapshot_label,
            "as_of": tables.snapshot_metadata.get("as_of"),
            "created_at": tables.snapshot_metadata.get("created_at"),
            "product_count_all": int(len(tables.products_all)),
            "product_count_filtered": int(len(products)),
            "tag_count_all": int(tables.snapshot_metadata.get("tag_count") or len(tables.tags)),
            "tag_count_filtered": int(len(tables.tags)),
            "taxonomy_count_all": int(tables.snapshot_metadata.get("taxonomy_count") or products["taxonomy_id"].nunique()),
            "taxonomy_count_filtered": int(products["taxonomy_id"].nunique()),
            "prime_base": int(base),
            "max_digit": int(max_digit),
            "battle_count": int(len(battles)),
            "folds": [int(fold) for fold in folds],
        },
        "models": {
            "rows": model_rows,
        },
        "ablation": ablation,
        "narrative": {
            "best_ablation_strategy": ablation.get("best_strategy"),
            "best_ablation_mean_padic_loss": ablation.get("best_mean_padic_loss"),
            "best_ablation_delta_vs_baseline": ablation.get("best_delta_vs_baseline"),
            "battle_elo_mean_padic_loss": ablation.get("baseline_mean_padic_loss"),
            "umllr_mean_padic_loss": next(
                row["mean_padic_loss"] for row in model_rows if row["model_key"] == "umllr"
            ),
            "umllr_mean_params": next(
                row["params"] for row in model_rows if row["model_key"] == "umllr"
            ),
            "umllr_mean_scoring_ops": next(
                row["mean_scoring_ops"] for row in model_rows if row["model_key"] == "umllr"
            ),
            "levelwise_mean_scoring_ops": next(
                row["mean_scoring_ops"] for row in model_rows if row["model_key"] == "levelwise"
            ),
        },
    }
