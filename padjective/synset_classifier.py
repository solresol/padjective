"""Train a logistic regression model to predict synsets from product tags.

This module provides a command line utility that looks at the
``product_synsets`` SQLite database (produced by :mod:`padjective.product_synsets`),
trains a multinomial logistic regression model using the tags as binary
features, evaluates it with stratified cross-validation, and stores the learned
weights in an output SQLite database.  A static HTML report summarises the most
influential tags to aid manual inspection.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _split_tags(tags: str) -> tuple[str, ...]:
    """Normalise a comma separated tag string into individual tags.

    Tags are upper-cased to match the rest of the repository's conventions and
    stripped of leading/trailing whitespace.  Empty fragments are ignored.
    """

    if not tags:
        return ()

    parts = []
    for fragment in tags.split(","):
        normalised = fragment.strip()
        if normalised:
            parts.append(normalised.upper())
    return tuple(parts)


@dataclass(slots=True)
class TrainingStats:
    """Basic metadata recorded after training the classifier."""

    samples: int
    synsets: int
    unique_tags: int
    training_accuracy: float
    cross_validation_folds: int | None = None
    cross_validation_mean_accuracy: float | None = None
    cross_validation_std_accuracy: float | None = None


def _create_pipeline() -> Pipeline:
    """Create the scikit-learn pipeline used for training and evaluation."""

    return Pipeline(
        steps=[
            (
                "tags_to_dicts",
                FunctionTransformer(_tag_lists_to_dicts, validate=False),
            ),
            ("vectorizer", DictVectorizer(sparse=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="auto",
                    n_jobs=None,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def load_training_data(
    database_path: Path,
    *,
    min_samples_per_synset: int = 5,
) -> pd.DataFrame:
    """Load training examples from the ``product_synsets`` database.

    Parameters
    ----------
    database_path:
        Path to the SQLite database created by :mod:`padjective.product_synsets`.
    min_samples_per_synset:
        Minimum number of training examples required for a synset to be kept in
        the dataset.  Rare classes are removed because logistic regression tends
        to perform poorly on extremely small classes, and the coefficient
        magnitudes become noisy.

    Returns
    -------
    pandas.DataFrame
        Data frame containing ``product_id``, ``synset_id`` and ``tag_list``
        columns.
    """

    with sqlite3.connect(database_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT product_id, tags, synset_id
            FROM product_synsets
            WHERE not_found = 0 AND synset_id IS NOT NULL
            """,
            conn,
        )

    if df.empty:
        raise ValueError(
            "No labelled synset data found in the database. Run padjective.product_synsets first."
        )

    df["tag_list"] = df["tags"].fillna("").map(_split_tags)
    df = df[df["tag_list"].map(len) > 0]

    if df.empty:
        raise ValueError("No products with usable tags were found in the database.")

    counts = df.groupby("synset_id").size()
    allowed_synsets = counts[counts >= max(1, min_samples_per_synset)].index
    filtered = df[df["synset_id"].isin(allowed_synsets)].copy()

    if filtered.empty:
        raise ValueError(
            "No synsets met the minimum sample threshold. Lower --min-samples-per-synset."
        )

    filtered.reset_index(drop=True, inplace=True)
    return filtered


def _tag_lists_to_dicts(data: Sequence[Sequence[str]]) -> np.ndarray:
    """Convert an iterable of tag sequences to dictionaries for ``DictVectorizer``."""

    return np.array([{tag: 1.0 for tag in tags} for tags in data], dtype=object)


def train_classifier(data: pd.DataFrame) -> tuple[Pipeline, TrainingStats]:
    """Train a multinomial logistic regression model using the provided data."""

    if data.empty:
        raise ValueError("Training data is empty; nothing to fit.")

    tag_sequences = data["tag_list"].to_numpy(dtype=object)
    labels = data["synset_id"].to_numpy()

    if len(set(labels)) < 2:
        raise ValueError("Need at least two synsets to train a classifier.")

    pipeline = _create_pipeline()

    pipeline.fit(tag_sequences, labels)

    accuracy = float(pipeline.score(tag_sequences, labels))
    vectorizer: DictVectorizer = pipeline.named_steps["vectorizer"]

    stats = TrainingStats(
        samples=len(data),
        synsets=len(pipeline.named_steps["classifier"].classes_),
        unique_tags=len(vectorizer.get_feature_names_out()),
        training_accuracy=accuracy,
    )
    return pipeline, stats


def cross_validate_classifier(
    data: pd.DataFrame,
    *,
    folds: int = 5,
    random_state: int = 0,
) -> list[float]:
    """Evaluate the classifier using stratified k-fold cross-validation."""

    if data.empty:
        return []

    labels = data["synset_id"].to_numpy()
    counts = pd.Series(labels).value_counts()
    max_splits = int(counts.min()) if not counts.empty else 0
    n_splits = min(folds, max_splits)

    if n_splits < 2:
        return []

    tag_sequences = data["tag_list"].to_numpy(dtype=object)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        _create_pipeline(),
        tag_sequences,
        labels,
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
    )
    return [float(score) for score in scores]


def compute_tag_coefficients(
    feature_names: Sequence[str],
    classes: Sequence[str],
    coef_matrix: np.ndarray,
) -> pd.DataFrame:
    """Create a summary table describing coefficient magnitudes per tag."""

    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)

    abs_coef = np.abs(coef_matrix)
    max_indices = abs_coef.argmax(axis=0)
    max_values = abs_coef[max_indices, range(abs_coef.shape[1])]
    sum_values = abs_coef.sum(axis=0)

    rows = []
    for idx, tag in enumerate(feature_names):
        class_index = int(max_indices[idx])
        weight = coef_matrix[class_index, idx]
        rows.append(
            {
                "tag": tag,
                "top_synset": classes[class_index],
                "top_weight": float(weight),
                "max_abs_coef": float(max_values[idx]),
                "sum_abs_coef": float(sum_values[idx]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.sort_values(["max_abs_coef", "sum_abs_coef"], ascending=[False, False], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def summarise_coefficients(model: Pipeline) -> pd.DataFrame:
    """Extract the coefficient summary table from a fitted pipeline."""

    vectorizer: DictVectorizer = model.named_steps["vectorizer"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    return compute_tag_coefficients(feature_names, classifier.classes_, classifier.coef_)


def _expand_binary_coefficients(
    classes: Sequence[str],
    coef_matrix: np.ndarray,
    intercepts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure binary classifiers expose one row per class."""

    if coef_matrix.ndim == 1:
        coef_matrix = coef_matrix.reshape(1, -1)

    intercepts = np.asarray(intercepts, dtype=float)

    if coef_matrix.shape[0] == len(classes):
        return coef_matrix, intercepts

    if len(classes) == 2 and coef_matrix.shape[0] == 1:
        coef_row = coef_matrix[0]
        intercept_value = float(intercepts[0]) if intercepts.size else 0.0
        expanded_coef = np.vstack([-coef_row, coef_row])
        expanded_intercepts = np.array([-intercept_value, intercept_value], dtype=float)
        return expanded_coef, expanded_intercepts

    raise ValueError(
        "Coefficient matrix shape does not match the number of synset classes."
    )


def save_model_to_database(
    database_path: Path,
    model: Pipeline,
    stats: TrainingStats,
    summary: pd.DataFrame,
    cv_scores: Iterable[float] | None = None,
) -> int:
    """Persist classifier weights, metadata, and summaries into SQLite."""

    database_path.parent.mkdir(parents=True, exist_ok=True)

    vectorizer: DictVectorizer = model.named_steps["vectorizer"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    classes = classifier.classes_
    coef_matrix = classifier.coef_
    intercepts = classifier.intercept_

    coef_matrix, intercepts = _expand_binary_coefficients(classes, coef_matrix, intercepts)
    coef_matrix = np.asarray(coef_matrix, dtype=float)
    intercepts = np.asarray(intercepts, dtype=float)

    cv_scores_list = [float(score) for score in (cv_scores or [])]
    cv_folds = len(cv_scores_list) if cv_scores_list else None
    cv_mean = float(np.mean(cv_scores_list)) if cv_scores_list else None
    cv_std = float(np.std(cv_scores_list, ddof=0)) if cv_scores_list else None

    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(database_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_classifier_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at TEXT NOT NULL,
                samples INTEGER NOT NULL,
                synsets INTEGER NOT NULL,
                unique_tags INTEGER NOT NULL,
                training_accuracy REAL NOT NULL,
                cv_folds INTEGER,
                cv_mean_accuracy REAL,
                cv_std_accuracy REAL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_classifier_cv_scores (
                model_id INTEGER NOT NULL,
                fold INTEGER NOT NULL,
                accuracy REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES synset_classifier_models(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_classifier_coefficients (
                model_id INTEGER NOT NULL,
                synset_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                weight REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES synset_classifier_models(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_classifier_intercepts (
                model_id INTEGER NOT NULL,
                synset_id TEXT NOT NULL,
                intercept REAL NOT NULL,
                FOREIGN KEY(model_id) REFERENCES synset_classifier_models(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS synset_classifier_tag_summary (
                model_id INTEGER NOT NULL,
                tag TEXT NOT NULL,
                top_synset TEXT NOT NULL,
                top_weight REAL NOT NULL,
                max_abs_coef REAL NOT NULL,
                sum_abs_coef REAL NOT NULL,
                PRIMARY KEY (model_id, tag),
                FOREIGN KEY(model_id) REFERENCES synset_classifier_models(id) ON DELETE CASCADE
            )
            """
        )

        cursor = conn.execute(
            """
            INSERT INTO synset_classifier_models (
                trained_at,
                samples,
                synsets,
                unique_tags,
                training_accuracy,
                cv_folds,
                cv_mean_accuracy,
                cv_std_accuracy
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                stats.samples,
                stats.synsets,
                stats.unique_tags,
                stats.training_accuracy,
                cv_folds,
                cv_mean,
                cv_std,
            ),
        )
        model_id = int(cursor.lastrowid)

        if cv_scores_list:
            conn.executemany(
                """
                INSERT INTO synset_classifier_cv_scores (model_id, fold, accuracy)
                VALUES (?, ?, ?)
                """,
                [
                    (model_id, index + 1, float(score))
                    for index, score in enumerate(cv_scores_list)
                ],
            )

        coefficient_rows = []
        for class_index, class_label in enumerate(classes):
            for feature_index, tag in enumerate(feature_names):
                coefficient_rows.append(
                    (
                        model_id,
                        str(class_label),
                        str(tag),
                        float(coef_matrix[class_index, feature_index]),
                    )
                )

        if coefficient_rows:
            conn.executemany(
                """
                INSERT INTO synset_classifier_coefficients (model_id, synset_id, tag, weight)
                VALUES (?, ?, ?, ?)
                """,
                coefficient_rows,
            )

        intercept_rows = [
            (model_id, str(class_label), float(intercept_value))
            for class_label, intercept_value in zip(classes, intercepts)
        ]
        if intercept_rows:
            conn.executemany(
                """
                INSERT INTO synset_classifier_intercepts (model_id, synset_id, intercept)
                VALUES (?, ?, ?)
                """,
                intercept_rows,
            )

        if not summary.empty:
            conn.executemany(
                """
                INSERT INTO synset_classifier_tag_summary (
                    model_id,
                    tag,
                    top_synset,
                    top_weight,
                    max_abs_coef,
                    sum_abs_coef
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        model_id,
                        str(row["tag"]),
                        str(row["top_synset"]),
                        float(row["top_weight"]),
                        float(row["max_abs_coef"]),
                        float(row["sum_abs_coef"]),
                    )
                    for row in summary.to_dict(orient="records")
                ],
            )

        conn.commit()

    return model_id


def _render_table(title: str, dataframe: pd.DataFrame, metric_column: str, top_n: int) -> str:
    records = dataframe.head(top_n).to_dict(orient="records")
    if not records:
        return f"<section><h2>{html.escape(title)}</h2><p>No data available.</p></section>"

    header = """
      <tr>
        <th>Rank</th>
        <th>Tag</th>
        <th>Synset with largest weight</th>
        <th>Weight (signed)</th>
        <th>Max |coef|</th>
        <th>Sum |coef|</th>
      </tr>
    """

    rows = []
    for index, record in enumerate(records, start=1):
        rows.append(
            "  <tr>"
            f"<td>{index}</td>"
            f"<td>{html.escape(record['tag'])}</td>"
            f"<td>{html.escape(str(record['top_synset']))}</td>"
            f"<td>{record['top_weight']:.4f}</td>"
            f"<td>{record['max_abs_coef']:.4f}</td>"
            f"<td>{record['sum_abs_coef']:.4f}</td>"
            "</tr>"
        )

    table = (
        f"<section><h2>{html.escape(title)}</h2>"
        "<table class=\"coeff-table\">"
        "<thead>"
        f"{header}"
        "</thead>"
        "<tbody>"
        f"{''.join(rows)}"
        "</tbody>"
        "</table></section>"
    )
    return table


def render_coefficients_html(
    summary: pd.DataFrame,
    stats: TrainingStats,
    output_path: Path,
    *,
    top_n: int = 50,
) -> None:
    """Render a standalone HTML report summarising coefficient magnitudes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    intro = (
        "<p>This report summarises a multinomial logistic regression model trained "
        "to predict WordNet synsets from Shopify product tags. The table below "
        "lists the tags with the strongest coefficients across all synset classes.</p>"
    )

    metadata_items = [
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>",
        f"<li><strong>Synsets:</strong> {stats.synsets:,}</li>",
        f"<li><strong>Unique tags:</strong> {stats.unique_tags:,}</li>",
        f"<li><strong>Training accuracy:</strong> {stats.training_accuracy:.3f}</li>",
    ]
    if (
        stats.cross_validation_folds
        and stats.cross_validation_mean_accuracy is not None
    ):
        cv_accuracy = stats.cross_validation_mean_accuracy
        if stats.cross_validation_std_accuracy is not None:
            metadata_items.append(
                "<li><strong>Cross-validated accuracy:</strong> "
                f"{cv_accuracy:.3f} ± {stats.cross_validation_std_accuracy:.3f}"
                f" ({stats.cross_validation_folds} folds)</li>"
            )
        else:
            metadata_items.append(
                "<li><strong>Cross-validated accuracy:</strong> "
                f"{cv_accuracy:.3f} ({stats.cross_validation_folds} folds)</li>"
            )

    metadata = "<ul class=\"stats\">" + "".join(metadata_items) + "</ul>"

    max_table = _render_table(
        "Tags ranked by maximum absolute coefficient",
        summary.sort_values(["max_abs_coef", "sum_abs_coef"], ascending=[False, False]),
        "max_abs_coef",
        top_n,
    )
    sum_table = _render_table(
        "Tags ranked by sum of absolute coefficients",
        summary.sort_values(["sum_abs_coef", "max_abs_coef"], ascending=[False, False]),
        "sum_abs_coef",
        top_n,
    )

    html_page = f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Synset tag coefficients</title>
    <style>
      body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
      h1 {{ margin-top: 0; }}
      .stats {{ list-style: none; padding: 0; display: flex; gap: 1.5rem; }}
      .stats li {{ background: white; padding: 0.75rem 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); }}
      section {{ margin-top: 2rem; }}
      table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); border-radius: 0.75rem; overflow: hidden; }}
      th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
      th {{ background: #0b6ce3; color: white; font-weight: 600; }}
      tr:nth-child(even) td {{ background: #f1f5f9; }}
    </style>
  </head>
  <body>
    <h1>Synset tag coefficients</h1>
    {intro}
    {metadata}
    {max_table}
    {sum_table}
  </body>
</html>"""

    output_path.write_text(html_page, encoding="utf-8")


def save_summary_json(summary: pd.DataFrame, stats: TrainingStats, output_path: Path) -> None:
    """Persist the coefficient data and metadata to JSON for downstream use."""

    payload = {
        "stats": {
            "samples": stats.samples,
            "synsets": stats.synsets,
            "unique_tags": stats.unique_tags,
            "training_accuracy": stats.training_accuracy,
        },
        "coefficients": summary.to_dict(orient="records"),
    }
    if stats.cross_validation_folds and stats.cross_validation_mean_accuracy is not None:
        payload["stats"]["cross_validation"] = {
            "folds": stats.cross_validation_folds,
            "mean_accuracy": stats.cross_validation_mean_accuracy,
            "std_accuracy": stats.cross_validation_std_accuracy,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a logistic regression model that predicts WordNet synsets from tags.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/product_synsets.sqlite"),
        help="SQLite database produced by padjective.product_synsets",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("synset_classifier"),
        help="Directory where the HTML report will be written.",
    )
    parser.add_argument(
        "--model-database",
        type=Path,
        default=Path("data/synset_classifier.sqlite"),
        help="SQLite database where classifier weights and metadata will be stored.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds to evaluate (limited by smallest class).",
    )
    parser.add_argument(
        "--min-samples-per-synset",
        type=int,
        default=5,
        help="Minimum number of labelled examples required to keep a synset in training.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of rows to include in the HTML tables.",
    )
    args = parser.parse_args()

    data = load_training_data(
        args.database, min_samples_per_synset=args.min_samples_per_synset
    )
    cv_scores = cross_validate_classifier(data, folds=max(2, args.cv_folds))
    model, stats = train_classifier(data)
    if cv_scores:
        stats.cross_validation_folds = len(cv_scores)
        stats.cross_validation_mean_accuracy = float(np.mean(cv_scores))
        stats.cross_validation_std_accuracy = float(np.std(cv_scores, ddof=0))
    summary = summarise_coefficients(model)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "tag_coefficients.html"

    save_model_to_database(args.model_database, model, stats, summary, cv_scores)
    render_coefficients_html(summary, stats, html_path, top_n=args.top_n)

    print(f"Trained on {stats.samples} samples covering {stats.synsets} synsets.")
    if stats.cross_validation_mean_accuracy is not None:
        print(
            "Cross-validated accuracy: "
            f"{stats.cross_validation_mean_accuracy:.3f}"
            + (
                f" ± {stats.cross_validation_std_accuracy:.3f}"
                if stats.cross_validation_std_accuracy is not None
                else ""
            )
            + f" ({stats.cross_validation_folds} folds)"
        )
    print(f"Classifier weights written to {args.model_database}")
    print(f"HTML report saved to {html_path}")


if __name__ == "__main__":
    main()
