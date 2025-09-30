"""Train a logistic regression model to predict synsets from product tags.

This module provides a small command line utility that looks at the
``product_synsets`` SQLite database (produced by :mod:`padjective.product_synsets`),
trains a multinomial logistic regression model using the tags as binary
features, and produces a static HTML report describing the most influential
tags.  The intention is to offer a lightweight baseline model that can be run
on the full dataset without involving large language models once the initial
synset annotations have been gathered.
"""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


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

    pipeline = Pipeline(
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

    metadata = (
        "<ul class=\"stats\">"
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>"
        f"<li><strong>Synsets:</strong> {stats.synsets:,}</li>"
        f"<li><strong>Unique tags:</strong> {stats.unique_tags:,}</li>"
        f"<li><strong>Training accuracy:</strong> {stats.training_accuracy:.3f}</li>"
        "</ul>"
    )

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
        help="Directory where the model, CSV, JSON, and HTML report will be written.",
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
    model, stats = train_classifier(data)
    summary = summarise_coefficients(model)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "synset_classifier.joblib"
    csv_path = output_dir / "tag_coefficients.csv"
    json_path = output_dir / "tag_coefficients.json"
    html_path = output_dir / "tag_coefficients.html"

    joblib.dump(model, model_path)
    summary.to_csv(csv_path, index=False)
    save_summary_json(summary, stats, json_path)
    render_coefficients_html(summary, stats, html_path, top_n=args.top_n)

    print(f"Trained on {stats.samples} samples covering {stats.synsets} synsets.")
    print(f"Model saved to {model_path}")
    print(f"Coefficient tables saved to {csv_path} and {html_path}")


if __name__ == "__main__":
    main()
