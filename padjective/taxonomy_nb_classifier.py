"""Train a Complement Naive Bayes model to predict Shopify taxonomy from tags.

This module trains a probabilistic classifier that maps Shopify product tags to
Shopify taxonomy identifiers.  The classifier operates directly on data stored
in the Shopify Postgres database and persists model artefacts back to the
``padjective`` schema.  A lightweight HTML report summarises the most
informative tags for manual review.
"""

from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from psycopg import Connection
from scipy import sparse
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB

from . import db, tag_features


@dataclass(slots=True)
class TrainingStats:
    """Metadata captured during training."""

    samples: int
    taxonomies: int
    unique_tags: int
    training_accuracy: float
    cross_validation_folds: int | None = None
    cross_validation_mean_accuracy: float | None = None
    cross_validation_std_accuracy: float | None = None


def load_training_data(
    conn: Connection,
    *,
    product_table: str = "cantbuymelove.product",
    min_tag_count: int = 2,
    min_samples_per_taxonomy: int = 5,
) -> tuple[sparse.csr_matrix, np.ndarray, list[str], pd.DataFrame]:
    """Extract tag features and taxonomy labels from Postgres."""

    features, metadata, feature_names = tag_features.extract_tag_features(
        conn,
        product_table=product_table,
        include_taxonomy=True,
        min_tag_count=min_tag_count,
    )

    if "taxonomy_id" not in metadata.columns:
        raise ValueError("Tag feature extraction must include taxonomy identifiers")

    valid_taxonomy_mask = metadata["taxonomy_id"].notna()
    features = features[valid_taxonomy_mask.to_numpy(dtype=bool, copy=False)]
    metadata = metadata.loc[valid_taxonomy_mask].reset_index(drop=True)

    taxonomy_counts = metadata["taxonomy_id"].value_counts()
    allowed_taxonomies = taxonomy_counts[
        taxonomy_counts >= max(1, min_samples_per_taxonomy)
    ].index
    filtered_mask = metadata["taxonomy_id"].isin(allowed_taxonomies)
    features = features[filtered_mask.to_numpy(dtype=bool, copy=False)]
    metadata = metadata.loc[filtered_mask].reset_index(drop=True)

    if len(metadata) == 0:
        raise ValueError(
            "No taxonomy classes satisfy the minimum sample threshold."
        )

    labels = metadata["taxonomy_id"].to_numpy()
    return features, labels, feature_names, metadata


def train_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    alpha: float = 0.1,
) -> tuple[ComplementNB, TrainingStats]:
    """Fit the Complement Naive Bayes classifier."""

    if features.shape[0] == 0:
        raise ValueError("Cannot train classifier on an empty dataset")

    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        raise ValueError("Need at least two taxonomy classes to train")

    model = ComplementNB(alpha=alpha, norm=False)
    model.fit(features, labels)

    predictions = model.predict(features)
    accuracy = float(accuracy_score(labels, predictions))

    stats = TrainingStats(
        samples=len(labels),
        taxonomies=unique_labels.size,
        unique_tags=features.shape[1],
        training_accuracy=accuracy,
    )
    return model, stats


def cross_validate_classifier(
    features: sparse.csr_matrix,
    labels: np.ndarray,
    *,
    folds: int = 5,
    random_state: int = 0,
    alpha: float = 0.1,
) -> list[float]:
    """Evaluate the classifier using stratified k-fold cross-validation."""

    if len(labels) == 0:
        return []

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return []

    max_folds = int(counts.min())
    n_splits = min(max_folds, folds)

    if n_splits < 2:
        return []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = ComplementNB(alpha=alpha, norm=False)
    scores = cross_val_score(model, features, labels, cv=cv, n_jobs=None)
    return [float(score) for score in scores]


def _taxonomy_path_lookup(metadata: pd.DataFrame) -> Mapping[str, str | None]:
    paths = {}
    for taxonomy_id, group in metadata.groupby("taxonomy_id"):
        values = (
            value for value in group["taxonomy_path"].dropna().unique() if value
        )
        paths[str(taxonomy_id)] = next(iter(values), None)
    return paths


def compute_tag_summary(
    feature_names: Sequence[str],
    classes: Sequence[str],
    feature_log_prob: np.ndarray,
    *,
    taxonomy_paths: Mapping[str, str | None] | None = None,
) -> pd.DataFrame:
    """Summarise the most informative taxonomy for each tag."""

    taxonomy_paths = taxonomy_paths or {}
    records = []
    for column, tag in enumerate(feature_names):
        column_scores = feature_log_prob[:, column]
        top_index = int(np.argmax(column_scores))
        sorted_indices = np.argsort(column_scores)[::-1]
        top_score = float(column_scores[top_index])
        second_score = (
            float(column_scores[sorted_indices[1]])
            if len(sorted_indices) > 1
            else float("-inf")
        )
        margin = top_score - second_score if np.isfinite(second_score) else float("inf")
        taxonomy_id = str(classes[top_index])
        records.append(
            {
                "tag": str(tag),
                "top_taxonomy_id": taxonomy_id,
                "top_taxonomy_path": taxonomy_paths.get(taxonomy_id),
                "log_probability": top_score,
                "probability": float(np.exp(top_score)),
                "margin": margin,
            }
        )
    summary = pd.DataFrame.from_records(records)
    summary.sort_values(["margin", "probability"], ascending=[False, False], inplace=True)
    summary.reset_index(drop=True, inplace=True)
    return summary


def compute_taxonomy_top_tags(
    feature_names: Sequence[str],
    classes: Sequence[str],
    feature_log_prob: np.ndarray,
    *,
    taxonomy_paths: Mapping[str, str | None] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Rank tags within each taxonomy."""

    taxonomy_paths = taxonomy_paths or {}
    rows = []
    for class_index, taxonomy_id in enumerate(classes):
        column_scores = feature_log_prob[class_index]
        order = np.argsort(column_scores)[::-1][:top_k]
        for rank, feature_index in enumerate(order, start=1):
            score = float(column_scores[feature_index])
            rows.append(
                {
                    "taxonomy_id": str(taxonomy_id),
                    "taxonomy_path": taxonomy_paths.get(str(taxonomy_id)),
                    "tag": str(feature_names[feature_index]),
                    "log_probability": score,
                    "probability": float(np.exp(score)),
                    "rank": rank,
                }
            )
    top_tags = pd.DataFrame.from_records(rows)
    top_tags.sort_values(["taxonomy_id", "rank"], inplace=True)
    top_tags.reset_index(drop=True, inplace=True)
    return top_tags


def compute_taxonomy_priors(
    classes: Sequence[str], class_log_prior: np.ndarray, taxonomy_paths: Mapping[str, str | None]
) -> pd.DataFrame:
    rows = []
    for taxonomy_id, log_prior in zip(classes, class_log_prior):
        taxonomy_id = str(taxonomy_id)
        rows.append(
            {
                "taxonomy_id": taxonomy_id,
                "taxonomy_path": taxonomy_paths.get(taxonomy_id),
                "log_prior": float(log_prior),
                "prior": float(np.exp(log_prior)),
            }
        )
    df = pd.DataFrame.from_records(rows)
    df.sort_values("prior", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def render_report_html(
    summary: pd.DataFrame,
    top_tags: pd.DataFrame,
    stats: TrainingStats,
    output_path: Path,
    *,
    top_n: int = 50,
) -> None:
    """Write a standalone HTML report describing classifier behaviour."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_items = [
        f"<li><strong>Training samples:</strong> {stats.samples:,}</li>",
        f"<li><strong>Taxonomy classes:</strong> {stats.taxonomies:,}</li>",
        f"<li><strong>Unique tags:</strong> {stats.unique_tags:,}</li>",
        f"<li><strong>Training accuracy:</strong> {stats.training_accuracy:.3f}</li>",
    ]
    if (
        stats.cross_validation_folds
        and stats.cross_validation_mean_accuracy is not None
    ):
        mean = stats.cross_validation_mean_accuracy
        std = stats.cross_validation_std_accuracy or 0.0
        stats_items.append(
            "<li><strong>Cross-validated accuracy:</strong> "
            f"{mean:.3f} ± {std:.3f} ({stats.cross_validation_folds} folds)</li>"
        )

    stats_block = "<ul class=\"stats\">" + "".join(stats_items) + "</ul>"

    summary_table = summary.head(top_n).to_html(
        index=False,
        escape=False,
        classes="tag-summary",
        columns=[
            "tag",
            "top_taxonomy_id",
            "top_taxonomy_path",
            "probability",
            "margin",
        ],
        formatters={
            "probability": lambda v: f"{v:.4f}",
            "margin": lambda v: f"{v:.4f}",
        },
    )

    top_tags_sections = []
    for taxonomy_id, group in top_tags.groupby("taxonomy_id"):
        path = html.escape(group["taxonomy_path"].iloc[0] or "Unknown path")
        table = group[["rank", "tag", "probability"]].copy()
        table["probability"] = table["probability"].map(lambda v: f"{v:.4f}")
        table_html = table.to_html(index=False, classes="taxonomy-top-tags")
        top_tags_sections.append(
            f"<section class=\"taxonomy-section\"><h3>{html.escape(taxonomy_id)} — {path}</h3>{table_html}</section>"
        )

    top_tags_html = "".join(top_tags_sections)

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <title>Shopify taxonomy classifier report</title>
    <style>
      body {{ font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif; margin: 2rem; background: #f8fafc; color: #0f172a; }}
      h1 {{ margin-top: 0; }}
      .stats {{ list-style: none; padding: 0; display: flex; flex-wrap: wrap; gap: 1rem; }}
      .stats li {{ background: white; padding: 0.75rem 1rem; border-radius: 0.75rem; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); }}
      table {{ border-collapse: collapse; width: 100%; background: white; box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12); border-radius: 0.75rem; overflow: hidden; }}
      th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid #e2e8f0; }}
      th {{ background: #0b6ce3; color: white; font-weight: 600; }}
      tr:nth-child(even) td {{ background: #f1f5f9; }}
      section.taxonomy-section {{ margin-top: 2rem; }}
    </style>
  </head>
  <body>
    <h1>Shopify taxonomy classifier</h1>
    <p>This Complement Naive Bayes model estimates the most likely Shopify taxonomy for each product based on its tags. Review the most influential tags below.</p>
    {stats_block}
    <section>
      <h2>Tags with the strongest taxonomy association</h2>
      {summary_table}
    </section>
    <section>
      <h2>Top tags per taxonomy</h2>
      {top_tags_html}
    </section>
  </body>
</html>"""

    output_path.write_text(html_content, encoding="utf-8")


def save_summary_json(
    summary: pd.DataFrame,
    top_tags: pd.DataFrame,
    priors: pd.DataFrame,
    stats: TrainingStats,
    output_path: Path,
) -> None:
    payload = {
        "stats": {
            "samples": stats.samples,
            "taxonomies": stats.taxonomies,
            "unique_tags": stats.unique_tags,
            "training_accuracy": stats.training_accuracy,
        },
        "tag_summary": summary.to_dict(orient="records"),
        "top_tags": top_tags.to_dict(orient="records"),
        "taxonomy_priors": priors.to_dict(orient="records"),
    }
    if stats.cross_validation_folds and stats.cross_validation_mean_accuracy is not None:
        payload["stats"]["cross_validation"] = {
            "folds": stats.cross_validation_folds,
            "mean_accuracy": stats.cross_validation_mean_accuracy,
            "std_accuracy": stats.cross_validation_std_accuracy,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _ensure_tables(conn: Connection) -> None:
    db.ensure_schema(conn, "padjective")
    db.ensure_table(
        conn,
        "padjective",
        "taxonomy_nb_models",
        [
            "id SERIAL PRIMARY KEY",
            "trained_at TIMESTAMPTZ NOT NULL",
            "classifier TEXT NOT NULL",
            "samples INTEGER NOT NULL",
            "taxonomies INTEGER NOT NULL",
            "unique_tags INTEGER NOT NULL",
            "training_accuracy REAL NOT NULL",
            "cv_folds INTEGER",
            "cv_mean_accuracy REAL",
            "cv_std_accuracy REAL",
        ],
    )
    db.ensure_table(
        conn,
        "padjective",
        "taxonomy_nb_cv_scores",
        [
            "model_id INTEGER NOT NULL REFERENCES padjective.taxonomy_nb_models(id) ON DELETE CASCADE",
            "fold INTEGER NOT NULL",
            "accuracy REAL NOT NULL",
            "PRIMARY KEY (model_id, fold)",
        ],
    )
    db.ensure_table(
        conn,
        "padjective",
        "taxonomy_nb_tag_summary",
        [
            "model_id INTEGER NOT NULL REFERENCES padjective.taxonomy_nb_models(id) ON DELETE CASCADE",
            "tag TEXT NOT NULL",
            "top_taxonomy_id TEXT NOT NULL",
            "top_taxonomy_path TEXT",
            "probability REAL NOT NULL",
            "margin REAL NOT NULL",
            "log_probability REAL NOT NULL",
            "PRIMARY KEY (model_id, tag)",
        ],
    )
    db.ensure_table(
        conn,
        "padjective",
        "taxonomy_nb_top_tags",
        [
            "model_id INTEGER NOT NULL REFERENCES padjective.taxonomy_nb_models(id) ON DELETE CASCADE",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT",
            "tag TEXT NOT NULL",
            "rank INTEGER NOT NULL",
            "probability REAL NOT NULL",
            "log_probability REAL NOT NULL",
            "PRIMARY KEY (model_id, taxonomy_id, rank)",
        ],
    )
    db.ensure_table(
        conn,
        "padjective",
        "taxonomy_nb_priors",
        [
            "model_id INTEGER NOT NULL REFERENCES padjective.taxonomy_nb_models(id) ON DELETE CASCADE",
            "taxonomy_id TEXT NOT NULL",
            "taxonomy_path TEXT",
            "log_prior REAL NOT NULL",
            "prior REAL NOT NULL",
            "PRIMARY KEY (model_id, taxonomy_id)",
        ],
    )


def save_model_to_database(
    conn: Connection,
    model: ComplementNB,
    summary: pd.DataFrame,
    top_tags: pd.DataFrame,
    priors: pd.DataFrame,
    stats: TrainingStats,
    *,
    cv_scores: Iterable[float] | None = None,
) -> int:
    """Persist classifier artefacts into Postgres."""

    _ensure_tables(conn)

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO padjective.taxonomy_nb_models (
                trained_at,
                classifier,
                samples,
                taxonomies,
                unique_tags,
                training_accuracy,
                cv_folds,
                cv_mean_accuracy,
                cv_std_accuracy
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                datetime.now(timezone.utc),
                "ComplementNB",
                stats.samples,
                stats.taxonomies,
                stats.unique_tags,
                stats.training_accuracy,
                stats.cross_validation_folds,
                stats.cross_validation_mean_accuracy,
                stats.cross_validation_std_accuracy,
            ),
        )
        model_id = int(cur.fetchone()[0])

        if cv_scores:
            cur.executemany(
                """
                INSERT INTO padjective.taxonomy_nb_cv_scores (model_id, fold, accuracy)
                VALUES (%s, %s, %s)
                """,
                [
                    (model_id, index + 1, float(score))
                    for index, score in enumerate(cv_scores)
                ],
            )

        if not summary.empty:
            cur.executemany(
                """
                INSERT INTO padjective.taxonomy_nb_tag_summary (
                    model_id, tag, top_taxonomy_id, top_taxonomy_path, probability, margin, log_probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        model_id,
                        row.tag,
                        row.top_taxonomy_id,
                        row.top_taxonomy_path,
                        float(row.probability),
                        float(row.margin),
                        float(row.log_probability),
                    )
                    for row in summary.itertuples()
                ],
            )

        if not top_tags.empty:
            cur.executemany(
                """
                INSERT INTO padjective.taxonomy_nb_top_tags (
                    model_id, taxonomy_id, taxonomy_path, tag, rank, probability, log_probability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        model_id,
                        row.taxonomy_id,
                        row.taxonomy_path,
                        row.tag,
                        int(row.rank),
                        float(row.probability),
                        float(row.log_probability),
                    )
                    for row in top_tags.itertuples()
                ],
            )

        if not priors.empty:
            cur.executemany(
                """
                INSERT INTO padjective.taxonomy_nb_priors (
                    model_id, taxonomy_id, taxonomy_path, log_prior, prior
                ) VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        model_id,
                        row.taxonomy_id,
                        row.taxonomy_path,
                        float(row.log_prior),
                        float(row.prior),
                    )
                    for row in priors.itertuples()
                ],
            )
    conn.commit()
    return model_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a Complement Naive Bayes model that predicts Shopify taxonomy IDs from tags.",
    )
    parser.add_argument("--dsn", help="Postgres DSN. Defaults to SHOPIFY_DB_DSN or DATABASE_URL if unset.")
    parser.add_argument(
        "--product-table",
        default="cantbuymelove.product",
        help="Qualified product table containing Shopify products.",
    )
    parser.add_argument(
        "--min-tag-count",
        type=int,
        default=2,
        help="Minimum number of products a tag must appear in to be included.",
    )
    parser.add_argument(
        "--min-samples-per-taxonomy",
        type=int,
        default=5,
        help="Minimum labelled products per taxonomy class.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (limited by smallest class).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/taxonomy_nb_classifier"),
        help="Directory where reports will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of tag rows to include in the HTML report.",
    )
    args = parser.parse_args()

    conn = db.get_connection(args.dsn)
    try:
        features, labels, feature_names, metadata = load_training_data(
            conn,
            product_table=args.product_table,
            min_tag_count=args.min_tag_count,
            min_samples_per_taxonomy=args.min_samples_per_taxonomy,
        )

        taxonomy_paths = _taxonomy_path_lookup(metadata)

        cv_scores = cross_validate_classifier(
            features,
            labels,
            folds=max(2, args.cv_folds),
        )

        model, stats = train_classifier(features, labels)
        if cv_scores:
            stats.cross_validation_folds = len(cv_scores)
            stats.cross_validation_mean_accuracy = float(np.mean(cv_scores))
            stats.cross_validation_std_accuracy = float(np.std(cv_scores, ddof=0))

        summary = compute_tag_summary(
            feature_names,
            model.classes_,
            model.feature_log_prob_,
            taxonomy_paths=taxonomy_paths,
        )
        top_tags = compute_taxonomy_top_tags(
            feature_names,
            model.classes_,
            model.feature_log_prob_,
            taxonomy_paths=taxonomy_paths,
        )
        priors = compute_taxonomy_priors(
            model.classes_,
            model.class_log_prior_,
            taxonomy_paths,
        )

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "taxonomy_nb_report.html"
        json_path = output_dir / "taxonomy_nb_summary.json"

        render_report_html(summary, top_tags, stats, html_path, top_n=args.top_n)
        save_summary_json(summary, top_tags, priors, stats, json_path)
        save_model_to_database(
            conn,
            model,
            summary,
            top_tags,
            priors,
            stats,
            cv_scores=cv_scores,
        )

        print(
            f"Trained ComplementNB classifier on {stats.samples} samples across {stats.taxonomies} taxonomy classes."
        )
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
        print(f"Report written to {html_path}")
        print("Summary stored in Postgres schema padjective.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
