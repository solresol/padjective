from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from padjective.synset_classifier import (
    TrainingStats,
    compute_tag_coefficients,
    load_training_data,
    render_coefficients_html,
    save_model_to_database,
    summarise_coefficients,
    train_classifier,
)


def _write_synset_rows(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE product_synsets (
                product_id INTEGER PRIMARY KEY,
                title TEXT,
                tags TEXT,
                synset_id TEXT,
                not_found INTEGER,
                reason TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO product_synsets (product_id, title, tags, synset_id, not_found, reason) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def test_load_training_data_filters_not_found(tmp_path: Path) -> None:
    db_path = tmp_path / "synsets.sqlite"
    _write_synset_rows(
        db_path,
        [
            (1, "Widget", "red,small", "n1", 0, ""),
            (2, "Gadget", "blue,small", "n2", 0, ""),
            (3, "Thing", "red,heavy", "n1", 0, ""),
            (4, "Mystery", "", "n3", 0, ""),
            (5, "Lost", "tiny", None, 0, ""),
            (6, "Skip", "green", "n4", 1, "no match"),
        ],
    )

    df = load_training_data(db_path, min_samples_per_synset=2)

    assert set(df["synset_id"]) == {"n1"}
    assert df.iloc[0]["tag_list"] == ("RED", "SMALL")
    assert len(df) == 2


def test_compute_tag_coefficients_ranks_by_maximum() -> None:
    feature_names = np.array(["RED", "SMALL", "BLUE"])
    classes = np.array(["n1", "n2"])
    coef = np.array([[1.0, -0.5, 0.1], [-0.2, 0.9, -1.5]])

    summary = compute_tag_coefficients(feature_names, classes, coef)

    assert list(summary["tag"][:2]) == ["BLUE", "RED"]
    blue_row = summary.loc[summary["tag"] == "BLUE"].iloc[0]
    assert blue_row["top_synset"] == "n2"
    assert blue_row["max_abs_coef"] == pytest.approx(1.5)


def test_render_coefficients_html(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "tag": "RED",
                "top_synset": "n1",
                "top_weight": 0.8,
                "max_abs_coef": 0.8,
                "sum_abs_coef": 1.2,
            },
            {
                "tag": "BLUE",
                "top_synset": "n2",
                "top_weight": -1.1,
                "max_abs_coef": 1.1,
                "sum_abs_coef": 1.6,
            },
        ]
    )
    stats = TrainingStats(
        samples=10,
        synsets=2,
        unique_tags=5,
        training_accuracy=0.9,
        cross_validation_folds=3,
        cross_validation_mean_accuracy=0.85,
        cross_validation_std_accuracy=0.02,
    )
    output = tmp_path / "report.html"

    render_coefficients_html(summary, stats, output, top_n=2)

    text = output.read_text(encoding="utf-8")
    assert "Synset tag coefficients" in text
    assert "Training samples" in text
    assert "BLUE" in text
    assert "Cross-validated accuracy" in text


def test_save_model_to_database_persists_weights(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "product_id": [1, 2, 3, 4],
            "synset_id": ["n1", "n1", "n2", "n2"],
            "tag_list": [
                ("RED", "ROUND"),
                ("BLUE",),
                ("RED",),
                ("BLUE", "ROUND"),
            ],
        }
    )

    model, stats = train_classifier(data)
    summary = summarise_coefficients(model)
    db_path = tmp_path / "classifier.sqlite"

    model_id = save_model_to_database(db_path, model, stats, summary, cv_scores=[0.6, 0.7, 0.8])

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute(
            "SELECT samples, synsets, training_accuracy, cv_folds, cv_mean_accuracy FROM synset_classifier_models"
        )
        row = cur.fetchone()
        assert row is not None
        samples, synsets, training_accuracy, cv_folds, cv_mean = row
        assert samples == stats.samples
        assert synsets == stats.synsets
        assert training_accuracy == pytest.approx(stats.training_accuracy)
        assert cv_folds == 3
        assert cv_mean == pytest.approx(0.7)

        coef_count = conn.execute(
            "SELECT COUNT(*) FROM synset_classifier_coefficients WHERE model_id = ?",
            (model_id,),
        ).fetchone()[0]
        assert coef_count == len(summary) * len(model.named_steps["classifier"].classes_)

        intercept_count = conn.execute(
            "SELECT COUNT(*) FROM synset_classifier_intercepts WHERE model_id = ?",
            (model_id,),
        ).fetchone()[0]
        assert intercept_count == len(model.named_steps["classifier"].classes_)

        summary_count = conn.execute(
            "SELECT COUNT(*) FROM synset_classifier_tag_summary WHERE model_id = ?",
            (model_id,),
        ).fetchone()[0]
        assert summary_count == len(summary)
    finally:
        conn.close()

