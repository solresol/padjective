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
    stats = TrainingStats(samples=10, synsets=2, unique_tags=5, training_accuracy=0.9)
    output = tmp_path / "report.html"

    render_coefficients_html(summary, stats, output, top_n=2)

    text = output.read_text(encoding="utf-8")
    assert "Synset tag coefficients" in text
    assert "Training samples" in text
    assert "BLUE" in text

