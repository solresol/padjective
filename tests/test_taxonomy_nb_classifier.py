import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from padjective.taxonomy_nb_classifier import (
    TrainingStats,
    compute_tag_summary,
    compute_taxonomy_top_tags,
    render_report_html,
    save_summary_json,
    train_classifier,
)


def test_compute_tag_summary_orders_by_margin() -> None:
    feature_names = ["RED", "BLUE", "GREEN"]
    classes = np.array(["100", "200"])
    feature_log_prob = np.array(
        [
            [0.0, -2.0, -1.0],
            [-3.0, -0.5, -4.0],
        ]
    )

    summary = compute_tag_summary(feature_names, classes, feature_log_prob)

    assert summary.iloc[0]["tag"] == "RED"
    assert summary.iloc[0]["top_taxonomy_id"] == "100"
    assert summary.iloc[-1]["tag"] == "BLUE"


def test_compute_taxonomy_top_tags_ranks_per_class() -> None:
    feature_names = ["ALPHA", "BETA", "GAMMA"]
    classes = np.array(["A", "B"])
    feature_log_prob = np.array(
        [
            [-0.1, -1.0, -3.0],
            [-2.0, -0.2, -0.4],
        ]
    )

    top_tags = compute_taxonomy_top_tags(feature_names, classes, feature_log_prob, top_k=2)

    a_rows = top_tags[top_tags["taxonomy_id"] == "A"]
    assert list(a_rows["tag"]) == ["ALPHA", "BETA"]
    assert list(a_rows["rank"]) == [1, 2]

    b_rows = top_tags[top_tags["taxonomy_id"] == "B"]
    assert list(b_rows["tag"]) == ["BETA", "GAMMA"]


def test_render_report_and_json(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        [
            {
                "tag": "BLUE",
                "top_taxonomy_id": "200",
                "top_taxonomy_path": "Clothing / Tops",
                "log_probability": -0.5,
                "probability": float(np.exp(-0.5)),
                "margin": 1.5,
            },
            {
                "tag": "RED",
                "top_taxonomy_id": "100",
                "top_taxonomy_path": "Accessories / Hats",
                "log_probability": -0.2,
                "probability": float(np.exp(-0.2)),
                "margin": 0.8,
            },
        ]
    )
    top_tags = pd.DataFrame(
        [
            {
                "taxonomy_id": "100",
                "taxonomy_path": "Accessories / Hats",
                "tag": "RED",
                "log_probability": -0.2,
                "probability": float(np.exp(-0.2)),
                "rank": 1,
            }
        ]
    )
    priors = pd.DataFrame(
        [
            {
                "taxonomy_id": "100",
                "taxonomy_path": "Accessories / Hats",
                "log_prior": -0.7,
                "prior": float(np.exp(-0.7)),
            }
        ]
    )
    stats = TrainingStats(
        samples=120,
        taxonomies=12,
        unique_tags=400,
        training_accuracy=0.82,
        cross_validation_folds=3,
        cross_validation_mean_accuracy=0.79,
        cross_validation_std_accuracy=0.03,
    )

    html_path = tmp_path / "report.html"
    json_path = tmp_path / "summary.json"

    render_report_html(summary, top_tags, stats, html_path, top_n=5)
    save_summary_json(summary, top_tags, priors, stats, json_path)

    html_text = html_path.read_text(encoding="utf-8")
    assert "Shopify taxonomy classifier" in html_text
    assert "Top tags per taxonomy" in html_text
    assert "BLUE" in html_text

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["stats"]["samples"] == 120
    assert payload["tag_summary"][0]["tag"] == "BLUE"
    assert payload["top_tags"][0]["taxonomy_id"] == "100"


def test_train_classifier_requires_multiple_classes() -> None:
    features = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array(["100", "100"])

    with pytest.raises(ValueError):
        train_classifier(features, labels)
