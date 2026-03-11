import numpy as np
from scipy import sparse

from padjective.taxonomy_levelwise_classifier import (
    predict_levelwise_taxonomy,
    train_levelwise_models,
)


def test_levelwise_classifier_predicts_valid_taxonomy_leaf() -> None:
    features = sparse.csr_matrix(
        [
            [3.0, 0.0, 0.0, 0.0],
            [2.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 2.5, 0.5],
        ]
    )
    labels = np.array(["A1", "A2", "B1", "B2"])
    taxonomy_paths = {
        "A1": ("A", "A1"),
        "A2": ("A", "A2"),
        "B1": ("B", "B1"),
        "B2": ("B", "B2"),
    }

    models = train_levelwise_models(features, labels, taxonomy_paths, max_iter=500)
    prediction, scoring_ops = predict_levelwise_taxonomy(features[0], models)

    assert prediction in taxonomy_paths
    assert prediction == "A1"
    assert scoring_ops > 0


def test_levelwise_classifier_handles_single_child_subtree() -> None:
    features = sparse.csr_matrix(
        [
            [2.0, 0.0, 0.0],
            [1.8, 0.2, 0.0],
            [0.0, 2.0, 1.0],
        ]
    )
    labels = np.array(["A1", "A2", "B1"])
    taxonomy_paths = {
        "A1": ("A", "A1"),
        "A2": ("A", "A2"),
        "B1": ("B", "B1"),
    }

    models = train_levelwise_models(features, labels, taxonomy_paths, max_iter=500)
    prediction, _ = predict_levelwise_taxonomy(features[2], models)

    assert prediction == "B1"
