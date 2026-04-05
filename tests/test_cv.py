"""Tests for cross-validation helper utilities."""

from __future__ import annotations

import warnings
from typing import Iterable, List

import numpy as np
import pytest

from padjective import cv


class FakeCursor:
    """Minimal cursor stub that replays predefined result sets."""

    def __init__(self, results: List[Iterable], row_factory=None):
        self._results = results
        self._row_factory = row_factory
        self._current_rows: List = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _query, _params=None):
        if not self._results:
            raise AssertionError("No more canned results available for cursor.execute")
        next_rows = self._results.pop(0)
        self._current_rows = list(next_rows)

    def fetchall(self):
        return [self._apply_row_factory(row) for row in self._current_rows]

    def __iter__(self):
        for row in self._current_rows:
            yield self._apply_row_factory(row)

    def _apply_row_factory(self, row):
        if self._row_factory is None:
            return row
        return self._row_factory(row)


class FakeConnection:
    """Connection stub returning :class:`FakeCursor` instances."""

    def __init__(self, *result_sets: Iterable):
        self._result_sets = [list(result) for result in result_sets]

    def cursor(self, *_, row_factory=None):
        return FakeCursor(self._result_sets, row_factory=row_factory)


def dict_identity(row):
    """Return ``row`` unchanged. Used to patch ``dict_row`` in tests."""

    return row


def test_calculate_cv_folds_requires_taxonomy_path(monkeypatch):
    """We surface upstream issues if ``taxonomy_path`` is missing."""

    info_schema_rows = [("taxonomy_label",), ("raw_output",)]
    product_rows: List[Iterable] = []

    fake_conn = FakeConnection(info_schema_rows, product_rows)

    monkeypatch.setattr(cv, "dict_row", dict_identity)

    with pytest.raises(RuntimeError, match="taxonomy_path"):
        cv.calculate_cv_folds(fake_conn, n_splits=3, random_state=0)


def test_calculate_cv_folds_uses_taxonomy_path(monkeypatch):
    """Ensure CV splits succeed when ``taxonomy_path`` data is available."""

    # ``calculate_cv_folds`` requests table columns first and then streams rows.
    info_schema_rows = [("taxonomy_path",)]
    product_rows = [
        {"id": 1, "taxonomy_path": "A/B"},
        {"id": 2, "taxonomy_path": "A/B"},
        {"id": 3, "taxonomy_path": "A/B"},
        {"id": 4, "taxonomy_path": "C/D"},
        {"id": 5, "taxonomy_path": "C/D"},
        {"id": 6, "taxonomy_path": "C/D"},
        {"id": 7, "taxonomy_path": "E/F"},
        {"id": 8, "taxonomy_path": "E/F"},
        {"id": 9, "taxonomy_path": "E/F"},
    ]

    fake_conn = FakeConnection(info_schema_rows, product_rows)

    # ``dict_row`` normally converts tuples into dictionaries using cursor
    # metadata. Our stub already yields dictionaries, so return rows unchanged.
    monkeypatch.setattr(cv, "dict_row", dict_identity)

    folds = cv.calculate_cv_folds(fake_conn, n_splits=3, random_state=0)

    assert set(folds.keys()) == set(range(1, 10))
    # Every fold should receive at least one product from each taxonomy bucket.
    counts = np.bincount(list(folds.values()))
    assert counts.shape[0] == 3
    assert np.all(counts > 0)


def test_calculate_cv_folds_degenerates_to_kfold_when_needed(monkeypatch):
    """Ensure we fall back to simple KFold when stratification is impossible."""

    info_schema_rows = [("taxonomy_path",)]
    product_rows = [
        {"id": idx, "taxonomy_path": f"{idx}"} for idx in range(1, 9)
    ]

    fake_conn = FakeConnection(info_schema_rows, product_rows)

    monkeypatch.setattr(cv, "dict_row", dict_identity)

    class FailStratified:
        def __init__(self, *_, **__):  # pragma: no cover - defensive
            raise AssertionError("StratifiedKFold should not be used for unique labels")

    monkeypatch.setattr(cv, "StratifiedKFold", FailStratified)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = cv.calculate_cv_folds(fake_conn, n_splits=4, random_state=0)

    assert set(folds.keys()) == set(range(1, 9))
    counts = np.bincount(list(folds.values()))
    assert counts.shape[0] == 4
    assert np.all(counts == 2)
    assert caught == []


def test_calculate_cv_folds_can_warn_on_fallback(monkeypatch):
    """Callers can opt into a warning when stratification is impossible."""

    info_schema_rows = [("taxonomy_path",)]
    product_rows = [
        {"id": idx, "taxonomy_path": f"{idx}"} for idx in range(1, 9)
    ]

    fake_conn = FakeConnection(info_schema_rows, product_rows)

    monkeypatch.setattr(cv, "dict_row", dict_identity)

    with pytest.warns(RuntimeWarning, match="Taxonomy distribution too sparse"):
        folds = cv.calculate_cv_folds(
            fake_conn,
            n_splits=4,
            random_state=0,
            warn_on_fallback=True,
        )

    assert set(folds.keys()) == set(range(1, 9))
