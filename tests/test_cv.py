"""Tests for cross-validation helper utilities."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

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


def test_calculate_cv_folds_falls_back_to_taxonomy_path(monkeypatch):
    """Ensure CV splits succeed when only ``taxonomy_path`` is available."""

    # ``calculate_cv_folds`` requests table columns first and then streams rows.
    info_schema_rows = [("taxonomy_path",), ("raw_output",)]
    product_rows = [
        {"id": 1, "taxonomy_label": "A/B"},
        {"id": 2, "taxonomy_label": "A/B"},
        {"id": 3, "taxonomy_label": "A/B"},
        {"id": 4, "taxonomy_label": "C/D"},
        {"id": 5, "taxonomy_label": "C/D"},
        {"id": 6, "taxonomy_label": "C/D"},
        {"id": 7, "taxonomy_label": "E/F"},
        {"id": 8, "taxonomy_label": "E/F"},
        {"id": 9, "taxonomy_label": "E/F"},
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
