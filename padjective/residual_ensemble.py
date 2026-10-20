"""Training-error masks and exact, incrementally accumulated consensus votes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .paper_validate_randomised import distance_matrix
from .randomised_linear import (
    apply_default, binary_design, check_precision, fit_coordinates, fit_default,
    linear_predictions,
)


class CumulativeConsensus:
    """Retain integer candidate costs; ties choose the smallest path code."""

    def __init__(self, rows, candidates, *, p=71, precision=7, max_members=243):
        self.options = np.unique(np.asarray(candidates, dtype=np.int64))
        self.p, self.precision = p, precision
        self.q = check_precision(p, precision, max_members)
        if rows < 1 or max_members < 1 or not len(self.options):
            raise ValueError('Positive row/member counts and candidate paths required')
        if np.any(self.options < 0) or np.any(self.options >= self.q):
            raise ValueError('Canonical candidate paths required')
        self.costs = np.zeros((rows, len(self.options)), dtype=np.int64)
        self.members, self.max_members = 0, max_members

    def add(self, predictions):
        values = np.asarray(predictions, dtype=np.int64)
        if values.shape != (len(self.costs),):
            raise ValueError('One prediction per row required')
        if np.any(values < 0) or np.any(values >= self.q):
            raise ValueError('Canonical predictions required')
        if self.members >= self.max_members:
            raise ValueError('Member bound exceeded')
        unique, inverse = np.unique(values, return_inverse=True)
        self.costs += distance_matrix(unique, self.options, self.p, self.precision)[inverse]
        self.members += 1
        return self.predict()

    def predict(self):
        if not self.members:
            raise ValueError('A vote requires at least one member')
        return self.options[np.argmin(self.costs, axis=1)]


def error_rows(targets, previous_predictions=None):
    """Compute a new mask each time; previously removed rows may re-enter."""
    actual = np.asarray(targets, dtype=np.int64)
    if actual.ndim != 1 or not len(actual):
        raise ValueError('Nonempty aligned training labels required')
    if previous_predictions is None:
        return np.arange(len(actual))
    previous = np.asarray(previous_predictions, dtype=np.int64)
    if previous.shape != actual.shape:
        raise ValueError('Aligned predictions required')
    return np.flatnonzero(previous != actual)


@dataclass(frozen=True)
class ResidualMember:
    selected_rows: np.ndarray
    coefficients: np.ndarray
    default: int
    training_raw: np.ndarray
    predictions_raw: np.ndarray
    training_predictions: np.ndarray
    predictions: np.ndarray
    fit: object


def fit_error_member(x_train, y_train, x_predict, *, previous_predictions=None,
                     seed, p=71, precision=7, max_sweeps=100, seconds=300):
    """Fit only training errors. No held-out target argument exists."""
    train, predict = binary_design(x_train).tocsr(), binary_design(x_predict).tocsr()
    targets = np.asarray(y_train, dtype=np.int64)
    if targets.shape != (train.shape[0],) or predict.shape[1] != train.shape[1]:
        raise ValueError('Aligned training rows and prediction columns required')
    selected = error_rows(targets, previous_predictions)
    if not len(selected):
        return None
    fit = fit_coordinates(train[selected], targets[selected], seed=seed, p=p,
                          precision=precision, max_sweeps=max_sweeps, seconds=seconds)
    if fit.status != 'coordinate_optimum':
        raise RuntimeError(f'Incomplete member: {fit.status}')
    raw_train = linear_predictions(train, fit.coefficients, p, precision)
    raw_predict = linear_predictions(predict, fit.coefficients, p, precision)
    default = fit_default(targets[selected], raw_train[selected], p=p, precision=precision)
    return ResidualMember(selected, fit.coefficients, default, raw_train, raw_predict,
        apply_default(raw_train, default), apply_default(raw_predict, default), fit)
