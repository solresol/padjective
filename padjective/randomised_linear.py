"""Exact finite-precision coordinate fits and taxonomy-consensus experiments.

These isolated algorithms do not change the historical floating-point greedy
fit. Strict integer comparisons and canonical residues are explicit protocol
choices. A first-pass checkpoint supplies a paired control.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy import sparse

from .published_mihara import validate_prime


def check_precision(p: int, precision: int, n: int) -> int:
    validate_prime(p)
    if precision < 1 or n < 1 or n * p**precision >= 2**62:
        raise ValueError("Positive precision/count within exact int64 range required")
    return p**precision


def loss_units(residuals, p: int, precision: int) -> np.ndarray:
    """Integer multiples of 1/p**precision; zero means exact residue agreement."""
    q = check_precision(p, precision, max(1, np.size(residuals)))
    delta = np.asarray(residuals, dtype=np.int64) % q
    units = np.full(delta.shape, q, dtype=np.int64)
    units[delta == 0] = 0
    active = (delta != 0) & (delta % p == 0)
    while np.any(active):
        units[active] //= p
        delta[active] //= p
        active = (delta != 0) & (delta % p == 0)
    return units


def medoid(values, p: int, precision: int) -> tuple[int, int]:
    """Exact residual medoid in O(E u log u), versus quadratic candidate scans.

    For each candidate, count responses sharing each successively longer
    prefix. Observations leaving the prefix at depth d contribute p**(E-d).
    The smallest canonical medoid breaks ties between candidate residues.
    """
    q = check_precision(p, precision, len(values))
    unique, counts = np.unique(np.asarray(values, dtype=np.int64) % q, return_counts=True)
    previous = np.full(len(unique), len(values), dtype=np.int64)
    costs = np.zeros(len(unique), dtype=np.int64)
    modulus, weight = 1, q
    for _ in range(precision):
        modulus *= p
        _, inverse = np.unique(unique % modulus, return_inverse=True)
        totals = np.zeros(int(inverse.max()) + 1, dtype=np.int64)
        np.add.at(totals, inverse, counts)
        matching = totals[inverse]
        costs += (previous - matching) * weight
        previous = matching
        weight //= p
    index = int(np.argmin(costs))
    return int(unique[index]), int(costs[index])


def binary_design(features) -> sparse.csc_matrix:
    matrix = sparse.csc_matrix(features, dtype=np.int64, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("A nonempty two-dimensional binary design is required")
    if np.any(matrix.data != 1):
        raise ValueError("Only binary features are supported")
    matrix.sort_indices()
    return matrix


def linear_predictions(features, coefficients, p: int, precision: int) -> np.ndarray:
    matrix = binary_design(features)
    q = check_precision(p, precision, max(matrix.shape))
    weights = np.asarray(coefficients, dtype=np.int64)
    if weights.shape != (matrix.shape[1],) or np.any(weights < 0) or np.any(weights >= q):
        raise ValueError("Aligned canonical coefficients required")
    return np.asarray(matrix @ weights).reshape(-1) % q


@dataclass(frozen=True)
class CoordinateFit:
    coefficients: np.ndarray
    first_pass_coefficients: np.ndarray | None
    residuals: np.ndarray
    initial_units: int
    final_units: int
    history: tuple[dict, ...]
    status: str
    elapsed_seconds: float
    coordinate_visits: int


def fit_coordinates(features, targets, *, p: int, precision: int, seed: int,
                    max_sweeps: int = 100, seconds: float = 300,
                    first_order=None, callback=None) -> CoordinateFit:
    started = time.monotonic()
    matrix = binary_design(features)
    q = check_precision(p, precision, max(matrix.shape))
    y = np.asarray(targets, dtype=np.int64)
    if y.shape != (matrix.shape[0],) or np.any(y < 0) or np.any(y >= q):
        raise ValueError("Aligned canonical targets required")
    if max_sweeps < 1 or seconds <= 0:
        raise ValueError("Positive resource limits required")
    active = np.flatnonzero(np.diff(matrix.indptr))
    if first_order is not None:
        first_order = np.asarray(first_order, dtype=np.int64)
        if sorted(first_order.tolist()) != active.tolist():
            raise ValueError("First order must contain each supported feature once")
    rng = np.random.default_rng(seed)
    weights = np.zeros(matrix.shape[1], dtype=np.int64)
    residuals = y.copy()
    initial = total = int(loss_units(residuals, p, precision).sum())
    history, first = [], None
    visits = 0
    status = "sweep_limit"
    for sweep in range(1, max_sweeps + 1):
        order = first_order if sweep == 1 and first_order is not None else rng.permutation(active)
        changes = 0
        complete = True
        for column in order:
            if time.monotonic() - started >= seconds:
                complete = False
                status = "time_limit"
                break
            ids = matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]]
            old = int(weights[column])
            before = int(loss_units(residuals[ids], p, precision).sum())
            adjusted = (residuals[ids] + old) % q
            candidate, after = medoid(adjusted, p, precision)
            if after < before:
                weights[column] = candidate
                residuals[ids] = (adjusted - candidate) % q
                total += after - before
                changes += 1
            visits += 1
        assert total == int(loss_units(residuals, p, precision).sum())
        item = dict(sweep=sweep, complete=complete, changes=changes,
                    training_loss_units=total, elapsed_seconds=time.monotonic()-started)
        history.append(item)
        if callback:
            callback(item)
        if sweep == 1 and complete:
            first = weights.copy()
        if not complete:
            break
        if not changes:
            status = "coordinate_optimum"
            break
    return CoordinateFit(weights, first, residuals, initial, total, tuple(history),
                         status, time.monotonic()-started, visits)


def coordinate_certificate(features, targets, coefficients, *, p: int, precision: int) -> dict:
    """Separate complete pass, without making updates."""
    matrix = binary_design(features)
    q = check_precision(p, precision, max(matrix.shape))
    predictions = linear_predictions(matrix, coefficients, p, precision)
    residuals = (np.asarray(targets, dtype=np.int64) - predictions) % q
    improving = []
    examined = 0
    for column in range(matrix.shape[1]):
        ids = matrix.indices[matrix.indptr[column]:matrix.indptr[column+1]]
        if not len(ids):
            continue
        examined += 1
        before = int(loss_units(residuals[ids], p, precision).sum())
        _, after = medoid((residuals[ids] + coefficients[column]) % q, p, precision)
        if after < before:
            improving.append(column)
    return dict(status="coordinate_optimum" if not improving else "improving_coordinates",
                examined=examined, improving_columns=improving)


def fit_default(targets, raw_predictions, *, p: int, precision: int) -> int:
    values = np.asarray(targets, dtype=np.int64)
    zero = np.asarray(raw_predictions) == 0
    if np.any(zero):
        # A residual medoid of these labels is itself a valid training path.
        return medoid(values[zero], p, precision)[0]
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def apply_default(raw_predictions, default: int) -> np.ndarray:
    values = np.asarray(raw_predictions, dtype=np.int64)
    return np.where(values == 0, default, values)


def consensus_predictions(member_predictions, candidates, *, p: int, precision: int) -> tuple[np.ndarray, dict]:
    """Choose a valid training path minimising total distance to member outputs.

    Prefix histograms evaluate the objective exactly without a dense
    products-by-members-by-paths array. Ties choose the smallest valid code.
    """
    predictions = np.asarray(member_predictions, dtype=np.int64)
    options = np.unique(np.asarray(candidates, dtype=np.int64))
    if predictions.ndim != 2 or not predictions.size or not len(options):
        raise ValueError("Nonempty product-by-member predictions and paths required")
    q = check_precision(p, precision, predictions.shape[1] * max(1, len(options)))
    if np.any(predictions < 0) or np.any(predictions >= q) or np.any(options < 0) or np.any(options >= q):
        raise ValueError("Canonical predictions and candidates required")
    members = predictions.shape[1]
    option_prefixes = [options % p**d for d in range(1, precision+1)]
    output = []
    for row in predictions:
        previous = np.full(len(options), members, dtype=np.int64)
        costs = np.zeros(len(options), dtype=np.int64)
        weight = q
        for depth, prefixes in enumerate(option_prefixes, 1):
            keys, counts = np.unique(row % p**depth, return_counts=True)
            indices = np.searchsorted(keys, prefixes)
            safe = np.minimum(indices, len(keys)-1)
            matching = np.where((indices < len(keys)) & (keys[safe] == prefixes), counts[safe], 0)
            costs += (previous - matching) * weight
            previous = matching
            weight //= p
        output.append(int(options[int(np.argmin(costs))]))
    return np.array(output, dtype=np.int64), dict(
        members=members, candidate_paths=len(options),
        candidate_prefix_evaluations_per_prediction=precision*len(options),
        member_prefix_observations_per_prediction=precision*members)
