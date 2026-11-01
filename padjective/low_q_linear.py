"""Depth-weighted medoids and coordinate fits with the encoding prime fixed.

The historical fitting modules are deliberately not modified. For sqrt(2),
integer weights use a Pell convergent whose ordering is certified for the
requested sample size. Thus optimisation decisions need no floating tolerance.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .randomised_linear import binary_design, check_precision, linear_predictions


SQRT2_NUMERATOR = 131836323
SQRT2_DENOMINATOR = 93222358


def depth_weights(kind: str, precision: int, n: int) -> tuple[np.ndarray, int]:
    if precision < 1 or n < 1:
        raise ValueError("Positive precision and observation count required")
    if kind == "sqrt2":
        # |P/D-sqrt(2)| < 1/D^2. For differences a+b*sqrt(2),
        # |a|,|b| <= n*scale, a nonzero difference exceeds 1/(3*n*scale).
        # This bound guarantees the rational surrogate has the same sign.
        scale = 2 ** (precision // 2)
        if 3 * (n * scale) ** 2 >= SQRT2_DENOMINATOR ** 2:
            raise ValueError("Sample/precision exceeds the certified sqrt(2) ordering bound")
        weights = [
            (scale // 2 ** (d // 2)) * SQRT2_DENOMINATOR if d % 2 == 0
            else (scale // 2 ** ((d + 1) // 2)) * SQRT2_NUMERATOR
            for d in range(precision)
        ]
    elif kind in {"2", "71"}:
        base = int(kind)
        weights = [base ** (precision - 1 - d) for d in range(precision)]
    else:
        raise ValueError("Supported depth weights are sqrt2, 2 and 71")
    if n * weights[0] >= 2 ** 62:
        raise ValueError("Integer objective exceeds the supported range")
    return np.asarray(weights, dtype=np.int64), weights[0]


def weight_value(kind: str) -> float:
    return math.sqrt(2) if kind == "sqrt2" else float(kind)


def loss_units(residuals, *, kind: str, p: int = 71, precision: int = 7):
    delta = np.asarray(residuals, dtype=np.int64) % p ** precision
    weights, _ = depth_weights(kind, precision, max(1, delta.size))
    output = np.zeros(delta.shape, dtype=np.int64)
    remaining = delta != 0
    for weight in weights:
        different = remaining & (delta % p != 0)
        output[different] = weight
        remaining &= ~different
        delta[remaining] //= p
    assert not np.any(remaining)
    return output


def best_choice(values, *, kind: str, p: int = 71, precision: int | None = None,
                candidates=None) -> tuple[int, int]:
    observations, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    if not len(observations):
        raise ValueError("Nonempty observations required")
    options = observations if candidates is None else np.unique(np.asarray(candidates, dtype=np.int64))
    if not len(options):
        raise ValueError("Nonempty candidate set required")
    span = max(int(observations[-1]), int(options[-1])) - min(int(observations[0]), int(options[0]))
    if precision is None:
        precision, modulus = 1, p
        while modulus <= span:
            precision += 1
            modulus *= p
    weights, _ = depth_weights(kind, precision, int(counts.sum()))
    previous = np.full(len(options), int(counts.sum()), dtype=np.int64)
    costs = np.zeros(len(options), dtype=np.int64)
    modulus = 1
    for weight in weights:
        modulus *= p
        if modulus > span:
            keys, totals = observations, counts
            prefixes = options
        else:
            keys, inverse = np.unique(observations % modulus, return_inverse=True)
            totals = np.zeros(len(keys), dtype=np.int64)
            np.add.at(totals, inverse, counts)
            prefixes = options % modulus
        positions = np.searchsorted(keys, prefixes)
        safe = np.minimum(positions, len(keys) - 1)
        matching = np.where((positions < len(keys)) & (keys[safe] == prefixes), totals[safe], 0)
        costs += (previous - matching) * weight
        previous = matching
    best = int(np.argmin(costs))
    return int(options[best]), int(costs[best])


def fit_coordinates(features, targets, *, kind: str, p: int = 71, precision: int = 7,
                    seed: int, max_sweeps: int = 100, seconds: float = 300):
    started = time.monotonic()
    matrix = binary_design(features)
    modulus = check_precision(p, precision, max(matrix.shape))
    y = np.asarray(targets, dtype=np.int64)
    if y.shape != (matrix.shape[0],) or np.any(y < 0) or np.any(y >= modulus):
        raise ValueError("Aligned canonical targets required")
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(np.diff(matrix.indptr))
    coefficients = np.zeros(matrix.shape[1], dtype=np.int64)
    residuals = y.copy()
    total = int(loss_units(residuals, kind=kind, p=p, precision=precision).sum())
    history = []
    status = "sweep_limit"
    for sweep in range(1, max_sweeps + 1):
        changes, complete = 0, True
        for column in rng.permutation(active):
            if time.monotonic() - started >= seconds:
                status, complete = "time_limit", False
                break
            rows = matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]]
            before = int(loss_units(residuals[rows], kind=kind, p=p, precision=precision).sum())
            adjusted = (residuals[rows] + coefficients[column]) % modulus
            candidate, after = best_choice(adjusted, kind=kind, p=p, precision=precision)
            if after < before:
                coefficients[column] = candidate
                residuals[rows] = (adjusted - candidate) % modulus
                total += after - before
                changes += 1
        assert total == int(loss_units(residuals, kind=kind, p=p, precision=precision).sum())
        history.append(dict(sweep=sweep, complete=complete, changes=changes, objective_units=total))
        if not complete:
            break
        if not changes:
            status = "coordinate_optimum"
            break
    return dict(coefficients=coefficients, history=history, status=status,
                elapsed_seconds=time.monotonic() - started)


def certificate(features, targets, coefficients, *, kind: str, p=71, precision=7):
    matrix = binary_design(features)
    residuals = (np.asarray(targets) - linear_predictions(matrix, coefficients, p, precision)) % p ** precision
    improving = []
    for column in np.flatnonzero(np.diff(matrix.indptr)):
        rows = matrix.indices[matrix.indptr[column]:matrix.indptr[column + 1]]
        before = int(loss_units(residuals[rows], kind=kind, p=p, precision=precision).sum())
        _, after = best_choice((residuals[rows] + coefficients[column]) % p ** precision,
                               kind=kind, p=p, precision=precision)
        if after < before:
            improving.append(int(column))
    return dict(status="coordinate_optimum" if not improving else "improving_coordinates",
                improving=improving)


def fit_default(targets, raw_predictions, *, kind: str, p=71, precision=7):
    zero = np.asarray(raw_predictions) == 0
    values = np.asarray(targets)
    if np.any(zero):
        return best_choice(values[zero], kind=kind, p=p, precision=precision)[0]
    unique, counts = np.unique(values, return_counts=True)
    return int(unique[int(np.argmax(counts))])


def consensus_predictions(member_predictions, candidates, *, kind: str, p=71, precision=7):
    options = np.unique(np.asarray(candidates, dtype=np.int64))
    return np.asarray([best_choice(row, kind=kind, p=p, precision=precision, candidates=options)[0]
                       for row in np.asarray(member_predictions)], dtype=np.int64)
