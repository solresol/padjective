"""Exact coordinate fits for deeper live taxonomies, separate from paper code.

Loss is stored in units of p**-(E-1), removing the unnecessary common factor p
from the paper implementation. Python integers are used when a sum can exceed
int64. Residues remain int64; unrepresentable moduli fail, never wrap.
"""
from __future__ import annotations

import time
import numpy as np

from .published_mihara import validate_prime
from .randomised_linear import binary_design

LIMIT = 2**62


def parameters(p, precision):
    validate_prime(p)
    if precision < 1 or p**precision >= LIMIT:
        raise ValueError('Positive precision and modulus below 2**62 required')
    return p**precision, p**(precision-1)


def total(values):
    """Reduce without int64 overflow, even for very large labelled cohorts."""
    a = np.asarray(values)
    if a.dtype == object or (a.size and int(a.max())*a.size >= LIMIT):
        return sum(map(int, a.flat))
    return int(a.sum())


def units(residuals, p, precision):
    q, scale = parameters(p, precision)
    delta = np.asarray(residuals, dtype=np.int64) % q
    out = np.full(delta.shape, scale, dtype=np.int64)
    out[delta == 0] = 0
    for depth in range(1, precision):
        out[(delta != 0) & (delta % p**depth == 0)] = p**(precision-1-depth)
    return out


def medoid(values, p, precision):
    q, scale = parameters(p, precision)
    values = np.asarray(values, dtype=np.int64) % q
    if not len(values):
        raise ValueError('Nonempty medoid input required')
    unique, counts = np.unique(values, return_counts=True)
    dtype = object if len(values)*scale >= LIMIT else np.int64
    previous = np.full(len(unique), len(values), dtype=np.int64)
    costs = np.zeros(len(unique), dtype=dtype)
    for depth in range(1, precision+1):
        _, inverse = np.unique(unique % p**depth, return_inverse=True)
        totals = np.zeros(int(inverse.max())+1, dtype=np.int64)
        np.add.at(totals, inverse, counts)
        matching = totals[inverse]
        costs += (previous-matching).astype(dtype)*p**(precision-depth)
        previous = matching
    index = int(np.argmin(costs))
    return int(unique[index]), int(costs[index])


def predict(features, coefficients, p, precision):
    q, _ = parameters(p, precision)
    x = features.tocsr()
    weights = np.asarray(coefficients, dtype=np.int64)
    if len(weights) != x.shape[1] or np.any(weights < 0) or np.any(weights >= q):
        raise ValueError('Aligned canonical coefficients required')
    if np.any(x.data != 1):
        raise ValueError('Binary features required')
    if int(np.diff(x.indptr).max(initial=0))*(q-1) < LIMIT:
        return np.asarray(x @ weights).reshape(-1) % q
    return direct_predict(x, weights, q)


def direct_predict(features, coefficients, q):
    x = features.tocsr()
    return np.array([sum(int(coefficients[j]) for j in
        x.indices[x.indptr[i]:x.indptr[i+1]]) % q for i in range(x.shape[0])], dtype=np.int64)


def fit(features, targets, *, p, precision, seed, max_sweeps=100, seconds=600):
    started = time.monotonic()
    x = binary_design(features)
    q, _ = parameters(p, precision)
    y = np.asarray(targets, dtype=np.int64)
    if y.shape != (x.shape[0],) or np.any(y < 0) or np.any(y >= q):
        raise ValueError('Aligned canonical targets required')
    if seconds <= 0 or max_sweeps < 1:
        raise ValueError('Positive fitting caps required')
    weights = np.zeros(x.shape[1], dtype=np.int64)
    residual = y.copy()
    active = np.flatnonzero(np.diff(x.indptr))
    rng = np.random.default_rng(seed)
    history = []
    status = 'sweep_limit'
    for sweep in range(1, max_sweeps+1):
        changes = 0
        for column in rng.permutation(active):
            if time.monotonic()-started >= seconds:
                status = 'time_limit'
                break
            ids = x.indices[x.indptr[column]:x.indptr[column+1]]
            before = total(units(residual[ids], p, precision))
            adjusted = (residual[ids]+weights[column]) % q
            candidate, after = medoid(adjusted, p, precision)
            if after < before:
                weights[column] = candidate
                residual[ids] = (adjusted-candidate) % q
                changes += 1
        history.append(dict(sweep=sweep, changes=changes,
            complete=status != 'time_limit', loss_units=total(units(residual, p, precision))))
        if status == 'time_limit':
            break
        if not changes:
            status = 'coordinate_optimum'
            break
    return dict(coefficients=weights.tolist(), history=history, status=status,
                seconds=time.monotonic()-started)


def certificate(features, targets, coefficients, p, precision):
    """Independent candidate-by-candidate valuation check, bounded temporary RAM.

    Each residual candidate is checked directly in blocks, without the fitter's
    prefix histogram. This is quadratic in distinct coordinate residuals.
    """
    q, scale = parameters(p, precision)
    x = features.tocsc()
    raw = direct_predict(x, coefficients, q)
    residual = (np.asarray(targets)-raw) % q
    for column in range(x.shape[1]):
        ids = x.indices[x.indptr[column]:x.indptr[column+1]]
        if not len(ids):
            continue
        adjusted, counts = np.unique((residual[ids]+coefficients[column]) % q, return_counts=True)
        before = total(units(residual[ids], p, precision))
        dtype = object if len(ids)*scale >= LIMIT else np.int64
        for start in range(0, len(adjusted), 32):
            delta = np.abs(adjusted[start:start+32, None]-adjusted[None, :])
            costs = np.full(delta.shape, scale, dtype=dtype)
            costs[delta == 0] = 0
            for depth in range(1, precision):
                costs[(delta != 0) & (delta % p**depth == 0)] = p**(precision-1-depth)
            if np.any(costs @ counts < before):
                raise AssertionError(f'Improving coordinate {column}')
    return raw


def default_value(targets, raw, p, precision):
    values = np.asarray(targets)[np.asarray(raw) == 0]
    if len(values):
        return medoid(values, p, precision)[0]
    unique, counts = np.unique(targets, return_counts=True)
    return int(unique[np.argmax(counts)])


def consensus(bank, candidates, p, precision):
    """Valid-path medoid of raw voters using bounded prefix-count state."""
    q, scale = parameters(p, precision)
    bank = np.asarray(bank, dtype=np.int64)
    options = np.unique(candidates)
    if bank.ndim != 2 or not bank.size or not len(options):
        raise ValueError('Nonempty product-by-member matrix and candidates required')
    if np.any(bank < 0) or np.any(bank >= q) or np.any(options < 0) or np.any(options >= q):
        raise ValueError('Canonical residues required')
    dtype = object if bank.shape[1]*scale >= LIMIT else np.int64
    prefixes = [options % p**d for d in range(1, precision+1)]
    result = []
    for row in bank:
        previous = np.full(len(options), len(row), dtype=np.int64)
        costs = np.zeros(len(options), dtype=dtype)
        for depth, prefix in enumerate(prefixes, 1):
            keys, counts = np.unique(row % p**depth, return_counts=True)
            positions = np.searchsorted(keys, prefix)
            safe = np.minimum(positions, len(keys)-1)
            matching = np.where((positions < len(keys)) & (keys[safe] == prefix), counts[safe], 0)
            costs += (previous-matching).astype(dtype)*p**(precision-depth)
            previous = matching
        result.append(int(options[np.argmin(costs)]))
    return np.array(result, dtype=np.int64)


def scores(targets, predictions, p, precision):
    y, pred = np.asarray(targets), np.asarray(predictions)
    if y.shape != pred.shape or y.ndim != 1 or not len(y):
        raise ValueError('Nonempty aligned outcomes required')
    loss = total(units(y-pred, p, precision))
    return dict(n=len(y), loss_units=loss, loss_scale=p**(precision-1),
                mean_padic_loss=loss/(len(y)*p**(precision-1)),
                exact_correct=int(np.sum(y == pred)), root_correct=int(np.sum(y % p == pred % p)),
                exact_accuracy=float(np.mean(y == pred)), root_accuracy=float(np.mean(y % p == pred % p)))
