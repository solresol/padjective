"""Uniform half-reference sampling and vectorised exact taxonomy consensus."""
from __future__ import annotations

import numpy as np

from .randomised_linear import check_precision

SAMPLING_SEED = 20261020


def reference_members(previous_count, *, fold, roster, percent=50):
    """One reproducible uniform sample, independent of labels and fit RNGs."""
    if previous_count < 0 or fold < 0 or roster < 0 or not isinstance(percent, int) or not 1 <= percent <= 100:
        raise ValueError('Nonnegative counts and stream identifiers required')
    if not previous_count:
        return np.array([], dtype=np.int64)
    rng = np.random.default_rng(np.random.SeedSequence(
        [SAMPLING_SEED, fold, roster, previous_count]))
    return np.sort(rng.choice(previous_count, (percent*previous_count+99)//100, replace=False))


def reference_consensus(predictions, candidates, *, p=71, precision=7):
    """Exact candidate costs via vectorised, per-row prefix agreement counts.

    Scratch space scales with rows times candidates plus rows times members;
    there is no rows-times-members-times-candidates tensor or stored cost bank.
    """
    bank = np.asarray(predictions, dtype=np.int64)
    options = np.unique(np.asarray(candidates, dtype=np.int64))
    if bank.ndim != 2 or not bank.size or not len(options):
        raise ValueError('Nonempty rows-by-members predictions and candidates required')
    q = check_precision(p, precision, bank.shape[1])
    if np.any(bank < 0) or np.any(bank >= q) or np.any(options < 0) or np.any(options >= q):
        raise ValueError('Canonical predictions and candidates required')
    rows, members = bank.shape
    costs = np.full((rows, len(options)), members*q, dtype=np.int64)
    for depth in range(1, precision+1):
        prefixes, inverse = np.unique(options % p**depth, return_inverse=True)
        observed = bank % p**depth
        positions = np.searchsorted(prefixes, observed)
        safe = np.minimum(positions, len(prefixes)-1)
        valid = (positions < len(prefixes)) & (prefixes[safe] == observed)
        indices = (np.arange(rows)[:, None]*len(prefixes)+safe)[valid]
        counts = np.bincount(indices, minlength=rows*len(prefixes)).reshape(rows, len(prefixes))
        bonus = p**(precision-depth+1) - (p**(precision-depth) if depth < precision else 0)
        costs -= counts[:, inverse]*bonus
    return options[np.argmin(costs, axis=1)]
