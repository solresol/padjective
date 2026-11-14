"""Mihara arXiv:2604.13137v2, Algorithms 1--8, over exact finite fields.

External resource limits are interrupt conditions, not acceptance rules. The
old bounded-consensus diagnostic in taxonomy_mihara_comparison is separate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time
from typing import Callable, Sequence

import numpy as np


class InterruptedFit(RuntimeError):
    """External limit or a certified undefined/impossible computation."""


def validate_prime(p: int) -> None:
    if p < 2 or any(p % d == 0 for d in range(2, math.isqrt(p) + 1)):
        raise ValueError("p must be prime")
    if p > 32749:
        raise ValueError("This exact int64 implementation supports primes <=32749")


@dataclass
class ResourceBudget:
    max_draws: int = 1_000_000
    seconds: float = 300.0
    clock: Callable[[], float] = time.monotonic
    draws: int = 0
    restarts: int = 0
    inclusion_tests: int = 0
    started: float = field(init=False)

    def __post_init__(self) -> None:
        if self.max_draws < 1 or self.seconds <= 0:
            raise ValueError("Resource limits must be positive")
        self.started = self.clock()

    def check(self) -> None:
        if self.clock() - self.started >= self.seconds:
            raise InterruptedFit("time_limit")

    def draw(self, rng: random.Random, count: int) -> int:
        self.check()
        if self.draws >= self.max_draws:
            raise InterruptedFit("draw_limit")
        self.draws += 1
        return rng.randrange(count)


@dataclass(frozen=True)
class Echelon:
    """Extended reduced row echelon form of (x, 1 | y)."""

    rows: np.ndarray
    pivots: tuple[int, ...]
    p: int

    @classmethod
    def empty(cls, dimension: int, p: int) -> Echelon:
        return cls(np.empty((0, dimension + 2), dtype=np.int64), (), p)

    @property
    def rank(self) -> int:
        return len(self.pivots)

    def add(self, row: Sequence[int]) -> tuple[bool, Echelon]:
        """Algorithm 1; immutable input supplies Algorithm 5's copy semantics."""
        v = np.asarray(row, dtype=np.int64).copy() % self.p
        if v.shape != (self.rows.shape[1],):
            raise ValueError("Extended row has the wrong dimension")
        if self.rank:
            v = (v - v[list(self.pivots)] @ self.rows) % self.p
        nonzero = np.flatnonzero(v)
        if not len(nonzero):
            return True, self
        pivot = int(nonzero[0])
        if pivot == len(v) - 1:
            return False, self
        v = v * pow(int(v[pivot]), -1, self.p) % self.p
        updated = (self.rows - self.rows[:, pivot, None] * v[None, :]) % self.p
        return True, Echelon(np.vstack((updated, v)), self.pivots + (pivot,), self.p)

    def contains(self, rows: np.ndarray) -> np.ndarray:
        """Membership of augmented rows in the span, equivalent to all C tests."""
        values = np.asarray(rows, dtype=np.int64) % self.p
        free = [i for i in range(self.rows.shape[1]) if i not in self.pivots]
        if not self.rank:
            return np.all(values == 0, axis=1)
        residual = (values[:, free] - values[:, list(self.pivots)] @ self.rows[:, free]) % self.p
        return np.all(residual == 0, axis=1)

    def solution(self) -> tuple[int, ...]:
        dimension = self.rows.shape[1] - 2
        if self.rank != dimension + 1:
            raise ValueError("No unique affine coefficient vector")
        result = [0] * (dimension + 1)
        for i, pivot in enumerate(self.pivots):
            result[pivot] = int(self.rows[i, -1])
        return tuple(result)


def noise_free_matrix(extended: np.ndarray, form: Echelon, budget: ResourceBudget) -> bool:
    """Algorithm 2, including its rank-dependent 9/10 threshold exactly."""
    budget.check()
    budget.inclusion_tests += 1
    count = 0
    for start in range(0, len(extended), 512):
        budget.check()
        count += int(form.contains(extended[start:start + 512]).sum())
    n, rank = len(extended), form.rank
    if n <= rank:
        raise InterruptedFit("insufficient_validation_rows")
    dimension = extended.shape[1] - 2
    # Avoid floating-point rounding at a strict inequality boundary.
    return 10 * (count - rank) * pow(form.p, dimension + 1 - rank) > 9 * (n - rank)


def floor_log(value: int, base: int) -> int:
    exponent = 0
    while value >= base:
        value //= base
        exponent += 1
    return exponent


def linear_regression_modulo(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    p: int,
    rep: int,
    rng: random.Random,
    budget: ResourceBudget,
) -> tuple[int, ...]:
    """Algorithms 4--6, without candidate-best selection or forced acceptance."""
    dimension = features.shape[1]
    extended = np.column_stack((features % p, np.ones(len(features), dtype=np.int64), targets % p))
    threshold = max(1, min(len(features) - 1, dimension + 1 - floor_log(len(features), p)))
    while True:  # Algorithm 6's outer restart loop, externally interruptible.
        budget.check()
        budget.restarts += 1
        form = Echelon.empty(dimension, p)
        while form.rank < threshold:  # Algorithm 4.
            index = budget.draw(rng, len(features))
            solvable, candidate = form.add(extended[index])
            if not solvable:
                break
            form = candidate
        if form.rank < threshold or not noise_free_matrix(extended, form, budget):
            continue
        failed = 0
        while form.rank < dimension + 1 and failed == 0:  # Algorithm 5.
            while failed < rep:
                index = budget.draw(rng, len(features))
                solvable, candidate = form.add(extended[index])
                if (not solvable or candidate.rank == form.rank
                        or not noise_free_matrix(extended, candidate, budget)):
                    failed += 1
                    continue
                failed = 0
                form = candidate
                break
        if form.rank == dimension + 1:
            return form.solution()


@dataclass(frozen=True)
class PublishedMiharaFit:
    coefficients: tuple[int, ...]  # Intercept last; partial on interrupted runs.
    completed_digits: int
    requested_digits: int
    status: str
    draws: int
    restarts: int
    inclusion_tests: int
    active_rows_by_digit: tuple[int, ...]
    elapsed_seconds: float

    @property
    def completed(self) -> bool:
        return self.status == "precision_complete"


def fit_published_mihara(
    features: Sequence[Sequence[int]] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    *,
    p: int,
    precision: int,
    rep: int = 3,
    seed: int = 42,
    budget: ResourceBudget | None = None,
) -> PublishedMiharaFit:
    """Algorithms 7--8. Higher input digits are retained in the recursion."""
    validate_prime(p)
    if precision < 1 or rep < 1:
        raise ValueError("precision and rep must be positive")
    x = np.asarray(features, dtype=np.int64)
    y = np.asarray(targets, dtype=np.int64)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x):
        raise ValueError("Expected aligned nonempty feature matrix and targets")
    budget = budget or ResourceBudget()
    rng = random.Random(seed)
    modulus = p**precision
    if modulus > 2**52:
        raise ValueError("Requested precision is outside this implementation's range")
    # Object dot products prevent int64 overflow in higher-digit residuals.
    exact_x = np.column_stack((x, np.ones(len(x), dtype=np.int64))).astype(object)
    coefficients = np.zeros(x.shape[1] + 1, dtype=object)
    active = np.arange(len(x))
    counts: list[int] = []
    completed_digits = 0
    status = "precision_complete"
    try:
        for exponent in range(precision):
            budget.check()
            counts.append(len(active))
            if not len(active):
                raise InterruptedFit("empty_active_set")
            residual = y[active].astype(object) - exact_x[active] @ coefficients
            divisor = p**exponent
            if any(int(value) % divisor for value in residual):
                raise AssertionError("Digit recursion lost its congruence invariant")
            digits = np.asarray([(int(value) // divisor) % p for value in residual], dtype=np.int64)
            theta = linear_regression_modulo(x[active], digits, p=p, rep=rep, rng=rng, budget=budget)
            coefficients += divisor * np.asarray(theta, dtype=object)
            completed_digits += 1
            residual_after = y[active].astype(object) - exact_x[active] @ coefficients
            active = active[np.asarray([int(value) % (divisor * p) == 0 for value in residual_after])]
    except InterruptedFit as exc:
        status = str(exc)
    return PublishedMiharaFit(
        tuple(int(v) for v in coefficients), completed_digits, precision, status,
        budget.draws, budget.restarts, budget.inclusion_tests, tuple(counts),
        budget.clock() - budget.started,
    )


def predict_published_mihara(fit: PublishedMiharaFit, features: np.ndarray, *, p: int) -> list[int]:
    if not fit.completed:
        raise ValueError("An interrupted fit is not a completed prediction model")
    x = np.asarray(features, dtype=object)
    coefficients = np.asarray(fit.coefficients, dtype=object)
    values = x @ coefficients[:-1] + coefficients[-1]
    return [int(value) % (p**fit.requested_digits) for value in values]
