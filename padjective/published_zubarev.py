"""Finite-precision Zubarev polynomial model and equation (17) transition.

Source: arXiv:2503.23488v2. The input is the digit-interleaving homeomorphism,
not a fitted linear tag score. Transitions use exact rejection sampling of the
finite-quotient Gibbs mass, not a single local Metropolis proposal.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from .published_mihara import floor_log, validate_prime


def interleave_residue(values: Sequence[int], *, p: int, digits: int) -> int:
    """Equations (5)--(6), reduced to the requested number of output digits."""
    if not len(values) or digits < 0:
        raise ValueError("A nonempty input and nonnegative digit count are required")
    result, place = 0, 1
    dimension = len(values)
    for position in range(digits):
        coordinate, input_digit = position % dimension, position // dimension
        digit = (int(values[coordinate]) // p**input_digit) % p
        result += digit * place
        place *= p
    return result


def binomial_basis_residues(value: int, degree: int, *, p: int, precision: int) -> np.ndarray:
    """Compute binom(value,k) modulo p^E, retaining units even across zero residues."""
    if value < 0 or degree < 0 or precision < 1:
        raise ValueError("Nonnegative value/degree and positive precision required")
    modulus = p**precision
    result = np.zeros(degree + 1, dtype=np.int64)
    result[0] = 1
    unit, valuation = 1, 0
    for k in range(1, min(degree, value) + 1):
        numerator, denominator = value - k + 1, k
        while numerator % p == 0:
            numerator //= p
            valuation += 1
        while denominator % p == 0:
            denominator //= p
            valuation -= 1
        unit = unit * (numerator % modulus) * pow(denominator, -1, modulus) % modulus
        if valuation < precision:
            result[k] = unit * p**valuation % modulus
    return result


def modular_matvec(matrix: np.ndarray, coefficients: np.ndarray, modulus: int) -> np.ndarray:
    """Exact modular dot product without overflowing int64 at p^7 precision."""
    a = np.asarray(matrix, dtype=np.int64)
    weights = np.asarray(coefficients, dtype=np.int64) % modulus
    if a.ndim != 2 or weights.shape != (a.shape[1],):
        raise ValueError("Matrix and coefficient dimensions differ")
    if not 1 < modulus <= 2**52:
        raise ValueError("modulus must be in (1, 2^52]")
    radix = min(4096, (2**62) // modulus)
    # Limit each block's sum rather than reducing the radix for a wide basis.
    block = max(1, min(a.shape[1], (2**62) // (modulus * (radix - 1))))
    digits = []
    while np.any(weights):
        digits.append(weights % radix)
        weights = weights // radix
    result = np.zeros(a.shape[0], dtype=np.int64)
    for digit in reversed(digits):
        subtotal = np.zeros_like(result)
        for start in range(0, a.shape[1], block):
            subtotal = (subtotal + (a[:, start:start + block] @ digit[start:start + block]) % modulus) % modulus
        result = (result * radix + subtotal) % modulus
    return result


def padic_losses(actual: Sequence[int], predicted: Sequence[int], p: int) -> np.ndarray:
    differences = np.abs(np.asarray(actual, dtype=np.int64) - np.asarray(predicted, dtype=np.int64))
    losses = np.ones(len(differences), dtype=np.float64)
    zero = differences == 0
    losses[zero] = 0
    active = (~zero) & (differences % p == 0)
    while np.any(active):
        losses[active] /= p
        differences[active] //= p
        active = active & (differences % p == 0)
    return losses


@dataclass
class MahlerDesign:
    p: int
    precision: int
    degree: int
    input_dimension: int
    input_digits: int
    basis: np.ndarray  # One row per distinct relevant interleaved input.
    row_groups: np.ndarray
    interleaved_values: tuple[int, ...]
    root_columns: np.ndarray

    @classmethod
    def build(cls, features: Sequence[Sequence[int]] | np.ndarray, *, p: int,
              precision: int, degree: int) -> MahlerDesign:
        validate_prime(p)
        x = np.asarray(features)
        if x.ndim != 2 or not x.shape[1] or not len(x) or degree < 0 or precision < 1:
            raise ValueError("Invalid design, degree or precision")
        if p**precision > 2**52:
            raise ValueError("Requested precision exceeds the exact int64 implementation")
        needed = 0 if degree == 0 else precision + floor_log(degree, p)
        groups: dict[int, int] = {}
        inverse: list[int] = []
        for row in x:
            value = interleave_residue(row, p=p, digits=needed)
            if value not in groups:
                groups[value] = len(groups)
            inverse.append(groups[value])
        basis = np.vstack([binomial_basis_residues(h, degree, p=p, precision=precision) for h in groups])
        root_columns = np.flatnonzero(np.any(basis % p, axis=0))
        return cls(p, precision, degree, x.shape[1], needed, basis,
                   np.asarray(inverse, dtype=np.int64), tuple(groups), root_columns)

    @property
    def modulus(self) -> int:
        return self.p**self.precision

    def predict(self, coefficients: np.ndarray) -> np.ndarray:
        return modular_matvec(self.basis, coefficients, self.modulus)[self.row_groups]

    def predict_new(self, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        if np.asarray(features).shape[1] != self.input_dimension:
            raise ValueError("Prediction input dimension differs from fitting design")
        other = MahlerDesign.build(features, p=self.p, precision=self.precision, degree=self.degree)
        return other.predict(coefficients)


@dataclass
class GibbsProblem:
    design: MahlerDesign
    targets: np.ndarray
    root_basis: np.ndarray
    root_histogram: np.ndarray
    root_lower_bound: float
    root_right_inverse: np.ndarray | None

    @classmethod
    def create(cls, design: MahlerDesign, targets: Sequence[int]) -> GibbsProblem:
        y = np.asarray(targets, dtype=np.int64)
        if y.shape != design.row_groups.shape or np.any(y < 0) or np.any(y >= design.modulus):
            raise ValueError("Targets must be aligned canonical residues at the chosen precision")
        roots = design.basis[:, design.root_columns] % design.p
        unique, inverse = np.unique(roots, axis=0, return_inverse=True)
        hist = np.zeros((len(unique), design.p), dtype=np.int64)
        for group, target in zip(inverse[design.row_groups], y, strict=True):
            hist[int(group), int(target % design.p)] += 1
        # Each root-signature group must share a predicted first digit. Letting
        # groups choose independently relaxes polynomial constraints, hence LB.
        lower = 1.0 - float(hist.max(axis=1).sum()) / len(y)
        lower = max(0.0, math.nextafter(lower, -math.inf))
        # A surjective evaluation map makes root predictions independent under
        # Haar measure. Retain its right inverse to sample their Gibbs marginal
        # directly, then reject only on the remaining (at most 1/p) loss.
        reduced = unique.copy()
        transform = np.eye(len(unique), dtype=np.int64)
        pivots = []
        for column in range(reduced.shape[1]):
            rank = len(pivots)
            candidates = np.flatnonzero(reduced[rank:, column])
            if not len(candidates):
                continue
            pivot = rank + int(candidates[0])
            reduced[[rank, pivot]] = reduced[[pivot, rank]]
            transform[[rank, pivot]] = transform[[pivot, rank]]
            inverse = pow(int(reduced[rank, column]), -1, design.p)
            reduced[rank] = reduced[rank] * inverse % design.p
            transform[rank] = transform[rank] * inverse % design.p
            factors = reduced[:, column].copy()
            factors[rank] = 0
            reduced = (reduced - factors[:, None] * reduced[rank]) % design.p
            transform = (transform - factors[:, None] * transform[rank]) % design.p
            pivots.append(column)
            if len(pivots) == len(unique):
                break
        right = None
        if len(pivots) == len(unique):
            right = np.zeros((unique.shape[1], len(unique)), dtype=np.int64)
            right[pivots] = transform
            assert np.array_equal(unique @ right % design.p, np.eye(len(unique), dtype=np.int64))
        return cls(design, y, unique, hist, lower, right)

    @property
    def sampler(self) -> str:
        return "root_conditioned_rejection" if self.root_right_inverse is not None else "haar_rejection"

    def root_loss(self, root_coefficients: np.ndarray) -> float:
        predicted = self.root_basis @ root_coefficients % self.design.p
        correct = self.root_histogram[np.arange(len(predicted)), predicted].sum()
        return 1.0 - float(correct) / len(self.targets)

    def loss(self, coefficients: np.ndarray) -> float:
        return float(padic_losses(self.targets, self.design.predict(coefficients), self.design.p).mean())


@dataclass(frozen=True)
class GibbsDraw:
    coefficients: np.ndarray | None
    loss: float | None
    proposals: int
    full_evaluations: int
    status: str


def draw_gibbs_transition(problem: GibbsProblem, *, beta: float, rng: np.random.Generator,
                         max_proposals: int, deadline: float = math.inf,
                         condition_roots: bool = True) -> GibbsDraw:
    """Equation (17) on (Z/p^E)^(K+1), using a proven rejection envelope.

    Root-digit rejection is lazy evaluation of the SAME uniform proposal and
    uniform acceptance variate. Unneeded coefficient digits are never drawn.
    The old state cancels from the normalized distribution by translation.
    """
    if beta < 0 or not math.isfinite(beta) or max_proposals < 1:
        raise ValueError("Finite nonnegative beta and positive proposal budget required")
    design = problem.design
    full_evaluations = 0
    conditioned = condition_roots and problem.root_right_inverse is not None
    if conditioned:
        logits = beta * (problem.root_histogram - problem.root_histogram.max(axis=1, keepdims=True)) / len(problem.targets)
        weights = np.exp(logits)
        cdf = np.cumsum(weights / weights.sum(axis=1, keepdims=True), axis=1)
        cdf[:, -1] = 1.0
    for proposal in range(max_proposals):
        if time.monotonic() >= deadline:
            return GibbsDraw(None, None, proposal, full_evaluations, "time_limit")
        root = rng.integers(0, design.p, size=len(design.root_columns), dtype=np.int64)
        if conditioned:
            desired = (rng.random(len(cdf))[:, None] >= cdf).sum(axis=1)
            correction = desired - problem.root_basis @ root
            root = (root + problem.root_right_inverse @ (correction % design.p)) % design.p
        uniform = max(float(rng.random()), np.finfo(float).tiny)
        root_loss = problem.root_loss(root)
        envelope = root_loss if conditioned else problem.root_lower_bound
        allowed_loss = math.inf if beta == 0 else envelope - math.log(uniform) / beta
        if root_loss > allowed_loss:
            continue
        coefficients = rng.integers(0, design.modulus, size=design.degree + 1, dtype=np.int64)
        coefficients[design.root_columns] = root + design.p * rng.integers(
            0, design.modulus // design.p, size=len(root), dtype=np.int64)
        loss = problem.loss(coefficients)
        full_evaluations += 1
        if loss <= allowed_loss:
            return GibbsDraw(coefficients, loss, proposal + 1, full_evaluations, "accepted")
    return GibbsDraw(None, None, max_proposals, full_evaluations, "proposal_limit")


@dataclass(frozen=True)
class PublishedZubarevFit:
    coefficients: np.ndarray
    initial_loss: float
    best_loss: float
    status: str
    accepted_transitions: int
    proposals: int
    full_evaluations: int
    history: tuple[dict, ...]
    elapsed_seconds: float
    initialisation: str
    seed: int


def fit_published_zubarev(problem: GibbsProblem, *, seed: int = 42,
                          initialisation: str = "zeros", betas: Sequence[float] = (0, 4, 16, 64, 256),
                          draws_per_beta: int = 64, proposals_per_beta: int = 200000,
                          seconds: float = 300.0) -> PublishedZubarevFit:
    if initialisation not in {"zeros", "random"}:
        raise ValueError("Only independent zeros/random starts are supported")
    if draws_per_beta < 1 or proposals_per_beta < 1 or seconds <= 0 or not betas:
        raise ValueError("Positive limits and a nonempty beta schedule are required")
    if any(beta < 0 or not math.isfinite(beta) for beta in betas):
        raise ValueError("Invalid beta schedule")
    started = time.monotonic()
    deadline = started + seconds
    rng = np.random.default_rng(seed)
    size, modulus = problem.design.degree + 1, problem.design.modulus
    best = (np.zeros(size, dtype=np.int64) if initialisation == "zeros"
            else rng.integers(0, modulus, size=size, dtype=np.int64))
    initial_loss = best_loss = problem.loss(best)
    history: list[dict] = []
    total_proposals = total_evaluations = total_accepted = 0
    final_status = "schedule_complete"
    for beta in betas:
        stage_proposals = stage_accepted = stage_evaluations = 0
        stage_status = "complete"
        while stage_accepted < draws_per_beta:
            remaining = proposals_per_beta - stage_proposals
            if remaining <= 0:
                stage_status = "proposal_limit"
                break
            draw = draw_gibbs_transition(problem, beta=beta, rng=rng,
                                         max_proposals=remaining, deadline=deadline)
            stage_proposals += draw.proposals
            stage_evaluations += draw.full_evaluations
            if draw.status != "accepted":
                stage_status = draw.status
                break
            stage_accepted += 1
            if draw.loss < best_loss:
                best, best_loss = draw.coefficients, float(draw.loss)
        total_proposals += stage_proposals
        total_evaluations += stage_evaluations
        total_accepted += stage_accepted
        history.append(dict(beta=float(beta), accepted=stage_accepted, proposals=stage_proposals,
                            full_evaluations=stage_evaluations, best_training_loss=best_loss,
                            status=stage_status, elapsed_seconds=time.monotonic() - started))
        if stage_status != "complete":
            final_status = stage_status
            # A failed draw is not a sampled transition. Do not silently skip
            # the difficult temperature and pretend the schedule completed.
            break
    return PublishedZubarevFit(best, initial_loss, best_loss, final_status, total_accepted,
                               total_proposals, total_evaluations, tuple(history),
                               time.monotonic() - started, initialisation, seed)
