"""Bounded binary-affine implementation of Salazar et al. (2609.25501v1).

Equations 7, 9, 10 and Appendix F supply exact one-sided slopes. Sections
4.1/4.2 and G.1 supply grouped Momentum/Adam. This is an independent adapter,
not the unreleased library optimizer: conservative tie groups are not split
on cancellation, and states are restricted to Z_p disks with seven-digit (or
user-selected) capacity. Centers are canonical finite residues. Radii use
float64 logarithms, snapped exactly to integer logs at clipped vertices.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .randomised_linear import binary_design, check_precision, loss_units


def vertex_depth(log_radius: float) -> int | None:
    # Only a clipped vertex is stored as an integer. Do not turn a nearby edge
    # into a vertex with a tolerance: that would change its tangent cone.
    return -int(log_radius) if float(log_radius).is_integer() else None


def forward(matrix, centers, log_radii, *, p, precision):
    q = check_precision(p, precision, max(matrix.shape))
    prediction = np.asarray(matrix @ centers).reshape(-1) % q
    radii = np.zeros(matrix.shape[0], dtype=np.float64)
    for i in range(matrix.shape[0]):
        ids = matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]
        if len(ids):
            radii[i] = p ** float(np.max(log_radii[ids]))
    return prediction, radii


def direct_loss(matrix, targets, centers, log_radii, *, p, precision):
    predictions, radii = forward(matrix, centers, log_radii, p=p, precision=precision)
    errors = loss_units((targets - predictions) % p**precision, p, precision) / p**precision
    return float(np.mean(np.maximum(errors, radii) - radii / 2))


def directions(matrix, targets, centers, logs, column, *, p, precision,
               predictions=None, row_ids=None):
    """Analytic slope along up and each available downward tree germ.

    A strict active maximum moves down; a tied maximum cannot move down when
    its coordinate moves alone. Upward movement changes the max at ties too.
    Child recentering preserves higher digits of the retained representative.
    """
    q = p**precision
    r = p ** float(logs[column])
    depth = vertex_depth(float(logs[column]))
    children = p if depth is not None and depth < precision else 1
    slopes = np.zeros(1 + children, dtype=np.float64)
    pred = (np.asarray(matrix @ centers).reshape(-1) % q
            if predictions is None else predictions)
    common_children = 0.0
    if row_ids is None:
        row_ids = range(matrix.shape[0])
    for i in row_ids:
        ids = matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]
        if column not in ids:
            continue
        others = ids[ids != column]
        other_r = p ** float(np.max(logs[others])) if len(others) else 0.0
        delta = (int(targets[i])-int(pred[i])) % q
        if delta == 0:
            error = 0.0
        else:
            valuation, residue = 0, delta
            while residue % p == 0:
                valuation, residue = valuation+1, residue//p
            error = p**(-valuation)
        if r >= other_r:
            slopes[0] += 0.5 if error <= r else -0.5
        if r <= other_r:
            continue
        if depth is None or depth >= precision:
            slopes[1] += -0.5 if error < r else 0.5
        else:
            power = p**depth
            old_digit = int(centers[column]) // power % p
            # Appendix F.1 sparse accumulation: common slope + one exception.
            common_children += 0.5
            if delta % power == 0:
                digit = (delta // power + old_digit) % p
                slopes[1 + digit] -= 1.0
    if depth is not None and depth < precision:
        slopes[1:] += common_children
    return slopes / matrix.shape[0]


def groups(matrix, logs, mask):
    """Conservative connected components of attaining terms (Section 4.2)."""
    parent = np.arange(len(logs))
    def root(j):
        while parent[j] != j:
            parent[j] = parent[parent[j]]
            j = int(parent[j])
        return j
    for i in range(matrix.shape[0]):
        ids = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        if not len(ids):
            continue
        active = ids[logs[ids] == np.max(logs[ids])]
        for j in active[1:]:
            parent[root(int(j))] = root(int(active[0]))
    components = {}
    for j in np.flatnonzero(mask):
        components.setdefault(root(int(j)), []).append(int(j))
    return list(components.values())


def child_centers(center, depth, p):
    power = p**depth
    old_digit = int(center) // power % p
    return int(center) + (np.arange(p, dtype=np.int64) - old_digit) * power


def limits(log_r, *, p, precision):
    """Distance and destination log for up/down, restricted to [p^-E, 1]."""
    r = p**float(log_r)
    at_vertex = vertex_depth(float(log_r)) is not None
    up = int(log_r) + 1 if at_vertex else math.ceil(log_r)
    down = int(log_r) - 1 if at_vertex else math.floor(log_r)
    return ((max(0.0, p**up-r), up) if log_r < 0 else (0.0, 0),
            (max(0.0, r-p**down), down) if log_r > -precision else (0.0, -precision))


def vertex_average(old, values, clock, beta):
    """G.1 weighted observations at this vertex, decayed by coordinate clock."""
    if old is None:
        numerator, weight = np.asarray(values, dtype=float).copy(), 1.0
    else:
        numerator, weight, previous = old
        decay = beta ** (clock-previous)
        numerator, weight = decay*numerator + values, decay*weight + 1
    state = (numerator, weight, clock)
    return (1-beta**clock)*numerator/weight, state


@dataclass
class TreeFit:
    centers: np.ndarray
    log_radii: np.ndarray
    updates: int
    coordinate_updates: int
    moves: int
    digit_changes: int
    elapsed_seconds: float
    status: str
    history: list[dict]


def fit_tree(features, targets, initial, *, p=71, precision=7, seed=42,
             optimizer="adam", initial_log_radius=None, updates=1000,
             seconds=120, batch_size=32, kappa=1.0, beta=0.9,
             beta2=0.999, epsilon=1e-8, checkpoint=None):
    if optimizer not in {"momentum", "adam", "gd"}:
        raise ValueError("Unknown optimizer")
    if updates < 1 or seconds <= 0 or batch_size < 1 or kappa <= 0:
        raise ValueError("Positive resource and optimizer settings required")
    matrix = binary_design(features).tocsr()
    q = check_precision(p, precision, max(matrix.shape))
    centers = np.asarray(initial, dtype=np.int64).copy()
    targets = np.asarray(targets, dtype=np.int64)
    if centers.shape != (matrix.shape[1],) or targets.shape != (matrix.shape[0],):
        raise ValueError("Aligned centers/targets required")
    if np.any(centers < 0) or np.any(centers >= q) or np.any(targets < 0) or np.any(targets >= q):
        raise ValueError("Canonical centers/targets required")
    mask = np.asarray(matrix.sum(axis=0)).reshape(-1) > 0
    if np.any(centers[~mask]):
        raise ValueError("Unsupported features must remain zero")
    initial_log_radius = -precision if initial_log_radius is None else initial_log_radius
    if not -precision <= initial_log_radius <= 0:
        raise ValueError("Initial radius outside capacity")
    logs = np.where(mask, float(initial_log_radius), -np.inf)
    rng_batch, rng_queue = np.random.default_rng(seed), np.random.default_rng(seed+1000000)
    upward, downward, second = (np.zeros(len(centers)) for _ in range(3))
    clocks = np.zeros(len(centers), dtype=np.int64)
    awaiting = mask.copy()
    visits = {}
    history = []
    moves = digit_changes = coordinate_updates = 0
    alpha = kappa*(1-1/p)  # Paper G.3 regression reference radius 1.
    started = time.monotonic()
    status = "update_limit"
    completed = 0
    for step in range(1, updates+1):
        if time.monotonic()-started >= seconds:
            status = "time_limit"
            break
        ids = rng_batch.choice(len(targets), size=min(batch_size, len(targets)), replace=False)
        batch, y = matrix[ids], targets[ids]
        components = groups(batch, logs, mask)
        selected = []
        for group in components:
            available = [j for j in group if awaiting[j]]
            if not available:
                awaiting[group] = True
                available = group
            selected.append(available[0] if len(available) == 1 else int(rng_queue.choice(available)))
        proposals = []
        row_map = {}
        for i in range(batch.shape[0]):
            for j in batch.indices[batch.indptr[i]:batch.indptr[i+1]]:
                row_map.setdefault(int(j), []).append(i)
        pred = np.asarray(batch @ centers).reshape(-1) % q
        for j in selected:
            depth = vertex_depth(float(logs[j]))
            # Zero-feature minibatch derivatives are exactly zero. Stored
            # momentum is still updated and may move the coordinate.
            if j in row_map:
                slopes = directions(batch, y, centers, logs, j, p=p, precision=precision,
                                    predictions=pred, row_ids=row_map[j])
            else:
                slopes = np.zeros(1+(p if depth is not None and depth < precision else 1))
            up, down = limits(float(logs[j]), p=p, precision=precision)
            # The finite quotient does not specify derivatives below its
            # terminal disk. Use zero for that unavailable direction when
            # updating histories; this capacity convention is an explicit
            # difference from an optimizer with unbounded exact centers.
            if down[0] == 0:
                slopes[1:] = 0
            clocks[j] += 1
            clock = int(clocks[j])
            coordinate_updates += 1
            awaiting[j] = False
            if optimizer == "gd":
                moment = slopes.copy()
            else:
                upward[j] = beta*upward[j] + (1-beta)*slopes[0]
                down_min = float(np.min(slopes[1:]))
                downward[j] = beta*downward[j] + (1-beta)*down_min
                moment = np.array([upward[j], downward[j]])
                if len(slopes) > 2:
                    key = (j, depth, int(centers[j]) % p**depth)
                    # Identifies the geometric vertex, independent of the
                    # representative's discarded digits.
                    observation = np.stack([slopes[1:]-down_min, slopes[1:]])
                    averaged, stored = vertex_average(visits.get(key), observation, clock, beta)
                    visits[key] = stored
                    moment = np.r_[upward[j], downward[j] + averaged[0]]
                    moment[1:][averaged[1] >= 0] = np.inf
            speeds = np.maximum(0.0, -moment)
            if optimizer == "adam":
                feasible_slopes = slopes.copy()
                if up[0] == 0:
                    feasible_slopes[0] = 0
                minimum = float(np.min(feasible_slopes))
                second[j] = beta2*second[j] + (1-beta2)*max(0.0, -minimum)**2
                speeds /= (1-beta**clock)*(math.sqrt(second[j]/(1-beta2**clock))+epsilon)
            if up[0] == 0:
                speeds[0] = 0
            if down[0] == 0:
                speeds[1:] = 0
            speed = float(np.max(speeds))
            if speed <= 0:
                continue
            choices = np.flatnonzero(speeds == speed)
            direction = int(choices[0] if len(choices)==1 else rng_queue.choice(choices))
            boundary = up if direction == 0 else down
            center = int(centers[j])
            if direction > 0 and len(slopes) > 2:
                center = int(child_centers(center, depth, p)[direction-1]) % q
            proposals.append((j, direction, speed, boundary, center))
        # Shared clipping time (Eq 12): stop EVERY selected coordinate at
        # the first vertex encountered, rather than clipping independently.
        dt = min([alpha]+[boundary[0]/speed for _,_,speed,boundary,_ in proposals])
        for j, direction, speed, boundary, center in proposals:
            distance = dt*speed
            if dt == boundary[0]/speed:
                new_log = float(boundary[1])
            else:
                r = p**float(logs[j]) + (distance if direction == 0 else -distance)
                new_log = math.log(r, p)
            digit_changes += center != centers[j]
            centers[j], logs[j] = center, new_log
            moves += 1
        completed = step
        if step == 1 or step % 100 == 0 or step == updates:
            item = dict(update=step, groups=len(components), selected=len(selected),
                        moves=moves, digit_changes=digit_changes,
                        coordinate_updates=coordinate_updates,
                        elapsed_seconds=time.monotonic()-started,
                        training_direct_loss=direct_loss(matrix, targets, centers, logs, p=p, precision=precision),
                        training_point_loss=float(loss_units((targets-np.asarray(matrix @ centers).reshape(-1)) % q, p, precision).sum())/(len(targets)*q),
                        maximum_log_radius=float(np.max(logs)),
                        minimum_log_radius=float(np.min(logs[mask])))
            history.append(item)
            if checkpoint:
                checkpoint(item, centers.copy(), logs.copy())
    return TreeFit(centers, logs, completed, coordinate_updates, moves,
                   digit_changes, time.monotonic()-started, status, history)
