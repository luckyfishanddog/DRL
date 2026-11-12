from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools
import math
import random
import time
from typing import Mapping, Sequence

import numpy as np
from numba import njit

from mrta_reference.geometry import (
    blocks_for_pattern,
    build_legal_pattern_catalog,
    generate_y_split_patterns,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    FormalScope,
    ParentWeld,
    Route,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.solution import block_map, canonicalize
from mrta_reference.scope import FORMAL_SCOPE_V2

from .common import (
    BaselineAccounting,
    BaselineResult,
    CandidateEvaluator,
    CommonBaselineEvaluator,
    EvaluatedCandidate,
    canonical_config_hash,
)


METHOD_ID = "ADAPTED_WAG_VNS_V1"
METHOD_ID_V2 = "ADAPTED_WAG_VNS_V2"
PAPER_CORE_ID = "WAG_PAPER_CORE_V1"


@dataclass(frozen=True)
class AdaptedWAGConfig:
    n: int = 10
    p: float = 0.50
    imax: int = 500
    factorial_window: int = 5
    factorial_mode: str = "PAPER_EXACT"
    max_windows: int = 8
    max_factorial_calls: int = 30720
    max_wag_variants: int | None = None
    max_route_combinations_per_wag: int | None = None

    def __post_init__(self) -> None:
        if min(
            self.n,
            self.imax,
            self.factorial_window,
            self.max_windows,
            self.max_factorial_calls,
        ) <= 0:
            raise ValueError("WAG integer parameters must be positive")
        if any(value is not None and value <= 0 for value in
               (self.max_wag_variants, self.max_route_combinations_per_wag)):
            raise ValueError("debug candidate bounds must be positive or None")
        if not 0.0 < self.p <= 1.0:
            raise ValueError("p must be in (0,1]")
        if self.factorial_mode not in {"PAPER_EXACT", "COMMON_MODEL_BOUNDED"}:
            raise ValueError("unknown factorial mode")



def _canonicalize_cached(parents, patterns, routes, config, *, scope=None):
    if isinstance(routes, Mapping):
        normalized_routes = tuple(Route(r, tuple(routes.get(r, ()))) for r in range(4))
    else:
        normalized_routes = tuple(routes)
    return _canonical_assignment(tuple(parents), tuple(patterns), normalized_routes, config, scope)


@lru_cache(maxsize=8192)
def _canonical_assignment(parents, patterns, routes, config, scope):
    # Same full canonicalizer and safeguards, once per identical input.
    return canonicalize(parents, patterns, routes, config, scope=scope)



def _initial_patterns(
    parents: Sequence[ParentWeld], config: ScientificConfig,
    *,
    scope: FormalScope | None = None,
) -> tuple[SplitPattern, ...]:
    catalog = (
        None if scope is None else build_legal_pattern_catalog(parents, config, scope)
    )
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        options = None if catalog is None else catalog[parent.parent_id]
        whole = next(
            (item for item in options or () if item.kind is SplitKind.WHOLE),
            SplitPattern(parent.parent_id, SplitKind.WHOLE),
        )
        if whole_eligible_rails(parent.start, parent.end, config):
            result.append(whole)
            continue
        splits = tuple(
            item for item in (options or generate_y_split_patterns(parent, config))
            if item.kind is SplitKind.Y_SPLIT
        )
        if not splits:
            raise ValueError(f"{parent.parent_id}: no legal WHOLE/Y_SPLIT pattern")
        result.append(splits[0])
    return tuple(result)


def _blocks_for(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
) -> dict[str, WeldingBlock]:
    by_parent = {parent.parent_id: parent for parent in parents}
    return {
        block.block_id: block
        for pattern in patterns
        for block in blocks_for_pattern(by_parent[pattern.parent_id], pattern, config)
    }


def _route_cost(
    route: Sequence[str], blocks: Mapping[str, WeldingBlock], config: ScientificConfig
) -> float:
    if not route:
        return 0.0
    states = {direction: config.process_time(blocks[route[0]].length) for direction in (0, 1)}
    for index in range(1, len(route)):
        old = blocks[route[index - 1]]
        current = blocks[route[index]]
        next_states = {}
        for direction in (0, 1):
            start = current.start if direction == 0 else current.end
            next_states[direction] = min(
                cost
                + math.dist(old.end if previous == 0 else old.start, start)
                / config.empty_speed
                + config.process_time(current.length)
                for previous, cost in states.items()
            )
        states = next_states
    return min(states.values())


def _home(robot: int, config: ScientificConfig) -> tuple[float, float]:
    x = config.workspace_x[0] if robot in (0, 2) else config.workspace_x[1]
    y = config.by[1] if robot < 2 else config.by[0]
    return (x, y)


@njit
def _array_route_cost(route, travel, processing):
    if len(route) == 0:
        return 0.0
    a = b = processing[route[0]]
    for k in range(1, len(route)):
        old, new = route[k - 1], route[k]
        a, b = (min(a + travel[old, new, 0, 0], b + travel[old, new, 1, 0]) + processing[new],
                min(a + travel[old, new, 0, 1], b + travel[old, new, 1, 1]) + processing[new])
    return min(a, b)


@njit
def _route_prefix_suffix(route, travel, processing):
    """Costs conditional on the boundary direction; both ends stay free."""
    n = len(route)
    prefix = np.zeros((n + 1, 2))
    suffix = np.zeros((n + 1, 2))
    for k in range(n):
        node = route[k]
        if k == 0:
            prefix[k + 1, :] = processing[node]
        else:
            old = route[k - 1]
            for d in range(2):
                prefix[k + 1, d] = processing[node] + min(
                    prefix[k, 0] + travel[old, node, 0, d],
                    prefix[k, 1] + travel[old, node, 1, d])
    for k in range(n - 1, -1, -1):
        node = route[k]
        for d in range(2):
            suffix[k, d] = processing[node]
            if k + 1 < n:
                new = route[k + 1]
                suffix[k, d] += min(travel[node, new, d, 0] + suffix[k + 1, 0],
                                    travel[node, new, d, 1] + suffix[k + 1, 1])
    return prefix, suffix


@njit
def _splice_cost(route, prefix, suffix, start, stop, inserted, travel, processing):
    """Exact O(len(inserted)) replacement, including the two boundary arcs."""
    a, b = prefix[start, 0], prefix[start, 1]
    old = route[start - 1] if start else -1
    for node in inserted:
        if old < 0:
            a = b = processing[node]
        else:
            a, b = (min(a + travel[old, node, 0, 0], b + travel[old, node, 1, 0]) + processing[node],
                    min(a + travel[old, node, 0, 1], b + travel[old, node, 1, 1]) + processing[node])
        old = node
    if stop == len(route):
        return min(a, b)
    if old < 0:
        return min(suffix[stop, 0], suffix[stop, 1])
    new = route[stop]
    return min(a + travel[old, new, 0, 0] + suffix[stop, 0],
               a + travel[old, new, 0, 1] + suffix[stop, 1],
               b + travel[old, new, 1, 0] + suffix[stop, 0],
               b + travel[old, new, 1, 1] + suffix[stop, 1])


@njit
def _interval_costs(route, travel, processing):
    """Min-plus 2x2 summaries for every unchanged contiguous segment."""
    n = len(route)
    result = np.full((n, n, 2, 2), np.inf)
    for i in range(n):
        result[i, i, 0, 0] = result[i, i, 1, 1] = processing[route[i]]
        for j in range(i + 1, n):
            old, new = route[j - 1], route[j]
            for first in range(2):
                for last in range(2):
                    result[i, j, first, last] = processing[new] + min(
                        result[i, j - 1, first, 0] + travel[old, new, 0, last],
                        result[i, j - 1, first, 1] + travel[old, new, 1, last])
    return result


@njit
def _join_segment(a, b, old, route, matrices, lo, hi, reverse, travel):
    if hi <= lo:
        return a, b, old
    first = route[hi - 1] if reverse else route[lo]
    if old < 0:
        a = b = 0.0
    else:
        a, b = (min(a + travel[old, first, 0, 0], b + travel[old, first, 1, 0]),
                min(a + travel[old, first, 0, 1], b + travel[old, first, 1, 1]))
    if reverse:
        # Reverse order AND complement endpoint directions. Euclidean empty
        # travel is symmetric, so this is the exact reversed segment matrix.
        c = min(a + matrices[lo, hi - 1, 1, 1], b + matrices[lo, hi - 1, 1, 0])
        d = min(a + matrices[lo, hi - 1, 0, 1], b + matrices[lo, hi - 1, 0, 0])
        return c, d, route[lo]
    c = min(a + matrices[lo, hi - 1, 0, 0], b + matrices[lo, hi - 1, 1, 0])
    d = min(a + matrices[lo, hi - 1, 0, 1], b + matrices[lo, hi - 1, 1, 1])
    return c, d, route[hi - 1]


@njit
def _two_opt_array(route, travel, processing, consider_final_edge):
    current = route.copy()
    cost = _array_route_cost(current, travel, processing)
    upper = len(current) if consider_final_edge else max(0, len(current) - 1)
    while True:
        summaries = _interval_costs(current, travel, processing)
        changed = False
        for left in range(max(0, upper - 1)):
            for right in range(left + 1, upper):
                a, b, old = _join_segment(0., 0., -1, current, summaries, 0, left, False, travel)
                a, b, old = _join_segment(a, b, old, current, summaries, left, right + 1, True, travel)
                a, b, old = _join_segment(a, b, old, current, summaries, right + 1, len(current), False, travel)
                value = min(a, b)
                if value < cost - 1.e-9:
                    current[left:right + 1] = current[left:right + 1][::-1]
                    cost, changed = value, True
                    break
            if changed:
                break
        if not changed:
            return current


@njit
def _factorial_array(route, travel, processing, permutations, size, final_edge,
                     exact, max_windows, max_calls):
    current = route.copy()
    n = len(current)
    if n < 2:
        return current, 0, 0, False, False
    last = n - size - (0 if final_edge else 1)
    count = max(0, last + 1)
    window_hit = not exact and count > max_windows
    if not exact:
        count = min(count, max_windows)
    calls = windows = 0
    best_perm = np.empty(size, np.int64)
    inserted = np.empty(size, np.int64)
    while True:
        changed = False
        for start in range(count):
            prefix, suffix = _route_prefix_suffix(current, travel, processing)
            best_cost = min(prefix[n, 0], prefix[n, 1])
            windows += 1
            found = False
            for permutation in permutations:
                if not exact and calls + 2 ** size > max_calls:
                    return current, calls, windows, window_hit, True
                calls += 2 ** size
                for k in range(size):
                    inserted[k] = current[start + permutation[k]]
                value = _splice_cost(current, prefix, suffix, start, start + size,
                                     inserted, travel, processing)
                if value < best_cost - 1.e-9:
                    best_cost, found = value, True
                    best_perm[:] = inserted
            if found:
                current[start:start + size] = best_perm
                changed = True
                break
        if not exact or not changed:
            return current, calls, windows, window_hit, False


@njit
def _farthest_array(ids, travel, processing, endpoint_x):
    if len(ids) == 0:
        return ids.copy()
    pending = list(ids)
    first = pending[0]
    for node in pending:
        if endpoint_x[node] > endpoint_x[first]:
            first = node
    # Resolve equal x with the lexicographic block-id index, like the reference.
    for node in pending:
        if (endpoint_x[node] == endpoint_x[first] and node > first):
            first = node
    current = np.array([first], np.int64)
    pending.remove(first)
    one = np.empty(1, np.int64)
    while pending:
        prefix, suffix = _route_prefix_suffix(current, travel, processing)
        before = min(prefix[len(current), 0], prefix[len(current), 1])
        far_delta, selected, position = -np.inf, -1, 0
        for node in pending:
            one[0] = node
            delta, best_position = np.inf, 0
            for k in range(len(current) + 1):
                value = _splice_cost(current, prefix, suffix, k, k, one, travel, processing) - before
                if value < delta:
                    delta, best_position = value, k
            if delta > far_delta or (delta == far_delta and node > selected):
                far_delta, selected, position = delta, node, best_position
        current = np.concatenate((current[:position], np.array([selected]), current[position:]))
        pending.remove(selected)
    return current


@lru_cache(maxsize=32)
def _route_tables(blocks: tuple[WeldingBlock, ...], config: ScientificConfig):
    ids = tuple(block.block_id for block in blocks)
    points = np.array([(block.start, block.end) for block in blocks], dtype=float).reshape((-1, 2, 2))
    processing = np.array([config.process_time(block.length) for block in blocks])
    travel = np.empty((len(blocks), len(blocks), 2, 2))
    for d in range(2):
        for e in range(2):
            delta = points[:, 1 - d, None, :] - points[None, :, e, :]
            travel[:, :, d, e] = np.sqrt(np.sum(delta * delta, axis=-1)) / config.empty_speed
    return ids, {name: i for i, name in enumerate(ids)}, points, travel, processing


def _tables_for(blocks, config):
    return _route_tables(tuple(blocks[k] for k in sorted(blocks)), config)


@lru_cache(maxsize=8)
def _permutations(size):
    return np.array(tuple(itertools.permutations(range(size))), dtype=np.int64)


@njit
def _direction_array(route, travel, processing, first):
    n = len(route)
    bits = np.zeros(n, np.int64)
    if not n:
        return 0., bits
    back = np.zeros((n, 2), np.int64)
    value0 = processing[route[0]] if first == 0 else np.inf
    value1 = processing[route[0]] if first == 1 else np.inf
    rank0, rank1 = 0, 1
    for k in range(1, n):
        old, new = route[k - 1], route[k]
        a, b = value0 + travel[old, new, 0, 0], value1 + travel[old, new, 1, 0]
        previous0 = 0 if a < b or (a == b and rank0 <= rank1) else 1
        next0 = min(a, b) + processing[new]
        a, b = value0 + travel[old, new, 0, 1], value1 + travel[old, new, 1, 1]
        previous1 = 0 if a < b or (a == b and rank0 <= rank1) else 1
        next1 = min(a, b) + processing[new]
        back[k, 0], back[k, 1] = previous0, previous1
        if (rank0 if previous0 == 0 else rank1) <= (rank0 if previous1 == 0 else rank1):
            rank0, rank1 = 0, 1
        else:
            rank0, rank1 = 1, 0
        value0, value1 = next0, next1
    last = 0 if value0 < value1 or (value0 == value1 and rank0 < rank1) else 1
    value = value0 if last == 0 else value1
    bits[n - 1] = last
    for k in range(n - 1, 0, -1):
        bits[k - 1] = back[k, bits[k]]
    return value, bits


@njit
def _numeric_templates(route, bits, points, processing, empty_speed, pre, post):
    # duration,x0,y0,x1,y1,start,end,node,kind (MOVE=0,SETUP=1,WELD=2,POST=3,WAIT=4)
    ops = np.zeros((4 * len(route), 9))
    count, cursor = 0, 0.
    for k in range(len(route)):
        node, direction = route[k], bits[k]
        start, end = points[node, direction], points[node, 1 - direction]
        if k:
            old = points[route[k - 1], 1 - bits[k - 1]]
            duration = math.sqrt((old[0] - start[0]) ** 2 + (old[1] - start[1]) ** 2) / empty_speed
            ops[count, 0], ops[count, 1], ops[count, 2] = duration, old[0], old[1]
            ops[count, 3], ops[count, 4] = start[0], start[1]
            ops[count, 5], ops[count, 6] = cursor, cursor + duration
            ops[count, 7], ops[count, 8] = node, 0.
            count += 1
            cursor += duration
        for kind in range(1, 4):
            duration = pre if kind == 1 else post if kind == 3 else processing[node] - pre - post
            a = end if kind == 3 else start
            b = start if kind == 1 else end
            ops[count, 0], ops[count, 1], ops[count, 2] = duration, a[0], a[1]
            ops[count, 3], ops[count, 4] = b[0], b[1]
            ops[count, 5], ops[count, 6] = cursor, cursor + duration
            ops[count, 7], ops[count, 8] = node, float(kind)
            count += 1
            cursor += duration
    return ops[:count]


@njit
def _numeric_conflict(a, b, ra, rb, dx, dy, eps):
    lo, hi = max(a[5], b[5]), min(a[6], b[6])
    if hi < lo:
        return False
    near_lo, near_hi = lo, hi
    for axis in range(2):
        va = (a[axis + 3] - a[axis + 1]) / a[0] if a[0] else 0.
        vb = (b[axis + 3] - b[axis + 1]) / b[0] if b[0] else 0.
        slope = va - vb
        intercept = a[axis + 1] - va * a[5] - b[axis + 1] + vb * b[5]
        if axis == 0 and ra // 2 == rb // 2:
            sign = 1. if ra % 2 == 0 else -1.
            if max(sign * (slope * lo + intercept), sign * (slope * hi + intercept)) + dx > eps:
                return True
        limit = (dx if axis == 0 else dy) + eps
        if abs(slope) <= 1.e-15:
            if abs(intercept) > limit:
                near_hi = -np.inf
        else:
            t0, t1 = (-limit - intercept) / slope, (limit - intercept) / slope
            near_lo, near_hi = max(near_lo, min(t0, t1)), min(near_hi, max(t0, t1))
    return near_lo <= near_hi


@njit
def _numeric_forbidden(a, b, ra, rb, ready, dx, dy, eps, order):
    constraints = np.empty((10, 3))
    return _numeric_forbidden_buffer(a, b, ra, rb, ready, dx, dy, eps, order, constraints)


@njit(inline='always')
def _temporal_vertex_valid(s, t, duration, begin, end, bound, geometry, eps):
    if (s - t > eps or t - s > duration + eps or -t > -begin + eps
            or t > end + eps or -s > -bound + eps or s > end + eps):
        return False
    for aa, bb, cc in geometry:
        if aa * s + bb * t > cc + eps:
            return False
    return True


@njit
def _fixed_temporal_projection(duration, begin, end, bound, geometry, eps):
    # The six unchanged temporal half-planes have twelve nonparallel pairs.
    # Three yield (end, end); enumerate their ten distinct vertices directly.
    low, high = np.inf, -np.inf
    for s, t in ((begin, begin), (end, end), (bound, bound), (begin - duration, begin),
                 (end - duration, end), (bound, bound + duration), (end, end + duration),
                 (bound, begin), (end, begin), (bound, end)):
        if _temporal_vertex_valid(s, t, duration, begin, end, bound, geometry, eps):
            low, high = min(low, s), max(high, s)
    temporal = ((1., -1., 0.), (-1., 1., duration), (0., -1., -begin),
                (0., 1., end), (-1., 0., -bound), (1., 0., end))
    for x, y, z in temporal:
        for aa, bb, cc in geometry:
            determinant = x * bb - aa * y
            if abs(determinant) <= 1.e-15:
                continue
            s = (z * bb - cc * y) / determinant
            t = (x * cc - aa * z) / determinant
            if _temporal_vertex_valid(s, t, duration, begin, end, bound, geometry, eps):
                low, high = min(low, s), max(high, s)
    for i in range(len(geometry)):
        x, y, z = geometry[i]
        for j in range(i + 1, len(geometry)):
            aa, bb, cc = geometry[j]
            determinant = x * bb - aa * y
            if abs(determinant) <= 1.e-15:
                continue
            s = (z * bb - cc * y) / determinant
            t = (x * cc - aa * z) / determinant
            if _temporal_vertex_valid(s, t, duration, begin, end, bound, geometry, eps):
                low, high = min(low, s), max(high, s)
    return low, high


@njit
def _numeric_forbidden_specialized(a, b, ra, ready, dx, dy, eps, order):
    duration, begin, end = a[0], b[5], b[6]
    bound = max(ready, begin - duration)
    vc = (a[3] - a[1]) / duration if duration else 0.
    vf = (b[3] - b[1]) / b[0] if b[0] else 0.
    constant = a[1] - b[1] + vf * begin
    if order:
        sign = 1. if ra % 2 == 0 else -1.
        return _fixed_temporal_projection(duration, begin, end, bound,
            ((sign * vc, sign * (vf - vc), sign * constant + dx - eps),), eps)
    vy = (a[4] - a[2]) / duration if duration else 0.
    wy = (b[4] - b[2]) / b[0] if b[0] else 0.
    cy = a[2] - b[2] + wy * begin
    lx, ly = dx + eps, dy + eps
    geometry = ((-vc, vc - vf, lx - constant), (vc, vf - vc, lx + constant),
                (-vy, vy - wy, ly - cy), (vy, wy - vy, ly + cy))
    return _fixed_temporal_projection(duration, begin, end, bound, geometry, eps)


@njit
def _numeric_forbidden_buffer(a, b, ra, rb, ready, dx, dy, eps, order, constraints, specialized=False):
    if specialized:
        return _numeric_forbidden_specialized(a, b, ra, ready, dx, dy, eps, order)
    duration = a[0]
    constraints[0] = (1., -1., 0.)
    constraints[1] = (-1., 1., duration)
    constraints[2] = (0., -1., -b[5])
    constraints[3] = (0., 1., b[6])
    constraints[4] = (-1., 0., -max(ready, b[5] - duration))
    constraints[5] = (1., 0., b[6])
    count = 6
    for axis in range(1 if order else 2):
        vc = (a[axis + 3] - a[axis + 1]) / duration if duration else 0.
        vf = (b[axis + 3] - b[axis + 1]) / b[0] if b[0] else 0.
        constant = a[axis + 1] - b[axis + 1] + vf * b[5]
        if order:
            sign = 1. if ra % 2 == 0 else -1.
            constraints[count] = (sign * vc, sign * (vf - vc), sign * constant + dx - eps)
            count += 1
        else:
            limit = (dx if axis == 0 else dy) + eps
            constraints[count] = (-vc, vc - vf, limit - constant)
            constraints[count + 1] = (vc, vf - vc, limit + constant)
            count += 2
    low, high = np.inf, -np.inf
    for i in range(count):
        for j in range(i + 1, count):
            determinant = constraints[i, 0] * constraints[j, 1] - constraints[j, 0] * constraints[i, 1]
            if abs(determinant) <= 1.e-15:
                continue
            s = (constraints[i, 2] * constraints[j, 1] - constraints[j, 2] * constraints[i, 1]) / determinant
            t = (constraints[i, 0] * constraints[j, 2] - constraints[j, 0] * constraints[i, 2]) / determinant
            valid = True
            for k in range(count):
                if constraints[k, 0] * s + constraints[k, 1] * t > constraints[k, 2] + eps:
                    valid = False
                    break
            if valid:
                low, high = min(low, s), max(high, s)
    return low, high


@njit
def _forbidden_may_contain(a, b, ra, ready, start, dx, dy, eps, order):
    """Conservative section of the original EPS-relaxed projection polygon.

    Every vertex admitted by _numeric_forbidden satisfies those half-planes
    with EPS slack. Their convex hull contains its entire returned start-time
    interval. If this looser section is empty, that interval cannot contain
    start. Near boundaries the original vertex enumeration remains decisive.
    """
    duration = a[0]
    slack = 32. * eps + 1.e-10 * max(1., abs(start), abs(ready), abs(b[5]), abs(b[6]))
    if start < max(ready, b[5] - duration) - slack or start > b[6] + slack:
        return False
    lo, hi = max(start, b[5]) - slack, min(start + duration, b[6]) + slack
    if lo > hi:
        return False
    for axis in range(1 if order else 2):
        vc = (a[axis + 3] - a[axis + 1]) / duration if duration else 0.
        vf = (b[axis + 3] - b[axis + 1]) / b[0] if b[0] else 0.
        constant = a[axis + 1] - b[axis + 1] + vf * b[5]
        if order:
            sign = 1. if ra % 2 == 0 else -1.
            coefficient = sign * (vf - vc)
            rhs = sign * constant + dx - eps - sign * vc * start
            margin = slack * (1. + abs(vc) + abs(vf))
            if coefficient > 0.:
                hi = min(hi, (rhs + margin) / coefficient)
            elif coefficient < 0.:
                lo = max(lo, (rhs + margin) / coefficient)
            elif rhs + margin < 0.:
                return False
        else:
            limit = (dx if axis == 0 else dy) + eps
            for side in (-1., 1.):
                coefficient = side * (vc - vf)
                rhs = limit - side * constant + side * vc * start
                margin = slack * (1. + abs(vc) + abs(vf))
                if coefficient > 0.:
                    hi = min(hi, (rhs + margin) / coefficient)
                elif coefficient < 0.:
                    lo = max(lo, (rhs + margin) / coefficient)
                elif rhs + margin < 0.:
                    return False
        if lo > hi:
            return False
    return True


@njit
def _follow_reusable_prefix(fixed, following, old_fixed, old_following, old_output, eps):
    """A prefix strictly before either fixed timeline changes is identical.

    The saved output must come from this same fixed/following priority. Commit
    only whole following operations, including their complete preceding WAIT.
    No inference from the winning priority or from absence of WAIT is used.
    """
    k = min(len(fixed), len(old_fixed)) if fixed.ctypes.data == old_fixed.ctypes.data else 0
    while k < min(len(fixed), len(old_fixed)):
        equal = True
        for field in range(9):
            if fixed[k, field] != old_fixed[k, field]:
                equal = False
                break
        if not equal:
            break
        k += 1
    change = np.inf
    if k < len(fixed):
        change = fixed[k, 5]
    if k < len(old_fixed):
        change = min(change, old_fixed[k, 5])
    margin = max(1.e-8, 32. * eps * max(1., abs(change))) if math.isfinite(change) else 0.
    matched = (min(len(following), len(old_following))
               if following.ctypes.data == old_following.ctypes.data else 0)
    while matched < min(len(following), len(old_following)):
        equal = True
        for field in range(9):
            if following[matched, field] != old_following[matched, field]:
                equal = False
                break
        if not equal:
            break
        matched += 1
    count, complete, ready, total_wait, pending_wait = 0, 0, 0., 0., 0.
    for j in range(len(old_output)):
        op = old_output[j]
        if complete >= matched or op[6] >= change - margin:
            break
        if op[8] == 4.:
            pending_wait += op[0]
        else:
            complete += 1
            count, ready = j + 1, op[6]
            total_wait += pending_wait
            pending_wait = 0.
    return count, complete, ready, total_wait


@njit
def _numeric_follow(fixed, following, rf, rr, dx, dy, eps, spatial_skip=False, time_window=False,
                    cmax_limit=np.inf, lazy_projection=False, reuse=None, failure=None, early_reject=False):
    output = np.empty((2 * len(following), 9))
    relevant = np.empty(len(fixed) if spatial_skip else 0, np.int64)
    intervals = np.empty((2 * len(fixed), 2))
    pending = np.empty((2 * len(fixed) if lazy_projection else 0, 2), np.int64)
    constraints = np.empty((10, 3))
    candidate, wait = np.empty(9), np.empty(9)
    guard = 1.e-8 * max(1., abs(cmax_limit)) if math.isfinite(cmax_limit) else 0.
    count, ready, total_wait = 0, 0., 0.
    first, past = 0, 0
    if failure is not None:
        failure[0], failure[1] = -2., np.nan  # unknown or a Cmax-only cutoff
    if reuse is not None:
        old_fixed, old_following, old_output = reuse
        count, first, ready, total_wait = _follow_reusable_prefix(
            fixed, following, old_fixed, old_following, old_output, eps)
        output[:count] = old_output[:count]
    for op_index in range(first, len(following)):
        op = following[op_index]
        # Positive remaining durations give a lower bound after every committed
        # prefix. A finite limit is used only to reject non-improving VND moves;
        # the ordinary native scorer still constructs its complete timeline.
        remaining = following[-1, 6] - op[5]
        if ready + remaining >= cmax_limit + guard:
            return output[:0], np.inf
        pending_count = 0
        if time_window:
            # ready is monotone. Retire each completed fixed operation once,
            # instead of walking the whole elapsed prefix for every next op.
            while past < len(fixed):
                end = fixed[past, 6]
                if end >= ready - eps * max(1., abs(ready), abs(end)):
                    break
                past += 1
        n, relevant_count, cursor = 0, 0, past
        start = ready
        # Each advance passes an interval endpoint. At most 2*len(fixed)
        # intervals exist: this bound completes the union search, not a cap.
        for _ in range(2 * len(fixed) + 2):
            for other_index in range(cursor, len(fixed)):
                other = fixed[other_index]
                if time_window:
                    # A projected forbidden interval containing start must
                    # overlap [start, start + duration], up to relaxed time
                    # inequalities. Later fixed operations are revisited after
                    # each WAIT advance, in their original chronological order.
                    edge = start + op[0]
                    margin = max(1.e-8, 16. * eps * max(1., abs(edge), abs(other[5])))
                    if other[5] > edge + margin:
                        break
                cursor = other_index + 1
                if other[6] < ready - eps * max(1., abs(ready), abs(other[6])):
                    continue
                if rr % 2 == 0:
                    separated = max(op[1], op[3]) + dx + eps < min(other[1], other[3])
                else:
                    separated = max(other[1], other[3]) + dx + eps < min(op[1], op[3])
                if separated:
                    continue
                if spatial_skip:
                    relevant[relevant_count] = other_index
                    relevant_count += 1
                for order in (True, False):
                    if spatial_skip and not order:
                        # Conservative spatial exclusion of the proximity polygon.
                        # Its epsilon-relaxed time bounds can extend each segment
                        # by at most epsilon * speed. Keep an additional numerical
                        # margin; near boundaries still use the original polygon.
                        disjoint = False
                        for axis in range(2):
                            va = abs((op[axis + 3] - op[axis + 1]) / op[0]) if op[0] else 0.
                            vb = abs((other[axis + 3] - other[axis + 1]) / other[0]) if other[0] else 0.
                            margin = max(1.e-8, 16. * eps * (1. + va + vb))
                            distance = dx if axis == 0 else dy
                            if (max(op[axis + 1], op[axis + 3]) + distance + margin < min(other[axis + 1], other[axis + 3])
                                    or max(other[axis + 1], other[axis + 3]) + distance + margin < min(op[axis + 1], op[axis + 3])):
                                disjoint = True
                                break
                        if disjoint:
                            continue
                    if lazy_projection:
                        pending[pending_count] = (other_index, int(order))
                        pending_count += 1
                        continue
                    lo, hi = _numeric_forbidden_buffer(op, other, rr, rf, ready, dx, dy, eps, order, constraints, early_reject)
                    if lo <= hi:
                        intervals[n] = (lo, hi)
                        n += 1
                        # Adding forbidden intervals can only delay this start.
                        # A forced WAIT prefix already in conflict stays invalid.
                        if early_reject and lo <= start <= hi:
                            forced = max(np.nextafter(hi, np.inf), hi + eps * max(1., abs(hi)))
                            if forced + remaining >= cmax_limit + guard:
                                return output[:0], np.inf
                            if hi > ready:
                                wait[:] = op
                                wait[0], wait[3], wait[4], wait[5], wait[6], wait[8] = hi - ready, op[1], op[2], ready, hi, 4.
                                if _numeric_conflict(wait, other, rr, rf, dx, dy, eps):
                                    if failure is not None:
                                        failure[0], failure[1] = float(op_index), max(hi + op[0], other[6])
                                    return output[:0], np.inf
            for j in range(pending_count):
                index = pending[j, 0]
                if index < 0:
                    continue
                order = pending[j, 1] == 1
                other = fixed[index]
                if _forbidden_may_contain(op, other, rr, ready, start, dx, dy, eps, order):
                    lo, hi = _numeric_forbidden_buffer(op, other, rr, rf, ready, dx, dy, eps, order, constraints, early_reject)
                    pending[j, 0] = -1
                    if lo <= hi:
                        intervals[n] = (lo, hi)
                        n += 1
                        if early_reject and lo <= start <= hi:
                            forced = max(np.nextafter(hi, np.inf), hi + eps * max(1., abs(hi)))
                            if forced + remaining >= cmax_limit + guard:
                                return output[:0], np.inf
                            if hi > ready:
                                wait[:] = op
                                wait[0], wait[3], wait[4], wait[5], wait[6], wait[8] = hi - ready, op[1], op[2], ready, hi, 4.
                                if _numeric_conflict(wait, other, rr, rf, dx, dy, eps):
                                    if failure is not None:
                                        failure[0], failure[1] = float(op_index), max(hi + op[0], other[6])
                                    return output[:0], np.inf
            furthest = -np.inf
            for j in range(n):
                if intervals[j, 0] <= start <= intervals[j, 1]:
                    furthest = max(furthest, intervals[j, 1])
            if furthest == -np.inf:
                break
            start = max(np.nextafter(furthest, np.inf), furthest + eps * max(1., abs(furthest)))
            if start + remaining >= cmax_limit + guard:
                return output[:0], np.inf
        candidate[:] = op
        candidate[5], candidate[6] = start, start + op[0]
        wait[:] = op
        wait[0], wait[3], wait[4], wait[5], wait[6], wait[8] = start - ready, op[1], op[2], ready, start, 4.
        # Operations excluded by the original time/strict same-rail spatial
        # tests cannot conflict with either the candidate or its stationary WAIT.
        for j in range(relevant_count if spatial_skip else len(fixed)):
            other = fixed[relevant[j] if spatial_skip else j]
            if _numeric_conflict(candidate, other, rr, rf, dx, dy, eps):
                if failure is not None:
                    failure[0], failure[1] = float(op_index), candidate[6]
                return output[:0], np.inf
            if start > ready and _numeric_conflict(wait, other, rr, rf, dx, dy, eps):
                if failure is not None:
                    failure[0], failure[1] = float(op_index), candidate[6]
                return output[:0], np.inf
        if start > ready + eps:
            output[count] = wait
            count += 1
            total_wait += start - ready
        output[count] = candidate
        count += 1
        ready = candidate[6]
    if failure is not None:
        failure[0] = -1.  # complete; no failed operation
    return output[:count], total_wait


@njit
def _priority_arrays(routes, travel, processing, points, parameters, fixed_directions=None):
    """Same two-priority rule as WAG, plus four-robot initial/cross checks.

    This is a native scorer: no dispatch tree, recovery, certification or B32.
    """
    speed, pre, post, dx, dy, eps = parameters
    directions = ([np.empty(0, np.int64) for _ in range(4)] if fixed_directions is None
                  else [bits.copy() for bits in fixed_directions])
    operations = [np.empty((0, 9)) for _ in range(4)]
    completion = np.zeros(4)
    for left in (0, 2):
        right = left + 1
        if fixed_directions is not None:
            bit0, bit1 = fixed_directions[left], fixed_directions[right]
            best = 0.
            if len(bit0) and len(bit1):
                a, b = points[routes[left][0], bit0[0]], points[routes[right][0], bit1[0]]
                if (a[0] + dx > b[0] + eps or
                        (abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps)):
                    return np.inf, directions, operations, completion
        else:
            best, bit0, bit1 = _priority_pair_directions(routes, left, travel, processing, points, parameters)
        if not math.isfinite(best):
            return np.inf, directions, operations, completion
        directions[left], directions[right] = bit0, bit1
        a = _numeric_templates(routes[left], bit0, points, processing, speed, pre, post)
        b = _numeric_templates(routes[right], bit1, points, processing, speed, pre, post)
        if not len(a) or not len(b):
            operations[left], operations[right] = a, b
        else:
            followed, wait = _numeric_follow(a, b, left, right, dx, dy, eps)
            reverse, reverse_wait = np.empty((0, 9)), np.inf
            if not math.isfinite(wait) or wait > 0.:
                reverse, reverse_wait = _numeric_follow(b, a, right, left, dx, dy, eps)
            if not math.isfinite(wait) and not math.isfinite(reverse_wait):
                return np.inf, directions, operations, completion
            forward_cmax = max(a[-1, 6], followed[-1, 6]) if math.isfinite(wait) else np.inf
            reverse_cmax = max(b[-1, 6], reverse[-1, 6]) if math.isfinite(reverse_wait) else np.inf
            if reverse_cmax < forward_cmax:
                operations[left], operations[right] = reverse, b
            else:
                operations[left], operations[right] = a, followed
    for r in range(4):
        completion[r] = operations[r][-1, 6] if len(operations[r]) else 0.
    # A native feasible score must not hide interference across the two rails.
    for r in range(2):
        for s in range(2, 4):
            if not len(routes[r]) or not len(routes[s]):
                continue
            a, b = points[routes[r][0], directions[r][0]], points[routes[s][0], directions[s][0]]
            if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                return np.inf, directions, operations, completion
            j = 0
            for aop in operations[r]:
                while j < len(operations[s]) and operations[s][j, 6] < aop[5]:
                    j += 1
                k = j
                while k < len(operations[s]) and operations[s][k, 5] <= aop[6]:
                    if _numeric_conflict(aop, operations[s][k], r, s, dx, dy, eps):
                        return np.inf, directions, operations, completion
                    k += 1
    return max(completion), directions, operations, completion


@njit
def _priority_pair_directions(routes, left, travel, processing, points, parameters):
    right = left + 1
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    best, bit0, bit1 = np.inf, np.empty(0, np.int64), np.empty(0, np.int64)
    for first0 in range(2 if len(routes[left]) else 1):
        c0, d0 = _direction_array(routes[left], travel, processing, first0)
        for first1 in range(2 if len(routes[right]) else 1):
            c1, d1 = _direction_array(routes[right], travel, processing, first1)
            if len(d0) and len(d1):
                a, b = points[routes[left][0], d0[0]], points[routes[right][0], d1[0]]
                if a[0] + dx > b[0] + eps:
                    continue
                if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                    continue
            if c0 + c1 < best:
                best, bit0, bit1 = c0 + c1, d0, d1
    return best, bit0, bit1


def _schedule_from_arrays(value, directions, operations, completion, ids, policy):
    from mrta_reference.model import Operation, OperationKind, ScheduleResult, ScheduleStatus
    vectors = tuple(tuple(int(d) for d in route) for route in directions)
    if not math.isfinite(value):
        return ScheduleResult(ScheduleStatus.DEADLOCK, directions=vectors,
                              diagnostics=("native priority schedule blocked or initial order infeasible",),
                              reference_policy_id=policy)
    kinds = (OperationKind.MOVE, OperationKind.SETUP, OperationKind.WELD, OperationKind.POST, OperationKind.WAIT)
    result = tuple(Operation(f"R{r}:NATIVE:{k}", r, kinds[int(op[8])], float(op[5]), float(op[6]),
                             (float(op[1]), float(op[2])), (float(op[3]), float(op[4])), k, ids[int(op[7])])
                   for r, route in enumerate(operations) for k, op in enumerate(route))
    return ScheduleResult(ScheduleStatus.FEASIBLE, result, float(value), tuple(float(x) for x in completion),
                          directions=vectors, reference_policy_id=policy)


def priority_schedule(solution, config, directions=None):
    """Shared lightweight four-robot native objective for the two baselines."""
    blocks = block_map(solution, config)
    ids, index, points, travel, processing = _tables_for(blocks, config)
    routes = [np.array([index[b] for b in route.block_ids], np.int64) for route in solution.routes]
    parameters = np.array([config.empty_speed, config.t_pre, config.t_post,
                           config.interference_dx, config.interference_dy, config.numeric_epsilon])
    if directions is not None and any(bit not in (0, 1) for bits in directions for bit in bits):
        raise ValueError("explicit directions must use 0/1 values")
    vectors = None if directions is None else [np.array(bits, np.int64) for bits in directions]
    if vectors is not None and (len(vectors) != 4 or any(len(vectors[r]) != len(routes[r])
            or np.any((vectors[r] != 0) & (vectors[r] != 1)) for r in range(4))):
        raise ValueError("explicit directions must contain one 0/1 value per routed block")
    policy = "NATIVE_PRIORITY_FOUR_ROBOTS" + ("_EXPLICIT_DIRECTIONS" if vectors is not None else "")
    return _schedule_from_arrays(*_priority_arrays(routes, travel, processing, points, parameters, vectors), ids, policy)


def nearest_addition(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    pending = set(block_ids)
    result = []
    point = _home(robot, config)
    while pending:
        options = []
        for block_id in pending:
            block = blocks[block_id]
            for direction, (start, end) in enumerate(
                ((block.start, block.end), (block.end, block.start))
            ):
                options.append((math.dist(point, start), block_id, direction, end))
        _, selected, _, point = min(options, key=lambda item: item[:3])
        result.append(selected)
        pending.remove(selected)
    return tuple(result)


def farthest_insertion(
    block_ids: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    ids, index, points, travel, processing = _tables_for(blocks, config)
    nodes = np.array([index[name] for name in block_ids], dtype=np.int64)
    result = _farthest_array(nodes, travel, processing, points[:, :, 0].max(axis=1))
    return tuple(ids[i] for i in result)


def two_opt(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    consider_final_edge: bool,
) -> tuple[str, ...]:
    ids, index, _, travel, processing = _tables_for(blocks, config)
    nodes = np.array([index[name] for name in route], dtype=np.int64)
    result = _two_opt_array(nodes, travel, processing, consider_final_edge)
    return tuple(ids[i] for i in result)


def factorial_edge_combination(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    window_size: int,
    consider_final_edge: bool,
    exact: bool,
    max_windows: int,
    max_factorial_calls: int,
    deadline: float | None = None,
    accounting: BaselineAccounting | None = None,
) -> tuple[tuple[str, ...], int, int]:
    """Exact orientation DP on each permutation, touching only its window."""
    if deadline is not None and time.perf_counter() >= deadline:
        return tuple(route), 0, 0
    ids, index, _, travel, processing = _tables_for(blocks, config)
    nodes = np.array([index[name] for name in route], dtype=np.int64)
    size = min(window_size, len(nodes))
    if size < 2:
        return tuple(route), 0, 0
    result, calls, windows, window_hit, call_hit = _factorial_array(
        nodes, travel, processing, _permutations(size), size, consider_final_edge,
        exact, max_windows, max_factorial_calls)
    if accounting is not None:
        accounting.factorial_window_cap_hits += int(window_hit)
        accounting.factorial_call_cap_hits += int(call_hit)
    return tuple(ids[i] for i in result), calls, windows


def _improved_route_set_uncached(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    deadline: float | None = None,
    accounting: BaselineAccounting | None = None,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int]:
    exact = wag_config.factorial_mode == "PAPER_EXACT"
    calls = 0
    def improve(route, final_edge):
        nonlocal calls
        while True:
            before = _route_cost(route, blocks, config)
            route = two_opt(route, blocks, config, consider_final_edge=final_edge)
            route, count, _ = factorial_edge_combination(
                route, blocks, config, window_size=wag_config.factorial_window,
                consider_final_edge=final_edge, exact=exact, max_windows=wag_config.max_windows,
                max_factorial_calls=wag_config.max_factorial_calls, deadline=deadline, accounting=accounting)
            calls += count
            if _route_cost(route, blocks, config) >= before - 1.e-9:
                return route
    nearest = nearest_addition(block_ids, robot, blocks, config)
    farthest = farthest_insertion(block_ids, blocks, config)
    i_r1 = improve(nearest, True)
    i_r2 = improve(farthest, False)
    i_r3 = improve(i_r2, True)
    return (i_r1, i_r2, i_r3), calls


@lru_cache(maxsize=8192)
def _improved_routes_cached(assigned, robot, config, settings):
    blocks = {b.block_id: b for b in assigned}
    return _improved_route_set_uncached(tuple(blocks), robot, blocks, config, settings)


def improved_route_set(block_ids, robot, blocks, config, wag_config, *, deadline=None, accounting=None):
    if deadline is not None and time.perf_counter() >= deadline:
        return ((tuple(block_ids),) * 3), 0
    # Construction depends on the assignment set, not on its inherited order.
    # Repeated WAGs and unchanged robot assignments therefore reuse exact routes.
    assigned = tuple(blocks[name] for name in sorted(block_ids))
    return _improved_routes_cached(assigned, robot, config, wag_config)


def _balanced_boundary_assign(
    block_ids: Sequence[str],
    pair: tuple[int, int],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    fixed_loads: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
) -> tuple[int, tuple[str, ...]]:
    ordered = tuple(
        sorted(
            block_ids,
            key=lambda item: (
                min(blocks[item].start[0], blocks[item].end[0]), item
            ),
        )
    )
    best = None
    for boundary in range(len(ordered) + 1):
        left = fixed_loads[pair[0]] + sum(config.process_time(blocks[item].length) for item in ordered[:boundary])
        right = fixed_loads[pair[1]] + sum(config.process_time(blocks[item].length) for item in ordered[boundary:])
        candidate = (max(left, right), abs(left - right), boundary)
        if best is None or candidate < best:
            best = candidate
    return best[2], ordered  # type: ignore[index]


def generate_wag_assignments(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    config: ScientificConfig,
    *,
    max_variants: int | None,
    accounting: BaselineAccounting | None = None,
    scope: FormalScope | None = None,
    retained_assignments: Mapping[str, int] | None = None,
) -> tuple[CanonicalSolution, ...]:
    """Generate at most 25 local boundary-offset WAGs, not all legal assignments.

    In an LNS repair, retained task owners and prescribed X children are fixed;
    only the destroyed tasks participate in residual balancing and offsets.
    """
    blocks = _blocks_for(parents, patterns, config)
    fixed_robot = {
        f"{pattern.parent_id}::{child}": robot
        for pattern in patterns
        if pattern.kind is SplitKind.X_SPLIT
        for child, robot in enumerate(
            (0, 1)
            if pattern.rail is not None and pattern.rail.value == "UPPER"
            else (2, 3)
        )
    }
    for block_id, robot in (retained_assignments or {}).items():
        if block_id not in blocks or robot not in range(4) or not robot_is_eligible(blocks[block_id], robot, config):
            raise ValueError(f"{block_id}: invalid retained assignment")
        if block_id in fixed_robot and fixed_robot[block_id] != robot:
            raise ValueError(f"{block_id}: retained owner violates prescribed X assignment")
        fixed_robot[block_id] = robot
    # Retained tasks and prescribed X children contribute their existing loads;
    # none of them participate in assigning the destroyed task set.
    fixed_loads = [0.0] * 4
    for block_id, robot in fixed_robot.items():
        fixed_loads[robot] += config.process_time(blocks[block_id].length)
    rail_groups: dict[str, list[str]] = {"UPPER": [], "LOWER": []}
    flexible: list[str] = []
    for block_id, block in blocks.items():
        if block_id in fixed_robot:
            continue
        upper = any(robot_is_eligible(block, robot, config) for robot in (0, 1))
        lower = any(robot_is_eligible(block, robot, config) for robot in (2, 3))
        if upper and lower:
            flexible.append(block_id)
        elif upper:
            rail_groups["UPPER"].append(block_id)
        elif lower:
            rail_groups["LOWER"].append(block_id)
        else:
            raise ValueError(f"{block_id}: no eligible rail")
    rail_load = {
        key: sum(config.process_time(blocks[item].length) for item in values)
        for key, values in rail_groups.items()
    }
    rail_load["UPPER"] += fixed_loads[0] + fixed_loads[1]
    rail_load["LOWER"] += fixed_loads[2] + fixed_loads[3]
    for block_id in sorted(
        flexible,
        key=lambda item: (
            min(blocks[item].start[0], blocks[item].end[0]), item
        ),
    ):
        rail = min(("UPPER", "LOWER"), key=lambda key: (rail_load[key], key))
        rail_groups[rail].append(block_id)
        rail_load[rail] += config.process_time(blocks[block_id].length)

    upper_boundary, upper_order = _balanced_boundary_assign(
        rail_groups["UPPER"], (0, 1), blocks, config, fixed_loads=fixed_loads
    )
    lower_boundary, lower_order = _balanced_boundary_assign(
        rail_groups["LOWER"], (2, 3), blocks, config, fixed_loads=fixed_loads
    )
    offsets = sorted(
        itertools.product(range(-2, 3), repeat=2),
        key=lambda item: (abs(item[0]) + abs(item[1]), item),
    )
    result = []
    seen = set()
    for offset_index, (upper_offset, lower_offset) in enumerate(offsets):
        ub = min(len(upper_order), max(0, upper_boundary + upper_offset))
        lb = min(len(lower_order), max(0, lower_boundary + lower_offset))
        route_lists = [
            list(upper_order[:ub]),
            list(upper_order[ub:]),
            list(lower_order[:lb]),
            list(lower_order[lb:]),
        ]
        for block_id, robot in sorted(fixed_robot.items()):
            route_lists[robot].append(block_id)
        routes = tuple(
            Route(robot, tuple(route_lists[robot])) for robot in range(4)
        )
        solution = _canonicalize_cached(
            parents, patterns, routes, config, scope=scope
        )
        if solution.canonical_hash not in seen:
            seen.add(solution.canonical_hash)
            result.append(solution)
        if max_variants is not None and len(result) >= max_variants:
            if offset_index + 1 < len(offsets) and accounting is not None:
                accounting.wag_variant_cap_hits += 1
            break
    return tuple(result)


@lru_cache(maxsize=512)
def _wag_assignments_cached(parents, patterns, config, max_variants, scope):
    return generate_wag_assignments(parents, patterns, config, max_variants=max_variants, scope=scope)


def _route_combinations(
    assignment: CanonicalSolution,
    config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    deadline: float,
    accounting: BaselineAccounting | None = None,
    scope: FormalScope | None = None,
) -> tuple[list[tuple[float, tuple[tuple[str, ...], ...]]], int]:
    blocks = block_map(assignment, config)
    variants = []
    factorial_calls = 0
    for route in assignment.routes:
        route_set, calls = improved_route_set(
            route.block_ids,
            route.robot_id,
            blocks,
            config,
            wag_config,
            deadline=deadline,
            accounting=accounting,
        )
        variants.append(route_set)
        factorial_calls += calls
    combinations = []
    costs = [[_route_cost(route, blocks, config) for route in route_set] for route_set in variants]
    has_split = any(pattern.kind is not SplitKind.WHOLE for pattern in assignment.patterns)
    for indices in itertools.product(range(3), repeat=4):
        routes = tuple(variants[robot][indices[robot]] for robot in range(4))
        cost = max(costs[robot][indices[robot]] for robot in range(4))
        candidate = routes
        if has_split and any(a.split('::', 1)[0] == b.split('::', 1)[0]
                             for route in routes for a, b in zip(route, route[1:])):
            # Adjacent split siblings are merged by the existing canonicalizer.
            # Its reduced setup/post time must be reflected BEFORE pruning.
            candidate = _canonicalize_cached(assignment.parents, assignment.patterns, dict(enumerate(routes)),
                                     config, scope=scope)
            actual_blocks = block_map(candidate, config)
            cost = max(_route_cost(r.block_ids, actual_blocks, config) for r in candidate.routes)
        combinations.append((cost, indices, candidate))
    combinations.sort(key=lambda item: (item[0], item[1]))
    return [(item[0], item[2]) for item in combinations], factorial_calls


def _same_rail_neighbors(robot: int) -> tuple[int, ...]:
    return ({0: (1,), 1: (0,), 2: (3,), 3: (2,)})[robot]


def wag_move_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    n: int,
    heavy_robot: int,
    scope: FormalScope | None = None,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config)
    if not solution.routes[heavy_robot].block_ids:
        return ()
    result = []
    for _ in range(n):
        routes = [list(route.block_ids) for route in solution.routes]
        block_id = routes[heavy_robot][rng.randrange(len(routes[heavy_robot]))]
        pattern = next(
            item for item in solution.patterns
            if item.parent_id == block_id.split("::", 1)[0]
        )
        if pattern.kind is SplitKind.X_SPLIT:
            continue
        destinations = [
            robot
            for robot in _same_rail_neighbors(heavy_robot)
            if robot_is_eligible(blocks[block_id], robot, config)
        ]
        if sum(robot_is_eligible(blocks[block_id], robot, config) for robot in range(4)) == 4:
            destinations = list(range(4))
            destinations.remove(heavy_robot)
        if not destinations:
            continue
        destination = destinations[rng.randrange(len(destinations))]
        routes[heavy_robot].remove(block_id)
        routes[destination].append(block_id)
        result.append(
            _canonicalize_cached(
                solution.parents,
                solution.patterns,
                {robot: tuple(route) for robot, route in enumerate(routes)},
                config,
                scope=scope,
            )
        )
    return tuple(result)


def wag_swap_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    n: int,
    heavy_robot: int,
    scope: FormalScope | None = None,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config)
    result = []
    if not solution.routes[heavy_robot].block_ids:
        return ()
    for _ in range(n):
        neighbor = _same_rail_neighbors(heavy_robot)[0]
        if not solution.routes[neighbor].block_ids:
            continue
        routes = [list(route.block_ids) for route in solution.routes]
        left = routes[heavy_robot][rng.randrange(len(routes[heavy_robot]))]
        right = routes[neighbor][rng.randrange(len(routes[neighbor]))]
        pattern_by_parent = {
            item.parent_id: item for item in solution.patterns
        }
        if any(
            pattern_by_parent[block.split("::", 1)[0]].kind is SplitKind.X_SPLIT
            for block in (left, right)
        ):
            continue
        if not (
            robot_is_eligible(blocks[left], neighbor, config)
            and robot_is_eligible(blocks[right], heavy_robot, config)
        ):
            continue
        li = routes[heavy_robot].index(left)
        ri = routes[neighbor].index(right)
        routes[heavy_robot][li], routes[neighbor][ri] = right, left
        result.append(
            _canonicalize_cached(
                solution.parents,
                solution.patterns,
                {robot: tuple(route) for robot, route in enumerate(routes)},
                config,
                scope=scope,
            )
        )
    return tuple(result)


def wag_lns_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    p: float,
    max_variants: int | None,
    scope: FormalScope | None = None,
) -> tuple[CanonicalSolution, ...]:
    routes = [list(route.block_ids) for route in solution.routes]
    removed = []
    for robot in range(4):
        count = max(1, math.ceil(len(routes[robot]) * p)) if routes[robot] else 0
        rng.shuffle(routes[robot])
        removed.extend(routes[robot][:count])
        routes[robot] = routes[robot][count:]
    retained = {block_id: robot for robot, route in enumerate(routes) for block_id in route}
    blocks = block_map(solution, config)
    removed.sort(
        key=lambda item: (
            min(blocks[item].start[0], blocks[item].end[0]), item
        )
    )
    loads = [
        sum(config.process_time(blocks[item].length) for item in route) for route in routes
    ]
    pattern_by_parent = {item.parent_id: item for item in solution.patterns}
    for block_id in removed:
        pattern = pattern_by_parent[block_id.split("::", 1)[0]]
        if pattern.kind is SplitKind.X_SPLIT:
            child = int(block_id.rsplit("::", 1)[1])
            robot = (
                (0, 1)[child]
                if pattern.rail is not None and pattern.rail.value == "UPPER"
                else (2, 3)[child]
            )
        else:
            eligible = [
                robot for robot in range(4)
                if robot_is_eligible(blocks[block_id], robot, config)
            ]
            robot = min(eligible, key=lambda item: (loads[item], item))
        routes[robot].append(block_id)
        loads[robot] += config.process_time(blocks[block_id].length)
    base = _canonicalize_cached(
        solution.parents,
        solution.patterns,
        {robot: tuple(route) for robot, route in enumerate(routes)},
        config,
        scope=scope,
    )
    repairs = generate_wag_assignments(solution.parents, solution.patterns, config,
                                      max_variants=max_variants, scope=scope, retained_assignments=retained)
    return repairs + (base,)


def toggle_optional_y(
    solution: CanonicalSolution,
    config: ScientificConfig,
) -> CanonicalSolution | None:
    current = {pattern.parent_id: pattern for pattern in solution.patterns}
    for parent in solution.parents:
        splits = generate_y_split_patterns(parent, config)
        if not splits or not whole_eligible_rails(parent.start, parent.end, config):
            continue
        current[parent.parent_id] = (
            splits[0]
            if current[parent.parent_id].kind is SplitKind.WHOLE
            else SplitPattern(parent.parent_id, SplitKind.WHOLE)
        )
        patterns = tuple(current[item.parent_id] for item in solution.parents)
        return generate_wag_assignments(
            solution.parents, patterns, config, max_variants=1
        )[0]
    return None


def transition_legal_pattern(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    scope: FormalScope,
    *,
    family_ordinal: int,
    cyclic_family_priority: bool = False,
) -> tuple[CanonicalSolution | None, SplitKind | None]:
    catalog = build_legal_pattern_catalog(solution.parents, config, scope)
    current = {pattern.parent_id: pattern for pattern in solution.patterns}
    by_kind: dict[SplitKind, list[tuple[ParentWeld, SplitPattern]]] = {
        kind: [] for kind in SplitKind
    }
    for parent in solution.parents:
        for option in catalog[parent.parent_id]:
            if option.pattern_id != current[parent.parent_id].pattern_id:
                by_kind[option.kind].append((parent, option))
    families = tuple(
        kind
        for kind in (SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT)
        if by_kind[kind]
    )
    if not families:
        return None, None
    if cyclic_family_priority:
        order = (SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT)
        offset = family_ordinal % len(order)
        target_kind = next(kind for kind in order[offset:] + order[:offset] if by_kind[kind])
    else:
        target_kind = families[family_ordinal % len(families)]
    options = sorted(
        by_kind[target_kind], key=lambda item: (item[0].parent_id, item[1].pattern_id)
    )
    parent, target = options[rng.randrange(len(options))]
    current[parent.parent_id] = target
    patterns = tuple(current[item.parent_id] for item in solution.parents)
    generated = generate_wag_assignments(
        solution.parents,
        patterns,
        config,
        max_variants=1,
        scope=scope,
    )
    return (generated[0] if generated else None), target_kind


def _evaluate_routed_assignment(
    assignment: CanonicalSolution,
    scientific_config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    evaluator: CommonBaselineEvaluator,
    *,
    source: str,
) -> None:
    key = assignment.patterns, tuple(tuple(sorted(r.block_ids)) for r in assignment.routes)
    if not hasattr(evaluator, '_wag_routed_assignments'):
        evaluator._wag_routed_assignments = set()
    if key in evaluator._wag_routed_assignments:
        return
    construction_started = time.perf_counter()
    combinations, factorial_calls = _route_combinations(
        assignment,
        scientific_config,
        wag_config,
        deadline=evaluator.deadline,
        accounting=evaluator.accounting,
        scope=evaluator.scope if evaluator.scope.x_split_rule_id is not None else None,
    )
    elapsed = time.perf_counter() - construction_started
    evaluator.accounting.construction_time += elapsed
    evaluator.accounting.factorial_local_search_time += elapsed
    evaluator.accounting.cheap_candidate_operations += factorial_calls
    native = (getattr(evaluator, 'candidate_evaluator', None) is priority_schedule
              and not evaluator.certify)
    if native:
        if not hasattr(evaluator, '_wag_native_contexts'):
            evaluator._wag_native_contexts = {}
        context = evaluator._wag_native_contexts.get(assignment.patterns)
        if context is None:
            table_ids, index, points, travel, processing = _tables_for(_blocks_for(
                assignment.parents, assignment.patterns, scientific_config), scientific_config)
            parameters = np.array([scientific_config.empty_speed, scientific_config.t_pre, scientific_config.t_post,
                                   scientific_config.interference_dx, scientific_config.interference_dy,
                                   scientific_config.numeric_epsilon])
            context = table_ids, index, points, travel, processing, parameters, {}, set()
            evaluator._wag_native_contexts[assignment.patterns] = context
        table_ids, index, points, travel, processing, parameters, encoded, scored = context
    for index, (lower, routes) in enumerate(combinations):
        native_arrays = None
        if evaluator.expired:
            break
        # Free-end process/travel omits only nonnegative waiting.
        if evaluator.best is not None and lower >= evaluator.best.metrics.cmax + 1.e-8:
            break
        if native and not isinstance(routes, CanonicalSolution):
            table_ids, table_index, points, travel, processing, parameters, encoded, scored = context
            if routes in scored:
                continue
            arrays = []
            for route in routes:
                if route not in encoded:
                    encoded[route] = np.array([table_index[b] for b in route], np.int64)
                arrays.append(encoded[route])
            scheduler_started = time.perf_counter()
            native_arrays = _priority_arrays(arrays, travel, processing, points, parameters)
            evaluator.accounting.scheduler_time += time.perf_counter() - scheduler_started
            evaluator.accounting.reference_calls += 1
            value = native_arrays[0]
            scored.add(routes)
            if not math.isfinite(value) or (evaluator.best is not None and value >= evaluator.best.metrics.cmax - 1.e-8):
                continue
        candidate = routes if isinstance(routes, CanonicalSolution) else _canonicalize_cached(
            assignment.parents, assignment.patterns, dict(enumerate(routes)), scientific_config,
            scope=evaluator.scope if evaluator.scope.x_split_rule_id else None)
        native_schedule = None
        if (native_arrays is not None and candidate.patterns == assignment.patterns
                and tuple(route.block_ids for route in candidate.routes) == routes):
            native_schedule = _schedule_from_arrays(*native_arrays, table_ids, "NATIVE_PRIORITY_FOUR_ROBOTS")
        evaluator.evaluate(candidate, source=f"{source}_ROUTE_COMBINATION_{index}",
                           **({'native_schedule': native_schedule} if native_schedule is not None else {}))
    if not evaluator.expired:
        evaluator._wag_routed_assignments.add(key)


def run_adapted_wag_vns(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    seed: int,
    time_limit: float,
    scope: FormalScope,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    common_seed: EvaluatedCandidate | None = None,
    method_id: str = METHOD_ID,
    pattern_access_policy: bool | None = None,
    certify: bool = True,
    candidate_evaluator: CandidateEvaluator | None = None,
) -> BaselineResult:
    rng = random.Random(seed)
    v2 = scope.x_split_rule_id is not None
    repaired_access = (v2 and method_id == METHOD_ID_V2) if pattern_access_policy is None else pattern_access_policy
    if repaired_access and not v2:
        raise ValueError("pattern-first access requires FORMAL_SCOPE_V2")
    evaluator = CommonBaselineEvaluator(
        parents,
        scientific_config,
        scope,
        time_limit=time_limit,
        checkpoints=checkpoints,
        certify=certify,
        candidate_evaluator=candidate_evaluator,
    )
    if v2:
        evaluator.accounting.legal_x_pattern_count = sum(
            pattern.kind is SplitKind.X_SPLIT
            for patterns in build_legal_pattern_catalog(
                parents, scientific_config, scope
            ).values()
            for pattern in patterns
        )
    init_started = time.perf_counter()
    if common_seed is not None:
        evaluator.inject_certified(
            common_seed, source="WAG_COMMON_CERTIFIED_SEED"
        )
    else:
        try:
            patterns = _initial_patterns(
                parents, scientific_config, scope=scope if v2 else None
            )
            assignments = generate_wag_assignments(
                parents,
                patterns,
                scientific_config,
                max_variants=wag_config.max_wag_variants,
                accounting=evaluator.accounting,
                scope=scope if v2 else None,
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            evaluator.reject_construction(f"WAG initialization: {error}")
            assignments = ()
        for index, assignment in enumerate(assignments):
            if evaluator.expired:
                break
            evaluator.evaluate(assignment, source=f"WAG_INITIAL_{index}_NATIVE")
            if evaluator.expired:
                break
            _evaluate_routed_assignment(
                assignment,
                scientific_config,
                wag_config,
                evaluator,
                source=f"WAG_INITIAL_{index}",
            )
    initialization_time = time.perf_counter() - init_started
    initial_candidate = evaluator.best
    initial_reference_calls = evaluator.accounting.reference_calls

    iterations = 0
    while (
        evaluator.best is not None
        and not evaluator.expired
        and iterations < wag_config.imax
    ):
        iterations += 1
        if iterations % 50 == 0:
            print('WAG_PROGRESS', iterations, round(evaluator.elapsed, 3),
                  evaluator.best.metrics.cmax if evaluator.best else None, flush=True)
        best_solution = evaluator.best.solution
        if repaired_access:
            transition_started = evaluator.elapsed
            optional, proposed_kind = transition_legal_pattern(
                best_solution, scientific_config, rng, scope,
                family_ordinal=seed + iterations - 1,
                cyclic_family_priority=True,
            )
            evaluator.accounting.construction_time += evaluator.elapsed - transition_started
            evaluator.accounting.pattern_transition_proposals += int(proposed_kind is not None)
            evaluator.accounting.x_pattern_proposals += int(proposed_kind is SplitKind.X_SPLIT)
            evaluator.accounting.y_pattern_proposals += int(proposed_kind is SplitKind.Y_SPLIT)
            if proposed_kind is SplitKind.X_SPLIT and evaluator.accounting.first_x_proposal_time is None:
                evaluator.accounting.first_x_proposal_time = evaluator.elapsed
            trace = {
                "iteration": iterations - 1,
                "family_ordinal": (seed + iterations - 1) % 3,
                "target_kind": None if proposed_kind is None else proposed_kind.value,
                "proposal_time": evaluator.elapsed,
                "constructed": optional is not None,
                "evaluation_started": None,
                "reference_evaluated": False,
                "status": None,
            }
            evaluator.accounting.pattern_transition_trace.append(trace)
            if optional is not None and not evaluator.expired:
                trace["evaluation_started"] = evaluator.elapsed
                calls_before = evaluator.accounting.reference_calls
                evaluated = evaluator.evaluate(optional, source=f"WAG_PATTERN_TRANSITION_{iterations}_DIRECT")
                trace["reference_evaluated"] = evaluator.accounting.reference_calls > calls_before
                trace["status"] = evaluated.status.value
                trace["completed_at"] = evaluated.completed_at
            if evaluator.expired:
                break
            best_solution = evaluator.best.solution
        if evaluator.best.schedule is not None:
            heavy = max(
                range(4),
                key=lambda robot: (
                    evaluator.best.schedule.robot_completion[robot], -robot
                ),
            )
        else:
            blocks = block_map(best_solution, scientific_config)
            heavy = max(
                range(4),
                key=lambda robot: (
                    _route_cost(best_solution.routes[robot].block_ids, blocks, scientific_config),
                    -robot,
                ),
            )
        operator = rng.randrange(3)
        if operator == 0:
            evaluator.accounting.move_calls += 1
            candidates = wag_move_candidates(
                best_solution,
                scientific_config,
                rng,
                n=wag_config.n,
                heavy_robot=heavy,
                scope=scope if v2 else None,
            )
            name = "MOVE"
        elif operator == 1:
            evaluator.accounting.swap_calls += 1
            candidates = wag_swap_candidates(
                best_solution,
                scientific_config,
                rng,
                n=wag_config.n,
                heavy_robot=heavy,
                scope=scope if v2 else None,
            )
            name = "SWAP"
        else:
            evaluator.accounting.lns_calls += 1
            candidates = wag_lns_candidates(
                best_solution,
                scientific_config,
                rng,
                p=wag_config.p,
                max_variants=wag_config.max_wag_variants,
                scope=scope if v2 else None,
            )
            name = "LNS"
        if not repaired_access and iterations % 3 == 0:
            evaluator.accounting.optional_y_toggle_attempts += 1
            if v2:
                optional, proposed_kind = transition_legal_pattern(
                    best_solution,
                    scientific_config,
                    rng,
                    scope,
                    family_ordinal=iterations // 3,
                )
                evaluator.accounting.pattern_transition_proposals += int(
                    proposed_kind is not None
                )
                evaluator.accounting.x_pattern_proposals += int(
                    proposed_kind is SplitKind.X_SPLIT
                )
                evaluator.accounting.y_pattern_proposals += int(
                    proposed_kind is SplitKind.Y_SPLIT
                )
            else:
                optional = toggle_optional_y(best_solution, scientific_config)
            if optional is not None:
                candidates = candidates + (optional,)
        seen = set()
        for index, assignment in enumerate(candidates):
            if evaluator.expired:
                break
            if assignment.canonical_hash in seen:
                continue
            seen.add(assignment.canonical_hash)
            _evaluate_routed_assignment(
                assignment,
                scientific_config,
                wag_config,
                evaluator,
                source=f"WAG_{name}_{iterations}_{index}",
            )

    if evaluator.best is None and evaluator.accounting.numeric_failure:
        termination_reason = "NUMERIC_FAILURE"
    elif evaluator.best is None:
        termination_reason = "INITIALIZATION_FAILED"
    elif evaluator.expired:
        termination_reason = "TIME_LIMIT"
    elif iterations >= wag_config.imax:
        termination_reason = "ITERATION_LIMIT"
    else:
        termination_reason = "COMPLETED_OTHER"

    return evaluator.finish(
        method_id=method_id,
        method_config_hash=canonical_config_hash(wag_config),
        iterations=iterations,
        initialization_time=initialization_time,
        termination_reason=termination_reason,
        initial_candidate=initial_candidate,
        initial_reference_calls=initial_reference_calls,
        diagnostics=(
            "paper three-robot conflict scheduler is excluded from common-model fitness",
            "ADAPTED_WAG_ASSIGNMENT_V1",
        ),
    )


def run_adapted_wag_vns_v2(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    wag_config: AdaptedWAGConfig,
    *,
    seed: int,
    time_limit: float,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    certify: bool = True,
    candidate_evaluator: CandidateEvaluator | None = None,
) -> BaselineResult:
    return run_adapted_wag_vns(
        parents,
        scientific_config,
        wag_config,
        seed=seed,
        time_limit=time_limit,
        scope=FORMAL_SCOPE_V2,
        checkpoints=checkpoints,
        method_id=METHOD_ID_V2,
        certify=certify,
        candidate_evaluator=candidate_evaluator,
    )
