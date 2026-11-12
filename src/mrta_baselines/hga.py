from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Mapping, Sequence

import numpy as np
from numba import njit

from .wag_vns import (
    _array_route_cost, _interval_costs,
    _priority_arrays, _numeric_conflict, _numeric_follow, _numeric_templates,
    _direction_array, _route_prefix_suffix, _splice_cost,
    _tables_for, _schedule_from_arrays, priority_schedule,
)

from mrta_reference.geometry import (
    blocks_for_pattern,
    build_legal_pattern_catalog,
    generate_y_split_patterns,
    oriented_endpoints,
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
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize

from .common import (
    BaselineResult,
    CandidateEvaluator,
    CommonBaselineEvaluator,
    EvaluatedCandidate,
    canonical_config_hash,
)


METHOD_ID = "ADAPTED_HGA_V1"
METHOD_ID_V2 = "ADAPTED_HGA_V2"
PAPER_CORE_ID = "HGA_PAPER_CORE_V1"


@dataclass(frozen=True)
class AdaptedHGAConfig:
    mu: int = 20
    lambda_size: int = 10
    alpha: int = 20
    max_iterations: int = 200000
    initialization_reference_limit: int = 20
    vnd_max_passes: int | None = None
    vnd_candidate_cap: int | None = None
    pattern_mutation_probability: float = 0.15
    direction_encoding: str = "EXPLICIT"

    def __post_init__(self) -> None:
        if min(self.mu, self.lambda_size, self.alpha, self.max_iterations) <= 0:
            raise ValueError("HGA integer parameters must be positive")
        if self.initialization_reference_limit <= 0:
            raise ValueError("initialization_reference_limit must be positive")
        if any(x is not None and x <= 0 for x in (self.vnd_max_passes, self.vnd_candidate_cap)):
            raise ValueError("VND bounds must be positive")
        if not 0.0 <= self.pattern_mutation_probability <= 1.0:
            raise ValueError("pattern_mutation_probability must be in [0,1]")
        if self.direction_encoding not in ("EXPLICIT", "LEGACY_ROUTE_DP"):
            raise ValueError("unknown HGA direction encoding")


@dataclass
class _Individual:
    solution: CanonicalSolution
    proxy: tuple[float, float, str]
    evaluated: EvaluatedCandidate | None = None


def legal_pattern_options(
    parent: ParentWeld,
    config: ScientificConfig,
    *,
    catalog: Mapping[str, Sequence[SplitPattern]] | None = None,
) -> tuple[SplitPattern, ...]:
    if catalog is not None:
        return tuple(catalog[parent.parent_id])
    y = tuple(generate_y_split_patterns(parent, config))
    if whole_eligible_rails(parent.start, parent.end, config):
        return (SplitPattern(parent.parent_id, SplitKind.WHOLE),) + y
    if not y:
        raise ValueError(f"{parent.parent_id}: no legal WHOLE/Y_SPLIT pattern")
    return y


def initial_patterns(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    *,
    catalog: Mapping[str, Sequence[SplitPattern]] | None = None,
) -> tuple[SplitPattern, ...]:
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        options = legal_pattern_options(parent, config, catalog=catalog)
        whole = next((value for value in options if value.kind is SplitKind.WHOLE), None)
        result.append(whole if whole is not None else options[0])
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
    states = {
        direction: (config.process_time(blocks[route[0]].length), (direction,))
        for direction in (0, 1)
    }
    for index in range(1, len(route)):
        current = blocks[route[index]]
        previous = blocks[route[index - 1]]
        next_states = {}
        for direction in (0, 1):
            start, _ = oriented_endpoints(current, direction)
            options = []
            for old_direction, (cost, vector) in states.items():
                _, end = oriented_endpoints(previous, old_direction)
                options.append(
                    (
                        cost
                        + math.dist(end, start) / config.empty_speed
                        + config.process_time(current.length),
                        vector + (direction,),
                    )
                )
            next_states[direction] = min(options, key=lambda item: item)
        states = next_states
    return min(states.values(), key=lambda item: item)[0]


def hga_surrogate(
    solution: CanonicalSolution, config: ScientificConfig
) -> tuple[float, float, str]:
    blocks = block_map(solution, config)
    costs = tuple(_route_cost(route.block_ids, blocks, config) for route in solution.routes)
    return (max(costs, default=0.0), sum(costs), solution.canonical_hash)


def _best_insertion(
    block_id: str,
    routes: list[list[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    shortest_route_only: bool,
    excluded_robot: int | None = None,
) -> tuple[int, int]:
    eligible = [
        robot
        for robot in range(4)
        if robot_is_eligible(blocks[block_id], robot, config)
    ]
    if not eligible:
        raise ValueError(f"{block_id}: no eligible robot")
    remaining = [r for r in eligible if r != excluded_robot]
    if remaining:
        eligible = remaining
    if shortest_route_only:
        loads = {robot: _route_cost(routes[robot], blocks, config) for robot in eligible}
        minimum = min(loads.values())
        eligible = [robot for robot in eligible if abs(loads[robot] - minimum) <= 1.0e-12]
    options = []
    route_costs = [_route_cost(route, blocks, config) for route in routes]
    for robot in eligible:
        before = route_costs[robot]
        for position in range(len(routes[robot]) + 1):
            candidate = routes[robot][:position] + [block_id] + routes[robot][position:]
            value = _route_cost(candidate, blocks, config)
            delta = value - before
            primary = delta if shortest_route_only else max(
                value, *(route_costs[r] for r in range(4) if r != robot))
            options.append((primary, delta, robot, position))
    _, _, robot, position = min(options)
    return robot, position


def initialize_hga_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    rng: random.Random,
    *,
    patterns: Sequence[SplitPattern] | None = None,
    scope: FormalScope | None = None,
) -> CanonicalSolution:
    catalog = None if scope is None else build_legal_pattern_catalog(parents, config, scope)
    selected = tuple(patterns or initial_patterns(parents, config, catalog=catalog))
    blocks = _blocks_for(parents, selected, config)
    pending = list(blocks)
    routes: list[list[str]] = [[] for _ in range(4)]
    while pending:
        index = rng.randrange(len(pending))
        block_id = pending.pop(index)
        robot, position = _best_insertion(
            block_id, routes, blocks, config, shortest_route_only=True
        )
        routes[robot].insert(position, block_id)
    solution = canonicalize(
        parents, selected, {robot: tuple(route) for robot, route in enumerate(routes)}, config,
        scope=scope,
    )
    return repair_first_tasks(solution, config, scope=scope)


def initialize_hga_region_seed(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    *,
    patterns: Sequence[SplitPattern] | None = None,
    scope: FormalScope | None = None,
) -> CanonicalSolution:
    """Adapt the paper's region decomposition to two rails and four robots."""
    catalog = None if scope is None else build_legal_pattern_catalog(parents, config, scope)
    selected = tuple(patterns or initial_patterns(parents, config, catalog=catalog))
    blocks = _blocks_for(parents, selected, config)
    rail_groups: list[list[str]] = [[], []]
    flexible: list[str] = []
    rail_loads = [0.0, 0.0]
    for block_id, block in blocks.items():
        eligible_rails = [
            rail
            for rail, pair in enumerate(((0, 1), (2, 3)))
            if any(robot_is_eligible(block, robot, config) for robot in pair)
        ]
        if len(eligible_rails) == 2:
            flexible.append(block_id)
        elif eligible_rails:
            rail_groups[eligible_rails[0]].append(block_id)
            rail_loads[eligible_rails[0]] += config.process_time(block.length)
        else:
            raise ValueError(f"{block_id}: no eligible rail")
    for block_id in sorted(flexible):
        block = blocks[block_id]
        rail = min(range(2), key=lambda value: (rail_loads[value], value))
        rail_groups[rail].append(block_id)
        rail_loads[rail] += config.process_time(block.length)
    routes: list[list[str]] = [[] for _ in range(4)]
    for rail, block_ids in enumerate(rail_groups):
        ordered = sorted(
            block_ids,
            key=lambda item: (min(blocks[item].start[0], blocks[item].end[0]), item),
        )
        boundary = min(
            range(len(ordered) + 1),
            key=lambda position: (
                max(
                    sum(
                        config.process_time(blocks[item].length)
                        for item in ordered[:position]
                    ),
                    sum(
                        config.process_time(blocks[item].length)
                        for item in ordered[position:]
                    ),
                ),
                position,
            ),
        )
        routes[2 * rail] = ordered[:boundary]
        routes[2 * rail + 1] = ordered[boundary:]
    solution = canonicalize(
        parents, selected, {robot: tuple(route) for robot, route in enumerate(routes)}, config,
        scope=scope,
    )
    return repair_first_tasks(solution, config, scope=scope)


def _eligible_routes(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> bool:
    return all(
        robot_is_eligible(blocks[block_id], robot, config)
        for robot, route in enumerate(routes)
        for block_id in route
    )


def _location(routes: Sequence[Sequence[str]], block_id: str) -> tuple[int, int]:
    for robot, route in enumerate(routes):
        if block_id in route:
            return robot, route.index(block_id)
    raise ValueError(f"unknown block {block_id}")


def apply_hga_neighborhood(
    solution: CanonicalSolution,
    move: int,
    u: str,
    v: str,
    config: ScientificConfig,
    *,
    scope: FormalScope | None = None,
) -> CanonicalSolution | None:
    """Apply paper neighborhoods M1-M6 to the common four-route encoding."""
    routes = [list(route.block_ids) for route in solution.routes]
    blocks = block_map(solution, config)
    ru, iu = _location(routes, u)
    rv, iv = _location(routes, v)
    longest = max(range(4), key=lambda robot: (_route_cost(routes[robot], blocks, config), -robot))
    if move == 1:
        if ru == rv or ru != longest:
            return None
        routes[ru].pop(iu)
        iv = routes[rv].index(v)
        routes[rv].insert(iv + 1, u)
    elif move == 2:
        if ru == rv or longest not in (ru, rv):
            return None
        routes[ru][iu], routes[rv][iv] = routes[rv][iv], routes[ru][iu]
    elif move == 3:
        if ru == rv or longest not in (ru, rv):
            return None
        if iu + 1 >= len(routes[ru]) or iv + 1 >= len(routes[rv]):
            return None
        left = routes[ru][iu : iu + 2]
        right = routes[rv][iv : iv + 2]
        routes[ru][iu : iu + 2] = right
        routes[rv][iv : iv + 2] = left
    elif move == 4:
        if ru != rv or iu == iv:
            return None
        lo, hi = sorted((iu, iv))
        # Replace (u,x),(v,y) by (u,v),(x,y): reverse x..v.
        routes[ru][lo + 1 : hi + 1] = reversed(routes[ru][lo + 1 : hi + 1])
    elif move == 5:
        if ru == rv:
            return None
        left_prefix, left_tail = routes[ru][:iu + 1], routes[ru][iu + 1:]
        right_prefix, right_tail = routes[rv][:iv + 1], routes[rv][iv + 1:]
        # (u,v),(x,y): join the two prefixes, then the two tails.
        routes[ru] = left_prefix + list(reversed(right_prefix))
        routes[rv] = list(reversed(left_tail)) + right_tail
    elif move == 6:
        if ru == rv:
            return None
        left_tail = routes[ru][iu + 1 :]
        right_tail = routes[rv][iv + 1 :]
        # (u,y),(v,x): exchange tails without reversing them.
        routes[ru][iu + 1 :] = right_tail
        routes[rv][iv + 1 :] = left_tail
    else:
        raise ValueError("move must be M1..M6")
    if not _eligible_routes(routes, blocks, config):
        return None
    try:
        return canonicalize(
            solution.parents,
            solution.patterns,
            {robot: tuple(route) for robot, route in enumerate(routes)},
            config,
            scope=scope,
        )
    except ValueError:
        return None


def alpha_nearest(
    solution: CanonicalSolution, block_id: str, alpha: int, config: ScientificConfig
) -> tuple[str, ...]:
    blocks = block_map(solution, config)
    origin = blocks[block_id]
    result = []
    for other_id, other in blocks.items():
        if other_id == block_id:
            continue
        distance = min(
            math.dist(left, right)
            for left in (origin.start, origin.end)
            for right in (other.start, other.end)
        )
        result.append((distance, other_id))
    result.sort()
    return tuple(item[1] for item in result[: min(alpha, len(result))])


@njit
def _move_segments(routes, move, ru, iu, rv, iv):
    nu, nv = len(routes[ru]), len(routes[rv])
    if move == 1:
        a = [(ru, 0, iu, False), (ru, iu + 1, nu, False)]
        b = [(rv, 0, iv + 1, False), (ru, iu, iu + 1, False), (rv, iv + 1, nv, False)]
    elif move == 2 or move == 3:
        count = 1 if move == 2 else 2
        a = [(ru, 0, iu, False), (rv, iv, iv + count, False), (ru, iu + count, nu, False)]
        b = [(rv, 0, iv, False), (ru, iu, iu + count, False), (rv, iv + count, nv, False)]
    elif move == 4:
        lo, hi = min(iu, iv) + 1, max(iu, iv)
        a = [(ru, 0, lo, False), (ru, lo, hi + 1, True), (ru, hi + 1, nu, False)]
        b = [(rv, 0, 0, False)]
    elif move == 5:
        a = [(ru, 0, iu + 1, False), (rv, 0, iv + 1, True)]
        b = [(ru, iu + 1, nu, True), (rv, iv + 1, nv, False)]
    else:
        a = [(ru, 0, iu + 1, False), (rv, iv + 1, nv, False)]
        b = [(rv, 0, iv + 1, False), (ru, iu + 1, nu, False)]
    return a, b


@njit
def _materialize_segments(segments, routes):
    n = 0
    for _, lo, hi, _ in segments:
        n += hi - lo
    output = np.empty(n, np.int64)
    k = 0
    for source, lo, hi, reverse in segments:
        for j in range(hi - lo):
            output[k] = routes[source][hi - 1 - j if reverse else lo + j]
            k += 1
    return output


@njit(inline='always')
def _raw_segment(a, b, old, source, lo, hi, reverse, target,
                 nodes, matrices, invalid, parent_index, travel):
    if not math.isfinite(min(a, b)):
        return np.inf, np.inf, old
    if hi <= lo:
        return a, b, old
    first = nodes[source, hi - 1] if reverse else nodes[source, lo]
    if invalid[source, target, hi] != invalid[source, target, lo]:
        return np.inf, np.inf, old
    if old >= 0 and parent_index[old] == parent_index[first]:
        return np.inf, np.inf, old
    if old >= 0:
        a, b = (min(a + travel[old, first, 0, 0], b + travel[old, first, 1, 0]),
                min(a + travel[old, first, 0, 1], b + travel[old, first, 1, 1]))
    if reverse:
        c = min(a + matrices[source, lo, hi - 1, 1, 1], b + matrices[source, lo, hi - 1, 1, 0])
        d = min(a + matrices[source, lo, hi - 1, 0, 1], b + matrices[source, lo, hi - 1, 0, 0])
        return c, d, nodes[source, lo]
    c = min(a + matrices[source, lo, hi - 1, 0, 0], b + matrices[source, lo, hi - 1, 1, 0])
    d = min(a + matrices[source, lo, hi - 1, 0, 1], b + matrices[source, lo, hi - 1, 1, 1])
    return c, d, nodes[source, hi - 1]


@njit
def _raw_move_costs(nodes, lengths, summaries, invalid, parent_index, travel, move, ru, iu, rv, iv):
    """M1-M6 exact costs without allocating a candidate or segment lists."""
    cu, cv = 0., 0.
    for side in range(1 if move == 4 else 2):
        r, i, s, j = (ru, iu, rv, iv) if side == 0 else (rv, iv, ru, iu)
        a, b, old = 0., 0., -1
        n, m = lengths[r], lengths[s]
        if move == 1:
            stop = i if side == 0 else i + 1
            a, b, old = _raw_segment(a, b, old, r, 0, stop, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
            if side == 1:
                a, b, old = _raw_segment(a, b, old, s, j, j + 1, False, r,
                                             nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, r, i + 1, n, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
        elif move in (2, 3):
            count = 1 if move == 2 else 2
            a, b, old = _raw_segment(a, b, old, r, 0, i, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, s, j, j + count, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, r, i + count, n, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
        elif move == 4:
            lo, hi = min(i, j) + 1, max(i, j) + 1
            a, b, old = _raw_segment(a, b, old, r, 0, lo, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, r, lo, hi, True, r,
                                         nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, r, hi, n, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
        elif move == 5:
            if side == 0:
                a, b, old = _raw_segment(a, b, old, r, 0, i + 1, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
                a, b, old = _raw_segment(a, b, old, s, 0, j + 1, True, r,
                                         nodes, summaries, invalid, parent_index, travel)
            else:
                a, b, old = _raw_segment(a, b, old, s, j + 1, m, True, r,
                                         nodes, summaries, invalid, parent_index, travel)
                a, b, old = _raw_segment(a, b, old, r, i + 1, n, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
        else:
            a, b, old = _raw_segment(a, b, old, r, 0, i + 1, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
            a, b, old = _raw_segment(a, b, old, s, j + 1, m, False, r,
                                         nodes, summaries, invalid, parent_index, travel)
        if side == 0:
            cu = min(a, b)
        else:
            cv = min(a, b)
    return cu, cv


@njit
def _legal_first_arrays(routes, points, parameters):
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for mask in range(16):
        valid = True
        for r in range(4):
            if not len(routes[r]):
                continue
            a = points[routes[r][0], (mask >> r) & 1]
            for s in range(r + 1, 4):
                if not len(routes[s]):
                    continue
                b = points[routes[s][0], (mask >> s) & 1]
                if r // 2 == s // 2 and a[0] + dx > b[0] + eps:
                    valid = False
                    break
                if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return True
    return False


@njit
def _segment_head_costs(segments, nodes, summaries, invalid, parent_index, travel, target):
    """First-direction route costs from at most three unchanged intervals."""
    head = -1
    values = np.zeros(2)
    for first in range(2):
        a, b, old = (0. if first == 0 else np.inf), (0. if first == 1 else np.inf), -1
        for source, lo, hi, reverse in segments:
            if hi > lo and head < 0:
                head = nodes[source, hi - 1 if reverse else lo]
            a, b, old = _raw_segment(a, b, old, source, lo, hi, reverse, target,
                                     nodes, summaries, invalid, parent_index, travel)
        values[first] = min(a, b)
    return head, values




@njit
def _pair_head_bound(left, heads, head_costs, points, parameters,
                      native_ops, directions, changed, incumbent):
    """Reject only if every native minimum-sum head choice cannot improve.

    The native policy first minimizes the pair's sum of independent route
    costs subject to legal initial positions. Its coordinated Cmax is at least
    each selected route cost. Interval arithmetic can associate additions
    differently, so retain all near-tied head choices and a generous guard.
    An unchanged route with no WAIT also supplies its exact nominal endpoint.
    """
    right = left + 1
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    sums = np.full((2, 2), np.inf)
    best = np.inf
    for d0 in range(2 if heads[left] >= 0 else 1):
        for d1 in range(2 if heads[right] >= 0 else 1):
            if heads[left] >= 0 and heads[right] >= 0:
                a, b = points[heads[left], d0], points[heads[right], d1]
                if a[0] + dx > b[0] + eps:
                    continue
                if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                    continue
            value = head_costs[left, d0] + head_costs[right, d1]
            sums[d0, d1] = value
            best = min(best, value)
    if not math.isfinite(best):
        return False
    if not math.isfinite(incumbent):
        return True
    guard = 1.e-8 * max(1., abs(best), abs(incumbent))
    for d0 in range(2 if heads[left] >= 0 else 1):
        for d1 in range(2 if heads[right] >= 0 else 1):
            if sums[d0, d1] > best + 2. * guard:
                continue
            lower = max(head_costs[left, d0], head_costs[right, d1])
            if lower >= incumbent + guard:
                continue
            exact_block = False
            for r in (left, right):
                if changed[r] or not len(native_ops[r]):
                    continue
                d = d0 if r == left else d1
                if d != directions[r][0]:
                    continue
                has_wait, cursor = False, 0.
                for op in native_ops[r]:
                    if op[8] == 4. or op[5] != cursor:
                        has_wait = True
                        break
                    cursor = op[6]
                if not has_wait and native_ops[r][-1, 6] >= incumbent - 1.e-8:
                    exact_block = True
                    break
            if not exact_block:
                return True
    return False




@njit
def _unique_pair_heads(routes, left, costs, points, parameters):
    """Use interval costs only when they prove a unique native head choice.

    Near ties retain the original full direction DP. The guard is deliberately
    much larger than positive-sum reassociation errors in these route tables.
    """
    best, second, selected0, selected1 = np.inf, np.inf, -1, -1
    right = left + 1
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for d0 in range(2 if len(routes[left]) else 1):
        for d1 in range(2 if len(routes[right]) else 1):
            if len(routes[left]) and len(routes[right]):
                a, b = points[routes[left][0], d0], points[routes[right][0], d1]
                if a[0] + dx > b[0] + eps:
                    continue
                if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                    continue
            value = costs[left, d0] + costs[right, d1]
            if value < best:
                second, best, selected0, selected1 = best, value, d0, d1
            elif value < second:
                second = value
    if not math.isfinite(best):
        return -1, -1
    guard = 1.e-8 * max(1., abs(best)) * max(1, len(routes[left]) + len(routes[right]))
    if second <= best + 2. * guard:
        return -1, -1
    return selected0, selected1




@njit
def _native_pair(routes, left, travel, processing, points, parameters, incumbent,
                  route_cache=None, dirty_left=True, dirty_right=True, candidate_costs=None):
    """Identical pair policy, without repeated right-route direction DP.

    A separated pair needs no forbidden-interval construction. The strict
    bounding-box test implies both rail order and proximity constraints for
    every MOVE, processing operation and possible WAIT.
    """
    right = left + 1
    speed, pre, post, dx, dy, eps = parameters
    best, bit0, bit1 = np.inf, np.empty(0, np.int64), np.empty(0, np.int64)
    selected0, selected1 = (-1, -1) if candidate_costs is None else _unique_pair_heads(
        routes, left, candidate_costs, points, parameters)
    if selected0 >= 0:
        if route_cache is not None and not dirty_left:
            c0, bit0 = route_cache[0][left, selected0], route_cache[1][2 * left + selected0]
        else:
            c0, bit0 = _direction_array(routes[left], travel, processing, selected0)
        if route_cache is not None and not dirty_right:
            c1, bit1 = route_cache[0][right, selected1], route_cache[1][2 * right + selected1]
        else:
            c1, bit1 = _direction_array(routes[right], travel, processing, selected1)
        best = c0 + c1
    else:
        costs1 = np.empty(2)
        bits1 = [np.empty(0, np.int64) for _ in range(2)]
        for first1 in range(2 if len(routes[right]) else 1):
            if route_cache is not None and not dirty_right:
                costs1[first1], bits1[first1] = route_cache[0][right, first1], route_cache[1][2 * right + first1]
            else:
                costs1[first1], bits1[first1] = _direction_array(routes[right], travel, processing, first1)
        for first0 in range(2 if len(routes[left]) else 1):
            if route_cache is not None and not dirty_left:
                c0, d0 = route_cache[0][left, first0], route_cache[1][2 * left + first0]
            else:
                c0, d0 = _direction_array(routes[left], travel, processing, first0)
            for first1 in range(2 if len(routes[right]) else 1):
                c1, d1 = costs1[first1], bits1[first1]
                if len(d0) and len(d1):
                    a, b = points[routes[left][0], d0[0]], points[routes[right][0], d1[0]]
                    if a[0] + dx > b[0] + eps:
                        continue
                    if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                        continue
                if c0 + c1 < best:
                    best, bit0, bit1 = c0 + c1, d0, d1
    empty = np.empty((0, 9))
    if not math.isfinite(best):
        return np.inf, bit0, bit1, empty, empty
    if route_cache is not None and not dirty_left:
        a = route_cache[2][2 * left + (bit0[0] if len(bit0) else 0)]
    else:
        a = _numeric_templates(routes[left], bit0, points, processing, speed, pre, post)
    if route_cache is not None and not dirty_right:
        b = route_cache[2][2 * right + (bit1[0] if len(bit1) else 0)]
    else:
        b = _numeric_templates(routes[right], bit1, points, processing, speed, pre, post)
    nominal = max(a[-1, 6] if len(a) else 0., b[-1, 6] if len(b) else 0.)
    if not len(a) or not len(b):
        return nominal, bit0, bit1, a, b
    if nominal >= incumbent - 1.e-8:
        return nominal, bit0, bit1, a, b  # bound only; cannot be accepted
    left_max, right_min = -np.inf, np.inf
    for node in routes[left]:
        left_max = max(left_max, points[node, 0, 0], points[node, 1, 0])
    for node in routes[right]:
        right_min = min(right_min, points[node, 0, 0], points[node, 1, 0])
    if left_max + dx + eps < right_min:
        return nominal, bit0, bit1, a, b
    reuse_forward = None if route_cache is None else (route_cache[3][left], route_cache[3][right], route_cache[4][right])
    followed, wait = _numeric_follow(a, b, left, right, dx, dy, eps, True, True, incumbent, True, reuse_forward)
    reverse, reverse_wait = empty, np.inf
    if not math.isfinite(wait) or wait > 0.:
        reuse_reverse = None if route_cache is None else (route_cache[3][right], route_cache[3][left], route_cache[4][left])
        reverse, reverse_wait = _numeric_follow(b, a, right, left, dx, dy, eps, True, True, incumbent, True, reuse_reverse)
    forward_cmax = max(a[-1, 6], followed[-1, 6]) if math.isfinite(wait) else np.inf
    reverse_cmax = max(b[-1, 6], reverse[-1, 6]) if math.isfinite(reverse_wait) else np.inf
    if reverse_cmax < forward_cmax:
        return reverse_cmax, bit0, bit1, reverse, b
    return forward_cmax, bit0, bit1, a, followed


@njit
def _native_cross_feasible(routes, directions, operations, points, parameters):
    """The cross-rail checks of priority_schedule, with exact spatial skipping."""
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for r in range(2):
        for s in range(2, 4):
            if not len(routes[r]) or not len(routes[s]):
                continue
            a, b = points[routes[r][0], directions[r][0]], points[routes[s][0], directions[s][0]]
            if abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps:
                return False
            j = 0
            for aop in operations[r]:
                while j < len(operations[s]) and operations[s][j, 6] < aop[5]:
                    j += 1
                k = j
                while k < len(operations[s]) and operations[s][k, 5] <= aop[6]:
                    if _numeric_conflict(aop, operations[s][k], r, s, dx, dy, eps):
                        return False
                    k += 1
    return True


@njit
def _native_update(routes, travel, processing, points, parameters,
                   directions, operations, pair_scores, dirty, incumbent, route_cache=None, candidate_costs=None):
    """Reuse the unchanged same-rail pair; never reuse a changed partner's WAIT.

    Each dirty pair uses the same native direction and two-priority rule. Its
    policy is independent of the other rail, which only rejects cross-rail
    interference after both pair timelines have been constructed.
    """
    bits, ops, scores = list(directions), list(operations), pair_scores.copy()
    for pair in range(2):
        if not dirty[2 * pair] and not dirty[2 * pair + 1]:
            continue
        left = 2 * pair
        value, d0, d1, o0, o1 = _native_pair(routes, left, travel, processing, points, parameters,
                                          incumbent, route_cache, dirty[left], dirty[left + 1], candidate_costs)
        scores[pair] = value
        bits[left], bits[left + 1], ops[left], ops[left + 1] = d0, d1, o0, o1
        if math.isfinite(incumbent) and (not math.isfinite(value) or value >= incumbent - 1.e-8):
            return value, bits, ops, scores
    value = max(scores)
    if not math.isfinite(value) or value >= incumbent - 1.e-8:
        return value, bits, ops, scores
    if not _native_cross_feasible(routes, bits, ops, points, parameters):
        value = np.inf
    return value, bits, ops, scores


def _vnd_scratch(capacity, universe):
    """Reusable storage, sized by possible tasks, not a candidate/search cap."""
    width = capacity + 1
    return (np.empty((4, width), np.int64), np.zeros(4, np.int64),
            np.empty((4, width, width, 2, 2)), np.zeros((4, 4, width), np.int64),
            np.zeros((4, width)), np.empty(universe, np.int64), np.empty(universe, np.int64))


@njit
def _refresh_vnd_route(r, route, travel, processing, eligible, nodes, lengths,
                       summaries, invalid, prefix, location_robot, location_index):
    n = len(route)
    nodes[r, :n], lengths[r] = route, n
    prefix[r, 0] = 0.
    invalid[r, :, 0] = 0
    for i in range(n):
        node = route[i]
        location_robot[node], location_index[node] = r, i
        prefix[r, i + 1] = prefix[r, i] + processing[node]
        for target in range(4):
            invalid[r, target, i + 1] = invalid[r, target, i] + int(not eligible[node, target])
        for first in range(2):
            for last in range(2):
                summaries[r, i, i, first, last] = processing[node] if first == last else np.inf
        for j in range(i + 1, n):
            old, new = route[j - 1], route[j]
            for first in range(2):
                for last in range(2):
                    summaries[r, i, j, first, last] = processing[new] + min(
                        summaries[r, i, j - 1, first, 0] + travel[old, new, 0, last],
                        summaries[r, i, j - 1, first, 1] + travel[old, new, 1, last])


@njit
def _vnd_arrays(routes, travel, processing, points, eligible, parent_index, nearest,
                parameters, alpha, max_passes, candidate_cap, conflict_aware, scratch, move_cache_hits=None):
    current = [route.copy() for route in routes]
    directions = [np.empty(0, np.int64) for _ in range(4)]
    native_ops = [np.empty((0, 9)) for _ in range(4)]
    pair_scores = np.zeros(2)
    dirty = np.ones(4, np.bool_)
    if conflict_aware:
        current_score, directions, native_ops, pair_scores = _native_update(
            current, travel, processing, points, parameters, directions, native_ops, pair_scores,
            dirty, np.inf)
    else:
        current_score = max(np.array([_array_route_cost(route, travel, processing) for route in current]))
    nodes, lengths, raw_summaries, invalid, processing_prefix, location_robot, location_index = scratch
    costs = np.zeros(4)
    heads = np.full(4, -1, np.int64)
    head_costs = np.zeros((4, 2))
    route_cache = (np.zeros((4, 2)), [np.empty(0, np.int64) for _ in range(8)],
                   [np.empty((0, 9)) for _ in range(8)], [np.empty((0, 9)) for _ in range(4)],
                   [np.empty((0, 9)) for _ in range(4)])
    passes, operations, accepted = 0, 0, 0
    active = np.zeros(len(processing), np.bool_)
    for route in current:
        for node in route:
            active[node] = True
    active_ids = np.concatenate((current[0], current[1], current[2], current[3]))
    candidate_index = np.full(len(processing), -1, np.int64)
    for index, node in enumerate(active_ids):
        candidate_index[node] = index
    seen_width = len(active_ids) if move_cache_hits is not None else 0
    seen = np.zeros((seen_width, seen_width), np.int64)
    active_nearest = np.full((len(processing), alpha), -1, np.int64)
    for route in current:
        for node in route:
            count = 0
            for other in nearest[node]:
                if other != node and active[other]:
                    active_nearest[node, count] = other
                    count += 1
                    if count == alpha:
                        break
    while True:
        if max_passes > 0 and passes >= max_passes:
            return current, current_score, operations, accepted, True, False
        passes += 1
        for r in range(4):
            if dirty[r]:
                _refresh_vnd_route(r, current[r], travel, processing, eligible, nodes, lengths,
                                   raw_summaries, invalid, processing_prefix, location_robot, location_index)
                costs[r] = _array_route_cost(current[r], travel, processing)
                heads[r] = current[r][0] if lengths[r] else -1
                for first in range(2):
                    head_costs[r, first] = min(raw_summaries[r, 0, lengths[r] - 1, first, 0],
                                               raw_summaries[r, 0, lengths[r] - 1, first, 1]) if lengths[r] else 0.
                    if conflict_aware:
                        option_value, option_bits = _direction_array(current[r], travel, processing, first)
                        route_cache[0][r, first], route_cache[1][2 * r + first] = option_value, option_bits
                        route_cache[2][2 * r + first] = _numeric_templates(
                            current[r], option_bits, points, processing, parameters[0], parameters[1], parameters[2])
        if conflict_aware:
            for left in (0, 2):
                right = left + 1
                if not dirty[left] and not dirty[right]:
                    continue
                for r in (left, right):
                    route_cache[3][r] = (route_cache[2][2 * r + (directions[r][0] if len(directions[r]) else 0)]
                                          if len(directions[r]) == len(current[r]) else np.empty((0, 9)))
                    route_cache[4][r] = np.empty((0, 9))
                a, b = route_cache[3][left], route_cache[3][right]
                if len(a) and len(b):
                    followed, wait = _numeric_follow(a, b, left, right, parameters[3], parameters[4],
                                                     parameters[5], True, True, np.inf, True)
                    route_cache[4][right] = followed
                    if not math.isfinite(wait) or wait > 0.:
                        reverse, _ = _numeric_follow(b, a, right, left, parameters[3], parameters[4],
                                                     parameters[5], True, True, np.inf, True)
                        route_cache[4][left] = reverse
        dirty[:] = False
        # Keep the paper's r_m role based on route cycle length for M1--M3.
        # Move acceptance is exclusively the coordinated native Cmax below.
        longest = int(np.argmax(costs))
        all_ids = np.concatenate((current[0], current[1], current[2], current[3]))
        improved = False
        for move in range(1, 7):
            for u in all_ids:
                ru, iu = location_robot[u], location_index[u]
                if move == 1 and ru != longest:
                    continue
                for v in active_nearest[u]:
                    if v < 0:
                        break
                    rv, iv = location_robot[v], location_index[v]
                    if move == 1 and (ru == rv or ru != longest):
                        continue
                    if move in (2, 3) and (ru == rv or longest not in (ru, rv)):
                        continue
                    if move == 3 and (iu + 1 >= len(current[ru]) or iv + 1 >= len(current[rv])):
                        continue
                    if move == 4 and (ru != rv or iu == iv):
                        continue
                    if move in (5, 6) and ru == rv:
                        continue
                    if candidate_cap > 0 and operations >= candidate_cap:
                        return current, current_score, operations, accepted, False, True
                    # Count actual structurally valid M1--M6 proposals, including
                    # proposals subsequently rejected by eligibility or bounds.
                    operations += 1
                    # M2/M3/M4/M6(u,v) and (v,u) produce the exact same four
                    # routes. Reuse that result only inside this unchanged
                    # pass; alpha membership and first-improvement order stay
                    # intact. M1 and M5 are directional and are never folded.
                    if move_cache_hits is not None and move in (2, 3, 4, 6):
                        index_u, index_v = candidate_index[u], candidate_index[v]
                        stamp = 7 * passes + move
                        if seen[index_u, index_v] == stamp:
                            if move_cache_hits is not None:
                                move_cache_hits[0] += 1
                            continue
                        seen[index_u, index_v], seen[index_v, index_u] = stamp, stamp
                    if move == 4 and abs(iu - iv) == 1:
                        continue  # reversing a singleton leaves the solution identical
                    if move == 6 and iu + 1 == lengths[ru] and iv + 1 == lengths[rv]:
                        continue  # exchanging two empty tails is also identical
                    # Safe, constant-time lower bounds before allocating even
                    # segment descriptors: untouched routes and process alone.
                    untouched = 0.
                    for r in range(4):
                        if r != ru and r != rv:
                            untouched = max(untouched, costs[r])
                    if untouched >= current_score - 1.e-8:
                        continue
                    if conflict_aware and ru // 2 == rv // 2 and pair_scores[1 - ru // 2] >= current_score - 1.e-8:
                        continue
                    pu, pv = processing_prefix[ru, lengths[ru]], processing_prefix[rv, lengths[rv]]
                    if move == 1:
                        pu -= processing[u]
                        pv += processing[u]
                    elif move in (2, 3):
                        width = 1 if move == 2 else 2
                        delta = (processing_prefix[rv, iv + width] - processing_prefix[rv, iv]
                                 - processing_prefix[ru, iu + width] + processing_prefix[ru, iu])
                        pu += delta
                        pv -= delta
                    elif move == 5:
                        before_u, before_v = processing_prefix[ru, iu + 1], processing_prefix[rv, iv + 1]
                        pu, pv = before_u + before_v, pu + pv - before_u - before_v
                    elif move == 6:
                        delta = ((processing_prefix[rv, lengths[rv]] - processing_prefix[rv, iv + 1])
                                 - (processing_prefix[ru, lengths[ru]] - processing_prefix[ru, iu + 1]))
                        pu += delta
                        pv -= delta
                    if max(pu, pv) >= current_score - 1.e-8:
                        continue
                    cu, cv = _raw_move_costs(nodes, lengths, raw_summaries, invalid, parent_index, travel, move, ru, iu, rv, iv)
                    if cu >= current_score - 1.e-8:
                        continue
                    lower = max(cu, cv) if move != 4 else cu
                    for r in range(4):
                        if r != ru and r != rv:
                            lower = max(lower, costs[r])
                    if not math.isfinite(lower) or lower >= current_score - 1.e-8:
                        continue
                    seg_a, seg_b = _move_segments(current, move, ru, iu, rv, iv)
                    changed = np.zeros(4, np.bool_)
                    changed[ru], changed[rv] = True, True
                    if conflict_aware:
                        candidate_heads, candidate_head_costs = heads.copy(), head_costs.copy()
                        candidate_heads[ru], candidate_head_costs[ru] = _segment_head_costs(
                            seg_a, nodes, raw_summaries, invalid, parent_index, travel, ru)
                        if move != 4:
                            candidate_heads[rv], candidate_head_costs[rv] = _segment_head_costs(
                                seg_b, nodes, raw_summaries, invalid, parent_index, travel, rv)
                        bound_ok = True
                        for left in (0, 2):
                            if changed[left] or changed[left + 1]:
                                if not _pair_head_bound(left, candidate_heads, candidate_head_costs, points,
                                                        parameters, native_ops, directions, changed, current_score):
                                    bound_ok = False
                                    break
                        if not bound_ok:
                            continue
                    candidate = list(current)
                    candidate[ru] = _materialize_segments(seg_a, current)
                    if move != 4:
                        candidate[rv] = _materialize_segments(seg_b, current)
                    if not _legal_first_arrays(candidate, points, parameters):
                        continue
                    if conflict_aware:
                        score, bits, ops, pairs = _native_update(
                            candidate, travel, processing, points, parameters,
                            directions, native_ops, pair_scores, changed, current_score, route_cache, candidate_head_costs)
                    else:
                        score = lower
                    if score < current_score - 1.e-8:
                        if conflict_aware:
                            directions, native_ops, pair_scores = bits, ops, pairs
                        current, current_score, improved = candidate, score, True
                        dirty[ru], dirty[rv] = True, True
                        accepted += 1
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return current, current_score, operations, accepted, False, False


@njit
def _oriented_route_cost(route, directions, travel, processing):
    cost = 0.
    for k in range(len(route)):
        if k:
            cost += travel[route[k - 1], route[k], directions[k - 1], directions[k]]
        cost += processing[route[k]]
    return cost


@njit
def _materialize_directions(segments, directions):
    n = 0
    for _, lo, hi, _ in segments:
        n += hi - lo
    result = np.empty(n, np.int64)
    k = 0
    for source, lo, hi, reverse in segments:
        for j in range(hi - lo):
            bit = directions[source][hi - 1 - j if reverse else lo + j]
            result[k] = 1 - bit if reverse else bit
            k += 1
    return result


def _oriented_scratch(capacity, universe):
    width = capacity + 1
    return (np.empty((4, width), np.int64), np.empty((4, width), np.int64),
            np.zeros(4, np.int64), np.zeros((4, width)),
            np.zeros((4, 4, width), np.int64), np.empty(universe, np.int64),
            np.empty(universe, np.int64))


@njit
def _refresh_oriented_route(r, route, bits, travel, processing, eligible, scratch):
    nodes, directions, lengths, prefix, invalid, robot, position = scratch
    n = len(route)
    nodes[r, :n], directions[r, :n], lengths[r] = route, bits, n
    prefix[r, 0], invalid[r, :, 0] = 0., 0
    for i in range(n):
        node = route[i]
        robot[node], position[node] = r, i
        value = processing[node]
        if i:
            value += travel[route[i - 1], node, bits[i - 1], bits[i]]
        prefix[r, i + 1] = prefix[r, i] + value
        for target in range(4):
            invalid[r, target, i + 1] = invalid[r, target, i] + int(not eligible[node, target])


@njit
def _oriented_segment_cost(segments, target, scratch, travel, parent_index):
    nodes, bits, _, prefix, invalid, _, _ = scratch
    value, previous, previous_bit = 0., -1, 0
    for source, lo, hi, reverse in segments:
        if hi <= lo:
            continue
        if invalid[source, target, hi] != invalid[source, target, lo]:
            return np.inf
        first, last = (hi - 1, lo) if reverse else (lo, hi - 1)
        node = nodes[source, first]
        direction = 1 - bits[source, first] if reverse else bits[source, first]
        if previous >= 0:
            if parent_index[previous] == parent_index[node]:
                return np.inf
            value += travel[previous, node, previous_bit, direction]
        value += prefix[source, hi] - prefix[source, lo]
        if lo:
            value -= travel[nodes[source, lo - 1], nodes[source, lo], bits[source, lo - 1], bits[source, lo]]
        previous = nodes[source, last]
        previous_bit = 1 - bits[source, last] if reverse else bits[source, last]
    return value


@njit
def _fixed_follow_progress(nominal, following):
    """Completed operation endpoints, accumulated delay and last gap index."""
    table = np.full((len(nominal) + 1, 3), np.nan)
    table[0] = (0., 0., 0.)
    count, ready, last_gap = 0, 0., 0
    for op in following:
        if op[8] == 4.:
            continue
        if count >= len(nominal):
            break
        if op[5] != ready:
            last_gap = count + 1
        count += 1
        table[count] = (op[6], op[6] - nominal[count - 1, 6], float(last_gap))
        ready = op[6]
    return table


@njit
def _fixed_prefix_reject(fixed, following, prefixes, changed, nominal, progress,
                         costs, incumbent, eps, failures=None):
    """An identical scheduled prefix supplies a safe per-priority Cmax bound."""
    fixed_count = len(nominal[fixed]) if not changed[fixed] else max(0, 4 * prefixes[fixed] - 1)
    following_count = len(nominal[following]) if not changed[following] else max(0, 4 * prefixes[following] - 1)
    if fixed_count == 0 or following_count == 0:
        return False
    table = progress[following]
    if not len(table):
        return False
    until = np.inf if not changed[fixed] else nominal[fixed][fixed_count - 1, 6]
    margin = max(1.e-8, 32. * eps * max(1., abs(until))) if math.isfinite(until) else 0.
    if failures is not None:
        failed = failures[following, 0]
        if 0. <= failed < following_count and failures[following, 1] < until - margin:
            return True  # the same complete operation prefix must fail again
    lo, hi = 0, min(following_count, len(table) - 1)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if math.isfinite(table[mid, 0]) and table[mid, 0] < until - margin:
            lo = mid
        else:
            hi = mid - 1
    guard = 1.e-8 * max(1., abs(incumbent), abs(costs[following]))
    if costs[following] + table[lo, 1] >= incumbent + guard:
        return True
    # When an unchanged follower has already accumulated every gap, its
    # identical, positive-duration tail has the exact old endpoint as a lower
    # bound, even if the changed fixed route later adds another wait.
    return (not changed[following] and math.isfinite(table[-1, 0])
            and table[-1, 2] <= lo and table[-1, 0] >= incumbent - 1.e-8)


@njit
def _fixed_native_pair(routes, directions, left, points, processing, parameters,
                        nominal, follows, dirty, incumbent, skip_priorities=None, failure_out=None):
    """The same two-priority scheduling rule, with explicit direction genes."""
    right = left + 1
    speed, pre, post, dx, dy, eps = parameters
    if failure_out is not None:
        failure_out[left] = (-2., np.nan)
        failure_out[right] = (-2., np.nan)
    a = (_numeric_templates(routes[left], directions[left], points, processing, speed, pre, post)
         if dirty[left] else nominal[left])
    b = (_numeric_templates(routes[right], directions[right], points, processing, speed, pre, post)
         if dirty[right] else nominal[right])
    empty = np.empty((0, 9))
    if len(a) and len(b):
        x, y = points[routes[left][0], directions[left][0]], points[routes[right][0], directions[right][0]]
        if x[0] + dx > y[0] + eps or (abs(x[0] - y[0]) <= dx + eps and abs(x[1] - y[1]) <= dy + eps):
            return np.inf, empty, empty, a, b, empty, empty
    lower = max(a[-1, 6] if len(a) else 0., b[-1, 6] if len(b) else 0.)
    if not len(a) or not len(b) or lower >= incumbent - 1.e-8:
        return lower, a, b, a, b, b, empty
    left_max, right_min = -np.inf, np.inf
    for node in routes[left]:
        left_max = max(left_max, points[node, 0, 0], points[node, 1, 0])
    for node in routes[right]:
        right_min = min(right_min, points[node, 0, 0], points[node, 1, 0])
    if left_max + dx + eps < right_min:
        return lower, a, b, a, b, b, empty
    forward, wait = empty, np.inf
    if skip_priorities is None or not skip_priorities[right]:
        forward, wait = _numeric_follow(a, b, left, right, dx, dy, eps, True, True, incumbent, True,
                                         (nominal[left], nominal[right], follows[right]),
                                         None if failure_out is None else failure_out[right], True)
    reverse, reverse_wait = empty, np.inf
    if (not math.isfinite(wait) or wait > 0.) and (skip_priorities is None or not skip_priorities[left]):
        reverse, reverse_wait = _numeric_follow(b, a, right, left, dx, dy, eps, True, True, incumbent, True,
                                                 (nominal[right], nominal[left], follows[left]),
                                                 None if failure_out is None else failure_out[left], True)
    c0 = max(a[-1, 6], forward[-1, 6]) if math.isfinite(wait) else np.inf
    c1 = max(b[-1, 6], reverse[-1, 6]) if math.isfinite(reverse_wait) else np.inf
    if c1 < c0:
        return c1, reverse, b, a, b, forward, reverse
    return c0, a, forward, a, b, forward, reverse


@njit
def _fixed_native_update(routes, directions, points, processing, parameters,
                          operations, pair_scores, nominal, follows, dirty, incumbent, skip_priorities=None,
                          failure_out=None):
    ops, scores, templates, following = list(operations), pair_scores.copy(), list(nominal), list(follows)
    for pair in range(2):
        left, right = 2 * pair, 2 * pair + 1
        if not dirty[left] and not dirty[right]:
            continue
        value, o0, o1, a, b, forward, reverse = _fixed_native_pair(
            routes, directions, left, points, processing, parameters, nominal, follows, dirty, incumbent,
            skip_priorities, failure_out)
        scores[pair], ops[left], ops[right] = value, o0, o1
        templates[left], templates[right], following[left], following[right] = a, b, reverse, forward
        if math.isfinite(incumbent) and (not math.isfinite(value) or value >= incumbent - 1.e-8):
            return value, ops, scores, templates, following
    value = max(scores)
    if math.isfinite(value) and value < incumbent - 1.e-8:
        if not _native_cross_feasible(routes, directions, ops, points, parameters):
            value = np.inf
    return value, ops, scores, templates, following


@njit
def _vnd_oriented_arrays(routes, directions, travel, processing, points, eligible, parent_index,
                           nearest, parameters, alpha, max_passes, candidate_cap, scratch,
                           move_cache_hits=None, use_schedule_bound=True):
    """Complete oriented M1--M6 VND, scored by the fixed-gene native Cmax.

    Inf-to-inf moves are accepted only when fewer native robot pairs remain
    rejected. This lets two separately blocked pairs be repaired in successive
    moves, without claiming that a native policy rejection proves infeasibility.
    """
    current, bits = [r.copy() for r in routes], [d.copy() for d in directions]
    ops = [np.empty((0, 9)) for _ in range(4)]
    nominal, follows = list(ops), list(ops)
    dirty, pairs = np.ones(4, np.bool_), np.zeros(2)
    failures = np.full((4, 2), np.nan)
    score, ops, pairs, nominal, follows = _fixed_native_update(
        current, bits, points, processing, parameters, ops, pairs, nominal, follows, dirty, np.inf,
        None, failures)
    rejected = int(not math.isfinite(pairs[0])) + int(not math.isfinite(pairs[1]))
    nodes, _, lengths, prefix, _, robot, position = scratch
    costs = np.zeros(4)
    progress = [np.empty((0, 3)) for _ in range(4)]
    active_ids = np.concatenate((current[0], current[1], current[2], current[3]))
    active = np.zeros(len(processing), np.bool_)
    index = np.full(len(processing), -1, np.int64)
    for i, node in enumerate(active_ids):
        active[node], index[node] = True, i
    near = np.full((len(processing), alpha), -1, np.int64)
    for u in active_ids:
        count = 0
        for v in nearest[u]:
            if v != u and active[v]:
                near[u, count] = v
                count += 1
                if count == alpha:
                    break
    width = len(active_ids) if move_cache_hits is not None else 0
    seen = np.zeros((width, width), np.int64)
    passes, operations, accepted = 0, 0, 0
    while True:
        if max_passes > 0 and passes >= max_passes:
            return current, bits, score, operations, accepted, True, False, rejected
        passes += 1
        for r in range(4):
            if dirty[r]:
                _refresh_oriented_route(r, current[r], bits[r], travel, processing, eligible, scratch)
                costs[r] = _oriented_route_cost(current[r], bits[r], travel, processing)
        if use_schedule_bound and math.isfinite(score):
            for left in (0, 2):
                if dirty[left] or dirty[left + 1] or not len(progress[left]):
                    progress[left] = _fixed_follow_progress(nominal[left], follows[left])
                    progress[left + 1] = _fixed_follow_progress(nominal[left + 1], follows[left + 1])
        dirty[:] = False
        longest = int(np.argmax(costs))
        all_ids = np.concatenate((current[0], current[1], current[2], current[3]))
        improved = False
        for move in range(1, 7):
            for u in all_ids:
                ru, iu = robot[u], position[u]
                if move == 1 and ru != longest:
                    continue
                for v in near[u]:
                    if v < 0:
                        break
                    rv, iv = robot[v], position[v]
                    if move == 1 and ru == rv:
                        continue
                    if move in (2, 3) and (ru == rv or longest not in (ru, rv)):
                        continue
                    if move == 3 and (iu + 1 >= lengths[ru] or iv + 1 >= lengths[rv]):
                        continue
                    if move == 4 and (ru != rv or iu == iv):
                        continue
                    if move in (5, 6) and ru == rv:
                        continue
                    if candidate_cap > 0 and operations >= candidate_cap:
                        return current, bits, score, operations, accepted, False, True, rejected
                    operations += 1
                    if move_cache_hits is not None and move in (2, 3, 4, 6):
                        a, b, stamp = index[u], index[v], 7 * passes + move
                        if seen[a, b] == stamp:
                            move_cache_hits[0] += 1
                            continue
                        seen[a, b], seen[b, a] = stamp, stamp
                    if move == 6 and iu + 1 == lengths[ru] and iv + 1 == lengths[rv]:
                        continue
                    if math.isfinite(score):
                        if ru // 2 == rv // 2 and pairs[1 - ru // 2] >= score - 1.e-8:
                            continue
                        untouched = 0.
                        for r in range(4):
                            if r != ru and r != rv and len(nominal[r]):
                                untouched = max(untouched, nominal[r][-1, 6])
                        if untouched >= score - 1.e-8:
                            continue
                    seg_a, seg_b = _move_segments(current, move, ru, iu, rv, iv)
                    cu = _oriented_segment_cost(seg_a, ru, scratch, travel, parent_index)
                    cv = 0. if move == 4 else _oriented_segment_cost(seg_b, rv, scratch, travel, parent_index)
                    lower = max(cu, cv)
                    if not math.isfinite(lower):
                        continue
                    guard = 1.e-8 * max(1., abs(score), abs(lower)) if math.isfinite(score) else 0.
                    if math.isfinite(score) and lower >= score + guard:
                        continue
                    changed = np.zeros(4, np.bool_)
                    changed[ru], changed[rv] = True, True
                    skip_priorities = np.zeros(4, np.bool_)
                    if use_schedule_bound and math.isfinite(score):
                        prefixes, candidate_costs = lengths.copy(), costs.copy()
                        source, lo, hi, reverse = seg_a[0]
                        prefixes[ru] = hi if source == ru and lo == 0 and not reverse else 0
                        candidate_costs[ru] = cu
                        if move != 4:
                            source, lo, hi, reverse = seg_b[0]
                            prefixes[rv] = hi if source == rv and lo == 0 and not reverse else 0
                            candidate_costs[rv] = cv
                        blocked = False
                        for left in (0, 2):
                            right = left + 1
                            if not changed[left] and not changed[right]:
                                continue
                            skip_priorities[right] = _fixed_prefix_reject(left, right, prefixes, changed,
                                nominal, progress, candidate_costs, score, parameters[5], failures)
                            skip_priorities[left] = _fixed_prefix_reject(right, left, prefixes, changed,
                                nominal, progress, candidate_costs, score, parameters[5], failures)
                            if move_cache_hits is not None and len(move_cache_hits) > 2:
                                move_cache_hits[1] += int(skip_priorities[left]) + int(skip_priorities[right])
                            if skip_priorities[left] and skip_priorities[right]:
                                blocked = True
                                break
                        if blocked:
                            if move_cache_hits is not None and len(move_cache_hits) > 2:
                                move_cache_hits[2] += 1
                            continue
                    candidate, candidate_bits = list(current), list(bits)
                    candidate[ru] = _materialize_segments(seg_a, current)
                    candidate_bits[ru] = _materialize_directions(seg_a, bits)
                    if move != 4:
                        candidate[rv] = _materialize_segments(seg_b, current)
                        candidate_bits[rv] = _materialize_directions(seg_b, bits)
                    new_failures = failures.copy()
                    if move_cache_hits is not None and len(move_cache_hits) > 3:
                        move_cache_hits[3] += 1
                    value, new_ops, new_pairs, new_nominal, new_follows = _fixed_native_update(
                        candidate, candidate_bits, points, processing, parameters, ops, pairs,
                        nominal, follows, changed, score, skip_priorities, new_failures)
                    new_rejected = int(not math.isfinite(new_pairs[0])) + int(not math.isfinite(new_pairs[1]))
                    if (value < score - 1.e-8 or (not math.isfinite(score) and not math.isfinite(value)
                                                 and new_rejected < rejected)):
                        current, bits, score = candidate, candidate_bits, value
                        ops, pairs, nominal, follows = new_ops, new_pairs, new_nominal, new_follows
                        failures = new_failures
                        rejected, improved = new_rejected, True
                        dirty[ru], dirty[rv] = True, True
                        accepted += 1
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return current, bits, score, operations, accepted, False, False, rejected


@njit
def _repair_first_arrays(routes, points, parameters):
    current = [route.copy() for route in routes]
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for left in (0, 2):
        right = left + 1
        if not len(current[left]) or not len(current[right]):
            continue
        found, bi, bj, best = False, 0, 0, 2 ** 30
        for i in range(len(current[left])):
            for j in range(len(current[right])):
                if i + j >= best:
                    continue
                for d in range(2):
                    for e in range(2):
                        a, b = points[current[left][i], d], points[current[right][j], e]
                        if a[0] + dx <= b[0] + eps and not (
                                abs(a[0] - b[0]) <= dx + eps and abs(a[1] - b[1]) <= dy + eps):
                            found, bi, bj, best = True, i, j, i + j
        if found:
            for r, pos in ((left, bi), (right, bj)):
                if pos:
                    node = current[r][pos]
                    current[r][1:pos + 1] = current[r][:pos].copy()
                    current[r][0] = node
    return current


def repair_first_tasks(solution, config, *, scope=None):
    blocks = block_map(solution, config)
    ids, index, points, _, _ = _tables_for(blocks, config)
    routes = [np.array([index[b] for b in route.block_ids], np.int64) for route in solution.routes]
    parameters = np.array([config.empty_speed, config.t_pre, config.t_post,
                           config.interference_dx, config.interference_dy, config.numeric_epsilon])
    repaired = _repair_first_arrays(routes, points, parameters)
    if all(np.array_equal(a, b) for a, b in zip(routes, repaired)):
        return solution
    return canonicalize(solution.parents, solution.patterns,
                        {r: tuple(ids[i] for i in route) for r, route in enumerate(repaired)}, config, scope=scope)


def hga_vnd(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    alpha: int,
    max_passes: int | None = None,
    candidate_cap: int | None = None,
    deadline: float | None = None,
    accounting=None,
    scope: FormalScope | None = None,
) -> CanonicalSolution:
    """Legacy route-order audit helper; explicit HGA uses _vnd_oriented_arrays.

    A CanonicalSolution alone does not store direction genes. The formal native
    runner retains them in each oriented individual and in the returned result.
    """
    if deadline is not None and time.perf_counter() >= deadline:
        return solution
    workspace = _HGAWorkspace(solution.parents, config, scope, patterns=solution.patterns)
    routes = workspace.encode(solution)
    move_cache_hits = np.zeros(1, np.int64)
    result, _, operations, _, pass_hit, candidate_hit = _vnd_arrays(
        routes, workspace.travel, workspace.processing, workspace.points, workspace.eligible,
        workspace.parent_index, workspace.nearest, workspace.parameters, alpha,
        max_passes or 0, candidate_cap or 0, True, workspace.vnd_scratch, move_cache_hits)
    if accounting is not None:
        accounting.cheap_candidate_operations += operations
        accounting.vnd_unique_calls += 1
        accounting.vnd_candidate_operations += operations
        accounting.vnd_move_cache_hits += int(move_cache_hits[0])
        accounting.vnd_pass_cap_hits += int(pass_hit)
        accounting.vnd_candidate_cap_hits += int(candidate_hit)
    return workspace.solution(result, workspace.initial_selected)


def _repair_routes(
    parents: Sequence[ParentWeld],
    patterns: Sequence[SplitPattern],
    inherited: Sequence[Sequence[str]],
    config: ScientificConfig,
    rng: random.Random,
    *,
    shuffle_missing: bool,
    scope: FormalScope | None = None,
    excluded_robot: int | None = None,
) -> CanonicalSolution:
    blocks = _blocks_for(parents, patterns, config)
    expected = set(blocks)
    routes: list[list[str]] = [[] for _ in range(4)]
    fixed_x_robot = {
        f"{pattern.parent_id}::{child}": robot
        for pattern in patterns
        if pattern.kind is SplitKind.X_SPLIT
        for child, robot in enumerate(
            (0, 1)
            if pattern.rail is not None and pattern.rail.value == "UPPER"
            else (2, 3)
        )
    }
    seen: set[str] = set()
    for robot in range(4):
        for block_id in inherited[robot]:
            if (
                block_id in expected
                and block_id not in seen
                and robot_is_eligible(blocks[block_id], robot, config)
                and fixed_x_robot.get(block_id, robot) == robot
            ):
                routes[robot].append(block_id)
                seen.add(block_id)
    missing = sorted(expected - seen)
    if shuffle_missing:
        rng.shuffle(missing)
    for block_id in missing:
        if block_id in fixed_x_robot:
            robot = fixed_x_robot[block_id]
            before = _route_cost(routes[robot], blocks, config)
            _, position = min(
                (
                    _route_cost(
                        routes[robot][:position] + [block_id] + routes[robot][position:],
                        blocks,
                        config,
                    ) - before,
                    position,
                )
                for position in range(len(routes[robot]) + 1)
            )
        else:
            robot, position = _best_insertion(
                block_id, routes, blocks, config, shortest_route_only=False, excluded_robot=excluded_robot
            )
        routes[robot].insert(position, block_id)
    # The common canonicalizer intentionally collapses consecutive siblings on
    # one robot to WHOLE.  A selected split chromosome must therefore repair
    # that representation artifact before canonicalization.
    for pattern in patterns:
        if pattern.kind is SplitKind.WHOLE:
            continue
        if pattern.kind is SplitKind.X_SPLIT:
            continue
        siblings = (f"{pattern.parent_id}::0", f"{pattern.parent_id}::1")
        locations = [_location(routes, block_id) for block_id in siblings]
        if locations[0][0] != locations[1][0] or abs(locations[0][1] - locations[1][1]) != 1:
            continue
        source = locations[1][0]
        routes[source].remove(siblings[1])
        choices = []
        for robot in range(4):
            if robot == source or not robot_is_eligible(blocks[siblings[1]], robot, config):
                continue
            before = _route_cost(routes[robot], blocks, config)
            for position in range(len(routes[robot]) + 1):
                candidate = routes[robot][:position] + [siblings[1]] + routes[robot][position:]
                choices.append(
                    (_route_cost(candidate, blocks, config) - before, robot, position)
                )
        if not choices:
            raise ValueError(f"{pattern.parent_id}: cannot preserve split sibling consistency")
        _, robot, position = min(choices)
        routes[robot].insert(position, siblings[1])
    solution = canonicalize(
        parents,
        patterns,
        {robot: tuple(route) for robot, route in enumerate(routes)},
        config,
        scope=scope,
    )
    return repair_first_tasks(solution, config, scope=scope)


def route_based_crossover(
    parent_a: CanonicalSolution,
    parent_b: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    scope: FormalScope | None = None,
) -> CanonicalSolution:
    if parent_a.parents != parent_b.parents:
        raise ValueError("parents must describe the same instance")
    pattern_a = {pattern.parent_id: pattern for pattern in parent_a.patterns}
    pattern_b = {pattern.parent_id: pattern for pattern in parent_b.patterns}
    route_a = rng.randrange(4)
    route_b = rng.randrange(4)
    for block_id in parent_b.routes[route_b].block_ids:
        parent_id = block_id.split('::', 1)[0]
        pattern_a[parent_id] = pattern_b[parent_id]
    patterns = tuple(pattern_a[parent.parent_id] for parent in parent_a.parents)
    inherited = [list(route.block_ids) for route in parent_a.routes]
    inherited[route_a] = list(parent_b.routes[route_b].block_ids)
    replaced = set(inherited[route_a])
    for robot in range(4):
        if robot != route_a:
            inherited[robot] = [item for item in inherited[robot] if item not in replaced]
    return _repair_routes(
        parent_a.parents,
        patterns,
        inherited,
        config,
        rng,
        shuffle_missing=True,
        scope=scope,
        excluded_robot=route_a,
    )


def mutate_optional_y(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
) -> CanonicalSolution:
    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}
    candidates = []
    for parent in solution.parents:
        options = legal_pattern_options(parent, config)
        if any(value.kind is SplitKind.WHOLE for value in options) and any(
            value.kind is SplitKind.Y_SPLIT for value in options
        ):
            candidates.append((parent, options))
    if not candidates:
        return solution
    parent, options = candidates[rng.randrange(len(candidates))]
    current = patterns[parent.parent_id]
    if current.kind is SplitKind.WHOLE:
        patterns[parent.parent_id] = next(
            value for value in options if value.kind is SplitKind.Y_SPLIT
        )
    else:
        patterns[parent.parent_id] = next(
            value for value in options if value.kind is SplitKind.WHOLE
        )
    selected = tuple(patterns[parent.parent_id] for parent in solution.parents)
    inherited = [list(route.block_ids) for route in solution.routes]
    return _repair_routes(
        solution.parents,
        selected,
        inherited,
        config,
        rng,
        shuffle_missing=False,
    )


def mutate_legal_pattern(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    scope: FormalScope,
    *,
    family_ordinal: int,
) -> tuple[CanonicalSolution, SplitKind | None]:
    """One bounded, family-balanced transition from the shared formal catalog."""
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
        return solution, None
    target_kind = families[family_ordinal % len(families)]
    options = sorted(
        by_kind[target_kind], key=lambda item: (item[0].parent_id, item[1].pattern_id)
    )
    parent, target = options[rng.randrange(len(options))]
    current[parent.parent_id] = target
    selected = tuple(current[item.parent_id] for item in solution.parents)
    inherited = [list(route.block_ids) for route in solution.routes]
    return (
        _repair_routes(
            solution.parents,
            selected,
            inherited,
            config,
            rng,
            shuffle_missing=False,
            scope=scope,
        ),
        target_kind,
    )


def route_edge_set(solution: CanonicalSolution) -> frozenset[tuple[int, str, str]]:
    edges = set()
    for route in solution.routes:
        nodes = (f"R{route.robot_id}:START",) + route.block_ids + (
            f"R{route.robot_id}:END",
        )
        edges.update((route.robot_id, left, right) for left, right in zip(nodes, nodes[1:]))
    return frozenset(edges)


def normalized_route_edge_distance(
    left: CanonicalSolution, right: CanonicalSolution
) -> float:
    a = route_edge_set(left)
    b = route_edge_set(right)
    union = a | b
    return 0.0 if not union else len(a ^ b) / len(union)


def _objective_key(individual: _Individual):
    if individual.evaluated is not None and individual.evaluated.metrics is not None:
        metrics = individual.evaluated.metrics
        if individual.evaluated.native_scoring:
            return (0, metrics.cmax)
        return (0, metrics.cmax, metrics.optional_split_count, metrics.process_imbalance,
                metrics.total_empty_travel, metrics.total_waiting,
                metrics.deterministic_id_order, individual.solution.canonical_hash)
    return (1, *individual.proxy)


def binary_tournament(population: Sequence[_Individual], rng: random.Random) -> _Individual:
    first = population[rng.randrange(len(population))]
    second = population[rng.randrange(len(population))]
    return min((first, second), key=_objective_key)


def manage_population(population: list[_Individual], mu: int) -> list[_Individual]:
    objective_order = {
        id(item): rank for rank, item in enumerate(sorted(population, key=_objective_key))
    }
    diversity_values = {}
    for item in population:
        distances = [
            normalized_route_edge_distance(item.solution, other.solution)
            for other in population
            if other is not item
        ]
        diversity_values[id(item)] = sum(distances) / len(distances) if distances else 0.0
    diversity_order = {
        id(item): rank
        for rank, item in enumerate(
            sorted(
                population,
                key=lambda value: (
                    -diversity_values[id(value)], value.solution.canonical_hash
                ),
            )
        )
    }
    return sorted(
        population,
        key=lambda item: (
            _objective_key(item)[0],
            objective_order[id(item)] + diversity_order[id(item)],
            objective_order[id(item)],
            diversity_order[id(item)],
            item.solution.canonical_hash,
        ),
    )[:mu]


@njit
def _insert_arrays(node, routes, eligible, travel, processing, shortest, excluded=-1):
    costs = np.array([_array_route_cost(route, travel, processing) for route in routes])
    minimum = np.inf
    for r in range(4):
        if eligible[node, r] and r != excluded:
            minimum = min(minimum, costs[r])
    if not math.isfinite(minimum):
        excluded = -1  # an X child has one prescribed robot
        for r in range(4):
            if eligible[node, r]:
                minimum = min(minimum, costs[r])
    best, best_delta, robot, position = np.inf, np.inf, -1, 0
    inserted = np.array([node], np.int64)
    for r in range(4):
        if not eligible[node, r] or r == excluded or (shortest and costs[r] > minimum + 1.e-12):
            continue
        prefix, suffix = _route_prefix_suffix(routes[r], travel, processing)
        other_peak = 0.
        for other in range(4):
            if other != r:
                other_peak = max(other_peak, costs[other])
        for k in range(len(routes[r]) + 1):
            value = _splice_cost(routes[r], prefix, suffix, k, k, inserted, travel, processing)
            delta = value - costs[r]
            primary = delta if shortest else max(value, other_peak)
            if primary < best or (primary == best and delta < best_delta):
                best, best_delta, robot, position = primary, delta, r, k
    return robot, position


@njit
def _put_node(routes, node, robot, position):
    route = routes[robot]
    routes[robot] = np.concatenate((route[:position], np.array([node], np.int64), route[position:]))


@njit
def _insert_oriented(node, routes, directions, eligible, travel, processing,
                       parent_index, shortest, excluded=-1):
    costs = np.array([_oriented_route_cost(routes[r], directions[r], travel, processing) for r in range(4)])
    available = np.zeros(4, np.bool_)
    for r in range(4):
        if eligible[node, r] and r != excluded:
            available[r] = True
    if not np.any(available):
        available[:] = eligible[node]
    minimum = np.inf
    for r in range(4):
        if not available[r]:
            continue
        has_position = False
        for k in range(len(routes[r]) + 1):
            if k and parent_index[routes[r][k - 1]] == parent_index[node]:
                continue
            if k < len(routes[r]) and parent_index[routes[r][k]] == parent_index[node]:
                continue
            has_position = True
            break
        available[r] = has_position
        if has_position:
            minimum = min(minimum, costs[r])
    best, best_delta, robot, position, direction = np.inf, np.inf, -1, 0, 0
    for r in range(4):
        if not available[r] or (shortest and costs[r] > minimum + 1.e-12):
            continue
        route, bits = routes[r], directions[r]
        other_peak = 0.
        for other in range(4):
            if other != r:
                other_peak = max(other_peak, costs[other])
        for k in range(len(route) + 1):
            if k and parent_index[route[k - 1]] == parent_index[node]:
                continue
            if k < len(route) and parent_index[route[k]] == parent_index[node]:
                continue
            for bit in range(2):
                delta = processing[node]
                if k:
                    delta += travel[route[k - 1], node, bits[k - 1], bit]
                if k < len(route):
                    delta += travel[node, route[k], bit, bits[k]]
                if k and k < len(route):
                    delta -= travel[route[k - 1], route[k], bits[k - 1], bits[k]]
                primary = delta if shortest else max(costs[r] + delta, other_peak)
                if primary < best or (primary == best and delta < best_delta):
                    best, best_delta, robot, position, direction = primary, delta, r, k, bit
    return robot, position, direction


@njit
def _put_oriented(routes, directions, node, bit, robot, position):
    routes[robot] = np.concatenate((routes[robot][:position], np.array([node], np.int64), routes[robot][position:]))
    directions[robot] = np.concatenate((directions[robot][:position], np.array([bit], np.int64), directions[robot][position:]))


@njit
def _greedy_route_directions(route, travel):
    bits = np.zeros(len(route), np.int64)
    for k in range(1, len(route)):
        a, b = travel[route[k - 1], route[k], bits[k - 1], 0], travel[route[k - 1], route[k], bits[k - 1], 1]
        bits[k] = 0 if a <= b else 1
    return bits


@njit
def _repair_oriented_first(routes, directions, points, travel, parameters, universe, allow_reorder=True):
    current, bits = list(routes), [d.copy() for d in directions]
    dx, dy, eps = parameters[3], parameters[4], parameters[5]
    for stage in range(2 if allow_reorder else 1):
        best_mask, best_flips, best_delta = -1, 5, np.inf
        for mask in range(16):
            valid, flips, delta = True, 0, 0.
            for r in range(4):
                if not len(current[r]):
                    continue
                bit = (mask >> r) & 1
                flips += int(bit != bits[r][0])
                if len(current[r]) > 1:
                    a, b = current[r][0], current[r][1]
                    delta += travel[a, b, bit, bits[r][1]] - travel[a, b, bits[r][0], bits[r][1]]
                start = points[current[r][0], bit]
                for s in range(r + 1, 4):
                    if not len(current[s]):
                        continue
                    other = points[current[s][0], (mask >> s) & 1]
                    if ((r // 2 == s // 2 and start[0] + dx > other[0] + eps) or
                            (abs(start[0] - other[0]) <= dx + eps and abs(start[1] - other[1]) <= dy + eps)):
                        valid = False
                        break
                if not valid:
                    break
            if valid and (flips < best_flips or (flips == best_flips and delta < best_delta)):
                best_mask, best_flips, best_delta = mask, flips, delta
        if best_mask >= 0:
            for r in range(4):
                if len(bits[r]):
                    bits[r][0] = (best_mask >> r) & 1
            return current, bits
        if stage == 0 and allow_reorder:
            by_node = np.zeros(universe, np.int64)
            for r in range(4):
                for k in range(len(current[r])):
                    by_node[current[r][k]] = bits[r][k]
            current = _repair_first_arrays(current, points, parameters)
            bits = [np.array([by_node[node] for node in route], np.int64) for route in current]
    return current, bits  # The native scorer, not this repair, decides rejection.


@njit
def _repair_oriented_numeric(inherited, inherited_bits, selected, pattern_nodes, node_child, parent_index,
                              eligible, travel, processing, points, parameters, excluded):
    routes = [np.empty(0, np.int64) for _ in range(4)]
    directions = [np.empty(0, np.int64) for _ in range(4)]
    seen = np.zeros(len(processing), np.bool_)
    for order in range(4):
        r = excluded if order == 0 and excluded >= 0 else order - 1 if excluded >= 0 and order <= excluded else order
        if r < 0:
            r = order
        retained, bits = [], []
        for k in range(len(inherited[r])):
            old = inherited[r][k]
            pattern, child = selected[parent_index[old]], node_child[old]
            count = 1 if pattern_nodes[pattern, 1] < 0 else 2
            if (child == 0) != (count == 1):
                continue
            node = pattern_nodes[pattern, 0 if child == 0 else child - 1]
            if not seen[node] and eligible[node, r]:
                retained.append(node)
                bits.append(inherited_bits[r][k])
                seen[node] = True
        routes[r], directions[r] = np.array(retained, np.int64), np.array(bits, np.int64)
    missing = []
    for pattern in selected:
        for node in pattern_nodes[pattern]:
            if node >= 0 and not seen[node]:
                missing.append(node)
    pending = np.array(missing, np.int64)
    np.random.shuffle(pending)
    for node in pending:
        robot, position, bit = _insert_oriented(node, routes, directions, eligible, travel, processing,
                                                parent_index, False, excluded)
        if robot < 0:
            raise ValueError("no legal oriented insertion in crossover")
        _put_oriented(routes, directions, node, bit, robot, position)
    routes, directions = _repair_oriented_first(routes, directions, points, travel, parameters, len(processing))
    # Extract adjacent second children before reinsertion. The insertion rule
    # then preserves separation for every selected split family.
    extracted, owners = [], []
    for pattern in selected:
        a, b = pattern_nodes[pattern]
        if b < 0:
            continue
        ra = rb = ia = ib = -1
        for r in range(4):
            for k in range(len(routes[r])):
                if routes[r][k] == a:
                    ra, ia = r, k
                if routes[r][k] == b:
                    rb, ib = r, k
        if rb >= 0 and ra == rb and abs(ia - ib) == 1:
            routes[rb] = np.concatenate((routes[rb][:ib], routes[rb][ib + 1:]))
            directions[rb] = np.concatenate((directions[rb][:ib], directions[rb][ib + 1:]))
            extracted.append(b)
            owners.append(rb)
    for i in range(len(extracted)):
        b = extracted[i]
        robot, position, bit = _insert_oriented(b, routes, directions, eligible, travel, processing,
                                                parent_index, False, owners[i])
        if robot < 0:
            raise ValueError("no legal separated-sibling oriented insertion")
        _put_oriented(routes, directions, b, bit, robot, position)
    return _repair_oriented_first(routes, directions, points, travel, parameters, len(processing), False)


@njit
def _repair_numeric(inherited, selected, pattern_nodes, node_pattern, node_child, parent_index,
                    eligible, travel, processing, points, parameters, excluded):
    routes = [np.empty(0, np.int64) for _ in range(4)]
    seen = np.zeros(len(processing), np.bool_)
    replaced = np.zeros(len(processing), np.bool_)
    # The replacement route wins duplicate removal, as in route-based crossover.
    for order in range(4):
        r = excluded if order == 0 and excluded >= 0 else order - 1 if excluded >= 0 and order <= excluded else order
        if r < 0:
            r = order
        retained = []
        for old in inherited[r]:
            pattern = selected[parent_index[old]]
            child = node_child[old]
            count = 1 if pattern_nodes[pattern, 1] < 0 else 2
            if (child == 0) != (count == 1):
                continue
            node = pattern_nodes[pattern, 0 if child == 0 else child - 1]
            if not seen[node] and eligible[node, r]:
                retained.append(node)
                seen[node] = True
        routes[r] = np.array(retained, np.int64)
    missing = []
    for pattern in selected:
        for node in pattern_nodes[pattern]:
            if node >= 0 and not seen[node]:
                missing.append(node)
    # RNG is seeded at the beginning of each run; this shuffle belongs to crossover.
    missing_array = np.array(missing, np.int64)
    np.random.shuffle(missing_array)
    for node in missing_array:
        robot, position = _insert_arrays(node, routes, eligible, travel, processing, False, excluded)
        if robot < 0:
            raise ValueError("no eligible insertion in crossover")
        _put_node(routes, node, robot, position)
    routes = _repair_first_arrays(routes, points, parameters)
    # Preserve selected split blocks if bringing a first task forward joined siblings.
    for pattern in selected:
        a, b = pattern_nodes[pattern]
        if b < 0:
            continue
        ra = rb = ia = ib = -1
        for r in range(4):
            for k in range(len(routes[r])):
                if routes[r][k] == a:
                    ra, ia = r, k
                if routes[r][k] == b:
                    rb, ib = r, k
        if ra == rb and abs(ia - ib) == 1:
            routes[rb] = np.concatenate((routes[rb][:ib], routes[rb][ib + 1:]))
            robot, position = _insert_arrays(b, routes, eligible, travel, processing, False, rb)
            _put_node(routes, b, robot, position)
    return routes


class _HGAWorkspace:
    """Fixed geometry and qualifications for one layout, shared by every child."""

    def __init__(self, parents, config, scope, *, patterns=None):
        self.parents = tuple(sorted(parents, key=lambda p: p.parent_id))
        self.config, self.scope = config, scope
        catalog = build_legal_pattern_catalog(self.parents, config, scope) if scope is not None else None
        self.patterns = tuple(option for parent in self.parents for option in
                              (legal_pattern_options(parent, config, catalog=catalog) if patterns is None else
                               (next(p for p in patterns if p.parent_id == parent.parent_id),)))
        lookup = {p: i for i, p in enumerate(self.patterns)}
        initial = initial_patterns(self.parents, config, catalog=catalog) if patterns is None else patterns
        self.initial_selected = np.array([lookup[p] for p in initial], np.int64)
        self.options = tuple(tuple(i for i, p in enumerate(self.patterns) if p.parent_id == parent.parent_id)
                             for parent in self.parents)
        parent_lookup = {p.parent_id: i for i, p in enumerate(self.parents)}
        self.pattern_nodes = np.full((len(self.patterns), 2), -1, np.int64)
        blocks, node_pattern, node_child, parent_index = [], [], [], []
        eligible = []
        for i, pattern in enumerate(self.patterns):
            parent = self.parents[parent_lookup[pattern.parent_id]]
            for child, block in enumerate(blocks_for_pattern(parent, pattern, config)):
                self.pattern_nodes[i, child] = len(blocks)
                blocks.append(block)
                node_pattern.append(i)
                node_child.append(0 if pattern.kind is SplitKind.WHOLE else child + 1)
                parent_index.append(parent_lookup[pattern.parent_id])
                mask = [robot_is_eligible(block, r, config) for r in range(4)]
                if pattern.kind is SplitKind.X_SPLIT:
                    fixed = (0 if pattern.rail.value == 'UPPER' else 2) + child
                    mask = [r == fixed for r in range(4)]
                eligible.append(mask)
        self.ids = tuple(b.block_id for b in blocks)
        self.points = np.array([(b.start, b.end) for b in blocks], dtype=float).reshape((-1, 2, 2))
        self.processing = np.array([config.process_time(b.length) for b in blocks])
        self.travel = np.empty((len(blocks), len(blocks), 2, 2))
        for d in range(2):
            for e in range(2):
                delta = self.points[:, 1 - d, None, :] - self.points[None, :, e, :]
                self.travel[:, :, d, e] = np.sqrt(np.sum(delta * delta, axis=-1)) / config.empty_speed
        self.eligible = np.array(eligible, np.bool_)
        self.parent_index = np.array(parent_index, np.int64)
        self.node_pattern, self.node_child = np.array(node_pattern, np.int64), np.array(node_child, np.int64)
        distance = self.travel.min(axis=(2, 3))
        np.fill_diagonal(distance, np.inf)
        # Stable block-id tie breaking; inactive alternative-pattern nodes are
        # filtered before counting the 20 actual alpha-neighbors.
        self.nearest = np.array([sorted(range(len(blocks)), key=lambda j: (distance[i, j], self.ids[j], j))
                                 for i in range(len(blocks))], np.int64)
        self.parameters = np.array([config.empty_speed, config.t_pre, config.t_post,
                                    config.interference_dx, config.interference_dy, config.numeric_epsilon])
        self.vnd_scratch = _vnd_scratch(2 * len(parents), len(self.processing))
        self.oriented_scratch = _oriented_scratch(2 * len(parents), len(self.processing))

    def encode(self, solution):
        selected = {p.parent_id: self.patterns.index(p) for p in solution.patterns}
        lookup = {(self.patterns[self.node_pattern[i]].parent_id, self.ids[i]): i for i in range(len(self.ids))
                  if self.node_pattern[i] == selected[self.patterns[self.node_pattern[i]].parent_id]}
        return [np.array([lookup[(name.split('::', 1)[0], name)] for name in r.block_ids], np.int64)
                for r in solution.routes]

    def solution(self, routes, selected):
        return canonicalize(self.parents, tuple(self.patterns[i] for i in selected),
                            {r: tuple(self.ids[i] for i in route) for r, route in enumerate(routes)},
                            self.config, scope=self.scope)

    def repair(self, routes, selected, excluded=-1, directions=None):
        if directions is not None:
            return _repair_oriented_numeric(routes, directions, selected, self.pattern_nodes, self.node_child,
                                            self.parent_index, self.eligible, self.travel, self.processing,
                                            self.points, self.parameters, excluded)
        return _repair_numeric(routes, selected, self.pattern_nodes, self.node_pattern, self.node_child,
                               self.parent_index, self.eligible, self.travel, self.processing, self.points,
                               self.parameters, excluded)


@njit
def _seed_numeric_rng(seed):
    np.random.seed(seed)


@njit
def _numeric_edges(routes, node_count):
    result = np.empty(len(routes[0]) + len(routes[1]) + len(routes[2]) + len(routes[3]) + 4, np.int64)
    k = 0
    for r in range(4):
        previous = node_count
        for node in routes[r]:
            result[k] = (r * (node_count + 1) + previous) * (node_count + 1) + node
            k += 1
            previous = node
        result[k] = (r * (node_count + 1) + previous) * (node_count + 1) + node_count
        k += 1
    return np.sort(result)


@njit
def _population_diversity(edges):
    n = len(edges)
    diversity = np.zeros(n)
    for a in range(n):
        for b in range(a + 1, n):
            i = j = common = 0
            while i < len(edges[a]) and j < len(edges[b]):
                if edges[a][i] == edges[b][j]:
                    common += 1
                    i += 1
                    j += 1
                elif edges[a][i] < edges[b][j]:
                    i += 1
                else:
                    j += 1
            total = len(edges[a]) + len(edges[b])
            distance = (total - 2 * common) / (total - common)
            diversity[a] += distance
            diversity[b] += distance
    return diversity / max(1, n - 1)


@dataclass
class _NumericIndividual:
    routes: list
    selected: np.ndarray
    cmax: float
    lower: float
    identity: tuple
    edges: np.ndarray
    directions: list | None = None
    rejected_pairs: int = 0


def _remember_vnd(cache, identity, result_identity, individual):
    # One completed VND may introduce TWO distinct keys. Evict after both
    # insertions; evicting just once beforehand lets the cache grow unbounded.
    for key in (identity, result_identity):
        cache.pop(key, None)
        cache[key] = individual
    while len(cache) > 4096:
        cache.pop(next(iter(cache)))


def _run_hga_route_dp(parents, config, settings, *, seed, scope, checkpoints, method_id):
    """The paper's child/VND loop with no CanonicalSolution for rejected moves."""
    rng = random.Random(seed)
    evaluator = CommonBaselineEvaluator(parents, config, scope, time_limit=math.inf,
                                        checkpoints=checkpoints, certify=False,
                                        candidate_evaluator=priority_schedule)
    workspace = _HGAWorkspace(parents, config, scope)
    _seed_numeric_rng(seed % (2 ** 32))
    population, cache = [], {}
    accounting = evaluator.accounting
    last_progress = time.perf_counter()

    def local(routes, selected):
        started = time.perf_counter()
        identity = tuple(tuple(int(n) for n in route) for route in routes)
        # Exact, layout-local memoization: an identical child has the identical
        # deterministic complete VND. Eviction affects memory/time only.
        previous = cache.get(identity)
        if previous is None:
            move_cache_hits = np.zeros(1, np.int64)
            result, score, operations, _, pass_hit, candidate_hit = _vnd_arrays(
                routes, workspace.travel, workspace.processing, workspace.points, workspace.eligible,
                workspace.parent_index, workspace.nearest, workspace.parameters, settings.alpha,
                settings.vnd_max_passes or 0, settings.vnd_candidate_cap or 0, True, workspace.vnd_scratch,
                move_cache_hits)
            accounting.cheap_candidate_operations += operations
            accounting.vnd_unique_calls += 1
            accounting.vnd_candidate_operations += operations
            accounting.vnd_move_cache_hits += int(move_cache_hits[0])
            accounting.vnd_pass_cap_hits += int(pass_hit)
            accounting.vnd_candidate_cap_hits += int(candidate_hit)
            lower = max(_array_route_cost(route, workspace.travel, workspace.processing) for route in result)
            # VND, tournament and survival all use the same native coordinated
            # Cmax. Free route costs supply bounds, the paper's r_m role and
            # population tie breaking, rather than an acceptance objective.
            result_identity = tuple(tuple(int(n) for n in route) for route in result)
            previous = _NumericIndividual(result, selected.copy(), float(score), lower, result_identity,
                                           _numeric_edges(result, len(workspace.processing)))
            # The terminal state also has a proven complete VND result.
            _remember_vnd(cache, identity, result_identity, previous)
        else:
            accounting.vnd_cache_hits += 1
            cache.pop(identity)
            cache[identity] = previous
        accounting.local_search_time += time.perf_counter() - started
        accounting.candidate_count += 1
        if math.isfinite(previous.cmax) and (evaluator.best is None or previous.cmax < evaluator.best.metrics.cmax - 1.e-8):
            solution = workspace.solution(previous.routes, previous.selected)
            evaluator.evaluate(solution, source=f'HGA_NATIVE_{accounting.candidate_count}')
        return previous

    init_started = time.perf_counter()
    for index in range(settings.mu):
        selected = workspace.initial_selected.copy()
        if index == 0:
            solution = initialize_hga_region_seed(parents, config, scope=scope)
            routes = workspace.encode(solution)
        else:
            pending = [int(n) for p in selected for n in workspace.pattern_nodes[p] if n >= 0]
            rng.shuffle(pending)
            routes = [np.empty(0, np.int64) for _ in range(4)]
            for node in pending:
                robot, position = _insert_arrays(node, routes, workspace.eligible, workspace.travel,
                                                 workspace.processing, True)
                _put_node(routes, node, robot, position)
        routes = workspace.repair(routes, selected)
        # Retain the constructed seed as well as its coordinated VND result.
        if index < settings.initialization_reference_limit:
            value = _priority_arrays(routes, workspace.travel, workspace.processing,
                                     workspace.points, workspace.parameters)[0]
            if math.isfinite(value) and (evaluator.best is None or value < evaluator.best.metrics.cmax - 1.e-8):
                evaluator.evaluate(workspace.solution(routes, selected), source=f'HGA_INITIAL_{index}_NATIVE')
        population.append(local(routes, selected))
        now = time.perf_counter()
        if now - last_progress >= 60.:
            print('HGA_INITIALIZATION', len(parents), seed, index + 1, round(evaluator.elapsed, 3),
                  'vnd_unique/cache/operations', accounting.vnd_unique_calls, accounting.vnd_cache_hits,
                  accounting.vnd_candidate_operations, flush=True)
            last_progress = now
    initialization_time = time.perf_counter() - init_started
    initial_candidate = evaluator.best
    initial_calls = accounting.reference_calls
    iterations, attempts = 0, 0
    while iterations < settings.max_iterations:
        attempts += 1
        started = time.perf_counter()
        def tournament():
            a, b = population[rng.randrange(len(population))], population[rng.randrange(len(population))]
            return min((a, b), key=lambda x: (x.cmax, x.lower, x.identity))
        first, second = tournament(), tournament()
        ra, rb = rng.randrange(4), rng.randrange(4)
        selected = first.selected.copy()
        for node in second.routes[rb]:
            parent = workspace.parent_index[node]
            selected[parent] = second.selected[parent]
        inherited = list(first.routes)
        inherited[ra] = second.routes[rb]
        if rng.random() < settings.pattern_mutation_probability:
            accounting.optional_y_mutations_attempted += 1
            by_kind = {kind: [] for kind in SplitKind}
            for parent, options in enumerate(workspace.options):
                for option in options:
                    if option != selected[parent]:
                        by_kind[workspace.patterns[option].kind].append((parent, option))
            families = [k for k in (SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT) if by_kind[k]]
            if families:
                kind = families[attempts % len(families)]
                parent, option = rng.choice(by_kind[kind])
                selected[parent] = option
                accounting.optional_y_mutations_accepted += 1
                accounting.pattern_transition_proposals += 1
                accounting.x_pattern_proposals += int(kind is SplitKind.X_SPLIT)
                accounting.y_pattern_proposals += int(kind is SplitKind.Y_SPLIT)
        try:
            routes = workspace.repair(inherited, selected, ra)
            accounting.crossover_time += time.perf_counter() - started
            child = local(routes, selected)
        except ValueError as error:
            evaluator.reject_construction(f'HGA child {attempts}: {error}')
            continue
        # A valid child has now completed VND (or reused its exact terminal
        # result). Failed construction attempts never consume this budget.
        iterations += 1
        population.append(child)
        if len(population) >= settings.mu + settings.lambda_size:
            started = time.perf_counter()
            objective = sorted(range(len(population)), key=lambda i: (population[i].cmax, population[i].lower,
                                                                       population[i].identity, i))
            objective_rank = {i: rank for rank, i in enumerate(objective)}
            diversity = _population_diversity([x.edges for x in population])
            diversity_order = sorted(range(len(population)), key=lambda i: (-diversity[i], population[i].identity, i))
            diversity_rank = {i: rank for rank, i in enumerate(diversity_order)}
            survivors = sorted(range(len(population)), key=lambda i: (
                not math.isfinite(population[i].cmax), objective_rank[i] + diversity_rank[i],
                objective_rank[i], diversity_rank[i], i))[:settings.mu]
            population = [population[i] for i in survivors]
            accounting.population_management_time += time.perf_counter() - started
            accounting.population_survival_events += 1
        now = time.perf_counter()
        if iterations % 10000 == 0 or now - last_progress >= 60.:
            print('HGA_PROGRESS', len(parents), seed, iterations, round(evaluator.elapsed, 3),
                  evaluator.best.metrics.cmax if evaluator.best else None,
                  'vnd_unique/cache/operations', accounting.vnd_unique_calls, accounting.vnd_cache_hits,
                  accounting.vnd_candidate_operations, flush=True)
            last_progress = now
    return evaluator.finish(method_id=method_id, method_config_hash=canonical_config_hash(settings),
                            iterations=iterations, initialization_time=initialization_time,
                            termination_reason='ITERATION_LIMIT', initial_candidate=initial_candidate,
                            initial_reference_calls=initial_calls,
                            diagnostics=('full M1-M6 VND; native four-robot priority objective; no internal B32',))


def _oriented_identity(routes, directions):
    return tuple(tuple((int(n), int(d)) for n, d in zip(route, bits))
                 for route, bits in zip(routes, directions))


def _run_hga_oriented_native(parents, config, settings, *, seed, scope, checkpoints, method_id,
                             time_limit=math.inf):
    """Explicit oriented individuals; the final experiment wrapper runs B32."""
    rng = random.Random(seed)
    evaluator = CommonBaselineEvaluator(parents, config, scope, time_limit=time_limit,
                                        checkpoints=checkpoints, certify=False, candidate_evaluator=priority_schedule)
    workspace = _HGAWorkspace(parents, config, scope)
    _seed_numeric_rng(seed % (2 ** 32))
    population, cache = [], {}
    accounting = evaluator.accounting
    last_progress = time.perf_counter()

    def record(routes, directions, selected, score, source):
        if not math.isfinite(score) or (evaluator.best is not None and score >= evaluator.best.metrics.cmax - 1.e-8):
            return
        solution = workspace.solution(routes, selected)
        expected = tuple(tuple(workspace.ids[n] for n in route) for route in routes)
        if tuple(route.block_ids for route in solution.routes) != expected:
            raise RuntimeError('oriented individual changed during canonicalization')
        started = time.perf_counter()
        value, vectors, ops, completion = _priority_arrays(
            routes, workspace.travel, workspace.processing, workspace.points, workspace.parameters, directions)
        accounting.scheduler_time += time.perf_counter() - started
        accounting.reference_calls += 1
        if not math.isfinite(value) or abs(value - score) > 1.e-8:
            raise RuntimeError('oriented VND and full native objective disagree')
        schedule = _schedule_from_arrays(value, vectors, ops, completion, workspace.ids,
                                         'NATIVE_PRIORITY_FOUR_ROBOTS_EXPLICIT_DIRECTIONS')
        evaluator.evaluate(solution, source=source, native_schedule=schedule)

    def local(routes, directions, selected):
        started = time.perf_counter()
        identity = _oriented_identity(routes, directions)
        previous = cache.get(identity)
        if previous is None:
            hits = np.zeros(4, np.int64)
            result, bits, score, operations, _, pass_hit, candidate_hit, rejected = _vnd_oriented_arrays(
                routes, directions, workspace.travel, workspace.processing, workspace.points, workspace.eligible,
                workspace.parent_index, workspace.nearest, workspace.parameters, settings.alpha,
                settings.vnd_max_passes or 0, settings.vnd_candidate_cap or 0, workspace.oriented_scratch, hits)
            accounting.cheap_candidate_operations += operations
            accounting.vnd_candidate_operations += operations
            accounting.vnd_native_candidate_evaluations += int(hits[3])
            accounting.vnd_unique_calls += 1
            accounting.vnd_move_cache_hits += int(hits[0])
            accounting.vnd_priority_bound_rejections += int(hits[1])
            accounting.vnd_schedule_bound_prunes += int(hits[2])
            accounting.vnd_pass_cap_hits += int(pass_hit)
            accounting.vnd_candidate_cap_hits += int(candidate_hit)
            terminal = _oriented_identity(result, bits)
            lower = max(_oriented_route_cost(r, d, workspace.travel, workspace.processing) for r, d in zip(result, bits))
            encoded = [2 * r + d for r, d in zip(result, bits)]
            previous = _NumericIndividual(result, selected.copy(), float(score), lower, terminal,
                                           _numeric_edges(encoded, 2 * len(workspace.processing)), bits, rejected)
            _remember_vnd(cache, identity, terminal, previous)
        else:
            accounting.vnd_cache_hits += 1
            cache.pop(identity)
            cache[identity] = previous
        accounting.local_search_time += time.perf_counter() - started
        accounting.candidate_count += 1
        accounting.remaining_deadlock += int(not math.isfinite(previous.cmax))
        record(previous.routes, previous.directions, previous.selected, previous.cmax,
               f'HGA_ORIENTED_NATIVE_{accounting.candidate_count}')
        return previous

    init_started = time.perf_counter()
    for index in range(settings.mu):
        if evaluator.expired:
            break
        selected = workspace.initial_selected.copy()
        if index == 0:
            solution = initialize_hga_region_seed(parents, config, scope=scope)
            routes = workspace.encode(solution)
            directions = [_greedy_route_directions(route, workspace.travel) for route in routes]
        else:
            pending = [int(n) for p in selected for n in workspace.pattern_nodes[p] if n >= 0]
            rng.shuffle(pending)
            routes = [np.empty(0, np.int64) for _ in range(4)]
            directions = [np.empty(0, np.int64) for _ in range(4)]
            for node in pending:
                robot, position, bit = _insert_oriented(node, routes, directions, workspace.eligible,
                    workspace.travel, workspace.processing, workspace.parent_index, True)
                if robot < 0:
                    raise ValueError('no legal oriented initialization insertion')
                _put_oriented(routes, directions, node, bit, robot, position)
        routes, directions = workspace.repair(routes, selected, directions=directions)
        if index < settings.initialization_reference_limit:
            value = _priority_arrays(routes, workspace.travel, workspace.processing,
                                      workspace.points, workspace.parameters, directions)[0]
            record(routes, directions, selected, value, f'HGA_ORIENTED_INITIAL_{index}')
        population.append(local(routes, directions, selected))
        now = time.perf_counter()
        if now - last_progress >= 60.:
            print('HGA_ORIENTED_INITIALIZATION', len(parents), seed, index + 1, round(evaluator.elapsed, 3),
                  'vnd_unique/cache/operations', accounting.vnd_unique_calls, accounting.vnd_cache_hits,
                  accounting.vnd_candidate_operations, flush=True)
            last_progress = now
    initialization_time = time.perf_counter() - init_started
    initial_candidate, initial_calls = evaluator.best, accounting.reference_calls
    iterations, attempts = 0, 0

    def objective(item):
        return item.cmax, item.rejected_pairs, item.lower, item.identity

    while iterations < settings.max_iterations and population and not evaluator.expired:
        attempts += 1
        started = time.perf_counter()
        def tournament():
            return min((population[rng.randrange(len(population))], population[rng.randrange(len(population))]), key=objective)
        first, second = tournament(), tournament()
        ra, rb = rng.randrange(4), rng.randrange(4)
        selected = first.selected.copy()
        for node in second.routes[rb]:
            parent = workspace.parent_index[node]
            selected[parent] = second.selected[parent]
        inherited, inherited_bits = list(first.routes), list(first.directions)
        inherited[ra], inherited_bits[ra] = second.routes[rb], second.directions[rb]
        if rng.random() < settings.pattern_mutation_probability:
            accounting.optional_y_mutations_attempted += 1
            by_kind = {kind: [] for kind in SplitKind}
            for parent, options in enumerate(workspace.options):
                for option in options:
                    if option != selected[parent]:
                        by_kind[workspace.patterns[option].kind].append((parent, option))
            families = [kind for kind in (SplitKind.WHOLE, SplitKind.Y_SPLIT, SplitKind.X_SPLIT) if by_kind[kind]]
            if families:
                kind = families[attempts % len(families)]
                parent, option = rng.choice(by_kind[kind])
                selected[parent] = option
                accounting.optional_y_mutations_accepted += 1
                accounting.pattern_transition_proposals += 1
                accounting.x_pattern_proposals += int(kind is SplitKind.X_SPLIT)
                accounting.y_pattern_proposals += int(kind is SplitKind.Y_SPLIT)
        try:
            routes, directions = workspace.repair(inherited, selected, ra, directions=inherited_bits)
            accounting.crossover_time += time.perf_counter() - started
            child = local(routes, directions, selected)
        except ValueError as error:
            evaluator.reject_construction(f'HGA oriented child {attempts}: {error}')
            continue
        iterations += 1  # only a successfully constructed, completely improved child
        population.append(child)
        if len(population) >= settings.mu + settings.lambda_size:
            started = time.perf_counter()
            order = sorted(range(len(population)), key=lambda i: (*objective(population[i]), i))
            objective_rank = {i: rank for rank, i in enumerate(order)}
            diversity = _population_diversity([item.edges for item in population])
            order = sorted(range(len(population)), key=lambda i: (-diversity[i], population[i].identity, i))
            diversity_rank = {i: rank for rank, i in enumerate(order)}
            survivors = sorted(range(len(population)), key=lambda i: (
                not math.isfinite(population[i].cmax), objective_rank[i] + diversity_rank[i],
                objective_rank[i], diversity_rank[i], i))[:settings.mu]
            population = [population[i] for i in survivors]
            accounting.population_management_time += time.perf_counter() - started
            accounting.population_survival_events += 1
        now = time.perf_counter()
        if iterations % 10000 == 0 or now - last_progress >= 60.:
            print('HGA_ORIENTED_PROGRESS', len(parents), seed, iterations, round(evaluator.elapsed, 3),
                  evaluator.best.metrics.cmax if evaluator.best else None, 'vnd_unique/cache/operations',
                  accounting.vnd_unique_calls, accounting.vnd_cache_hits, accounting.vnd_candidate_operations, flush=True)
            last_progress = now
    return evaluator.finish(method_id=method_id, method_config_hash=canonical_config_hash(settings),
        iterations=iterations, initialization_time=initialization_time,
        termination_reason='TIME_LIMIT' if evaluator.expired else 'ITERATION_LIMIT',
        initial_candidate=initial_candidate, initial_reference_calls=initial_calls,
        diagnostics=('explicit oriented genes; complete M1-M6 VND; same fixed-gene native Cmax in VND and population',
                     'only native rejected-pair progress breaks nonfinite-Cmax ties; no internal B32'))


def _run_hga_native(parents, config, settings, *, seed, scope, checkpoints, method_id):
    run = _run_hga_oriented_native if settings.direction_encoding == 'EXPLICIT' else _run_hga_route_dp
    return run(parents, config, settings, seed=seed, scope=scope, checkpoints=checkpoints, method_id=method_id)


def run_adapted_hga(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    hga_config: AdaptedHGAConfig,
    *,
    seed: int,
    time_limit: float,
    scope: FormalScope,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    common_seed: EvaluatedCandidate | None = None,
    method_id: str = METHOD_ID,
    certify: bool = False,
    candidate_evaluator: CandidateEvaluator | None = priority_schedule,
) -> BaselineResult:
    if candidate_evaluator is priority_schedule and not certify and common_seed is None:
        if hga_config.direction_encoding == 'EXPLICIT':
            return _run_hga_oriented_native(parents, scientific_config, hga_config, seed=seed,
                scope=scope, checkpoints=checkpoints, method_id=method_id, time_limit=time_limit)
        if math.isinf(time_limit):
            return _run_hga_native(parents, scientific_config, hga_config, seed=seed,
                                   scope=scope, checkpoints=checkpoints, method_id=method_id)
    if hga_config.direction_encoding == 'EXPLICIT':
        raise ValueError('explicit-direction HGA requires native priority scoring without certification or an external seed; '
                         'use direction_encoding=LEGACY_ROUTE_DP for the old certified/evaluator API')
    rng = random.Random(seed)
    v2 = scope.x_split_rule_id is not None
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
    population: list[_Individual] = []
    init_started = time.perf_counter()
    if common_seed is not None:
        injected = evaluator.inject_certified(
            common_seed, source="HGA_COMMON_CERTIFIED_SEED"
        )
        population.append(
            _Individual(
                injected.solution,
                hga_surrogate(injected.solution, scientific_config),
                injected,
            )
        )
    for index in range(1 if common_seed is not None else 0, hga_config.mu):
        if evaluator.expired:
            break
        construct_started = time.perf_counter()
        try:
            solution = (
                initialize_hga_region_seed(
                    parents, scientific_config, scope=scope if v2 else None
                )
                if index == 0
                else initialize_hga_solution(
                    parents, scientific_config, rng, scope=scope if v2 else None
                )
            )
            native_evaluated = None
            if (
                common_seed is None
                and index < hga_config.initialization_reference_limit
                and not evaluator.expired
            ):
                native_evaluated = evaluator.evaluate(
                    solution, source=f"HGA_INITIALIZATION_{index}_NATIVE"
                )
            elif common_seed is None and index >= hga_config.initialization_reference_limit:
                evaluator.accounting.initialization_reference_limit_hits += 1
            native_hash = solution.canonical_hash
            local_started = time.perf_counter()
            solution = hga_vnd(
                solution,
                scientific_config,
                alpha=hga_config.alpha,
                max_passes=hga_config.vnd_max_passes,
                candidate_cap=hga_config.vnd_candidate_cap,
                deadline=evaluator.deadline,
                accounting=evaluator.accounting,
                scope=scope if v2 else None,
            )
            evaluator.accounting.local_search_time += time.perf_counter() - local_started
        except (ArithmeticError, OverflowError, ValueError) as error:
            evaluator.reject_construction(f"HGA initialization {index}: {error}")
            continue
        evaluator.accounting.construction_time += time.perf_counter() - construct_started
        individual = _Individual(solution, hga_surrogate(solution, scientific_config))
        if solution.canonical_hash == native_hash:
            individual.evaluated = native_evaluated
        elif (
            common_seed is None
            and index < hga_config.initialization_reference_limit
            and not evaluator.expired
        ):
            individual.evaluated = evaluator.evaluate(
                solution, source=f"HGA_INITIALIZATION_{index}_VND"
            )
        population.append(individual)
    initialization_time = time.perf_counter() - init_started
    initial_candidate = evaluator.best
    initial_reference_calls = evaluator.accounting.reference_calls

    iterations = 0
    while (
        population
        and not evaluator.expired
        and iterations < hga_config.max_iterations
    ):
        first = binary_tournament(population, rng)
        second = binary_tournament(population, rng)
        crossover_started = time.perf_counter()
        try:
            offspring = route_based_crossover(
                first.solution,
                second.solution,
                scientific_config,
                rng,
                scope=scope if v2 else None,
            )
            if rng.random() < hga_config.pattern_mutation_probability:
                evaluator.accounting.optional_y_mutations_attempted += 1
                before_patterns = offspring.patterns
                if v2:
                    offspring, proposed_kind = mutate_legal_pattern(
                        offspring,
                        scientific_config,
                        rng,
                        scope,
                        family_ordinal=iterations,
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
                    offspring = mutate_optional_y(offspring, scientific_config, rng)
                if offspring.patterns != before_patterns:
                    evaluator.accounting.optional_y_mutations_accepted += 1
        except (ArithmeticError, OverflowError, ValueError) as error:
            evaluator.reject_construction(f"HGA crossover iteration {iterations}: {error}")
            continue
        evaluator.accounting.crossover_time += time.perf_counter() - crossover_started
        local_started = time.perf_counter()
        offspring = hga_vnd(
            offspring,
            scientific_config,
            alpha=hga_config.alpha,
            max_passes=hga_config.vnd_max_passes,
            candidate_cap=hga_config.vnd_candidate_cap,
            deadline=evaluator.deadline,
            accounting=evaluator.accounting,
            scope=scope if v2 else None,
        )
        evaluator.accounting.local_search_time += time.perf_counter() - local_started
        iterations += 1
        evaluated = evaluator.evaluate(offspring, source=f"HGA_OFFSPRING_{iterations}")
        population.append(
            _Individual(offspring, hga_surrogate(offspring, scientific_config), evaluated)
        )
        if len(population) >= hga_config.mu + hga_config.lambda_size:
            management_started = time.perf_counter()
            population = manage_population(population, hga_config.mu)
            evaluator.accounting.population_management_time += (
                time.perf_counter() - management_started
            )
            evaluator.accounting.population_survival_events += 1

    if evaluator.best is None and evaluator.accounting.numeric_failure:
        termination_reason = "NUMERIC_FAILURE"
    elif evaluator.best is None:
        termination_reason = "INITIALIZATION_FAILED"
    elif evaluator.expired:
        termination_reason = "TIME_LIMIT"
    elif iterations >= hga_config.max_iterations:
        termination_reason = "ITERATION_LIMIT"
    else:
        termination_reason = "COMPLETED_OTHER"

    return evaluator.finish(
        method_id=method_id,
        method_config_hash=canonical_config_hash(hga_config),
        iterations=iterations,
        initialization_time=initialization_time,
        termination_reason=termination_reason,
        initial_candidate=initial_candidate,
        initial_reference_calls=initial_reference_calls,
        diagnostics=(
            "one four-quadrant HGA region seed adapts paper region division; paper start/return physics excluded",
            "ADAPTED_HGA_BIASED_FITNESS_V1",
        ),
    )


def run_adapted_hga_v2(
    parents: Sequence[ParentWeld],
    scientific_config: ScientificConfig,
    hga_config: AdaptedHGAConfig,
    *,
    seed: int,
    time_limit: float,
    checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    certify: bool = False,
    candidate_evaluator: CandidateEvaluator | None = priority_schedule,
) -> BaselineResult:
    return run_adapted_hga(
        parents,
        scientific_config,
        hga_config,
        seed=seed,
        time_limit=time_limit,
        scope=FORMAL_SCOPE_V2,
        checkpoints=checkpoints,
        method_id=METHOD_ID_V2,
        certify=certify,
        candidate_evaluator=candidate_evaluator,
    )
