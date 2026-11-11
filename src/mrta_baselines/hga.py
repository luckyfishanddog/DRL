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
    _priority_arrays, _route_prefix_suffix, _splice_cost,
    _tables_for, priority_schedule,
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

    def __post_init__(self) -> None:
        if min(self.mu, self.lambda_size, self.alpha, self.max_iterations) <= 0:
            raise ValueError("HGA integer parameters must be positive")
        if self.initialization_reference_limit <= 0:
            raise ValueError("initialization_reference_limit must be positive")
        if any(x is not None and x <= 0 for x in (self.vnd_max_passes, self.vnd_candidate_cap)):
            raise ValueError("VND bounds must be positive")
        if not 0.0 <= self.pattern_mutation_probability <= 1.0:
            raise ValueError("pattern_mutation_probability must be in [0,1]")


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
def _vnd_arrays(routes, travel, processing, points, eligible, parent_index, nearest,
                parameters, alpha, max_passes, candidate_cap, conflict_aware):
    current = [route.copy() for route in routes]
    if conflict_aware:
        current_score = _priority_arrays(current, travel, processing, points, parameters)[0]
    else:
        current_score = 0.
        for route in current:
            current_score = max(current_score, _array_route_cost(route, travel, processing))
    passes, operations, accepted = 0, 0, 0
    active = np.zeros(len(processing), np.bool_)
    for route in current:
        for node in route:
            active[node] = True
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
        summaries = [_interval_costs(route, travel, processing) for route in current]
        costs = np.array([_array_route_cost(route, travel, processing) for route in current])
        processing_prefix = [np.zeros(len(route) + 1) for route in current]
        for r in range(4):
            for k in range(len(current[r])):
                processing_prefix[r][k + 1] = processing_prefix[r][k] + processing[current[r][k]]
        longest = int(np.argmax(costs))
        location_robot = np.full(len(processing), -1, np.int64)
        location_index = np.zeros(len(processing), np.int64)
        width = max(len(current[0]), len(current[1]), len(current[2]), len(current[3])) + 1
        invalid = np.zeros((4, 4, width), np.int64)
        all_ids = np.concatenate((current[0], current[1], current[2], current[3]))
        nodes = np.full((4, width), -1, np.int64)
        lengths = np.array([len(route) for route in current], np.int64)
        raw_summaries = np.zeros((4, width, width, 2, 2))
        for r in range(4):
            nodes[r, :lengths[r]] = current[r]
            raw_summaries[r, :lengths[r], :lengths[r], :, :] = summaries[r]
        for r in range(4):
            for i in range(len(current[r])):
                node = current[r][i]
                location_robot[node], location_index[node] = r, i
                for target in range(4):
                    invalid[r, target, i + 1] = invalid[r, target, i] + int(not eligible[node, target])
        improved = False
        for move in range(1, 7):
            for u in all_ids:
                ru, iu = location_robot[u], location_index[u]
                if move in (1, 4) and ru != longest:
                    continue
                for v in active_nearest[u]:
                    if v < 0:
                        break
                    rv, iv = location_robot[v], location_index[v]
                    if candidate_cap > 0 and operations >= candidate_cap:
                        return current, current_score, operations, accepted, False, True
                    operations += 1
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
                    # Safe, constant-time lower bounds before allocating even
                    # segment descriptors: untouched routes and process alone.
                    untouched = 0.
                    for r in range(4):
                        if r != ru and r != rv:
                            untouched = max(untouched, costs[r])
                    if untouched >= current_score - 1.e-8:
                        continue
                    pu, pv = processing_prefix[ru][-1], processing_prefix[rv][-1]
                    if move == 1:
                        pu -= processing[u]
                        pv += processing[u]
                    elif move in (2, 3):
                        width = 1 if move == 2 else 2
                        delta = (processing_prefix[rv][iv + width] - processing_prefix[rv][iv]
                                 - processing_prefix[ru][iu + width] + processing_prefix[ru][iu])
                        pu += delta
                        pv -= delta
                    elif move == 5:
                        before_u, before_v = processing_prefix[ru][iu + 1], processing_prefix[rv][iv + 1]
                        pu, pv = before_u + before_v, pu + pv - before_u - before_v
                    elif move == 6:
                        delta = ((processing_prefix[rv][-1] - processing_prefix[rv][iv + 1])
                                 - (processing_prefix[ru][-1] - processing_prefix[ru][iu + 1]))
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
                    candidate = list(current)
                    candidate[ru] = _materialize_segments(seg_a, current)
                    if move != 4:
                        candidate[rv] = _materialize_segments(seg_b, current)
                    if not _legal_first_arrays(candidate, points, parameters):
                        continue
                    score = _priority_arrays(candidate, travel, processing, points, parameters)[0] if conflict_aware else lower
                    if score < current_score - 1.e-8:
                        current, current_score, improved = candidate, score, True
                        accepted += 1
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            return current, current_score, operations, accepted, False, False


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
    if deadline is not None and time.perf_counter() >= deadline:
        return solution
    workspace = _HGAWorkspace(solution.parents, config, scope, patterns=solution.patterns)
    routes = workspace.encode(solution)
    result, _, operations, _, pass_hit, candidate_hit = _vnd_arrays(
        routes, workspace.travel, workspace.processing, workspace.points, workspace.eligible,
        workspace.parent_index, workspace.nearest, workspace.parameters, alpha,
        max_passes or 0, candidate_cap or 0, False)
    if accounting is not None:
        accounting.cheap_candidate_operations += operations
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

    def repair(self, routes, selected, excluded=-1):
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


def _run_hga_native(parents, config, settings, *, seed, scope, checkpoints, method_id):
    """The paper's child/VND loop with no CanonicalSolution for rejected moves."""
    rng = random.Random(seed)
    evaluator = CommonBaselineEvaluator(parents, config, scope, time_limit=math.inf,
                                        checkpoints=checkpoints, certify=False,
                                        candidate_evaluator=priority_schedule)
    workspace = _HGAWorkspace(parents, config, scope)
    _seed_numeric_rng(seed % (2 ** 32))
    population, cache = [], {}
    accounting = evaluator.accounting

    def local(routes, selected):
        started = time.perf_counter()
        identity = tuple(tuple(int(n) for n in route) for route in routes)
        # Exact, layout-local memoization: an identical child has the identical
        # deterministic complete VND. Eviction affects memory/time only.
        previous = cache.get(identity)
        if previous is None:
            result, score, operations, _, pass_hit, candidate_hit = _vnd_arrays(
                routes, workspace.travel, workspace.processing, workspace.points, workspace.eligible,
                workspace.parent_index, workspace.nearest, workspace.parameters, settings.alpha,
                settings.vnd_max_passes or 0, settings.vnd_candidate_cap or 0, False)
            accounting.cheap_candidate_operations += operations
            accounting.vnd_pass_cap_hits += int(pass_hit)
            accounting.vnd_candidate_cap_hits += int(candidate_hit)
            lower = max(_array_route_cost(route, workspace.travel, workspace.processing) for route in result)
            # Preserve the paper's cycle-time VND; score every complete child
            # for MOVE/WAIT/interference before tournament and survival.
            score = _priority_arrays(result, workspace.travel, workspace.processing,
                                     workspace.points, workspace.parameters)[0]
            result_identity = tuple(tuple(int(n) for n in route) for route in result)
            previous = _NumericIndividual(result, selected.copy(), float(score), lower, result_identity,
                                           _numeric_edges(result, len(workspace.processing)))
            if len(cache) >= 4096:
                cache.pop(next(iter(cache)))
            cache[identity] = previous
            # The terminal state also has a proven complete VND result.
            cache[result_identity] = previous
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
        # As in the original initialization path, retain a feasible constructed
        # seed as well as its cycle-time VND result. A route-only improvement
        # must not erase the best conflict-feasible complete initial solution.
        if index < settings.initialization_reference_limit:
            value = _priority_arrays(routes, workspace.travel, workspace.processing,
                                     workspace.points, workspace.parameters)[0]
            if math.isfinite(value) and (evaluator.best is None or value < evaluator.best.metrics.cmax - 1.e-8):
                evaluator.evaluate(workspace.solution(routes, selected), source=f'HGA_INITIAL_{index}_NATIVE')
        population.append(local(routes, selected))
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
        if iterations % 10000 == 0:
            print('HGA_PROGRESS', len(parents), seed, iterations, round(evaluator.elapsed, 3),
                  evaluator.best.metrics.cmax if evaluator.best else None, flush=True)
    return evaluator.finish(method_id=method_id, method_config_hash=canonical_config_hash(settings),
                            iterations=iterations, initialization_time=initialization_time,
                            termination_reason='ITERATION_LIMIT', initial_candidate=initial_candidate,
                            initial_reference_calls=initial_calls,
                            diagnostics=('full M1-M6 VND; native four-robot priority objective; no internal B32',))


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
    certify: bool = True,
    candidate_evaluator: CandidateEvaluator | None = None,
) -> BaselineResult:
    if (candidate_evaluator is priority_schedule and not certify
            and math.isinf(time_limit) and common_seed is None):
        return _run_hga_native(parents, scientific_config, hga_config, seed=seed,
                               scope=scope, checkpoints=checkpoints, method_id=method_id)
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
    certify: bool = True,
    candidate_evaluator: CandidateEvaluator | None = None,
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
