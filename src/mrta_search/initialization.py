from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
import random
import time

from mrta_reference.certifier import CertificationReport, certify_schedule
from mrta_reference.model import FormalScope
from mrta_reference.scheduler import resolve_reference_evaluator
from mrta_reference.geometry import (
    blocks_for_pattern,
    finite_x_split_validator,
    generate_y_split_patterns,
    oriented_endpoints,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    OperationKind,
    ParentWeld,
    Route,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.scheduler import reference_schedule
from mrta_reference.solution import canonicalize
from mrta_reference.solution import block_map, official_metrics

from .direction import (
    ConstrainedDirectionResult,
    DirectionStatus,
    DirectionVectors,
    _initial_error,
    optimize_directions_with_initial_feasibility,
)
from .stats import SearchStats


class InitializationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


class InitializationStrategy(str, Enum):
    LOAD_FIRST = "LOAD_FIRST"
    RAIL_BALANCED = "RAIL_BALANCED"
    X_ORDER_AWARE = "X_ORDER_AWARE"
    SPATIAL_SPREAD = "SPATIAL_SPREAD"
    RAIL_MONOTONE_BALANCED = "RAIL_MONOTONE_BALANCED_BOOTSTRAP"
    REGION_COOPERATIVE_INITIALIZATION = "REGION_COOPERATIVE_INITIALIZATION"


RAIL_SERIAL_BOOTSTRAP = "RAIL_SERIAL_BOOTSTRAP"
V3_DIRECTION_FEASIBILITY_REFERENCE_BUDGET = 4
CLEARANCE_BALANCED_BOOTSTRAP = "CLEARANCE_BALANCED_BOOTSTRAP"


@dataclass(frozen=True)
class InitializationAttempt:
    construction_index: int
    solution: CanonicalSolution | None
    directions: ConstrainedDirectionResult | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    diagnostics: tuple[str, ...] = ()
    duplicate: bool = False
    strategy: str = InitializationStrategy.LOAD_FIRST.value


@dataclass(frozen=True)
class InitializationResult:
    status: InitializationStatus
    solution: CanonicalSolution | None
    directions: DirectionVectors | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    attempts: tuple[InitializationAttempt, ...]
    winning_strategy: str | None = None


@dataclass(frozen=True)
class JRRBCandidate:
    """One complete, cheap-only joint rail/region/route reconstruction."""

    solution: CanonicalSolution
    cheap_score: tuple[float, ...]
    components: tuple[float, ...]
    b_up: float
    b_low: float
    window: tuple[float, float]
    scheme: str
    changed_parent_ids: tuple[str, ...]
    cross_rail_parent_changes: int
    same_rail_robot_changes: int
    reordered_routes: int
    b_up_changed: bool
    b_low_changed: bool


@dataclass(frozen=True)
class JRRBConstruction:
    attempted: int
    constructed: int
    valid_unique: int
    candidates: tuple[JRRBCandidate, ...]
    rejection_reasons: tuple[tuple[str, int], ...] = ()


ReferenceEvaluator = Callable[..., ScheduleResult]
Certifier = Callable[..., CertificationReport]


def bounded_insertion_positions(length: int, limit: int) -> tuple[int, ...]:
    if limit < 2:
        raise ValueError("I_init must be at least 2")
    if length + 1 <= limit:
        return tuple(range(length + 1))
    return tuple(sorted({math.floor(index * length / (limit - 1)) for index in range(limit)}))


def _patterns(parents: Sequence[ParentWeld], config: ScientificConfig):
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        if whole_eligible_rails(parent.start, parent.end, config):
            result.append(SplitPattern(parent.parent_id, SplitKind.WHOLE))
            continue
        legal = []
        for pattern in generate_y_split_patterns(parent, config):
            blocks = blocks_for_pattern(parent, pattern, config)
            if all(any(robot_is_eligible(block, robot, config) for robot in range(4)) for block in blocks):
                legal.append(pattern)
        if not legal:
            raise ValueError(f"{parent.parent_id}: no legal mandatory Y pattern")
        result.append(min(legal, key=lambda item: (item.t, item.point_id)))
    return tuple(result)


def _local_delta(
    route: list[str],
    hints: list[int],
    position: int,
    block: WeldingBlock,
    hint: int,
    blocks: dict[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    start, end = oriented_endpoints(block, hint)
    delta = 0.0
    if position > 0:
        _, left_end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        delta += math.dist(left_end, start) / config.empty_speed
    if position < len(route):
        right_start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        delta += math.dist(end, right_start) / config.empty_speed
    if 0 < position < len(route):
        _, left_end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        right_start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        delta -= math.dist(left_end, right_start) / config.empty_speed
    return delta


def _construct(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
    *,
    insertion_limit: int,
    strategy: InitializationStrategy,
    legacy_robot_tie_first: bool = False,
    clearance_aware: bool = False,
) -> CanonicalSolution:
    if strategy is InitializationStrategy.REGION_COOPERATIVE_INITIALIZATION:
        return _construct_region_cooperative_initialization(
            parents, patterns, config
        )
    if strategy is InitializationStrategy.RAIL_MONOTONE_BALANCED:
        return _construct_rail_monotone_balanced_bootstrap(
            parents, patterns, config, clearance_aware=clearance_aware
        )
    parent_by_id = {parent.parent_id: parent for parent in parents}
    all_blocks = {
        block.block_id: block
        for pattern in patterns
        for block in blocks_for_pattern(parent_by_id[pattern.parent_id], pattern, config)
    }
    ordered_blocks = sorted(
        all_blocks.values(),
        key=lambda block: (-config.process_time(block.length), block.parent_id, block.block_id),
    )
    routes: list[list[str]] = [[], [], [], []]
    hints: list[list[int]] = [[], [], [], []]
    loads = [0.0, 0.0, 0.0, 0.0]
    for block in ordered_blocks:
        candidates = []
        duration = config.process_time(block.length)
        for robot in range(4):
            if not robot_is_eligible(block, robot, config):
                continue
            for position in bounded_insertion_positions(len(routes[robot]), insertion_limit):
                for hint in (0, 1):
                    delta = _local_delta(
                        routes[robot], hints[robot], position, block, hint, all_blocks, config
                    )
                    projected = max(
                        loads[index] + (duration if index == robot else 0.0)
                        for index in range(4)
                    )
                    projected_loads = tuple(
                        loads[index] + (duration if index == robot else 0.0)
                        for index in range(4)
                    )
                    if strategy is InitializationStrategy.LOAD_FIRST:
                        score = (projected, delta, robot, position, hint)
                    elif legacy_robot_tie_first:
                        score = (projected, robot, delta, position, hint)
                    elif strategy is InitializationStrategy.RAIL_BALANCED:
                        pair_spread = max(
                            abs(projected_loads[0] - projected_loads[1]),
                            abs(projected_loads[2] - projected_loads[3]),
                        )
                        rail_spread = abs(
                            projected_loads[0]
                            + projected_loads[1]
                            - projected_loads[2]
                            - projected_loads[3]
                        )
                        score = (
                            pair_spread,
                            rail_spread,
                            projected,
                            delta,
                            robot,
                            position,
                            hint,
                        )
                    elif strategy is InitializationStrategy.X_ORDER_AWARE:
                        provisional_routes = [list(route) for route in routes]
                        provisional_hints = [list(vector) for vector in hints]
                        provisional_routes[robot].insert(position, block.block_id)
                        provisional_hints[robot].insert(position, hint)
                        first_points = {}
                        for active_robot in range(4):
                            if provisional_routes[active_robot]:
                                first_block = all_blocks[provisional_routes[active_robot][0]]
                                first_points[active_robot] = oriented_endpoints(
                                    first_block, provisional_hints[active_robot][0]
                                )[0]
                        violations = 0
                        crossing = 0.0
                        for left, right in ((0, 1), (2, 3)):
                            if left in first_points and right in first_points:
                                excess = (
                                    first_points[left][0]
                                    + config.interference_dx
                                    - first_points[right][0]
                                )
                                if excess > config.numeric_epsilon:
                                    violations += 1
                                    crossing += excess
                        score = (
                            violations,
                            crossing,
                            projected,
                            robot,
                            delta,
                            position,
                            hint,
                        )
                    else:
                        other_points = []
                        start, _ = oriented_endpoints(block, hint)
                        for other_robot in range(4):
                            if routes[other_robot]:
                                first_block = all_blocks[routes[other_robot][0]]
                                other_points.append(
                                    oriented_endpoints(first_block, hints[other_robot][0])[0]
                                )
                        nearest = min(
                            (math.dist(start, point) for point in other_points),
                            default=math.inf,
                        )
                        score = (
                            projected,
                            -nearest,
                            delta,
                            robot,
                            position,
                            hint,
                        )
                    candidates.append((score, robot, position, hint))
        if not candidates:
            raise ValueError(f"{block.block_id}: no eligible insertion")
        _, robot, position, hint = min(candidates, key=lambda item: item[0])
        routes[robot].insert(position, block.block_id)
        hints[robot].insert(position, hint)
        loads[robot] += duration
    return canonicalize(
        parents,
        patterns,
        tuple(Route(robot, tuple(routes[robot])) for robot in range(4)),
        config,
        revision=0,
    )


def _construct_rail_serial_bootstrap(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
) -> CanonicalSolution:
    """Build one deterministic, feasibility-oriented fallback.

    The normal portfolio uses both robots on each rail.  If every selected
    construction deadlocks, this bounded fallback activates only the left
    robot on each rail and orders its blocks monotonically by X.  Flexible
    blocks are assigned to the less-loaded rail.  Geometry and split patterns
    are unchanged.
    """

    parent_by_id = {parent.parent_id: parent for parent in parents}
    blocks = tuple(
        block
        for pattern in patterns
        for block in blocks_for_pattern(parent_by_id[pattern.parent_id], pattern, config)
    )
    assigned: dict[int, list[WeldingBlock]] = {0: [], 2: []}
    loads = {0: 0.0, 2: 0.0}
    for block in sorted(
        blocks,
        key=lambda item: (
            -config.process_time(item.length),
            item.parent_id,
            item.block_id,
        ),
    ):
        eligible = tuple(
            robot for robot in (0, 2) if robot_is_eligible(block, robot, config)
        )
        if not eligible:
            raise ValueError(f"{block.block_id}: no eligible rail for bootstrap")
        robot = min(eligible, key=lambda item: (loads[item], item))
        assigned[robot].append(block)
        loads[robot] += config.process_time(block.length)

    for robot in (0, 2):
        assigned[robot].sort(
            key=lambda item: (
                (item.start[0] + item.end[0]) / 2.0,
                min(item.start[0], item.end[0]),
                max(item.start[0], item.end[0]),
                item.block_id,
            )
        )
    return canonicalize(
        parents,
        patterns,
        (
            Route(0, tuple(block.block_id for block in assigned[0])),
            Route(1, ()),
            Route(2, tuple(block.block_id for block in assigned[2])),
            Route(3, ()),
        ),
        config,
        revision=0,
    )


def _block_x_key(block: WeldingBlock) -> tuple[float, float, float, str]:
    return (
        min(block.start[0], block.end[0]),
        (block.start[0] + block.end[0]) / 2.0,
        max(block.start[0], block.end[0]),
        block.block_id,
    )


def _route_empty_proxy(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    """Minimum empty travel for a fixed block order, excluding the common home leg."""
    if len(route) < 2:
        return 0.0
    states = {direction: 0.0 for direction in (0, 1)}
    for index in range(1, len(route)):
        previous = blocks[route[index - 1]]
        current = blocks[route[index]]
        next_states = {}
        for direction in (0, 1):
            start = current.start if direction == 0 else current.end
            next_states[direction] = min(
                cost
                + math.dist(
                    previous.end if old_direction == 0 else previous.start,
                    start,
                )
                / config.empty_speed
                for old_direction, cost in states.items()
            )
        states = next_states
    return min(states.values())


def _nearest_route(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    pending = set(block_ids)
    result: list[str] = []
    point = (
        config.workspace_x[0] if robot in (0, 2) else config.workspace_x[1],
        config.by[1] if robot < 2 else config.by[0],
    )
    while pending:
        options = []
        for block_id in sorted(pending):
            block = blocks[block_id]
            for direction, (start, end) in enumerate(
                ((block.start, block.end), (block.end, block.start))
            ):
                options.append((math.dist(point, start), block_id, direction, end))
        _, selected, _, point = min(options, key=lambda item: item[:3])
        result.append(selected)
        pending.remove(selected)
    return tuple(result)


def _farthest_insertion_route(
    block_ids: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    if not block_ids:
        return ()
    first = max(
        block_ids,
        key=lambda item: (max(blocks[item].start[0], blocks[item].end[0]), item),
    )
    route = [first]
    pending = set(block_ids) - {first}
    while pending:
        before = _route_empty_proxy(route, blocks, config)
        options = []
        for block_id in sorted(pending):
            insertions = []
            for position in range(len(route) + 1):
                candidate = route[:position] + [block_id] + route[position:]
                insertions.append(
                    (_route_empty_proxy(candidate, blocks, config) - before, position)
                )
            delta, position = min(insertions)
            options.append((delta, block_id, position))
        _, selected, position = max(options, key=lambda item: (item[0], item[1]))
        route.insert(position, selected)
        pending.remove(selected)
    return tuple(route)


def _bounded_two_opt(
    route: Sequence[str],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    max_passes: int = 2,
) -> tuple[str, ...]:
    current = tuple(route)
    current_cost = _route_empty_proxy(current, blocks, config)
    for _ in range(max_passes):
        best = (current_cost, current)
        for left in range(max(0, len(current) - 1)):
            for right in range(left + 1, len(current)):
                candidate = (
                    current[:left]
                    + tuple(reversed(current[left : right + 1]))
                    + current[right + 1 :]
                )
                cost = _route_empty_proxy(candidate, blocks, config)
                if (cost, candidate) < best:
                    best = (cost, candidate)
        if best[1] == current:
            break
        current_cost, current = best
    return current


def _route_order_risk(
    route: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    centers = [
        (blocks[item].start[0] + blocks[item].end[0]) / 2.0 for item in route
    ]
    sign = 1.0 if robot in (0, 2) else -1.0
    return sum(
        abs(right - left) / config.empty_speed
        for left, right in zip(centers, centers[1:])
        if sign * (right - left) < -config.numeric_epsilon
    )


def _improve_region_route(
    block_ids: Sequence[str],
    robot: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[str, ...]:
    """Choose one bounded WAG-inspired ordering without running WAG/VNS."""
    monotone = tuple(
        sorted(
            block_ids,
            key=lambda item: _block_x_key(blocks[item]),
            reverse=robot in (1, 3),
        )
    )
    nearest = _nearest_route(block_ids, robot, blocks, config)
    farthest = _farthest_insertion_route(block_ids, blocks, config)
    candidates = {
        monotone,
        _bounded_two_opt(nearest, blocks, config),
        _bounded_two_opt(farthest, blocks, config),
    }
    return min(
        candidates,
        key=lambda route: (
            _route_empty_proxy(route, blocks, config)
            + 0.10 * _route_order_risk(route, robot, blocks, config),
            _route_order_risk(route, robot, blocks, config),
            route,
        ),
    )


def _jrrb_robot_by_block(solution: CanonicalSolution) -> dict[str, int]:
    return {
        block_id: route.robot_id
        for route in solution.routes
        for block_id in route.block_ids
    }


def _jrrb_boundary_value(
    routes: Sequence[Sequence[str]],
    pair: tuple[int, int],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    left = routes[pair[0]]
    right = routes[pair[1]]
    if not left:
        return config.workspace_x[0]
    if not right:
        return config.workspace_x[1]
    left_x = max((blocks[item].start[0] + blocks[item].end[0]) / 2.0 for item in left)
    right_x = min((blocks[item].start[0] + blocks[item].end[0]) / 2.0 for item in right)
    return 0.5 * (left_x + right_x)


def _jrrb_same_rail_overlap(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    total = 0.0
    pairs = ((0, 1),) if len(routes) == 2 else ((0, 1), (2, 3))
    for left, right in pairs:
        if not routes[left] or not routes[right]:
            continue
        left_max = max(
            max(blocks[item].start[0], blocks[item].end[0]) for item in routes[left]
        )
        right_min = min(
            min(blocks[item].start[0], blocks[item].end[0]) for item in routes[right]
        )
        total += max(0.0, left_max + config.interference_dx - right_min)
    return total


def _jrrb_boundary_options(
    block_ids: Sequence[str],
    pair: tuple[int, int],
    fixed: Mapping[int, Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    limit: int = 3,
) -> tuple[tuple[tuple[str, ...], tuple[str, ...], float], ...]:
    """Return a fixed small set of boundary choices for one rail."""
    ordered = tuple(sorted(block_ids, key=lambda item: _block_x_key(blocks[item])))
    ranked = []
    for boundary in range(len(ordered) + 1):
        left = tuple(fixed.get(pair[0], ())) + ordered[:boundary]
        right = ordered[boundary:] + tuple(fixed.get(pair[1], ()))
        left = tuple(sorted(left, key=lambda item: _block_x_key(blocks[item])))
        right = tuple(sorted(right, key=lambda item: _block_x_key(blocks[item]), reverse=True))
        routes = (left, right)
        loads = tuple(
            sum(config.process_time(blocks[item].length) for item in route)
            for route in routes
        )
        empty = tuple(_route_empty_proxy(route, blocks, config) for route in routes)
        overlap = _jrrb_same_rail_overlap(routes, blocks, config)
        boundary_x = _jrrb_boundary_value(routes, (0, 1), blocks, config)
        ranked.append((
            (
                max(loads, default=0.0) + 0.10 * max(empty, default=0.0)
                + 0.02 * overlap / config.empty_speed,
                abs(loads[0] - loads[1]),
                sum(empty),
                overlap,
                boundary,
            ),
            left,
            right,
            boundary_x,
        ))
    return tuple((left, right, boundary_x) for _, left, right, boundary_x in sorted(ranked)[:limit])


def _jrrb_route_components(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *,
    current_max_load: float,
    current_empty: float,
    wait_scale: float,
) -> tuple[float, ...]:
    loads = tuple(
        sum(config.process_time(blocks[item].length) for item in route)
        for route in routes
    )
    rail_totals = (loads[0] + loads[1], loads[2] + loads[3])
    empty = sum(_route_empty_proxy(route, blocks, config) for route in routes)
    overlap = _jrrb_same_rail_overlap(routes, blocks, config)
    order_risk = sum(
        _route_order_risk(route, robot, blocks, config)
        for robot, route in enumerate(routes)
    )
    load_scale = max(current_max_load, 1.0)
    rail_scale = max(sum(loads), 1.0)
    empty_scale = max(current_empty, 1.0)
    workspace_width = max(config.workspace_x[1] - config.workspace_x[0], 1.0)
    normalized = (
        0.45 * max(loads, default=0.0) / load_scale
        + 0.15 * (max(loads) - min(loads)) / load_scale
        + 0.12 * abs(rail_totals[0] - rail_totals[1]) / rail_scale
        + 0.18 * empty / empty_scale
        + 0.05 * overlap / workspace_width * (1.0 + wait_scale)
        + 0.05 * order_risk / empty_scale
    )
    return (
        normalized,
        max(loads, default=0.0),
        max(loads) - min(loads),
        abs(rail_totals[0] - rail_totals[1]),
        empty,
        overlap,
        order_risk,
    )


def _jrrb_representatives(rows: Sequence[dict[str, object]], limit: int) -> tuple[dict[str, object], ...]:
    """Round-robin fixed proxy views so the cap is not a single-weight Top-K."""
    rankings = (
        lambda row: (row["components"][0], row["identity"]),
        lambda row: (row["components"][1], row["components"][3], row["identity"]),
        lambda row: (row["components"][4], row["components"][6], row["identity"]),
        lambda row: (row["components"][5], row["components"][6], row["identity"]),
    )
    ordered = [sorted(rows, key=key) for key in rankings]
    selected = []
    seen = set()
    for rank in range(max((len(items) for items in ordered), default=0)):
        for items in ordered:
            if rank >= len(items):
                continue
            row = items[rank]
            identity = row["identity"]
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(row)
            if len(selected) == limit:
                return tuple(selected)
    return tuple(selected)


def build_joint_rail_region_rebuild_candidates(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *,
    max_candidates: int = 16,
) -> JRRBConstruction:
    """Build one bounded JRRB pool without direction DP or reference access.

    Cross-rail parent assignment and the two rail boundaries are proposed as
    one state.  Boundary pairs are scored together; neither rail is committed
    before the other.  Split patterns, including fixed X children, are frozen.
    """
    del current_directions  # directions are deliberately unavailable to the cheap pool
    if not 0 <= max_candidates <= 16:
        raise ValueError("JRRB complete-candidate cap must be in 0..16")
    if max_candidates == 0:
        return JRRBConstruction(0, 0, 0, ())
    blocks = block_map(current, config)
    patterns = {pattern.parent_id: pattern for pattern in current.patterns}
    current_routes = tuple(route.block_ids for route in current.routes)
    robot_by_block = _jrrb_robot_by_block(current)
    fixed: dict[int, list[str]] = {robot: [] for robot in range(4)}
    rail_blocks: list[list[str]] = [[], []]
    blocks_by_parent: dict[str, list[str]] = {}
    for block_id, block in blocks.items():
        blocks_by_parent.setdefault(block.parent_id, []).append(block_id)
        if patterns[block.parent_id].kind is SplitKind.X_SPLIT:
            fixed[robot_by_block[block_id]].append(block_id)
        else:
            rail_blocks[robot_by_block[block_id] // 2].append(block_id)
    movable = []
    for parent_id, parent_blocks in sorted(blocks_by_parent.items()):
        if patterns[parent_id].kind is SplitKind.X_SPLIT:
            continue
        current_rails = {robot_by_block[item] // 2 for item in parent_blocks}
        if len(current_rails) != 1:
            continue
        if all(
            any(robot_is_eligible(blocks[item], robot, config) for robot in pair)
            for item in parent_blocks
            for pair in ((0, 1), (2, 3))
        ):
            x = sum(
                (blocks[item].start[0] + blocks[item].end[0]) / 2.0
                for item in parent_blocks
            ) / len(parent_blocks)
            duration = sum(config.process_time(blocks[item].length) for item in parent_blocks)
            movable.append((parent_id, x, duration, next(iter(current_rails))))
    if len(movable) < 2:
        return JRRBConstruction(0, 0, 0, (), (("NO_MULTI_PARENT_CROSS_RAIL_REGION", 1),))

    loads = tuple(
        sum(config.process_time(blocks[item].length) for item in route)
        for route in current_routes
    )
    current_empty = sum(_route_empty_proxy(route, blocks, config) for route in current_routes)
    waits = [0.0, 0.0, 0.0, 0.0]
    wait_x = []
    for operation in schedule.operations:
        if operation.kind is OperationKind.WAIT:
            waits[operation.robot_id] += operation.duration
            wait_x.append(0.5 * (operation.start[0] + operation.end[0]))
    critical_robot = max(range(4), key=lambda robot: (loads[robot], waits[robot], -robot))
    critical_x = [
        (blocks[item].start[0] + blocks[item].end[0]) / 2.0
        for item in current_routes[critical_robot]
    ]
    lo = min(row[1] for row in movable)
    hi = max(row[1] for row in movable)
    span = max(hi - lo, config.delta_x)
    anchor = (
        sum(wait_x) / len(wait_x)
        if wait_x
        else sum(critical_x) / len(critical_x)
        if critical_x
        else 0.5 * (lo + hi)
    )
    windows = (
        (lo, hi),
        (lo, min(hi, lo + 0.55 * span)),
        (max(lo, hi - 0.55 * span), hi),
        (max(lo, anchor - 0.28 * span), min(hi, anchor + 0.28 * span)),
        (
            max(lo, min(critical_x, default=anchor) - 0.08 * span),
            min(hi, max(critical_x, default=anchor) + 0.08 * span),
        ),
    )
    unique_windows = []
    seen_windows = set()
    for a, b in windows:
        key = (round(a, 12), round(b, 12))
        if a <= b and key not in seen_windows:
            seen_windows.add(key)
            unique_windows.append((a, b))
    windows = tuple(unique_windows)
    current_rail_by_parent = {parent_id: rail for parent_id, _, _, rail in movable}
    current_rail_totals = [loads[0] + loads[1], loads[2] + loads[3]]
    raw: list[dict[str, object]] = []
    raw_seen = set()
    for window in windows:
        selected = [row for row in movable if window[0] <= row[1] <= window[1]]
        if len(selected) < 2:
            continue
        selected_ids = {row[0] for row in selected}
        base_totals = list(current_rail_totals)
        for parent_id, _, duration, rail in selected:
            base_totals[rail] -= duration
        schemes: dict[str, dict[str, int]] = {}
        schemes["WINDOW_FLIP"] = {
            parent_id: 1 - rail for parent_id, _, _, rail in selected
        }
        greedy = {}
        totals = list(base_totals)
        for parent_id, _, duration, _ in sorted(selected, key=lambda row: (-row[2], row[1], row[0])):
            rail = min((0, 1), key=lambda item: (totals[item], item))
            greedy[parent_id] = rail
            totals[rail] += duration
        schemes["JOINT_LOAD_BALANCE"] = greedy
        overloaded = int(current_rail_totals[1] > current_rail_totals[0])
        underloaded = 1 - overloaded
        schemes["OVERLOAD_REGION_TRANSFER"] = {
            parent_id: (underloaded if rail == overloaded else rail)
            for parent_id, _, _, rail in selected
        }
        alternating = {}
        first = int(base_totals[1] < base_totals[0])
        for index, (parent_id, _, _, _) in enumerate(sorted(selected, key=lambda row: (row[1], row[0]))):
            alternating[parent_id] = (first + index) % 2
        schemes["X_BAND_ALTERNATE"] = alternating
        for scheme, changed in schemes.items():
            assignment = dict(current_rail_by_parent)
            assignment.update(changed)
            if not any(assignment[parent_id] != current_rail_by_parent[parent_id] for parent_id in selected_ids):
                continue
            proposed = [[], []]
            for block_id in rail_blocks[0] + rail_blocks[1]:
                parent_id = blocks[block_id].parent_id
                rail = assignment.get(parent_id, robot_by_block[block_id] // 2)
                proposed[rail].append(block_id)
            up_options = _jrrb_boundary_options(proposed[0], (0, 1), fixed, blocks, config)
            low_options = _jrrb_boundary_options(proposed[1], (2, 3), fixed, blocks, config)
            for up in up_options:
                for low in low_options:
                    routes = (up[0], up[1], low[0], low[1])
                    identity = tuple(tuple(route) for route in routes)
                    if identity in raw_seen or identity == current_routes:
                        continue
                    raw_seen.add(identity)
                    components = _jrrb_route_components(
                        routes, blocks, config,
                        current_max_load=max(loads), current_empty=current_empty,
                        wait_scale=min(sum(waits) / max(max(loads), 1.0), 1.0),
                    )
                    raw.append({
                        "routes": routes,
                        "components": components,
                        "identity": identity,
                        "b_up": up[2],
                        "b_low": low[2],
                        "window": window,
                        "scheme": scheme,
                    })
    selected_raw = _jrrb_representatives(raw, max_candidates)
    current_b_up = _jrrb_boundary_value(current_routes, (0, 1), blocks, config)
    current_b_low = _jrrb_boundary_value(current_routes, (2, 3), blocks, config)
    candidates = []
    seen_hashes = set()
    rejections: dict[str, int] = {}
    constructed_count = 0
    for row in selected_raw:
        routes = tuple(
            _improve_region_route(route, robot, blocks, config)
            for robot, route in enumerate(row["routes"])
        )
        try:
            solution = canonicalize(
                current.parents,
                current.patterns,
                tuple(Route(robot, routes[robot]) for robot in range(4)),
                config,
                revision=current.revision + 1,
                x_split_validator=(
                    finite_x_split_validator(current.parents, config)
                    if any(pattern.kind is SplitKind.X_SPLIT for pattern in current.patterns)
                    else None
                ),
            )
            if tuple(pattern.pattern_id for pattern in solution.patterns) != tuple(
                pattern.pattern_id for pattern in current.patterns
            ):
                raise ValueError("JRRB changed split pattern")
            solution_blocks = block_map(solution, config)
            if any(
                not robot_is_eligible(solution_blocks[item], route.robot_id, config)
                for route in solution.routes
                for item in route.block_ids
            ):
                raise ValueError("JRRB robot eligibility failure")
        except (ValueError, KeyError, IndexError) as error:
            key = str(error)
            rejections[key] = rejections.get(key, 0) + 1
            continue
        constructed_count += 1
        if solution.canonical_hash == current.canonical_hash or solution.canonical_hash in seen_hashes:
            rejections["IDENTITY_OR_DUPLICATE"] = rejections.get("IDENTITY_OR_DUPLICATE", 0) + 1
            continue
        seen_hashes.add(solution.canonical_hash)
        new_robot_by_block = _jrrb_robot_by_block(solution)
        cross_rail_parents = {
            blocks[item].parent_id
            for item in blocks
            if new_robot_by_block[item] // 2 != robot_by_block[item] // 2
        }
        if not cross_rail_parents:
            rejections["NO_CROSS_RAIL_CHANGE"] = rejections.get("NO_CROSS_RAIL_CHANGE", 0) + 1
            continue
        same_rail_changes = sum(
            new_robot_by_block[item] != robot_by_block[item]
            and new_robot_by_block[item] // 2 == robot_by_block[item] // 2
            for item in blocks
        )
        changed_parents = tuple(sorted({
            blocks[item].parent_id
            for item in blocks
            if new_robot_by_block[item] != robot_by_block[item]
        }))
        reordered = sum(
            tuple(current_routes[robot]) != tuple(solution.routes[robot].block_ids)
            for robot in range(4)
        )
        components = _jrrb_route_components(
            tuple(route.block_ids for route in solution.routes), solution_blocks, config,
            current_max_load=max(loads), current_empty=current_empty,
            wait_scale=min(sum(waits) / max(max(loads), 1.0), 1.0),
        )
        candidates.append(JRRBCandidate(
            solution=solution,
            cheap_score=components,
            components=components,
            b_up=float(row["b_up"]),
            b_low=float(row["b_low"]),
            window=tuple(row["window"]),
            scheme=str(row["scheme"]),
            changed_parent_ids=changed_parents,
            cross_rail_parent_changes=len(cross_rail_parents),
            same_rail_robot_changes=same_rail_changes,
            reordered_routes=reordered,
            b_up_changed=abs(float(row["b_up"]) - current_b_up) > config.numeric_epsilon,
            b_low_changed=abs(float(row["b_low"]) - current_b_low) > config.numeric_epsilon,
        ))
    return JRRBConstruction(
        attempted=len(selected_raw),
        constructed=constructed_count,
        valid_unique=len(candidates),
        candidates=tuple(candidates),
        rejection_reasons=tuple(sorted(rejections.items())),
    )


def _region_boundary_plan(
    block_ids: Sequence[str],
    pair: tuple[int, int],
    fixed: Mapping[int, Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[tuple[object, ...], tuple[tuple[str, ...], tuple[str, ...]]]:
    """Cheaply select a contiguous X boundary; overlap is a penalty, not a veto."""
    ordered = tuple(sorted(block_ids, key=lambda item: _block_x_key(blocks[item])))
    best = None
    for boundary in range(len(ordered) + 1):
        left = tuple(fixed.get(pair[0], ())) + ordered[:boundary]
        right = ordered[boundary:] + tuple(fixed.get(pair[1], ()))
        left = tuple(sorted(left, key=lambda item: _block_x_key(blocks[item])))
        right = tuple(
            sorted(right, key=lambda item: _block_x_key(blocks[item]), reverse=True)
        )
        loads = tuple(
            sum(config.process_time(blocks[item].length) for item in route)
            for route in (left, right)
        )
        empty = tuple(
            _route_empty_proxy(route, blocks, config) for route in (left, right)
        )
        overlap = 0.0
        if left and right:
            left_max = max(
                max(blocks[item].start[0], blocks[item].end[0]) for item in left
            )
            right_min = min(
                min(blocks[item].start[0], blocks[item].end[0]) for item in right
            )
            overlap = max(0.0, left_max + config.interference_dx - right_min)
        objective = (
            max(loads, default=0.0)
            + 0.10 * max(empty, default=0.0)
            + 0.02 * overlap / config.empty_speed
        )
        score = (
            objective,
            max(loads, default=0.0),
            abs(loads[0] - loads[1]),
            sum(empty),
            overlap,
            boundary,
        )
        candidate = (score, (left, right))
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    return best


def _construct_region_cooperative_initialization(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
) -> CanonicalSolution:
    """Deterministic rail allocation, X-boundary balancing and bounded route ordering."""
    parent_by_id = {parent.parent_id: parent for parent in parents}
    blocks = {
        block.block_id: block
        for pattern in patterns
        for block in blocks_for_pattern(parent_by_id[pattern.parent_id], pattern, config)
    }
    pattern_by_parent = {pattern.parent_id: pattern for pattern in patterns}
    fixed: dict[int, list[str]] = {robot: [] for robot in range(4)}
    rail_fixed: list[list[str]] = [[], []]
    flexible: list[str] = []
    for block_id, block in blocks.items():
        pattern = pattern_by_parent[block.parent_id]
        if pattern.kind is SplitKind.X_SPLIT:
            child = int(block_id.rsplit("::", 1)[1])
            upper = pattern.rail is not None and pattern.rail.value == "UPPER"
            robot = (0, 1)[child] if upper else (2, 3)[child]
            if not robot_is_eligible(block, robot, config):
                raise ValueError(f"{block_id}: fixed X child is not robot eligible")
            fixed[robot].append(block_id)
            continue
        eligible_rails = tuple(
            rail
            for rail, pair in enumerate(((0, 1), (2, 3)))
            if any(robot_is_eligible(block, robot, config) for robot in pair)
        )
        if len(eligible_rails) == 2:
            flexible.append(block_id)
        elif len(eligible_rails) == 1:
            rail_fixed[eligible_rails[0]].append(block_id)
        else:
            raise ValueError(f"{block_id}: no eligible rail for region initialization")

    orders = (
        tuple(
            sorted(
                flexible,
                key=lambda item: (
                    -config.process_time(blocks[item].length),
                    _block_x_key(blocks[item]),
                ),
            )
        ),
        tuple(sorted(flexible, key=lambda item: _block_x_key(blocks[item]))),
        tuple(
            sorted(flexible, key=lambda item: _block_x_key(blocks[item]), reverse=True)
        ),
    )
    candidates = []
    for order in dict.fromkeys(orders):
        groups = [list(rail_fixed[0]), list(rail_fixed[1])]
        for block_id in order:
            proposals = []
            for rail in (0, 1):
                proposed = [list(groups[0]), list(groups[1])]
                proposed[rail].append(block_id)
                plans = tuple(
                    _region_boundary_plan(
                        proposed[index],
                        ((0, 1), (2, 3))[index],
                        fixed,
                        blocks,
                        config,
                    )
                    for index in (0, 1)
                )
                loads = tuple(
                    sum(config.process_time(blocks[item].length) for item in route)
                    for _, routes in plans
                    for route in routes
                )
                rail_totals = (loads[0] + loads[1], loads[2] + loads[3])
                proposals.append(
                    (
                        (
                            max(plan[0][0] for plan in plans),
                            max(loads, default=0.0),
                            abs(rail_totals[0] - rail_totals[1]),
                            rail,
                        ),
                        rail,
                    )
                )
            _, selected_rail = min(proposals)
            groups[selected_rail].append(block_id)

        plans = tuple(
            _region_boundary_plan(
                groups[index],
                ((0, 1), (2, 3))[index],
                fixed,
                blocks,
                config,
            )
            for index in (0, 1)
        )
        route_ids = [route for _, routes in plans for route in routes]
        improved = tuple(
            _improve_region_route(route, robot, blocks, config)
            for robot, route in enumerate(route_ids)
        )
        loads = tuple(
            sum(config.process_time(blocks[item].length) for item in route)
            for route in improved
        )
        empty = tuple(
            _route_empty_proxy(route, blocks, config) for route in improved
        )
        score = (
            max(loads, default=0.0) + 0.10 * max(empty, default=0.0),
            max(loads, default=0.0),
            max(abs(loads[0] - loads[1]), abs(loads[2] - loads[3])),
            sum(empty),
            improved,
        )
        candidates.append((score, improved))
    _, routes = min(candidates)
    return canonicalize(
        parents,
        patterns,
        tuple(Route(robot, routes[robot]) for robot in range(4)),
        config,
        revision=0,
        x_split_validator=(
            finite_x_split_validator(parents, config)
            if any(pattern.kind is SplitKind.X_SPLIT for pattern in patterns)
            else None
        ),
    )


def _construct_rail_monotone_balanced_bootstrap(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
    *,
    clearance_aware: bool = False,
) -> CanonicalSolution:
    """Build the deterministic four-robot V3 monotone/load-balanced seed."""
    parent_by_id = {parent.parent_id: parent for parent in parents}
    blocks = tuple(
        block
        for pattern in patterns
        for block in blocks_for_pattern(parent_by_id[pattern.parent_id], pattern, config)
    )
    rail_blocks: dict[int, list[WeldingBlock]] = {0: [], 1: []}
    flexible: list[WeldingBlock] = []
    rail_loads = [0.0, 0.0]
    for block in blocks:
        upper = any(robot_is_eligible(block, robot, config) for robot in (0, 1))
        lower = any(robot_is_eligible(block, robot, config) for robot in (2, 3))
        if upper and lower:
            flexible.append(block)
        elif upper:
            rail_blocks[0].append(block)
            rail_loads[0] += config.process_time(block.length)
        elif lower:
            rail_blocks[1].append(block)
            rail_loads[1] += config.process_time(block.length)
        else:
            raise ValueError(f"{block.block_id}: no eligible rail for V3 bootstrap")

    for block in sorted(flexible, key=_block_x_key):
        rail = min((0, 1), key=lambda item: (rail_loads[item], item))
        rail_blocks[rail].append(block)
        rail_loads[rail] += config.process_time(block.length)

    routes: list[tuple[str, ...]] = [(), (), (), ()]
    for rail, (left_robot, right_robot) in enumerate(((0, 1), (2, 3))):
        ordered = sorted(rail_blocks[rail], key=_block_x_key)
        prefix_loads = [0.0]
        for block in ordered:
            prefix_loads.append(
                prefix_loads[-1] + config.process_time(block.length)
            )
        total = prefix_loads[-1]
        boundaries = range(len(ordered) + 1)
        if clearance_aware:
            # A load-balanced cut can put overlapping weld envelopes on opposite
            # robots, which monotone route ordering alone does not prevent.
            # Prefer separated, nonempty partitions when geometry permits one;
            # otherwise preserve the original construction (no false guarantee).
            separated = [
                i for i in range(1, len(ordered))
                if max(max(b.start[0], b.end[0]) for b in ordered[:i])
                + config.interference_dx
                <= min(min(b.start[0], b.end[0]) for b in ordered[i:])
                + config.numeric_epsilon
            ]
            if separated:
                boundaries = separated
        boundary = min(
            boundaries,
            key=lambda index: (
                max(prefix_loads[index], total - prefix_loads[index]),
                abs(prefix_loads[index] - (total - prefix_loads[index])),
                index,
            ),
        )
        routes[left_robot] = tuple(block.block_id for block in ordered[:boundary])
        routes[right_robot] = tuple(
            block.block_id for block in reversed(ordered[boundary:])
        )

    return canonicalize(
        parents,
        patterns,
        tuple(Route(robot, routes[robot]) for robot in range(4)),
        config,
        revision=0,
    )


def _direction_empty_travel(
    solution: CanonicalSolution,
    directions: DirectionVectors,
    config: ScientificConfig,
) -> float:
    blocks = block_map(solution, config)
    total = 0.0
    for robot, route in enumerate(solution.routes):
        for position in range(len(route.block_ids) - 1):
            left = blocks[route.block_ids[position]]
            right = blocks[route.block_ids[position + 1]]
            _, left_end = oriented_endpoints(left, directions[robot][position])
            right_start, _ = oriented_endpoints(
                right, directions[robot][position + 1]
            )
            total += math.dist(left_end, right_start) / config.empty_speed
    return total


def _v3_direction_feasibility_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    budget: int = V3_DIRECTION_FEASIBILITY_REFERENCE_BUDGET,
) -> tuple[DirectionVectors, ...]:
    """Return deterministic directions for feasibility only, not Cmax selection."""
    if budget <= 0:
        return ()
    blocks = block_map(solution, config)
    rng = random.Random(int(solution.canonical_hash[:16], 16))
    result: list[DirectionVectors] = []
    seen: set[DirectionVectors] = set()
    attempts = 0
    while len(result) < budget and attempts < 32 * budget:
        attempts += 1
        directions: DirectionVectors = tuple(
            tuple(rng.randrange(2) for _ in route.block_ids)
            for route in solution.routes
        )  # type: ignore[assignment]
        if directions in seen:
            continue
        seen.add(directions)
        points = {
            robot: oriented_endpoints(
                blocks[route.block_ids[0]], directions[robot][0]
            )[0]
            for robot, route in enumerate(solution.routes)
            if route.block_ids
        }
        if _initial_error(points, config) is not None:
            continue
        result.append(directions)
    return tuple(result)


def build_initial_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    stats: SearchStats,
    *,
    insertion_limit: int = 8,
    construction_budget: int = 2,
    kinit_ref: int = 2,
    feasibility_bootstrap_budget: int = 1,
    portfolio: bool = False,
    clearance_aware: bool = False,
    region_cooperative: bool = False,
    reference_evaluator: ReferenceEvaluator | None = None,
    certifier: Certifier = certify_schedule,
    scope: FormalScope | None = None,
) -> InitializationResult:
    reference_evaluator = resolve_reference_evaluator(scope, reference_evaluator)
    if scope is not None and certifier is not certify_schedule:
        raise ValueError("formal initialization requires the common certifier")
    baseline_strategies = (
        InitializationStrategy.LOAD_FIRST,
        InitializationStrategy.RAIL_BALANCED,
        InitializationStrategy.X_ORDER_AWARE,
        InitializationStrategy.SPATIAL_SPREAD,
        InitializationStrategy.RAIL_MONOTONE_BALANCED,
    )
    cooperative_strategies = (
        InitializationStrategy.LOAD_FIRST,
        InitializationStrategy.RAIL_BALANCED,
        InitializationStrategy.X_ORDER_AWARE,
        InitializationStrategy.REGION_COOPERATIVE_INITIALIZATION,
        InitializationStrategy.RAIL_MONOTONE_BALANCED,
    )
    strategy_pool = cooperative_strategies if region_cooperative else baseline_strategies
    strategies = strategy_pool[:construction_budget]
    if construction_budget < 1 or construction_budget > len(baseline_strategies):
        raise ValueError(
            f"B_init_pool must be between 1 and {len(baseline_strategies)}"
        )
    if not (1 <= kinit_ref <= construction_budget):
        raise ValueError("require 1 <= Kinit_ref <= B_init_pool")
    if feasibility_bootstrap_budget not in (0, 1):
        raise ValueError("B_init_bootstrap must be 0 or 1")
    started = time.perf_counter()
    attempts: list[InitializationAttempt] = []
    seen: set[str] = set()
    try:
        selected_patterns = _patterns(parents, config)
    except (ValueError, ArithmeticError, OverflowError) as error:
        stats.init_time += time.perf_counter() - started
        return InitializationResult(
            InitializationStatus.INITIALIZATION_FAILED,
            None,
            None,
            None,
            None,
            (InitializationAttempt(0, None, None, None, None, (str(error),)),),
        )

    constructed: list[tuple[int, CanonicalSolution, ConstrainedDirectionResult, tuple[object, ...]]] = []
    for construction_index, strategy in enumerate(strategies):
        stats.construction_attempts += 1
        try:
            solution = _construct(
                parents,
                selected_patterns,
                config,
                insertion_limit=insertion_limit,
                strategy=strategy,
                legacy_robot_tie_first=(not portfolio and construction_index == 1),
                clearance_aware=clearance_aware,
            )
        except (ValueError, ArithmeticError, OverflowError) as error:
            attempts.append(
                InitializationAttempt(
                    construction_index,
                    None,
                    None,
                    None,
                    None,
                    (str(error),),
                    False,
                    strategy.value,
                )
            )
            continue
        if solution.canonical_hash in seen:
            attempts.append(
                InitializationAttempt(
                    construction_index,
                    solution,
                    None,
                    None,
                    None,
                    ("duplicate construction; reference evaluation skipped",),
                    True,
                    strategy.value,
                )
            )
            continue
        seen.add(solution.canonical_hash)

        dp_started = time.perf_counter()
        direction = optimize_directions_with_initial_feasibility(solution, config)
        stats.direction_dp_time += time.perf_counter() - dp_started
        attempts.append(
            InitializationAttempt(
                construction_index,
                solution,
                direction,
                None,
                None,
                direction.diagnostics,
                False,
                strategy.value,
            )
        )
        if direction.status is not DirectionStatus.FEASIBLE:
            stats.init_direction_failures += 1
            continue
        blocks = block_map(solution, config)
        loads = tuple(
            sum(config.process_time(blocks[item].length) for item in route.block_ids)
            for route in solution.routes
        )
        constructed.append(
            (
                len(attempts) - 1,
                solution,
                direction,
                (
                    max(loads, default=0.0),
                    direction.total_empty_travel,
                    construction_index,
                    solution.canonical_hash,
                ),
            )
        )

    feasible = []
    for attempt_index, solution, direction, _ in sorted(
        constructed, key=(lambda item: item[3]) if portfolio else (lambda item: item[0])
    )[:kinit_ref]:
        orientation_map = {robot: direction.directions[robot] for robot in range(4)}
        ref_started = time.perf_counter()
        try:
            schedule = reference_evaluator(
                solution,
                config,
                orientations=orientation_map,
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            schedule = ScheduleResult(
                ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),)
            )
        ref_duration = time.perf_counter() - ref_started
        ref_ended = time.perf_counter()
        certification = None
        if schedule.status is ScheduleStatus.FEASIBLE:
            cert_started = time.perf_counter()
            certification = (certifier(solution, schedule, config, scope=scope)
                             if scope is not None else certifier(solution, schedule, config))
            stats.certifier_time += time.perf_counter() - cert_started
            if not certification.certified:
                schedule = replace(
                    schedule, status=ScheduleStatus.NUMERIC_FAILURE, cmax=None,
                    diagnostics=(
                        "initial FEASIBLE schedule failed certification: "
                        + "; ".join(certification.errors),
                    ),
                    directions=direction.directions,
                )
        stats.record_reference(
            schedule.status,
            ref_duration,
            initialization=True,
            schedule=schedule,
            reference_start=(
                None if stats.run_started is None else ref_started - stats.run_started
            ),
            reference_end=(
                None if stats.run_started is None else ref_ended - stats.run_started
            ),
        )
        old_attempt = attempts[attempt_index]
        if (
            schedule.status is ScheduleStatus.DEADLOCK
            and old_attempt.strategy
            == InitializationStrategy.RAIL_MONOTONE_BALANCED.value
        ):
            for fallback_directions in _v3_direction_feasibility_candidates(
                solution, config
            ):
                fallback_started = time.perf_counter()
                try:
                    fallback_schedule = reference_evaluator(
                        solution,
                        config,
                        orientations={
                            robot: fallback_directions[robot]
                            for robot in range(4)
                        },
                    )
                except (ArithmeticError, OverflowError, ValueError) as error:
                    fallback_schedule = ScheduleResult(
                        ScheduleStatus.NUMERIC_FAILURE,
                        diagnostics=(str(error),),
                    )
                fallback_duration = time.perf_counter() - fallback_started
                fallback_ended = time.perf_counter()
                fallback_certification = None
                if fallback_schedule.status is ScheduleStatus.FEASIBLE:
                    cert_started = time.perf_counter()
                    fallback_certification = (
                        certifier(solution, fallback_schedule, config, scope=scope)
                        if scope is not None
                        else certifier(solution, fallback_schedule, config)
                    )
                    stats.certifier_time += time.perf_counter() - cert_started
                    if not fallback_certification.certified:
                        fallback_schedule = replace(
                            fallback_schedule,
                            status=ScheduleStatus.NUMERIC_FAILURE,
                            cmax=None,
                            diagnostics=(
                                "V3 direction fallback certification failure: "
                                + "; ".join(fallback_certification.errors),
                            ),
                            directions=fallback_directions,
                        )
                stats.record_reference(
                    fallback_schedule.status,
                    fallback_duration,
                    initialization=True,
                    schedule=fallback_schedule,
                    reference_start=(
                        None
                        if stats.run_started is None
                        else fallback_started - stats.run_started
                    ),
                    reference_end=(
                        None
                        if stats.run_started is None
                        else fallback_ended - stats.run_started
                    ),
                )
                schedule = fallback_schedule
                certification = fallback_certification
                if (
                    schedule.status is ScheduleStatus.FEASIBLE
                    and certification is not None
                    and certification.certified
                ):
                    direction = ConstrainedDirectionResult(
                        DirectionStatus.FEASIBLE,
                        fallback_directions,
                        _direction_empty_travel(
                            solution, fallback_directions, config
                        ),
                        ("V3_DIRECTION_FEASIBILITY_FALLBACK",),
                        direction.legal_first_combinations,
                        direction.route_dp_calls,
                    )
                    break
                if schedule.status is ScheduleStatus.NUMERIC_FAILURE:
                    break
        attempt = InitializationAttempt(
            old_attempt.construction_index,
            solution,
            direction,
            schedule,
            certification,
            direction.diagnostics + schedule.diagnostics,
            False,
            old_attempt.strategy,
        )
        attempts[attempt_index] = attempt
        if schedule.status is ScheduleStatus.NUMERIC_FAILURE:
            stats.init_time += time.perf_counter() - started
            return InitializationResult(
                InitializationStatus.NUMERIC_FAILURE,
                None,
                None,
                attempt.schedule,
                attempt.certification,
                tuple(item for item in attempts if item.schedule is not None),
            )
        if (
            direction.status is DirectionStatus.FEASIBLE
            and schedule.status is ScheduleStatus.FEASIBLE
            and certification is not None
            and certification.certified
        ):
            feasible.append((official_metrics(solution, schedule, config), attempt))

    if clearance_aware:
        attempts = [replace(a, strategy=CLEARANCE_BALANCED_BOOTSTRAP)
                    if a.strategy == InitializationStrategy.RAIL_MONOTONE_BALANCED.value
                    else a for a in attempts]
        feasible = [(m, replace(a, strategy=CLEARANCE_BALANCED_BOOTSTRAP)
                     if a.strategy == InitializationStrategy.RAIL_MONOTONE_BALANCED.value else a)
                    for m, a in feasible]

    if feasible:
        best_metrics, best_attempt = feasible[0]
        for metrics, attempt in feasible[1:]:
            if metrics.compare(best_metrics) < 0:
                best_metrics, best_attempt = metrics, attempt
        stats.init_time += time.perf_counter() - started
        stats.initial_strategy_wins[best_attempt.strategy] += 1
        return InitializationResult(
            InitializationStatus.SUCCESS,
            best_attempt.solution,
            best_attempt.directions.directions if best_attempt.directions else None,
            best_attempt.schedule,
            best_attempt.certification,
            tuple(attempts),
            best_attempt.strategy,
        )

    if feasibility_bootstrap_budget:
        construction_index = len(strategies)
        stats.construction_attempts += 1
        try:
            solution = _construct_rail_serial_bootstrap(
                parents,
                selected_patterns,
                config,
            )
        except (ValueError, ArithmeticError, OverflowError) as error:
            attempts.append(
                InitializationAttempt(
                    construction_index,
                    None,
                    None,
                    None,
                    None,
                    (str(error),),
                    False,
                    RAIL_SERIAL_BOOTSTRAP,
                )
            )
        else:
            if solution.canonical_hash in seen:
                attempts.append(
                    InitializationAttempt(
                        construction_index,
                        solution,
                        None,
                        None,
                        None,
                        ("duplicate bootstrap; reference evaluation skipped",),
                        True,
                        RAIL_SERIAL_BOOTSTRAP,
                    )
                )
            else:
                dp_started = time.perf_counter()
                direction = optimize_directions_with_initial_feasibility(solution, config)
                stats.direction_dp_time += time.perf_counter() - dp_started
                if direction.status is not DirectionStatus.FEASIBLE:
                    stats.init_direction_failures += 1
                    attempts.append(
                        InitializationAttempt(
                            construction_index,
                            solution,
                            direction,
                            None,
                            None,
                            direction.diagnostics,
                            False,
                            RAIL_SERIAL_BOOTSTRAP,
                        )
                    )
                else:
                    orientation_map = {
                        robot: direction.directions[robot] for robot in range(4)
                    }
                    ref_started = time.perf_counter()
                    try:
                        schedule = reference_evaluator(
                            solution,
                            config,
                            orientations=orientation_map,
                        )
                    except (ArithmeticError, OverflowError, ValueError) as error:
                        schedule = ScheduleResult(
                            ScheduleStatus.NUMERIC_FAILURE,
                            diagnostics=(str(error),),
                        )
                    ref_duration = time.perf_counter() - ref_started
                    ref_ended = time.perf_counter()
                    certification = None
                    if schedule.status is ScheduleStatus.FEASIBLE:
                        cert_started = time.perf_counter()
                        certification = (
                            certifier(solution, schedule, config, scope=scope)
                            if scope is not None
                            else certifier(solution, schedule, config)
                        )
                        stats.certifier_time += time.perf_counter() - cert_started
                        if not certification.certified:
                            schedule = replace(
                                schedule,
                                status=ScheduleStatus.NUMERIC_FAILURE,
                                cmax=None,
                                diagnostics=(
                                    "bootstrap FEASIBLE schedule failed certification: "
                                    + "; ".join(certification.errors),
                                ),
                                directions=direction.directions,
                            )
                    stats.record_reference(
                        schedule.status,
                        ref_duration,
                        initialization=True,
                        schedule=schedule,
                        reference_start=(
                            None
                            if stats.run_started is None
                            else ref_started - stats.run_started
                        ),
                        reference_end=(
                            None
                            if stats.run_started is None
                            else ref_ended - stats.run_started
                        ),
                    )
                    attempt = InitializationAttempt(
                        construction_index,
                        solution,
                        direction,
                        schedule,
                        certification,
                        direction.diagnostics + schedule.diagnostics,
                        False,
                        RAIL_SERIAL_BOOTSTRAP,
                    )
                    attempts.append(attempt)
                    if schedule.status is ScheduleStatus.NUMERIC_FAILURE:
                        stats.init_time += time.perf_counter() - started
                        return InitializationResult(
                            InitializationStatus.NUMERIC_FAILURE,
                            None,
                            None,
                            schedule,
                            certification,
                            tuple(attempts),
                        )
                    if (
                        schedule.status is ScheduleStatus.FEASIBLE
                        and certification is not None
                        and certification.certified
                    ):
                        stats.init_time += time.perf_counter() - started
                        stats.initial_strategy_wins[RAIL_SERIAL_BOOTSTRAP] += 1
                        return InitializationResult(
                            InitializationStatus.SUCCESS,
                            solution,
                            direction.directions,
                            schedule,
                            certification,
                            tuple(attempts),
                            RAIL_SERIAL_BOOTSTRAP,
                        )

    stats.init_time += time.perf_counter() - started
    return InitializationResult(
        InitializationStatus.INITIALIZATION_FAILED,
        None,
        None,
        None,
        None,
        tuple(attempts),
    )
