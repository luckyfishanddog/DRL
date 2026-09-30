from __future__ import annotations

from collections.abc import Callable, Sequence
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
    generate_y_split_patterns,
    oriented_endpoints,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
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


RAIL_SERIAL_BOOTSTRAP = "RAIL_SERIAL_BOOTSTRAP"
V3_DIRECTION_FEASIBILITY_REFERENCE_BUDGET = 4


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
) -> CanonicalSolution:
    if strategy is InitializationStrategy.RAIL_MONOTONE_BALANCED:
        return _construct_rail_monotone_balanced_bootstrap(
            parents, patterns, config
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


def _construct_rail_monotone_balanced_bootstrap(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
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
        boundary = min(
            range(len(ordered) + 1),
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
    reference_evaluator: ReferenceEvaluator | None = None,
    certifier: Certifier = certify_schedule,
    scope: FormalScope | None = None,
) -> InitializationResult:
    reference_evaluator = resolve_reference_evaluator(scope, reference_evaluator)
    if scope is not None and certifier is not certify_schedule:
        raise ValueError("formal initialization requires the common certifier")
    strategies = tuple(InitializationStrategy)[:construction_budget]
    if construction_budget < 1 or construction_budget > len(InitializationStrategy):
        raise ValueError(
            f"B_init_pool must be between 1 and {len(InitializationStrategy)}"
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
