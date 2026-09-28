from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
import time

from .geometry import (
    XSplitValidator,
    forbidden_start_intervals,
    operations_conflict,
    optimize_directions,
    oriented_endpoints,
    robot_is_eligible,
)
from .model import (
    CanonicalSolution,
    Operation,
    OperationKind,
    RobotRoute,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    FormalScope,
    FORMAL_SCOPE_V1,
    SplitKind,
)
from .solution import block_map, canonicalize

DeadlockRepair = Callable[
    [
        CanonicalSolution,
        ScientificConfig,
        ScheduleResult,
        tuple[tuple[int, tuple[int, ...]], ...],
        tuple[str, ...],
    ],
    ScheduleResult | None,
]


@dataclass
class SchedulerProfile:
    """Development-only counters/timers; it never participates in decisions."""

    number_of_operations: int = 0
    ess_calls: int = 0
    fixed_operations_scanned: int = 0
    relevant_fixed_operations_scanned: int = 0
    forbidden_interval_calls: int = 0
    forbidden_intervals_generated: int = 0
    operations_conflict_calls: int = 0
    wait_conflict_calls: int = 0
    total_time: float = 0.0
    canonical_time: float = 0.0
    route_build_time: float = 0.0
    operation_template_time: float = 0.0
    scheduler_main_loop_time: float = 0.0
    ess_time: float = 0.0
    forbidden_interval_time: float = 0.0
    interval_sort_sweep_time: float = 0.0
    conflict_validation_time: float = 0.0
    wait_conflict_validation_time: float = 0.0
    remaining_processing_time: float = 0.0

    @property
    def conflict_time(self) -> float:
        return self.conflict_validation_time + self.wait_conflict_validation_time

    @property
    def other_time(self) -> float:
        accounted = (
            self.canonical_time
            + self.route_build_time
            + self.operation_template_time
            + self.scheduler_main_loop_time
        )
        return max(0.0, self.total_time - accounted)

    @property
    def route_template_time(self) -> float:
        return self.route_build_time + self.operation_template_time

    def as_dict(self) -> dict[str, int | float]:
        return {
            "number_of_operations": self.number_of_operations,
            "number_of_ESS_calls": self.ess_calls,
            "fixed_operations_scanned": self.fixed_operations_scanned,
            "relevant_fixed_operations_scanned": self.relevant_fixed_operations_scanned,
            "forbidden_interval_calls": self.forbidden_interval_calls,
            "forbidden_intervals_generated": self.forbidden_intervals_generated,
            "operations_conflict_calls": self.operations_conflict_calls,
            "wait_conflict_calls": self.wait_conflict_calls,
            "total": self.total_time,
            "canonical": self.canonical_time,
            "route_template": self.route_template_time,
            "route_build": self.route_build_time,
            "operation_template": self.operation_template_time,
            "scheduler_main_loop": self.scheduler_main_loop_time,
            "ESS": self.ess_time,
            "forbidden_interval": self.forbidden_interval_time,
            "interval_sort_sweep": self.interval_sort_sweep_time,
            "conflict": self.conflict_time,
            "remaining_processing": self.remaining_processing_time,
            "other": self.other_time,
        }


def _expired_before_ready(
    operation: Operation, ready: float, config: ScientificConfig
) -> bool:
    tolerance = config.numeric_epsilon * max(1.0, abs(ready), abs(operation.end_time))
    return operation.end_time < ready - tolerance


def relevant_fixed_operations(
    fixed_operations: Sequence[Operation],
    robot_id: int,
    ready: float,
    config: ScientificConfig,
) -> tuple[Operation, ...]:
    """Return only cross-robot operations that can overlap start>=ready or its WAIT."""
    return tuple(
        operation
        for operation in fixed_operations
        if operation.robot_id != robot_id
        and not _expired_before_ready(operation, ready, config)
    )


class _RelevantFixedIndex:
    def __init__(self) -> None:
        self.operations: list[list[Operation]] = [[], [], [], []]
        self.end_times: list[list[float]] = [[], [], [], []]

    def add(self, operation: Operation) -> None:
        robot = operation.robot_id
        if self.end_times[robot] and operation.end_time < self.end_times[robot][-1]:
            raise ValueError("per-robot fixed operation end times must be monotone")
        self.operations[robot].append(operation)
        self.end_times[robot].append(operation.end_time)

    def relevant(
        self, robot_id: int, ready: float, config: ScientificConfig
    ) -> tuple[Operation, ...]:
        threshold = ready - config.numeric_epsilon * max(1.0, abs(ready))
        result: list[Operation] = []
        for other in range(4):
            if other == robot_id:
                continue
            start = bisect_left(self.end_times[other], threshold)
            if start:
                start -= 1
            result.extend(
                operation
                for operation in self.operations[other][start:]
                if not _expired_before_ready(operation, ready, config)
            )
        return tuple(result)


def build_robot_routes(
    solution: CanonicalSolution,
    config: ScientificConfig,
    orientations: Mapping[int, Sequence[int]] | None = None,
    *,
    x_split_validator: XSplitValidator | None = None,
) -> tuple[RobotRoute, ...]:
    blocks = block_map(solution, config, x_split_validator=x_split_validator)
    result: list[RobotRoute] = []
    for route in solution.routes:
        route_blocks = tuple(blocks[block_id] for block_id in route.block_ids)
        vector = (
            tuple(orientations[route.robot_id])
            if orientations is not None and route.robot_id in orientations
            else optimize_directions(route_blocks, config).orientations
        )
        result.append(RobotRoute(route.robot_id, route_blocks, vector))
    return tuple(result)


def build_operation_templates(
    route: RobotRoute, config: ScientificConfig
) -> tuple[Operation, ...]:
    templates: list[Operation] = []
    cursor = 0.0
    previous_end = None
    sequence_index = 0
    for block_index, (block, orientation) in enumerate(zip(route.blocks, route.orientations)):
        weld_start, weld_end = oriented_endpoints(block, orientation)
        if previous_end is not None:
            duration = math.dist(previous_end, weld_start) / config.empty_speed
            templates.append(
                Operation(
                    f"R{route.robot_id}:B{block_index}:MOVE",
                    route.robot_id,
                    OperationKind.MOVE,
                    cursor,
                    cursor + duration,
                    previous_end,
                    weld_start,
                    sequence_index,
                )
            )
            cursor += duration
            sequence_index += 1
        templates.append(
            Operation(
                f"R{route.robot_id}:B{block_index}:SETUP",
                route.robot_id,
                OperationKind.SETUP,
                cursor,
                cursor + config.t_pre,
                weld_start,
                weld_start,
                sequence_index,
                block.block_id,
            )
        )
        cursor += config.t_pre
        sequence_index += 1
        weld_duration = block.length / config.weld_speed
        templates.append(
            Operation(
                f"R{route.robot_id}:B{block_index}:WELD",
                route.robot_id,
                OperationKind.WELD,
                cursor,
                cursor + weld_duration,
                weld_start,
                weld_end,
                sequence_index,
                block.block_id,
            )
        )
        cursor += weld_duration
        sequence_index += 1
        templates.append(
            Operation(
                f"R{route.robot_id}:B{block_index}:POST",
                route.robot_id,
                OperationKind.POST,
                cursor,
                cursor + config.t_post,
                weld_end,
                weld_end,
                sequence_index,
                block.block_id,
            )
        )
        cursor += config.t_post
        sequence_index += 1
        previous_end = weld_end
    return tuple(templates)


def _at_start(template: Operation, start_time: float) -> Operation:
    return replace(
        template,
        start_time=start_time,
        end_time=start_time + template.duration,
    )


def earliest_safe_start_slow(
    template: Operation,
    ready: float,
    fixed_operations: Sequence[Operation],
    config: ScientificConfig,
    *,
    profile: SchedulerProfile | None = None,
) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
    ess_started = time.perf_counter()
    if profile is not None:
        profile.ess_calls += 1
        profile.fixed_operations_scanned += len(fixed_operations)
    intervals: list[tuple[float, float, int, str]] = []
    for fixed in fixed_operations:
        if fixed.robot_id == template.robot_id:
            continue
        interval_started = time.perf_counter()
        generated = forbidden_start_intervals(
            template.robot_id,
            template.duration,
            template.start,
            template.end,
            fixed,
            ready,
            config,
        )
        if profile is not None:
            profile.forbidden_interval_calls += 1
            profile.forbidden_intervals_generated += len(generated)
            profile.forbidden_interval_time += time.perf_counter() - interval_started
        for lo, hi in generated:
            intervals.append((lo, hi, fixed.robot_id, fixed.operation_id))
    sweep_started = time.perf_counter()
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    start = ready
    blockers: set[int] = set()
    blocker_ids: set[str] = set()
    for _ in range(len(intervals) + 2):
        # Spatial epsilon is already embedded in the projected interval. Do not
        # widen it again on the start-time axis.
        overlaps = [item for item in intervals if item[0] <= start <= item[1]]
        if not overlaps:
            break
        furthest = max(item[1] for item in overlaps)
        blockers.update(item[2] for item in overlaps)
        blocker_ids.update(item[3] for item in overlaps)
        next_start = max(
            math.nextafter(furthest, math.inf),
            furthest + config.numeric_epsilon * max(1.0, abs(furthest)),
        )
        if not math.isfinite(next_start) or next_start <= start:
            return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
        start = next_start
    else:
        if profile is not None:
            profile.interval_sort_sweep_time += time.perf_counter() - sweep_started
            profile.ess_time += time.perf_counter() - ess_started
        return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
    if profile is not None:
        profile.interval_sort_sweep_time += time.perf_counter() - sweep_started

    candidate = _at_start(template, start)
    validation_started = time.perf_counter()
    for fixed in fixed_operations:
        if fixed.robot_id == template.robot_id:
            continue
        if profile is not None:
            profile.operations_conflict_calls += 1
        if operations_conflict(candidate, fixed, config):
            if profile is not None:
                profile.conflict_validation_time += time.perf_counter() - validation_started
                profile.ess_time += time.perf_counter() - ess_started
            return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
    if profile is not None:
        profile.conflict_validation_time += time.perf_counter() - validation_started

    if start > ready:
        waiting = Operation(
            f"R{template.robot_id}:PROBE_WAIT",
            template.robot_id,
            OperationKind.WAIT,
            ready,
            start,
            template.start,
            template.start,
            template.sequence_index,
            template.block_id,
        )
        wait_started = time.perf_counter()
        for fixed in fixed_operations:
            if fixed.robot_id == template.robot_id:
                continue
            if profile is not None:
                profile.wait_conflict_calls += 1
            if operations_conflict(waiting, fixed, config):
                blockers.add(fixed.robot_id)
                blocker_ids.add(fixed.operation_id)
                if profile is not None:
                    profile.wait_conflict_validation_time += time.perf_counter() - wait_started
                    profile.ess_time += time.perf_counter() - ess_started
                return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
        if profile is not None:
            profile.wait_conflict_validation_time += time.perf_counter() - wait_started
    if profile is not None:
        profile.ess_time += time.perf_counter() - ess_started
    return start, tuple(sorted(blockers)), tuple(sorted(blocker_ids))


def earliest_safe_start_optimized(
    template: Operation,
    ready: float,
    relevant_fixed: Sequence[Operation],
    config: ScientificConfig,
    *,
    profile: SchedulerProfile | None = None,
) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
    """Equivalent ESS using expired filtering and one deterministic interval sweep."""
    ess_started = time.perf_counter()
    if profile is not None:
        profile.ess_calls += 1
        profile.relevant_fixed_operations_scanned += len(relevant_fixed)
    intervals: list[tuple[float, float, int, str]] = []
    for fixed in relevant_fixed:
        interval_started = time.perf_counter()
        generated = forbidden_start_intervals(
            template.robot_id,
            template.duration,
            template.start,
            template.end,
            fixed,
            ready,
            config,
        )
        if profile is not None:
            profile.forbidden_interval_calls += 1
            profile.forbidden_intervals_generated += len(generated)
            profile.forbidden_interval_time += time.perf_counter() - interval_started
        intervals.extend(
            (lo, hi, fixed.robot_id, fixed.operation_id) for lo, hi in generated
        )

    sweep_started = time.perf_counter()
    intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    start = ready
    blockers: set[int] = set()
    blocker_ids: set[str] = set()
    index = 0
    while index < len(intervals):
        while index < len(intervals) and intervals[index][1] < start:
            index += 1
        if index >= len(intervals) or intervals[index][0] > start:
            break
        furthest = start
        covered = False
        while index < len(intervals) and intervals[index][0] <= start:
            lo, hi, blocker, blocker_id = intervals[index]
            if lo <= start <= hi:
                covered = True
                furthest = max(furthest, hi)
                blockers.add(blocker)
                blocker_ids.add(blocker_id)
            index += 1
        if not covered:
            break
        next_start = max(
            math.nextafter(furthest, math.inf),
            furthest + config.numeric_epsilon * max(1.0, abs(furthest)),
        )
        if not math.isfinite(next_start) or next_start <= start:
            if profile is not None:
                profile.interval_sort_sweep_time += time.perf_counter() - sweep_started
                profile.ess_time += time.perf_counter() - ess_started
            return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
        start = next_start
    if profile is not None:
        profile.interval_sort_sweep_time += time.perf_counter() - sweep_started

    candidate = _at_start(template, start)
    validation_started = time.perf_counter()
    for fixed in relevant_fixed:
        if profile is not None:
            profile.operations_conflict_calls += 1
        if operations_conflict(candidate, fixed, config):
            if profile is not None:
                profile.conflict_validation_time += time.perf_counter() - validation_started
                profile.ess_time += time.perf_counter() - ess_started
            return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
    if profile is not None:
        profile.conflict_validation_time += time.perf_counter() - validation_started

    if start > ready:
        waiting = Operation(
            f"R{template.robot_id}:PROBE_WAIT",
            template.robot_id,
            OperationKind.WAIT,
            ready,
            start,
            template.start,
            template.start,
            template.sequence_index,
            template.block_id,
        )
        wait_started = time.perf_counter()
        for fixed in relevant_fixed:
            if profile is not None:
                profile.wait_conflict_calls += 1
            if operations_conflict(waiting, fixed, config):
                blockers.add(fixed.robot_id)
                blocker_ids.add(fixed.operation_id)
                if profile is not None:
                    profile.wait_conflict_validation_time += time.perf_counter() - wait_started
                    profile.ess_time += time.perf_counter() - ess_started
                return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
        if profile is not None:
            profile.wait_conflict_validation_time += time.perf_counter() - wait_started
    if profile is not None:
        profile.ess_time += time.perf_counter() - ess_started
    return start, tuple(sorted(blockers)), tuple(sorted(blocker_ids))


def earliest_safe_start(
    template: Operation,
    ready: float,
    fixed_operations: Sequence[Operation],
    config: ScientificConfig,
) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
    """Production ESS; preserves the original public full-fixed input contract."""
    relevant = relevant_fixed_operations(
        fixed_operations, template.robot_id, ready, config
    )
    return earliest_safe_start_optimized(template, ready, relevant, config)


def build_wait_for_graph(
    blockers_by_robot: Mapping[int, Sequence[int]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (robot, tuple(sorted(set(blockers))))
        for robot, blockers in sorted(blockers_by_robot.items())
        if blockers
    )


def wait_for_cycles(graph: Sequence[tuple[int, Sequence[int]]]) -> tuple[tuple[int, ...], ...]:
    adjacency = {node: tuple(neighbors) for node, neighbors in graph}
    cycles: set[tuple[int, ...]] = set()

    def visit(start: int, node: int, path: tuple[int, ...]) -> None:
        for neighbor in adjacency.get(node, ()):
            if neighbor == start:
                body = path
                rotations = [body[index:] + body[:index] for index in range(len(body))]
                cycles.add(min(rotations) + (min(rotations)[0],))
            elif neighbor not in path and len(path) <= len(adjacency):
                visit(start, neighbor, path + (neighbor,))

    for node in sorted(adjacency):
        visit(node, node, (node,))
    return tuple(sorted(cycles))


def _infeasible(message: str) -> ScheduleResult:
    return ScheduleResult(ScheduleStatus.INFEASIBLE, diagnostics=(message,))


def _template_semantics_error(
    operation: Operation, config: ScientificConfig
) -> str | None:
    tolerance = config.numeric_epsilon * max(1.0, operation.duration)
    distance = math.dist(operation.start, operation.end)
    if operation.kind is OperationKind.MOVE:
        expected = distance / config.empty_speed
        return None if abs(operation.duration - expected) <= tolerance else "bad MOVE duration"
    if operation.kind is OperationKind.WELD:
        expected = distance / config.weld_speed
        return None if abs(operation.duration - expected) <= tolerance else "bad WELD duration"
    if operation.kind is OperationKind.SETUP:
        if distance > config.numeric_epsilon or abs(operation.duration - config.t_pre) > tolerance:
            return "bad SETUP geometry/duration"
    elif operation.kind is OperationKind.POST:
        if distance > config.numeric_epsilon or abs(operation.duration - config.t_post) > tolerance:
            return "bad POST geometry/duration"
    elif operation.kind is OperationKind.WAIT and distance > config.numeric_epsilon:
        return "WAIT must be stationary"
    return None


def _validated_reference_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
) -> tuple[dict[int, tuple[Operation, ...]], ScheduleResult | None]:
    if any(robot not in range(4) for robot in templates_by_robot):
        return {}, _infeasible("template robot IDs must be 0..3")
    templates = {
        robot: tuple(templates_by_robot.get(robot, ())) for robot in range(4)
    }
    for robot, items in templates.items():
        if any(item.robot_id != robot for item in items):
            return {}, _infeasible(f"R{robot}: template robot identity mismatch")
        for operation in items:
            error = _template_semantics_error(operation, config)
            if error is not None:
                return {}, _infeasible(f"R{robot}/{operation.operation_id}: {error}")
        for previous, current in zip(items, items[1:]):
            if math.dist(previous.end, current.start) > config.numeric_epsilon:
                return {}, _infeasible(f"R{robot}: templates are spatially discontinuous")

    first_positions = {
        robot: items[0].start for robot, items in templates.items() if items
    }
    for left, right in ((0, 1), (2, 3)):
        if left in first_positions and right in first_positions:
            if (
                first_positions[left][0] + config.interference_dx
                > first_positions[right][0] + config.numeric_epsilon
            ):
                return {}, _infeasible(
                    f"illegal first-task same-rail order R{left}/R{right}"
                )
    initial_robots = sorted(first_positions)
    for index, first_robot in enumerate(initial_robots):
        for second_robot in initial_robots[index + 1 :]:
            first = first_positions[first_robot]
            second = first_positions[second_robot]
            if (
                abs(first[0] - second[0])
                <= config.interference_dx + config.numeric_epsilon
                and abs(first[1] - second[1])
                <= config.interference_dy + config.numeric_epsilon
            ):
                return {}, _infeasible(
                    f"initial theoretical interference R{first_robot}/R{second_robot}"
                )
    return templates, None


def reference_schedule_from_templates_slow(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ] = ((), (), (), ()),
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """Legacy Phase 1.1 list scheduler retained as the equivalence oracle."""
    main_started = time.perf_counter()
    templates, invalid = _validated_reference_templates(templates_by_robot, config)
    if invalid is not None:
        return invalid
    if profile is not None:
        profile.number_of_operations = sum(len(items) for items in templates.values())

    next_index = [0, 0, 0, 0]
    completion = [0.0, 0.0, 0.0, 0.0]
    current_point: list[tuple[float, float] | None] = [None, None, None, None]
    fixed: list[Operation] = []
    wait_count = [0, 0, 0, 0]

    try:
        while any(next_index[robot] < len(templates[robot]) for robot in range(4)):
            choices = []
            blockers_by_robot: dict[int, tuple[int, ...]] = {}
            blocker_ids_by_robot: dict[int, tuple[str, ...]] = {}
            for robot in range(4):
                if next_index[robot] >= len(templates[robot]):
                    continue
                template = templates[robot][next_index[robot]]
                start, blockers, blocker_ids = earliest_safe_start_slow(
                    template, completion[robot], fixed, config, profile=profile
                )
                blockers_by_robot[robot] = blockers
                blocker_ids_by_robot[robot] = blocker_ids
                remaining_started = time.perf_counter()
                remaining_processing = sum(
                    item.duration
                    for item in templates[robot][next_index[robot] :]
                    if item.kind
                    in (OperationKind.SETUP, OperationKind.WELD, OperationKind.POST)
                )
                if profile is not None:
                    profile.remaining_processing_time += time.perf_counter() - remaining_started
                if math.isfinite(start):
                    choices.append(
                        (
                            start,
                            -remaining_processing,
                            -completion[robot],
                            robot,
                            template,
                        )
                    )
            if not choices:
                graph = build_wait_for_graph(blockers_by_robot)
                cycles = wait_for_cycles(graph)
                result = ScheduleResult(
                    ScheduleStatus.DEADLOCK,
                    tuple(sorted(fixed, key=_operation_sort_key)),
                    None,
                    tuple(completion),
                    graph,
                    tuple(
                        [f"wait-for cycles={cycles}"]
                        + [
                            f"R{robot} blocked by {blocker_ids_by_robot.get(robot, ())}"
                            for robot in sorted(blocker_ids_by_robot)
                        ]
                    ),
                    directions,
                )
                if profile is not None:
                    profile.scheduler_main_loop_time += time.perf_counter() - main_started
                return result

            start, _, _, robot, template = min(choices, key=lambda item: item[:4])
            ready = completion[robot]
            if start > ready:
                point = (
                    current_point[robot]
                    if current_point[robot] is not None
                    else template.start
                )
                fixed.append(
                    Operation(
                        f"R{robot}:WAIT:{wait_count[robot]}",
                        robot,
                        OperationKind.WAIT,
                        ready,
                        start,
                        point,
                        point,
                        template.sequence_index,
                        template.block_id,
                    )
                )
                wait_count[robot] += 1
            operation = _at_start(template, start)
            fixed.append(operation)
            completion[robot] = operation.end_time
            current_point[robot] = operation.end
            next_index[robot] += 1
    except ValueError as error:
        return _infeasible(str(error))
    except (OverflowError, ArithmeticError) as error:
        return ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),))

    result = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        tuple(sorted(fixed, key=_operation_sort_key)),
        max(completion, default=0.0),
        tuple(completion),
        directions=directions,
    )
    if profile is not None:
        profile.scheduler_main_loop_time += time.perf_counter() - main_started
    return result


def _remaining_processing_suffix(
    templates: Mapping[int, Sequence[Operation]],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    suffixes = []
    for robot in range(4):
        items = templates[robot]
        suffix = [0.0] * (len(items) + 1)
        for index in range(len(items) - 1, -1, -1):
            suffix[index] = suffix[index + 1] + (
                items[index].duration
                if items[index].kind
                in (OperationKind.SETUP, OperationKind.WELD, OperationKind.POST)
                else 0.0
            )
        suffixes.append(tuple(suffix))
    return tuple(suffixes)  # type: ignore[return-value]


def reference_schedule_from_templates_optimized(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ] = ((), (), (), ()),
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """Semantics-equivalent list scheduler with indexed relevant fixed operations."""
    main_started = time.perf_counter()
    templates, invalid = _validated_reference_templates(templates_by_robot, config)
    if invalid is not None:
        return invalid
    if profile is not None:
        profile.number_of_operations = sum(len(items) for items in templates.values())
    suffix_started = time.perf_counter()
    remaining_suffix = _remaining_processing_suffix(templates)
    if profile is not None:
        profile.remaining_processing_time += time.perf_counter() - suffix_started

    next_index = [0, 0, 0, 0]
    completion = [0.0, 0.0, 0.0, 0.0]
    current_point: list[tuple[float, float] | None] = [None, None, None, None]
    fixed: list[Operation] = []
    fixed_index = _RelevantFixedIndex()
    wait_count = [0, 0, 0, 0]

    try:
        while any(next_index[robot] < len(templates[robot]) for robot in range(4)):
            choices = []
            blockers_by_robot: dict[int, tuple[int, ...]] = {}
            blocker_ids_by_robot: dict[int, tuple[str, ...]] = {}
            for robot in range(4):
                if next_index[robot] >= len(templates[robot]):
                    continue
                template = templates[robot][next_index[robot]]
                relevant = fixed_index.relevant(robot, completion[robot], config)
                start, blockers, blocker_ids = earliest_safe_start_optimized(
                    template,
                    completion[robot],
                    relevant,
                    config,
                    profile=profile,
                )
                blockers_by_robot[robot] = blockers
                blocker_ids_by_robot[robot] = blocker_ids
                remaining_started = time.perf_counter()
                remaining_processing = remaining_suffix[robot][next_index[robot]]
                if profile is not None:
                    profile.remaining_processing_time += time.perf_counter() - remaining_started
                if math.isfinite(start):
                    choices.append(
                        (
                            start,
                            -remaining_processing,
                            -completion[robot],
                            robot,
                            template,
                        )
                    )
            if not choices:
                graph = build_wait_for_graph(blockers_by_robot)
                cycles = wait_for_cycles(graph)
                result = ScheduleResult(
                    ScheduleStatus.DEADLOCK,
                    tuple(sorted(fixed, key=_operation_sort_key)),
                    None,
                    tuple(completion),
                    graph,
                    tuple(
                        [f"wait-for cycles={cycles}"]
                        + [
                            f"R{robot} blocked by {blocker_ids_by_robot.get(robot, ())}"
                            for robot in sorted(blocker_ids_by_robot)
                        ]
                    ),
                    directions,
                )
                if profile is not None:
                    profile.scheduler_main_loop_time += time.perf_counter() - main_started
                return result

            start, _, _, robot, template = min(choices, key=lambda item: item[:4])
            ready = completion[robot]
            if start > ready:
                point = current_point[robot] if current_point[robot] is not None else template.start
                waiting = Operation(
                    f"R{robot}:WAIT:{wait_count[robot]}",
                    robot,
                    OperationKind.WAIT,
                    ready,
                    start,
                    point,
                    point,
                    template.sequence_index,
                    template.block_id,
                )
                fixed.append(waiting)
                fixed_index.add(waiting)
                wait_count[robot] += 1
            operation = _at_start(template, start)
            fixed.append(operation)
            fixed_index.add(operation)
            completion[robot] = operation.end_time
            current_point[robot] = operation.end
            next_index[robot] += 1
    except ValueError as error:
        return _infeasible(str(error))
    except (OverflowError, ArithmeticError) as error:
        return ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),))

    result = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        tuple(sorted(fixed, key=_operation_sort_key)),
        max(completion, default=0.0),
        tuple(completion),
        directions=directions,
    )
    if profile is not None:
        profile.scheduler_main_loop_time += time.perf_counter() - main_started
    return result


def reference_schedule_from_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ] = ((), (), (), ()),
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    return reference_schedule_from_templates_optimized(
        templates_by_robot, config, directions=directions, profile=profile
    )


def _reference_schedule_with(
    solution: CanonicalSolution,
    config: ScientificConfig,
    template_scheduler,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    deadlock_repair: DeadlockRepair | None = None,
    x_split_validator: XSplitValidator | None = None,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    total_started = time.perf_counter()

    def finish(result: ScheduleResult) -> ScheduleResult:
        if profile is not None:
            profile.total_time += time.perf_counter() - total_started
        return result

    canonical_started = time.perf_counter()
    try:
        canonical = canonicalize(
            solution.parents,
            solution.patterns,
            solution.routes,
            config,
            revision=solution.revision,
            x_split_validator=x_split_validator,
        )
    except ValueError as error:
        if profile is not None:
            profile.canonical_time += time.perf_counter() - canonical_started
        return finish(_infeasible(str(error)))
    if profile is not None:
        profile.canonical_time += time.perf_counter() - canonical_started
    route_started = time.perf_counter()
    route_recorded = False
    template_started: float | None = None
    template_recorded = False
    try:
        routes = build_robot_routes(
            canonical,
            config,
            orientations,
            x_split_validator=x_split_validator,
        )
        for route in routes:
            for block in route.blocks:
                if not robot_is_eligible(block, route.robot_id, config):
                    if profile is not None:
                        profile.route_build_time += time.perf_counter() - route_started
                        route_recorded = True
                    return finish(_infeasible(f"robot R{route.robot_id} is ineligible for {block.block_id}"))
        if profile is not None:
            profile.route_build_time += time.perf_counter() - route_started
            route_recorded = True
        template_started = time.perf_counter()
        templates = {route.robot_id: build_operation_templates(route, config) for route in routes}
        if profile is not None:
            profile.operation_template_time += time.perf_counter() - template_started
            template_recorded = True
        result = template_scheduler(
            templates,
            config,
            directions=tuple(route.orientations for route in routes),
            profile=profile,
        )
        if result.status is ScheduleStatus.DEADLOCK and deadlock_repair is not None:
            repaired = deadlock_repair(
                canonical,
                config,
                result,
                result.wait_for_graph,
                result.diagnostics,
            )
            if repaired is not None:
                return finish(repaired)
        return finish(result)
    except ValueError as error:
        if profile is not None:
            if not route_recorded:
                profile.route_build_time += time.perf_counter() - route_started
            elif template_started is not None and not template_recorded:
                profile.operation_template_time += time.perf_counter() - template_started
        return finish(_infeasible(str(error)))
    except (OverflowError, ArithmeticError) as error:
        if profile is not None:
            if not route_recorded:
                profile.route_build_time += time.perf_counter() - route_started
            elif template_started is not None and not template_recorded:
                profile.operation_template_time += time.perf_counter() - template_started
        return finish(ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),)))


def reference_schedule_slow(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    deadlock_repair: DeadlockRepair | None = None,
    x_split_validator: XSplitValidator | None = None,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    return _reference_schedule_with(
        solution,
        config,
        reference_schedule_from_templates_slow,
        orientations=orientations,
        deadlock_repair=deadlock_repair,
        x_split_validator=x_split_validator,
        profile=profile,
    )


def reference_schedule_optimized(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    deadlock_repair: DeadlockRepair | None = None,
    x_split_validator: XSplitValidator | None = None,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    return _reference_schedule_with(
        solution,
        config,
        reference_schedule_from_templates_optimized,
        orientations=orientations,
        deadlock_repair=deadlock_repair,
        x_split_validator=x_split_validator,
        profile=profile,
    )


def reference_schedule(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    deadlock_repair: DeadlockRepair | None = None,
    x_split_validator: XSplitValidator | None = None,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """Production reference evaluator; scientifically identical optimized path."""
    return reference_schedule_optimized(
        solution,
        config,
        orientations=orientations,
        deadlock_repair=deadlock_repair,
        x_split_validator=x_split_validator,
        profile=profile,
    )


def _operation_sort_key(operation: Operation):
    return (
        operation.start_time,
        operation.end_time,
        operation.robot_id,
        operation.sequence_index,
        operation.kind.value,
        operation.operation_id,
    )


def _bounded_dispatch_recovery(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    baseline: ScheduleResult,
    *,
    state_budget: int,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """DFS over finite ESS dispatch choices; one popped prefix is one state.

    The baseline priority child is visited first. This is a bounded policy, not
    an infeasibility proof or an exact optimization procedure. No incumbent from
    an algorithm, time limit, RNG, or cross-call state participates.
    """
    if state_budget < 1:
        raise ValueError("deadlock state budget must be positive")
    if baseline.status is not ScheduleStatus.DEADLOCK:
        return baseline
    templates, invalid = _validated_reference_templates(templates_by_robot, config)
    if invalid is not None:
        return invalid
    suffix = _remaining_processing_suffix(templates)
    # next indices, completion, committed operations, per-robot WAIT counters.
    stack = [((0, 0, 0, 0), (0.0, 0.0, 0.0, 0.0), (), (0, 0, 0, 0))]
    expanded = 0
    best = None
    try:
        while stack and expanded < state_budget:
            indices, completion, fixed, waits = stack.pop()
            expanded += 1
            if all(indices[r] == len(templates[r]) for r in range(4)):
                candidate = ScheduleResult(
                    ScheduleStatus.FEASIBLE,
                    tuple(sorted(fixed, key=_operation_sort_key)),
                    max(completion), completion, directions=baseline.directions,
                )
                if best is None or (candidate.cmax, candidate.canonical_json()) < (
                    best.cmax, best.canonical_json()
                ):
                    best = candidate
                continue
            choices = []
            for robot in range(4):
                if indices[robot] == len(templates[robot]):
                    continue
                template = templates[robot][indices[robot]]
                relevant = relevant_fixed_operations(fixed, robot, completion[robot], config)
                start, _, _ = earliest_safe_start_optimized(
                    template, completion[robot], relevant, config, profile=profile
                )
                if math.isfinite(start):
                    choices.append((start, -suffix[robot][indices[robot]],
                                    -completion[robot], robot, template))
            for start, _, _, robot, template in sorted(
                choices, key=lambda choice: choice[:4], reverse=True
            ):
                new_indices, new_completion, new_waits = list(indices), list(completion), list(waits)
                operations = fixed
                if start > completion[robot]:
                    point = (templates[robot][indices[robot] - 1].end
                             if indices[robot] else template.start)
                    operations += (Operation(
                        f"R{robot}:WAIT:{waits[robot]}", robot, OperationKind.WAIT,
                        completion[robot], start, point, point,
                        template.sequence_index, template.block_id,
                    ),)
                    new_waits[robot] += 1
                operation = _at_start(template, start)
                operations += (operation,)
                new_indices[robot] += 1
                new_completion[robot] = operation.end_time
                stack.append((tuple(new_indices), tuple(new_completion), operations, tuple(new_waits)))
    except (ValueError, ArithmeticError, OverflowError) as error:
        return replace(baseline, status=ScheduleStatus.NUMERIC_FAILURE,
                       diagnostics=baseline.diagnostics + (f"bounded recovery numeric failure: {error}",),
                       baseline_deadlock=True, expanded_states=expanded, state_budget=state_budget)
    diagnostics = baseline.diagnostics + (
        "baseline DEADLOCK", f"expanded_states={expanded}", f"budget={state_budget}",
        f"recovery_exhausted={bool(stack)}", f"frontier_exhausted={not stack}",
    )
    return replace(
        best if best is not None else baseline,
        source="BOUNDED_DEADLOCK_RECOVERY" if best is not None else "BASELINE",
        diagnostics=diagnostics, baseline_deadlock=True, expanded_states=expanded,
        state_budget=state_budget, recovery_exhausted=bool(stack), frontier_exhausted=not stack,
    )


def reference_schedule_from_templates_formal(
    templates_by_robot, config: ScientificConfig, *, scope: FormalScope = FORMAL_SCOPE_V1,
    directions=((), (), (), ()), profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """Formal dispatch policy for operation fixtures; no parent-coverage claim."""
    scope.validate_implemented()
    baseline = reference_schedule_from_templates_optimized(
        templates_by_robot, config, directions=directions, profile=profile
    )
    result = _bounded_dispatch_recovery(
        templates_by_robot, config, baseline,
        state_budget=scope.deadlock_state_budget, profile=profile,
    ) if baseline.status is ScheduleStatus.DEADLOCK else baseline
    return replace(result, reference_policy_id=scope.reference_scheduler_policy_id,
                   scope_id=scope.scope_id, scope_hash=scope.scope_hash,
                   state_budget=scope.deadlock_state_budget)


def reference_schedule_formal(
    solution: CanonicalSolution, config: ScientificConfig, *,
    scope: FormalScope = FORMAL_SCOPE_V1,
    orientations: Mapping[int, Sequence[int]] | None = None,
    profile: SchedulerProfile | None = None,
) -> ScheduleResult:
    """The common formal evaluator. Development callbacks/X providers are absent."""
    scope.validate_implemented()
    if any(pattern.kind is SplitKind.X_SPLIT for pattern in solution.patterns):
        result = _infeasible("FORMAL_SCOPE_V1: optional X_SPLIT is EXCLUDED")
    else:
        def dispatch(templates, cfg, *, directions, profile):
            return reference_schedule_from_templates_formal(
                templates, cfg, scope=scope, directions=directions, profile=profile
            )
        result = _reference_schedule_with(
            solution, config, dispatch, orientations=orientations, profile=profile
        )
    return replace(result, reference_policy_id=scope.reference_scheduler_policy_id,
                   scope_id=scope.scope_id, scope_hash=scope.scope_hash,
                   state_budget=scope.deadlock_state_budget)


@dataclass(frozen=True)
class FormalReferenceEvaluator:
    scope: FormalScope = FORMAL_SCOPE_V1

    def __call__(self, solution, config, *, orientations=None):
        return reference_schedule_formal(solution, config, scope=self.scope, orientations=orientations)


def resolve_reference_evaluator(scope: FormalScope | None, evaluator=None):
    if scope is None:
        if isinstance(evaluator, FormalReferenceEvaluator) or evaluator is reference_schedule_formal:
            raise ValueError("formal evaluator requires explicit scope")
        return reference_schedule if evaluator is None else evaluator
    scope.validate_implemented()
    if evaluator is None or evaluator is reference_schedule_formal:
        return FormalReferenceEvaluator(scope)
    if isinstance(evaluator, FormalReferenceEvaluator) and evaluator.scope == scope:
        return evaluator
    raise ValueError("formal run requires the matching formal evaluator; development callbacks are unsupported")
