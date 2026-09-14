from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
import math

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


def earliest_safe_start(
    template: Operation,
    ready: float,
    fixed_operations: Sequence[Operation],
    config: ScientificConfig,
) -> tuple[float, tuple[int, ...], tuple[str, ...]]:
    intervals: list[tuple[float, float, int, str]] = []
    for fixed in fixed_operations:
        if fixed.robot_id == template.robot_id:
            continue
        for lo, hi in forbidden_start_intervals(
            template.robot_id,
            template.duration,
            template.start,
            template.end,
            fixed,
            ready,
            config,
        ):
            intervals.append((lo, hi, fixed.robot_id, fixed.operation_id))
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
        return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))

    candidate = _at_start(template, start)
    if any(
        operations_conflict(candidate, fixed, config)
        for fixed in fixed_operations
        if fixed.robot_id != template.robot_id
    ):
        return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))

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
        for fixed in fixed_operations:
            if fixed.robot_id != template.robot_id and operations_conflict(waiting, fixed, config):
                blockers.add(fixed.robot_id)
                blocker_ids.add(fixed.operation_id)
                return math.inf, tuple(sorted(blockers)), tuple(sorted(blocker_ids))
    return start, tuple(sorted(blockers)), tuple(sorted(blocker_ids))


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


def reference_schedule_from_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ] = ((), (), (), ()),
) -> ScheduleResult:
    """Run the reference list policy on caller-supplied precedence templates."""
    if any(robot not in range(4) for robot in templates_by_robot):
        return _infeasible("template robot IDs must be 0..3")
    templates = {
        robot: tuple(templates_by_robot.get(robot, ())) for robot in range(4)
    }
    for robot, items in templates.items():
        if any(item.robot_id != robot for item in items):
            return _infeasible(f"R{robot}: template robot identity mismatch")
        for operation in items:
            error = _template_semantics_error(operation, config)
            if error is not None:
                return _infeasible(f"R{robot}/{operation.operation_id}: {error}")
        for previous, current in zip(items, items[1:]):
            if math.dist(previous.end, current.start) > config.numeric_epsilon:
                return _infeasible(f"R{robot}: templates are spatially discontinuous")

    first_positions = {
        robot: items[0].start for robot, items in templates.items() if items
    }
    for left, right in ((0, 1), (2, 3)):
        if left in first_positions and right in first_positions:
            if (
                first_positions[left][0] + config.interference_dx
                > first_positions[right][0] + config.numeric_epsilon
            ):
                return _infeasible(
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
                return _infeasible(
                    f"initial theoretical interference R{first_robot}/R{second_robot}"
                )

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
                start, blockers, blocker_ids = earliest_safe_start(
                    template, completion[robot], fixed, config
                )
                blockers_by_robot[robot] = blockers
                blocker_ids_by_robot[robot] = blocker_ids
                remaining_processing = sum(
                    item.duration
                    for item in templates[robot][next_index[robot] :]
                    if item.kind
                    in (OperationKind.SETUP, OperationKind.WELD, OperationKind.POST)
                )
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
                return ScheduleResult(
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

    return ScheduleResult(
        ScheduleStatus.FEASIBLE,
        tuple(sorted(fixed, key=_operation_sort_key)),
        max(completion, default=0.0),
        tuple(completion),
        directions=directions,
    )


def reference_schedule(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    deadlock_repair: DeadlockRepair | None = None,
    x_split_validator: XSplitValidator | None = None,
) -> ScheduleResult:
    """Deterministic operation-level list scheduler with no default repair policy."""
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
        return _infeasible(str(error))
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
                    return _infeasible(f"robot R{route.robot_id} is ineligible for {block.block_id}")
        templates = {route.robot_id: build_operation_templates(route, config) for route in routes}
        result = reference_schedule_from_templates(
            templates,
            config,
            directions=tuple(route.orientations for route in routes),
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
                return repaired
        return result
    except ValueError as error:
        return _infeasible(str(error))
    except (OverflowError, ArithmeticError) as error:
        return ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),))


def _operation_sort_key(operation: Operation):
    return (
        operation.start_time,
        operation.end_time,
        operation.robot_id,
        operation.sequence_index,
        operation.kind.value,
        operation.operation_id,
    )
