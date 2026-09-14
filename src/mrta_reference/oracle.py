from __future__ import annotations

from dataclasses import dataclass, replace
import math
from collections.abc import Mapping, Sequence

from .geometry import XSplitValidator
from .model import (
    CanonicalSolution,
    Operation,
    OperationKind,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    robot_rail,
)
from .scheduler import (
    build_operation_templates,
    build_robot_routes,
    reference_schedule,
    reference_schedule_from_templates,
)


@dataclass(frozen=True)
class TinyOracleResult:
    status: ScheduleStatus
    best_cmax: float | None
    schedule: ScheduleResult | None
    explored_dispatches: int
    reference_status: ScheduleStatus
    reference_cmax: float | None
    feasibility_agreement: bool
    scheduler_gap: float | None


def _line(operation: Operation, axis: int) -> tuple[float, float]:
    if operation.duration == 0.0:
        return 0.0, operation.start[axis]
    velocity = (operation.end[axis] - operation.start[axis]) / operation.duration
    return velocity, operation.start[axis] - velocity * operation.start_time


def _band_interval(
    slope: float,
    intercept: float,
    limit: float,
    lower: float,
    upper: float,
) -> tuple[float, float] | None:
    if abs(slope) <= 1.0e-15:
        return (lower, upper) if abs(intercept) <= limit else None
    first = (-limit - intercept) / slope
    second = (limit - intercept) / slope
    interval = max(lower, min(first, second)), min(upper, max(first, second))
    return interval if interval[0] <= interval[1] else None


def _conflicts(first: Operation, second: Operation, config: ScientificConfig) -> bool:
    lower = max(first.start_time, second.start_time)
    upper = min(first.end_time, second.end_time)
    if upper < lower:
        return False
    overlap = (lower, upper)
    for axis, threshold in ((0, config.interference_dx), (1, config.interference_dy)):
        first_velocity, first_intercept = _line(first, axis)
        second_velocity, second_intercept = _line(second, axis)
        interval = _band_interval(
            first_velocity - second_velocity,
            first_intercept - second_intercept,
            threshold + config.numeric_epsilon,
            overlap[0],
            overlap[1],
        )
        if interval is None:
            return False
        overlap = max(overlap[0], interval[0]), min(overlap[1], interval[1])
        if overlap[0] > overlap[1]:
            return False
    return True


def _order_conflict(first: Operation, second: Operation, config: ScientificConfig) -> bool:
    if robot_rail(first.robot_id) is not robot_rail(second.robot_id):
        return False
    left, right = (first, second) if first.robot_id in (0, 2) else (second, first)
    lower = max(left.start_time, right.start_time)
    upper = min(left.end_time, right.end_time)
    if upper < lower:
        return False
    left_velocity, left_intercept = _line(left, 0)
    right_velocity, right_intercept = _line(right, 0)
    slope = left_velocity - right_velocity
    intercept = left_intercept - right_intercept + config.interference_dx
    return max(slope * lower + intercept, slope * upper + intercept) > config.numeric_epsilon


def _operations_conflict(
    first: Operation, second: Operation, config: ScientificConfig
) -> bool:
    return _conflicts(first, second, config) or _order_conflict(first, second, config)


def _project_to_start_axis(
    constraints: Sequence[tuple[float, float, float]],
    epsilon: float,
) -> tuple[float, float] | None:
    """Eliminate time by Fourier-Motzkin, independently of the reference LP path."""
    lower_lines: list[tuple[float, float]] = []
    upper_lines: list[tuple[float, float]] = []
    start_inequalities: list[tuple[float, float]] = []
    for start_coefficient, time_coefficient, bound in constraints:
        if time_coefficient > 1.0e-15:
            upper_lines.append(
                (bound / time_coefficient, -start_coefficient / time_coefficient)
            )
        elif time_coefficient < -1.0e-15:
            lower_lines.append(
                (bound / time_coefficient, -start_coefficient / time_coefficient)
            )
        else:
            start_inequalities.append((start_coefficient, bound))

    for lower_constant, lower_slope in lower_lines:
        for upper_constant, upper_slope in upper_lines:
            start_inequalities.append(
                (lower_slope - upper_slope, upper_constant - lower_constant)
            )

    lower_bound = -math.inf
    upper_bound = math.inf
    for coefficient, bound in start_inequalities:
        if abs(coefficient) <= 1.0e-15:
            if bound < -epsilon:
                return None
        elif coefficient > 0.0:
            upper_bound = min(upper_bound, (bound + epsilon) / coefficient)
        else:
            lower_bound = max(lower_bound, (bound + epsilon) / coefficient)
    if lower_bound > upper_bound:
        return None
    return lower_bound, upper_bound


def _base_overlap_constraints(
    duration: float,
    fixed: Operation,
    ready: float,
) -> list[tuple[float, float, float]]:
    return [
        (1.0, -1.0, 0.0),
        (-1.0, 1.0, duration),
        (0.0, -1.0, -fixed.start_time),
        (0.0, 1.0, fixed.end_time),
        (-1.0, 0.0, -max(ready, fixed.start_time - duration)),
        (1.0, 0.0, fixed.end_time),
    ]


def _forbidden_start_intervals(
    robot_id: int,
    duration: float,
    start_point: tuple[float, float],
    end_point: tuple[float, float],
    fixed: Operation,
    ready: float,
    config: ScientificConfig,
) -> tuple[tuple[float, float], ...]:
    overlap = _base_overlap_constraints(duration, fixed, ready)
    proximity = list(overlap)
    for axis, threshold in ((0, config.interference_dx), (1, config.interference_dy)):
        candidate_velocity = (
            0.0
            if duration == 0.0
            else (end_point[axis] - start_point[axis]) / duration
        )
        fixed_velocity, fixed_intercept = _line(fixed, axis)
        constant = start_point[axis] - fixed_intercept
        limit = threshold + config.numeric_epsilon
        proximity.append(
            (-candidate_velocity, candidate_velocity - fixed_velocity, limit - constant)
        )
        proximity.append(
            (candidate_velocity, fixed_velocity - candidate_velocity, limit + constant)
        )

    intervals: list[tuple[float, float]] = []
    projected = _project_to_start_axis(proximity, config.numeric_epsilon)
    if projected is not None:
        intervals.append(projected)

    if robot_rail(robot_id) is robot_rail(fixed.robot_id) and robot_id != fixed.robot_id:
        candidate_velocity = (
            0.0 if duration == 0.0 else (end_point[0] - start_point[0]) / duration
        )
        fixed_velocity, fixed_intercept = _line(fixed, 0)
        constant = start_point[0] - fixed_intercept
        order_constraints = list(overlap)
        if robot_id in (0, 2):
            order_constraints.append(
                (
                    candidate_velocity,
                    fixed_velocity - candidate_velocity,
                    constant + config.interference_dx - config.numeric_epsilon,
                )
            )
        else:
            order_constraints.append(
                (
                    -candidate_velocity,
                    candidate_velocity - fixed_velocity,
                    config.interference_dx - constant - config.numeric_epsilon,
                )
            )
        projected = _project_to_start_axis(order_constraints, config.numeric_epsilon)
        if projected is not None:
            intervals.append(projected)
    return tuple(intervals)


def _earliest_safe_start(
    template: Operation,
    ready: float,
    fixed_operations: Sequence[Operation],
    config: ScientificConfig,
) -> float:
    intervals: list[tuple[float, float]] = []
    for fixed in fixed_operations:
        if fixed.robot_id == template.robot_id:
            continue
        intervals.extend(
            _forbidden_start_intervals(
                template.robot_id,
                template.duration,
                template.start,
                template.end,
                fixed,
                ready,
                config,
            )
        )
    intervals.sort()
    start = ready
    for _ in range(len(intervals) + 2):
        containing = [interval for interval in intervals if interval[0] <= start <= interval[1]]
        if not containing:
            break
        end = max(interval[1] for interval in containing)
        start = max(
            math.nextafter(end, math.inf),
            end + config.numeric_epsilon * max(1.0, abs(end)),
        )
        if not math.isfinite(start):
            return math.inf
    else:
        return math.inf

    candidate = replace(template, start_time=start, end_time=start + template.duration)
    if any(
        _operations_conflict(candidate, fixed, config)
        for fixed in fixed_operations
        if fixed.robot_id != template.robot_id
    ):
        return math.inf
    if start > ready:
        wait = Operation(
            "ORACLE:WAIT:PROBE",
            template.robot_id,
            OperationKind.WAIT,
            ready,
            start,
            template.start,
            template.start,
            template.sequence_index,
            template.block_id,
        )
        if any(
            _operations_conflict(wait, fixed, config)
            for fixed in fixed_operations
            if fixed.robot_id != template.robot_id
        ):
            return math.inf
    return start


def _initially_valid(
    templates: Mapping[int, Sequence[Operation]], config: ScientificConfig
) -> bool:
    first_positions = {robot: items[0].start for robot, items in templates.items() if items}
    for left, right in ((0, 1), (2, 3)):
        if left in first_positions and right in first_positions:
            if (
                first_positions[left][0] + config.interference_dx
                > first_positions[right][0] + config.numeric_epsilon
            ):
                return False
    robots = sorted(first_positions)
    for index, first_robot in enumerate(robots):
        for second_robot in robots[index + 1 :]:
            first = first_positions[first_robot]
            second = first_positions[second_robot]
            if (
                abs(first[0] - second[0]) <= config.interference_dx + config.numeric_epsilon
                and abs(first[1] - second[1]) <= config.interference_dy + config.numeric_epsilon
            ):
                return False
    return True


def _validate_templates(
    templates: Mapping[int, Sequence[Operation]], config: ScientificConfig
) -> tuple[int, int]:
    active = tuple(sorted(robot for robot, items in templates.items() if items))
    if len(active) != 2:
        raise ValueError("tiny scheduler oracle requires exactly two active robots")
    for robot in active:
        items = templates[robot]
        if any(item.robot_id != robot for item in items):
            raise ValueError(f"R{robot}: template robot identity mismatch")
        for item in items:
            distance = math.dist(item.start, item.end)
            tolerance = config.numeric_epsilon * max(1.0, item.duration)
            if item.kind is OperationKind.MOVE:
                valid = abs(item.duration - distance / config.empty_speed) <= tolerance
            elif item.kind is OperationKind.WELD:
                valid = abs(item.duration - distance / config.weld_speed) <= tolerance
            elif item.kind is OperationKind.SETUP:
                valid = (
                    distance <= config.numeric_epsilon
                    and abs(item.duration - config.t_pre) <= tolerance
                )
            elif item.kind is OperationKind.POST:
                valid = (
                    distance <= config.numeric_epsilon
                    and abs(item.duration - config.t_post) <= tolerance
                )
            else:
                valid = distance <= config.numeric_epsilon
            if not valid:
                raise ValueError(
                    f"R{robot}/{item.operation_id}: invalid {item.kind.value} template semantics"
                )
        for previous, current in zip(items, items[1:]):
            if math.dist(previous.end, current.start) > config.numeric_epsilon:
                raise ValueError(f"R{robot}: precedence templates are spatially discontinuous")
    return active


def _search_dispatches(
    templates: Mapping[int, Sequence[Operation]],
    active: tuple[int, int],
    config: ScientificConfig,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ],
) -> tuple[ScheduleResult | None, int]:
    if not _initially_valid(templates, config):
        return None, 1
    best: ScheduleResult | None = None
    explored = 0

    def visit(indices, completion, points, fixed, wait_count) -> None:
        nonlocal best, explored
        if all(indices[robot] == len(templates[robot]) for robot in active):
            explored += 1
            values = tuple(completion.get(robot, 0.0) for robot in range(4))
            candidate = ScheduleResult(
                ScheduleStatus.FEASIBLE,
                tuple(
                    sorted(
                        fixed,
                        key=lambda operation: (
                            operation.start_time,
                            operation.end_time,
                            operation.robot_id,
                            operation.operation_id,
                        ),
                    )
                ),
                max(values),
                values,
                directions=directions,
            )
            if best is None or (candidate.cmax, candidate.canonical_json()) < (
                best.cmax,
                best.canonical_json(),
            ):
                best = candidate
            return
        progressed = False
        for robot in active:
            if indices[robot] >= len(templates[robot]):
                continue
            template = templates[robot][indices[robot]]
            start = _earliest_safe_start(template, completion[robot], fixed, config)
            if not math.isfinite(start):
                continue
            progressed = True
            next_fixed = list(fixed)
            next_wait_count = dict(wait_count)
            if start > completion[robot]:
                point = points.get(robot, template.start)
                next_fixed.append(
                    Operation(
                        f"ORACLE:R{robot}:WAIT:{next_wait_count[robot]}",
                        robot,
                        OperationKind.WAIT,
                        completion[robot],
                        start,
                        point,
                        point,
                        template.sequence_index,
                        template.block_id,
                    )
                )
                next_wait_count[robot] += 1
            operation = replace(template, start_time=start, end_time=start + template.duration)
            next_fixed.append(operation)
            next_indices = dict(indices)
            next_indices[robot] += 1
            next_completion = dict(completion)
            next_completion[robot] = operation.end_time
            next_points = dict(points)
            next_points[robot] = operation.end
            visit(
                next_indices,
                next_completion,
                next_points,
                next_fixed,
                next_wait_count,
            )
        if not progressed:
            explored += 1

    visit(
        {robot: 0 for robot in active},
        {robot: 0.0 for robot in active},
        {},
        [],
        {robot: 0 for robot in active},
    )
    return best, explored


def _comparison_result(
    best: ScheduleResult | None,
    explored: int,
    reference: ScheduleResult,
) -> TinyOracleResult:
    if best is None:
        return TinyOracleResult(
            ScheduleStatus.INFEASIBLE,
            None,
            None,
            explored,
            reference.status,
            reference.cmax,
            reference.status is not ScheduleStatus.FEASIBLE,
            None,
        )
    gap = reference.cmax - best.cmax if reference.cmax is not None else None
    return TinyOracleResult(
        ScheduleStatus.FEASIBLE,
        best.cmax,
        best,
        explored,
        reference.status,
        reference.cmax,
        reference.status is ScheduleStatus.FEASIBLE,
        gap,
    )


def tiny_scheduler_oracle_from_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: tuple[
        tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
    ] = ((), (), (), ()),
) -> TinyOracleResult:
    """Enumerate all interleavings of two manual routes with 2..4 operations each."""
    templates = {robot: tuple(items) for robot, items in templates_by_robot.items()}
    active = _validate_templates(templates, config)
    if any(not 2 <= len(templates[robot]) <= 4 for robot in active):
        raise ValueError("manual tiny oracle requires 2..4 operations per active robot")
    best, explored = _search_dispatches(templates, active, config, directions)
    reference = reference_schedule_from_templates(templates, config, directions=directions)
    return _comparison_result(best, explored, reference)


def tiny_scheduler_oracle(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
    x_split_validator: XSplitValidator | None = None,
) -> TinyOracleResult:
    """Convenience wrapper supporting up to two welding blocks (about 7 ops/route)."""
    robot_routes = build_robot_routes(
        solution,
        config,
        orientations,
        x_split_validator=x_split_validator,
    )
    active = tuple(route.robot_id for route in robot_routes if route.blocks)
    if len(active) != 2:
        raise ValueError("tiny scheduler oracle requires exactly two non-empty robot routes")
    templates = {
        route.robot_id: build_operation_templates(route, config)
        for route in robot_routes
        if route.robot_id in active
    }
    if any(not 2 <= len(items) <= 7 for items in templates.values()):
        raise ValueError("solution oracle supports at most two welding blocks per active robot")
    active_pair = _validate_templates(templates, config)
    directions = tuple(route.orientations for route in robot_routes)
    best, explored = _search_dispatches(templates, active_pair, config, directions)
    reference = reference_schedule(
        solution,
        config,
        orientations=orientations,
        x_split_validator=x_split_validator,
    )
    return _comparison_result(best, explored, reference)
