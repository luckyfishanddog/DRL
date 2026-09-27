from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math

from mrta_reference.model import (
    Operation,
    OperationKind,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    robot_rail,
)


DirectionVectors = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]


@dataclass(frozen=True)
class ExactScheduleResult:
    status: ScheduleStatus
    schedule: ScheduleResult | None
    dispatch_states: int
    pruned_states: int
    diagnostics: tuple[str, ...] = ()

    @property
    def cmax(self) -> float | None:
        return None if self.schedule is None else self.schedule.cmax


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
    result = max(lower, min(first, second)), min(upper, max(first, second))
    return result if result[0] <= result[1] else None


def _proximity_conflict(
    first: Operation, second: Operation, config: ScientificConfig
) -> bool:
    lower = max(first.start_time, second.start_time)
    upper = min(first.end_time, second.end_time)
    if upper < lower:
        return False
    overlap = (lower, upper)
    for axis, threshold in (
        (0, config.interference_dx),
        (1, config.interference_dy),
    ):
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


def _order_conflict(
    first: Operation, second: Operation, config: ScientificConfig
) -> bool:
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
    return _proximity_conflict(first, second, config) or _order_conflict(
        first, second, config
    )


def _project_to_start_axis(
    constraints: Sequence[tuple[float, float, float]], epsilon: float
) -> tuple[float, float] | None:
    """Project linear (start,time) constraints onto start via Fourier-Motzkin."""
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
    duration: float, fixed: Operation, ready: float
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
    for axis, threshold in (
        (0, config.interference_dx),
        (1, config.interference_dy),
    ):
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
                    constant
                    + config.interference_dx
                    - config.numeric_epsilon,
                )
            )
        else:
            order_constraints.append(
                (
                    -candidate_velocity,
                    candidate_velocity - fixed_velocity,
                    config.interference_dx
                    - constant
                    - config.numeric_epsilon,
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
    """Independent analytical ESS; no reference-scheduler helper is called."""
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
        containing = [item for item in intervals if item[0] <= start <= item[1]]
        if not containing:
            break
        end = max(item[1] for item in containing)
        next_start = max(
            math.nextafter(end, math.inf),
            end + config.numeric_epsilon * max(1.0, abs(end)),
        )
        if not math.isfinite(next_start) or next_start <= start:
            return math.inf
        start = next_start
    else:
        return math.inf

    candidate = replace(
        template, start_time=start, end_time=start + template.duration
    )
    if any(
        _operations_conflict(candidate, fixed, config)
        for fixed in fixed_operations
        if fixed.robot_id != template.robot_id
    ):
        return math.inf
    if start > ready:
        wait = Operation(
            "EXACT:WAIT:PROBE",
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


def _template_error(operation: Operation, config: ScientificConfig) -> str | None:
    if operation.robot_id not in range(4):
        return "robot ID must be 0..3"
    if not math.isfinite(operation.duration) or operation.duration < 0.0:
        return "duration must be finite and non-negative"
    tolerance = config.numeric_epsilon * max(1.0, operation.duration)
    distance = math.dist(operation.start, operation.end)
    if operation.kind is OperationKind.MOVE:
        if abs(operation.duration - distance / config.empty_speed) > tolerance:
            return "bad MOVE duration"
    elif operation.kind is OperationKind.WELD:
        if abs(operation.duration - distance / config.weld_speed) > tolerance:
            return "bad WELD duration"
    elif operation.kind is OperationKind.SETUP:
        if distance > config.numeric_epsilon or abs(
            operation.duration - config.t_pre
        ) > tolerance:
            return "bad SETUP geometry/duration"
    elif operation.kind is OperationKind.POST:
        if distance > config.numeric_epsilon or abs(
            operation.duration - config.t_post
        ) > tolerance:
            return "bad POST geometry/duration"
    elif operation.kind is OperationKind.WAIT and distance > config.numeric_epsilon:
        return "WAIT must be stationary"
    return None


def _normalize_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
) -> tuple[dict[int, tuple[Operation, ...]], str | None]:
    if any(robot not in range(4) for robot in templates_by_robot):
        return {}, "template robot IDs must be 0..3"
    templates = {
        robot: tuple(templates_by_robot.get(robot, ())) for robot in range(4)
    }
    for robot, items in templates.items():
        if any(item.robot_id != robot for item in items):
            return {}, f"R{robot}: template robot identity mismatch"
        for item in items:
            error = _template_error(item, config)
            if error is not None:
                return {}, f"R{robot}/{item.operation_id}: {error}"
        for previous, current in zip(items, items[1:]):
            if math.dist(previous.end, current.start) > config.numeric_epsilon:
                return {}, f"R{robot}: templates are spatially discontinuous"
    return templates, None


def _initial_error(
    templates: Mapping[int, Sequence[Operation]], config: ScientificConfig
) -> str | None:
    first_positions = {
        robot: items[0].start for robot, items in templates.items() if items
    }
    for left, right in ((0, 1), (2, 3)):
        if left in first_positions and right in first_positions:
            if (
                first_positions[left][0] + config.interference_dx
                > first_positions[right][0] + config.numeric_epsilon
            ):
                return f"illegal first-task same-rail order R{left}/R{right}"
    robots = sorted(first_positions)
    for index, first_robot in enumerate(robots):
        for second_robot in robots[index + 1 :]:
            first = first_positions[first_robot]
            second = first_positions[second_robot]
            if (
                abs(first[0] - second[0])
                <= config.interference_dx + config.numeric_epsilon
                and abs(first[1] - second[1])
                <= config.interference_dy + config.numeric_epsilon
            ):
                return f"initial theoretical interference R{first_robot}/R{second_robot}"
    return None


def _operation_sort_key(operation: Operation) -> tuple[object, ...]:
    return (
        operation.start_time,
        operation.end_time,
        operation.robot_id,
        operation.sequence_index,
        operation.kind.value,
        operation.operation_id,
    )


def exact_schedule_from_templates(
    templates_by_robot: Mapping[int, Sequence[Operation]],
    config: ScientificConfig,
    *,
    directions: DirectionVectors = ((), (), (), ()),
    branch_and_bound: bool = True,
) -> ExactScheduleResult:
    """Enumerate every robot-local-precedence-preserving dispatch interleaving."""
    templates, error = _normalize_templates(templates_by_robot, config)
    if error is not None:
        return ExactScheduleResult(
            ScheduleStatus.INFEASIBLE, None, 0, 0, (error,)
        )
    error = _initial_error(templates, config)
    if error is not None:
        return ExactScheduleResult(
            ScheduleStatus.INFEASIBLE, None, 1, 0, (error,)
        )
    active = tuple(robot for robot in range(4) if templates[robot])
    if not active:
        schedule = ScheduleResult(
            ScheduleStatus.FEASIBLE,
            (),
            0.0,
            (0.0, 0.0, 0.0, 0.0),
            directions=directions,
        )
        return ExactScheduleResult(ScheduleStatus.FEASIBLE, schedule, 1, 0)

    best: ScheduleResult | None = None
    dispatch_states = 0
    pruned_states = 0

    def visit(
        indices: tuple[int, int, int, int],
        completion: tuple[float, float, float, float],
        points: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ],
        fixed: tuple[Operation, ...],
        wait_count: tuple[int, int, int, int],
    ) -> None:
        nonlocal best, dispatch_states, pruned_states
        dispatch_states += 1
        if branch_and_bound and best is not None and best.cmax is not None:
            remaining_bound = max(
                completion[robot]
                + sum(
                    item.duration
                    for item in templates[robot][indices[robot] :]
                )
                for robot in active
            )
            tolerance = 1.0e-9 * max(1.0, abs(best.cmax))
            current_wait = sum(
                operation.duration
                for operation in fixed
                if operation.kind is OperationKind.WAIT
            )
            best_wait = sum(
                operation.duration
                for operation in best.operations
                if operation.kind is OperationKind.WAIT
            )
            # A Cmax tie is pruned only when accumulated WAIT cannot improve
            # the incumbent. Future WAIT is non-negative; schedule JSON is only
            # a deterministic representative after the formal metrics tie.
            if (
                remaining_bound > best.cmax + tolerance
                or (
                    remaining_bound >= best.cmax - tolerance
                    and current_wait >= best_wait - tolerance
                )
            ):
                pruned_states += 1
                return

        if all(indices[robot] == len(templates[robot]) for robot in active):
            candidate = ScheduleResult(
                ScheduleStatus.FEASIBLE,
                tuple(sorted(fixed, key=_operation_sort_key)),
                max(completion),
                completion,
                directions=directions,
            )
            candidate_wait = sum(
                operation.duration
                for operation in candidate.operations
                if operation.kind is OperationKind.WAIT
            )
            best_wait = (
                math.inf
                if best is None
                else sum(
                    operation.duration
                    for operation in best.operations
                    if operation.kind is OperationKind.WAIT
                )
            )
            tolerance = (
                0.0
                if best is None or best.cmax is None
                else 1.0e-9 * max(1.0, abs(candidate.cmax), abs(best.cmax))
            )
            if (
                best is None
                or candidate.cmax < best.cmax - tolerance
                or (
                    abs(candidate.cmax - best.cmax) <= tolerance
                    and (candidate_wait, candidate.canonical_json())
                    < (best_wait, best.canonical_json())
                )
            ):
                best = candidate
            return

        for robot in active:
            if indices[robot] >= len(templates[robot]):
                continue
            template = templates[robot][indices[robot]]
            start = _earliest_safe_start(
                template, completion[robot], fixed, config
            )
            if not math.isfinite(start):
                continue
            next_fixed = list(fixed)
            next_wait_count = list(wait_count)
            if start > completion[robot]:
                point = points[robot]
                if point is None:
                    point = template.start
                next_fixed.append(
                    Operation(
                        f"EXACT:R{robot}:WAIT:{next_wait_count[robot]}",
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
            operation = replace(
                template,
                start_time=start,
                end_time=start + template.duration,
            )
            next_fixed.append(operation)
            next_indices = list(indices)
            next_indices[robot] += 1
            next_completion = list(completion)
            next_completion[robot] = operation.end_time
            next_points = list(points)
            next_points[robot] = operation.end
            visit(
                tuple(next_indices),
                tuple(next_completion),
                tuple(next_points),
                tuple(next_fixed),
                tuple(next_wait_count),
            )

    try:
        visit(
            (0, 0, 0, 0),
            (0.0, 0.0, 0.0, 0.0),
            (None, None, None, None),
            (),
            (0, 0, 0, 0),
        )
    except (ArithmeticError, OverflowError, ValueError) as exception:
        return ExactScheduleResult(
            ScheduleStatus.NUMERIC_FAILURE,
            None,
            dispatch_states,
            pruned_states,
            (str(exception),),
        )
    if best is None:
        return ExactScheduleResult(
            ScheduleStatus.INFEASIBLE,
            None,
            dispatch_states,
            pruned_states,
            ("all dispatch interleavings are infeasible",),
        )
    return ExactScheduleResult(
        ScheduleStatus.FEASIBLE,
        best,
        dispatch_states,
        pruned_states,
    )
