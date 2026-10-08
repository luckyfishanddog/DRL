from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Callable

from .geometry import (
    build_legal_pattern_catalog,
    finite_x_split_validator,
    pattern_in_catalog,
)

from .model import (
    CanonicalSolution,
    Operation,
    OperationKind,
    ParentWeld,
    Rail,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    ScientificAmbiguityError,
    robot_rail,
    FormalScope,
)

XSplitValidator = Callable[[ParentWeld, SplitPattern, ScientificConfig], bool]


@dataclass(frozen=True)
class CertificationReport:
    certified: bool
    errors: tuple[str, ...]
    recomputed_cmax: float | None


def certify_template_schedule(templates_by_robot, schedule, config) -> CertificationReport:
    """Independent trajectory certificate for development E1-E4 MOVE fixtures.

    This validates the supplied templates, not parent weld coverage. Production
    weld schedules must use certify_schedule as well.
    """
    if schedule.status is not ScheduleStatus.FEASIBLE or schedule.cmax is None:
        return CertificationReport(False, ("schedule is not FEASIBLE",), None)
    errors = []
    completion = []
    first = {}
    for robot in range(4):
        templates = tuple(templates_by_robot.get(robot, ()))
        actual = sorted((op for op in schedule.operations if op.robot_id == robot),
                        key=lambda op: (op.start_time, op.end_time, op.sequence_index,
                                        0 if op.kind is OperationKind.WAIT else 1))
        nonwait = tuple(op for op in actual if op.kind is not OperationKind.WAIT)
        if len(nonwait) != len(templates):
            errors.append(f"R{robot}: template coverage")
        for expected, op in zip(templates, nonwait):
            if (op.operation_id, op.kind, op.sequence_index, op.block_id) != (
                expected.operation_id, expected.kind, expected.sequence_index, expected.block_id
            ) or not _point_close(op.start, expected.start, config) or not _point_close(
                op.end, expected.end, config
            ) or not _close(op.duration, expected.duration, config):
                errors.append(f"R{robot}: template mismatch")
            if op.kind in (OperationKind.MOVE, OperationKind.WELD):
                speed = config.empty_speed if op.kind is OperationKind.MOVE else config.weld_speed
                if not _close(op.duration, math.dist(op.start, op.end) / speed, config):
                    errors.append(f"R{robot}: incorrect motion duration")
            elif (not _point_close(op.start, op.end, config)
                  or not _close(op.duration, config.t_pre if op.kind is OperationKind.SETUP else config.t_post, config)):
                errors.append(f"R{robot}: incorrect stationary operation")
        ready = 0.0
        point = templates[0].start if templates else None
        if point is not None:
            first[robot] = point
        for op in actual:
            if not _close(op.start_time, ready, config):
                errors.append(f"R{robot}: overlap or uncovered idle")
            if point is None or not _point_close(point, op.start, config):
                errors.append(f"R{robot}: discontinuity")
            if op.kind is OperationKind.WAIT and not _point_close(op.start, op.end, config):
                errors.append(f"R{robot}: moving WAIT")
            ready, point = op.end_time, op.end
        if actual and actual[-1].kind is OperationKind.WAIT:
            errors.append(f"R{robot}: terminal WAIT outside task horizon")
        completion.append(ready)
    for left, right in ((0, 1), (2, 3)):
        if left in first and right in first and (
            first[left][0] + config.interference_dx > first[right][0] + config.numeric_epsilon
        ):
            errors.append("illegal initial rail order")
    for index, a in enumerate(schedule.operations):
        for b in schedule.operations[index + 1:]:
            if a.robot_id != b.robot_id and (_interferes(a, b, config) or _order_violation(a, b, config)):
                errors.append(f"trajectory conflict: {a.operation_id}/{b.operation_id}")
    cmax = max(completion)
    if not _close(cmax, schedule.cmax, config) or any(
        not _close(a, b, config) for a, b in zip(completion, schedule.robot_completion)
    ):
        errors.append("completion mismatch")
    return CertificationReport(not errors, tuple(errors), cmax)


def _close(left: float, right: float, config: ScientificConfig) -> bool:
    return abs(left - right) <= config.numeric_epsilon * max(1.0, abs(left), abs(right))


def _point_close(left, right, config: ScientificConfig) -> bool:
    return _close(left[0], right[0], config) and _close(left[1], right[1], config)


def _line(operation: Operation, axis: int) -> tuple[float, float]:
    if operation.duration == 0.0:
        return 0.0, operation.start[axis]
    velocity = (operation.end[axis] - operation.start[axis]) / operation.duration
    return velocity, operation.start[axis] - velocity * operation.start_time


def _band(a: float, b: float, limit: float, lo: float, hi: float):
    if abs(a) <= 1.0e-15:
        return (lo, hi) if abs(b) <= limit else None
    x = (-limit - b) / a
    y = (limit - b) / a
    return max(lo, min(x, y)), min(hi, max(x, y))


def _interferes(first: Operation, second: Operation, config: ScientificConfig) -> bool:
    lo = max(first.start_time, second.start_time)
    hi = min(first.end_time, second.end_time)
    if hi < lo:
        return False
    intersection = (lo, hi)
    for axis, threshold in ((0, config.interference_dx), (1, config.interference_dy)):
        va, ba = _line(first, axis)
        vb, bb = _line(second, axis)
        interval = _band(va - vb, ba - bb, threshold + config.numeric_epsilon, *intersection)
        if interval is None or interval[0] > interval[1]:
            return False
        intersection = (max(intersection[0], interval[0]), min(intersection[1], interval[1]))
    return intersection[0] <= intersection[1]


def _order_violation(first: Operation, second: Operation, config: ScientificConfig) -> bool:
    if robot_rail(first.robot_id) is not robot_rail(second.robot_id):
        return False
    left, right = (first, second) if first.robot_id in (0, 2) else (second, first)
    lo = max(left.start_time, right.start_time)
    hi = min(left.end_time, right.end_time)
    if hi < lo:
        return False
    vl, bl = _line(left, 0)
    vr, br = _line(right, 0)
    a = vl - vr
    b = bl - br + config.interference_dx
    return max(a * lo + b, a * hi + b) > config.numeric_epsilon


def _eligible(start, end, robot: int, config: ScientificConfig) -> bool:
    lower, upper = config.by
    rail = robot_rail(robot)
    if rail is Rail.UPPER:
        return min(start[1], end[1]) >= lower - config.numeric_epsilon
    return max(start[1], end[1]) <= upper + config.numeric_epsilon


def certify_schedule(
    solution: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
    scope: FormalScope | None = None,
) -> CertificationReport:
    """Independently reconstruct geometry and timing; scheduler helpers are not called."""
    errors: list[str] = []
    formal_catalog = None
    if scope is not None:
        scope.validate_implemented()
        formal_catalog = build_legal_pattern_catalog(solution.parents, config, scope)
        if (schedule.scope_id, schedule.scope_hash, schedule.reference_policy_id) != (
            scope.scope_id, scope.scope_hash, scope.reference_scheduler_policy_id
        ):
            errors.append("formal evaluator identity mismatch")
        if (
            scope.optional_x_split_policy == "EXCLUDED"
            and any(p.kind is SplitKind.X_SPLIT for p in solution.patterns)
        ):
            return CertificationReport(
                False,
                (f"{scope.scope_id}: optional X_SPLIT is EXCLUDED",),
                None,
            )
        # TASK_HORIZON_RELEASE_V1 includes final POST (closed endpoint), then
        # neither TCP nor rail occupancy. No invented parking/terminal WAIT.
        for route in solution.routes:
            rows = sorted((op for op in schedule.operations if op.robot_id == route.robot_id),
                          key=lambda op: (op.start_time, op.end_time, op.sequence_index))
            if not route.block_ids:
                if rows or schedule.robot_completion[route.robot_id] != 0.0:
                    errors.append("empty route must be undeployed with completion zero")
            elif not rows or rows[-1].kind is not OperationKind.POST:
                errors.append("formal task horizon must end at final POST")
    if schedule.status is not ScheduleStatus.FEASIBLE or schedule.cmax is None:
        return CertificationReport(False, ("schedule is not FEASIBLE",), None)

    parent_by_id = {parent.parent_id: parent for parent in solution.parents}
    if len(parent_by_id) != len(solution.parents):
        errors.append("duplicate parent")
    pattern_by_id = {}
    blocks = {}
    for pattern in solution.patterns:
        if pattern.parent_id in pattern_by_id:
            errors.append(f"{pattern.parent_id}: more than one pattern")
            continue
        pattern_by_id[pattern.parent_id] = pattern
        parent = parent_by_id.get(pattern.parent_id)
        if parent is None:
            errors.append(f"{pattern.parent_id}: unknown parent")
            continue
        if formal_catalog is not None and not pattern_in_catalog(
            parent, pattern, formal_catalog, config
        ):
            errors.append(
                f"{scope.scope_id}: pattern is absent from legal catalog: "
                f"{pattern.pattern_id}"
            )
        if pattern.kind is SplitKind.WHOLE:
            blocks[f"{parent.parent_id}::whole"] = (
                parent.parent_id, parent.start, parent.end, parent.length
            )
        else:
            if pattern.t is None or not (0.0 < pattern.t < 1.0):
                errors.append(f"{parent.parent_id}: invalid split parameter")
                continue
            point = parent.point(pattern.t)
            first_length = math.dist(parent.start, point)
            second_length = math.dist(point, parent.end)
            if min(first_length, second_length) + config.numeric_epsilon < config.min_child_length:
                errors.append(f"{parent.parent_id}: child below Lmin")
            if not _close(first_length + second_length, parent.length, config):
                errors.append(f"{parent.parent_id}: length is not conserved")
            if pattern.kind is SplitKind.Y_SPLIT:
                upper_whole = min(parent.start[1], parent.end[1]) >= config.by[0] - config.numeric_epsilon
                lower_whole = max(parent.start[1], parent.end[1]) <= config.by[1] + config.numeric_epsilon
                if pattern.mandatory != (not (upper_whole or lower_whole)):
                    errors.append(f"{parent.parent_id}: incorrect mandatory-split identity")
                if not (config.by[0] - config.numeric_epsilon <= point[1] <= config.by[1] + config.numeric_epsilon):
                    errors.append(f"{parent.parent_id}: Y split outside By")
                candidate_t = []
                dy = parent.end[1] - parent.start[1]
                if dy != 0.0:
                    # Independent reconstruction: retain legacy inner lines and
                    # add outer lines without trusting the generator or flag.
                    inner = min(0.20, config.delta_y)
                    y_lines = [(6.0-inner, "BY_LOWER"), (6.0, "BY_CENTER"),
                               (6.0+inner, "BY_UPPER")]
                    if config.delta_y > 0.20:
                        y_lines.extend(((config.by[0], "BY_OUTER_LOWER"),
                                        (config.by[1], "BY_OUTER_UPPER")))
                    candidate_t.extend(
                        ((y - parent.start[1]) / dy, point_id)
                        for y, point_id in y_lines
                    )
                midpoint = parent.point(0.5)
                if config.by[0] - config.numeric_epsilon <= midpoint[1] <= config.by[1] + config.numeric_epsilon:
                    candidate_t.append((0.5, "MIDPOINT"))
                kept = []
                for value, point_id in candidate_t:
                    if not (0.0 < value < 1.0):
                        continue
                    if value * parent.length + config.numeric_epsilon < config.min_child_length:
                        continue
                    if (1.0 - value) * parent.length + config.numeric_epsilon < config.min_child_length:
                        continue
                    if any(abs(value - old_value) <= config.numeric_epsilon for old_value, _ in kept):
                        continue
                    kept.append((value, point_id))
                if not any(
                    abs(pattern.t - value) <= config.numeric_epsilon and pattern.point_id == point_id
                    for value, point_id in kept
                ):
                    errors.append(f"{parent.parent_id}: Y split is not a specified candidate")
            elif pattern.kind is SplitKind.X_SPLIT:
                if not (
                    min(parent.start[1], parent.end[1]) >= config.by[0] - config.numeric_epsilon
                    or max(parent.start[1], parent.end[1]) <= config.by[1] + config.numeric_epsilon
                ):
                    errors.append(f"{parent.parent_id}: X split cannot replace mandatory Y handover")
                if scope is None and x_split_validator is None:
                    raise ScientificAmbiguityError("retained X_SPLIT requires an explicit formal XSplitValidator")
                if scope is None and x_split_validator is not None and not x_split_validator(parent, pattern, config):
                    errors.append(
                        f"{pattern.pattern_id}: rejected by explicit XSplitValidator"
                    )
            first = (parent.parent_id, parent.start, point, first_length)
            second = (parent.parent_id, point, parent.end, second_length)
            if pattern.kind is SplitKind.X_SPLIT and (
                (parent.start[0] + point[0]) / 2.0
                > (point[0] + parent.end[0]) / 2.0
            ):
                first, second = second, first
            blocks[f"{parent.parent_id}::0"] = first
            blocks[f"{parent.parent_id}::1"] = second

    if set(pattern_by_id) != set(parent_by_id):
        errors.append("parent pattern coverage mismatch")
    assigned = [block for route in solution.routes for block in route.block_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(blocks):
        errors.append("welding block coverage mismatch")
    assigned_robot = {
        block_id: route.robot_id
        for route in solution.routes
        for block_id in route.block_ids
    }
    for pattern in solution.patterns:
        if pattern.kind is SplitKind.X_SPLIT:
            expected = (0, 1) if pattern.rail is Rail.UPPER else (2, 3)
            actual = (
                assigned_robot.get(f"{pattern.parent_id}::0"),
                assigned_robot.get(f"{pattern.parent_id}::1"),
            )
            if actual != expected:
                errors.append(
                    f"{pattern.parent_id}: X_SPLIT spatial assignment must be R{expected[0]}/R{expected[1]}"
                )
    for route in solution.routes:
        for left, right in zip(route.block_ids, route.block_ids[1:]):
            if left.endswith("::0") and right == left[:-1] + "1":
                errors.append(f"{left.split('::')[0]}: consecutive same-robot children not canonicalized")
            if left.endswith("::1") and right == left[:-1] + "0":
                errors.append(f"{left.split('::')[0]}: consecutive same-robot children not canonicalized")
        for block_id in route.block_ids:
            if block_id in blocks and not _eligible(blocks[block_id][1], blocks[block_id][2], route.robot_id, config):
                errors.append(f"R{route.robot_id}: ineligible for {block_id}")

    operations_by_robot = {
        robot: sorted(
            (operation for operation in schedule.operations if operation.robot_id == robot),
            key=lambda operation: (operation.start_time, operation.end_time, operation.operation_id),
        )
        for robot in range(4)
    }
    for robot, operations in operations_by_robot.items():
        route = solution.routes[robot]
        declared_directions = schedule.directions[robot]
        if len(declared_directions) != len(route.block_ids) or any(
            direction not in (0, 1) for direction in declared_directions
        ):
            errors.append(f"R{robot}: invalid declared direction vector")
        expected_endpoints = {}
        for block_index, block_id in enumerate(route.block_ids):
            geometry = blocks.get(block_id)
            if geometry is None or block_index >= len(declared_directions):
                continue
            expected_endpoints[block_id] = (
                (geometry[1], geometry[2])
                if declared_directions[block_index] == 0
                else (geometry[2], geometry[1])
            )
        non_wait = [operation for operation in operations if operation.kind is not OperationKind.WAIT]
        expected_kinds = []
        for index, block_id in enumerate(route.block_ids):
            if index:
                expected_kinds.append((OperationKind.MOVE, None))
            expected_kinds.extend(
                ((OperationKind.SETUP, block_id), (OperationKind.WELD, block_id), (OperationKind.POST, block_id))
            )
        if [(operation.kind, operation.block_id) for operation in non_wait] != expected_kinds:
            errors.append(f"R{robot}: operation/route precedence mismatch")
            continue
        previous_end_time = 0.0
        previous_point = None
        for operation in operations:
            if operation.start_time < previous_end_time - config.numeric_epsilon:
                errors.append(f"R{robot}: operation overlap")
            if operation.start_time > previous_end_time + config.numeric_epsilon:
                errors.append(f"R{robot}: uncovered idle gap; WAIT required")
            if previous_point is not None and not _point_close(
                previous_point, operation.start, config
            ):
                errors.append(
                    f"R{robot}: spatial discontinuity before {operation.operation_id}"
                )
            if operation.kind is OperationKind.WAIT:
                if not _point_close(operation.start, operation.end, config):
                    errors.append(f"{operation.operation_id}: WAIT moves")
                if previous_point is not None and not _point_close(operation.start, previous_point, config):
                    errors.append(f"{operation.operation_id}: WAIT not at prior endpoint")
            elif operation.kind is OperationKind.MOVE:
                expected = math.dist(operation.start, operation.end) / config.empty_speed
                if not _close(operation.duration, expected, config):
                    errors.append(f"{operation.operation_id}: bad MOVE duration")
                if previous_point is not None and not _point_close(operation.start, previous_point, config):
                    errors.append(f"{operation.operation_id}: MOVE discontinuity")
            elif operation.kind is OperationKind.SETUP:
                if not _point_close(operation.start, operation.end, config) or not _close(operation.duration, config.t_pre, config):
                    errors.append(f"{operation.operation_id}: bad SETUP")
                expected = expected_endpoints.get(operation.block_id)
                if expected is None or not _point_close(operation.start, expected[0], config):
                    errors.append(
                        f"{operation.operation_id}: SETUP is not at declared weld start"
                    )
            elif operation.kind is OperationKind.POST:
                if not _point_close(operation.start, operation.end, config) or not _close(operation.duration, config.t_post, config):
                    errors.append(f"{operation.operation_id}: bad POST")
                expected = expected_endpoints.get(operation.block_id)
                if expected is None or not _point_close(operation.start, expected[1], config):
                    errors.append(
                        f"{operation.operation_id}: POST is not at declared weld end"
                    )
            elif operation.kind is OperationKind.WELD:
                geometry = blocks.get(operation.block_id)
                if geometry is None:
                    errors.append(f"{operation.operation_id}: unknown WELD block")
                else:
                    expected = expected_endpoints.get(operation.block_id)
                    if expected is not None:
                        expected_start, expected_end = expected
                        endpoints_ok = _point_close(operation.start, expected_start, config) and _point_close(
                            operation.end, expected_end, config
                        )
                    else:
                        endpoints_ok = False
                    if not endpoints_ok or not _close(operation.duration, geometry[3] / config.weld_speed, config):
                        errors.append(f"{operation.operation_id}: bad WELD geometry/duration")
            previous_end_time = operation.end_time
            previous_point = operation.end

    # Independent initial-order, continuous interference, and non-passing audit.
    first_points = {}
    for robot, operations in operations_by_robot.items():
        setup = next((op for op in operations if op.kind is OperationKind.SETUP), None)
        if setup is not None:
            first_points[robot] = setup.start
    for left, right in ((0, 1), (2, 3)):
        if left in first_points and right in first_points:
            if first_points[left][0] + config.interference_dx > first_points[right][0] + config.numeric_epsilon:
                errors.append(f"R{left}/R{right}: illegal initial order")

    all_operations = tuple(schedule.operations)
    if len({operation.operation_id for operation in all_operations}) != len(all_operations):
        errors.append("duplicate operation ID")
    for index, first in enumerate(all_operations):
        for second in all_operations[index + 1 :]:
            if first.robot_id == second.robot_id:
                continue
            if _interferes(first, second, config):
                errors.append(f"interference: {first.operation_id}/{second.operation_id}")
            if _order_violation(first, second, config):
                errors.append(f"same-rail non-passing: {first.operation_id}/{second.operation_id}")

    completion = tuple(
        max((operation.end_time for operation in operations_by_robot[robot]), default=0.0)
        for robot in range(4)
    )
    if any(not _close(completion[robot], schedule.robot_completion[robot], config) for robot in range(4)):
        errors.append("per-robot completion mismatch")
    recomputed_cmax = max(completion, default=0.0)
    cmax_epsilon = 1.0e-9 * max(1.0, abs(recomputed_cmax))
    if abs(recomputed_cmax - schedule.cmax) > cmax_epsilon:
        errors.append("reported Cmax mismatch")
    return CertificationReport(not errors, tuple(errors), recomputed_cmax)
