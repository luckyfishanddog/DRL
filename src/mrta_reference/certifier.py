from __future__ import annotations

from dataclasses import dataclass
import math

from .model import (
    CanonicalSolution,
    Operation,
    OperationKind,
    Rail,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    robot_rail,
)


@dataclass(frozen=True)
class CertificationReport:
    certified: bool
    errors: tuple[str, ...]
    recomputed_cmax: float | None


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
) -> CertificationReport:
    """Independently reconstruct geometry and timing; scheduler helpers are not called."""
    errors: list[str] = []
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
                    candidate_t.extend(
                        ((y - parent.start[1]) / dy, point_id)
                        for y, point_id in (
                            (config.by[0], "BY_LOWER"),
                            (6.0, "BY_CENTER"),
                            (config.by[1], "BY_UPPER"),
                        )
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
            blocks[f"{parent.parent_id}::0"] = (
                parent.parent_id, parent.start, point, first_length
            )
            blocks[f"{parent.parent_id}::1"] = (
                parent.parent_id, point, parent.end, second_length
            )

    if set(pattern_by_id) != set(parent_by_id):
        errors.append("parent pattern coverage mismatch")
    assigned = [block for route in solution.routes for block in route.block_ids]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(blocks):
        errors.append("welding block coverage mismatch")
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
        weld_rank = 0
        for operation in operations:
            if operation.start_time < previous_end_time - config.numeric_epsilon:
                errors.append(f"R{robot}: operation overlap")
            if operation.start_time > previous_end_time + config.numeric_epsilon:
                errors.append(f"R{robot}: uncovered idle gap; WAIT required")
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
            elif operation.kind is OperationKind.POST:
                if not _point_close(operation.start, operation.end, config) or not _close(operation.duration, config.t_post, config):
                    errors.append(f"{operation.operation_id}: bad POST")
            elif operation.kind is OperationKind.WELD:
                geometry = blocks.get(operation.block_id)
                if geometry is None:
                    errors.append(f"{operation.operation_id}: unknown WELD block")
                else:
                    if weld_rank < len(declared_directions):
                        expected_start, expected_end = (
                            (geometry[1], geometry[2])
                            if declared_directions[weld_rank] == 0
                            else (geometry[2], geometry[1])
                        )
                        endpoints_ok = _point_close(operation.start, expected_start, config) and _point_close(
                            operation.end, expected_end, config
                        )
                    else:
                        endpoints_ok = False
                    if not endpoints_ok or not _close(operation.duration, geometry[3] / config.weld_speed, config):
                        errors.append(f"{operation.operation_id}: bad WELD geometry/duration")
                weld_rank += 1
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
