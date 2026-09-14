from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from .model import CanonicalSolution, Operation, OperationKind, ScheduleResult, ScheduleStatus, ScientificConfig
from .scheduler import build_operation_templates, build_robot_routes, earliest_safe_start, reference_schedule


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


def tiny_scheduler_oracle(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    orientations: Mapping[int, Sequence[int]] | None = None,
) -> TinyOracleResult:
    """Exhaustively enumerate operation-dispatch alternatives for two tiny routes."""
    robot_routes = build_robot_routes(solution, config, orientations)
    reference = reference_schedule(solution, config, orientations=orientations)
    active = tuple(route.robot_id for route in robot_routes if route.blocks)
    if len(active) != 2:
        raise ValueError("tiny scheduler oracle requires exactly two non-empty robot routes")
    templates = {route.robot_id: build_operation_templates(route, config) for route in robot_routes if route.robot_id in active}
    if any(not 2 <= len(items) <= 4 for items in templates.values()):
        raise ValueError("tiny oracle supports 2..4 operations per active robot")

    best: ScheduleResult | None = None
    explored = 0

    def search(indices, completion, points, fixed, wait_count) -> None:
        nonlocal best, explored
        if all(indices[robot] == len(templates[robot]) for robot in active):
            explored += 1
            values = tuple(completion.get(robot, 0.0) for robot in range(4))
            candidate = ScheduleResult(
                ScheduleStatus.FEASIBLE,
                tuple(sorted(fixed, key=lambda op: (op.start_time, op.end_time, op.robot_id, op.operation_id))),
                max(values),
                values,
                directions=tuple(route.orientations for route in robot_routes),
            )
            if best is None or (candidate.cmax, candidate.canonical_json()) < (best.cmax, best.canonical_json()):
                best = candidate
            return
        if best is not None and max(completion.values(), default=0.0) >= best.cmax:
            return
        progressed = False
        for robot in active:
            if indices[robot] >= len(templates[robot]):
                continue
            template = templates[robot][indices[robot]]
            start, _, _ = earliest_safe_start(template, completion[robot], fixed, config)
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
            search(next_indices, next_completion, next_points, next_fixed, next_wait_count)
        if not progressed:
            explored += 1

    search(
        {robot: 0 for robot in active},
        {robot: 0.0 for robot in active},
        {},
        [],
        {robot: 0 for robot in active},
    )
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
