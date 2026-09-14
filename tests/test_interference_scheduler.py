from __future__ import annotations

import math

import pytest

from mrta_reference.geometry import continuous_interference, operations_conflict, same_rail_order_violation
from mrta_reference.model import (
    Operation,
    OperationKind,
    ParentWeld,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.scheduler import (
    build_operation_templates,
    build_robot_routes,
    earliest_safe_start,
    reference_schedule,
    wait_for_cycles,
)
from mrta_reference.solution import canonicalize


CONFIG = ScientificConfig()


def _operation(
    name: str,
    robot: int,
    kind: OperationKind,
    times,
    start,
    end,
) -> Operation:
    return Operation(name, robot, kind, times[0], times[1], start, end, 0)


def test_continuous_moving_moving_crossing_and_nearly_parallel_separation() -> None:
    crossing_a = _operation("a", 0, OperationKind.MOVE, (0, 10), (0, 0), (10, 0))
    crossing_b = _operation("b", 2, OperationKind.WELD, (0, 10), (5, -5), (5, 5))
    assert continuous_interference(crossing_a, crossing_b, CONFIG)
    parallel = _operation("p", 2, OperationKind.MOVE, (0, 10), (0, 0.500000001), (10, 0.500000002))
    assert not continuous_interference(crossing_a, parallel, CONFIG)


@pytest.mark.parametrize(
    "kind",
    (OperationKind.SETUP, OperationKind.POST, OperationKind.WAIT),
)
def test_stationary_operation_kinds_occupy_space(kind: OperationKind) -> None:
    stationary = _operation(kind.value, 2, kind, (2, 8), (5, 0), (5, 0))
    moving = _operation("moving", 0, OperationKind.MOVE, (0, 10), (0, 0), (10, 0))
    assert continuous_interference(stationary, moving, CONFIG)


def test_stationary_stationary_endpoint_touching_and_zero_velocity_components() -> None:
    first = _operation("first", 0, OperationKind.SETUP, (0, 1), (0, 0), (0, 0))
    second = _operation("second", 2, OperationKind.POST, (1, 2), (0.5, 0.5), (0.5, 0.5))
    assert continuous_interference(first, second, CONFIG)
    outside = _operation("outside", 2, OperationKind.POST, (0, 1), (0.500001, 0), (0.500001, 0))
    assert not continuous_interference(first, outside, CONFIG)


def test_same_rail_non_passing_is_checked_even_without_2d_proximity() -> None:
    left = _operation("left", 0, OperationKind.WELD, (0, 10), (4, 10), (7, 10))
    right = _operation("right", 1, OperationKind.WELD, (0, 10), (6, 11), (6, 11))
    assert not continuous_interference(left, right, CONFIG)
    assert same_rail_order_violation(left, right, CONFIG)
    assert operations_conflict(left, right, CONFIG)
    boundary_left = _operation("boundary-left", 0, OperationKind.POST, (0, 1), (5.5, 10), (5.5, 10))
    boundary_right = _operation("boundary-right", 1, OperationKind.POST, (0, 1), (6.0, 11), (6.0, 11))
    assert not same_rail_order_violation(boundary_left, boundary_right, CONFIG)


def test_earliest_safe_start_is_finite_and_analytic() -> None:
    fixed = _operation("fixed", 2, OperationKind.POST, (0, 5), (5, 0), (5, 0))
    template = _operation("candidate", 0, OperationKind.MOVE, (0, 10), (0, 0), (10, 0))
    start, blockers, blocker_ids = earliest_safe_start(template, 0.0, (fixed,), CONFIG)
    assert start > 0.5
    assert start == pytest.approx(0.5, abs=1.0e-9)
    assert blockers == (2,)
    assert blocker_ids == ("fixed",)


def _four_robot_solution(*, bad_order: bool = False):
    coordinates = {
        "u0": ((8.0 if bad_order else 1.0), 10.0),
        "u1": ((2.0 if bad_order else 10.0), 10.0),
        "l0": (1.0, 2.0),
        "l1": (10.0, 2.0),
    }
    parents = tuple(
        ParentWeld(name, start, (start[0] + 1.0, start[1])) for name, start in coordinates.items()
    )
    patterns = tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents)
    return canonicalize(
        parents,
        patterns,
        {
            0: ("u0::whole",),
            1: ("u1::whole",),
            2: ("l0::whole",),
            3: ("l1::whole",),
        },
        CONFIG,
    )


def test_operation_timeline_has_no_home_move_and_is_non_preemptive() -> None:
    solution = _four_robot_solution()
    robot_route = build_robot_routes(solution, CONFIG)[0]
    templates = build_operation_templates(robot_route, CONFIG)
    assert [operation.kind for operation in templates] == [
        OperationKind.SETUP,
        OperationKind.WELD,
        OperationKind.POST,
    ]
    assert templates[0].start_time == 0.0
    assert templates[0].duration == CONFIG.t_pre
    assert templates[1].duration == pytest.approx(1.0 / CONFIG.weld_speed)
    assert templates[2].duration == CONFIG.t_post
    assert all(left.end_time == right.start_time for left, right in zip(templates, templates[1:]))


def test_reference_scheduler_is_feasible_deterministic_and_parallel_when_safe() -> None:
    solution = _four_robot_solution()
    first = reference_schedule(solution, CONFIG)
    second = reference_schedule(solution, CONFIG)
    assert first.status is ScheduleStatus.FEASIBLE
    assert first.canonical_json() == second.canonical_json()
    setups = [operation for operation in first.operations if operation.kind is OperationKind.SETUP]
    assert {operation.start_time for operation in setups} == {0.0}
    assert first.cmax == pytest.approx(CONFIG.process_time(1.0))


def test_illegal_first_task_same_rail_order_is_infeasible() -> None:
    result = reference_schedule(_four_robot_solution(bad_order=True), CONFIG)
    assert result.status is ScheduleStatus.INFEASIBLE
    assert "first-task" in result.diagnostics[0]


def test_constructed_wait_for_cycle_detection_is_deterministic() -> None:
    graph = ((0, (1,)), (1, (0,)), (2, (3,)), (3, (2,)))
    assert wait_for_cycles(graph) == ((0, 1, 0), (2, 3, 2))


def test_constructed_route_deadlock_reports_blockers_without_hidden_repair() -> None:
    parents = (
        ParentWeld("left-to-right", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("right-to-left", (10.0, 8.0), (2.0, 8.0)),
    )
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("left-to-right::whole",), 1: ("right-to-left::whole",)},
        CONFIG,
    )
    result = reference_schedule(solution, CONFIG, orientations={0: (0,), 1: (0,)})
    assert result.status is ScheduleStatus.DEADLOCK
    assert result.wait_for_graph
    assert any("blocked by" in item for item in result.diagnostics)
