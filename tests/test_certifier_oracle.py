from __future__ import annotations

from dataclasses import replace

import pytest

from mrta_reference.certifier import certify_schedule
from mrta_reference.model import (
    Operation,
    OperationKind,
    ParentWeld,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.oracle import tiny_scheduler_oracle, tiny_scheduler_oracle_from_templates
from mrta_reference.scheduler import (
    build_operation_templates,
    build_robot_routes,
    reference_schedule,
)
from mrta_reference.solution import canonicalize, official_metrics


CONFIG = ScientificConfig()
ORACLE_CONFIG = ScientificConfig(
    weld_speed=1.0,
    empty_speed=1.0,
    t_pre=1.0,
    t_post=1.0,
)


def _solution():
    parents = (
        ParentWeld("left", (1.0, 10.0), (2.0, 10.0)),
        ParentWeld("right", (10.0, 10.0), (11.0, 10.0)),
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("left::whole",), 1: ("right::whole",)},
        CONFIG,
    )


def _two_block_solution():
    parents = (
        ParentWeld("left-a", (1.0, 10.0), (2.0, 10.0)),
        ParentWeld("left-b", (3.0, 10.0), (4.0, 10.0)),
        ParentWeld("right", (10.0, 10.0), (11.0, 10.0)),
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {
            0: ("left-a::whole", "left-b::whole"),
            1: ("right::whole",),
        },
        CONFIG,
    )


def _two_by_two_solution():
    parents = tuple(
        ParentWeld(name, start, end)
        for name, start, end in (
            ("l0", (1.0, 10.0), (2.0, 10.0)),
            ("l1", (3.0, 10.0), (4.0, 10.0)),
            ("r0", (10.0, 10.0), (11.0, 10.0)),
            ("r1", (12.0, 10.0), (13.0, 10.0)),
        )
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {
            0: ("l0::whole", "l1::whole"),
            1: ("r0::whole", "r1::whole"),
        },
        CONFIG,
    )


def _move_templates(robot_id, points):
    operations = []
    cursor = 0.0
    for index, (start, end) in enumerate(zip(points, points[1:])):
        duration = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5
        operations.append(
            Operation(
                f"R{robot_id}:{index}",
                robot_id,
                OperationKind.MOVE,
                cursor,
                cursor + duration,
                start,
                end,
                index,
            )
        )
        cursor += duration
    return tuple(operations)


def test_independent_certifier_accepts_reference_and_recomputes_cmax() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    report = certify_schedule(solution, schedule, CONFIG)
    assert report.certified, report.errors
    assert report.recomputed_cmax == schedule.cmax


def test_certifier_rejects_tampered_cmax_duration_and_operation_overlap() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    bad_cmax = replace(schedule, cmax=schedule.cmax + 1.0)
    assert "reported Cmax mismatch" in certify_schedule(solution, bad_cmax, CONFIG).errors

    operations = list(schedule.operations)
    weld_index = next(index for index, op in enumerate(operations) if op.kind is OperationKind.WELD)
    operations[weld_index] = replace(operations[weld_index], end_time=operations[weld_index].end_time + 1.0)
    bad_duration = replace(schedule, operations=tuple(operations))
    report = certify_schedule(solution, bad_duration, CONFIG)
    assert not report.certified
    assert any("bad WELD" in error for error in report.errors)
    assert any("overlap" in error for error in report.errors)

    reversed_directions = list(schedule.directions)
    reversed_directions[0] = (1,)
    bad_direction = replace(schedule, directions=tuple(reversed_directions))
    assert any("bad WELD" in error for error in certify_schedule(solution, bad_direction, CONFIG).errors)


def test_certifier_accepts_explicit_safe_initial_wait_and_checks_wait_occupancy() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    shifted = []
    first_r1 = next(op for op in schedule.operations if op.robot_id == 1)
    shifted.append(
        Operation(
            "R1:WAIT:test",
            1,
            OperationKind.WAIT,
            0.0,
            5.0,
            first_r1.start,
            first_r1.start,
            first_r1.sequence_index,
            first_r1.block_id,
        )
    )
    for operation in schedule.operations:
        shifted.append(
            replace(
                operation,
                start_time=operation.start_time + (5.0 if operation.robot_id == 1 else 0.0),
                end_time=operation.end_time + (5.0 if operation.robot_id == 1 else 0.0),
            )
        )
    completion = list(schedule.robot_completion)
    completion[1] += 5.0
    waited = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        tuple(sorted(shifted, key=lambda op: (op.start_time, op.end_time, op.robot_id, op.operation_id))),
        max(completion),
        tuple(completion),
        directions=schedule.directions,
    )
    assert certify_schedule(solution, waited, CONFIG).certified

    bad_wait_ops = list(waited.operations)
    wait_index = next(index for index, op in enumerate(bad_wait_ops) if op.kind is OperationKind.WAIT)
    bad_wait_ops[wait_index] = replace(bad_wait_ops[wait_index], end=(9.0, 9.0))
    assert any("WAIT moves" in error for error in certify_schedule(solution, replace(waited, operations=tuple(bad_wait_ops)), CONFIG).errors)


def test_certifier_rejects_all_spatial_teleportation_interfaces() -> None:
    solution = _two_block_solution()
    schedule = reference_schedule(solution, CONFIG)
    assert certify_schedule(solution, schedule, CONFIG).certified

    def tamper(kind, occurrence, **changes):
        operations = list(schedule.operations)
        matches = [index for index, operation in enumerate(operations) if operation.robot_id == 0 and operation.kind is kind]
        index = matches[occurrence]
        operations[index] = replace(operations[index], **changes)
        return replace(schedule, operations=tuple(operations))

    first_weld = next(
        operation
        for operation in schedule.operations
        if operation.robot_id == 0 and operation.kind is OperationKind.WELD
    )
    shifted_weld = tamper(
        OperationKind.WELD,
        0,
        start=(first_weld.start[0], first_weld.start[1] + 1.0),
        end=(first_weld.end[0], first_weld.end[1] + 1.0),
    )
    assert any(
        "spatial discontinuity" in error or "bad WELD geometry" in error
        for error in certify_schedule(solution, shifted_weld, CONFIG).errors
    )

    first_post = next(
        operation
        for operation in schedule.operations
        if operation.robot_id == 0 and operation.kind is OperationKind.POST
    )
    shifted_post = tamper(
        OperationKind.POST,
        0,
        start=(first_post.start[0], first_post.start[1] + 1.0),
        end=(first_post.end[0], first_post.end[1] + 1.0),
    )
    assert any(
        "POST is not at declared weld end" in error
        for error in certify_schedule(solution, shifted_post, CONFIG).errors
    )

    move = next(
        operation
        for operation in schedule.operations
        if operation.robot_id == 0 and operation.kind is OperationKind.MOVE
    )
    bad_move = tamper(
        OperationKind.MOVE,
        0,
        end=(move.end[0], move.end[1] + 1.0),
    )
    assert any(
        "spatial discontinuity" in error
        for error in certify_schedule(solution, bad_move, CONFIG).errors
    )

    # Insert a real, time-contiguous WAIT, then move its stationary point.
    first_r1 = next(operation for operation in schedule.operations if operation.robot_id == 1)
    wait = Operation(
        "R1:WAIT:adversarial",
        1,
        OperationKind.WAIT,
        0.0,
        1.0,
        (9.0, 9.0),
        (9.0, 9.0),
        first_r1.sequence_index,
        first_r1.block_id,
    )
    shifted_operations = [wait]
    shifted_operations.extend(
        replace(
            operation,
            start_time=operation.start_time + (1.0 if operation.robot_id == 1 else 0.0),
            end_time=operation.end_time + (1.0 if operation.robot_id == 1 else 0.0),
        )
        for operation in schedule.operations
    )
    completion = list(schedule.robot_completion)
    completion[1] += 1.0
    bad_wait = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        tuple(shifted_operations),
        max(completion),
        tuple(completion),
        directions=schedule.directions,
    )
    assert any(
        "spatial discontinuity" in error
        for error in certify_schedule(solution, bad_wait, CONFIG).errors
    )


def test_pre_post_count_and_robot_eligibility_are_independently_rejected() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    operations = tuple(op for op in schedule.operations if not (op.robot_id == 0 and op.kind is OperationKind.POST))
    report = certify_schedule(solution, replace(schedule, operations=operations), CONFIG)
    assert any("operation/route precedence" in error for error in report.errors)

    operations = tuple(
        op
        for op in schedule.operations
        if not (op.robot_id == 0 and op.kind is OperationKind.SETUP)
    )
    report = certify_schedule(solution, replace(schedule, operations=operations), CONFIG)
    assert any("operation/route precedence" in error for error in report.errors)

    ineligible = canonicalize(solution.parents, solution.patterns, {2: ("left::whole",), 1: ("right::whole",)}, CONFIG)
    result = reference_schedule(ineligible, CONFIG)
    assert result.status is ScheduleStatus.INFEASIBLE
    assert "ineligible" in result.diagnostics[0]


def test_certifier_rejects_bad_move_hidden_gap_and_same_robot_overlap() -> None:
    solution = _two_block_solution()
    schedule = reference_schedule(solution, CONFIG)
    operations = list(schedule.operations)
    move_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.robot_id == 0 and operation.kind is OperationKind.MOVE
    )
    operations[move_index] = replace(
        operations[move_index], end_time=operations[move_index].end_time + 0.5
    )
    assert any(
        "bad MOVE duration" in error
        for error in certify_schedule(
            solution, replace(schedule, operations=tuple(operations)), CONFIG
        ).errors
    )

    operations = list(schedule.operations)
    weld_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.robot_id == 0 and operation.kind is OperationKind.WELD
    )
    post_index = next(
        index
        for index, operation in enumerate(operations)
        if operation.robot_id == 0 and operation.kind is OperationKind.POST
    )
    operations[weld_index] = replace(
        operations[weld_index],
        start_time=operations[weld_index].start_time + 1.0,
        end_time=operations[weld_index].end_time + 1.0,
    )
    operations[post_index] = replace(
        operations[post_index],
        start_time=operations[post_index].start_time + 1.0,
        end_time=operations[post_index].end_time + 1.0,
    )
    completion = list(schedule.robot_completion)
    completion[0] += 1.0
    gap_schedule = replace(
        schedule,
        operations=tuple(operations),
        cmax=max(completion),
        robot_completion=tuple(completion),
    )
    assert any(
        "uncovered idle gap" in error
        for error in certify_schedule(solution, gap_schedule, CONFIG).errors
    )

    operations = list(schedule.operations)
    operations[weld_index] = replace(
        operations[weld_index],
        start_time=operations[weld_index].start_time - 1.0,
        end_time=operations[weld_index].end_time - 1.0,
    )
    overlap_schedule = replace(schedule, operations=tuple(operations))
    assert any(
        "operation overlap" in error
        for error in certify_schedule(solution, overlap_schedule, CONFIG).errors
    )


def test_certifier_rejects_cross_robot_interference_and_same_rail_passing() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    operations = list(schedule.operations)
    r0_by_kind = {
        operation.kind: operation
        for operation in operations
        if operation.robot_id == 0
    }
    for index, operation in enumerate(operations):
        if operation.robot_id == 1:
            reference = r0_by_kind[operation.kind]
            operations[index] = replace(
                operation, start=reference.start, end=reference.end
            )
    interference_report = certify_schedule(
        solution, replace(schedule, operations=tuple(operations)), CONFIG
    )
    assert any("interference:" in error for error in interference_report.errors)

    crossing_parents = (
        ParentWeld("a", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("b", (10.0, 8.0), (2.0, 8.0)),
    )
    crossing_solution = canonicalize(
        crossing_parents,
        tuple(
            SplitPattern(parent.parent_id, SplitKind.WHOLE)
            for parent in crossing_parents
        ),
        {0: ("a::whole",), 1: ("b::whole",)},
        CONFIG,
    )
    routes = build_robot_routes(
        crossing_solution, CONFIG, orientations={0: (0,), 1: (0,)}
    )
    crossing_operations = tuple(
        operation
        for route in routes
        for operation in build_operation_templates(route, CONFIG)
    )
    completion = tuple(
        max(
            (
                operation.end_time
                for operation in crossing_operations
                if operation.robot_id == robot
            ),
            default=0.0,
        )
        for robot in range(4)
    )
    fake = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        crossing_operations,
        max(completion),
        completion,
        directions=((0,), (0,), (), ()),
    )
    passing_report = certify_schedule(crossing_solution, fake, CONFIG)
    assert any("same-rail non-passing" in error for error in passing_report.errors)


def test_official_metrics_use_formal_definitions() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    metrics = official_metrics(solution, schedule, CONFIG)
    assert metrics.cmax == schedule.cmax
    assert metrics.optional_split_count == 0
    assert metrics.total_empty_travel == 0.0
    assert metrics.total_waiting == 0.0
    assert metrics.process_imbalance == pytest.approx(CONFIG.process_time(1.0))


def test_tiny_scheduler_oracle_agrees_on_feasibility_and_bounds_reference_cmax() -> None:
    solution = _solution()
    reference = reference_schedule(solution, CONFIG)
    oracle = tiny_scheduler_oracle(solution, CONFIG)
    assert reference.status is oracle.status is ScheduleStatus.FEASIBLE
    assert oracle.explored_dispatches > 0
    assert oracle.reference_status is reference.status
    assert oracle.reference_cmax == reference.cmax
    assert oracle.feasibility_agreement
    assert oracle.best_cmax <= reference.cmax
    assert oracle.best_cmax == pytest.approx(reference.cmax)
    assert oracle.scheduler_gap == pytest.approx(0.0)
    assert oracle.schedule is not None
    assert certify_schedule(solution, oracle.schedule, CONFIG).certified


def test_solution_oracle_supports_two_welding_blocks_per_robot() -> None:
    solution = _two_by_two_solution()
    oracle = tiny_scheduler_oracle(solution, CONFIG)
    assert oracle.status is ScheduleStatus.FEASIBLE
    assert oracle.reference_status is ScheduleStatus.FEASIBLE
    assert oracle.feasibility_agreement
    assert oracle.explored_dispatches > 1
    assert oracle.schedule is not None
    assert certify_schedule(solution, oracle.schedule, CONFIG).certified


def test_manual_oracle_case_a_reference_is_optimal() -> None:
    templates = {
        0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        2: _move_templates(2, ((0.0, 6.0), (1.0, 6.0), (2.0, 6.0))),
    }
    result = tiny_scheduler_oracle_from_templates(templates, ORACLE_CONFIG)
    assert result.status is result.reference_status is ScheduleStatus.FEASIBLE
    assert result.feasibility_agreement
    assert result.best_cmax == pytest.approx(2.0)
    assert result.scheduler_gap == pytest.approx(0.0)
    assert result.explored_dispatches > 1


def test_manual_oracle_case_b_detects_list_priority_cmax_loss() -> None:
    templates = {
        0: _move_templates(
            0,
            ((2.0, 6.0), (2.0, 1.0), (4.0, 6.0), (3.0, 4.0)),
        ),
        2: _move_templates(
            2,
            ((1.0, 4.0), (5.0, 4.0), (1.0, 1.0), (2.0, 4.0)),
        ),
    }
    result = tiny_scheduler_oracle_from_templates(templates, ORACLE_CONFIG)
    assert result.status is result.reference_status is ScheduleStatus.FEASIBLE
    assert result.feasibility_agreement
    assert result.scheduler_gap is not None and result.scheduler_gap > 0.4
    assert result.best_cmax < result.reference_cmax
    assert result.explored_dispatches > 1


def test_manual_oracle_case_c_deadlock_is_not_mathematical_infeasibility() -> None:
    templates = {
        0: _move_templates(
            0,
            ((4.0, 2.0), (4.0, 5.0), (1.0, 0.0), (6.0, 3.0)),
        ),
        2: _move_templates(
            2,
            ((0.0, 2.0), (0.0, 0.0), (3.0, 3.0), (0.0, 4.0)),
        ),
    }
    result = tiny_scheduler_oracle_from_templates(templates, ORACLE_CONFIG)
    assert result.status is ScheduleStatus.FEASIBLE
    assert result.reference_status is ScheduleStatus.DEADLOCK
    assert not result.feasibility_agreement
    assert result.best_cmax is not None
    assert result.reference_cmax is None
    assert result.explored_dispatches > 1


def test_manual_oracle_case_d_all_dispatches_are_infeasible() -> None:
    templates = {
        0: _move_templates(
            0,
            ((5.0, 0.0), (3.0, 4.0), (1.0, 1.0), (1.0, 0.0)),
        ),
        2: _move_templates(
            2,
            ((4.0, 1.0), (5.0, 0.0), (3.0, 0.0), (1.0, 0.0)),
        ),
    }
    result = tiny_scheduler_oracle_from_templates(templates, ORACLE_CONFIG)
    assert result.status is ScheduleStatus.INFEASIBLE
    assert result.reference_status is ScheduleStatus.DEADLOCK
    assert result.feasibility_agreement
    assert result.best_cmax is None
    assert result.reference_cmax is None
    assert result.scheduler_gap is None
    assert result.explored_dispatches > 1
