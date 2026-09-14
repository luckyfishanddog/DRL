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
from mrta_reference.oracle import tiny_scheduler_oracle
from mrta_reference.scheduler import reference_schedule
from mrta_reference.solution import canonicalize, official_metrics


CONFIG = ScientificConfig()


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


def test_pre_post_count_and_robot_eligibility_are_independently_rejected() -> None:
    solution = _solution()
    schedule = reference_schedule(solution, CONFIG)
    operations = tuple(op for op in schedule.operations if not (op.robot_id == 0 and op.kind is OperationKind.POST))
    report = certify_schedule(solution, replace(schedule, operations=operations), CONFIG)
    assert any("operation/route precedence" in error for error in report.errors)

    ineligible = canonicalize(solution.parents, solution.patterns, {2: ("left::whole",), 1: ("right::whole",)}, CONFIG)
    result = reference_schedule(ineligible, CONFIG)
    assert result.status is ScheduleStatus.INFEASIBLE
    assert "ineligible" in result.diagnostics[0]


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
