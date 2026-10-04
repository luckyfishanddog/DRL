from __future__ import annotations

import math
from dataclasses import replace

import pytest

from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import (
    continuous_interference,
    finite_x_split_validator,
    generate_x_split_patterns,
    operations_conflict,
    same_rail_order_violation,
)
from mrta_reference.model import (
    Operation,
    OperationKind,
    ParentWeld,
    Rail,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    EXPERIMENTAL_X_SPLIT_SCOPE_V1,
    FORMAL_SCOPE_V1_1,
)
from mrta_reference.scheduler import (
    build_operation_templates,
    build_robot_routes,
    earliest_safe_start,
    reference_schedule,
    reference_schedule_formal,
    reference_schedule_from_templates,
    wait_for_cycles,
)
from mrta_reference.solution import canonicalize


CONFIG = ScientificConfig()


def test_formal_task_horizon_empty_initial_post_and_same_rail_release():
    from mrta_reference.model import FORMAL_SCOPE_V1_1
    from mrta_reference.scheduler import reference_schedule_formal, reference_schedule_slow
    cfg = ScientificConfig(weld_speed=1, empty_speed=1, t_pre=1, t_post=1)
    parents = (ParentWeld("left", (2, 10), (3, 10)),
               ParentWeld("right", (10, 10), (1, 10)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ("left::whole",), 1: ("right::whole",)}, cfg)
    schedule = reference_schedule_formal(solution, cfg, orientations={0: (0,), 1: (0,)})
    assert schedule.feasible
    assert schedule.canonical_json() == reference_schedule_slow(
        solution, cfg, orientations={0: (0,), 1: (0,)}).canonical_json()
    assert schedule.robot_completion == (3, 11, 0, 0)
    assert not any(op.robot_id >= 2 for op in schedule.operations)
    assert certify_schedule(solution, schedule, cfg, scope=FORMAL_SCOPE_V1_1).certified
    assert all(min(op.start_time for op in schedule.operations if op.robot_id == r) == 0 for r in (0, 1))
    assert all(max((op for op in schedule.operations if op.robot_id == r),
                   key=lambda op: op.end_time).kind is OperationKind.POST for r in (0, 1))
    # Completed R0 would obstruct R1 under terminal occupancy. Formal release
    # deliberately removes that future occupancy, including the rail constraint.
    parked = Operation("fake-park", 0, OperationKind.WAIT, 3, 11, (3, 10), (3, 10), 3)
    assert any(operations_conflict(parked, op, cfg) for op in schedule.operations if op.robot_id == 1)
    bad = replace(schedule, operations=schedule.operations + (parked,), robot_completion=(11, 11, 0, 0))
    assert not certify_schedule(solution, bad, cfg, scope=FORMAL_SCOPE_V1_1).certified
    post = next(op for op in schedule.operations if op.robot_id == 0 and op.kind is OperationKind.POST)
    touching = Operation("touch", 1, OperationKind.WAIT, 3, 3, (3, 10), (3, 10), 0)
    assert operations_conflict(post, touching, cfg)  # closed final POST endpoint
    empty = canonicalize((), (), {}, cfg)
    result = reference_schedule_formal(empty, cfg)
    assert result.operations == () and result.cmax == 0
    assert certify_schedule(empty, result, cfg, scope=FORMAL_SCOPE_V1_1).certified
    wrong_order = canonicalize(parents, solution.patterns,
                              {1: ("left::whole",), 0: ("right::whole",)}, cfg)
    assert reference_schedule_formal(wrong_order, cfg, orientations={0: (0,), 1: (0,)}).status is ScheduleStatus.INFEASIBLE
    single = canonicalize(parents[:1], solution.patterns[:1], {1: ("left::whole",)}, cfg)
    result = reference_schedule_formal(single, cfg)
    assert result.feasible and result.robot_completion[0] == 0
    assert certify_schedule(single, result, cfg, scope=FORMAL_SCOPE_V1_1).certified


def test_formal_x_excluded_even_with_valid_legacy_validator():
    from mrta_reference.model import CanonicalSolution, Route, FORMAL_SCOPE_V1_1
    from mrta_reference.scheduler import reference_schedule_formal
    parent = ParentWeld("x", (1, 8), (5, 8))
    pattern = SplitPattern("x", SplitKind.X_SPLIT, 0.5, "explicit-x", rail=Rail.UPPER)
    solution = CanonicalSolution((parent,), (pattern,), (Route(0, ("x::0",)), Route(1, ("x::1",)), Route(2, ()), Route(3, ())))
    formal = reference_schedule_formal(solution, CONFIG)
    assert formal.status is ScheduleStatus.INFEASIBLE
    assert "EXCLUDED" in formal.diagnostics[0]
    fake = replace(formal, status=ScheduleStatus.FEASIBLE, cmax=0)
    assert not certify_schedule(solution, fake, CONFIG, scope=FORMAL_SCOPE_V1_1,
                                x_split_validator=lambda *args: True).certified


def test_experimental_x_scope_certifies_processing_and_shared_point_wait() -> None:
    parent = ParentWeld("xgate", (0.0, 8.0), (4.0, 8.0))
    pattern = next(
        item
        for item in generate_x_split_patterns(parent, 2.0, CONFIG, rail=Rail.UPPER)
        if item.point_id == "BX_CENTER"
    )
    validator = finite_x_split_validator((parent,), CONFIG)
    solution = canonicalize(
        (parent,),
        (pattern,),
        {0: ("xgate::0",), 1: ("xgate::1",)},
        CONFIG,
        x_split_validator=validator,
    )
    schedule = reference_schedule_formal(
        solution,
        CONFIG,
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
        orientations={0: (0,), 1: (1,), 2: (), 3: ()},
    )
    assert schedule.status is ScheduleStatus.FEASIBLE
    report = certify_schedule(
        solution,
        schedule,
        CONFIG,
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
    )
    assert report.certified, report.errors
    assert sum(op.kind is OperationKind.SETUP for op in schedule.operations) == 2
    assert sum(op.kind is OperationKind.POST for op in schedule.operations) == 2
    split_process = sum(
        op.duration
        for op in schedule.operations
        if op.kind in (OperationKind.SETUP, OperationKind.WELD, OperationKind.POST)
    )
    assert split_process == pytest.approx(CONFIG.process_time(parent.length) + CONFIG.t_pre + CONFIG.t_post)
    assert any(op.kind is OperationKind.WAIT and op.duration > 0.0 for op in schedule.operations)
    rejected = reference_schedule_formal(solution, CONFIG, scope=FORMAL_SCOPE_V1_1)
    assert rejected.status is ScheduleStatus.INFEASIBLE
    assert "EXCLUDED" in rejected.diagnostics[0]


def test_phase3x_controlled_fixtures_f1_to_f8() -> None:
    def whole_and_x(length: float):
        parent = ParentWeld(f"p{length}", (0.0, 8.0), (length, 8.0))
        whole = canonicalize(
            (parent,),
            (SplitPattern(parent.parent_id, SplitKind.WHOLE),),
            {0: (f"{parent.parent_id}::whole",)},
            CONFIG,
        )
        patterns = generate_x_split_patterns(
            parent, length / 2.0, CONFIG, rail=Rail.UPPER
        )
        center = next((item for item in patterns if item.point_id == "BX_CENTER"), None)
        if center is None:
            return parent, whole, None
        split = canonicalize(
            (parent,),
            (center,),
            {0: (f"{parent.parent_id}::0",), 1: (f"{parent.parent_id}::1",)},
            CONFIG,
            x_split_validator=finite_x_split_validator((parent,), CONFIG),
        )
        return parent, whole, split

    # F1: long X-span remains legally WHOLE; X_SPLIT is optional, never forced.
    long_parent, long_whole, long_split = whole_and_x(12.0)
    assert long_whole.patterns[0].kind is SplitKind.WHOLE
    assert long_split is not None

    # F2/F3: a sub-10m parent can benefit from finite X load sharing.
    short_parent, short_whole, short_split = whole_and_x(8.0)
    assert short_parent.length < 10.0 and short_split is not None
    whole_schedule = reference_schedule_formal(
        short_whole, CONFIG, scope=FORMAL_SCOPE_V1_1
    )
    outward = reference_schedule_formal(
        short_split,
        CONFIG,
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
        orientations={0: (0,), 1: (0,), 2: (), 3: ()},
    )
    assert outward.feasible and outward.cmax < whole_schedule.cmax

    # F4: minimum-size children plus coordination can make X no better.
    _, tiny_whole, tiny_split = whole_and_x(0.4)
    assert tiny_split is not None
    tiny_x = reference_schedule_formal(
        tiny_split,
        CONFIG,
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
        orientations={0: (0,), 1: (0,), 2: (), 3: ()},
    )
    assert reference_schedule_formal(tiny_whole, CONFIG, scope=FORMAL_SCOPE_V1_1).feasible
    assert not tiny_x.feasible

    # F5/F6: the shared point is checked; direction changes WAIT without changing legality.
    _, _, medium_split = whole_and_x(4.0)
    assert medium_split is not None
    inward = reference_schedule_formal(
        medium_split,
        CONFIG,
        scope=EXPERIMENTAL_X_SPLIT_SCOPE_V1,
        orientations={0: (0,), 1: (1,), 2: (), 3: ()},
    )
    assert any(op.kind is OperationKind.WAIT for op in inward.operations)
    independent = canonicalize(
        (
            ParentWeld("ind-left", (0.0, 8.0), (2.0, 8.0)),
            ParentWeld("ind-right", (2.0, 8.0), (4.0, 8.0)),
        ),
        (
            SplitPattern("ind-left", SplitKind.WHOLE),
            SplitPattern("ind-right", SplitKind.WHOLE),
        ),
        {0: ("ind-left::whole",), 1: ("ind-right::whole",)},
        CONFIG,
    )
    aligned_without_x = reference_schedule_formal(
        independent,
        CONFIG,
        scope=FORMAL_SCOPE_V1_1,
        orientations={0: (0,), 1: (0,), 2: (), 3: ()},
    )
    assert not any(pattern.kind is SplitKind.X_SPLIT for pattern in independent.patterns)
    assert aligned_without_x.feasible
    assert not any(op.kind is OperationKind.WAIT for op in aligned_without_x.operations)

    # F7/F8: negligible X-span and sub-Lmin children generate no candidates.
    vertical = ParentWeld("vertical-fixture", (2.0, 7.0), (2.0, 9.0))
    too_short = ParentWeld("short-fixture", (0.0, 8.0), (0.3, 8.0))
    assert generate_x_split_patterns(vertical, 2.0, CONFIG, rail=Rail.UPPER) == ()
    assert generate_x_split_patterns(too_short, 0.15, CONFIG, rail=Rail.UPPER) == ()


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


def test_reference_scheduler_inserts_explicit_wait_for_delayed_operation() -> None:
    manual_config = replace(CONFIG, t_pre=5.0, empty_speed=1.0)
    templates = {
        0: (
            _operation("R0:hold", 0, OperationKind.SETUP, (0.0, 5.0), (0.0, 0.0), (0.0, 0.0)),
        ),
        2: (
            _operation("R2:move", 2, OperationKind.MOVE, (0.0, 2.0), (2.0, 0.0), (0.0, 0.0)),
        ),
    }
    result = reference_schedule_from_templates(templates, manual_config)
    assert result.status is ScheduleStatus.FEASIBLE
    waits = [operation for operation in result.operations if operation.kind is OperationKind.WAIT]
    assert len(waits) == 1
    assert waits[0].robot_id == 2
    assert waits[0].start == waits[0].end == (2.0, 0.0)
    moved = next(operation for operation in result.operations if operation.operation_id == "R2:move")
    assert waits[0].end_time == moved.start_time


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


def _deadlock_solution():
    parents = (
        ParentWeld("left-to-right", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("right-to-left", (10.0, 8.0), (2.0, 8.0)),
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("left-to-right::whole",), 1: ("right-to-left::whole",)},
        CONFIG,
    )


def test_constructed_route_deadlock_reports_blockers_without_hidden_repair() -> None:
    solution = _deadlock_solution()
    result = reference_schedule(solution, CONFIG, orientations={0: (0,), 1: (0,)})
    assert result.status is ScheduleStatus.DEADLOCK
    assert result.wait_for_graph
    assert any("blocked by" in item for item in result.diagnostics)


def test_deadlock_repair_hook_receives_full_context_and_none_preserves_result() -> None:
    solution = _deadlock_solution()
    captured = {}

    def no_repair(canonical, config, deadlock, graph, diagnostics):
        captured.update(
            canonical=canonical,
            config=config,
            deadlock=deadlock,
            graph=graph,
            diagnostics=diagnostics,
        )
        return None

    result = reference_schedule(
        solution,
        CONFIG,
        orientations={0: (0,), 1: (0,)},
        deadlock_repair=no_repair,
    )
    assert result.status is ScheduleStatus.DEADLOCK
    assert captured["canonical"].canonical_hash == solution.canonical_hash
    assert captured["config"] == CONFIG
    assert captured["deadlock"] == result
    assert captured["graph"] == result.wait_for_graph
    assert captured["diagnostics"] == result.diagnostics


def test_caller_supplied_deadlock_repair_can_return_independently_certified_result() -> None:
    solution = _deadlock_solution()

    def caller_policy(canonical, config, deadlock, graph, diagnostics):
        assert deadlock.status is ScheduleStatus.DEADLOCK
        assert graph == deadlock.wait_for_graph
        assert diagnostics == deadlock.diagnostics
        return reference_schedule(
            canonical,
            config,
            orientations={0: (0,), 1: (1,)},
        )

    repaired = reference_schedule(
        solution,
        CONFIG,
        orientations={0: (0,), 1: (0,)},
        deadlock_repair=caller_policy,
    )
    assert repaired.status is ScheduleStatus.FEASIBLE
    assert certify_schedule(solution, repaired, CONFIG).certified
