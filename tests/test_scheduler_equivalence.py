from __future__ import annotations

from dataclasses import replace
import math
import random

import pytest

import mrta_reference.scheduler as scheduler_module
from mrta_reference.geometry import generate_y_split_patterns, robot_is_eligible
from mrta_reference.model import Operation, OperationKind, ParentWeld, Route, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import (
    _remaining_processing_suffix,
    earliest_safe_start_optimized,
    earliest_safe_start_slow,
    reference_schedule_from_templates_optimized,
    reference_schedule_from_templates_slow,
    reference_schedule_optimized,
    reference_schedule_slow,
    relevant_fixed_operations,
)
from mrta_reference.solution import block_map, canonicalize
from mrta_search import SearchConfig, run_bounded_sa_oi


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _assert_schedule_equivalent(slow: ScheduleResult, optimized: ScheduleResult, config=FAST) -> None:
    assert slow.status is optimized.status
    assert slow.directions == optimized.directions
    assert slow.wait_for_graph == optimized.wait_for_graph
    assert slow.diagnostics == optimized.diagnostics
    if slow.cmax is None:
        assert optimized.cmax is None
    else:
        assert slow.cmax == pytest.approx(optimized.cmax)
    assert slow.robot_completion == pytest.approx(optimized.robot_completion)
    assert len(slow.operations) == len(optimized.operations)
    for left, right in zip(slow.operations, optimized.operations):
        assert left.robot_id == right.robot_id
        assert left.kind is right.kind
        assert left.start == pytest.approx(right.start)
        assert left.end == pytest.approx(right.end)
        assert left.start_time == pytest.approx(right.start_time)
        assert left.end_time == pytest.approx(right.end_time)
        assert left.sequence_index == right.sequence_index
        assert left.block_id == right.block_id
        assert left.operation_id == right.operation_id


def _move_templates(robot_id: int, points):
    operations = []
    cursor = 0.0
    for index, (start, end) in enumerate(zip(points, points[1:])):
        duration = math.dist(start, end)
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


def _oracle_cases():
    return (
        {
            0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            2: _move_templates(2, ((0.0, 6.0), (1.0, 6.0), (2.0, 6.0))),
        },
        {
            0: _move_templates(0, ((2.0, 6.0), (2.0, 1.0), (4.0, 6.0), (3.0, 4.0))),
            2: _move_templates(2, ((1.0, 4.0), (5.0, 4.0), (1.0, 1.0), (2.0, 4.0))),
        },
        {
            0: _move_templates(0, ((4.0, 2.0), (4.0, 5.0), (1.0, 0.0), (6.0, 3.0))),
            2: _move_templates(2, ((0.0, 2.0), (0.0, 0.0), (3.0, 3.0), (0.0, 4.0))),
        },
        {
            0: _move_templates(0, ((5.0, 0.0), (3.0, 4.0), (1.0, 1.0), (1.0, 0.0))),
            2: _move_templates(2, ((4.0, 1.0), (5.0, 0.0), (3.0, 0.0), (1.0, 0.0))),
        },
    )


def test_expired_filter_closed_boundary_and_epsilon_edges() -> None:
    ready = 10.0
    epsilon = FAST.numeric_epsilon

    def fixed(name, end):
        return Operation(name, 1, OperationKind.WAIT, end - 1.0, end, (1.0, 1.0), (1.0, 1.0), 0)

    rows = (
        fixed("equal", ready),
        fixed("minus-half-eps", ready - 0.5 * epsilon),
        fixed("plus-eps", ready + epsilon),
        fixed("expired", ready - 100.0 * epsilon),
        Operation("zero-at-ready", 1, OperationKind.WAIT, ready, ready, (1.0, 1.0), (1.0, 1.0), 0),
    )
    relevant = relevant_fixed_operations(rows, 0, ready, FAST)
    assert {item.operation_id for item in relevant} == {
        "equal",
        "minus-half-eps",
        "plus-eps",
        "zero-at-ready",
    }


def test_interval_sweep_matches_legacy_for_nested_touching_and_random_cases() -> None:
    rng = random.Random(20260928)
    for case in range(500):
        robot = rng.randrange(4)
        ready = rng.uniform(0.0, 10.0)
        duration = rng.choice((0.0, rng.uniform(0.01, 5.0)))
        start = (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
        end = start if duration == 0.0 else (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
        template = Operation(f"candidate-{case}", robot, OperationKind.MOVE, 0.0, duration, start, end, 0)
        fixed = []
        for index in range(rng.randrange(0, 12)):
            other = rng.randrange(4)
            begin = rng.uniform(0.0, 15.0)
            fixed_duration = rng.choice((0.0, rng.uniform(0.01, 5.0)))
            point_a = (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
            point_b = point_a if fixed_duration == 0.0 else (rng.uniform(0.0, 20.0), rng.uniform(0.0, 12.0))
            fixed.append(Operation(f"fixed-{case}-{index}", other, OperationKind.MOVE, begin, begin + fixed_duration, point_a, point_b, index))
        slow = earliest_safe_start_slow(template, ready, fixed, FAST)
        relevant = relevant_fixed_operations(fixed, robot, ready, FAST)
        optimized = earliest_safe_start_optimized(template, ready, relevant, FAST)
        if math.isfinite(slow[0]):
            assert optimized[0] == pytest.approx(slow[0])
        else:
            assert math.isinf(optimized[0])
        assert optimized[1:] == slow[1:]


def test_interval_sweep_explicit_nested_touching_identical_and_infinite(monkeypatch) -> None:
    template = Operation(
        "candidate", 0, OperationKind.MOVE, 0.0, 1.0, (0.0, 0.0), (1.0, 0.0), 0
    )
    fixed = tuple(
        Operation(
            name,
            1,
            OperationKind.WAIT,
            0.0,
            20.0,
            (10.0, 10.0),
            (10.0, 10.0),
            index,
        )
        for index, name in enumerate(("outer", "nested", "chain", "identical"))
    )
    intervals = {
        "outer": ((0.0, 2.0),),
        "nested": ((0.0, 1.0),),
        "chain": ((1.5, 4.0),),
        "identical": ((1.5, 4.0),),
    }

    def projected(*args):
        operation = args[4]
        return intervals[operation.operation_id]

    monkeypatch.setattr(scheduler_module, "forbidden_start_intervals", projected)
    slow = earliest_safe_start_slow(template, 0.0, fixed, FAST)
    optimized = earliest_safe_start_optimized(template, 0.0, fixed, FAST)
    assert slow == optimized
    assert slow[0] > 4.0
    assert slow[1] == (1,)
    assert slow[2] == ("chain", "identical", "nested", "outer")

    monkeypatch.setattr(
        scheduler_module,
        "forbidden_start_intervals",
        lambda *args: ((0.0, math.inf),),
    )
    slow_infinite = earliest_safe_start_slow(template, 0.0, fixed[:1], FAST)
    optimized_infinite = earliest_safe_start_optimized(
        template, 0.0, fixed[:1], FAST
    )
    assert slow_infinite == optimized_infinite
    assert math.isinf(slow_infinite[0])


def test_remaining_processing_suffix_excludes_move_and_matches_legacy_sum() -> None:
    templates = {robot: () for robot in range(4)}
    templates[0] = (
        Operation("setup", 0, OperationKind.SETUP, 0.0, 2.0, (0.0, 0.0), (0.0, 0.0), 0),
        Operation("move", 0, OperationKind.MOVE, 2.0, 7.0, (0.0, 0.0), (5.0, 0.0), 1),
        Operation("weld", 0, OperationKind.WELD, 7.0, 10.0, (5.0, 0.0), (8.0, 0.0), 2),
        Operation("post", 0, OperationKind.POST, 10.0, 14.0, (8.0, 0.0), (8.0, 0.0), 3),
    )
    suffix = _remaining_processing_suffix(templates)[0]
    assert suffix == pytest.approx((9.0, 7.0, 7.0, 4.0, 0.0))


@pytest.mark.parametrize("templates", _oracle_cases())
def test_phase2a_e1_e4_template_differential(templates) -> None:
    slow = reference_schedule_from_templates_slow(templates, FAST)
    optimized = reference_schedule_from_templates_optimized(templates, FAST)
    _assert_schedule_equivalent(slow, optimized)


def test_deadlock_infeasible_and_numeric_failure_differential(monkeypatch) -> None:
    parents = (
        ParentWeld("left-to-right", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("right-to-left", (10.0, 8.0), (2.0, 8.0)),
    )
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("left-to-right::whole",), 1: ("right-to-left::whole",)},
        FAST,
    )
    slow = reference_schedule_slow(solution, FAST, orientations={0: (0,), 1: (0,)})
    optimized = reference_schedule_optimized(solution, FAST, orientations={0: (0,), 1: (0,)})
    assert slow.status is ScheduleStatus.DEADLOCK
    _assert_schedule_equivalent(slow, optimized)
    assert slow.wait_for_graph
    assert any("blocked by" in item for item in slow.diagnostics)

    bad = {0: (Operation("bad", 1, OperationKind.WAIT, 0.0, 1.0, (0.0, 0.0), (0.0, 0.0), 0),)}
    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(bad, FAST),
        reference_schedule_from_templates_optimized(bad, FAST),
    )

    valid = {0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0)))}

    def explode(*args, **kwargs):
        raise OverflowError("forced numeric differential")

    monkeypatch.setattr(scheduler_module, "earliest_safe_start_slow", explode)
    slow_numeric = reference_schedule_from_templates_slow(valid, FAST)
    monkeypatch.setattr(scheduler_module, "earliest_safe_start_optimized", explode)
    optimized_numeric = reference_schedule_from_templates_optimized(valid, FAST)
    assert slow_numeric.status is ScheduleStatus.NUMERIC_FAILURE
    _assert_schedule_equivalent(slow_numeric, optimized_numeric)


def _random_solution(rng: random.Random, case: int):
    count = rng.randrange(0, 7)
    parents = []
    patterns = []
    routes = {robot: [] for robot in range(4)}
    for index in range(count):
        y = rng.choice((2.0, 6.0, 10.0))
        x = rng.uniform(0.5, 18.5)
        parent = ParentWeld(f"c{case}-p{index}", (x, y), (x + rng.uniform(0.2, 1.0), y))
        parents.append(parent)
        patterns.append(SplitPattern(parent.parent_id, SplitKind.WHOLE))
    provisional = canonicalize(parents, patterns, {robot: () for robot in range(4)} if not parents else {0: ()}, FAST) if not parents else None
    if parents:
        for parent in parents:
            block_id = f"{parent.parent_id}::whole"
            eligible = [robot for robot in range(4) if (robot < 2 and parent.start[1] >= FAST.by[0]) or (robot >= 2 and parent.start[1] <= FAST.by[1])]
            routes[rng.choice(eligible)].append(block_id)
        for route in routes.values():
            rng.shuffle(route)
        provisional = canonicalize(parents, patterns, routes, FAST)
    directions = {
        robot: tuple(rng.randrange(2) for _ in route.block_ids)
        for robot, route in enumerate(provisional.routes)
    }
    return provisional, directions


def test_hundreds_of_deterministic_random_solution_differentials() -> None:
    rng = random.Random(73013)
    for case in range(300):
        solution, directions = _random_solution(rng, case)
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        formal = scheduler_module.reference_schedule_formal(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert formal.canonical_json() == baseline.canonical_json()
            assert formal.expanded_states == 0


@pytest.mark.parametrize("index", range(4))
def test_formal_e1_e4_policy_and_independent_template_certificate(index):
    from mrta_reference.certifier import certify_template_schedule
    templates = _oracle_cases()[index]
    baseline = reference_schedule_from_templates_optimized(templates, FAST)
    formal = scheduler_module.reference_schedule_from_templates_formal(templates, FAST)
    if index < 2:
        assert formal.canonical_json() == baseline.canonical_json()
        assert formal.source == "BASELINE" and formal.expanded_states == 0
    elif index == 2:
        assert baseline.status is ScheduleStatus.DEADLOCK
        assert formal.feasible and formal.source == "BOUNDED_DEADLOCK_RECOVERY"
        assert formal.expanded_states == formal.state_budget == 16
        assert formal.cmax == pytest.approx(15.5287634621)
    else:
        assert formal.status is ScheduleStatus.DEADLOCK
        assert formal.cmax is None and formal.expanded_states == 7
        assert formal.frontier_exhausted
    if formal.feasible:
        assert certify_template_schedule(templates, formal, FAST).certified
        assert not certify_template_schedule(templates, replace(formal, cmax=formal.cmax + 1), FAST).certified
        assert not certify_template_schedule(templates, replace(formal, operations=formal.operations[1:]), FAST).certified


def test_bounded_exhaustion_stays_deadlock_and_numeric_failure_is_not_swallowed(monkeypatch):
    templates = _oracle_cases()[2]
    baseline = reference_schedule_from_templates_optimized(templates, FAST)
    exhausted = scheduler_module._bounded_dispatch_recovery(templates, FAST, baseline, state_budget=1)
    assert exhausted.status is ScheduleStatus.DEADLOCK
    assert exhausted.recovery_exhausted and exhausted.expanded_states == 1
    def explode(*args, **kwargs):
        raise ArithmeticError("forced recovery failure")
    monkeypatch.setattr(scheduler_module, "earliest_safe_start_optimized", explode)
    numeric = scheduler_module._bounded_dispatch_recovery(templates, FAST, baseline, state_budget=16)
    assert numeric.status is ScheduleStatus.NUMERIC_FAILURE
    assert any("forced recovery failure" in d for d in numeric.diagnostics)


def test_y_split_empty_route_wait_and_same_rail_boundary_differentials() -> None:
    cases = []
    mandatory = ParentWeld("mandatory", (2.0, 5.0), (2.0, 7.0))
    mandatory_pattern = generate_y_split_patterns(mandatory, FAST)[0]
    cases.append(
        canonicalize(
            (mandatory,),
            (mandatory_pattern,),
            {0: ("mandatory::1",), 2: ("mandatory::0",)},
            FAST,
        )
    )
    optional = ParentWeld("optional", (0.0, 6.0), (4.0, 6.0))
    optional_pattern = generate_y_split_patterns(optional, FAST)[0]
    cases.append(
        canonicalize(
            (optional,),
            (optional_pattern,),
            {0: ("optional::0",), 3: ("optional::1",)},
            FAST,
        )
    )
    for solution in cases:
        directions = {robot: tuple(0 for _ in route.block_ids) for robot, route in enumerate(solution.routes)}
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert scheduler_module.reference_schedule_formal(
                solution, FAST, orientations=directions).canonical_json() == baseline.canonical_json()

    touching = {
        0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
        1: _move_templates(1, ((2.5, 0.0), (1.5, 0.0), (3.0, 0.0))),
    }
    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(touching, FAST),
        reference_schedule_from_templates_optimized(touching, FAST),
    )


def test_phase2a_m1_m7_solution_differentials() -> None:
    rows = []

    def whole(parents, routes, directions):
        solution = canonicalize(
            parents,
            tuple(
                SplitPattern(parent.parent_id, SplitKind.WHOLE)
                for parent in parents
            ),
            routes,
            FAST,
        )
        rows.append((solution, directions))

    m1 = (ParentWeld("m1", (1.0, 2.0), (3.0, 2.0)),)
    whole(m1, {2: ("m1::whole",)}, {2: (0,)})

    assignment = (
        ParentWeld("left", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("right", (9.0, 2.0), (10.0, 2.0)),
    )
    whole(
        assignment,
        {2: ("left::whole",), 3: ("right::whole",)},
        {2: (0,), 3: (0,)},
    )
    whole(
        assignment,
        {2: ("left::whole", "right::whole")},
        {2: (0, 0)},
    )

    route = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (5.0, 2.0), (4.0, 2.0)),
        ParentWeld("c", (9.0, 2.0), (10.0, 2.0)),
    )
    whole(
        route,
        {2: ("a::whole", "b::whole", "c::whole")},
        {2: (0, 1, 0)},
    )
    whole(
        route,
        {2: ("a::whole", "c::whole", "b::whole")},
        {2: (0, 0, 1)},
    )
    whole(
        route[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 0)},
    )
    whole(
        route[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 1)},
    )

    optional = ParentWeld("optional-m5", (0.0, 6.0), (4.0, 6.0))
    whole(
        (optional,),
        {0: ("optional-m5::whole",)},
        {0: (0,)},
    )
    optional_pattern = generate_y_split_patterns(optional, FAST)[0]
    optional_solution = canonicalize(
        (optional,),
        (optional_pattern,),
        {0: ("optional-m5::0",), 1: ("optional-m5::1",)},
        FAST,
    )
    rows.append((optional_solution, {0: (0,), 1: (0,)}))

    mandatory = ParentWeld("mandatory-m7", (1.0, 5.0), (1.0, 7.0))
    mandatory_pattern = generate_y_split_patterns(mandatory, FAST)[0]
    mandatory_solution = canonicalize(
        (mandatory,),
        (mandatory_pattern,),
        {0: ("mandatory-m7::1",), 2: ("mandatory-m7::0",)},
        FAST,
    )
    rows.append((mandatory_solution, {0: (0,), 2: (0,)}))

    for solution, directions in rows:
        _assert_schedule_equivalent(
            reference_schedule_slow(solution, FAST, orientations=directions),
            reference_schedule_optimized(solution, FAST, orientations=directions),
        )
        baseline = reference_schedule_optimized(solution, FAST, orientations=directions)
        if baseline.feasible:
            assert scheduler_module.reference_schedule_formal(
                solution, FAST, orientations=directions).canonical_json() == baseline.canonical_json()

    _assert_schedule_equivalent(
        reference_schedule_from_templates_slow(_oracle_cases()[1], FAST),
        reference_schedule_from_templates_optimized(_oracle_cases()[1], FAST),
    )


def test_fixed_budget_search_trajectory_is_identical() -> None:
    parents = (
        ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),
        ParentWeld("lower-a", (1.0, 2.0), (3.0, 2.0)),
        ParentWeld("lower-b", (8.0, 2.0), (10.0, 2.0)),
    )
    search_config = SearchConfig(m=42, kdp=6, kref=2, max_iterations=12, time_limit=None)
    slow = run_bounded_sa_oi(parents, FAST, search_config, seed=123, reference_evaluator=reference_schedule_slow)
    optimized = run_bounded_sa_oi(parents, FAST, search_config, seed=123, reference_evaluator=reference_schedule_optimized)
    assert slow.status is optimized.status
    assert slow.best_solution.canonical_hash == optimized.best_solution.canonical_hash
    assert slow.best_directions == optimized.best_directions
    assert slow.best_schedule.cmax == pytest.approx(optimized.best_schedule.cmax)
    assert slow.stats.reference_status_sequence == optimized.stats.reference_status_sequence
    assert slow.stats.proposal_trajectory == optimized.stats.proposal_trajectory
    assert slow.stats.best_improvement_cmax == optimized.stats.best_improvement_cmax
