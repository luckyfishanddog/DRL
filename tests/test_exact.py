from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mrta_exact import (
    EXACT_Y_SCOPE_CURRENT_SEMANTICS,
    ExactSolveStatus,
    YOnlyPatternProvider,
    analytical_lower_bound,
    exact_schedule_from_templates,
    solve_exact_micro,
)
from mrta_reference.certifier import certify_schedule
from mrta_reference.model import (
    Operation,
    OperationKind,
    ParentWeld,
    Rail,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.oracle import tiny_scheduler_oracle_from_templates
from mrta_reference.scheduler import build_operation_templates, build_robot_routes
from mrta_reference.solution import canonicalize


FAST_CONFIG = ScientificConfig(
    weld_speed=1.0,
    empty_speed=1.0,
    t_pre=1.0,
    t_post=1.0,
)


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


def _manual_oracle_cases():
    return {
        "E1": {
            0: _move_templates(0, ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))),
            2: _move_templates(2, ((0.0, 6.0), (1.0, 6.0), (2.0, 6.0))),
        },
        "E2": {
            0: _move_templates(
                0, ((2.0, 6.0), (2.0, 1.0), (4.0, 6.0), (3.0, 4.0))
            ),
            2: _move_templates(
                2, ((1.0, 4.0), (5.0, 4.0), (1.0, 1.0), (2.0, 4.0))
            ),
        },
        "E3": {
            0: _move_templates(
                0, ((4.0, 2.0), (4.0, 5.0), (1.0, 0.0), (6.0, 3.0))
            ),
            2: _move_templates(
                2, ((0.0, 2.0), (0.0, 0.0), (3.0, 3.0), (0.0, 4.0))
            ),
        },
        "E4": {
            0: _move_templates(
                0, ((5.0, 0.0), (3.0, 4.0), (1.0, 1.0), (1.0, 0.0))
            ),
            2: _move_templates(
                2, ((4.0, 1.0), (5.0, 0.0), (3.0, 0.0), (1.0, 0.0))
            ),
        },
    }


def _parallel_solution(active_robots: int):
    specifications = (
        ("u-left", (1.0, 10.0), 0),
        ("u-right", (10.0, 10.0), 1),
        ("l-left", (1.0, 2.0), 2),
        ("l-right", (10.0, 2.0), 3),
    )[:active_robots]
    parents = tuple(
        ParentWeld(name, start, (start[0] + 1.0, start[1]))
        for name, start, _ in specifications
    )
    routes = {robot: () for robot in range(4)}
    for name, _, robot in specifications:
        routes[robot] = (f"{name}::whole",)
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        routes,
        FAST_CONFIG,
    )


def _fixed_whole_schedule(parents, routes, directions):
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        routes,
        FAST_CONFIG,
    )
    robot_routes = build_robot_routes(solution, FAST_CONFIG, directions)
    templates = {
        route.robot_id: build_operation_templates(route, FAST_CONFIG)
        for route in robot_routes
    }
    result = exact_schedule_from_templates(
        templates,
        FAST_CONFIG,
        directions=tuple(route.orientations for route in robot_routes),
        branch_and_bound=False,
    )
    assert result.schedule is not None
    assert certify_schedule(solution, result.schedule, FAST_CONFIG).certified
    return result


def test_lb0_manual_empty_and_parameter_sensitive_calculations() -> None:
    config = replace(FAST_CONFIG, t_pre=2.0, t_post=3.0)
    parents = (
        ParentWeld("long", (0.0, 2.0), (4.0, 2.0)),
        ParentWeld("short", (0.0, 3.0), (2.0, 3.0)),
    )
    bound = analytical_lower_bound(parents, config)
    assert bound.total_base_processing_work == pytest.approx(16.0)
    assert bound.lb_work == pytest.approx(4.0)
    assert bound.max_parent_parallel_lb == pytest.approx(7.0)
    assert bound.lb0 == pytest.approx(7.0)
    empty = analytical_lower_bound((), config)
    assert empty.total_base_processing_work == 0.0
    assert empty.lb_work == 0.0
    assert empty.max_parent_parallel_lb == 0.0
    assert empty.lb0 == 0.0
    faster = analytical_lower_bound(parents, replace(config, weld_speed=2.0))
    assert faster.lb0 == pytest.approx(6.0)


def test_y_only_pattern_provider_is_deterministic_and_excludes_x_split() -> None:
    provider = YOnlyPatternProvider()
    optional = ParentWeld("optional", (0.0, 6.0), (4.0, 6.0))
    first = provider.patterns_for(optional, FAST_CONFIG)
    assert first == provider.patterns_for(optional, FAST_CONFIG)
    assert first[0].kind is SplitKind.WHOLE
    assert {pattern.kind for pattern in first} == {
        SplitKind.WHOLE,
        SplitKind.Y_SPLIT,
    }
    assert len({pattern.pattern_id for pattern in first}) == len(first)

    mandatory = ParentWeld("mandatory", (1.0, 5.0), (1.0, 7.0))
    required = provider.patterns_for(mandatory, FAST_CONFIG)
    assert required
    assert all(pattern.kind is SplitKind.Y_SPLIT for pattern in required)
    assert all(pattern.mandatory for pattern in required)
    assert all(pattern.kind is not SplitKind.X_SPLIT for pattern in first + required)


@pytest.mark.parametrize("case_name", ("E1", "E2", "E3", "E4"))
def test_exact_scheduler_matches_independent_two_robot_oracle(case_name: str) -> None:
    templates = _manual_oracle_cases()[case_name]
    oracle = tiny_scheduler_oracle_from_templates(templates, FAST_CONFIG)
    exact = exact_schedule_from_templates(
        templates, FAST_CONFIG, branch_and_bound=False
    )
    assert exact.status is oracle.status
    assert exact.dispatch_states > 1
    if oracle.status is ScheduleStatus.FEASIBLE:
        assert exact.schedule is not None
        assert exact.cmax == pytest.approx(oracle.best_cmax)
    else:
        assert exact.schedule is None
    if case_name == "E1":
        assert oracle.scheduler_gap == pytest.approx(0.0)
    elif case_name == "E2":
        assert oracle.scheduler_gap is not None and oracle.scheduler_gap > 0.4
        assert exact.cmax < oracle.reference_cmax
    elif case_name == "E3":
        assert exact.status is ScheduleStatus.FEASIBLE
        assert oracle.reference_status is ScheduleStatus.DEADLOCK
    else:
        assert exact.status is ScheduleStatus.INFEASIBLE


@pytest.mark.parametrize("active_robots", (1, 3, 4))
def test_one_three_and_four_robot_exact_schedules_pass_independent_certifier(
    active_robots: int,
) -> None:
    solution = _parallel_solution(active_robots)
    robot_routes = build_robot_routes(
        solution,
        FAST_CONFIG,
        {robot: (0,) if robot < active_robots else () for robot in range(4)},
    )
    templates = {
        route.robot_id: build_operation_templates(route, FAST_CONFIG)
        for route in robot_routes
    }
    directions = tuple(route.orientations for route in robot_routes)
    exact = exact_schedule_from_templates(
        templates, FAST_CONFIG, directions=directions
    )
    assert exact.status is ScheduleStatus.FEASIBLE
    assert exact.schedule is not None
    report = certify_schedule(solution, exact.schedule, FAST_CONFIG)
    assert report.certified, report.errors
    assert exact.cmax == pytest.approx(FAST_CONFIG.process_time(1.0))


def test_empty_exact_scheduler_is_feasible_under_current_idle_semantics() -> None:
    result = exact_schedule_from_templates({}, FAST_CONFIG)
    assert result.status is ScheduleStatus.FEASIBLE
    assert result.cmax == 0.0
    assert result.schedule is not None
    assert result.schedule.operations == ()


def test_exact_scheduler_rejects_illegal_same_rail_initial_order() -> None:
    templates = {
        0: _move_templates(0, ((8.0, 10.0), (9.0, 10.0))),
        1: _move_templates(1, ((2.0, 10.0), (3.0, 10.0))),
    }
    result = exact_schedule_from_templates(templates, FAST_CONFIG)
    assert result.status is ScheduleStatus.INFEASIBLE
    assert "same-rail order" in result.diagnostics[0]


def test_micro_solver_m1_is_hand_calculable_certified_and_deterministic() -> None:
    parents = (ParentWeld("p", (1.0, 2.0), (3.0, 2.0)),)
    first = solve_exact_micro(parents, FAST_CONFIG, exhaustive_mode=True)
    second = solve_exact_micro(parents, FAST_CONFIG, exhaustive_mode=True)
    assert first.scope_id == EXACT_Y_SCOPE_CURRENT_SEMANTICS
    assert first.status is ExactSolveStatus.OPTIMAL
    assert first.optimal and first.certified
    assert first.best_cmax == pytest.approx(4.0)
    assert first.lb0 == pytest.approx(3.0)
    assert first.lb0 <= first.best_cmax
    assert first.best_solution is not None and first.best_schedule is not None
    assert first.best_solution.routes[0].block_ids == ()
    assert first.best_solution.routes[1].block_ids == ()
    assert certify_schedule(
        first.best_solution, first.best_schedule, FAST_CONFIG
    ).certified
    assert first.best_solution.canonical_hash == second.best_solution.canonical_hash
    assert first.best_schedule.canonical_json() == second.best_schedule.canonical_json()


def test_optional_y_split_is_beneficial_and_canonical_duplicates_are_removed() -> None:
    parents = (ParentWeld("p", (0.0, 6.0), (4.0, 6.0)),)
    result = solve_exact_micro(parents, FAST_CONFIG, exhaustive_mode=True)
    assert result.status is ExactSolveStatus.OPTIMAL
    assert result.best_solution is not None
    assert result.best_solution.patterns[0].kind is SplitKind.Y_SPLIT
    assert result.best_cmax < FAST_CONFIG.process_time(4.0)
    assert result.pattern_combinations == 2
    assert result.canonical_duplicates > 0
    assert result.raw_discrete_count == result.route_states
    assert result.lb0 <= result.best_cmax + 1.0e-9


def test_full_enumeration_covers_assignment_routes_and_directions() -> None:
    parents = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (5.0, 2.0), (4.0, 2.0)),
        ParentWeld("c", (9.0, 2.0), (10.0, 2.0)),
    )
    result = solve_exact_micro(parents, FAST_CONFIG)
    assert result.status is ExactSolveStatus.OPTIMAL
    assert result.assignment_states == 8
    assert result.route_states > result.assignment_states
    assert result.direction_states > result.canonical_unique_solutions
    assert result.schedule_evaluations > 0
    assert result.best_schedule is not None
    assert any(
        direction == 1
        for vector in result.best_schedule.directions
        for direction in vector
    )
    assert result.lb0 <= result.best_cmax + 1.0e-9


def test_m2_assignment_m3_route_order_and_m4_direction_are_material() -> None:
    assignment_parents = (
        ParentWeld("left", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("right", (9.0, 2.0), (10.0, 2.0)),
    )
    split_assignment = _fixed_whole_schedule(
        assignment_parents,
        {2: ("left::whole",), 3: ("right::whole",)},
        {2: (0,), 3: (0,)},
    )
    same_robot = _fixed_whole_schedule(
        assignment_parents,
        {2: ("left::whole", "right::whole")},
        {2: (0, 0)},
    )
    assert split_assignment.cmax == pytest.approx(3.0)
    assert split_assignment.cmax < same_robot.cmax

    route_parents = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (5.0, 2.0), (4.0, 2.0)),
        ParentWeld("c", (9.0, 2.0), (10.0, 2.0)),
    )
    short_route = _fixed_whole_schedule(
        route_parents,
        {2: ("a::whole", "b::whole", "c::whole")},
        {2: (0, 1, 0)},
    )
    long_route = _fixed_whole_schedule(
        route_parents,
        {2: ("a::whole", "c::whole", "b::whole")},
        {2: (0, 0, 1)},
    )
    assert short_route.cmax == pytest.approx(15.0)
    assert long_route.cmax == pytest.approx(22.0)

    forward = _fixed_whole_schedule(
        route_parents[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 0)},
    )
    reversed_second = _fixed_whole_schedule(
        route_parents[:2],
        {2: ("a::whole", "b::whole")},
        {2: (0, 1)},
    )
    assert forward.cmax == pytest.approx(9.0)
    assert reversed_second.cmax == pytest.approx(8.0)


@pytest.mark.parametrize("count", (2, 3))
def test_brute_force_and_branch_and_bound_have_identical_optimum(count: int) -> None:
    parents = tuple(
        ParentWeld(f"p{index}", (1.0 + 3.0 * index, 2.0), (2.0 + 3.0 * index, 2.0))
        for index in range(count)
    )
    brute = solve_exact_micro(parents, FAST_CONFIG, exhaustive_mode=True)
    bounded = solve_exact_micro(parents, FAST_CONFIG, exhaustive_mode=False)
    assert brute.status is bounded.status is ExactSolveStatus.OPTIMAL
    assert brute.best_cmax == pytest.approx(bounded.best_cmax)
    assert brute.lb0 <= brute.best_cmax + 1.0e-9
    assert bounded.pruned_states > 0


def test_mandatory_y_solution_is_legal_and_certified() -> None:
    parents = (ParentWeld("cross", (1.0, 5.0), (1.0, 7.0)),)
    result = solve_exact_micro(parents, FAST_CONFIG)
    assert result.status is ExactSolveStatus.OPTIMAL
    assert result.best_solution is not None and result.best_schedule is not None
    assert result.best_solution.patterns[0].kind is SplitKind.Y_SPLIT
    assert result.best_solution.patterns[0].mandatory
    assert certify_schedule(
        result.best_solution, result.best_schedule, FAST_CONFIG
    ).certified


def test_x_split_provider_is_rejected_by_current_scope() -> None:
    class BadProvider:
        scope_id = EXACT_Y_SCOPE_CURRENT_SEMANTICS

        def patterns_for(self, parent, config):
            return (SplitPattern(parent.parent_id, SplitKind.X_SPLIT, 0.5, "x", rail=Rail.UPPER),)

    parent = ParentWeld("p", (0.0, 2.0), (2.0, 2.0))
    with pytest.raises(ValueError, match="X_SPLIT is excluded"):
        solve_exact_micro((parent,), FAST_CONFIG, pattern_provider=BadProvider())


def test_no_legal_pattern_reports_no_feasible_solution() -> None:
    class EmptyProvider:
        scope_id = EXACT_Y_SCOPE_CURRENT_SEMANTICS

        def patterns_for(self, parent, config):
            return ()

    parent = ParentWeld("p", (0.0, 2.0), (2.0, 2.0))
    result = solve_exact_micro(
        (parent,), FAST_CONFIG, pattern_provider=EmptyProvider()
    )
    assert result.status is ExactSolveStatus.NO_FEASIBLE_SOLUTION
    assert not result.optimal
    assert result.best_cmax is None


def test_limit_reached_is_never_reported_optimal() -> None:
    parents = (ParentWeld("p", (1.0, 2.0), (3.0, 2.0)),)
    result = solve_exact_micro(
        parents, FAST_CONFIG, max_schedule_evaluations=0
    )
    assert result.status is ExactSolveStatus.LIMIT_REACHED
    assert not result.optimal
    assert result.best_cmax is None
    assert "max_schedule_evaluations reached" in result.diagnostics


def test_micro_size_guard_requires_explicit_override() -> None:
    parents = tuple(
        ParentWeld(f"p{index}", (index * 2.0, 2.0), (index * 2.0 + 1.0, 2.0))
        for index in range(5)
    )
    with pytest.raises(ValueError, match="micro exact guard"):
        solve_exact_micro(parents, FAST_CONFIG)
