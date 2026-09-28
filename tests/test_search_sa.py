from __future__ import annotations

import random

import pytest

from mrta_exact import exact_schedule_from_templates, solve_exact_micro
from mrta_reference.model import CandidateKey, CandidateMove, MoveType, OperationKind, ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import build_operation_templates, build_robot_routes, reference_schedule
from mrta_reference.solution import canonicalize
from mrta_search.direction import ConstrainedDirectionResult, DirectionStatus, optimize_directions_with_initial_feasibility
from mrta_search.neighborhood import ScreenedCandidate
from mrta_search.pipeline import DirectionEvaluatedCandidate, SearchConfig, SearchStatus, evaluate_iteration, micro_gap_decomposition, rerank_c3, run_bounded_sa_oi, sa_accept
from mrta_search.stats import SearchStats


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _base():
    parents = tuple(
        ParentWeld(name, (x, 2.0), (x + 1.0, 2.0))
        for name, x in (("a", 1.0), ("b", 4.0), ("c", 10.0), ("d", 14.0))
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {2: ("a::whole", "b::whole"), 3: ("c::whole", "d::whole")},
        FAST,
    )


def test_c3_exact_direction_rerank_changes_kref_shortlist() -> None:
    solution = _base()
    rows = []
    for rank, exact_empty in enumerate((9.0, 8.0, 1.0)):
        key = CandidateKey(0, MoveType.INTRA_RELOCATE, ("a",), 2, 2, (0,), (rank,))
        screened = ScreenedCandidate(
            CandidateMove(key),
            solution,
            (10.0, float(rank), 0, (rank,)),
            10.0,
            float(rank),
            0,
        )
        direction = ConstrainedDirectionResult(
            DirectionStatus.FEASIBLE,
            ((0,), (), (), ()),
            exact_empty,
        )
        rows.append(DirectionEvaluatedCandidate(screened, rank, direction))
    shortlist = rerank_c3(rows, 2)
    assert [item.cheap_rank for item in shortlist] == [2, 1]


def test_pipeline_hard_kdp_kref_explicit_directions_and_four_statuses() -> None:
    solution = _base()
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    search_config = SearchConfig(m=32, kdp=4, kref=1, max_iterations=1)
    for status in ScheduleStatus:
        stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 3)
        supplied = []

        def evaluator(candidate, config, *, orientations):
            supplied.append(tuple(tuple(orientations[robot]) for robot in range(4)))
            if status is ScheduleStatus.FEASIBLE:
                return reference_schedule(candidate, config, orientations=orientations)
            return ScheduleResult(status, diagnostics=(status.value,))

        result = evaluate_iteration(
            solution,
            directions,
            FAST,
            search_config,
            stats,
            seed=3,
            reference_evaluator=evaluator,
        )
        assert len(result.c3_candidates) <= 4
        assert len(supplied) <= 1
        assert stats.per_iteration_kdp[-1] <= 4
        assert stats.per_iteration_nref[-1] <= 1
        if supplied:
            assert result.c4_candidates[0].schedule.directions == supplied[0] or status is not ScheduleStatus.FEASIBLE
            assert (
                stats.n_feasible,
                stats.n_deadlock,
                stats.n_infeasible,
                stats.n_numeric_failure,
            )[list(ScheduleStatus).index(status)] == 1
        stats.assert_invariants(kdp=4, kref=1)


def test_sa_acceptance_improve_equal_worse_seeded_and_zero_temperature() -> None:
    assert sa_accept(10.0, 9.0, temperature_value=1.0, cscale=1.0, rng=random.Random(1))[0]
    assert sa_accept(10.0, 10.0 + 1e-10, temperature_value=1.0, cscale=1.0, rng=random.Random(1))[0]
    first = sa_accept(10.0, 11.0, temperature_value=1.0, cscale=10.0, rng=random.Random(42))
    second = sa_accept(10.0, 11.0, temperature_value=1.0, cscale=10.0, rng=random.Random(42))
    assert first == second
    assert 0.0 < first[1] < 1.0
    assert not sa_accept(10.0, 11.0, temperature_value=0.0, cscale=10.0, rng=random.Random(1))[0]


def test_optional_y_activation_enters_search_and_improves_whole_only() -> None:
    parents = (ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),)
    result = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=10),
        seed=0,
    )
    assert result.status is SearchStatus.COMPLETED
    assert result.initialization.schedule.cmax == pytest.approx(6.0)
    assert result.best_schedule.cmax == pytest.approx(4.0)
    assert result.best_solution.patterns[0].kind is SplitKind.Y_SPLIT
    assert result.stats.accepted_by_move[MoveType.SPLIT_ACTIVATE.value] >= 1
    assert result.final_certification.certified


@pytest.mark.parametrize("count", (1, 2, 3, 4))
def test_micro_exact_reference_gap_decomposition_and_certified_best(count: int) -> None:
    specs = (
        ("u0", (1.0, 10.0)),
        ("u1", (10.0, 10.0)),
        ("l0", (1.0, 2.0)),
        ("l1", (10.0, 2.0)),
    )[:count]
    parents = tuple(ParentWeld(name, start, (start[0] + 1.0, start[1])) for name, start in specs)
    exact = solve_exact_micro(parents, FAST)
    search = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=28, kdp=4, kref=1, max_iterations=2),
        seed=5,
    )
    gap = micro_gap_decomposition(search, exact, FAST)
    assert gap.total_gap == pytest.approx(gap.search_gap + gap.scheduler_gap)
    assert gap.search_gap >= -1e-9
    assert gap.c_coord <= gap.c_ref + 1e-9
    assert search.final_certification.certified


def test_deadlock_and_numeric_failure_are_never_sa_proposals() -> None:
    solution = _base()
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    for status in (ScheduleStatus.DEADLOCK, ScheduleStatus.NUMERIC_FAILURE, ScheduleStatus.INFEASIBLE):
        stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 9)

        def evaluator(*args, **kwargs):
            return ScheduleResult(status)

        iteration = evaluate_iteration(
            solution,
            directions,
            FAST,
            SearchConfig(m=28, kdp=4, kref=1, max_iterations=1),
            stats,
            seed=9,
            reference_evaluator=evaluator,
        )
        assert iteration.proposal is None


def _nontrivial_quality_cases():
    return {
        "Q1_assignment": (
            (
                ParentWeld("p0", (11.14, 10.0), (12.54, 10.0)),
                ParentWeld("p1", (1.73, 10.0), (3.44, 10.0)),
                ParentWeld("p2", (3.70, 10.0), (4.88, 10.0)),
            ),
            7,
            1,
            MoveType.INTER_RELOCATE,
        ),
        "Q2_route_order": (
            (
                ParentWeld("p0", (7.133, 8.048), (10.143, 11.5)),
                ParentWeld("p1", (10.03, 8.264), (10.365, 8.484)),
                ParentWeld("p2", (13.385, 9.169), (12.862, 9.463)),
                ParentWeld("p3", (12.928, 8.209), (13.724, 8.133)),
            ),
            43,
            8,
            MoveType.TWO_OPT,
        ),
        "Q3_direction": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
            ),
            3,
            20,
            None,
        ),
        "Q4_optional_y": (
            (ParentWeld("optional", (0.0, 6.0), (4.0, 6.0)),),
            0,
            10,
            MoveType.SPLIT_ACTIVATE,
        ),
        "Q5_interference_wait": (
            (
                ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
                ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
                ParentWeld("d", (2.31, 10.0), (2.59, 10.0)),
            ),
            37,
            12,
            MoveType.INTER_RELOCATE,
        ),
    }


def _fixed_coordination_cmax(solution, directions) -> float:
    routes = build_robot_routes(
        solution, FAST, {robot: directions[robot] for robot in range(4)}
    )
    exact = exact_schedule_from_templates(
        {
            route.robot_id: build_operation_templates(route, FAST)
            for route in routes
        },
        FAST,
        directions=directions,
    )
    assert exact.schedule is not None
    assert exact.schedule.cmax is not None
    return exact.schedule.cmax


@pytest.mark.parametrize(
    "case_name", tuple(_nontrivial_quality_cases())
)
def test_q1_q5_nontrivial_initial_gap_and_backbone_quality(case_name: str) -> None:
    parents, seed, iterations, expected_move = _nontrivial_quality_cases()[case_name]
    exact = solve_exact_micro(parents, FAST)
    search = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(m=64, kdp=8, kref=2, max_iterations=iterations),
        seed=seed,
    )
    assert search.initialization.solution is not None
    assert search.initialization.directions is not None
    assert search.initialization.schedule is not None
    assert search.initialization.schedule.cmax is not None
    initial_coord = _fixed_coordination_cmax(
        search.initialization.solution, search.initialization.directions
    )
    assert exact.best_cmax is not None
    initial_gap = initial_coord - exact.best_cmax
    assert initial_gap > 1.0e-6
    final_gap = micro_gap_decomposition(search, exact, FAST)

    if expected_move is None:
        assert case_name == "Q3_direction"
        assert final_gap.search_gap == pytest.approx(initial_gap)
        assert exact.best_schedule is not None
        assert exact.best_schedule.directions != search.best_directions
    else:
        assert final_gap.search_gap < initial_gap - 1.0e-6
        assert search.stats.best_improvement_by_move[expected_move.value] >= 1

    if case_name == "Q2_route_order":
        assert [set(route.block_ids) for route in search.initialization.solution.routes] == [
            set(route.block_ids) for route in search.best_solution.routes
        ]
    if case_name == "Q5_interference_wait":
        assert any(
            operation.kind is OperationKind.WAIT
            for operation in search.initialization.schedule.operations
        )
    assert search.final_certification is not None
    assert search.final_certification.certified
