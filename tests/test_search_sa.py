from __future__ import annotations

import random

import pytest

from mrta_exact import solve_exact_micro
from mrta_reference.model import CandidateKey, CandidateMove, MoveType, ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scheduler import reference_schedule
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

