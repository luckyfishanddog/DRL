from __future__ import annotations

import pytest

from mrta_exact import solve_exact_micro
from mrta_reference.model import ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig
from mrta_reference.scheduler import reference_schedule
from mrta_search import SearchConfig, run_bounded_sa_oi
from mrta_search.direction import refine_directions_bounded
from mrta_search.stats import SearchStats


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)
Q3 = (
    ParentWeld("u", (1.26, 6.33), (5.33, 6.21)),
    ParentWeld("l", (6.26, 5.68), (1.99, 5.79)),
)


def test_q3_bounded_single_flip_improves_direction_gap_to_current_scope_optimum() -> None:
    baseline = run_bounded_sa_oi(
        Q3,
        FAST,
        SearchConfig(
            max_iterations=0,
            direction_refinement_budget=0,
            kref_total=2,
        ),
        seed=3,
    )
    assert baseline.best_solution is not None
    assert baseline.best_directions is not None
    assert baseline.best_schedule is not None
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 3)
    refined = refine_directions_bounded(
        baseline.best_solution,
        baseline.best_directions,
        baseline.best_schedule,
        FAST,
        max_calls=4,
        reference_evaluator=reference_schedule,
        stats=stats,
    )
    exact = solve_exact_micro(Q3, FAST)
    assert refined.calls <= 4
    assert refined.improved
    assert refined.schedule.cmax is not None and exact.best_cmax is not None
    assert refined.schedule.cmax < baseline.best_schedule.cmax - 1.0e-6
    assert refined.schedule.cmax == pytest.approx(exact.best_cmax)
    assert stats.nref == refined.calls == stats.direction_refinement_calls


def test_refinement_status_isolated_and_budget_is_hard() -> None:
    baseline = run_bounded_sa_oi(Q3, FAST, SearchConfig(max_iterations=0), seed=3)
    assert baseline.best_solution is not None
    assert baseline.best_directions is not None
    assert baseline.best_schedule is not None

    def deadlock(*args, **kwargs):
        return ScheduleResult(ScheduleStatus.DEADLOCK, diagnostics=("blocked",))

    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 3)
    result = refine_directions_bounded(
        baseline.best_solution,
        baseline.best_directions,
        baseline.best_schedule,
        FAST,
        max_calls=1,
        reference_evaluator=deadlock,
        stats=stats,
    )
    assert result.calls == 1
    assert result.statuses == (ScheduleStatus.DEADLOCK,)
    assert not result.improved
    assert result.directions == baseline.best_directions
    assert stats.n_deadlock == 1 and stats.n_infeasible == 0


def test_search_iteration_total_reference_cap_includes_refinement_and_replays() -> None:
    config = SearchConfig(
        m=48,
        m_atomic=32,
        m_lns=16,
        kdp=8,
        kref=2,
        kref_total=3,
        direction_refinement_budget=4,
        max_iterations=4,
    )
    first = run_bounded_sa_oi(Q3, FAST, config, seed=19)
    second = run_bounded_sa_oi(Q3, FAST, config, seed=19)
    assert max(first.stats.per_iteration_nref, default=0) <= 3
    assert first.best_solution.canonical_json == second.best_solution.canonical_json
    assert first.best_directions == second.best_directions
    assert first.best_schedule.cmax == pytest.approx(second.best_schedule.cmax)
    assert first.stats.operator_sequence == second.stats.operator_sequence
    assert first.stats.proposal_trajectory == second.stats.proposal_trajectory
    assert first.stats.final_operator_weights == second.stats.final_operator_weights
    assert first.final_certification is not None and first.final_certification.certified
