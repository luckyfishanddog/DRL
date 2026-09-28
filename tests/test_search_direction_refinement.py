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


def test_refinement_certification_failure_is_effective_numeric_status(monkeypatch):
    from mrta_reference import certifier
    baseline = run_bounded_sa_oi(Q3, FAST, SearchConfig(max_iterations=0), seed=3)
    monkeypatch.setattr(certifier, "certify_schedule", lambda *a, **k:
                        certifier.CertificationReport(False, ("injected failure",), None))
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 3)
    result = refine_directions_bounded(
        baseline.best_solution, baseline.best_directions, baseline.best_schedule,
        FAST, max_calls=1, reference_evaluator=lambda *a, **k: baseline.best_schedule,
        stats=stats,
    )
    assert result.statuses == (ScheduleStatus.NUMERIC_FAILURE,)
    assert stats.reference_status_sequence == ["NUMERIC_FAILURE"]
    assert stats.nref == stats.n_numeric_failure == 1
    assert stats.n_feasible == 0
    assert "injected failure" in result.diagnostics[0]
    assert not result.improved


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


def test_formal_fixed_budget_replay_history_independence_and_shared_policy():
    from dataclasses import replace
    from mrta_reference.model import FORMAL_SCOPE_V1
    from mrta_reference.scheduler import reference_schedule_formal
    config = SearchConfig(max_iterations=4)
    results = []
    for algorithm_label, seed in (("SA-OI-ALNS", 19), ("unrelated-caller", 99), ("replay", 19)):
        result = run_bounded_sa_oi(Q3, FAST, config, seed=seed,
                                  scope=FORMAL_SCOPE_V1, source_commit="test-commit")
        results.append(result)
        assert result.final_certification.certified
        assert result.stats.direction_refinement_calls > 0
        assert result.stats.scientific_identity.scope_hash == FORMAL_SCOPE_V1.scope_hash
        assert {row["initialization"] for row in result.stats.reference_records} == {True, False}
        assert all(row["scope_hash"] == FORMAL_SCOPE_V1.scope_hash
                   and row["reference_policy_id"] == FORMAL_SCOPE_V1.reference_scheduler_policy_id
                   for row in result.stats.reference_records)
    first, _, replay = results
    assert first.best_solution.canonical_json == replay.best_solution.canonical_json
    assert first.best_directions == replay.best_directions
    assert first.best_schedule.canonical_json() == replay.best_schedule.canonical_json()
    assert first.stats.reference_status_sequence == replay.stats.reference_status_sequence
    assert first.stats.proposal_trajectory == replay.stats.proposal_trajectory
    assert first.stats.operator_sequence == replay.stats.operator_sequence
    assert first.stats.final_operator_weights == replay.stats.final_operator_weights
    fixed = first.initialization.solution
    directions = dict(enumerate(first.initialization.directions))
    expected = reference_schedule_formal(fixed, FAST, orientations=directions)
    for other in reversed(results):
        reference_schedule_formal(other.best_solution, FAST, orientations=dict(enumerate(other.best_directions)))
        actual = reference_schedule_formal(replace(fixed, revision=99), FAST,
                                            orientations=dict(reversed(tuple(directions.items()))))
        assert actual == expected
    with pytest.raises(ValueError, match="matching formal"):
        run_bounded_sa_oi(Q3, FAST, config, scope=FORMAL_SCOPE_V1, source_commit="test",
                          reference_evaluator=reference_schedule)


def test_formal_refinement_certifier_failure_preserves_scope_and_numeric_diagnostics(monkeypatch):
    from mrta_reference import certifier
    from mrta_reference.scope import FORMAL_SCOPE_V1
    from mrta_reference.scheduler import FormalReferenceEvaluator
    baseline = run_bounded_sa_oi(Q3, FAST, SearchConfig(max_iterations=0),
                                 scope=FORMAL_SCOPE_V1, source_commit="test")
    monkeypatch.setattr(certifier, "certify_schedule", lambda *a, **k:
                        certifier.CertificationReport(False, ("forced formal failure",), None))
    stats = SearchStats(FORMAL_SCOPE_V1.scope_id, 0)
    result = refine_directions_bounded(
        baseline.best_solution, baseline.best_directions, baseline.best_schedule, FAST,
        max_calls=1, reference_evaluator=FormalReferenceEvaluator(), scope=FORMAL_SCOPE_V1, stats=stats,
    )
    assert result.statuses == (ScheduleStatus.NUMERIC_FAILURE,)
    assert stats.nref == stats.n_numeric_failure == 1 and stats.n_feasible == 0
    assert stats.reference_records[0]["scope_hash"] == FORMAL_SCOPE_V1.scope_hash
    assert "forced formal failure" in stats.reference_records[0]["diagnostics"]


@pytest.mark.parametrize("name,expected", (
    ("Q1_assignment_trap", 7.15), ("Q2_route_order_trap", 11.468135899374975),
    ("Q3_direction_trap", 6.271416626834708), ("Q4_optional_y_split_trap", 4.0),
    ("Q5_interference_wait_trap", 6.271416626834708), ("Q6_lns_basin_trap", 3.557181856472053),
))
def test_formal_q1_q6_regression(name, expected, monkeypatch):
    from pathlib import Path
    from mrta_reference.scope import FORMAL_SCOPE_V1
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from profile_scheduler import quality_fixture
    parents, seed, iterations = quality_fixture(name)
    result = run_bounded_sa_oi(parents, FAST, SearchConfig(max_iterations=iterations),
                               seed=seed, scope=FORMAL_SCOPE_V1, source_commit="test")
    assert result.final_certification.certified
    assert result.best_schedule.cmax == pytest.approx(expected)
    assert result.best_schedule.cmax < result.initialization.schedule.cmax
    assert result.stats.n_numeric_failure == 0
    if name.startswith("Q6"):
        assert result.stats.improvements_by_family["LNS_REPAIRED"] > 0
