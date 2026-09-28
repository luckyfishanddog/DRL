from __future__ import annotations

from mrta_reference.model import ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.scheduler import reference_schedule
from mrta_search.initialization import InitializationStatus, bounded_insertion_positions, build_initial_solution
from mrta_search.stats import SearchStats


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _stats():
    return SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 7)


def test_initialization_whole_mandatory_y_bounded_positions_and_replay() -> None:
    parents = (
        ParentWeld("whole", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("mandatory", (8.0, 5.0), (8.0, 7.0)),
    )
    first = build_initial_solution(parents, FAST, _stats())
    second = build_initial_solution(parents, FAST, _stats())
    assert first.status is InitializationStatus.SUCCESS
    assert first.solution == second.solution
    assert first.directions == second.directions
    patterns = {pattern.parent_id: pattern for pattern in first.solution.patterns}
    assert patterns["whole"].kind is SplitKind.WHOLE
    assert patterns["mandatory"].kind is SplitKind.Y_SPLIT
    assert patterns["mandatory"].mandatory
    assert bounded_insertion_positions(3, 8) == (0, 1, 2, 3)
    positions = bounded_insertion_positions(20, 8)
    assert positions[0] == 0 and positions[-1] == 20 and len(positions) <= 8


def test_initialization_fallback_is_bounded_and_does_not_hide_numeric_failure() -> None:
    xs = [2.5529206381, 16.1012410018, 14.5117177606, 4.8463114890, 9.4132666547, 8.5403302310, 12.3802664817]
    config = ScientificConfig(
        weld_speed=1.0,
        empty_speed=1.0,
        t_pre=1.0,
        t_post=1.0,
        interference_dx=0.01,
        interference_dy=0.01,
    )
    parents = tuple(ParentWeld(str(i), (x, 2.0), (x + 0.5, 2.0)) for i, x in enumerate(xs))
    calls = []

    def first_deadlocks(solution, scientific_config, *, orientations):
        calls.append(solution.canonical_hash)
        if len(calls) == 1:
            return ScheduleResult(ScheduleStatus.DEADLOCK)
        return reference_schedule(solution, scientific_config, orientations=orientations)

    stats = _stats()
    result = build_initial_solution(parents, config, stats, reference_evaluator=first_deadlocks)
    assert result.status is InitializationStatus.SUCCESS
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert stats.construction_attempts == 2
    assert stats.raw_candidate_full_dp_calls == 0
    assert stats.raw_candidate_reference_calls == 0

    def numeric(*args, **kwargs):
        return ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=("boom",))

    failed = build_initial_solution(parents, config, _stats(), reference_evaluator=numeric)
    assert failed.status is InitializationStatus.NUMERIC_FAILURE
    assert len(failed.attempts) == 1


def test_initialization_failed_is_run_outcome_and_duplicate_fallback_is_not_reevaluated() -> None:
    parents = (ParentWeld("p", (1.0, 2.0), (2.0, 2.0)),)
    calls = 0

    def deadlock(*args, **kwargs):
        nonlocal calls
        calls += 1
        return ScheduleResult(ScheduleStatus.DEADLOCK)

    result = build_initial_solution(parents, FAST, _stats(), reference_evaluator=deadlock)
    assert result.status is InitializationStatus.INITIALIZATION_FAILED
    assert calls == 1
    assert len(result.attempts) == 2
    assert result.attempts[1].duplicate
