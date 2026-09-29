from __future__ import annotations

from mrta_reference.model import ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.scheduler import reference_schedule
import pytest

from mrta_search.initialization import (
    RAIL_SERIAL_BOOTSTRAP,
    InitializationStatus,
    bounded_insertion_positions,
    build_initial_solution,
)
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
    assert len(result.attempts) == 3
    assert result.attempts[1].duplicate
    assert result.attempts[2].duplicate
    assert result.attempts[2].strategy == RAIL_SERIAL_BOOTSTRAP


def test_all_four_diagnostics_kinit_filter_and_bounded_bootstrap_are_reproducible() -> None:
    parents = tuple(
        ParentWeld(str(index), (x, 2.0), (x + 0.5, 2.0))
        for index, x in enumerate(
            (2.55, 16.10, 14.51, 4.85, 9.41, 8.54, 12.38, 6.25)
        )
    )
    config = ScientificConfig(
        weld_speed=1.0,
        empty_speed=1.0,
        t_pre=1.0,
        t_post=1.0,
        interference_dx=0.01,
        interference_dy=0.01,
    )

    def run_once():
        calls = []

        def evaluator(solution, scientific_config, *, orientations):
            calls.append(solution.canonical_hash)
            if not solution.routes[1].block_ids and not solution.routes[3].block_ids:
                return reference_schedule(
                    solution, scientific_config, orientations=orientations
                )
            return ScheduleResult(ScheduleStatus.DEADLOCK)

        result = build_initial_solution(
            parents,
            config,
            _stats(),
            construction_budget=4,
            kinit_ref=2,
            feasibility_bootstrap_budget=1,
            portfolio=True,
            reference_evaluator=evaluator,
        )
        return result, calls

    first, first_calls = run_once()
    second, second_calls = run_once()
    assert first.status is InitializationStatus.SUCCESS
    assert first.winning_strategy == RAIL_SERIAL_BOOTSTRAP
    assert first.solution == second.solution
    assert first.directions == second.directions
    assert first_calls == second_calls
    assert len(first_calls) == 3  # Kinit_ref=2 plus one bounded bootstrap call.
    assert [attempt.strategy for attempt in first.attempts[:4]] == [
        "LOAD_FIRST",
        "RAIL_BALANCED",
        "X_ORDER_AWARE",
        "SPATIAL_SPREAD",
    ]
    assert sum(attempt.schedule is not None for attempt in first.attempts[:4]) == 2
    assert first.attempts[-1].strategy == RAIL_SERIAL_BOOTSTRAP
    assert not first.solution.routes[1].block_ids
    assert not first.solution.routes[3].block_ids


def test_feasibility_bootstrap_budget_can_be_disabled_and_is_validated() -> None:
    parents = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (10.0, 2.0), (11.0, 2.0)),
    )

    def deadlock(*args, **kwargs):
        return ScheduleResult(ScheduleStatus.DEADLOCK)

    result = build_initial_solution(
        parents,
        FAST,
        _stats(),
        feasibility_bootstrap_budget=0,
        reference_evaluator=deadlock,
    )
    assert all(attempt.strategy != RAIL_SERIAL_BOOTSTRAP for attempt in result.attempts)
    with pytest.raises(ValueError, match="B_init_bootstrap"):
        build_initial_solution(
            parents,
            FAST,
            _stats(),
            feasibility_bootstrap_budget=2,
        )
