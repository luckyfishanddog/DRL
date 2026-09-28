from __future__ import annotations

import pytest

from mrta_reference.model import MoveType, ScheduleStatus
from mrta_search.stats import SearchStats


def test_stats_status_invariant_percentiles_moves_and_anytime_completion_semantics() -> None:
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 4)
    stats.record_reference(
        ScheduleStatus.FEASIBLE,
        10.0,
        initialization=True,
        reference_start=0.1,
        reference_end=10.1,
    )
    for status, duration in (
        (ScheduleStatus.FEASIBLE, 1.0),
        (ScheduleStatus.DEADLOCK, 2.0),
        (ScheduleStatus.INFEASIBLE, 3.0),
        (ScheduleStatus.NUMERIC_FAILURE, 4.0),
    ):
        stats.record_reference(
            status,
            duration,
            initialization=False,
            reference_start=duration,
            reference_end=duration + 0.5,
        )
    stats.attempted_by_move[MoveType.SWAP.value] = 2
    stats.cheap_valid_by_move[MoveType.SWAP.value] = 1
    stats.c4_by_move[MoveType.SWAP.value] = 1
    stats.record_best(0.8, 12.0)
    stats.record_best(1.2, 10.0)
    stats.per_iteration_kdp.append(8)
    stats.per_iteration_nref.append(2)
    stats.assert_invariants(kdp=8, kref=2)
    assert stats.init_reference_calls == 1
    assert stats.nref == 4
    assert stats.init_scheduler_mean == pytest.approx(10.0)
    assert stats.init_scheduler_p50 == pytest.approx(10.0)
    assert stats.search_scheduler_mean == pytest.approx(2.5)
    assert stats.search_scheduler_p50 == pytest.approx(2.5)
    assert stats.search_scheduler_p95 == pytest.approx(3.85)
    assert stats.scheduler_mean == pytest.approx(4.0)
    assert stats.scheduler_p50 == pytest.approx(3.0)
    assert stats.scheduler_p95 == pytest.approx(8.8)
    assert stats.last_reference_start == pytest.approx(4.0)
    assert stats.last_reference_end == pytest.approx(4.5)
    checkpoints = stats.anytime((0.5, 1.0, 5.0))
    assert checkpoints[0.5]["cmax"] is None
    assert checkpoints[1.0]["cmax"] == 12.0
    assert checkpoints[5.0]["cmax"] == 10.0
    assert stats.valid_by_move[MoveType.SWAP.value] == 1
