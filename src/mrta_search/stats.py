from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Iterable

from mrta_reference.model import MoveType, ScheduleStatus


ACTIVE_MOVE_TYPES = (
    MoveType.INTRA_RELOCATE,
    MoveType.INTER_RELOCATE,
    MoveType.SWAP,
    MoveType.TWO_OPT,
    MoveType.SPLIT_ACTIVATE,
    MoveType.SPLIT_DEACTIVATE,
    MoveType.SPLIT_POINT_SWITCH,
)


def _move_counter() -> dict[str, int]:
    return {move.value: 0 for move in ACTIVE_MOVE_TYPES}


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass
class SearchStats:
    scope_id: str
    seed: int
    iterations: int = 0

    construction_attempts: int = 0
    init_reference_calls: int = 0
    init_status_counts: Counter[str] = field(default_factory=Counter)
    init_time: float = 0.0

    raw_attempts: int = 0
    constructed: int = 0
    cheap_feasible: int = 0
    duplicates: int = 0
    kdp_count: int = 0

    nref: int = 0
    n_feasible: int = 0
    n_deadlock: int = 0
    n_infeasible: int = 0
    n_numeric_failure: int = 0

    candidate_generation_time: float = 0.0
    cheap_screen_time: float = 0.0
    direction_dp_time: float = 0.0
    reference_scheduler_time: float = 0.0
    certifier_time: float = 0.0

    attempted_by_move: dict[str, int] = field(default_factory=_move_counter)
    constructed_by_move: dict[str, int] = field(default_factory=_move_counter)
    cheap_valid_by_move: dict[str, int] = field(default_factory=_move_counter)
    duplicate_by_move: dict[str, int] = field(default_factory=_move_counter)
    c3_by_move: dict[str, int] = field(default_factory=_move_counter)
    c4_by_move: dict[str, int] = field(default_factory=_move_counter)
    accepted_by_move: dict[str, int] = field(default_factory=_move_counter)
    best_improvement_by_move: dict[str, int] = field(default_factory=_move_counter)
    rejection_reasons: Counter[str] = field(default_factory=Counter)

    raw_candidate_full_dp_calls: int = 0
    raw_candidate_reference_calls: int = 0
    per_iteration_kdp: list[int] = field(default_factory=list)
    per_iteration_nref: list[int] = field(default_factory=list)
    scheduler_durations: list[float] = field(default_factory=list)
    best_events: list[tuple[float, float]] = field(default_factory=list)

    @property
    def valid_by_move(self) -> dict[str, int]:
        return self.cheap_valid_by_move

    @property
    def improvements_by_move(self) -> dict[str, int]:
        return self.best_improvement_by_move

    @property
    def scheduler_mean(self) -> float | None:
        if not self.scheduler_durations:
            return None
        return sum(self.scheduler_durations) / len(self.scheduler_durations)

    @property
    def scheduler_p50(self) -> float | None:
        return _percentile(self.scheduler_durations, 0.50)

    @property
    def scheduler_p95(self) -> float | None:
        return _percentile(self.scheduler_durations, 0.95)

    def record_reference(
        self,
        status: ScheduleStatus,
        duration: float,
        *,
        initialization: bool,
    ) -> None:
        self.scheduler_durations.append(duration)
        self.reference_scheduler_time += duration
        if initialization:
            self.init_reference_calls += 1
            self.init_status_counts[status.value] += 1
            return
        self.nref += 1
        if status is ScheduleStatus.FEASIBLE:
            self.n_feasible += 1
        elif status is ScheduleStatus.DEADLOCK:
            self.n_deadlock += 1
        elif status is ScheduleStatus.INFEASIBLE:
            self.n_infeasible += 1
        elif status is ScheduleStatus.NUMERIC_FAILURE:
            self.n_numeric_failure += 1

    def record_best(self, elapsed: float, cmax: float) -> None:
        self.best_events.append((elapsed, cmax))

    def anytime(self, checkpoints: Iterable[float]) -> dict[float, dict[str, object]]:
        result: dict[float, dict[str, object]] = {}
        for deadline in checkpoints:
            eligible = [cmax for elapsed, cmax in self.best_events if elapsed <= deadline]
            result[deadline] = (
                {"cmax": min(eligible), "reason": None}
                if eligible
                else {"cmax": None, "reason": "NO_FEASIBLE_BEFORE_DEADLINE"}
            )
        return result

    def assert_invariants(self, *, kdp: int, kref: int) -> None:
        if self.nref != (
            self.n_feasible
            + self.n_deadlock
            + self.n_infeasible
            + self.n_numeric_failure
        ):
            raise AssertionError("reference status counts do not sum to Nref")
        if self.raw_candidate_full_dp_calls != 0:
            raise AssertionError("raw candidates invoked full direction DP")
        if self.raw_candidate_reference_calls != 0:
            raise AssertionError("raw candidates invoked reference scheduler")
        if any(value > kdp for value in self.per_iteration_kdp):
            raise AssertionError("per-iteration Kdp hard cap exceeded")
        if any(value > kref for value in self.per_iteration_nref):
            raise AssertionError("per-iteration Kref hard cap exceeded")

