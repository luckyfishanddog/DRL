from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Iterable

from mrta_reference.model import MoveType, ScheduleStatus, RunScientificIdentity, ScheduleResult


ACTIVE_MOVE_TYPES = (
    MoveType.INTRA_RELOCATE,
    MoveType.INTER_RELOCATE,
    MoveType.SWAP,
    MoveType.TWO_OPT,
    MoveType.SPLIT_ACTIVATE,
    MoveType.SPLIT_DEACTIVATE,
    MoveType.SPLIT_POINT_SWITCH,
)
TRACKED_MOVE_TYPES = ACTIVE_MOVE_TYPES + (MoveType.TWO_OPT_STAR,)


def _move_counter() -> dict[str, int]:
    return {move.value: 0 for move in TRACKED_MOVE_TYPES}


def _family_counter() -> dict[str, int]:
    return {"ATOMIC": 0, "LNS_REPAIRED": 0, "DIRECTION_REFINEMENT": 0}


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
    scientific_identity: RunScientificIdentity | None = None
    development_only: bool = True
    reference_records: list[dict[str, object]] = field(default_factory=list)

    construction_attempts: int = 0
    init_reference_calls: int = 0
    init_status_counts: Counter[str] = field(default_factory=Counter)
    init_time: float = 0.0
    init_direction_failures: int = 0
    initial_strategy_wins: Counter[str] = field(default_factory=Counter)

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
    repair_time: float = 0.0
    repair_insertion_evaluations: int = 0
    repair_reference_calls: int = 0
    repair_full_direction_dp_calls: int = 0

    attempted_by_family: dict[str, int] = field(default_factory=_family_counter)
    constructed_by_family: dict[str, int] = field(default_factory=_family_counter)
    valid_by_family: dict[str, int] = field(default_factory=_family_counter)
    c4_by_family: dict[str, int] = field(default_factory=_family_counter)
    accepted_by_family: dict[str, int] = field(default_factory=_family_counter)
    improvements_by_family: dict[str, int] = field(default_factory=_family_counter)
    direction_refinement_calls: int = 0
    direction_improvements: int = 0
    operator_pair_uses: Counter[str] = field(default_factory=Counter)
    operator_pair_rewards: Counter[str] = field(default_factory=Counter)
    operator_pair_global_bests: Counter[str] = field(default_factory=Counter)
    final_operator_weights: dict[str, float] = field(default_factory=dict)
    operator_sequence: list[tuple[str, str]] = field(default_factory=list)

    attempted_by_move: dict[str, int] = field(default_factory=_move_counter)
    applicable_by_move: dict[str, int] = field(default_factory=_move_counter)
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
    per_iteration_attempts: list[int] = field(default_factory=list)
    init_scheduler_durations: list[float] = field(default_factory=list)
    search_scheduler_durations: list[float] = field(default_factory=list)
    best_events: list[tuple[float, float]] = field(default_factory=list)
    run_started: float | None = None
    requested_budget: float | None = None
    actual_runtime: float | None = None
    overshoot: float | None = None
    last_reference_start: float | None = None
    last_reference_end: float | None = None
    reference_status_sequence: list[str] = field(default_factory=list)
    proposal_trajectory: list[tuple[int, str | None, bool, float | None]] = field(
        default_factory=list
    )
    best_improvement_cmax: list[float] = field(default_factory=list)
    x_pattern_candidates_generated: int = 0
    x_pattern_candidates_cheap_feasible: int = 0
    x_pattern_candidates_c3: int = 0
    x_pattern_candidates_reference_evaluated: int = 0
    x_pattern_candidates_certified: int = 0
    x_pattern_candidates_accepted: int = 0
    x_pattern_global_best_updates: int = 0
    x_pattern_source_counts: Counter[str] = field(default_factory=Counter)
    y_pattern_candidates_generated: int = 0
    y_pattern_candidates_cheap_feasible: int = 0
    y_pattern_candidates_c3: int = 0
    y_pattern_candidates_reference_evaluated: int = 0
    y_pattern_candidates_certified: int = 0
    y_pattern_candidates_accepted: int = 0
    y_pattern_global_best_updates: int = 0
    decision_family_funnel: dict[str, Counter[str]] = field(default_factory=lambda: {
        family: Counter({stage: 0 for stage in (
            "generated", "constructed", "cheap_valid", "C2_selected",
            "direction_evaluated", "direction_feasible", "C4_selected",
            "reference_evaluated", "certified", "accepted", "global_best_update",
        )}) for family in ("STRUCTURAL", "TARGET_WHOLE", "TARGET_Y", "TARGET_X")
    })
    first_x_proposal_time: float | None = None
    first_x_c2_time: float | None = None
    first_x_reference_time: float | None = None

    def mark_first_x(self, stage: str) -> None:
        import time
        field_name = f"first_x_{stage}_time"
        if self.run_started is not None and getattr(self, field_name) is None:
            setattr(self, field_name, time.perf_counter() - self.run_started)

    @property
    def valid_by_move(self) -> dict[str, int]:
        return self.cheap_valid_by_move

    @property
    def improvements_by_move(self) -> dict[str, int]:
        return self.best_improvement_by_move

    @property
    def all_scheduler_durations(self) -> list[float]:
        return self.init_scheduler_durations + self.search_scheduler_durations

    @property
    def scheduler_durations(self) -> list[float]:
        """Backward-compatible alias for all actual scheduler calls."""
        return self.all_scheduler_durations

    @staticmethod
    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def scheduler_mean(self) -> float | None:
        return self._mean(self.all_scheduler_durations)

    @property
    def scheduler_p50(self) -> float | None:
        return _percentile(self.all_scheduler_durations, 0.50)

    @property
    def scheduler_p95(self) -> float | None:
        return _percentile(self.all_scheduler_durations, 0.95)

    @property
    def init_scheduler_mean(self) -> float | None:
        return self._mean(self.init_scheduler_durations)

    @property
    def init_scheduler_p50(self) -> float | None:
        return _percentile(self.init_scheduler_durations, 0.50)

    @property
    def init_scheduler_p95(self) -> float | None:
        return _percentile(self.init_scheduler_durations, 0.95)

    @property
    def search_scheduler_mean(self) -> float | None:
        return self._mean(self.search_scheduler_durations)

    @property
    def search_scheduler_p50(self) -> float | None:
        return _percentile(self.search_scheduler_durations, 0.50)

    @property
    def search_scheduler_p95(self) -> float | None:
        return _percentile(self.search_scheduler_durations, 0.95)

    def record_reference(
        self,
        status: ScheduleStatus,
        duration: float,
        *,
        initialization: bool,
        reference_start: float | None = None,
        reference_end: float | None = None,
        schedule: ScheduleResult | None = None,
    ) -> None:
        if schedule is not None:
            self.reference_records.append({
                "initialization": initialization, "status": status.value,
                "source": schedule.source, "diagnostics": schedule.diagnostics,
                "scope_id": schedule.scope_id, "scope_hash": schedule.scope_hash,
                "reference_policy_id": schedule.reference_policy_id,
                "expanded_states": schedule.expanded_states,
                "state_budget": schedule.state_budget,
                "recovery_rollouts": schedule.recovery_rollouts,
                "rollout_budget": schedule.rollout_budget,
                "max_discrepancies_used": schedule.max_discrepancies_used,
                "branch_points_considered": schedule.branch_points_considered,
                "frontier_exhausted": schedule.frontier_exhausted,
                "baseline_deadlock": schedule.baseline_deadlock,
            })
        if initialization:
            self.init_scheduler_durations.append(duration)
        else:
            self.search_scheduler_durations.append(duration)
        self.reference_scheduler_time += duration
        if reference_start is not None:
            self.last_reference_start = reference_start
        if reference_end is not None:
            self.last_reference_end = reference_end
        if initialization:
            self.init_reference_calls += 1
            self.init_status_counts[status.value] += 1
            return
        self.nref += 1
        self.reference_status_sequence.append(status.value)
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

    def assert_invariants(self, *, kdp: int, kref: int, m: int | None = None) -> None:
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
        if self.repair_reference_calls != 0:
            raise AssertionError("repair invoked reference scheduler")
        if self.repair_full_direction_dp_calls != 0:
            raise AssertionError("repair invoked full direction DP")
        if any(value > kdp for value in self.per_iteration_kdp):
            raise AssertionError("per-iteration Kdp hard cap exceeded")
        if any(value > kref for value in self.per_iteration_nref):
            raise AssertionError("per-iteration Kref hard cap exceeded")
        if m is not None and any(value > m for value in self.per_iteration_attempts):
            raise AssertionError("per-iteration total candidate attempt cap exceeded")
