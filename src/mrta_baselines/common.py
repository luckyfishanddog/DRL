from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

from mrta_reference.certifier import CertificationReport, certify_schedule
from mrta_reference.model import (
    CanonicalSolution,
    FormalScope,
    OfficialMetrics,
    ParentWeld,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
)
from mrta_reference.scheduler import FormalReferenceEvaluator
from mrta_reference.solution import official_metrics
from mrta_reference.geometry import oriented_endpoints
from mrta_reference.solution import block_map
from mrta_search.direction import (
    ConstrainedDirectionResult,
    DirectionStatus,
    optimize_directions_with_initial_feasibility,
)


DirectionVectors = tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]
]


class BaselineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    TIME_LIMIT_NO_FEASIBLE = "TIME_LIMIT_NO_FEASIBLE"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    CONSTRUCTION_REJECTED = "CONSTRUCTION_REJECTED"
    DIRECTION_INFEASIBLE = "DIRECTION_INFEASIBLE"
    DEADLOCK = "DEADLOCK"
    INFEASIBLE = "INFEASIBLE"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


@dataclass(frozen=True)
class BaselineBestEvent:
    elapsed: float
    cmax: float
    source: str


@dataclass
class BaselineAccounting:
    candidate_count: int = 0
    cheap_candidate_operations: int = 0
    direction_dp_calls: int = 0
    reference_calls: int = 0
    certifier_calls: int = 0
    construction_rejected: int = 0
    direction_infeasible: int = 0
    baseline_deadlock: int = 0
    recovered: int = 0
    remaining_deadlock: int = 0
    infeasible: int = 0
    numeric_failure: int = 0
    scheduler_time: float = 0.0
    certifier_time: float = 0.0
    direction_time: float = 0.0
    local_search_time: float = 0.0
    crossover_time: float = 0.0
    construction_time: float = 0.0
    factorial_local_search_time: float = 0.0
    population_management_time: float = 0.0
    vnd_pass_cap_hits: int = 0
    vnd_candidate_cap_hits: int = 0
    initialization_reference_limit_hits: int = 0
    population_survival_events: int = 0
    optional_y_mutations_attempted: int = 0
    optional_y_mutations_accepted: int = 0
    factorial_window_cap_hits: int = 0
    factorial_call_cap_hits: int = 0
    wag_variant_cap_hits: int = 0
    route_combination_cap_hits: int = 0
    move_calls: int = 0
    swap_calls: int = 0
    lns_calls: int = 0
    optional_y_toggle_attempts: int = 0
    pattern_transition_proposals: int = 0
    legal_x_pattern_count: int = 0
    x_pattern_proposals: int = 0
    x_pattern_constructed: int = 0
    x_pattern_direction_evaluated: int = 0
    x_pattern_reference_evaluated: int = 0
    x_pattern_certified: int = 0
    x_pattern_accepted: int = 0
    x_pattern_global_best_updates: int = 0
    y_pattern_proposals: int = 0
    y_pattern_constructed: int = 0
    y_pattern_direction_evaluated: int = 0
    y_pattern_reference_evaluated: int = 0
    y_pattern_certified: int = 0
    y_pattern_accepted: int = 0
    y_pattern_global_best_updates: int = 0


@dataclass(frozen=True)
class EvaluatedCandidate:
    status: BaselineStatus
    solution: CanonicalSolution
    directions: DirectionVectors | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    metrics: OfficialMetrics | None
    source: str
    completed_at: float
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class BaselineResult:
    method_id: str
    status: BaselineStatus
    solution: CanonicalSolution | None
    directions: DirectionVectors | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    metrics: OfficialMetrics | None
    best_events: tuple[BaselineBestEvent, ...]
    time_to_first_certified: float | None
    iterations: int
    candidate_count: int
    reference_calls: int
    certifier_calls: int
    initialization_time: float
    search_time: float
    actual_runtime: float
    overshoot: float
    checkpoints: Mapping[float, float | None]
    diagnostics: tuple[str, ...]
    accounting: Mapping[str, int | float]
    best_source: str | None
    scope_id: str
    scope_hash: str
    method_config_hash: str
    termination_reason: str
    initial_candidate: EvaluatedCandidate | None
    initial_reference_calls: int

    @property
    def final_certified(self) -> bool:
        return bool(self.certification and self.certification.certified)


def canonical_config_hash(config: Any) -> str:
    payload = asdict(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def solution_telemetry(
    solution: CanonicalSolution,
    directions: DirectionVectors,
    config: ScientificConfig,
) -> dict[str, Any]:
    """Return method-independent structural telemetry for one certified seed."""
    blocks = block_map(solution, config)
    process_loads = []
    route_empty = []
    route_hashes = []
    for route in solution.routes:
        process_loads.append(
            sum(config.process_time(blocks[item].length) for item in route.block_ids)
        )
        empty = 0.0
        for index in range(1, len(route.block_ids)):
            previous = blocks[route.block_ids[index - 1]]
            current = blocks[route.block_ids[index]]
            _, previous_end = oriented_endpoints(
                previous, directions[route.robot_id][index - 1]
            )
            current_start, _ = oriented_endpoints(
                current, directions[route.robot_id][index]
            )
            empty += math.dist(previous_end, current_start) / config.empty_speed
        route_empty.append(empty)
        encoded = json.dumps(
            list(route.block_ids), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        route_hashes.append(hashlib.sha256(encoded).hexdigest())
    return {
        "initial_solution_hash": solution.canonical_hash,
        "initial_pattern_count": len(solution.patterns),
        "initial_optional_split_count": sum(
            pattern.kind.value != "WHOLE" and not pattern.mandatory
            for pattern in solution.patterns
        ),
        "initial_robot_block_counts": [
            len(route.block_ids) for route in solution.routes
        ],
        "initial_robot_process_loads": process_loads,
        "initial_route_proxy_costs": route_empty,
        "initial_route_empty_travel_proxy": route_empty,
        "initial_total_empty_travel_proxy": sum(route_empty),
        "initial_route_hashes": route_hashes,
        "initial_routes": [list(route.block_ids) for route in solution.routes],
        "initial_directions": [list(vector) for vector in directions],
        "initial_patterns": [
            {
                "parent_id": pattern.parent_id,
                "kind": pattern.kind.value,
                "t": pattern.t,
                "point_id": pattern.point_id,
                "mandatory": pattern.mandatory,
            }
            for pattern in solution.patterns
        ],
    }


class CommonBaselineEvaluator:
    """The sole complete-candidate scientific evaluation path for baselines."""

    def __init__(
        self,
        parents: Sequence[ParentWeld],
        config: ScientificConfig,
        scope: FormalScope,
        *,
        time_limit: float,
        checkpoints: Sequence[float] = (5.0, 30.0, 60.0),
    ) -> None:
        if time_limit <= 0.0:
            raise ValueError("time_limit must be positive")
        scope.validate_implemented()
        self.parents = tuple(parents)
        self.config = config
        self.scope = scope
        self.time_limit = float(time_limit)
        self.checkpoint_deadlines = tuple(float(value) for value in checkpoints)
        self.started = time.perf_counter()
        self.deadline = self.started + self.time_limit
        self.accounting = BaselineAccounting()
        self.best: EvaluatedCandidate | None = None
        self.best_events: list[BaselineBestEvent] = []
        self.first_certified: float | None = None
        self.failure_diagnostics: list[str] = []
        self.reference_evaluator = FormalReferenceEvaluator(scope)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def reject_construction(self, message: str) -> None:
        self.accounting.construction_rejected += 1
        self.failure_diagnostics.append(message)

    def inject_certified(
        self, candidate: EvaluatedCandidate, *, source: str
    ) -> EvaluatedCandidate:
        """Install an externally certified seed without charging evaluator calls."""
        if (
            candidate.status is not BaselineStatus.COMPLETED
            or candidate.directions is None
            or candidate.schedule is None
            or candidate.certification is None
            or not candidate.certification.certified
            or candidate.metrics is None
        ):
            raise ValueError("common seed must be a certified completed candidate")
        if candidate.solution.parents != self.parents:
            raise ValueError("common seed parents do not match evaluator instance")
        injected = replace(
            candidate,
            source=source,
            completed_at=self.elapsed,
            diagnostics=(),
        )
        self._record_best(injected)
        return injected

    def _record_best(self, candidate: EvaluatedCandidate) -> None:
        if candidate.metrics is None:
            return
        if self.first_certified is None:
            self.first_certified = candidate.completed_at
        if self.best is None or candidate.metrics.compare(self.best.metrics) < 0:  # type: ignore[arg-type]
            self.best = candidate
            has_x = any(
                pattern.kind is SplitKind.X_SPLIT
                for pattern in candidate.solution.patterns
            )
            has_y = any(
                pattern.kind is SplitKind.Y_SPLIT
                for pattern in candidate.solution.patterns
            )
            self.accounting.x_pattern_global_best_updates += int(has_x)
            self.accounting.y_pattern_global_best_updates += int(has_y)
            self.best_events.append(
                BaselineBestEvent(candidate.completed_at, candidate.metrics.cmax, candidate.source)
            )

    def evaluate(self, solution: CanonicalSolution, *, source: str) -> EvaluatedCandidate:
        self.accounting.candidate_count += 1
        has_x = any(pattern.kind is SplitKind.X_SPLIT for pattern in solution.patterns)
        has_y = any(pattern.kind is SplitKind.Y_SPLIT for pattern in solution.patterns)
        self.accounting.x_pattern_constructed += int(has_x)
        self.accounting.y_pattern_constructed += int(has_y)
        direction_started = time.perf_counter()
        direction: ConstrainedDirectionResult
        try:
            direction = optimize_directions_with_initial_feasibility(solution, self.config)
        except (ArithmeticError, OverflowError, ValueError) as error:
            self.accounting.numeric_failure += 1
            completed = self.elapsed
            result = EvaluatedCandidate(
                BaselineStatus.NUMERIC_FAILURE,
                solution,
                None,
                None,
                None,
                None,
                source,
                completed,
                (str(error),),
            )
            self.failure_diagnostics.append(f"{source}: {error}")
            return result
        self.accounting.direction_time += time.perf_counter() - direction_started
        self.accounting.direction_dp_calls += direction.route_dp_calls
        self.accounting.x_pattern_direction_evaluated += int(has_x)
        self.accounting.y_pattern_direction_evaluated += int(has_y)
        if direction.status is not DirectionStatus.FEASIBLE:
            self.accounting.direction_infeasible += 1
            return EvaluatedCandidate(
                BaselineStatus.DIRECTION_INFEASIBLE,
                solution,
                direction.directions,
                None,
                None,
                None,
                source,
                self.elapsed,
                direction.diagnostics,
            )

        orientations = {robot: direction.directions[robot] for robot in range(4)}
        scheduler_started = time.perf_counter()
        try:
            schedule = self.reference_evaluator(
                solution, self.config, orientations=orientations
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            schedule = ScheduleResult(
                ScheduleStatus.NUMERIC_FAILURE,
                diagnostics=(str(error),),
                directions=direction.directions,
                reference_policy_id=self.scope.reference_scheduler_policy_id,
                scope_id=self.scope.scope_id,
                scope_hash=self.scope.scope_hash,
            )
        self.accounting.scheduler_time += time.perf_counter() - scheduler_started
        self.accounting.reference_calls += 1
        self.accounting.x_pattern_reference_evaluated += int(has_x)
        self.accounting.y_pattern_reference_evaluated += int(has_y)
        if schedule.baseline_deadlock:
            self.accounting.baseline_deadlock += 1
            if schedule.status is ScheduleStatus.FEASIBLE:
                self.accounting.recovered += 1
        if schedule.status is ScheduleStatus.DEADLOCK:
            self.accounting.remaining_deadlock += 1
            return EvaluatedCandidate(
                BaselineStatus.DEADLOCK,
                solution,
                direction.directions,
                schedule,
                None,
                None,
                source,
                self.elapsed,
                schedule.diagnostics,
            )
        if schedule.status is ScheduleStatus.INFEASIBLE:
            self.accounting.infeasible += 1
            return EvaluatedCandidate(
                BaselineStatus.INFEASIBLE,
                solution,
                direction.directions,
                schedule,
                None,
                None,
                source,
                self.elapsed,
                schedule.diagnostics,
            )
        if schedule.status is not ScheduleStatus.FEASIBLE:
            self.accounting.numeric_failure += 1
            return EvaluatedCandidate(
                BaselineStatus.NUMERIC_FAILURE,
                solution,
                direction.directions,
                schedule,
                None,
                None,
                source,
                self.elapsed,
                schedule.diagnostics,
            )

        cert_started = time.perf_counter()
        certification = certify_schedule(solution, schedule, self.config, scope=self.scope)
        self.accounting.certifier_time += time.perf_counter() - cert_started
        self.accounting.certifier_calls += 1
        if not certification.certified:
            self.accounting.numeric_failure += 1
            schedule = replace(
                schedule,
                status=ScheduleStatus.NUMERIC_FAILURE,
                cmax=None,
                diagnostics=(
                    "common baseline certification failure: "
                    + "; ".join(certification.errors),
                ),
            )
            result = EvaluatedCandidate(
                BaselineStatus.NUMERIC_FAILURE,
                solution,
                direction.directions,
                schedule,
                certification,
                None,
                source,
                self.elapsed,
                schedule.diagnostics,
            )
            self.failure_diagnostics.extend(result.diagnostics)
            return result

        metrics = official_metrics(solution, schedule, self.config)
        self.accounting.x_pattern_certified += int(has_x)
        self.accounting.y_pattern_certified += int(has_y)
        self.accounting.x_pattern_accepted += int(has_x)
        self.accounting.y_pattern_accepted += int(has_y)
        result = EvaluatedCandidate(
            BaselineStatus.COMPLETED,
            solution,
            direction.directions,
            schedule,
            certification,
            metrics,
            source,
            self.elapsed,
            (),
        )
        self._record_best(result)
        return result

    def checkpoint_values(self) -> dict[float, float | None]:
        return {
            deadline: min(
                (event.cmax for event in self.best_events if event.elapsed <= deadline),
                default=None,
            )
            if deadline <= self.time_limit
            else None
            for deadline in self.checkpoint_deadlines
        }

    def finish(
        self,
        *,
        method_id: str,
        method_config_hash: str,
        iterations: int,
        initialization_time: float,
        termination_reason: str = "COMPLETED_OTHER",
        initial_candidate: EvaluatedCandidate | None = None,
        initial_reference_calls: int = 0,
        diagnostics: Sequence[str] = (),
    ) -> BaselineResult:
        runtime = self.elapsed
        best = self.best
        if best is not None:
            status = BaselineStatus.COMPLETED
        elif self.accounting.numeric_failure:
            status = BaselineStatus.NUMERIC_FAILURE
        elif self.accounting.candidate_count == 0:
            status = BaselineStatus.INITIALIZATION_FAILED
        else:
            status = BaselineStatus.TIME_LIMIT_NO_FEASIBLE
        return BaselineResult(
            method_id=method_id,
            status=status,
            solution=None if best is None else best.solution,
            directions=None if best is None else best.directions,
            schedule=None if best is None else best.schedule,
            certification=None if best is None else best.certification,
            metrics=None if best is None else best.metrics,
            best_events=tuple(self.best_events),
            time_to_first_certified=self.first_certified,
            iterations=iterations,
            candidate_count=self.accounting.candidate_count,
            reference_calls=self.accounting.reference_calls,
            certifier_calls=self.accounting.certifier_calls,
            initialization_time=initialization_time,
            search_time=max(0.0, runtime - initialization_time),
            actual_runtime=runtime,
            overshoot=max(0.0, runtime - self.time_limit),
            checkpoints=self.checkpoint_values(),
            diagnostics=tuple(diagnostics) + tuple(self.failure_diagnostics),
            accounting=asdict(self.accounting),
            best_source=None if best is None else best.source,
            scope_id=self.scope.scope_id,
            scope_hash=self.scope.scope_hash,
            method_config_hash=method_config_hash,
            termination_reason=termination_reason,
            initial_candidate=initial_candidate,
            initial_reference_calls=initial_reference_calls,
        )
