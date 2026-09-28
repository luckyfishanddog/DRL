from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import math
import random
import time

from mrta_exact import EXACT_Y_SCOPE_CURRENT_SEMANTICS
from mrta_exact import ExactResult, ExactSolveStatus, exact_schedule_from_templates
from mrta_reference.certifier import CertificationReport, certify_schedule
from mrta_reference.model import (
    CandidateMove,
    CanonicalSolution,
    OfficialMetrics,
    ParentWeld,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
)
from mrta_reference.scheduler import reference_schedule
from mrta_reference.scheduler import build_operation_templates, build_robot_routes
from mrta_reference.solution import official_metrics

from .direction import (
    ConstrainedDirectionResult,
    DirectionVectors,
    optimize_directions_with_initial_feasibility,
)
from .initialization import (
    InitializationResult,
    InitializationStatus,
    build_initial_solution,
)
from .neighborhood import (
    ScreenedCandidate,
    candidate_key_order,
    generate_raw_attempts,
    screen_raw_attempts,
)
from .stats import SearchStats


REFERENCE_POLICY_ID = "REFERENCE_LIST_SCHEDULER_NO_REPAIR_V1"


@dataclass(frozen=True)
class SearchConfig:
    m: int = 64
    kdp: int = 8
    kref: int = 2
    insertion_limit: int = 8
    construction_budget: int = 2
    t0: float = 0.05
    cooling_rate: float = 0.995
    tmin: float = 1.0e-6
    max_iterations: int = 100
    time_limit: float | None = None
    checkpoints: tuple[float, ...] = (1.0, 5.0, 30.0)

    def __post_init__(self) -> None:
        if not (1 <= self.kref <= self.kdp <= self.m):
            raise ValueError("require 1 <= Kref <= Kdp <= M")
        if self.insertion_limit < 2:
            raise ValueError("I_init must be at least 2")
        if self.construction_budget not in (1, 2):
            raise ValueError("B_init must be 1 or 2")
        if self.t0 < 0.0 or self.tmin < 0.0:
            raise ValueError("temperatures must be non-negative")
        if not (0.0 < self.cooling_rate <= 1.0):
            raise ValueError("cooling_rate must be in (0,1]")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.time_limit is not None and self.time_limit < 0.0:
            raise ValueError("time_limit must be non-negative")


@dataclass(frozen=True)
class DirectionEvaluatedCandidate:
    screened: ScreenedCandidate
    cheap_rank: int
    direction: ConstrainedDirectionResult

    @property
    def rerank_key(self) -> tuple[object, ...]:
        return (
            self.screened.projected_process_makespan,
            math.inf
            if self.direction.total_empty_travel is None
            else self.direction.total_empty_travel,
            self.screened.split_count_delta,
            self.cheap_rank,
            candidate_key_order(self.screened.candidate.key),
        )


@dataclass(frozen=True)
class ReferenceEvaluatedCandidate:
    direction_candidate: DirectionEvaluatedCandidate
    schedule: ScheduleResult
    certification: CertificationReport | None
    metrics: OfficialMetrics | None
    cache_hit: bool = False


@dataclass(frozen=True)
class IterationResult:
    c1_candidates: tuple[ScreenedCandidate, ...]
    c3_candidates: tuple[DirectionEvaluatedCandidate, ...]
    c4_candidates: tuple[ReferenceEvaluatedCandidate, ...]
    proposal: ReferenceEvaluatedCandidate | None


class SearchStatus(str, Enum):
    COMPLETED = "COMPLETED"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


@dataclass(frozen=True)
class SearchResult:
    status: SearchStatus
    best_solution: CanonicalSolution | None
    best_directions: DirectionVectors | None
    best_schedule: ScheduleResult | None
    best_metrics: OfficialMetrics | None
    final_certification: CertificationReport | None
    initialization: InitializationResult
    stats: SearchStats
    anytime: dict[float, dict[str, object]]
    runtime: float


@dataclass(frozen=True)
class MicroGapResult:
    c_star: float
    c_coord: float
    c_ref: float
    search_gap: float
    scheduler_gap: float
    total_gap: float


def cscale_for(parents: Sequence[ParentWeld], config: ScientificConfig) -> float:
    return max(
        1.0,
        sum(config.process_time(parent.length) for parent in parents) / 4.0,
    )


def temperature(config: SearchConfig, reference_count: int) -> float:
    return max(config.tmin, config.t0 * (config.cooling_rate**reference_count))


def sa_accept(
    current_cmax: float,
    new_cmax: float,
    *,
    temperature_value: float,
    cscale: float,
    rng: random.Random,
) -> tuple[bool, float, float | None]:
    epsilon = 1.0e-9 * max(1.0, abs(current_cmax), abs(new_cmax))
    if new_cmax <= current_cmax + epsilon:
        return True, 1.0, None
    if temperature_value <= 0.0:
        return False, 0.0, None
    probability = math.exp(
        -(new_cmax - current_cmax) / (temperature_value * cscale)
    )
    draw = rng.random()
    return draw < probability, probability, draw


def rerank_c3(
    candidates: Sequence[DirectionEvaluatedCandidate], kref: int
) -> tuple[DirectionEvaluatedCandidate, ...]:
    return tuple(sorted(candidates, key=lambda item: item.rerank_key)[:kref])


def _proposal(
    candidates: Sequence[ReferenceEvaluatedCandidate],
) -> ReferenceEvaluatedCandidate | None:
    feasible = [item for item in candidates if item.metrics is not None]
    if not feasible:
        return None
    best = feasible[0]
    for item in feasible[1:]:
        assert best.metrics is not None and item.metrics is not None
        comparison = item.metrics.compare(best.metrics)
        if comparison < 0 or (
            comparison == 0
            and candidate_key_order(item.direction_candidate.screened.candidate.key)
            < candidate_key_order(best.direction_candidate.screened.candidate.key)
        ):
            best = item
    return best


ReferenceEvaluator = Callable[..., ScheduleResult]


def evaluate_iteration(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    config: ScientificConfig,
    search_config: SearchConfig,
    stats: SearchStats,
    *,
    seed: int,
    reference_evaluator: ReferenceEvaluator = reference_schedule,
) -> IterationResult:
    raw = generate_raw_attempts(
        current,
        config,
        m=search_config.m,
        seed=seed,
        stats=stats,
    )
    screened = screen_raw_attempts(current, current_directions, raw, config, stats)
    cheap_ranked = tuple(sorted(screened, key=lambda item: item.cheap_score))
    c2 = cheap_ranked[: search_config.kdp]
    stats.kdp_count += len(c2)
    stats.per_iteration_kdp.append(len(c2))

    c3 = []
    for cheap_rank, candidate in enumerate(c2):
        started = time.perf_counter()
        direction = optimize_directions_with_initial_feasibility(
            candidate.solution, config
        )
        stats.direction_dp_time += time.perf_counter() - started
        stats.c3_by_move[candidate.candidate.key.move_type.value] += 1
        c3.append(DirectionEvaluatedCandidate(candidate, cheap_rank, direction))
    shortlist = rerank_c3(c3, search_config.kref)

    cache: dict[tuple[object, ...], ReferenceEvaluatedCandidate] = {}
    c4 = []
    calls_before = stats.nref
    for item in shortlist:
        move_name = item.screened.candidate.key.move_type.value
        stats.c4_by_move[move_name] += 1
        cache_key = (
            EXACT_Y_SCOPE_CURRENT_SEMANTICS,
            config.scientific_hash,
            item.screened.solution.canonical_hash,
            item.direction.directions,
            REFERENCE_POLICY_ID,
        )
        if cache_key in cache:
            cached = cache[cache_key]
            evaluated = ReferenceEvaluatedCandidate(
                item,
                cached.schedule,
                cached.certification,
                cached.metrics,
                True,
            )
            c4.append(evaluated)
            continue
        orientation_map = {
            robot: item.direction.directions[robot] for robot in range(4)
        }
        started = time.perf_counter()
        try:
            schedule = reference_evaluator(
                item.screened.solution,
                config,
                orientations=orientation_map,
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            schedule = ScheduleResult(
                ScheduleStatus.NUMERIC_FAILURE,
                diagnostics=(str(error),),
                directions=item.direction.directions,
            )
        scheduler_duration = time.perf_counter() - started
        reference_ended = time.perf_counter()
        certification = None
        metrics = None
        effective_status = schedule.status
        if schedule.status is ScheduleStatus.FEASIBLE:
            cert_started = time.perf_counter()
            certification = certify_schedule(item.screened.solution, schedule, config)
            stats.certifier_time += time.perf_counter() - cert_started
            if certification.certified:
                metrics = official_metrics(item.screened.solution, schedule, config)
            else:
                effective_status = ScheduleStatus.NUMERIC_FAILURE
                schedule = ScheduleResult(
                    ScheduleStatus.NUMERIC_FAILURE,
                    diagnostics=(
                        "candidate FEASIBLE schedule failed certification: "
                        + "; ".join(certification.errors),
                    ),
                    directions=item.direction.directions,
                )
        stats.record_reference(
            effective_status,
            scheduler_duration,
            initialization=False,
            reference_start=(
                None if stats.run_started is None else started - stats.run_started
            ),
            reference_end=(
                None
                if stats.run_started is None
                else reference_ended - stats.run_started
            ),
        )
        evaluated = ReferenceEvaluatedCandidate(
            item, schedule, certification, metrics, False
        )
        cache[cache_key] = evaluated
        c4.append(evaluated)
    stats.per_iteration_nref.append(stats.nref - calls_before)
    return IterationResult(tuple(screened), tuple(c3), tuple(c4), _proposal(c4))


def run_bounded_sa_oi(
    parents: Sequence[ParentWeld],
    config: ScientificConfig = ScientificConfig(),
    search_config: SearchConfig = SearchConfig(),
    *,
    seed: int = 0,
    reference_evaluator: ReferenceEvaluator = reference_schedule,
) -> SearchResult:
    started = time.perf_counter()
    stats = SearchStats(EXACT_Y_SCOPE_CURRENT_SEMANTICS, seed)
    stats.run_started = started
    stats.requested_budget = search_config.time_limit
    rng = random.Random(seed)
    initialization = build_initial_solution(
        parents,
        config,
        stats,
        insertion_limit=search_config.insertion_limit,
        construction_budget=search_config.construction_budget,
        reference_evaluator=reference_evaluator,
    )
    if initialization.status is not InitializationStatus.SUCCESS:
        runtime = time.perf_counter() - started
        stats.actual_runtime = runtime
        stats.overshoot = (
            None
            if search_config.time_limit is None
            else max(0.0, runtime - search_config.time_limit)
        )
        status = (
            SearchStatus.NUMERIC_FAILURE
            if initialization.status is InitializationStatus.NUMERIC_FAILURE
            else SearchStatus.INITIALIZATION_FAILED
        )
        stats.assert_invariants(kdp=search_config.kdp, kref=search_config.kref)
        return SearchResult(
            status,
            None,
            None,
            None,
            None,
            None,
            initialization,
            stats,
            stats.anytime(search_config.checkpoints),
            runtime,
        )

    assert initialization.solution is not None
    assert initialization.directions is not None
    assert initialization.schedule is not None
    current_solution = initialization.solution
    current_directions = initialization.directions
    current_schedule = initialization.schedule
    current_metrics = official_metrics(current_solution, current_schedule, config)
    best_solution = current_solution
    best_directions = current_directions
    best_schedule = current_schedule
    best_metrics = current_metrics
    stats.record_best(time.perf_counter() - started, best_metrics.cmax)
    scale = cscale_for(parents, config)

    for iteration in range(search_config.max_iterations):
        if (
            search_config.time_limit is not None
            and time.perf_counter() - started >= search_config.time_limit
        ):
            break
        result = evaluate_iteration(
            current_solution,
            current_directions,
            config,
            search_config,
            stats,
            seed=seed + iteration * 65_537,
            reference_evaluator=reference_evaluator,
        )
        stats.iterations += 1
        proposal = result.proposal
        if proposal is None or proposal.metrics is None:
            stats.proposal_trajectory.append((iteration, None, False, None))
            continue
        accepted, _, _ = sa_accept(
            current_metrics.cmax,
            proposal.metrics.cmax,
            temperature_value=temperature(search_config, stats.nref),
            cscale=scale,
            rng=rng,
        )
        if not accepted:
            stats.proposal_trajectory.append(
                (
                    iteration,
                    proposal.direction_candidate.screened.solution.canonical_hash,
                    False,
                    proposal.metrics.cmax,
                )
            )
            continue
        stats.proposal_trajectory.append(
            (
                iteration,
                proposal.direction_candidate.screened.solution.canonical_hash,
                True,
                proposal.metrics.cmax,
            )
        )
        move_name = proposal.direction_candidate.screened.candidate.key.move_type.value
        stats.accepted_by_move[move_name] += 1
        current_solution = proposal.direction_candidate.screened.solution
        current_directions = proposal.direction_candidate.direction.directions
        current_schedule = proposal.schedule
        current_metrics = proposal.metrics
        if current_metrics.compare(best_metrics) < 0:
            best_solution = current_solution
            best_directions = current_directions
            best_schedule = current_schedule
            best_metrics = current_metrics
            stats.best_improvement_by_move[move_name] += 1
            stats.best_improvement_cmax.append(best_metrics.cmax)
            stats.record_best(time.perf_counter() - started, best_metrics.cmax)

    cert_started = time.perf_counter()
    final_certification = certify_schedule(best_solution, best_schedule, config)
    stats.certifier_time += time.perf_counter() - cert_started
    runtime = time.perf_counter() - started
    stats.actual_runtime = runtime
    stats.overshoot = (
        None
        if search_config.time_limit is None
        else max(0.0, runtime - search_config.time_limit)
    )
    status = (
        SearchStatus.COMPLETED
        if final_certification.certified
        else SearchStatus.NUMERIC_FAILURE
    )
    stats.assert_invariants(kdp=search_config.kdp, kref=search_config.kref)
    return SearchResult(
        status,
        best_solution,
        best_directions,
        best_schedule,
        best_metrics,
        final_certification,
        initialization,
        stats,
        stats.anytime(search_config.checkpoints),
        runtime,
    )


def micro_gap_decomposition(
    search: SearchResult,
    exact: ExactResult,
    config: ScientificConfig,
) -> MicroGapResult:
    if (
        search.status is not SearchStatus.COMPLETED
        or search.best_solution is None
        or search.best_directions is None
        or search.best_schedule is None
        or search.best_schedule.cmax is None
    ):
        raise ValueError("micro decomposition requires a completed FEASIBLE search")
    if exact.status is not ExactSolveStatus.OPTIMAL or exact.best_cmax is None:
        raise ValueError("micro decomposition requires current-scope exact optimum")
    routes = build_robot_routes(
        search.best_solution,
        config,
        {robot: search.best_directions[robot] for robot in range(4)},
    )
    templates = {
        route.robot_id: build_operation_templates(route, config) for route in routes
    }
    coordination = exact_schedule_from_templates(
        templates,
        config,
        directions=search.best_directions,
    )
    if coordination.status is not ScheduleStatus.FEASIBLE or coordination.schedule is None or coordination.schedule.cmax is None:
        raise RuntimeError("fixed search solution has no exact-coordination schedule")
    c_star = exact.best_cmax
    c_coord = coordination.schedule.cmax
    c_ref = search.best_schedule.cmax
    tolerance = 1.0e-9 * max(1.0, abs(c_star), abs(c_coord), abs(c_ref))
    if c_coord < c_star - tolerance:
        raise RuntimeError("negative search gap: exact scope/directions/semantics mismatch")
    if c_coord > c_ref + tolerance:
        raise RuntimeError("exact coordination is worse than reference: implementation inconsistency")
    search_gap = c_coord - c_star
    scheduler_gap = c_ref - c_coord
    total_gap = c_ref - c_star
    if abs(total_gap - search_gap - scheduler_gap) > tolerance:
        raise RuntimeError("micro gap decomposition identity failed")
    return MicroGapResult(
        c_star,
        c_coord,
        c_ref,
        search_gap,
        scheduler_gap,
        total_gap,
    )
