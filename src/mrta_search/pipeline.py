from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
import random
import time
from pathlib import Path

from mrta_exact import EXACT_Y_SCOPE_CURRENT_SEMANTICS
from mrta_exact import ExactResult, ExactSolveStatus, exact_schedule_from_templates
from mrta_reference.certifier import CertificationReport, certify_schedule
from mrta_reference.model import (
    CandidateMove,
    CanonicalSolution,
    OfficialMetrics,
    ParentWeld,
    MoveType,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
)
from mrta_reference.scheduler import reference_schedule
from mrta_reference.scheduler import resolve_reference_evaluator
from mrta_reference.model import FormalScope, RunScientificIdentity, DEVELOPMENT_NO_REPAIR_V1
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.provenance import (
    SourceProvenance,
    SourceProvenanceError,
    resolve_source_provenance,
)
from mrta_reference.scheduler import build_operation_templates, build_robot_routes
from mrta_reference.solution import official_metrics

from .direction import (
    ConstrainedDirectionResult,
    DirectionVectors,
    optimize_directions_with_initial_feasibility,
    refine_directions_bounded,
)
from .initialization import (
    InitializationResult,
    InitializationStatus,
    InitializationStrategy,
    build_initial_solution,
)
from .lns import (
    AdaptiveOperatorState,
    CandidateSourceKind,
    CompleteSearchCandidate,
    DestroyOperator,
    RepairOperator,
    atomic_complete_candidate,
    destroy_parents,
    destroy_size,
    lns_pair,
    repair_partial_state,
)
from .neighborhood import (
    ScreenedCandidate,
    candidate_key_order,
    generate_raw_attempts,
    screen_raw_attempts,
)
from .stats import ACTIVE_MOVE_TYPES, SearchStats


REFERENCE_POLICY_ID = DEVELOPMENT_NO_REPAIR_V1
SA_OI_ALNS_V2_METHOD_ID = "SA_OI_ALNS_V2"
PATTERN_TRANSITION_BALANCED_FAMILY_V1 = "PATTERN_TRANSITION_BALANCED_FAMILY_V1"
V2_C2_PATTERN_FAMILY_STRATIFIED_V1 = "V2_C2_PATTERN_FAMILY_STRATIFIED_V1"
V2_C4_FAMILY_EXPLORE_EXPLOIT_V1 = "V2_C4_FAMILY_EXPLORE_EXPLOIT_V1"
DECISION_FAMILIES = ("STRUCTURAL", "TARGET_WHOLE", "TARGET_Y", "TARGET_X")


@dataclass(frozen=True)
class SearchConfig:
    m: int = 64
    kdp: int = 8
    kref: int = 2
    insertion_limit: int = 8
    construction_budget: int = 4
    kinit_ref: int = 2
    feasibility_bootstrap_budget: int = 1
    m_atomic: int | None = None
    m_lns: int = 16
    kref_total: int = 4
    destroy_rho: float = 0.15
    destroy_q_min: int = 2
    destroy_q_max: int = 8
    repair_insertion_limit: int = 8
    repair_evaluation_cap: int = 4096
    adaptive_reaction: float = 0.2
    adaptive_segment_length: int = 20
    reward_global_best: float = 8.0
    reward_accepted_improvement: float = 4.0
    reward_accepted_non_improvement: float = 1.0
    reward_rejected: float = 0.0
    reward_evaluation_failed: float = 0.0
    direction_refinement_budget: int = 4
    t0: float = 0.05
    cooling_rate: float = 0.995
    tmin: float = 1.0e-6
    max_iterations: int = 100
    time_limit: float | None = None
    checkpoints: tuple[float, ...] = (1.0, 5.0, 30.0)
    enable_two_opt_star: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.kref <= self.kdp <= self.m):
            raise ValueError("require 1 <= Kref <= Kdp <= M")
        if self.insertion_limit < 2:
            raise ValueError("I_init must be at least 2")
        if not (1 <= self.construction_budget <= len(InitializationStrategy)):
            raise ValueError(
                f"B_init_pool must be between 1 and {len(InitializationStrategy)}"
            )
        if not (1 <= self.kinit_ref <= self.construction_budget):
            raise ValueError("require 1 <= Kinit_ref <= B_init_pool")
        if self.feasibility_bootstrap_budget not in (0, 1):
            raise ValueError("B_init_bootstrap must be 0 or 1")
        atomic = self.m - self.m_lns if self.m_atomic is None else self.m_atomic
        if min(atomic, self.m_lns) < 0 or atomic + self.m_lns > self.m:
            raise ValueError("atomic + LNS attempts must not exceed M")
        if self.kref_total < self.kref:
            raise ValueError("Kref_total must be at least Kref")
        if self.destroy_q_min < 1 or self.destroy_q_max < self.destroy_q_min:
            raise ValueError("invalid destroy q bounds")
        if self.repair_insertion_limit < 2 or self.repair_evaluation_cap < 1:
            raise ValueError("invalid repair bounds")
        if self.adaptive_segment_length < 1 or not (0.0 < self.adaptive_reaction <= 1.0):
            raise ValueError("invalid adaptive configuration")
        if self.direction_refinement_budget < 0:
            raise ValueError("Bdir must be non-negative")
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
    screened: CompleteSearchCandidate | ScreenedCandidate
    cheap_rank: int
    direction: ConstrainedDirectionResult

    @property
    def rerank_key(self) -> tuple[object, ...]:
        identity = (
            self.screened.candidate_identity
            if isinstance(self.screened, CompleteSearchCandidate)
            else candidate_key_order(self.screened.candidate.key)
        )
        return (
            self.screened.projected_process_makespan,
            math.inf
            if self.direction.total_empty_travel is None
            else self.direction.total_empty_travel,
            self.screened.split_count_delta,
            self.cheap_rank,
            identity,
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
    complete_candidates: tuple[CompleteSearchCandidate, ...]
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
    termination_reason: str


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


def decision_family(candidate: CompleteSearchCandidate | ScreenedCandidate | CandidateMove) -> str:
    """Classify the explicit transition, never patterns retained in the solution."""
    if isinstance(candidate, CompleteSearchCandidate):
        move = candidate.atomic_move
    elif isinstance(candidate, ScreenedCandidate):
        move = candidate.candidate
    else:
        move = candidate
    if move is None:
        return "STRUCTURAL"
    if move.key.move_type is MoveType.SPLIT_DEACTIVATE:
        return "TARGET_WHOLE"
    if move.split_pattern is not None:
        return {
            SplitKind.WHOLE: "TARGET_WHOLE",
            SplitKind.Y_SPLIT: "TARGET_Y",
            SplitKind.X_SPLIT: "TARGET_X",
        }[move.split_pattern.kind]
    return "STRUCTURAL"


def select_c2_by_family(candidates, kdp: int, *, seed: int, iteration: int):
    ranked = sorted(candidates, key=lambda item: item.cheap_score)
    families = DECISION_FAMILIES
    if kdp < len(families):
        offset = (seed + iteration) % len(families)
        families = families[offset:] + families[:offset]
    selected = []
    for family in families:
        candidate = next((row for row in ranked if decision_family(row) == family), None)
        if candidate is not None and len(selected) < kdp:
            selected.append(candidate)
    selected_ids = {id(row) for row in selected}
    selected.extend(row for row in ranked if id(row) not in selected_ids)
    return tuple(selected[:kdp])


def select_c4_by_family(candidates, kref: int, *, seed: int, iteration: int):
    ranked = sorted(candidates, key=lambda item: item.rerank_key)
    if kref <= 1 or len(ranked) <= 1:
        return tuple(ranked[:kref])
    selected = [ranked[0]]
    offset = (seed + iteration) % len(DECISION_FAMILIES)
    for family in DECISION_FAMILIES[offset:] + DECISION_FAMILIES[:offset]:
        candidate = next((row for row in ranked if row is not selected[0]
                          and decision_family(row.screened) == family), None)
        if candidate is not None:
            selected.append(candidate)
            break
    selected_ids = {id(row) for row in selected}
    selected.extend(row for row in ranked if id(row) not in selected_ids)
    return tuple(selected[:kref])


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
            and _complete_identity(item.direction_candidate.screened)
            < _complete_identity(best.direction_candidate.screened)
        ):
            best = item
    return best


def _complete_identity(
    candidate: CompleteSearchCandidate | ScreenedCandidate,
) -> tuple[object, ...] | object:
    if isinstance(candidate, CompleteSearchCandidate):
        return candidate.candidate_identity
    return candidate_key_order(candidate.candidate.key)


def _solution_of(
    candidate: CompleteSearchCandidate | ScreenedCandidate,
) -> CanonicalSolution:
    return candidate.solution


def _family_of(candidate: CompleteSearchCandidate | ScreenedCandidate) -> str:
    if isinstance(candidate, CompleteSearchCandidate):
        return candidate.source_kind.value
    return CandidateSourceKind.ATOMIC.value


def _move_name(candidate: CompleteSearchCandidate | ScreenedCandidate) -> str | None:
    if isinstance(candidate, CompleteSearchCandidate):
        return (
            None
            if candidate.atomic_move is None
            else candidate.atomic_move.key.move_type.value
        )
    return candidate.candidate.key.move_type.value


def _x_patterns(solution: CanonicalSolution):
    return tuple(
        pattern for pattern in solution.patterns if pattern.kind is SplitKind.X_SPLIT
    )


def _y_patterns(solution: CanonicalSolution):
    return tuple(
        pattern for pattern in solution.patterns if pattern.kind is SplitKind.Y_SPLIT
    )


ReferenceEvaluator = Callable[..., ScheduleResult]


def evaluate_iteration(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    config: ScientificConfig,
    search_config: SearchConfig,
    stats: SearchStats,
    *,
    seed: int,
    reference_evaluator: ReferenceEvaluator | None = None,
    current_schedule: ScheduleResult | None = None,
    adaptive_state: AdaptiveOperatorState | None = None,
    scope: FormalScope | None = None,
    enable_x_split: bool = False,
    family_access_policy: bool | None = None,
) -> IterationResult:
    family_access = scope == FORMAL_SCOPE_V2 if family_access_policy is None else family_access_policy
    if family_access and scope != FORMAL_SCOPE_V2:
        raise ValueError("decision-family access policies require FORMAL_SCOPE_V2")
    reference_evaluator = resolve_reference_evaluator(scope, reference_evaluator)
    attempts_before = stats.raw_attempts
    atomic_budget = (
        search_config.m
        if current_schedule is None
        else (
            search_config.m - search_config.m_lns
            if search_config.m_atomic is None
            else search_config.m_atomic
        )
    )
    raw = generate_raw_attempts(
        current,
        config,
        m=atomic_budget,
        seed=seed,
        stats=stats,
        moves=(
            ACTIVE_MOVE_TYPES + (MoveType.TWO_OPT_STAR,)
            if search_config.enable_two_opt_star
            else ACTIVE_MOVE_TYPES
        ),
        enable_x_split=enable_x_split,
        scope=scope,
    )
    stats.attempted_by_family[CandidateSourceKind.ATOMIC.value] += len(raw)
    stats.constructed_by_family[CandidateSourceKind.ATOMIC.value] += sum(
        attempt.candidate is not None for attempt in raw
    )
    for attempt in raw:
        pattern = None if attempt.candidate is None else attempt.candidate.split_pattern
        if attempt.candidate is not None:
            stats.decision_family_funnel[decision_family(attempt.candidate)]["generated"] += 1
        if pattern is not None and pattern.kind is SplitKind.X_SPLIT:
            stats.mark_first_x("proposal")
            stats.x_pattern_candidates_generated += 1
            if pattern.point_id is not None:
                stats.x_pattern_source_counts[pattern.point_id] += 1
        if pattern is not None and pattern.kind is SplitKind.Y_SPLIT:
            stats.y_pattern_candidates_generated += 1
    screened = screen_raw_attempts(
        current,
        current_directions,
        raw,
        config,
        stats,
        enable_x_split=enable_x_split,
        scope=scope,
    )
    stats.x_pattern_candidates_cheap_feasible += sum(
        bool(_x_patterns(item.solution)) for item in screened
    )
    stats.y_pattern_candidates_cheap_feasible += sum(
        bool(_y_patterns(item.solution)) for item in screened
    )
    complete: list[CompleteSearchCandidate] = []
    seen_solutions: set[str] = {current.canonical_hash}
    for item in screened:
        stats.decision_family_funnel[decision_family(item)]["constructed"] += 1
        candidate = atomic_complete_candidate(current, current_directions, item, config)
        if candidate.solution.canonical_hash in seen_solutions:
            stats.duplicates += 1
            continue
        seen_solutions.add(candidate.solution.canonical_hash)
        complete.append(candidate)
        stats.decision_family_funnel[decision_family(candidate)]["cheap_valid"] += 1
        stats.valid_by_family[CandidateSourceKind.ATOMIC.value] += 1

    if current_schedule is not None and search_config.m_lns:
        if adaptive_state is None:
            adaptive_state = AdaptiveOperatorState(
                search_config.adaptive_reaction,
                search_config.adaptive_segment_length,
            )
        lns_rng = random.Random(seed + 7_919_119 * (current.revision + 1))
        q = destroy_size(
            len(current.parents),
            rho=search_config.destroy_rho,
            q_min=search_config.destroy_q_min,
            q_max=search_config.destroy_q_max,
        )
        for ordinal in range(search_config.m_lns):
            stats.decision_family_funnel["STRUCTURAL"]["generated"] += 1
            stats.raw_attempts += 1
            stats.attempted_by_family[CandidateSourceKind.LNS_REPAIRED.value] += 1
            pair = adaptive_state.select(lns_rng)
            generation_started = time.perf_counter()
            partial = destroy_parents(
                current,
                current_schedule,
                pair[0],
                q=q,
                seed=seed + 104_729 * ordinal,
                config=config,
            )
            repair_started = time.perf_counter()
            repaired = repair_partial_state(
                current,
                current_directions,
                partial,
                pair[0],
                pair[1],
                config,
                insertion_limit=search_config.repair_insertion_limit,
                max_insertion_evaluations=search_config.repair_evaluation_cap,
            )
            stats.repair_time += time.perf_counter() - repair_started
            stats.candidate_generation_time += repair_started - generation_started
            stats.repair_insertion_evaluations += repaired.insertion_evaluations
            if repaired.candidate is None:
                stats.rejection_reasons[
                    f"LNS:{repaired.rejection_reason or 'REPAIR_FAILED'}"
                ] += 1
                continue
            stats.constructed += 1
            stats.constructed_by_family[CandidateSourceKind.LNS_REPAIRED.value] += 1
            candidate = repaired.candidate
            stats.decision_family_funnel["STRUCTURAL"]["constructed"] += 1
            if candidate.solution.canonical_hash in seen_solutions:
                stats.duplicates += 1
                stats.rejection_reasons["DUPLICATE_CANONICAL_SOLUTION"] += 1
                continue
            seen_solutions.add(candidate.solution.canonical_hash)
            complete.append(candidate)
            stats.decision_family_funnel["STRUCTURAL"]["cheap_valid"] += 1
            stats.cheap_feasible += 1
            stats.valid_by_family[CandidateSourceKind.LNS_REPAIRED.value] += 1

    cheap_ranked = tuple(sorted(complete, key=lambda item: item.cheap_score))
    c2 = (
        select_c2_by_family(cheap_ranked, search_config.kdp, seed=stats.seed, iteration=stats.iterations)
        if family_access else cheap_ranked[: search_config.kdp]
    )
    stats.kdp_count += len(c2)
    stats.per_iteration_kdp.append(len(c2))

    c3 = []
    global_ranks = {id(row): rank for rank, row in enumerate(cheap_ranked)}
    for candidate in c2:
        cheap_rank = global_ranks[id(candidate)]
        decision = decision_family(candidate)
        funnel = stats.decision_family_funnel[decision]
        funnel["C2_selected"] += 1
        funnel["direction_evaluated"] += 1
        if decision == "TARGET_X":
            stats.mark_first_x("c2")
        started = time.perf_counter()
        direction = optimize_directions_with_initial_feasibility(
            candidate.solution, config
        )
        stats.direction_dp_time += time.perf_counter() - started
        move_name = _move_name(candidate)
        if move_name is not None:
            stats.c3_by_move[move_name] += 1
        c3.append(DirectionEvaluatedCandidate(candidate, cheap_rank, direction))
        funnel["direction_feasible"] += int(direction.total_empty_travel is not None)
        if _x_patterns(candidate.solution):
            stats.x_pattern_candidates_c3 += 1
        if _y_patterns(candidate.solution):
            stats.y_pattern_candidates_c3 += 1
    feasible_c3 = [item for item in c3 if item.direction.total_empty_travel is not None]
    shortlist = (
        select_c4_by_family(feasible_c3, search_config.kref, seed=stats.seed, iteration=stats.iterations)
        if family_access else rerank_c3(feasible_c3, search_config.kref)
    )

    cache: dict[tuple[object, ...], ReferenceEvaluatedCandidate] = {}
    c4 = []
    calls_before = stats.nref
    for item in shortlist:
        decision = decision_family(item.screened)
        funnel = stats.decision_family_funnel[decision]
        funnel["C4_selected"] += 1
        move_name = _move_name(item.screened)
        family = _family_of(item.screened)
        has_x = bool(_x_patterns(_solution_of(item.screened)))
        if has_x:
            stats.x_pattern_candidates_reference_evaluated += 1
        has_y = bool(_y_patterns(_solution_of(item.screened)))
        if has_y:
            stats.y_pattern_candidates_reference_evaluated += 1
        stats.c4_by_family[family] += 1
        if move_name is not None:
            stats.c4_by_move[move_name] += 1
        cache_key = (
            EXACT_Y_SCOPE_CURRENT_SEMANTICS if scope is None else scope.scope_hash,
            config.scientific_hash,
            _solution_of(item.screened).canonical_hash,
            item.direction.directions,
            REFERENCE_POLICY_ID if scope is None else scope.reference_scheduler_policy_id,
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
        funnel["reference_evaluated"] += 1
        if decision == "TARGET_X":
            stats.mark_first_x("reference")
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
            certification = certify_schedule(_solution_of(item.screened), schedule, config, scope=scope)
            stats.certifier_time += time.perf_counter() - cert_started
            if certification.certified:
                funnel["certified"] += 1
                metrics = official_metrics(_solution_of(item.screened), schedule, config)
                if has_x:
                    stats.x_pattern_candidates_certified += 1
                if has_y:
                    stats.y_pattern_candidates_certified += 1
            else:
                effective_status = ScheduleStatus.NUMERIC_FAILURE
                schedule = replace(
                    schedule, status=ScheduleStatus.NUMERIC_FAILURE, cmax=None,
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
            schedule=schedule,
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
    stats.per_iteration_attempts.append(stats.raw_attempts - attempts_before)
    return IterationResult(
        tuple(screened), tuple(complete), tuple(c3), tuple(c4), _proposal(c4)
    )


def run_bounded_sa_oi(
    parents: Sequence[ParentWeld],
    config: ScientificConfig = ScientificConfig(),
    search_config: SearchConfig = SearchConfig(),
    *,
    seed: int = 0,
    reference_evaluator: ReferenceEvaluator | None = None,
    scope: FormalScope | None = None,
    source_provenance: SourceProvenance | None = None,
    source_commit: str | None = None,
    allow_unverified_source: bool = False,
    formal_result: bool = False,
    initialization_override: InitializationResult | None = None,
    enable_x_split: bool = False,
    family_access_policy: bool | None = None,
) -> SearchResult:
    started = time.perf_counter()
    if enable_x_split and (
        scope is None or scope.optional_x_split_policy != "FINITE_GEOMETRIC_X_SPLIT_V1"
    ):
        raise ValueError("X_SPLIT search requires EXPERIMENTAL_X_SPLIT_SCOPE_V1")
    reference_evaluator = resolve_reference_evaluator(scope, reference_evaluator)
    stats = SearchStats(EXACT_Y_SCOPE_CURRENT_SEMANTICS if scope is None else scope.scope_id, seed)
    if scope is not None:
        root = Path(__file__).resolve().parents[2]
        if source_provenance is not None and source_commit is not None:
            raise ValueError("provide source_provenance or source_commit, not both")
        if source_provenance is None:
            source_provenance = resolve_source_provenance(
                root,
                source_commit=source_commit,
                allow_unverified_source=allow_unverified_source,
            )
        if formal_result:
            verified = resolve_source_provenance(root)
            if source_provenance != verified:
                raise SourceProvenanceError(
                    "formal result provenance does not match the executing DRL tree"
                )
            source_provenance.require_formal_result()
        stats.scientific_identity = RunScientificIdentity.from_scope(
            scope, config, source_provenance
        )
        stats.development_only = not formal_result
    stats.run_started = started
    stats.requested_budget = search_config.time_limit
    rng = random.Random(seed)
    if initialization_override is None:
        initialization = build_initial_solution(
            parents,
            config,
            stats,
            insertion_limit=search_config.insertion_limit,
            construction_budget=search_config.construction_budget,
            kinit_ref=search_config.kinit_ref,
            feasibility_bootstrap_budget=search_config.feasibility_bootstrap_budget,
            portfolio=True,
            reference_evaluator=reference_evaluator,
            scope=scope,
        )
    else:
        initialization = initialization_override
        if (
            initialization.status is not InitializationStatus.SUCCESS
            or initialization.solution is None
            or initialization.solution.parents != tuple(parents)
            or initialization.directions is None
            or initialization.schedule is None
            or initialization.certification is None
            or not initialization.certification.certified
        ):
            raise ValueError("initialization_override must be a matching certified seed")
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
        stats.assert_invariants(
            kdp=search_config.kdp, kref=search_config.kref_total, m=search_config.m
        )
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
            status.value,
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
    adaptive = AdaptiveOperatorState(
        search_config.adaptive_reaction,
        search_config.adaptive_segment_length,
    )

    def record_lns_observation(
        evaluated: ReferenceEvaluatedCandidate,
        reward: float | None,
        *,
        global_best: bool = False,
    ) -> None:
        candidate = evaluated.direction_candidate.screened
        if not isinstance(candidate, CompleteSearchCandidate):
            return
        pair = lns_pair(candidate)
        if pair is None:
            return
        label = f"{pair[0].value}+{pair[1].value}"
        stats.operator_pair_uses[label] += 1
        if reward is not None:
            stats.operator_pair_rewards[label] += reward
        if global_best:
            stats.operator_pair_global_bests[label] += 1
        adaptive.record(pair, reward)

    termination_reason = "ITERATION_LIMIT"
    for iteration in range(search_config.max_iterations):
        if (
            iteration > 0
            and
            search_config.time_limit is not None
            and time.perf_counter() - started >= search_config.time_limit
        ):
            termination_reason = "TIME_LIMIT"
            break
        iteration_calls_before = stats.nref
        result = evaluate_iteration(
            current_solution,
            current_directions,
            config,
            search_config,
            stats,
            seed=seed + iteration * 65_537,
            reference_evaluator=reference_evaluator,
            current_schedule=current_schedule,
            adaptive_state=adaptive,
            scope=scope,
            enable_x_split=enable_x_split,
            family_access_policy=family_access_policy,
        )
        stats.iterations += 1
        proposal = result.proposal
        for evaluated in result.c4_candidates:
            if evaluated is proposal:
                continue
            if evaluated.schedule.status is ScheduleStatus.NUMERIC_FAILURE:
                reward = None
            elif evaluated.metrics is None:
                reward = search_config.reward_evaluation_failed
            else:
                reward = search_config.reward_rejected
            record_lns_observation(evaluated, reward)
        if proposal is None or proposal.metrics is None:
            stats.proposal_trajectory.append((iteration, None, False, None))
            stats.per_iteration_nref[-1] = stats.nref - iteration_calls_before
            continue
        remaining_refinement = min(
            search_config.direction_refinement_budget,
            max(0, search_config.kref_total - (stats.nref - iteration_calls_before)),
        )
        proposal_refined = False
        if remaining_refinement:
            base_candidate = proposal.direction_candidate
            refined = refine_directions_bounded(
                _solution_of(base_candidate.screened),
                base_candidate.direction.directions,
                proposal.schedule,
                config,
                max_calls=remaining_refinement,
                reference_evaluator=reference_evaluator,
                stats=stats,
                scope=scope,
            )
            if refined.improved:
                proposal_refined = True
                refined_metrics = official_metrics(
                    _solution_of(base_candidate.screened), refined.schedule, config
                )
                refined_direction = ConstrainedDirectionResult(
                    base_candidate.direction.status,
                    refined.directions,
                    refined_metrics.total_empty_travel,
                    base_candidate.direction.diagnostics,
                    base_candidate.direction.legal_first_combinations,
                    base_candidate.direction.route_dp_calls,
                )
                refined_candidate = DirectionEvaluatedCandidate(
                    base_candidate.screened,
                    base_candidate.cheap_rank,
                    refined_direction,
                )
                certification = certify_schedule(
                    _solution_of(base_candidate.screened), refined.schedule, config, scope=scope
                )
                proposal = ReferenceEvaluatedCandidate(
                    refined_candidate,
                    refined.schedule,
                    certification,
                    refined_metrics,
                    False,
                )
        stats.per_iteration_nref[-1] = stats.nref - iteration_calls_before
        current_before = current_metrics
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
            record_lns_observation(proposal, search_config.reward_rejected)
            continue
        stats.proposal_trajectory.append(
            (
                iteration,
                proposal.direction_candidate.screened.solution.canonical_hash,
                True,
                proposal.metrics.cmax,
            )
        )
        candidate = proposal.direction_candidate.screened
        family = _family_of(candidate)
        move_name = _move_name(candidate)
        stats.accepted_by_family[family] += 1
        stats.decision_family_funnel[decision_family(candidate)]["accepted"] += 1
        accepted_has_x = bool(_x_patterns(_solution_of(candidate)))
        accepted_has_y = bool(_y_patterns(_solution_of(candidate)))
        if accepted_has_x:
            stats.x_pattern_candidates_accepted += 1
        if accepted_has_y:
            stats.y_pattern_candidates_accepted += 1
        if move_name is not None:
            stats.accepted_by_move[move_name] += 1
        current_solution = _solution_of(candidate)
        current_directions = proposal.direction_candidate.direction.directions
        current_schedule = proposal.schedule
        current_metrics = proposal.metrics
        global_best = current_metrics.compare(best_metrics) < 0
        if global_best:
            stats.decision_family_funnel[decision_family(candidate)]["global_best_update"] += 1
            best_solution = current_solution
            best_directions = current_directions
            best_schedule = current_schedule
            best_metrics = current_metrics
            best_family = (
                "DIRECTION_REFINEMENT" if proposal_refined else family
            )
            stats.improvements_by_family[best_family] += 1
            if move_name is not None:
                stats.best_improvement_by_move[move_name] += 1
            stats.best_improvement_cmax.append(best_metrics.cmax)
            stats.record_best(time.perf_counter() - started, best_metrics.cmax)
            if accepted_has_x:
                stats.x_pattern_global_best_updates += 1
            if accepted_has_y:
                stats.y_pattern_global_best_updates += 1
        if global_best:
            reward = search_config.reward_global_best
        elif current_metrics.cmax < current_before.cmax - 1.0e-9 * max(
            1.0, abs(current_before.cmax), abs(current_metrics.cmax)
        ):
            reward = search_config.reward_accepted_improvement
        else:
            reward = search_config.reward_accepted_non_improvement
        record_lns_observation(proposal, reward, global_best=global_best)

    cert_started = time.perf_counter()
    final_certification = certify_schedule(best_solution, best_schedule, config, scope=scope)
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
    if status is SearchStatus.NUMERIC_FAILURE:
        termination_reason = "NUMERIC_FAILURE"
    stats.operator_sequence = list(adaptive.selection_history)
    stats.final_operator_weights = {
        f"{pair[0].value}+{pair[1].value}": weight
        for pair, weight in adaptive.weights.items()
    }
    stats.assert_invariants(
        kdp=search_config.kdp, kref=search_config.kref_total, m=search_config.m
    )
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
        termination_reason,
    )


def run_sa_oi_alns_v2(
    parents: Sequence[ParentWeld],
    config: ScientificConfig = ScientificConfig(),
    search_config: SearchConfig = SearchConfig(),
    *,
    seed: int = 0,
    source_provenance: SourceProvenance | None = None,
    source_commit: str | None = None,
    allow_unverified_source: bool = False,
    formal_result: bool = False,
) -> SearchResult:
    return run_bounded_sa_oi(
        parents,
        config,
        search_config,
        seed=seed,
        scope=FORMAL_SCOPE_V2,
        source_provenance=source_provenance,
        source_commit=source_commit,
        allow_unverified_source=allow_unverified_source,
        formal_result=formal_result,
        enable_x_split=True,
    )


def micro_gap_decomposition(
    search: SearchResult,
    exact: ExactResult,
    config: ScientificConfig,
) -> MicroGapResult:
    if search.stats.scope_id != EXACT_Y_SCOPE_CURRENT_SEMANTICS:
        raise ValueError("development exact gaps cannot label a formal-scope run")
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
