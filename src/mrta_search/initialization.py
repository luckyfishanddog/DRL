from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
import math
import time

from mrta_reference.certifier import CertificationReport, certify_schedule
from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    oriented_endpoints,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    ParentWeld,
    Route,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.scheduler import reference_schedule
from mrta_reference.solution import canonicalize

from .direction import (
    ConstrainedDirectionResult,
    DirectionStatus,
    DirectionVectors,
    optimize_directions_with_initial_feasibility,
)
from .stats import SearchStats


class InitializationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


@dataclass(frozen=True)
class InitializationAttempt:
    construction_index: int
    solution: CanonicalSolution | None
    directions: ConstrainedDirectionResult | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    diagnostics: tuple[str, ...] = ()
    duplicate: bool = False


@dataclass(frozen=True)
class InitializationResult:
    status: InitializationStatus
    solution: CanonicalSolution | None
    directions: DirectionVectors | None
    schedule: ScheduleResult | None
    certification: CertificationReport | None
    attempts: tuple[InitializationAttempt, ...]


ReferenceEvaluator = Callable[..., ScheduleResult]
Certifier = Callable[..., CertificationReport]


def bounded_insertion_positions(length: int, limit: int) -> tuple[int, ...]:
    if limit < 2:
        raise ValueError("I_init must be at least 2")
    if length + 1 <= limit:
        return tuple(range(length + 1))
    return tuple(sorted({math.floor(index * length / (limit - 1)) for index in range(limit)}))


def _patterns(parents: Sequence[ParentWeld], config: ScientificConfig):
    result = []
    for parent in sorted(parents, key=lambda item: item.parent_id):
        if whole_eligible_rails(parent.start, parent.end, config):
            result.append(SplitPattern(parent.parent_id, SplitKind.WHOLE))
            continue
        legal = []
        for pattern in generate_y_split_patterns(parent, config):
            blocks = blocks_for_pattern(parent, pattern, config)
            if all(any(robot_is_eligible(block, robot, config) for robot in range(4)) for block in blocks):
                legal.append(pattern)
        if not legal:
            raise ValueError(f"{parent.parent_id}: no legal mandatory Y pattern")
        result.append(min(legal, key=lambda item: (item.t, item.point_id)))
    return tuple(result)


def _local_delta(
    route: list[str],
    hints: list[int],
    position: int,
    block: WeldingBlock,
    hint: int,
    blocks: dict[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    start, end = oriented_endpoints(block, hint)
    delta = 0.0
    if position > 0:
        _, left_end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        delta += math.dist(left_end, start) / config.empty_speed
    if position < len(route):
        right_start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        delta += math.dist(end, right_start) / config.empty_speed
    if 0 < position < len(route):
        _, left_end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        right_start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        delta -= math.dist(left_end, right_start) / config.empty_speed
    return delta


def _construct(
    parents: Sequence[ParentWeld],
    patterns: tuple[SplitPattern, ...],
    config: ScientificConfig,
    *,
    insertion_limit: int,
    fallback_order: bool,
) -> CanonicalSolution:
    parent_by_id = {parent.parent_id: parent for parent in parents}
    all_blocks = {
        block.block_id: block
        for pattern in patterns
        for block in blocks_for_pattern(parent_by_id[pattern.parent_id], pattern, config)
    }
    ordered_blocks = sorted(
        all_blocks.values(),
        key=lambda block: (-config.process_time(block.length), block.parent_id, block.block_id),
    )
    routes: list[list[str]] = [[], [], [], []]
    hints: list[list[int]] = [[], [], [], []]
    loads = [0.0, 0.0, 0.0, 0.0]
    for block in ordered_blocks:
        candidates = []
        duration = config.process_time(block.length)
        for robot in range(4):
            if not robot_is_eligible(block, robot, config):
                continue
            for position in bounded_insertion_positions(len(routes[robot]), insertion_limit):
                for hint in (0, 1):
                    delta = _local_delta(
                        routes[robot], hints[robot], position, block, hint, all_blocks, config
                    )
                    projected = max(
                        loads[index] + (duration if index == robot else 0.0)
                        for index in range(4)
                    )
                    if fallback_order:
                        score = (projected, robot, delta, position, hint)
                    else:
                        score = (projected, delta, robot, position, hint)
                    candidates.append((score, robot, position, hint))
        if not candidates:
            raise ValueError(f"{block.block_id}: no eligible insertion")
        _, robot, position, hint = min(candidates, key=lambda item: item[0])
        routes[robot].insert(position, block.block_id)
        hints[robot].insert(position, hint)
        loads[robot] += duration
    return canonicalize(
        parents,
        patterns,
        tuple(Route(robot, tuple(routes[robot])) for robot in range(4)),
        config,
        revision=0,
    )


def build_initial_solution(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    stats: SearchStats,
    *,
    insertion_limit: int = 8,
    construction_budget: int = 2,
    reference_evaluator: ReferenceEvaluator = reference_schedule,
    certifier: Certifier = certify_schedule,
) -> InitializationResult:
    if construction_budget < 1 or construction_budget > 2:
        raise ValueError("Phase 2B-1 B_init must be 1 or 2")
    started = time.perf_counter()
    attempts: list[InitializationAttempt] = []
    seen: set[str] = set()
    try:
        selected_patterns = _patterns(parents, config)
    except (ValueError, ArithmeticError, OverflowError) as error:
        stats.init_time += time.perf_counter() - started
        return InitializationResult(
            InitializationStatus.INITIALIZATION_FAILED,
            None,
            None,
            None,
            None,
            (InitializationAttempt(0, None, None, None, None, (str(error),)),),
        )

    for construction_index in range(construction_budget):
        stats.construction_attempts += 1
        try:
            solution = _construct(
                parents,
                selected_patterns,
                config,
                insertion_limit=insertion_limit,
                fallback_order=construction_index == 1,
            )
        except (ValueError, ArithmeticError, OverflowError) as error:
            attempts.append(
                InitializationAttempt(
                    construction_index, None, None, None, None, (str(error),)
                )
            )
            continue
        if solution.canonical_hash in seen:
            attempts.append(
                InitializationAttempt(
                    construction_index,
                    solution,
                    None,
                    None,
                    None,
                    ("duplicate construction; reference evaluation skipped",),
                    True,
                )
            )
            continue
        seen.add(solution.canonical_hash)

        dp_started = time.perf_counter()
        direction = optimize_directions_with_initial_feasibility(solution, config)
        stats.direction_dp_time += time.perf_counter() - dp_started
        orientation_map = {robot: direction.directions[robot] for robot in range(4)}
        ref_started = time.perf_counter()
        try:
            schedule = reference_evaluator(
                solution,
                config,
                orientations=orientation_map,
            )
        except (ArithmeticError, OverflowError, ValueError) as error:
            schedule = ScheduleResult(
                ScheduleStatus.NUMERIC_FAILURE, diagnostics=(str(error),)
            )
        ref_duration = time.perf_counter() - ref_started
        ref_ended = time.perf_counter()
        stats.record_reference(
            schedule.status,
            ref_duration,
            initialization=True,
            reference_start=(
                None if stats.run_started is None else ref_started - stats.run_started
            ),
            reference_end=(
                None if stats.run_started is None else ref_ended - stats.run_started
            ),
        )
        certification = None
        if schedule.status is ScheduleStatus.FEASIBLE:
            cert_started = time.perf_counter()
            certification = certifier(solution, schedule, config)
            stats.certifier_time += time.perf_counter() - cert_started
            if not certification.certified:
                schedule = ScheduleResult(
                    ScheduleStatus.NUMERIC_FAILURE,
                    diagnostics=(
                        "initial FEASIBLE schedule failed certification: "
                        + "; ".join(certification.errors),
                    ),
                    directions=direction.directions,
                )
                stats.init_status_counts[ScheduleStatus.FEASIBLE.value] -= 1
                stats.init_status_counts[ScheduleStatus.NUMERIC_FAILURE.value] += 1
        attempt = InitializationAttempt(
            construction_index,
            solution,
            direction,
            schedule,
            certification,
            direction.diagnostics + schedule.diagnostics,
        )
        attempts.append(attempt)
        if schedule.status is ScheduleStatus.NUMERIC_FAILURE:
            stats.init_time += time.perf_counter() - started
            return InitializationResult(
                InitializationStatus.NUMERIC_FAILURE,
                None,
                None,
                schedule,
                certification,
                tuple(attempts),
            )
        if (
            direction.status is DirectionStatus.FEASIBLE
            and schedule.status is ScheduleStatus.FEASIBLE
            and certification is not None
            and certification.certified
        ):
            stats.init_time += time.perf_counter() - started
            return InitializationResult(
                InitializationStatus.SUCCESS,
                solution,
                direction.directions,
                schedule,
                certification,
                tuple(attempts),
            )

    stats.init_time += time.perf_counter() - started
    return InitializationResult(
        InitializationStatus.INITIALIZATION_FAILED,
        None,
        None,
        None,
        None,
        tuple(attempts),
    )
