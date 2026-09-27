from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import itertools
import math
import time
from typing import Protocol

from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CanonicalSolution,
    OfficialMetrics,
    ParentWeld,
    Route,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.scheduler import (
    build_operation_templates,
    build_robot_routes,
    reference_schedule,
)
from mrta_reference.solution import block_map, canonicalize, official_metrics

from .lower_bounds import AnalyticalLowerBound, analytical_lower_bound
from .scheduler import DirectionVectors, exact_schedule_from_templates


EXACT_Y_SCOPE_CURRENT_SEMANTICS = "EXACT_Y_SCOPE_CURRENT_SEMANTICS"


class PatternProvider(Protocol):
    scope_id: str

    def patterns_for(
        self, parent: ParentWeld, config: ScientificConfig
    ) -> tuple[SplitPattern, ...]: ...


@dataclass(frozen=True)
class YOnlyPatternProvider:
    """WHOLE plus every legal Y split, with mandatory-Y fail-closed behavior."""

    scope_id: str = EXACT_Y_SCOPE_CURRENT_SEMANTICS

    def patterns_for(
        self, parent: ParentWeld, config: ScientificConfig
    ) -> tuple[SplitPattern, ...]:
        y_patterns = generate_y_split_patterns(parent, config)
        if whole_eligible_rails(parent.start, parent.end, config):
            candidates = (SplitPattern(parent.parent_id, SplitKind.WHOLE),) + y_patterns
        else:
            candidates = tuple(pattern for pattern in y_patterns if pattern.mandatory)
        seen: set[str] = set()
        deterministic: list[SplitPattern] = []
        for pattern in candidates:
            if pattern.pattern_id not in seen:
                deterministic.append(pattern)
                seen.add(pattern.pattern_id)
        return tuple(deterministic)


class ExactSolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    NO_FEASIBLE_SOLUTION = "NO_FEASIBLE_SOLUTION"
    LIMIT_REACHED = "LIMIT_REACHED"
    NUMERIC_FAILURE = "NUMERIC_FAILURE"


@dataclass(frozen=True)
class ExactResult:
    scope_id: str
    status: ExactSolveStatus
    optimal: bool
    best_cmax: float | None
    lb0: float
    lower_bound: AnalyticalLowerBound
    best_solution: CanonicalSolution | None
    best_schedule: ScheduleResult | None
    best_metrics: OfficialMetrics | None
    certified: bool
    pattern_combinations: int
    assignment_states: int
    route_states: int
    direction_states: int
    schedule_evaluations: int
    raw_discrete_count: int
    canonical_unique_solutions: int
    canonical_duplicates: int
    duplicate_count: int
    states_explored: int
    pruned_states: int
    exact_scheduler_dispatch_states: int
    runtime_total: float
    runtime_scheduler: float
    reference_status: ScheduleStatus | None
    reference_cmax: float | None
    scheduler_only_gap: float | None
    diagnostics: tuple[str, ...] = ()


@dataclass
class _Counters:
    pattern_combinations: int = 0
    assignment_states: int = 0
    route_states: int = 0
    direction_states: int = 0
    schedule_evaluations: int = 0
    raw_discrete_count: int = 0
    canonical_unique_solutions: int = 0
    canonical_duplicates: int = 0
    states_explored: int = 0
    pruned_states: int = 0
    exact_scheduler_dispatch_states: int = 0
    runtime_scheduler: float = 0.0


class _LimitReached(RuntimeError):
    pass


def _direction_vectors(solution: CanonicalSolution):
    total = sum(len(route.block_ids) for route in solution.routes)
    for flat in itertools.product((0, 1), repeat=total):
        cursor = 0
        vectors: list[tuple[int, ...]] = []
        for route in solution.routes:
            length = len(route.block_ids)
            vectors.append(tuple(flat[cursor : cursor + length]))
            cursor += length
        yield tuple(vectors)


def _route_products(blocks_by_robot: dict[int, list[str]]):
    choices = []
    for robot in range(4):
        block_ids = tuple(blocks_by_robot[robot])
        choices.append(tuple(itertools.permutations(block_ids)) if block_ids else ((),))
    yield from itertools.product(*choices)


def _is_better(
    metrics: OfficialMetrics,
    solution: CanonicalSolution,
    directions: DirectionVectors,
    best_metrics: OfficialMetrics | None,
    best_solution: CanonicalSolution | None,
    best_directions: DirectionVectors | None,
) -> bool:
    if best_metrics is None or best_solution is None or best_directions is None:
        return True
    comparison = metrics.compare(best_metrics)
    if comparison != 0:
        return comparison < 0
    return (solution.canonical_hash, directions) < (
        best_solution.canonical_hash,
        best_directions,
    )


def solve_exact_micro(
    parents: Sequence[ParentWeld],
    config: ScientificConfig,
    *,
    pattern_provider: PatternProvider | None = None,
    exhaustive_mode: bool = False,
    max_parents: int = 4,
    allow_larger: bool = False,
    max_states: int | None = None,
    max_schedule_evaluations: int | None = None,
    time_limit: float | None = None,
) -> ExactResult:
    """Solve the complete discrete micro problem in the current Y-only scope."""
    started = time.perf_counter()
    ordered_parents = tuple(sorted(parents, key=lambda parent: parent.parent_id))
    if len({parent.parent_id for parent in ordered_parents}) != len(ordered_parents):
        raise ValueError("duplicate parent_id")
    if max_parents < 0:
        raise ValueError("max_parents must be non-negative")
    if len(ordered_parents) > max_parents and not allow_larger:
        raise ValueError(
            f"micro exact guard: {len(ordered_parents)} parents exceeds max_parents={max_parents}"
        )
    for name, value in (
        ("max_states", max_states),
        ("max_schedule_evaluations", max_schedule_evaluations),
    ):
        if value is not None and value < 0:
            raise ValueError(f"{name} must be non-negative or None")
    if time_limit is not None and time_limit < 0.0:
        raise ValueError("time_limit must be non-negative or None")

    provider = pattern_provider or YOnlyPatternProvider()
    if provider.scope_id != EXACT_Y_SCOPE_CURRENT_SEMANTICS:
        raise ValueError(
            "solve_exact_micro currently supports only EXACT_Y_SCOPE_CURRENT_SEMANTICS"
        )
    pattern_choices = tuple(
        tuple(provider.patterns_for(parent, config)) for parent in ordered_parents
    )
    for parent, choices in zip(ordered_parents, pattern_choices):
        for pattern in choices:
            if pattern.parent_id != parent.parent_id:
                raise ValueError("PatternProvider returned a parent-mismatched pattern")
            if pattern.kind is SplitKind.X_SPLIT:
                raise ValueError("X_SPLIT is excluded from the current exact scope")

    lower_bound = analytical_lower_bound(ordered_parents, config)
    counters = _Counters()
    seen_canonical: set[str] = set()
    diagnostics: list[str] = []
    best_solution: CanonicalSolution | None = None
    best_schedule: ScheduleResult | None = None
    best_metrics: OfficialMetrics | None = None
    best_directions: DirectionVectors | None = None
    best_reference: ScheduleResult | None = None
    numeric_failure = False
    limit_reached = False

    def check_time() -> None:
        if time_limit is not None and time.perf_counter() - started >= time_limit:
            raise _LimitReached("time_limit reached")

    def enter_state() -> None:
        check_time()
        if max_states is not None and counters.states_explored >= max_states:
            raise _LimitReached("max_states reached")
        counters.states_explored += 1

    def before_schedule() -> None:
        check_time()
        if (
            max_schedule_evaluations is not None
            and counters.schedule_evaluations >= max_schedule_evaluations
        ):
            raise _LimitReached("max_schedule_evaluations reached")

    try:
        if any(not choices for choices in pattern_choices):
            diagnostics.append("at least one parent has no legal pattern in the current scope")
        else:
            for patterns in itertools.product(*pattern_choices):
                enter_state()
                counters.pattern_combinations += 1
                blocks = tuple(
                    block
                    for parent, pattern in zip(ordered_parents, patterns)
                    for block in blocks_for_pattern(parent, pattern, config)
                )
                eligible_by_block = tuple(
                    tuple(
                        robot
                        for robot in range(4)
                        if robot_is_eligible(block, robot, config)
                    )
                    for block in blocks
                )
                if any(not eligible for eligible in eligible_by_block):
                    counters.pruned_states += 1
                    continue
                for assignment in itertools.product(*eligible_by_block):
                    enter_state()
                    counters.assignment_states += 1
                    blocks_by_robot = {robot: [] for robot in range(4)}
                    for block, robot in zip(blocks, assignment):
                        blocks_by_robot[robot].append(block.block_id)
                    for route_orders in _route_products(blocks_by_robot):
                        enter_state()
                        counters.route_states += 1
                        counters.raw_discrete_count += 1
                        try:
                            solution = canonicalize(
                                ordered_parents,
                                patterns,
                                tuple(
                                    Route(robot, tuple(route_orders[robot]))
                                    for robot in range(4)
                                ),
                                config,
                            )
                        except ValueError:
                            counters.pruned_states += 1
                            continue
                        if solution.canonical_hash in seen_canonical:
                            counters.canonical_duplicates += 1
                            continue
                        seen_canonical.add(solution.canonical_hash)
                        counters.canonical_unique_solutions += 1

                        if not exhaustive_mode and best_schedule is not None:
                            blocks_by_id = block_map(solution, config)
                            process_load = max(
                                sum(
                                    config.process_time(blocks_by_id[block_id].length)
                                    for block_id in route.block_ids
                                )
                                for route in solution.routes
                            )
                            assert best_schedule.cmax is not None
                            cmax_tolerance = 1.0e-9 * max(
                                1.0, abs(best_schedule.cmax)
                            )
                            if process_load > best_schedule.cmax + cmax_tolerance:
                                counters.pruned_states += 1
                                continue

                        for raw_directions in _direction_vectors(solution):
                            enter_state()
                            directions: DirectionVectors = raw_directions
                            counters.direction_states += 1
                            before_schedule()
                            orientation_map = {
                                robot: directions[robot] for robot in range(4)
                            }
                            robot_routes = build_robot_routes(
                                solution, config, orientation_map
                            )
                            if any(
                                not robot_is_eligible(block, route.robot_id, config)
                                for route in robot_routes
                                for block in route.blocks
                            ):
                                counters.pruned_states += 1
                                continue
                            templates = {
                                route.robot_id: build_operation_templates(route, config)
                                for route in robot_routes
                            }
                            scheduler_started = time.perf_counter()
                            exact = exact_schedule_from_templates(
                                templates,
                                config,
                                directions=directions,
                                branch_and_bound=not exhaustive_mode,
                            )
                            counters.runtime_scheduler += (
                                time.perf_counter() - scheduler_started
                            )
                            counters.schedule_evaluations += 1
                            counters.exact_scheduler_dispatch_states += (
                                exact.dispatch_states
                            )
                            counters.pruned_states += exact.pruned_states
                            if exact.status is ScheduleStatus.NUMERIC_FAILURE:
                                numeric_failure = True
                                diagnostics.extend(exact.diagnostics)
                                continue
                            if (
                                exact.status is not ScheduleStatus.FEASIBLE
                                or exact.schedule is None
                            ):
                                continue
                            certification = certify_schedule(
                                solution, exact.schedule, config
                            )
                            if not certification.certified:
                                numeric_failure = True
                                diagnostics.append(
                                    "exact schedule failed independent certification: "
                                    + "; ".join(certification.errors)
                                )
                                continue
                            metrics = official_metrics(
                                solution, exact.schedule, config
                            )
                            if _is_better(
                                metrics,
                                solution,
                                directions,
                                best_metrics,
                                best_solution,
                                best_directions,
                            ):
                                best_solution = solution
                                best_schedule = exact.schedule
                                best_metrics = metrics
                                best_directions = directions
                                best_reference = reference_schedule(
                                    solution,
                                    config,
                                    orientations=orientation_map,
                                )
    except _LimitReached as exception:
        limit_reached = True
        diagnostics.append(str(exception))
    except (ArithmeticError, OverflowError) as exception:
        numeric_failure = True
        diagnostics.append(str(exception))

    if limit_reached:
        status = ExactSolveStatus.LIMIT_REACHED
    elif numeric_failure:
        status = ExactSolveStatus.NUMERIC_FAILURE
    elif best_schedule is None:
        status = ExactSolveStatus.NO_FEASIBLE_SOLUTION
    else:
        status = ExactSolveStatus.OPTIMAL

    reference_status = None if best_reference is None else best_reference.status
    reference_cmax = None if best_reference is None else best_reference.cmax
    scheduler_gap = None
    if (
        best_schedule is not None
        and best_schedule.cmax is not None
        and reference_status is ScheduleStatus.FEASIBLE
        and reference_cmax is not None
    ):
        scheduler_gap = reference_cmax - best_schedule.cmax

    return ExactResult(
        scope_id=EXACT_Y_SCOPE_CURRENT_SEMANTICS,
        status=status,
        optimal=status is ExactSolveStatus.OPTIMAL,
        best_cmax=None if best_schedule is None else best_schedule.cmax,
        lb0=lower_bound.lb0,
        lower_bound=lower_bound,
        best_solution=best_solution,
        best_schedule=best_schedule,
        best_metrics=best_metrics,
        certified=best_solution is not None and best_schedule is not None,
        pattern_combinations=counters.pattern_combinations,
        assignment_states=counters.assignment_states,
        route_states=counters.route_states,
        direction_states=counters.direction_states,
        schedule_evaluations=counters.schedule_evaluations,
        raw_discrete_count=counters.raw_discrete_count,
        canonical_unique_solutions=counters.canonical_unique_solutions,
        canonical_duplicates=counters.canonical_duplicates,
        duplicate_count=counters.canonical_duplicates,
        states_explored=counters.states_explored,
        pruned_states=counters.pruned_states,
        exact_scheduler_dispatch_states=counters.exact_scheduler_dispatch_states,
        runtime_total=time.perf_counter() - started,
        runtime_scheduler=counters.runtime_scheduler,
        reference_status=reference_status,
        reference_cmax=reference_cmax,
        scheduler_only_gap=scheduler_gap,
        diagnostics=tuple(dict.fromkeys(diagnostics)),
    )
