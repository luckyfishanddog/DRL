"""Small fixed-split VNS evaluated only by direction DP, B32 and certification."""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Iterable, Sequence

from mrta_baselines.common import (
    BaselineStatus,
    CommonBaselineEvaluator,
    EvaluatedCandidate,
)
from mrta_baselines.wag_vns import _route_cost
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import CanonicalSolution, Route, ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize


NEIGHBORHOODS = ("N1_RELOCATE", "N2_SWAP", "N3_ROUTE")
GLOBAL_BALANCED = "GLOBAL_BALANCED"
LEGACY_LOCAL_ABSOLUTE = "LEGACY_LOCAL_ABSOLUTE"


@dataclass
class VNSStats:
    iterations: int = 0
    attempted: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in NEIGHBORHOODS}
    )
    certified: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in NEIGHBORHOODS}
    )
    improvements: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in NEIGHBORHOODS}
    )
    generated: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in NEIGHBORHOODS}
    )
    explored_types: dict[str, dict[str, int]] = field(
        default_factory=lambda: {name: {} for name in NEIGHBORHOODS}
    )
    late_certified: int = 0
    failures: dict[str, int] = field(
        default_factory=lambda: {
            BaselineStatus.DEADLOCK.value: 0,
            BaselineStatus.DIRECTION_INFEASIBLE.value: 0,
            BaselineStatus.INFEASIBLE.value: 0,
            BaselineStatus.NUMERIC_FAILURE.value: 0,
        }
    )


@dataclass(frozen=True)
class VNSResult:
    initial: EvaluatedCandidate | None
    best: EvaluatedCandidate | None
    stats: VNSStats
    initialization_attempts: int
    initialization_failures: tuple[str, ...]
    elapsed: float


@dataclass(frozen=True)
class _CandidateBatch:
    candidates: tuple[CanonicalSolution, ...]
    move_types: tuple[str, ...]
    generated_count: int


def _replace_routes(
    solution: CanonicalSolution,
    routes: Sequence[Sequence[str]],
    config: ScientificConfig,
) -> CanonicalSolution:
    return canonicalize(
        solution.parents,
        solution.patterns,
        tuple(Route(robot, tuple(routes[robot])) for robot in range(4)),
        config,
        revision=solution.revision + 1,
        scope=FORMAL_SCOPE_V2,
    )


def _route_lists(solution: CanonicalSolution) -> list[list[str]]:
    return [list(route.block_ids) for route in solution.routes]


def _route_repair_variants(
    solution: CanonicalSolution, config: ScientificConfig
) -> Iterable[CanonicalSolution]:
    """A shared, bounded spatial-boundary/order repair for a failed initial schedule."""
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    base = _route_lists(solution)
    seen = {tuple(tuple(route) for route in base)}
    candidates: list[list[list[str]]] = []
    # Same-rail spatial boundary shifts are the smallest assignment repair that
    # addresses a common B32 deadlock without changing split patterns.  The exact
    # offset order and budget are identical for RULE and GNN.
    for left_robot, right_robot in ((0, 1), (2, 3)):
        combined = sorted(
            base[left_robot] + base[right_robot],
            key=lambda block_id: (
                (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0,
                block_id,
            ),
        )
        boundary = len(base[left_robot])
        for offset in (1, -1, 2, -2):
            shifted = min(len(combined), max(0, boundary + offset))
            routes = [list(route) for route in base]
            routes[left_robot] = combined[:shifted]
            routes[right_robot] = combined[shifted:]
            candidates.append(routes)
    for robot in range(4):
        routes = [list(route) for route in base]
        routes[robot].reverse()
        candidates.append(routes)
    spatial = [list(route) for route in base]
    for robot in range(4):
        spatial[robot].sort(
            key=lambda block_id: (
                (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0,
                block_id,
            ),
        )
    candidates.append(spatial)
    for routes in candidates:
        signature = tuple(tuple(route) for route in routes)
        if signature in seen:
            continue
        seen.add(signature)
        yield _replace_routes(solution, routes, config)


def evaluate_initial_with_shared_repair(
    solution: CanonicalSolution,
    evaluator: CommonBaselineEvaluator,
    config: ScientificConfig,
    *,
    accept_completed_by: float | None = None,
) -> tuple[EvaluatedCandidate | None, int, tuple[str, ...]]:
    attempts = 1
    failures: list[str] = []
    result = evaluator.evaluate(solution, source="INITIAL_COMMON_ROUTE")
    if result.status is BaselineStatus.COMPLETED and (
        accept_completed_by is None or result.completed_at <= accept_completed_by
    ):
        return result, attempts, ()
    failures.append(
        "LATE_CERTIFIED"
        if result.status is BaselineStatus.COMPLETED
        else result.status.value
    )
    for index, repaired in enumerate(_route_repair_variants(solution, config)):
        if evaluator.expired:
            break
        attempts += 1
        result = evaluator.evaluate(repaired, source=f"INITIAL_COMMON_REPAIR_{index}")
        if result.status is BaselineStatus.COMPLETED and (
            accept_completed_by is None or result.completed_at <= accept_completed_by
        ):
            return result, attempts, tuple(failures)
        failures.append(
            "LATE_CERTIFIED"
            if result.status is BaselineStatus.COMPLETED
            else result.status.value
        )
    return None, attempts, tuple(failures)


def _loads(solution: CanonicalSolution, config: ScientificConfig) -> list[float]:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    return [
        sum(config.process_time(blocks[item].length) for item in route.block_ids)
        for route in solution.routes
    ]


def _ranked_insertions(
    block_id: str,
    route: Sequence[str],
    blocks,
    config: ScientificConfig,
    limit: int = 2,
) -> tuple[int, ...]:
    ranked = []
    for position in range(len(route) + 1):
        candidate = tuple(route[:position]) + (block_id,) + tuple(route[position:])
        ranked.append((_route_cost(candidate, blocks, config), position))
    return tuple(position for _, position in sorted(ranked)[:limit])


@dataclass(frozen=True)
class _Proposal:
    key: tuple[object, ...]
    group: str
    move_type: str
    routes: tuple[tuple[str, ...], ...]


def _route_signature(routes: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(route) for route in routes)


def _processing_loads_from_routes(routes, blocks, config: ScientificConfig) -> tuple[float, ...]:
    return tuple(
        sum(config.process_time(blocks[block_id].length) for block_id in route)
        for route in routes
    )


def _route_costs_from_routes(routes, blocks, config: ScientificConfig) -> tuple[float, ...]:
    return tuple(_route_cost(route, blocks, config) for route in routes)


def _same_rail_inversions(routes, blocks) -> int:
    """Cheap coordination proxy; it is never treated as a feasibility rule."""
    total = 0
    for left, right in ((0, 1), (2, 3)):
        left_x = [
            (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0
            for block_id in routes[left]
        ]
        right_x = [
            (blocks[block_id].start[0] + blocks[block_id].end[0]) / 2.0
            for block_id in routes[right]
        ]
        total += sum(left_value > right_value for left_value in left_x for right_value in right_x)
    return total


def _select_proposals(
    proposals: Sequence[_Proposal],
    cap: int,
    *,
    diverse: bool,
) -> tuple[_Proposal, ...]:
    """Select strong global candidates while reserving four slots for move diversity."""
    if cap <= 0:
        return ()
    unique: dict[tuple[tuple[str, ...], ...], _Proposal] = {}
    for proposal in proposals:
        previous = unique.get(proposal.routes)
        if previous is None or proposal.key < previous.key:
            unique[proposal.routes] = proposal
    ranked = sorted(unique.values(), key=lambda proposal: proposal.key)
    if not diverse or len(ranked) <= cap:
        return tuple(ranked[:cap])

    reserve = min(4, max(1, cap // 3))
    selected = list(ranked[: max(0, cap - reserve)])
    selected_signatures = {proposal.routes for proposal in selected}
    represented_groups = {proposal.group for proposal in selected}

    group_best: dict[str, _Proposal] = {}
    for proposal in ranked:
        group_best.setdefault(proposal.group, proposal)
    for proposal in sorted(group_best.values(), key=lambda item: item.key):
        if len(selected) >= cap:
            break
        if proposal.group in represented_groups or proposal.routes in selected_signatures:
            continue
        selected.append(proposal)
        selected_signatures.add(proposal.routes)
        represented_groups.add(proposal.group)
    for proposal in ranked:
        if len(selected) >= cap:
            break
        if proposal.routes not in selected_signatures:
            selected.append(proposal)
            selected_signatures.add(proposal.routes)
    return tuple(selected)


def _materialize_batch(
    solution: CanonicalSolution,
    config: ScientificConfig,
    proposals: Sequence[_Proposal],
    cap: int,
    *,
    diverse: bool,
) -> _CandidateBatch:
    selected = _select_proposals(proposals, cap, diverse=diverse)
    return _CandidateBatch(
        tuple(_replace_routes(solution, proposal.routes, config) for proposal in selected),
        tuple(proposal.move_type for proposal in selected),
        len({proposal.routes for proposal in proposals}),
    )


def _n1_relocate_batch(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str,
) -> _CandidateBatch:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    loads = _processing_loads_from_routes(routes, blocks, config)
    route_costs = _route_costs_from_routes(routes, blocks, config)
    critical = {max(range(4), key=loads.__getitem__), max(range(4), key=route_costs.__getitem__)}
    sources = list(range(4))
    rng.shuffle(sources)
    sources.sort(key=lambda robot: -loads[robot])
    proposals: list[_Proposal] = []
    for source in sources:
        block_ids = list(routes[source])
        rng.shuffle(block_ids)
        for block_id in block_ids:
            for destination in range(4):
                if destination == source or not robot_is_eligible(
                    blocks[block_id], destination, config
                ):
                    continue
                positions: Iterable[int]
                if selection_policy == LEGACY_LOCAL_ABSOLUTE:
                    positions = _ranked_insertions(
                        block_id, routes[destination], blocks, config
                    )
                else:
                    positions = range(len(routes[destination]) + 1)
                for position in positions:
                    candidate_routes = [list(route) for route in routes]
                    candidate_routes[source].remove(block_id)
                    candidate_routes[destination].insert(position, block_id)
                    candidate_costs = list(route_costs)
                    candidate_costs[source] = _route_cost(
                        candidate_routes[source], blocks, config
                    )
                    candidate_costs[destination] = _route_cost(
                        candidate_routes[destination], blocks, config
                    )
                    if selection_policy == LEGACY_LOCAL_ABSOLUTE:
                        key = (
                            max(candidate_costs[source], candidate_costs[destination]),
                            source,
                            destination,
                            block_id,
                            position,
                        )
                    else:
                        candidate_loads = list(loads)
                        duration = config.process_time(blocks[block_id].length)
                        candidate_loads[source] -= duration
                        candidate_loads[destination] += duration
                        key = (
                            max(candidate_loads),
                            max(candidate_costs),
                            max(candidate_loads) - max(loads),
                            max(candidate_costs) - max(route_costs),
                            0 if source in critical else 1,
                            _same_rail_inversions(candidate_routes, blocks),
                            sum(candidate_costs) - sum(route_costs),
                            source,
                            destination,
                            block_id,
                            position,
                        )
                    proposals.append(
                        _Proposal(
                            key,
                            f"R{source}->R{destination}",
                            f"R{source}->R{destination}",
                            _route_signature(candidate_routes),
                        )
                    )
    return _materialize_batch(
        solution,
        config,
        proposals,
        cap,
        diverse=selection_policy != LEGACY_LOCAL_ABSOLUTE,
    )


def n1_relocate_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str = GLOBAL_BALANCED,
) -> tuple[CanonicalSolution, ...]:
    return _n1_relocate_batch(
        solution,
        config,
        rng,
        cap=cap,
        selection_policy=selection_policy,
    ).candidates


def _n2_swap_batch(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str,
) -> _CandidateBatch:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    loads = _processing_loads_from_routes(routes, blocks, config)
    route_costs = _route_costs_from_routes(routes, blocks, config)
    critical = {max(range(4), key=loads.__getitem__), max(range(4), key=route_costs.__getitem__)}
    proposals: list[_Proposal] = []
    pairs = [(left, right) for left in range(4) for right in range(left + 1, 4)]
    rng.shuffle(pairs)
    for left, right in pairs:
        for left_position, left_id in enumerate(routes[left]):
            for right_position, right_id in enumerate(routes[right]):
                if not robot_is_eligible(blocks[left_id], right, config):
                    continue
                if not robot_is_eligible(blocks[right_id], left, config):
                    continue
                candidate_routes = [list(route) for route in routes]
                candidate_routes[left][left_position], candidate_routes[right][right_position] = (
                    right_id,
                    left_id,
                )
                candidate_costs = list(route_costs)
                candidate_costs[left] = _route_cost(candidate_routes[left], blocks, config)
                candidate_costs[right] = _route_cost(candidate_routes[right], blocks, config)
                if selection_policy == LEGACY_LOCAL_ABSOLUTE:
                    key = (
                        max(candidate_costs[left], candidate_costs[right]),
                        left,
                        right,
                        left_position,
                        right_position,
                    )
                else:
                    candidate_loads = list(loads)
                    left_duration = config.process_time(blocks[left_id].length)
                    right_duration = config.process_time(blocks[right_id].length)
                    candidate_loads[left] += right_duration - left_duration
                    candidate_loads[right] += left_duration - right_duration
                    key = (
                        max(candidate_loads),
                        max(candidate_costs),
                        max(candidate_loads) - max(loads),
                        max(candidate_costs) - max(route_costs),
                        0 if left in critical or right in critical else 1,
                        _same_rail_inversions(candidate_routes, blocks),
                        sum(candidate_costs) - sum(route_costs),
                        left,
                        right,
                        left_position,
                        right_position,
                    )
                proposals.append(
                    _Proposal(
                        key,
                        f"R{left}<->R{right}",
                        f"R{left}<->R{right}",
                        _route_signature(candidate_routes),
                    )
                )
    return _materialize_batch(
        solution,
        config,
        proposals,
        cap,
        diverse=selection_policy != LEGACY_LOCAL_ABSOLUTE,
    )


def n2_swap_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str = GLOBAL_BALANCED,
) -> tuple[CanonicalSolution, ...]:
    return _n2_swap_batch(
        solution,
        config,
        rng,
        cap=cap,
        selection_policy=selection_policy,
    ).candidates


def _n3_route_batch(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str,
) -> _CandidateBatch:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    route_costs = _route_costs_from_routes(routes, blocks, config)
    critical = {max(range(4), key=route_costs.__getitem__)}
    proposals: list[_Proposal] = []
    robot_order = list(range(4))
    rng.shuffle(robot_order)

    def append_proposal(robot: int, candidate: tuple[str, ...], operation: str) -> None:
        if candidate == tuple(routes[robot]):
            return
        candidate_routes = [list(route) for route in routes]
        candidate_routes[robot] = list(candidate)
        new_cost = _route_cost(candidate, blocks, config)
        candidate_costs = list(route_costs)
        candidate_costs[robot] = new_cost
        if selection_policy == LEGACY_LOCAL_ABSOLUTE:
            key = (new_cost, robot, candidate)
        else:
            key = (
                max(candidate_costs),
                new_cost - route_costs[robot],
                0 if robot in critical else 1,
                _same_rail_inversions(candidate_routes, blocks),
                sum(candidate_costs) - sum(route_costs),
                robot,
                operation,
                candidate,
            )
        proposals.append(
            _Proposal(
                key,
                f"R{robot}:{operation}",
                f"R{robot}:{operation}",
                _route_signature(candidate_routes),
            )
        )

    for robot in robot_order:
        route = routes[robot]
        for left in range(len(route) - 1):
            for right in range(left + 1, min(len(route), left + 6)):
                candidate = tuple(route[:left]) + tuple(
                    reversed(route[left : right + 1])
                ) + tuple(route[right + 1 :])
                append_proposal(robot, candidate, "2OPT")
        for source in range(len(route)):
            removed = route[source]
            remainder = route[:source] + route[source + 1 :]
            if selection_policy == LEGACY_LOCAL_ABSOLUTE:
                destinations: Iterable[int] = _ranked_insertions(
                    removed, remainder, blocks, config
                )
            else:
                destinations = range(len(remainder) + 1)
            for destination in destinations:
                candidate_list = list(remainder)
                candidate_list.insert(destination, removed)
                append_proposal(robot, tuple(candidate_list), "REINSERT")
    return _materialize_batch(
        solution,
        config,
        proposals,
        cap,
        diverse=selection_policy != LEGACY_LOCAL_ABSOLUTE,
    )


def n3_route_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
    selection_policy: str = GLOBAL_BALANCED,
) -> tuple[CanonicalSolution, ...]:
    return _n3_route_batch(
        solution,
        config,
        rng,
        cap=cap,
        selection_policy=selection_policy,
    ).candidates


def _record_failure(stats: VNSStats, result: EvaluatedCandidate) -> None:
    key = result.status.value
    if key in stats.failures:
        stats.failures[key] += 1


def run_vns(
    initial_solution: CanonicalSolution,
    evaluator: CommonBaselineEvaluator,
    config: ScientificConfig,
    *,
    seed: int,
    per_neighborhood_budget: int = 12,
    selection_policy: str = GLOBAL_BALANCED,
    certified_initial: EvaluatedCandidate | None = None,
    accept_completed_by: float | None = None,
) -> VNSResult:
    """Run deterministic-seed VND and accept only strictly better certified Cmax."""
    if selection_policy not in (GLOBAL_BALANCED, LEGACY_LOCAL_ABSOLUTE):
        raise ValueError(f"unknown VNS selection policy: {selection_policy}")
    started = time.perf_counter()
    stats = VNSStats()
    if certified_initial is None:
        initial, attempts, init_failures = evaluate_initial_with_shared_repair(
            initial_solution,
            evaluator,
            config,
            accept_completed_by=accept_completed_by,
        )
    else:
        initial = evaluator.inject_certified(
            certified_initial, source="PAIRED_CERTIFIED_INITIAL"
        )
        attempts = 0
        init_failures = ()
    if initial is None:
        return VNSResult(
            None,
            None,
            stats,
            attempts,
            init_failures,
            time.perf_counter() - started,
        )
    current = initial
    rng = random.Random(seed)
    neighborhood = 0
    builders = (_n1_relocate_batch, _n2_swap_batch, _n3_route_batch)
    while not evaluator.expired and neighborhood < len(NEIGHBORHOODS):
        stats.iterations += 1
        name = NEIGHBORHOODS[neighborhood]
        batch = builders[neighborhood](
            current.solution,
            config,
            rng,
            cap=per_neighborhood_budget,
            selection_policy=selection_policy,
        )
        stats.generated[name] += batch.generated_count
        best_neighbor: EvaluatedCandidate | None = None
        for candidate, move_type in zip(batch.candidates, batch.move_types):
            if evaluator.expired:
                break
            stats.attempted[name] += 1
            type_counts = stats.explored_types[name]
            type_counts[move_type] = type_counts.get(move_type, 0) + 1
            evaluated = evaluator.evaluate(candidate, source=name)
            if evaluated.status is not BaselineStatus.COMPLETED:
                _record_failure(stats, evaluated)
                continue
            if (
                accept_completed_by is not None
                and evaluated.completed_at > accept_completed_by
            ):
                stats.late_certified += 1
                continue
            stats.certified[name] += 1
            assert evaluated.metrics is not None and current.metrics is not None
            epsilon = 1.0e-9 * max(
                1.0, abs(evaluated.metrics.cmax), abs(current.metrics.cmax)
            )
            if evaluated.metrics.cmax < current.metrics.cmax - epsilon and (
                best_neighbor is None
                or evaluated.metrics.cmax < best_neighbor.metrics.cmax  # type: ignore[union-attr]
            ):
                best_neighbor = evaluated
        if best_neighbor is not None:
            current = best_neighbor
            stats.improvements[name] += 1
            neighborhood = 0
        else:
            neighborhood += 1
    return VNSResult(
        initial,
        current,
        stats,
        attempts,
        init_failures,
        time.perf_counter() - started,
    )

