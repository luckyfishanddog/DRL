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
) -> tuple[EvaluatedCandidate | None, int, tuple[str, ...]]:
    attempts = 1
    failures: list[str] = []
    result = evaluator.evaluate(solution, source="INITIAL_COMMON_ROUTE")
    if result.status is BaselineStatus.COMPLETED:
        return result, attempts, ()
    failures.append(result.status.value)
    for index, repaired in enumerate(_route_repair_variants(solution, config)):
        if evaluator.expired:
            break
        attempts += 1
        result = evaluator.evaluate(repaired, source=f"INITIAL_COMMON_REPAIR_{index}")
        if result.status is BaselineStatus.COMPLETED:
            return result, attempts, tuple(failures)
        failures.append(result.status.value)
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


def n1_relocate_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    loads = _loads(solution, config)
    sources = list(range(4))
    rng.shuffle(sources)
    sources.sort(key=lambda robot: -loads[robot])
    proposals: list[tuple[float, int, int, str, int]] = []
    for source in sources:
        block_ids = list(routes[source])
        rng.shuffle(block_ids)
        for block_id in block_ids:
            for destination in range(4):
                if destination == source or not robot_is_eligible(
                    blocks[block_id], destination, config
                ):
                    continue
                source_route = [item for item in routes[source] if item != block_id]
                for position in _ranked_insertions(
                    block_id, routes[destination], blocks, config
                ):
                    destination_route = list(routes[destination])
                    destination_route.insert(position, block_id)
                    cheap = max(
                        _route_cost(source_route, blocks, config),
                        _route_cost(destination_route, blocks, config),
                    )
                    proposals.append(
                        (cheap, source, destination, block_id, position)
                    )
    result = []
    for _, source, destination, block_id, position in sorted(proposals)[:cap]:
        candidate_routes = [list(route) for route in routes]
        candidate_routes[source].remove(block_id)
        candidate_routes[destination].insert(position, block_id)
        result.append(_replace_routes(solution, candidate_routes, config))
    return tuple(result)


def n2_swap_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    proposals: list[tuple[float, int, int, int, int]] = []
    pairs = [(left, right) for left in range(4) for right in range(left + 1, 4)]
    rng.shuffle(pairs)
    for left, right in pairs:
        for left_position, left_id in enumerate(routes[left]):
            for right_position, right_id in enumerate(routes[right]):
                if not robot_is_eligible(blocks[left_id], right, config):
                    continue
                if not robot_is_eligible(blocks[right_id], left, config):
                    continue
                left_route = list(routes[left])
                right_route = list(routes[right])
                left_route[left_position], right_route[right_position] = (
                    right_id,
                    left_id,
                )
                cheap = max(
                    _route_cost(left_route, blocks, config),
                    _route_cost(right_route, blocks, config),
                )
                proposals.append(
                    (cheap, left, right, left_position, right_position)
                )
    result = []
    for _, left, right, left_position, right_position in sorted(proposals)[:cap]:
        candidate_routes = [list(route) for route in routes]
        candidate_routes[left][left_position], candidate_routes[right][right_position] = (
            candidate_routes[right][right_position],
            candidate_routes[left][left_position],
        )
        result.append(_replace_routes(solution, candidate_routes, config))
    return tuple(result)


def n3_route_candidates(
    solution: CanonicalSolution,
    config: ScientificConfig,
    rng: random.Random,
    *,
    cap: int,
) -> tuple[CanonicalSolution, ...]:
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    routes = _route_lists(solution)
    proposals: list[tuple[float, int, tuple[str, ...]]] = []
    robot_order = list(range(4))
    rng.shuffle(robot_order)
    for robot in robot_order:
        route = routes[robot]
        for left in range(len(route) - 1):
            for right in range(left + 1, min(len(route), left + 6)):
                candidate = tuple(route[:left]) + tuple(
                    reversed(route[left : right + 1])
                ) + tuple(route[right + 1 :])
                proposals.append((_route_cost(candidate, blocks, config), robot, candidate))
        for source in range(len(route)):
            removed = route[source]
            remainder = route[:source] + route[source + 1 :]
            for destination in _ranked_insertions(removed, remainder, blocks, config):
                candidate_list = list(remainder)
                candidate_list.insert(destination, removed)
                candidate = tuple(candidate_list)
                if candidate != tuple(route):
                    proposals.append(
                        (_route_cost(candidate, blocks, config), robot, candidate)
                    )
    result = []
    seen = set()
    for _, robot, route in sorted(proposals):
        key = (robot, route)
        if key in seen:
            continue
        seen.add(key)
        candidate_routes = [list(items) for items in routes]
        candidate_routes[robot] = list(route)
        result.append(_replace_routes(solution, candidate_routes, config))
        if len(result) >= cap:
            break
    return tuple(result)


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
) -> VNSResult:
    """Run deterministic-seed VND and accept only strictly better certified Cmax."""
    started = time.perf_counter()
    stats = VNSStats()
    initial, attempts, init_failures = evaluate_initial_with_shared_repair(
        initial_solution, evaluator, config
    )
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
    builders = (n1_relocate_candidates, n2_swap_candidates, n3_route_candidates)
    while not evaluator.expired and neighborhood < len(NEIGHBORHOODS):
        stats.iterations += 1
        name = NEIGHBORHOODS[neighborhood]
        candidates = builders[neighborhood](
            current.solution,
            config,
            rng,
            cap=per_neighborhood_budget,
        )
        best_neighbor: EvaluatedCandidate | None = None
        for candidate in candidates:
            if evaluator.expired:
                break
            stats.attempted[name] += 1
            evaluated = evaluator.evaluate(candidate, source=name)
            if evaluated.status is not BaselineStatus.COMPLETED:
                _record_failure(stats, evaluated)
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

