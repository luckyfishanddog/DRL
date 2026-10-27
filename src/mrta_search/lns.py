from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
import random
from typing import Mapping, Sequence

from mrta_reference.geometry import (
    finite_x_split_validator,
    oriented_endpoints,
    robot_is_eligible,
)
from mrta_reference.model import (
    CandidateMove,
    CanonicalSolution,
    OperationKind,
    ParentWeld,
    Route,
    ScheduleResult,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    WeldingBlock,
)
from mrta_reference.solution import block_map, canonicalize

from .direction import DirectionVectors, _initial_error
from .initialization import bounded_insertion_positions


class CandidateSourceKind(str, Enum):
    ATOMIC = "ATOMIC"
    LNS_REPAIRED = "LNS_REPAIRED"


class DestroyOperator(str, Enum):
    RANDOM_REMOVAL = "RANDOM_REMOVAL"
    CRITICAL_LOAD_REMOVAL = "CRITICAL_LOAD_REMOVAL"


class RepairOperator(str, Enum):
    GREEDY_REPAIR = "GREEDY_REPAIR"
    REGRET_2_REPAIR = "REGRET_2_REPAIR"


class BottleneckSelection(str, Enum):
    """Offline pilot selectors; deliberately outside production OPERATOR_PAIRS."""

    RANDOM = "RANDOM"
    LOAD = "LOAD"
    TRAVEL = "TRAVEL"
    WAIT_ANCHORED = "WAIT_ANCHORED"


class CRRRBottleneck(str, Enum):
    WAIT_BOUNDARY = "WAIT_BOUNDARY"
    LOAD_BOUNDARY = "LOAD_BOUNDARY"


CRRR_OPERATOR = "CRRR"


OPERATOR_PAIRS = tuple(
    (destroy, repair)
    for destroy in DestroyOperator
    for repair in RepairOperator
)


@dataclass(frozen=True, order=True)
class CompleteCandidateIdentity:
    current_revision: int
    source_kind: str
    operator_identity: tuple[object, ...]


@dataclass(frozen=True)
class CompleteSearchCandidate:
    source_kind: CandidateSourceKind
    canonical_solution: CanonicalSolution
    candidate_identity: CompleteCandidateIdentity
    projected_process_makespan: float
    local_directed_travel_delta: float
    split_count_delta: int
    load_spread_delta: float
    provenance: tuple[tuple[str, object], ...]
    atomic_move: CandidateMove | None = None

    @property
    def cheap_score(self) -> tuple[object, ...]:
        return (
            self.projected_process_makespan,
            self.local_directed_travel_delta,
            self.split_count_delta,
            self.candidate_identity,
        )

    @property
    def solution(self) -> CanonicalSolution:
        return self.canonical_solution


@dataclass(frozen=True)
class PartialSearchState:
    parents: tuple[ParentWeld, ...]
    patterns: tuple[SplitPattern, ...]
    routes: tuple[tuple[str, ...], ...]
    removed_parent_ids: tuple[str, ...]
    source_revision: int


@dataclass(frozen=True)
class RepairResult:
    candidate: CompleteSearchCandidate | None
    insertion_evaluations: int
    rejection_reason: str | None = None
    directions: DirectionVectors | None = None
    actions: tuple[OrientedInsertionAction, ...] = ()


@dataclass(frozen=True)
class OrientedInsertionAction:
    block_id: str
    robot: int
    position: int
    direction: int
    neighbour_changes: tuple[tuple[str, int, int], ...] = ()


@dataclass(frozen=True)
class OrientedRouteState:
    routes: tuple[tuple[str, ...], ...]
    directions: tuple[tuple[int, ...], ...]
    loads: tuple[float, ...]
    arc_costs: tuple[tuple[float, ...], ...]

    @property
    def travel(self) -> float:
        return sum(sum(arcs) for arcs in self.arc_costs)


@dataclass(frozen=True)
class CRRRRegion:
    bottleneck_type: CRRRBottleneck
    affected_parent_ids: tuple[str, ...]
    critical_robot: int
    partner_robot: int
    anchor_x: float
    wait_duration: float


@dataclass(frozen=True)
class _InsertionAlternative:
    parent_id: str
    routes: tuple[tuple[str, ...], ...]
    hints: tuple[tuple[int, ...], ...]
    score: tuple[object, ...]
    trace: tuple[tuple[str, int, int, int], ...]
    actions: tuple[OrientedInsertionAction, ...] = ()
    oriented_state: OrientedRouteState | None = None


def _parent_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def _optional_split_count(solution: CanonicalSolution, config: ScientificConfig) -> int:
    from mrta_reference.geometry import whole_eligible_rails

    parents = {parent.parent_id: parent for parent in solution.parents}
    return sum(
        pattern.kind is not SplitKind.WHOLE
        and bool(
            whole_eligible_rails(
                parents[pattern.parent_id].start,
                parents[pattern.parent_id].end,
                config,
            )
        )
        for pattern in solution.patterns
    )


def _directed_proxy(
    routes: Sequence[Sequence[str]],
    hints: Sequence[Sequence[int]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> float:
    total = 0.0
    for robot, route in enumerate(routes):
        for index in range(len(route) - 1):
            _, end = oriented_endpoints(blocks[route[index]], hints[robot][index])
            start, _ = oriented_endpoints(
                blocks[route[index + 1]], hints[robot][index + 1]
            )
            total += math.dist(end, start) / config.empty_speed
    return total


def _loads(
    routes: Sequence[Sequence[str]],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, float, float, float]:
    return tuple(
        sum(config.process_time(blocks[item].length) for item in route)
        for route in routes
    )  # type: ignore[return-value]


def _route_arc_costs(
    route: Sequence[str],
    hints: Sequence[int],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, ...]:
    """Cache the directed costs of one route without changing proxy semantics."""
    costs = []
    for index in range(len(route) - 1):
        _, end = oriented_endpoints(blocks[route[index]], hints[index])
        start, _ = oriented_endpoints(blocks[route[index + 1]], hints[index + 1])
        costs.append(math.dist(end, start) / config.empty_speed)
    return tuple(costs)


def _proxy_from_arc_costs(costs: Sequence[Sequence[float]]) -> float:
    # Keep the exact robot/arc accumulation order used by _directed_proxy.  A
    # sum-of-route-totals would be faster but can perturb floating-point ties.
    total = 0.0
    for route_costs in costs:
        for value in route_costs:
            total += value
    return total


def _insert_route_arc_costs(
    route: Sequence[str],
    hints: Sequence[int],
    costs: Sequence[float],
    block: WeldingBlock,
    position: int,
    hint: int,
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
) -> tuple[float, ...]:
    """Update only the one or two arcs touched by an insertion."""
    if not route:
        return ()
    before = tuple(costs[: max(0, position - 1)])
    after = tuple(costs[position:])
    inserted: list[float] = []
    if position:
        _, end = oriented_endpoints(blocks[route[position - 1]], hints[position - 1])
        start, _ = oriented_endpoints(block, hint)
        inserted.append(math.dist(end, start) / config.empty_speed)
    if position < len(route):
        _, end = oriented_endpoints(block, hint)
        start, _ = oriented_endpoints(blocks[route[position]], hints[position])
        inserted.append(math.dist(end, start) / config.empty_speed)
    return before + tuple(inserted) + after


def _insert_tuple(values: tuple[object, ...], position: int, value: object) -> tuple[object, ...]:
    return values[:position] + (value,) + values[position:]


def _hints_from_current(
    solution: CanonicalSolution, directions: DirectionVectors
) -> dict[str, int]:
    return {
        block_id: directions[robot][position]
        for robot, route in enumerate(solution.routes)
        for position, block_id in enumerate(route.block_ids)
        if position < len(directions[robot])
    }


def atomic_complete_candidate(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    screened,
    config: ScientificConfig,
) -> CompleteSearchCandidate:
    current_blocks = block_map(current, config)
    provisional_blocks = block_map(screened.solution, config)
    current_routes = tuple(route.block_ids for route in current.routes)
    provisional_routes = tuple(route.block_ids for route in screened.solution.routes)
    hints_by_block = _hints_from_current(current, current_directions)
    current_hints = tuple(
        tuple(hints_by_block.get(item, 0) for item in route) for route in current_routes
    )
    provisional_hints = tuple(
        tuple(hints_by_block.get(item, 0) for item in route)
        for route in provisional_routes
    )
    current_loads = _loads(current_routes, current_blocks, config)
    new_loads = _loads(provisional_routes, provisional_blocks, config)
    from .neighborhood import candidate_key_order

    identity = CompleteCandidateIdentity(
        current.revision,
        CandidateSourceKind.ATOMIC.value,
        candidate_key_order(screened.candidate.key),
    )
    return CompleteSearchCandidate(
        CandidateSourceKind.ATOMIC,
        screened.solution,
        identity,
        max(new_loads),
        _directed_proxy(provisional_routes, provisional_hints, provisional_blocks, config)
        - _directed_proxy(current_routes, current_hints, current_blocks, config),
        _optional_split_count(screened.solution, config)
        - _optional_split_count(current, config),
        (max(new_loads) - min(new_loads))
        - (max(current_loads) - min(current_loads)),
        (("move_type", screened.candidate.key.move_type.value),),
        screened.candidate,
    )


def destroy_size(
    parent_count: int, *, rho: float, q_min: int, q_max: int
) -> int:
    if parent_count <= 0:
        return 0
    if rho < 0.0 or q_min < 1 or q_max < q_min:
        raise ValueError("invalid destroy-size configuration")
    q = max(q_min, min(q_max, round(rho * parent_count)))
    if parent_count > 1:
        return min(q, parent_count - 1)
    return 1


def _critical_parent_order(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
) -> tuple[str, ...]:
    blocks = block_map(current, config)
    parent_robots: dict[str, set[int]] = {
        parent.parent_id: set() for parent in current.parents
    }
    for robot, route in enumerate(current.routes):
        for block_id in route.block_ids:
            parent_robots[_parent_id(block_id)].add(robot)
    loads = _loads(tuple(route.block_ids for route in current.routes), blocks, config)
    waits = [0.0, 0.0, 0.0, 0.0]
    for operation in schedule.operations:
        if operation.kind is OperationKind.WAIT:
            waits[operation.robot_id] += operation.duration
    completions = schedule.robot_completion
    critical_robot = max(range(4), key=lambda robot: (completions[robot], -robot))

    def score(parent_id: str) -> tuple[object, ...]:
        robots = parent_robots[parent_id]
        return (
            -int(critical_robot in robots),
            -max((completions[robot] for robot in robots), default=0.0),
            -max((loads[robot] for robot in robots), default=0.0),
            -max((waits[robot] for robot in robots), default=0.0),
            parent_id,
        )

    return tuple(sorted(parent_robots, key=score))


def destroy_parents(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    operator: DestroyOperator,
    *,
    q: int,
    seed: int,
    config: ScientificConfig,
) -> PartialSearchState:
    parent_ids = tuple(parent.parent_id for parent in current.parents)
    if not (1 <= q <= len(parent_ids)):
        raise ValueError("q must select one or more existing parents")
    if operator is DestroyOperator.RANDOM_REMOVAL:
        removed = tuple(sorted(random.Random(seed).sample(parent_ids, q)))
    else:
        removed = tuple(sorted(_critical_parent_order(current, schedule, config)[:q]))
    return _remove_parent_set(current, removed)


def _remove_parent_set(
    current: CanonicalSolution, removed: Sequence[str]
) -> PartialSearchState:
    removed = tuple(sorted(set(removed)))
    removed_set = set(removed)
    routes = tuple(
        tuple(item for item in route.block_ids if _parent_id(item) not in removed_set)
        for route in current.routes
    )
    return PartialSearchState(
        current.parents,
        current.patterns,
        routes,
        removed,
        current.revision,
    )


def select_bottleneck_parents(
    current: CanonicalSolution,
    directions: DirectionVectors,
    schedule: ScheduleResult,
    selection: BottleneckSelection,
    *,
    q: int,
    seed: int,
    config: ScientificConfig,
) -> PartialSearchState | None:
    """Select closed parent groups using only the current certified timeline.

    WAIT_ANCHORED uses observed waits and temporal/spatial neighbours, not
    inferred causal blockers. None means no positive WAIT exists. Directions
    are required so TRAVEL uses the actual exit/entry endpoints.
    """
    if not schedule.feasible:
        raise ValueError("bottleneck selection requires a feasible current schedule")
    if not 1 <= q <= len(current.parents):
        raise ValueError("q must select one or more existing parents")
    if selection is BottleneckSelection.RANDOM:
        return destroy_parents(current, schedule, DestroyOperator.RANDOM_REMOVAL,
                               q=q, seed=seed, config=config)
    rng = random.Random(seed)
    blocks = block_map(current, config)
    patterns = {p.parent_id: p for p in current.patterns}
    owners = {item: r for r, route in enumerate(current.routes) for item in route.block_ids}
    parent_blocks = {p.parent_id: [] for p in current.parents}
    for block in blocks.values():
        parent_blocks[block.parent_id].append(block)
    legal = {}
    for item, block in blocks.items():
        pattern = patterns[block.parent_id]
        if pattern.kind is SplitKind.X_SPLIT:
            base = 0 if pattern.rail.value == "UPPER" else 2
            legal[item] = {base + int(item.endswith("::1"))}
        else:
            legal[item] = {r for r in range(4) if robot_is_eligible(block, r, config)}
    movable = {p: any(legal[b.block_id] - {owners[b.block_id]} for b in bs)
               for p, bs in parent_blocks.items()}
    loads = _loads(tuple(r.block_ids for r in current.routes), blocks, config)
    completion = schedule.robot_completion
    max_completion = max(max(completion), config.numeric_epsilon)

    def choose(rows):
        values, weights = zip(*rows)
        return rng.choices(values, weights=weights, k=1)[0]

    anchors: list[str] = []
    wait = None
    if selection is BottleneckSelection.LOAD:
        # Both completion and processing matter. All legal robot moves count,
        # including left/right moves on the same rail.
        robot = choose([(r, max(loads[r], 1e-9) * (completion[r] / max_completion) ** 2)
                        for r in range(4) if current.routes[r].block_ids])
        candidates = sorted({_parent_id(b) for b in current.routes[robot].block_ids})
        flexible = [p for p in candidates if movable[p]]
        anchors = [choose([(p, sum(config.process_time(b.length)
                                  for b in parent_blocks[p] if owners[b.block_id] == robot))
                           for p in (flexible or candidates)])]
    elif selection is BottleneckSelection.TRAVEL:
        arcs = []
        for r, route in enumerate(current.routes):
            for i, cost in enumerate(_route_arc_costs(route.block_ids, directions[r], blocks, config)):
                arcs.append(((route.block_ids[i], route.block_ids[i + 1]),
                             max(cost, 1e-9) * (0.25 + completion[r] / max_completion)))
        if arcs:
            anchors = list(dict.fromkeys(_parent_id(b) for b in choose(arcs)))
        else:
            anchors = [rng.choice(current.parents).parent_id]
    elif selection is BottleneckSelection.WAIT_ANCHORED:
        waits = [op for op in schedule.operations
                 if op.kind is OperationKind.WAIT and op.duration > config.numeric_epsilon]
        if not waits:
            return None
        wait = choose([(op, op.duration * (0.25 + completion[op.robot_id] / max_completion) ** 2)
                       for op in waits])
        own = [op for op in schedule.operations if op.robot_id == wait.robot_id
               and op.block_id in blocks and op.kind is not OperationKind.WAIT]
        before = [op for op in own if op.end_time <= wait.start_time + config.numeric_epsilon]
        after = [op for op in own if op.start_time >= wait.end_time - config.numeric_epsilon]
        for op in ([max(before, key=lambda op: op.end_time)] if before else []) + (
            [min(after, key=lambda op: op.start_time)] if after else []
        ):
            if _parent_id(op.block_id) not in anchors:
                anchors.append(_parent_id(op.block_id))
        if not anchors and wait.block_id in blocks:
            anchors.append(_parent_id(wait.block_id))
    else:
        raise ValueError(f"unknown bottleneck selection: {selection}")

    anchor_blocks = [b for p in anchors for b in parent_blocks[p]]
    receiver_robots = set().union(*(legal[b.block_id] for b in anchor_blocks))
    adjacent = set()
    for route in current.routes:
        for i, item in enumerate(route.block_ids):
            if _parent_id(item) in anchors:
                adjacent.update(_parent_id(b) for b in route.block_ids[max(0, i - 1):i + 2])
    wait_related: dict[str, float] = {}
    if wait is not None:
        for op in schedule.operations:
            if op.robot_id == wait.robot_id or op.block_id not in blocks:
                continue
            overlap = max(0.0, min(op.end_time, wait.end_time) - max(op.start_time, wait.start_time))
            if overlap:
                distance = min(math.dist(wait.start, op.start), math.dist(wait.start, op.end))
                p = _parent_id(op.block_id)
                wait_related[p] = max(wait_related.get(p, 0.0), overlap / wait.duration / (0.5 + distance))
    related = []
    for p, bs in parent_blocks.items():
        if p in anchors:
            continue
        distance = min((math.dist(x, y) for b in bs for a in anchor_blocks
                        for x in (b.start, b.end) for y in (a.start, a.end)), default=20.0)
        receiver = any(owners[b.block_id] in receiver_robots for b in bs)
        weight = (0.1 + (1.0 + movable[p] + receiver) / (0.5 + distance)
                  + 2.0 * (p in adjacent) + 8.0 * wait_related.get(p, 0.0))
        related.append((p, weight))
    # Weighted sampling without replacement gives different anchors/context
    # across attempts, rather than repeatedly taking a deterministic top-q.
    ranked = sorted(related, key=lambda row: -math.log(max(rng.random(), 1e-15)) / row[1])
    removed = anchors[:q] + [p for p, _ in ranked[:max(0, q - len(anchors))]]
    return _remove_parent_set(current, removed)


def _insert_one(
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    block: WeldingBlock,
    robot: int,
    position: int,
    hint: int,
) -> tuple[tuple[tuple[str, ...], ...], tuple[tuple[int, ...], ...]]:
    new_routes = [list(route) for route in routes]
    new_hints = [list(vector) for vector in hints]
    new_routes[robot].insert(position, block.block_id)
    new_hints[robot].insert(position, hint)
    return (
        tuple(tuple(route) for route in new_routes),
        tuple(tuple(vector) for vector in new_hints),
    )


def _parent_alternatives(
    parent_id: str,
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    blocks: Mapping[str, WeldingBlock],
    pattern: SplitPattern,
    config: ScientificConfig,
    *,
    insertion_limit: int,
    remaining_budget: int,
    incremental_scoring: bool = False,
    direction_mode: str | None = None,
    original_directions: Mapping[str, int] | None = None,
    seed: int = 0,
    oriented_state: OrientedRouteState | None = None,
) -> tuple[tuple[_InsertionAlternative, ...], int]:
    if direction_mode is not None:
        return _oriented_parent_alternatives(
            parent_id, routes, hints, blocks, pattern, config,
            insertion_limit=insertion_limit, remaining_budget=remaining_budget,
            direction_mode=direction_mode, original_directions=original_directions or {},
            seed=seed, state=oriented_state,
        )
    parent_blocks = tuple(
        sorted(
            (block for block in blocks.values() if block.parent_id == parent_id),
            key=lambda block: block.block_id,
        )
    )
    base_proxy = _directed_proxy(routes, hints, blocks, config)
    if incremental_scoring:
        load_terms = tuple(
            tuple(config.process_time(blocks[item].length) for item in route)
            for route in routes
        )
        cached_loads = tuple(sum(terms) for terms in load_terms)
        arc_costs = tuple(
            _route_arc_costs(route, hints[robot], blocks, config)
            for robot, route in enumerate(routes)
        )
        # This assertion protects the tie-sensitive score ordering while the
        # optimized path is still an explicitly selected development option.
        assert _proxy_from_arc_costs(arc_costs) == base_proxy
    alternatives: list[_InsertionAlternative] = []
    evaluated = 0

    placements: list[tuple[int, int, int]] = []
    first = parent_blocks[0]
    first_robots = range(4)
    if pattern.kind is SplitKind.X_SPLIT:
        assert pattern.rail is not None
        first_robots = (0,) if pattern.rail.value == "UPPER" else (2,)
    for robot in first_robots:
        if not robot_is_eligible(first, robot, config):
            continue
        for position in bounded_insertion_positions(len(routes[robot]), insertion_limit):
            for hint in (0, 1):
                placements.append((robot, position, hint))

    for robot, position, hint in placements:
        if evaluated >= remaining_budget:
            break
        routes1, hints1 = _insert_one(routes, hints, first, robot, position, hint)
        if incremental_scoring:
            load_terms1 = list(load_terms)
            load_terms1[robot] = _insert_tuple(
                load_terms[robot], position, config.process_time(first.length)
            )
            loads1 = list(cached_loads)
            loads1[robot] = sum(load_terms1[robot])
            arc_costs1 = list(arc_costs)
            arc_costs1[robot] = _insert_route_arc_costs(
                routes[robot], hints[robot], arc_costs[robot], first,
                position, hint, blocks, config,
            )
        if len(parent_blocks) == 1:
            evaluated += 1
            if incremental_scoring:
                loads = tuple(loads1)
                delta = _proxy_from_arc_costs(arc_costs1) - base_proxy
            else:
                loads = _loads(routes1, blocks, config)
                delta = _directed_proxy(routes1, hints1, blocks, config) - base_proxy
            trace = ((first.block_id, robot, position, hint),)
            alternatives.append(
                _InsertionAlternative(
                    parent_id,
                    routes1,
                    hints1,
                    (max(loads), delta, robot, position, hint, trace),
                    trace,
                )
            )
            continue

        second = parent_blocks[1]
        second_robots = range(4)
        if pattern.kind is SplitKind.X_SPLIT:
            assert pattern.rail is not None
            second_robots = (1,) if pattern.rail.value == "UPPER" else (3,)
        for robot2 in second_robots:
            if evaluated >= remaining_budget:
                break
            if not robot_is_eligible(second, robot2, config):
                continue
            positions2 = bounded_insertion_positions(
                len(routes1[robot2]), insertion_limit
            )
            for position2 in positions2:
                if evaluated >= remaining_budget:
                    break
                for hint2 in (0, 1):
                    if evaluated >= remaining_budget:
                        break
                    # Insertion indices refer to different intermediate routes.
                    # Only final canonicalization may decide split collapse.
                    evaluated += 1
                    routes2, hints2 = _insert_one(
                        routes1, hints1, second, robot2, position2, hint2
                    )
                    if incremental_scoring:
                        load_terms2 = list(load_terms1)
                        load_terms2[robot2] = _insert_tuple(
                            load_terms1[robot2], position2,
                            config.process_time(second.length),
                        )
                        loads2 = list(loads1)
                        loads2[robot2] = sum(load_terms2[robot2])
                        arc_costs2 = list(arc_costs1)
                        arc_costs2[robot2] = _insert_route_arc_costs(
                            routes1[robot2], hints1[robot2], arc_costs1[robot2],
                            second, position2, hint2, blocks, config,
                        )
                        loads = tuple(loads2)
                        delta = _proxy_from_arc_costs(arc_costs2) - base_proxy
                    else:
                        loads = _loads(routes2, blocks, config)
                        delta = _directed_proxy(routes2, hints2, blocks, config) - base_proxy
                    trace = (
                        (first.block_id, robot, position, hint),
                        (second.block_id, robot2, position2, hint2),
                    )
                    alternatives.append(
                        _InsertionAlternative(
                            parent_id,
                            routes2,
                            hints2,
                            (
                                max(loads),
                                delta,
                                robot,
                                position,
                                hint,
                                robot2,
                                position2,
                                hint2,
                                trace,
                            ),
                            trace,
                        )
                    )
    return tuple(sorted(alternatives, key=lambda item: item.score)), evaluated


def oriented_route_state(routes, directions, blocks, config) -> OrientedRouteState:
    if len(routes) != 4 or len(directions) != 4:
        raise ValueError("four explicit routes and direction vectors are required")
    if any(len(route) != len(bits) or any(bit not in (0, 1) for bit in bits)
           for route, bits in zip(routes, directions)):
        raise ValueError("direction length/value mismatch")
    routes = tuple(tuple(route) for route in routes)
    directions = tuple(tuple(bits) for bits in directions)
    return OrientedRouteState(routes, directions, _loads(routes, blocks, config),
                              tuple(_route_arc_costs(route, directions[r], blocks, config)
                                    for r, route in enumerate(routes)))


def oriented_insert_variants(
    state: OrientedRouteState, block: WeldingBlock, robot: int, position: int,
    blocks: Mapping[str, WeldingBlock], config: ScientificConfig, *,
    fixed_direction: int | None = None, locked_directions: Mapping[str, int] | None = None,
    max_evaluations: int = 8, keep: int = 2, seed: int = 0,
) -> tuple[tuple[tuple[OrientedRouteState, OrientedInsertionAction], ...], int]:
    """One (block, robot, position, direction) decision with adjacent flips.

    Only arcs incident to the inserted block or its immediate neighbours are
    recomputed. Other routes and non-adjacent directions are retained verbatim.
    No route DP, scheduler or certifier is called here.
    """
    locked = locked_directions or {}
    route = state.routes[robot]
    if not 0 <= position <= len(route) or not robot_is_eligible(block, robot, config):
        raise ValueError("ineligible robot or invalid insertion position")
    new_route = route[:position] + (block.block_id,) + route[position:]
    original = dict(zip(route, state.directions[robot]))
    local = tuple(i for i in (position - 1, position, position + 1) if 0 <= i < len(new_route))
    choices = []
    for i in local:
        item = new_route[i]
        if item in locked:
            choices.append((locked[item],))
        elif fixed_direction is not None:
            choices.append((fixed_direction if i == position else original[item],))
        else:
            choices.append((0, 1))
    combinations = list(itertools.product(*choices))
    random.Random(seed).shuffle(combinations)
    rows = []
    used = 0
    for bits in combinations[:max(0, max_evaluations)]:
        used += 1
        hint = bits[local.index(position)]
        routes, directions = _insert_one(state.routes, state.directions, block, robot, position, hint)
        vector = list(directions[robot])
        for i, bit in zip(local, bits):
            vector[i] = bit
        directions = tuple(tuple(vector) if r == robot else directions[r] for r in range(4))
        points = {r: oriented_endpoints(blocks[items[0]], directions[r][0])[0]
                  for r, items in enumerate(routes) if items}
        if _initial_error(points, config) is not None:
            continue
        # Reuse old arcs, splice the insertion, then refresh only up to four
        # arcs touched by the predecessor/new block/successor direction tuple.
        costs = list(_insert_route_arc_costs(route, state.directions[robot], state.arc_costs[robot],
                                            block, position, hint, blocks, config))
        for i in range(max(0, position - 2), min(len(new_route) - 1, position + 2)):
            _, end = oriented_endpoints(blocks[new_route[i]], vector[i])
            start, _ = oriented_endpoints(blocks[new_route[i + 1]], vector[i + 1])
            costs[i] = math.dist(end, start) / config.empty_speed
        arcs = tuple(tuple(costs) if r == robot else state.arc_costs[r] for r in range(4))
        loads = tuple(value + (config.process_time(block.length) if r == robot else 0.0)
                      for r, value in enumerate(state.loads))
        changes = tuple((new_route[i], original[new_route[i]], vector[i]) for i in local
                        if i != position and original[new_route[i]] != vector[i])
        action = OrientedInsertionAction(block.block_id, robot, position, hint, changes)
        candidate = OrientedRouteState(routes, directions, loads, arcs)
        rows.append((candidate, action))
    rows.sort(key=lambda row: (row[0].travel, row[0].directions))
    return tuple(rows[:keep]), used


def _oriented_parent_alternatives(
    parent_id, routes, hints, blocks, pattern, config, *, insertion_limit,
    remaining_budget, direction_mode, original_directions, seed, state=None,
):
    state = state or oriented_route_state(routes, hints, blocks, config)
    children = sorted((b for b in blocks.values() if b.parent_id == parent_id), key=lambda b: b.block_id)
    rng = random.Random(seed)
    legal = []
    for index, child in enumerate(children):
        robots = [r for r in range(4) if robot_is_eligible(child, r, config)]
        if pattern.kind is SplitKind.X_SPLIT:
            robots = [(0 if pattern.rail.value == "UPPER" else 2) + index]
        legal.append(robots)
    groups = list(itertools.product(*legal))
    rng.shuffle(groups)
    alternatives = []
    evaluated = 0
    for group_index, robots in enumerate(groups):
        # Every legal robot (or child robot pair) receives its own quota. With
        # very small budgets the seeded group rotation avoids persistent R0 bias.
        quota = remaining_budget // len(groups) + (group_index < remaining_budget % len(groups))
        used_group = 0
        positions = [bounded_insertion_positions(len(routes[robots[0]]), insertion_limit)]
        if len(children) == 2:
            positions.append(bounded_insertion_positions(len(routes[robots[1]]) + int(robots[0] == robots[1]), insertion_limit))
        plans = list(itertools.product(*positions))
        rng.shuffle(plans)
        for plan in plans:
            left = quota - used_group
            if left < len(children):
                break
            allowance = min(left, 8 if len(children) == 1 else 24)
            first_budget = allowance if len(children) == 1 else max(1, allowance // 3)
            first_rows, used = oriented_insert_variants(
                state, children[0], robots[0], plan[0], blocks, config,
                fixed_direction=(original_directions[children[0].block_id] if direction_mode == "FIXED" else None),
                max_evaluations=first_budget, keep=2, seed=rng.randrange(2**32),
            )
            used_group += used
            complete = [(s, (a,)) for s, a in first_rows]
            if len(children) == 2:
                complete = []
                for j, (first_state, first_action) in enumerate(first_rows):
                    second_budget = (allowance - (used_group - (quota - left))) // (len(first_rows) - j)
                    second_rows, used = oriented_insert_variants(
                        first_state, children[1], robots[1], plan[1], blocks, config,
                        fixed_direction=(original_directions[children[1].block_id] if direction_mode == "FIXED" else None),
                        max_evaluations=max(0, second_budget), keep=2, seed=rng.randrange(2**32),
                    )
                    used_group += used
                    complete.extend((s, (first_action, a)) for s, a in second_rows)
            for final, actions in complete:
                # Adjacent children would be merged by canonicalize. Exclude
                # this structural action before it can crowd out fixed-mode alternatives.
                if len(children) == 2 and any(
                    children[0].block_id in route and children[1].block_id in route
                    and abs(route.index(children[0].block_id) - route.index(children[1].block_id)) == 1
                    for route in final.routes
                ):
                    continue
                trace = tuple((a.block_id, a.robot, a.position, a.direction) for a in actions)
                score = (max(final.loads), final.travel - state.travel, final.routes, final.directions)
                alternatives.append(_InsertionAlternative(parent_id, final.routes, final.directions,
                                                           score, trace, actions, final))
        evaluated += used_group
    # Regret compares distinct route structures, never two direction variants
    # of the same assignment and insertion location.
    by_structure = {}
    for alternative in sorted(alternatives, key=lambda a: a.score):
        by_structure.setdefault(alternative.routes, alternative)
    return tuple(sorted(by_structure.values(), key=lambda a: a.score)), evaluated


def repair_partial_state(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    partial: PartialSearchState,
    destroy_operator: DestroyOperator,
    repair_operator: RepairOperator,
    config: ScientificConfig,
    *,
    insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096,
    incremental_scoring: bool = False,
    preserve_patterns: bool = False,
    direction_mode: str | None = None,
    seed: int = 0,
) -> RepairResult:
    if partial.source_revision != current.revision:
        raise ValueError("stale partial state revision")
    if max_insertion_evaluations < 1:
        raise ValueError("repair insertion budget must be positive")
    patterns = {pattern.parent_id: pattern for pattern in partial.patterns}
    blocks = block_map(current, config)
    current_hints_by_block = _hints_from_current(current, current_directions)
    if direction_mode is not None:
        if direction_mode not in ("FIXED", "JOINT"):
            raise ValueError("unknown oriented repair mode")
        oriented_route_state(tuple(r.block_ids for r in current.routes), current_directions, blocks, config)
    routes = partial.routes
    hints = tuple(
        tuple(current_hints_by_block.get(item, 0) for item in route)
        for route in routes
    )
    oriented_state = (oriented_route_state(routes, hints, blocks, config)
                      if direction_mode is not None else None)
    pending = list(partial.removed_parent_ids)
    parents = {parent.parent_id: parent for parent in current.parents}
    trace: list[tuple[str, tuple[tuple[str, int, int, int], ...]]] = []
    evaluations = 0
    actions: list[OrientedInsertionAction] = []
    rng = random.Random(seed)

    while pending:
        if repair_operator is RepairOperator.GREEDY_REPAIR:
            selected_parent = min(
                pending,
                key=lambda parent_id: (
                    -config.process_time(parents[parent_id].length),
                    parent_id,
                ),
            )
            parents_to_evaluate = (selected_parent,)
        else:
            parents_to_evaluate = tuple(sorted(pending))
        alternatives_by_parent: dict[str, tuple[_InsertionAlternative, ...]] = {}
        future_calls = max(
            1,
            len(pending)
            if repair_operator is RepairOperator.GREEDY_REPAIR
            else len(pending) * (len(pending) + 1) // 2,
        )
        per_call_budget = max(
            1, (max_insertion_evaluations - evaluations) // future_calls
        )
        for parent_id in parents_to_evaluate:
            oriented_options = ({} if direction_mode is None else dict(
                direction_mode=direction_mode, original_directions=current_hints_by_block,
                seed=rng.randrange(2**32), oriented_state=oriented_state,
            ))
            alternatives, used = _parent_alternatives(
                parent_id,
                routes,
                hints,
                blocks,
                patterns[parent_id],
                config,
                insertion_limit=insertion_limit,
                remaining_budget=min(
                    per_call_budget, max_insertion_evaluations - evaluations
                ),
                incremental_scoring=incremental_scoring,
                **oriented_options,
            )
            evaluations += used
            alternatives_by_parent[parent_id] = alternatives
            if evaluations >= max_insertion_evaluations:
                break
        viable = {
            parent_id: alternatives
            for parent_id, alternatives in alternatives_by_parent.items()
            if alternatives
        }
        if not viable:
            return RepairResult(None, evaluations, "NO_LEGAL_REPAIR_INSERTION")
        if repair_operator is RepairOperator.GREEDY_REPAIR:
            selected_parent = next(iter(viable))
        else:
            regret_rows = []
            for parent_id, alternatives in viable.items():
                best = alternatives[0]
                second = alternatives[1] if len(alternatives) > 1 else None
                primary = (
                    math.inf
                    if second is None
                    else float(second.score[0]) - float(best.score[0])
                )
                secondary = (
                    math.inf
                    if second is None
                    else float(second.score[1]) - float(best.score[1])
                )
                regret_rows.append(
                    (
                        -primary,
                        -secondary,
                        -config.process_time(parents[parent_id].length),
                        parent_id,
                    )
                )
            selected_parent = min(regret_rows)[3]
        selected = viable[selected_parent][0]
        routes, hints = selected.routes, selected.hints
        oriented_state = selected.oriented_state
        trace.append((selected_parent, selected.trace))
        actions.extend(selected.actions)
        pending.remove(selected_parent)

    try:
        solution = canonicalize(
            partial.parents,
            partial.patterns,
            tuple(Route(robot, routes[robot]) for robot in range(4)),
            config,
            revision=current.revision + 1,
            x_split_validator=(
                finite_x_split_validator(current.parents, config)
                if any(pattern.kind is SplitKind.X_SPLIT for pattern in partial.patterns)
                else None
            ),
        )
        solution_blocks = block_map(solution, config)
        if preserve_patterns and solution.patterns != current.patterns:
            return RepairResult(None, evaluations, "FIXED_PATTERN_CHANGED_BY_CANONICALIZATION")
        if any(
            not robot_is_eligible(solution_blocks[item], route.robot_id, config)
            for route in solution.routes
            for item in route.block_ids
        ):
            raise ValueError("robot eligibility failure")
    except (ValueError, KeyError, IndexError) as error:
        return RepairResult(None, evaluations, f"CANONICAL_REPAIR_FAILURE:{error}")
    current_blocks = block_map(current, config)
    solution_hints_by_block = {
        item: hints[robot][index]
        for robot, route in enumerate(routes)
        for index, item in enumerate(route)
    }
    final_routes = tuple(route.block_ids for route in solution.routes)
    final_hints = tuple(
        tuple(solution_hints_by_block.get(item, 0) for item in route)
        for route in final_routes
    )
    current_routes = tuple(route.block_ids for route in current.routes)
    current_hints = tuple(
        tuple(current_hints_by_block.get(item, 0) for item in route)
        for route in current_routes
    )
    current_loads = _loads(current_routes, current_blocks, config)
    final_loads = _loads(final_routes, solution_blocks, config)
    repair_identity = tuple(trace)
    identity = CompleteCandidateIdentity(
        current.revision,
        CandidateSourceKind.LNS_REPAIRED.value,
        (
            destroy_operator.value,
            repair_operator.value,
            partial.removed_parent_ids,
            repair_identity,
        ),
    )
    provenance = (
        ("destroy_operator", destroy_operator.value),
        ("repair_operator", repair_operator.value),
        ("removed_parent_ids", partial.removed_parent_ids),
        ("q", len(partial.removed_parent_ids)),
        ("repair_trace", repair_identity),
    )
    candidate = CompleteSearchCandidate(
        CandidateSourceKind.LNS_REPAIRED,
        solution,
        identity,
        max(final_loads),
        _directed_proxy(final_routes, final_hints, solution_blocks, config)
        - _directed_proxy(current_routes, current_hints, current_blocks, config),
        _optional_split_count(solution, config) - _optional_split_count(current, config),
        (max(final_loads) - min(final_loads))
        - (max(current_loads) - min(current_loads)),
        provenance,
    )
    return RepairResult(candidate, evaluations, directions=(final_hints if direction_mode is not None else None),
                        actions=tuple(actions))


def repair_partial_state_oriented(
    current: CanonicalSolution, current_directions: DirectionVectors,
    partial: PartialSearchState, config: ScientificConfig, *,
    fixed_directions: bool = False, insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096, seed: int = 0,
) -> RepairResult:
    """The existing Regret-2 loop with explicit, persistent action directions."""
    return repair_partial_state(
        current, current_directions, partial, DestroyOperator.RANDOM_REMOVAL,
        RepairOperator.REGRET_2_REPAIR, config, insertion_limit=insertion_limit,
        max_insertion_evaluations=max_insertion_evaluations, preserve_patterns=True,
        direction_mode="FIXED" if fixed_directions else "JOINT", seed=seed,
    )


def _block_mid_x(block: WeldingBlock) -> float:
    return 0.5 * (block.start[0] + block.end[0])


def _parent_mid_x(parent_id: str, blocks: Mapping[str, WeldingBlock]) -> float:
    rows = [block for block in blocks.values() if block.parent_id == parent_id]
    return sum(_block_mid_x(block) for block in rows) / len(rows)


def _robot_by_block(solution: CanonicalSolution) -> dict[str, int]:
    return {
        block_id: robot
        for robot, route in enumerate(solution.routes)
        for block_id in route.block_ids
    }


def _weld_parent_neighbours(
    schedule: ScheduleResult, robot: int, wait_operation
) -> tuple[str, ...]:
    """Associate WAIT with adjacent real WELDs; this is not a causal claim."""
    welds = sorted(
        (
            operation
            for operation in schedule.operations
            if operation.robot_id == robot
            and operation.kind is OperationKind.WELD
            and operation.block_id is not None
        ),
        key=lambda operation: (
            operation.start_time,
            operation.end_time,
            operation.sequence_index,
            operation.operation_id,
        ),
    )
    before = [op for op in welds if op.end_time <= wait_operation.start_time + 1.0e-9]
    after = [op for op in welds if op.start_time >= wait_operation.end_time - 1.0e-9]
    adjacent = []
    if before:
        adjacent.append(before[-1])
    if after:
        adjacent.append(after[0])
    if not adjacent and welds:
        adjacent.append(min(welds, key=lambda op: (
            abs(op.sequence_index - wait_operation.sequence_index),
            abs(op.start_time - wait_operation.start_time), op.operation_id,
        )))
    return tuple(dict.fromkeys(_parent_id(op.block_id) for op in adjacent))


def identify_crrr_region(
    current: CanonicalSolution,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *,
    attempt_index: int,
    min_parents: int = 2,
    max_parents: int = 6,
) -> CRRRRegion | None:
    """Locate one bounded schedule- and geometry-anchored parent region."""
    if attempt_index < 0 or attempt_index >= 8:
        raise ValueError("CRRR attempt_index must be in 0..7")
    if not (2 <= min_parents <= max_parents <= 6):
        raise ValueError("CRRR parent bounds must satisfy 2 <= min <= max <= 6")
    blocks = block_map(current, config)
    robot_by_block = _robot_by_block(current)
    critical_robot = max(
        range(4), key=lambda robot: (schedule.robot_completion[robot], -robot)
    )
    partner_robot = critical_robot ^ 1
    waits = sorted(
        (
            operation
            for operation in schedule.operations
            if operation.kind is OperationKind.WAIT
            and operation.duration > config.numeric_epsilon
        ),
        key=lambda operation: (
            -int(operation.robot_id == critical_robot), -operation.duration,
            operation.robot_id, operation.start_time, operation.operation_id,
        ),
    )
    use_wait = bool(waits) and attempt_index % 2 == 0
    anchors: list[str] = []
    wait_duration = 0.0
    if use_wait:
        wait = waits[(attempt_index // 2) % len(waits)]
        critical_robot = wait.robot_id
        partner_robot = critical_robot ^ 1
        wait_duration = wait.duration
        anchors.extend(_weld_parent_neighbours(schedule, critical_robot, wait))
        anchor_points = [
            _parent_mid_x(parent_id, blocks)
            for parent_id in anchors
            if any(block.parent_id == parent_id for block in blocks.values())
        ]
        anchor_x = (
            sum(anchor_points) / len(anchor_points)
            if anchor_points else 0.5 * (wait.start[0] + wait.end[0])
        )
        partner_welds = sorted(
            (
                operation
                for operation in schedule.operations
                if operation.robot_id == partner_robot
                and operation.kind is OperationKind.WELD
                and operation.block_id is not None
                and operation.start_time <= wait.end_time + 1.0e-9
                and operation.end_time >= wait.start_time - 1.0e-9
            ),
            key=lambda operation: (
                abs(0.5 * (operation.start[0] + operation.end[0]) - anchor_x),
                operation.sequence_index,
            ),
        )
        anchors.extend(_parent_id(operation.block_id) for operation in partner_welds[:2])
        bottleneck = CRRRBottleneck.WAIT_BOUNDARY
    else:
        left_robot = 0 if critical_robot < 2 else 2
        right_robot = left_robot + 1
        left_x = [_block_mid_x(blocks[item]) for item in current.routes[left_robot].block_ids]
        right_x = [_block_mid_x(blocks[item]) for item in current.routes[right_robot].block_ids]
        if left_x and right_x:
            anchor_x = 0.5 * (max(left_x) + min(right_x))
        else:
            rail_x = left_x + right_x
            anchor_x = sorted(rail_x)[len(rail_x) // 2] if rail_x else 0.0
        bottleneck = CRRRBottleneck.LOAD_BOUNDARY
    rail_parents = {
        blocks[item].parent_id
        for robot in (critical_robot, partner_robot)
        for item in current.routes[robot].block_ids
    }
    ordered = sorted(
        (parent.parent_id for parent in current.parents),
        key=lambda parent_id: (
            -int(parent_id in anchors), -int(parent_id in rail_parents),
            abs(_parent_mid_x(parent_id, blocks) - anchor_x),
            -int(any(
                robot_by_block[block.block_id] == critical_robot
                for block in blocks.values() if block.parent_id == parent_id
            )),
            parent_id,
        ),
    )
    target_count = min(max_parents, max(min_parents, 2 + attempt_index % 5))
    selected = tuple(sorted(ordered[:target_count]))
    if len(selected) < min_parents:
        return None
    return CRRRRegion(
        bottleneck, selected, critical_robot, partner_robot, anchor_x, wait_duration
    )


def _adjacency_by_parent(routes: Sequence[Sequence[str]]) -> dict[str, set[tuple[str, str]]]:
    result: dict[str, set[tuple[str, str]]] = {}
    for route in routes:
        parents = tuple(_parent_id(item) for item in route)
        for index, parent_id in enumerate(parents):
            neighbours = result.setdefault(parent_id, set())
            if index:
                neighbours.add((parents[index - 1], parent_id))
            if index + 1 < len(parents):
                neighbours.add((parent_id, parents[index + 1]))
    return result


def _route_overlap_proxy(
    routes: Sequence[Sequence[str]], blocks: Mapping[str, WeldingBlock]
) -> float:
    envelopes = []
    for route in routes:
        xs = [x for item in route for x in (blocks[item].start[0], blocks[item].end[0])]
        envelopes.append(None if not xs else (min(xs), max(xs)))
    total = 0.0
    for left, right in ((0, 1), (2, 3)):
        if envelopes[left] is not None and envelopes[right] is not None:
            total += max(0.0, min(envelopes[left][1], envelopes[right][1])
                         - max(envelopes[left][0], envelopes[right][0]))
    return total


def _first_task_order_risk(
    routes: Sequence[Sequence[str]], blocks: Mapping[str, WeldingBlock]
) -> float:
    return sum(
        max(0.0, _block_mid_x(blocks[routes[left][0]])
            - _block_mid_x(blocks[routes[right][0]]))
        for left, right in ((0, 1), (2, 3))
        if routes[left] and routes[right]
    )


def _crrr_components(
    routes: tuple[tuple[str, ...], ...],
    hints: tuple[tuple[int, ...], ...],
    blocks: Mapping[str, WeldingBlock],
    config: ScientificConfig,
    *, current_max_load: float, current_travel: float, wait_duration: float,
) -> tuple[float, float, float, float, float, float]:
    loads = _loads(routes, blocks, config)
    travel = _directed_proxy(routes, hints, blocks, config)
    scale = max(current_max_load, 1.0)
    travel_scale = max(current_travel, 1.0)
    overlap = _route_overlap_proxy(routes, blocks) / max(config.workspace_x[1] - config.workspace_x[0], 1.0)
    order_risk = _first_task_order_risk(routes, blocks) / max(config.workspace_x[1] - config.workspace_x[0], 1.0)
    normalized = (
        0.45 * max(loads) / scale
        + 0.20 * (max(loads) - min(loads)) / scale
        + 0.20 * travel / travel_scale
        + 0.10 * overlap
        + 0.05 * order_risk * (1.0 + min(wait_duration / scale, 1.0))
    )
    return normalized, max(loads), max(loads) - min(loads), travel, overlap, order_risk


def build_crrr_candidate(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    schedule: ScheduleResult,
    config: ScientificConfig,
    *, attempt_index: int, insertion_limit: int = 8,
    max_insertion_evaluations: int = 4096,
) -> RepairResult:
    """Build one joint reassignment/reordering without direction or B32 access."""
    region = identify_crrr_region(current, schedule, config, attempt_index=attempt_index)
    if region is None:
        return RepairResult(None, 0, "CRRR_NO_REGION")
    selected = set(region.affected_parent_ids)
    blocks = block_map(current, config)
    patterns = {pattern.parent_id: pattern for pattern in current.patterns}
    robot_by_block = _robot_by_block(current)
    current_routes = tuple(route.block_ids for route in current.routes)
    hints_by_block = _hints_from_current(current, current_directions)
    current_hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in current_routes)
    routes = tuple(tuple(item for item in route if _parent_id(item) not in selected) for route in current_routes)
    hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in routes)
    current_loads = _loads(current_routes, blocks, config)
    current_travel = _directed_proxy(current_routes, current_hints, blocks, config)
    scheme = attempt_index % 4
    scheme_names = (
        "SAME_RAIL_CONTIGUOUS_TRANSFER", "SAME_RAIL_BOUNDARY_EXCHANGE",
        "WAIT_REGION_LOCAL_REORDER", "CROSS_RAIL_REALLOCATION",
    )
    projected_loads = list(_loads(routes, blocks, config))
    placements: list[tuple[str, int]] = []
    parent_order = sorted(selected, key=lambda parent_id: (
        abs(_parent_mid_x(parent_id, blocks) - region.anchor_x), parent_id,
    ))
    if scheme == 2:
        parent_order.reverse()
    for parent_rank, parent_id in enumerate(parent_order):
        parent_blocks = sorted(
            (block for block in blocks.values() if block.parent_id == parent_id),
            key=lambda block: (_block_mid_x(block), block.block_id),
            reverse=scheme == 2 and parent_rank % 2 == 0,
        )
        pattern = patterns[parent_id]
        for block in parent_blocks:
            original_robot = robot_by_block[block.block_id]
            eligible = [robot for robot in range(4) if robot_is_eligible(block, robot, config)]
            if pattern.kind is SplitKind.X_SPLIT:
                eligible = [original_robot]
            if scheme == 0:
                preferred = [region.partner_robot, original_robot]
            elif scheme == 1:
                swap = (region.partner_robot if original_robot == region.critical_robot
                        else region.critical_robot if original_robot == region.partner_robot
                        else original_robot)
                preferred = [swap, original_robot]
            elif scheme == 2:
                preferred = [original_robot]
            else:
                preferred = sorted(
                    (robot for robot in eligible if robot // 2 != original_robot // 2),
                    key=lambda robot: (projected_loads[robot], robot),
                ) + [original_robot]
            target = next((robot for robot in preferred if robot in eligible), None)
            if target is None:
                target = min(eligible, key=lambda robot: (projected_loads[robot], robot))
            placements.append((block.block_id, target))
            projected_loads[target] += config.process_time(block.length)
    evaluations = 0
    trace: list[tuple[str, int, int, int]] = []
    for block_id, target in placements:
        if evaluations >= max_insertion_evaluations:
            return RepairResult(None, evaluations, "CRRR_REPAIR_BUDGET_EXHAUSTED")
        alternatives = []
        for position in bounded_insertion_positions(len(routes[target]), insertion_limit):
            if evaluations >= max_insertion_evaluations:
                break
            evaluations += 1
            candidate_routes, candidate_hints = _insert_one(
                routes, hints, blocks[block_id], target, position, hints_by_block.get(block_id, 0)
            )
            components = _crrr_components(
                candidate_routes, candidate_hints, blocks, config,
                current_max_load=max(current_loads), current_travel=current_travel,
                wait_duration=region.wait_duration,
            )
            alternatives.append((components, position, candidate_routes, candidate_hints))
        if not alternatives:
            return RepairResult(None, evaluations, "CRRR_NO_LEGAL_INSERTION")
        _, position, routes, hints = min(alternatives, key=lambda row: (row[0], row[1]))
        trace.append((block_id, target, position, hints_by_block.get(block_id, 0)))
    try:
        solution = canonicalize(
            current.parents, current.patterns,
            tuple(Route(robot, routes[robot]) for robot in range(4)), config,
            revision=current.revision + 1,
            x_split_validator=(finite_x_split_validator(current.parents, config)
                if any(pattern.kind is SplitKind.X_SPLIT for pattern in current.patterns) else None),
        )
        if tuple(pattern.pattern_id for pattern in solution.patterns) != tuple(
            pattern.pattern_id for pattern in current.patterns
        ):
            raise ValueError("CRRR changed split pattern")
        solution_blocks = block_map(solution, config)
        if any(not robot_is_eligible(solution_blocks[item], route.robot_id, config)
               for route in solution.routes for item in route.block_ids):
            raise ValueError("robot eligibility failure")
    except (ValueError, KeyError, IndexError) as error:
        return RepairResult(None, evaluations, f"CRRR_CANONICAL_FAILURE:{error}")
    final_routes = tuple(route.block_ids for route in solution.routes)
    final_hints = tuple(tuple(hints_by_block.get(item, 0) for item in route) for route in final_routes)
    final_robot_by_block = _robot_by_block(solution)
    assignment_parent_ids = {
        parent_id for parent_id in selected
        if any(final_robot_by_block[block.block_id] != robot_by_block[block.block_id]
               for block in blocks.values() if block.parent_id == parent_id)
    }
    before_adjacency = _adjacency_by_parent(current_routes)
    after_adjacency = _adjacency_by_parent(final_routes)
    adjacency_parent_ids = {
        parent_id for parent_id in selected
        if before_adjacency.get(parent_id, set()) != after_adjacency.get(parent_id, set())
    }
    changed_parent_ids = assignment_parent_ids | adjacency_parent_ids
    if len(changed_parent_ids) < 2 or solution.canonical_hash == current.canonical_hash:
        return RepairResult(None, evaluations, "CRRR_NO_JOINT_CHANGE")
    final_loads = _loads(final_routes, solution_blocks, config)
    components = _crrr_components(
        final_routes, final_hints, solution_blocks, config,
        current_max_load=max(current_loads), current_travel=current_travel,
        wait_duration=region.wait_duration,
    )
    identity = CompleteCandidateIdentity(
        current.revision, CandidateSourceKind.LNS_REPAIRED.value,
        (CRRR_OPERATOR, region.bottleneck_type.value, attempt_index,
         region.affected_parent_ids, tuple(tuple(route) for route in final_routes)),
    )
    provenance = (
        ("operator", CRRR_OPERATOR),
        ("bottleneck_type", region.bottleneck_type.value),
        ("affected_parent_ids", region.affected_parent_ids),
        ("removed_parent_ids", region.affected_parent_ids),
        ("critical_robot", region.critical_robot),
        ("partner_robot", region.partner_robot),
        ("changed_parent_count", len(changed_parent_ids)),
        ("assignment_changes", len(assignment_parent_ids)),
        ("route_adjacency_changes", len(adjacency_parent_ids)),
        ("scheme", scheme_names[scheme]),
        ("cheap_score", tuple(round(value, 9) for value in components)),
        ("repair_trace", tuple(
            (parent_id, tuple(row for row in trace if _parent_id(row[0]) == parent_id))
            for parent_id in region.affected_parent_ids
        )),
    )
    return RepairResult(
        CompleteSearchCandidate(
            CandidateSourceKind.LNS_REPAIRED, solution, identity, max(final_loads),
            _directed_proxy(final_routes, final_hints, solution_blocks, config) - current_travel,
            0, (max(final_loads) - min(final_loads)) - (max(current_loads) - min(current_loads)),
            provenance,
        ),
        evaluations,
    )


@dataclass
class AdaptiveOperatorState:
    reaction: float = 0.2
    segment_length: int = 20
    weights: dict[tuple[DestroyOperator, RepairOperator], float] = field(
        default_factory=lambda: {pair: 1.0 for pair in OPERATOR_PAIRS}
    )
    segment_uses: dict[tuple[DestroyOperator, RepairOperator], int] = field(
        default_factory=lambda: {pair: 0 for pair in OPERATOR_PAIRS}
    )
    segment_rewards: dict[tuple[DestroyOperator, RepairOperator], float] = field(
        default_factory=lambda: {pair: 0.0 for pair in OPERATOR_PAIRS}
    )
    observations: int = 0
    selection_history: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 < self.reaction <= 1.0):
            raise ValueError("adaptive reaction must be in (0,1]")
        if self.segment_length < 1:
            raise ValueError("adaptive segment length must be positive")

    def select(self, rng: random.Random) -> tuple[DestroyOperator, RepairOperator]:
        pairs = OPERATOR_PAIRS
        total = sum(self.weights[pair] for pair in pairs)
        draw = rng.random() * total
        cumulative = 0.0
        selected = pairs[-1]
        for pair in pairs:
            cumulative += self.weights[pair]
            if draw < cumulative:
                selected = pair
                break
        self.selection_history.append((selected[0].value, selected[1].value))
        return selected

    def record(
        self,
        pair: tuple[DestroyOperator, RepairOperator],
        reward: float | None,
    ) -> None:
        if reward is None:
            return
        self.segment_uses[pair] += 1
        self.segment_rewards[pair] += reward
        self.observations += 1
        if self.observations % self.segment_length:
            return
        for operator_pair in OPERATOR_PAIRS:
            uses = self.segment_uses[operator_pair]
            if uses:
                average = self.segment_rewards[operator_pair] / uses
                self.weights[operator_pair] = (
                    (1.0 - self.reaction) * self.weights[operator_pair]
                    + self.reaction * average
                )
            self.segment_uses[operator_pair] = 0
            self.segment_rewards[operator_pair] = 0.0


def lns_pair(candidate: CompleteSearchCandidate) -> tuple[DestroyOperator, RepairOperator] | None:
    if candidate.source_kind is not CandidateSourceKind.LNS_REPAIRED:
        return None
    values = dict(candidate.provenance)
    if values.get("operator") == CRRR_OPERATOR:
        return None
    if "destroy_operator" not in values or "repair_operator" not in values:
        return None
    return (
        DestroyOperator(str(values["destroy_operator"])),
        RepairOperator(str(values["repair_operator"])),
    )
