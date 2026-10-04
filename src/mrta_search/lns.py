from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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

from .direction import DirectionVectors
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


@dataclass(frozen=True)
class _InsertionAlternative:
    parent_id: str
    routes: tuple[tuple[str, ...], ...]
    hints: tuple[tuple[int, ...], ...]
    score: tuple[object, ...]
    trace: tuple[tuple[str, int, int, int], ...]


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
) -> tuple[tuple[_InsertionAlternative, ...], int]:
    parent_blocks = tuple(
        sorted(
            (block for block in blocks.values() if block.parent_id == parent_id),
            key=lambda block: block.block_id,
        )
    )
    base_proxy = _directed_proxy(routes, hints, blocks, config)
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
        if len(parent_blocks) == 1:
            evaluated += 1
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
) -> RepairResult:
    if partial.source_revision != current.revision:
        raise ValueError("stale partial state revision")
    if max_insertion_evaluations < 1:
        raise ValueError("repair insertion budget must be positive")
    patterns = {pattern.parent_id: pattern for pattern in partial.patterns}
    blocks = block_map(current, config)
    current_hints_by_block = _hints_from_current(current, current_directions)
    routes = partial.routes
    hints = tuple(
        tuple(current_hints_by_block.get(item, 0) for item in route)
        for route in routes
    )
    pending = list(partial.removed_parent_ids)
    parents = {parent.parent_id: parent for parent in current.parents}
    trace: list[tuple[str, tuple[tuple[str, int, int, int], ...]]] = []
    evaluations = 0

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
        trace.append((selected_parent, selected.trace))
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
    return RepairResult(candidate, evaluations)


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
    return (
        DestroyOperator(str(values["destroy_operator"])),
        RepairOperator(str(values["repair_operator"])),
    )
