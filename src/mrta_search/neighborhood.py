from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from mrta_reference.candidate import apply_candidate
from mrta_reference.geometry import (
    finite_x_split_validator,
    oriented_endpoints,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CandidateKey,
    CandidateMove,
    CanonicalSolution,
    FormalScope,
    MoveType,
    ScientificConfig,
    SplitKind,
    SplitPattern,
    robot_rail,
)
from mrta_reference.solution import block_map

from .direction import DirectionVectors
from .stats import ACTIVE_MOVE_TYPES, TRACKED_MOVE_TYPES, SearchStats


@dataclass(frozen=True)
class RawAttempt:
    move_type: MoveType
    candidate: CandidateMove | None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ScreenedCandidate:
    candidate: CandidateMove
    solution: CanonicalSolution
    cheap_score: tuple[object, ...]
    projected_process_makespan: float
    local_directed_travel_delta: float
    split_count_delta: int


def candidate_key_order(key: CandidateKey) -> tuple[object, ...]:
    return (
        key.current_solution_revision,
        key.move_type.value,
        key.affected_parent_ids,
        -1 if key.source_robot_id is None else key.source_robot_id,
        -1 if key.destination_robot_id is None else key.destination_robot_id,
        key.source_positions,
        key.destination_positions,
        key.split_pattern_id or "",
        key.split_point_id or "",
    )


def balanced_move_attempt_order(
    m: int,
    seed: int,
    moves: tuple[MoveType, ...] = ACTIVE_MOVE_TYPES,
) -> tuple[MoveType, ...]:
    if m < 0:
        raise ValueError("M must be non-negative")
    if not moves:
        return ()
    base, remainder = divmod(m, len(moves))
    start = seed % len(moves)
    rotated = moves[start:] + moves[:start]
    quotas = {move: base for move in moves}
    for move in rotated[:remainder]:
        quotas[move] += 1
    result = []
    while len(result) < m:
        progressed = False
        for move in rotated:
            used = result.count(move)
            if used < quotas[move]:
                result.append(move)
                progressed = True
        if not progressed:
            break
    return tuple(result)


def applicable_move_mask(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    enable_two_opt_star: bool = False,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
) -> dict[MoveType, bool]:
    routes = [route.block_ids for route in solution.routes]
    blocks = block_map(solution, config)
    parents = {parent.parent_id: parent for parent in solution.parents}
    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}
    total_blocks = sum(len(route) for route in routes)

    intra = any(len(route) >= 2 for route in routes)
    inter = any(
        robot_is_eligible(blocks[block_id], destination, config)
        for source, route in enumerate(routes)
        for block_id in route
        for destination in range(4)
        if destination != source
    )
    swap = intra
    if not swap and total_blocks >= 2:
        occurrences = [
            (robot, block_id)
            for robot, route in enumerate(routes)
            for block_id in route
        ]
        swap = any(
            first_robot != second_robot
            and robot_is_eligible(blocks[first_block], second_robot, config)
            and robot_is_eligible(blocks[second_block], first_robot, config)
            for index, (first_robot, first_block) in enumerate(occurrences)
            for second_robot, second_block in occurrences[index + 1 :]
        )

    return {
        MoveType.INTRA_RELOCATE: intra,
        MoveType.INTER_RELOCATE: inter,
        MoveType.SWAP: swap,
        MoveType.TWO_OPT: intra,
        MoveType.TWO_OPT_STAR: enable_two_opt_star and any(
            len(routes[left]) + len(routes[right]) > 0
            for left, right in ((0, 1), (2, 3))
        ),
    }


def _parent_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def _local_rng(seed: int, revision: int, move: MoveType, ordinal: int) -> random.Random:
    move_index = TRACKED_MOVE_TYPES.index(move)
    return random.Random(seed + 1_000_003 * revision + 10_007 * move_index + 97 * ordinal)


def make_raw_candidate(
    solution: CanonicalSolution,
    config: ScientificConfig,
    move: MoveType,
    *,
    seed: int,
    ordinal: int,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
) -> RawAttempt:
    if move not in TRACKED_MOVE_TYPES:
        return RawAttempt(move, None, "RETIRED_DYNAMIC_MOVE")
    rng = _local_rng(seed, solution.revision, move, ordinal)
    routes = [route.block_ids for route in solution.routes]
    occurrences = [(robot, position, block) for robot, route in enumerate(routes) for position, block in enumerate(route)]
    def key(
        affected,
        source=None,
        destination=None,
        source_positions=(),
        destination_positions=(),
        pattern=None,
    ):
        return CandidateKey(
            solution.revision,
            move,
            tuple(sorted(set(affected))),
            source,
            destination,
            tuple(source_positions),
            tuple(destination_positions),
            None if pattern is None else pattern.pattern_id,
            None if pattern is None else pattern.point_id,
        )

    if move is MoveType.INTRA_RELOCATE:
        choices = [robot for robot, route in enumerate(routes) if route]
        if not choices:
            return RawAttempt(move, None, "NO_NONEMPTY_ROUTE")
        robot = rng.choice(choices)
        source = rng.randrange(len(routes[robot]))
        destination = rng.randrange(len(routes[robot]) + 1)
        block = routes[robot][source]
        candidate = CandidateMove(
            key((_parent_id(block),), robot, robot, (source,), (destination,)),
            (block,),
        )
    elif move is MoveType.INTER_RELOCATE:
        choices = [robot for robot, route in enumerate(routes) if route]
        if not choices:
            return RawAttempt(move, None, "NO_NONEMPTY_ROUTE")
        source_robot = rng.choice(choices)
        destinations = [robot for robot in range(4) if robot != source_robot]
        destination_robot = rng.choice(destinations)
        source = rng.randrange(len(routes[source_robot]))
        destination = rng.randrange(len(routes[destination_robot]) + 1)
        block = routes[source_robot][source]
        candidate = CandidateMove(
            key(
                (_parent_id(block),),
                source_robot,
                destination_robot,
                (source,),
                (destination,),
            ),
            (block,),
        )
    elif move is MoveType.SWAP:
        if len(occurrences) < 2:
            return RawAttempt(move, None, "TOO_FEW_BLOCKS")
        first, second = rng.sample(occurrences, 2)
        candidate = CandidateMove(
            key(
                (_parent_id(first[2]), _parent_id(second[2])),
                first[0],
                second[0],
                (first[1],),
                (second[1],),
            ),
            (first[2], second[2]),
        )
    elif move is MoveType.TWO_OPT:
        choices = [robot for robot, route in enumerate(routes) if len(route) >= 2]
        if not choices:
            return RawAttempt(move, None, "NO_ROUTE_WITH_TWO_BLOCKS")
        robot = rng.choice(choices)
        lo, hi = sorted(rng.sample(range(len(routes[robot])), 2))
        affected = tuple(_parent_id(block) for block in routes[robot][lo : hi + 1])
        candidate = CandidateMove(key(affected, robot, robot, (lo, hi), ()))
    elif move is MoveType.TWO_OPT_STAR:
        choices = [
            pair
            for pair in ((0, 1), (2, 3))
            if len(routes[pair[0]]) + len(routes[pair[1]]) > 0
        ]
        if not choices:
            return RawAttempt(move, None, "NO_ACTIVE_SAME_RAIL_PAIR")
        source_robot, destination_robot = rng.choice(choices)
        source_cut = rng.randrange(len(routes[source_robot]) + 1)
        destination_cut = rng.randrange(len(routes[destination_robot]) + 1)
        affected = tuple(
            _parent_id(block)
            for block in (
                routes[source_robot][source_cut:]
                + routes[destination_robot][destination_cut:]
            )
        )
        if not affected:
            return RawAttempt(move, None, "EMPTY_SUFFIX_EXCHANGE")
        candidate = CandidateMove(
            key(
                affected,
                source_robot,
                destination_robot,
                (source_cut,),
                (destination_cut,),
            )
        )
    else:
        return RawAttempt(move, None, "INACTIVE_MOVE")
    return RawAttempt(move, candidate)


def generate_raw_attempts(
    solution: CanonicalSolution,
    config: ScientificConfig,
    *,
    m: int,
    seed: int,
    stats: SearchStats,
    moves: tuple[MoveType, ...] = ACTIVE_MOVE_TYPES,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
    start_attempt: int = 0,
) -> tuple[RawAttempt, ...]:
    started = time.perf_counter()
    mask = applicable_move_mask(
        solution,
        config,
        enable_two_opt_star=MoveType.TWO_OPT_STAR in moves,
        enable_x_split=enable_x_split,
        scope=scope,
    )
    applicable = tuple(move for move in moves if mask[move])
    for move in applicable:
        stats.applicable_by_move[move.value] += 1
    order = balanced_move_attempt_order(
        m + start_attempt, seed + solution.revision, applicable
    )
    ordinals = {move: 0 for move in moves}
    for move in order[:start_attempt]:
        ordinals[move] += 1
    attempts = []
    for move in order[start_attempt:]:
        stats.raw_attempts += 1
        stats.attempted_by_move[move.value] += 1
        attempt = make_raw_candidate(
            solution,
            config,
            move,
            seed=seed,
            ordinal=ordinals[move],
            enable_x_split=enable_x_split,
            scope=scope,
        )
        ordinals[move] += 1
        if attempt.candidate is not None:
            stats.constructed += 1
            stats.constructed_by_move[move.value] += 1
        elif attempt.rejection_reason:
            stats.rejection_reasons[attempt.rejection_reason] += 1
        attempts.append(attempt)
    stats.candidate_generation_time += time.perf_counter() - started
    return tuple(attempts)


def _direction_hints(
    solution: CanonicalSolution, directions: DirectionVectors
) -> dict[str, int]:
    return {
        block_id: directions[robot][position]
        for robot, route in enumerate(solution.routes)
        for position, block_id in enumerate(route.block_ids)
        if position < len(directions[robot])
    }


def _directed_proxy(
    solution: CanonicalSolution,
    hint_by_block: dict[str, int],
    config: ScientificConfig,
) -> float:
    blocks = block_map(solution, config)
    total = 0.0
    for route in solution.routes:
        for left, right in zip(route.block_ids, route.block_ids[1:]):
            _, end = oriented_endpoints(blocks[left], hint_by_block.get(left, 0))
            start, _ = oriented_endpoints(blocks[right], hint_by_block.get(right, 0))
            total += math.dist(end, start) / config.empty_speed
    return total


def _optional_split_count(solution, config: ScientificConfig) -> int:
    parents = {parent.parent_id: parent for parent in solution.parents}
    return sum(
        pattern.kind is not SplitKind.WHOLE
        and bool(whole_eligible_rails(parents[pattern.parent_id].start, parents[pattern.parent_id].end, config))
        for pattern in solution.patterns
    )


def _obvious_move_error(
    solution: CanonicalSolution, candidate: CandidateMove
) -> str | None:
    key = candidate.key
    routes = [route.block_ids for route in solution.routes]
    source = key.source_robot_id
    destination = key.destination_robot_id
    if key.move_type in (MoveType.INTRA_RELOCATE, MoveType.INTER_RELOCATE):
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            return "bad relocate shape"
        src, dst = key.source_positions[0], key.destination_positions[0]
        if src >= len(routes[source]) or dst > len(routes[destination]):
            return "relocate route index out of range"
        if candidate.block_ids and candidate.block_ids != (routes[source][src],):
            return "relocate block identity mismatch"
    elif key.move_type is MoveType.SWAP:
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            return "bad swap shape"
        src, dst = key.source_positions[0], key.destination_positions[0]
        if src >= len(routes[source]) or dst >= len(routes[destination]):
            return "swap route index out of range"
        expected = (routes[source][src], routes[destination][dst])
        if candidate.block_ids and candidate.block_ids != expected:
            return "swap block identity mismatch"
    elif key.move_type is MoveType.TWO_OPT:
        if source is None or len(key.source_positions) != 2:
            return "bad 2-opt shape"
        lo, hi = key.source_positions
        if not (0 <= lo <= hi < len(routes[source])):
            return "2-opt route index out of range"
    elif key.move_type is MoveType.TWO_OPT_STAR:
        if (
            source is None
            or destination is None
            or source == destination
            or {source, destination} not in ({0, 1}, {2, 3})
            or len(key.source_positions) != 1
            or len(key.destination_positions) != 1
        ):
            return "bad same-rail 2-opt* shape"
        source_cut = key.source_positions[0]
        destination_cut = key.destination_positions[0]
        if not (
            0 <= source_cut <= len(routes[source])
            and 0 <= destination_cut <= len(routes[destination])
        ):
            return "2-opt* cut index out of range"
    else:
        return "retired dynamic or unsupported move"
    return None


def screen_raw_attempts(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    attempts: tuple[RawAttempt, ...],
    config: ScientificConfig,
    stats: SearchStats,
    *,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
    observer=None,
) -> tuple[ScreenedCandidate, ...]:
    started = time.perf_counter()
    seen_keys: set[CandidateKey] = set()
    seen_solutions: set[str] = set()
    direction_hints = _direction_hints(current, current_directions)
    current_proxy = _directed_proxy(current, direction_hints, config)
    current_splits = _optional_split_count(current, config)
    result = []
    x_validator = (
        finite_x_split_validator(current.parents, config) if enable_x_split else None
    )
    def emit(attempt, status, reason=None, solution=None):
        if observer is not None:
            observer("atomic_attempt", (attempt, status, reason, solution))

    for attempt in attempts:
        candidate = attempt.candidate
        if candidate is None:
            emit(attempt, "RAW_REJECTED", attempt.rejection_reason)
            continue
        if attempt.move_type not in TRACKED_MOVE_TYPES:
            stats.rejection_reasons["RETIRED_DYNAMIC_MOVE"] += 1
            emit(attempt, "CONSTRUCTION_REJECTED", "RETIRED_DYNAMIC_MOVE")
            continue
        move_name = attempt.move_type.value
        if candidate.key in seen_keys:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["DUPLICATE_CANDIDATE_KEY"] += 1
            emit(attempt, "DUPLICATE", "DUPLICATE_CANDIDATE_KEY")
            continue
        seen_keys.add(candidate.key)
        obvious_error = _obvious_move_error(current, candidate)
        if obvious_error is not None:
            stats.rejection_reasons[f"CHEAP_INVALID:{obvious_error}"] += 1
            emit(attempt, "CONSTRUCTION_REJECTED", obvious_error)
            continue
        try:
            provisional = apply_candidate(
                current,
                candidate,
                config,
                x_split_validator=x_validator,
                scope=scope,
            )
            blocks = block_map(provisional, config, scope=scope)
            if any(
                not robot_is_eligible(blocks[block_id], route.robot_id, config)
                for route in provisional.routes
                for block_id in route.block_ids
            ):
                raise ValueError("robot eligibility failure")
        except (ValueError, IndexError, KeyError) as error:
            stats.rejection_reasons[f"CHEAP_INVALID:{error}"] += 1
            emit(attempt, "CONSTRUCTION_REJECTED", str(error))
            continue
        if provisional.canonical_hash == current.canonical_hash:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["IDENTITY_MOVE"] += 1
            emit(attempt, "IDENTITY", "IDENTITY_MOVE", provisional)
            continue
        if provisional.canonical_hash in seen_solutions:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["DUPLICATE_CANONICAL_SOLUTION"] += 1
            emit(attempt, "DUPLICATE", "DUPLICATE_CANONICAL_SOLUTION", provisional)
            continue
        seen_solutions.add(provisional.canonical_hash)
        provisional_blocks = block_map(provisional, config, scope=scope)
        loads = [
            sum(config.process_time(provisional_blocks[item].length) for item in route.block_ids)
            for route in provisional.routes
        ]
        projected = max(loads, default=0.0)
        travel_delta = _directed_proxy(provisional, direction_hints, config) - current_proxy
        split_delta = _optional_split_count(provisional, config) - current_splits
        order = candidate_key_order(candidate.key)
        score = (projected, travel_delta, split_delta, order)
        result.append(
            ScreenedCandidate(
                candidate,
                provisional,
                score,
                projected,
                travel_delta,
                split_delta,
            )
        )
        emit(attempt, "VALID", solution=provisional)
        stats.cheap_feasible += 1
        stats.cheap_valid_by_move[move_name] += 1
    stats.cheap_screen_time += time.perf_counter() - started
    return tuple(result)
