from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from mrta_reference.candidate import apply_candidate
from mrta_reference.geometry import (
    blocks_for_pattern,
    generate_y_split_patterns,
    oriented_endpoints,
    robot_is_eligible,
    whole_eligible_rails,
)
from mrta_reference.model import (
    CandidateKey,
    CandidateMove,
    CanonicalSolution,
    MoveType,
    ScientificConfig,
    SplitKind,
    robot_rail,
)
from mrta_reference.solution import block_map

from .direction import DirectionVectors
from .stats import ACTIVE_MOVE_TYPES, SearchStats


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
    solution: CanonicalSolution, config: ScientificConfig
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

    activate = False
    deactivate = False
    switch = False
    child_robots = {
        block_id: robot
        for robot, route in enumerate(routes)
        for block_id in route
    }
    for parent_id, pattern in patterns.items():
        legal = generate_y_split_patterns(parents[parent_id], config)
        if pattern.kind is SplitKind.WHOLE and legal:
            activate = True
        elif pattern.kind is SplitKind.Y_SPLIT:
            if not pattern.mandatory and whole_eligible_rails(
                parents[parent_id].start, parents[parent_id].end, config
            ):
                deactivate = True
            for alternative in legal:
                if alternative.pattern_id == pattern.pattern_id:
                    continue
                alternative_blocks = blocks_for_pattern(
                    parents[parent_id], alternative, config
                )
                if all(
                    block.block_id in child_robots
                    and robot_is_eligible(
                        block, child_robots[block.block_id], config
                    )
                    for block in alternative_blocks
                ):
                    switch = True
                    break

    return {
        MoveType.INTRA_RELOCATE: intra,
        MoveType.INTER_RELOCATE: inter,
        MoveType.SWAP: swap,
        MoveType.TWO_OPT: intra,
        MoveType.SPLIT_ACTIVATE: activate,
        MoveType.SPLIT_DEACTIVATE: deactivate,
        MoveType.SPLIT_POINT_SWITCH: switch,
    }


def _parent_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def _local_rng(seed: int, revision: int, move: MoveType, ordinal: int) -> random.Random:
    move_index = ACTIVE_MOVE_TYPES.index(move)
    return random.Random(seed + 1_000_003 * revision + 10_007 * move_index + 97 * ordinal)


def make_raw_candidate(
    solution: CanonicalSolution,
    config: ScientificConfig,
    move: MoveType,
    *,
    seed: int,
    ordinal: int,
) -> RawAttempt:
    rng = _local_rng(seed, solution.revision, move, ordinal)
    routes = [route.block_ids for route in solution.routes]
    occurrences = [(robot, position, block) for robot, route in enumerate(routes) for position, block in enumerate(route)]
    parents = {parent.parent_id: parent for parent in solution.parents}
    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}

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
    elif move is MoveType.SPLIT_ACTIVATE:
        choices = []
        for parent_id, pattern in patterns.items():
            if pattern.kind is not SplitKind.WHOLE:
                continue
            alternatives = generate_y_split_patterns(parents[parent_id], config)
            if alternatives:
                location = next(
                    (
                        (robot, position)
                        for robot, route in enumerate(routes)
                        for position, block in enumerate(route)
                        if block == f"{parent_id}::whole"
                    ),
                    None,
                )
                if location is not None:
                    choices.append((parent_id, alternatives, location))
        if not choices:
            return RawAttempt(move, None, "NO_OPTIONAL_Y_ACTIVATION")
        parent_id, alternatives, (source_robot, source_position) = rng.choice(choices)
        pattern = rng.choice(alternatives)
        destination_robot = rng.randrange(4)
        destination_position = rng.randrange(len(routes[destination_robot]) + 1)
        candidate = CandidateMove(
            key(
                (parent_id,),
                source_robot,
                destination_robot,
                (source_position,),
                (destination_position,),
                pattern,
            ),
            (f"{parent_id}::whole",),
            pattern,
        )
    elif move is MoveType.SPLIT_DEACTIVATE:
        choices = [
            pattern
            for pattern in solution.patterns
            if pattern.kind is SplitKind.Y_SPLIT and not pattern.mandatory
        ]
        if not choices:
            return RawAttempt(move, None, "NO_OPTIONAL_Y_DEACTIVATION")
        pattern = rng.choice(choices)
        eligible_robots = [
            robot
            for robot in range(4)
            if robot_rail(robot)
            in whole_eligible_rails(
                parents[pattern.parent_id].start,
                parents[pattern.parent_id].end,
                config,
            )
        ]
        destination_robot = rng.choice(eligible_robots)
        destination_position = rng.randrange(len(routes[destination_robot]) + 1)
        candidate = CandidateMove(
            key(
                (pattern.parent_id,),
                None,
                destination_robot,
                (),
                (destination_position,),
            )
        )
    elif move is MoveType.SPLIT_POINT_SWITCH:
        choices = []
        for pattern in solution.patterns:
            if pattern.kind is not SplitKind.Y_SPLIT:
                continue
            alternatives = tuple(
                item
                for item in generate_y_split_patterns(parents[pattern.parent_id], config)
                if item.pattern_id != pattern.pattern_id
            )
            if alternatives:
                choices.append((pattern, alternatives))
        if not choices:
            return RawAttempt(move, None, "NO_Y_POINT_SWITCH")
        old, alternatives = rng.choice(choices)
        pattern = rng.choice(alternatives)
        candidate = CandidateMove(key((old.parent_id,), pattern=pattern), split_pattern=pattern)
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
) -> tuple[RawAttempt, ...]:
    started = time.perf_counter()
    mask = applicable_move_mask(solution, config)
    applicable = tuple(move for move in ACTIVE_MOVE_TYPES if mask[move])
    for move in applicable:
        stats.applicable_by_move[move.value] += 1
    order = balanced_move_attempt_order(
        m, seed + solution.revision, applicable
    )
    ordinals = {move: 0 for move in ACTIVE_MOVE_TYPES}
    attempts = []
    for move in order:
        stats.raw_attempts += 1
        stats.attempted_by_move[move.value] += 1
        attempt = make_raw_candidate(
            solution,
            config,
            move,
            seed=seed,
            ordinal=ordinals[move],
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
    elif key.move_type is MoveType.SPLIT_ACTIVATE:
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            return "bad split activation shape"
        src, dst = key.source_positions[0], key.destination_positions[0]
        if src >= len(routes[source]) or dst > len(routes[destination]):
            return "split activation route index out of range"
    elif key.move_type is MoveType.SPLIT_DEACTIVATE:
        if destination is None or len(key.destination_positions) != 1:
            return "bad split deactivation shape"
        parent_id = key.affected_parent_ids[0]
        child_ids = {f"{parent_id}::0", f"{parent_id}::1"}
        remaining = sum(block not in child_ids for block in routes[destination])
        if key.destination_positions[0] > remaining:
            return "split deactivation route index out of range"
    return None


def screen_raw_attempts(
    current: CanonicalSolution,
    current_directions: DirectionVectors,
    attempts: tuple[RawAttempt, ...],
    config: ScientificConfig,
    stats: SearchStats,
) -> tuple[ScreenedCandidate, ...]:
    started = time.perf_counter()
    seen_keys: set[CandidateKey] = set()
    seen_solutions: set[str] = set()
    direction_hints = _direction_hints(current, current_directions)
    current_proxy = _directed_proxy(current, direction_hints, config)
    current_splits = _optional_split_count(current, config)
    result = []
    for attempt in attempts:
        candidate = attempt.candidate
        if candidate is None:
            continue
        move_name = attempt.move_type.value
        if candidate.key in seen_keys:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["DUPLICATE_CANDIDATE_KEY"] += 1
            continue
        seen_keys.add(candidate.key)
        obvious_error = _obvious_move_error(current, candidate)
        if obvious_error is not None:
            stats.rejection_reasons[f"CHEAP_INVALID:{obvious_error}"] += 1
            continue
        if (
            candidate.split_pattern is not None
            and candidate.split_pattern.kind is SplitKind.X_SPLIT
        ):
            stats.rejection_reasons["X_SPLIT_FORBIDDEN"] += 1
            continue
        if candidate.key.move_type is MoveType.SPLIT_DEACTIVATE:
            pattern = next(
                item
                for item in current.patterns
                if item.parent_id == candidate.key.affected_parent_ids[0]
            )
            if pattern.mandatory:
                stats.rejection_reasons["MANDATORY_Y_CANNOT_DEACTIVATE"] += 1
                continue
        try:
            provisional = apply_candidate(current, candidate, config)
            blocks = block_map(provisional, config)
            if any(
                not robot_is_eligible(blocks[block_id], route.robot_id, config)
                for route in provisional.routes
                for block_id in route.block_ids
            ):
                raise ValueError("robot eligibility failure")
        except (ValueError, IndexError, KeyError) as error:
            stats.rejection_reasons[f"CHEAP_INVALID:{error}"] += 1
            continue
        if provisional.canonical_hash == current.canonical_hash:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["IDENTITY_MOVE"] += 1
            continue
        if provisional.canonical_hash in seen_solutions:
            stats.duplicates += 1
            stats.duplicate_by_move[move_name] += 1
            stats.rejection_reasons["DUPLICATE_CANONICAL_SOLUTION"] += 1
            continue
        seen_solutions.add(provisional.canonical_hash)
        provisional_blocks = block_map(provisional, config)
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
        stats.cheap_feasible += 1
        stats.cheap_valid_by_move[move_name] += 1
    stats.cheap_screen_time += time.perf_counter() - started
    return tuple(result)
