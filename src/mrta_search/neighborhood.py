from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time

from mrta_reference.candidate import apply_candidate
from mrta_reference.geometry import (
    blocks_for_pattern,
    build_legal_pattern_catalog,
    finite_x_split_validator,
    frozen_handover_centers,
    generate_x_split_patterns,
    generate_y_split_patterns,
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

    activate = False
    deactivate = False
    switch = False
    child_robots = {
        block_id: robot
        for robot, route in enumerate(routes)
        for block_id in route
    }
    catalog = (
        build_legal_pattern_catalog(solution.parents, config, scope)
        if scope is not None
        else None
    )
    x_up, x_low = frozen_handover_centers(solution.parents, config)
    for parent_id, pattern in patterns.items():
        options = None if catalog is None else catalog[parent_id]
        y_legal = tuple(
            item for item in (options or generate_y_split_patterns(parents[parent_id], config))
            if item.kind is SplitKind.Y_SPLIT
        )
        x_legal = (
            tuple(item for item in options if item.kind is SplitKind.X_SPLIT)
            if options is not None
            else tuple(
                    candidate
                    for rail, center in ((robot_rail(0), x_up), (robot_rail(2), x_low))
                    for candidate in generate_x_split_patterns(
                        parents[parent_id], center, config, rail=rail
                    )
                )
            if enable_x_split
            else ()
        )
        legal = y_legal + x_legal
        if pattern.kind is SplitKind.WHOLE and legal:
            activate = True
        elif pattern.kind is not SplitKind.WHOLE:
            if not pattern.mandatory and whole_eligible_rails(
                parents[parent_id].start, parents[parent_id].end, config
            ):
                deactivate = True
            alternatives = (
                y_legal
                if pattern.kind is SplitKind.Y_SPLIT
                else tuple(item for item in x_legal if item.rail is pattern.rail)
            )
            for alternative in alternatives:
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
        MoveType.TWO_OPT_STAR: enable_two_opt_star and any(
            len(routes[left]) + len(routes[right]) > 0
            for left, right in ((0, 1), (2, 3))
        ),
        MoveType.SPLIT_ACTIVATE: activate,
        MoveType.SPLIT_DEACTIVATE: deactivate,
        MoveType.SPLIT_POINT_SWITCH: switch,
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
    rng = _local_rng(seed, solution.revision, move, ordinal)
    routes = [route.block_ids for route in solution.routes]
    occurrences = [(robot, position, block) for robot, route in enumerate(routes) for position, block in enumerate(route)]
    parents = {parent.parent_id: parent for parent in solution.parents}
    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}
    catalog = (
        build_legal_pattern_catalog(solution.parents, config, scope)
        if scope is not None
        else None
    )
    x_up, x_low = frozen_handover_centers(solution.parents, config)

    def x_patterns(parent_id: str) -> tuple[SplitPattern, ...]:
        if not enable_x_split:
            return ()
        if catalog is not None:
            return tuple(
                item for item in catalog[parent_id] if item.kind is SplitKind.X_SPLIT
            )
        parent = parents[parent_id]
        return tuple(
            candidate
            for rail, center in ((robot_rail(0), x_up), (robot_rail(2), x_low))
            for candidate in generate_x_split_patterns(parent, center, config, rail=rail)
        )

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
    elif move is MoveType.SPLIT_ACTIVATE:
        choices = []
        for parent_id, pattern in patterns.items():
            if pattern.kind is not SplitKind.WHOLE:
                continue
            alternatives = (
                tuple(
                    item for item in catalog[parent_id]
                    if item.kind is not SplitKind.WHOLE
                )
                if catalog is not None
                else generate_y_split_patterns(parents[parent_id], config) + x_patterns(parent_id)
            )
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
        families = tuple(
            kind
            for kind in (SplitKind.Y_SPLIT, SplitKind.X_SPLIT)
            if any(any(item.kind is kind for item in choice[1]) for choice in choices)
        )
        target_kind = families[ordinal % len(families)]
        family_choices = [
            choice for choice in choices
            if any(item.kind is target_kind for item in choice[1])
        ]
        parent_id, alternatives, (source_robot, source_position) = rng.choice(family_choices)
        pattern = rng.choice(tuple(item for item in alternatives if item.kind is target_kind))
        if pattern.kind is SplitKind.X_SPLIT:
            assert pattern.rail is not None
            left_robot, right_robot = ((0, 1) if pattern.rail.value == "UPPER" else (2, 3))
            left_length = len(routes[left_robot]) - int(source_robot == left_robot)
            right_length = len(routes[right_robot]) - int(source_robot == right_robot)
            destination_robot = left_robot
            destination_positions = (
                rng.randrange(left_length + 1),
                rng.randrange(right_length + 1),
            )
        else:
            destination_robot = rng.randrange(4)
            destination_positions = (rng.randrange(len(routes[destination_robot]) + 1),)
        candidate = CandidateMove(
            key(
                (parent_id,),
                source_robot,
                destination_robot,
                (source_position,),
                destination_positions,
                pattern,
            ),
            (f"{parent_id}::whole",),
            pattern,
        )
    elif move is MoveType.SPLIT_DEACTIVATE:
        choices = [
            pattern
            for pattern in solution.patterns
            if pattern.kind is not SplitKind.WHOLE and not pattern.mandatory
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
            if pattern.kind is SplitKind.WHOLE:
                continue
            family = (
                tuple(
                    item for item in catalog[pattern.parent_id]
                    if item.kind is not SplitKind.WHOLE
                )
                if catalog is not None
                else (
                    generate_y_split_patterns(parents[pattern.parent_id], config)
                    if pattern.kind is SplitKind.Y_SPLIT
                    else tuple(item for item in x_patterns(pattern.parent_id) if item.rail is pattern.rail)
                )
            )
            alternatives = tuple(
                item
                for item in family
                if item.pattern_id != pattern.pattern_id
            )
            if alternatives:
                choices.append((pattern, alternatives))
        if not choices:
            return RawAttempt(move, None, "NO_Y_POINT_SWITCH")
        families = tuple(
            kind
            for kind in (SplitKind.Y_SPLIT, SplitKind.X_SPLIT)
            if any(any(item.kind is kind for item in choice[1]) for choice in choices)
        )
        target_kind = families[ordinal % len(families)]
        family_choices = [
            choice for choice in choices
            if any(item.kind is target_kind for item in choice[1])
        ]
        old, alternatives = rng.choice(family_choices)
        pattern = rng.choice(tuple(item for item in alternatives if item.kind is target_kind))
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
    moves: tuple[MoveType, ...] = ACTIVE_MOVE_TYPES,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
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
        m, seed + solution.revision, applicable
    )
    ordinals = {move: 0 for move in moves}
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
    elif key.move_type is MoveType.SPLIT_ACTIVATE:
        required_destinations = (
            2
            if candidate.split_pattern is not None
            and candidate.split_pattern.kind is SplitKind.X_SPLIT
            else 1
        )
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != required_destinations:
            return "bad split activation shape"
        src = key.source_positions[0]
        if src >= len(routes[source]):
            return "split activation route index out of range"
        if required_destinations == 2:
            assert candidate.split_pattern is not None and candidate.split_pattern.rail is not None
            left_robot, right_robot = ((0, 1) if candidate.split_pattern.rail.value == "UPPER" else (2, 3))
            limits = (
                len(routes[left_robot]) - int(source == left_robot),
                len(routes[right_robot]) - int(source == right_robot),
            )
            if any(position > limit for position, limit in zip(key.destination_positions, limits)):
                return "split activation route index out of range"
        elif key.destination_positions[0] > len(routes[destination]):
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
    *,
    enable_x_split: bool = False,
    scope: FormalScope | None = None,
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
            and not enable_x_split
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
        stats.cheap_feasible += 1
        stats.cheap_valid_by_move[move_name] += 1
    stats.cheap_screen_time += time.perf_counter() - started
    return tuple(result)
