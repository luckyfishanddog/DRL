from __future__ import annotations

from typing import Iterable

from .geometry import XSplitValidator
from .model import CandidateMove, CanonicalSolution, MoveType, Route, ScientificConfig, SplitKind, SplitPattern
from .solution import canonicalize


def _parent_id(block_id: str) -> str:
    return block_id.split("::", 1)[0]


def _require_affected(candidate: CandidateMove, actual_parent_ids) -> None:
    actual = tuple(sorted(set(actual_parent_ids)))
    if actual != candidate.key.affected_parent_ids:
        raise ValueError(
            f"candidate affected-parent identity mismatch: key={candidate.key.affected_parent_ids}, actual={actual}"
        )


def apply_candidate(
    solution: CanonicalSolution,
    candidate: CandidateMove,
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
) -> CanonicalSolution:
    if candidate.key.current_solution_revision != solution.revision:
        raise ValueError("candidate revision does not match current solution")
    routes = [list(route.block_ids) for route in solution.routes]
    key = candidate.key
    source = key.source_robot_id
    destination = key.destination_robot_id
    if candidate.split_pattern is None:
        if key.split_pattern_id is not None or key.split_point_id is not None:
            raise ValueError("candidate key declares split identity without a split pattern")
    elif (
        key.split_pattern_id != candidate.split_pattern.pattern_id
        or key.split_point_id != candidate.split_pattern.point_id
    ):
        raise ValueError("candidate split identity does not match CandidateKey")

    if key.move_type in (MoveType.INTRA_RELOCATE, MoveType.INTER_RELOCATE):
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            raise ValueError("relocate requires exact source/destination robot and position")
        source_position, destination_position = key.source_positions[0], key.destination_positions[0]
        block = routes[source].pop(source_position)
        _require_affected(candidate, (_parent_id(block),))
        if candidate.block_ids and candidate.block_ids != (block,):
            raise ValueError("candidate block identity mismatch")
        if source == destination and destination_position > source_position:
            destination_position -= 1
        routes[destination].insert(destination_position, block)
    elif key.move_type is MoveType.SWAP:
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            raise ValueError("swap requires exact endpoints")
        a, b = key.source_positions[0], key.destination_positions[0]
        _require_affected(candidate, (_parent_id(routes[source][a]), _parent_id(routes[destination][b])))
        routes[source][a], routes[destination][b] = routes[destination][b], routes[source][a]
    elif key.move_type is MoveType.TWO_OPT:
        if source is None or len(key.source_positions) != 2:
            raise ValueError("2-opt requires a robot and inclusive bounds")
        lo, hi = key.source_positions
        if not (0 <= lo <= hi < len(routes[source])):
            raise ValueError("invalid 2-opt bounds")
        _require_affected(candidate, (_parent_id(block) for block in routes[source][lo : hi + 1]))
        routes[source][lo : hi + 1] = reversed(routes[source][lo : hi + 1])
    elif key.move_type is MoveType.TWO_OPT_STAR:
        if source is None or destination is None or len(key.source_positions) != 1 or len(key.destination_positions) != 1:
            raise ValueError("2-opt* requires two exact cut positions")
        if source == destination:
            raise ValueError("2-opt* requires two distinct robot routes")
        a, b = key.source_positions[0], key.destination_positions[0]
        _require_affected(
            candidate,
            (_parent_id(block) for block in routes[source][a:] + routes[destination][b:]),
        )
        routes[source][a:], routes[destination][b:] = routes[destination][b:], routes[source][a:]
    elif key.move_type is MoveType.SPLIT_ACTIVATE:
        pattern = candidate.split_pattern
        if pattern is None or pattern.kind is SplitKind.WHOLE or source is None or destination is None:
            raise ValueError("split activation requires a concrete non-WHOLE pattern and destinations")
        required_destinations = 2 if pattern.kind is SplitKind.X_SPLIT else 1
        if len(key.source_positions) != 1 or len(key.destination_positions) != required_destinations:
            raise ValueError("split activation requires exact positions")
        _require_affected(candidate, (pattern.parent_id,))
        whole = f"{pattern.parent_id}::whole"
        if routes[source][key.source_positions[0]] != whole:
            raise ValueError("split activation source is not the parent WHOLE block")
        if pattern.kind is SplitKind.X_SPLIT:
            routes[source].pop(key.source_positions[0])
            assert pattern.rail is not None
            left_robot, right_robot = ((0, 1) if pattern.rail.value == "UPPER" else (2, 3))
            routes[left_robot].insert(key.destination_positions[0], f"{pattern.parent_id}::0")
            routes[right_robot].insert(key.destination_positions[1], f"{pattern.parent_id}::1")
        else:
            routes[source][key.source_positions[0]] = f"{pattern.parent_id}::0"
            routes[destination].insert(key.destination_positions[0], f"{pattern.parent_id}::1")
    elif key.move_type is MoveType.SPLIT_DEACTIVATE:
        parent_id = key.affected_parent_ids[0]
        _require_affected(candidate, (parent_id,))
        occurrences: list[tuple[int, int]] = []
        for robot, route in enumerate(routes):
            for position, block_id in enumerate(route):
                if block_id in (f"{parent_id}::0", f"{parent_id}::1"):
                    occurrences.append((robot, position))
        if len(occurrences) != 2 or destination is None or len(key.destination_positions) != 1:
            raise ValueError("split deactivation requires both children and one exact destination")
        for robot, position in sorted(occurrences, reverse=True):
            routes[robot].pop(position)
        routes[destination].insert(key.destination_positions[0], f"{parent_id}::whole")
    elif key.move_type is MoveType.SPLIT_POINT_SWITCH:
        pattern = candidate.split_pattern
        if pattern is None or pattern.kind is SplitKind.WHOLE:
            raise ValueError("split point switch requires a concrete split pattern")
        _require_affected(candidate, (pattern.parent_id,))
        # Canonical child identities remain rank-based; only geometry changes.
    else:
        raise ValueError(f"unsupported move type: {key.move_type}")

    patterns = {pattern.parent_id: pattern for pattern in solution.patterns}
    if key.move_type in (MoveType.SPLIT_ACTIVATE, MoveType.SPLIT_POINT_SWITCH):
        assert candidate.split_pattern is not None
        patterns[candidate.split_pattern.parent_id] = candidate.split_pattern
    elif key.move_type is MoveType.SPLIT_DEACTIVATE:
        parent_id = key.affected_parent_ids[0]
        patterns[parent_id] = SplitPattern(parent_id, SplitKind.WHOLE)

    return canonicalize(
        solution.parents,
        tuple(patterns.values()),
        tuple(Route(robot, tuple(routes[robot])) for robot in range(4)),
        config,
        revision=solution.revision + 1,
        x_split_validator=x_split_validator,
    )


def deduplicate_candidates(
    solution: CanonicalSolution,
    candidates: Iterable[CandidateMove],
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
) -> tuple[tuple[CandidateMove, CanonicalSolution], ...]:
    seen_keys = set()
    seen_hashes = set()
    result = []
    for candidate in candidates:
        if candidate.key in seen_keys:
            continue
        seen_keys.add(candidate.key)
        provisional = apply_candidate(
            solution,
            candidate,
            config,
            x_split_validator=x_split_validator,
        )
        if provisional.canonical_hash in seen_hashes:
            continue
        seen_hashes.add(provisional.canonical_hash)
        result.append((candidate, provisional))
    return tuple(result)
