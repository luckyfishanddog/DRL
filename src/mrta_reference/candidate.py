from __future__ import annotations

from typing import Iterable

from .geometry import XSplitValidator
from .model import CandidateMove, CanonicalSolution, FormalScope, MoveType, Route, ScientificConfig, SplitKind, SplitPattern
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
    scope: FormalScope | None = None,
) -> CanonicalSolution:
    if candidate.key.current_solution_revision != solution.revision:
        raise ValueError("candidate revision does not match current solution")
    routes = [list(route.block_ids) for route in solution.routes]
    key = candidate.key
    source = key.source_robot_id
    destination = key.destination_robot_id
    if candidate.split_pattern is None:
        if key.split_pattern_id is not None or key.split_point_id is not None:
            raise ValueError('candidate key declares split identity without a split pattern')
    elif (key.split_pattern_id != candidate.split_pattern.pattern_id
          or key.split_point_id != candidate.split_pattern.point_id):
        raise ValueError('candidate split identity does not match CandidateKey')
    if candidate.split_pattern is not None:
        raise ValueError('dynamic split search is retired; task boundaries cannot change')

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
    else:
        raise ValueError(f"unsupported move type: {key.move_type}")

    return canonicalize(
        solution.parents,
        solution.patterns,
        tuple(Route(robot, tuple(routes[robot])) for robot in range(4)),
        config,
        revision=solution.revision + 1,
        x_split_validator=x_split_validator,
        scope=scope,
    )


def deduplicate_candidates(
    solution: CanonicalSolution,
    candidates: Iterable[CandidateMove],
    config: ScientificConfig,
    *,
    x_split_validator: XSplitValidator | None = None,
    scope: FormalScope | None = None,
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
            scope=scope,
        )
        if provisional.canonical_hash in seen_hashes:
            continue
        seen_hashes.add(provisional.canonical_hash)
        result.append((candidate, provisional))
    return tuple(result)
