from __future__ import annotations

import pytest

from mrta_reference.candidate import apply_candidate, deduplicate_candidates
from mrta_reference.model import (
    CandidateKey,
    CandidateMove,
    MoveType,
    OperationKind,
    ParentWeld,
    Route,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.solution import canonicalize
from mrta_reference.scheduler import reference_schedule


CONFIG = ScientificConfig()


def _whole_solution():
    parents = tuple(
        ParentWeld(name, (x, 8.0), (x + 1.0, 8.0))
        for name, x in (("a", 1.0), ("b", 3.0), ("c", 10.0), ("d", 12.0))
    )
    patterns = tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents)
    return canonicalize(parents, patterns, {0: ("a::whole", "b::whole"), 1: ("c::whole", "d::whole")}, CONFIG)


def _key(move_type, affected, src, dst, src_pos, dst_pos, *, revision=0, pattern=None):
    return CandidateKey(
        revision,
        move_type,
        tuple(sorted(affected)),
        src,
        dst,
        tuple(src_pos),
        tuple(dst_pos),
        pattern.pattern_id if pattern else None,
        pattern.point_id if pattern else None,
    )


def test_parent_has_exactly_one_mutually_exclusive_pattern_and_complete_coverage() -> None:
    parent = ParentWeld("p", (0, 8), (1, 8))
    with pytest.raises(ValueError, match="more than one"):
        canonicalize(
            (parent,),
            (SplitPattern("p", SplitKind.WHOLE), SplitPattern("p", SplitKind.X_SPLIT, 0.5, "x")),
            {0: ("p::whole",)},
            CONFIG,
        )
    with pytest.raises(ValueError, match="coverage"):
        canonicalize((parent,), (SplitPattern("p", SplitKind.WHOLE),), {}, CONFIG)


def test_same_robot_consecutive_children_canonicalize_to_whole_before_processing() -> None:
    parent = ParentWeld("p", (1, 8), (3, 8))
    solution = canonicalize(
        (parent,),
        (SplitPattern("p", SplitKind.X_SPLIT, 0.5, "given"),),
        {0: ("p::0", "p::1")},
        CONFIG,
    )
    assert solution.patterns == (SplitPattern("p", SplitKind.WHOLE),)
    assert solution.routes[0].block_ids == ("p::whole",)
    schedule = reference_schedule(solution, CONFIG)
    assert sum(operation.kind is OperationKind.SETUP for operation in schedule.operations) == 1
    assert sum(operation.kind is OperationKind.POST for operation in schedule.operations) == 1


def test_mandatory_y_handover_is_derived_and_x_cannot_replace_it() -> None:
    parent = ParentWeld("cross", (2.0, 5.0), (2.0, 7.0))
    y_pattern = SplitPattern("cross", SplitKind.Y_SPLIT, 0.5, "BY_CENTER", False)
    solution = canonicalize(
        (parent,), (y_pattern,), {0: ("cross::1",), 2: ("cross::0",)}, CONFIG
    )
    assert solution.patterns[0].mandatory is True
    with pytest.raises(ValueError, match="requires Y_SPLIT"):
        canonicalize(
            (parent,),
            (SplitPattern("cross", SplitKind.X_SPLIT, 0.5, "x"),),
            {0: ("cross::0",), 1: ("cross::1",)},
            CONFIG,
        )


def test_canonical_serialization_and_hash_ignore_revision_but_not_logic() -> None:
    solution = _whole_solution()
    replay_revision = canonicalize(
        solution.parents, solution.patterns, solution.routes, CONFIG, revision=99
    )
    assert solution.canonical_json == replay_revision.canonical_json
    assert solution.canonical_hash == replay_revision.canonical_hash


def test_candidate_replay_is_deterministic() -> None:
    solution = _whole_solution()
    candidate = CandidateMove(
        _key(MoveType.INTER_RELOCATE, ("b",), 0, 1, (1,), (1,)),
        ("b::whole",),
    )
    first = apply_candidate(solution, candidate, CONFIG)
    second = apply_candidate(solution, candidate, CONFIG)
    assert first.canonical_json == second.canonical_json
    assert first.canonical_hash == second.canonical_hash
    assert first.routes[0].block_ids == ("a::whole",)
    assert first.routes[1].block_ids == ("c::whole", "b::whole", "d::whole")


def test_all_route_move_types_are_complete_atomic_transforms() -> None:
    solution = _whole_solution()
    intra = CandidateMove(_key(MoveType.INTRA_RELOCATE, ("a",), 0, 0, (0,), (2,)), ("a::whole",))
    assert apply_candidate(solution, intra, CONFIG).routes[0].block_ids == ("b::whole", "a::whole")
    swap = CandidateMove(_key(MoveType.SWAP, ("a", "c"), 0, 1, (0,), (0,)))
    swapped = apply_candidate(solution, swap, CONFIG)
    assert swapped.routes[0].block_ids[0] == "c::whole"
    two_opt = CandidateMove(_key(MoveType.TWO_OPT, ("a", "b"), 0, None, (0, 1), ()))
    assert apply_candidate(solution, two_opt, CONFIG).routes[0].block_ids == ("b::whole", "a::whole")
    star = CandidateMove(_key(MoveType.TWO_OPT_STAR, ("b", "d"), 0, 1, (1,), (1,)))
    starred = apply_candidate(solution, star, CONFIG)
    assert starred.routes[0].block_ids == ("a::whole", "d::whole")
    assert starred.routes[1].block_ids == ("c::whole", "b::whole")


def test_split_activate_switch_deactivate_and_canonical_duplicate_detection() -> None:
    solution = _whole_solution()
    activated_pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.4, "x40")
    activate = CandidateMove(
        _key(MoveType.SPLIT_ACTIVATE, ("a",), 0, 1, (0,), (0,), pattern=activated_pattern),
        split_pattern=activated_pattern,
    )
    split_solution = apply_candidate(solution, activate, CONFIG)
    assert split_solution.routes[0].block_ids == ("a::0", "b::whole")
    assert split_solution.routes[1].block_ids[0] == "a::1"

    switched_pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.6, "x60")
    switch = CandidateMove(
        _key(MoveType.SPLIT_POINT_SWITCH, ("a",), None, None, (), (), revision=1, pattern=switched_pattern),
        split_pattern=switched_pattern,
    )
    switched = apply_candidate(split_solution, switch, CONFIG)
    assert next(pattern for pattern in switched.patterns if pattern.parent_id == "a") == switched_pattern

    deactivate = CandidateMove(
        _key(MoveType.SPLIT_DEACTIVATE, ("a",), None, 0, (), (0,), revision=2)
    )
    whole_again = apply_candidate(switched, deactivate, CONFIG)
    assert whole_again.canonical_hash == solution.canonical_hash

    noop_a = CandidateMove(_key(MoveType.INTRA_RELOCATE, ("a",), 0, 0, (0,), (0,)), ("a::whole",))
    noop_b = CandidateMove(_key(MoveType.INTRA_RELOCATE, ("a",), 0, 0, (0,), (1,)), ("a::whole",))
    deduplicated = deduplicate_candidates(solution, (noop_a, noop_a, noop_b), CONFIG)
    assert len(deduplicated) == 1
    assert deduplicated[0][1].canonical_hash == solution.canonical_hash


def test_candidate_key_requires_sorted_parent_identity() -> None:
    with pytest.raises(ValueError, match="sorted"):
        CandidateKey(0, MoveType.SWAP, ("z", "a"), 0, 1, (0,), (0,))


def test_candidate_payload_cannot_disagree_with_its_key() -> None:
    solution = _whole_solution()
    wrong_parent = CandidateMove(
        _key(MoveType.INTER_RELOCATE, ("a",), 0, 1, (1,), (0,)),
        ("b::whole",),
    )
    with pytest.raises(ValueError, match="affected-parent identity"):
        apply_candidate(solution, wrong_parent, CONFIG)

    pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.4, "x40")
    wrong_split_key = CandidateKey(
        0,
        MoveType.SPLIT_ACTIVATE,
        ("a",),
        0,
        1,
        (0,),
        (0,),
        "a:X_SPLIT:different",
        "different",
    )
    with pytest.raises(ValueError, match="split identity"):
        apply_candidate(solution, CandidateMove(wrong_split_key, split_pattern=pattern), CONFIG)
