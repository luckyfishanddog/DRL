from __future__ import annotations

import pytest

from mrta_reference.candidate import apply_candidate, deduplicate_candidates
from mrta_reference.certifier import certify_schedule
from mrta_reference.model import (
    CandidateKey,
    CandidateMove,
    MoveType,
    OperationKind,
    ParentWeld,
    Rail,
    Route,
    ScientificAmbiguityError,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.solution import canonicalize
from mrta_reference.scheduler import reference_schedule
from mrta_reference.geometry import build_legal_pattern_catalog
from mrta_reference.scope import FORMAL_SCOPE_V2


CONFIG = ScientificConfig()


def _allow_test_x_split(parent, pattern, config) -> bool:
    """Test-only candidate set; production intentionally has no default."""
    del parent, config
    return pattern.point_id in {"x40", "x60"}


def test_v2_catalog_membership_and_fixed_x_assignment_are_internal():
    parent = ParentWeld("formal-x", (0.0, 8.0), (4.0, 8.0))
    catalog = build_legal_pattern_catalog((parent,), CONFIG, FORMAL_SCOPE_V2)
    pattern = next(
        item for item in catalog[parent.parent_id]
        if item.kind is SplitKind.X_SPLIT and item.point_id == "BX_CENTER"
    )
    accepted = canonicalize(
        (parent,),
        (pattern,),
        {0: ("formal-x::0",), 1: ("formal-x::1",)},
        CONFIG,
        scope=FORMAL_SCOPE_V2,
        x_split_validator=lambda *_: False,
    )
    assert accepted.patterns == (pattern,)
    with pytest.raises(ValueError, match="spatial children"):
        canonicalize(
            (parent,),
            (pattern,),
            {0: ("formal-x::1",), 1: ("formal-x::0",)},
            CONFIG,
            scope=FORMAL_SCOPE_V2,
        )
    forged = SplitPattern(
        parent.parent_id, SplitKind.X_SPLIT, 0.4, "FORGED", rail=Rail.UPPER
    )
    with pytest.raises(ValueError, match="absent from legal catalog"):
        canonicalize(
            (parent,),
            (forged,),
            {0: ("formal-x::0",), 1: ("formal-x::1",)},
            CONFIG,
            scope=FORMAL_SCOPE_V2,
            x_split_validator=lambda *_: True,
        )


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
            (SplitPattern("p", SplitKind.WHOLE), SplitPattern("p", SplitKind.X_SPLIT, 0.5, "x", rail=Rail.UPPER)),
            {0: ("p::whole",)},
            CONFIG,
        )
    with pytest.raises(ValueError, match="coverage"):
        canonicalize((parent,), (SplitPattern("p", SplitKind.WHOLE),), {}, CONFIG)


def test_duplicate_parent_block_missing_and_double_assignment_are_rejected() -> None:
    parent = ParentWeld("p", (0.0, 8.0), (1.0, 8.0))
    pattern = SplitPattern("p", SplitKind.WHOLE)
    with pytest.raises(ValueError, match="duplicate parent_id"):
        canonicalize((parent, parent), (pattern,), {0: ("p::whole",)}, CONFIG)
    with pytest.raises(ValueError, match="missing"):
        canonicalize((parent,), (pattern,), {0: ()}, CONFIG)
    with pytest.raises(ValueError, match="more than once"):
        canonicalize(
            (parent,),
            (pattern,),
            {0: ("p::whole",), 1: ("p::whole",)},
            CONFIG,
        )


def test_workspace_boundary_is_inclusive_but_outside_is_rejected() -> None:
    boundary = ParentWeld("edge", (0.0, 0.0), (20.0, 12.0))
    solution = canonicalize(
        (boundary,),
        (SplitPattern("edge", SplitKind.Y_SPLIT, 0.5, "BY_CENTER"),),
        {0: ("edge::1",), 2: ("edge::0",)},
        CONFIG,
    )
    assert solution.parents == (boundary,)
    outside = ParentWeld("outside", (-1.0e-6, 8.0), (1.0, 8.0))
    with pytest.raises(ValueError, match="outside"):
        canonicalize(
            (outside,),
            (SplitPattern("outside", SplitKind.WHOLE),),
            {0: ("outside::whole",)},
            CONFIG,
        )


def test_same_robot_consecutive_children_canonicalize_to_whole_before_processing() -> None:
    parent = ParentWeld("p", (1, 8), (3, 8))
    solution = canonicalize(
        (parent,),
        (SplitPattern("p", SplitKind.X_SPLIT, 0.5, "given", rail=Rail.UPPER),),
        {0: ("p::0", "p::1")},
        CONFIG,
    )
    assert solution.patterns == (SplitPattern("p", SplitKind.WHOLE),)
    assert solution.routes[0].block_ids == ("p::whole",)
    schedule = reference_schedule(solution, CONFIG)
    assert sum(operation.kind is OperationKind.SETUP for operation in schedule.operations) == 1
    assert sum(operation.kind is OperationKind.POST for operation in schedule.operations) == 1


def test_retained_x_split_formal_validation_is_fail_closed() -> None:
    parent = ParentWeld("p", (1.0, 8.0), (3.0, 8.0))
    pattern = SplitPattern("p", SplitKind.X_SPLIT, 0.4, "test-t40", rail=Rail.UPPER)
    routes = {0: ("p::0",), 1: ("p::1",)}

    with pytest.raises(ScientificAmbiguityError, match="XSplitValidator"):
        canonicalize((parent,), (pattern,), routes, CONFIG)

    def allow_t40(candidate_parent, candidate_pattern, candidate_config) -> bool:
        assert candidate_parent == parent
        assert candidate_config == CONFIG
        return candidate_pattern.point_id == "test-t40"

    approved = canonicalize(
        (parent,), (pattern,), routes, CONFIG, x_split_validator=allow_t40
    )
    schedule = reference_schedule(
        approved, CONFIG, x_split_validator=allow_t40
    )
    assert schedule.status.value == "FEASIBLE"
    assert certify_schedule(
        approved, schedule, CONFIG, x_split_validator=allow_t40
    ).certified
    with pytest.raises(ScientificAmbiguityError, match="XSplitValidator"):
        certify_schedule(approved, schedule, CONFIG)

    with pytest.raises(ValueError, match="rejected"):
        canonicalize(
            (parent,),
            (pattern,),
            routes,
            CONFIG,
            x_split_validator=lambda *_: False,
        )


def test_x_split_merged_to_whole_does_not_require_validator() -> None:
    parent = ParentWeld("p", (1.0, 8.0), (3.0, 8.0))
    pattern = SplitPattern("p", SplitKind.X_SPLIT, 0.4, "unfrozen", rail=Rail.UPPER)
    merged = canonicalize(
        (parent,), (pattern,), {0: ("p::1", "p::0")}, CONFIG
    )
    assert merged.patterns == (SplitPattern("p", SplitKind.WHOLE),)
    assert merged.routes[0].block_ids == ("p::whole",)


def test_x_children_only_merge_when_same_robot_and_consecutive() -> None:
    parents = (
        ParentWeld("p", (1.0, 8.0), (3.0, 8.0)),
        ParentWeld("separator", (5.0, 8.0), (6.0, 8.0)),
    )
    patterns = (
        SplitPattern("p", SplitKind.X_SPLIT, 0.5, "test-mid", rail=Rail.UPPER),
        SplitPattern("separator", SplitKind.WHOLE),
    )
    validator = lambda _parent, pattern, _config: pattern.point_id == "test-mid"
    with pytest.raises(ValueError, match="spatial children"):
        canonicalize(
            parents,
            patterns,
            {0: ("p::0", "separator::whole", "p::1")},
            CONFIG,
            x_split_validator=validator,
        )
    different_robots = canonicalize(
        parents,
        patterns,
        {0: ("p::0", "separator::whole"), 1: ("p::1",)},
        CONFIG,
        x_split_validator=validator,
    )
    assert different_robots.patterns[0].kind is SplitKind.X_SPLIT


def test_equivalent_input_order_has_identical_canonical_hash() -> None:
    solution = _whole_solution()
    reordered = canonicalize(
        tuple(reversed(solution.parents)),
        tuple(reversed(solution.patterns)),
        {1: solution.routes[1].block_ids, 0: solution.routes[0].block_ids},
        CONFIG,
        revision=42,
    )
    assert reordered.canonical_json == solution.canonical_json
    assert reordered.canonical_hash == solution.canonical_hash


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
            (SplitPattern("cross", SplitKind.X_SPLIT, 0.5, "x", rail=Rail.UPPER),),
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
    activated_pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.4, "x40", rail=Rail.UPPER)
    activate = CandidateMove(
        _key(MoveType.SPLIT_ACTIVATE, ("a",), 0, 0, (0,), (0, 0), pattern=activated_pattern),
        split_pattern=activated_pattern,
    )
    split_solution = apply_candidate(
        solution, activate, CONFIG, x_split_validator=_allow_test_x_split
    )
    assert split_solution.routes[0].block_ids == ("a::0", "b::whole")
    assert split_solution.routes[1].block_ids[0] == "a::1"

    switched_pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.6, "x60", rail=Rail.UPPER)
    switch = CandidateMove(
        _key(MoveType.SPLIT_POINT_SWITCH, ("a",), None, None, (), (), revision=1, pattern=switched_pattern),
        split_pattern=switched_pattern,
    )
    switched = apply_candidate(
        split_solution, switch, CONFIG, x_split_validator=_allow_test_x_split
    )
    assert next(pattern for pattern in switched.patterns if pattern.parent_id == "a") == switched_pattern

    deactivate = CandidateMove(
        _key(MoveType.SPLIT_DEACTIVATE, ("a",), None, 0, (), (0,), revision=2)
    )
    whole_again = apply_candidate(
        switched, deactivate, CONFIG, x_split_validator=_allow_test_x_split
    )
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

    pattern = SplitPattern("a", SplitKind.X_SPLIT, 0.4, "x40", rail=Rail.UPPER)
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
