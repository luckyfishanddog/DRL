from __future__ import annotations

import pytest

from mrta_reference.candidate import apply_candidate
from mrta_reference.geometry import generate_y_split_patterns
from mrta_reference.model import CandidateKey, CandidateMove, MoveType, ParentWeld, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.solution import canonicalize
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.neighborhood import RawAttempt, balanced_move_attempt_order, generate_raw_attempts, screen_raw_attempts
from mrta_search.stats import ACTIVE_MOVE_TYPES, SearchStats


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _key(revision, move, affected, source=None, destination=None, source_positions=(), destination_positions=(), pattern=None):
    return CandidateKey(
        revision,
        move,
        tuple(sorted(affected)),
        source,
        destination,
        tuple(source_positions),
        tuple(destination_positions),
        None if pattern is None else pattern.pattern_id,
        None if pattern is None else pattern.point_id,
    )


def _base():
    parents = tuple(
        ParentWeld(name, (x, 2.0), (x + 1.0, 2.0))
        for name, x in (("a", 1.0), ("b", 4.0), ("c", 10.0), ("d", 14.0))
    )
    return canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {2: ("a::whole", "b::whole"), 3: ("c::whole", "d::whole")},
        FAST,
    )


def _split_pair():
    parent = ParentWeld("p", (0.0, 5.9), (4.0, 7.0))
    patterns = generate_y_split_patterns(parent, FAST)
    assert len(patterns) >= 2
    solution = canonicalize(
        (parent,),
        (patterns[0],),
        {0: ("p::1",), 1: ("p::0",)},
        FAST,
    )
    return solution, patterns


def test_four_route_moves_preserve_coverage_and_replay_and_reject_old_revision() -> None:
    solution = _base()
    moves = (
        CandidateMove(_key(0, MoveType.INTRA_RELOCATE, ("a",), 2, 2, (0,), (2,)), ("a::whole",)),
        CandidateMove(_key(0, MoveType.INTER_RELOCATE, ("b",), 2, 3, (1,), (1,)), ("b::whole",)),
        CandidateMove(_key(0, MoveType.SWAP, ("a", "c"), 2, 3, (0,), (0,))),
        CandidateMove(_key(0, MoveType.TWO_OPT, ("a", "b"), 2, 2, (0, 1), ())),
    )
    expected = {f"{name}::whole" for name in "abcd"}
    for move in moves:
        first = apply_candidate(solution, move, FAST)
        second = apply_candidate(solution, move, FAST)
        assert first == second
        assigned = [block for route in first.routes for block in route.block_ids]
        assert set(assigned) == expected
        assert len(assigned) == len(set(assigned))
        with pytest.raises(ValueError, match="revision"):
            apply_candidate(first, move, FAST)


def test_three_y_moves_valid_boundary_and_deterministic_replay() -> None:
    parent = ParentWeld("p", (0.0, 5.9), (4.0, 7.0))
    whole = canonicalize(
        (parent,),
        (SplitPattern("p", SplitKind.WHOLE),),
        {0: ("p::whole",)},
        FAST,
    )
    patterns = generate_y_split_patterns(parent, FAST)
    activate = CandidateMove(
        _key(0, MoveType.SPLIT_ACTIVATE, ("p",), 0, 1, (0,), (0,), patterns[0]),
        ("p::whole",),
        patterns[0],
    )
    split = apply_candidate(whole, activate, FAST)
    assert split == apply_candidate(whole, activate, FAST)
    assert split.patterns[0].kind is SplitKind.Y_SPLIT

    switch = CandidateMove(
        _key(1, MoveType.SPLIT_POINT_SWITCH, ("p",), pattern=patterns[1]),
        split_pattern=patterns[1],
    )
    switched = apply_candidate(split, switch, FAST)
    assert switched == apply_candidate(split, switch, FAST)
    assert switched.patterns[0].pattern_id == patterns[1].pattern_id

    deactivate = CandidateMove(
        _key(1, MoveType.SPLIT_DEACTIVATE, ("p",), None, 0, (), (0,))
    )
    deactivated = apply_candidate(split, deactivate, FAST)
    assert deactivated.patterns[0].kind is SplitKind.WHOLE
    assert deactivated == apply_candidate(split, deactivate, FAST)

    illegal_point = SplitPattern("p", SplitKind.Y_SPLIT, 0.51, "invented")
    invalid = CandidateMove(
        _key(1, MoveType.SPLIT_POINT_SWITCH, ("p",), pattern=illegal_point),
        split_pattern=illegal_point,
    )
    with pytest.raises(ValueError, match="deterministic legal"):
        apply_candidate(split, invalid, FAST)
    with pytest.raises(ValueError, match="revision"):
        apply_candidate(switched, switch, FAST)


def test_mandatory_y_deactivation_is_cheap_rejected_not_scheduler_infeasible() -> None:
    parent = ParentWeld("cross", (2.0, 5.0), (2.0, 7.0))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    solution = canonicalize(
        (parent,),
        (pattern,),
        {0: ("cross::1",), 2: ("cross::0",)},
        FAST,
    )
    candidate = CandidateMove(
        _key(0, MoveType.SPLIT_DEACTIVATE, ("cross",), None, 0, (), (0,))
    )
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 0)
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    screened = screen_raw_attempts(
        solution,
        directions,
        (RawAttempt(MoveType.SPLIT_DEACTIVATE, candidate),),
        FAST,
        stats,
    )
    assert screened == ()
    assert stats.rejection_reasons["MANDATORY_Y_CANNOT_DEACTIVATE"] == 1
    assert stats.nref == 0


def test_balanced_quota_hard_m_duplicate_accounting_and_seeded_replay() -> None:
    solution = _base()
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 11)
    order = balanced_move_attempt_order(64, 11)
    assert len(order) == 64
    counts = [order.count(move) for move in ACTIVE_MOVE_TYPES]
    assert max(counts) - min(counts) <= 1
    first = generate_raw_attempts(solution, FAST, m=64, seed=11, stats=stats)
    second_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 11)
    second = generate_raw_attempts(solution, FAST, m=64, seed=11, stats=second_stats)
    assert first == second
    assert stats.raw_attempts == 64
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    screen_raw_attempts(solution, directions, first, FAST, stats)
    assert stats.raw_candidate_full_dp_calls == 0
    assert stats.raw_candidate_reference_calls == 0
    assert stats.duplicates > 0

