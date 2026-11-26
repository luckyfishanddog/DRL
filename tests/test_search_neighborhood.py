from __future__ import annotations

import pytest

from mrta_reference.candidate import apply_candidate
from mrta_reference.geometry import generate_y_split_patterns
from mrta_reference.model import CandidateKey, CandidateMove, MoveType, ParentWeld, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.solution import canonicalize
from mrta_search.direction import optimize_directions_with_initial_feasibility
from mrta_search.neighborhood import RawAttempt, applicable_move_mask, balanced_move_attempt_order, generate_raw_attempts, screen_raw_attempts
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


def test_retired_y_moves_are_rejected_without_changing_saved_patterns():
    from mrta_search.neighborhood import make_raw_candidate
    solution,patterns=_split_pair()
    for kind in (MoveType.SPLIT_ACTIVATE,MoveType.SPLIT_POINT_SWITCH,MoveType.SPLIT_DEACTIVATE):
        pattern=patterns[1] if kind is not MoveType.SPLIT_DEACTIVATE else None
        candidate=CandidateMove(_key(0,kind,('p',),0,1,(0,),(0,),pattern),split_pattern=pattern)
        with pytest.raises(ValueError,match='retired|unsupported'):
            apply_candidate(solution,candidate,FAST)
        assert make_raw_candidate(solution,FAST,kind,seed=3,ordinal=0).candidate is None
    assert solution.patterns==(patterns[0],)


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
    assert stats.rejection_reasons["RETIRED_DYNAMIC_MOVE"] == 1
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


def test_state_aware_applicable_mask_balances_only_possible_moves_and_keeps_hard_m() -> None:
    solution = _base()
    mask = applicable_move_mask(solution, FAST)
    assert mask[MoveType.INTRA_RELOCATE]
    assert mask[MoveType.INTER_RELOCATE]
    assert mask[MoveType.SWAP]
    assert mask[MoveType.TWO_OPT]
    assert MoveType.SPLIT_ACTIVATE not in mask
    assert MoveType.SPLIT_DEACTIVATE not in mask
    assert MoveType.SPLIT_POINT_SWITCH not in mask

    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 17)
    attempts = generate_raw_attempts(solution, FAST, m=23, seed=17, stats=stats)
    assert len(attempts) == 23
    assert sum(stats.attempted_by_move.values()) == 23
    active_counts = [
        stats.attempted_by_move[move.value] for move, applies in mask.items() if applies
    ]
    assert max(active_counts) - min(active_counts) <= 1
    for move, applies in mask.items():
        assert stats.applicable_by_move[move.value] == int(applies)
        if not applies:
            assert stats.attempted_by_move[move.value] == 0

    optional = ParentWeld("optional", (0.0, 5.9), (4.0, 7.0))
    whole = canonicalize(
        (optional,),
        (SplitPattern("optional", SplitKind.WHOLE),),
        {0: ("optional::whole",)},
        FAST,
    )
    assert MoveType.SPLIT_ACTIVATE not in applicable_move_mask(whole, FAST)

    patterns = generate_y_split_patterns(optional, FAST)
    split = canonicalize(
        (optional,),
        (patterns[0],),
        {0: ("optional::0",), 1: ("optional::1",)},
        FAST,
    )
    split_mask = applicable_move_mask(split, FAST)
    assert MoveType.SPLIT_DEACTIVATE not in split_mask
    assert MoveType.SPLIT_POINT_SWITCH not in split_mask


def test_two_opt_star_swaps_same_rail_suffixes_without_pattern_mutation():
    solution = _base()
    move = CandidateMove(
        _key(0, MoveType.TWO_OPT_STAR, ("b", "d"), 2, 3, (1,), (1,))
    )
    candidate = apply_candidate(solution, move, FAST)
    assert candidate.routes[2].block_ids == ("a::whole", "d::whole")
    assert candidate.routes[3].block_ids == ("c::whole", "b::whole")
    assert candidate.patterns == solution.patterns
    assigned = [block for route in candidate.routes for block in route.block_ids]
    assert len(assigned) == len(set(assigned)) == 4
    assert candidate == apply_candidate(solution, move, FAST)


def test_two_opt_star_is_deterministic_ablatable_same_rail_and_cheap_rejects_cross_rail():
    solution = _base()
    off_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 17)
    off = generate_raw_attempts(solution, FAST, m=16, seed=17, stats=off_stats)
    assert all(item.move_type is not MoveType.TWO_OPT_STAR for item in off)

    moves = ACTIVE_MOVE_TYPES + (MoveType.TWO_OPT_STAR,)
    first_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 17)
    second_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 17)
    first = generate_raw_attempts(
        solution, FAST, m=16, seed=17, stats=first_stats, moves=moves
    )
    second = generate_raw_attempts(
        solution, FAST, m=16, seed=17, stats=second_stats, moves=moves
    )
    assert first == second
    generated = [item for item in first if item.move_type is MoveType.TWO_OPT_STAR]
    assert generated
    assert all(
        item.candidate is None
        or {item.candidate.key.source_robot_id, item.candidate.key.destination_robot_id}
        in ({0, 1}, {2, 3})
        for item in generated
    )

    cross_rail = CandidateMove(
        _key(0, MoveType.TWO_OPT_STAR, ("a", "c"), 2, 0, (0,), (0,))
    )
    directions = optimize_directions_with_initial_feasibility(solution, FAST).directions
    screened = screen_raw_attempts(
        solution,
        directions,
        (RawAttempt(MoveType.TWO_OPT_STAR, cross_rail),),
        FAST,
        first_stats,
    )
    assert screened == ()
    assert first_stats.rejection_reasons["CHEAP_INVALID:bad same-rail 2-opt* shape"] == 1


def test_old_x_enable_flag_cannot_reactivate_split_search():
    parent=ParentWeld('x',(1.,8.),(5.,8.))
    solution=canonicalize((parent,),(SplitPattern('x',SplitKind.WHOLE),),{0:('x::whole',)},FAST)
    stats=SearchStats('HISTORICAL_METADATA',20261004)
    attempts=generate_raw_attempts(solution,FAST,m=14,seed=20261004,stats=stats,enable_x_split=True)
    assert len(attempts)==14
    assert all(item.candidate is None or item.candidate.split_pattern is None for item in attempts)
    assert all(item.move_type in (MoveType.INTRA_RELOCATE,MoveType.INTER_RELOCATE,MoveType.SWAP,MoveType.TWO_OPT)
               for item in attempts)
