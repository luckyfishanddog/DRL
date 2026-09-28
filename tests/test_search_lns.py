from __future__ import annotations

import random

import pytest

from mrta_exact import solve_exact_micro
from mrta_reference.model import ParentWeld, ScientificConfig, SplitKind
from mrta_reference.solution import canonicalize
from mrta_search import (
    AdaptiveOperatorState,
    CandidateSourceKind,
    DestroyOperator,
    RepairOperator,
    SearchConfig,
    destroy_parents,
    repair_partial_state,
    run_bounded_sa_oi,
)
from mrta_search.pipeline import evaluate_iteration
from mrta_search.stats import SearchStats


FAST = ScientificConfig(
    weld_speed=1.0,
    empty_speed=1.0,
    t_pre=1.0,
    t_post=1.0,
    interference_dx=0.2,
    interference_dy=0.2,
)


def test_mandatory_pair_insertion_uses_final_route_not_old_index_difference(monkeypatch):
    from mrta_reference.geometry import blocks_for_pattern, generate_y_split_patterns
    from mrta_reference.model import SplitPattern
    from mrta_search import lns

    parent = ParentWeld("c", (2.0, 5.0), (2.0, 7.0))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    blocks = {b.block_id: b for b in blocks_for_pattern(parent, pattern, FAST)}
    for name, x in (("a", 1.0), ("b", 4.0)):
        p = ParentWeld(name, (x, 6.0), (x + 0.5, 6.0))
        blocks.update({b.block_id: b for b in blocks_for_pattern(
            p, SplitPattern(name, SplitKind.WHOLE), FAST
        )})
    # Isolate structural insertion from rail eligibility: actual mandatory-Y
    # geometry normally requires different rails. Final evaluation still checks it.
    monkeypatch.setattr(lns, "robot_is_eligible", lambda *args: True)
    alternatives, _ = lns._parent_alternatives(
        "c", (("a::whole", "b::whole"), (), (), ()), ((0, 0), (), (), ()),
        blocks, pattern, FAST, insertion_limit=8, remaining_budget=10000,
    )
    assert any(a.routes[0] == ("c::1", "a::whole", "c::0", "b::whole")
               and a.trace[0][1:3] == (0, 1) and a.trace[1][1:3] == (0, 0)
               for a in alternatives)


def test_collapsed_mandatory_y_is_rejected_by_scientific_evaluator():
    from mrta_reference.geometry import generate_y_split_patterns
    from mrta_reference.model import CanonicalSolution, Route, ScheduleStatus
    from mrta_reference.scheduler import reference_schedule
    parent = ParentWeld("c", (2.0, 5.0), (2.0, 7.0))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    raw = CanonicalSolution((parent,), (pattern,),
                            tuple(Route(r, ("c::0", "c::1") if r == 0 else ())
                                  for r in range(4)))
    assert reference_schedule(raw, FAST).status is ScheduleStatus.INFEASIBLE
    from mrta_reference.scheduler import reference_schedule_formal
    assert reference_schedule_formal(raw, FAST).status is ScheduleStatus.INFEASIBLE


def _parents() -> tuple[ParentWeld, ...]:
    return tuple(
        ParentWeld(name, start, end)
        for name, start, end in (
            ("a", (1.0, 9.0), (2.0, 10.0)),
            ("b", (6.0, 9.0), (7.0, 10.0)),
            ("c", (1.0, 3.0), (2.0, 2.0)),
            ("d", (6.0, 3.0), (7.0, 2.0)),
            ("e", (12.0, 9.0), (13.0, 10.0)),
            ("f", (12.0, 3.0), (13.0, 2.0)),
        )
    )


def _initial(parents=None):
    result = run_bounded_sa_oi(
        _parents() if parents is None else parents,
        FAST,
        SearchConfig(max_iterations=0),
        seed=17,
    )
    assert result.best_solution is not None
    assert result.best_directions is not None
    assert result.best_schedule is not None
    return result


def _handover_heavy(count: int) -> tuple[ParentWeld, ...]:
    parents = []
    groups = (count + 3) // 4
    for index in range(count):
        local = index // 4
        offset = 7.0 * local / max(1, groups - 1)
        x = {
            0: 0.7 + offset,
            1: 19.2 - offset,
            2: 1.3 + offset,
            3: 18.6 - offset,
        }[index % 4]
        parents.append(
            ParentWeld(f"handover_heavy-{index:03d}", (x, 5.8), (x, 6.2))
        )
    return tuple(parents)


@pytest.mark.parametrize("repair", tuple(RepairOperator))
def test_parent_destroy_repair_identity_replay_and_hard_budget(repair) -> None:
    initial = _initial()
    partial = destroy_parents(
        initial.best_solution,
        initial.best_schedule,
        DestroyOperator.RANDOM_REMOVAL,
        q=2,
        seed=91,
        config=FAST,
    )
    assert len(partial.removed_parent_ids) == 2
    for parent_id in partial.removed_parent_ids:
        assert all(
            not block_id.startswith(parent_id + "::")
            for route in partial.routes
            for block_id in route
        )
    first = repair_partial_state(
        initial.best_solution,
        initial.best_directions,
        partial,
        DestroyOperator.RANDOM_REMOVAL,
        repair,
        FAST,
        max_insertion_evaluations=200,
    )
    second = repair_partial_state(
        initial.best_solution,
        initial.best_directions,
        partial,
        DestroyOperator.RANDOM_REMOVAL,
        repair,
        FAST,
        max_insertion_evaluations=200,
    )
    assert first.insertion_evaluations <= 200
    assert first.candidate is not None
    assert second.candidate is not None
    assert first.candidate.candidate_identity == second.candidate.candidate_identity
    assert first.candidate.solution.canonical_json == second.candidate.solution.canonical_json
    assert first.candidate.source_kind is CandidateSourceKind.LNS_REPAIRED
    actual = [
        block_id
        for route in first.candidate.solution.routes
        for block_id in route.block_ids
    ]
    assert len(actual) == len(set(actual))
    assert {block_id.split("::", 1)[0] for block_id in actual} == {
        parent.parent_id for parent in _parents()
    }

    newer = canonicalize(
        initial.best_solution.parents,
        initial.best_solution.patterns,
        initial.best_solution.routes,
        FAST,
        revision=initial.best_solution.revision + 1,
    )
    with pytest.raises(ValueError, match="stale"):
        repair_partial_state(
            newer,
            initial.best_directions,
            partial,
            DestroyOperator.RANDOM_REMOVAL,
            repair,
            FAST,
        )


def test_mandatory_y_parent_is_destroyed_and_repaired_as_one_unit() -> None:
    parents = (ParentWeld("mandatory", (2.0, 5.0), (2.0, 7.0)),)
    initial = _initial(parents)
    assert initial.best_solution.patterns[0].kind is SplitKind.Y_SPLIT
    assert initial.best_solution.patterns[0].mandatory
    partial = destroy_parents(
        initial.best_solution,
        initial.best_schedule,
        DestroyOperator.CRITICAL_LOAD_REMOVAL,
        q=1,
        seed=0,
        config=FAST,
    )
    assert all(not route for route in partial.routes)
    repaired = repair_partial_state(
        initial.best_solution,
        initial.best_directions,
        partial,
        DestroyOperator.CRITICAL_LOAD_REMOVAL,
        RepairOperator.REGRET_2_REPAIR,
        FAST,
        max_insertion_evaluations=256,
    )
    assert repaired.candidate is not None
    pattern = repaired.candidate.solution.patterns[0]
    assert pattern.kind is SplitKind.Y_SPLIT and pattern.mandatory
    blocks = [
        block_id
        for route in repaired.candidate.solution.routes
        for block_id in route.block_ids
    ]
    assert sorted(blocks) == ["mandatory::0", "mandatory::1"]


def test_adaptive_pair_selection_and_reaction_are_reproducible() -> None:
    first = AdaptiveOperatorState(reaction=0.5, segment_length=4)
    second = AdaptiveOperatorState(reaction=0.5, segment_length=4)
    rng1, rng2 = random.Random(8), random.Random(8)
    selected1 = [first.select(rng1) for _ in range(8)]
    selected2 = [second.select(rng2) for _ in range(8)]
    assert selected1 == selected2
    for index, pair in enumerate(selected1[:4]):
        first.record(pair, float(index + 1))
        second.record(pair, float(index + 1))
    assert first.weights == second.weights
    assert any(weight != 1.0 for weight in first.weights.values())


def test_bounded_initial_portfolio_closes_known_handover_n50_failure() -> None:
    for seed in (3, 7, 11):
        result = run_bounded_sa_oi(
            _handover_heavy(50),
            ScientificConfig(),
            SearchConfig(max_iterations=0),
            seed=seed,
        )
        assert result.best_schedule is not None
        assert result.final_certification is not None
        assert result.final_certification.certified
        assert result.stats.construction_attempts == 4
        assert result.stats.init_reference_calls <= 2
        assert result.initialization.winning_strategy == "SPATIAL_SPREAD"


def test_atomic_and_lns_share_complete_candidate_pipeline() -> None:
    initial = _initial()
    stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 31)
    iteration = evaluate_iteration(
        initial.best_solution,
        initial.best_directions,
        FAST,
        SearchConfig(
            m=32,
            m_atomic=16,
            m_lns=16,
            kdp=16,
            kref=4,
            kref_total=4,
            max_iterations=1,
        ),
        stats,
        seed=31,
        current_schedule=initial.best_schedule,
        adaptive_state=AdaptiveOperatorState(),
    )
    families = {candidate.source_kind for candidate in iteration.complete_candidates}
    assert CandidateSourceKind.ATOMIC in families
    assert CandidateSourceKind.LNS_REPAIRED in families
    assert len(iteration.c3_candidates) <= 16
    assert stats.per_iteration_nref[-1] <= 4
    assert stats.per_iteration_attempts[-1] == 32
    assert stats.repair_reference_calls == 0
    assert stats.repair_full_direction_dp_calls == 0


def test_q6_destroy_repair_crosses_a_basin_atomic_short_budget_cannot() -> None:
    parents = (
        ParentWeld("p0", (1.6988031761970745, 8.06251909281191), (2.507267666737879, 8.024513064718768)),
        ParentWeld("p1", (10.876415580121574, 3.0587891677034196), (11.955019966147066, 3.7709109070254483)),
        ParentWeld("p2", (16.84513432535822, 8.602342619881437), (15.951444969122434, 9.165047149162183)),
        ParentWeld("p3", (6.854207738122211, 1.5241986081119485), (5.40597202291001, 2.096412420693884)),
    )
    exact = solve_exact_micro(parents, FAST)
    atomic = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(
            m=64,
            m_atomic=64,
            m_lns=0,
            kref_total=2,
            direction_refinement_budget=0,
            max_iterations=1,
        ),
        seed=1,
    )
    alns = run_bounded_sa_oi(
        parents,
        FAST,
        SearchConfig(
            m=64,
            m_atomic=0,
            m_lns=64,
            kref_total=2,
            direction_refinement_budget=0,
            max_iterations=4,
        ),
        seed=1,
    )
    assert exact.best_cmax is not None
    assert atomic.initialization.schedule is not None
    assert atomic.best_schedule is not None and alns.best_schedule is not None
    assert atomic.best_schedule.cmax == pytest.approx(atomic.initialization.schedule.cmax)
    assert alns.best_schedule.cmax < atomic.best_schedule.cmax - 1.0e-6
    assert alns.best_schedule.cmax == pytest.approx(exact.best_cmax)
    assert alns.stats.improvements_by_family[CandidateSourceKind.LNS_REPAIRED.value] >= 1
    assert alns.final_certification is not None and alns.final_certification.certified
