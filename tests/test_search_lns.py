from __future__ import annotations

import random
from dataclasses import replace

import pytest

from mrta_exact import solve_exact_micro
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import Operation, OperationKind, ParentWeld, ScientificConfig, SplitKind
from mrta_reference.solution import block_map, canonicalize
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
from mrta_search.lns import (
    BottleneckSelection,
    CRRRBottleneck,
    CRRR_OPERATOR,
    CompleteCandidateIdentity,
    CompleteSearchCandidate,
    build_crrr_candidate,
    identify_crrr_region,
    lns_pair,
    select_bottleneck_parents,
)
from mrta_search.stats import SearchStats


FAST = ScientificConfig(
    weld_speed=1.0,
    empty_speed=1.0,
    t_pre=1.0,
    t_post=1.0,
    interference_dx=0.2,
    interference_dy=0.2,
)


@pytest.mark.parametrize('kind', (SplitKind.Y_SPLIT, SplitKind.X_SPLIT))
def test_oriented_mode_repair_rebuilds_children_and_charges_setup(kind):
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.model import SplitPattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import _remove_parent_set, repair_partial_state_oriented
    config = ScientificConfig()
    parents = (ParentWeld('p', (2., 5.8), (6., 6.4)),
               ParentWeld('a', (0.5, 9.), (1., 9.)),
               ParentWeld('b', (17., 3.), (18., 3.)))
    current = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                           {0: ('a::whole',), 1: (), 2: ('p::whole', 'b::whole'), 3: ()},
                           config, scope=FORMAL_SCOPE_V2)
    partial = _remove_parent_set(current, ('p',))
    target = next(p for p in build_legal_pattern_catalog(current.parents, config, FORMAL_SCOPE_V2)['p']
                  if p.kind is kind)
    partial = replace(partial, patterns=tuple(target if p.parent_id=='p' else p for p in partial.patterns))
    result = repair_partial_state_oriented(current, ((0,), (), (0, 0), ()), partial, config,
                coordination_proxy=True, direction_versions=2, allow_pattern_changes=True)
    assert result.candidate is not None, result.rejection_reason
    solution = result.candidate.solution
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    assert 'p::whole' not in blocks and {'p::0', 'p::1'} <= set(blocks)
    assert sum(config.process_time(b.length) for b in blocks.values() if b.parent_id=='p') == pytest.approx(
        config.process_time(parents[0].length)+50.)
    for old, new in zip(partial.routes, solution.routes):
        assert tuple(b for b in new.block_ids if blocks[b].parent_id!='p') == old
    if kind is SplitKind.X_SPLIT:
        base = 0 if target.rail.value=='UPPER' else 2
        assert 'p::0' in solution.routes[base].block_ids
        assert 'p::1' in solution.routes[base+1].block_ids


def test_two_parent_mode_proposals_and_joint_repair_preserve_closed_coverage():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.model import SplitPattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import _remove_parent_set,rank_parent_mode_pairs,repair_partial_state_oriented
    config = ScientificConfig()
    parents = (ParentWeld('a',(2.,5.6),(4.,6.4)),ParentWeld('b',(14.,5.6),(16.,6.4)),
               ParentWeld('keep',(7.,9.),(8.,9.)))
    solution = canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                            {0:('a::whole','keep::whole'),3:('b::whole',)},config,scope=FORMAL_SCOPE_V2)
    partial = _remove_parent_set(solution,('a','b'))
    proposals = rank_parent_mode_pairs(solution,partial,config)
    catalog = build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)
    assert proposals and proposals == rank_parent_mode_pairs(solution,partial,config)
    assert len(set(proposals)) == len(proposals) <= 3
    for pair in proposals:
        assert {p.parent_id for p in pair} == {'a','b'}
        assert all(p in catalog[p.parent_id] and p.kind is not SplitKind.WHOLE for p in pair)
    assert rank_parent_mode_pairs(solution,_remove_parent_set(solution,('a',)),config) == ()
    # Explicitly exercise simultaneous WHOLE -> optional Y for both parents.
    changed={p:next(v for v in catalog[p] if v.kind is SplitKind.Y_SPLIT) for p in ('a','b')}
    partial=replace(partial,patterns=tuple(changed.get(p.parent_id,p) for p in partial.patterns))
    repaired=repair_partial_state_oriented(solution,((0,0),(),(),(0,)),partial,config,
        seed=11,insertion_limit=8,max_insertion_evaluations=4096,
        coordination_proxy=True,direction_versions=2,allow_pattern_changes=True)
    assert repaired.candidate is not None,repaired.rejection_reason
    after=repaired.candidate.solution
    blocks=block_map(after,config,scope=FORMAL_SCOPE_V2)
    assert set(blocks)=={'a::0','a::1','b::0','b::1','keep::whole'}
    assert sorted(x for route in after.routes for x in route.block_ids)==sorted(blocks)
    assert sum(config.process_time(b.length) for b in blocks.values()) == pytest.approx(
        sum(config.process_time(p.length) for p in parents)+100.)
    assert 'keep::whole' in after.routes[0].block_ids


def test_pair_diversity_preserves_best_and_covers_distinct_parent_pairs():
    from mrta_reference.model import SplitPattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import _remove_parent_set,rank_parent_mode_pairs
    config=ScientificConfig()
    parents=tuple(ParentWeld(pid,(x,5.6),(x+2.,6.4)) for pid,x in (('a',2.),('b',8.),('c',14.)))
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
        {0:('a::whole',),1:('b::whole',),3:('c::whole',)},config,scope=FORMAL_SCOPE_V2)
    partial=_remove_parent_set(solution,('a','b','c'))
    ranked=rank_parent_mode_pairs(solution,partial,config,limit=1000)
    diverse=rank_parent_mode_pairs(solution,partial,config,limit=3,diverse_pairs=True)
    assert diverse[0]==ranked[0]
    assert len({tuple(p.parent_id for p in pair) for pair in diverse})==3
    for pair in diverse:
        key=tuple(p.parent_id for p in pair)
        assert pair==next(v for v in ranked if tuple(p.parent_id for p in v)==key)
    partial=_remove_parent_set(solution,('a','b'))
    assert rank_parent_mode_pairs(solution,partial,config,limit=3,diverse_pairs=True)==rank_parent_mode_pairs(
        solution,partial,config,limit=3)
    assert rank_parent_mode_pairs(solution,partial,config,limit=0,diverse_pairs=True)==()


def test_mode_repair_uses_new_cut_when_child_ids_are_unchanged():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import _remove_parent_set, repair_partial_state_oriented
    config = ScientificConfig()
    parent = ParentWeld('p', (4., 3.), (4., 9.))
    patterns = build_legal_pattern_catalog((parent,), config, FORMAL_SCOPE_V2)['p']
    old, target = patterns[0], patterns[-1]
    current = canonicalize((parent,), (old,), {0: ('p::1',), 2: ('p::0',)},
                           config, scope=FORMAL_SCOPE_V2)
    partial = replace(_remove_parent_set(current, ('p',)), patterns=(target,))
    result = repair_partial_state_oriented(current, ((0,), (), (0,), ()), partial, config,
                coordination_proxy=True, direction_versions=2, allow_pattern_changes=True)
    assert result.candidate is not None, result.rejection_reason
    before = block_map(current, config, scope=FORMAL_SCOPE_V2)
    after = block_map(result.candidate.solution, config, scope=FORMAL_SCOPE_V2)
    assert set(before) == set(after) == {'p::0', 'p::1'}
    assert after['p::0'].end == pytest.approx(parent.point(target.t))
    assert after['p::0'].length != pytest.approx(before['p::0'].length)
    assert result.candidate.projected_process_makespan == pytest.approx(
        max(sum(config.process_time(after[b].length) for b in route.block_ids)
            for route in result.candidate.solution.routes))


def test_mode_repair_rejects_changed_geometry_of_retained_parent():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.model import SplitPattern
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import _remove_parent_set, repair_partial_state_oriented
    config = ScientificConfig()
    parents = (ParentWeld('p', (2., 6.), (6., 6.)), ParentWeld('q', (10., 3.), (11., 3.)))
    current = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                           {0: ('p::whole',), 1: (), 2: ('q::whole',), 3: ()}, config, scope=FORMAL_SCOPE_V2)
    partial = _remove_parent_set(current, ('q',))
    target = next(p for p in build_legal_pattern_catalog(parents, config, FORMAL_SCOPE_V2)['p']
                  if p.kind is SplitKind.Y_SPLIT)
    partial = replace(partial, patterns=(target, partial.patterns[1]))
    with pytest.raises(ValueError, match='complete removal'):
        repair_partial_state_oriented(current, ((0,), (), (0,), ()), partial, config, allow_pattern_changes=True)


@pytest.mark.parametrize("selection", tuple(BottleneckSelection))
def test_bottleneck_selection_is_closed_replayable_and_wait_optional(selection):
    initial = _initial()
    solution, schedule = initial.best_solution, initial.best_schedule
    schedule = replace(schedule, operations=tuple(op for op in schedule.operations
                                                if op.kind is not OperationKind.WAIT))
    if selection is BottleneckSelection.SHARED_Y_PAIR:
        # This fixture has only upper-rail welds; a missing link is explicit.
        assert select_bottleneck_parents(solution, initial.best_directions, schedule,
                                         selection, q=2, seed=9, config=FAST) is None
        return
    if selection is BottleneckSelection.WAIT_ANCHORED:
        assert select_bottleneck_parents(solution, initial.best_directions, schedule,
                                         selection, q=2, seed=9, config=FAST) is None
        wait = Operation("observed-wait", 0, OperationKind.WAIT, 0, 1,
                         (1, 9), (1, 9), 0, solution.routes[0].block_ids[0])
        schedule = replace(schedule, operations=schedule.operations + (wait,))
    one = select_bottleneck_parents(solution, initial.best_directions, schedule,
                                    selection, q=2, seed=9, config=FAST)
    two = select_bottleneck_parents(solution, initial.best_directions, schedule,
                                    selection, q=2, seed=9, config=FAST)
    assert one == two
    assert len(one.removed_parent_ids) == 2
    for r, route in enumerate(solution.routes):
        assert one.routes[r] == tuple(b for b in route.block_ids
                                      if b.split("::", 1)[0] not in one.removed_parent_ids)
    if selection is BottleneckSelection.RANDOM:
        assert one == destroy_parents(solution, schedule, DestroyOperator.RANDOM_REMOVAL,
                                      q=2, seed=9, config=FAST)


def test_load_selector_includes_same_rail_mobility_without_cross_rail():
    initial = _initial()
    choices = [select_bottleneck_parents(initial.best_solution, initial.best_directions,
                                        initial.best_schedule, BottleneckSelection.LOAD,
                                        q=2, seed=s, config=FAST).removed_parent_ids for s in range(12)]
    assert len(set(choices)) > 1
    blocks = block_map(initial.best_solution, FAST)
    old = {b: r for r, route in enumerate(initial.best_solution.routes) for b in route.block_ids}
    for selected in choices:
        assert any(robot_is_eligible(blocks[b], old[b] ^ 1, FAST)
                   for b in blocks if blocks[b].parent_id in selected)
    assert all(not robot_is_eligible(blocks[b], (old[b] + 2) % 4, FAST) for b in blocks)


def test_travel_selection_weights_use_oriented_endpoints(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus, SplitPattern
    from mrta_search import lns
    parents = (ParentWeld("a", (1, 9), (5, 9)), ParentWeld("b", (6, 9), (10, 9)),
               ParentWeld("c", (11, 9), (12, 9)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ("a::whole", "b::whole", "c::whole")}, FAST)
    recorded = []
    original_choices = random.Random.choices
    def capture(self, population, weights=None, **kwargs):
        recorded.append(tuple(weights))
        return original_choices(self, population, weights=weights, **kwargs)
    monkeypatch.setattr(random.Random, "choices", capture)
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10, robot_completion=(10, 0, 0, 0))
    for hints in (((0, 0, 0), (), (), ()), ((1, 0, 0), (), (), ())):
        lns.select_bottleneck_parents(solution, hints, schedule, BottleneckSelection.TRAVEL,
                                      q=2, seed=1, config=FAST)
    # Same segment midpoints, different exit point of a: 1m vs 5m travel.
    assert recorded[1][0] == pytest.approx(5 * recorded[0][0])
    assert recorded[1][1] == recorded[0][1]


def test_fixed_pattern_repair_rejects_canonical_child_collapse(monkeypatch):
    from mrta_reference.geometry import generate_y_split_patterns
    from mrta_reference.model import Route, SplitPattern
    from mrta_search import lns
    parent = ParentWeld("a", (2, 6), (4, 6))
    neighbour = ParentWeld("b", (5, 6), (6, 6))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    solution = canonicalize((parent, neighbour), (pattern, SplitPattern("b", SplitKind.WHOLE)),
                            {0: ("a::0", "b::whole", "a::1")}, FAST)
    partial = lns._remove_parent_set(solution, ("a",))
    def collapse(*args, **kwargs):
        return ((lns._InsertionAlternative("a", (("a::0", "a::1", "b::whole"), (), (), ()),
                                          ((0, 0, 0), (), (), ()), (0, 0), ()),), 1)
    monkeypatch.setattr(lns, "_parent_alternatives", collapse)
    result = repair_partial_state(solution, ((0, 0, 0), (), (), ()), partial,
                                   DestroyOperator.RANDOM_REMOVAL, RepairOperator.REGRET_2_REPAIR,
                                   FAST, preserve_patterns=True)
    assert result.candidate is None
    assert result.rejection_reason == "FIXED_PATTERN_CHANGED_BY_CANONICALIZATION"
    legacy = repair_partial_state(solution, ((0, 0, 0), (), (), ()), partial,
                                  DestroyOperator.RANDOM_REMOVAL, RepairOperator.REGRET_2_REPAIR, FAST)
    assert legacy.candidate is not None
    assert legacy.candidate.solution.patterns[0].kind is SplitKind.WHOLE


@pytest.mark.parametrize("selection", tuple(s for s in BottleneckSelection
                                           if s is not BottleneckSelection.SHARED_Y_PAIR))
def test_bottleneck_selection_closes_split_children(selection):
    from mrta_reference.geometry import generate_y_split_patterns
    from mrta_reference.model import ScheduleResult, ScheduleStatus, SplitPattern
    parent = ParentWeld("a", (2, 5), (2, 7))
    other = ParentWeld("b", (5, 9), (6, 9))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    solution = canonicalize((parent, other), (pattern, SplitPattern("b", SplitKind.WHOLE)),
                            {0: ("a::1", "b::whole"), 2: ("a::0",)}, FAST)
    wait = Operation("observed-wait", 0, OperationKind.WAIT, 2, 3,
                     (2, 7), (2, 7), 0, "a::1")
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, operations=(wait,), cmax=10,
                              robot_completion=(10, 0, 8, 0))
    partial = select_bottleneck_parents(solution, ((0, 0), (), (0,), ()), schedule,
                                       selection, q=2, seed=1, config=FAST)
    assert partial.removed_parent_ids == ("a", "b")
    assert partial.routes == ((), (), (), ())
    assert partial.patterns == solution.patterns


def test_wait_selector_keeps_actual_before_after_anchors():
    from mrta_reference.model import ScheduleResult, ScheduleStatus, SplitPattern
    parents = tuple(ParentWeld(p, (x, 9), (x + 1, 9))
                    for p, x in (("a", 1), ("b", 3), ("c", 17)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ("a::whole", "b::whole"), 1: ("c::whole",)}, FAST)
    operations = (
        Operation("post-a", 0, OperationKind.POST, 0, 5, (2, 9), (2, 9), 0, "a::whole"),
        Operation("wait-b", 0, OperationKind.WAIT, 5, 10, (2, 9), (2, 9), 1, "b::whole"),
        Operation("setup-b", 0, OperationKind.SETUP, 10, 11, (3, 9), (3, 9), 1, "b::whole"),
        Operation("other", 1, OperationKind.WELD, 5, 10, (17, 9), (18, 9), 0, "c::whole"),
    )
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, operations=operations, cmax=11,
                              robot_completion=(11, 10, 0, 0))
    for seed in range(5):
        partial = select_bottleneck_parents(solution, ((0, 0), (0,), (), ()), schedule,
                                           BottleneckSelection.WAIT_ANCHORED,
                                           q=2, seed=seed, config=FAST)
        assert partial.removed_parent_ids == ("a", "b")


def test_shared_y_pair_closes_both_parents_and_retains_unrelated_routes():
    from mrta_reference.geometry import generate_y_split_patterns
    from mrta_reference.model import ScheduleResult, ScheduleStatus, SplitPattern
    from mrta_search.lns import shared_y_parent_links
    parents = (ParentWeld("a", (2, 5), (2, 7)),
               ParentWeld("b", (1, 6), (3, 6)),
               ParentWeld("c", (10, 9), (11, 9)))
    pattern = next(p for p in generate_y_split_patterns(parents[0], FAST) if abs(p.t - 0.5) < 1e-8)
    solution = canonicalize(parents, (pattern, SplitPattern("b", SplitKind.WHOLE),
                                     SplitPattern("c", SplitKind.WHOLE)),
                            {0: ("a::1", "c::whole"), 2: ("a::0", "b::whole")}, FAST)
    # Intersecting interiors, no close endpoints. Serialized safe WELDs still
    # form a proximity link; the link itself is not a collision or blocker.
    ops = (Operation("a-upper", 0, OperationKind.WELD, 0, 1, (2, 6), (2, 7), 0, "a::1"),
           Operation("a-lower", 2, OperationKind.WELD, 0, 1, (2, 5), (2, 6), 0, "a::0"),
           Operation("b-lower", 2, OperationKind.WELD, 2, 4, (1, 6), (3, 6), 1, "b::whole"))
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, operations=ops, cmax=5,
                              robot_completion=(5, 0, 4, 0))
    links = shared_y_parent_links(solution, schedule, FAST)
    assert [(a, b) for a, b, _ in links] == [("a", "b")]
    for seed in range(4):
        partial = select_bottleneck_parents(solution, ((0, 0), (), (0, 0), ()), schedule,
                    BottleneckSelection.SHARED_Y_PAIR, q=2, seed=seed, config=FAST)
        assert partial.removed_parent_ids == ("a", "b")
        assert partial.routes == (("c::whole",), (), (), ())
        assert partial.patterns == solution.patterns
    delayed = replace(schedule, operations=ops[:2] + (replace(ops[2], start_time=20, end_time=22),))
    assert shared_y_parent_links(solution, delayed, FAST)[0][2] < links[0][2]
    outside = replace(schedule, operations=ops[:2] + (replace(ops[2], start=(1, 4), end=(3, 4)),))
    assert shared_y_parent_links(solution, outside, FAST) == ()
    # Repeated phases must not multiply the edge score.
    repeated = replace(schedule, operations=ops + (replace(ops[2], operation_id="duplicate"),))
    assert shared_y_parent_links(solution, repeated, FAST) == links


def _oriented_fixture():
    from mrta_reference.model import SplitPattern
    parents = tuple(ParentWeld(p, (x, 9), (x + 1, 9))
                    for p, x in (("a", 1), ("b", 3), ("c", 11), ("d", 13), ("e", 5)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ("a::whole", "b::whole", "e::whole"), 1: ("c::whole", "d::whole")}, FAST)
    return solution, ((0, 1, 0), (1, 0), (), ())


def test_oriented_local_arcs_match_full_cost_and_keep_unrelated_directions():
    from mrta_search import lns
    solution, directions = _oriented_fixture()
    blocks = block_map(solution, FAST)
    routes = (("a::whole", "b::whole"), ("c::whole", "d::whole"), (), ())
    bits = ((0, 1), (1, 0), (), ())
    state = lns.oriented_route_state(routes, bits, blocks, FAST)
    rows, used = lns.oriented_insert_variants(state, blocks["e::whole"], 0, 1, blocks, FAST, keep=8)
    assert used == 8
    assert rows
    for after, action in rows:
        assert after.directions[1:] == state.directions[1:]
        assert after.arc_costs[1:] == state.arc_costs[1:]
        assert after.travel == pytest.approx(lns._directed_proxy(after.routes, after.directions, blocks, FAST))
        assert after.loads == pytest.approx(lns._loads(after.routes, blocks, FAST))
        assert action.block_id == "e::whole"
        assert all(b in ("a::whole", "b::whole") for b, _, _ in action.neighbour_changes)


@pytest.mark.parametrize("fixed", (True, False))
@pytest.mark.parametrize("followup", (True, False))
def test_oriented_repair_actions_replay_and_fixed_bits_survive(fixed, followup):
    from mrta_search import lns
    solution, directions = _oriented_fixture()
    partial = lns._remove_parent_set(solution, ("b", "e"))
    result = lns.repair_partial_state_oriented(solution, directions, partial, FAST,
                                              fixed_directions=fixed, seed=61, max_insertion_evaluations=256,
                                              coordination_proxy=followup, direction_versions=2 if followup else 1)
    assert result.candidate is not None
    assert result.insertion_evaluations <= 256
    assert 1 <= len(result.direction_candidates) <= (2 if followup else 1)
    old_bits = {b: d for route, bits in zip(solution.routes, directions) for b, d in zip(route.block_ids, bits)}
    routes = [list(r) for r in partial.routes]
    bits = [[old_bits[b] for b in r] for r in routes]
    for action in result.actions:
        routes[action.robot].insert(action.position, action.block_id)
        bits[action.robot].insert(action.position, action.direction)
        for b, old, new in action.neighbour_changes:
            i = routes[action.robot].index(b)
            assert bits[action.robot][i] == old
            bits[action.robot][i] = new
    assert tuple(tuple(r) for r in routes) == tuple(r.block_ids for r in result.candidate.solution.routes)
    assert tuple(tuple(b) for b in bits) == result.directions
    if fixed:
        assert all(old_bits[b] == d for route, ds in zip(result.candidate.solution.routes, result.directions)
                   for b, d in zip(route.block_ids, ds))


def test_oriented_budget_visits_all_legal_robots_and_distinct_structures(monkeypatch):
    from mrta_reference.model import SplitPattern
    from mrta_search import lns
    parent = ParentWeld("free", (4, 6), (5, 6))
    solution = canonicalize((parent,), (SplitPattern("free", SplitKind.WHOLE),), {0: ("free::whole",)}, FAST)
    blocks = block_map(solution, FAST)
    visited = []
    original = lns.oriented_insert_variants
    def capture(state, block, robot, *args, **kwargs):
        visited.append(robot)
        return original(state, block, robot, *args, **kwargs)
    monkeypatch.setattr(lns, "oriented_insert_variants", capture)
    alternatives, used = lns._parent_alternatives("free", ((), (), (), ()), ((), (), (), ()), blocks,
                                                 solution.patterns[0], FAST, insertion_limit=8, remaining_budget=8,
                                                 direction_mode="JOINT", original_directions={"free::whole": 0}, seed=3)
    assert set(visited) == {0, 1, 2, 3}
    assert used <= 8
    assert len(alternatives) == len({a.routes for a in alternatives}) == 4


def _oriented_pilot_module():
    from pathlib import Path
    import runpy
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/run_joint_oriented_repair_pilot.py"))


def test_new_initialization_and_x_smoke_never_call_dp_or_certifier(monkeypatch):
    from mrta_reference import geometry, scheduler, certifier
    from mrta_search import direction, initialization
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden DP/certifier path called")
    monkeypatch.setattr(geometry, "optimize_directions", forbidden)
    monkeypatch.setattr(scheduler, "optimize_directions", forbidden)
    monkeypatch.setattr(direction, "optimize_directions_with_initial_feasibility", forbidden)
    monkeypatch.setattr(direction, "refine_directions_bounded", forbidden)
    monkeypatch.setattr(initialization, "optimize_directions_with_initial_feasibility", forbidden)
    monkeypatch.setattr(initialization, "_route_empty_proxy", forbidden)
    monkeypatch.setattr(initialization, "_improve_region_route", forbidden)
    monkeypatch.setattr(certifier, "certify_schedule", forbidden)
    monkeypatch.setattr(initialization, "certify_schedule", forbidden)
    module = _oriented_pilot_module()
    smoke = module["x_split_smoke"](ScientificConfig())
    assert smoke["status"] == "FEASIBLE"
    assert smoke["directions"][2:] == ((), ())
    initial, attempts = module["common_initial"](_parents(), FAST)
    assert initial is not None
    assert attempts[-1]["status"] == "FEASIBLE"
    from mrta_search import lns
    partial = lns._remove_parent_set(initial.solution, (initial.solution.parents[0].parent_id,))
    result = lns.repair_partial_state_oriented(initial.solution, initial.directions, partial, FAST, seed=4)
    assert result.candidate is not None
    module["schedule_explicit"](result.candidate.solution, result.directions, FAST)
    beam_initial, attempts = module['common_initial_beam'](_parents(), FAST)
    assert beam_initial is not None
    assert sum(a['status'] in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE') for a in attempts) <= 8
    smoke_new = module['x_split_smoke'](ScientificConfig(), repair_options=dict(coordination_proxy=True, direction_versions=2))
    assert smoke_new['status'] == 'FEASIBLE'


def test_direct_schedule_rejects_missing_vectors_and_missing_welds(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    module = _oriented_pilot_module()
    solution, directions = _oriented_fixture()
    with pytest.raises(ValueError, match="four explicit"):
        module["schedule_explicit"](solution, directions[:2], FAST)
    captured = []
    def fake_schedule(solution, config, **kwargs):
        captured.append(kwargs["orientations"])
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=1, directions=directions)
    monkeypatch.setitem(module["schedule_explicit"].__globals__, "reference_schedule_formal", fake_schedule)
    with pytest.raises(ValueError, match="omitted or repeated"):
        module["schedule_explicit"](solution, directions, FAST)
    assert captured == [{r: directions[r] for r in range(4)}]


def test_oriented_mandatory_y_keeps_two_children_and_full_directions():
    from mrta_reference.geometry import generate_y_split_patterns
    from mrta_search import lns
    parent = ParentWeld("mandatory", (2, 5), (2, 7))
    pattern = generate_y_split_patterns(parent, FAST)[0]
    solution = canonicalize((parent,), (pattern,), {0: ("mandatory::1",), 2: ("mandatory::0",)}, FAST)
    result = lns.repair_partial_state_oriented(solution, ((0,), (), (0,), ()),
                                              lns._remove_parent_set(solution, (parent.parent_id,)),
                                              FAST, seed=8, max_insertion_evaluations=128)
    assert result.candidate is not None
    assert result.candidate.solution.patterns == (pattern,)
    assert result.insertion_evaluations <= 128
    assert len(result.actions) == 2
    blocks = block_map(result.candidate.solution, FAST)
    assert set(blocks) == {"mandatory::0", "mandatory::1"}
    for route, bits in zip(result.candidate.solution.routes, result.directions):
        assert len(route.block_ids) == len(bits)
        assert all(robot_is_eligible(blocks[b], route.robot_id, FAST) for b in route.block_ids)


def test_nominal_coordination_matches_route_time_without_interference():
    from mrta_search.lns import nominal_coordination_features, oriented_route_state, oriented_time_score
    solution, directions = _oriented_fixture()
    from mrta_reference.model import Route
    solution = replace(solution, routes=(Route(0, tuple(b for r in solution.routes for b in r.block_ids)),
                                        Route(1, ()), Route(2, ()), Route(3, ())))
    directions = (tuple(d for bits in directions for d in bits), (), (), ())
    features = nominal_coordination_features(solution, directions, FAST)
    state = oriented_route_state(tuple(r.block_ids for r in solution.routes), directions,
                                 block_map(solution, FAST), FAST)
    assert features['nominal_cmax'] == pytest.approx(oriented_time_score(state)[0])
    assert features['exposure_seconds'] == (0, 0, 0, 0)
    assert features['exposure_score'] == features['nominal_cmax']


def test_nominal_coordination_measures_continuous_cross_rail_proximity():
    from mrta_reference.model import SplitPattern
    from mrta_search.lns import nominal_coordination_features
    config = replace(FAST, t_pre=0, t_post=0)
    parents = (ParentWeld('a', (0, 6), (2, 6)), ParentWeld('b', (2, 6), (0, 6)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ('a::whole',), 2: ('b::whole',)}, config)
    features = nominal_coordination_features(solution, ((0,), (), (0,), ()), config)
    # Opposing speed 1: |2t-2| <= 0.2 lasts 0.2 seconds, at the interiors.
    assert features['proximity_seconds'] == pytest.approx((0.2, 0, 0.2, 0))
    assert features['same_rail_order_seconds'] == (0, 0, 0, 0)
    assert features['exposure_score'] == pytest.approx(2.2)
    reverse = nominal_coordination_features(solution, ((0,), (), (1,), ()), config)
    assert reverse['exposure_seconds'] == pytest.approx((2, 0, 2, 0))


def test_nominal_coordination_counts_same_rail_order_and_unions_overlaps():
    from mrta_reference.model import SplitPattern
    from mrta_search.lns import nominal_coordination_features
    config = replace(FAST, t_pre=0, t_post=0)
    parents = (ParentWeld('a', (5, 6), (7, 6)), ParentWeld('b', (1, 9), (3, 9)),
               ParentWeld('c', (5, 6.1), (7, 6.1)))
    solution = canonicalize(parents, tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents),
                            {0: ('a::whole',), 1: ('b::whole',), 2: ('c::whole',)}, config)
    features = nominal_coordination_features(solution, ((0,), (0,), (0,), ()), config)
    assert features['same_rail_order_seconds'] == pytest.approx((2, 2, 0, 0))
    assert features['proximity_seconds'] == pytest.approx((2, 0, 2, 0))
    assert features['exposure_seconds'] == pytest.approx((2, 2, 2, 0))


def test_followup_proxy_and_two_direction_versions_keep_distinct_regret_structures():
    from collections import Counter
    from mrta_search import lns
    solution, directions = _oriented_fixture()
    blocks = block_map(solution, FAST)
    partial = lns._remove_parent_set(solution, ("b",))
    hint_by_id = {b:d for route,ds in zip(solution.routes,directions) for b,d in zip(route.block_ids,ds)}
    hints = tuple(tuple(hint_by_id[b] for b in route) for route in partial.routes)
    alternatives, used = lns._parent_alternatives(
        "b", partial.routes, hints, blocks, solution.patterns[1], FAST,
        insertion_limit=8, remaining_budget=256, direction_mode="JOINT",
        original_directions=hint_by_id, seed=8, coordination_proxy=True, direction_versions=2,
    )
    assert used <= 256
    counts = Counter(a.routes for a in alternatives)
    assert max(counts.values()) == 2
    assert len(counts) > 1
    for a in alternatives:
        assert a.score[:2] == lns.oriented_time_score(a.oriented_state)
    best = alternatives[0]
    second_structure = next(a for a in alternatives if a.routes != best.routes)
    assert second_structure.routes != best.routes


def test_joint_insertion_beam_has_bounded_route_diversity_and_full_coverage():
    from mrta_search.initialization import build_oriented_insertion_beam
    solution, _ = _oriented_fixture()
    beam = build_oriented_insertion_beam(solution, FAST, seed=7)
    assert 1 < len(beam) <= 4
    assert len({state.routes for state, _ in beam}) > 1
    expected = {b for route in solution.routes for b in route.block_ids}
    for state, actions in beam:
        actual = [b for route in state.routes for b in route]
        assert len(actual) == len(set(actual)) == len(expected)
        assert set(actual) == expected
        assert len(actions) == len(expected)
        assert all(len(route)==len(ds) for route,ds in zip(state.routes,state.directions))
    with pytest.raises(ValueError, match="between 1 and 4"):
        build_oriented_insertion_beam(solution, FAST, beam_width=5)


def test_beam_initial_evaluates_at_most_four_and_selects_best_not_first(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    module = _oriented_pilot_module()
    solution, _ = _oriented_fixture()
    calls = []
    monkeypatch.setitem(module['common_initial_beam'].__globals__, 'initial_reference', lambda *args: solution)
    def fake_schedule(candidate, directions, config):
        calls.append((candidate,directions))
        # Every complete candidate succeeds; the fourth must still be evaluated.
        schedule = ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100-len(calls),directions=directions)
        return module['DirectedSolution'](candidate,directions,schedule)
    monkeypatch.setitem(module['common_initial_beam'].__globals__, 'schedule_explicit', fake_schedule)
    initial, attempts = module['common_initial_beam'](solution.parents,FAST)
    assert len(calls) == len(attempts) == 4
    assert initial.schedule.cmax == 96


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
    accelerated, accelerated_count = lns._parent_alternatives(
        "c", (("a::whole", "b::whole"), (), (), ()), ((0, 0), (), (), ()),
        blocks, pattern, FAST, insertion_limit=8, remaining_budget=10000,
        incremental_scoring=True,
    )
    assert accelerated_count == len(alternatives)
    assert accelerated == alternatives


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
    accelerated = repair_partial_state(
        initial.best_solution,
        initial.best_directions,
        partial,
        DestroyOperator.RANDOM_REMOVAL,
        repair,
        FAST,
        max_insertion_evaluations=200,
        incremental_scoring=True,
    )
    assert accelerated == first
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
    search = SearchConfig(
        m=32,
        m_atomic=16,
        m_lns=16,
        kdp=16,
        kref=4,
        kref_total=4,
        max_iterations=1,
    )
    iteration = evaluate_iteration(
        initial.best_solution,
        initial.best_directions,
        FAST,
        search,
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

    accelerated_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 31)
    accelerated = evaluate_iteration(
        initial.best_solution,
        initial.best_directions,
        FAST,
        replace(search, lns_incremental_scoring=True),
        accelerated_stats,
        seed=31,
        current_schedule=initial.best_schedule,
        adaptive_state=AdaptiveOperatorState(),
    )
    assert not SearchConfig().lns_incremental_scoring
    assert accelerated.complete_candidates == iteration.complete_candidates
    assert accelerated.c3_candidates == iteration.c3_candidates
    assert accelerated.c4_candidates == iteration.c4_candidates
    assert accelerated.proposal == iteration.proposal
    assert accelerated_stats.repair_insertion_evaluations == stats.repair_insertion_evaluations


def test_upgraded_frozen_alns_profiles_change_only_authorized_switches() -> None:
    from mrta_ranker.ranking import upgraded_method_spec, upgraded_search_config

    baseline = upgraded_search_config("BASELINE_HEURISTIC")
    heuristic = upgraded_search_config("REGION_FAST_HEURISTIC")
    gat = upgraded_search_config("REGION_FAST_GAT_V05")
    assert not baseline.initialization_region_cooperative
    assert not baseline.lns_incremental_scoring
    assert heuristic == gat
    assert heuristic.initialization_region_cooperative
    assert heuristic.lns_incremental_scoring
    for search in (baseline, heuristic, gat):
        assert (search.m, search.m_atomic, search.m_lns) == (192, 144, 48)
        assert (search.kdp, search.kref, search.kref_total) == (8, 2, 4)
        assert not search.enable_crrr
        assert search.c2_mode == "BASELINE_C2"
        assert not search.enable_two_opt_star
    assert upgraded_method_spec("REGION_FAST_HEURISTIC")["ranker"] is None
    assert upgraded_method_spec("REGION_FAST_GAT_V05")["ranker"] == "GAT_V05"


def test_final_coordination_profiles_are_frozen_and_mutually_exclusive() -> None:
    from mrta_ranker.ranking import (
        FINAL_COORDINATION_METHODS,
        final_coordination_method_spec,
        final_coordination_search_config,
    )

    base = final_coordination_search_config("REGION_FAST_HEURISTIC")
    route = final_coordination_search_config("REGION_FAST_ROUTE_REORDER")
    same = final_coordination_search_config("REGION_FAST_SAME_RAIL")
    assert FINAL_COORDINATION_METHODS == (
        "REGION_FAST_HEURISTIC", "REGION_FAST_ROUTE_REORDER",
        "REGION_FAST_SAME_RAIL", "ADAPTED_HGA_V2", "ADAPTED_WAG_VNS_V2",
    )
    for search in (base, route, same):
        assert (search.m, search.m_atomic, search.m_lns) == (192, 144, 48)
        assert (search.kdp, search.kref, search.kref_total) == (8, 2, 4)
        assert (search.construction_budget, search.kinit_ref) == (5, 5)
        assert search.initialization_region_cooperative
        assert search.lns_incremental_scoring and search.hard_reference_deadline
        assert not search.enable_crrr and not search.enable_jrrb
        assert not search.enable_two_opt_star and search.c2_mode == "BASELINE_C2"
    assert not base.initialization_same_rail_boundary_rebalance
    assert not base.initialization_coordination_route_reorder
    assert route.initialization_coordination_route_reorder
    assert not route.initialization_same_rail_boundary_rebalance
    assert same.initialization_same_rail_boundary_rebalance
    assert not same.initialization_coordination_route_reorder
    for method in FINAL_COORDINATION_METHODS:
        assert final_coordination_method_spec(method)["reference_recovery_rollouts"] == 32


def test_crrr_wait_boundary_uses_adjacent_weld_and_joint_change_is_legal() -> None:
    initial = _initial()
    weld = next(
        operation for operation in initial.best_schedule.operations
        if operation.kind is OperationKind.WELD and operation.block_id is not None
    )
    wait = Operation(
        "crrr-test-wait", weld.robot_id, OperationKind.WAIT,
        max(0.0, weld.start_time - 0.5), weld.start_time,
        weld.start, weld.start, weld.sequence_index,
    )
    schedule = replace(
        initial.best_schedule,
        operations=initial.best_schedule.operations + (wait,),
    )
    region = identify_crrr_region(
        initial.best_solution, schedule, FAST, attempt_index=0
    )
    assert region is not None
    assert region.bottleneck_type is CRRRBottleneck.WAIT_BOUNDARY
    assert weld.block_id.split("::", 1)[0] in region.affected_parent_ids
    assert 2 <= len(region.affected_parent_ids) <= 6

    repairs = [
        build_crrr_candidate(
            initial.best_solution, initial.best_directions, schedule, FAST,
            attempt_index=index,
        )
        for index in range(8)
    ]
    candidate = next(row.candidate for row in repairs if row.candidate is not None)
    provenance = dict(candidate.provenance)
    assert provenance["operator"] == "CRRR"
    assert provenance["changed_parent_count"] >= 2
    assert lns_pair(candidate) is None
    assert tuple(pattern.pattern_id for pattern in candidate.solution.patterns) == tuple(
        pattern.pattern_id for pattern in initial.best_solution.patterns
    )
    blocks = block_map(candidate.solution, FAST)
    actual = [item for route in candidate.solution.routes for item in route.block_ids]
    assert len(actual) == len(set(actual)) == len(blocks)
    assert all(
        robot_is_eligible(blocks[item], route.robot_id, FAST)
        for route in candidate.solution.routes for item in route.block_ids
    )
    from mrta_ranker.features import extract_c2_features
    features = extract_c2_features(
        initial.best_solution, initial.best_directions,
        initial.best_schedule.cmax, candidate, FAST,
    )
    assert features["source_lns_repaired"] == 1.0
    assert features["affected_parent_count"] >= 2.0


def test_crrr_replaces_eight_original_lns_attempts_and_default_is_unchanged() -> None:
    initial = _initial()
    base = dict(
        m=32, m_atomic=16, m_lns=16, kdp=8, kref=2,
        kref_total=4, max_iterations=1,
    )
    disabled_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 31)
    evaluate_iteration(
        initial.best_solution, initial.best_directions, FAST,
        SearchConfig(**base), disabled_stats, seed=31,
        current_schedule=initial.best_schedule,
        adaptive_state=AdaptiveOperatorState(),
    )
    assert disabled_stats.per_iteration_attempts == [32]
    assert disabled_stats.crrr_attempted == 0
    assert len(disabled_stats.operator_sequence) == 0

    enabled_stats = SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 31)
    adaptive = AdaptiveOperatorState()
    iteration = evaluate_iteration(
        initial.best_solution, initial.best_directions, FAST,
        SearchConfig(**base, enable_crrr=True), enabled_stats, seed=31,
        current_schedule=initial.best_schedule, adaptive_state=adaptive,
    )
    assert enabled_stats.per_iteration_attempts == [32]
    assert enabled_stats.attempted_by_family == {
        "ATOMIC": 16, "LNS_REPAIRED": 16, "DIRECTION_REFINEMENT": 0,
    }
    assert enabled_stats.crrr_attempted == 8
    assert len(adaptive.selection_history) == 8
    assert enabled_stats.crrr_valid_unique > 0
    assert any(dict(candidate.provenance).get("operator") == "CRRR"
               for candidate in iteration.complete_candidates)
    assert enabled_stats.repair_reference_calls == 0
    assert enabled_stats.repair_full_direction_dp_calls == 0


def test_crrr_access_c2_is_guarded_family_safe_and_limited_to_one(monkeypatch) -> None:
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    from mrta_search.coordination_risk import CoordinationRisk
    from mrta_search import pipeline
    from mrta_search.pipeline import select_c2_by_family, select_c2_crrr_access

    current = _initial().best_solution
    normal = tuple(
        CompleteSearchCandidate(
            CandidateSourceKind.LNS_REPAIRED,
            current,
            CompleteCandidateIdentity(0, CandidateSourceKind.LNS_REPAIRED.value, ("OLD", index)),
            float(index + 1),
            10.0 + index,
            0,
            0.0,
            (("operator", "OLD"),),
        )
        for index in range(8)
    )
    changed = replace(current, revision=current.revision + 1)
    challenger = CompleteSearchCandidate(
        CandidateSourceKind.LNS_REPAIRED,
        changed,
        CompleteCandidateIdentity(0, CandidateSourceKind.LNS_REPAIRED.value, (CRRR_OPERATOR, 0)),
        9.0,
        -1.0,
        0,
        0.0,
        (("operator", CRRR_OPERATOR),),
    )
    monkeypatch.setattr(pipeline, "_projected_process_makespan", lambda *args: 10.0)
    monkeypatch.setattr(
        pipeline,
        "coordination_risk",
        lambda solution, config: CoordinationRisk(4.0, 0.0)
        if solution is changed else CoordinationRisk(5.0, 0.0),
    )
    monkeypatch.setattr(
        pipeline, "optimize_directions_with_initial_feasibility",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("direction called")),
    )
    monkeypatch.setattr(
        pipeline, "reference_schedule",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("reference called")),
    )
    schedule = ScheduleResult(
        ScheduleStatus.FEASIBLE, cmax=100.0, robot_completion=(100.0,) * 4,
    )
    baseline = select_c2_by_family(normal + (challenger,), 8, seed=3, iteration=0)
    access = select_c2_crrr_access(
        normal + (challenger,), 8, current, schedule, FAST, seed=3, iteration=0,
    )
    assert len(access.selected) == 8
    assert access.injected is challenger
    assert sum(dict(row.provenance).get("operator") == CRRR_OPERATOR
               for row in access.selected) == 1
    assert baseline[0] in access.selected  # protected STRUCTURAL representative

    rejected = replace(challenger, projected_process_makespan=11.0,
                       local_directed_travel_delta=1.0)
    no_access = select_c2_crrr_access(
        normal + (rejected,), 8, current, schedule, FAST, seed=3, iteration=0,
    )
    assert no_access.injected is None
    assert no_access.selected == select_c2_by_family(
        normal + (rejected,), 8, seed=3, iteration=0,
    )


def test_crrr_preserves_x_split_child_robot_identity() -> None:
    from mrta_reference.geometry import finite_x_split_validator, generate_x_split_patterns
    from mrta_reference.model import Rail, Route, ScheduleResult, ScheduleStatus
    parents = (
        ParentWeld("x1", (2.0, 9.0), (18.0, 9.0)),
        ParentWeld("x2", (2.5, 10.0), (17.5, 10.0)),
    )
    patterns = tuple(
        next(pattern for pattern in generate_x_split_patterns(
            parent, 10.0, FAST, rail=Rail.UPPER
        ) if pattern.point_id == "BX_CENTER")
        for parent in parents
    )
    current = canonicalize(
        parents, patterns,
        (Route(0, ("x1::0", "x2::0")), Route(1, ("x1::1", "x2::1")),
         Route(2, ()), Route(3, ())),
        FAST, x_split_validator=finite_x_split_validator(parents, FAST),
    )
    schedule = ScheduleResult(
        ScheduleStatus.FEASIBLE, cmax=20.0,
        robot_completion=(20.0, 19.0, 0.0, 0.0),
    )
    repaired = build_crrr_candidate(
        current, ((0, 0), (0, 0), (), ()), schedule, FAST, attempt_index=0
    )
    assert repaired.candidate is not None
    routes = repaired.candidate.solution.routes
    assert set(routes[0].block_ids) == {"x1::0", "x2::0"}
    assert set(routes[1].block_ids) == {"x1::1", "x2::1"}
    assert not routes[2].block_ids and not routes[3].block_ids
    assert repaired.candidate.solution.patterns == patterns

    partial = destroy_parents(
        current, schedule, DestroyOperator.CRITICAL_LOAD_REMOVAL,
        q=1, seed=0, config=FAST,
    )
    legacy = repair_partial_state(
        current, ((0, 0), (0, 0), (), ()), partial,
        DestroyOperator.CRITICAL_LOAD_REMOVAL, RepairOperator.REGRET_2_REPAIR,
        FAST, max_insertion_evaluations=512,
    )
    accelerated = repair_partial_state(
        current, ((0, 0), (0, 0), (), ()), partial,
        DestroyOperator.CRITICAL_LOAD_REMOVAL, RepairOperator.REGRET_2_REPAIR,
        FAST, max_insertion_evaluations=512, incremental_scoring=True,
    )
    assert accelerated == legacy


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
