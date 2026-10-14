from __future__ import annotations

from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import ParentWeld, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.scheduler import reference_schedule, reference_schedule_formal
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_reference.solution import block_map
import pytest

from mrta_search.initialization import (
    RAIL_SERIAL_BOOTSTRAP,
    InitializationStatus,
    _construct_region_cooperative_initialization,
    _construct_rail_monotone_balanced_bootstrap,
    _patterns,
    bounded_insertion_positions,
    build_joint_rail_region_rebuild_candidates,
    build_initial_solution,
)
from mrta_search.stats import SearchStats


FAST = ScientificConfig(weld_speed=1.0, empty_speed=1.0, t_pre=1.0, t_post=1.0)


def _stats():
    return SearchStats("EXACT_Y_SCOPE_CURRENT_SEMANTICS", 7)


def test_initialization_whole_mandatory_y_bounded_positions_and_replay() -> None:
    parents = (
        ParentWeld("whole", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("mandatory", (8.0, 5.0), (8.0, 7.0)),
    )
    first = build_initial_solution(parents, FAST, _stats())
    second = build_initial_solution(parents, FAST, _stats())
    assert first.status is InitializationStatus.SUCCESS
    assert first.solution == second.solution
    assert first.directions == second.directions
    patterns = {pattern.parent_id: pattern for pattern in first.solution.patterns}
    assert patterns["whole"].kind is SplitKind.WHOLE
    assert patterns["mandatory"].kind is SplitKind.Y_SPLIT
    assert patterns["mandatory"].mandatory
    assert bounded_insertion_positions(3, 8) == (0, 1, 2, 3)
    positions = bounded_insertion_positions(20, 8)
    assert positions[0] == 0 and positions[-1] == 20 and len(positions) <= 8


def test_initialization_fallback_is_bounded_and_does_not_hide_numeric_failure() -> None:
    xs = [2.5529206381, 16.1012410018, 14.5117177606, 4.8463114890, 9.4132666547, 8.5403302310, 12.3802664817]
    config = ScientificConfig(
        weld_speed=1.0,
        empty_speed=1.0,
        t_pre=1.0,
        t_post=1.0,
        interference_dx=0.01,
        interference_dy=0.01,
    )
    parents = tuple(ParentWeld(str(i), (x, 2.0), (x + 0.5, 2.0)) for i, x in enumerate(xs))
    calls = []

    def first_deadlocks(solution, scientific_config, *, orientations):
        calls.append(solution.canonical_hash)
        if len(calls) == 1:
            return ScheduleResult(ScheduleStatus.DEADLOCK)
        return reference_schedule(solution, scientific_config, orientations=orientations)

    stats = _stats()
    result = build_initial_solution(parents, config, stats, reference_evaluator=first_deadlocks)
    assert result.status is InitializationStatus.SUCCESS
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert stats.construction_attempts == 2
    assert stats.raw_candidate_full_dp_calls == 0
    assert stats.raw_candidate_reference_calls == 0

    def numeric(*args, **kwargs):
        return ScheduleResult(ScheduleStatus.NUMERIC_FAILURE, diagnostics=("boom",))

    failed = build_initial_solution(parents, config, _stats(), reference_evaluator=numeric)
    assert failed.status is InitializationStatus.NUMERIC_FAILURE
    assert len(failed.attempts) == 1


def test_initialization_failed_is_run_outcome_and_duplicate_fallback_is_not_reevaluated() -> None:
    parents = (ParentWeld("p", (1.0, 2.0), (2.0, 2.0)),)
    calls = 0

    def deadlock(*args, **kwargs):
        nonlocal calls
        calls += 1
        return ScheduleResult(ScheduleStatus.DEADLOCK)

    result = build_initial_solution(parents, FAST, _stats(), reference_evaluator=deadlock)
    assert result.status is InitializationStatus.INITIALIZATION_FAILED
    assert calls == 1
    assert len(result.attempts) == 3
    assert result.attempts[1].duplicate
    assert result.attempts[2].duplicate
    assert result.attempts[2].strategy == RAIL_SERIAL_BOOTSTRAP


def test_all_four_diagnostics_kinit_filter_and_bounded_bootstrap_are_reproducible() -> None:
    parents = tuple(
        ParentWeld(str(index), (x, 2.0), (x + 0.5, 2.0))
        for index, x in enumerate(
            (2.55, 16.10, 14.51, 4.85, 9.41, 8.54, 12.38, 6.25)
        )
    )
    config = ScientificConfig(
        weld_speed=1.0,
        empty_speed=1.0,
        t_pre=1.0,
        t_post=1.0,
        interference_dx=0.01,
        interference_dy=0.01,
    )

    def run_once():
        calls = []

        def evaluator(solution, scientific_config, *, orientations):
            calls.append(solution.canonical_hash)
            if not solution.routes[1].block_ids and not solution.routes[3].block_ids:
                return reference_schedule(
                    solution, scientific_config, orientations=orientations
                )
            return ScheduleResult(ScheduleStatus.DEADLOCK)

        result = build_initial_solution(
            parents,
            config,
            _stats(),
            construction_budget=4,
            kinit_ref=2,
            feasibility_bootstrap_budget=1,
            portfolio=True,
            reference_evaluator=evaluator,
        )
        return result, calls

    first, first_calls = run_once()
    second, second_calls = run_once()
    assert first.status is InitializationStatus.SUCCESS
    assert first.winning_strategy == RAIL_SERIAL_BOOTSTRAP
    assert first.solution == second.solution
    assert first.directions == second.directions
    assert first_calls == second_calls
    assert len(first_calls) == 3  # Kinit_ref=2 plus one bounded bootstrap call.
    assert [attempt.strategy for attempt in first.attempts[:4]] == [
        "LOAD_FIRST",
        "RAIL_BALANCED",
        "X_ORDER_AWARE",
        "SPATIAL_SPREAD",
    ]
    assert sum(attempt.schedule is not None for attempt in first.attempts[:4]) == 2
    assert first.attempts[-1].strategy == RAIL_SERIAL_BOOTSTRAP
    assert not first.solution.routes[1].block_ids
    assert not first.solution.routes[3].block_ids


def test_feasibility_bootstrap_budget_can_be_disabled_and_is_validated() -> None:
    parents = (
        ParentWeld("a", (1.0, 2.0), (2.0, 2.0)),
        ParentWeld("b", (10.0, 2.0), (11.0, 2.0)),
    )

    def deadlock(*args, **kwargs):
        return ScheduleResult(ScheduleStatus.DEADLOCK)

    result = build_initial_solution(
        parents,
        FAST,
        _stats(),
        feasibility_bootstrap_budget=0,
        reference_evaluator=deadlock,
    )
    assert all(attempt.strategy != RAIL_SERIAL_BOOTSTRAP for attempt in result.attempts)
    with pytest.raises(ValueError, match="B_init_bootstrap"):
        build_initial_solution(
            parents,
            FAST,
            _stats(),
            feasibility_bootstrap_budget=2,
        )


def _v3_fixture():
    upper = tuple(
        ParentWeld(f"u{i}", (1.0 + i * 3.0, 9.0), (2.0 + i * 3.0, 9.0))
        for i in range(4)
    )
    lower = tuple(
        ParentWeld(f"l{i}", (1.5 + i * 3.0, 3.0), (2.5 + i * 3.0, 3.0))
        for i in range(4)
    )
    return upper + lower


def test_v3_monotone_balanced_bootstrap_is_deterministic_canonical_and_certified():
    parents = _v3_fixture()
    patterns = _patterns(parents, FAST)
    first = _construct_rail_monotone_balanced_bootstrap(parents, patterns, FAST)
    second = _construct_rail_monotone_balanced_bootstrap(parents, patterns, FAST)
    assert first == second
    assert first.patterns == patterns
    assert all(first.routes[robot].block_ids for robot in range(4))
    blocks = block_map(first, FAST)
    assigned = [block for route in first.routes for block in route.block_ids]
    assert len(assigned) == len(set(assigned)) == len(blocks)
    assert all(
        robot_is_eligible(blocks[block], route.robot_id, FAST)
        for route in first.routes
        for block in route.block_ids
    )
    midpoint = lambda block_id: sum(blocks[block_id].start[0:1] + blocks[block_id].end[0:1]) / 2.0
    assert list(first.routes[0].block_ids) == sorted(first.routes[0].block_ids, key=midpoint)
    assert list(first.routes[1].block_ids) == sorted(first.routes[1].block_ids, key=midpoint, reverse=True)
    assert list(first.routes[2].block_ids) == sorted(first.routes[2].block_ids, key=midpoint)
    assert list(first.routes[3].block_ids) == sorted(first.routes[3].block_ids, key=midpoint, reverse=True)

    result = build_initial_solution(
        parents,
        FAST,
        _stats(),
        construction_budget=5,
        kinit_ref=5,
        portfolio=True,
        scope=FORMAL_SCOPE_V1_1,
    )
    assert result.status is InitializationStatus.SUCCESS
    assert result.certification is not None and result.certification.certified
    assert result.solution is not None
    assert result.solution.patterns == patterns


def test_v3_flexible_rail_assignment_is_deterministic_and_preserves_patterns():
    parents = _v3_fixture() + (
        ParentWeld("flex", (6.0, 6.0), (7.0, 6.0)),
    )
    patterns = _patterns(parents, FAST)
    first = _construct_rail_monotone_balanced_bootstrap(parents, patterns, FAST)
    second = _construct_rail_monotone_balanced_bootstrap(parents, patterns, FAST)
    assert first == second
    assert first.patterns == patterns
    occurrences = [
        route.robot_id
        for route in first.routes
        if "flex::whole" in route.block_ids
    ]
    assert len(occurrences) == 1
    assert occurrences[0] in (0, 1, 2, 3)



def test_clearance_construction_is_geometry_only_deterministic_and_formally_legal(monkeypatch):
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.solution import canonicalize
    parents=_v3_fixture() + (ParentWeld('mandatory',(8.,5.),(8.,7.)),
                              ParentWeld('shared',(12.,6.),(13.,6.)))
    # No workbook, historical solution, or seed-specific input is available here.
    def forbidden(*args,**kwargs):
        raise AssertionError('initializer must construct from supplied geometry')
    monkeypatch.setattr('builtins.open',forbidden)
    patterns=_patterns(parents,FAST)
    one=_construct_rail_monotone_balanced_bootstrap(parents,patterns,FAST,clearance_aware=True)
    two=_construct_rail_monotone_balanced_bootstrap(parents,patterns,FAST,clearance_aware=True)
    assert one==two
    catalog=build_legal_pattern_catalog(parents,FAST,FORMAL_SCOPE_V2)
    assert all(p in catalog[p.parent_id] for p in one.patterns)
    assert next(p for p in one.patterns if p.parent_id=='mandatory').mandatory
    assert canonicalize(parents,one.patterns,one.routes,FAST,scope=FORMAL_SCOPE_V2)==one
    blocks=block_map(one,FAST,scope=FORMAL_SCOPE_V2)
    assert all(b.length>=.2-FAST.numeric_epsilon for b in blocks.values())
    assert all(robot_is_eligible(blocks[b],r.robot_id,FAST) for r in one.routes for b in r.block_ids)


def test_clearance_partition_prefers_separation_but_preserves_no_gap_fallback():
    parents=(ParentWeld('a',(1.,9.),(4.,9.)),ParentWeld('b',(3.,9.),(5.,9.)),
             ParentWeld('c',(8.,9.),(9.,9.)),ParentWeld('d',(10.,9.),(11.,9.)))
    patterns=_patterns(parents,FAST)
    sol=_construct_rail_monotone_balanced_bootstrap(parents,patterns,FAST,clearance_aware=True)
    blocks=block_map(sol,FAST)
    left=[blocks[b] for b in sol.routes[0].block_ids];right=[blocks[b] for b in sol.routes[1].block_ids]
    assert left and right
    assert max(max(b.start[0],b.end[0]) for b in left)+FAST.interference_dx<=min(min(b.start[0],b.end[0]) for b in right)
    overlap=(ParentWeld('x',(1.,9.),(10.,9.)),ParentWeld('y',(2.,9.),(11.,9.)))
    patterns=_patterns(overlap,FAST)
    assert _construct_rail_monotone_balanced_bootstrap(overlap,patterns,FAST,clearance_aware=True)==_construct_rail_monotone_balanced_bootstrap(overlap,patterns,FAST)


def test_clearance_portfolio_preserves_five_slots_and_existing_reference_limits(monkeypatch):
    from mrta_search import initialization as init
    calls=[];fallback_calls=[]
    original=init._v3_direction_feasibility_candidates
    def bounded(*args,**kwargs):
        rows=original(*args,**kwargs);fallback_calls.append(len(rows));return rows
    monkeypatch.setattr(init,'_v3_direction_feasibility_candidates',bounded)
    def deadlock(*args,**kwargs):
        calls.append(1);return ScheduleResult(ScheduleStatus.DEADLOCK)
    stats=_stats()
    result=build_initial_solution(_v3_fixture(),FAST,stats,construction_budget=5,kinit_ref=5,
        feasibility_bootstrap_budget=1,portfolio=True,clearance_aware=True,reference_evaluator=deadlock)
    assert result.status is InitializationStatus.INITIALIZATION_FAILED
    assert {a.construction_index for a in result.attempts if a.construction_index<5}==set(range(5))
    assert sum(a.construction_index==5 for a in result.attempts)<=1
    assert len(fallback_calls)<=1 and sum(fallback_calls)<=4
    normal=sum(a.schedule is not None for a in result.attempts if a.construction_index<5)
    assert normal<=5 and len(calls)<=5+4+1
    assert len(calls)==stats.init_reference_calls and stats.nref==0
    assert stats.construction_attempts<=6  # five normal + original serial fallback


def test_clearance_accepted_initializations_are_independently_certified_and_repeatable():
    from mrta_reference.scope import FORMAL_SCOPE_V2
    results=[build_initial_solution(_v3_fixture(),FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,clearance_aware=True,scope=FORMAL_SCOPE_V2) for _ in range(2)]
    assert results[0].solution==results[1].solution
    assert results[0].directions==results[1].directions
    for result in results:
        assert result.status is InitializationStatus.SUCCESS
        assert certify_schedule(result.solution,result.schedule,FAST,scope=FORMAL_SCOPE_V2).certified
        assert all(a.certification and a.certification.certified for a in result.attempts
                   if a.schedule and a.schedule.status is ScheduleStatus.FEASIBLE)


def test_clearance_default_remains_baseline_and_never_hides_certifier_failure():
    from mrta_search.pipeline import SearchConfig
    from mrta_reference.certifier import CertificationReport
    assert SearchConfig().initialization_clearance_aware is False
    def reject(*args,**kwargs):
        return CertificationReport(False,('deliberate independent certification rejection',),None)
    result=build_initial_solution(_v3_fixture(),FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,clearance_aware=True,certifier=reject)
    assert result.status is InitializationStatus.NUMERIC_FAILURE
    assert result.solution is None


def test_region_cooperative_is_deterministic_balanced_eligible_and_certified():
    from mrta_reference.scope import FORMAL_SCOPE_V2
    parents=_v3_fixture() + (
        ParentWeld('mandatory',(8.,5.),(8.,7.)),
        ParentWeld('shared',(12.,6.),(13.,6.)),
    )
    patterns=_patterns(parents,FAST)
    first=_construct_region_cooperative_initialization(parents,patterns,FAST)
    second=_construct_region_cooperative_initialization(parents,patterns,FAST)
    assert first==second and first.patterns==patterns
    blocks=block_map(first,FAST,scope=FORMAL_SCOPE_V2)
    assert all(robot_is_eligible(blocks[item],route.robot_id,FAST)
               for route in first.routes for item in route.block_ids)
    assert all(route.block_ids for route in first.routes)
    result=build_initial_solution(parents,FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,region_cooperative=True,scope=FORMAL_SCOPE_V2)
    assert result.status is InitializationStatus.SUCCESS
    assert result.certification is not None and result.certification.certified


def test_region_cooperative_replaces_only_spatial_slot_and_preserves_budget(monkeypatch):
    from mrta_search import initialization as init
    calls=[];fallback=[];original=init._v3_direction_feasibility_candidates
    def bounded(*args,**kwargs):
        rows=original(*args,**kwargs);fallback.append(len(rows));return rows
    monkeypatch.setattr(init,'_v3_direction_feasibility_candidates',bounded)
    def deadlock(*args,**kwargs):
        calls.append(1);return ScheduleResult(ScheduleStatus.DEADLOCK)
    baseline=build_initial_solution(_v3_fixture(),FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,reference_evaluator=deadlock)
    enhanced=build_initial_solution(_v3_fixture(),FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,region_cooperative=True,reference_evaluator=deadlock)
    assert [row.strategy for row in baseline.attempts[:5]]==[
        'LOAD_FIRST','RAIL_BALANCED','X_ORDER_AWARE','SPATIAL_SPREAD',
        'RAIL_MONOTONE_BALANCED_BOOTSTRAP']
    assert [row.strategy for row in enhanced.attempts[:5]]==[
        'LOAD_FIRST','RAIL_BALANCED','X_ORDER_AWARE','REGION_COOPERATIVE_INITIALIZATION',
        'RAIL_MONOTONE_BALANCED_BOOTSTRAP']
    assert baseline.status is enhanced.status is InitializationStatus.INITIALIZATION_FAILED
    assert len(calls)<=2*(5+4+1)
    assert all(value<=4 for value in fallback)


def test_region_cooperative_default_switch_is_off_and_certifier_rejection_is_fatal():
    from mrta_reference.certifier import CertificationReport
    from mrta_search.pipeline import SearchConfig
    assert SearchConfig().initialization_region_cooperative is False
    def reject(*args,**kwargs):
        return CertificationReport(False,('deliberate rejection',),None)
    result=build_initial_solution(_v3_fixture(),FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,region_cooperative=True,certifier=reject)
    assert result.status is InitializationStatus.NUMERIC_FAILURE
    assert result.solution is None


def test_region_cooperative_preserves_x_split_child_robot_ownership():
    from mrta_reference.geometry import generate_x_split_patterns
    from mrta_reference.model import Rail
    parent=ParentWeld('x',(2.,9.),(18.,9.))
    pattern=next(row for row in generate_x_split_patterns(parent,10.,FAST,rail=Rail.UPPER)
                 if row.point_id=='BX_CENTER')
    solution=_construct_region_cooperative_initialization((parent,),(pattern,),FAST)
    assert solution.routes[0].block_ids==('x::0',)
    assert solution.routes[1].block_ids==('x::1',)
    assert not solution.routes[2].block_ids and not solution.routes[3].block_ids
    blocks=block_map(solution,FAST)
    assert all(block.length>=FAST.min_child_length-FAST.numeric_epsilon for block in blocks.values())


def _jrrb_state():
    parents=tuple(ParentWeld(f'j{i}',(1.+2*i,6.),(2.+2*i,6.)) for i in range(8))
    result=build_initial_solution(parents,FAST,_stats(),construction_budget=5,kinit_ref=5,
        portfolio=True,region_cooperative=True)
    assert result.status is InitializationStatus.SUCCESS
    return parents,result


def test_jrrb_joint_boundary_pool_is_bounded_structural_and_legal():
    parents,initial=_jrrb_state()
    pool=build_joint_rail_region_rebuild_candidates(
        initial.solution,initial.directions,initial.schedule,FAST)
    assert 0<pool.attempted<=16
    assert pool.constructed<=pool.attempted and pool.valid_unique<=pool.constructed
    assert len(pool.candidates)==pool.valid_unique
    assert len({row.solution.canonical_hash for row in pool.candidates})==pool.valid_unique
    assert any(row.b_up_changed and row.b_low_changed for row in pool.candidates)
    assert any(row.reordered_routes==4 for row in pool.candidates)
    for candidate in pool.candidates:
        assert candidate.cross_rail_parent_changes>0
        assert candidate.solution.canonical_hash!=initial.solution.canonical_hash
        assert candidate.solution.parents==tuple(sorted(parents,key=lambda row:row.parent_id))
        assert tuple(p.pattern_id for p in candidate.solution.patterns)==tuple(
            p.pattern_id for p in initial.solution.patterns)
        blocks=block_map(candidate.solution,FAST)
        assigned=[item for route in candidate.solution.routes for item in route.block_ids]
        assert len(assigned)==len(set(assigned))==len(blocks)
        assert all(robot_is_eligible(blocks[item],route.robot_id,FAST)
                   for route in candidate.solution.routes for item in route.block_ids)


def test_jrrb_preserves_mandatory_y_x_children_and_scientific_split_limits():
    from mrta_reference.geometry import generate_x_split_patterns
    from mrta_reference.model import Rail
    flexible=tuple(ParentWeld(f'f{i}',(1.+2*i,6.),(2.+2*i,6.)) for i in range(8))
    x_parent=ParentWeld('x',(2.,9.),(18.,9.))
    mandatory=ParentWeld('mandatory',(9.,5.),(9.,7.))
    parents=flexible+(x_parent,mandatory)
    patterns=list(_patterns(parents,FAST))
    x_pattern=next(row for row in generate_x_split_patterns(x_parent,10.,FAST,rail=Rail.UPPER)
                   if row.point_id=='BX_CENTER')
    patterns=tuple(x_pattern if row.parent_id=='x' else row for row in patterns)
    current=_construct_region_cooperative_initialization(parents,patterns,FAST)
    pool=build_joint_rail_region_rebuild_candidates(
        current,tuple(tuple(0 for _ in route.block_ids) for route in current.routes),
        ScheduleResult(ScheduleStatus.FEASIBLE),FAST)
    assert pool.valid_unique>0
    for candidate in pool.candidates:
        by_robot={item:route.robot_id for route in candidate.solution.routes for item in route.block_ids}
        assert by_robot['x::0']==0 and by_robot['x::1']==1
        mandatory_pattern=next(row for row in candidate.solution.patterns if row.parent_id=='mandatory')
        assert mandatory_pattern.kind is SplitKind.Y_SPLIT and mandatory_pattern.mandatory
        assert tuple(row.pattern_id for row in candidate.solution.patterns)==tuple(row.pattern_id for row in patterns)
        assert all(block.length>=FAST.min_child_length-FAST.numeric_epsilon
                   for block in block_map(candidate.solution,FAST).values())
    assert FAST.delta_x==pytest.approx(.5) and FAST.delta_y==pytest.approx(.5)
    assert FAST.min_child_length==pytest.approx(.2)
