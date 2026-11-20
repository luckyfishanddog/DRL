from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_joint_oriented_repair_pilot as oriented
from mrta_reference import scheduler
from mrta_reference.geometry import generate_y_split_patterns
from mrta_reference.model import DeadlockBlockage,Operation,OperationKind,ParentWeld,ScheduleResult,ScheduleStatus,ScientificConfig,SplitKind,SplitPattern
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize
from mrta_search.lns import select_schedule_blocker_parents


def test_observer_preserves_b32_and_keeps_selected_recovery_order():
    from test_scheduler_equivalence import _known_recoverable_large_case
    solution,config,directions,templates,baseline=_known_recoverable_large_case()
    original=scheduler.reference_schedule_from_templates_formal(templates,config,scope=FORMAL_SCOPE_V2,directions=directions)
    orders={}
    actual=scheduler.reference_schedule_from_templates_formal(templates,config,scope=FORMAL_SCOPE_V2,directions=directions,
        feasible_dispatch_observer=lambda result,order:orders.setdefault(result.operations,order))
    assert asdict(actual)==asdict(original)
    assert actual.feasible and actual.baseline_deadlock and orders
    selected=orders[actual.operations]
    events=scheduler.selected_schedule_wait_blockers(solution,directions,actual,selected,config)
    assert events
    for event in events:
        prefix=selected[:selected.index(event.waiting)]
        assert all(op_id in {op.operation_id for op in prefix} for op_id,_,_ in event.blocker_intervals)
    wrong=baseline.terminal_state.fixed_operations
    with pytest.raises(ValueError,match='final selected'):
        scheduler.selected_schedule_wait_blockers(solution,directions,actual,wrong,config)


def test_real_wait_before_move_maps_to_destination_and_actual_blocker():
    config=ScientificConfig(weld_speed=1.,empty_speed=100.,t_pre=1.,t_post=1.)
    parents=(ParentWeld('a',(1,9),(2,9)),ParentWeld('b',(17,9),(18,9)),ParentWeld('c',(14,9),(20,9)))
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                          {0:('a::whole','b::whole'),1:('c::whole',)},config,scope=FORMAL_SCOPE_V2)
    directions=((0,0),(0,),(),());orders={}
    actual=oriented.schedule_explicit(solution,directions,config,
        feasible_dispatch_observer=lambda result,order:orders.setdefault(result.operations,order))
    assert actual.schedule.feasible
    events=scheduler.selected_schedule_wait_blockers(solution,directions,actual.schedule,orders[actual.schedule.operations],config)
    event=next(e for e in events if e.delayed_operation.kind is OperationKind.MOVE)
    assert event.waiting.block_id is None and event.delayed_block_id=='b::whole'
    assert any(block_id=='c::whole' and op_id.startswith('R1:') for op_id,block_id,_ in event.blocker_intervals)


def _selection_fixture():
    config=ScientificConfig()
    parents=(ParentWeld('a',(2,5),(2,7)),ParentWeld('b',(5,9),(6,9)),
             ParentWeld('far',(17,9),(18,9)),ParentWeld('isolated',(18,2),(19,2)))
    pattern=generate_y_split_patterns(parents[0],config)[0]
    solution=canonicalize(parents,(pattern,)+tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents[1:]),
        {0:('a::1',),1:('b::whole','far::whole'),2:('a::0',),3:('isolated::whole',)},config,scope=FORMAL_SCOPE_V2)
    wait=Operation('wait',0,OperationKind.WAIT,2,7,(2,7),(2,7),0,'a::1')
    delayed=Operation('delayed',0,OperationKind.WELD,7,9,(2,7),(2,8),0,'a::1')
    event=scheduler.ScheduleWaitBlockers(wait,delayed,'a::1',(('blocking-post','b::whole',((2.,7.),)),))
    schedule=ScheduleResult(ScheduleStatus.FEASIBLE,(wait,),10.,(10.,8.,5.,1.))
    return solution,schedule,event,config


def test_blocker_core_closes_parent_and_only_fills_direct_route_neighbours():
    solution,schedule,event,config=_selection_fixture()
    partial,selected=select_schedule_blocker_parents(solution,schedule,(event,),q=4,seed=20261103,config=config)
    assert selected==event and partial.removed_parent_ids==('a','b','far')
    assert partial.routes==((),(),(),('isolated::whole',))
    assert partial.patterns==solution.patterns
    one,_=select_schedule_blocker_parents(solution,schedule,(event,),q=1,seed=20261103,config=config)
    assert one.removed_parent_ids==('a',)
    assert 'a::0' not in one.routes[2] and 'a::1' not in one.routes[0]


def test_blocker_priority_uses_interval_union_then_stable_parent_id():
    solution,schedule,event,config=_selection_fixture()
    event=replace(event,blocker_intervals=(('b1','b::whole',((2.,4.),)),('b2','b::whole',((2.,4.),)),
        ('f','far::whole',((2.,5.),)),('i','isolated::whole',((2.,5.),))))
    partial,_=select_schedule_blocker_parents(solution,schedule,(event,),q=2,seed=1,config=config)
    assert partial.removed_parent_ids==('a','far')


def test_no_wait_or_no_mappable_blocker_is_not_applicable():
    solution,schedule,event,config=_selection_fixture()
    for events in ((),(replace(event,waiting=replace(event.waiting,end_time=2.)),),
                   (replace(event,blocker_intervals=(('unknown','missing',((2.,7.),)),)),)):
        assert select_schedule_blocker_parents(solution,schedule,events,q=4,seed=1,config=config)==(None,None)


def _deadlock_templates():
    from mrta_reference.model import RobotRoute
    from mrta_reference.solution import block_map
    config=ScientificConfig(weld_speed=1.,empty_speed=1.,t_pre=1.,t_post=1.)
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(17.,9.),(18.,9.)),
             ParentWeld('c',(19.,9.),(11.,9.)))
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
        {0:('a::whole','b::whole'),1:('c::whole',)},config,scope=FORMAL_SCOPE_V2)
    blocks=block_map(solution,config)
    templates={r.robot_id:scheduler.build_operation_templates(RobotRoute(r.robot_id,
        tuple(blocks[b] for b in r.block_ids),(0,)*len(r.block_ids)),config) for r in solution.routes}
    return solution,config,templates


def test_deadlock_structured_evidence_matches_real_slow_and_optimized_cursors():
    from test_scheduler_equivalence import _known_recoverable_large_case
    solution,config,directions,templates,baseline=_known_recoverable_large_case()
    slow=scheduler.reference_schedule_from_templates_slow(templates,config)
    fast=scheduler.reference_schedule_from_templates_optimized(templates,config)
    assert slow.status is fast.status is ScheduleStatus.DEADLOCK
    assert slow.canonical_json()==fast.canonical_json()
    assert slow.deadlock_blockages==fast.deadlock_blockages and fast.deadlock_blockages
    assert replace(fast,deadlock_blockages=(),deadlock_evidence_source=None).canonical_json()==fast.canonical_json()
    committed={op.operation_id:op for op in fast.operations}
    for blockage in fast.deadlock_blockages:
        robot=blockage.robot_id
        cursor=sum(op.robot_id==robot and op.kind is not OperationKind.WAIT for op in fast.operations)
        assert blockage.next_template_index==cursor
        assert blockage.blocked_operation==templates[robot][cursor]
        assert blockage.ready_time==fast.robot_completion[robot]
        assert blockage.blocked_parent_ids
        for identifier,op in zip(blockage.blocker_operation_ids,blockage.blocker_operations):
            assert op==committed[identifier]


def test_deadlock_move_wait_and_weld_mapping_keeps_arc_source_and_destination():
    solution,config,templates=_deadlock_templates()
    move=next(op for op in templates[0] if op.kind is OperationKind.MOVE)
    assert move.block_id is None
    assert scheduler.operation_task_blocks(move,templates)==('a::whole','b::whole')
    wait=Operation('R0:WAIT:17',0,OperationKind.WAIT,3.,4.,move.start,move.start,move.sequence_index)
    assert scheduler.operation_task_blocks(wait,templates)==('a::whole','b::whole')
    weld=next(op for op in templates[0] if op.kind is OperationKind.WELD and op.block_id=='b::whole')
    assert scheduler.operation_task_blocks(weld,templates)==('b::whole',)
    evidence=scheduler._deadlock_blockages(templates,(3,0,0,0),(3.,0.,0.,0.),(wait,),
                                         {0:('R0:WAIT:17','unknown-blocker')})[0]
    assert evidence.blocked_parent_ids==('a','b')
    assert evidence.blocker_parent_ids==(('a','b'),())
    assert evidence.blocker_operation_ids[-1]=='unknown-blocker' and evidence.blocker_operations[-1] is None


def test_deadlock_recovery_reports_last_failure_without_changing_scientific_result(monkeypatch):
    from mrta_reference.dispatch_recovery import (DispatchState,DispatchChoice,BranchPoint,
        RolloutOutcome,RolloutTrace,limited_discrepancy_recovery)
    from test_scheduler_equivalence import _known_recoverable_large_case
    solution,config,directions,templates,baseline=_known_recoverable_large_case()
    first=scheduler.reference_schedule_from_templates_optimized(templates,config)
    later=replace(first,deadlock_blockages=tuple(replace(e,ready_time=e.ready_time+100.) for e in first.deadlock_blockages))
    state=DispatchState()
    choices=(DispatchChoice((0.,0.,0.,0),0,0,'chosen',0.),DispatchChoice((0.,0.,0.,1),1,0,'alternative',0.))
    branch=BranchPoint(0,0,state,choices,0)
    baseline=RolloutOutcome(first,RolloutTrace((branch,),0,1),state)
    summary=limited_discrepancy_recovery(baseline,rollout_budget=32,
        rollout_from_snapshot=lambda *a:RolloutOutcome(later,RolloutTrace((),0,1),state))
    assert summary.recovery_rollouts==1 and summary.result is first and summary.last_deadlock is later
    monkeypatch.setattr(scheduler,'limited_discrepancy_recovery',lambda *a,**k:summary)
    prepared=scheduler.prepare_dispatch_problem(templates,config)
    slow=scheduler._limited_discrepancy_dispatch_recovery_slow(templates,config,baseline,rollout_budget=32)
    fast=scheduler._limited_discrepancy_dispatch_recovery_optimized(prepared,baseline,rollout_budget=32)
    assert slow.deadlock_blockages==fast.deadlock_blockages==later.deadlock_blockages
    assert slow.canonical_json()==fast.canonical_json()
    assert fast.deadlock_evidence_source=='last recovery deadlock'
    assert fast.operations==first.operations and fast.wait_for_graph==first.wait_for_graph


def test_deadlock_parent_selection_closes_split_siblings_and_direct_neighbours():
    from mrta_search.lns import select_deadlock_blocker_parents
    solution,old,event,config=_selection_fixture()
    blocking=Operation('direct',1,OperationKind.POST,1.,2.,(6.,9.),(6.,9.),0,'b::whole')
    item=DeadlockBlockage(0,0,0.,event.delayed_operation,('a::1',),('a',),('direct',),
                         (blocking,),(('b::whole',),),(('b',),))
    schedule=ScheduleResult(ScheduleStatus.DEADLOCK,deadlock_blockages=(item,))
    partial=select_deadlock_blocker_parents(solution,schedule,config)
    assert partial.removed_parent_ids==('a','b','far')
    assert partial.routes==((),(),(),('isolated::whole',))
    unknown=replace(item,blocker_operation_ids=('unknown',),blocker_operations=(None,),
                    blocker_block_ids=((),),blocker_parent_ids=((),))
    assert select_deadlock_blocker_parents(solution,replace(schedule,deadlock_blockages=(unknown,)),config) is None


def test_deadlock_gnn_continuation_keeps_untouched_owners_order_and_directions(monkeypatch):
    import torch
    from test_gnn_construct_vns import _imitation_fixture
    from mrta_search.lns import _remove_parent_set
    e,c,parents,model,record=_imitation_fixture()
    space=c.parent_assignment_space(parents,e.CONFIG)
    current=e.deadlock_plan(parents,record['plan'],space)
    partial=_remove_parent_set(current.solution,('a',))
    monkeypatch.setattr(e,'b32',lambda *a,**k:pytest.fail('B32 called on a partial construction'))
    with torch.no_grad():
        rebuilt=e.deadlock_reconstruct(current,partial,model,space,model.encode_static(space.graph))
    assert rebuilt.solution is not None and 1<=len(rebuilt.candidates)<=2
    before={b:(r,d) for r,(route,bits) in enumerate(zip(current.solution.routes,current.directions))
            for b,d in zip(route.block_ids,bits) if not b.startswith('a::')}
    for plan in rebuilt.candidates:
        after={b:(r,d) for r,(route,bits) in enumerate(zip(plan.solution.routes,plan.directions))
               for b,d in zip(route.block_ids,bits)}
        assert all(after[b]==owner for b,owner in before.items())
        assert {p.parent_id for p in plan.solution.patterns}=={p.parent_id for p in parents}
        assert all(route.block_ids for route in plan.solution.routes)


def test_deadlock_fixed_bits_apply_existing_head_preference_before_pruning(monkeypatch):
    import torch
    from test_gnn_construct_vns import _imitation_fixture
    from mrta_search.direction import _initial_error
    from mrta_reference.geometry import oriented_endpoints
    from mrta_reference.solution import block_map
    e,c,_,model,_=_imitation_fixture()
    parents=tuple(ParentWeld(name,start,end) for name,start,end in
        [('k',(10.,9.),(11.,9.)),('b',(17.,9.),(18.,9.)),('c',(8.,9.),(12.,9.)),
         ('u',(1.,2.),(2.,2.)),('v',(17.,2.),(18.,2.))])
    space=c.parent_assignment_space(parents,e.CONFIG)
    patterns={p.parent_id:next(a.pattern for a in space.actions if a.parent_id==p.parent_id and
                             a.pattern.kind is SplitKind.WHOLE) for p in parents}
    blocks={b.block_id:b for p in patterns.values() for b in space.children[p]}
    routes=(('k::whole',),('b::whole',),('u::whole',),('v::whole',))
    state=e.lns.oriented_route_state(routes,((0,),)*4,blocks,e.CONFIG)
    nodes={b:space.block_nodes[patterns[value.parent_id],b] for b,value in blocks.items()}
    target=next(i for i,a in enumerate(space.actions) if a.parent_id=='c' and
                a.pattern.kind is SplitKind.WHOLE and a.robots==(1,))
    original=c.structural_child_indices;visits=[]
    def inspect(variants,**kwargs):
        visits.append(len(variants))
        for after,_ in variants:
            heads={r:oriented_endpoints(blocks[route[0]],after.directions[r][0])[0]
                   for r,route in enumerate(after.routes) if route}
            assert _initial_error(heads,e.CONFIG) is None
        return original(variants,**kwargs)
    monkeypatch.setattr(c,'structural_child_indices',inspect)
    with torch.no_grad():
        plans,_=c.vnext_joint_actions(space,model.encode_static(space.graph),model,state,
            {b:v for b,v in blocks.items() if v.parent_id!='c'},nodes,{'c'},e.CONFIG,
            assignment_indices=(target,),fixed_directions={b:0 for route in routes for b in route})
    assert plans and visits
