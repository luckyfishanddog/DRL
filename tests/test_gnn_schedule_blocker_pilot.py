from dataclasses import asdict, replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_joint_oriented_repair_pilot as oriented
from mrta_reference import scheduler
from mrta_reference.geometry import generate_y_split_patterns
from mrta_reference.model import Operation,OperationKind,ParentWeld,ScheduleResult,ScheduleStatus,ScientificConfig,SplitKind,SplitPattern
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
