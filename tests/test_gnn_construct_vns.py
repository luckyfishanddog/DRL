from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import random
import sys
import time
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gnn_construct_vns_pilot as pilot_script


def test_ranked_parent_budgets_keep_gnn_priority_and_two_distinct_supplements():
    from mrta_search.gnn_constructor import compact_parent_ids
    pending=tuple(f'p{i:02d}' for i in range(24))
    orders=dict(gnn=pending,mrv=tuple(reversed(pending)),lpt=pending[12:]+pending[:12])
    for limit in (4,6,8,12,16):
        for name in ('gnn','mrv','lpt'):
            assert compact_parent_ids({},pending,parent_limit=limit,strategy=name,orders=orders)==orders[name][:limit]
        ids=compact_parent_ids({},pending,parent_limit=limit,strategy='gnn_mrv_lpt',orders=orders)
        assert len(ids)==len(set(ids))==limit
        assert ids[:limit-2]==pending[:limit-2]
        assert ids[-2]=='p23' and ids[-1]==next(p for p in orders['lpt'] if p not in ids[:-1])


def test_existing_parent_head_scores_all_pending_without_future_teacher_modes(monkeypatch):
    from mrta_search import gnn_constructor as c
    from mrta_search import lns
    space=c.complete_parent_space(_parents(),CONFIG,compact=True)
    _,blocks,patterns=c.complete_parent_view(space,{})
    state=lns.oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    model=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    pending=tuple(p.parent_id for p in _parents());seen=[]
    monkeypatch.setattr(lns,'has_joint_block_candidate',lambda *a,**kw:False)
    def score(graph,emb,ids,features):
        assert set(ids)==set(pending) and features.shape[1]==20
        assert {b.block_id for b in graph.blocks}==set(blocks)
        seen.append(ids);return torch.arange(len(ids),dtype=torch.float32)
    monkeypatch.setattr(model,'parent_logits',score)
    monkeypatch.setattr(model,'mode_logits',lambda *a:(_ for _ in ()).throw(AssertionError('independent mode head')))
    orders=c.gnn_parent_orders(space,state,pending,{},model,CONFIG)
    assert len(seen)==1 and orders['gnn']==tuple(reversed(sorted(pending)))
    assert not c.joint_parent_inputs(state,pending,blocks,patterns,CONFIG)[0]


def test_after_parent_workload_changes_only_the_four_existing_input_columns():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state,joint_action_key
    parents=(ParentWeld('upper',(1.,9.),(3.,9.)),ParentWeld('lower',(14.,3.),(16.,3.)))
    space=c.complete_parent_space(parents,CONFIG,compact=True)
    _,blocks,_=c.complete_parent_view(space,{})
    state=oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    args=dict(parent_limit=4,first_limit=8,second_limit=4,include_remaining_workload=True,
              candidate_parent_ids=('upper','lower'))
    before=c.complete_parent_candidates(space,state,('upper','lower'),{},CONFIG,**args)
    after=c.complete_parent_candidates(space,state,('upper','lower'),{},CONFIG,remaining_after_parent=True,**args)
    assert [(p['pattern'],tuple(map(joint_action_key,p['actions']))) for p in before]==[(p['pattern'],tuple(map(joint_action_key,p['actions']))) for p in after]
    for p in after:
        expected=torch.tensor([0.,0.,.5,.5]) if p['parent_id']=='upper' else torch.tensor([.5,.5,0.,0.])
        assert torch.equal(p['remaining_eligible_workload'],expected)
    model=c.CompleteParentGNN(16,4,remaining_workload_features=True)
    weights={k:v.clone() for k,v in model.state_dict().items()}
    inputs_old=c.complete_parent_inputs(before);inputs_new=c.complete_parent_inputs(after)
    embedded=model.encode_static(space['graph'])
    rows_old=model.parent_action_rows(space['graph'],embedded,inputs_old)
    rows_new=model.parent_action_rows(space['graph'],embedded,inputs_new)
    assert torch.equal(rows_old[:,:-4],rows_new[:,:-4])
    assert torch.equal(rows_new[:,-4:],inputs_new['remaining_eligible_workload'])
    assert rows_new.shape[1]==model.parent_action_head[0].in_features
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in weights.items())


def test_ranked_joint_decoder_uses_existing_parent_and_complete_heads_atomically(monkeypatch):
    from mrta_search import gnn_constructor as c
    parents=(ParentWeld('x',(0.,8.),(4.,8.)),)
    model=c.CompleteParentGNN(16,4,remaining_workload_features=True);seen=[]
    def score(graph,h,inputs):
        assert torch.equal(inputs['remaining_eligible_workload'],torch.zeros(len(inputs['parent_ids']),4))
        seen.append(inputs);return inputs['mode_features'][:,2]*100-inputs['child_features'][:,:,19].sum(1)
    monkeypatch.setattr(model,'parent_action_scores',score)
    monkeypatch.setattr(model,'mode_logits',lambda *a:(_ for _ in ()).throw(AssertionError('independent mode head')))
    result=c.construct_complete_parent_gnn(parents,model,CONFIG,parent_limit=4,parent_strategy='gnn_mrv_lpt',remaining_after_parent=True)
    assert len(seen)==1 and result.solution is not None and len(result.actions)==2
    assert result.solution.patterns[0].kind is SplitKind.X_SPLIT
    assert result.solution.routes[0].block_ids==('x::0',) and result.solution.routes[1].block_ids==('x::1',)


def test_route_progress_decoder_keeps_the_measured_whole_variants_and_residual_rows():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_action_key, oriented_route_state

    parents = tuple(ParentWeld(f'w{i}', (1. + 2 * i, 9.), (2. + 2 * i, 9.)) for i in range(4))
    space = c.complete_parent_space(parents, CONFIG, compact=True)
    selected = {p.parent_id: space['catalog'][p.parent_id][0] for p in parents[:3]}
    pending = {'w3'}
    _, blocks, _ = c.complete_parent_view(space, selected)
    route = tuple(space['options'][pattern][0].block_id for pattern in selected.values())
    state = oriented_route_state((route, (), (), ()), ((0, 0, 0), (), (), ()), blocks, CONFIG)
    model = c.CompleteParentGNN(16, 4, remaining_workload_features=True).eval()
    model.compact_parent_limit = 16

    def keys(plans):
        return [(plan['pattern'], tuple(map(joint_action_key, plan['actions']))) for plan in plans]

    with torch.inference_mode():
        expected = c.route_progress_complete_candidates(space, state, pending, selected, model, CONFIG)
    assert sum(p['pattern'].kind is SplitKind.WHOLE for p in expected) > 16
    seen = []

    def observe(_, __, ___, ____, plans, best):
        seen.append(keys(plans))
        assert torch.equal(best['remaining_eligible_workload'], torch.zeros(4))

    result = c.construct_complete_parent_gnn(parents, model, CONFIG, space=space, state=state, pending=pending,
        selected=selected, observer=observe)
    assert result.solution is not None
    assert seen == [keys(expected)]


def test_residual_preferences_distinguish_cmax_and_feasibility():
    from run_gnn_cmax_finetune import residual_feedback_pairs

    outcomes = [
        dict(status='FEASIBLE', cmax=20.),
        dict(status='DEADLOCK', cmax=None),
        dict(status='FEASIBLE', cmax=19.),
        dict(status='DEADLOCK', cmax=None),
        dict(status='NUMERIC_FAILURE', cmax=None),
    ]
    assert {(p['better'], p['worse'], p['kind']) for p in residual_feedback_pairs(outcomes)} == {
        (2, 0, 'cmax'),
        (0, 1, 'feasibility'), (0, 3, 'feasibility'),
        (2, 1, 'feasibility'), (2, 3, 'feasibility'),
    }


def test_compact_mrv_lpt_flexibility_observes_forced_x_robots_and_no_neural_parent_head():
    from mrta_search import gnn_constructor as c
    space=c.complete_parent_space(_parents(),CONFIG,compact=True)
    for parent in _parents():
        expected=0
        for pattern in space['catalog'][parent.parent_id]:
            possibilities=1
            for child in space['options'][pattern]:
                possibilities*=1 if pattern.kind is SplitKind.X_SPLIT else sum(robot_is_eligible(child,r,CONFIG) for r in range(4))
            expected+=possibilities
        assert space['flexibility'][parent.parent_id]==expected
    ids=tuple(p.parent_id for p in _parents())
    scarce=sorted(ids,key=lambda p:(space['flexibility'][p],-space['original_processing'][p],p))[:2]
    long=sorted(ids,key=lambda p:(-space['original_processing'][p],space['flexibility'][p],p))[:2]
    assert c.compact_parent_ids(space,ids)==tuple(dict.fromkeys(scarce+long))
    assert len(c.compact_parent_ids(space,ids,parent_limit=6))<=6


def test_remaining_workload_is_category_averaged_normalized_and_excludes_committed_parents():
    from mrta_search import gnn_constructor as c
    parents=(ParentWeld('upper',(2.,9.),(8.,9.)),ParentWeld('lower',(2.,3.),(3.,3.)))
    space=c.complete_parent_space(parents,CONFIG,compact=True)
    # Each real X cut uses R0/R1 exclusively, even when a child is reachable
    # from more robots geometrically; WHOLE divides upper work equally.
    patterns=space['catalog']['upper'];xs=[p for p in patterns if p.kind is SplitKind.X_SPLIT]
    assert xs and {p.kind for p in patterns}=={SplitKind.WHOLE,SplitKind.X_SPLIT}
    whole=CONFIG.process_time(6.)
    left=sum(CONFIG.process_time(space['options'][p][0].length) for p in xs)/len(xs)
    right=sum(CONFIG.process_time(space['options'][p][1].length) for p in xs)/len(xs)
    expected=torch.tensor([(whole/2+left)/2,(whole/2+right)/2,0.,0.])
    assert torch.allclose(torch.tensor(space['remaining_parent_workload']['upper']),expected)
    assert torch.allclose(c.remaining_eligible_workload(space,('lower',)),torch.tensor([0.,0.,.5,.5]))
    assert torch.isclose(c.remaining_eligible_workload(space,('upper','lower')).sum(),torch.tensor(1.))
    assert torch.equal(c.remaining_eligible_workload(space,()),torch.zeros(4))


def test_remaining_feature_expansion_preserves_old_scores_freezes_encoder_and_roundtrips():
    import io
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state
    source=c.CompleteParentGNN(16,4)
    model=shared.expand_remaining_workload_model(source)
    space=c.complete_parent_space(_parents(),CONFIG,compact=True)
    _,blocks,_=c.complete_parent_view(space,{})
    state=oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    plans=c.complete_parent_candidates(space,state,tuple(p.parent_id for p in _parents()),{},CONFIG,
        first_limit=8,second_limit=4,parent_limit=6,include_remaining_workload=True)
    inputs=c.complete_parent_inputs(plans)
    with torch.no_grad():
        emb=model.encode_static(space['graph']);rows=model.parent_action_rows(space['graph'],emb,inputs).clone()
        repeated=torch.stack([emb[space['graph'].parent_block_indices[p]].mean(0) for p in inputs['parent_ids']])
        assert torch.equal(rows[:,:model.hidden_dim],repeated)
        old=source.parent_action_scores(space['graph'],source.encode_static(space['graph']),inputs)
        assert torch.allclose(model.parent_action_head(rows).squeeze(1),old,atol=1e-6)
    assert torch.count_nonzero(model.parent_action_head[0].weight[:,-4:])==0
    assert all(p.requires_grad==n.startswith('parent_action_head.') for n,p in model.named_parameters())
    before={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('parent_action_head.')}
    optimizer=torch.optim.AdamW(model.parent_action_head.parameters(),lr=1e-4)
    (-torch.log_softmax(model.parent_action_head(rows).squeeze(1),0)[0]).backward();optimizer.step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    branch=dict(model_type='CompleteParentGNN',hidden_dim=16,heads=4,normalize_inputs=True,
        opportunity_features=True,parent_action_features=True,remaining_workload_features=True,
        compact_parent_limit=6,state_dict=model.state_dict(),training=dict(epochs=10))
    stream=io.BytesIO();shared.save_neural_checkpoint(dict(branch,compact_workload_candidate=branch),stream)
    stream.seek(0);full=shared.read_neural_checkpoint(stream)
    assert full['compact_workload_candidate']['training']==dict(epochs=10)
    stream.seek(0);thin=shared.read_neural_checkpoint(stream,include_research=False)
    loaded=shared._model_from_checkpoint(thin['compact_workload_candidate'])
    assert loaded.remaining_workload_features and all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())


def test_complete_parent_pool_covers_all_parents_and_legal_modes_without_policy_heads():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state
    parents=_parents()
    space=c.complete_parent_space(parents,CONFIG)
    _,blocks,_=c.complete_parent_view(space,{})
    state=oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    plans=c.complete_parent_candidates(space,state,{p.parent_id for p in parents},{},CONFIG)
    assert {p['parent_id'] for p in plans}=={p.parent_id for p in parents}
    assert {p['pattern'] for p in plans}=={p for ps in space['catalog'].values() for p in ps}
    assert {p['pattern'].kind for p in plans}=={SplitKind.WHOLE,SplitKind.X_SPLIT,SplitKind.Y_SPLIT}
    assert all(len(p['actions'])==(1 if p['pattern'].kind is SplitKind.WHOLE else 2) for p in plans)
    assert all(len({a.block_id for a in p['actions']})==len(p['actions']) for p in plans)
    assert not any(state.routes) and not any(state.loads)
    for pattern in space['catalog']['mandatory']:
        for block in space['options'][pattern]:
            assert space['graph'].blocks[space['indices'][pattern,block.block_id]].start==block.start


def test_complete_parent_second_child_uses_temporary_routes_and_failed_mode_is_not_committed(monkeypatch):
    from mrta_search import gnn_constructor as c
    from mrta_search import lns
    parents=(ParentWeld('x',(0.,8.),(4.,8.)),)
    space=c.complete_parent_space(parents,CONFIG)
    _,blocks,_=c.complete_parent_view(space,{})
    state=lns.oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    original=lns.joint_block_candidates;seen=[]
    def reject_second(partial,child,pattern,*args,**kwargs):
        if child.block_id.endswith('::1'):
            seen.append(partial)
            assert any('x::0' in route for route in partial.routes)
            return ()
        return original(partial,child,pattern,*args,**kwargs)
    monkeypatch.setattr(lns,'joint_block_candidates',reject_second)
    plans=c.complete_parent_candidates(space,state,('x',),{},CONFIG)
    assert seen and all(p['pattern'].kind is SplitKind.WHOLE for p in plans)
    assert state.routes==((),)*4


@pytest.mark.parametrize('compact',(False,True))
def test_complete_parent_neural_mode_atomic_assignment_and_directions_enter_real_b32(monkeypatch,compact):
    from mrta_search import gnn_constructor as c
    import run_joint_oriented_repair_pilot as direct
    model=c.CompleteParentGNN(16,4,remaining_workload_features=compact)
    def forbidden(*args):raise AssertionError('independent old policy head was called')
    monkeypatch.setattr(model,'parent_logits',forbidden)
    monkeypatch.setattr(model,'joint_action_logits',forbidden)
    monkeypatch.setattr(model,'mode_logits',forbidden)
    monkeypatch.setattr(model,'parent_action_scores',lambda graph,h,x:
        x['mode_features'][:,2]*100-x['child_features'][:,:,19].sum(1))
    observed=[]
    result=c.construct_complete_parent_gnn((ParentWeld('x',(0.,8.),(4.,8.)),),model,CONFIG,
        observer=lambda sp,s,pending,selected,plans,best:observed.append((s,pending,best)))
    assert result.solution is not None and len(observed)==1
    assert observed[0][0].routes==((),)*4 and observed[0][1]==('x',)
    assert result.solution.patterns[0].kind is SplitKind.X_SPLIT
    assert result.solution.routes[0].block_ids==('x::0',)
    assert result.solution.routes[1].block_ids==('x::1',)
    assert result.directions[:2]==((0,),(0,)) and len(result.actions)==2
    assert direct.schedule_explicit(result.solution,result.directions,CONFIG).schedule.status is ScheduleStatus.FEASIBLE


def test_complete_parent_scalar_gradients_reach_both_gat_layers_and_archive_roundtrips():
    import io
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state
    model=c.CompleteParentGNN(16,4);space=c.complete_parent_space(_parents(),CONFIG)
    _,blocks,_=c.complete_parent_view(space,{})
    state=oriented_route_state(((),)*4,((),)*4,blocks,CONFIG)
    plans=c.complete_parent_candidates(space,state,tuple(p.parent_id for p in _parents()),{},CONFIG)
    inputs=c.complete_parent_inputs(plans)
    scores=model.parent_action_scores(space['graph'],model.encode_static(space['graph']),inputs)
    (-torch.log_softmax(scores,0)[0]).backward()
    for prefix in ('layer1.','layer2.','parent_action_head.'):
        assert any(p.grad is not None and torch.count_nonzero(p.grad) for n,p in model.named_parameters() if n.startswith(prefix))
    branch=dict(model_type='CompleteParentGNN',hidden_dim=16,heads=4,normalize_inputs=True,
                opportunity_features=True,parent_action_features=True,state_dict=model.state_dict(),
                training=dict(epochs=10),feedback_state_dict=model.state_dict())
    stream=io.BytesIO();shared.save_neural_checkpoint(dict(branch,complete_parent_candidate=branch),stream)
    stream.seek(0);full=shared.read_neural_checkpoint(stream)
    assert full['complete_parent_candidate']['training']==dict(epochs=10)
    stream.seek(0);thin=shared.read_neural_checkpoint(stream,include_research=False)
    loaded=shared._model_from_checkpoint(thin['complete_parent_candidate'])
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())


def test_complete_parent_cmax_pairs_only_use_true_same_state_feasible_values():
    import run_gnn_cmax_finetune as study
    outcomes=[dict(status='FEASIBLE',cmax=20.),dict(status='DEADLOCK',cmax=None),
              dict(status='FEASIBLE',cmax=10.),dict(status='CONSTRUCTION_FAILED',cmax=None)]
    assert study.complete_cmax_pairs(outcomes)==[(2,0)]


def test_diverse_child_pool_keeps_a_different_robot_then_position():
    from types import SimpleNamespace as NS
    from mrta_search.gnn_constructor import diverse_child_indices
    variants=[(NS(loads=(10.,0.,0.,0.)),NS(robot=r,position=p,direction=d))
              for r,p,d in ((0,0,0),(0,0,1),(0,1,0),(1,0,0))]
    logits=torch.tensor([4.,3.,2.,1.])
    assert diverse_child_indices(variants,logits)==(0,3)
    assert diverse_child_indices(variants[:3],logits[:3])==(0,2)
    assert diverse_child_indices(variants,logits,limit=1)==(0,)


def test_diverse_parent_pool_preserves_top_four_and_adds_bounded_legal_representatives():
    from mrta_search import gnn_constructor as c
    parents=tuple(ParentWeld(p,(x,9.),(x+1.,9.)) for p,x in (('a',1.),('b',4.),('c',7.),('d',10.)))+(
        ParentWeld('e',(12.,9.),(13.,9.)),ParentWeld('z',(2.,3.),(17.,3.)),ParentWeld('y',(3.,6.),(4.,6.)))
    ref=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    _,blocks,patterns=c.joint_graph(ref,CONFIG)
    pending=tuple(p.parent_id for p in ref.parents);state=c.joint_start_state(ref,((),)*4,((),)*4,pending,blocks,CONFIG)
    ids,_=c.joint_parent_inputs(state,pending,blocks,patterns,CONFIG)
    values=torch.tensor([10.-i for i in range(len(ids))])
    selected=c.diverse_parent_indices(state,ids,values,blocks,patterns,CONFIG)
    assert selected[:4]==(0,1,2,3) and len(selected)<=6
    assert ids.index('z') in selected and len(set(selected))==len(selected)


def test_complete_action_head_is_length_neutral_at_zero_and_rank_reaches_existing_gat():
    import run_gnn_shared_construction as shared
    import run_gnn_cmax_finetune as study
    from mrta_search import gnn_constructor as c
    torch.manual_seed(17)
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('cross',(10.,9.),(10.,3.)))
    ref=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    graph,blocks,patterns=c.joint_graph(ref,CONFIG);pending=tuple(p.parent_id for p in ref.parents)
    state=c.joint_start_state(ref,((),)*4,((),)*4,pending,blocks,CONFIG)
    source=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    model=shared.expand_parent_action_model(source)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
    plans=study.cmax_plans(ref,graph,blocks,patterns,state,pending,source)
    inputs=c.parent_action_inputs(graph,state,plans,blocks,CONFIG)
    assert set(inputs['child_mask'].sum(1).tolist())=={1.,2.}
    record=dict(graph_key='g',inputs=inputs)
    loss,parts=shared.opportunity_ranking_loss(model,{'g':graph},[],[(record,0,len(plans)-1)],
        complete_actions=True,imitation_weight=.1,load_weight=0.)
    assert torch.equal(loss,parts['rank'])
    loss.backward()
    for prefix in ('layer1','layer2','parent_action_head'):
        assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix+'.') and p.grad is not None)>0
    with torch.no_grad():
        for parameter in model.parent_action_head.parameters():parameter.zero_()
        scores=model.parent_action_scores(graph,model.encode_static(graph),inputs)
    assert torch.equal(scores,torch.zeros(len(plans)))


def test_complete_action_model_archive_retains_old_weights_and_new_head():
    import io
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import ModeJointInsertionGNN
    source=ModeJointInsertionGNN(16,4,opportunity_features=True);model=shared.expand_parent_action_model(source)
    base=dict(model_type='ModeJointInsertionGNN',hidden_dim=16,heads=4,normalize_inputs=True,opportunity_features=True,
              use_dynamic_context=False,state_dict=source.state_dict())
    payload=dict(base,parent_cmax_candidate=dict(base,parent_action_features=True,state_dict=model.state_dict(),training=dict(epochs=10)),
        parent_cmax_macro_il_candidate=dict(base,parent_action_features=True,state_dict=model.state_dict(),training=dict(frozen_base=True)))
    stream=io.BytesIO();shared.save_neural_checkpoint(payload,stream);stream.seek(0)
    full=shared.read_neural_checkpoint(stream)
    assert full['parent_cmax_candidate']['training']==dict(epochs=10)
    assert full['parent_cmax_macro_il_candidate']['training']==dict(frozen_base=True)
    stream.seek(0);thin=shared.read_neural_checkpoint(stream,include_research=False)
    loaded=shared._model_from_checkpoint(thin['parent_cmax_candidate'])
    assert loaded.parent_action_features and 'training' not in thin['parent_cmax_candidate']
    assert all(torch.equal(v,thin['state_dict'][k]) for k,v in source.state_dict().items())
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    corrected=shared._model_from_checkpoint(thin['parent_cmax_macro_il_candidate'])
    assert all(torch.equal(v,corrected.state_dict()[k]) for k,v in model.state_dict().items())


def test_direct_macro_imitation_replays_both_children_and_only_trains_scalar_head():
    from dataclasses import asdict
    import run_gnn_cmax_finetune as study
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    torch.manual_seed(17)
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('cross',(10.,9.),(10.,3.)))
    reference=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    source=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    pending=tuple(p.parent_id for p in parents)
    result=c.construct_or_repair_with_gnn(reference,((),)*4,((),)*4,pending,source,CONFIG)
    assert result.solution is not None
    graph,steps,_=c.prepare_joint_teacher_steps(reference,((),)*4,((),)*4,pending,
        [asdict(a) for a in result.actions],CONFIG,include_opportunities=True)
    key=study.op_graph_key(reference)
    examples=[]
    for step in steps:
        target=torch.zeros(len(step['candidate_actions']));target[step['target']]=1.
        examples.append(dict(kind='insertion',graph_key=key,step=step,target=target,weight=1.))
    macro,coverage=study.cmax_expert_macro_examples(source,{key:graph},examples)
    assert coverage['expert_macro_examples']==len(macro)==2
    assert sorted(int(e['inputs']['child_mask'][e['target']].sum()) for e in macro)==[1,2]
    assert max(e['progress'] for e in macro)==.5
    model=shared.expand_parent_action_model(source)
    model.requires_grad_(False);model.parent_action_head.requires_grad_(True)
    optimizer=torch.optim.AdamW(model.parent_action_head.parameters(),lr=1e-4)
    head_before={k:v.clone() for k,v in model.parent_action_head.state_dict().items()}
    losses=[]
    with torch.no_grad():
        embedded=model.encode_static(graph)
        cached=[model.parent_action_rows(graph,embedded,e['inputs']).clone() for e in macro]
    for e,rows in zip(macro,cached):
        scores=model.parent_action_head(rows).squeeze(1)
        assert torch.equal(scores,model.parent_action_scores(graph,embedded,e['inputs']))
        losses.append(-torch.log_softmax(scores,0)[e['target']])
    torch.stack(losses).mean().backward();optimizer.step()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items())
    assert any(not torch.equal(v,model.parent_action_head.state_dict()[k]) for k,v in head_before.items())


def test_complete_parent_feedback_never_reschedules_legacy_and_adds_at_most_one_b32(monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace as NS
    import run_gnn_cmax_finetune as study
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    reference=_joint_simple_reference();model=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    ref=c.joint_geometry(reference.parents,reference.patterns,CONFIG)
    pending=tuple(p.parent_id for p in ref.parents)
    graph,blocks,patterns=c.joint_graph(ref,CONFIG)
    state=c.joint_start_state(ref,((),)*4,((),)*4,pending,blocks,CONFIG)
    actions=study.op_representative_actions(ref,state,pending,model,CONFIG,early=True)
    outcomes=[]
    for i,action in enumerate(actions):
        result=c.construct_or_repair_with_gnn(ref,((),)*4,((),)*4,pending,model,CONFIG,forced_action=action)
        assert result.solution is not None
        outcomes.append(dict(action=asdict(action),status='FEASIBLE',cmax=100.+i,
            complete=dict(solution=asdict(result.solution),directions=result.directions)))
    row=dict(phase='early_counterfactual',instance='unit',payload=dict(fraction=0.,parents=[asdict(p) for p in ref.parents],
        prefix=dict(routes=((),)*4,directions=((),)*4,pending=pending),outcomes=outcomes))
    written=[];calls=[]
    monkeypatch.setattr(study,'op_read',lambda:[row]);monkeypatch.setattr(study,'op_save',written.append)
    monkeypatch.setattr(shared,'read_neural_checkpoint',lambda *a,**k:{study.OP_VARIANT:{}})
    monkeypatch.setattr(shared,'_model_from_checkpoint',lambda *a:model)
    def schedule(solution,directions,config):
        calls.append((solution,directions))
        return NS(schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=100.))
    monkeypatch.setattr(study.pilot.oriented,'schedule_explicit',schedule)
    study.op_parent_cmax_data()
    assert len(written)==1 and len(calls)<=1 and written[0]['b32_calls']==len(calls)
    assert sum(o['origin']=='one_extra' for o in written[0]['payload']['outcomes'])<=1
    assert all(o['reused'] and o['b32_calls']==0 for o in written[0]['payload']['outcomes'] if o['origin']=='legacy')


def test_opportunity_counts_use_actual_rail_and_initial_head_legality():
    from mrta_reference.model import ParentWeld
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(5.,9.),(6.,9.)))
    ref=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    _,blocks,patterns=c.joint_graph(ref,CONFIG)
    empty=c.joint_start_state(ref,((),)*4,((),)*4,('a','b'),blocks,CONFIG)
    before=c.remaining_robot_work(empty,blocks,patterns,CONFIG)
    assert before[:4]==pytest.approx((1.,1.,0.,0.)) and before[4:]==(0.,)*4
    child=next(b for b in blocks.values() if b.parent_id=='a')
    after=next(s for s,a in joint_block_candidates(empty,child,patterns['a'],blocks,CONFIG)
               if a.robot==1 and a.direction==0)
    remaining=next(b for b in blocks.values() if b.parent_id=='b')
    actual={a.robot for _,a in joint_block_candidates(after,remaining,patterns['b'],blocks,CONFIG)}
    assert actual=={1}
    work=c.remaining_robot_work(after,blocks,patterns,CONFIG)
    assert work[0]==work[2]==work[3]==0. and work[1]>0.
    assert work[4:]==pytest.approx((0.,work[1],0.,0.))


def test_action_delta_excludes_current_child_from_both_opportunity_sums():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    parent=ParentWeld('only',(1.,9.),(2.,9.))
    geometry=c.joint_geometry((parent,),c.fixed_split_patterns((parent,),CONFIG),CONFIG)
    _,blocks,patterns=c.joint_graph(geometry,CONFIG)
    empty=c.joint_start_state(geometry,((),)*4,((),)*4,('only',),blocks,CONFIG)
    variants=joint_block_candidates(empty,next(iter(blocks.values())),patterns['only'],blocks,CONFIG)
    assert c.joint_opportunity_deltas(empty,variants,blocks,patterns,CONFIG,normalize=False)==((0.,)*4,)*len(variants)


def test_action_delta_observes_loss_on_the_other_robot_and_uses_process_time():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(5.,9.),(6.,9.)))
    geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    _,blocks,patterns=c.joint_graph(geometry,CONFIG)
    empty=c.joint_start_state(geometry,((),)*4,((),)*4,('a','b'),blocks,CONFIG)
    chosen=next(v for v in joint_block_candidates(empty,blocks['a::whole'],patterns['a'],blocks,CONFIG)
                if v[1].robot==1 and v[1].direction==0)
    raw=c.joint_opportunity_deltas(empty,[chosen],blocks,patterns,CONFIG,normalize=False)[0]
    assert raw==pytest.approx((-CONFIG.process_time(blocks['b::whole'].length),0.,0.,0.))
    scaled=c.joint_opportunity_deltas(empty,[chosen],blocks,patterns,CONFIG)[0]
    scale=sum(CONFIG.process_time(b.length) for b in blocks.values())
    assert scaled==pytest.approx(tuple(x/scale for x in raw))


@pytest.mark.parametrize('kind',('whole','x_split','mandatory_y'))
@pytest.mark.parametrize('check_start',(False,True))
def test_action_delta_local_reuse_matches_full_calculation_with_heads_and_siblings(kind,check_start):
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    from mrta_reference.geometry import build_legal_pattern_catalog
    p=ParentWeld('p',(10.,9.),(10.,3.)) if kind=='mandatory_y' else ParentWeld('p',(0.,8.),(4.,8.))
    parents=(p,ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(4.,9.),(5.,9.)),
             ParentWeld('d',(7.,9.),(8.,9.)),ParentWeld('q',(12.,9.),(13.,9.)))
    patterns=list(c.fixed_split_patterns(parents,CONFIG))
    if kind=='x_split':
        pattern=next(v for v in build_legal_pattern_catalog((p,),CONFIG,FORMAL_SCOPE_V2)['p']
                     if v.kind is SplitKind.X_SPLIT and v.point_id=='BX_CENTER')
        patterns=[pattern if v.parent_id=='p' else v for v in patterns]
    geometry=c.joint_geometry(parents,tuple(patterns),CONFIG)
    _,blocks,patterns=c.joint_graph(geometry,CONFIG)
    state=c.joint_start_state(geometry,(('a::whole','b::whole','d::whole'),(),(),()),
                            ((0,1,0),(),(),()),('p','q'),blocks,CONFIG)
    children=sorted((b for b in blocks.values() if b.parent_id=='p'),key=lambda b:b.block_id)
    variants=joint_block_candidates(state,children[0],patterns['p'],blocks,CONFIG,check_initial_feasibility=check_start)
    assert variants
    stats={}
    actual=c.joint_opportunity_deltas(state,variants,blocks,patterns,CONFIG,
        check_initial_feasibility=check_start,reuse=True,stats=stats)
    expected=c.joint_opportunity_deltas(state,variants,blocks,patterns,CONFIG,
        check_initial_feasibility=check_start,reuse=False)
    assert actual==expected and stats['cache_hits']>0
    if len(children)==2:
        for after,_ in variants[:3]:
            second=joint_block_candidates(after,children[1],patterns['p'],blocks,CONFIG,
                                          check_initial_feasibility=check_start)
            assert c.joint_opportunity_deltas(after,second,blocks,patterns,CONFIG,
                check_initial_feasibility=check_start,reuse=True)==c.joint_opportunity_deltas(
                after,second,blocks,patterns,CONFIG,check_initial_feasibility=check_start,reuse=False)


def test_opportunity_expansion_preserves_old_parameters_and_scalers():
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import ModeJointInsertionGNN
    old=ModeJointInsertionGNN(16,4,normalize_inputs=True);new=shared.expand_opportunity_model(old)
    for name,value in old.state_dict().items():
        target=new.state_dict()[name]
        if value.shape==target.shape:assert torch.equal(target,value)
        elif value.ndim==2:
            assert torch.equal(target[:,:value.shape[1]],value)
            assert torch.count_nonzero(target[:,value.shape[1]:])==0
        else:assert torch.equal(target[:len(value)],value)
    new.fit_opportunity_scaling([dict(features=torch.randn(5,49),parent_features=torch.randn(5,20))])
    for prefix,n in (('joint',41),('parent',12)):
        assert torch.equal(getattr(new,prefix+'_mean')[:n],getattr(old,prefix+'_mean'))
        assert torch.equal(getattr(new,prefix+'_scale')[:n],getattr(old,prefix+'_scale'))
    assert new.use_dynamic_context is False


def test_counterfactual_pair_labels_never_rank_two_failures():
    from run_gnn_cmax_finetune import op_pair_labels
    rows=[dict(status='DEADLOCK',cmax=None),dict(status='CONSTRUCTION_FAILED',cmax=None),
          dict(status='FEASIBLE',cmax=20.),dict(status='FEASIBLE',cmax=10.)]
    pairs=op_pair_labels(rows)
    assert (0,1) not in pairs and (1,0) not in pairs
    assert (3,2) in pairs and (2,0) in pairs and (3,1) in pairs


def test_forced_joint_decision_uses_the_same_pool_then_neural_completion():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    reference=_joint_simple_reference();torch.manual_seed(8)
    model=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    geometry=c.joint_geometry(reference.parents,reference.patterns,CONFIG)
    _,blocks,patterns=c.joint_graph(geometry,CONFIG)
    pending=tuple(p.parent_id for p in reference.parents)
    state=c.joint_start_state(geometry,((),)*4,((),)*4,pending,blocks,CONFIG)
    block=next(iter(blocks.values()))
    action=joint_block_candidates(state,block,patterns[block.parent_id],blocks,CONFIG)[-1][1]
    snapshots=[]
    result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,pending,model,CONFIG,
        forced_action=action,parent_observer=lambda s,p,a:snapshots.append((len(p),len(a))))
    assert result.solution is not None and result.actions[0]==action
    assert snapshots[0]==(2,0) and snapshots[1]==(1,1)
    with pytest.raises(ValueError,match='unchanged legal pool'):
        c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,pending,model,CONFIG,
            forced_action=replace(action,position=999))


def test_cmax_rank_loss_backpropagates_through_both_existing_policy_heads():
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    reference=_joint_simple_reference();torch.manual_seed(17)
    model=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    geometry=c.joint_geometry(reference.parents,reference.patterns,CONFIG)
    graph,blocks,patterns=c.joint_graph(geometry,CONFIG)
    pending=tuple(p.parent_id for p in reference.parents)
    state=c.joint_start_state(geometry,((),)*4,((),)*4,pending,blocks,CONFIG)
    ids,pf=c.joint_parent_inputs(state,pending,blocks,patterns,CONFIG,include_opportunities=True)
    options=[]
    for index,parent in enumerate(ids):
        child=next(b for b in blocks.values() if b.parent_id==parent)
        variants=joint_block_candidates(state,child,patterns[parent],blocks,CONFIG)
        indices,features=c.joint_candidate_inputs(graph,state,variants,blocks,CONFIG,include_opportunities=True,patterns=patterns)
        options.append(dict(parent_index=index,indices=indices,features=features,action_index=0))
    record=dict(graph_key='g',parent_ids=ids,parent_features=pf,options=options)
    loss,parts=shared.opportunity_ranking_loss(model,{'g':graph},[],[(record,0,1)])
    assert loss==parts['rank'] and parts['il']==parts['load']==0.
    loss.backward()
    for name in ('layer1','layer2','parent_head','insertion_head'):
        assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(name+'.') and p.grad is not None)>0.


@pytest.mark.parametrize('first_decode',('greedy','joint_greedy','parent_joint'))
def test_single_model_three_proposals_have_no_old_model_or_rule_fallback(monkeypatch,first_decode):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();model=object();seen=[]
    def construct(*args,**kwargs):
        seen.append((args[4],kwargs['decode_mode'],kwargs['seed']))
        return SimpleNamespace(solution=None,actions=(),rejection_reason='NO_ACTION')
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    def forbidden(*a,**k):raise AssertionError('fallback or independent certification')
    monkeypatch.setattr(shared.pilot.oriented,'common_initial_beam',forbidden)
    monkeypatch.setattr(shared,'research_certify',forbidden)
    best,info=shared.neural_single_model_portfolio(reference.parents,model,CONFIG,seed=7,greedy_decode=first_decode)
    assert best is None and info['b32_calls']==0
    assert seen==[(model,first_decode,7),(model,'sample',7),(model,'sample',8)]


@pytest.mark.parametrize('decode',('joint_greedy','parent_joint'))
def test_joint_greedy_normalizes_full_parent_pool_and_each_conditional_pool(monkeypatch,decode):
    from mrta_search import gnn_constructor as c
    parents=tuple(ParentWeld(p,(x,9.),(x+1.,9.)) for p,x in (('a',1.),('b',6.),('c',12.)))
    geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    graph,_,_=c.joint_graph(geometry,CONFIG)
    by_index={graph.block_index[b.block_id]:b.parent_id for b in graph.blocks}
    model=c.ModeJointInsertionGNN(16,4)
    normalized=[];original=torch.log_softmax
    def logsoftmax(logits,*args,**kwargs):
        normalized.append(logits.clone());return original(logits,*args,**kwargs)
    monkeypatch.setattr(torch,'log_softmax',logsoftmax)
    monkeypatch.setattr(model,'parent_logits',lambda g,e,ids,f:torch.tensor([{'a':.4,'b':0.,'c':-1.}[p] for p in ids]))
    def insertion(emb,indices,features):
        parent=by_index[int(indices[0,0])]
        # A raw-logit sum would incorrectly choose a; its normalized action
        # probability is diffuse. Parent c is outside the finite top-two set.
        logits=torch.full((len(indices),),1000. if parent=='a' else 0.)
        if parent!='a':logits[0]=100.
        return logits
    monkeypatch.setattr(model,'joint_action_logits',insertion)
    result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,('a','b','c'),model,CONFIG,
        decode_mode=decode,parent_candidate_limit=2)
    assert result.solution is not None and result.actions[0].block_id.startswith('b::')
    assert normalized[0].tolist()==pytest.approx([.4,0.,-1.])
    assert len(normalized[1])>1 and len(normalized[2])>1
    assert torch.all(normalized[1]==1000.) and normalized[2].max()==100.


def test_parent_joint_scores_second_child_in_each_actual_branch_and_commits_once(monkeypatch):
    from mrta_search import gnn_constructor as c,lns
    from mrta_reference.geometry import build_legal_pattern_catalog
    parent=ParentWeld('split',(0.,8.),(4.,8.))
    pattern=next(p for p in build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)['split']
                 if p.kind is SplitKind.X_SPLIT and p.point_id=='BX_CENTER')
    geometry=c.joint_geometry((parent,),(pattern,),CONFIG)
    graph,blocks,patterns=c.joint_graph(geometry,CONFIG)
    empty=c.joint_start_state(geometry,((),)*4,((),)*4,('split',),blocks,CONFIG)
    model=c.ModeJointInsertionGNN(16,4);second_prefixes=[]
    def logits(embedding,indices,features):
        if int(indices[0,0])==graph.block_index['split::0']:
            return -.2*features[:,19]  # Prefer first-child direction 0 slightly.
        second_prefixes.append(features[:,:4].clone())
        return torch.zeros(len(indices))
    monkeypatch.setattr(model,'joint_action_logits',logits)
    with torch.inference_mode():
        emb=model.encode_static(graph)
        plans=c.joint_parent_action_candidates(graph,empty,('split',),torch.zeros(1),emb,model,blocks,patterns,CONFIG)
    assert len(plans)==3  # First branch has 2 legal seconds; second has 1.
    best=max(plans,key=lambda p:p['score'])
    assert best['child_ranks'][0]==2 and best['actions'][0].direction==1
    assert best['child_pool_sizes']==(2,1)
    assert best['score']==pytest.approx(best['parent_log_probability']+sum(best['child_log_probabilities']))
    assert all(torch.all(f[:,0]>0) and torch.all(f[:,1:]==0) for f in second_prefixes)
    for plan in plans:
        state=empty
        for action in plan['actions']:
            variants=lns.joint_block_candidates(state,blocks[action.block_id],patterns['split'],blocks,CONFIG)
            state=next(s for s,a in variants if a==action)
            assert action.direction in (0,1) and len([b for route in state.routes for b in route])==len(set(b for route in state.routes for b in route))
        assert state==plan['state'] and len(plan['actions'])==2
    assert empty.routes==((),)*4 and empty.directions==((),)*4 and empty.loads==(0.,)*4
    result=c.construct_or_repair_with_gnn(geometry,empty.routes,empty.directions,('split',),model,CONFIG,
        decode_mode='parent_joint')
    assert result.solution is not None and len(result.actions)==2
    assert result.actions[0].direction==1
    assert {b for route in result.solution.routes for b in route.block_ids}=={'split::0','split::1'}


@pytest.mark.parametrize('all_branches_fail',(False,True))
def test_parent_joint_discards_failed_child_branch_without_partial_commit(monkeypatch,all_branches_fail):
    from mrta_search import gnn_constructor as c,lns
    from mrta_reference.geometry import build_legal_pattern_catalog
    parent=ParentWeld('split',(0.,8.),(4.,8.))
    pattern=next(p for p in build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)['split']
                 if p.kind is SplitKind.X_SPLIT and p.point_id=='BX_CENTER')
    geometry=c.joint_geometry((parent,),(pattern,),CONFIG)
    original=lns.joint_block_candidates
    def candidates(state,child,*args,**kwargs):
        if child.block_id=='split::1' and (all_branches_fail or state.directions[0][0]==1):return ()
        return original(state,child,*args,**kwargs)
    monkeypatch.setattr(lns,'joint_block_candidates',candidates)
    result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,('split',),c.ModeJointInsertionGNN(16,4),CONFIG,
        decode_mode='parent_joint')
    if all_branches_fail:
        assert result.solution is None and not result.actions
        assert result.rejection_reason=='NO_LEGAL_PARENT_COMPLETION'
    else:
        assert result.solution is not None and len(result.actions)==2 and result.actions[0].direction==0


def test_early_prefix_targets_include_real_empty_routes_before_any_action():
    from mrta_search import gnn_constructor as c
    from run_gnn_cmax_finetune import op_prefix_targets
    reference=_joint_simple_reference()
    geometry=c.joint_geometry(reference.parents,reference.patterns,CONFIG)
    pending=tuple(p.parent_id for p in reference.parents);model=c.ModeJointInsertionGNN(16,4)
    seen=[]
    c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,pending,model,CONFIG,
        parent_observer=lambda s,p,a:seen.append((s.routes,p,a)))
    targets=op_prefix_targets(21,early=True)
    assert targets=={0.:0,.1:2,.25:5}
    routes,remaining,actions=seen[targets[0.]]
    assert routes==((),)*4 and set(remaining)==set(pending) and not actions


def test_early_representatives_use_load_for_robot_and_scarcity_for_other_parent(monkeypatch):
    import run_gnn_cmax_finetune as study
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates,has_joint_block_candidate
    parents=tuple(ParentWeld(p,(x,9.),(x+length,9.)) for p,x,length in
                  (('a',1.,1.),('b',6.,1.),('c',12.,4.)))
    geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
    _,blocks,patterns=c.joint_graph(geometry,CONFIG)
    pending=tuple(p.parent_id for p in parents)
    state=c.joint_start_state(geometry,((),)*4,((),)*4,pending,blocks,CONFIG)
    model=c.ModeJointInsertionGNN(16,4,opportunity_features=True)
    monkeypatch.setattr(model,'parent_logits',lambda g,e,ids,f:torch.tensor([3.,2.,1.]))
    monkeypatch.setattr(model,'joint_action_logits',lambda e,i,f:torch.arange(len(i),dtype=torch.float))
    actions=study.op_representative_actions(geometry,state,pending,model,CONFIG,early=True)
    assert len(actions)==3 and blocks[actions[0].block_id].parent_id=='a'
    assert actions[1].robot!=actions[0].robot and blocks[actions[2].block_id].parent_id=='c'
    child=blocks[actions[0].block_id]
    variants=joint_block_candidates(state,child,patterns['a'],blocks,CONFIG)
    alternatives=[s for s,a in variants if a.robot!=actions[0].robot]
    selected=next(s for s,a in variants if a==actions[1])
    assert max(selected.loads)==min(max(s.loads) for s in alternatives)
    for action in actions:
        child=blocks[action.block_id]
        assert has_joint_block_candidate(state,child,patterns[child.parent_id],blocks,CONFIG,robot_id=action.robot)
        assert action in [a for _,a in joint_block_candidates(state,child,patterns[child.parent_id],blocks,CONFIG)]


def test_counterfactual_evaluation_reuses_exact_complete_schedule_once(monkeypatch):
    from types import SimpleNamespace as NS
    import run_gnn_cmax_finetune as study
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    reference=_joint_simple_reference();model=c.ModeJointInsertionGNN(16,4)
    geometry=c.joint_geometry(reference.parents,reference.patterns,CONFIG)
    completion=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
        tuple(p.parent_id for p in reference.parents),model,CONFIG)
    assert completion.solution is not None
    monkeypatch.setattr(c,'construct_or_repair_with_gnn',lambda *a,**k:completion)
    monkeypatch.setattr(shared,'read_neural_checkpoint',lambda *a,**k:{study.OP_VARIANT:{}})
    monkeypatch.setattr(shared,'_model_from_checkpoint',lambda *a:model)
    calls=[]
    def schedule(solution,directions,config):
        calls.append((solution,directions))
        return NS(solution=solution,directions=directions,
            schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=100.,operations=(),diagnostics=()))
    monkeypatch.setattr(study.pilot.oriented,'schedule_explicit',schedule)
    cache={}
    info=study.op_measure_decodes(shared,CONFIG,reference.parents,study.OP_VARIANT,('greedy','sample','sample'),cache)
    assert len(calls)==1 and info['b32_calls']==1
    assert [a['b32_calls'] for a in info['attempts']]==[1,0,0]
    again=study.op_measure_decodes(shared,CONFIG,reference.parents,study.OP_VARIANT,('joint_greedy',),cache)
    assert len(calls)==1 and again['b32_calls']==0 and again['reused_b32_seconds']>0.
    raw=again['attempts'][0]
    reversed_raw=dict(raw,directions=tuple(tuple(1-d for d in route) for route in raw['directions']))
    assert study.op_complete_key(raw)!=study.op_complete_key(reversed_raw)


def test_joint_greedy_rejects_a_split_when_second_child_cannot_be_inserted(monkeypatch):
    from mrta_search import gnn_constructor as c,lns
    from mrta_reference.geometry import build_legal_pattern_catalog
    parent=ParentWeld('split',(0.,8.),(4.,8.))
    pattern=next(p for p in build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)['split']
                 if p.kind is SplitKind.X_SPLIT and p.point_id=='BX_CENTER')
    geometry=c.joint_geometry((parent,),(pattern,),CONFIG)
    original=lns.joint_block_candidates
    monkeypatch.setattr(lns,'joint_block_candidates',lambda s,b,*a,**k:[] if b.block_id.endswith('::1') else original(s,b,*a,**k))
    result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,('split',),c.ModeJointInsertionGNN(16,4),CONFIG,
        decode_mode='joint_greedy')
    assert result.solution is None and len(result.actions)==1
    assert result.rejection_reason=='NO_LEGAL_CHILD_INSERTION'


def test_final_geometry_pairs_exclude_failures_and_average_within_geometry():
    import run_gnn_multi_initial as experiment
    rows=[]
    for instance,scale in (('a',1),('b',100)):
        for seed in range(3):
            for method,cmax in (('A',90*scale),('B',100*scale)):
                rows.append(dict(instance=instance,tier='SMALL',seed=seed,method=method,
                                 status='FEASIBLE',cmax=cmax))
    pair=experiment._final_pair(rows,'A','B',3)
    assert pair['wins']==2 and pair['median_pct']==pytest.approx(10.)
    rows[-1].update(status='FAILED',cmax=None)
    pair=experiment._final_pair(rows,'A','B',3)
    assert pair['wins']==1 and pair['excluded']==['b']


def test_final_gnn_keep_requires_five_geometry_wins_and_two_tiers():
    import run_gnn_multi_initial as experiment
    rows=[]
    for i in range(9):
        for seed in range(3):
            for method,cmax in (('GNN-V2-Multi4',99.),('RULE/BEAM',100.)):
                rows.append(dict(instance=str(i),tier=('SMALL','MEDIUM','LARGE')[i//3],seed=seed,
                                 method=method,status='FEASIBLE',cmax=cmax))
    assert experiment._final_initial_decision(rows,'GNN-V2')['keep']
    for r in rows:
        if r['method']=='GNN-V2-Multi4' and int(r['instance'])>=3:r['cmax']=101.
    assert not experiment._final_initial_decision(rows,'GNN-V2')['keep']


def test_final_four_repair_arms_share_one_initial_and_its_elapsed_time(monkeypatch):
    import run_gnn_multi_initial as experiment
    initial=object();calls=[];seen=[]
    def build(*args,**kwargs):
        calls.append(1)
        return initial,[]
    def search(parents,method,seed,budget,iterations,**kwargs):
        seen.append((method,kwargs))
        return dict(status='FEASIBLE',cmax=10.,actual_runtime=0.,reference_calls=0)
    monkeypatch.setattr(experiment.shared.pilot.oriented,'common_initial_beam',build)
    monkeypatch.setattr(experiment,'_mode_search',search)
    entry=dict(instance='fixture',tier='SMALL',N=1,geometry_seed=1,
               parents=[dict(parent_id='p',start=(1.,9.),end=(2.,9.))])
    rows=experiment._final_experiment_worker((entry,7,experiment.FINAL_ARMS[:4],30.))
    assert len(calls)==1 and len(rows)==4
    assert all(k['common_initial'] is initial and k['config'].min_child_length==.5
               and k['certify'] is False for _,k in seen)
    assert len({k['initial_seconds'] for _,k in seen})==1


def test_final_csv_resume_preserves_failure_and_timeline_payload(tmp_path):
    import run_gnn_multi_initial as experiment
    path=tmp_path/'runs.csv'
    rows=[dict(record='run',instance='p',seed=1,method='MODE_NEW',status='EXECUTION_ERROR',cmax=None),
          dict(record='run',instance='p',seed=2,method='MODE_NEW',status='FEASIBLE',cmax=10.,
               details=dict(operations=[dict(kind='WAIT',t0=1.,t1=2.)]))]
    for row in rows:experiment._final_append(path,row)
    assert experiment._final_read(path)==rows


def test_final_choice_requires_useful_mode_component_for_gnn(monkeypatch):
    import run_gnn_multi_initial as experiment
    monkeypatch.setattr(experiment,'_final_stable',lambda rows,a,b:(a,b)!=('MODE_GNN','FIXED_GNN'))
    choice=experiment._final_choice([])
    assert choice['gnn_pairwise_positive'] and not choice['gnn_stable']
    assert choice['method']=='MODE_NEW' and choice['fixed']=='FIXED_NEW'


def test_region_formal_comparator_uses_05_and_b32_without_certification(monkeypatch):
    import run_gnn_multi_initial as experiment
    from mrta_ranker import ranking
    from mrta_search import pipeline
    from mrta_reference import certifier
    def forbidden(*args,**kwargs):
        raise AssertionError('experiment performed independent certification')
    monkeypatch.setattr(pipeline,'certify_schedule',forbidden)
    monkeypatch.setattr(certifier,'certify_schedule',forbidden)
    monkeypatch.setattr(ranking,'final_coordination_search_config',lambda method:
        pipeline.SearchConfig(max_iterations=1,hard_reference_deadline=False))
    parents=(ParentWeld('p',(1.,9.),(2.,9.)),ParentWeld('q',(14.,3.),(15.,3.)))
    out=experiment._final_region(parents,1,10.)
    assert out['status']=='FEASIBLE' and out['cmax']>0
    assert out['certifier_seconds']==0.
    assert out['final_state']['robot_completion']
    assert experiment.CONFIG.min_child_length==.2
    assert experiment.CONFIG_05.min_child_length==.5


@pytest.mark.parametrize('limit,expected', [(0,0),(1,1),(4,4),(None,8)])
def test_common_beam_optional_reference_limit(monkeypatch,limit,expected):
    from types import SimpleNamespace
    import run_joint_oriented_repair_pilot as oriented
    config=ScientificConfig()
    parent=ParentWeld('p',(1.,9.),(2.,9.))
    solution=canonicalize((parent,),(SplitPattern('p',SplitKind.WHOLE),),
                           {0:('p::whole',)},config,scope=FORMAL_SCOPE_V2)
    state=SimpleNamespace(routes=(('p::whole',),(),(),()),directions=((0,),(),(),()))
    monkeypatch.setattr(oriented,'initial_reference',lambda *args:solution)
    monkeypatch.setattr(oriented,'build_oriented_insertion_beam',lambda *args,**kwargs:((state,()),)*4)
    calls=[]
    def schedule(*args,**kwargs):
        calls.append(1)
        return oriented.DirectedSolution(solution,state.directions,ScheduleResult(ScheduleStatus.DEADLOCK))
    monkeypatch.setattr(oriented,'schedule_explicit',schedule)
    best,attempts=oriented.common_initial_beam((parent,),config,max_reference_calls=limit)
    assert best is None and len(calls)==expected


@pytest.mark.parametrize('late_second', (False,True))
def test_beam_round_robin_preserves_timely_incumbent(monkeypatch,late_second):
    from types import SimpleNamespace
    import run_joint_oriented_repair_pilot as oriented
    config=ScientificConfig()
    parent=ParentWeld('p',(1.,9.),(2.,9.))
    solution=canonicalize((parent,),(SplitPattern('p',SplitKind.WHOLE),),
                           {0:('p::whole',)},config,scope=FORMAL_SCOPE_V2)
    state=SimpleNamespace(routes=(('p::whole',),(),(),()),directions=((0,),(),(),()))
    clock=[0.]
    monkeypatch.setattr(oriented.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(oriented,'initial_reference',lambda *args:solution)
    monkeypatch.setattr(oriented,'build_oriented_insertion_beam',lambda *args,**kwargs:((state,()),)*4)
    def scheduled(sol,directions,cfg):
        first=clock[0]==0.
        clock[0]=1. if first else (4. if late_second else 2.)
        result=ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100. if first else 50.)
        return oriented.DirectedSolution(sol,directions,result)
    monkeypatch.setattr(oriented,'schedule_explicit',scheduled)
    current,attempts=oriented.common_initial_beam((parent,),config,deadline=3.,interleave=True,
                                                 accept_candidate=lambda candidate:clock[0]<=3.)
    assert [a['scheme'] for a in attempts]==['REGION_LOCAL','RULE_X_LOCAL']
    assert current.schedule.cmax==(100. if late_second else 50.)
    assert attempts[-1]['admitted'] is not late_second


def test_joint_policy_rejects_normalized_siblings_without_crashing_solver():
    from mrta_search.gnn_constructor import JointInsertionGNN,joint_geometry,construct_or_repair_with_gnn
    from mrta_reference.geometry import build_legal_pattern_catalog
    config=ScientificConfig()
    parents=(ParentWeld('p',(2.,6.),(6.,6.)),)
    pattern=next(p for p in build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)['p']
                 if p.kind is SplitKind.Y_SPLIT)
    geometry=joint_geometry(parents,(pattern,),config)
    result=construct_or_repair_with_gnn(geometry,(('p::0','p::1'),(),(),()),((0,0),(),(),()),
                                      (),JointInsertionGNN(64,4),config)
    assert result.solution is None and result.directions is None
    assert result.rejection_reason=='FIXED_PATTERN_CHANGED'

def test_destroyed_head_conflict_can_be_repaired_without_accepting_illegal_final_start(monkeypatch):
    from mrta_search.gnn_constructor import JointInsertionGNN,joint_geometry,construct_or_repair_with_gnn
    from mrta_reference.geometry import build_legal_pattern_catalog
    import run_joint_oriented_repair_pilot as oriented
    config=ScientificConfig()
    parents=(ParentWeld('bridge',(2.,9.),(2.,3.)),ParentWeld('keep',(8.,2.),(9.,2.)),
             ParentWeld('right',(5.,1.),(6.,1.)))
    catalog=build_legal_pattern_catalog(parents,config,FORMAL_SCOPE_V2)
    cut=next(p for p in catalog['bridge'] if p.point_id=='BY_CENTER')
    patterns=(cut,SplitPattern('keep',SplitKind.WHOLE),SplitPattern('right',SplitKind.WHOLE))
    source=canonicalize(parents,patterns,{0:('bridge::0',),2:('bridge::1','keep::whole'),3:('right::whole',)},
        config,scope=FORMAL_SCOPE_V2)
    original=oriented.schedule_explicit(source,((0,),(),(0,0),(0,)),config)
    assert certify_schedule(source,original.schedule,config,scope=FORMAL_SCOPE_V2).certified
    model=JointInsertionGNN(64,4)
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    # Prefer late insertion positions: merely permitting invalid partial starts
    # would leave the conflict unresolved, so restoration must take priority.
    monkeypatch.setattr(model,'joint_action_logits',lambda embeddings,indices,dynamic:
        torch.arange(dynamic.shape[0],dtype=embeddings.dtype,device=embeddings.device))
    args=(joint_geometry(parents,patterns,config),((),(),('keep::whole',),('right::whole',)),
          ((),(),(0,),(0,)),('bridge',),model,config)
    strict=construct_or_repair_with_gnn(*args)
    assert strict.solution is None and strict.rejection_reason=='NO_LEGAL_PARENT_ACTION'
    assert not strict.actions  # Upper child must precede the child repairing R2's head.
    permissive=construct_or_repair_with_gnn(*args,allow_partial_start_conflicts=True,prefer_legal_start_recovery=False)
    assert permissive.solution is None and permissive.rejection_reason=='UNRESOLVED_PARTIAL_START_CONFLICT'
    repaired=construct_or_repair_with_gnn(*args,allow_partial_start_conflicts=True)
    assert repaired.solution is not None,repaired.rejection_reason
    assert len(repaired.actions)==2 and repaired.actions[0].block_id=='bridge::0'
    scheduled=oriented.schedule_explicit(repaired.solution,repaired.directions,config)
    assert certify_schedule(repaired.solution,scheduled.schedule,config,scope=FORMAL_SCOPE_V2).certified
    # With no missing parent left, the same invalid R2/R3 deployment must fail.
    unresolved=construct_or_repair_with_gnn(joint_geometry(parents[1:],patterns[1:],config),
        args[1],args[2],(),model,config,allow_partial_start_conflicts=True)
    assert unresolved.solution is None and unresolved.rejection_reason=='UNRESOLVED_PARTIAL_START_CONFLICT'


def test_outer_deadline_does_not_restart_local_search_budget(monkeypatch):
    import run_gnn_multi_initial as study
    import run_joint_oriented_repair_pilot as oriented
    parent=ParentWeld('p',(1.,9.),(2.,9.))
    solution=canonicalize((parent,),(SplitPattern('p',SplitKind.WHOLE),),
        {0:('p::whole',)},study.CONFIG,scope=FORMAL_SCOPE_V2)
    current=oriented.schedule_explicit(solution,((0,),(),(),()),study.CONFIG)
    def forbidden(*args,**kwargs):raise AssertionError('search started after outer deadline')
    monkeypatch.setattr(study,'_evaluate_mode_bundle',forbidden)
    result=study._online_bundle_search(current,'NEW','SINGLE',1,30.,None,absolute_deadline=time.perf_counter()-1.)
    assert result['trace']==[] and result['reference_calls']==0 and result['budget']==0.
    assert result['cmax']==current.schedule.cmax and result['status']=='CERTIFIED'


def test_recovery_sampling_deduplicates_states_preserves_directions_and_first_context():
    import copy
    import run_gnn_multi_initial as study
    def trial(block,bit):
        return dict(status='DEADLOCK',changed_modes=[],incumbent_before=100.,late=False,
            patterns=[{'parent_id':'p'}],routes=[[block],[],[],[]],directions=[[bit],[],[],[]],
            frontier_exhausted=False,recovery_rollouts=32)
    first=[trial('a',0),trial('a',1),trial('b',0),trial('b',1)]
    repeated=copy.deepcopy(first*8)
    for row in repeated:row['incumbent_before']=90.
    def run(seed,trace):
        return dict(record='sparse_run',version=study.SPARSE_VERSION,instance='sample',repairer='GNN',
            policy='SINGLE',seed=seed,solution={'parents':[]},trace=trace)
    rows=[run(2,repeated),run(1,first)]
    cases,population=study._stratified_recovery_sample(rows)
    assert study._stratified_recovery_sample(list(reversed(rows)))==(cases,population)
    assert len(cases)==3 and population[0]['unique']==4 and population[0]['occurrences']==36
    # Directions distinguish complete candidates. Repeated proposals must not
    # gain extra chances or replace the preselected first comparison context.
    assert all(len(c['occurrences'])==9 and c['occurrences'][0]['seed']==1 for c in cases)
    assert all(c['occurrences'][0]['incumbent']==100. for c in cases)
    duplicated=copy.deepcopy(rows)
    duplicated[0]['trace']*=2
    extra,_=study._stratified_recovery_sample(duplicated)
    assert [(c['routes'],c['directions']) for c in extra]==[(c['routes'],c['directions']) for c in cases]


def test_fixed_route_bound_skips_only_nonimproving_candidates_and_keeps_real_evaluation(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_multi_initial as study
    import run_joint_oriented_repair_pilot as oriented
    import mrta_search.lns as lns
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(3.,9.),(4.,9.)))
    patterns=tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents)
    solution=canonicalize(parents,patterns,{0:('a::whole','b::whole')},study.CONFIG,scope=FORMAL_SCOPE_V2)
    directions=((0,0),(),(),())
    current=oriented.schedule_explicit(solution,directions,study.CONFIG)
    bound=study._fixed_route_time_bound(solution,directions)
    assert bound==pytest.approx(current.schedule.cmax)
    partial=lns._remove_parent_set(solution,('a',))
    monkeypatch.setattr(lns,'repair_partial_state_oriented',lambda *args,**kwargs:
        SimpleNamespace(candidate=SimpleNamespace(solution=solution),directions=directions,rejection_reason=None))
    calls=[];original=oriented.schedule_explicit
    def measured(*args,**kwargs):
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(oriented,'schedule_explicit',measured)
    skipped=study._evaluate_mode_bundle(current,partial,(),'NEW',1,None,improvement_ceiling=bound-1.)
    assert skipped['status']=='NONIMPROVING_ROUTE_BOUND' and skipped['reference_calls']==0
    assert not calls and skipped['cmax'] is None and not skipped['certified']
    # A promising candidate and the conservative numerical boundary still go
    # through the actual scheduler and independent certificate.
    for ceiling in (bound+1.,bound-0.5e-7,None):
        evaluated=study._evaluate_mode_bundle(current,partial,(),'NEW',1,None,improvement_ceiling=ceiling)
        assert evaluated['certified'] and evaluated['reference_calls']==1
        assert evaluated['cmax']==pytest.approx(bound)
    assert len(calls)==3


from mrta_baselines.common import CommonBaselineEvaluator
from mrta_reference.certifier import certify_schedule
from mrta_reference.geometry import robot_is_eligible
from mrta_reference.model import (
    OperationKind,
    ParentWeld,
    ScheduleResult,
    ScheduleStatus,
    ScientificConfig,
    SplitKind,
    SplitPattern,
)
from mrta_reference.scheduler import build_operation_templates, build_robot_routes
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_search.gnn_constructor import (
    BlockAssignmentGNN,
    BlockAssignmentRankingModel,
    build_block_robot_graph,
    construct_ranked_solution,
    construct_solution,
    dag_safe_ordered_routes,
    dynamic_features,
    fixed_blocks,
    fixed_split_patterns,
    rule_assignment,
    solution_from_assignment,
)
from mrta_search.vns import (
    GLOBAL_BALANCED,
    LEGACY_LOCAL_ABSOLUTE,
    n1_relocate_candidates,
    n2_swap_candidates,
    n3_route_candidates,
    evaluate_initial_with_shared_repair,
    run_vns,
)


CONFIG = ScientificConfig()


def _joint_simple_reference():
    parents=(ParentWeld('joint-a',(1,9),(2,9)),ParentWeld('joint-b',(14,9),(15,9)))
    return canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                        {0:('joint-a::whole',),1:('joint-b::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)


def test_coordination_context_observes_route_order_direction_and_partner():
    from mrta_search.gnn_constructor import CoordinationJointInsertionGNN,joint_graph
    torch.manual_seed(14)
    reference=_joint_simple_reference()
    graph,_,_=joint_graph(reference,CONFIG)
    model=CoordinationJointInsertionGNN()
    embeddings=model.encode_static(graph)
    forward=(('joint-a::whole','joint-b::whole'),(),(),())
    reverse=(('joint-b::whole','joint-a::whole'),(),(),())
    directions=((0,0),(),(),())
    original=model.contextualize_state(graph,embeddings,forward,directions)
    reordered=model.contextualize_state(graph,embeddings,reverse,directions)
    flipped=model.contextualize_state(graph,embeddings,forward,((1,0),(),(),()))
    # R1 must see changes in its R0 partner, even though its own route is empty.
    partner=graph.robot_offset+1
    assert not torch.allclose(original[partner],flipped[partner])
    assert not torch.allclose(original[partner],reordered[partner])
    original.square().mean().backward()
    for module in (model.route_direction,model.route_messages,model.route_norm):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in module.parameters())
    model.use_dynamic_context=False
    assert model.contextualize_state(graph,embeddings,forward,directions) is embeddings


def test_coordination_decoding_refreshes_only_the_actual_partial_routes(monkeypatch):
    from mrta_search.gnn_constructor import CoordinationJointInsertionGNN,construct_or_repair_with_gnn
    reference=_joint_simple_reference()
    model=CoordinationJointInsertionGNN()
    observed=[]
    original=model.contextualize_state
    def observe(graph,embeddings,routes,directions):
        observed.append(sum(map(len,routes)))
        return original(graph,embeddings,routes,directions)
    monkeypatch.setattr(model,'contextualize_state',observe)
    result=construct_or_repair_with_gnn(reference,((),(),(),()),((),(),(),()),
                                      tuple(p.parent_id for p in reference.parents),model,CONFIG)
    assert result.solution is not None
    assert observed==[0,1]


def test_joint_teacher_labels_replay_in_the_actual_student_pool():
    from mrta_search.lns import _remove_parent_set,repair_partial_state_oriented
    from mrta_search.gnn_constructor import prepare_joint_teacher_steps
    reference=_joint_simple_reference()
    partial=_remove_parent_set(reference,tuple(p.parent_id for p in reference.parents))
    teacher=repair_partial_state_oriented(reference,((0,),(0,),(),()),partial,CONFIG,
                                          coordination_proxy=True,direction_versions=2,seed=4)
    assert teacher.candidate is not None
    actions=[dict(block_id=a.block_id,robot=a.robot,position=a.position,direction=a.direction,
                  neighbour_changes=a.neighbour_changes) for a in teacher.actions]
    graph,steps,final=prepare_joint_teacher_steps(reference,partial.routes,((),(),(),()),
                                                 partial.removed_parent_ids,actions,CONFIG)
    assert final.routes==tuple(r.block_ids for r in teacher.candidate.solution.routes)
    assert final.directions==teacher.directions
    assert len(steps)==len(actions)==2
    assert all(s['indices'].shape[1]==4 and s['features'].shape[1]==41 for s in steps)
    broken=[dict(a) for a in actions]
    broken[0]['position']=999
    with pytest.raises(ValueError,match='missing/ambiguous'):
        prepare_joint_teacher_steps(reference,partial.routes,((),(),(),()),partial.removed_parent_ids,broken,CONFIG)


def test_joint_neural_logits_control_robot_choice_and_complete_real_b32(monkeypatch):
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn
    from mrta_search import lns
    import run_joint_oriented_repair_pilot as direct
    def forbidden(*args,**kwargs):
        raise AssertionError('GNN must not fall back to heuristic repair')
    monkeypatch.setattr(lns,'repair_partial_state_oriented',forbidden)
    parent=ParentWeld('one',(1,9),(2,9))
    reference=canonicalize((parent,),(SplitPattern('one',SplitKind.WHOLE),),{0:('one::whole',)},CONFIG)
    for sign,expected in ((1,0),(-1,1)):
        model=JointInsertionGNN()
        with torch.no_grad():
            for p in model.insertion_head.parameters():
                p.zero_()
            # A learned-head feature, not a decoder rule, controls left/right.
            model.insertion_head[0].weight[0,4*64+31]=1
            model.insertion_head[2].weight[0,0]=sign
        result=construct_or_repair_with_gnn(reference,((),(),(),()),((),(),(),()),('one',),model,CONFIG)
        assert result.solution is not None
        assert result.actions[0].robot==expected
        assert direct.schedule_explicit(result.solution,result.directions,CONFIG).schedule.status is ScheduleStatus.FEASIBLE


@pytest.mark.parametrize('parent_selection_mode,decode_mode',
    (('learned','greedy'),('learned','sample'),('mrv','greedy'),('mrv','sample'),('learned','joint_greedy')))
def test_joint_empty_routes_and_fixed_x_children_use_the_same_neural_decoder(decode_mode,parent_selection_mode):
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn
    from mrta_reference.geometry import build_legal_pattern_catalog
    import run_joint_oriented_repair_pilot as direct
    parent=ParentWeld('joint-x',(0,8),(4,8))
    pattern=next(p for p in build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)[parent.parent_id]
                 if p.kind is SplitKind.X_SPLIT and p.point_id=='BX_CENTER')
    reference=canonicalize((parent,),(pattern,),{0:('joint-x::0',),1:('joint-x::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    torch.manual_seed(20261027)
    result=construct_or_repair_with_gnn(reference,((),(),(),()),((),(),(),()),('joint-x',),JointInsertionGNN(),CONFIG,
                                      decode_mode=decode_mode,seed=20261027,parent_selection_mode=parent_selection_mode)
    assert result.solution is not None
    assert result.solution.patterns==(pattern,)
    assert result.solution.routes[0].block_ids==('joint-x::0',)
    assert result.solution.routes[1].block_ids==('joint-x::1',)
    assert result.directions[2:]==((),())
    assert len(result.actions)==2
    assert direct.schedule_explicit(result.solution,result.directions,CONFIG).schedule.status is ScheduleStatus.FEASIBLE


def test_joint_default_greedy_matches_explicit_greedy_without_consuming_rng():
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn,joint_geometry
    reference=_joint_simple_reference();model=JointInsertionGNN()
    geometry=joint_geometry(reference.parents,reference.patterns,CONFIG)
    args=(geometry,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in reference.parents),model,CONFIG)
    before=torch.get_rng_state().clone()
    first=construct_or_repair_with_gnn(*args)
    second=construct_or_repair_with_gnn(*args,decode_mode='greedy',seed=98765,parent_selection_mode='learned')
    assert first.solution is not None and second.solution is not None
    assert first.solution.routes==second.solution.routes and first.directions==second.directions
    assert first.actions==second.actions and torch.equal(before,torch.get_rng_state())


def test_mrv_uses_child_count_product_then_lpt_then_stable_id(monkeypatch):
    from types import SimpleNamespace as NS
    import mrta_search.gnn_constructor as joint
    state=NS(routes=((),('r',)*3,('r',)*7,('r',)*15))
    config=NS(process_time=lambda length:length)
    blocks={name:NS(block_id=name,parent_id=parent,length=length)
            for name,parent,length in (('a0','a',9),('a1','a',9),('b0','b',1),('c0','c',2),('d0','d',2))}
    patterns={p:NS(kind=SplitKind.WHOLE) for p in ('a','b','c','d')}
    legal={'a0':(1,),'a1':(1,),'b0':(0,2),'c0':(0,2),'d0':(0,2)}
    monkeypatch.setattr(joint,'robot_is_eligible',lambda block,robot,config:robot in legal[block.block_id])
    # a has 4*4 choices, versus b's 1+8; an additive child count would choose a.
    assert joint.mrv_parent(state,('a','b'),blocks,patterns,config)=='b'
    assert joint.mrv_parent(state,('b','c'),blocks,patterns,config)=='c'
    assert joint.mrv_parent(state,('d','c'),blocks,patterns,config)=='c'
    # X children are forced to upper robots 0/1: 1*4, not all eligible robots.
    blocks={'x::0':NS(block_id='x::0',parent_id='x',length=1),
            'x::1':NS(block_id='x::1',parent_id='x',length=1),'b0':blocks['b0']}
    monkeypatch.setattr(joint,'robot_is_eligible',lambda *args:True)
    patterns={'x':NS(kind=SplitKind.X_SPLIT,rail=joint.Rail.UPPER),'b':patterns['b']}
    assert joint.mrv_parent(state,('x','b'),blocks,patterns,config)=='x'


def test_mrv_skips_parent_head_and_preserves_the_common_insertion_pool(monkeypatch):
    import mrta_search.gnn_constructor as joint
    from mrta_search import lns
    reference=_joint_simple_reference();model=joint.JointInsertionGNN()
    monkeypatch.setattr(model,'parent_logits',lambda graph,embeddings,ids,features:torch.zeros(len(ids)))
    pools=[]
    real_candidates=lns.joint_block_candidates
    def candidates(state,block,*args,**kwargs):
        rows=real_candidates(state,block,*args,**kwargs)
        pools.append((state.routes,state.directions,block.block_id,kwargs,
                      tuple(lns.joint_action_key(a) for _,a in rows)))
        return rows
    monkeypatch.setattr(lns,'joint_block_candidates',candidates)
    args=(reference,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in reference.parents),model,CONFIG)
    learned=joint.construct_or_repair_with_gnn(*args)
    original_pools=pools[:];pools.clear()
    def forbidden(*args):
        raise AssertionError('MRV must not call parent_head')
    monkeypatch.setattr(model,'parent_logits',forbidden)
    before=torch.get_rng_state().clone()
    mrv=joint.construct_or_repair_with_gnn(*args,parent_selection_mode='mrv')
    assert original_pools==pools
    assert learned.solution is not None and mrv.solution is not None
    assert learned.actions==mrv.actions and learned.directions==mrv.directions
    assert learned.solution.routes==mrv.solution.routes and torch.equal(before,torch.get_rng_state())
    with pytest.raises(ValueError,match='parent_selection_mode'):
        joint.construct_or_repair_with_gnn(*args,parent_selection_mode='unknown')


def test_joint_sampling_uses_softmax_for_both_heads_and_a_local_seed(monkeypatch):
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn,joint_geometry
    parents=tuple(ParentWeld(str(i),(1+3*i,9),(1.5+3*i,9)) for i in range(6))
    geometry=joint_geometry(parents,fixed_split_patterns(parents,CONFIG),CONFIG)
    model=JointInsertionGNN();calls=[];original=torch.multinomial
    def logits(size):
        assert not model.training and torch.is_inference_mode_enabled()
        return torch.linspace(-1,1,size)
    monkeypatch.setattr(model,'parent_logits',lambda graph,embeddings,ids,features:logits(len(ids)))
    monkeypatch.setattr(model,'joint_action_logits',lambda embeddings,indices,features:logits(len(indices)))
    def draw(probabilities,num_samples,*,generator):
        assert generator is not None and num_samples==1
        assert torch.allclose(probabilities,torch.softmax(torch.linspace(-1,1,len(probabilities)),dim=0))
        calls.append(len(probabilities))
        return original(probabilities,num_samples,generator=generator)
    monkeypatch.setattr(torch,'multinomial',draw)
    before=torch.get_rng_state().clone()
    args=(geometry,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in parents),model,CONFIG)
    first=construct_or_repair_with_gnn(*args,decode_mode='sample',seed=20261028)
    second=construct_or_repair_with_gnn(*args,decode_mode='sample',seed=20261028)
    third=construct_or_repair_with_gnn(*args,decode_mode='sample',seed=20261029)
    assert first.solution is not None and second.solution is not None and third.solution is not None
    assert first.actions==second.actions and first.directions==second.directions
    assert first.actions!=third.actions
    assert len(calls)==3*2*len(parents)
    assert torch.equal(before,torch.get_rng_state())
    blocks=block_map(first.solution,CONFIG)
    import run_gnn_multi_initial as multi
    multi.check_complete(first)
    assert {b for route in first.solution.routes for b in route.block_ids}==set(blocks)
    for route,bits in zip(first.solution.routes,first.directions):
        assert len(route.block_ids)==len(bits) and all(d in (0,1) for d in bits)
        assert all(robot_is_eligible(blocks[b],route.robot_id,CONFIG) for b in route.block_ids)


def test_joint_training_rollout_matches_sample_and_has_both_policy_gradients():
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn,rollout_joint_repair
    torch.manual_seed(20261027)
    model=JointInsertionGNN();reference=_joint_simple_reference()
    args=(reference,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in reference.parents),model,CONFIG)
    inference=construct_or_repair_with_gnn(*args,decode_mode='sample',seed=20261027)
    before=torch.get_rng_state().clone()
    rollout=rollout_joint_repair(*args,seed=20261027)
    assert rollout.completion.solution is not None
    assert rollout.completion.actions==inference.actions and rollout.completion.directions==inference.directions
    assert len(rollout.parent_log_probs)==len(rollout.insertion_log_probs)==2
    assert torch.equal(before,torch.get_rng_state())
    assert all(v.requires_grad for v in rollout.parent_log_probs+rollout.insertion_log_probs)
    assert torch.equal(rollout.log_prob,torch.stack(rollout.parent_log_probs+rollout.insertion_log_probs).sum())
    (-.25*rollout.log_prob).backward()
    for head in ('layer1','layer2','parent_head','insertion_head'):
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in getattr(model,head).parameters())


def test_joint_training_rollout_rejects_numeric_and_data_errors(monkeypatch):
    from mrta_search.gnn_constructor import JointInsertionGNN,rollout_joint_repair
    reference=_joint_simple_reference();model=JointInsertionGNN()
    args=(reference,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in reference.parents),model,CONFIG)
    with pytest.raises(ValueError,match='remove each pending parent'):
        rollout_joint_repair(reference,((),(),(),()),((),(),(),()),('joint-a',),model,CONFIG)
    monkeypatch.setattr(model,'parent_logits',lambda graph,embeddings,ids,features:torch.full((len(ids),),float('nan')))
    with pytest.raises(ValueError,match='NONFINITE_PARENT_LOGITS'):
        rollout_joint_repair(*args)
    with torch.inference_mode(),pytest.raises(RuntimeError,match='inference_mode'):
        rollout_joint_repair(*args)


def test_joint_training_dead_end_retains_sampled_prefix(monkeypatch):
    from mrta_search import lns
    from mrta_search.gnn_constructor import JointInsertionGNN,rollout_joint_repair
    original=lns.joint_block_candidates;count=0
    def candidates(*args,**kwargs):
        nonlocal count
        if not kwargs.get('first_only'):
            count+=1
            if count==2:return ()
        return original(*args,**kwargs)
    monkeypatch.setattr(lns,'joint_block_candidates',candidates)
    model=JointInsertionGNN();reference=_joint_simple_reference()
    result=rollout_joint_repair(reference,((),(),(),()),((),(),(),()),('joint-a','joint-b'),model,CONFIG,seed=17)
    assert result.completion.solution is None and result.completion.rejection_reason=='NO_LEGAL_CHILD_INSERTION'
    assert len(result.completion.actions)==len(result.insertion_log_probs)==1
    assert len(result.parent_log_probs)==2 and result.log_prob.requires_grad


def test_cmax_return_and_pair_budget_use_real_status_without_fake_cost(monkeypatch):
    from types import SimpleNamespace as NS
    import run_gnn_cmax_finetune as rl
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn
    def feasible(cmax):return NS(schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=cmax))
    assert rl.terminal_return(feasible(80),100)==pytest.approx(20/180)
    assert rl.terminal_return(feasible(120),100)==pytest.approx(-20/220)
    assert rl.terminal_return(None,100)==-1
    assert rl.terminal_return(NS(schedule=NS(status=ScheduleStatus.DEADLOCK,cmax=None)),100)==-1
    with pytest.raises(ValueError,match='positive finite Cmax'):
        rl.terminal_return(feasible(float('nan')),100)
    reference=_joint_simple_reference()
    result=construct_or_repair_with_gnn(reference,((),(),(),()),((),(),(),()),('joint-a','joint-b'),JointInsertionGNN(),CONFIG)
    calls=[]
    monkeypatch.setattr(rl.pilot.oriented,'schedule_explicit',lambda *args:(calls.append(args) or feasible(80)))
    outcomes,budget=rl.score_pair(result,result,100)
    assert budget==len(calls)==1 and outcomes[0][0] is outcomes[1][0]
    assert outcomes[0][1]-outcomes[1][1]==0
    def broken(*args):raise ValueError('numeric/data failure')
    monkeypatch.setattr(rl.pilot.oriented,'schedule_explicit',broken)
    with pytest.raises(ValueError,match='numeric/data failure'):
        rl.score_pair(result,result,100)


def test_multi4_reuses_duplicate_schedules_and_keeps_best_feasible(monkeypatch):
    import run_gnn_multi_initial as multi
    from types import SimpleNamespace as NS
    from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn,joint_geometry
    reference=_joint_simple_reference();geometry=joint_geometry(reference.parents,reference.patterns,CONFIG)
    model=JointInsertionGNN()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
        model.insertion_head[0].weight[0,4*64+31]=1
        model.insertion_head[2].weight[0,0]=1
    args=(geometry,((),(),(),()),((),(),(),()),tuple(p.parent_id for p in reference.parents),model,CONFIG)
    a=construct_or_repair_with_gnn(*args)
    with torch.no_grad():model.insertion_head[2].weight[0,0]=-1
    b=construct_or_repair_with_gnn(*args)
    assert a.solution is not None and b.solution is not None and a.solution.routes!=b.solution.routes
    results=iter((a,a,b,b));modes=[];calls=[]
    def construct(*args,**kwargs):
        modes.append((kwargs['decode_mode'],kwargs['seed']));return next(results)
    def schedule(solution,directions,config):
        calls.append(solution.routes)
        return NS(solution=solution,directions=directions,
                  schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=100. if len(calls)==1 else 90.))
    monkeypatch.setattr(multi,'construct_or_repair_with_gnn',construct)
    monkeypatch.setattr(multi.shared.pilot.oriented,'schedule_explicit',schedule)
    greedy,combined,best=multi.neural_multi4(reference.parents,model,'GNN','TRAIN','unit',20261027)
    assert modes==[('greedy',20261027),('sample',20261028),('sample',20261029),('sample',20261030)]
    assert len(calls)==combined['b32_calls']==combined['unique_routes']==2
    assert greedy['b32_calls']==1 and greedy['best_cmax']==100.
    assert combined['best_cmax']==best.schedule.cmax==90.


def test_shared_raw_geometry_initial_has_no_routes_or_heuristic_input(monkeypatch):
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import JointInsertionGNN,joint_geometry
    from mrta_search import lns
    def forbidden(*args,**kwargs):
        raise AssertionError('pure neural construction must not call Beam/Regret')
    monkeypatch.setattr(shared.pilot.oriented,'common_initial_beam',forbidden)
    monkeypatch.setattr(lns,'repair_partial_state_oriented',forbidden)
    reference=_joint_simple_reference()
    geometry=joint_geometry(reference.parents,reference.patterns,CONFIG)
    assert not hasattr(geometry,'routes')
    model=JointInsertionGNN()
    with torch.no_grad():
        for parameter in model.parameters():parameter.zero_()
    scheduled,info=shared.neural_initial(reference.parents,model)
    assert info['constructed']==info['b32_calls']==1
    assert info['actions']==len(reference.parents)
    assert scheduled is not None and scheduled.schedule.status is ScheduleStatus.FEASIBLE


def test_shared_replay_is_parent_atomic_and_every_prefix_can_continue():
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import joint_geometry,prepare_joint_teacher_steps
    parents=(ParentWeld('cross',(4,4),(4,8)),ParentWeld('u',(1,9),(2,9)),ParentWeld('l',(1,2),(2,2)))
    patterns=fixed_split_patterns(parents,CONFIG)
    target=canonicalize(parents,patterns,{0:('u::whole','cross::1'),2:('l::whole','cross::0')},CONFIG,scope=FORMAL_SCOPE_V2)
    directions=((0,0),(),(0,0),())
    result,info=shared.construction_replay(target,directions)
    assert info['replay_status']=='REPLAYED'
    state,actions,snapshots=result
    cross=[i for i,a in enumerate(actions) if a.block_id.startswith('cross::')]
    assert cross[1]==cross[0]+1
    geometry=joint_geometry(parents,patterns,CONFIG)
    from dataclasses import asdict
    for prefix,pending,offset in snapshots:
        _,_,final=prepare_joint_teacher_steps(geometry,prefix.routes,prefix.directions,pending,
                                             [asdict(a) for a in actions[offset:]],CONFIG)
        assert final.routes==state.routes and final.directions==directions


def test_shared_replay_failure_is_bounded_and_does_not_inject_an_action():
    import run_gnn_shared_construction as shared
    result,info=shared.construction_replay(_joint_simple_reference(),((0,),(0,),(),()),max_nodes=0)
    assert result is None and info['replay_status']=='BOUNDED_REPLAY_EXHAUSTED'


def test_joint_data_preparation_rejects_dev_supervision():
    import run_gnn_joint_repair_pilot as joint
    with pytest.raises(ValueError,match='DEV supervision'):
        joint.prepare_record(dict(entry=dict(split='MLP_DEV')))


def test_joint_record_identity_keeps_different_partial_learning_states():
    from copy import deepcopy
    import run_gnn_joint_repair_pilot as joint
    record=dict(parents=[dict(parent_id='a',start=(1,9),end=(2,9))],
                patterns=[dict(parent_id='a',kind='WHOLE',t=None,point_id=None,rail=None)],
                partial_routes=[[],[],[],[]],partial_directions=[[],[],[],[]],pending_parents=['a'],
                final_routes=[['a::whole'],[],[],[]],final_directions=[[0],[],[],[]])
    same=deepcopy(record)
    other=deepcopy(record);other['partial_routes'][1]=['retained::whole'];other['partial_directions'][1]=[1]
    assert joint.record_key(record)==joint.record_key(same)
    assert joint.record_key(record)!=joint.record_key(other)
    other=deepcopy(record);other['parents'][0]['end']=(3,9)
    assert joint.record_key(record)!=joint.record_key(other)


def test_joint_train_scaling_roundtrips_and_v1_still_loads_strictly():
    from mrta_search.gnn_constructor import JointInsertionGNN
    torch.manual_seed(7)
    legacy=JointInsertionGNN()
    restored=JointInsertionGNN();restored.load_state_dict(legacy.state_dict(),strict=True)
    embeddings=torch.randn(8,64);indices=torch.tensor([[0,4,-1,2],[0,5,1,-1]])
    features=torch.randn(2,41)*0.001
    assert torch.equal(legacy.joint_action_logits(embeddings,indices,features),
                       restored.joint_action_logits(embeddings,indices,features))
    normalized=JointInsertionGNN(normalize_inputs=True)
    normalized.fit_input_scaling([dict(features=features,parent_features=torch.randn(3,12))])
    assert torch.isfinite(normalized.scale_inputs(features,'joint')).all()
    normalized_copy=JointInsertionGNN(normalize_inputs=True)
    normalized_copy.load_state_dict(normalized.state_dict())
    assert torch.equal(normalized.joint_action_logits(embeddings,indices,features),
                       normalized_copy.joint_action_logits(embeddings,indices,features))


def test_joint_feedback_selects_from_updated_incumbent_and_bounds_b32(monkeypatch):
    from types import SimpleNamespace as NS
    import run_gnn_joint_repair_pilot as joint
    selections=[];scheduled=[]
    def select(solution,directions,schedule,selection,**kwargs):
        selections.append((solution,schedule.cmax))
        return NS(routes=(),removed_parent_ids=('a',))
    def repair(*args,**kwargs):
        return NS(solution='candidate-'+str(len(selections)),directions=(),actions=())
    def schedule(solution,directions,config):
        scheduled.append(solution)
        cmax={1:90.,5:80.}.get(len(scheduled),95.)
        return NS(solution=solution,directions=(),schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=cmax,operations=()))
    monkeypatch.setattr(joint,'select_bottleneck_parents',select)
    monkeypatch.setattr(joint,'construct_or_repair_with_gnn',repair)
    monkeypatch.setattr(joint,'partial_directions',lambda *args:())
    monkeypatch.setattr(joint.oriented,'schedule_explicit',schedule)
    monkeypatch.setattr(joint.oriented,'summary',lambda *args:{})
    initial=NS(solution='initial',directions=(),schedule=NS(cmax=100.,operations=()))
    row=joint.iterative_group(dict(sheet_name='g21_w035'),initial,'GNN-V2',None,NS(seed=20261027))
    assert row['accepted']==2 and row['best_cmax']==80.
    assert row['b32_calls']==len(scheduled)==24
    assert selections[0]==('initial',100.)
    assert selections[1:5]==[('candidate-1',90.)]*4
    assert selections[5:]==[('candidate-5',80.)]*19


def test_joint_wait_feedback_without_wait_retains_load_search_and_records_fallback(monkeypatch):
    import json
    from types import SimpleNamespace as NS
    import run_gnn_joint_repair_pilot as joint
    selections=[]
    def select(solution,directions,schedule,selection,**kwargs):
        selections.append(selection)
        if selection is joint.BottleneckSelection.WAIT_ANCHORED:
            return None
        return NS(routes=(),removed_parent_ids=('a',))
    monkeypatch.setattr(joint,'select_bottleneck_parents',select)
    monkeypatch.setattr(joint,'construct_or_repair_with_gnn',lambda *args,**kwargs:
                        NS(solution='same',directions=(),actions=()))
    monkeypatch.setattr(joint,'partial_directions',lambda *args:())
    initial=NS(solution='same',directions=(),schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=100.,operations=()))
    monkeypatch.setattr(joint.oriented,'schedule_explicit',lambda *args:initial)
    row=joint.iterative_group(dict(sheet_name='g21_w035'),initial,'GNN-V2',None,NS(seed=20261027),
                              selectors=(joint.BottleneckSelection.WAIT_ANCHORED,),iterations=8)
    assert selections==[joint.BottleneckSelection.WAIT_ANCHORED,joint.BottleneckSelection.LOAD]*8
    assert row['attempts']==row['b32_calls']==8
    assert row['accepted']==0 and row['best_cmax']==100.
    trace=json.loads(row['trace'])
    assert [item['q'] for item in trace]==[4,4,4,8,8,8,4,4]
    assert all(item['selection']=='LOAD' and item['requested_selection']=='WAIT_ANCHORED'
               and item['selection_fallback']=='NO_POSITIVE_WAIT' for item in trace)


def test_adaptive_wait_without_positive_wait_matches_base_budget_and_selections(monkeypatch):
    import json
    from types import SimpleNamespace as NS
    import run_gnn_joint_repair_pilot as joint
    ops=tuple(NS(kind=joint.OperationKind.WAIT,duration=d) for d in (0.,joint.CONFIG.numeric_epsilon/2))
    initial=NS(solution='same',directions=(),schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=100.,operations=ops))
    selections=[];calls=[]
    def select(solution,directions,schedule,selection,**kwargs):
        selections.append((selection,kwargs['q'],kwargs['seed']))
        return NS(routes=(),removed_parent_ids=('a',))
    def schedule(*args):
        calls.append(args)
        return initial
    monkeypatch.setattr(joint,'select_bottleneck_parents',select)
    monkeypatch.setattr(joint,'construct_or_repair_with_gnn',lambda *args,**kwargs:
                        NS(solution='same',directions=(),actions=()))
    monkeypatch.setattr(joint,'partial_directions',lambda *args:())
    monkeypatch.setattr(joint.oriented,'schedule_explicit',schedule)
    rows=[joint.iterative_group(dict(sheet_name='g21_w035'),initial,'GNN-V2',None,NS(seed=20261027),
                                wait_feedback=feedback) for feedback in (False,True)]
    assert selections[:24]==selections[24:]
    assert [q for _,q,_ in selections[:24]]==[4,4,4,8,8,8]*4
    assert [seed for _,_,seed in selections[:24]]==list(range(20261027,20261027+24))
    assert len(calls)==48
    assert all(r['attempts']==r['b32_calls']==24 and r['accepted']==0 for r in rows)
    assert rows[1]['no_wait_rounds']==24 and rows[1]['wait_used']==rows[1]['wait_fallbacks']==0
    for base,wait in zip(*(json.loads(r['trace']) for r in rows)):
        assert {k:v for k,v in base.items() if k!='no_wait'}=={k:v for k,v in wait.items() if k!='no_wait'}


@pytest.mark.parametrize('empty_partial',(False,True))
def test_adaptive_wait_reads_only_accepted_schedule_and_recovers_original_selector(monkeypatch,empty_partial):
    import json
    from types import SimpleNamespace as NS
    import run_gnn_joint_repair_pilot as joint
    def state(cmax,wait):
        return NS(solution=NS(routes=()),directions=(),schedule=NS(status=ScheduleStatus.FEASIBLE,cmax=cmax,
                   operations=(NS(kind=joint.OperationKind.WAIT,duration=wait),)))
    initial=state(100.,0.)
    candidates=[state(90.,5.),state(90.,0.),state(80.,0.)]
    calls=[];selections=[]
    def select(solution,directions,schedule,selection,**kwargs):
        selections.append((selection,schedule.cmax,joint.positive_wait_duration(schedule)))
        if len(calls)==1 and selection is joint.BottleneckSelection.WAIT_ANCHORED:
            return NS(routes=(),removed_parent_ids=()) if empty_partial else None
        return NS(routes=(),removed_parent_ids=('a',))
    def schedule(*args):
        calls.append(args)
        return candidates[min(len(calls)-1,2)]
    monkeypatch.setattr(joint,'select_bottleneck_parents',select)
    monkeypatch.setattr(joint,'construct_or_repair_with_gnn',lambda *args,**kwargs:
                        NS(solution=NS(routes=()),directions=(),actions=()))
    monkeypatch.setattr(joint,'partial_directions',lambda *args:())
    monkeypatch.setattr(joint.oriented,'schedule_explicit',schedule)
    monkeypatch.setattr(joint.oriented,'summary',lambda *args:{})
    row=joint.iterative_group(dict(sheet_name='g21_w035'),initial,'GNN-V2-WAIT',None,NS(seed=20261027),wait_feedback=True)
    trace=json.loads(row['trace'])
    assert len(calls)==row['b32_calls']==row['attempts']==24
    assert row['accepted']==2 and row['best_cmax']==80.
    assert row['wait_used']==row['wait_accepted']==row['wait_fallbacks']==1
    assert row['no_wait_rounds']==22
    assert trace[1]['requested_selection']==trace[1]['selection']=='TRAVEL' and trace[1]['wait_fallback']
    assert trace[2]['selection']=='WAIT_ANCHORED' and trace[2]['wait_before']==5. and trace[2]['accepted']
    assert trace[3]['selection']=='RANDOM' and trace[3]['no_wait']
    assert all(t['after']==t['before'] for t in trace if not t['accepted'])
    assert all(t['after']<t['before'] for t in trace if t['accepted'])


def test_joint_teacher_neighbour_flips_are_executable_student_actions():
    from dataclasses import asdict
    from mrta_search.lns import oriented_route_state,joint_block_candidates
    from mrta_search.gnn_constructor import prepare_joint_teacher_steps
    parents=tuple(ParentWeld(p,(x,9),(x+1,9)) for p,x in (('a',1),('b',4),('c',7)))
    reference=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                            {0:('a::whole','b::whole','c::whole')},CONFIG)
    blocks=block_map(reference,CONFIG)
    routes=(('a::whole','c::whole'),(),(),());directions=((1,1),(),(),())
    state=oriented_route_state(routes,directions,blocks,CONFIG)
    variants=joint_block_candidates(state,blocks['b::whole'],reference.patterns[1],blocks,CONFIG)
    after,action=next((s,a) for s,a in variants if a.robot==0 and a.position==1 and a.neighbour_changes)
    _,steps,replayed=prepare_joint_teacher_steps(reference,routes,directions,('b',),(asdict(action),),CONFIG)
    assert replayed.routes==after.routes and replayed.directions==after.directions
    assert steps[0]['features'][steps[0]['target'],24]>0


def _parents():
    return (
        ParentWeld("mandatory", (4.0, 4.0), (4.0, 8.0)),
        ParentWeld("shared-a", (5.0, 6.0), (7.0, 6.0)),
        ParentWeld("shared-b", (12.0, 6.0), (14.0, 6.0)),
        ParentWeld("upper", (2.0, 9.0), (3.0, 9.0)),
        ParentWeld("lower", (16.0, 3.0), (17.0, 3.0)),
    )


def _solution():
    return construct_solution(_parents(), CONFIG).solution


def _imbalanced_solution():
    parents = (
        ParentWeld("a", (1.0, 6.0), (4.5, 6.0)),
        ParentWeld("b", (5.0, 6.0), (8.5, 6.0)),
        ParentWeld("c", (9.0, 6.0), (12.5, 6.0)),
        ParentWeld("d", (13.0, 6.0), (16.5, 6.0)),
        ParentWeld("e", (2.0, 6.0), (2.5, 6.0)),
        ParentWeld("f", (3.0, 6.0), (3.5, 6.0)),
        ParentWeld("g", (10.0, 6.0), (10.5, 6.0)),
        ParentWeld("h", (15.0, 6.0), (15.5, 6.0)),
    )
    patterns = tuple(
        SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents
    )
    solution = canonicalize(
        parents,
        patterns,
        {
            0: ("a::whole", "c::whole", "b::whole", "d::whole"),
            1: ("e::whole", "f::whole"),
            2: ("g::whole",),
            3: ("h::whole",),
        },
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    return parents, solution


def _assert_assignment_legal(solution):
    blocks = block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2)
    actual = []
    for route in solution.routes:
        for block_id in route.block_ids:
            actual.append(block_id)
            assert robot_is_eligible(blocks[block_id], route.robot_id, CONFIG)
    assert len(actual) == len(set(actual)) == len(blocks)
    assert set(actual) == set(blocks)


def test_fixed_blocks_only_allow_eligible_robots_and_shared_blocks_allow_both_rails():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    upper = next(block for block in blocks if block.parent_id == "upper")
    shared = next(block for block in blocks if block.parent_id == "shared-a")
    assert [robot_is_eligible(upper, robot, CONFIG) for robot in range(4)] == [True, True, False, False]
    assert all(robot_is_eligible(shared, robot, CONFIG) for robot in range(4))


def test_mandatory_y_split_children_are_distinct_complete_and_not_whole():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    mandatory = next(pattern for pattern in patterns if pattern.parent_id == "mandatory")
    assert mandatory.kind is SplitKind.Y_SPLIT and mandatory.mandatory
    solution = _solution()
    assigned = [item for route in solution.routes for item in route.block_ids]
    assert assigned.count("mandatory::0") == 1
    assert assigned.count("mandatory::1") == 1
    assert "mandatory::whole" not in assigned


def test_gnn_constructor_covers_every_block_once_with_only_legal_actions():
    model = BlockAssignmentGNN()
    result = construct_solution(_parents(), CONFIG, model=model)
    assert result.legal_action_fraction == 1.0
    _assert_assignment_legal(result.solution)


def test_ranking_model_direct_constructor_is_deterministic_complete_and_legal():
    import torch

    torch.manual_seed(71)
    model = BlockAssignmentRankingModel()
    first = construct_ranked_solution(
        _parents(), CONFIG, model, assignment_kind="MODEL"
    )
    second = construct_ranked_solution(
        _parents(), CONFIG, model, assignment_kind="MODEL"
    )
    assert first.solution.canonical_json == second.solution.canonical_json
    _assert_assignment_legal(first.solution)


def test_weld_edge_ablation_removes_only_weld_relations():
    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    full = build_block_robot_graph(blocks, patterns, CONFIG)
    ablated = build_block_robot_graph(
        blocks, patterns, CONFIG, include_weld_edges=False
    )
    assert any(int(kind) in (1, 2) for kind in full.edge_type)
    assert not any(int(kind) in (1, 2) for kind in ablated.edge_type)
    assert any(int(kind) == 3 for kind in ablated.edge_type)
    assert ablated.node_features.equal(full.node_features)


def test_joint_assignment_and_pairwise_ranking_heads_both_receive_gradients():
    import torch

    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    graph = build_block_robot_graph(blocks, patterns, CONFIG)
    by_id = {block.block_id: block for block in blocks}
    assignment = rule_assignment(blocks, CONFIG)
    routes = [[] for _ in range(4)]
    model = BlockAssignmentRankingModel()
    embeddings = model.encode_static(graph)
    first = blocks[0]
    rows = dynamic_features(first, routes, by_id, CONFIG, 0, len(blocks))
    mask = torch.tensor(
        [robot_is_eligible(first, robot, CONFIG) for robot in range(4)],
        dtype=torch.bool,
    )
    target = dict(assignment)[first.block_id]
    assignment_loss = torch.nn.functional.cross_entropy(
        model.action_logits(graph, embeddings, first.block_id, rows, mask).unsqueeze(0),
        torch.tensor([target]),
    )
    same_robot = [block_id for block_id, robot in assignment if robot == target]
    if len(same_robot) < 2:
        same_robot = [block.block_id for block in blocks[:2]]
        target = next(
            robot
            for robot in range(4)
            if all(robot_is_eligible(by_id[item], robot, CONFIG) for item in same_robot)
        )
    score0 = model.ordering_score(
        graph,
        embeddings,
        same_robot[0],
        target,
        dynamic_features(by_id[same_robot[0]], routes, by_id, CONFIG, 0, len(blocks)),
    )
    score1 = model.ordering_score(
        graph,
        embeddings,
        same_robot[1],
        target,
        dynamic_features(by_id[same_robot[1]], routes, by_id, CONFIG, 0, len(blocks)),
    )
    (assignment_loss + torch.nn.functional.softplus(-(score0 - score1))).backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.assignment_head.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.ranking_head.parameters()
    )


def test_dag_safe_constructor_rejects_precedence_cycle():
    import torch

    patterns = fixed_split_patterns(_parents(), CONFIG)
    blocks = fixed_blocks(_parents(), patterns, CONFIG)
    graph = build_block_robot_graph(blocks, patterns, CONFIG)
    model = BlockAssignmentRankingModel(encoder_kind="mlp")
    with torch.inference_mode():
        embeddings = model.encode_static(graph)
        assignment = rule_assignment(blocks, CONFIG)
        first, second = blocks[0].block_id, blocks[1].block_id
        try:
            dag_safe_ordered_routes(
                model,
                graph,
                embeddings,
                assignment,
                CONFIG,
                predecessors={first: (second,), second: (first,)},
            )
        except ValueError as error:
            assert "cycle" in str(error)
        else:
            raise AssertionError("cyclic precedence was accepted")


def test_solution_constructor_rejects_an_illegal_robot_before_canonicalization():
    parents = _parents()
    patterns = fixed_split_patterns(parents, CONFIG)
    blocks = fixed_blocks(parents, patterns, CONFIG)
    assignment = []
    for block in blocks:
        robot = 2 if block.parent_id == "upper" else next(
            robot for robot in range(4) if robot_is_eligible(block, robot, CONFIG)
        )
        assignment.append((block.block_id, robot))
    try:
        solution_from_assignment(parents, patterns, blocks, assignment, CONFIG)
    except ValueError as error:
        assert "illegal assignment" in str(error)
    else:
        raise AssertionError("illegal upper-to-lower assignment was accepted")


def test_vns_relocate_and_swap_keep_assignment_legal():
    solution = _solution()
    relocate = n1_relocate_candidates(solution, CONFIG, random.Random(7), cap=10_000)
    swap = n2_swap_candidates(solution, CONFIG, random.Random(7), cap=10_000)
    assert relocate
    assert swap
    for candidate in (*relocate, *swap):
        _assert_assignment_legal(candidate)


def test_route_neighborhood_preserves_each_robot_task_set():
    solution = _solution()
    candidates = n3_route_candidates(solution, CONFIG, random.Random(9), cap=20)
    assert candidates
    expected = [set(route.block_ids) for route in solution.routes]
    signatures = set()
    for candidate in candidates:
        assert [set(route.block_ids) for route in candidate.routes] == expected
        signatures.add(candidate.canonical_json)
    assert len(signatures) == len(candidates)


def test_global_n1_and_n2_selection_reaches_the_critical_robot():
    _, solution = _imbalanced_solution()
    for builder in (n1_relocate_candidates, n2_swap_candidates):
        candidate = builder(
            solution,
            CONFIG,
            random.Random(4),
            cap=1,
            selection_policy=GLOBAL_BALANCED,
        )[0]
        changed = [
            robot
            for robot in range(4)
            if candidate.routes[robot].block_ids != solution.routes[robot].block_ids
        ]
        assert 0 in changed
        _assert_assignment_legal(candidate)


def test_n3_global_selection_uses_critical_route_and_improvement_not_absolute_cost():
    _, solution = _imbalanced_solution()
    global_candidate = n3_route_candidates(
        solution,
        CONFIG,
        random.Random(5),
        cap=1,
        selection_policy=GLOBAL_BALANCED,
    )[0]
    legacy_candidate = n3_route_candidates(
        solution,
        CONFIG,
        random.Random(5),
        cap=1,
        selection_policy=LEGACY_LOCAL_ABSOLUTE,
    )[0]
    global_changed = [
        robot
        for robot in range(4)
        if global_candidate.routes[robot].block_ids != solution.routes[robot].block_ids
    ]
    legacy_changed = [
        robot
        for robot in range(4)
        if legacy_candidate.routes[robot].block_ids != solution.routes[robot].block_ids
    ]
    assert global_changed == [0]
    assert legacy_changed == [1]


def test_same_rail_passing_cannot_pass_independent_certification():
    parents = (
        ParentWeld("a", (1.0, 8.0), (9.0, 8.0)),
        ParentWeld("b", (10.0, 8.0), (2.0, 8.0)),
    )
    solution = canonicalize(
        parents,
        tuple(SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents),
        {0: ("a::whole",), 1: ("b::whole",)},
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    robot_routes = build_robot_routes(solution, CONFIG, orientations={0: (0,), 1: (0,)})
    operations = tuple(
        operation
        for route in robot_routes
        for operation in build_operation_templates(route, CONFIG)
    )
    completion = tuple(
        max((op.end_time for op in operations if op.robot_id == robot), default=0.0)
        for robot in range(4)
    )
    fake = ScheduleResult(
        ScheduleStatus.FEASIBLE,
        operations,
        max(completion),
        completion,
        directions=((0,), (0,), (), ()),
    )
    report = certify_schedule(solution, fake, CONFIG, scope=FORMAL_SCOPE_V2)
    assert not report.certified
    assert any("same-rail non-passing" in error for error in report.errors)


def test_same_seed_gnn_inference_and_vns_are_reproducible():
    torch_seed = 123
    import torch

    torch.manual_seed(torch_seed)
    model = BlockAssignmentGNN()
    first = construct_solution(_parents(), CONFIG, model=model)
    second = construct_solution(_parents(), CONFIG, model=model)
    assert first.solution.canonical_json == second.solution.canonical_json

    results = []
    for _ in range(2):
        evaluator = CommonBaselineEvaluator(
            _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
        )
        result = run_vns(
            first.solution,
            evaluator,
            CONFIG,
            seed=77,
            per_neighborhood_budget=2,
        )
        results.append(result)
    assert results[0].best is not None and results[1].best is not None
    assert results[0].initial is not None and results[1].initial is not None
    assert results[0].best.solution.canonical_json == results[1].best.solution.canonical_json
    assert results[0].stats.improvements == results[1].stats.improvements
    assert results[0].best.metrics.cmax <= results[0].initial.metrics.cmax
    assert results[0].best.certification is not None
    assert results[0].best.certification.certified


def test_late_initial_certification_is_not_accepted_at_hard_deadline():
    evaluator = CommonBaselineEvaluator(
        _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
    )
    completed = evaluator.evaluate(_solution(), source="ON_TIME_REFERENCE")
    assert completed.status.value == "COMPLETED"

    class LateEvaluator:
        expired = True

        def evaluate(self, solution, *, source):
            return replace(completed, solution=solution, source=source, completed_at=1.01)

    initial, attempts, failures = evaluate_initial_with_shared_repair(
        _solution(), LateEvaluator(), CONFIG, accept_completed_by=1.0
    )
    assert initial is None
    assert attempts == 1
    assert failures == ("LATE_CERTIFIED",)


def test_end_to_end_budget_charges_model_load_construction_and_uses_remaining_time(monkeypatch):
    entry = {
        "instance_id": "unit:tiny",
        "N": 5,
        "split": "MLP_DEV",
    }
    original_construct = pilot_script.construct_solution
    captured = {}

    def delayed_load_model():
        time.sleep(0.02)
        return BlockAssignmentGNN(), {}

    def delayed_construct(parents, config, *, model=None):
        time.sleep(0.02)
        return original_construct(parents, config, model=model)

    def fake_run_vns(initial_solution, evaluator, config, **kwargs):
        captured["time_limit"] = evaluator.time_limit
        captured["accept_completed_by"] = kwargs["accept_completed_by"]
        return pilot_script.VNSResult(
            None, None, pilot_script.VNSStats(), 0, (), evaluator.elapsed
        )

    monkeypatch.setattr(pilot_script.ranker_dataset, "load_parents", lambda _: _parents())
    monkeypatch.setattr(pilot_script, "load_model", delayed_load_model)
    monkeypatch.setattr(pilot_script, "construct_solution", delayed_construct)
    monkeypatch.setattr(pilot_script, "run_vns", fake_run_vns)
    row = pilot_script.run_end_to_end(entry, "GNN", 7, CONFIG, 0.2)
    assert row["timing_scope"] == "END_TO_END_EXCLUDING_INPUT_READ"
    assert row["model_load_seconds"] >= 0.015
    assert row["construction_seconds"] >= 0.015
    assert captured["time_limit"] < 0.17
    assert captured["accept_completed_by"] == captured["time_limit"]


def test_assignment_change_diagnostic_ignores_route_order_only_changes():
    evaluator = CommonBaselineEvaluator(
        _parents(), CONFIG, FORMAL_SCOPE_V2, time_limit=5.0, checkpoints=(5.0,)
    )
    initial = evaluator.evaluate(_solution(), source="INITIAL")
    assert initial.status.value == "COMPLETED"
    reordered = canonicalize(
        initial.solution.parents,
        initial.solution.patterns,
        {
            route.robot_id: tuple(reversed(route.block_ids))
            for route in initial.solution.routes
        },
        CONFIG,
        scope=FORMAL_SCOPE_V2,
    )
    final = replace(initial, solution=reordered)
    diagnostics = pilot_script._knowledge_diagnostics(initial, final, CONFIG)
    assert diagnostics["assignment_changes"] == 0
    assert diagnostics["changed_route_robots"] > 0
    assert diagnostics["pure_order_changed_robots"] > 0


def test_existing_role_manifest_keeps_train_and_dev_disjoint():
    representatives = pilot_script.ranker_dataset.representatives_v05()
    train = {
        entry["instance_id"] for entry in representatives if entry["split"] == "MLP_TRAIN"
    }
    dev = {
        entry["instance_id"] for entry in representatives if entry["split"] == "MLP_DEV"
    }
    assert len(train) == 17
    assert len(dev) == 6
    assert train.isdisjoint(dev)


def test_expanded_joint_sheets_use_roles_and_geometry_instead_of_directory_names():
    import statistics
    from mrta_ranker import dataset
    candidates=dataset.train_sheet_candidates()
    split=dataset.read(dataset.SPLIT)
    assert set(candidates)==set(split['MLP_TRAIN'])
    assert any('/ID_TEST/' in path for path in candidates)
    assert set(candidates).isdisjoint(split['MLP_DEV']+split['ORACLE_DEV_CONSUMED'])
    for path,rows in candidates.items():
        assert len({r['instance_id'] for r in rows[:3]})==3
        for r in rows:
            assert r['relative_path']==path and r['split']=='MLP_TRAIN'
            assert r['validation_status']=='VALID' and r['duplicate_of'] is None
            assert r['formal_compatibility']=='PASS' and 10<=r['N']<=90
        counts=[r['N'] for r in rows]
        assert rows[0]['N']==min(counts) and rows[2]['N']==max(counts)
        assert abs(rows[1]['N']-statistics.median(counts))==min(abs(n-statistics.median(counts)) for n in counts)


def _research_retained_state(solution, directions, cmax):
    from dataclasses import asdict
    return dict(solution=dict(parents=[asdict(p) for p in solution.parents],
                              patterns=[asdict(p) for p in solution.patterns],
                              routes=[asdict(r) for r in solution.routes]),
                directions=directions, cmax=cmax)


def test_research_graph_identity_separates_geometry_and_patterns_from_route_labels():
    import run_gnn_shared_construction as shared
    from mrta_reference.geometry import build_legal_pattern_catalog
    parents=(ParentWeld('p',(1.,5.8),(3.,6.2)),)
    whole=canonicalize(parents,(SplitPattern('p',SplitKind.WHOLE),),
                       {0:('p::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
    other_owner=canonicalize(parents,whole.patterns,{1:('p::whole',)},CONFIG,
                             scope=FORMAL_SCOPE_V2)
    pattern=next(p for p in build_legal_pattern_catalog(parents,CONFIG,FORMAL_SCOPE_V2)['p']
                 if p.kind is SplitKind.Y_SPLIT)
    split=canonicalize(parents,(pattern,),{0:('p::0',),2:('p::1',)},CONFIG,
                       scope=FORMAL_SCOPE_V2)
    shifted_parents=(replace(parents[0],start=(2.,5.8),end=(4.,6.2)),)
    shifted=canonicalize(shifted_parents,whole.patterns,{0:('p::whole',)},CONFIG,
                         scope=FORMAL_SCOPE_V2)
    restored=[shared.research_restore(_research_retained_state(s,bits,100.),CONFIG)
              for s,bits in ((whole,((0,),(),(),())),(other_owner,((),(1,),(),())),
                             (split,((0,),(),(0,),())),(shifted,((0,),(),(),())))]
    assert restored[0][2]==restored[1][2]==restored[2][2]
    assert restored[0][3]==restored[1][3]
    assert restored[0][4]!=restored[1][4]
    assert restored[0][3]!=restored[2][3]
    assert restored[0][2]!=restored[3][2]
    assert len({r[3] for r in restored})==3
    with pytest.raises(ValueError,match='invalid explicit directions'):
        shared.research_restore(_research_retained_state(whole,((),(),(),()),100.),CONFIG)


@pytest.mark.parametrize('kind',('parent','insertion'))
def test_research_multiteacher_targets_keep_quality_weight_at_shared_prefix(monkeypatch,kind):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),)
    if kind=='parent':
        parents+=(ParentWeld('b',(4.,9.),(5.,9.)),)
        route_pairs=({0:('a::whole','b::whole')},{0:('b::whole','a::whole')})
        bits_pairs=(((0,0),(),(),()),((0,0),(),(),()))
    else:
        route_pairs=({0:('a::whole',)},{1:('a::whole',)})
        bits_pairs=(((0,),(),(),()),((),(1,),(),()))
    patterns=tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents)
    solutions=[canonicalize(parents,patterns,routes,CONFIG,scope=FORMAL_SCOPE_V2)
               for routes in route_pairs]
    rows=[dict(instance='same-geometry',method=method,seed=seed,
               details=dict(final_state=_research_retained_state(solution,bits,cmax)))
          for seed,(method,solution,bits,cmax) in enumerate(zip(
              ('HGA','WAG'),solutions,bits_pairs,(100.,200.)))]
    teachers,coverage=shared.research_teachers(rows,CONFIG)
    assert coverage['selected_teachers']==2
    assert {t['teacher'] for t in teachers}=={'HGA','WAG'}
    cmax_by_solution={s.canonical_json:c for s,c in zip(solutions,(100.,200.))}
    # Keep the real action generator and constructive teacher replay. Scheduling
    # is isolated here so unequal quality weights can be checked independently.
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',lambda solution,bits,config:
        SimpleNamespace(solution=solution,directions=bits,schedule=ScheduleResult(
            ScheduleStatus.FEASIBLE,cmax=cmax_by_solution[solution.canonical_json])))
    monkeypatch.setattr(shared,'research_certify',lambda scheduled,config:True)
    graphs,examples,steps,diagnostics=shared.research_prepare(teachers,CONFIG)
    assert len(graphs)==1 and len(diagnostics)==2
    assert all(d['replay_status']=='REPLAYED' for d in diagnostics)
    at_empty=[e for e in examples if e['kind']==kind
              and all(not route for route in e['step']['state_routes'])]
    assert len(at_empty)==1
    target=at_empty[0]['target']
    assert target.sum().item()==pytest.approx(1.)
    assert sorted(target[target>0].tolist())==pytest.approx([1/3,2/3])
    assert at_empty[0]['weight']==pytest.approx(.75)
    assert len(steps)==sum(len(p) for p in route_pairs[0].values())*2


@pytest.mark.parametrize('alias_parent',(False,True))
def test_research_holdout_rejects_geometry_alias_before_model_loading(monkeypatch,alias_parent):
    import run_gnn_shared_construction as shared
    parent=ParentWeld('p',(1.,9.),(2.,9.))
    solution=canonicalize((parent,),(SplitPattern('p',SplitKind.WHOLE),),
                           {0:('p::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
    state=_research_retained_state(solution,((0,),(),(),()),100.)
    geometry_key=shared.research_restore(state,CONFIG)[2]
    if alias_parent:
        alias=replace(parent,parent_id='renamed-parent')
        renamed=canonicalize((alias,),(SplitPattern(alias.parent_id,SplitKind.WHOLE),),
                              {0:(alias.parent_id+'::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
        state=_research_retained_state(renamed,((0,),(),(),()),100.)
    def forbidden(*args,**kwargs):
        raise AssertionError('holdout overlap reached model loading')
    monkeypatch.setattr(shared,'load_model',forbidden)
    rows=[dict(instance='renamed-evaluation-instance',tier='SMALL',
               details=dict(final_state=state))]
    with pytest.raises(ValueError,match='training/evaluation geometry overlap'):
        shared.research_evaluate(None,rows,CONFIG,1,training_geometry_keys={geometry_key})


def test_all_mode_teachers_keep_superior_region_and_mode_final_states():
    import run_gnn_shared_construction as shared
    from mrta_reference.geometry import build_legal_pattern_catalog
    parents=(ParentWeld('p',(1.,5.8),(3.,6.2)),)
    whole=canonicalize(parents,(SplitPattern('p',SplitKind.WHOLE),),{0:('p::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
    pattern=next(p for p in build_legal_pattern_catalog(parents,CONFIG,FORMAL_SCOPE_V2)['p'] if p.kind is SplitKind.Y_SPLIT)
    split=canonicalize(parents,(pattern,),{0:('p::0',),2:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    rows=[dict(instance='SMALL_01',method=method,seed=1,details=dict(
        initial_state=_research_retained_state(whole,((0,),(),(),()),200.),
        final_state=_research_retained_state(split,((0,),(),(0,),()),100.))) for method in ('MODE_NEW','REGION')]
    fixed,_=shared.research_teachers(rows,CONFIG)
    all_modes,coverage=shared.research_teachers(rows,CONFIG,deployment_only=False)
    assert len(fixed)==len(all_modes)==1
    assert fixed[0]['cmax']==200. and fixed[0]['source_state']=='initial_state'
    assert all_modes[0]['cmax']==100. and all_modes[0]['source_state']=='final_state'
    assert set(all_modes[0]['teachers'])=={'MODE_NEW','REGION'}
    assert coverage['nonfixed_pattern_states']==2
    assert coverage.get('excluded_nondeployment_patterns',0)==0


def test_refinement_split_keeps_all_method_seeds_and_detects_physical_alias():
    import run_gnn_shared_construction as shared
    def row(name,offset,method,seed):
        p=ParentWeld('p',(1.+offset,9.),(2.+offset,9.))
        solution=canonicalize((p,),(SplitPattern('p',SplitKind.WHOLE),),{0:('p::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
        return dict(instance=name,tier='SMALL',method=method,seed=seed,
                    details=dict(final_state=_research_retained_state(solution,((0,),(),(),()),100.)))
    rows=[row(name,offset,method,seed) for name,offset in (('SMALL_01',0.),('SMALL_02',5.))
          for method in ('MODE_NEW','REGION') for seed in (1,2)]
    train,holdout,info=shared.refinement_split(rows)
    assert len(train)==len(holdout)==4
    assert {r['instance'] for r in train}=={'SMALL_01'}
    assert {r['instance'] for r in holdout}=={'SMALL_02'}
    assert info['training_geometries']==info['holdout_geometries']==1
    rows.append(row('SMALL_02',0.,'HGA',3))
    with pytest.raises(ValueError,match='training/evaluation geometry overlap'):
        shared.refinement_split(rows)


def test_best_mode_labels_do_not_average_away_a_superior_split():
    import run_gnn_shared_construction as shared
    from mrta_reference.geometry import build_legal_pattern_catalog
    parents=(ParentWeld('p',(1.,5.8),(3.,6.2)),)
    whole=SplitPattern('p',SplitKind.WHOLE)
    cut=next(p for p in build_legal_pattern_catalog(parents,CONFIG,FORMAL_SCOPE_V2)['p'] if p.kind is SplitKind.Y_SPLIT)
    rows=[]
    for robot,method in enumerate(('HGA','WAG','FIXED_NEW')):
        solution=canonicalize(parents,(whole,),{robot:('p::whole',)},CONFIG,scope=FORMAL_SCOPE_V2)
        bits=tuple((0,) if r==robot else () for r in range(4))
        rows.append(dict(instance='SMALL_01',method=method,seed=1,
            details=dict(final_state=_research_retained_state(solution,bits,200.))))
    split=canonicalize(parents,(cut,),{0:('p::0',),2:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    rows.append(dict(instance='SMALL_01',method='MODE_NEW',seed=1,
        details=dict(final_state=_research_retained_state(split,((0,),(),(0,),()),100.))))
    teachers,_=shared.research_teachers(rows,CONFIG,deployment_only=False)
    averaged=shared.refinement_mode_examples(teachers,CONFIG)[0]
    best=shared.refinement_mode_examples(teachers,CONFIG,best_only=True)[0]
    assert averaged['target'][averaged['features'][:,0]==1].sum()>.5
    assert best['target'][best['features'][:,1]==1].sum()==pytest.approx(1.)
    assert best['changed']


def test_neural_portfolio_preserves_both_robust_proposals_when_new_policy_fails(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();old=object();new=object();scheduled=[]
    def construct(geometry,routes,directions,pending,model,config,**options):
        if model is new:return SimpleNamespace(solution=None,directions=None,rejection_reason='NO_LEGAL_CHILD_INSERTION')
        bits=((0,),(0,),(),()) if options['decode_mode']=='greedy' else ((1,),(0,),(),())
        return SimpleNamespace(solution=reference,directions=bits,rejection_reason=None)
    def schedule(solution,bits,config):
        scheduled.append(bits)
        return shared.pilot.oriented.DirectedSolution(solution,bits,ScheduleResult(
            ScheduleStatus.FEASIBLE,cmax=100. if bits[0][0] else 123.))
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',schedule)
    monkeypatch.setattr(shared,'research_certify',lambda candidate,config:True)
    best,info=shared.neural_initial_portfolio(reference.parents,old,new,CONFIG,reference_calls=3)
    assert best.schedule.cmax==100.
    assert len(scheduled)==info['b32_calls']==2
    assert len(info['attempts'])==3 and info['attempts'][-1]['status']=='CONSTRUCTION_FAILED'


def test_neural_lb_pruning_never_skips_feasibility_search_and_labels_skips(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    from mrta_search.lns import oriented_route_state, oriented_route_lower_bound
    reference=_joint_simple_reference();proposals=[];scheduled=[]
    def construct(*args,**kwargs):
        bits=(((0,),(0,),(),()),((1,),(0,),(),()),((0,),(1,),(),()))[len(proposals)]
        proposals.append(bits)
        state=oriented_route_state(tuple(r.block_ids for r in reference.routes),bits,block_map(reference,CONFIG),CONFIG)
        return SimpleNamespace(solution=reference,directions=bits,rejection_reason=None,oriented_state=state)
    def schedule(solution,bits,config):
        scheduled.append(bits)
        state=oriented_route_state(tuple(r.block_ids for r in reference.routes),bits,block_map(reference,CONFIG),CONFIG)
        return shared.pilot.oriented.DirectedSolution(solution,bits,ScheduleResult(
            ScheduleStatus.FEASIBLE,cmax=oriented_route_lower_bound(state)))
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',schedule)
    best,info=shared.neural_initial_portfolio(reference.parents,object(),object(),CONFIG,
                                            certify=False,prune_cmax_lb=True)
    assert best is not None and len(proposals)==3 and len(scheduled)==info['b32_calls']==1
    assert [a['status'] for a in info['attempts']]==['FEASIBLE','PRUNED_BY_CMAX_LB','PRUNED_BY_CMAX_LB']
    scheduled.clear();proposals.clear()
    def deadlock(solution,bits,config):
        scheduled.append(bits)
        return shared.pilot.oriented.DirectedSolution(solution,bits,ScheduleResult(ScheduleStatus.DEADLOCK))
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',deadlock)
    best,info=shared.neural_initial_portfolio(reference.parents,object(),object(),CONFIG,
                                            certify=False,prune_cmax_lb=True)
    assert best is None and len(scheduled)==info['b32_calls']==3


def test_rule_guided_rank_ties_use_rule_cost_and_keep_all_actions():
    from types import SimpleNamespace
    from mrta_search.gnn_constructor import rule_guided_action_logits
    variants=[]
    for robot,load in ((1,0.),(0,3.),(0,2.)):
        state=SimpleNamespace(loads=(load,0.,0.,0.),arc_costs=((),)*4)
        variants.append((state,SimpleNamespace(robot=robot,block_id='b')))
    scores=rule_guided_action_logits(torch.tensor([9.,1.,0.]),variants,{'b':0})
    assert scores.shape==(3,) and torch.isfinite(scores).all()
    assert int(scores.argmax())==2
    assert sorted(scores.tolist())==[-2.,-1.,0.]


def test_rule_assignment_is_a_soft_preference_in_rank_fusion():
    from types import SimpleNamespace
    from mrta_search.gnn_constructor import rule_guided_action_logits
    variants=[]
    for i in range(5):
        state=SimpleNamespace(loads=(float(i),0.,0.,0.),arc_costs=((),)*4)
        variants.append((state,SimpleNamespace(robot=0 if i==4 else 1,block_id='b')))
    scores=rule_guided_action_logits(torch.tensor([5.,4.,3.,2.,1.]),variants,{'b':0})
    assert len(scores)==len(variants) and variants[int(scores.argmax())][1].robot==1


def test_rule_anchor_neural_tie_break_cannot_override_lower_rule_cost():
    from types import SimpleNamespace
    from mrta_search.gnn_constructor import rule_anchored_action_logits
    variants=[]
    for robot,load in ((0,2.),(0,2.),(0,3.),(1,0.)):
        state=SimpleNamespace(loads=(load,0.,0.,0.),arc_costs=((),)*4)
        variants.append((state,SimpleNamespace(robot=robot,block_id='b')))
    scores=rule_anchored_action_logits(torch.tensor([1.,2.,100.,1000.]),variants,{'b':0})
    assert int(scores.argmax())==1
    assert sorted(scores.tolist())==[-3.,-2.,-1.,0.]
    # Even the nonpreferred robot remains in the sampling distribution.
    assert torch.all(torch.softmax(scores,0)>0.)


def test_diagnostic_action_observer_preserves_decoding_and_same_pool_rankings():
    import mrta_search.gnn_constructor as constructor
    from mrta_search.lns import joint_action_key
    reference=_joint_simple_reference();torch.manual_seed(29)
    model=constructor.JointInsertionGNN()
    geometry=constructor.joint_geometry(reference.parents,reference.patterns,CONFIG)
    observed=[]
    def observe(graph,state,variants,logits,selected,owners):
        keys=tuple(joint_action_key(a) for _,a in variants)
        costs=constructor.rule_action_costs(variants,owners)
        fusion=constructor.rule_guided_action_logits(logits,variants,owners)
        assert len(keys)==len(costs)==len(fusion)
        assert keys==tuple(joint_action_key(a) for _,a in variants)
        observed.append(keys[selected])
    options=dict(parent_selection_mode='mrv',action_selection_mode='rule_guided')
    plain=constructor.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
        tuple(p.parent_id for p in reference.parents),model,CONFIG,**options)
    audited=constructor.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
        tuple(p.parent_id for p in reference.parents),model,CONFIG,action_observer=observe,**options)
    assert plain.solution==audited.solution and plain.directions==audited.directions
    assert tuple(observed)==tuple(joint_action_key(a) for a in plain.actions)


def test_diagnostic_milestones_are_exact_seeded_search_prefixes():
    import run_gnn_multi_initial as experiment
    reference=_joint_simple_reference();bits=((0,),(0,),(),())
    initial=experiment.shared.pilot.oriented.schedule_explicit(reference,bits,CONFIG)
    options=dict(config=CONFIG,certify=False,common_initial=initial,prune_cmax_lb=True)
    short=experiment._mode_search(reference.parents,'MODE_NEW',23,float('inf'),2,**options)
    long=experiment._mode_search(reference.parents,'MODE_NEW',23,float('inf'),4,
        iteration_checkpoints=(2,4),compact_trace=True,**options)
    milestone=long['iteration_checkpoints'][0]
    assert milestone['solution']==short['solution'] and milestone['directions']==short['directions']
    assert milestone['cmax']==short['cmax'] and milestone['reference_calls']==short['reference_calls']
    assert milestone['accepted']==short['accepted']
    assert len(long['trace'])==4 and long['iterations']==4


@pytest.mark.parametrize('parent,action', [('learned','gnn'),('mrv','gnn'),('mrv','rule_guided'),
                                        ('mrv','rule'),('mrv','gnn_rank'),('mrv','rule_anchored')])
def test_portfolio_passes_rule_guidance_without_extra_proposals(monkeypatch,parent,action):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();calls=[]
    def construct(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(solution=None,rejection_reason='FIXTURE')
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    _,info=shared.neural_initial_portfolio(reference.parents,object(),object(),CONFIG,certify=False,
        parent_selection_mode=parent,action_selection_mode=action)
    assert len(calls)==len(info['attempts'])==3
    for options in calls:
        assert options.get('parent_selection_mode','learned')==parent
        assert options.get('action_selection_mode','gnn')==action
        if parent=='learned':assert set(options)=={'decode_mode','seed'}


def test_diagnostic_checkpoint_uses_completion_time_and_never_backdates():
    import run_rule_guided_gnn as experiment
    events=[[10.,150.,'INITIAL'],[60.01,100.,'MODE'],[300.01,90.,'MODE']]
    assert experiment.diagnostic_value(events,5.) is None
    assert experiment.diagnostic_value(events,60.)==150.
    assert experiment.diagnostic_value(events,300.)==100.
    assert experiment.diagnostic_value(events,600.)==90.


def test_real_mrv_rule_guided_decode_bypasses_parent_head_and_preserves_legal_pool(monkeypatch):
    import mrta_search.gnn_constructor as constructor
    from mrta_search.lns import joint_action_key
    reference=_joint_simple_reference();torch.manual_seed(9)
    model=constructor.JointInsertionGNN()
    def forbidden(*args,**kwargs):raise AssertionError('MRV called the learned parent head')
    monkeypatch.setattr(model,'parent_logits',forbidden)
    original=constructor.rule_guided_action_logits;pool_sizes=[]
    def fusion(logits,variants,owners):
        before=tuple(joint_action_key(a) for _,a in variants)
        out=original(logits,variants,owners)
        assert before==tuple(joint_action_key(a) for _,a in variants) and len(out)==len(variants)
        pool_sizes.append(len(variants));return out
    monkeypatch.setattr(constructor,'rule_guided_action_logits',fusion)
    geometry=constructor.joint_geometry(reference.parents,reference.patterns,CONFIG)
    result=constructor.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
        tuple(p.parent_id for p in reference.parents),model,CONFIG,
        parent_selection_mode='mrv',action_selection_mode='rule_guided')
    assert result.solution is not None and len(result.actions)==len(pool_sizes)==2
    assert result.solution.patterns==reference.patterns


def test_rule_guided_dataset_loader_never_opens_reserved_holdout(monkeypatch):
    import run_rule_guided_gnn as experiment
    def forbidden(*args,**kwargs):raise AssertionError('reserved workbook was opened')
    monkeypatch.setattr(experiment,'load_ppo_platform_instance',forbidden)
    with pytest.raises(ValueError,match='not development inputs'):
        experiment.load_entry(dict(purpose='holdout',role='V2_TRAIN_POOL'))
    with pytest.raises(ValueError,match='not development inputs'):
        experiment.load_entry(dict(purpose='train',role='ID_TEST_SEALED'))


def test_retained_b32_teacher_replay_needs_no_second_schedule_or_certifier(monkeypatch):
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();bits=((0,),(0,),(),())
    initial=shared.pilot.oriented.schedule_explicit(reference,bits,CONFIG)
    row=dict(instance='TRAIN_fixture',method='RULE',seed=1,
        details=dict(final_state=_research_retained_state(reference,bits,initial.schedule.cmax)))
    teachers,_=shared.research_teachers([row],CONFIG,deployment_only=False)
    for teacher in teachers:teacher['b32_status']='FEASIBLE'
    def forbidden(*args,**kwargs):raise AssertionError('duplicate schedule or independent certifier')
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',forbidden)
    monkeypatch.setattr(shared,'research_certify',forbidden)
    graphs,examples,steps,replays=shared.research_prepare(teachers,CONFIG,certify=False)
    assert graphs and examples and steps and replays[0]['replay_status']=='REPLAYED'
    teachers[0]['b32_status']='DEADLOCK'
    with pytest.raises(ValueError,match='no FEASIBLE B32 record'):
        shared.research_prepare(teachers,CONFIG,certify=False)


@pytest.mark.parametrize('prune', [False, True])
def test_mode_search_bound_keeps_fixed_iterations_and_strict_incumbent(monkeypatch, prune):
    from types import SimpleNamespace
    import run_gnn_multi_initial as experiment
    import mrta_search.lns as lns
    reference=_joint_simple_reference();bits=((0,),(0,),(),())
    initial=experiment.shared.pilot.oriented.schedule_explicit(reference,bits,CONFIG)
    state=lns.oriented_route_state(tuple(r.block_ids for r in reference.routes),bits,block_map(reference,CONFIG),CONFIG)
    assert initial.schedule.cmax==pytest.approx(lns.oriented_route_lower_bound(state))
    monkeypatch.setattr(lns,'repair_partial_state_oriented',lambda *a,**k:SimpleNamespace(
        candidate=SimpleNamespace(solution=reference),directions=bits,oriented_state=state))
    scheduled=[]
    def schedule(*args):scheduled.append(1);return initial
    monkeypatch.setattr(experiment.shared.pilot.oriented,'schedule_explicit',schedule)
    out=experiment._mode_search(reference.parents,'MODE_NEW',1,float('inf'),4,
        config=CONFIG,certify=False,common_initial=initial,prune_cmax_lb=prune)
    assert out['iterations']==4 and out['accepted']==0 and out['cmax']==initial.schedule.cmax
    assert len(scheduled)==out['reference_calls']==(0 if prune else 4)
    assert out['pruned_by_cmax_lb']==(4 if prune else 0)
    if prune:assert all(t['status']=='PRUNED_BY_CMAX_LB' for t in out['trace'])


@pytest.mark.parametrize('old_sampling', [False, True])
def test_neural_portfolio_can_use_b32_only_and_preserves_comparator_proposals(monkeypatch, old_sampling):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();old=object();new=object();proposals=[]
    def construct(geometry,routes,directions,pending,model,config,**options):
        proposals.append((model,options['decode_mode'],options['seed']))
        bits=(((0,),(0,),(),()),((1,),(0,),(),()),((0,),(1,),(),()))[len(proposals)-1]
        return SimpleNamespace(solution=reference,directions=bits,rejection_reason=None)
    def schedule(solution,bits,config):
        return shared.pilot.oriented.DirectedSolution(solution,bits,ScheduleResult(
            ScheduleStatus.FEASIBLE,cmax=400.-len(proposals)*100.))
    def forbidden(*args,**kwargs):raise AssertionError('independent certification was called')
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',schedule)
    monkeypatch.setattr(shared,'research_certify',forbidden)
    best,info=shared.neural_initial_portfolio(reference.parents,old,new,CONFIG,seed=7,
                                            old_sampling=old_sampling,certify=False)
    assert proposals==[(old,'greedy',7),(old,'sample',7),
                       (old,'sample',8) if old_sampling else (new,'greedy',7)]
    assert info['b32_calls']==3 and best.schedule.cmax==100.


@pytest.mark.parametrize('late_construction', [False, True])
def test_neural_portfolio_deadline_rejects_late_b32_or_construction(monkeypatch, late_construction):
    from types import SimpleNamespace
    import run_gnn_shared_construction as shared
    reference=_joint_simple_reference();clock=[100.];proposals=[];schedules=[]
    monkeypatch.setattr(shared,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    def construct(*args,**kwargs):
        proposals.append(1)
        if late_construction:clock[0]=161.
        bits=((len(proposals)-1,),(0,),(),())
        return SimpleNamespace(solution=reference,directions=bits,rejection_reason=None)
    def schedule(solution,bits,config):
        schedules.append(1);clock[0]=110. if len(schedules)==1 else 161.
        return shared.pilot.oriented.DirectedSolution(solution,bits,ScheduleResult(
            ScheduleStatus.FEASIBLE,cmax=300.-len(schedules)*100.))
    monkeypatch.setattr(shared,'construct_or_repair_with_gnn',construct)
    monkeypatch.setattr(shared.pilot.oriented,'schedule_explicit',schedule)
    best,info=shared.neural_initial_portfolio(reference.parents,object(),object(),CONFIG,
                                            certify=False,deadline=160.)
    if late_construction:
        assert best is None and not schedules and len(proposals)==1
        assert info['attempts'][0]['status']=='CONSTRUCTION_FINISHED_AFTER_DEADLINE'
    else:
        assert best.schedule.cmax==200. and info['selected_seconds']==10.
        assert len(schedules)==2 and info['attempts'][-1]['cmax'] is None
        assert not info['attempts'][-1]['admitted'] and info['attempts'][-1]['late']


def test_initial_mode_new_failure_never_falls_back_and_charges_model_load(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_initial_mode_new as study
    clock=[100.]
    monkeypatch.setattr(study,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    def load(*args):clock[0]+=5.;return object(),object()
    def portfolio(*args,**options):
        assert options['deadline']==160. and options['certify'] is False
        assert options['reference_calls']==3 and options['use_learned_modes'] is False
        clock[0]+=1.
        return None,dict(attempts=[dict(status='DEADLOCK',b32_calls=1)]*3,b32_calls=3,selected_seconds=None)
    def forbidden(*args,**kwargs):raise AssertionError('hidden rule initialization/search after GNN failure')
    monkeypatch.setattr(study.shared,'load_neural_initial_models',load)
    monkeypatch.setattr(study.shared,'neural_initial_portfolio',portfolio)
    monkeypatch.setattr(study,'_mode_search',forbidden)
    monkeypatch.setattr(study.shared.pilot.oriented,'common_initial_beam',forbidden)
    entry=dict(instance='fixture',tier='SMALL',N=1,geometry_seed=1,workbook='fixture.xlsx',sheet='raw',
               parents=[dict(parent_id='p',start=(1.,9.),end=(2.,9.))])
    row=study.run_job((entry,'NEW_GNN_PORTFOLIO',7,'smoke'))
    assert row['status']=='INITIALIZATION_FAILED' and row['final_cmax'] is None
    assert row['model_load_seconds']==5. and row['actual_runtime']==6. and row['b32_calls']==3


def test_initial_mode_new_passes_full_initialization_cost_to_unchanged_backend(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_initial_mode_new as study
    clock=[100.];initial=SimpleNamespace(schedule=SimpleNamespace(cmax=123.));calls=[]
    monkeypatch.setattr(study,'time',SimpleNamespace(perf_counter=lambda:clock[0]))
    def load(*args):clock[0]+=5.;return object(),object()
    def portfolio(*args,**kwargs):
        clock[0]+=2.
        return initial,dict(attempts=[dict(status='FEASIBLE',b32_calls=1)],b32_calls=1,selected_seconds=1.)
    def search(*args,**kwargs):
        calls.append(kwargs)
        assert kwargs['common_initial'] is initial and kwargs['initial_seconds']==7.
        assert kwargs['method']=='MODE_NEW' and kwargs['budget']==60. and kwargs['certify'] is False
        assert kwargs['config'] is study.CONFIG_05 and kwargs['max_iterations']==1_000_000
        clock[0]=160.5
        return dict(status='FEASIBLE',cmax=100.,reference_calls=4,iterations=3,accepted=1,mode_accepted=1,
                    best_events=[[7.,123.,'INITIAL'],[59.,100.,'MODE']],trace=[dict(late=True)])
    monkeypatch.setattr(study.shared,'load_neural_initial_models',load)
    monkeypatch.setattr(study.shared,'neural_initial_portfolio',portfolio)
    monkeypatch.setattr(study,'_mode_search',search)
    entry=dict(instance='fixture',tier='SMALL',N=1,geometry_seed=1,workbook='fixture.xlsx',sheet='raw',
               parents=[dict(parent_id='p',start=(1.,9.),end=(2.,9.))])
    row=study.run_job((entry,'NEW_GNN_PORTFOLIO',7,'smoke'))
    assert len(calls)==1 and row['actual_runtime']==60.5 and row['initial_found_seconds']==6.
    assert row['last_improvement_seconds']==59. and row['b32_calls']==4 and row['search_b32_calls']==3


def test_initial_mode_new_pairs_use_geometry_medians_and_keep_missing_pairs_out():
    import run_gnn_initial_mode_new as study
    rows=[]
    for i,(rule,new) in enumerate(((100.,90.),(100.,90.),(1000.,1100.),(None,None),(None,None))):
        rows.extend([dict(instance='a',method='RULE',final_cmax=rule),
                     dict(instance='a',method='NEW_GNN_PORTFOLIO',final_cmax=new)])
    rows.extend([dict(instance='b',method='RULE',final_cmax=100.),
                 dict(instance='b',method='NEW_GNN_PORTFOLIO',final_cmax=None)])
    counts,change,pairs=study.pair_summary(rows,'final_cmax','RULE')
    assert counts==[1,0,0] and change==pytest.approx(10.)
    assert len(pairs)==1 and pairs[0][1:3]==(100.,90.)
    assert len(rows)==12


def test_initial_mode_new_resume_keeps_saved_failures_and_separates_smoke(monkeypatch):
    import run_gnn_initial_mode_new as study
    entries=[dict(instance=tier+'_01',tier=tier) for tier in ('SMALL','LARGE','HIGH_WORKLOAD')]
    saved=[dict(phase='smoke',instance='SMALL_01',solver_seed='20261900',method='OLD_GNN_PORTFOLIO',
                status='INITIALIZATION_FAILED'),
           dict(phase='formal',instance='SMALL_01',solver_seed='20261900',method='NEW_GNN_PORTFOLIO',
                status='FEASIBLE')]
    submitted=[];written=[]
    class Future:
        def __init__(self,job):self.job=job
        def result(self):
            entry,method,seed,phase=self.job
            return dict(instance=entry['instance'],method=method,solver_seed=seed,phase=phase,status='FEASIBLE',
                        initial_cmax=100.,final_cmax=90.,actual_runtime=60.,b32_calls=3)
    class Executor:
        def __init__(self,**kwargs):assert kwargs['max_workers']==4
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,job):submitted.append(job);return Future(job)
    monkeypatch.setattr(study,'read_rows',lambda:saved)
    monkeypatch.setattr(study,'ProcessPoolExecutor',Executor)
    monkeypatch.setattr(study,'as_completed',lambda futures:iter(futures))
    monkeypatch.setattr(study,'append_row',written.append)
    study.run_phase(entries,'smoke',4)
    assert len(submitted)==len(written)==8
    assert not any(e['instance']=='SMALL_01' and m=='OLD_GNN_PORTFOLIO' for e,m,s,p in submitted)
    assert any(e['instance']=='SMALL_01' and m=='NEW_GNN_PORTFOLIO' for e,m,s,p in submitted)


def test_neural_archive_retains_exact_weights_and_skips_diagnostics_at_inference(monkeypatch):
    import io
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import CoordinationJointInsertionGNN,ModeJointInsertionGNN
    old=CoordinationJointInsertionGNN();new=ModeJointInsertionGNN()
    payload=dict(model_type='CoordinationJointInsertionGNN',state_dict=old.state_dict(),
        normalize_inputs=True,use_dynamic_context=False,training=dict(loss=[2.,1.],failures=[None,'DEADLOCK']),
        refinement_candidate=dict(model_type='ModeJointInsertionGNN',state_dict=new.state_dict(),
            normalize_inputs=True,use_dynamic_context=False,refinement=dict(methods=['REGION','MODE_NEW'],cmax=123.45)))
    opportunity=shared.expand_opportunity_model(new)
    payload['opportunity_cmax_candidate']=dict(model_type='ModeJointInsertionGNN',state_dict=opportunity.state_dict(),
        opportunity_features=True,normalize_inputs=True,use_dynamic_context=False,training=dict(prefixes=[.25,.5,.75]))
    payload['early_cmax_candidate']=dict(payload['opportunity_cmax_candidate'],training=dict(prefixes=[0.,.1,.25]))
    stream=io.BytesIO();shared.save_neural_checkpoint(payload,stream)
    stream.seek(0);full=shared.read_neural_checkpoint(stream)
    assert full['training']==payload['training']
    assert full['refinement_candidate']['refinement']==payload['refinement_candidate']['refinement']
    assert full['early_cmax_candidate']['training']==dict(prefixes=[0.,.1,.25])
    assert full['opportunity_cmax_candidate']['training']==dict(prefixes=[.25,.5,.75])
    for a,b in ((payload['state_dict'],full['state_dict']),
                (payload['refinement_candidate']['state_dict'],full['refinement_candidate']['state_dict'])):
        assert all(torch.equal(a[k],b[k]) for k in a)
    stream.seek(0);thin=shared.read_neural_checkpoint(stream,include_research=False)
    early=shared._model_from_checkpoint(thin['early_cmax_candidate'])
    assert 'training' not in thin['early_cmax_candidate']
    assert all(torch.equal(v,early.state_dict()[k]) for k,v in opportunity.state_dict().items())
    monkeypatch.setattr(shared,'_expand_neural_checkpoint',lambda *args:
        (_ for _ in ()).throw(AssertionError('inference expanded research diagnostics')))
    stream.seek(0);loaded_old,loaded_new=shared.load_neural_initial_models(stream)
    assert not loaded_old.use_dynamic_context and not loaded_new.use_dynamic_context
    assert all(torch.equal(old.state_dict()[k],loaded_old.state_dict()[k]) for k in old.state_dict())
    assert all(torch.equal(new.state_dict()[k],loaded_new.state_dict()[k]) for k in new.state_dict())


def test_mode_policy_exposes_current_legal_catalog_and_stable_parent_ties():
    from mrta_search.gnn_constructor import ModeJointInsertionGNN,legal_mode_inputs,select_gnn_patterns
    from mrta_reference.geometry import build_legal_pattern_catalog,blocks_for_pattern
    config=ScientificConfig()
    parents=(ParentWeld('flex',(2.,6.),(8.,6.)),ParentWeld('cross',(10.,9.),(10.,3.)))
    graph,candidates,features=legal_mode_inputs(parents,config)
    catalog=build_legal_pattern_catalog(tuple(sorted(parents,key=lambda p:p.parent_id)),config,FORMAL_SCOPE_V2)
    expected=tuple(pattern for parent in sorted(parents,key=lambda p:p.parent_id)
                   for pattern in catalog[parent.parent_id])
    assert candidates==expected and features.shape==(len(expected),12)
    assert torch.isfinite(features).all()
    assert any(pattern.kind is SplitKind.WHOLE for pattern in candidates)
    assert any(pattern.kind is SplitKind.Y_SPLIT and pattern.mandatory for pattern in candidates)
    assert any(pattern.kind is SplitKind.X_SPLIT for pattern in candidates)
    by_id={parent.parent_id:parent for parent in parents}
    assert all(child.length>=config.min_child_length-config.numeric_epsilon for pattern in candidates
               for child in blocks_for_pattern(by_id[pattern.parent_id],pattern,config))
    model=ModeJointInsertionGNN(16,4)
    assert model.use_dynamic_context is False
    with torch.no_grad():
        for parameter in model.mode_head.parameters():parameter.zero_()
    assert select_gnn_patterns(parents,model,config)==tuple(catalog[p.parent_id][0]
        for p in sorted(parents,key=lambda p:p.parent_id))
    assert set(graph.parent_block_indices)==set(by_id)


def test_mode_logits_accept_repeated_parent_candidates_and_backpropagate():
    from mrta_search.gnn_constructor import ModeJointInsertionGNN,legal_mode_inputs
    torch.manual_seed(27)
    config=ScientificConfig();parents=(ParentWeld('flex',(2.,6.),(8.,6.)),)
    graph,candidates,features=legal_mode_inputs(parents,config)
    assert len(candidates)>1
    model=ModeJointInsertionGNN(16,4)
    features=features.clone().requires_grad_()
    logits=model.mode_logits(graph,model.encode_static(graph),tuple(p.parent_id for p in candidates),features)
    assert logits.shape==(len(candidates),) and torch.isfinite(logits).all()
    assert logits.unique().numel()>1
    logits.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum()>0
    for module in (model.mode_head,model.geometry_skip):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
                   for p in module.parameters())


def test_mode_checkpoint_roundtrip_preserves_type_weights_and_context_without_files(monkeypatch):
    import io
    import run_gnn_shared_construction as shared
    from mrta_search.gnn_constructor import ModeJointInsertionGNN,select_gnn_patterns
    model=ModeJointInsertionGNN(16,4,normalize_inputs=True)
    checkpoint=dict(model_type='ModeJointInsertionGNN',hidden_dim=16,heads=4,normalize_inputs=True,
                    use_dynamic_context=False,state_dict=model.state_dict())
    buffer=io.BytesIO();torch.save(checkpoint,buffer);serialized=buffer.getvalue()
    original_load=torch.load;requested=Path('in-memory-mode-checkpoint.pt')
    def load_memory(path,**kwargs):
        assert path==requested and kwargs==dict(map_location='cpu',weights_only=True)
        return original_load(io.BytesIO(serialized),**kwargs)
    monkeypatch.setattr(shared.torch,'load',load_memory)
    restored,metadata=shared.load_model(requested)
    assert isinstance(restored,ModeJointInsertionGNN) and restored.use_dynamic_context is False
    assert metadata['model_type']=='ModeJointInsertionGNN'
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in model.state_dict().items())
    parents=(ParentWeld('flex',(2.,6.),(8.,6.)),)
    assert select_gnn_patterns(parents,restored,ScientificConfig())==select_gnn_patterns(parents,model,ScientificConfig())


@pytest.mark.parametrize('kind',(SplitKind.WHOLE,SplitKind.Y_SPLIT,SplitKind.X_SPLIT,'mandatory_y'))
@pytest.mark.parametrize('check_start',(False,True))
@pytest.mark.parametrize('limit',(2,8))
def test_joint_existence_matches_full_pool_for_modes_siblings_and_partial_starts(kind,check_start,limit):
    from mrta_reference.geometry import build_legal_pattern_catalog,blocks_for_pattern
    from mrta_reference.model import Rail
    from mrta_search import lns
    config=ScientificConfig()
    parent=(ParentWeld('p',(10.,9.),(10.,3.)) if kind=='mandatory_y'
            else ParentWeld('p',(7.2,5.8),(11.2,6.2)))
    others=(ParentWeld('a',(.5,9.),(1.5,9.)),ParentWeld('b',(17.,3.),(18.,3.)),
            ParentWeld('c',(10.,9.),(11.,9.)),ParentWeld('d',(5.,9.),(6.,9.)))
    catalog=build_legal_pattern_catalog((parent,*others),config,FORMAL_SCOPE_V2)
    pattern=next(p for p in catalog['p'] if
        (p.kind is SplitKind.Y_SPLIT and p.mandatory if kind=='mandatory_y' else p.kind is kind)
        and (p.kind is not SplitKind.X_SPLIT or p.rail is Rail.UPPER))
    blocks={child.block_id:child for p in (parent,*others)
            for child in blocks_for_pattern(p,pattern if p.parent_id=='p' else SplitPattern(p.parent_id,SplitKind.WHOLE),config)}
    block=blocks['p::whole' if kind is SplitKind.WHOLE else 'p::1']
    routes=[((),(),(),()),(('a::whole',),('c::whole',),('b::whole',),()),
            (('c::whole',),('d::whole',),('b::whole',),()),
            (('a::whole','d::whole','c::whole'),(),('b::whole',),())]
    if kind is not SplitKind.WHOLE:
        routes.extend(((('a::whole','p::0'),('c::whole',),('b::whole',),()),
                       (('a::whole',),('p::0','c::whole'),('b::whole',),())))
    for route_set in routes:
        for bit in (0,1):
            bits=tuple((bit,)*len(route) for route in route_set)
            state=lns.oriented_route_state(route_set,bits,blocks,config)
            expected=bool(lns.joint_block_candidates(state,block,pattern,blocks,config,
                insertion_limit=limit,first_only=True,check_initial_feasibility=check_start))
            assert lns.has_joint_block_candidate(state,block,pattern,blocks,config,
                insertion_limit=limit,check_initial_feasibility=check_start) is expected
            if not check_start:
                for recovery in (False,True):
                    assert bool(lns.joint_block_candidates(state,block,pattern,blocks,config,
                        insertion_limit=limit,check_initial_feasibility=False,
                        prefer_legal_start_recovery=recovery)) is expected


def test_joint_existence_does_not_build_or_sort_direction_variants(monkeypatch):
    from mrta_search import lns
    from mrta_search.gnn_constructor import joint_graph,joint_start_state
    reference=_joint_simple_reference();graph,blocks,patterns=joint_graph(reference,CONFIG)
    state=joint_start_state(reference,((),(),(),()),((),(),(),()),tuple(patterns),blocks,CONFIG)
    block=next(iter(blocks.values()))
    def forbidden(*args,**kwargs):raise AssertionError('existence query constructed arc costs')
    monkeypatch.setattr(lns,'oriented_insert_variants',forbidden)
    assert lns.has_joint_block_candidate(state,block,patterns[block.parent_id],blocks,CONFIG)

