from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import random
import sys
import time
import pytest
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gnn_construct_vns_pilot as pilot_script


def _feedback_fixture():
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c
    experiment.configure_cpu()
    torch.manual_seed(81)
    parents = (ParentWeld('a', (1., 9.), (6., 9.)), ParentWeld('b', (14., 9.), (18., 9.)),
               ParentWeld('c', (2., 5.), (2., 7.)), ParentWeld('d', (14., 2.), (18., 2.)))
    model = c.VNextGNN(16, 4, workload_lookahead=True, joint_actions=True, schedule_feedback=True)
    return experiment, c, parents, model


@pytest.mark.parametrize('block,destination,sibling', [('a::0',3,'a::1'),('a::1',1,'a::0')])
def test_child_transfer_moves_y_child_independently_and_keeps_sibling_exactly(block,destination,sibling):
    from collections import Counter
    from test_gnn_schedule_blocker_pilot import _selection_fixture
    from mrta_search import lns
    solution,_,_,config=_selection_fixture()
    bits=((1,),(0,1),(0,),(1,))
    before={b:(r,i,bits[r][i]) for r,route in enumerate(solution.routes) for i,b in enumerate(route.block_ids)}
    transfers=lns.child_transfer_candidates(solution,bits,block,config,destination_robot=destination)
    assert transfers
    for transfer in transfers:
        after={b:(r,i,transfer.state.directions[r][i]) for r,route in enumerate(transfer.state.routes)
               for i,b in enumerate(route)}
        assert after[sibling]==before[sibling]
        assert after[block][0]==destination and transfer.move.block_ids==(block,)
        assert transfer.solution.patterns==solution.patterns
        assert Counter(b for route in transfer.state.routes for b in route)==Counter(before.keys())
        assert transfer.state.directions[destination][transfer.action.position]==transfer.action.direction
        assert len(transfer.partial.routes[before[block][0]])==len(solution.routes[before[block][0]].block_ids)-1
        assert sibling in transfer.partial.routes[before[sibling][0]]


def _child_whole_fixture():
    from mrta_search import lns
    parents=tuple(ParentWeld(name,(x,y),(x+1,y)) for name,x,y in
                  [('a',2.,9.),('b',15.,9.),('c',2.,2.),('d',15.,2.)])
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
        {i:(p.parent_id+'::whole',) for i,p in enumerate(parents)},CONFIG,scope=FORMAL_SCOPE_V2)
    bits=((0,),)*4
    state=lns.oriented_route_state(tuple(r.block_ids for r in solution.routes),bits,block_map(solution,CONFIG),CONFIG)
    return solution,bits,state


def test_child_whole_transfer_uses_existing_qualification_and_joint_directions():
    from mrta_search import lns
    from mrta_reference.model import MoveType
    solution,bits,_=_child_whole_fixture()
    assert not lns.child_transfer_candidates(solution,bits,'a::whole',CONFIG,destination_robot=2)
    transfers=lns.child_transfer_candidates(solution,bits,'a::whole',CONFIG,destination_robot=1)
    assert transfers and {t.action.direction for t in transfers}=={0,1}
    assert {t.action.position for t in transfers}=={0,1}
    assert any(t.action.neighbour_changes for t in transfers)
    for t in transfers:
        assert t.move.key.move_type is MoveType.INTER_RELOCATE
        assert t.state.routes[0]==() and t.state.routes[2:]==tuple(r.block_ids for r in solution.routes[2:])
        assert t.state.directions[2:]==bits[2:]
        assert t.solution.patterns==solution.patterns


def test_child_x_split_cannot_change_pinned_robot_ownership():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.model import Rail
    from mrta_search import lns
    parent=ParentWeld('p',(7.2,5.8),(11.2,6.2))
    pattern=next(p for p in build_legal_pattern_catalog((parent,),CONFIG,FORMAL_SCOPE_V2)['p']
                 if p.kind is SplitKind.X_SPLIT and p.rail is Rail.UPPER)
    solution=canonicalize((parent,),(pattern,),{0:('p::0',),1:('p::1',)},CONFIG,scope=FORMAL_SCOPE_V2)
    for block in ('p::0','p::1'):
        assert lns.child_transfer_candidates(solution,((0,),(0,),(),()),block,CONFIG)==()


def test_child_transfer_rejects_canonical_whole_merge_exposed_on_source_route():
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_search import lns
    parents=(ParentWeld('q',(7.2,5.8),(11.2,6.2)),ParentWeld('a',(2.,9.),(3.,9.)),
             ParentWeld('b',(17.,9.),(18.,9.)))
    pattern=next(p for p in build_legal_pattern_catalog(parents,CONFIG,FORMAL_SCOPE_V2)['q']
                 if p.kind is SplitKind.Y_SPLIT)
    patterns=(pattern,SplitPattern('a',SplitKind.WHOLE),SplitPattern('b',SplitKind.WHOLE))
    solution=canonicalize(parents,patterns,{0:('q::0','a::whole','q::1'),1:('b::whole',)},
                          CONFIG,scope=FORMAL_SCOPE_V2)
    assert any(p.kind is SplitKind.Y_SPLIT for p in solution.patterns)
    assert not lns.child_transfer_candidates(solution,((0,0,0),(0,),(),()),'a::whole',CONFIG,destination_robot=1)


def test_child_summary_maps_only_observed_wait_and_preserves_unmapped_robot_wait():
    from mrta_search import lns
    from mrta_reference.model import Operation,OperationKind,ScheduleResult,ScheduleStatus
    solution,_,_=_child_whole_fixture()
    operations=(Operation('R0:B0:WELD',0,OperationKind.WELD,0.,2.,(2.,9.),(3.,9.),0,'a::whole'),
        Operation('R0:B0:MOVE',0,OperationKind.MOVE,2.,5.,(2.,9.),(3.,9.),0,None),
        Operation('R0:B0:MOVE:WAIT',0,OperationKind.WAIT,5.,9.,(2.,9.),(2.,9.),0,None),
        Operation('unknown:B0:WAIT',0,OperationKind.WAIT,9.,16.,(2.,9.),(2.,9.),0,None))
    schedule=ScheduleResult(ScheduleStatus.FEASIBLE,operations,16.,(16.,0.,0.,0.))
    feedback=lns.schedule_feedback_summary(solution,schedule,(),CONFIG)
    assert feedback['block_processing']['a::whole']==2.
    assert feedback['block_move']['a::whole']==3.
    assert feedback['block_wait']['a::whole']==4. and feedback['robot_wait'][0]==11.
    assert sum(feedback['block_wait'].values())==4.


def test_child_preference_labels_exclude_ties_double_deadlocks_and_numeric_failures():
    import run_gnn_weld_layout as e
    outcomes=[dict(status='FEASIBLE',cmax_s=10.),dict(status='FEASIBLE',cmax_s=9.),
              dict(status='FEASIBLE',cmax_s=10.),dict(status='DEADLOCK',cmax_s=None),
              dict(status='DEADLOCK',cmax_s=None),dict(status='NUMERIC_FAILURE',cmax_s=None)]
    pairs,kinds=e.child_preference_pairs(outcomes)
    assert len(pairs)==8 and kinds.count('CMAX')==2 and kinds.count('FEASIBILITY')==6
    assert (0,2) not in pairs and (3,4) not in pairs and all(5 not in pair for pair in pairs)
    assert (1,0) in pairs and (1,2) in pairs


def test_child_head_fits_same_state_pairs_without_updating_old_parameters():
    import json
    experiment,c,_,model=_feedback_fixture()
    model.requires_grad_(False).enable_child_transfer_head()
    width=model.child_transfer_mean.numel()
    before={k:v.clone() for k,v in model.state_dict().items()}
    records=[dict(layout_id='train_01',seed=experiment.SEED,status='FEASIBLE',diagnostic=json.dumps(
        dict(features=[[-1.]*width,[1.]*width],pairs=[[1,0]],pair_kinds=['CMAX'])))]
    summary=experiment.fit_child_head(model,records)
    assert summary['preferences']==1 and summary['final_loss']<summary['initial_loss']
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()
               if not k.startswith('child_transfer_'))
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('child_transfer_head.'))
    assert model.cmax_value_head is None


def test_child_search_keeps_current_and_best_when_b32_deadlocks_and_reuses_full_plan(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    experiment,c,_,model=_feedback_fixture()
    model.requires_grad_(False).enable_child_transfer_head()
    solution,bits,state=_child_whole_fixture()
    result=c.JointCompletionResult(solution,bits,(),0.,oriented_state=state)
    transfers=experiment.lns.child_transfer_candidates(solution,bits,'a::whole',CONFIG,destination_robot=1)[:1]
    context=dict(started=time.perf_counter(),space=None,embeddings=None,seen={},blockers={},rng=random.Random(1),
        result=result,schedule=ScheduleResult(ScheduleStatus.FEASIBLE,cmax=100.,robot_completion=(100.,0.,0.,0.)),
        initial_attempts=1,initial_status='FEASIBLE',initial_cmax_s=100.,construct_s=0.,last_status='FEASIBLE',failure='')
    context['counts']=dict(reconstruction_attempts=0,reconstructions=0,accepted=0,b32_calls=1,cache_hits=0,
        deadlocks=0,numeric_failures=0,incomplete=0,late_candidates=0)
    monkeypatch.setattr(experiment,'child_initial_state',lambda *a:context)
    monkeypatch.setattr(experiment.lns,'select_feedback_child',lambda *a,**kw:'a::whole')
    def generate(current,*args,**kwargs):
        assert current is solution
        return transfers
    monkeypatch.setattr(experiment.lns,'child_transfer_candidates',generate)
    monkeypatch.setattr(c,'child_transfer_inputs',lambda *a:torch.zeros((1,model.child_transfer_mean.numel())))
    calls=[]
    monkeypatch.setattr(experiment,'b32',lambda *a,**kw:calls.append(1) or ScheduleResult(ScheduleStatus.DEADLOCK))
    row=experiment.run_child_feedback_gnn(solution.parents,model,1,attempts=3)
    assert row['status']=='FEASIBLE' and row['cmax_s']==100. and row['accepted']==0
    assert row['reconstruction_attempts']==row['reconstructions']==3
    assert row['b32_calls']==2 and row['cache_hits']==2 and len(calls)==1


def test_feedback_joint_trajectory_preserves_coverage_legality_and_replays_exact_logits():
    from collections import Counter
    from mrta_reference.geometry import robot_is_eligible
    experiment, c, parents, model = _feedback_fixture()
    space = c.parent_assignment_space(parents, CONFIG)
    trace, observed = [], []
    result = c.construct_vnext_gnn(parents, model, CONFIG, space=space,
        step_observer=lambda state, selected, plans, scores: observed.append(scores.detach().clone()),
        policy_observer=lambda inputs, choice: trace.append((inputs, choice)))
    assert result.solution is not None and len(trace) == len(parents)
    blocks = block_map(result.solution, CONFIG)
    assert Counter(b for route in result.solution.routes for b in route.block_ids) == Counter(blocks.keys())
    for robot, (route, directions) in enumerate(zip(result.solution.routes, result.directions)):
        assert len(route.block_ids) == len(directions)
        assert all(d in (0, 1) and robot_is_eligible(blocks[b], robot, CONFIG) for b, d in zip(route.block_ids, directions))
    embeddings = model.encode_static(space.graph)
    for logits, (inputs, choice) in zip(observed, trace):
        assert torch.allclose(model.feedback_scores(space, embeddings, inputs), logits, atol=1e-6)
        assert choice == int(logits.argmax())
    before = {key: value.clone() for key, value in model.state_dict().items()}
    loss, norm = experiment.feedback_update(model, space, trace, .2,
                                           torch.optim.AdamW(model.parameters(), lr=1e-3))
    assert math.isfinite(loss) and math.isfinite(norm)
    for prefix in ('layer1.', 'policy.', 'insertion_head.'):
        assert any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items() if key.startswith(prefix))
    assert model.cmax_value_head is None


def test_feedback_sampling_uses_all_legal_joint_candidates():
    experiment, c, parents, model = _feedback_fixture()
    class LastChoice:
        def choices(self, population, *, weights):
            assert len(population) == len(weights) and all(w >= 0. for w in weights)
            return [population[-1]]
    trace = []
    result = c.construct_vnext_gnn(parents, model, CONFIG, rng=LastChoice(),
        policy_observer=lambda inputs, choice: trace.append((len(inputs['indices']), choice)))
    assert result.solution is not None
    assert all(choice == count - 1 for count, choice in trace)
    assert any(count > 3 for count, _ in trace)


def test_feedback_shortlist_keeps_robot_and_mode_representatives_without_outcome_penalties():
    from mrta_search.lns import oriented_route_state
    experiment, c, parents, model = _feedback_fixture()
    space = c.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    state = oriented_route_state(((),) * 4, ((),) * 4, {}, CONFIG)
    indices, features = model.assignment_features(space, embeddings, state, {}, set(space.parent_ids), CONFIG)
    chosen = c.feedback_shortlist(space, indices, torch.arange(len(indices), dtype=torch.float32))
    available = [space.actions[i] for i in indices.tolist()]
    kept = [space.actions[i] for i in chosen]
    assert {r for a in kept for r in a.robots} == {r for a in available for r in a.robots}
    assert {a.pattern.kind for a in kept} == {a.pattern.kind for a in available}
    assert len(chosen) == min(12, len(indices)) and c.COARSE_ASSIGNMENT_LIMIT == 8
    assert len({a.parent_id for a in kept}) > 1
    assert any(a.parent_id == b.parent_id and a.robots != b.robots for a in kept for b in kept)


def test_feedback_parent_removal_closes_siblings_and_freezes_unremoved_directions():
    from test_gnn_schedule_blocker_pilot import _selection_fixture
    solution, schedule, event, config = _selection_fixture()
    experiment, c, _, model = _feedback_fixture()
    space = c.parent_assignment_space(solution.parents, config)
    blocks = block_map(solution, config)
    bits = ((1,), (0, 1), (0,), (1,))
    state = experiment.lns.oriented_route_state(tuple(r.block_ids for r in solution.routes), bits, blocks, config)
    result = c.JointCompletionResult(solution, bits, (), 0., oriented_state=state)
    feedback = experiment.lns.schedule_feedback_summary(solution, schedule, (event,), config)
    feedback['scores'] = dict.fromkeys(feedback['scores'], 0.)
    feedback['scores']['a'] = 100.
    partial, selected = experiment.feedback_prefix(result, space, feedback, 1, 1)
    assert 'a' not in selected and all(not b.startswith('a::') for route in partial.routes for b in route)
    repaired = c.construct_vnext_gnn(solution.parents, model, config, space=space, state=partial,
                                      selected=selected, feedback=feedback)
    assert repaired.solution is not None
    retained = {b for route in partial.routes for b in route}
    for old_route, old_bits, route, new_bits in zip(partial.routes, partial.directions,
                                                   repaired.oriented_state.routes, repaired.directions):
        assert tuple((b, d) for b, d in zip(route, new_bits) if b in retained) == tuple(zip(old_route, old_bits))
    assert feedback['parent_wait']['a'] == 5. and feedback['parent_blocked']['b'] == 5.
    unrelated = experiment.lns.schedule_feedback_summary(solution, schedule, (), config)
    assert unrelated['parent_blocked'] == {}


@pytest.mark.parametrize('status,cmax,accepted', [('FEASIBLE', 9., True), ('FEASIBLE', 10., False),
    ('FEASIBLE', 11., False), ('DEADLOCK', None, False), ('NUMERIC_FAILURE', None, False)])
def test_feedback_accepts_only_strict_feasible_cmax_improvements(status, cmax, accepted):
    import run_gnn_weld_layout as experiment
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    incumbent = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10.)
    candidate = ScheduleResult(ScheduleStatus(status), cmax=cmax)
    assert experiment.feedback_improves(candidate, incumbent) is accepted
    if status == 'NUMERIC_FAILURE':
        assert experiment.feedback_reward(candidate, 10.) is None
    elif status == 'DEADLOCK':
        assert experiment.feedback_reward(candidate, 10.) == -5.


def test_feedback_complete_schedule_and_blocker_evidence_are_evaluated_once(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    calls = []
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10.)
    monkeypatch.setattr(experiment, 'b32', lambda result, **kwargs: calls.append(result) or schedule)
    seen, blockers = {}, {}
    assert experiment.evaluate_once(result, seen, blocker_cache=blockers) == (schedule, True)
    assert experiment.evaluate_once(result, seen, blocker_cache=blockers) == (schedule, False)
    assert len(calls) == len(blockers) == 1


@pytest.mark.parametrize('status,cmax', [('DEADLOCK', None), ('NUMERIC_FAILURE', None),
                                      ('FEASIBLE', 10.), ('FEASIBLE', 11.)])
def test_feedback_search_retains_best_after_failed_or_nonimproving_repair(monkeypatch, status, cmax):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    clock = [0.]
    constructions = []
    def construct(*args, **kwargs):
        clock[0] += 1. if not constructions else 2.
        constructions.append(1)
        return result
    outcomes = [ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10.)]
    def evaluate(*args, **kwargs):
        schedule = outcomes[0] if len(constructions) == 1 else ScheduleResult(ScheduleStatus(status), cmax=cmax)
        return schedule, True
    monkeypatch.setattr(experiment.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', evaluate)
    row = experiment.run_feedback_gnn(parents, model, 1, budget=10.)
    assert row['status'] == 'FEASIBLE' and row['cmax_s'] == row['initial_cmax_s'] == 10.
    assert row['reconstructions'] > 0
    assert experiment.json.loads(row['diagnostic'])['best_improvements'] == 0
    if status == 'FEASIBLE' and cmax == 10.:
        assert row['accepted'] == row['reconstructions']
    assert row['b32_calls'] == len(constructions) and row['runtime_s'] <= 10.


def test_feedback_continuation_uses_packed_120_train_and_retains_original_validation(monkeypatch):
    import copy
    from types import SimpleNamespace
    experiment, c, _, model = _feedback_fixture()
    parent = ParentWeld('p', (1., 9.), (2., 9.))
    base = [(f'train_{i + 1:02d}', (parent,) * (94 if i < 29 else 93)) for i in range(40)]
    expanded = base + [(f'train_{i + 41:02d}', (parent,) * (94 if i < 78 else 93)) for i in range(80)]
    validation = [(f'validation_{i + 1:02d}', (parent,) * (94 if i < 6 else 93)) for i in range(10)]
    calls, trained, stored = [], [], []
    def load(split):
        calls.append(split)
        return dict(train=base, train_expanded=expanded, validation=validation)[split]
    weights = copy.deepcopy(model.state_dict())
    saved = dict(state_dict=weights, last_state_dict=weights, selected_epoch=0, completed_epoch=1,
        training_complete=True, training_layouts=[name for name, _ in base],
        history=[dict(epoch=0), dict(epoch=1)], test_eligible=False)
    reference = [dict(stage='validation', epoch=0, layout_id=name, seed=experiment.SEED,
                      status='FEASIBLE', cmax_s=100., runtime_s=119.) for name, _ in validation]
    monkeypatch.setattr(experiment, 'load_layouts', load)
    monkeypatch.setattr(experiment, 'FEEDBACK_MODEL', SimpleNamespace(exists=lambda: True))
    monkeypatch.setattr(experiment, 'FEEDBACK_TEST_CSV', SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(experiment.torch, 'load', lambda *a, **kw: saved)
    monkeypatch.setattr(experiment.torch, 'save', lambda state, path: stored.append(copy.deepcopy(state)))
    monkeypatch.setattr(experiment, 'feedback_model', lambda *a: c.VNextGNN(16, 4,
        workload_lookahead=True, joint_actions=True, schedule_feedback=True))
    monkeypatch.setattr(experiment, 'read_rows', lambda path: reference)
    monkeypatch.setattr(experiment, 'feedback_write', lambda *a: None)
    def train(model, baseline, layout, epoch, optimizer, **kwargs):
        trained.append((layout[0], epoch, kwargs['training_stage']))
        return []
    monkeypatch.setattr(experiment, 'feedback_train_layout', train)
    monkeypatch.setattr(experiment, 'feedback_evaluate', lambda *a, **kw:
                        [dict(row, epoch=2, cmax_s=110.) for row in reference])
    result = experiment.train_feedback(1, 2, expanded=True)
    assert calls == ['train', 'train_expanded', 'validation']
    assert len(trained) == 120 and len({name for name, _, _ in trained}) == 120
    assert all(epoch == 2 and stage == 'train_expanded' for _, epoch, stage in trained)
    assert result['history'][:2] == [dict(epoch=0), dict(epoch=1)]
    assert result['history'][-1]['training_parents'] == result['training_parents'] == 11267
    assert result['selected_epoch'] == 0 and result['training_complete']
    assert 'forty_layout_state_dict' in stored[-1]


def test_feedback_late_initial_schedule_is_not_reported_as_timely_feasible(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    clock = [0.]
    def construct(*args, **kwargs):
        clock[0] += 1.
        return result
    def evaluate(*args, **kwargs):
        clock[0] += 5.
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10.), True
    monkeypatch.setattr(experiment.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', evaluate)
    row = experiment.run_feedback_gnn(parents, model, 1, budget=5.)
    assert row['status'] == 'TIME_LIMIT' and row['cmax_s'] is None
    assert row['b32_calls'] == row['late_candidates'] == 1


def test_feedback_fixed_reconstruction_count_finishes_after_120_seconds(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    clock = [0.]
    def construct(*args, **kwargs):
        clock[0] += 60.
        return result
    monkeypatch.setattr(experiment.time, 'perf_counter', lambda: clock[0])
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', lambda *a, **kw:
                        (ScheduleResult(ScheduleStatus.FEASIBLE, cmax=10.), True))
    row = experiment.run_feedback_gnn(parents, model, 1, budget=None, reconstruction_limit=3)
    assert row['reconstruction_attempts'] == row['reconstructions'] == 3
    assert row['runtime_s'] == 240. and row['status'] == 'FEASIBLE'
    assert row['b32_calls'] == 4


def test_feedback_full_validation_uses_only_selected_weights_and_leaves_checkpoint_unchanged(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment
    weights = {i: {'weight': torch.tensor(float(i))} for i in (0, 2, 3)}
    saved = dict(training_complete=True, initial_state_dict=weights[0], state_dict=weights[2],
        last_state_dict=weights[3], completed_epoch=3, selected_epoch=2, round_state_dicts={2: weights[2]})
    reference = [dict(layout_id=f'validation_{i:02d}', method='GNN_WIDTH1', seed=experiment.SEED,
        status='FEASIBLE' if i <= 6 else 'DEADLOCK', cmax_s=110. if i <= 6 else '', runtime_s=10.)
        for i in range(1, 11)]
    evaluated, stored, loaded = [], [], []
    def load(split):
        loaded.append(split)
        return [(r['layout_id'], ()) for r in reference]
    def evaluate(model, layouts, epoch, workers, **kwargs):
        evaluated.append((epoch, kwargs['reconstruction_limit'], model))
        return [dict(r, status='DEADLOCK' if epoch == 3 and i == 9 else 'FEASIBLE',
                     cmax_s=100. if epoch == 0 else (105. if epoch == 2 else ('' if i == 9 else 90.)))
                for i, r in enumerate(reference)]
    monkeypatch.setattr(experiment, 'FEEDBACK_TEST_CSV', SimpleNamespace(exists=lambda: False))
    monkeypatch.setattr(experiment.torch, 'load', lambda *a, **kw: saved)
    monkeypatch.setattr(experiment.torch, 'save', lambda state, path: stored.append(state))
    monkeypatch.setattr(experiment, 'feedback_model', lambda state: SimpleNamespace(eval=lambda: state))
    monkeypatch.setattr(experiment, 'feedback_evaluate', evaluate)
    monkeypatch.setattr(experiment, 'load_layouts', load)
    monkeypatch.setattr(experiment, 'read_rows', lambda path: reference)
    monkeypatch.setattr(experiment, 'summarize_feedback', lambda **kwargs: dict(split=kwargs['split']))
    experiment.validate_feedback_complete(1)
    assert evaluated == [(2, 500, weights[2])] and loaded == ['validation']
    assert stored == [] and saved['state_dict'] is weights[2]


@pytest.mark.parametrize('status,cmax,draw,expected', [
    ('FEASIBLE', 99., .99, True), ('FEASIBLE', 100., .99, True),
    ('FEASIBLE', 101., .3, True), ('FEASIBLE', 101., .4, False),
    ('DEADLOCK', None, 0., False), ('NUMERIC_FAILURE', None, 0., False)])
def test_feedback_sa_uses_probability_and_rejects_failed_candidates(status, cmax, draw, expected):
    from types import SimpleNamespace
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    import run_gnn_weld_layout as experiment
    current = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100.)
    candidate = ScheduleResult(ScheduleStatus(status), cmax=cmax)
    assert experiment.feedback_accepts(candidate, current, 1., SimpleNamespace(random=lambda: draw)) is expected
    assert not experiment.feedback_accepts(None, current, 1., SimpleNamespace(random=lambda: 0.))


def test_feedback_sa_searches_from_accepted_current_and_returns_global_best(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    candidates = [replace(result, inference_seconds=float(i)) for i in range(5)]
    # Candidate identity is observable independently of the mocked objective.
    completions = [100., 100.1, 99., None, 101.]
    built, evaluated, prefixes, temperatures = [], [], [], []
    class AcceptUphill:
        def __init__(self, seed): pass
        def randrange(self, stop): return 1
        def random(self): return 0.
    def construct(*args, **kwargs):
        answer = candidates[len(built)]
        built.append(answer)
        return answer
    def evaluate(candidate, *args, **kwargs):
        value = completions[len(evaluated)]
        evaluated.append(candidate)
        status = ScheduleStatus.DEADLOCK if value is None else ScheduleStatus.FEASIBLE
        return ScheduleResult(status, cmax=value), True
    def prefix(current, space, feedback, q, seed):
        prefixes.append(current)
        return result.oriented_state, {}
    original_accepts = experiment.feedback_accepts
    def accept(candidate, current, temperature, rng):
        temperatures.append(temperature)
        return original_accepts(candidate, current, temperature, rng)
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', evaluate)
    monkeypatch.setattr(experiment, 'feedback_prefix', prefix)
    monkeypatch.setattr(experiment, 'feedback_accepts', accept)
    monkeypatch.setattr(experiment, 'Random', AcceptUphill)
    row = experiment.run_feedback_gnn(parents, model, 1, reconstruction_limit=4)
    assert prefixes == [candidates[0], candidates[1], candidates[2], candidates[2]]
    assert row['cmax_s'] == 99. and row['accepted'] == 3 and row['b32_calls'] == 5
    detail = experiment.json.loads(row['diagnostic'])
    assert detail['current_cmax_s'] == 101. and detail['accepted_uphill'] == 2
    assert detail['best_improvements'] == 1
    assert temperatures == pytest.approx([.01 * 100. * .01 ** (k / 499.) for k in range(4)])


def test_feedback_all_500_attempts_count_incomplete_and_expand_then_reset(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, _, model = _feedback_fixture()
    parents = tuple(ParentWeld(f'p{i:02d}', (1. + i / 10., 9.), (1.5 + i / 10., 9.)) for i in range(20))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    solution = canonicalize(parents, patterns, {0: tuple(p.parent_id + '::whole' for p in parents)},
                            CONFIG, scope=FORMAL_SCOPE_V2)
    blocks = block_map(solution, CONFIG)
    directions = ((0,) * len(parents), (), (), ())
    state = experiment.lns.oriented_route_state(tuple(r.block_ids for r in solution.routes), directions, blocks, CONFIG)
    result = c.JointCompletionResult(solution, directions, (), 0., oriented_state=state)
    incomplete = c.JointCompletionResult(None, None, (), 0., rejection_reason='NO_COMPLETE_PARENT_ACTION')
    built, evaluated, sizes, temperatures = [], [], [], []
    def construct(*args, **kwargs):
        built.append(1)
        return result if len(built) in (1, 28) else incomplete
    def evaluate(candidate, *args, **kwargs):
        assert candidate.solution is not None
        evaluated.append(1)
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100. if len(evaluated) == 1 else 99.), True
    def prefix(current, space, feedback, q, seed):
        sizes.append(q)
        return state, {}
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', evaluate)
    monkeypatch.setattr(experiment, 'feedback_prefix', prefix)
    original_accepts = experiment.feedback_accepts
    def accept(candidate, current, temperature, rng):
        temperatures.append(temperature)
        return original_accepts(candidate, current, temperature, rng)
    monkeypatch.setattr(experiment, 'feedback_accepts', accept)
    row = experiment.run_feedback_gnn(parents, model, 7)
    assert len(built) == 501 and len(evaluated) == row['b32_calls'] == 2
    assert row['reconstruction_attempts'] == 500 and row['reconstructions'] == 1
    assert row['incomplete'] == 499 and row['cmax_s'] == 99.
    assert sizes[:25] == [8] * 25 and sizes[25:27] == [12, 12]
    assert sizes[27:52] == [8] * 25 and sizes[52:] == [12] * 448
    assert temperatures[0] == 1. and temperatures[-1] == pytest.approx(.01)


def test_feedback_critical_quota_uses_real_completion_and_fills_insufficient_routes():
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, _, _ = _feedback_fixture()
    parents = tuple(ParentWeld(f'p{i:02d}', (2., 9. if i < 14 else 2.), (3., 9. if i < 14 else 2.))
                    for i in range(20))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    routes = {0: tuple(f'p{i:02d}::whole' for i in range(5)),
              1: tuple(f'p{i:02d}::whole' for i in range(5, 14)),
              2: tuple(f'p{i:02d}::whole' for i in range(14, 17)),
              3: tuple(f'p{i:02d}::whole' for i in range(17, 20))}
    solution = canonicalize(parents, patterns, routes, CONFIG, scope=FORMAL_SCOPE_V2)
    schedule = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100., robot_completion=(50., 100., 30., 40.))
    feedback = experiment.lns.schedule_feedback_summary(solution, schedule, (), CONFIG)
    feedback['scores'] = {p.parent_id: float(i + 1) for i, p in enumerate(parents)}
    space = c.parent_assignment_space(parents, CONFIG)
    select = lambda q, seed: experiment.lns.select_feedback_bottleneck(solution, feedback, q=q, seed=seed,
                                                                       assignment_space=space)
    partial = select(10, 12)
    critical = {f'p{i:02d}' for i in range(5, 14)}
    assert len(set(partial.removed_parent_ids) & critical) == 7
    assert partial == select(10, 12)
    assert len(select(19, 12).removed_parent_ids) == 19
    assert all(b.split('::')[0] not in partial.removed_parent_ids for route in partial.routes for b in route)
    with pytest.raises(ValueError, match='invalid parent removal'):
        select(0, 12)
    singleton = canonicalize(parents[:1], patterns[:1], {0: ('p00::whole',)}, CONFIG, scope=FORMAL_SCOPE_V2)
    empty = experiment.lns.select_feedback_bottleneck(singleton, {}, q=0, seed=12)
    assert empty.removed_parent_ids == () and empty.routes[0] == ('p00::whole',)


def test_feedback_validation_three_seeds_resumes_only_new_source(monkeypatch):
    from types import SimpleNamespace
    experiment, _, _, model = _feedback_fixture()
    names = [f'validation_{i:02d}' for i in range(1, 11)]
    def row(name, seed, source):
        return dict(layout_id=name, seed=seed, stage='full_validation', epoch=0, source=source)
    old = [row(name, seed, 'COMPLETE_100_RECONSTRUCTIONS') for name in names for seed in experiment.TEST_SEEDS]
    current = [row(names[0], experiment.TEST_SEEDS[0], experiment.FEEDBACK_FULL_SOURCE)]
    jobs, written = [], []
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def submit(self, fn, job):
            jobs.append(job)
            return SimpleNamespace(result=lambda: row(job[0], job[2], experiment.FEEDBACK_FULL_SOURCE))
    monkeypatch.setattr(experiment, 'read_rows', lambda path: old + current)
    monkeypatch.setattr(experiment, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(experiment, 'as_completed', iter)
    monkeypatch.setattr(experiment, 'feedback_write', lambda path, rows: written.extend(rows))
    rows = experiment.feedback_evaluate(model, [(name, ()) for name in names], 0, 4, reconstruction_limit=500)
    assert len(rows) == 30 and len(jobs) == len(written) == 29
    assert {job[2] for job in jobs} == set(experiment.TEST_SEEDS) and all(job[-1] == 500 for job in jobs)
    assert all(r['source'] == experiment.FEEDBACK_FULL_SOURCE for r in rows)


@pytest.mark.parametrize('failure,eligible', [(None, True), ('feasibility', False),
    ('wag_cmax', False), ('hga_cmax', False), ('wag_wins', False), ('wag_layouts', False)])
def test_feedback_test_requires_all_four_validation_conditions(monkeypatch, failure, eligible):
    import json
    import run_gnn_weld_layout as experiment
    rows = [dict(stage='full_validation', epoch=0, source=experiment.FEEDBACK_FULL_SOURCE,
        layout_id=f'validation_{i:02d}', seed=seed, status='FEASIBLE', cmax_s=99., runtime_s=200.,
        initial_status='FEASIBLE', initial_cmax_s=110., reconstruction_attempts=500,
        reconstructions=500, accepted=1, b32_calls=501, cache_hits=0, incomplete=0, deadlocks=0,
        numeric_failures=0) for i in range(1, 11) for seed in experiment.TEST_SEEDS]
    if failure == 'feasibility':
        rows[0] = dict(rows[0], status='DEADLOCK', cmax_s='')
    references = {}
    for method in experiment.BASELINES:
        records = [dict(row, method=method, cmax_s=100., b32_calls=1,
            source=experiment.FORMAL_BASELINE_SOURCES[method],
            iterations=experiment.BASELINE_ITERATIONS[method],
            diagnostic=json.dumps(dict(phase='FULL', run_status='COMPLETED'))) for row in rows]
        for i, record in enumerate(records):
            if failure == ('wag_cmax' if method == 'WAG+VNS' else 'hga_cmax'):
                record['cmax_s'] = 98.
            if method == 'WAG+VNS' and failure == 'wag_wins':
                # Even-sized pairs have a negative median, yet wins == losses.
                record['cmax_s'] = 100. if i < 15 else 98.5
            if method == 'WAG+VNS' and failure == 'wag_layouts':
                # Five layouts improve on all seeds, five regress on two seeds.
                record['cmax_s'] = 100. if i < 15 or i % 3 == 0 else 98.
        references[experiment.BASELINES[method][2]] = records
    old = [dict(row, source='COMPLETE_100_RECONSTRUCTIONS', cmax_s=1.) for row in rows]
    def read(path):
        return rows + old if path == experiment.FEEDBACK_TRAIN_CSV else references.get(path.name, [])
    monkeypatch.setattr(experiment, 'read_rows', read)
    summary = experiment.summarize_feedback(split='validation')
    assert summary['gnn']['runs'] == 30 and summary['test_eligible'] is eligible
    if not eligible:
        loaded = []
        monkeypatch.setattr(experiment, 'load_layouts', lambda split: loaded.append(split))
        monkeypatch.setattr(experiment.torch, 'load', lambda *a, **kw: pytest.fail('TEST loaded the model after failure'))
        experiment.test_feedback(4)
        assert loaded == []


def test_feedback_initial_deadlock_gets_at_most_two_gnn_sampling_retries(monkeypatch):
    from mrta_reference.model import ScheduleResult, ScheduleStatus
    experiment, c, parents, model = _feedback_fixture()
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    sampling = []
    def construct(*args, **kwargs):
        sampling.append(kwargs.get('rng'))
        return result
    monkeypatch.setattr(c, 'construct_vnext_gnn', construct)
    monkeypatch.setattr(experiment, 'evaluate_once', lambda *a, **kw:
        (ScheduleResult(ScheduleStatus.DEADLOCK), True))
    row = experiment.run_feedback_gnn(parents, model, 12)
    assert len(sampling) == row['b32_calls'] == 3 and sampling[0] is None
    assert sampling[1] is sampling[2] and sampling[1] is not None
    assert row['status'] == 'DEADLOCK' and row['cmax_s'] is None and row['reconstruction_attempts'] == 0


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


def test_fast_assignment_decode_preserves_ownership_and_passes_explicit_directions_to_b32(monkeypatch):
    from mrta_reference import scheduler
    from mrta_search import gnn_constructor as c
    from run_joint_oriented_repair_pilot import schedule_explicit

    parents = (ParentWeld('left', (1., 9.), (3., 9.)),
               ParentWeld('right', (2., 9.), (18., 9.)),
               ParentWeld('lower', (1., 2.), (4., 2.)))
    owners = {'left': 0, 'right': 1, 'lower': 2}
    labels = tuple(c.ParentAssignment(p.parent_id, p, (owners[p.parent_id],))
                   for p in c.fixed_split_patterns(parents, CONFIG))
    decoded = c.decode_parent_assignments(parents, labels, CONFIG)
    reversed_input = c.decode_parent_assignments(parents[::-1], labels[::-1], CONFIG)
    assert decoded.solution == reversed_input.solution
    assert decoded.directions == reversed_input.directions
    assert decoded.actions[0].block_id == 'right::whole'
    assert c.parent_assignment_targets(decoded.solution, CONFIG) == labels
    assert decoded.directions[1] == (1,)

    def unexpected(*args, **kwargs):
        pytest.fail('an explicit direction must never invoke route direction optimization')

    monkeypatch.setattr(scheduler, 'optimize_directions', unexpected)
    schedule = schedule_explicit(decoded.solution, decoded.directions, CONFIG).schedule
    assert schedule.status is ScheduleStatus.FEASIBLE
    assert schedule.directions == decoded.directions
    assert len([op for op in schedule.operations if op.kind is OperationKind.WELD]) == len(parents)


def test_coordination_aware_assignment_selects_direction_by_nominal_exposure():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import nominal_coordination_features
    from run_joint_oriented_repair_pilot import schedule_explicit

    parents = (ParentWeld('left', (0., 9.), (8., 9.)), ParentWeld('right', (9., 9.), (3., 9.)))
    labels = tuple(c.ParentAssignment(p.parent_id, p, (r,))
                   for r, p in enumerate(c.fixed_split_patterns(parents, CONFIG)))
    original = c.decode_parent_assignments(parents, labels, CONFIG)
    coordinated = c.decode_parent_assignments(parents, labels, CONFIG, coordination_aware=True)
    assert c.parent_assignment_targets(coordinated.solution, CONFIG) == labels
    assert original.solution.routes == coordinated.solution.routes
    assert original.directions == ((0,), (0,), (), ())
    assert coordinated.directions == ((0,), (1,), (), ())
    old = nominal_coordination_features(original.solution, original.directions, CONFIG)
    new = nominal_coordination_features(coordinated.solution, coordinated.directions, CONFIG)
    assert old['same_rail_order_seconds'][0] > 0
    assert new['same_rail_order_seconds'] == (0.,) * 4
    assert new['exposure_score'] < old['exposure_score']
    assert schedule_explicit(coordinated.solution, coordinated.directions, CONFIG).schedule.status is ScheduleStatus.FEASIBLE


@pytest.mark.parametrize('coordination_aware', (False, True))
def test_fast_assignment_decode_keeps_split_siblings_and_requires_complete_valid_ownership(coordination_aware):
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.model import Rail
    from mrta_search import gnn_constructor as c

    long = ParentWeld('long', (2., 5.7), (10., 6.3))
    other = ParentWeld('other', (3., 9.), (4., 9.))
    parents = (long, other)
    catalog = build_legal_pattern_catalog(parents, CONFIG, FORMAL_SCOPE_V2)
    split = next(p for p in catalog['long'] if p.kind is SplitKind.Y_SPLIT)
    split_label = c.ParentAssignment('long', split, (0, 0))
    separator = c.ParentAssignment('other', catalog['other'][0], (0,))
    decoded = c.decode_parent_assignments(parents, (split_label, separator), CONFIG,
                                          coordination_aware=coordination_aware)
    route = decoded.solution.routes[0].block_ids
    assert set(route) == {'long::0', 'long::1', 'other::whole'}
    assert route[1] == 'other::whole'
    assert c.parent_assignment_targets(decoded.solution, CONFIG) == (split_label, separator)

    with pytest.raises(ValueError, match='separating block'):
        c.decode_parent_assignments((long,), (split_label,), CONFIG, coordination_aware=coordination_aware)
    with pytest.raises(ValueError, match='one assignment per parent'):
        c.decode_parent_assignments(parents, (separator, separator), CONFIG)
    with pytest.raises(ValueError, match='one assignment per parent'):
        c.decode_parent_assignments(parents, (split_label,), CONFIG)
    x_split = next(p for p in catalog['long'] if p.kind is SplitKind.X_SPLIT)
    x_owners = (0, 1) if x_split.rail is Rail.UPPER else (2, 3)
    x_labels = (c.ParentAssignment('long', x_split, x_owners), separator)
    x_decoded = c.decode_parent_assignments(parents, x_labels, CONFIG, coordination_aware=coordination_aware)
    assert c.parent_assignment_targets(x_decoded.solution, CONFIG) == x_labels
    with pytest.raises(ValueError, match='illegal assignment'):
        c.decode_parent_assignments(parents, (c.ParentAssignment('long', x_split, x_owners[::-1]), separator),
                                    CONFIG, coordination_aware=coordination_aware)


def test_joint_assignment_head_scores_all_and_only_eligible_tuples_with_one_static_encoding(monkeypatch):
    from itertools import product
    from mrta_reference.geometry import blocks_for_pattern, build_legal_pattern_catalog
    from mrta_reference.model import Rail
    from mrta_search import gnn_constructor as c, lns

    parents = (ParentWeld('optional', (14., 6.2), (2., 5.8)),
               ParentWeld('mandatory', (1., 9.), (1., 3.)))
    catalog = build_legal_pattern_catalog(parents, CONFIG, FORMAL_SCOPE_V2)
    expected = set()
    for parent in parents:
        for pattern in catalog[parent.parent_id]:
            blocks = blocks_for_pattern(parent, pattern, CONFIG)
            robots = ((0, 1),) if pattern.rail is Rail.UPPER else ((2, 3),) if pattern.rail is Rail.LOWER else product(
                *(tuple(r for r in range(4) if robot_is_eligible(b, r, CONFIG)) for b in blocks))
            expected.update(c.ParentAssignment(parent.parent_id, pattern, owners) for owners in robots)

    def unexpected(*args, **kwargs):
        pytest.fail('joint ownership scoring must never generate an insertion state or re-encode the graph')

    monkeypatch.setattr(lns, 'joint_block_candidates', unexpected)
    space = c.parent_assignment_space(parents, CONFIG)
    assert set(space.actions) == expected
    assert len(space.actions) == len(expected)
    assert {a.pattern.kind for a in space.actions} == set(SplitKind)
    model = c.ParentAssignmentGNN()
    embeddings = model.encode_static(space.graph)
    assert embeddings.shape == (len(space.graph.node_features), 64)
    assert model.layer1.heads == model.layer2.heads == 4
    monkeypatch.setattr(model, 'encode_static', unexpected)
    empty = ((),) * 4
    state = lns.oriented_route_state(empty, empty, {}, CONFIG)
    ids, scores = model.assignment_logits(space, embeddings, state, {}, set(catalog), CONFIG)
    assert ids.tolist() == list(range(len(expected)))
    assert scores.shape == (len(expected),) and torch.isfinite(scores).all()
    subset, logits = model.assignment_logits(space, embeddings, state, {}, {'mandatory'}, CONFIG)
    assert all(space.actions[i].parent_id == 'mandatory' for i in subset.tolist())
    assert torch.allclose(logits, scores[subset])
    assert model.assignment_logits(space, embeddings, state, {}, set(), CONFIG)[1].numel() == 0

    placed = blocks_for_pattern(parents[0], catalog['optional'][0], CONFIG)[0]
    blocks = {placed.block_id: placed}
    occupied = lns.oriented_route_state(((placed.block_id,), (), (), ()), ((0,), (), (), ()), blocks, CONFIG)
    _, updated = model.assignment_logits(space, embeddings, occupied, blocks, {'mandatory'}, CONFIG)
    assert not torch.allclose(updated, logits)


def test_coarse_to_fine_gnn_uses_both_heads_to_assign_and_direct(monkeypatch):
    from mrta_search import gnn_constructor as c
    from run_joint_oriented_repair_pilot import schedule_explicit

    parents = (ParentWeld('p', (2., 9.), (6., 9.)),)
    model = c.CoarseToFineGNN()
    coarse_width = model.policy[0].in_features
    direction_column = coarse_width + 4 * 64 + 1
    monkeypatch.setattr(model.policy, 'forward', lambda rows: 100. * rows[:, -12:-11])
    monkeypatch.setattr(model.insertion_head, 'forward', lambda rows: rows[:, direction_column:direction_column + 1])
    trace = []

    def observe(state, blocks, selected, pending, nominal, ids, rows, order, plans, fine, logits, best):
        assert state.routes == ((),) * 4 and selected == {} and pending == {'p'}
        assert len(order) <= c.COARSE_ASSIGNMENT_LIMIT
        assert len(plans) <= c.COARSE_ASSIGNMENT_LIMIT * c.FINE_INSERTION_LIMIT
        trace.append(plans[best])

    result = c.construct_coarse_to_fine_gnn(parents, model, CONFIG, observer=observe)
    assert len(trace) == 1 and trace[0].actions == result.actions
    assert c.parent_assignment_targets(result.solution, CONFIG)[0].robots == (1,)
    assert result.directions == ((), (1,), (), ())
    assert schedule_explicit(result.solution, result.directions, CONFIG).schedule.status is ScheduleStatus.FEASIBLE


def test_coarse_to_fine_never_returns_a_partial_split(monkeypatch):
    from mrta_search import gnn_constructor as c, lns

    parents = (ParentWeld('cross', (2., 4.), (2., 8.)),)
    space = c.parent_assignment_space(parents, CONFIG)
    model = c.CoarseToFineGNN()
    result = c.construct_coarse_to_fine_gnn(parents, model, CONFIG, space=space)
    assert len(result.actions) == 2
    labels = c.parent_assignment_targets(result.solution, CONFIG)
    assert labels[0].pattern.kind is SplitKind.Y_SPLIT
    assert labels[0] in space.action_index
    original = lns.joint_block_candidates

    def no_second(state, block, *args, **kwargs):
        return () if block.block_id.endswith('::1') else original(state, block, *args, **kwargs)

    monkeypatch.setattr(lns, 'joint_block_candidates', no_second)
    failure = c.construct_coarse_to_fine_gnn(parents, model, CONFIG, space=space)
    assert failure.solution is None and failure.actions == ()


def test_coarse_to_fine_owner_slots_keep_capacity_and_offer_different_robots():
    from mrta_search import gnn_constructor as c

    parents = tuple(ParentWeld(f'p{i}', (i, 9.), (i + .6, 9.)) for i in range(10))
    space = c.parent_assignment_space(parents, CONFIG)
    ids = torch.arange(len(space.actions))
    logits = torch.tensor([-int(a.parent_id[1:]) - 100 * a.robots[0] for a in space.actions], dtype=torch.float32)
    original = [space.actions[i] for i in c.shortlist_assignments(space, ids, logits)]
    expanded = [space.actions[i] for i in c.shortlist_assignments(space, ids, logits, owner_alternatives=True)]
    assert len(original) == len(expanded) == c.COARSE_ASSIGNMENT_LIMIT
    assert all(a.robots == (0,) for a in original)
    assert {(a.parent_id, a.robots) for a in expanded if a.robots != (0,)} == {('p0', (1,)), ('p1', (1,))}


def test_vnext_expands_one_learned_assignment_and_directs_without_exact_scans(monkeypatch):
    from mrta_search import gnn_constructor as c, lns
    from run_joint_oriented_repair_pilot import schedule_explicit

    parents = (ParentWeld('p', (2., 9.), (2.6, 9.)),)
    model = c.VNextGNN()
    direction = 4 * 64 + 5
    monkeypatch.setattr(model.policy, 'forward', lambda rows: 100. * rows[:, -12:-11])
    monkeypatch.setattr(model.insertion_head, 'forward', lambda rows: rows[:, direction:direction + 1])
    original = lns.joint_block_candidates
    expanded = []

    def record(state, block, pattern, blocks, config, **kwargs):
        expanded.append(kwargs['robot_id'])
        return original(state, block, pattern, blocks, config, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail('VNext inference invoked the exact B teacher')

    monkeypatch.setattr(lns, 'joint_block_candidates', record)
    monkeypatch.setattr(lns.NominalRouteCoordination, 'from_state', forbidden)
    result = c.construct_vnext_gnn(parents, model, CONFIG)
    assert expanded == [1]
    assert c.parent_assignment_targets(result.solution, CONFIG)[0].robots == (1,)
    assert result.directions == ((), (1,), (), ())
    assert schedule_explicit(result.solution, result.directions, CONFIG).schedule.status is ScheduleStatus.FEASIBLE


def test_vnext_delays_same_robot_split_and_commits_both_children(monkeypatch):
    from mrta_search import gnn_constructor as c, lns

    parents = (ParentWeld('seam', (2., 5.7), (10., 6.3)),
               ParentWeld('separator', (3., 9.), (3.6, 9.)))
    model = c.VNextGNN()
    original_logits = model.assignment_logits

    def scores(space, embeddings, state, blocks, pending, config):
        indices, _ = original_logits(space, embeddings, state, blocks, pending, config)
        weights = []
        for i in indices.tolist():
            action = space.actions[i]
            target = action.parent_id == 'seam' and action.pattern.kind is SplitKind.Y_SPLIT
            separator = action.parent_id == 'separator' and action.robots == (0,)
            weights.append(2. if target and action.robots == (0, 0) else float(separator))
        return indices, torch.tensor(weights)

    monkeypatch.setattr(model, 'assignment_logits', scores)
    commits = []
    result = c.construct_vnext_gnn(parents, model, CONFIG,
                                   decision_observer=lambda assignment, actions, state: commits.append((assignment, actions)))
    assert [a.parent_id for a, _ in commits] == ['separator', 'seam']
    assert commits[1][0].robots == (0, 0) and len(commits[1][1]) == 2
    assert result.solution.routes[0].block_ids[1] == 'separator::whole'
    assert c.parent_assignment_targets(result.solution, CONFIG)[0].pattern.kind is SplitKind.Y_SPLIT
    original_candidates = lns.joint_block_candidates

    def no_second(state, block, *args, **kwargs):
        return () if block.block_id.endswith('::1') else original_candidates(state, block, *args, **kwargs)

    monkeypatch.setattr(lns, 'joint_block_candidates', no_second)
    failed = c.construct_vnext_gnn(parents, model, CONFIG)
    assert failed.solution is None and [a.block_id for a in failed.actions] == ['separator::whole']


def test_vnext_b_distillation_accepts_near_ties():
    from run_gnn_cmax_finetune import vnext_rank_targets

    target = vnext_rank_targets(((10., 3., 5., 0), (10.2, 3.1, 5.2, 1), (11., 1., 2., 0)))
    assert target[0] == target[1] and target[1] > target[2]
    assert target.sum().item() == pytest.approx(1.)


@pytest.mark.parametrize('reconstruct', (False, True))
def test_complete_gnn_keeps_late_completion_and_evaluates_each_plan_once(monkeypatch, reconstruct):
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment

    solution = _solution()
    clock, builds, schedules = [0.], [], []
    model = SimpleNamespace(encode_static=lambda graph: None)
    monkeypatch.setattr(experiment, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(experiment.gnn, 'parent_assignment_space', lambda *args: SimpleNamespace(graph=None))

    def build(*args, **kwargs):
        assert kwargs.get('deadline') is None
        clock[0] += 90.
        direction = int(bool(builds))
        builds.append(kwargs)
        directions = tuple(tuple(direction for _ in route.block_ids) for route in solution.routes)
        return experiment.gnn.JointCompletionResult(solution, directions, (), 90.)

    def schedule(result):
        schedules.append(result)
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=110. - 10. * len(schedules))

    monkeypatch.setattr(experiment, 'build_joint', build)
    monkeypatch.setattr(experiment, 'b32', schedule)
    monkeypatch.setattr(experiment, 'repair_prefix', lambda *args: (object(), {}))
    row, stats = experiment.complete_gnn(_parents(), model, 1, (60.,), reconstruct=reconstruct)

    assert len(builds) == (3 if reconstruct else 1)
    assert len(schedules) == row['b32_calls'] == (2 if reconstruct else 1)
    assert stats == ((1, 2, 1) if reconstruct else (1, 0, 0))
    assert row['status'] == 'FEASIBLE' and row['cmax_s'] == (90. if reconstruct else 100.)
    assert row['construct_s'] == 90. and row['runtime_s'] == (270. if reconstruct else 90.)
    assert row['cmax_60_s'] is None and row['source'] == experiment.GNN_RESULT_SOURCE


@pytest.mark.parametrize('status', (ScheduleStatus.DEADLOCK, ScheduleStatus.INFEASIBLE))
def test_complete_gnn_reports_actual_schedule_failure(monkeypatch, status):
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment

    result = experiment.gnn.JointCompletionResult(_solution(), ((),) * 4, (), 0.)
    builds = []
    model = SimpleNamespace(encode_static=lambda graph: None)
    monkeypatch.setattr(experiment.gnn, 'parent_assignment_space', lambda *args: SimpleNamespace(graph=None))

    def build(*args, **kwargs):
        builds.append(kwargs)
        return result

    monkeypatch.setattr(experiment, 'build_joint', build)
    monkeypatch.setattr(experiment, 'b32', lambda result: ScheduleResult(status, diagnostics=('blocked',)))
    row, stats = experiment.complete_gnn(_parents(), model, 1, (), reconstruct=True)

    assert row['status'] == status.value and row['cmax_s'] is None
    assert row['diagnostic'] == status.value + ' blocked'
    assert len(builds) == row['b32_calls'] == 1 and stats == (1, 0, 0)


def test_weld_layout_writer_preserves_rows_when_extending_existing_columns():
    import csv
    import io
    from unittest.mock import Mock, mock_open
    import run_gnn_weld_layout as experiment

    fields = [field for field in experiment.RESULT_FIELDS if not field.startswith('native_cmax')]
    original = dict(layout_id='test_01', method='GNN_JOINT', seed='1', cmax_s='100', source='RUN')
    previous = io.StringIO(newline='')
    writer = csv.DictWriter(previous, fieldnames=fields)
    writer.writeheader()
    writer.writerow(original)
    path = Mock()
    path.exists.return_value = True
    path.stat.return_value.st_size = len(previous.getvalue())
    path.open = mock_open(read_data=previous.getvalue())
    latest = dict(original, seed='2', native_cmax_s='90', source=experiment.GNN_RESULT_SOURCE)

    experiment.write_rows(path, [latest], append=True)

    text = ''.join(call.args[0] for call in path.open.return_value.write.call_args_list)
    reader = csv.DictReader(io.StringIO(text, newline=''))
    rows = list(reader)
    assert reader.fieldnames == list(experiment.RESULT_FIELDS)
    assert len(rows) == 2 and all(rows[0][field] == value for field, value in original.items())
    assert rows[0]['native_cmax_s'] == ''
    assert all(rows[1][field] == value for field, value in latest.items())


def _controlled_vnext_beam(monkeypatch, failed_robots=()):
    from types import SimpleNamespace
    from mrta_search import gnn_constructor as c, lns

    parents = (ParentWeld('p0', (2., 9.), (2.6, 9.)), ParentWeld('p1', (4., 9.), (4.6, 9.)))
    model = SimpleNamespace(joint_weights=torch.zeros(6), encode_static=lambda graph: None)
    visits = []

    def proposals(space, embeddings, model, state, blocks, nodes, pending, config):
        assert set(blocks) == set(nodes) == {b for route in state.routes for b in route}
        assert len(pending) + len(blocks) == len(parents)
        visits.append(state)
        first_robot = next((r for r, route in enumerate(state.routes) if 'p0::whole' in route), None)
        if first_robot in failed_robots:
            return (), torch.empty(0)
        parent = min(pending)
        plans = []
        for robot in (0, 1):
            index = next(i for i, a in enumerate(space.actions)
                         if a.parent_id == parent and a.pattern.kind is SplitKind.WHOLE and a.robots == (robot,))
            assignment = space.actions[index]
            child, = space.children[assignment.pattern]
            local = {**blocks, child.block_id: child}
            after, action = lns.joint_block_candidates(state, child, assignment.pattern, local, config,
                                                       robot_id=robot, check_initial_feasibility=False)[0]
            plans.append(c.VNextJointAction(index, after, (action,), torch.empty(0), torch.empty(0)))
        # Raw cumulative logits would keep both R1 branches; normalized scores
        # retain the confident R1 prefix and one R0 prefix.
        scores = [0.2, 0.] if parent == 'p0' else ([100., 100.] if first_robot == 0 else [1000., 990.])
        return tuple(plans), torch.tensor(scores)

    monkeypatch.setattr(c, 'vnext_joint_actions', proposals)
    return parents, model, visits


def test_vnext_width_one_preserves_greedy_and_width_two_ranks_global_log_probabilities(monkeypatch):
    from mrta_search import gnn_constructor as c
    import mrta_reference.scheduler as scheduler

    parents, model, visits = _controlled_vnext_beam(monkeypatch)
    monkeypatch.setattr(scheduler, 'reference_schedule_formal',
                        lambda *a, **k: pytest.fail('B32 evaluated a partial construction state'))
    greedy = c.construct_vnext_gnn(parents, model, CONFIG)
    explicit = c.construct_vnext_gnn(parents, model, CONFIG, beam_width=1)
    assert (greedy.solution, greedy.directions, greedy.actions) == (explicit.solution, explicit.directions, explicit.actions)
    assert not greedy.candidates and greedy.solution.routes[0].block_ids == ('p1::whole', 'p0::whole')
    visits.clear()
    beam = c.construct_vnext_gnn(parents, model, CONFIG, beam_width=2)
    assert len(visits) == 3 and len(beam.candidates) == 2
    first_owners = [next(r for r, route in enumerate(plan.solution.routes) if 'p0::whole' in route.block_ids)
                    for plan in beam.candidates]
    assert first_owners == [1, 0]
    assert all(len(plan.actions) == len(parents) for plan in beam.candidates)


@pytest.mark.parametrize('failed_robots', ((0,), (0, 1)))
def test_vnext_beam_discards_only_the_unextendable_paths(monkeypatch, failed_robots):
    from mrta_search import gnn_constructor as c

    parents, model, _ = _controlled_vnext_beam(monkeypatch, failed_robots)
    result = c.construct_vnext_gnn(parents, model, CONFIG, beam_width=2)
    if failed_robots == (0,):
        assert result.solution is not None and len(result.actions) == 2
        assert all('p0::whole' in p.solution.routes[1].block_ids for p in result.candidates)
    else:
        assert result.solution is None and not result.candidates
        assert 'NO_COMPLETE_PARENT_ACTION' in result.rejection_reason and 'p1' in result.rejection_reason


def test_vnext_beam_deduplicates_convergent_states_but_preserves_directions(monkeypatch):
    from types import SimpleNamespace
    from mrta_search import gnn_constructor as c, lns

    parents = (ParentWeld('p0', (2., 9.), (2.6, 9.)), ParentWeld('p1', (4., 9.), (4.6, 9.)))
    model = SimpleNamespace(joint_weights=torch.zeros(6), encode_static=lambda graph: None)

    def proposals(space, embeddings, model, state, blocks, nodes, pending, config):
        plans = []
        options = [(parent, 0) for parent in sorted(pending)] if len(pending) == 2 else [(min(pending), 0), (min(pending), 1)]
        for parent, direction in options:
            index = next(i for i, a in enumerate(space.actions)
                         if a.parent_id == parent and a.pattern.kind is SplitKind.WHOLE and a.robots == (0,))
            child, = space.children[space.actions[index].pattern]
            local = {**blocks, child.block_id: child}
            route = tuple(sorted(local))
            bits = tuple(direction if b == child.block_id else 0 for b in route)
            after = lns.oriented_route_state((route, (), (), ()), (bits, (), (), ()), local, config)
            action = lns.OrientedInsertionAction(child.block_id, 0, route.index(child.block_id), direction)
            plans.append(c.VNextJointAction(index, after, (action,), torch.empty(0), torch.empty(0)))
        return tuple(plans), torch.tensor([0., -1.]) if len(pending) == 1 else torch.zeros(2)

    monkeypatch.setattr(c, 'vnext_joint_actions', proposals)
    result = c.construct_vnext_gnn(parents, model, CONFIG, beam_width=2)
    assert len(result.candidates) == 2
    a, b = result.candidates
    assert a.solution.patterns == b.solution.patterns and a.solution.routes == b.solution.routes
    assert a.directions != b.directions


def test_vnext_beam_real_joint_actions_preserve_complete_parents_loads_and_directions():
    from mrta_search import gnn_constructor as c

    torch.manual_seed(12)
    parents = (ParentWeld('seam', (2., 5.7), (10., 6.3)), ParentWeld('separator', (3., 9.), (3.6, 9.)))
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True).eval()
    result = c.construct_vnext_gnn(parents, model, CONFIG, beam_width=2)
    assert result.solution is not None and 1 <= len(result.candidates) <= 2
    for candidate in result.candidates:
        blocks = block_map(candidate.solution, CONFIG, scope=FORMAL_SCOPE_V2)
        assert set(blocks) == {a.block_id for a in candidate.actions}
        assert {p.parent_id for p in candidate.solution.patterns} == {p.parent_id for p in parents}
        for robot, route in enumerate(candidate.solution.routes):
            assert len(route.block_ids) == len(candidate.directions[robot])
            assert candidate.oriented_state.loads[robot] == pytest.approx(sum(CONFIG.process_time(blocks[b].length) for b in route.block_ids))
            assert all(robot_is_eligible(blocks[b], robot, CONFIG) for b in route.block_ids)


@pytest.mark.parametrize('outcomes, expected', (
    (((ScheduleStatus.FEASIBLE, 100.), (ScheduleStatus.FEASIBLE, 90.)), 1),
    (((ScheduleStatus.DEADLOCK, None), (ScheduleStatus.FEASIBLE, 120.)), 1),
    (((ScheduleStatus.FEASIBLE, 100.), (ScheduleStatus.DEADLOCK, None)), 0),
    (((ScheduleStatus.DEADLOCK, 123.), (ScheduleStatus.INFEASIBLE, None)), 0),
))
def test_complete_gnn_beam_selects_only_final_b32_results(monkeypatch, outcomes, expected):
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment

    solution = _solution()
    candidates = tuple(experiment.gnn.JointCompletionResult(solution,
                       tuple(tuple(bit for _ in r.block_ids) for r in solution.routes), (), 90.) for bit in (0, 1))
    clock = [0.]
    monkeypatch.setattr(experiment, 'time', SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(experiment.gnn, 'parent_assignment_space', lambda *a: SimpleNamespace(graph=None))
    builds, calls, chosen = [], [], []

    def build(*args, **kwargs):
        assert kwargs['beam_width'] == 2 and kwargs.get('deadline') is None
        clock[0] += 90.
        builds.append(kwargs)
        return replace(candidates[0], candidates=candidates)

    def schedule(result):
        calls.append(result)
        status, cmax = outcomes[len(calls) - 1]
        return ScheduleResult(status, cmax=cmax, diagnostics=('actual failure',))

    monkeypatch.setattr(experiment, 'build_joint', build)
    monkeypatch.setattr(experiment, 'b32', schedule)
    monkeypatch.setattr(experiment, 'repair_prefix', lambda *a: pytest.fail('beam construction invoked reconstruction'))
    row, stats = experiment.complete_gnn(_parents(), SimpleNamespace(encode_static=lambda g: None), 1, (),
                                        reconstruct=False, beam_width=2,
                                        result_observer=lambda result, schedule: chosen.append(result))
    assert len(builds) == 1 and row['b32_calls'] == len(calls) == 2 and stats == (1, 0, 0)
    assert row['runtime_s'] == row['construct_s'] == 90.
    assert chosen == [candidates[expected]]
    assert row['status'] == outcomes[expected][0].value
    assert row['cmax_s'] == (outcomes[expected][1] if outcomes[expected][0] is ScheduleStatus.FEASIBLE else None)
    if row['cmax_s'] is None:
        assert 'DEADLOCK' in row['diagnostic'] and 'INFEASIBLE' in row['diagnostic']


def test_complete_gnn_beam_reuses_identical_complete_plan_evaluation(monkeypatch):
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment

    result = experiment.gnn.JointCompletionResult(_solution(), ((),) * 4, (), 0.)
    monkeypatch.setattr(experiment.gnn, 'parent_assignment_space', lambda *a: SimpleNamespace(graph=None))
    monkeypatch.setattr(experiment, 'build_joint', lambda *a, **k: replace(result, candidates=(result, result)))
    calls = []

    def schedule(result):
        calls.append(result)
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100.)

    monkeypatch.setattr(experiment, 'b32', schedule)
    model = SimpleNamespace(encode_static=lambda graph: None)
    seen = {}
    row, _ = experiment.complete_gnn(_parents(), model, 1, (), reconstruct=False, beam_width=2, evaluation_cache=seen)
    assert row['b32_calls'] == len(calls) == 1
    row, _ = experiment.complete_gnn(_parents(), model, 1, (), reconstruct=False, beam_width=2, evaluation_cache=seen)
    assert row['b32_calls'] == 0 and len(calls) == 1


def test_cmax_value_old_checkpoint_loads_and_only_new_head_trains():
    import io
    import run_gnn_weld_layout as experiment

    saved = torch.load(experiment.FINAL_MODEL, map_location='cpu', weights_only=True)
    model = experiment.joint_model(saved['state_dict'])
    original = {k: v.clone() for k, v in model.state_dict().items()}
    assert model.cmax_value_head is None and model.load_state_dict(original).missing_keys == []
    model.enable_cmax_value()
    assert all(p.requires_grad == name.startswith('cmax_value_head.') for name, p in model.named_parameters())
    optimizer = torch.optim.AdamW(model.cmax_value_head.parameters(), lr=.001)
    features = torch.randn(3, model.cmax_value_head[0].in_features)
    for _ in range(3):
        scores = model.cmax_values(features)
        loss = torch.nn.functional.softplus(scores[1] - scores[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert any(torch.count_nonzero(p) for p in model.cmax_value_head[-1].parameters())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = experiment.joint_model(torch.load(buffer, weights_only=True), cmax_value=True)
    assert torch.equal(model.cmax_values(features), restored.cmax_values(features))


def test_cmax_value_diversity_uses_mode_owner_and_another_parent(monkeypatch):
    from types import SimpleNamespace
    from mrta_search import gnn_constructor as c

    parents, model, visits = _controlled_vnext_beam(monkeypatch)
    space = c.parent_assignment_space(parents, CONFIG)
    indices = [next(i for i, a in enumerate(space.actions)
                    if a.parent_id == parent and a.pattern.kind is SplitKind.WHOLE and a.robots == (robot,))
               for parent, robot in (('p0', 0), ('p0', 0), ('p0', 1), ('p1', 0))]
    states = [SimpleNamespace(routes=((str(i),), (), (), ()), directions=((i % 2,), (), (), ())) for i in range(4)]
    plans = tuple(c.VNextJointAction(index, state, (), torch.empty(0), torch.empty(0)) for index, state in zip(indices, states))
    assert c.vnext_value_candidate_indices(space, plans, torch.tensor([4., 3., 2., 1.])) == (0, 2, 3)
    assert c.vnext_value_candidate_indices(space, plans[:2], torch.tensor([2., 1.])) == (0, 1)
    duplicate = (plans[0], plans[0])
    assert c.vnext_value_candidate_indices(space, duplicate, torch.tensor([2., 1.])) == (0,)


def test_cmax_value_full_split_features_match_training_and_constructor(monkeypatch):
    from mrta_search import gnn_constructor as c, lns

    torch.manual_seed(43)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True, cmax_value=True).eval()
    parents = (ParentWeld('seam', (2., 5.7), (10., 6.3)),)
    space = c.parent_assignment_space(parents, CONFIG)
    target = next(i for i, a in enumerate(space.actions) if a.pattern.kind is SplitKind.Y_SPLIT and a.robots == (0, 2))
    monkeypatch.setattr(c, 'shortlist_assignments', lambda *a, **k: (target,))
    expected, actual = [], []

    def observe(state, selected, plans, scores):
        assert all(len(p.actions) == len(p.insertion_features) == 2 for p in plans)
        keep = c.vnext_value_candidate_indices(space, plans, scores)
        expected.append(model.cmax_value_inputs(space, model.encode_static(space.graph), state, {'seam'},
                                                tuple(plans[i] for i in keep), scores[list(keep)]))

    original = model.cmax_values

    def score(features):
        actual.append(features.clone())
        return original(features)

    monkeypatch.setattr(model, 'cmax_values', score)
    result = c.construct_vnext_gnn(parents, model, CONFIG, space=space, step_observer=observe)
    assert result.solution is not None and len(result.actions) == 2
    assert len(actual) == len(expected) == 1 and torch.equal(actual[0], expected[0])
    assert actual[0].shape[1] == model.cmax_value_head[0].in_features > 300


def test_cmax_value_zero_head_preserves_legacy_width_one_then_changes_actual_choice(monkeypatch):
    from mrta_search import gnn_constructor as c

    torch.manual_seed(10)
    parents = (ParentWeld('p', (2., 9.), (2.6, 9.)),)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True).eval()
    legacy = c.construct_vnext_gnn(parents, model, CONFIG)
    model.enable_cmax_value()
    zero = c.construct_vnext_gnn(parents, model, CONFIG)
    assert (legacy.solution, legacy.directions, legacy.actions) == (zero.solution, zero.directions, zero.actions)
    monkeypatch.setattr(model, 'cmax_values', lambda features: torch.arange(len(features), dtype=torch.float32))
    reranked = c.construct_vnext_gnn(parents, model, CONFIG)
    assert (reranked.solution.routes, reranked.directions) != (legacy.solution.routes, legacy.directions)


def test_cmax_value_b32_preferences_ignore_both_failed_partial_cmax():
    from run_gnn_weld_layout import value_preferences

    failed = ScheduleResult(ScheduleStatus.DEADLOCK, cmax=1.)
    infeasible = ScheduleResult(ScheduleStatus.INFEASIBLE, cmax=2.)
    good = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100.)
    bad = ScheduleResult(ScheduleStatus.FEASIBLE, cmax=120.)
    assert value_preferences([failed, infeasible]) == []
    assert value_preferences([failed, good, bad]) == [(1, 0, 'feasibility'), (2, 0, 'feasibility'), (1, 2, 'cmax')]
    assert value_preferences([good, good]) == []
    assert value_preferences([None, good]) == []


def test_cmax_value_collector_uses_three_real_prefixes_and_only_original_continuations(monkeypatch):
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c

    experiment.configure_cpu()
    torch.manual_seed(19)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True).eval()
    monkeypatch.setattr(experiment, 'worker_model', model, raising=False)
    parents = tuple(ParentWeld(f'p{i}', (1. + i * 2., 9. if i < 4 else 3.),
                               (1.6 + i * 2., 9. if i < 4 else 3.)) for i in range(8))
    builds, evaluated = [], []
    original = experiment.build_joint

    def build(*args, **kwargs):
        assert args[1].cmax_value_head is None and kwargs['beam_width'] == 1 and kwargs.get('deadline') is None
        builds.append(kwargs)
        return original(*args, **kwargs)

    def b32(result):
        key = experiment.plan_key(result.solution, result.directions)
        assert key not in evaluated
        evaluated.append(key)
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=max(result.oriented_state.loads) + result.oriented_state.travel)

    monkeypatch.setattr(experiment, 'build_joint', build)
    monkeypatch.setattr(experiment, 'b32', b32)
    monkeypatch.setattr(experiment, 'rule_beam', lambda *a: pytest.fail('repeated RULE teacher'))
    monkeypatch.setattr(experiment, 'baseline_run', lambda *a: pytest.fail('repeated WAG/HGA teacher'))
    sample = experiment.collect_value_branches(('train_fixture', parents))
    assert [group['assigned_parents'] for group in sample['groups']] == [2, 4, 6]
    assert len(sample['rows']) == 9 and len(builds) == 10 and builds[0].get('state') is None
    assert len(evaluated) == sum(row['b32_calls'] for row in sample['rows']) <= 9
    for group in sample['groups']:
        assert len(group['pending']) + len(group['selected']) == len(parents)
        assert len(group['candidates']) == group['features'].shape[0] == 3
        assert group['features'].shape[1] == model.policy[0].in_features + c._VNEXT_VALUE_EXTRA_DIM


@pytest.mark.parametrize('rule_feasible', (True, False))
def test_formal_teacher_uses_full_wag_b32_and_only_positive_demonstrations(monkeypatch, rule_feasible):
    import io
    import math
    import pickle
    from types import SimpleNamespace
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c

    experiment.configure_cpu()
    solution = _solution()
    directions = tuple(tuple(0 for _ in route.block_ids) for route in solution.routes)
    reversed_bits = tuple(tuple(1 for _ in route.block_ids) for route in solution.routes)
    monkeypatch.setattr(experiment, 'rule_beam', lambda *a: c.JointCompletionResult(solution, directions, (), 0.))

    def search(parents, method, seed, budget, checkpoints, **kwargs):
        assert method == 'WAG+VNS' and budget == math.inf and checkpoints == ()
        assert kwargs['iterations'] == 500 and kwargs['candidate_evaluator'] is experiment.priority_schedule
        best = SimpleNamespace(solution=solution, directions=reversed_bits,
                               schedule=ScheduleResult(ScheduleStatus.FEASIBLE, cmax=1.))
        return SimpleNamespace(timely_best=best, iterations=500, termination_reason='COMPLETED')

    def b32(result):
        if result.directions == directions and rule_feasible:
            return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=100.)
        return ScheduleResult(ScheduleStatus.DEADLOCK, cmax=None, diagnostics=('actual B32 failure',))

    monkeypatch.setattr(experiment, 'baseline_run', search)
    monkeypatch.setattr(experiment, 'b32', b32)
    records, seen = experiment.formal_teacher_records(('train_fixture', _parents()))
    assert records[1]['row']['native_cmax_s'] == 1. and records[1]['row']['cmax_s'] is None
    assert records[1]['row']['status'] == 'DEADLOCK' and len(seen) == 2
    buffer = io.BytesIO()
    torch.save(dict(seen=pickle.dumps(seen)), buffer)
    buffer.seek(0)
    assert len(pickle.loads(torch.load(buffer, weights_only=True)['seen'])) == 2
    monkeypatch.setattr(experiment, 'formal_teacher_records', lambda *a: (records, seen))
    monkeypatch.setattr(experiment, 'worker_model', c.VNextGNN(workload_lookahead=True, joint_actions=True).eval(), raising=False)
    monkeypatch.setattr(experiment, 'build_joint', lambda *a, **k: c.JointCompletionResult(None, None, (), 0.))
    examples = []
    monkeypatch.setattr(experiment, 'teacher_examples', lambda *a, **k: examples.append(a[2]))
    sample = pickle.loads(experiment.collect_teachers(('train_fixture', _parents())))
    assert len(examples) == int(rule_feasible)
    assert sample.evidence['teachers']['WAG+VNS'] == ('DEADLOCK', None)


def test_formal_teacher_training_accepts_preferences_without_positive_demonstrations(monkeypatch):
    import pickle
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c, lns

    experiment.configure_cpu()
    torch.manual_seed(47)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True)
    space = c.parent_assignment_space(_parents(), CONFIG)
    state = lns.oriented_route_state(((),) * 4, ((),) * 4, {}, CONFIG)
    with torch.no_grad():
        indices, features = model.assignment_features(space, model.encode_static(space.graph), state, {},
                                                       set(space.parent_ids), CONFIG)
    preference = (indices[:2], features[:2], torch.tensor([[1., 0., 0., 0., 0., 0.], [2., 0., 0., 0., 0., 0.]]))
    samples = [experiment.TrainingLayout(space, None, None, (preference,), {'name': 'preference_only'}),
               experiment.TrainingLayout(space, None, None, (), {'name': 'no_signal'})]
    serialized = [pickle.dumps(sample) for sample in samples]

    class InlinePool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, function, layouts):
            return serialized

    saved, before = {}, model.joint_weights.detach().clone()
    monkeypatch.setattr(experiment, 'ProcessPoolExecutor', InlinePool)
    monkeypatch.setattr(experiment, 'load_layouts', lambda split: [('fixture', _parents())])
    monkeypatch.setattr(experiment, 'joint_model', lambda: model)
    monkeypatch.setattr(experiment, 'EPOCHS', 1)
    monkeypatch.setattr(experiment, 'VALIDATE_EPOCHS', (1,))
    monkeypatch.setattr(experiment, 'validate', lambda *args: dict(feasible=1, median_cmax_s=100., median_runtime_s=1.))
    monkeypatch.setattr(experiment.torch, 'save', lambda value, path: saved.update(value))
    experiment.train(1)
    assert saved['selected_epoch'] == 1 and len(saved['training']) == 2
    assert saved['history'][0]['assignment_loss'] == saved['history'][0]['insertion_loss'] == 0.
    assert saved['history'][0]['cmax_rank_loss'] > 0.
    assert not torch.equal(saved['state_dict']['joint_weights'], before)


def test_dagger_stratification_covers_weld_count_and_packing_size():
    from run_gnn_weld_layout import stratified_dagger_layouts

    records = [dict(layout_id=f'train_{i:03d}', parents=[None] * i,
                    rectangle_utilization=.5 + (i % 40) / 100., parts=[None] * (1 + i % 7)) for i in range(1, 121)]
    names, strata = stratified_dagger_layouts(records)
    assert len(names) == len(set(names)) == 60 and len(strata) == 6
    assert all(len(stratum['layouts']) == 10 for stratum in strata)
    assert [sum(40 * b < int(name[-3:]) <= 40 * (b + 1) for name in names) for b in range(3)] == [20, 20, 20]


def test_dagger_uses_current_value_policy_for_both_states_and_continuations(monkeypatch):
    import pickle
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c

    experiment.configure_cpu()
    torch.manual_seed(29)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True, cmax_value=True).eval()
    monkeypatch.setattr(model, 'cmax_values', lambda features: -features[:, -1])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    monkeypatch.setattr(experiment, 'worker_model', model, raising=False)
    parents = tuple(ParentWeld(f'p{i:02d}', (.7 + i % 10 * 1.8, 9. if i < 10 else 3.),
                              (1.3 + i % 10 * 1.8, 9. if i < 10 else 3.)) for i in range(20))
    original = experiment.build_joint
    builds, evaluated = [], []

    def build(*args, **kwargs):
        assert args[1] is model and kwargs['beam_width'] == 1 and kwargs.get('deadline') is None
        builds.append(kwargs)
        return original(*args, **kwargs)

    def schedule(result):
        key = experiment.plan_key(result.solution, result.directions)
        assert key not in evaluated
        evaluated.append(key)
        return ScheduleResult(ScheduleStatus.FEASIBLE, cmax=max(result.oriented_state.loads) + result.oriented_state.travel)

    monkeypatch.setattr(experiment, 'build_joint', build)
    monkeypatch.setattr(experiment, 'b32', schedule)
    monkeypatch.setattr(experiment, 'baseline_run', lambda *a, **k: pytest.fail('DAgger inference called WAG'))
    sample = experiment.collect_value_branches(('train_fixture', parents), fractions=(.05, .15, .5, .85),
                                               candidate_limit=2, current_policy=True, retain_evaluations=True)
    assert sample['policy_has_value'] and sample['original_complete']
    assert [group['assigned_parents'] for group in sample['groups']] == [1, 3, 10, 17]
    assert len(sample['rows']) == 8 and len(builds) == 9
    assert all(model.cmax_values(group['features'])[0] >= model.cmax_values(group['features'])[1] for group in sample['groups'])
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    replay = experiment.collect_value_branches(('train_fixture', parents), fractions=(.05, .15, .5, .85),
                                               candidate_limit=2, current_policy=True,
                                               evaluation_cache=pickle.loads(sample['evaluations']))
    assert sum(row['b32_calls'] for row in replay['rows']) == 0


def test_dagger_second_round_merges_preferences_and_preserves_input_normalization(monkeypatch):
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c

    torch.manual_seed(41)
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True)
    base = {key: value.clone() for key, value in model.state_dict().items()}
    model.enable_cmax_value()
    model.cmax_value_mean.copy_(torch.randn_like(model.cmax_value_mean))
    model.cmax_value_scale.fill_(2.)
    policy = {key: value.clone() for key, value in model.state_dict().items()}
    width = model.cmax_value_head[0].in_features
    groups = [dict(features=torch.randn(2, width), preferences=[(i % 2, 1 - i % 2, 'cmax')]) for i in range(2)]
    row = dict(layout_id='validation_fixture', status='FEASIBLE', cmax_s=100., construct_s=1., runtime_s=1.,
               b32_calls=0, processing_r0_s=90., processing_r1_s=80., processing_r2_s=85., processing_r3_s=75., move_s=10.)
    references = [dict(row, method=method, cmax_s=110.) for method in ('RULE/BEAM', 'WAG+VNS')]
    saved = dict(base_state_dict=base, references=[dict(records=[dict(row=r) for r in references])],
                 rounds={1: dict(samples=[dict(layout_id='a', groups=[groups[0]])]),
                         2: dict(policy_state_dict=policy, samples=[dict(layout_id='b', groups=[groups[1]])],
                                 completed_epoch=0, history=[], validation={})})
    monkeypatch.setattr(experiment, 'EPOCHS', 1)
    monkeypatch.setattr(experiment, 'VALIDATE_EPOCHS', (1,))
    monkeypatch.setattr(experiment, 'expanded_validate', lambda *a, **k: [row])
    monkeypatch.setattr(experiment, 'save_expanded_experiment', lambda *a: None)
    experiment.train_dagger_round(saved, 2)
    trained = saved['rounds'][2]
    assert trained['history'][0]['labelled_prefixes'] == 2 and trained['base_parameters_unchanged']
    assert torch.equal(trained['state_dict']['cmax_value_mean'], policy['cmax_value_mean'])
    assert torch.equal(trained['state_dict']['cmax_value_scale'], policy['cmax_value_scale'])


def test_expanded_final_selection_prioritizes_feasibility_and_does_not_enter_test_when_failed(monkeypatch):
    import run_gnn_weld_layout as experiment

    def summary(feasible, change):
        return dict(feasible=feasible, median_construct_s=1., paired={method: dict(common=10, wins=5, losses=5,
                    median_change_pct=change) for method in ('WAG+VNS', 'RULE/BEAM')})

    best = summary(18, 1.)
    saved = dict(base_selected_epoch=4, base_state_dict={'value': torch.tensor(1.)}, base_selected_summary=best,
                 rounds={i: dict(selected_epoch=2, state_dict={'value': torch.tensor(float(i + 1))},
                                 selected_summary=summary(18 - i, -float(i))) for i in (1, 2)},
                 references=[dict(records=[dict(row=dict(method=method, status='FEASIBLE'))
                                            for method in ('RULE/BEAM', 'WAG+VNS')]) for _ in range(20)])
    monkeypatch.setattr(experiment, 'expanded_teachers', lambda *a: saved)
    monkeypatch.setattr(experiment, 'train_expanded_base', lambda *a: None)
    monkeypatch.setattr(experiment, 'collect_dagger_round', lambda *a: None)
    monkeypatch.setattr(experiment, 'train_dagger_round', lambda *a: None)
    monkeypatch.setattr(experiment, 'save_expanded_experiment', lambda *a: None)
    monkeypatch.setattr(experiment, 'expanded_validate', lambda *a, **k: pytest.fail('unaccepted model entered TEST'))
    experiment.expanded_dagger_experiment(1)
    assert saved['selected_round'] == 0 and saved['selected_epoch'] == 4
    assert saved['selected_summary'] is best and not saved['acceptance']['feasibility']
    assert saved['state_dict']['value'].item() == 1.


def _owner_cf_fixture():
    import run_gnn_weld_layout as experiment
    from mrta_search import gnn_constructor as c, lns

    experiment.configure_cpu()
    torch.manual_seed(53)
    parents = (ParentWeld('prefix', (1., 9.), (2., 9.)), ParentWeld('target', (5., 9.), (10., 9.)))
    model = c.VNextGNN(workload_lookahead=True, joint_actions=True).eval()
    space = c.parent_assignment_space(parents, CONFIG)
    pattern = next(p for p in space.children if p.parent_id == 'prefix' and p.kind is SplitKind.WHOLE)
    selected = {'prefix': pattern}
    block = space.children[pattern][0]
    state = lns.oriented_route_state(((block.block_id,), (), (), ()), ((0,), (), (), ()), {block.block_id: block}, CONFIG)
    assignment = next(a for a in space.actions if a.parent_id == 'target' and a.pattern.kind is SplitKind.WHOLE
                      and a.robots == (0,))
    return experiment, model, space, state, selected, assignment


def test_owner_cf_candidates_change_only_whole_owner_and_preserve_prefix():
    experiment, model, space, state, selected, assignment = _owner_cf_fixture()
    with torch.no_grad():
        pair = experiment.owner_cf_candidates(space, model.encode_static(space.graph), model, state, selected, assignment)
    assert len(pair) == 2
    actions = [space.actions[p.assignment_index] for p in pair]
    assert actions[0] == assignment and actions[1].robots != actions[0].robots
    assert actions[0].parent_id == actions[1].parent_id == 'target'
    assert actions[0].pattern == actions[1].pattern and actions[0].pattern.kind is SplitKind.WHOLE
    kept = {b for route in state.routes for b in route}
    for plan in pair:
        assert all(not action.neighbour_changes for action in plan.actions)
        projected = [tuple((b, d) for b, d in zip(route, bits) if b in kept)
                     for route, bits in zip(plan.state.routes, plan.state.directions)]
        assert projected == [tuple(zip(route, bits)) for route, bits in zip(state.routes, state.directions)]
    assert max(pair[1].state.loads) < max(pair[0].state.loads)


def test_owner_cf_bounded_residual_matches_shortlist_final_and_training(monkeypatch):
    from mrta_search import gnn_constructor as c

    experiment, model, space, state, selected, _ = _owner_cf_fixture()
    original_keys = set(model.state_dict())
    model.bounded_assignment = True
    monkeypatch.setattr(model, 'assignment_scores', lambda rows: torch.full((len(rows),), 1000.))
    original = c.shortlist_assignments
    recorded = []
    embeddings = model.encode_static(space.graph)
    blocks = {b.block_id: b for pattern in selected.values() for b in space.children[pattern]}
    nodes = {b.block_id: space.block_nodes[pattern, b.block_id] for pattern in selected.values() for b in space.children[pattern]}
    pending = {'target'}
    _, features = model.assignment_features(space, embeddings, state, blocks, pending, CONFIG)

    def shortlist(space_arg, indices, scores, **kwargs):
        residual = scores - features[:, -4:-2] @ model.joint_weights[:2]
        assert torch.allclose(residual, torch.full_like(residual, .35), atol=1e-6)
        recorded.append(True)
        return original(space_arg, indices, scores, **kwargs)

    monkeypatch.setattr(c, 'shortlist_assignments', shortlist)
    with torch.no_grad():
        plans, scores = c.vnext_joint_actions(space, embeddings, model, state, blocks, nodes, pending, CONFIG)
        assert recorded and plans
        outcomes = torch.stack([p.outcome for p in plans])
        expected = model.joint_scores(space, embeddings, torch.tensor([p.assignment_index for p in plans]),
                                      torch.stack([p.assignment_features for p in plans]), outcomes)
        assert torch.allclose(scores, expected)
        assert torch.allclose(scores - outcomes @ model.joint_weights, torch.full_like(scores, .35), atol=1e-6)
        rows = torch.zeros(2, model.policy[0].in_features)
        model.bounded_assignment = False
        assert torch.equal(model.assignment_contribution(rows), torch.full((2,), 350.))
    assert set(model.state_dict()) == original_keys and model.cmax_value_head is None


def test_owner_cf_training_updates_only_policy_and_keeps_six_epochs(monkeypatch):
    import math

    experiment, model, space, state, selected, assignment = _owner_cf_fixture()
    with torch.no_grad():
        embeddings = model.encode_static(space.graph)
        plans = experiment.owner_cf_candidates(space, embeddings, model, state, selected, assignment)
        rows = model.embed_assignment_rows(space, embeddings, torch.tensor([p.assignment_index for p in plans]),
                                           torch.stack([p.assignment_features for p in plans]))
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    group = dict(assignment_rows=rows, outcomes=torch.stack([p.outcome for p in plans]), preferences=[(1, 0, 'cmax')])
    saved = dict(initial_state_dict=initial, state_dict=initial, completed_epoch=0, history=[],
                 training=[dict(layout_id='train_fixture', groups=[group])])
    monkeypatch.setattr(experiment, 'save_owner_cf', lambda saved: None)
    monkeypatch.setattr(experiment, 'b32', lambda *a: pytest.fail('head training called B32'))
    monkeypatch.setattr(experiment, 'baseline_run', lambda *a, **k: pytest.fail('repeated WAG/HGA search'))
    experiment.train_owner_cf(saved)
    assert saved['completed_epoch'] == len(saved['history']) == 6 and saved['frozen_parameters_unchanged']
    assert all(record['pairs'] == 1 and math.isfinite(record['ranking_loss']) for record in saved['history'])
    assert any(not torch.equal(value, saved['state_dict'][key]) for key, value in initial.items() if key.startswith('policy.'))
    assert all(torch.equal(value, saved['state_dict'][key]) for key, value in initial.items() if not key.startswith('policy.'))
    assert set(initial) == set(saved['state_dict']) and 'cmax_value_mean' not in saved['state_dict']


def test_owner_cf_selects_forty_feasible_train_layouts_across_eight_strata():
    import run_gnn_weld_layout as experiment

    records = [dict(layout_id=f'train_{i:03d}', parents=[None] * (50 + i),
                    rectangle_utilization=.79 + (i % 14) / 1000., parts=[None] * (10 + i % 20)) for i in range(112)]
    source = dict(teachers=[dict(layout_id=row['layout_id'], records=[dict(row=dict(status='FEASIBLE', method='WAG+VNS',
                   iterations=500, cmax_s=5000., processing_r0_s=100. + i, processing_r1_s=100.,
                   processing_r2_s=100., processing_r3_s=100.))]) for i, row in enumerate(records)])
    names, strata, teachers = experiment.owner_cf_layouts(source, records)
    assert len(names) == len(set(names)) == 40 and len(strata) == 8
    assert all(len(stratum['layouts']) == 5 for stratum in strata)
    assert all(name.startswith('train_') and teachers[name]['row']['status'] == 'FEASIBLE' for name in names)


def test_vnd_graph_has_directed_routes_qualifications_and_no_outcome_leakage():
    import numpy as np
    from mrta_baselines.hga import _HGAWorkspace
    from mrta_search.gnn_constructor import hga_vnd_inputs,HGAVNDCandidateGNN,batch_hga_vnd_inputs
    parents=tuple(ParentWeld(chr(97+i),(1.+i*2.,9.),(2.+i*2.,9.)) for i in range(6))
    w=_HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    active=[int(w.pattern_nodes[p,0]) for p in w.initial_selected]
    routes=[np.array(active[:3],np.int64),np.array(active[3:],np.int64),np.empty(0,np.int64),np.empty(0,np.int64)]
    directions=[np.array([0,1,0],np.int64),np.array([1,0,1],np.int64),np.empty(0,np.int64),np.empty(0,np.int64)]
    action=np.array([[1,active[1],active[4],0,1,1,1,1,0],
                     [4,active[0],active[2],0,0,0,2,2,1]],np.int64)
    inputs=hga_vnd_inputs(w,routes,directions,action,1000.)
    action[:,8]^=1
    changed=hga_vnd_inputs(w,routes,directions,action,1000.)
    assert all(torch.equal(value,changed[key]) for key,value in inputs.items())
    assert inputs['nodes'].shape==(10,32) and inputs['features'].shape==(2,35)
    relations=set(inputs['relations'].tolist())
    assert {0,1,2,3,5}.issubset(relations)
    edges={(int(a),int(b),int(t)) for (a,b),t in zip(inputs['edges'].T,inputs['relations'])}
    assert (0,1,2) in edges and (1,0,3) in edges
    model=HGAVNDCandidateGNN().eval()
    with torch.no_grad():
        single=model(inputs)
        batched=model(batch_hga_vnd_inputs([inputs,inputs]))
    assert torch.allclose(batched[:2],single,atol=1.e-6)
    assert torch.allclose(batched[2:],single,atol=1.e-6)
    loss=model(inputs).square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_vnd_vectorized_attention_matches_existing_sparse_gat():
    from mrta_ranker.model import SparseGATLayer
    from mrta_search.gnn_constructor import _vnd_attention
    torch.manual_seed(66)
    layer=SparseGATLayer(6,8,2)
    x=torch.randn(5,6)
    edges=torch.tensor([[0,1,2,3,4,0,1,3],[0,1,2,3,4,1,0,1]])
    relations=torch.tensor([0,0,0,0,0,1,2,3])
    assert torch.allclose(_vnd_attention(layer,x,edges,relations),layer(x,edges,relations),atol=1.e-6)


def test_eg_teacher_native_labels_are_separate_from_hga_acceptance(monkeypatch):
    import numpy as np
    import run_eg_gnn_pvnd as e
    from mrta_baselines.hga import _HGAWorkspace
    parents=tuple(ParentWeld(chr(97+i),(1.+i*2.,9.),(2.+i*2.,9.)) for i in range(4))
    w=_HGAWorkspace(parents,ScientificConfig(),FORMAL_SCOPE_V2)
    nodes=[int(w.pattern_nodes[p,0]) for p in w.initial_selected]
    raw=dict(selected=w.initial_selected.tolist(),routes=[nodes,[],[],[]],directions=[[0]*4,[],[],[]],
        elapsed_s=0.,individual_index=0,pass_number=1,visited_legal_per_move=[0,0,0,3,0,0],
        actions=[[4,nodes[0],nodes[2],0,0,0,2,1,0],
                 [4,nodes[0],nodes[3],0,0,0,3,2,1],
                 [4,nodes[1],nodes[3],0,1,0,3,3,0]])
    scores=iter([100.,90.,110.,float('nan')])
    monkeypatch.setattr(e,'_priority_arrays',lambda *args:(next(scores),None,None,None))
    groups,counts=e.label_observations(w,[raw,raw],'synthetic')
    assert len(groups)==1 and counts['label_native_calls']==4
    a,b,c=groups[0]['candidates']
    assert a['positive'] and not a['teacher_accepted'] and a['improvement_s']==10.
    assert not b['positive'] and b['teacher_accepted'] and b['improvement_s']==-10.
    assert not c['label_known'] and c['native_status']=='NUMERIC_FAILURE'
    assert c['native_cmax_s'] is None


def test_eg_offline_first_improvement_uses_only_evaluated_candidates():
    import numpy as np
    import run_eg_gnn_pvnd as e
    candidates=[dict(positive=p,native_eval_s=.01,materialize_s=.001) for p in (False,False,True)]
    m=e.order_metrics(candidates,np.array([0.,0.,1.]))
    assert m['original_evals']==3 and m['gnn_evals']==1
    assert m['pairwise_accuracy']==1. and m['gnn_service_s']==pytest.approx(.011)
    for c in candidates:c['positive']=False
    m=e.order_metrics(candidates,np.array([0.,0.,1.]))
    assert not m['has_improvement'] and m['original_evals']==m['gnn_evals']==3
    assert m['pairwise_accuracy'] is None


def test_eg_train_split_is_layout_disjoint_and_true_cmax_drives_preferences():
    import run_eg_gnn_pvnd as e
    layouts=e.training_layouts()
    assert len({name for name,_,_ in layouts})==20
    assert sum(role=='fit' for _,_,role in layouts)==15
    assert sum(role=='holdout' for _,_,role in layouts)==5
    assert all(name.startswith('train_') for name,_,_ in layouts)
    scores=torch.zeros(2,requires_grad=True)
    torch.manual_seed(2)
    loss=e.ranking_loss(scores,torch.ones(2),[dict(native_cmax_s=90.),dict(native_cmax_s=80.)])
    loss.backward()
    assert scores.grad[1]<0 and scores.grad[0]>0


def test_ppo_scale_initial_actor_is_exact_and_critic_shares_two_gat_layers():
    import run_gnn_weld_layout as e
    from mrta_ranker.model import SparseGATLayer
    e.configure_cpu()
    model=e.ppo_model();actor=e.feedback_model()
    assert sum(isinstance(m,SparseGATLayer) for m in model.modules())==2
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in actor.state_dict().items())
    _,c,parents,_=_feedback_fixture()
    space=c.parent_assignment_space(parents,CONFIG)
    assert torch.allclose(model.encode_static(space.graph),actor.encode_static(space.graph),atol=2e-6)
    assert model.policy[0].out_features==64 and model.layer1.heads==model.layer2.heads==4
    assert torch.equal(model.value_head[-1].weight,torch.zeros_like(model.value_head[-1].weight))


def test_ppo_scale_terminal_gae_and_legal_action_mask_are_exact():
    import run_gnn_weld_layout as e
    advantage,returns=e.ppo_gae(torch.tensor([.5,.4,.2]),-1.)
    assert advantage.tolist()==pytest.approx([-1.373,-1.34,-1.2])
    assert returns.tolist()==pytest.approx([-.873,-.94,-1.])
    scores=torch.tensor([1.,2.,999.],requires_grad=True)
    d=e.ppo_distribution(scores,torch.tensor([True,True,False]))
    assert d.probs.tolist()==pytest.approx([.26894142,.73105858,0.])
    assert torch.isfinite(d.entropy())
    with pytest.raises(ArithmeticError):e.ppo_distribution(scores,torch.zeros(3,dtype=torch.bool))


def _ppo_scale_fixture(monkeypatch):
    import run_gnn_weld_layout as e
    from mrta_search import gnn_constructor as c
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    e.configure_cpu();torch.manual_seed(71)
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(9.,9.),(10.,9.)),
             ParentWeld('c',(2.,3.),(3.,3.)),ParentWeld('d',(13.,3.),(14.,3.)))
    model=c.PPOJointInsertionGNN(16,4)
    def evaluated(result,seen):
        assert result.solution is not None
        return ScheduleResult(ScheduleStatus.FEASIBLE,cmax=1000.,directions=result.directions),True
    monkeypatch.setattr(e,'evaluate_once',evaluated)
    return e,c,model,('fixture',parents)


def test_ppo_scale_replays_same_complete_joint_distribution_and_updates_shared_actor_critic(monkeypatch):
    e,c,model,layout=_ppo_scale_fixture(monkeypatch)
    before={k:v.clone() for k,v in model.state_dict().items()}
    episodes=[e.ppo_rollout(model,layout,seed) for seed in (7,8)]
    import pickle
    episodes=[pickle.loads(pickle.dumps(episode)) for episode in episodes]
    for episode in episodes:
        assert episode['reward'] is not None and episode['row']['b32_calls']==1
        assert len(episode['trace'])==len(layout[1])
        embeddings=model.encode_static(episode['space'].graph)
        for step in episode['trace']:
            scores=model.feedback_scores(episode['space'],embeddings,step['inputs'])
            d=e.ppo_distribution(scores,step['mask'])
            assert torch.allclose(d.probs,step['old_probabilities'],atol=2e-6)
            assert float(d.log_prob(torch.tensor(step['choice'])))==pytest.approx(step['old_log_probability'],abs=2e-6)
            assert len(step['action_keys'])==len(scores) and step['mask'].all()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    statistics=e.ppo_update(model,episodes,optimizer)
    assert statistics['updates']==8 and statistics['replay_probability_max_error']<2e-6
    assert statistics['gradient_norm']>0 and math.isfinite(statistics['loss'])
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('layer1.'))
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('policy.'))
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('insertion_head.'))
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items() if k.startswith('value_head.'))


def test_ppo_scale_detects_wrong_candidate_normalizer_before_update(monkeypatch):
    e,c,model,layout=_ppo_scale_fixture(monkeypatch)
    episode=e.ppo_rollout(model,layout,3)
    episode['trace'][0]['old_probabilities'][0]+=.01
    with pytest.raises(RuntimeError,match='probability mismatch'):
        e.ppo_update(model,[episode],torch.optim.AdamW(model.parameters()))


def test_ppo_scale_numeric_failure_is_skipped_and_deadlock_is_distinct(monkeypatch):
    from mrta_reference.model import ScheduleResult,ScheduleStatus
    e,c,model,layout=_ppo_scale_fixture(monkeypatch)
    monkeypatch.setattr(e,'evaluate_once',lambda *a:(ScheduleResult(ScheduleStatus.NUMERIC_FAILURE),True))
    numerical=e.ppo_rollout(model,layout,2)
    assert numerical['row']['status']=='NUMERIC_FAILURE' and numerical['reward'] is None
    assert e.ppo_update(model,[numerical],torch.optim.AdamW(model.parameters()))['updates']==0
    monkeypatch.setattr(e,'evaluate_once',lambda *a:(ScheduleResult(ScheduleStatus.DEADLOCK),True))
    deadlock=e.ppo_rollout(model,layout,2)
    assert deadlock['row']['status']=='DEADLOCK' and deadlock['reward']==-5.


def test_ppo_scale_twenty_representatives_are_stratified_before_any_outcome(monkeypatch):
    import run_gnn_weld_layout as e
    rows=[];layouts=[(f'v_{i:03}',[None]*(40+i)) for i in range(100)]
    monkeypatch.setattr(e,'ppo_layouts',lambda split:layouts)
    monkeypatch.setattr(e,'read_rows',lambda path:rows)
    monkeypatch.setattr(e,'ppo_write',lambda new:rows.extend(new))
    selected=e.ppo_representatives()
    assert [int(name.split('_')[1]) for name,_ in selected]==list(range(2,100,5))
    assert len(rows)==20 and all(r['stage']=='selection' for r in rows)
    assert e.ppo_representatives()==selected
    rows.clear();rows.append(dict(stage='validation'))
    with pytest.raises(ValueError,match='before viewing'):e.ppo_representatives()


def test_ppo_scale_summary_does_not_allow_test_for_incomplete_or_worse_validation(monkeypatch):
    import run_gnn_weld_layout as e
    monkeypatch.setattr(e,'read_rows',lambda path:[])
    assert not e.ppo_summary()['test_decision']['passed']
    rows=[]
    for i in range(100):
        for size,score,pmax in ((0,120.,100.),(120,110.,99.),(500,108.,98.),(1000,107.,97.)):
            rows.append(dict(stage='validation',size=str(size),layout_id=f'v{i}',seed='7',status='FEASIBLE',
                cmax_s=str(score),max_processing_s=str(pmax),move_s='10',wait_s='0',runtime_s='1',b32_calls='1'))
        if i<20:
            rows.append(dict(stage='selection',layout_id=f'v{i}'))
            for method in ('WAG+VNS','HGA'):
                rows.append(dict(stage='baseline',method=method,layout_id=f'v{i}',seed='7',status='FEASIBLE',
                    cmax_s='100',runtime_s='10',b32_calls='1'))
    monkeypatch.setattr(e,'read_rows',lambda path:rows)
    result=e.ppo_summary()['test_decision']
    assert result['complete'] and result['processing_passed'] and result['time_passed']
    assert not result['quality_passed'] and not result['passed']


def _imitation_fixture():
    import run_gnn_weld_layout as e
    from mrta_search import gnn_constructor as c
    e.configure_cpu();torch.manual_seed(23)
    parents=tuple(ParentWeld(name,(x,y),(x+2,y)) for name,x,y in
                  [('a',1.,9.),('b',16.,9.),('c',1.,2.),('d',16.,2.)])
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
        {r:(p.parent_id+'::whole',) for r,p in enumerate(parents)},CONFIG,scope=FORMAL_SCOPE_V2)
    result=c.JointCompletionResult(solution,((1,),)*4,(),0.)
    record=dict(row=dict(method='fixture',status='FEASIBLE',cmax_s=1000.),plan=e.pack_directed_plan(result))
    model=c.VNextGNN(16,4,workload_lookahead=True,joint_actions=True,imitation_balance=True)
    return e,c,parents,model,record


def test_imitation_coverage_dp_matches_brute_force_and_detects_coupled_impossibility():
    from itertools import product
    e,c,parents,model,record=_imitation_fixture()
    for layout in (parents,(ParentWeld('vertical',(2.,4.),(2.,8.)),),parents[:2]):
        space=c.parent_assignment_space(layout,CONFIG)
        masks=space.parent_robot_masks
        brute={0}
        for values in product(*(masks[p] for p in space.parent_ids)):
            covered=0
            for value in values:covered|=value
            brute.add(covered)
        assert space.four_robot_cover_possible==(15 in brute)
    single=c.parent_assignment_space((ParentWeld('vertical',(2.,4.),(2.,8.)),),CONFIG)
    assert not single.four_robot_cover_possible
    assert single.coverage_diagnostic()['robots_without_legal_task']==[]


def test_imitation_last_assignment_cannot_abandon_a_coverable_robot():
    e,c,parents,model,record=_imitation_fixture()
    space=c.parent_assignment_space(parents,CONFIG)
    solution=e.unpack_directed_plan(parents,record['plan'],space).solution
    blocks=block_map(solution,CONFIG)
    state=e.lns.oriented_route_state((('a::whole',),(),('c::whole',),('d::whole',)),
                                    ((1,),(),(1,),(1,)),blocks,CONFIG)
    indices=torch.tensor([i for i,a in enumerate(space.actions) if a.parent_id=='b'])
    keep=c.coverage_preserving_assignments(space,indices,state,{'b'})
    assert keep.any() and (~keep).any()
    assert all(bool(ok)==bool(space.robot_masks[i]&2) for i,ok in zip(indices.tolist(),keep.tolist()))


def test_imitation_processing_forecast_does_not_include_move():
    e,c,parents,model,record=_imitation_fixture()
    space=c.parent_assignment_space(parents,CONFIG)
    state=e.lns.oriented_route_state(((),)*4,((),)*4,{},CONFIG)
    indices=torch.arange(len(space.actions));projected=space.processing.clone()
    pending=torch.ones(len(parents),dtype=torch.bool)
    moved=replace(state,arc_costs=((10000.,),(9000.,),(8000.,),(7000.,)))
    first=c._forecast_workload(space,indices,projected,state,pending)
    second=c._forecast_workload(space,indices,projected,moved,pending)
    assert torch.equal(first[:,-8:],second[:,-8:])
    assert not torch.equal(first[:,4:8],second[:,4:8])
    legacy=c._forecast_workload(space,indices,projected,moved,pending,include_move=True)
    assert not torch.equal(first[:,-8:],legacy[:,-8:])


def test_imitation_shortlist_preserves_owner_modes_and_excludes_high_score_overload():
    from types import SimpleNamespace
    e,c,parents,model,record=_imitation_fixture()
    actions=[SimpleNamespace(parent_id=str(i),robots=owner,pattern=SimpleNamespace(kind=kind))
             for i,(owner,kind) in enumerate([((0,),SplitKind.WHOLE),((1,),SplitKind.WHOLE),
                 ((0,1),SplitKind.X_SPLIT),((0,2),SplitKind.Y_SPLIT),((3,),SplitKind.WHOLE)])]
    work=torch.tensor([[2.,2.,2.,2.]]*4+[[1.,1.,1.,6.]])
    chosen=c.shortlist_assignments(SimpleNamespace(actions=actions),torch.arange(5),
        torch.tensor([0.,1.,2.,3.,1e8]),processing=work)
    assert set(chosen)=={0,1,2,3}


def test_imitation_teacher_forcing_keeps_exact_routes_directions_and_pruned_positive(monkeypatch):
    e,c,parents,model,record=_imitation_fixture()
    monkeypatch.setattr(e,'b32',lambda *a,**k:pytest.fail('B32 called on a teacher prefix'))
    monkeypatch.setattr(c,'structural_child_indices',lambda variants,**k:(0,))
    sample=e.imitation_replay(('train_fixture',parents),record,model,joint_stride=2)
    assert sample['evidence']['parent_steps']==len(parents)
    assert sample['evidence']['exact_routes_and_directions']
    assert sample['evidence']['rescued_teacher_insertions']>0
    assert len(sample['insertion'].sizes)==4
    assert all(a[2]==0 and a[3]==1 for row in sample['trace'] for a in row['actions'])


def test_imitation_supervision_updates_both_gats_and_all_existing_actor_scores():
    e,c,parents,model,record=_imitation_fixture()
    sample=e.imitation_replay(('train_fixture',parents),record,model,joint_stride=1)
    embeddings=model.encode_static(sample['space'].graph)
    a=sample['assignment'];ins=sample['insertion']
    first,_=e.imitation_choice_loss(model.assignment_scores(model.embed_assignment_rows(
        sample['space'],embeddings,a.indices,a.features)),a)
    second,_=e.imitation_choice_loss(model.insertion_logits(embeddings,ins.indices,ins.features),ins)
    joint=torch.stack([torch.nn.functional.cross_entropy(c.imitation_joint_scores(
        sample['space'],embeddings,model,inputs)[None],torch.tensor([answer])) for inputs,answer in sample['joint']]).mean()
    (first+second+.5*joint).backward()
    for prefix in ('layer1.','layer2.','policy.','insertion_head.','joint_weights'):
        assert any(p.grad is not None and bool(p.grad.abs().sum()>0)
                   for name,p in model.named_parameters() if name.startswith(prefix))
    assert not any('value_head' in name or 'critic' in name for name,_ in model.named_parameters())


def test_imitation_real_constructor_covers_four_robots_without_changing_physical_rules():
    e,c,parents,model,record=_imitation_fixture()
    result=c.construct_vnext_gnn(parents,model,CONFIG)
    assert result.solution is not None and all(route.block_ids for route in result.solution.routes)
    assert {b.split('::')[0] for route in result.solution.routes for b in route.block_ids}=={p.parent_id for p in parents}
    for route in result.solution.routes:
        assert all(robot_is_eligible(block_map(result.solution,CONFIG)[b],route.robot_id,CONFIG) for b in route.block_ids)


def test_imitation_resume_reuses_completed_b32_after_csv_permission_failure(monkeypatch):
    e,c,parents,model,record=_imitation_fixture()
    saved=dict(training_complete=True,state_dict={},selected_epoch=2,pilot=[],legal_start_selection_complete=True)
    calls,written=[],[]
    monkeypatch.setattr(e,'imitation_saved',lambda:saved)
    monkeypatch.setattr(e,'imitation_model',lambda *a,**k:model)
    monkeypatch.setattr(e,'imitation_references',lambda:None)
    monkeypatch.setattr(e,'ppo_layouts',lambda split:[(name,parents) for name in e.IMITATION_CASES])
    monkeypatch.setattr(e,'read_rows',lambda path:[])
    monkeypatch.setattr(e,'imitation_save',lambda state:None)
    monkeypatch.setattr(e,'imitation_summary',lambda:{})
    def evaluate(layout,model):
        calls.append(layout[0])
        return dict(layout_id=layout[0],method='GNN_IMITATION',b32_calls=1)
    monkeypatch.setattr(e,'imitation_evaluate',evaluate)
    monkeypatch.setattr(e,'imitation_write',lambda rows:(_ for _ in ()).throw(PermissionError('fixture CSV denied')))
    with pytest.raises(PermissionError):
        e.imitation_pilot()
    assert len(calls)==len(saved['pilot'])==1
    monkeypatch.setattr(e,'imitation_write',lambda rows:written.extend(rows))
    e.imitation_pilot()
    assert len(calls)==len(set(calls))==len(written)==6

