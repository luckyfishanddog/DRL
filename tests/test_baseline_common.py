from __future__ import annotations

from mrta_baselines.common import (
    BaselineBestEvent,
    BaselineStatus,
    CommonBaselineEvaluator,
    solution_telemetry,
)
from mrta_reference.model import ParentWeld, Route, ScientificConfig, SplitKind, SplitPattern
from mrta_reference.scope import FORMAL_SCOPE_V1_1
from mrta_reference.solution import canonicalize


def _teacher_runner(monkeypatch):
    import importlib
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('run_fixed_taskset_experiment')


def _teacher_row(runner, name='train_A', method='wag_vns', status='FEASIBLE', value=100, seed=1):
    return dict(split='train',layout_id=name,method=method,seed=seed,status=status,
        iterations=500 if method=='wag_vns' else 200000,termination_reason='ITERATION_LIMIT',
        native_cmax_s=value if status=='FEASIBLE' else '',cmax_s=value if status=='FEASIBLE' else '',
        diagnostic=runner.json_text(dict(input_adapter=runner.WAG_INPUT_ADAPTER if method=='wag_vns'
                                        else runner.HGA_INPUT_ADAPTER)),
        solution=runner.json_text(dict(routes=[['a'],[],[],[]],directions=[[0],[],[],[]])))


def _scale_teacher_fixture():
    from types import SimpleNamespace
    rows=[(f'train_{i:02d}',SimpleNamespace(tasks=tuple(range(100+i)))) for i in range(40)]
    rows += [(f'scale_train_{i:04d}',SimpleNamespace(tasks=tuple(range(80+i%140)))) for i in range(2960)]
    return rows,{name for name,_ in rows[:40]}


def test_scale_teacher_loader_and_ten_strata_keep_originals_unique(monkeypatch):
    import pytest
    runner=_teacher_runner(monkeypatch)
    rows,originals=_scale_teacher_fixture()
    original={name:ts for name,ts in rows if name in originals}
    expanded=dict(rows)
    monkeypatch.setattr(runner,'_train_file',lambda path:expanded if path.parent.name=='scale1000' else original)
    loaded,old=runner.scale3000_tasksets()
    assert len(loaded)==3000 and old==originals
    chosen=runner.hga_training_selection(loaded,old)
    assert len(chosen)==2000 and originals <= chosen
    assert chosen==runner.hga_training_selection(list(reversed(loaded)),old)
    ranked=sorted(loaded,key=lambda item:(len(item[1].tasks),item[0]))
    previous=set(originals)
    for i in range(10):
        group=[name for name,_ in ranked[i*300:(i+1)*300] if name not in originals]
        assert len(chosen & set(group))==196
        previous.update(group[(2*k+1)*len(group)//192] for k in range(96))
    assert len(previous)==1000 and previous <= chosen and len(chosen-previous)==1000
    expanded['train_00']=object()
    with pytest.raises(ValueError,match='differs'):
        runner.scale3000_tasksets()


def test_scale_teacher_train_file_rejects_duplicate_and_nontrain(monkeypatch,tmp_path):
    import pytest
    runner=_teacher_runner(monkeypatch)
    record=dict(layout_id='train_A',split='train',parents=[dict(parent_id='p',start=[1.,9.],end=[2.,9.])])
    path=tmp_path/'train.jsonl'
    path.write_text(runner.json_text(record)+'\n'+runner.json_text(record)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='duplicate TRAIN'):
        runner._train_file(path)
    record['split']='test'
    path.write_text(runner.json_text(record)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='non-TRAIN'):
        runner._train_file(path)


def test_scale_teacher_budget_and_native_teacher_eligibility(monkeypatch):
    runner=_teacher_runner(monkeypatch)
    for method,budget in [('wag_vns',500),('hga',200000)]:
        row=_teacher_row(runner,method=method)
        assert runner.complete_budget(row,method) and runner.feasible_teacher(row,method)
        dead=dict(row,status='DEADLOCK',native_cmax_s='',solution='')
        assert runner.complete_budget(dead,method) and not runner.feasible_teacher(dead,method)
        for bad in [dict(row,iterations=budget-1),dict(row,termination_reason='TIME_LIMIT'),dict(row,status='ERROR'),
                    dict(row,status='INCOMPLETE'),dict(row,split='validation'),dict(row,split='test')]:
            assert not runner.complete_budget(bad,method)
        for value in ('','nan','inf','0','-1'):
            assert not runner.feasible_teacher(dict(row,native_cmax_s=value,cmax_s='1'),method)
        assert not runner.feasible_teacher(dict(row,solution=runner.json_text(dict(
            routes=[['a'],[],[],[]],directions=[[],[],[],[]]))),method)


def test_scale_teacher_coverage_merges_history_recent_and_retry_results(monkeypatch,tmp_path):
    runner=_teacher_runner(monkeypatch)
    monkeypatch.setattr(runner,'HISTORICAL_RESULTS',tmp_path/'history')
    monkeypatch.setattr(runner,'RESULTS',tmp_path/'recent')
    monkeypatch.setattr(runner,'TEACHER_RESULTS',tmp_path/'new')
    for row in [_teacher_row(runner,value=200),_teacher_row(runner,value=100,seed=2),
                _teacher_row(runner,'train_B',status='DEADLOCK')]:
        runner.append_row('wag_vns',row,directory=runner.HISTORICAL_RESULTS)
    runner.append_row('wag_vns',_teacher_row(runner,'train_C'),directory=runner.RESULTS)
    runner.append_row('wag_vns',dict(_teacher_row(runner,'train_D'),status='ERROR'),directory=runner.TEACHER_RESULTS)
    targets={'train_A','train_B','train_C','train_D'}
    completed,summary=runner.teacher_coverage('wag_vns',targets)
    assert len(completed)==3 and summary['historical_reused']==3 and summary['missing']==['train_D']
    assert summary['error_or_incomplete']==1 and completed['train_A']['native_cmax_s']=='100'
    runner.append_row('wag_vns',_teacher_row(runner,value=80),directory=runner.TEACHER_RESULTS)
    runner.append_row('wag_vns',_teacher_row(runner,'train_D',status='DEADLOCK'),directory=runner.TEACHER_RESULTS)
    completed,summary=runner.teacher_coverage('wag_vns',targets)
    assert len(completed)==4 and summary['DEADLOCK']==2 and summary['new_completed']==1
    assert summary['error_or_incomplete']==0 and completed['train_A']['native_cmax_s']=='80'


def test_scale_teacher_append_repairs_only_partial_tail_and_rotates(monkeypatch,tmp_path):
    runner=_teacher_runner(monkeypatch)
    monkeypatch.setattr(runner,'TEACHER_RESULTS',tmp_path)
    runner.append_row('wag_vns',_teacher_row(runner),directory=tmp_path)
    path=tmp_path/'wag_vns.csv'
    with path.open('ab') as stream:stream.write(b'train,interrupted,"partial\xe4\xb8')
    assert len(runner.read_rows('wag_vns',directory=tmp_path))==1
    runner.append_row('wag_vns',_teacher_row(runner,'train_B'),directory=tmp_path)
    size=path.stat().st_size
    monkeypatch.setattr(runner,'LIMIT',size+10)
    runner.append_row('wag_vns',_teacher_row(runner,'train_C'),directory=tmp_path)
    assert path.stat().st_size==size
    assert (tmp_path/'wag_vns_002.csv').stat().st_size<runner.LIMIT
    assert [r['layout_id'] for r in runner.read_rows('wag_vns',directory=tmp_path)]==['train_A','train_B','train_C']


def test_scale_teacher_batch_skips_completed_and_pairs_deterministic_seeds(monkeypatch):
    runner=_teacher_runner(monkeypatch)
    rows,originals=_scale_teacher_fixture();names={name for name,_ in rows}
    selected=runner.hga_training_selection(rows,originals)
    pending=sorted(selected)[:2];calls=[];saved=[]
    monkeypatch.setattr(runner,'scale3000_tasksets',lambda:(rows,originals))
    def coverage(method,targets):
        done=targets-set(pending) if sum(c[0]==method for c in calls)<2 else targets
        return {name:None for name in done},dict(method=method,missing=sorted(targets-done))
    monkeypatch.setattr(runner,'teacher_coverage',coverage)
    def job(item,**kwargs):
        method,split,name,ts,seed=item;calls.append((method,name,seed,kwargs['purpose']))
        return _teacher_row(runner,name,method,status='DEADLOCK',seed=seed)
    monkeypatch.setattr(runner,'baseline_job',job)
    monkeypatch.setattr(runner,'append_row',lambda method,row,**kwargs:saved.append(row))
    runner.scale3000_teachers(workers=1)
    assert len(calls)==len(saved)==4
    assert [c[0] for c in calls]==['wag_vns','wag_vns','hga','hga']
    assert {c[1]:c[2] for c in calls[:2]}=={c[1]:c[2] for c in calls[2:]}
    for _,name,seed,purpose in calls:
        assert seed==runner.BASE_SEED+sorted(names).index(name) and purpose=='scale3000_teachers'
    calls.clear()
    runner.scale3000_teachers(workers=56,check_only=True)
    assert not calls


def test_scale_teacher_baseline_job_preserves_exact_native_configuration(monkeypatch):
    from types import SimpleNamespace
    from mrta_reference.model import ScheduleStatus
    runner=_teacher_runner(monkeypatch);observed=[]
    def run(parents,config,settings,**kwargs):
        observed.append((settings,kwargs))
        schedule=SimpleNamespace(feasible=False,status=ScheduleStatus.DEADLOCK,robot_completion=(0.,)*4,
                                 waiting_time=0.,diagnostics=())
        return SimpleNamespace(schedule=schedule,solution=None,iterations=settings.imax if hasattr(settings,'imax')
            else settings.max_iterations,termination_reason='ITERATION_LIMIT',reference_calls=1,status=ScheduleStatus.DEADLOCK,
            accounting={},diagnostics=(),actual_runtime=0.,initialization_time=0.,search_time=0.)
    monkeypatch.setattr(runner,'run_adapted_wag_vns',run);monkeypatch.setattr(runner,'run_adapted_hga',run)
    ts=SimpleNamespace(parents=(),tasks=(1,))
    for method in ('wag_vns','hga'):
        result=runner.baseline_job((method,'train','train_A',ts,7),purpose='scale3000_teachers')
        assert result['status']=='DEADLOCK' and runner.complete_budget(result,method)
    wag,hga=observed[0][0],observed[1][0]
    assert wag.imax==500
    assert (hga.mu,hga.lambda_size,hga.alpha,hga.max_iterations)==(20,10,20,200000)
    assert hga.vnd_objective=='ROUTE_CYCLE' and hga.direction_encoding=='EXPLICIT'
    assert hga.vnd_max_passes is None and hga.vnd_candidate_cap is None
    for _,kw in observed:
        assert kw['time_limit']==float('inf') and kw['checkpoints']==() and kw['certify'] is False
        assert kw['candidate_evaluator'] is runner.priority_schedule and kw['fixed_taskset'] is ts


def test_scale_teacher_gnn_reads_new_csv_and_best_native_teacher(monkeypatch,tmp_path):
    import importlib
    runner=_teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    monkeypatch.setattr(runner,'HISTORICAL_RESULTS',tmp_path/'history')
    monkeypatch.setattr(runner,'RESULTS',tmp_path/'recent')
    monkeypatch.setattr(runner,'TEACHER_RESULTS',tmp_path/'new')
    monkeypatch.setattr(gnn,'TEACHER_RESULTS',runner.TEACHER_RESULTS)
    for method in ('wag_vns','hga'):
        runner.append_row(method,_teacher_row(runner,method=method,value=100),directory=runner.HISTORICAL_RESULTS)
        runner.append_row(method,_teacher_row(runner,method=method,value=80),directory=runner.TEACHER_RESULTS)
        runner.append_row(method,dict(_teacher_row(runner,method=method,value=1),iterations=1),directory=runner.TEACHER_RESULTS)
        runner.append_row(method,dict(_teacher_row(runner,'validation_A',method,value=1),split='validation'),directory=runner.TEACHER_RESULTS)
    teachers=gnn.read_training_teachers()
    assert len(teachers)==2 and {float(row['native_cmax_s']) for row in teachers.values()}=={80.}


def test_scale_teacher_incomplete_coverage_stops_before_labels_and_training(monkeypatch,tmp_path):
    import importlib
    import pytest
    from types import SimpleNamespace
    runner=_teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    rows,originals=_scale_teacher_fixture()
    monkeypatch.setattr(gnn,'ROOT',tmp_path)
    monkeypatch.setattr(gnn,'scale3000_tasksets',lambda:(rows,originals))
    def coverage(method,targets):
        missing=[sorted(targets)[0]]
        return {},dict(method=method,missing=missing,duplicate_complete_keys=[],invalid_feasible_teachers=[])
    monkeypatch.setattr(gnn,'teacher_coverage',coverage)
    def forbidden(*a,**kw):raise AssertionError('missing coverage reached labels/training')
    monkeypatch.setattr(gnn,'build_supervision',forbidden)
    monkeypatch.setattr(gnn,'fit_training_normalization',forbidden)
    monkeypatch.setattr(gnn,'train_ranker',forbidden)
    args=SimpleNamespace(weights=tmp_path/'new.pt',dataset='scale1000',layouts=None)
    with pytest.raises(ValueError,match='no labels/training started'):
        gnn.train(args)
    assert not args.weights.exists()


def test_scale_teacher_full_training_passes_all_3000_and_accepts_complete_deadlocks(monkeypatch,tmp_path):
    import importlib
    import pytest
    from types import SimpleNamespace
    runner=_teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    rows,originals=_scale_teacher_fixture()
    monkeypatch.setattr(gnn,'ROOT',tmp_path)
    monkeypatch.setattr(gnn,'scale3000_tasksets',lambda:(rows,originals))
    def coverage(method,targets):
        assert len(targets)==(3000 if method=='wag_vns' else 2000)
        return {name:dict(status='DEADLOCK') for name in targets},dict(method=method,missing=[],duplicate_complete_keys=[])
    monkeypatch.setattr(gnn,'teacher_coverage',coverage)
    def normalization(dataset,*,tasksets):
        assert dataset=='scale1000' and len(tasksets)==3000
        return gnn.FeatureConfig()
    monkeypatch.setattr(gnn,'fit_training_normalization',normalization)
    def supervision(**kwargs):
        assert kwargs['layouts'] is None and len(kwargs['tasksets'])==3000
        assert kwargs['label_workers']==56
        return [(name,None,None) for name,_ in kwargs['tasksets']],{}
    monkeypatch.setattr(gnn,'build_supervision',supervision)
    def train(examples,**kwargs):
        import torch
        assert torch.get_num_threads()==56
        assert len(examples)==3000 and len({e[0] for e in examples})==3000
        assert kwargs==dict(seed=20261126,epochs=24,variant='full_gnn')
        raise RuntimeError('verified training call without executing training')
    monkeypatch.setattr(gnn,'train_ranker',train)
    args=SimpleNamespace(weights=tmp_path/'new.pt',dataset='scale1000',layouts=None,
                         training_seed=20261126,epochs=24,variant='full_gnn',label_workers=56,train_threads=56)
    with pytest.raises(RuntimeError,match='verified training call'):
        gnn.train(args)
    assert not args.weights.exists()


def test_scale_teacher_gnn_existing_checkpoint_is_loaded_before_skipping(monkeypatch,tmp_path,capsys):
    import importlib
    from types import SimpleNamespace
    _teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    monkeypatch.setattr(gnn,'ROOT',tmp_path)
    path=tmp_path/'existing.pt';path.write_bytes(b'existing model')
    observed=[]
    monkeypatch.setattr(gnn,'load_model',lambda dataset,variant,**kwargs:observed.append((dataset,variant,kwargs)))
    gnn.train(SimpleNamespace(weights=path,dataset='scale1000',variant='full_gnn'))
    assert observed==[('scale1000','full_gnn',dict(path=path))]
    assert 'BR_TRAIN_ALREADY_SAVED' in capsys.readouterr().out
    assert path.read_bytes()==b'existing model'


def test_scale_teacher_gnn_interrupted_checkpoint_is_preserved_and_not_skipped(monkeypatch,tmp_path):
    import importlib
    import pytest
    from types import SimpleNamespace
    _teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    monkeypatch.setattr(gnn,'ROOT',tmp_path)
    path=tmp_path/'partial.pt';path.write_bytes(b'incomplete checkpoint')
    with pytest.raises(ValueError,match='existing checkpoint is incomplete or invalid; file kept unchanged'):
        gnn.train(SimpleNamespace(weights=path,dataset='scale1000',variant='full_gnn'))
    assert path.read_bytes()==b'incomplete checkpoint'


def test_scale_teacher_parallel_results_are_parent_written_and_errors_remain_retryable(monkeypatch):
    from concurrent.futures import Future
    import pytest
    runner=_teacher_runner(monkeypatch);rows,originals=_scale_teacher_fixture()
    names={n for n,_ in rows};pending=sorted(names)[:2];saved=[];workers=[]
    monkeypatch.setattr(runner,'scale3000_tasksets',lambda:(rows,originals))
    def coverage(method,targets):
        done=targets-set(pending)
        done |= {r['layout_id'] for r in saved if runner.complete_budget(r,method)}
        return {n:None for n in done},dict(method=method,missing=sorted(targets-done))
    monkeypatch.setattr(runner,'teacher_coverage',coverage)
    class Pool:
        def __init__(self,max_workers):workers.append(max_workers)
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,fn,item,**kwargs):
            assert fn is runner.baseline_job and kwargs['purpose']=='scale3000_teachers'
            future=Future()
            if item[2]==pending[0]:future.set_exception(RuntimeError('worker failed'))
            else:future.set_result(_teacher_row(runner,item[2],status='DEADLOCK'))
            return future
    monkeypatch.setattr(runner,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(runner,'append_row',lambda method,row,**kwargs:saved.append(row))
    with pytest.raises(RuntimeError,match='incomplete teacher coverage'):
        runner.scale3000_teachers(('wag_vns',),workers=56)
    assert workers==[56] and {r['status'] for r in saved}=={'ERROR','DEADLOCK'}
    completed,summary=coverage('wag_vns',names)
    assert summary['missing']==[pending[0]] and pending[1] in completed


def test_scale_teacher_cli_uses_56_workers_without_starting_solves(monkeypatch):
    import inspect
    import sys
    runner=_teacher_runner(monkeypatch)
    assert inspect.signature(runner.scale3000_teachers).parameters['workers'].default==56
    observed=[]
    monkeypatch.setattr(runner,'scale3000_teachers',lambda methods,**kwargs:observed.append((methods,kwargs)))
    monkeypatch.setattr(sys,'argv',['run_fixed_taskset_experiment.py','scale3000-teachers','--method','wag_vns','--check-only'])
    runner.main()
    assert observed==[(('wag_vns',),dict(workers=56,check_only=True))]


def test_scale_teacher_gnn_requires_new_2000_coverage_after_old_1000_complete(monkeypatch,capsys):
    import importlib
    import json
    import pytest
    runner=_teacher_runner(monkeypatch);gnn=importlib.import_module('run_gnn_vns_br')
    rows,originals=_scale_teacher_fixture()
    ranked=sorted(rows,key=lambda item:(len(item[1].tasks),item[0]))
    previous=set(originals)
    for i in range(10):
        group=[name for name,_ in ranked[i*300:(i+1)*300] if name not in originals]
        previous.update(group[(2*k+1)*len(group)//192] for k in range(96))
    assert len(previous)==1000
    def coverage(method,targets):
        done=set(targets) if method=='wag_vns' else previous
        return {name:dict(status='DEADLOCK') for name in done},dict(method=method,target_layouts=len(targets),
            complete_budget=len(done),missing=sorted(targets-done),duplicate_complete_keys=[])
    monkeypatch.setattr(gnn,'teacher_coverage',coverage)
    with pytest.raises(ValueError,match='no labels/training started'):
        gnn.check_training_teachers(rows,originals)
    summaries=[json.loads(line.split(' ',1)[1]) for line in capsys.readouterr().out.splitlines()
               if line.startswith('BR_TEACHER_COVERAGE ')]
    hga=next(s for s in summaries if s['method']=='hga')
    assert hga['target_layouts']==2000 and hga['complete_budget']==1000 and len(hga['missing'])==1000


def _solution():
    config = ScientificConfig()
    parents = (
        ParentWeld("upper", (2.0, 9.0), (3.0, 9.0)),
        ParentWeld("lower", (14.0, 3.0), (15.0, 3.0)),
    )
    patterns = tuple(
        SplitPattern(parent.parent_id, SplitKind.WHOLE) for parent in parents
    )
    solution = canonicalize(
        parents,
        patterns,
        (
            Route(0, ("upper::whole",)),
            Route(1, ()),
            Route(2, ("lower::whole",)),
            Route(3, ()),
        ),
        config,
    )
    return config, parents, solution


def _native_four_task_fixture():
    from mrta_reference.fixed_taskset import generate_taskset
    config=ScientificConfig()
    parents=(ParentWeld('p',(0.,9.),(6.6,9.)),ParentWeld('u',(13.,9.),(14.,9.)),
             ParentWeld('l',(1.,2.),(2.,2.)),ParentWeld('r',(13.,2.),(14.,2.)))
    taskset=generate_taskset(parents,config)
    groups={0:tuple(t.task_id for t in taskset.tasks if t.source_parent_id=='p'),
            1:('u::seg_000',),2:('l::seg_000',),3:('r::seg_000',)}
    return config,taskset,canonicalize(taskset.parents,taskset.patterns,groups,config)


def test_unified_native_interface_calls_only_one_compiled_kernel(monkeypatch):
    import pytest
    from mrta_baselines import wag_vns as wag
    from mrta_reference import scheduler
    from mrta_reference.model import OperationKind
    config,ts,solution=_native_four_task_fixture()
    calls=[];kernel=wag._priority_arrays
    def observed(*a,**k):calls.append(1);return kernel(*a,**k)
    def forbidden(*a,**k):raise AssertionError('Python reference scheduler called by current objective')
    monkeypatch.setattr(wag,'_priority_arrays',observed)
    monkeypatch.setattr(scheduler,'reference_schedule_optimized',forbidden)
    result=scheduler.native_schedule(solution,config,orientations={r:(0,) for r in range(4)})
    assert result.feasible and calls==[1]
    assert result.cmax==max(result.robot_completion)
    assert result.cmax==pytest.approx(config.process_time(6.6))
    assert len(result.directions)==4 and all(d==(0,) for d in result.directions)
    assert sum(op.kind is OperationKind.SETUP for op in result.operations)==4
    assert sum(op.kind is OperationKind.POST for op in result.operations)==4
    assert sum(op.duration for op in result.operations if op.kind is OperationKind.SETUP)==4*20.
    assert sum(op.duration for op in result.operations if op.kind is OperationKind.POST)==4*30.
    assert result.waiting_time==sum(result.robot_waiting_time)
    assert len(ts.tasks)>4 and len(solution.routes[0].block_ids)==1


def test_unified_native_gnn_array_and_public_results_match_without_final_call(monkeypatch):
    import numpy as np
    import pytest
    from mrta_search import gnn_vns_br as br
    from mrta_baselines.wag_vns import priority_schedule
    config,ts,solution=_native_four_task_fixture()
    workspace=br.FixedWorkspace(ts,config)
    routes=[np.array([workspace.index[b] for b in route.block_ids],np.int64) for route in solution.routes]
    bits=[np.zeros(len(route),np.int64) for route in routes]
    calls=[];kernel=br._priority_arrays
    def observed(*a,**k):calls.append(1);return kernel(*a,**k)
    monkeypatch.setattr(br,'_priority_arrays',observed)
    state=workspace.evaluate(routes,bits)
    result=workspace.schedule_result(state)
    assert calls==[1] and workspace.evaluations==1
    public=priority_schedule(solution,config,bits)
    assert result.status==public.status and result.cmax==pytest.approx(public.cmax)
    assert result.robot_completion==pytest.approx(public.robot_completion)
    assert result.directions==public.directions and result.operations==public.operations
    assert result.waiting_time==pytest.approx(public.waiting_time)


def test_unified_native_deadlock_has_no_alternative_dispatch(monkeypatch):
    from mrta_reference import scheduler
    from mrta_reference.model import ScheduleStatus
    from mrta_baselines import wag_vns as wag
    config=ScientificConfig()
    parents=(ParentWeld('a',(10.,9.),(11.,9.)),ParentWeld('b',(1.,9.),(2.,9.)))
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                          {0:('a::whole',),1:('b::whole',)},config)
    calls=[];kernel=wag._priority_arrays
    def observed(*a,**k):calls.append(1);return kernel(*a,**k)
    def forbidden(*a,**k):raise AssertionError('an alternative Python dispatch was attempted')
    monkeypatch.setattr(wag,'_priority_arrays',observed)
    monkeypatch.setattr(scheduler,'reference_schedule_optimized',forbidden)
    result=scheduler.native_schedule(solution,config,orientations={0:(0,),1:(0,)})
    assert result.status is ScheduleStatus.DEADLOCK and result.cmax is None
    assert calls==[1] and result.diagnostics


def test_native_wait_contributes_to_each_robot_completion():
    import pytest
    from mrta_reference.scheduler import native_schedule
    from mrta_reference.model import OperationKind
    config=ScientificConfig(weld_speed=1.,empty_speed=100.,t_pre=1.,t_post=1.)
    parents=(ParentWeld('a',(1.,9.),(2.,9.)),ParentWeld('b',(17.,9.),(18.,9.)),
             ParentWeld('c',(14.,9.),(20.,9.)))
    solution=canonicalize(parents,tuple(SplitPattern(p.parent_id,SplitKind.WHOLE) for p in parents),
                          {0:('a::whole','b::whole'),1:('c::whole',)},config)
    result=native_schedule(solution,config,orientations={0:(0,0),1:(0,)})
    assert result.feasible and result.waiting_time>0
    for robot in range(4):
        owned=[op for op in result.operations if op.robot_id==robot]
        assert result.robot_completion[robot]==pytest.approx(max((op.end_time for op in owned),default=0.))
        assert result.robot_completion[robot]==pytest.approx(sum(op.duration for op in owned))
    assert result.cmax==max(result.robot_completion)
    assert result.waiting_time==sum(op.duration for op in result.operations if op.kind is OperationKind.WAIT)


def test_native_distinguishes_numeric_failure_from_structural_infeasibility(monkeypatch):
    from mrta_reference import scheduler
    from mrta_reference.model import ScheduleStatus
    from mrta_baselines import wag_vns
    import numpy as np
    config,_,solution=_native_four_task_fixture()
    result=scheduler.native_schedule(solution,config,orientations={0:(2,)})
    assert result.status is ScheduleStatus.INFEASIBLE and result.cmax is None
    def broken(*a,**k):
        return float('nan'),[np.zeros(1,dtype=np.int64) for _ in range(4)],[],np.zeros(4)
    monkeypatch.setattr(wag_vns,'_priority_arrays',broken)
    result=scheduler.native_schedule(solution,config,orientations={r:(0,) for r in range(4)})
    assert result.status is ScheduleStatus.NUMERIC_FAILURE and result.cmax is None


def test_fixed_baseline_entries_reuse_native_schedule_without_final_evaluation(monkeypatch):
    import importlib
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from mrta_reference.scheduler import native_schedule
    from mrta_reference.model import ScheduleStatus
    from mrta_baselines import wag_vns
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_fixed_taskset_experiment')
    config,taskset,solution=_native_four_task_fixture()
    schedule=native_schedule(solution,config,orientations={r:(0,) for r in range(4)})
    def forbidden(*a,**k):raise AssertionError('baseline entry repeated native evaluation')
    monkeypatch.setattr(wag_vns,'_priority_arrays',forbidden)
    for method,attr,budget in [('wag_vns','run_adapted_wag_vns',500),('hga','run_adapted_hga',200000)]:
        def fake(parents,config,settings,**kwargs):
            assert kwargs['certify'] is False and kwargs['candidate_evaluator'] is runner.priority_schedule
            if method=='hga':assert settings.vnd_objective=='ROUTE_CYCLE'
            return SimpleNamespace(schedule=schedule,solution=solution,directions=schedule.directions,
                iterations=budget,termination_reason='ITERATION_LIMIT',reference_calls=1,
                status=ScheduleStatus.FEASIBLE,accounting={},diagnostics=(),actual_runtime=.01,
                initialization_time=.005,search_time=.005)
        monkeypatch.setattr(runner,attr,fake)
        row=runner.baseline_job((method,'train','toy',taskset,17))
        detail=json.loads(row['diagnostic'])
        assert row['status']=='FEASIBLE' and row['cmax_s']==schedule.cmax
        assert row['coordination_calls']==1 and detail['final_scheduler_extra_calls']==0
        assert detail['final_schedule_reused']


def test_legacy_baseline_entry_reuses_native_timeline(monkeypatch):
    import importlib
    from dataclasses import asdict
    from pathlib import Path
    from types import SimpleNamespace
    from mrta_reference.scheduler import native_schedule
    from mrta_baselines.common import BaselineAccounting
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    runner=importlib.import_module('run_gnn_weld_layout')
    config,_,solution=_native_four_task_fixture()
    schedule=native_schedule(solution,config,orientations={r:(0,) for r in range(4)})
    best=SimpleNamespace(solution=solution,directions=schedule.directions,schedule=schedule,completed_at=.01)
    def forbidden(*a,**k):raise AssertionError('legacy entry repeated native scheduling')
    monkeypatch.setattr(runner,'schedule_explicit',forbidden)
    for method in ('WAG+VNS','HGA'):
        result=SimpleNamespace(timely_best=best,iterations=runner.BASELINE_ITERATIONS[method],
            accounting=asdict(BaselineAccounting()),reference_calls=7,certifier_calls=0,
            actual_runtime=.02,initialization_time=.01,overshoot=0.,best_events=(),termination_reason='ITERATION_LIMIT')
        monkeypatch.setattr(runner,'baseline_run',lambda *a,**k:result)
        row=runner.baseline_job((('toy',solution.parents),method,17))
        assert row['cmax_s']==schedule.cmax and row['native_cmax_s']==schedule.cmax
        assert row['coordination_calls']==7 and row['wait_s']==schedule.waiting_time
    assert runner.coordination(best) is schedule


def test_common_baseline_path_uses_formal_scope_direction_and_certifier():
    config, parents, solution = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=2.0, checkpoints=(0.0, 2.0)
    )
    candidate = evaluator.evaluate(solution, source="FIXTURE")
    assert candidate.status is BaselineStatus.COMPLETED
    assert candidate.schedule is not None
    assert candidate.schedule.scope_id == FORMAL_SCOPE_V1_1.scope_id
    assert candidate.schedule.scope_hash == FORMAL_SCOPE_V1_1.scope_hash
    assert candidate.certification is not None and candidate.certification.certified
    assert evaluator.accounting.reference_calls == 1
    assert evaluator.accounting.certifier_calls == 1


def test_precomputed_native_schedule_preserves_metrics_and_avoids_second_call():
    from mrta_baselines.wag_vns import priority_schedule
    config, parents, solution = _solution()
    schedule = priority_schedule(solution, config)
    ordinary = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V1_1, time_limit=10.,
                                       certify=False, candidate_evaluator=priority_schedule)
    expected = ordinary.evaluate(solution, source='ORDINARY')
    def forbidden(*args):
        raise AssertionError('duplicate native scheduling')
    reused = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V1_1, time_limit=10.,
                                     certify=False, candidate_evaluator=forbidden)
    actual = reused.evaluate(solution, source='REUSED', native_schedule=schedule)
    assert actual.schedule == expected.schedule
    assert actual.directions == expected.directions
    assert actual.metrics == expected.metrics
    assert reused.accounting.reference_calls == 0
    assert reused.best is actual


def test_checkpoint_semantics_do_not_backfill_late_feasible_solution():
    config, parents, _ = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=30.0, checkpoints=(5.0, 30.0, 60.0)
    )
    evaluator.best_events.extend(
        (BaselineBestEvent(6.0, 100.0, "LATE"), BaselineBestEvent(20.0, 90.0, "BETTER"))
    )
    assert evaluator.checkpoint_values() == {5.0: None, 30.0: 90.0, 60.0: None}


def test_native_cache_distinguishes_explicit_directions_on_the_same_order():
    from mrta_baselines.wag_vns import priority_schedule
    config = ScientificConfig()
    parents = (ParentWeld('a', (1., 9.), (2., 9.)), ParentWeld('b', (5., 9.), (6., 9.)))
    patterns = tuple(SplitPattern(p.parent_id, SplitKind.WHOLE) for p in parents)
    solution = canonicalize(parents, patterns, (Route(0, ('a::whole', 'b::whole')),
                            Route(1, ()), Route(2, ()), Route(3, ())), config)
    a = priority_schedule(solution, config, ((0, 0), (), (), ()))
    b = priority_schedule(solution, config, ((1, 0), (), (), ()))
    assert a.feasible and b.feasible and a.cmax != b.cmax
    def forbidden(*args):
        raise AssertionError('precomputed explicit scores must not be recomputed')
    evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V1_1, time_limit=10.,
                                         certify=False, candidate_evaluator=forbidden)
    first = evaluator.evaluate(solution, source='A', native_schedule=a)
    second = evaluator.evaluate(solution, source='B', native_schedule=b)
    again = evaluator.evaluate(solution, source='A_AGAIN', native_schedule=a)
    assert first.schedule == again.schedule == a
    assert second.schedule == b and second.directions != first.directions
    assert evaluator.accounting.reference_calls == 0


def test_experiment_can_use_coordination_without_certifier(monkeypatch):
    import mrta_baselines.common as common
    config, parents, solution = _solution()
    def forbidden(*args, **kwargs):
        raise AssertionError('experiment must not call certify_schedule')
    monkeypatch.setattr(common, 'certify_schedule', forbidden)
    evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V1_1,
                                        time_limit=2., certify=False)
    result = evaluator.evaluate(solution, source='NATIVE_ONLY')
    assert result.status is BaselineStatus.COMPLETED
    assert result.schedule.feasible and result.certification is None
    assert evaluator.accounting.reference_calls == 1
    assert evaluator.accounting.certifier_calls == 0
    assert evaluator.timely_best == result


def test_baseline_keeps_timely_solution_when_late_candidate_updates_native_best():
    from dataclasses import replace

    config, parents, solution = _solution()
    evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V1_1, time_limit=30.0)
    candidate = evaluator.evaluate(solution, source="TIMELY")
    assert candidate.metrics is not None
    late = replace(candidate, completed_at=31.0, source="LATE",
                   metrics=replace(candidate.metrics, cmax=candidate.metrics.cmax - 1.0))
    evaluator._record_best(late)
    result = evaluator.finish(method_id="FIXTURE", method_config_hash="fixture",
                              iterations=1, initialization_time=0.0)
    assert result.best_source == "LATE"
    assert result.timely_best == candidate


def test_failure_has_no_penalty_cmax_and_is_separately_classified():
    config, parents, _ = _solution()
    evaluator = CommonBaselineEvaluator(
        parents, config, FORMAL_SCOPE_V1_1, time_limit=1.0
    )
    evaluator.reject_construction("fixture rejection")
    result = evaluator.finish(
        method_id="FIXTURE",
        method_config_hash="fixture",
        iterations=0,
        initialization_time=0.0,
    )
    assert result.status is BaselineStatus.INITIALIZATION_FAILED
    assert result.metrics is None
    assert result.solution is None
    assert result.termination_reason == "COMPLETED_OTHER"


def test_initial_solution_telemetry_hash_is_canonical_deterministic_and_source_free():
    config, _, solution = _solution()
    directions = ((0,), (), (0,), ())
    first = solution_telemetry(solution, directions, config)
    second = solution_telemetry(solution, directions, config)
    assert first == second
    assert first["initial_solution_hash"] == solution.canonical_hash
    assert first["initial_robot_block_counts"] == [1, 0, 1, 0]
    assert len(first["initial_route_hashes"]) == 4


def test_zr_observer_same_result_and_candidate_replay():
    from dataclasses import asdict
    from mrta_reference.scheduler import NativeReferenceEvaluator
    from mrta_reference.scope import FORMAL_SCOPE_V2
    config, parents, solution = _solution()
    expected = NativeReferenceEvaluator(FORMAL_SCOPE_V2)(solution, config)
    for level in (0, 1, 2):
        with ReferenceTrace("FIXTURE", level) as trace:
            evaluator = CommonBaselineEvaluator(parents, config, FORMAL_SCOPE_V2, time_limit=2.0)
            candidate = evaluator.evaluate(solution, source="FIXTURE_SOURCE")
        assert asdict(candidate.schedule) == asdict(expected)
        if level:
            row = trace.serialize({"instance_id": "fixture", "instance_geometry_hash": "fixture", "tier": "SMALL"})[0]
            assert row["scheduler_duration"] > 0
            assert row["certified"] is True
            assert row["candidate_source"] == "FIXTURE_SOURCE"
            assert row["native_evaluation_time"] == row["scheduler_duration"]
            recovered, cfg, directions = reconstruct(row["replay"])
            assert recovered.canonical_hash == solution.canonical_hash
            assert cfg.scientific_hash == config.scientific_hash
            assert tuple(directions.values()) == expected.directions
        else:
            assert trace.rows == []

def test_zr_observer_exception_explicitly_fails_closed_and_restores():
    import pytest
    from mrta_reference.scheduler import NativeReferenceEvaluator
    original = NativeReferenceEvaluator.__call__
    config, _, solution = _solution()
    def fail(row):
        raise RuntimeError("observer sink failed")
    with pytest.raises(RuntimeError, match="observer sink failed"):
        with ReferenceTrace("FIXTURE", sink=fail):
            NativeReferenceEvaluator(FORMAL_SCOPE_V1_1)(solution, config)
    assert NativeReferenceEvaluator.__call__ is original

def test_zr_observer_preserves_protected_certifier_identity_in_real_alns_initialization():
    from mrta_search.initialization import build_initial_solution, InitializationStatus
    from mrta_search.stats import SearchStats
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.certifier import certify_schedule
    import mrta_search.initialization as initialization
    config, parents, _ = _solution()
    with ReferenceTrace("SA_OI_ALNS_V2") as trace:
        result = build_initial_solution(parents, config, SearchStats(FORMAL_SCOPE_V2.scope_id, 0),
                    scope=FORMAL_SCOPE_V2, construction_budget=2, kinit_ref=2)
        assert initialization.certify_schedule is certify_schedule
    assert result.status is InitializationStatus.SUCCESS
    assert trace.rows and any(row["certified"] for row in trace.rows)

import time,traceback,json,hashlib
from functools import wraps
from dataclasses import asdict
from mrta_reference import scheduler
from mrta_reference.model import ParentWeld,Route,SplitPattern,SplitKind,Rail,ScientificConfig,ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize
from mrta_search.stats import SearchStats
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
def payload(solution, config, directions):
    return {"parents": [asdict(p) for p in solution.parents], "patterns": [asdict(p) for p in solution.patterns],
            "routes": [asdict(r) for r in solution.routes], "revision": solution.revision,
            "directions": directions, "scientific_config": asdict(config), "scope_id": FORMAL_SCOPE_V2.scope_id,
            "scope_hash": FORMAL_SCOPE_V2.scope_hash}

def reconstruct(data):
    if data["scope_hash"] != FORMAL_SCOPE_V2.scope_hash:
        raise ValueError("replay scope differs")
    config = ScientificConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in data["scientific_config"].items()})
    parents = tuple(ParentWeld(p["parent_id"], tuple(p["start"]), tuple(p["end"])) for p in data["parents"])
    patterns = tuple(SplitPattern(p["parent_id"], SplitKind(p["kind"]), p["t"], p["point_id"], p["mandatory"], None if p["rail"] is None else Rail(p["rail"])) for p in data["patterns"])
    routes = tuple(Route(r["robot_id"], tuple(r["block_ids"])) for r in data["routes"])
    solution = canonicalize(parents, patterns, routes, config, revision=data["revision"], scope=FORMAL_SCOPE_V2)
    return solution, config, {r: tuple(v) for r, v in enumerate(data["directions"])}

class ReferenceTrace:
    """Observe native calls and existing certification without replaying dispatch."""
    def __init__(self, method, level=1, sink=None):
        if level not in (0,1,2):raise ValueError('telemetry level must be 0,1 or 2')
        self.method,self.level,self.sink=method,level,sink
        self.rows,self.errors,self.patches=[],[],[]
        self.source='NATIVE_REFERENCE'

    def patch(self,obj,name,replacement):
        self.patches.append((obj,name,getattr(obj,name)))
        setattr(obj,name,replacement)

    def __enter__(self):
        if self.level==0:return self
        original=scheduler.NativeReferenceEvaluator.__call__
        @wraps(original)
        def reference(evaluator,solution,config,*,orientations=None):
            started=time.perf_counter()
            result=original(evaluator,solution,config,orientations=orientations)
            duration=time.perf_counter()-started
            row=dict(reference_call_index=len(self.rows)+1,method_id=self.method,
                candidate_source=self.source,scheduler_duration=duration,native_evaluation_time=duration,
                schedule_status=result.status.value,Cmax=result.cmax,certified=None,certification_errors=[],
                _solution=solution,_config=config,_schedule=result)
            self.rows.append(row)
            if self.sink is not None:self.sink(row)
            return result
        self.patch(scheduler.NativeReferenceEvaluator,'__call__',reference)
        original_evaluate=CommonBaselineEvaluator.evaluate
        @wraps(original_evaluate)
        def evaluate(evaluator,solution,*args,**kwargs):
            previous=self.source;self.source=kwargs.get('source','NATIVE_REFERENCE')
            before=len(self.rows)
            try:result=original_evaluate(evaluator,solution,*args,**kwargs)
            finally:self.source=previous
            if len(self.rows)>before:
                self.rows[-1]['certified']=bool(result.certification and result.certification.certified)
            return result
        self.patch(CommonBaselineEvaluator,'evaluate',evaluate)
        original_record=SearchStats.record_reference
        from inspect import signature
        record_signature=signature(original_record)
        @wraps(original_record)
        def recorded(stats,*args,**kwargs):
            result=original_record(stats,*args,**kwargs)
            values=record_signature.bind(stats,*args,**kwargs).arguments
            schedule=values.get('schedule')
            if self.rows and schedule is self.rows[-1]['_schedule']:
                self.rows[-1]['certified']=values.get('status') is ScheduleStatus.FEASIBLE
                self.rows[-1]['candidate_source']='ALNS_INITIALIZATION' if values.get('initialization') else 'ALNS_REFERENCE'
            return result
        self.patch(SearchStats,'record_reference',recorded)
        return self

    def __exit__(self,*exc):
        for obj,name,original in reversed(self.patches):setattr(obj,name,original)

    def serialize(self, entry):
        rows = []
        for raw in self.rows:
            row = {k: v for k, v in raw.items() if not k.startswith("_")}
            solution, config, schedule = raw["_solution"], raw["_config"], raw["_schedule"]
            if schedule.scope_hash != FORMAL_SCOPE_V2.scope_hash:
                raise ValueError("ZR replay artifacts require the frozen V2 scope")
            directions = schedule.directions
            row.update({"instance_id": entry["instance_id"], "instance_geometry_hash": entry["instance_geometry_hash"],
                "N": len(solution.parents), "tier": entry["tier"], "block_count": sum(len(r.block_ids) for r in solution.routes),
                "solution_canonical_hash": solution.canonical_hash, "direction_hash": digest(directions),
                "robot_route_lengths": [len(r.block_ids) for r in solution.routes],
                "WHOLE_count": sum(p.kind is SplitKind.WHOLE for p in solution.patterns),
                "X_count": sum(p.kind is SplitKind.X_SPLIT for p in solution.patterns),
                "Y_count": sum(p.kind is SplitKind.Y_SPLIT for p in solution.patterns),
                "replay": payload(solution, config, directions), "level": self.level})
            # Ordinary compound primary key from the requested identities; no
            # extra hash/contract is needed to identify a unique reference call.
            row["call_identity"] = ":".join((entry["instance_geometry_hash"], row["solution_canonical_hash"], row["direction_hash"]))
            if row["certified"] is None and not schedule.feasible:
                row["certified"] = False
            rows.append(json.loads(json.dumps(row)))
        return rows
