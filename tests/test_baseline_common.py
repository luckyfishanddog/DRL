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
