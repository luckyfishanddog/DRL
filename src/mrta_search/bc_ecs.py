"""Bottleneck/conflict-aware ejection chains after an in-run SAEAX seed.

Only the unchanged SAEAX initializer and one selected improvement can call
B32. Chain construction uses the existing numeric directed insertion kernel;
the initial feasible B32 result remains an in-memory fallback throughout.
"""
from __future__ import annotations

if __name__=='__main__':
    from pathlib import Path
    import sys
    _entry_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(_entry_root/'src'))
    sys.path.insert(0,str(_entry_root/'scripts'))
    from run_saeax import configure_cpu
    configure_cpu()

from dataclasses import dataclass, field
import itertools
import math
import random
import time

import numpy as np
from numba import njit

from mrta_baselines.hga import _HGAWorkspace
from mrta_baselines.wag_vns import _priority_arrays
from mrta_reference.model import OperationKind, ScientificConfig
from mrta_reference.scheduler import reference_schedule_formal
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map
from mrta_search.lns import (_remove_parent_set, schedule_feedback_summary,
    oriented_route_state, oriented_route_lower_bound, joint_block_candidates)
from mrta_search.saeax import (Individual, _fragment_position, _put_fragment,
    _heads_legal, _route_statistics, complete_identity, validate_individual, run_saeax)


@dataclass(frozen=True)
class BCECSConfig:
    max_candidates: int = 6000
    stagnation_candidates: int = 1500

    def __post_init__(self):
        for name in ('max_candidates','stagnation_candidates'):
            value = getattr(self,name)
            if not isinstance(value,int) or isinstance(value,bool) or value<=0:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class BCECSResult:
    status: str
    cmax: float | None
    solution: object | None
    directions: tuple | None
    schedule: object | None
    native_cmax: float | None
    runtime: float
    termination_reason: str
    accounting: dict
    initial_result: object
    progress: list = field(default_factory=list)
    accepted_actions: list = field(default_factory=list)
    final_candidate: dict | None = None


@njit
def assignment_loads(loads, nodes, owners, destinations, processing):
    result = loads.copy()
    for i,node in enumerate(nodes):result[owners[node]] -= processing[node]
    for i,node in enumerate(nodes):result[destinations[i]] += processing[node]
    return result


def cannot_improve(lower_bound, target, node_count):
    """Conservative floating guard; equal/near-equal bounds are NOT pruned.

    Processing and complete fixed-direction P+MOVE are lower bounds because
    neither evaluator can subtract processing, travel or add negative WAIT.
    No partial-route MOVE bound is used: subsequent insertion can reduce it.
    """
    guard = max(1.e-7,64*np.finfo(float).eps*(node_count+1)*max(1.,abs(lower_bound),abs(target)))
    return math.isfinite(target) and lower_bound>target+guard


def observed_native_waits(operations, workspace):
    waits = np.zeros(len(workspace.parents))
    for row in operations:
        for op in row:
            if int(op[8])==4 and 0<=int(op[7])<len(workspace.ids):
                waits[workspace.parent_index[int(op[7])]] += float(op[0])
    return waits


def chain_plans(item, workspace, waits, rng):
    """Finite, interleaved one-edge, two-edge and three-robot-cycle plans.

    The root is on max-P or max-(P+MOVE). A receiving robot can eject another
    parent to a third robot or exchange with the root. A three-edge cycle closes
    a three-robot chain. No parent can be removed twice within a plan.
    """
    arcs = [[] for _ in range(4)]
    for r,row in enumerate(item.routes):
        for node in row:
            node = int(node)
            for target in sorted((s for s in range(4) if s!=r and workspace.eligible[node,s]),
                                 key=lambda s:(item.processing[s],item.costs[s],s)):
                arcs[r].append((node,r,target))
    for r in range(4):
        ties = {node:rng.random() for node,_,_ in arcs[r]}
        arcs[r].sort(key=lambda a:(-bool(waits[workspace.parent_index[a[0]]]>0),
            -workspace.processing[a[0]],ties[a[0]],item.processing[a[2]],a))
    critical = list(dict.fromkeys((int(np.argmax(item.processing)),int(np.argmax(item.costs)))))
    roots = [a for r in critical for a in arcs[r]]
    def parents_differ(a,b):return workspace.parent_index[a[0]]!=workspace.parent_index[b[0]]
    def singles():
        for a in roots:yield (a,)
    def pairs():
        for a in roots:
            for b in arcs[a[2]]:
                if parents_differ(a,b):yield (a,b)
    def triples():
        for a in roots:
            for b in arcs[a[2]]:
                if b[2] in (a[1],a[2]) or not parents_differ(a,b):continue
                for c in arcs[b[2]]:
                    if c[2]==a[1] and parents_differ(a,c) and parents_differ(b,c):yield (a,b,c)
    pending = [iter(singles()),iter(pairs()),iter(triples())]
    while pending:
        retained = []
        for stream in pending:
            try:
                yield next(stream)
                retained.append(stream)
            except StopIteration:pass
        pending = retained


def chain_assignments(chain, item, workspace):
    """Close removal over parents; Y siblings retain independent eligibility."""
    parent_indices = tuple(sorted({int(workspace.parent_index[a[0]]) for a in chain}))
    nodes = np.array([int(n) for p in parent_indices for n in workspace.pattern_nodes[item.selected[p]] if n>=0],np.int64)
    forced = {node:target for node,_,target in chain}
    choices = [([forced[int(n)]] if int(n) in forced else
                [r for r in range(4) if workspace.eligible[n,r]]) for n in nodes]
    for assignment in itertools.product(*choices):
        yield parent_indices,nodes,np.array(assignment,np.int64)


def construct_chain(item, solution, workspace, parent_indices, nodes, destinations, chain, variant):
    """One atomic parent-closed repair with forced chain destinations.

    Two fixed construction orders are used, not another search framework.
    The existing numeric kernel jointly picks insertion position and the two
    directions; all unaffected directions stay fixed. Pending processing is
    reserved at its planned robot only for construction, never safe pruning.
    """
    if any(not workspace.eligible[node,robot] for node,robot in zip(nodes,destinations)):
        raise ValueError('illegal chain destination or fixed X ownership')
    parent_ids = tuple(workspace.parents[p].parent_id for p in parent_indices)
    partial = _remove_parent_set(solution,parent_ids)
    active = {workspace.ids[int(n)]:int(n) for row in item.routes for n in row}
    old_bits = {int(n):int(d) for row,bits in zip(item.routes,item.directions) for n,d in zip(row,bits)}
    routes = [np.array([active[name] for name in row],np.int64) for row in partial.routes]
    directions = [np.array([old_bits[int(n)] for n in row],np.int64) for row in routes]
    targets = {int(n):int(r) for n,r in zip(nodes,destinations)}
    forced_eligible = workspace.eligible.copy()
    remaining = np.zeros(4)
    for node,robot in targets.items():
        forced_eligible[node,:] = False
        forced_eligible[node,robot] = True
        remaining[robot] += workspace.processing[node]
    chain_order = {int(workspace.parent_index[a[0]]):i for i,a in enumerate(chain)}
    ordered = sorted((int(n) for n in nodes),key=lambda n:(
        (-workspace.processing[n],n) if variant==0 else
        (-chain_order[int(workspace.parent_index[n])],-workspace.processing[n],n)))
    for node in ordered:
        robot = targets[node]
        remaining[robot] -= workspace.processing[node]
        costs,_ = _route_statistics(routes,directions,workspace.travel,workspace.processing)
        fragment = np.array([node],np.int64)
        bits = np.array([old_bits[node] ^ variant],np.int64)
        position = _fragment_position(fragment,bits,routes,directions,remaining,costs,robot,
            workspace.travel,workspace.processing,forced_eligible,workspace.parent_index,
            workspace.points,workspace.parameters)
        if position[0]<0:return None
        _put_fragment(fragment,bits,routes,directions,position)
    try:validate_individual(workspace,routes,directions,item.selected)
    except ValueError:return None
    if not _heads_legal(routes,directions,workspace.points,workspace.parameters):return None
    return routes,directions


def chain_candidates(item, solution, workspace, waits, rng, target, accounting):
    owners = np.full(len(workspace.ids),-1,np.int64)
    for r,row in enumerate(item.routes):owners[row] = r
    lookup = {workspace.ids[int(n)]:int(n) for row in item.routes for n in row}
    blocks = block_map(solution,workspace.config)
    patterns = {p.parent_id:p for p in solution.patterns}
    attempted_assignments = set()
    for chain in chain_plans(item,workspace,waits,rng):
        accounting['chain_plans'] += 1
        for parents,nodes,destinations in chain_assignments(chain,item,workspace):
            identity = (parents,tuple((int(n),int(r)) for n,r in zip(nodes,destinations)))
            if identity in attempted_assignments:continue
            attempted_assignments.add(identity)
            accounting['assignment_combinations'] += 1
            projected = assignment_loads(item.processing,nodes,owners,destinations,workspace.processing)
            if cannot_improve(float(max(projected)),target,sum(map(len,item.routes))):
                accounting['processing_pruned'] += 1
                continue
            for variant in range(2):
                candidate = construct_chain(item,solution,workspace,parents,nodes,destinations,chain,variant)
                if candidate is None:
                    accounting['construction_rejected'] += 1
                    continue
                routes,bits = candidate
                # This is still one unscored joint chain construction. Enumerate
                # every root insertion location/direction (and the existing
                # primitive's immediate-neighbour flips), rather than scoring
                # a greedy chain and adding an independent refinement phase.
                root,_,destination = chain[0]
                names = tuple(tuple(workspace.ids[int(n)] for n in row if n!=root) for row in routes)
                vectors = tuple(tuple(int(d) for n,d in zip(row,db) if n!=root)
                    for row,db in zip(routes,bits))
                partial = oriented_route_state(names,vectors,blocks,workspace.config)
                block = blocks[workspace.ids[root]]
                options = joint_block_candidates(partial,block,patterns[block.parent_id],blocks,
                    workspace.config,insertion_limit=len(names[destination])+1,robot_id=destination)
                for completed,action in options:
                    accounting['complete_constructed'] += 1
                    lower_bound = oriented_route_lower_bound(completed)
                    if cannot_improve(lower_bound,target,sum(map(len,completed.routes))):
                        accounting['route_bound_pruned'] += 1
                        continue
                    rr = [np.array([lookup[name] for name in row],np.int64) for row in completed.routes]
                    dd = [np.array(row,np.int64) for row in completed.directions]
                    validate_individual(workspace,rr,dd,item.selected)
                    costs,loads = _route_statistics(rr,dd,workspace.travel,workspace.processing)
                    kind = ('SINGLE_TRANSFER' if len(chain)==1 else 'TWO_ROBOT_EXCHANGE'
                        if len(chain)==2 and chain[-1][2]==chain[0][1] else 'THREE_ROBOT_CHAIN')
                    detail = dict(kind=kind,parents=[workspace.parents[p].parent_id for p in parents],
                        transfers=[(workspace.ids[n],int(owners[n]),int(r)) for n,r in zip(nodes,destinations)
                            if owners[n]!=r],variant=variant,wait_anchored=any(waits[p]>0 for p in parents),
                        root_insertion=dict(block_id=action.block_id,robot=action.robot,
                            position=action.position,direction=action.direction,
                            neighbour_changes=action.neighbour_changes))
                    yield rr,dd,costs,loads,complete_identity(rr,dd,item.selected),detail


def run_bc_ecs(parents, config=None, settings=None, *, seed=20261110,
               scope=FORMAL_SCOPE_V2, progress_callback=None):
    config,settings = config or ScientificConfig(),settings or BCECSConfig()
    started = time.perf_counter()
    def initial_progress(row):
        if progress_callback is not None:progress_callback(dict(phase='SAEAX_INITIALIZATION',**row))
    initial = run_saeax(parents,config,seed=seed,scope=scope,progress_callback=initial_progress)
    initialization_s = time.perf_counter()-started
    accounting = dict(initialization_s=initialization_s,initial_saeax_runtime_s=initial.runtime,
        initial_native_calls=initial.accounting['native_calls'],initial_b32_calls=initial.accounting['b32_calls'],
        native_calls=initial.accounting['native_calls'],b32_calls=initial.accounting['b32_calls'],
        initial_b32_s=initial.accounting['b32_s'],b32_s=initial.accounting['b32_s'],
        bc_native_calls=0,final_b32_calls=0,final_b32_s=0.,bc_native_s=0.,chain_search_s=0.,
        chain_plans=0,assignment_combinations=0,processing_pruned=0,route_bound_pruned=0,
        construction_rejected=0,complete_constructed=0,duplicate_candidates=0,
        native_feasible_candidates=0,native_nonfinite_candidates=0,numeric_failures=0,
        accepted_improvements=0,wait_anchored_evaluations=0,stagnation_at_stop=0,
        single_transfer_evaluations=0,exchange_evaluations=0,three_robot_evaluations=0,
        initial_cmax_s=initial.cmax,final_fallback=False)
    if initial.status!='FEASIBLE' or initial.schedule is None:
        accounting.update(search_complete=False,return_reason='INITIAL_SAEAX_NOT_B32_FEASIBLE')
        return BCECSResult(initial.status,None,initial.solution,initial.directions,initial.schedule,
            initial.native_cmax,time.perf_counter()-started,'INITIALIZATION_FAILED',accounting,initial)
    chain_started = time.perf_counter()
    workspace = _HGAWorkspace(parents,config,scope)
    selected = np.array([workspace.patterns.index(p) for p in initial.solution.patterns],np.int64)
    routes = workspace.encode(initial.solution)
    bits = [np.array(row,np.int64) for row in initial.directions]
    validate_individual(workspace,routes,bits,selected)
    costs,loads = _route_statistics(routes,bits,workspace.travel,workspace.processing)
    identity = complete_identity(routes,bits,selected)
    native = initial.native_cmax if initial.native_cmax is not None else math.inf
    best = Individual(routes,bits,selected,native,costs,loads,identity,np.empty(0,np.int64))
    initial_identity,best_solution = identity,initial.solution
    feedback = schedule_feedback_summary(initial.solution,initial.schedule,(),config)
    waits = np.array([feedback['parent_wait'].get(p.parent_id,0.) for p in workspace.parents])
    accounting['initial_observed_b32_wait_s'] = float(sum(waits))
    seen,rng,progress,accepted = {identity:native},random.Random(seed),[],[]
    stale,termination = 0,'CHAIN_PLANS_EXHAUSTED'

    def snapshot():
        row = dict(phase='BC_ECS',evaluations=accounting['bc_native_calls'],stagnation=stale,
            native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
            max_processing_s=float(max(best.processing)),move_s=float(sum(best.costs-best.processing)),
            accepted=accounting['accepted_improvements'],elapsed_s=time.perf_counter()-started,
            processing_pruned=accounting['processing_pruned'],route_bound_pruned=accounting['route_bound_pruned'])
        progress.append(row)
        if progress_callback is not None:progress_callback(row)

    snapshot()
    stop = False
    while not stop:
        improved = False
        # Strict native acceptance is the search objective. The same exact
        # physical bounds also exclude candidates unable to beat initial B32.
        target = min(best.native_cmax,initial.cmax)
        for rr,dd,cc,pp,key,detail in chain_candidates(best,best_solution,workspace,waits,rng,target,accounting):
            if key in seen:
                accounting['duplicate_candidates'] += 1
                continue
            t = time.perf_counter()
            value,_,operations,_ = _priority_arrays(rr,workspace.travel,workspace.processing,
                workspace.points,workspace.parameters,dd)
            accounting['bc_native_s'] += time.perf_counter()-t
            accounting['bc_native_calls'] += 1
            accounting['native_calls'] += 1
            seen[key] = float(value)
            stale += 1
            counter = dict(SINGLE_TRANSFER='single_transfer_evaluations',
                TWO_ROBOT_EXCHANGE='exchange_evaluations',THREE_ROBOT_CHAIN='three_robot_evaluations')[detail['kind']]
            accounting[counter] += 1
            accounting['wait_anchored_evaluations'] += int(detail['wait_anchored'])
            if math.isnan(value):
                accounting['numeric_failures'] += 1
                termination,stop = 'NATIVE_NUMERIC_FAILURE',True
            elif math.isfinite(value):
                accounting['native_feasible_candidates'] += 1
                if value<best.native_cmax-1.e-8:
                    previous = best.native_cmax
                    best = Individual(rr,dd,selected,float(value),cc,pp,key,np.empty(0,np.int64))
                    best_solution = workspace.solution(rr,selected)
                    expected = tuple(tuple(workspace.ids[n] for n in row) for row in rr)
                    if (tuple(r.block_ids for r in best_solution.routes)!=expected or
                        best_solution.patterns!=tuple(workspace.patterns[p] for p in selected)):
                        raise RuntimeError('canonicalization changed an accepted chain')
                    waits = observed_native_waits(operations,workspace)
                    accounting['accepted_improvements'] += 1
                    accepted.append(dict(detail,evaluation=accounting['bc_native_calls'],
                        native_before_s=previous if math.isfinite(previous) else None,native_after_s=float(value),
                        processing_s=pp.tolist(),move_s=float(sum(cc-pp)),observed_native_wait_s=float(sum(waits))))
                    stale,improved = 0,True
            else:accounting['native_nonfinite_candidates'] += 1
            if accounting['bc_native_calls']>=settings.max_candidates:
                termination,stop = 'CANDIDATE_LIMIT',True
            elif stale>=settings.stagnation_candidates:
                termination,stop = 'STAGNATION',True
            if improved or accounting['bc_native_calls']%500==0:snapshot()
            if stop or improved:break
        if not improved and not stop:
            # Every finite chain/assignment and both construction orders was
            # visited. Do not invent evaluations by recycling identical plans.
            termination,stop = 'CHAIN_PLANS_EXHAUSTED',True
    if progress[-1]['evaluations']!=accounting['bc_native_calls']:snapshot()
    accounting.update(chain_search_s=time.perf_counter()-chain_started,stagnation_at_stop=stale,
        search_best_native_cmax_s=best.native_cmax if math.isfinite(best.native_cmax) else None,
        search_best_processing_s=best.processing.tolist(),search_best_move_s=float(sum(best.costs-best.processing)))
    chosen,final_record = initial,None
    if best.identity!=initial_identity and best.native_cmax<native-1.e-8:
        vectors = tuple(tuple(int(d) for d in row) for row in best.directions)
        t = time.perf_counter()
        schedule = reference_schedule_formal(best_solution,config,scope=scope,orientations=dict(enumerate(vectors)))
        elapsed = time.perf_counter()-t
        accounting.update(final_b32_calls=1,final_b32_s=elapsed,
            b32_calls=accounting['b32_calls']+1,b32_s=accounting['b32_s']+elapsed)
        final_record = dict(status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,
            native_cmax_s=best.native_cmax,max_processing_s=float(max(best.processing)),
            processing_s=best.processing.tolist(),move_s=float(sum(best.costs-best.processing)),
            wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
            wait_is_partial=not schedule.feasible,b32_s=elapsed,
            diagnostic=' | '.join(schedule.diagnostics[:2]))
        final_record['solution'] = dict(patterns=[dict(parent_id=p.parent_id,kind=p.kind.value,t=p.t,
            point_id=p.point_id,rail=p.rail.value if p.rail else None) for p in best_solution.patterns],
            routes=[list(r.block_ids) for r in best_solution.routes],directions=vectors)
        if schedule.feasible and schedule.cmax<initial.cmax-1.e-8:
            if schedule.directions!=vectors:raise RuntimeError('B32 changed explicit chain directions')
            accounting['return_reason'] = 'IMPROVED_B32'
            accounting.update(processing_s=best.processing.tolist(),move_s=final_record['move_s'],wait_s=final_record['wait_s'])
            accounting.update(search_complete=termination!='NATIVE_NUMERIC_FAILURE',
                search_s=initial.accounting['search_s']+accounting['chain_search_s'])
            return BCECSResult('FEASIBLE',schedule.cmax,best_solution,vectors,schedule,best.native_cmax,
                time.perf_counter()-started,termination,accounting,initial,progress,accepted,final_record)
        accounting['final_fallback'] = True
        accounting['return_reason'] = 'FINAL_B32_FAILED' if not schedule.feasible else 'INITIAL_B32_BETTER_OR_EQUAL'
    else:accounting['return_reason'] = 'NO_NATIVE_IMPROVEMENT'
    accounting.update(processing_s=list(initial.accounting['processing_s']),
        move_s=initial.accounting['move_s'],wait_s=initial.accounting['wait_s'],
        search_complete=termination!='NATIVE_NUMERIC_FAILURE',
        search_s=initial.accounting['search_s']+accounting['chain_search_s'])
    return BCECSResult('FEASIBLE',chosen.cmax,chosen.solution,chosen.directions,chosen.schedule,
        chosen.native_cmax,time.perf_counter()-started,termination,accounting,initial,progress,accepted,final_record)


# Shared experiment functions for the script and the existing file entry.
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from run_saeax import configure_cpu, load_layouts, read_rows, append_result, paired_summary, baseline_rows, SEEDS, SOURCE as SAEAX_SOURCE
configure_cpu()
from mrta_search.saeax import SAEAXConfig

SOURCE = 'BC_ECS_V1'
PILOTS = ('train_15','train_34','train_28')


def solution_record(solution,directions):
    if solution is None:return None
    return dict(patterns=[dict(parent_id=p.parent_id,kind=p.kind.value,t=p.t,point_id=p.point_id,
        rail=p.rail.value if p.rail else None) for p in solution.patterns],
        routes=[list(r.block_ids) for r in solution.routes],directions=directions)


def experiment_job(job):
    from numba import set_num_threads
    set_num_threads(1)
    stage,(name,parents),seed = job
    print('BC_START',stage,name,len(parents),seed,flush=True)
    def report(row):print('BC_PROGRESS',name,seed,json.dumps(row),flush=True)
    started = time.perf_counter()
    try:
        result = run_bc_ecs(parents,seed=seed,progress_callback=report)
    except Exception as error:
        row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,status='ERROR',
            runtime_s=time.perf_counter()-started,source=SOURCE,termination_reason='ERROR',
            diagnostic=json.dumps(dict(settings=asdict(BCECSConfig()),
                error=f'{type(error).__name__}: {error}'),ensure_ascii=False))
        print('BC_ERROR',name,seed,row['diagnostic'],flush=True)
        return row
    a,initial = result.accounting,result.initial_result
    detail = dict(settings=asdict(BCECSConfig()),accounting=a,progress=result.progress,
        accepted_actions=result.accepted_actions,final_candidate=result.final_candidate,
        solution=solution_record(result.solution,result.directions),
        initial_saeax=dict(settings=asdict(SAEAXConfig()),status=initial.status,cmax_s=initial.cmax,
            native_cmax_s=initial.native_cmax,runtime_s=initial.runtime,generations=initial.generations,
            valid_children=initial.valid_children,termination_reason=initial.termination_reason,
            accounting=initial.accounting,progress=initial.progress,final_candidates=initial.final_candidates,
            solution=solution_record(initial.solution,initial.directions)))
    loads = a.get('processing_s')
    # BC-ECS has no generations or offspring; the existing generic CSV slot
    # valid_children records completed BC native candidates. Initial SAEAX
    # generations/offspring and both phases' calls remain explicit above.
    row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,status=result.status,
        cmax_s=result.cmax,native_cmax_s=result.native_cmax,runtime_s=result.runtime,generations='',
        valid_children=a['bc_native_calls'],native_calls=a['native_calls'],b32_calls=a['b32_calls'],
        termination_reason=result.termination_reason,max_processing_s=max(loads) if loads else None,
        move_s=a.get('move_s'),wait_s=a.get('wait_s'),source=SOURCE,
        diagnostic=json.dumps(detail,ensure_ascii=False,allow_nan=False))
    if loads:row.update({f'processing_r{r}_s':p for r,p in enumerate(loads)})
    print('BC_RESULT',json.dumps({k:v for k,v in row.items() if k!='diagnostic'},ensure_ascii=False),flush=True)
    return row


def method_rows(stage):
    return [r for r in read_rows() if r['stage']==stage and r['source']==SOURCE]


def saeax_rows(stage):
    return {(r['layout_id'],int(r['seed'])):r for r in read_rows()
            if r['stage']==stage and r['source']==SAEAX_SOURCE}


def pilot_decision():
    rows = method_rows('pilot')
    expected = {(name,SEEDS[0]) for name in PILOTS}
    paired = paired_summary(rows,saeax_rows('pilot'))
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    complete = (len(rows)==3 and {(r['layout_id'],int(r['seed'])) for r in rows}==expected and
        all(json.loads(r['diagnostic']).get('accounting',{}).get('search_complete',False) for r in rows))
    feasible = sum(r['status']=='FEASIBLE' for r in rows)
    return dict(complete=complete,feasible=feasible,paired_saeax=paired,runtime_median_s=runtime,
        passed=(complete and feasible==3 and paired['pairs']==3 and paired['wins']>=2 and runtime<=103.59))


def validation_decision():
    rows = method_rows('validation')
    expected = {(name,seed) for name,_ in load_layouts('validation') for seed in SEEDS}
    paired = {method:paired_summary(rows,baseline_rows(method)) for method in ('WAG+VNS','HGA')}
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    complete = (len(rows)==30 and {(r['layout_id'],int(r['seed'])) for r in rows}==expected and
        all(json.loads(r['diagnostic']).get('accounting',{}).get('search_complete',False) for r in rows))
    feasible = sum(r['status']=='FEASIBLE' for r in rows)
    quality = all(p['baseline_runs']==30 and p['pairs']==p['baseline_feasible'] and
        p['median_cmax_change_pct'] is not None and p['median_cmax_change_pct']<0 and p['wins']>p['losses']
        for p in paired.values())
    return dict(complete=complete,feasible=feasible,paired=paired,runtime_median_s=runtime,
        passed=(complete and feasible==30 and quality and runtime<=103.59))


def summarize(stage):
    rows = method_rows(stage)
    if not rows:return None
    feasible = [r for r in rows if r['status']=='FEASIBLE']
    accounts = [json.loads(r['diagnostic']).get('accounting',{}) for r in rows]
    def timing(key):
        values = [a[key] for a in accounts if key in a]
        return median(values) if values else None
    summary = dict(stage=stage,runs=len(rows),feasible=len(feasible),
        statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        cmax_median_s=median(float(r['cmax_s']) for r in feasible) if feasible else None,
        cmax_mean_s=mean(float(r['cmax_s']) for r in feasible) if feasible else None,
        runtime_median_s=median(float(r['runtime_s']) for r in rows),
        runtime_mean_s=mean(float(r['runtime_s']) for r in rows),
        initialization_median_s=timing('initialization_s'),chain_search_median_s=timing('chain_search_s'),
        search_median_s=timing('search_s'),b32_median_s=timing('b32_s'),
        final_fallbacks=sum(a.get('final_fallback',False) for a in accounts),
        SAEAX=paired_summary(rows,saeax_rows(stage)))
    for method in ('WAG+VNS','HGA'):summary[method] = paired_summary(rows,baseline_rows(method))
    summary['decision'] = pilot_decision() if stage=='pilot' else validation_decision()
    print('BC_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    for r,a in sorted(zip(rows,accounts),key=lambda ra:(ra[0]['layout_id'],ra[0]['seed'])):
        print('BC_ROW',json.dumps(dict(layout_id=r['layout_id'],seed=r['seed'],status=r['status'],
            cmax_s=r['cmax_s'],max_processing_s=r['max_processing_s'],move_s=r['move_s'],wait_s=r['wait_s'],
            runtime_s=r['runtime_s'],bc_native_calls=a.get('bc_native_calls'),
            initial_native_calls=a.get('initial_native_calls'),b32_calls=r['b32_calls'],
            chain_search_s=a.get('chain_search_s'),termination_reason=r['termination_reason'],
            return_reason=a.get('return_reason')),ensure_ascii=False),flush=True)
    return summary


def run_experiment(stage,workers):
    if stage=='validation':
        decision = pilot_decision()
        print('BC_TRAIN_CHECK',json.dumps(decision),flush=True)
        if not decision['passed']:raise ValueError('TRAIN screening has not passed; VALIDATION is not run')
    layouts = load_layouts('train' if stage=='pilot' else 'validation')
    if stage=='pilot':
        lookup = dict(layouts)
        jobs = [(stage,(name,lookup[name]),SEEDS[0]) for name in PILOTS]
    else:jobs = [(stage,layout,seed) for layout in layouts for seed in SEEDS]
    completed = {(r['layout_id'],int(r['seed'])) for r in method_rows(stage)}
    pending = [job for job in jobs if (job[1][0],job[2]) not in completed]
    print('BC_BATCH',stage,'completed',len(jobs)-len(pending),'remaining',len(pending),'workers',workers,
        'settings',json.dumps(asdict(BCECSConfig())),flush=True)
    started = time.perf_counter()
    if workers==1:
        for job in pending:append_result(experiment_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(experiment_job,job) for job in pending]
            for future in as_completed(futures):append_result(future.result())
    print('BC_BATCH_SECONDS',time.perf_counter()-started,flush=True)
    return summarize(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pilot','validation','summary'))
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args(argv)
    if args.workers<1:parser.error('--workers must be positive')
    if args.stage=='summary':
        for stage in ('pilot','validation'):summarize(stage)
    else:run_experiment(args.stage,args.workers)


if __name__=='__main__':
    main()

