"""Frozen GNN construction and parent-mode reconstruction experiments."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import run_gnn_shared_construction as shared
from mrta_reference.model import ScheduleStatus
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map
from mrta_search.gnn_constructor import (
    construct_or_repair_with_gnn,fixed_split_patterns,joint_geometry,
)

CSV=ROOT/'results/gnn_multi_initial.csv'
REPORT=ROOT/'docs/GNN_MULTI_INITIAL.md'
CONFIG=shared.CONFIG
EMPTY=shared.EMPTY


def check_complete(result):
    """Basic executable structure checks, without any extra evaluator."""
    blocks=block_map(result.solution,CONFIG,scope=FORMAL_SCOPE_V2)
    actual=[b for route in result.solution.routes for b in route.block_ids]
    if Counter(actual)!=Counter(blocks.keys()) or len(result.directions)!=4:
        raise ValueError('incomplete/duplicate block coverage or direction vectors')
    for route,bits in zip(result.solution.routes,result.directions):
        if len(route.block_ids)!=len(bits) or any(d not in (0,1) for d in bits):
            raise ValueError('route and explicit direction mismatch')
    groups=[]
    for action in result.actions:
        parent=blocks[action.block_id].parent_id
        if not groups or groups[-1][0]!=parent:groups.append((parent,[]))
        groups[-1][1].append(action.block_id)
    if len({p for p,_ in groups})!=len(groups):
        raise ValueError('parent insertion was not atomic')
    for parent,children in groups:
        expected=sorted(b for b,block in blocks.items() if block.parent_id==parent)
        if children!=expected:
            raise ValueError('parent children were not completely inserted in block order')


def neural_multi4(parents,model,label,phase,instance,seed):
    """Return both nested-budget rows and the actual best scheduled candidate."""
    started=time.perf_counter()
    geometry=joint_geometry(parents,fixed_split_patterns(parents,CONFIG),CONFIG)
    pending=tuple(p.parent_id for p in parents)
    outcomes=[];scheduled_by_routes={};calls=0;greedy_row=None
    paths=[('greedy',seed)]+[('sample',seed+i) for i in (1,2,3)]
    for index,(mode,path_seed) in enumerate(paths):
        result=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,pending,model,CONFIG,
                                          insertion_limit=8,decode_mode=mode,seed=path_seed)
        scheduled=None;called=0;duplicate=False
        status='CONSTRUCTION_FAILED';cmax=None
        if result.solution is not None:
            check_complete(result)
            key=(tuple(r.block_ids for r in result.solution.routes),result.directions)
            if key in scheduled_by_routes:
                scheduled=scheduled_by_routes[key];duplicate=True
            else:
                scheduled=shared.pilot.oriented.schedule_explicit(result.solution,result.directions,CONFIG)
                scheduled_by_routes[key]=scheduled;calls+=1;called=1
            status=scheduled.schedule.status.value
            if scheduled.schedule.status is ScheduleStatus.FEASIBLE:cmax=scheduled.schedule.cmax
        outcomes.append(scheduled)
        if index==0:
            greedy_row=dict(phase=phase,instance=instance,method=label+'-Greedy',status=status,best_cmax=cmax,
                            b32_calls=calls,runtime_s=time.perf_counter()-started,unique_routes=len(scheduled_by_routes))
        print('CANDIDATE',phase,instance,label,index+1,mode,'seed',path_seed,status,'Cmax',cmax,
              'B32',called,'duplicate',duplicate,'actions',len(result.actions),
              'failure',result.rejection_reason,flush=True)
    feasible=[s for s in outcomes if s is not None and s.schedule.status is ScheduleStatus.FEASIBLE]
    best=min(feasible,key=lambda s:s.schedule.cmax) if feasible else None
    multi_row=dict(phase=phase,instance=instance,method=label+'-Multi4',status='FEASIBLE' if best else 'FAILED',
                   best_cmax=None if best is None else best.schedule.cmax,b32_calls=calls,
                   runtime_s=time.perf_counter()-started,unique_routes=len(scheduled_by_routes))
    if greedy_row['status']=='FEASIBLE' and (best is None or best.schedule.cmax>greedy_row['best_cmax']):
        raise ValueError('multi4 lost its own feasible greedy candidate')
    return greedy_row,multi_row,best


def save_results(rows,seed):
    with CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=('phase','instance','method','status','best_cmax','b32_calls','runtime_s','unique_routes'))
        writer.writeheader();writer.writerows(rows)
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    dev=[r for r in rows if r['phase']=='DEV']
    lines=['# GNN 四候选联合初解实验','',
           f'冻结原V2与Shared checkpoint，model.eval() + torch.inference_mode()；无训练或归一化更新。seed={seed}，候选1为原greedy，候选2—4采样种子为{seed+1}/{seed+2}/{seed+3}；父选择与联合插入均直接使用softmax(logits)，温度固定1，insertion_limit=8。',
           '全部神经候选从纯几何、固定合法模式和四条空路线开始；保持原子子块插入及同一步邻块方向翻转。每个完成且不同的路线+方向只调用一次现有B32；重复候选复用结果，四次解码之后不再补采。Greedy行直接复用Multi4的第一个候选，并非另外再运行一条。','',
           '## TRAIN 正确性运行','',
           '从已有TRAIN构造数据按N选择15/53/90三个规模，仍通过原MLP_TRAIN白名单加载PPO原几何。所有完成候选检查父焊缝覆盖、子块原子性及方向长度，随后实际调用B32。不同路线数同时考虑显式方向。','',
           '|TRAIN几何|方法|可调度|最佳Cmax/s|B32调用|构造+B32秒|不同完整路线/方向|',
           '|---|---|---|---:|---:|---:|---:|']
    for row in rows:
        if row['phase']=='TRAIN':
            lines.append(f"|{row['instance']}|{row['method']}|{row['status']}|{fmt(row['best_cmax'])}|{row['b32_calls']}|{row['runtime_s']:.3f}|{row['unique_routes']}|")
    lines+=['','## 六 DEV 结果','',
            '模型统一预加载，Excel读取不计时；计时包含几何模式/图、四次神经构造和实际B32。Beam保留现有实现，其4或8次B32预算如实列出。Beam在神经候选完成后独立运行，归属、路线与方向均未提供给GNN。不同预算不能称为严格等成本比较。','',
            '|方法|至少一条可调度/实例数|B32调用合计|构造+B32合计秒|','|---|---:|---:|---:|']
    for method in ('Beam-Init','GNN-V2-Greedy','GNN-V2-Multi4','GNN-Shared-Greedy','GNN-Shared-Multi4'):
        group=[r for r in dev if r['method']==method]
        if group:
            lines.append(f"|{method}|{sum(r['status']=='FEASIBLE' for r in group)}/{len(group)}|{sum(r['b32_calls'] for r in group)}|{sum(r['runtime_s'] for r in group):.3f}|")
    lines+=['','|实例|方法|结果|最佳真实Cmax/s|B32调用|构造+B32秒|不同完整路线/方向|','|---|---|---|---:|---:|---:|---:|']
    for row in dev:
        unique='—' if row['unique_routes'] is None else str(row['unique_routes'])
        lines.append(f"|{row['instance']}|{row['method']}|{row['status']}|{fmt(row['best_cmax'])}|{row['b32_calls']}|{row['runtime_s']:.3f}|{unique}|")
    lines+=['','DEADLOCK只表示现有B32在该完整方案与方向下未找到调度，不是数学不可行证明。失败无虚构Cmax，无Beam或Regret修复/回退。','']
    if len(dev)==30:
        lookup={(r['instance'],r['method']):r for r in dev}
        lines+=['## 判断','']
        multi_success={}
        for label in ('GNN-V2','GNN-Shared'):
            group=[r for r in dev if r['method']==label+'-Multi4']
            greedy=[lookup[r['instance'],label+'-Greedy'] for r in group]
            gained=[r['instance'] for r,g in zip(group,greedy) if r['status']=='FEASIBLE' and g['status']!='FEASIBLE']
            lowered=[r['instance'] for r,g in zip(group,greedy) if g['status']=='FEASIBLE' and r['best_cmax']<g['best_cmax']-1e-9*max(1,g['best_cmax'])]
            multi_success[label]=sum(r['status']=='FEASIBLE' for r in group)
            lines.append(f"- {label}：Greedy {sum(r['status']=='FEASIBLE' for r in greedy)}/6 → Multi4 {multi_success[label]}/6；新增可调度实例 {gained or '无'}；原Greedy已可调度实例中进一步降低Cmax {lowered or '无'}。实际不同完整路线/方向总数 {sum(r['unique_routes'] for r in group)}/24。")
        lines+=['','本轮固定四候选，未继续增加采样、调整温度、改变模型或进行强化学习。研究路线按本次真实成功数、Cmax与成本作出判断。']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--diagnostic',choices=('split','wait'),
                        help='Run consumed-DEV diagnostics to stdout; do not write results or models.')
    parser.add_argument('--mode-study',choices=('dev','independent','ppo-validation','matched',
                                               'neighborhood','neighborhood-summary','rank-diagnostic',
                                               'rank-validation','rank-summary','rank-audit','pair-diagnostic',
                                               'pair-extension','pair-summary','pair-audit','diversity-diagnostic',
                                               'diversity-online','diversity-summary','diversity-audit',
                                               'sparse-online','sparse-summary','sparse-audit','sparse-constructor-diagnostic',
                                               'sparse-recovery-budget','recovery-stratified','recovery-summary',
                                               'bound-shadow','bound-online','bound-summary','fresh-e2e','fresh-summary','summary'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--seconds',type=float,default=30.)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode_study:
        run_mode_study(args)
        return
    if args.diagnostic:
        run_diagnostic(args)
        return
    models={label:shared.load_model(path)[0].eval() for label,path in
            (('GNN-V2',shared.pilot.MODEL),('GNN-Shared',shared.MODEL))}
    saved_states={label:{k:v.clone() for k,v in model.state_dict().items()} for label,model in models.items()}
    records=[json.loads(l) for l in shared.DATA.read_text(encoding='utf-8').splitlines() if l]
    whitelist={e['instance_id']:e for group in shared.pilot.dataset.train_sheet_candidates().values() for e in group}
    rows=[]
    for record in shared.ordered_sources(records,3)[:3]:
        entry=whitelist[record['entry']['instance_id']]
        parents=shared.pilot.dataset.load_parents(entry)
        for label,model in models.items():
            greedy,multi,_=neural_multi4(parents,model,label,'TRAIN',entry['instance_id'],args.seed)
            rows.extend((greedy,multi));save_results(rows,args.seed)
    print('TRAIN THREE SCALES COMPLETE; beginning fixed six DEV',flush=True)
    historical=list(csv.DictReader(shared.CSV.open(encoding='utf-8-sig')))
    dev={e['sheet_name']:e for e in shared.pilot.entries()[1]}
    for name in shared.pilot.oriented.NAMES:
        parents=shared.pilot.dataset.load_parents(dev[name])
        for label,model in models.items():
            greedy,multi,_=neural_multi4(parents,model,label,'DEV',name,args.seed)
            old=next(r for r in historical if r['stage']=='DEV_INIT' and r['instance']==name and r['method']==label+'-Init')
            if greedy['status']!=old['status'] or (greedy['best_cmax'] is not None and abs(greedy['best_cmax']-float(old['final_cmax']))>1e-6):
                raise ValueError('default greedy no longer reproduces the previous single-path result')
            rows.extend((greedy,multi));save_results(rows,args.seed)
        start=time.perf_counter()
        beam,attempts=shared.pilot.oriented.common_initial_beam(parents,CONFIG,seed=args.seed)
        elapsed=time.perf_counter()-start
        if beam is not None and beam.solution.patterns!=fixed_split_patterns(parents,CONFIG):
            raise ValueError('Beam and GNN fixed patterns differ')
        rows.append(dict(phase='DEV',instance=name,method='Beam-Init',status='FEASIBLE' if beam else 'FAILED',
                    best_cmax=None if beam is None else beam.schedule.cmax,
                    b32_calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts),
                    runtime_s=elapsed,unique_routes=None))
        print('BEAM',name,rows[-1]['best_cmax'],'B32',rows[-1]['b32_calls'],'seconds',elapsed,flush=True)
        save_results(rows,args.seed)
    for label,model in models.items():
        if not all(torch.equal(value,model.state_dict()[key]) for key,value in saved_states[label].items()):
            raise ValueError('frozen model weights or normalization changed')
    print('DONE: frozen checkpoints unchanged; greedy reproduced; four fixed neural paths per instance/model.',flush=True)


def pattern_repair_diagnostic(initial,model,config):
    """Explore legal modes of three load-critical parents on one frozen seed.

    This is a bounded development neighborhood, not a learned split head.
    Every trial removes all old children of one parent. Unchanged-pattern
    trials separate ordinary reinsertion from gains requiring another mode.
    The incumbent is retained and all feasible trials are independently checked.
    """
    from mrta_reference.geometry import build_legal_pattern_catalog
    from mrta_reference.certifier import certify_schedule

    started=time.perf_counter()
    blocks=block_map(initial.solution,config,scope=FORMAL_SCOPE_V2)
    catalog=build_legal_pattern_catalog(initial.solution.parents,config,FORMAL_SCOPE_V2)
    old={p.parent_id:p for p in initial.solution.patterns}
    critical=max(range(4),key=lambda r:initial.schedule.robot_completion[r])
    loads={}
    for b in initial.solution.routes[critical].block_ids:
        pid=blocks[b].parent_id
        loads[pid]=loads.get(pid,0.)+config.process_time(blocks[b].length)
    selected=sorted((pid for pid in loads if len(catalog[pid])>1),
                    key=lambda pid:(-loads[pid],pid))[:3]
    trials=[]
    best=initial
    for pid in selected:
        routes=tuple(tuple(b for b in route.block_ids if blocks[b].parent_id!=pid)
                     for route in initial.solution.routes)
        directions=tuple(tuple(d for b,d in zip(route.block_ids,bits) if blocks[b].parent_id!=pid)
                         for route,bits in zip(initial.solution.routes,initial.directions))
        for option in catalog[pid]:
            patterns=tuple(option if p.parent_id==pid else p for p in initial.solution.patterns)
            geometry=joint_geometry(initial.solution.parents,patterns,config)
            result=construct_or_repair_with_gnn(geometry,routes,directions,(pid,),model,config,
                                              insertion_limit=8)
            trial=dict(parent=pid,pattern=option.pattern_id,changed=option!=old[pid],
                       status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,certified=False)
            if result.solution is not None:
                candidate=shared.pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                trial.update(status=candidate.schedule.status.value,b32_calls=1)
                if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                    certificate=certify_schedule(result.solution,candidate.schedule,config,scope=FORMAL_SCOPE_V2)
                    if not certificate.certified:
                        raise ValueError(f'pattern-repair certification failed: {certificate.errors}')
                    trial.update(cmax=candidate.schedule.cmax,certified=True)
                    if candidate.schedule.cmax<best.schedule.cmax-1e-9*max(1.,best.schedule.cmax):
                        best=candidate
            trials.append(trial)
    def summary(changed):
        group=[r for r in trials if r['changed']==changed]
        feasible=[r for r in group if r['cmax'] is not None]
        return dict(attempts=len(group),b32_calls=sum(r['b32_calls'] for r in group),
                    feasible=len(feasible),improved=sum(r['cmax']<initial.schedule.cmax-1e-7 for r in feasible),
                    best=min(feasible,key=lambda r:r['cmax']) if feasible else None)
    return dict(initial_cmax=initial.schedule.cmax,best_cmax=best.schedule.cmax,
                selected_parents=selected,fixed=summary(False),changed=summary(True),
                runtime_s=time.perf_counter()-started,trials=trials),best


def run_diagnostic(args):
    """Reuse consumed DEV and existing checkpoints without creating files."""
    import contextlib
    import io
    from mrta_reference.certifier import certify_schedule

    pilot=shared.pilot
    model,_=shared.load_model(pilot.MODEL)
    dev={e['sheet_name']:e for e in pilot.entries()[1]}
    for name in pilot.oriented.NAMES:
        started=time.perf_counter()
        initial,attempts=pilot.oriented.common_initial_beam(pilot.dataset.load_parents(dev[name]),CONFIG,seed=args.seed)
        if initial is None:
            print('DIAGNOSTIC',json.dumps(dict(instance=name,status='INITIALIZATION_FAILED')),flush=True)
            continue
        certificate=certify_schedule(initial.solution,initial.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
        if not certificate.certified:
            raise ValueError(certificate.errors)
        init_info=dict(initial_runtime_s=time.perf_counter()-started,
                       initial_b32_calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts))
        if args.diagnostic=='split':
            row,_=pattern_repair_diagnostic(initial,model,CONFIG)
            print('SPLIT_DIAGNOSTIC',json.dumps(dict(instance=name,seed=args.seed,**init_info,**row)),flush=True)
            continue
        # Independent validation is applied identically to both audit arms;
        # the legacy public experiment and scheduler are left unchanged.
        original=pilot.oriented.schedule_explicit
        certification_count=0
        def checked_schedule(solution,directions,config):
            nonlocal certification_count
            candidate=original(solution,directions,config)
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                check=certify_schedule(solution,candidate.schedule,config,scope=FORMAL_SCOPE_V2)
                if not check.certified:
                    raise ValueError(check.errors)
                certification_count+=1
            return candidate
        pilot.oriented.schedule_explicit=checked_schedule
        try:
            for label,selectors in (('GNN-V2',pilot.SELECTORS),
                                    ('GNN-V2-WAIT',pilot.SELECTORS+(pilot.BottleneckSelection.WAIT_ANCHORED,))):
                certification_count=0
                with contextlib.redirect_stdout(io.StringIO()):
                    row=pilot.iterative_group(dev[name],initial,label,model,args,selectors=selectors)
                trace=json.loads(row.pop('trace'))
                row.update(certified=certification_count,**init_info,
                           wait_attempts=sum(t['selection']=='WAIT_ANCHORED' for t in trace),
                           wait_fallbacks=sum(t.get('selection_fallback')=='NO_POSITIVE_WAIT' for t in trace),
                           accepted_steps=[{k:t[k] for k in ('iteration','selection','q','before','candidate_cmax')}
                                           for t in trace if t['accepted']])
                print('WAIT_DIAGNOSTIC',json.dumps(row),flush=True)
        finally:
            pilot.oriented.schedule_explicit=original


STUDY_PATH=ROOT/'results/parent_mode_reconstruction_20261030.jsonl'
STUDY_VERSION='parent-mode-v4-outer'
STUDY_ARMS=('FIXED_NEW','MODE_NEW','FIXED_GNN','MODE_GNN','ADAPT_NEW')


def _study_append(row):
    # The user explicitly authorized this one new result file. Geometry,
    # parameters and run traces share it; no workbook/checkpoint is created.
    with STUDY_PATH.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def _study_rows():
    return ([json.loads(line) for line in STUDY_PATH.read_text(encoding='utf-8').splitlines() if line]
            if STUDY_PATH.exists() else [])


def _mode_worker_init():
    torch.set_num_threads(1)


def _mode_search(parents,method,seed,budget,max_iterations):
    from dataclasses import asdict,replace
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import rank_parent_mode_changes,select_bottleneck_parents,repair_partial_state_oriented

    started=time.perf_counter()
    pilot=shared.pilot
    model=shared.load_model(pilot.MODEL)[0] if method.endswith('GNN') else None
    certificates=0
    initial_events=[]
    def accept_initial(candidate):
        nonlocal certificates
        check=certify_schedule(candidate.solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
        certificates+=1
        if not check.certified:raise ValueError(check.errors)
        elapsed=time.perf_counter()-started
        if elapsed>budget:return False
        if not initial_events or candidate.schedule.cmax<initial_events[-1][1]:
            initial_events.append([elapsed,candidate.schedule.cmax,'INITIAL'])
        return True
    current,attempts=pilot.oriented.common_initial_beam(parents,CONFIG,seed=seed,
        deadline=started+budget,accept_candidate=accept_initial,interleave=True)
    calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
    init_elapsed=time.perf_counter()-started
    row=dict(method=method,initial_cmax=None if current is None else current.schedule.cmax,
             initial_attempts=attempts,
             initialization_seconds=init_elapsed,initial_reference_calls=calls,reference_calls=calls,
             certifier_calls=certificates,accepted=0,mode_accepted=0,iterations=0,trace=[],best_events=initial_events)
    if current is None:
        return dict(row,status='TIME_LIMIT_NO_FEASIBLE' if init_elapsed>budget else 'INITIALIZATION_FAILED',
                    cmax=None,actual_runtime=init_elapsed,overshoot=max(0.,init_elapsed-budget),solution=None)
    iteration=0
    mode_period=2
    unsuccessful_modes=0
    last_mode_iteration=-1
    while iteration<max_iterations and time.perf_counter()-started<budget:
        local_seed=seed+iteration
        selector=pilot.SELECTORS[iteration%3]
        q=min(len(parents),(4,8)[(iteration//3)%2])
        partial=select_bottleneck_parents(current.solution,current.directions,current.schedule,selector,
                                          q=q,seed=local_seed,config=CONFIG)
        changed=[]
        try_mode=(method.startswith('MODE') and iteration%2==1) or (
            method=='ADAPT_NEW' and iteration-last_mode_iteration>=mode_period)
        if try_mode:
            alternatives=rank_parent_mode_changes(current.solution,partial,CONFIG)
            if alternatives:
                # One actual proposal per iteration, with gradual access to
                # lower-ranked legal alternatives. No hidden B32 screening.
                target=alternatives[(iteration//6)%len(alternatives)]
                partial=replace(partial,patterns=tuple(target if p.parent_id==target.parent_id else p
                                                      for p in partial.patterns))
                changed=[target.pattern_id]
                last_mode_iteration=iteration
        if time.perf_counter()-started>=budget:break
        if model is not None:
            geometry=joint_geometry(parents,partial.patterns,CONFIG)
            result=construct_or_repair_with_gnn(geometry,partial.routes,pilot.partial_directions(current,partial),
                                               partial.removed_parent_ids,model,CONFIG,insertion_limit=8)
            solution=result.solution
        else:
            result=repair_partial_state_oriented(current.solution,current.directions,partial,CONFIG,
                      seed=local_seed,insertion_limit=8,max_insertion_evaluations=4096,
                      coordination_proxy=True,direction_versions=2,allow_pattern_changes=bool(changed))
            solution=None if result.candidate is None else result.candidate.solution
        item=dict(iteration=iteration+1,selection=selector.value,q=q,removed=partial.removed_parent_ids,
                  proposed_modes=changed,before=current.schedule.cmax,status='CONSTRUCTION_FAILED',
                  cmax=None,accepted=False,certified=None,mode_period=mode_period)
        if solution is None:
            item['reason']=result.rejection_reason
        elif time.perf_counter()-started<budget:
            candidate=pilot.oriented.schedule_explicit(solution,result.directions,CONFIG)
            row['reference_calls']+=1
            item['status']=candidate.schedule.status.value
            item.update(patterns=[asdict(p) for p in solution.patterns],
                        routes=[r.block_ids for r in solution.routes],directions=result.directions,
                        recovery_rollouts=candidate.schedule.recovery_rollouts,
                        recovery_exhausted=candidate.schedule.recovery_exhausted,
                        frontier_exhausted=candidate.schedule.frontier_exhausted)
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                check=certify_schedule(solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                row['certifier_calls']+=1
                item['certified']=check.certified
                if not check.certified:raise ValueError(check.errors)
                item['cmax']=candidate.schedule.cmax
                elapsed=time.perf_counter()-started
                if elapsed<=budget and candidate.schedule.cmax<current.schedule.cmax-1e-9*max(1.,current.schedule.cmax):
                    current=candidate
                    row['accepted']+=1;row['mode_accepted']+=bool(changed)
                    item['accepted']=True
                    row['best_events'].append([elapsed,current.schedule.cmax,'MODE' if changed else 'FIXED'])
            if time.perf_counter()-started>budget:item['late']=True
        else:
            item['status']='CONSTRUCTION_FINISHED_AFTER_DEADLINE'
        item.update(after=current.schedule.cmax,elapsed=time.perf_counter()-started)
        if method=='ADAPT_NEW' and changed:
            if item['accepted']:
                unsuccessful_modes=0
                mode_period=2
            else:
                unsuccessful_modes+=1
                if unsuccessful_modes>=4:
                    mode_period=min(8,mode_period*2)
                    unsuccessful_modes=0
        row['trace'].append(item)
        iteration+=1
    elapsed=time.perf_counter()-started
    row.update(status='CERTIFIED',cmax=current.schedule.cmax,iterations=iteration,actual_runtime=elapsed,
               overshoot=max(0.,elapsed-budget),solution=asdict(current.solution),directions=current.directions,
               robot_completion=current.schedule.robot_completion,
               split_count=sum(p.kind.value!='WHOLE' for p in current.solution.patterns))
    return row


def _mode_study_worker(job):
    from dataclasses import asdict
    from mrta_reference.model import ParentWeld
    entry,method,seed,budget,iterations,stage=job
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in entry['parents'])
    common=dict(record='run',version=STUDY_VERSION,stage=stage,instance=entry['instance'],
                N=len(parents),seed=seed,method=method,budget=budget,max_iterations=iterations)
    cpu_started=time.process_time()
    try:
        if method in STUDY_ARMS:
            result=_mode_search(parents,method,seed,budget,iterations)
        else:
            if method=='WAG':
                from mrta_baselines.wag_vns import run_adapted_wag_vns_v2,AdaptedWAGConfig
                out=run_adapted_wag_vns_v2(parents,CONFIG,AdaptedWAGConfig(),seed=seed,time_limit=budget,checkpoints=(budget,))
            elif method=='HGA':
                from mrta_baselines.hga import run_adapted_hga_v2,AdaptedHGAConfig
                out=run_adapted_hga_v2(parents,CONFIG,AdaptedHGAConfig(),seed=seed,time_limit=budget,checkpoints=(budget,))
            else:raise ValueError(method)
            cmax=out.checkpoints[budget]
            result=dict(status=('CERTIFIED' if cmax is not None else
                                'TIME_LIMIT_NO_FEASIBLE' if out.status.value=='COMPLETED' else out.status.value),
                        final_status=out.status.value,cmax=cmax,
                        final_cmax=None if out.schedule is None else out.schedule.cmax,
                        initial_cmax=(None if out.initial_candidate is None or out.initial_candidate.schedule is None
                                      else out.initial_candidate.schedule.cmax),
                        actual_runtime=out.actual_runtime,overshoot=out.overshoot,
                        reference_calls=out.reference_calls,certifier_calls=out.certifier_calls,
                        initial_reference_calls=out.initial_reference_calls,
                        initialization_seconds=out.initialization_time,iterations=out.iterations,
                        accounting=out.accounting,best_events=[asdict(v) for v in out.best_events],
                        solution=None if out.solution is None else asdict(out.solution),directions=out.directions,
                        solution_cmax=None if out.schedule is None else out.schedule.cmax,
                        solution_completed_at=None if not out.best_events else out.best_events[-1].elapsed,
                        solution_within_budget=bool(out.best_events and out.best_events[-1].elapsed<=budget))
        return dict(common,cpu_seconds=time.process_time()-cpu_started,
                    **{k:v for k,v in result.items() if k not in common})
    except Exception as error:
        import traceback
        return dict(common,status='EXECUTION_ERROR',cmax=None,error=str(error),traceback=traceback.format_exc())


def _independent_mode_instances(*,seed_start=2026103001,count=9):
    """New layouts, selected by count only; no solver and no sealed workbook."""
    import importlib.util
    import contextlib
    import io
    source_module=ROOT.parent/'src/generate_welds.py'
    spec=importlib.util.spec_from_file_location('mode_study_generator',source_module)
    generator=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=generator
    spec.loader.exec_module(generator)
    source=generator.DEFAULT_SOURCE_EXCEL
    with contextlib.redirect_stdout(io.StringIO()):
        groups,_,_=generator.read_and_prepare_groups(source,None,'mm',0.01,min_weld_length_m=0.5)
    history=shared.pilot.dataset.DATASET.read_text(encoding='utf-8')
    entries=[]
    for index,seed in enumerate(range(seed_start,seed_start+count)):
        if str(seed) in history:raise ValueError('proposed geometry seed was already in the dataset manifest')
        target=(35,55,85)[index%3]
        with contextlib.redirect_stdout(io.StringIO()):
            placed,_=generator.layout_packing(groups,45,0.03,platform_w=20.,platform_h=12.,
                         allow_rotate=True,seed=seed,max_attempts_per_instance=2000)
            choices=[];count=0
            for item in placed:
                count+=len(groups[item['group_id']])
                choices.append((abs(count-target),int(item['instance_index'])+1,count))
            _,group_count,_=min(choices)
            welds,metadata=generator.generate_weld_instance_from_source(source,group_count,seed)
        parents=[dict(parent_id=str(w.id),start=(w.x1,w.y1),end=(w.x2,w.y2)) for w in welds]
        entries.append(dict(record='instance',version=STUDY_VERSION,stage='INDEPENDENT',
                            instance=f'N{target}_{index//3+1:02d}',target_N=target,N=len(parents),
                            geometry_seed=seed,group_count=group_count,parents=parents,
                            source_excel=source,generator_version=metadata['generator_version']))
    return entries


def summarize_mode_study(rows):
    """Aggregate seeds within geometry; never pool different time budgets."""
    import random
    import statistics
    rows=[r for r in rows if r.get('record')=='run' and r.get('version')==STUDY_VERSION]
    groups=sorted({(r['stage'],r['budget'],r['max_iterations']) for r in rows})
    output=[]
    pairs=(('ADAPT_NEW','MODE_NEW'),('ADAPT_NEW','FIXED_NEW'),('ADAPT_NEW','WAG'),
           ('MODE_NEW','FIXED_NEW'),('MODE_GNN','FIXED_GNN'),
           ('FIXED_GNN','FIXED_NEW'),('MODE_GNN','MODE_NEW'),
           ('MODE_GNN','WAG'),('MODE_GNN','HGA'),('MODE_NEW','WAG'),('FIXED_GNN','WAG'))
    for stage,budget,limit in groups:
        group=[r for r in rows if (r['stage'],r['budget'],r['max_iterations'])==(stage,budget,limit)]
        context=dict(version=STUDY_VERSION,stage=stage,budget=budget,max_iterations=limit)
        for method in STUDY_ARMS+('WAG','HGA'):
            selected=[r for r in group if r['method']==method]
            if not selected:continue
            good=[r for r in selected if r.get('cmax') is not None]
            runtimes=[r['actual_runtime'] for r in selected if 'actual_runtime' in r]
            output.append(dict(context,record='analysis_method',method=method,runs=len(selected),certified=len(good),
                mean_cmax=statistics.mean(r['cmax'] for r in good) if good else None,
                mean_cmax_population='successful runs only; report success denominator separately',
                reference_calls=sum(r.get('reference_calls',0) for r in selected),
                certifier_calls=sum(r.get('certifier_calls',0) for r in selected),
                mean_runtime=statistics.mean(runtimes) if runtimes else None,runtime_observed=len(runtimes),
                max_overshoot=max(r.get('overshoot',0.) for r in selected)))
        for first,second in pairs:
            a={(r['instance'],r['seed']):r.get('cmax') for r in group if r['method']==first}
            b={(r['instance'],r['seed']):r.get('cmax') for r in group if r['method']==second}
            if not a or not b:continue
            shared_keys=set(a)&set(b)
            feasibility=dict(both_success=sum(a[k] is not None and b[k] is not None for k in shared_keys),
                first_only=sum(a[k] is not None and b[k] is None for k in shared_keys),
                second_only=sum(a[k] is None and b[k] is not None for k in shared_keys),
                both_failed=sum(a[k] is None and b[k] is None for k in shared_keys))
            changes=[]
            for name in sorted({key[0] for key in a}|{key[0] for key in b}):
                ka={k for k in a if k[0]==name};kb={k for k in b if k[0]==name}
                if not ka or ka!=kb or any(a[k] is None or b[k] is None for k in ka):continue
                av=statistics.mean(a[k] for k in ka);bv=statistics.mean(b[k] for k in ka)
                changes.append(dict(instance=name,first_mean=av,second_mean=bv,percent=100*(av/bv-1)))
            ds=[r['percent'] for r in changes]
            if not ds:
                output.append(dict(context,record='analysis_pair',first=first,second=second,
                    paired_geometries=0,feasibility=feasibility,instances=[]))
                continue
            rng=random.Random(20261030)
            samples=sorted(statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(20000))
            output.append(dict(context,record='analysis_pair',first=first,second=second,paired_geometries=len(ds),
                feasibility=feasibility,
                wins=sum(d < -1e-7 for d in ds),ties=sum(abs(d)<=1e-7 for d in ds),losses=sum(d>1e-7 for d in ds),
                mean_percent=statistics.mean(ds),median_percent=statistics.median(ds),
                bootstrap95=[samples[499],samples[19499]],bootstrap_samples=20000,bootstrap_seed=20261030,
                interpretation='descriptive geometry-level paired bootstrap; unadjusted for multiple comparisons',
                instances=changes))
    return output


def _ppo_validation_mode_instances():
    """One sheet per existing validation workbook, chosen without solver results."""
    from dataclasses import asdict
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from mrta_data.ppo_instances import load_ppo_platform_instance,to_parent_welds
    dataset=shared.pilot.dataset
    roles=json.loads(dataset.ROLES.read_text(encoding='utf-8'))
    manifest=json.loads(dataset.DATASET.read_text(encoding='utf-8'))
    paths=sorted(w['relative_path'] for w in roles['workbooks'] if w['new_v2_role']=='V2_VALIDATION')
    assert_v2_solver_access_allowed(roles,paths,allowed_roles=('V2_VALIDATION',))
    groups=[]
    for path in paths:
        sheets=[e for e in manifest['instances'] if e['relative_path']==path and e['validation_status']=='VALID'
                and e['duplicate_of'] is None and e['formal_compatibility']=='PASS']
        groups.append((path,sheets))
    # Balanced scale allocation: choose four workbooks per target using only
    # absolute count error, then alternate high/low shared-band presence ties.
    chosen=[]
    for target in (85,55,35):
        for index in range(4):
            options=[]
            for path,sheets in groups:
                for e in sheets:
                    density=(e['n_mandatory_y_split']+e['n_whole_both_rails'])/e['actual_weld_count']
                    options.append(((abs(e['actual_weld_count']-target),
                                     -density if index%2==0 else density,path,e['sheet_name']),path,e))
            _,path,e=min(options,key=lambda x:x[0])
            chosen.append((target,e))
            groups=[g for g in groups if g[0]!=path]
    entries=[]
    for target,e in chosen:
        instance=load_ppo_platform_instance(dataset.PPO/e['relative_path'],e['sheet_name'],
                                           ppo_root=dataset.PPO,instance_id=e['instance_id'])
        parents=to_parent_welds(instance)
        entries.append(dict(record='instance',version=STUDY_VERSION,stage='PPO_VALIDATION',
            instance=e['instance_id'],relative_path=e['relative_path'],sheet_name=e['sheet_name'],
            target_N=target,N=len(parents),role='V2_VALIDATION',
            mandatory_y=e['n_mandatory_y_split'],whole_both_rails=e['n_whole_both_rails'],
            parents=[asdict(p) for p in parents]))
    return entries


def run_mode_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing
    import os
    if args.mode_study=='sparse-online':
        run_sparse_study(args)
        return
    if args.mode_study=='sparse-summary':
        summarize_sparse_study(save=True)
        return
    if args.mode_study=='sparse-audit':
        audit_sparse_study()
        return
    if args.mode_study=='sparse-constructor-diagnostic':
        diagnose_sparse_constructor()
        return
    if args.mode_study=='sparse-recovery-budget':
        diagnose_sparse_recovery_budget()
        return
    if args.mode_study=='recovery-stratified':
        run_stratified_recovery(args)
        return
    if args.mode_study=='recovery-summary':
        summarize_stratified_recovery()
        return
    if args.mode_study=='bound-shadow':
        audit_route_bound_shadow()
        return
    if args.mode_study=='bound-online':
        run_bound_study(args)
        return
    if args.mode_study=='bound-summary':
        summarize_bound_study()
        return
    if args.mode_study=='fresh-e2e':
        run_fresh_e2e(args)
        return
    if args.mode_study=='fresh-summary':
        summarize_fresh_e2e()
        return
    if args.mode_study in ('diversity-diagnostic','diversity-online'):
        run_diversity_study(args)
        return
    if args.mode_study=='diversity-summary':
        summarize_diversity_study(save=True)
        return
    if args.mode_study=='diversity-audit':
        audit_diversity_study()
        return
    if args.mode_study in ('pair-diagnostic','pair-extension'):
        run_pair_study(args)
        return
    if args.mode_study=='pair-summary':
        print(json.dumps(summarize_pair_study(_study_rows()),ensure_ascii=False,indent=2))
        return
    if args.mode_study=='pair-audit':
        audit_pair_study()
        return
    if args.mode_study=='rank-audit':
        audit_ranking_study()
        return
    if args.mode_study in ('rank-diagnostic','rank-validation'):
        run_ranking_study(args)
        return
    if args.mode_study=='rank-summary':
        print(json.dumps(summarize_ranking_study(_study_rows()),ensure_ascii=False,indent=2))
        return
    if args.mode_study in ('matched','neighborhood'):
        run_matched_study(args)
        return
    if args.mode_study=='neighborhood-summary':
        print(json.dumps(summarize_neighborhood_study(_study_rows()),ensure_ascii=False,indent=2))
        return
    if args.mode_study=='summary':
        for row in summarize_mode_study(_study_rows()):
            print(json.dumps(row,ensure_ascii=False),flush=True)
        return
    stage={'dev':'DEV','independent':'INDEPENDENT','ppo-validation':'PPO_VALIDATION'}[args.mode_study]
    budget=args.seconds
    iterations=24 if stage=='DEV' else 1000000
    methods=STUDY_ARMS if stage=='DEV' else STUDY_ARMS+('WAG','HGA')
    seeds=(20261030,) if stage=='DEV' else (20261031,20261032)
    existing=_study_rows()
    entries=[r for r in existing if r.get('record')=='instance' and r.get('version')==STUDY_VERSION and r.get('stage')==stage]
    if not entries:
        _study_append(dict(record='plan',version=STUDY_VERSION,stage=stage,methods=methods,seeds=seeds,
                           seconds=budget,max_iterations=iterations,workers=args.workers,
                           selection='RANDOM/TRAVEL/LOAD; q4/q8 every 3 iterations; one proposal and at most one B32 per iteration',
                           mode_selection='MODE: every 2 iterations; ADAPT_NEW: period 2, double after 4 unsuccessful mode attempts up to 8, reset after improvement; same workload ranking and index=(iteration//6)%count',
                           cut_policy='configured boundaries + center + legal midpoint; default Y=5.5/6/6.5, X=center +/-0.5; no extra +/-0.2 cuts',
                           data_selection='PPO_VALIDATION: one sheet per existing V2 validation workbook; four targets each 85/55/35; nearest N then alternate high/low (mandatory-Y + whole-both)/N; no solver outcomes',
                           initialization='round-robin REGION/RULE beam candidates; first round with any timely certified solution; preserve incumbent across deadline',
                           model='gnn_joint_repair_v2.pt; unchanged pretrained weights',scientific_config=CONFIG.scientific_mapping()))
        if stage=='DEV':
            from dataclasses import asdict
            entries=[dict(record='instance',version=STUDY_VERSION,stage=stage,instance=e['sheet_name'],
                          N=e['N'],parents=[asdict(p) for p in shared.pilot.dataset.load_parents(e)])
                     for e in shared.pilot.entries()[1]]
        elif stage=='PPO_VALIDATION':entries=_ppo_validation_mode_instances()
        else:entries=_independent_mode_instances()
        for entry in entries:_study_append(entry)
    finished={(r['instance'],r['method'],r['seed'],r['budget'],r['max_iterations']) for r in existing
              if r.get('record')=='run' and r.get('version')==STUDY_VERSION and r.get('stage')==stage}
    jobs=[(entry,method,seed,budget,iterations,stage) for entry in entries for seed in seeds for method in methods
          if (entry['instance'],method,seed,budget,iterations) not in finished]
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    print('MODE_STUDY',stage,'jobs',len(jobs),'workers',args.workers,'seconds',budget,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_mode_worker_init) as pool:
        futures={pool.submit(_mode_study_worker,job):job for job in jobs}
        for future in as_completed(futures):
            row=future.result()
            _study_append(row)
            print('MODE_RUN',row['stage'],row['instance'],row['method'],row['seed'],row['status'],
                  row.get('cmax'),'iterations',row.get('iterations'),'mode_accepts',row.get('mode_accepted'),flush=True)


def _matched_worker(source, family='LEGACY', version='matched-state-v1'):
    """Same incumbent, removed parents and mode proposals for both repairers."""
    from dataclasses import asdict,replace
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import (BottleneckSelection, select_bottleneck_parents, rank_parent_mode_changes,
                                 repair_partial_state_oriented, shared_y_parent_links, nominal_coordination_features)
    started=time.perf_counter()
    data=source['solution']
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
    patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
                               None if p['rail'] is None else Rail(p['rail'])) for p in data['patterns'])
    solution=canonicalize(parents,patterns,{r['robot_id']:r['block_ids'] for r in data['routes']},CONFIG,scope=FORMAL_SCOPE_V2)
    pilot=shared.pilot
    current=pilot.oriented.schedule_explicit(solution,source['directions'],CONFIG)
    check=certify_schedule(solution,current.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
    if not check.certified or abs(current.schedule.cmax-source['cmax'])>1e-7:
        raise ValueError('saved incumbent failed reproduction')
    model=shared.load_model(pilot.MODEL)[0]
    selectors={'LEGACY':pilot.SELECTORS, 'WAIT':(BottleneckSelection.WAIT_ANCHORED,)*3,
               'Y_PAIR':(BottleneckSelection.SHARED_Y_PAIR,)*3}[family]
    links=shared_y_parent_links(solution,current.schedule,CONFIG) if family=='Y_PAIR' else ()
    trials=[]
    for q in (4,8):
        for selector_index,selector in enumerate(selectors):
            local_seed=20261031+selector_index+q
            selection_started=time.perf_counter()
            partial=select_bottleneck_parents(solution,current.directions,current.schedule,selector,
                q=q,seed=local_seed,config=CONFIG)
            fallback=None
            if partial is None:
                fallback='NO_POSITIVE_WAIT' if family=='WAIT' else 'NO_SHARED_Y_LINK'
                partial=select_bottleneck_parents(solution,current.directions,current.schedule,BottleneckSelection.LOAD,
                    q=q,seed=local_seed,config=CONFIG)
            selection_seconds=time.perf_counter()-selection_started
            alternatives=rank_parent_mode_changes(solution,partial,CONFIG)[:3]
            for mode_index,target in enumerate((None,)+alternatives):
                proposed=partial if target is None else replace(partial,patterns=tuple(
                    target if p.parent_id==target.parent_id else p for p in partial.patterns))
                for method in ('NEW','GNN'):
                    tick=time.perf_counter()
                    row=dict(q=q,selection=selector.value,context_index=selector_index,seed=local_seed,
                        selection_fallback=fallback,selection_seconds=selection_seconds,
                        shared_y_links=len(links),selected_shared_y_links=[link for link in links
                            if link[0] in partial.removed_parent_ids and link[1] in partial.removed_parent_ids],
                        removed=partial.removed_parent_ids,
                        mode_rank=mode_index,mode=None if target is None else asdict(target),repairer=method,
                        incumbent_cmax=current.schedule.cmax,cmax=None,certified=False,reference_calls=0)
                    try:
                        if method=='GNN':
                            geometry=joint_geometry(parents,proposed.patterns,CONFIG)
                            result=construct_or_repair_with_gnn(geometry,proposed.routes,
                                pilot.partial_directions(current,proposed),proposed.removed_parent_ids,model,CONFIG,insertion_limit=8)
                            candidate_solution=result.solution
                        else:
                            result=repair_partial_state_oriented(solution,current.directions,proposed,CONFIG,
                                seed=local_seed,insertion_limit=8,max_insertion_evaluations=4096,
                                coordination_proxy=True,direction_versions=2,allow_pattern_changes=target is not None)
                            candidate_solution=None if result.candidate is None else result.candidate.solution
                        row['construction_seconds']=time.perf_counter()-tick
                        if candidate_solution is None:
                            row.update(status='CONSTRUCTION_FAILED',reason=result.rejection_reason)
                        else:
                            if version=='mode-ranking-v1':
                                feature_started=time.perf_counter()
                                row['ranking_features']=nominal_coordination_features(candidate_solution,result.directions,CONFIG)
                                row['feature_seconds']=time.perf_counter()-feature_started
                            row['reference_calls']=1
                            candidate=pilot.oriented.schedule_explicit(candidate_solution,result.directions,CONFIG)
                            row.update(status=candidate.schedule.status.value,
                                recovery_rollouts=candidate.schedule.recovery_rollouts,
                                frontier_exhausted=candidate.schedule.frontier_exhausted,
                                patterns=[asdict(p) for p in candidate_solution.patterns],
                                routes=[r.block_ids for r in candidate_solution.routes],directions=result.directions)
                            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                                check=certify_schedule(candidate_solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                                if not check.certified:raise ValueError(check.errors)
                                row.update(cmax=candidate.schedule.cmax,certified=True)
                    except Exception as error:
                        import traceback
                        row.update(status='EXECUTION_ERROR',error=str(error),traceback=traceback.format_exc())
                    row['seconds']=time.perf_counter()-tick
                    trials.append(row)
    return dict(record='matched_run',version=version,family=family,source_version=STUDY_VERSION,
                instance=source['instance'],source_seed=source['seed'],source_method=source['method'],
                N=source['N'],incumbent_cmax=current.schedule.cmax,source_reference_calls=1,
                source_certified=True,trials=trials,actual_runtime=time.perf_counter()-started)


def pair_portfolio(single_count, pair_count, policy):
    """A common allowance derived from the available single-mode controls."""
    singles=['S'+str(i+1) for i in range(single_count)]
    pairs=['P'+str(i+1) for i in range(pair_count)]
    quota=min(3,single_count)
    if policy=='SINGLE':return singles[:quota]
    if policy=='PAIR':return (pairs+singles)[:quota]
    if policy=='MIXED':return (singles[:1]+pairs+singles[1:])[:quota]
    raise ValueError(policy)


def _fixed_route_time_bound(solution,directions):
    """Lower bound for THIS fixed assignment/order/direction, including every setup/post."""
    from mrta_search.lns import oriented_route_state
    state=oriented_route_state(tuple(r.block_ids for r in solution.routes),directions,
        block_map(solution,CONFIG,scope=FORMAL_SCOPE_V2),CONFIG)
    return max(p+sum(arcs) for p,arcs in zip(state.loads,state.arc_costs))


def _evaluate_mode_bundle(current,partial,changed,repairer,seed,model,*,deadline=None,return_candidate=False,improvement_ceiling=None):
    from dataclasses import asdict,replace
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import repair_partial_state_oriented
    replacements={p.parent_id:p for p in changed}
    proposed=replace(partial,patterns=tuple(replacements.get(p.parent_id,p) for p in partial.patterns))
    tick=time.perf_counter()
    candidate=None
    row=dict(changed_modes=[asdict(p) for p in changed],repairer=repairer,cmax=None,certified=False,reference_calls=0)
    try:
        if repairer=='GNN':
            geometry=joint_geometry(current.solution.parents,proposed.patterns,CONFIG)
            result=construct_or_repair_with_gnn(geometry,proposed.routes,shared.pilot.partial_directions(current,proposed),
                proposed.removed_parent_ids,model,CONFIG,insertion_limit=8)
            solution=result.solution
        else:
            result=repair_partial_state_oriented(current.solution,current.directions,proposed,CONFIG,seed=seed,
                insertion_limit=8,max_insertion_evaluations=4096,coordination_proxy=True,direction_versions=2,
                allow_pattern_changes=bool(changed))
            solution=None if result.candidate is None else result.candidate.solution
        row['construction_seconds']=time.perf_counter()-tick
        if solution is None:
            row.update(status='CONSTRUCTION_FAILED',reason=result.rejection_reason)
        elif deadline is not None and time.perf_counter()>=deadline:
            row.update(status='CONSTRUCTION_FINISHED_AFTER_DEADLINE')
        else:
            if improvement_ceiling is not None:
                bound_tick=time.perf_counter();bound=_fixed_route_time_bound(solution,result.directions)
                row.update(route_lower_bound=bound,bound_seconds=time.perf_counter()-bound_tick,improvement_ceiling=improvement_ceiling)
                # Conservative numerical margin: prune only at/above the
                # incumbent, not inside its acceptance tolerance band.
                if bound>=improvement_ceiling+1e-7:
                    row.update(status='NONIMPROVING_ROUTE_BOUND',patterns=[asdict(p) for p in solution.patterns],
                        routes=[r.block_ids for r in solution.routes],directions=result.directions,seconds=time.perf_counter()-tick)
                    return (row,None) if return_candidate else row
                if deadline is not None and time.perf_counter()>=deadline:
                    row.update(status='CONSTRUCTION_FINISHED_AFTER_DEADLINE',seconds=time.perf_counter()-tick)
                    return (row,None) if return_candidate else row
            row['reference_calls']=1
            candidate=shared.pilot.oriented.schedule_explicit(solution,result.directions,CONFIG)
            row.update(status=candidate.schedule.status.value,recovery_rollouts=candidate.schedule.recovery_rollouts,
                frontier_exhausted=candidate.schedule.frontier_exhausted,patterns=[asdict(p) for p in solution.patterns],
                routes=[r.block_ids for r in solution.routes],directions=result.directions)
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                check=certify_schedule(solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                if not check.certified:raise ValueError(check.errors)
                row.update(cmax=candidate.schedule.cmax,certified=True)
    except Exception as error:
        import traceback
        row.update(status='EXECUTION_ERROR',error=str(error),traceback=traceback.format_exc())
    row['seconds']=time.perf_counter()-tick
    return (row,candidate) if return_candidate else row


def _pair_worker(job):
    from dataclasses import asdict
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import select_bottleneck_parents,rank_parent_mode_changes,rank_parent_mode_pairs
    source,control,cohort=job;started=time.perf_counter()
    data=source['solution']
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
    patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
        None if p['rail'] is None else Rail(p['rail'])) for p in data['patterns'])
    solution=canonicalize(parents,patterns,{r['robot_id']:r['block_ids'] for r in data['routes']},CONFIG,scope=FORMAL_SCOPE_V2)
    current=shared.pilot.oriented.schedule_explicit(solution,source['directions'],CONFIG)
    check=certify_schedule(solution,current.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
    if not check.certified or abs(current.schedule.cmax-source['cmax'])>1e-7:
        raise ValueError('pair study source replay failed')
    model=shared.load_model(shared.pilot.MODEL)[0]
    contexts=[]
    for q in (4,8):
        for selector_index,selector in enumerate(shared.pilot.SELECTORS):
            seed=20261031+selector_index+q
            partial=select_bottleneck_parents(solution,current.directions,current.schedule,selector,q=q,seed=seed,config=CONFIG)
            tick=time.perf_counter();singles=rank_parent_mode_changes(solution,partial,CONFIG)[:3]
            single_seconds=time.perf_counter()-tick
            tick=time.perf_counter();pairs=rank_parent_mode_pairs(solution,partial,CONFIG,limit=len(singles))
            pair_seconds=time.perf_counter()-tick
            for repairer in ('NEW','GNN'):
                originals=sorted([t for t in control['trials'] if t['q']==q and t['seed']==seed and t['repairer']==repairer],key=lambda t:t['mode_rank'])
                if len(originals)!=1+len(singles) or any(tuple(t['removed'])!=partial.removed_parent_ids for t in originals):
                    raise ValueError('single-mode control context differs')
                if [t['mode'] for t in originals[1:]]!=[asdict(p) for p in singles]:
                    raise ValueError('single-mode control proposals differ')
                trials=[dict(t,proposal_id='O' if t['mode_rank']==0 else 'S'+str(t['mode_rank']),reused=True) for t in originals]
                pair_trials=[]
                for index,changed in enumerate(pairs):
                    trial=_evaluate_mode_bundle(current,partial,changed,repairer,seed,model)
                    trial.update(proposal_id='P'+str(index+1),reused=False)
                    trials.append(trial);pair_trials.append(trial)
                base=min([current.schedule.cmax]+[t['cmax'] for t in originals[:1] if t['certified']])
                best_single=min([base]+[t['cmax'] for t in originals[1:] if t['certified']])
                feasible_pairs=[t for t in pair_trials if t['certified']]
                components=[];best_pair=None
                if feasible_pairs:
                    best_pair=min(feasible_pairs,key=lambda t:t['cmax'])
                    if best_pair['cmax']<best_single-1e-7:
                        changed=pairs[int(best_pair['proposal_id'][1:])-1]
                        for index,p in enumerate(changed):
                            prior=next((t for t in originals[1:] if t['mode']==asdict(p)),None)
                            if prior is not None:
                                component=dict(prior,reused=True,changed_modes=[asdict(p)])
                            else:
                                component=dict(_evaluate_mode_bundle(current,partial,(p,),repairer,seed,model),reused=False)
                            component.update(component_of=best_pair['proposal_id'],component_index=index)
                            components.append(component)
                contexts.append(dict(q=q,selection=selector.value,seed=seed,repairer=repairer,
                    removed=partial.removed_parent_ids,base=base,incumbent_cmax=current.schedule.cmax,
                    single_proposal_seconds=single_seconds,pair_proposal_seconds=pair_seconds,
                    single_count=len(singles),pair_count=len(pairs),trials=trials,components=components,
                    portfolios={policy:pair_portfolio(len(singles),len(pairs),policy) for policy in ('SINGLE','PAIR','MIXED')}))
    return dict(record='pair_run',version='parent-pair-v1',cohort=cohort,instance=source['instance'],N=len(parents),
        source_seed=source['seed'],source_cmax=source['cmax'],control_version=control['version'],
        source_reference_calls=1,source_certified=True,contexts=contexts,actual_runtime=time.perf_counter()-started)


def run_pair_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os
    rows=_study_rows();cohort='BASE12' if args.mode_study=='pair-diagnostic' else 'EXT9_SAME_FAMILY'
    if not any(r.get('record')=='pair_plan' and r.get('version')=='parent-pair-v1' for r in rows):
        _study_append(dict(record='pair_plan',version='parent-pair-v1',cohorts=['BASE12','EXT9_SAME_FAMILY'],
            sources='reuse twelve matched-state-v1 incumbents and nine mode-ranking-v1 incumbents; consumed development states',
            proposals='enumerate legal mode changes for two distinct removed parents; exact minimization over their robot assignments of '
                      'max process load, then spread, then sum of both parents process time; all others keep incumbent owners only for proxy',
            quota='k=min(3, available single-mode proposals); evaluate k or fewer pairs, no extra split points',
            portfolios={'SINGLE':'top k singles','PAIR':'top k pairs, pad missing slots with singles',
                        'MIXED':'best single then pairs, pad missing slots with remaining singles'},
            components='if best pair improves on single portfolio, evaluate both constituents separately from the same source and partial; '
                       'reuse exact control matches; these extra calls are diagnostic only and excluded from policy results',
            repairers=['NEW','GNN'],q=[4,8],selectors=['RANDOM','TRAVEL','LOAD'],seed='20261031+selector_index+q',
            statistics='report cohorts separately; pooled resampling grouped by workbook, not 21 independent geometries',
            cost='same B32 allowance per policy; pair proposal and repair overhead separately recorded, no equal-wall-time claim'))
    if cohort=='BASE12':
        sources={r['instance']:r for r in rows if r.get('record')=='run' and r.get('version')==STUDY_VERSION
            and r.get('stage')=='PPO_VALIDATION' and r.get('method')=='FIXED_NEW' and r.get('seed')==20261031}
        controls={r['instance']:r for r in rows if r.get('record')=='matched_run' and r.get('version')=='matched-state-v1'}
    else:
        sources={r['instance']:r for r in rows if r.get('record')=='ranking_source' and r.get('version')=='mode-ranking-v1'}
        controls={r['instance']:r for r in rows if r.get('record')=='ranking_run' and r.get('version')=='mode-ranking-v1'
            and r.get('stage')=='NEW_LAYOUT_VALIDATION'}
    done={r['instance'] for r in rows if r.get('record')=='pair_run' and r.get('version')=='parent-pair-v1' and r['cohort']==cohort}
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        jobs=[pool.submit(_pair_worker,(source,controls[name],cohort)) for name,source in sources.items() if name not in done]
        for future in as_completed(jobs):
            row=future.result();_study_append(row)
            pairs=[t for c in row['contexts'] for t in c['trials'] if not t['reused']]
            print('PAIR',cohort,row['instance'],len(pairs),Counter(t['status'] for t in pairs),
                  'component_checks',sum(len(c['components']) for c in row['contexts']),flush=True)


def summarize_pair_study(rows):
    from collections import defaultdict,Counter
    from statistics import mean
    import random
    runs=[r for r in rows if r.get('record')=='pair_run' and r.get('version')=='parent-pair-v1']
    summaries=[];comparisons=[];component_cases=[];execution=[]
    for cohort in ('BASE12','EXT9_SAME_FAMILY','POOLED_WORKBOOK'):
        selected=runs if cohort=='POOLED_WORKBOOK' else [r for r in runs if r['cohort']==cohort]
        if not selected:continue
        allpairs=[t for r in selected for c in r['contexts'] for t in c['trials'] if not t['reused']]
        added_components=[t for r in selected for c in r['contexts'] for t in c['components'] if not t['reused']]
        execution.append(dict(cohort=cohort,runs=len(selected),pair_candidates=len(allpairs),
            pair_statuses=dict(Counter(t['status'] for t in allpairs)),pair_calls=sum(t['reference_calls'] for t in allpairs),
            pair_certifications=sum(t['certified'] for t in allpairs),source_replays=len(selected),
            component_calls=sum(t['reference_calls'] for t in added_components),
            component_statuses=dict(Counter(t['status'] for t in added_components)),
            component_certifications=sum(t['certified'] for t in added_components),
            cumulative_worker_seconds=sum(r['actual_runtime'] for r in selected)))
        stats=defaultdict(Counter);values=defaultdict(lambda:defaultdict(list));delta=defaultdict(lambda:defaultdict(list))
        for run in selected:
            unit=run['instance'].split('::')[0] if cohort=='POOLED_WORKBOOK' else run['instance']
            for context in run['contexts']:
                repairer=context['repairer'];trials={t['proposal_id']:t for t in context['trials']}
                outcomes={}
                for policy,ids in context['portfolios'].items():
                    chosen=[trials[i] for i in ids]
                    best=min([context['base']]+[t['cmax'] for t in chosen if t['certified']])
                    outcomes[policy]=best;k=repairer,policy
                    stats[k].update(contexts=1,selected_candidates=len(chosen),
                        selected_calls=sum(t['reference_calls'] for t in chosen),selected_feasible=sum(t['certified'] for t in chosen),
                        mode_improved=int(best<context['base']-1e-7),no_pair_contexts=int(context['pair_count']==0),
                        single_padding=sum(i.startswith('S') for i in ids) if policy=='PAIR' else 0)
                    values[k][unit].append(100*(best/context['incumbent_cmax']-1))
                for policy in ('PAIR','MIXED'):
                    delta[repairer,policy][unit].append(100*(outcomes[policy]/outcomes['SINGLE']-1))
                if context['components'] and cohort!='POOLED_WORKBOOK':
                    pair_id=context['components'][0]['component_of'];pair=trials[pair_id]
                    component_best=min([context['base']]+[t['cmax'] for t in context['components'] if t['certified']])
                    component_cases.append(dict(cohort=cohort,instance=run['instance'],repairer=repairer,q=context['q'],seed=context['seed'],
                        pair_id=pair_id,base=context['base'],single_portfolio=outcomes['SINGLE'],pair_cmax=pair['cmax'],
                        component_results=[dict(status=t['status'],cmax=t['cmax'],reused=t['reused']) for t in context['components']],
                        pair_better_than_each_component=pair['cmax']<component_best-1e-7,
                        neither_component_improves_base=component_best>=context['base']-1e-7))
        for (repairer,policy),by_unit in values.items():
            summaries.append(dict(cohort=cohort,repairer=repairer,policy=policy,units=len(by_unit),
                **stats[repairer,policy],mean_pct=mean(mean(v) for v in by_unit.values())))
        for (repairer,policy),by_unit in delta.items():
            ds=[mean(v) for _,v in sorted(by_unit.items())]
            rng=random.Random(20261031);boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(20000))
            comparisons.append(dict(cohort=cohort,repairer=repairer,policy=policy+'/SINGLE',units=len(ds),mean_pct=mean(ds),
                win=sum(x<-1e-8 for x in ds),tie=sum(abs(x)<=1e-8 for x in ds),loss=sum(x>1e-8 for x in ds),
                descriptive_ci95=[boot[500],boot[19499]]))
    return dict(record='pair_analysis',version='parent-pair-v1',execution=execution,summary=summaries,
        comparisons=comparisons,component_cases=component_cases,
        interpretation='paired context changes averaged within geometry or workbook; bootstrap20000 seed20261031 unadjusted; '
                       'component tests are conditional diagnostics, not additional policy evaluations')


def audit_pair_study():
    """Check effective mode changes, replay conditional examples and account cost."""
    from collections import defaultdict
    from statistics import mean
    from dataclasses import asdict
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import nominal_coordination_features,oriented_route_state
    rows=_study_rows()
    if any(r.get('record')=='pair_audit' and r.get('version')=='parent-pair-v1' for r in rows):
        print('PAIR_AUDIT already saved',flush=True);return
    runs=[r for r in rows if r.get('record')=='pair_run' and r.get('version')=='parent-pair-v1']
    if Counter(r['cohort'] for r in runs)!=Counter(BASE12=12,EXT9_SAME_FAMILY=9):
        raise ValueError('complete both pair cohorts before final analysis')
    if len({(r['cohort'],r['instance']) for r in runs})!=len(runs):
        raise ValueError('duplicate pair runs')
    sources={(r['instance'],'BASE12'):r for r in rows if r.get('record')=='run'
        and r.get('version')==STUDY_VERSION and r.get('stage')=='PPO_VALIDATION'
        and r.get('method')=='FIXED_NEW' and r.get('seed')==20261031}
    sources.update({(r['instance'],'EXT9_SAME_FAMILY'):r for r in rows
        if r.get('record')=='ranking_source' and r.get('version')=='mode-ranking-v1'})

    def restore(source,trial=None):
        data=source['solution']
        parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
        pp=data['patterns'] if trial is None else trial['patterns']
        patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
            None if p['rail'] is None else Rail(p['rail'])) for p in pp)
        routes=[r['block_ids'] for r in data['routes']] if trial is None else trial['routes']
        return canonicalize(parents,patterns,dict(enumerate(routes)),CONFIG,scope=FORMAL_SCOPE_V2)

    def describe(solution,directions,cmax):
        blocks=block_map(solution,CONFIG,scope=FORMAL_SCOPE_V2)
        state=oriented_route_state(tuple(r.block_ids for r in solution.routes),directions,blocks,CONFIG)
        features=nominal_coordination_features(solution,directions,CONFIG)
        return dict(blocks=len(blocks),processing_loads=state.loads,total_processing=sum(state.loads),
            nominal_completion=features['nominal_completion'],nominal_cmax=features['nominal_cmax'],
            coordination_completion_penalty=None if cmax is None else cmax-features['nominal_cmax'])

    costs=defaultdict(list);effective=Counter();examples=[];contexts=0;replay_calls=0;replay_certified=0
    for run in sorted(runs,key=lambda r:(r['cohort'],r['instance'])):
        source=sources[run['instance'],run['cohort']];old={p['parent_id']:p for p in source['solution']['patterns']}
        source_features=None
        for context in run['contexts']:
            contexts+=1;trials={t['proposal_id']:t for t in context['trials']}
            if len({len(ids) for ids in context['portfolios'].values()})!=1:
                raise ValueError('unequal portfolio candidate allowance')
            for trial in trials.values():
                if trial['reused']:continue
                if trial['status']=='FEASIBLE' and not trial['certified']:
                    raise ValueError('uncertified feasible pair')
                if 'patterns' not in trial:continue
                solution=restore(source,trial);actual={p.parent_id:asdict(p) for p in solution.patterns}
                requested={p['parent_id']:p for p in trial['changed_modes']}
                changed={pid for pid in old if old[pid]!=actual[pid]}
                if len(requested)!=2 or changed!=set(requested) or not changed<=set(context['removed']):
                    raise ValueError('effective changes are not the two selected removed parents')
                if any(actual[p]!=pattern for p,pattern in requested.items()):
                    raise ValueError('requested pair pattern changed during construction')
                effective[context['repairer'],trial['status'],len(changed)]+=1
            for policy,ids in context['portfolios'].items():
                evaluated=[trials['O']]+[trials[i] for i in ids]
                construction=sum(t['construction_seconds'] for t in evaluated)
                dispatch=sum(t['seconds']-t['construction_seconds']-t.get('feature_seconds',0) for t in evaluated)
                proposal=context['single_proposal_seconds']+(context['pair_proposal_seconds'] if policy!='SINGLE' else 0)
                if dispatch < -1e-7:raise ValueError('invalid candidate timing decomposition')
                costs[run['cohort'],context['repairer'],policy].append((proposal,construction,dispatch))
            if not context['components']:continue
            if source_features is None:source_features=describe(restore(source),source['directions'],source['cmax'])
            pair=trials[context['components'][0]['component_of']]
            replayed=[]
            for name,trial in [('PAIR',pair)]+[('COMPONENT_'+str(i+1),t) for i,t in enumerate(context['components'])]:
                solution=restore(source,trial)
                scheduled=shared.pilot.oriented.schedule_explicit(solution,trial['directions'],CONFIG)
                replay_calls+=1
                if scheduled.schedule.status.value!=trial['status']:
                    raise ValueError('pair/component replay status differs')
                certified=False
                if trial['status']=='FEASIBLE':
                    check=certify_schedule(solution,scheduled.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                    if not check.certified or abs(scheduled.schedule.cmax-trial['cmax'])>1e-7:
                        raise ValueError('pair/component replay Cmax or certification differs')
                    certified=True;replay_certified+=1
                replayed.append(dict(name=name,status=trial['status'],cmax=trial['cmax'],certified=certified,
                    recovery_rollouts=scheduled.schedule.recovery_rollouts,
                    features=describe(solution,trial['directions'],trial['cmax'])))
            modes=[dict(parent_id=p['parent_id'],before=old[p['parent_id']],after=p) for p in pair['changed_modes']]
            examples.append(dict(cohort=run['cohort'],instance=run['instance'],q=context['q'],seed=context['seed'],
                selection=context['selection'],repairer=context['repairer'],proposal_id=pair['proposal_id'],
                base=context['base'],source_cmax=source['cmax'],source_features=source_features,modes=modes,replays=replayed))
            print('PAIR_REPLAY',run['cohort'],run['instance'],context['repairer'],context['q'],context['seed'],flush=True)
    accounted=[dict(cohort=cohort,repairer=repairer,policy=policy,contexts=len(values),
        mean_proposal_seconds=mean(v[0] for v in values),mean_construction_seconds=mean(v[1] for v in values),
        mean_dispatch_certification_seconds=mean(v[2] for v in values),mean_accounted_seconds=mean(sum(v) for v in values))
        for (cohort,repairer,policy),values in sorted(costs.items())]
    analysis=summarize_pair_study(rows)
    if not any(r.get('record')=='pair_analysis' and r.get('version')=='parent-pair-v1' for r in rows):
        _study_append(analysis)
    record=dict(record='pair_audit',version='parent-pair-v1',contexts=contexts,
        effective_mode_changes=[dict(repairer=k[0],status=k[1],changed_parents=k[2],count=v) for k,v in sorted(effective.items())],
        costs=accounted,examples=examples,reference_calls=replay_calls,certifications=replay_certified,
        note='all conditional positive examples and both constituents replayed; component comparisons are conditional on '
             'source, destroy set, repairer and seed, not global infeasibility or necessity proofs. Costs include common '
             'original-mode candidate, omit old diagnostic features, and reuse historical single timings; no equal-wall-time claim.')
    _study_append(record)
    print(json.dumps(dict(record='pair_audit',contexts=contexts,examples=len(examples),reference_calls=replay_calls,
        certifications=replay_certified,costs=accounted),ensure_ascii=False),flush=True)


DIVERSITY_VERSION='parent-diversity-v1'
DIVERSITY_POLICIES=('SINGLE','MIXED','DIVERSE')
DIVERSITY_SEEDS=(2026103101,2026103102)


def _pair_study_sources(rows):
    sources={(r['instance'],'BASE12'):r for r in rows if r.get('record')=='run'
        and r.get('version')==STUDY_VERSION and r.get('stage')=='PPO_VALIDATION'
        and r.get('method')=='FIXED_NEW' and r.get('seed')==20261031}
    sources.update({(r['instance'],'EXT9_SAME_FAMILY'):r for r in rows
        if r.get('record')=='ranking_source' and r.get('version')=='mode-ranking-v1'})
    return sources


def _replay_study_source(source):
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.certifier import certify_schedule
    data=source['solution']
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
    patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
        None if p['rail'] is None else Rail(p['rail'])) for p in data['patterns'])
    solution=canonicalize(parents,patterns,{r['robot_id']:r['block_ids'] for r in data['routes']},CONFIG,scope=FORMAL_SCOPE_V2)
    current=shared.pilot.oriented.schedule_explicit(solution,source['directions'],CONFIG)
    check=certify_schedule(solution,current.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
    if not check.certified or abs(current.schedule.cmax-source['cmax'])>1e-7:
        raise ValueError('diversity source replay differs')
    return current


def _diversity_worker(job):
    from dataclasses import asdict
    from mrta_search.lns import select_bottleneck_parents,rank_parent_mode_pairs
    source,control=job;started=time.perf_counter()
    current=_replay_study_source(source);model=shared.load_model(shared.pilot.MODEL)[0]
    contexts=[]
    for prior in control['contexts']:
        selector=next(s for s in shared.pilot.SELECTORS if s.value==prior['selection'])
        partial=select_bottleneck_parents(current.solution,current.directions,current.schedule,selector,
            q=prior['q'],seed=prior['seed'],config=CONFIG)
        if tuple(prior['removed'])!=partial.removed_parent_ids:raise ValueError('diversity destroy set differs')
        limit=max(0,prior['single_count']-1)
        tick=time.perf_counter()
        pairs=rank_parent_mode_pairs(current.solution,partial,CONFIG,limit=limit,diverse_pairs=True)
        proposal_seconds=time.perf_counter()-tick
        trials=[]
        for index,pair in enumerate(pairs):
            modes=[asdict(p) for p in pair]
            reuse=next((t for t in prior['trials'] if t.get('changed_modes')==modes),None)
            if reuse is None:
                trial=_evaluate_mode_bundle(current,partial,pair,prior['repairer'],prior['seed'],model)
                trial.update(reused=False)
            else:
                trial=dict(reuse,reused=True,reused_proposal_id=reuse['proposal_id'])
            trial['proposal_id']='D'+str(index+1);trials.append(trial)
        ids=[s.replace('P','D') for s in pair_portfolio(prior['single_count'],len(pairs),'MIXED')]
        contexts.append(dict(q=prior['q'],selection=prior['selection'],seed=prior['seed'],repairer=prior['repairer'],
            removed=prior['removed'],trials=trials,portfolio=ids,proposal_seconds=proposal_seconds))
    return dict(record='diversity_run',version=DIVERSITY_VERSION,cohort=control['cohort'],instance=control['instance'],
        source_reference_calls=1,source_certified=True,contexts=contexts,actual_runtime=time.perf_counter()-started)


def _online_bundle_search(current,repairer,policy,seed,budget,model,*,prune_route_bound=False,absolute_deadline=None):
    """A timed local-search continuation; each round shares one frozen partial."""
    from dataclasses import asdict
    from mrta_search.lns import select_bottleneck_parents,rank_parent_mode_changes,rank_parent_mode_pairs
    started=time.perf_counter();cpu_started=time.process_time();deadline=started+budget
    if absolute_deadline is not None:
        deadline=absolute_deadline;budget=max(0.,deadline-started)
    initial=current.schedule.cmax;trace=[];events=[[0.,initial,'SOURCE']];iteration=0;error=None
    while time.perf_counter()<deadline:
        origin=current;local_seed=seed+iteration
        selector=shared.pilot.SELECTORS[iteration%3];q=min(len(current.solution.parents),(4,8)[(iteration//3)%2])
        tick=time.perf_counter()
        partial=select_bottleneck_parents(origin.solution,origin.directions,origin.schedule,selector,
            q=q,seed=local_seed,config=CONFIG)
        singles=rank_parent_mode_changes(origin.solution,partial,CONFIG)[:3]
        pairs=() if policy=='SINGLE' else rank_parent_mode_pairs(origin.solution,partial,CONFIG,
            limit=max(0,len(singles)-1),diverse_pairs=policy=='DIVERSE')
        ids=pair_portfolio(len(singles),len(pairs),'SINGLE' if policy=='SINGLE' else 'MIXED')
        proposals=[('O',())]+[(name,(singles[int(name[1:])-1],) if name.startswith('S') else pairs[int(name[1:])-1]) for name in ids]
        proposal_seconds=time.perf_counter()-tick
        for position,(name,changed) in enumerate(proposals):
            if time.perf_counter()>=deadline:break
            before=current.schedule.cmax
            trial,candidate=_evaluate_mode_bundle(origin,partial,changed,repairer,local_seed,model,
                deadline=deadline,return_candidate=True,improvement_ceiling=before if prune_route_bound else None)
            elapsed=time.perf_counter()-started
            accepted=bool(trial['certified'] and elapsed<=budget and trial['cmax']<before-1e-9*max(1.,before))
            if accepted:
                current=candidate;events.append([elapsed,current.schedule.cmax,name])
            trial.update(iteration=iteration,position=position,proposal_id=name,q=q,selection=selector.value,
                seed=local_seed,removed=partial.removed_parent_ids,round_source_cmax=origin.schedule.cmax,
                incumbent_before=before,incumbent_after=current.schedule.cmax,accepted=accepted,late=elapsed>budget,
                elapsed=elapsed,proposal_seconds=proposal_seconds if position==0 else 0.)
            trace.append(trial)
            if trial['status']=='EXECUTION_ERROR':error=trial['error'];break
        iteration+=1
        if error is not None:break
    elapsed=time.perf_counter()-started
    return dict(status='CERTIFIED' if error is None else 'EXECUTION_ERROR',error=error,initial_cmax=initial,search_started=started,
        cmax=current.schedule.cmax,budget=budget,actual_runtime=elapsed,cpu_seconds=time.process_time()-cpu_started,
        overshoot=max(0.,elapsed-budget),iterations=iteration,trace=trace,best_events=events,
        reference_calls=sum(t['reference_calls'] for t in trace),certifier_calls=sum(t['certified'] for t in trace),
        accepted=sum(t['accepted'] for t in trace),mode_accepted=sum(t['accepted'] and bool(t['changed_modes']) for t in trace),
        pair_accepted=sum(t['accepted'] and len(t['changed_modes'])==2 for t in trace),
        late_candidates=sum(t['late'] for t in trace),solution=asdict(current.solution),directions=current.directions)


def _diversity_online_worker(job):
    source,repairer,seed,policies=job;tick=time.perf_counter()
    current=_replay_study_source(source)
    model=shared.load_model(shared.pilot.MODEL)[0] if repairer=='GNN' else None
    preparation=time.perf_counter()-tick
    rows=[]
    for index,policy in enumerate(policies):
        result=_online_bundle_search(current,repairer,policy,seed,20.,model)
        rows.append(dict(record='diversity_online_run',version=DIVERSITY_VERSION,instance=source['instance'],
            source_seed=source['seed'],repairer=repairer,seed=seed,policy=policy,policy_order=policies,
            source_reference_calls=int(index==0),source_certified=True,
            group_preparation_seconds=preparation if index==0 else 0.,**result))
    return rows


def run_diversity_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os,random
    rows=_study_rows();sources=_pair_study_sources(rows)
    if not any(r.get('record')=='diversity_plan' and r.get('version')==DIVERSITY_VERSION for r in rows):
        _study_append(dict(record='diversity_plan',version=DIVERSITY_VERSION,
            hypothesis='top load-ranked pair modes overconcentrate on the same unordered parent pair',
            diversity='best mode per distinct unordered parent pair first, fill remaining slots by original ranking; top1 unchanged',
            diagnostic='reuse all 21 saved pair-study states/contexts and overlapping trials; only evaluate new diverse pair modes',
            policies={'SINGLE':'O + top up to 3 singles','MIXED':'O + S1 + original top up to 2 pairs',
                'DIVERSE':'O + S1 + diverse top up to 2 pairs; both mixtures pad missing pairs with singles'},
            online=dict(instances=sorted(name for name,cohort in sources if cohort=='BASE12'),
                seeds=DIVERSITY_SEEDS,budget_seconds=20,repairers=['NEW','GNN'],policies=DIVERSITY_POLICIES,
                start='same saved FIXED_NEW incumbent for each instance; 20s local continuation, not end-to-end initialization',
                rounds='RANDOM/TRAVEL/LOAD, q4 then q8 every three rounds; origin fixed within round, accept certified improvement immediately',
                timeout='no B32 if construction ends after deadline; certify complete feasible calls, exclude late improvements',
                order='cyclic policy order balanced across instance and seed; shuffled groups, four single-thread workers',
                preparation='source replay and frozen model loading outside local timer, reported separately'),
            inference='development states only; matched analysis clustered by workbook, online average seeds per workbook; '
                      'no independent generalization or 3D collision claim; no B32 or split catalog changes'))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    if args.mode_study=='diversity-diagnostic':
        prior=[r for r in rows if r.get('record')=='pair_run' and r.get('version')=='parent-pair-v1']
        done={(r['cohort'],r['instance']) for r in rows if r.get('record')=='diversity_run' and r.get('version')==DIVERSITY_VERSION}
        jobs=[(sources[r['instance'],r['cohort']],r) for r in prior if (r['cohort'],r['instance']) not in done]
        worker=_diversity_worker
    else:
        done={(r['instance'],r['repairer'],r['seed'],r['policy']) for r in rows
            if r.get('record')=='diversity_online_run' and r.get('version')==DIVERSITY_VERSION}
        jobs=[]
        for instance_index,((instance,cohort),source) in enumerate(sorted((k,v) for k,v in sources.items() if k[1]=='BASE12')):
            for seed_index,seed in enumerate(DIVERSITY_SEEDS):
                for repairer_index,repairer in enumerate(('NEW','GNN')):
                    shift=(instance_index+seed_index+repairer_index)%3
                    order=DIVERSITY_POLICIES[shift:]+DIVERSITY_POLICIES[:shift]
                    policies=tuple(p for p in order if (instance,repairer,seed,p) not in done)
                    if policies:jobs.append((source,repairer,seed,policies))
        random.Random(20261031).shuffle(jobs);worker=_diversity_online_worker
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        pending=[pool.submit(worker,job) for job in jobs]
        for future in as_completed(pending):
            result=future.result()
            for row in result if isinstance(result,list) else [result]:
                _study_append(row)
                if row['record']=='diversity_run':
                    new=[t for c in row['contexts'] for t in c['trials'] if not t['reused']]
                    print('DIVERSITY',row['cohort'],row['instance'],'new',len(new),Counter(t['status'] for t in new),flush=True)
                else:
                    print('ONLINE',row['instance'],row['repairer'],row['seed'],row['policy'],row['status'],
                        'Cmax',round(row['cmax'],6),'calls',row['reference_calls'],'pair_accept',row['pair_accepted'],flush=True)


def summarize_diversity_study(*,save=False):
    from collections import defaultdict
    from statistics import mean
    import random
    rows=_study_rows()
    prior={(r['instance'],r['cohort']):r for r in rows if r.get('record')=='pair_run' and r.get('version')=='parent-pair-v1'}
    diagnostic=[r for r in rows if r.get('record')=='diversity_run' and r.get('version')==DIVERSITY_VERSION]
    online=[r for r in rows if r.get('record')=='diversity_online_run' and r.get('version')==DIVERSITY_VERSION]
    def compare(values):
        ds=[mean(v) for _,v in sorted(values.items())];rng=random.Random(20261031)
        boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(20000))
        return dict(units=len(ds),mean_pct=mean(ds),win=sum(v<-1e-8 for v in ds),tie=sum(abs(v)<=1e-8 for v in ds),
            loss=sum(v>1e-8 for v in ds),descriptive_ci95=[boot[500],boot[19499]])
    differences=defaultdict(lambda:defaultdict(list));stats=defaultdict(Counter);decisions=[];calls=0;certs=0;new_statuses=Counter()
    for run in diagnostic:
        old=prior[run['instance'],run['cohort']]
        lookup={(c['repairer'],c['q'],c['seed']):c for c in old['contexts']}
        for context in run['contexts']:
            base=lookup[context['repairer'],context['q'],context['seed']]
            trials={t['proposal_id']:t for t in base['trials']}
            trials.update({t['proposal_id']:t for t in context['trials']})
            for t in context['trials']:
                if not t['reused']:calls+=t['reference_calls'];certs+=t['certified'];new_statuses[t['status']]+=1
            portfolios={p:base['portfolios'][p] for p in ('SINGLE','MIXED')}
            portfolios['DIVERSE']=context['portfolio']
            if len({len(ids) for ids in portfolios.values()})!=1:raise ValueError('diverse matched budget differs')
            outcomes={}
            for policy,ids in portfolios.items():
                chosen=[trials[i] for i in ids]
                outcomes[policy]=min([base['base']]+[t['cmax'] for t in chosen if t['certified']])
                stats[run['cohort'],context['repairer'],policy].update(contexts=1,candidates=len(chosen),
                    feasible=sum(t['certified'] for t in chosen),improvements=int(outcomes[policy]<base['base']-1e-7))
            for reference in ('SINGLE','MIXED'):
                delta=100*(outcomes['DIVERSE']/outcomes[reference]-1)
                differences['MATCHED',run['cohort'],context['repairer'],'DIVERSE/'+reference][run['instance']].append(delta)
                differences['MATCHED','POOLED_WORKBOOK',context['repairer'],'DIVERSE/'+reference][run['instance'].split('::')[0]].append(delta)
            decisions.append(dict(instance=run['instance'],cohort=run['cohort'],repairer=context['repairer'],q=context['q'],
                seed=context['seed'],base=base['base'],outcomes=outcomes))
    online_lookup={(r['instance'],r['repairer'],r['seed'],r['policy']):r for r in online}
    groups=sorted({k[:3] for k in online_lookup});online_stats=[]
    for instance,repairer,seed in groups:
        group={p:online_lookup.get((instance,repairer,seed,p)) for p in DIVERSITY_POLICIES}
        if any(r is None for r in group.values()):continue
        if any(r['status']!='CERTIFIED' for r in group.values()):continue
        for policy,reference in (('MIXED','SINGLE'),('DIVERSE','SINGLE'),('DIVERSE','MIXED')):
            differences['ONLINE','BASE12',repairer,policy+'/'+reference][instance].append(100*(group[policy]['cmax']/group[reference]['cmax']-1))
    for repairer in ('NEW','GNN'):
        for policy in DIVERSITY_POLICIES:
            group=[r for r in online if r['repairer']==repairer and r['policy']==policy]
            if not group:continue
            online_stats.append(dict(repairer=repairer,policy=policy,runs=len(group),statuses=dict(Counter(r['status'] for r in group)),
                mean_source_pct=mean(100*(r['cmax']/r['initial_cmax']-1) for r in group),
                reference_calls=sum(r['reference_calls'] for r in group),certifications=sum(r['certifier_calls'] for r in group),
                accepted=sum(r['accepted'] for r in group),pair_accepted=sum(r['pair_accepted'] for r in group),
                late_candidates=sum(r['late_candidates'] for r in group),
                mean_actual_seconds=mean(r['actual_runtime'] for r in group),max_overshoot=max(r['overshoot'] for r in group),
                mean_cpu_seconds=mean(r['cpu_seconds'] for r in group)))
    result=dict(record='diversity_analysis',version=DIVERSITY_VERSION,diagnostic_runs=len(diagnostic),online_runs=len(online),
        new_diagnostic_calls=calls,new_diagnostic_certifications=certs,new_diagnostic_statuses=dict(new_statuses),
        source_replays=len(diagnostic)+sum(r['source_reference_calls'] for r in online),
        matched_stats=[dict(cohort=k[0],repairer=k[1],policy=k[2],**v) for k,v in sorted(stats.items())],
        comparisons=[dict(stage=k[0],cohort=k[1],repairer=k[2],comparison=k[3],**compare(v)) for k,v in sorted(differences.items())],
        online_stats=online_stats,matched_decisions=decisions,
        interpretation='matched averages six contexts per layout, pooled clusters workbook; online averages two seeds per workbook; '
            'bootstrap20000 seed20261031 descriptive and unadjusted, consumed development states; online is 20s local continuation')
    if save:
        if len(diagnostic)!=21 or len(online)!=144:raise ValueError('finish planned 21 diagnostic and 144 online runs before final summary')
        if any(r['status']!='CERTIFIED' for r in online):raise ValueError('online errors require inspection before aggregate conclusion')
        if not any(r.get('record')=='diversity_analysis' and r.get('version')==DIVERSITY_VERSION for r in rows):_study_append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='matched_decisions'},ensure_ascii=False),flush=True)
    return result


def audit_diversity_study():
    from collections import defaultdict
    from statistics import mean
    import os,platform
    rows=_study_rows()
    if any(r.get('record')=='diversity_audit' and r.get('version')==DIVERSITY_VERSION for r in rows):
        print('DIVERSITY_AUDIT already saved',flush=True);return
    online=[r for r in rows if r.get('record')=='diversity_online_run' and r.get('version')==DIVERSITY_VERSION]
    sources=_pair_study_sources(rows)
    expected={(name,repairer,seed,policy) for name,cohort in sources if cohort=='BASE12'
        for repairer in ('NEW','GNN') for seed in DIVERSITY_SEEDS for policy in DIVERSITY_POLICIES}
    actual={(r['instance'],r['repairer'],r['seed'],r['policy']) for r in online}
    if actual!=expected or len(online)!=len(expected):raise ValueError('online cohort incomplete or duplicated')
    actions=defaultdict(Counter);anytime=defaultdict(list);source_positions=defaultdict(Counter)
    replays=[];audited=0;late_better=0;late_seconds=0.
    for run in online:
        if run['status']!='CERTIFIED':raise ValueError('online execution error')
        source=sources[run['instance'],'BASE12']
        current=source['cmax'];patterns=source['solution']['patterns'];routes=[r['block_ids'] for r in source['solution']['routes']]
        directions=source['directions'];iteration=-1;origin_patterns=None;events=[[0.,current,'SOURCE']]
        if run['initial_cmax']!=current:raise ValueError('unequal online starting incumbent')
        if sum(t['reference_calls'] for t in run['trace'])!=run['reference_calls']:raise ValueError('online call accounting differs')
        if sum(t['certified'] for t in run['trace'])!=run['certifier_calls']:raise ValueError('online certification accounting differs')
        source_positions[run['repairer'],run['policy']][run['policy_order'].index(run['policy'])]+=1
        previous_elapsed=0.
        for trial in run['trace']:
            audited+=1
            if (trial['status']=='FEASIBLE')!=bool(trial['certified']):raise ValueError('online status/certification differs')
            if bool(trial['certified'])!=(trial['cmax'] is not None):raise ValueError('online status/objective differs')
            if trial['late']!=(trial['elapsed']>run['budget']):raise ValueError('online late flag differs')
            if trial['elapsed']<previous_elapsed:raise ValueError('nonmonotonic trace time')
            previous_elapsed=trial['elapsed']
            if trial['iteration']!=iteration:
                iteration=trial['iteration'];origin_patterns={p['parent_id']:p for p in patterns}
                if abs(trial['round_source_cmax']-current)>1e-7:raise ValueError('round origin differs')
            if abs(trial['incumbent_before']-current)>1e-7:raise ValueError('incumbent trace differs')
            if 'patterns' in trial:
                final={p['parent_id']:p for p in trial['patterns']}
                requested={p['parent_id']:p for p in trial['changed_modes']}
                changed={pid for pid in final if final[pid]!=origin_patterns[pid]}
                if changed!=set(requested) or not changed<=set(trial['removed']):raise ValueError('online effective mode changes differ')
                if any(final[pid]!=p for pid,p in requested.items()):raise ValueError('online requested mode differs')
            should_accept=bool(trial['certified'] and trial['elapsed']<=run['budget']
                and trial['cmax']<current-1e-9*max(1.,current))
            if trial['accepted']!=should_accept:raise ValueError('online deadline/acceptance differs')
            if trial['late'] and trial['certified'] and trial['cmax']<current-1e-7:
                late_better+=1;late_seconds+=current-trial['cmax']
            if should_accept:
                current=trial['cmax'];patterns=trial['patterns'];routes=trial['routes'];directions=trial['directions']
                events.append([trial['elapsed'],current,trial['proposal_id']])
            if abs(trial['incumbent_after']-current)>1e-7:raise ValueError('accepted solution trace differs')
            kind=('FIXED','SINGLE','PAIR')[len(trial['changed_modes'])]
            actions[run['repairer'],run['policy'],kind].update(candidates=1,calls=trial['reference_calls'],
                certified=trial['certified'],accepted=trial['accepted'],late=trial['late'],
                construction_seconds=trial['construction_seconds'],candidate_seconds=trial['seconds'],
                accepted_reduction_seconds=trial['incumbent_before']-trial['incumbent_after'])
        if abs(current-run['cmax'])>1e-7 or run['best_events']!=events:raise ValueError('final online objective/events differ')
        if run['solution']['patterns']!=patterns or [r['block_ids'] for r in run['solution']['routes']]!=routes or run['directions']!=directions:
            raise ValueError('saved final online solution differs')
        for checkpoint in (5.,10.,20.):
            value=min(e[1] for e in events if e[0]<=checkpoint)
            anytime[run['repairer'],run['policy'],checkpoint].append(100*(value/source['cmax']-1))
    for repairer in ('NEW','GNN'):
        for policy in DIVERSITY_POLICIES:
            run=min((r for r in online if r['repairer']==repairer and r['policy']==policy),
                key=lambda r:(r['cmax']/r['initial_cmax'],r['instance'],r['seed']))
            replay=_replay_study_source(run)
            replays.append(dict(instance=run['instance'],repairer=repairer,policy=policy,seed=run['seed'],
                cmax=replay.schedule.cmax,certified=True,relative_source_pct=100*(run['cmax']/run['initial_cmax']-1)))
    prefix=[]
    online_lookup={(r['instance'],r['repairer'],r['seed'],r['policy']):r for r in online}
    for instance,repairer,seed in sorted({k[:3] for k in online_lookup}):
        group={p:online_lookup[instance,repairer,seed,p] for p in DIVERSITY_POLICIES}
        called={p:[t for t in r['trace'] if t['reference_calls'] and not t['late']] for p,r in group.items()}
        quota=min(len(v) for v in called.values())
        outcomes={p:min([r['initial_cmax']]+[t['cmax'] for t in called[p][:quota] if t['certified']]) for p,r in group.items()}
        prefix.append(dict(instance=instance,repairer=repairer,seed=seed,common_completed_calls=quota,
            prefix_cmax=outcomes,timed_cmax={p:r['cmax'] for p,r in group.items()}))
    record=dict(record='diversity_audit',version=DIVERSITY_VERSION,platform=platform.platform(),logical_cpus=os.cpu_count(),
        online_runs=len(online),audited_candidates=audited,late_better_excluded=late_better,
        late_better_reductions_excluded_seconds=late_seconds,reference_calls=len(replays),certifications=len(replays),replays=replays,
        action_statistics=[dict(repairer=k[0],policy=k[1],kind=k[2],**v) for k,v in sorted(actions.items())],
        anytime=[dict(repairer=k[0],policy=k[1],checkpoint=k[2],mean_source_pct=mean(v)) for k,v in sorted(anytime.items())],
        policy_positions=[dict(repairer=k[0],policy=k[1],positions=dict(v)) for k,v in sorted(source_positions.items())],
        posthoc_common_prefix=prefix,
        note='all traces checked for same source, exact requested mode changes, deadline acceptance and saved final solution; '
             'six largest relative final improvements replayed, not an independent validation set; raw accepted step reductions are not causal component effects; '
             'common-call prefixes are retrospective trajectory truncations with a data-dependent shared quota, not another controlled benchmark')
    _study_append(record)
    print(json.dumps({k:v for k,v in record.items() if k!='posthoc_common_prefix'},ensure_ascii=False),flush=True)


SPARSE_VERSION='parent-sparse-v1'
SPARSE_POLICIES=('SINGLE','PERIODIC','STALL')
SPARSE_SEEDS=(2026103103,2026103104)


def _sparse_online_search(current,repairer,policy,seed,budget,model):
    """Retain every single-mode slot; optionally append one low-frequency pair."""
    from dataclasses import asdict
    from mrta_search.lns import select_bottleneck_parents,rank_parent_mode_changes,rank_parent_mode_pairs
    if policy=='SINGLE':
        return dict(_online_bundle_search(current,repairer,policy,seed,budget,model),round_log=[],pair_requests=[])
    if policy not in ('PERIODIC','STALL'):raise ValueError(policy)
    started=time.perf_counter();cpu_started=time.process_time();deadline=started+budget
    initial=current.schedule.cmax;trace=[];events=[[0.,initial,'SOURCE']]
    rounds=[];requests=[];iteration=0;completed_since=0;failed_since=0;error=None

    def evaluate(origin,partial,changed,name,local_seed,selector,q,position,phase,proposal_seconds):
        nonlocal current,error
        before=current.schedule.cmax
        trial,candidate=_evaluate_mode_bundle(origin,partial,changed,repairer,local_seed,model,
            deadline=deadline,return_candidate=True)
        elapsed=time.perf_counter()-started
        accepted=bool(trial['certified'] and elapsed<=budget and trial['cmax']<before-1e-9*max(1.,before))
        if accepted:
            current=candidate;events.append([elapsed,current.schedule.cmax,name])
        trial.update(iteration=iteration,position=position,phase=phase,proposal_id=name,q=q,
            selection=selector.value,seed=local_seed,removed=partial.removed_parent_ids,
            round_source_cmax=origin.schedule.cmax,incumbent_before=before,incumbent_after=current.schedule.cmax,
            accepted=accepted,late=elapsed>budget,elapsed=elapsed,proposal_seconds=proposal_seconds)
        trace.append(trial)
        if trial['status']=='EXECUTION_ERROR':error=trial['error']
        return trial

    while time.perf_counter()<deadline:
        origin=current;local_seed=seed+iteration;selector=shared.pilot.SELECTORS[iteration%3]
        q=min(len(current.solution.parents),(4,8)[(iteration//3)%2]);tick=time.perf_counter()
        partial=select_bottleneck_parents(origin.solution,origin.directions,origin.schedule,selector,
            q=q,seed=local_seed,config=CONFIG)
        singles=rank_parent_mode_changes(origin.solution,partial,CONFIG)[:3]
        proposals=[('O',())]+[('S'+str(i+1),(p,)) for i,p in enumerate(singles)]
        proposal_seconds=time.perf_counter()-tick;start_trace=len(trace)
        for position,(name,changed) in enumerate(proposals):
            if time.perf_counter()>=deadline:break
            evaluate(origin,partial,changed,name,local_seed,selector,q,position,'NORMAL',
                proposal_seconds if position==0 else 0.)
            if error is not None:break
        normal=trace[start_trace:]
        complete=(len(normal)==len(proposals) and all(not t['late'] for t in normal) and error is None)
        improved=any(t['accepted'] for t in normal)
        if complete:
            completed_since+=1;failed_since=0 if improved else failed_since+1
        due=complete and completed_since>=3 and (policy=='PERIODIC' or failed_since>=3)
        round_row=dict(iteration=iteration,proposal_ids=[p[0] for p in proposals],completed=complete,
            evaluated=len(normal),improved=improved,completed_since=completed_since,failed_since=failed_since,
            due=due,request_index=None,after_normal=current.schedule.cmax)
        if due and time.perf_counter()<deadline:
            attempt=len(requests);pair_seed=seed+100000+attempt
            pair_selector=shared.pilot.SELECTORS[attempt%3]
            pair_q=min(len(current.solution.parents),(4,8)[attempt%2])
            pair_origin=current;tick=time.perf_counter()
            pair_partial=select_bottleneck_parents(current.solution,current.directions,current.schedule,pair_selector,
                q=pair_q,seed=pair_seed,config=CONFIG)
            pairs=rank_parent_mode_pairs(current.solution,pair_partial,CONFIG,limit=1)
            proposal_seconds=time.perf_counter()-tick
            request=dict(attempt=attempt,iteration=iteration,q=pair_q,selection=pair_selector.value,seed=pair_seed,
                removed=pair_partial.removed_parent_ids,origin_cmax=current.schedule.cmax,available=bool(pairs),
                proposal_seconds=proposal_seconds,evaluated=False,status='NO_PAIR',completed_since=completed_since,
                failed_since=failed_since,changed_modes=[] if not pairs else [asdict(p) for p in pairs[0]])
            if pairs and time.perf_counter()<deadline:
                trial=evaluate(pair_origin,pair_partial,pairs[0],'P1',pair_seed,pair_selector,pair_q,0,'PAIR',proposal_seconds)
                request.update(evaluated=True,status=trial['status'],accepted=trial['accepted'],late=trial['late'])
            elif pairs:request['status']='PROPOSAL_FINISHED_AFTER_DEADLINE'
            request.update(elapsed=time.perf_counter()-started,after=current.schedule.cmax)
            requests.append(request);round_row['request_index']=attempt
            completed_since=failed_since=0
        rounds.append(round_row);iteration+=1
        if error is not None:break
    elapsed=time.perf_counter()-started
    return dict(status='CERTIFIED' if error is None else 'EXECUTION_ERROR',error=error,initial_cmax=initial,
        cmax=current.schedule.cmax,budget=budget,actual_runtime=elapsed,cpu_seconds=time.process_time()-cpu_started,
        overshoot=max(0.,elapsed-budget),iterations=iteration,trace=trace,best_events=events,round_log=rounds,pair_requests=requests,
        reference_calls=sum(t['reference_calls'] for t in trace),certifier_calls=sum(t['certified'] for t in trace),
        accepted=sum(t['accepted'] for t in trace),mode_accepted=sum(t['accepted'] and bool(t['changed_modes']) for t in trace),
        pair_accepted=sum(t['accepted'] and len(t['changed_modes'])==2 for t in trace),
        late_candidates=sum(t['late'] for t in trace),solution=asdict(current.solution),directions=current.directions)


def _sparse_study_worker(job):
    source,repairer,seed,policies=job;tick=time.perf_counter()
    current=_replay_study_source(source)
    model=shared.load_model(shared.pilot.MODEL)[0] if repairer=='GNN' else None
    preparation=time.perf_counter()-tick;rows=[]
    for index,policy in enumerate(policies):
        result=_sparse_online_search(current,repairer,policy,seed,30.,model)
        rows.append(dict(record='sparse_run',version=SPARSE_VERSION,instance=source['instance'],repairer=repairer,seed=seed,
            source_seed=source['seed'],policy=policy,policy_order=policies,source_reference_calls=int(index==0),
            source_certified=True,group_preparation_seconds=preparation if index==0 else 0.,**result))
    return rows


def run_sparse_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os,random
    rows=_study_rows();sources={k[0]:v for k,v in _pair_study_sources(rows).items() if k[1]=='BASE12'}
    if not any(r.get('record')=='sparse_plan' and r.get('version')==SPARSE_VERSION for r in rows):
        _study_append(dict(record='sparse_plan',version=SPARSE_VERSION,instances=sorted(sources),seeds=SPARSE_SEEDS,
            repairers=['NEW','GNN'],policies=SPARSE_POLICIES,budget_seconds=30,workers=args.workers,
            hypothesis='preserving S1/S2/S3 and reducing pair frequency may avoid high-frequency pair opportunity cost; '
                       'PERIODIC separates low frequency from stagnation conditioning',
            normal='same O+top3 singles, RANDOM/TRAVEL/LOAD, q4/q8, seeds and repair as prior SINGLE; '
                   'freeze origin within normal round, accept certified improvements on completion',
            periodic='after 3 complete timely normal rounds since last pair request, append one pair request',
            stall='same minimum spacing, plus 3 consecutive complete normal rounds without accepted improvement; '
                  'an improving normal round resets failure count; every pair request resets both counters',
            pair='fresh partial from current incumbent after normal round; attempt index j chooses RANDOM/TRAVEL/LOAD[j%3], '
                 'q=(4,8)[j%2], seed=run_seed+100000+j; top1 original load-ranked pair; no new cut points; '
                 'no available pair still consumes request and resets counters',
            timing='all proposal/repair/B32/certification within 30s; source replay/model preload outside local timer; '
                   'complete noninterruptible calls, exclude late improvements; model unchanged',
            design='12 workbooks x 2 new seeds x 2 repairers x 3 policies = 144 fresh runs; '
                   'policy order cyclically balanced, groups shuffled, single-thread workers',
            statistics='primary paired 30s final Cmax, average seeds within workbook; 10/20s descriptive secondary; '
                       'bootstrap20000 seed20261031 unadjusted; consumed development states, no independent test claim'))
    done={(r['instance'],r['repairer'],r['seed'],r['policy']) for r in rows
        if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION}
    jobs=[]
    for instance_index,(instance,source) in enumerate(sorted(sources.items())):
        for seed_index,seed in enumerate(SPARSE_SEEDS):
            for repairer_index,repairer in enumerate(('NEW','GNN')):
                shift=(instance_index+seed_index+repairer_index)%3
                order=SPARSE_POLICIES[shift:]+SPARSE_POLICIES[:shift]
                policies=tuple(p for p in order if (instance,repairer,seed,p) not in done)
                if policies:jobs.append((source,repairer,seed,policies))
    random.Random(20261031).shuffle(jobs)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        for future in as_completed([pool.submit(_sparse_study_worker,job) for job in jobs]):
            for row in future.result():
                _study_append(row)
                print('SPARSE',row['instance'],row['repairer'],row['seed'],row['policy'],row['status'],
                    'Cmax',round(row['cmax'],6),'calls',row['reference_calls'],'pair_requests',len(row['pair_requests']),
                    'pair_accepted',row['pair_accepted'],flush=True)


def summarize_sparse_study(*,save=False):
    from collections import defaultdict
    from statistics import mean
    import random
    rows=_study_rows();runs=[r for r in rows if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION]
    lookup={(r['instance'],r['repairer'],r['seed'],r['policy']):r for r in runs}
    paired=defaultdict(lambda:defaultdict(list));summary=[]
    for instance,repairer,seed in sorted({k[:3] for k in lookup}):
        group={p:lookup.get((instance,repairer,seed,p)) for p in SPARSE_POLICIES}
        if any(r is None or r['status']!='CERTIFIED' for r in group.values()):continue
        for checkpoint in (10.,20.,30.):
            values={p:min(e[1] for e in r['best_events'] if e[0]<=checkpoint) for p,r in group.items()}
            for policy,reference in (('PERIODIC','SINGLE'),('STALL','SINGLE'),('STALL','PERIODIC')):
                paired[repairer,policy+'/'+reference,checkpoint][instance].append(100*(values[policy]/values[reference]-1))
    comparisons=[]
    for (repairer,comparison,checkpoint),by_instance in sorted(paired.items()):
        ds=[mean(v) for _,v in sorted(by_instance.items())];rng=random.Random(20261031)
        boot=sorted(mean(rng.choices(ds,k=len(ds))) for _ in range(20000))
        comparisons.append(dict(repairer=repairer,comparison=comparison,checkpoint=checkpoint,units=len(ds),mean_pct=mean(ds),
            win=sum(x<-1e-8 for x in ds),tie=sum(abs(x)<=1e-8 for x in ds),loss=sum(x>1e-8 for x in ds),
            descriptive_ci95=[boot[500],boot[19499]]))
    for repairer in ('NEW','GNN'):
        for policy in SPARSE_POLICIES:
            group=[r for r in runs if r['repairer']==repairer and r['policy']==policy]
            if not group:continue
            pairs=[t for r in group for t in r['trace'] if len(t['changed_modes'])==2]
            summary.append(dict(repairer=repairer,policy=policy,runs=len(group),statuses=dict(Counter(r['status'] for r in group)),
                mean_source_pct=mean(100*(r['cmax']/r['initial_cmax']-1) for r in group),
                reference_calls=sum(r['reference_calls'] for r in group),certifications=sum(r['certifier_calls'] for r in group),
                accepted=sum(r['accepted'] for r in group),pair_requests=sum(len(r['pair_requests']) for r in group),
                pair_calls=sum(t['reference_calls'] for t in pairs),pair_certifications=sum(t['certified'] for t in pairs),
                pair_accepted=sum(t['accepted'] for t in pairs),pair_candidate_seconds=sum(t['seconds'] for t in pairs),
                total_candidate_seconds=sum(t['seconds'] for r in group for t in r['trace']),
                runs_with_pair_call=sum(any(len(t['changed_modes'])==2 and t['reference_calls'] for t in r['trace']) for r in group),
                mean_actual_seconds=mean(r['actual_runtime'] for r in group),max_overshoot=max(r['overshoot'] for r in group)))
    result=dict(record='sparse_analysis',version=SPARSE_VERSION,runs=len(runs),source_replays=sum(r['source_reference_calls'] for r in runs),
        candidates=sum(len(r['trace']) for r in runs),candidate_statuses=dict(Counter(t['status'] for r in runs for t in r['trace'])),
        summary=summary,comparisons=comparisons,
        note='30s primary; 10/20s descriptive secondary, not independent trials; new seeds and rerun SINGLE, '
             'average seeds per workbook, bootstrap20000 unadjusted; all data already used in development')
    if save:
        expected={(name,repairer,seed,p) for (name,cohort) in _pair_study_sources(rows) if cohort=='BASE12'
            for repairer in ('NEW','GNN') for seed in SPARSE_SEEDS for p in SPARSE_POLICIES}
        if set(lookup)!=expected or len(runs)!=len(expected):raise ValueError('complete the 144 planned sparse runs')
        if any(r['status']!='CERTIFIED' for r in runs):raise ValueError('inspect sparse execution errors first')
        if not any(r.get('record')=='sparse_analysis' and r.get('version')==SPARSE_VERSION for r in rows):_study_append(result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def audit_sparse_study():
    from collections import defaultdict
    import platform,os
    rows=_study_rows()
    if any(r.get('record')=='sparse_audit' and r.get('version')==SPARSE_VERSION for r in rows):
        print('SPARSE_AUDIT already saved',flush=True);return
    runs=[r for r in rows if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION]
    sources={k[0]:v for k,v in _pair_study_sources(rows).items() if k[1]=='BASE12'}
    expected={(name,repairer,seed,p) for name in sources for repairer in ('NEW','GNN')
        for seed in SPARSE_SEEDS for p in SPARSE_POLICIES}
    if len(runs)!=len(expected) or {(r['instance'],r['repairer'],r['seed'],r['policy']) for r in runs}!=expected:
        raise ValueError('sparse run identities incomplete or duplicated')
    actions=defaultdict(Counter);positions=defaultdict(Counter);late_better=[];replays=[];audited=0;rounds_checked=0
    for run in runs:
        if run['status']!='CERTIFIED':raise ValueError('inspect sparse execution error')
        source=sources[run['instance']];current=source['cmax'];patterns=source['solution']['patterns']
        routes=[r['block_ids'] for r in source['solution']['routes']];directions=source['directions']
        events=[[0.,current,'SOURCE']];phase_key=None;origin_patterns=None;phase_positions=0;previous_elapsed=0.
        if run['initial_cmax']!=current:raise ValueError('sparse source differs')
        positions[run['repairer'],run['policy']][run['policy_order'].index(run['policy'])]+=1
        normal_by_round=defaultdict(list);pairs_by_round=defaultdict(list)
        for trial in run['trace']:
            audited+=1;phase=trial.get('phase','NORMAL');key=(trial['iteration'],phase)
            if trial['elapsed']<previous_elapsed:raise ValueError('nonmonotonic sparse time')
            previous_elapsed=trial['elapsed']
            if key!=phase_key:
                phase_key=key;origin_patterns={p['parent_id']:p for p in patterns};phase_positions=0
                if abs(trial['round_source_cmax']-current)>1e-7:raise ValueError('sparse phase origin differs')
            if trial['position']!=phase_positions:raise ValueError('sparse candidate order differs')
            phase_positions+=1
            if phase=='NORMAL':
                normal_by_round[trial['iteration']].append(trial)
                name='O' if trial['position']==0 else 'S'+str(trial['position'])
                if trial['proposal_id']!=name or len(trial['changed_modes'])!=int(name!='O'):
                    raise ValueError('a single-mode slot was displaced')
            else:
                pairs_by_round[trial['iteration']].append(trial)
                if phase!='PAIR' or trial['proposal_id']!='P1' or len(trial['changed_modes'])!=2:
                    raise ValueError('unexpected sparse probe')
            if trial['status']=='FEASIBLE' and not trial['certified']:raise ValueError('uncertified sparse candidate')
            if bool(trial['certified'])!=(trial['cmax'] is not None):raise ValueError('sparse objective/certificate differs')
            if trial['late']!=(trial['elapsed']>run['budget']):raise ValueError('sparse late flag differs')
            if abs(trial['incumbent_before']-current)>1e-7:raise ValueError('sparse incumbent before differs')
            if 'patterns' in trial:
                actual={p['parent_id']:p for p in trial['patterns']};requested={p['parent_id']:p for p in trial['changed_modes']}
                changed={p for p in actual if actual[p]!=origin_patterns[p]}
                if changed!=set(requested) or not changed<=set(trial['removed']):raise ValueError('sparse effective changes differ')
                if any(actual[p]!=v for p,v in requested.items()):raise ValueError('sparse requested pattern differs')
            accept=bool(trial['certified'] and trial['elapsed']<=run['budget'] and trial['cmax']<current-1e-9*max(1.,current))
            if trial['accepted']!=accept:raise ValueError('sparse deadline acceptance differs')
            if trial['late'] and trial['certified'] and trial['cmax']<current-1e-7:
                late_better.append(dict(instance=run['instance'],repairer=run['repairer'],policy=run['policy'],seed=run['seed'],
                    elapsed=trial['elapsed'],before=current,excluded_cmax=trial['cmax']))
            if accept:
                current=trial['cmax'];patterns=trial['patterns'];routes=trial['routes'];directions=trial['directions']
                events.append([trial['elapsed'],current,trial['proposal_id']])
            if abs(trial['incumbent_after']-current)>1e-7:raise ValueError('sparse incumbent after differs')
            kind=('FIXED','SINGLE','PAIR')[len(trial['changed_modes'])]
            actions[run['repairer'],run['policy'],kind].update(candidates=1,calls=trial['reference_calls'],certified=trial['certified'],
                accepted=trial['accepted'],late=trial['late'],candidate_seconds=trial['seconds'],
                construction_seconds=trial['construction_seconds'],accepted_reduction_seconds=trial['incumbent_before']-current)
        if current!=run['cmax'] or events!=run['best_events']:raise ValueError('sparse final objective/events differ')
        if patterns!=run['solution']['patterns'] or routes!=[r['block_ids'] for r in run['solution']['routes']] or directions!=run['directions']:
            raise ValueError('sparse final solution differs')
        if sum(t['reference_calls'] for t in run['trace'])!=run['reference_calls']:raise ValueError('sparse call count differs')
        if sum(t['certified'] for t in run['trace'])!=run['certifier_calls']:raise ValueError('sparse certifier count differs')
        if run['policy']=='SINGLE':
            if run['pair_requests'] or pairs_by_round:raise ValueError('pair in SINGLE control')
            continue
        completed_since=failed_since=request_index=0
        for index,rr in enumerate(run['round_log']):
            rounds_checked+=1
            normal=normal_by_round[index]
            expected_ids=['O']+['S'+str(j) for j in range(1,len(rr['proposal_ids']))]
            if rr['iteration']!=index or rr['proposal_ids']!=expected_ids or len(expected_ids)>4:
                raise ValueError('sparse normal-round plan differs')
            if [t['proposal_id'] for t in normal]!=expected_ids[:len(normal)] or rr['evaluated']!=len(normal):
                raise ValueError('sparse normal-round prefix differs')
            complete=len(normal)==len(expected_ids) and all(not t['late'] for t in normal)
            improved=any(t['accepted'] for t in normal)
            if complete:completed_since+=1;failed_since=0 if improved else failed_since+1
            due=complete and completed_since>=3 and (run['policy']=='PERIODIC' or failed_since>=3)
            if (rr['completed'],rr['improved'],rr['completed_since'],rr['failed_since'],rr['due'])!=(complete,improved,completed_since,failed_since,due):
                raise ValueError('sparse trigger counters differ')
            if rr['request_index'] is not None:
                if not due or rr['request_index']!=request_index:raise ValueError('unscheduled sparse pair request')
                req=run['pair_requests'][request_index];pair_trials=pairs_by_round[index]
                if (req['attempt'],req['iteration'],req['completed_since'],req['failed_since'])!=(request_index,index,completed_since,failed_since):
                    raise ValueError('sparse request counters differ')
                if req['seed']!=run['seed']+100000+request_index or req['selection']!=shared.pilot.SELECTORS[request_index%3].value:
                    raise ValueError('sparse pair context seed/selector differs')
                if req['q']!=min(len(source['solution']['parents']),(4,8)[request_index%2]):raise ValueError('sparse pair q differs')
                if abs(req['origin_cmax']-rr['after_normal'])>1e-7:raise ValueError('probe did not use updated incumbent')
                if len(pair_trials)!=int(req['evaluated']):raise ValueError('sparse probe execution differs')
                if pair_trials:
                    trial=pair_trials[0]
                    if any(trial[k]!=req[k] for k in ('q','selection','seed','removed','changed_modes','status','accepted','late')):
                        raise ValueError('sparse probe metadata differs')
                request_index+=1;completed_since=failed_since=0
            elif pairs_by_round[index]:raise ValueError('unrecorded pair attempt')
        if request_index!=len(run['pair_requests']):raise ValueError('orphan sparse request')
    for repairer in ('NEW','GNN'):
        for policy in SPARSE_POLICIES:
            run=min((r for r in runs if r['repairer']==repairer and r['policy']==policy),
                key=lambda r:(r['cmax']/r['initial_cmax'],r['instance'],r['seed']))
            replay=_replay_study_source(run)
            replays.append(dict(instance=run['instance'],repairer=repairer,policy=policy,seed=run['seed'],cmax=replay.schedule.cmax,
                certified=True,relative_source_pct=100*(run['cmax']/run['initial_cmax']-1)))
    record=dict(record='sparse_audit',version=SPARSE_VERSION,platform=platform.platform(),logical_cpus=os.cpu_count(),
        audited_candidates=audited,sparse_rounds_checked=rounds_checked,source_runs=len(runs),
        actions=[dict(repairer=k[0],policy=k[1],kind=k[2],**v) for k,v in sorted(actions.items())],
        positions=[dict(repairer=k[0],policy=k[1],counts=dict(v)) for k,v in sorted(positions.items())],
        late_better_excluded=late_better,replays=replays,reference_calls=len(replays),certifications=len(replays),
        note='all traces checked; trigger counts recomputed from completed normal rounds; all single slots precede optional '
             'pair request; requests with no legal pair reset counters as specified; six best relative finals replayed, not independent validation')
    _study_append(record)
    print(json.dumps(record,ensure_ascii=False),flush=True)


def diagnose_sparse_constructor():
    """Post-hoc reconstruction of every distinct failed original-mode GNN context."""
    from dataclasses import asdict
    from mrta_search.lns import _remove_parent_set,_initial_error
    from mrta_reference.geometry import oriented_endpoints
    from mrta_reference.certifier import certify_schedule
    rows=_study_rows()
    if any(r.get('record')=='sparse_constructor_recovery' and r.get('version')==SPARSE_VERSION for r in rows):
        print('SPARSE_CONSTRUCTOR_DIAGNOSTIC already saved',flush=True);return
    runs=[r for r in rows if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION]
    if len(runs)!=144:raise ValueError('finish unchanged formal cohort before constructor diagnostic')
    sources={k[0]:v for k,v in _pair_study_sources(rows).items() if k[1]=='BASE12'}
    cases=[]
    for run in runs:
        if run['repairer']!='GNN':continue
        source=sources[run['instance']];data=source['solution'];patterns=data['patterns']
        routes=[r['block_ids'] for r in data['routes']];directions=source['directions'];cmax=source['cmax']
        for trial in run['trace']:
            if trial['proposal_id']=='O' and trial['status']=='CONSTRUCTION_FAILED' and trial.get('reason')=='NO_LEGAL_PARENT_ACTION':
                snapshot=dict(solution=dict(parents=data['parents'],patterns=patterns,
                    routes=[dict(robot_id=r,block_ids=route) for r,route in enumerate(routes)]),directions=directions,cmax=cmax)
                occurrence=dict(policy=run['policy'],seed=run['seed'],iteration=trial['iteration'])
                existing=next((c for c in cases if c['instance']==run['instance'] and c['snapshot']==snapshot and c['removed']==trial['removed']),None)
                if existing is not None:existing['occurrences'].append(occurrence)
                else:cases.append(dict(instance=run['instance'],snapshot=snapshot,removed=trial['removed'],
                    repair_seed=trial['seed'],occurrences=[occurrence]))
            if trial['accepted']:
                patterns=trial['patterns'];routes=trial['routes'];directions=trial['directions'];cmax=trial['cmax']
    model=shared.load_model(shared.pilot.MODEL)[0];evaluated=[];calls=certifications=0
    for case in cases:
        current=_replay_study_source(case['snapshot']);calls+=1;certifications+=1
        partial=_remove_parent_set(current.solution,tuple(case['removed']))
        directions=shared.pilot.partial_directions(current,partial)
        blocks=block_map(current.solution,CONFIG,scope=FORMAL_SCOPE_V2)
        def points(routes,bits):
            return {r:oriented_endpoints(blocks[route[0]],bits[r][0])[0] for r,route in enumerate(routes) if route}
        full_points=points(tuple(r.block_ids for r in current.solution.routes),current.directions)
        partial_points=points(partial.routes,directions)
        variants=[]
        for label,limit,relax,prefer in (('GNN_GREEDY8',8,False,False),('GNN_ALL_POSITIONS',len(blocks)+1,False,False),
            ('GNN_REPAIR_PARTIAL_START',8,True,False),('GNN_RESTORE_START_PRIORITY',8,True,True)):
            tick=time.perf_counter()
            result=construct_or_repair_with_gnn(joint_geometry(current.solution.parents,partial.patterns,CONFIG),
                partial.routes,directions,partial.removed_parent_ids,model,CONFIG,insertion_limit=limit,
                allow_partial_start_conflicts=relax,prefer_legal_start_recovery=prefer)
            row=dict(method=label,insertion_limit=limit,allow_partial_start_conflicts=relax,prefer_legal_start_recovery=prefer,
                construction_seconds=time.perf_counter()-tick,
                actions=len(result.actions),status='CONSTRUCTION_FAILED',reason=result.rejection_reason,
                cmax=None,certified=False,reference_calls=0)
            if result.solution is not None:
                candidate=shared.pilot.oriented.schedule_explicit(result.solution,result.directions,CONFIG);calls+=1
                row.update(status=candidate.schedule.status.value,reference_calls=1,
                    solution=asdict(result.solution),directions=result.directions,
                    scheduler_diagnostics=candidate.schedule.diagnostics,
                    wait_for_graph=candidate.schedule.wait_for_graph,
                    recovery_rollouts=candidate.schedule.recovery_rollouts,
                    frontier_exhausted=candidate.schedule.frontier_exhausted)
                if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                    check=certify_schedule(result.solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
                    if not check.certified:raise ValueError(check.errors)
                    row.update(cmax=candidate.schedule.cmax,certified=True);certifications+=1
            row['seconds']=time.perf_counter()-tick;variants.append(row)
        rule=_evaluate_mode_bundle(current,partial,(),'NEW',case['repair_seed'],None)
        rule['method']='NEW_ORIGINAL';calls+=rule['reference_calls'];certifications+=rule['certified'];variants.append(rule)
        evaluated.append(dict(case,full_start=full_points,full_start_error=_initial_error(full_points,CONFIG),
            partial_start=partial_points,partial_start_error=_initial_error(partial_points,CONFIG),variants=variants))
        print('CONSTRUCTOR_DIAGNOSTIC',case['instance'],[(v['method'],v['status'],v.get('actions'),v['cmax']) for v in variants],flush=True)
    record=dict(record='sparse_constructor_recovery',version=SPARSE_VERSION,cases=evaluated,
        reference_calls=calls,certifications=certifications,
        scope='post-hoc all distinct original-mode NO_LEGAL_PARENT_ACTION contexts; exact direct equality deduplication, '
              'no hashing/cache; formal 144 runs unchanged and did not enable optional partial-start repair; '
              'only temporarily conflicting partial starts may bypass the insertion prune, legal starts keep it, '
              'complete start validity, B32 and independent certification all retained; no model update')
    _study_append(record)
    print('CONSTRUCTOR_DIAGNOSTIC_SAVED','cases',len(evaluated),'B32',calls,'certifications',certifications,flush=True)


def diagnose_sparse_recovery_budget():
    """Development-only fixed-candidate budget sensitivity, never formal scores."""
    from dataclasses import asdict
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail,OperationKind
    from mrta_reference.solution import canonicalize
    from mrta_reference.geometry import finite_x_split_validator
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scheduler import (_reference_schedule_with,prepare_dispatch_problem,
        _prepared_dispatch_outcome,_limited_discrepancy_dispatch_recovery_optimized)
    rows=_study_rows()
    if any(r.get('record')=='sparse_recovery_budget' and r.get('version')==SPARSE_VERSION for r in rows):
        print('SPARSE_RECOVERY_BUDGET already saved',flush=True);return
    recovery=next(r for r in rows if r.get('record')=='sparse_constructor_recovery' and r.get('version')==SPARSE_VERSION)
    budgets=(32,128,512)
    if not any(r.get('record')=='sparse_recovery_budget_plan' and r.get('version')==SPARSE_VERSION for r in rows):
        _study_append(dict(record='sparse_recovery_budget_plan',version=SPARSE_VERSION,budgets=budgets,
            candidates='every completed GNN_RESTORE_START_PRIORITY candidate from sparse_constructor_recovery',
            scope='fixed routes, patterns and directions; independent fresh recovery per budget; no formal scope changes; '
                  'development physical certificate only, never a formal B32 result or an online improvement'))
    output=[]
    for case in recovery['cases']:
        candidate=next(v for v in case['variants'] if v['method']=='GNN_RESTORE_START_PRIORITY')
        if 'solution' not in candidate:continue
        data=candidate['solution'];directions=candidate['directions']
        parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
        patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
            None if p['rail'] is None else Rail(p['rail'])) for p in data['patterns'])
        solution=canonicalize(parents,patterns,{r['robot_id']:r['block_ids'] for r in data['routes']},CONFIG,scope=FORMAL_SCOPE_V2)
        validator=finite_x_split_validator(parents,CONFIG) if any(p.kind is SplitKind.X_SPLIT for p in patterns) else None
        for budget in budgets:
            def dispatch(templates,cfg,*,directions,profile):
                prepared=prepare_dispatch_problem(templates,cfg,directions=directions,profile=profile)
                baseline=_prepared_dispatch_outcome(prepared,profile=profile,collect_trace=True)
                return _limited_discrepancy_dispatch_recovery_optimized(prepared,baseline,rollout_budget=budget,profile=profile)
            tick=time.perf_counter()
            schedule=_reference_schedule_with(solution,CONFIG,dispatch,
                orientations={r:tuple(bits) for r,bits in enumerate(directions)},x_split_validator=validator)
            row=dict(instance=case['instance'],budget=budget,status=schedule.status.value,cmax=schedule.cmax,
                seconds=time.perf_counter()-tick,recovery_rollouts=schedule.recovery_rollouts,
                frontier_exhausted=schedule.frontier_exhausted,diagnostics=schedule.diagnostics,
                certified_physical=False,formal_b32_result=False)
            if budget==32 and (schedule.status.value!=candidate['status'] or schedule.cmax!=candidate['cmax']):
                raise ValueError('diagnostic B32 differs from saved formal evaluation')
            if schedule.status is ScheduleStatus.FEASIBLE:
                check=certify_schedule(solution,schedule,CONFIG,x_split_validator=validator)
                if not check.certified:raise ValueError(check.errors)
                if schedule.directions!=tuple(tuple(bits) for bits in directions):raise ValueError('directions changed')
                for route in solution.routes:
                    ops=sorted((op for op in schedule.operations if op.robot_id==route.robot_id),key=lambda op:op.sequence_index)
                    if route.block_ids and (not ops or ops[-1].kind is not OperationKind.POST):raise ValueError('terminal policy changed')
                    if not route.block_ids and (ops or schedule.robot_completion[route.robot_id]!=0):raise ValueError('empty route deployed')
                row.update(certified_physical=True,schedule=asdict(schedule))
            output.append(row)
            print('RECOVERY_BUDGET',case['instance'],budget,row['status'],row['cmax'],row['seconds'],flush=True)
    _study_append(dict(record='sparse_recovery_budget',version=SPARSE_VERSION,trials=output,
        diagnostic_dispatch_calls=len(output),certifications=sum(r['certified_physical'] for r in output),
        note='development budget sensitivity, no replacement of the formal evaluator or 144 online results'))


RECOVERY_VERSION='recovery-stratified-v1'
RECOVERY_BUDGETS=(32,128,512)


def _stratified_recovery_sample(rows):
    """Uniform within workbook/repairer/mode-count strata, before larger-budget outcomes."""
    from collections import defaultdict
    import random
    groups=defaultdict(list)
    runs=sorted((r for r in rows if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION),
        key=lambda r:(r['instance'],r['repairer'],r['seed'],r['policy']))
    for run in runs:
        for index,trial in enumerate(run['trace']):
            if trial['status']!='DEADLOCK':continue
            key=(run['instance'],run['repairer'],len(trial['changed_modes']))
            occurrence=dict(policy=run['policy'],seed=run['seed'],trace_index=index,
                incumbent=trial['incumbent_before'],late=trial['late'])
            state=(trial['patterns'],trial['routes'],trial['directions'])
            existing=next((c for c in groups[key] if (c['patterns'],c['routes'],c['directions'])==state),None)
            if existing is not None:
                if (existing['frontier_exhausted'],existing['recovery_rollouts'])!=(trial['frontier_exhausted'],trial['recovery_rollouts']):
                    raise ValueError('identical candidate has differing saved B32 outcome')
                existing['occurrences'].append(occurrence)
            else:
                groups[key].append(dict(instance=key[0],repairer=key[1],mode_count=key[2],
                    parents=run['solution']['parents'],patterns=state[0],routes=state[1],directions=state[2],
                    frontier_exhausted=trial['frontier_exhausted'],recovery_rollouts=trial['recovery_rollouts'],
                    occurrences=[occurrence]))
    rng=random.Random(20261101);cases=[];population=[]
    for key,group in sorted(groups.items()):
        n=min(3,len(group));chosen=sorted(rng.sample(range(len(group)),n))
        population.append(dict(instance=key[0],repairer=key[1],mode_count=key[2],unique=len(group),
            occurrences=sum(len(c['occurrences']) for c in group),sampled=n,
            exhausted_unique=sum(c['frontier_exhausted'] for c in group)))
        for index in chosen:
            cases.append(dict(group[index],case_id=len(cases)+1,stratum_unique=len(group),stratum_sampled=n))
    return cases,population


def _recovery_case_solution(case):
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in case['parents'])
    patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
        None if p['rail'] is None else Rail(p['rail'])) for p in case['patterns'])
    solution=canonicalize(parents,patterns,{r:ids for r,ids in enumerate(case['routes'])},CONFIG,scope=FORMAL_SCOPE_V2)
    if solution.patterns!=patterns or [list(r.block_ids) for r in solution.routes]!=case['routes']:
        raise ValueError('saved recovery candidate changed under canonicalization')
    return solution


def _development_budget_schedule(solution,directions,budget):
    from mrta_reference.geometry import finite_x_split_validator
    from mrta_reference.model import SplitKind
    from mrta_reference.scheduler import (_reference_schedule_with,prepare_dispatch_problem,
        _prepared_dispatch_outcome,_limited_discrepancy_dispatch_recovery_optimized)
    validator=finite_x_split_validator(solution.parents,CONFIG) if any(p.kind is SplitKind.X_SPLIT for p in solution.patterns) else None
    def dispatch(templates,cfg,*,directions,profile):
        prepared=prepare_dispatch_problem(templates,cfg,directions=directions,profile=profile)
        baseline=_prepared_dispatch_outcome(prepared,profile=profile,collect_trace=True)
        return _limited_discrepancy_dispatch_recovery_optimized(prepared,baseline,rollout_budget=budget,profile=profile)
    schedule=_reference_schedule_with(solution,CONFIG,dispatch,
        orientations={r:tuple(bits) for r,bits in enumerate(directions)},x_split_validator=validator)
    return schedule,validator


def _certify_development_budget(solution,directions,schedule,validator):
    """Independent physical certificate plus explicit formal terminal/direction semantics."""
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.model import OperationKind
    check=certify_schedule(solution,schedule,CONFIG,x_split_validator=validator)
    if not check.certified:raise ValueError(check.errors)
    if schedule.directions!=tuple(tuple(bits) for bits in directions):raise ValueError('directions changed')
    for route in solution.routes:
        ops=sorted((op for op in schedule.operations if op.robot_id==route.robot_id),key=lambda op:op.sequence_index)
        if route.block_ids and (not ops or ops[-1].kind is not OperationKind.POST):raise ValueError('terminal policy changed')
        if not route.block_ids and (ops or schedule.robot_completion[route.robot_id]!=0):raise ValueError('empty route deployed')


def _stratified_recovery_worker(case):
    from dataclasses import asdict
    from mrta_search.lns import oriented_route_state
    started=time.perf_counter();solution=_recovery_case_solution(case)
    blocks=block_map(solution,CONFIG,scope=FORMAL_SCOPE_V2)
    tick=time.perf_counter()
    state=oriented_route_state(case['routes'],case['directions'],blocks,CONFIG)
    lower_bound=max(p+sum(arcs) for p,arcs in zip(state.loads,state.arc_costs))
    bound_seconds=time.perf_counter()-tick
    offset=(case['case_id']-1)%3;order=RECOVERY_BUDGETS[offset:]+RECOVERY_BUDGETS[:offset]
    trials=[]
    for budget in order:
        tick=time.perf_counter()
        schedule,validator=_development_budget_schedule(solution,case['directions'],budget)
        dispatch_seconds=time.perf_counter()-tick
        row=dict(budget=budget,status=schedule.status.value,cmax=schedule.cmax,dispatch_seconds=dispatch_seconds,
            recovery_rollouts=schedule.recovery_rollouts,frontier_exhausted=schedule.frontier_exhausted,
            certified_physical=False,formal_b32_result=False,diagnostics=schedule.diagnostics)
        if budget==32 and (schedule.status is not ScheduleStatus.DEADLOCK or
            (schedule.recovery_rollouts,schedule.frontier_exhausted)!=(case['recovery_rollouts'],case['frontier_exhausted'])):
            raise ValueError('saved formal B32 failure does not reproduce')
        if schedule.status is ScheduleStatus.FEASIBLE:
            _certify_development_budget(solution,case['directions'],schedule,validator)
            if schedule.cmax<lower_bound-1e-7:raise ValueError('route lower bound exceeds certified makespan')
            row.update(certified_physical=True,schedule=asdict(schedule))
        elif schedule.status is not ScheduleStatus.DEADLOCK:raise ValueError('unexpected diagnostic status: '+schedule.status.value)
        row['seconds']=time.perf_counter()-tick
        row['improves_first']=bool(row['certified_physical'] and row['cmax']<case['occurrences'][0]['incumbent']-1e-7)
        row['improving_occurrences']=sum(row['certified_physical'] and row['cmax']<o['incumbent']-1e-7 for o in case['occurrences'])
        trials.append(row)
    return dict(record='recovery_case',version=RECOVERY_VERSION,**case,budget_order=order,
        lower_bound=lower_bound,bound_seconds=bound_seconds,
        bound_blocks_first=lower_bound>=case['occurrences'][0]['incumbent']-1e-7,
        bound_blocked_occurrences=sum(lower_bound>=o['incumbent']-1e-7 for o in case['occurrences']),
        trials=trials,actual_runtime=time.perf_counter()-started)


def run_stratified_recovery(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os,random
    rows=_study_rows();cases,population=_stratified_recovery_sample(rows)
    if not any(r.get('record')=='recovery_plan' and r.get('version')==RECOVERY_VERSION for r in rows):
        _study_append(dict(record='recovery_plan',version=RECOVERY_VERSION,source_version=SPARSE_VERSION,
            sampler_seed=20261101,max_per_stratum=3,population=population,budgets=RECOVERY_BUDGETS,workers=args.workers,
            selected=[dict(case_id=c['case_id'],instance=c['instance'],repairer=c['repairer'],mode_count=c['mode_count'],
                occurrences=c['occurrences']) for c in cases],
            design='direct equality dedup within workbook/repairer/mode-count; uniform without replacement; '
                'primary improvement relative to first occurrence in sorted instance/repairer/seed/policy/trace order; '
                'all occurrence improvements secondary; no online counterfactual claim; fresh recovery each budget; '
                'rotating order; all candidates evaluated regardless of route bound or frontier exhaustion; '
                'fixed-route bound=max(process+travel), no waiting, no fit; physical certificate, not formal B32 results'))
    done={r['case_id'] for r in rows if r.get('record')=='recovery_case' and r.get('version')==RECOVERY_VERSION}
    jobs=[c for c in cases if c['case_id'] not in done];random.Random(20261102).shuffle(jobs)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        futures=[pool.submit(_stratified_recovery_worker,c) for c in jobs]
        for future in as_completed(futures):
            row=future.result();_study_append(row)
            print('RECOVERY_CASE',row['case_id'],row['repairer'],row['mode_count'],
                [(t['budget'],t['status'],t['improves_first']) for t in row['trials']],flush=True)


def summarize_stratified_recovery():
    rows=_study_rows();cases,population=_stratified_recovery_sample(rows)
    runs=sorted((r for r in rows if r.get('record')=='recovery_case' and r.get('version')==RECOVERY_VERSION),key=lambda r:r['case_id'])
    if len(runs)!=len(cases):raise ValueError('complete all selected recovery candidates first')
    if any(r.get('record')=='recovery_analysis' and r.get('version')==RECOVERY_VERSION for r in rows):
        print('RECOVERY_ANALYSIS already saved',flush=True);return
    summary=[];checked=0
    for run,case in zip(runs,cases):
        if any(run[k]!=v for k,v in case.items()):raise ValueError('selected candidate differs')
        trials={t['budget']:t for t in run['trials']}
        if set(trials)!=set(RECOVERY_BUDGETS):raise ValueError('budget missing')
        for budget in (128,512):
            row=trials[budget];previous=trials[32 if budget==128 else 128]
            if previous['certified_physical'] and (not row['certified_physical'] or row['cmax']>previous['cmax']+1e-7):
                raise ValueError('larger recovery budget lost a recovered schedule')
            if run['frontier_exhausted'] and row['status']!='DEADLOCK':raise ValueError('exhausted B32 unexpectedly recovered')
            if row['improves_first'] and run['bound_blocks_first']:raise ValueError('bound would prune improvement')
        checked+=len(trials)
    for repairer in ('NEW','GNN'):
        group=[r for r in runs if r['repairer']==repairer]
        for budget in RECOVERY_BUDGETS:
            trial=[(r,next(t for t in r['trials'] if t['budget']==budget)) for r in group]
            summary.append(dict(repairer=repairer,budget=budget,cases=len(group),
                recovered=sum(t['certified_physical'] for r,t in trial),improves_first=sum(t['improves_first'] for r,t in trial),
                improving_workbooks=len({r['instance'] for r,t in trial if t['improves_first']}),
                improves_any_occurrence=sum(t['improving_occurrences']>0 for r,t in trial),
                total_seconds=sum(t['seconds'] for r,t in trial),
                exhausted_at_32=sum(r['frontier_exhausted'] for r in group),
                bound_blocks_first=sum(r['bound_blocks_first'] for r in group),
                eligible_extension=sum(not r['frontier_exhausted'] and not r['bound_blocks_first'] for r in group),
                selected_extension_seconds=sum(t['seconds'] for r,t in trial if not r['frontier_exhausted'] and not r['bound_blocks_first']),
                estimated_unique_recovery_rate=sum(t['certified_physical']*r['stratum_unique']/r['stratum_sampled'] for r,t in trial)/
                    sum(p['unique'] for p in population if p['repairer']==repairer),
                estimated_occurrence_improvement_rate=sum(t['improving_occurrences']*r['stratum_unique']/r['stratum_sampled'] for r,t in trial)/
                    sum(p['occurrences'] for p in population if p['repairer']==repairer)))
    record=dict(record='recovery_analysis',version=RECOVERY_VERSION,cases=len(runs),audited_trials=checked,
        diagnostic_calls=checked,certifications=sum(t['certified_physical'] for r in runs for t in r['trials']),
        worker_seconds=sum(r['actual_runtime'] for r in runs),summary=summary,
        note='stratified sample from consumed development trajectories, not independent instances or online gains; '
             'population estimates use N/n per stratum; first occurrence endpoint fixed before evaluation; '
             'selected extension timing is a retrospective cost projection, not an equal-wall-time solver experiment')
    _study_append(record);print(json.dumps(record,ensure_ascii=False),flush=True)


BOUND_VERSION='route-bound-v1'
BOUND_SEEDS=(2026110101,2026110102)
BOUND_POLICIES=('UNPRUNED','BOUND')


def audit_route_bound_shadow():
    rows=_study_rows()
    if any(r.get('record')=='bound_shadow' and r.get('version')==BOUND_VERSION for r in rows):
        print('BOUND_SHADOW already saved',flush=True);return
    stats={};started=time.perf_counter()
    for run in (r for r in rows if r.get('record')=='sparse_run' and r.get('version')==SPARSE_VERSION):
        key=run['repairer'];s=stats.setdefault(key,Counter())
        for trial in run['trace']:
            if not trial['reference_calls']:continue
            case=dict(parents=run['solution']['parents'],**{k:trial[k] for k in ('patterns','routes','directions')})
            solution=_recovery_case_solution(case);tick=time.perf_counter()
            bound=_fixed_route_time_bound(solution,trial['directions']);s['bound_seconds']+=time.perf_counter()-tick
            prune=bound>=trial['incumbent_before']+1e-7;s['evaluated']+=1
            if trial['certified'] and bound>trial['cmax']+1e-7:raise ValueError('invalid fixed-route bound')
            if prune:
                if trial['accepted']:raise ValueError('bound removes a recorded accepted improvement')
                s['pruned']+=1;s['pruned_'+trial['status']]+=1
                s['original_dispatch_cert_seconds']+=trial['seconds']-trial['construction_seconds']
    record=dict(record='bound_shadow',version=BOUND_VERSION,source_version=SPARSE_VERSION,
        summary=[dict(repairer=k,**v) for k,v in sorted(stats.items())],seconds=time.perf_counter()-started,
        note='all previously evaluated candidates checked; no accepted improvement pruned; '
             'recorded dispatch/certification cost is retrospective, not measured online speedup')
    _study_append(record);print(json.dumps(record),flush=True)


def _bound_worker(job):
    source,repairer,seed,policies=job;tick=time.perf_counter()
    current=_replay_study_source(source);model=shared.load_model(shared.pilot.MODEL)[0] if repairer=='GNN' else None
    prep=time.perf_counter()-tick;results=[]
    for index,policy in enumerate(policies):
        out=_online_bundle_search(current,repairer,'SINGLE',seed,20.,model,prune_route_bound=policy=='BOUND')
        results.append(dict(record='bound_run',version=BOUND_VERSION,instance=source['instance'],repairer=repairer,
            seed=seed,policy=policy,policy_order=policies,source_reference_calls=int(index==0),
            preparation_seconds=prep if index==0 else 0.,**out))
    return results


def run_bound_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os,random
    rows=_study_rows();sources={k[0]:v for k,v in _pair_study_sources(rows).items() if k[1]=='BASE12'}
    if not any(r.get('record')=='bound_shadow' and r.get('version')==BOUND_VERSION for r in rows):
        raise ValueError('run all-candidate bound-shadow validation before timed comparison')
    if not any(r.get('record')=='bound_plan' and r.get('version')==BOUND_VERSION for r in rows):
        _study_append(dict(record='bound_plan',version=BOUND_VERSION,instances=sorted(sources),seeds=BOUND_SEEDS,
            repairers=['NEW','GNN'],policies=BOUND_POLICIES,budget_seconds=20,workers=args.workers,
            design='12 workbooks x 2 seeds x 2 repairers x 2 policies=96; O+S1/S2/S3 normal search, no pairs; '
                'cyclic balanced policy order, shuffled groups, single-thread workers; same saved source per pair',
            bound='max per-robot process+travel for fixed routes/directions; prune only if bound>=current incumbent+1e-7; '
                'a skipped candidate has no invented Cmax or feasibility label; original B32 on remaining candidates',
            timing='all proposal/construction/bound/B32/certification within 20s; exclude late improvements; '
                'source replay and model preload outside local timer; no end-to-end or independent-test claim',
            statistics='20s primary paired endpoint; 10s secondary; average two seeds within workbook, '
                'bootstrap20000 workbook resamples seed20261101, descriptive unadjusted intervals'))
    done={(r['instance'],r['repairer'],r['seed'],r['policy']) for r in rows if r.get('record')=='bound_run' and r.get('version')==BOUND_VERSION}
    jobs=[]
    for i,(instance,source) in enumerate(sorted(sources.items())):
        for j,seed in enumerate(BOUND_SEEDS):
            for k,repairer in enumerate(('NEW','GNN')):
                shift=(i+j+k)%2;order=BOUND_POLICIES[shift:]+BOUND_POLICIES[:shift]
                policies=tuple(p for p in order if (instance,repairer,seed,p) not in done)
                if policies:jobs.append((source,repairer,seed,policies))
    random.Random(20261101).shuffle(jobs)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        for future in as_completed([pool.submit(_bound_worker,job) for job in jobs]):
            for row in future.result():
                _study_append(row)
                print('BOUND_RUN',row['instance'],row['repairer'],row['seed'],row['policy'],row['status'],
                    row['cmax'],'calls',row['reference_calls'],'pruned',sum(t['status']=='NONIMPROVING_ROUTE_BOUND' for t in row['trace']),flush=True)


def summarize_bound_study():
    from collections import defaultdict
    from statistics import mean
    import random
    rows=_study_rows();runs=[r for r in rows if r.get('record')=='bound_run' and r.get('version')==BOUND_VERSION]
    if len(runs)!=96:raise ValueError('finish all 96 bound runs')
    if any(r.get('record')=='bound_analysis' and r.get('version')==BOUND_VERSION for r in rows):
        print('BOUND_ANALYSIS already saved',flush=True);return
    summary=[];pairs=[];audited=0;late_improvements=0
    sources={k[0]:v for k,v in _pair_study_sources(rows).items() if k[1]=='BASE12'}
    for run in runs:
        source=sources[run['instance']];current=source['cmax'];events=[[0.,current,'SOURCE']]
        patterns=source['solution']['patterns'];routes=[r['block_ids'] for r in source['solution']['routes']];bits=source['directions']
        if run['status']!='CERTIFIED' or current!=run['initial_cmax']:raise ValueError('bad online source/status')
        for t in run['trace']:
            audited+=1
            if abs(current-t['incumbent_before'])>1e-7:raise ValueError('incumbent history differs')
            if t['status']=='NONIMPROVING_ROUTE_BOUND':
                case=dict(parents=source['solution']['parents'],**{k:t[k] for k in ('patterns','routes','directions')})
                bound=_fixed_route_time_bound(_recovery_case_solution(case),t['directions'])
                if (run['policy']!='BOUND' or abs(bound-t['route_lower_bound'])>1e-7 or bound<current+1e-7 or
                    t['reference_calls'] or t['certified'] or t['cmax'] is not None):raise ValueError('invalid skipped candidate')
            accept=bool(t['certified'] and t['elapsed']<=20. and t['cmax']<current-1e-9*max(1.,current))
            if t['accepted']!=accept or t['late']!=(t['elapsed']>20.):raise ValueError('acceptance/deadline differs')
            if t['late'] and t['certified'] and t['cmax']<current-1e-7:late_improvements+=1
            if accept:
                current=t['cmax'];events.append([t['elapsed'],current,t['proposal_id']])
                patterns=t['patterns'];routes=t['routes'];bits=t['directions']
        if current!=run['cmax'] or events!=run['best_events']:raise ValueError('saved objective/events differ')
        if (run['solution']['patterns']!=patterns or [r['block_ids'] for r in run['solution']['routes']]!=routes or run['directions']!=bits):
            raise ValueError('saved final solution differs')
    lookup={(r['instance'],r['repairer'],r['seed'],r['policy']):r for r in runs}
    prefix_candidates=0
    for instance in sorted(sources):
        for repairer in ('NEW','GNN'):
            for seed in BOUND_SEEDS:
                a=lookup[instance,repairer,seed,'BOUND'];b=lookup[instance,repairer,seed,'UNPRUNED']
                for ta,tb in zip(a['trace'],b['trace']):
                    if ta['late'] or tb['late']:break
                    for key in ('iteration','position','proposal_id','q','selection','seed','removed',
                                'round_source_cmax','incumbent_before','incumbent_after','accepted'):
                        if ta[key]!=tb[key]:raise ValueError('bound changed timely search prefix: '+key)
                    for key in ('patterns','routes','directions'):
                        if ta.get(key)!=tb.get(key):raise ValueError('bound changed candidate construction: '+key)
                    prefix_candidates+=1
    replays=[]
    for repairer in ('NEW','GNN'):
        for policy in BOUND_POLICIES:
            group=[r for r in runs if r['repairer']==repairer and r['policy']==policy]
            summary.append(dict(repairer=repairer,policy=policy,runs=len(group),
                mean_source_pct=mean(100*(r['cmax']/r['initial_cmax']-1) for r in group),
                calls=sum(r['reference_calls'] for r in group),certifications=sum(r['certifier_calls'] for r in group),
                candidates=sum(len(r['trace']) for r in group),accepted=sum(r['accepted'] for r in group),
                pruned=sum(t['status']=='NONIMPROVING_ROUTE_BOUND' for r in group for t in r['trace']),
                bound_seconds=sum(t.get('bound_seconds',0.) for r in group for t in r['trace']),
                runtime=sum(r['actual_runtime'] for r in group),max_overshoot=max(r['overshoot'] for r in group),
                positions=dict(Counter(r['policy_order'].index(policy) for r in group))))
            run=min(group,key=lambda r:(r['cmax']/r['initial_cmax'],r['instance'],r['seed']))
            replay=_replay_study_source(run)
            replays.append(dict(instance=run['instance'],repairer=repairer,policy=policy,seed=run['seed'],cmax=replay.schedule.cmax,certified=True))
        for checkpoint in (10.,20.):
            by_instance=defaultdict(list)
            for instance in sorted(sources):
                for seed in BOUND_SEEDS:
                    a=lookup[instance,repairer,seed,'BOUND'];b=lookup[instance,repairer,seed,'UNPRUNED']
                    av=min(e[1] for e in a['best_events'] if e[0]<=checkpoint);bv=min(e[1] for e in b['best_events'] if e[0]<=checkpoint)
                    by_instance[instance].append(100*(av/bv-1))
            values=[mean(v) for v in by_instance.values()];rng=random.Random(20261101)
            boot=sorted(mean(rng.choices(values,k=len(values))) for _ in range(20000))
            pairs.append(dict(repairer=repairer,checkpoint=checkpoint,mean_pct=mean(values),
                win=sum(v<-1e-8 for v in values),tie=sum(abs(v)<=1e-8 for v in values),loss=sum(v>1e-8 for v in values),
                descriptive_ci95=[boot[500],boot[19499]],workbook_pct=dict(zip(by_instance,values))))
    record=dict(record='bound_analysis',version=BOUND_VERSION,runs=len(runs),audited_candidates=audited,
        summary=summary,comparisons=pairs,late_better_excluded=late_improvements,replays=replays,
        identical_timely_prefix_candidates=prefix_candidates,
        reference_calls=len(replays),certifications=len(replays),source_replays=sum(r['source_reference_calls'] for r in runs))
    _study_append(record);print(json.dumps(record,ensure_ascii=False),flush=True)


FRESH_VERSION='fresh-e2e-bound-v1'


def _fresh_e2e_worker(job):
    from mrta_reference.model import ParentWeld
    from mrta_reference.certifier import certify_schedule
    entry,repairer,seed,policies=job
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in entry['parents'])
    rows=[]
    for policy in policies:
        started=time.perf_counter();deadline=started+30.;events=[];certs=0
        model=shared.load_model(shared.pilot.MODEL)[0] if repairer=='GNN' else None
        model_seconds=time.perf_counter()-started
        def accept(candidate):
            nonlocal certs
            check=certify_schedule(candidate.solution,candidate.schedule,CONFIG,scope=FORMAL_SCOPE_V2);certs+=1
            if not check.certified:raise ValueError(check.errors)
            elapsed=time.perf_counter()-started
            if elapsed>30:return False
            if not events or candidate.schedule.cmax<events[-1][1]:events.append([elapsed,candidate.schedule.cmax,'INITIAL'])
            return True
        current,attempts=shared.pilot.oriented.common_initial_beam(parents,CONFIG,seed=seed,
            deadline=deadline,accept_candidate=accept,interleave=True)
        init_seconds=time.perf_counter()-started
        calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
        row=dict(record='fresh_e2e_run',version=FRESH_VERSION,instance=entry['instance'],N=entry['N'],
            repairer=repairer,seed=seed,policy=policy,policy_order=policies,budget=30.,model_seconds=model_seconds,
            initialization_seconds=init_seconds,initial_attempts=attempts,initial_reference_calls=calls,initial_certifications=certs)
        if current is None:
            row.update(status='TIME_LIMIT_NO_FEASIBLE' if init_seconds>30. else 'INITIALIZATION_FAILED',cmax=None,initial_cmax=None,trace=[],best_events=events,
                reference_calls=calls,certifier_calls=certs,solution=None,directions=None)
        else:
            out=_online_bundle_search(current,repairer,'SINGLE',seed,30.,model,
                prune_route_bound=policy=='BOUND',absolute_deadline=deadline)
            offset=out.pop('search_started')-started
            for t in out['trace']:t['elapsed']+=offset
            combined=events+[[e[0]+offset,e[1],e[2]] for e in out['best_events'][1:]]
            row.update(out,budget=30.,best_events=combined,initialization_seconds=init_seconds,
                reference_calls=calls+out['reference_calls'],certifier_calls=certs+out['certifier_calls'])
        row.update(actual_runtime=time.perf_counter()-started)
        row['overshoot']=max(0.,row['actual_runtime']-30.)
        rows.append(row)
    return rows


def run_fresh_e2e(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os,random
    rows=_study_rows();entries=[r for r in rows if r.get('record')=='fresh_instance' and r.get('version')==FRESH_VERSION]
    if not entries:
        entries=_independent_mode_instances(seed_start=2026111001,count=6)
        def geometry(ps):return sorted(tuple(sorted((tuple(p['start']),tuple(p['end'])))) for p in ps)
        prior=[geometry(r['parents']) for r in rows if r.get('record') in ('instance','fresh_instance') and 'parents' in r]
        for entry in entries:
            g=geometry(entry['parents'])
            if g in prior:raise ValueError('new packing duplicates saved geometry')
            prior.append(g)
            entry.update(record='fresh_instance',version=FRESH_VERSION,instance='FRESH_'+str(entry['geometry_seed']))
            _study_append(entry)
    if len(entries)!=6:raise ValueError('requires six saved fresh geometries')
    seeds=(2026110201,2026110202)
    if not any(r.get('record')=='fresh_plan' and r.get('version')==FRESH_VERSION for r in rows):
        _study_append(dict(record='fresh_plan',version=FRESH_VERSION,instances=[e['instance'] for e in entries],
            seeds=seeds,repairers=['NEW','GNN'],policies=BOUND_POLICIES,budget=30,workers=args.workers,
            design='6 new packing seeds, two per target35/55/85; same component source, not new part families; '
                '48 runs, geometry loading and Python imports outside timer; model weights loaded inside each run; '
                'Beam initialization, certification and O+S1/S2/S3 search inside one absolute deadline; '
                'balanced order; report no-initial failures separately; no model tuning'))
    done={(r['instance'],r['repairer'],r['seed'],r['policy']) for r in rows if r.get('record')=='fresh_e2e_run' and r.get('version')==FRESH_VERSION}
    jobs=[]
    for i,entry in enumerate(sorted(entries,key=lambda e:e['instance'])):
        for j,seed in enumerate(seeds):
            for k,repairer in enumerate(('NEW','GNN')):
                shift=(i+j+k)%2;order=BOUND_POLICIES[shift:]+BOUND_POLICIES[:shift]
                todo=tuple(p for p in order if (entry['instance'],repairer,seed,p) not in done)
                if todo:jobs.append((entry,repairer,seed,todo))
    random.Random(20261103).shuffle(jobs)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        for future in as_completed([pool.submit(_fresh_e2e_worker,j) for j in jobs]):
            for row in future.result():
                _study_append(row)
                print('FRESH',row['instance'],row['repairer'],row['seed'],row['policy'],row['status'],row['cmax'],
                    'init_s',round(row['initialization_seconds'],3),'calls',row['reference_calls'],flush=True)


def summarize_fresh_e2e():
    from statistics import mean
    import random
    rows=_study_rows();runs=[r for r in rows if r.get('record')=='fresh_e2e_run' and r.get('version')==FRESH_VERSION]
    if len(runs)!=48:raise ValueError('finish 48 fresh runs')
    if any(r.get('record')=='fresh_analysis' and r.get('version')==FRESH_VERSION for r in rows):return
    summaries=[];comparisons=[];replays=[];audited=0
    for r in runs:
        initial=[e for e in r['best_events'] if e[2]=='INITIAL']
        if r['cmax'] is None:
            if initial or r['trace']:raise ValueError('failed initial has accepted trajectory')
            continue
        current=initial[-1][1]
        if current!=r['initial_cmax']:raise ValueError('initial value differs')
        for t in r['trace']:
            audited+=1
            if abs(t['incumbent_before']-current)>1e-7:raise ValueError('history differs')
            if abs(t['elapsed']-30.)>1e-8 and t['late']!=(t['elapsed']>30.):raise ValueError('deadline shifted')
            accept=bool(t['certified'] and t['elapsed']<=30. and t['cmax']<current-1e-9*max(1.,current))
            if accept!=t['accepted']:raise ValueError('invalid end-to-end acceptance')
            if accept:current=t['cmax']
        if current!=r['cmax'] or any(e[0]>30. for e in r['best_events']):raise ValueError('late final counted')
    for repairer in ('NEW','GNN'):
        for policy in BOUND_POLICIES:
            group=[r for r in runs if r['repairer']==repairer and r['policy']==policy];good=[r for r in group if r['cmax'] is not None]
            summaries.append(dict(repairer=repairer,policy=policy,runs=len(group),successful=len(good),
                initial_seconds=mean(r['initialization_seconds'] for r in group),model_seconds=mean(r['model_seconds'] for r in group),
                calls=sum(r['reference_calls'] for r in group),certifications=sum(r['certifier_calls'] for r in group),
                pruned=sum(t['status']=='NONIMPROVING_ROUTE_BOUND' for r in group for t in r['trace']),
                mean_initial_gain_pct=mean(100*(r['cmax']/r['initial_cmax']-1) for r in good) if good else None))
            if good:
                r=min(good,key=lambda r:r['cmax']/r['initial_cmax']);out=_replay_study_source(r)
                replays.append(dict(instance=r['instance'],repairer=repairer,policy=policy,seed=r['seed'],cmax=out.schedule.cmax,certified=True))
        lookup={(r['instance'],r['seed'],r['policy']):r for r in runs if r['repairer']==repairer}
        by_instance={};initial_equal=0;failed_pairs=0
        for instance in sorted({r['instance'] for r in runs}):
            values=[]
            for seed in (2026110201,2026110202):
                a=lookup[instance,seed,'BOUND'];b=lookup[instance,seed,'UNPRUNED']
                if a['cmax'] is None or b['cmax'] is None:failed_pairs+=1;continue
                initial_equal+=int(a['initial_cmax']==b['initial_cmax'])
                values.append(100*(a['cmax']/b['cmax']-1))
            if values:by_instance[instance]=mean(values)
        values=list(by_instance.values());rng=random.Random(20261104)
        boot=sorted(mean(rng.choices(values,k=len(values))) for _ in range(20000)) if values else []
        comparisons.append(dict(repairer=repairer,paired_layouts=len(values),failed_pairs=failed_pairs,equal_initial_pairs=initial_equal,
            mean_pct=mean(values) if values else None,win=sum(v<-1e-8 for v in values),tie=sum(abs(v)<=1e-8 for v in values),
            loss=sum(v>1e-8 for v in values),descriptive_ci95=[boot[500],boot[19499]] if boot else None,layout_pct=by_instance))
    record=dict(record='fresh_analysis',version=FRESH_VERSION,summary=summaries,comparisons=comparisons,
        audited_candidates=audited,replays=replays,reference_calls=len(replays),certifications=len(replays),
        note='new packing layouts from same component catalogue; model loading plus initialization included; six layouts only, '
             'descriptive unadjusted bootstrap, not unseen part families or cold Python startup performance')
    _study_append(record);print(json.dumps(record,ensure_ascii=False),flush=True)


def choose_mode_rank(candidates, rule):
    """Only pre-dispatch features enter a decision; outcome labels are absent."""
    if rule=='LOAD':
        return 1 if any(c['mode_rank']==1 for c in candidates) else None
    usable=[c for c in candidates if c['features'] is not None]
    if not usable:return None
    key='nominal_cmax' if rule=='PM' else 'exposure_score'
    if rule not in ('PM','EXPOSURE'):raise ValueError(rule)
    return min(usable,key=lambda c:(round(c['features'][key],8),
                                    c['features']['total_nominal_time'],c['mode_rank']))['mode_rank']


def _ranking_existing_worker(job):
    """Re-score saved complete routes; no B32 calls and no fitted parameters."""
    source,run=job
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_search.lns import nominal_coordination_features
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in source['solution']['parents'])
    trials=[]
    started=time.perf_counter()
    for old in run['trials']:
        trial=dict(old)
        if old.get('patterns') is not None:
            patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
                None if p['rail'] is None else Rail(p['rail'])) for p in old['patterns'])
            solution=canonicalize(parents,patterns,dict(enumerate(old['routes'])),CONFIG,scope=FORMAL_SCOPE_V2)
            tick=time.perf_counter()
            trial['ranking_features']=nominal_coordination_features(solution,old['directions'],CONFIG)
            trial['feature_seconds']=time.perf_counter()-tick
        trials.append(trial)
    return dict(run,record='ranking_run',version='mode-ranking-v1',stage='REUSED_DIAGNOSTIC',
                reused_version=run['version'],trials=trials,new_reference_calls=0,
                feature_runtime=time.perf_counter()-started)


def _ranking_validation_worker(entry):
    from mrta_reference.model import ParentWeld
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in entry['parents'])
    result=_mode_search(parents,'FIXED_NEW',20261031,30.,1000000)
    source=dict(result,record='ranking_source',version='mode-ranking-v1',stage='NEW_LAYOUT_VALIDATION',
                instance=entry['instance'],N=len(parents),seed=20261031,budget=30.)
    if source['cmax'] is None:
        return [source]
    run=_matched_worker(source,version='mode-ranking-v1')
    run.update(record='ranking_run',stage='NEW_LAYOUT_VALIDATION',
               new_reference_calls=run['source_reference_calls']+sum(t['reference_calls'] for t in run['trials']))
    return [source,run]


def _ranking_validation_entries(rows):
    """Nine previously unrecorded layouts, one per validation workbook."""
    from dataclasses import asdict
    from mrta_data.phase3_split import assert_v2_solver_access_allowed
    from mrta_data.ppo_instances import load_ppo_platform_instance,to_parent_welds
    dataset=shared.pilot.dataset
    roles=json.loads(dataset.ROLES.read_text(encoding='utf-8'))
    manifest=json.loads(dataset.DATASET.read_text(encoding='utf-8'))
    paths={w['relative_path'] for w in roles['workbooks'] if w['new_v2_role']=='V2_VALIDATION'}
    assert_v2_solver_access_allowed(roles,paths,allowed_roles=('V2_VALIDATION',))
    used={r['instance'] for r in rows if 'instance' in r}
    # Reuse existing manifest duplicate identities; do not introduce new hashes.
    used_geometry={e['instance_geometry_hash'] for e in manifest['instances'] if e['instance_id'] in used}
    available=[e for e in manifest['instances'] if e['relative_path'] in paths and e['instance_id'] not in used
        and e['instance_geometry_hash'] not in used_geometry and e['validation_status']=='VALID'
        and e['duplicate_of'] is None and e['formal_compatibility']=='PASS']
    entries=[]
    for target in (85,55,35):
        for index in range(3):
            def order(e):
                density=(e['n_mandatory_y_split']+e['n_whole_both_rails'])/e['actual_weld_count']
                return abs(e['actual_weld_count']-target),(-density if index%2==0 else density),e['relative_path'],e['sheet_name']
            e=min(available,key=order)
            available=[x for x in available if x['relative_path']!=e['relative_path']]
            instance=load_ppo_platform_instance(dataset.PPO/e['relative_path'],e['sheet_name'],
                ppo_root=dataset.PPO,instance_id=e['instance_id'])
            parents=to_parent_welds(instance)
            entries.append(dict(record='ranking_instance',version='mode-ranking-v1',stage='NEW_LAYOUT_VALIDATION',
                instance=e['instance_id'],relative_path=e['relative_path'],sheet_name=e['sheet_name'],
                N=len(parents),target_N=target,role='V2_VALIDATION',mandatory_y=e['n_mandatory_y_split'],
                whole_both_rails=e['n_whole_both_rails'],parents=[asdict(p) for p in parents]))
    return entries


def run_ranking_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os
    rows=_study_rows()
    if not any(r.get('record')=='ranking_plan' and r.get('version')=='mode-ranking-v1' for r in rows):
        _study_append(dict(record='ranking_plan',version='mode-ranking-v1',
            policies=['LOAD','PM','EXPOSURE'],proposals='first three original load-ranked alternatives',
            common_candidate='original mode repaired and evaluated once per context',
            selectors=['RANDOM','TRAVEL','LOAD'],q=[4,8],seed='20261031 + selector index + q',
            exposure='union per robot of continuous XY proximity and same-rail X order violation on independent time-zero routes; '
                     'score=max(nominal_completion+union_seconds), tie total_nominal_time then original mode rank',
            protocol='decisions use only route features, never B32 outcomes; evaluate complete pool to audit decisions; '
                     'each policy gets original-mode plus one selected alternative evaluation, but PM/EXPOSURE build up to three alternatives',
            validation='nine unrecorded V2_VALIDATION layouts, one per workbook, 3 per N85/55/35; old workbook families reused; '
                       '30s FIXED_NEW incumbents, one source seed; no sealed data or model updates',
            timing='candidate construction, feature extraction and B32 recorded separately; no equal wall-time claim'))
    # Select and save fresh layout IDs before either set of ranking outcomes is inspected.
    entries=[r for r in rows if r.get('record')=='ranking_instance' and r.get('version')=='mode-ranking-v1']
    if not entries:
        entries=_ranking_validation_entries(rows)
        for entry in entries:_study_append(entry)
    stage='REUSED_DIAGNOSTIC' if args.mode_study=='rank-diagnostic' else 'NEW_LAYOUT_VALIDATION'
    done={r['instance'] for r in rows if r.get('record')=='ranking_run' and r.get('version')=='mode-ranking-v1' and r.get('stage')==stage}
    done.update(r['instance'] for r in rows if r.get('record')=='ranking_source'
                and r.get('version')=='mode-ranking-v1' and r.get('stage')==stage and r['cmax'] is None)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        if stage=='REUSED_DIAGNOSTIC':
            sources={r['instance']:r for r in rows if r.get('record')=='run' and r.get('version')==STUDY_VERSION
                     and r.get('stage')=='PPO_VALIDATION' and r['method']=='FIXED_NEW' and r['seed']==20261031}
            jobs=[pool.submit(_ranking_existing_worker,(sources[r['instance']],r)) for r in rows
                  if r.get('record')=='matched_run' and r.get('version')=='matched-state-v1' and r['instance'] not in done]
        else:
            jobs=[pool.submit(_ranking_validation_worker,e) for e in entries if e['instance'] not in done]
        for future in as_completed(jobs):
            result=future.result()
            for row in result if isinstance(result,list) else [result]:
                _study_append(row)
                print('RANK',row['stage'],row['instance'],row['record'],len(row.get('trials',[])),
                      Counter(t['status'] for t in row.get('trials',[])),flush=True)


def summarize_ranking_study(rows):
    from collections import defaultdict,Counter
    from statistics import mean
    import random
    summaries=[];decisions=[];execution=[]
    for stage in ('REUSED_DIAGNOSTIC','NEW_LAYOUT_VALIDATION'):
        runs=[r for r in rows if r.get('record')=='ranking_run' and r.get('version')=='mode-ranking-v1' and r['stage']==stage]
        if not runs:continue
        sources=[r for r in rows if r.get('record')=='ranking_source' and r.get('version')=='mode-ranking-v1' and r['stage']==stage]
        execution.append(dict(stage=stage,runs=len(runs),source_runs=len(sources),
            source_statuses=dict(Counter(r['status'] for r in sources)),
            source_reference_calls=sum(r['reference_calls'] for r in sources),
            source_certifications=sum(r['certifier_calls'] for r in sources),
            source_actual_seconds=sum(r['actual_runtime'] for r in sources),
            replay_calls=0 if stage=='REUSED_DIAGNOSTIC' else sum(r['source_reference_calls'] for r in runs),
            candidate_calls=sum(t['reference_calls'] for r in runs for t in r['trials']),
            candidate_calls_reused=stage=='REUSED_DIAGNOSTIC',
            candidate_statuses=dict(Counter(t['status'] for r in runs for t in r['trials'])),
            candidate_certifications=sum(t['certified'] for r in runs for t in r['trials']),
            feature_seconds=sum(t.get('feature_seconds',0.) for r in runs for t in r['trials']),
            construction_seconds=sum(t['construction_seconds'] for r in runs for t in r['trials']),
            candidate_total_seconds=sum(t['seconds'] for r in runs for t in r['trials'])))
        geometry=defaultdict(lambda:defaultdict(list));counts=defaultdict(Counter)
        for run in runs:
            groups=defaultdict(list)
            for t in run['trials']:groups[t['repairer'],t['q'],t['seed']].append(t)
            for (repairer,q,seed),trials in groups.items():
                original=next(t for t in trials if t['mode_rank']==0)
                incumbent=original['incumbent_cmax']
                base=min(incumbent,original['cmax'] if original['certified'] else incumbent)
                alternatives=[t for t in trials if t['mode_rank']>0]
                # Deliberately pass no outcomes to the selectors.
                features=[dict(mode_rank=t['mode_rank'],features=t.get('ranking_features')) for t in alternatives]
                available={t['mode_rank']:t for t in alternatives}
                oracle=min([base]+[t['cmax'] for t in alternatives if t['certified']])
                outcomes={}
                for rule in ('LOAD','PM','EXPOSURE'):
                    rank=choose_mode_rank(features,rule)
                    picked=available.get(rank)
                    best=min(base,picked['cmax'] if picked and picked['certified'] else base)
                    outcomes[rule]=best
                    k=(repairer,rule)
                    counts[k].update(contexts=1,selected_feasible=int(bool(picked and picked['certified'])),
                        selected_calls=0 if picked is None else picked['reference_calls'],
                        different_from_load=int(rank!=1),improved_base=int(best<base-1e-7),
                        missed_opportunity=int(best>oracle+1e-7))
                    geometry[k][run['instance']].append((100*(best/incumbent-1),100*(best/base-1),best-oracle))
                    decisions.append(dict(instance=run['instance'],stage=stage,repairer=repairer,q=q,seed=seed,policy=rule,
                        selected_rank=rank,status=None if picked is None else picked['status'],base=base,cmax=best,oracle=oracle))
                for rule in ('PM','EXPOSURE'):
                    geometry[repairer,rule+'/LOAD'][run['instance']].append(100*(outcomes[rule]/outcomes['LOAD']-1))
        for (repairer,rule),by_instance in geometry.items():
            if rule.endswith('/LOAD'):
                values=[mean(v) for _,v in sorted(by_instance.items())]
                rng=random.Random(20261031);boot=sorted(mean(rng.choices(values,k=len(values))) for _ in range(20000))
                summaries.append(dict(stage=stage,repairer=repairer,policy=rule,geometries=len(values),
                    mean_pct=mean(values),win=sum(x<-1e-8 for x in values),tie=sum(abs(x)<=1e-8 for x in values),
                    loss=sum(x>1e-8 for x in values),descriptive_ci95=[boot[500],boot[19499]]))
            else:
                summaries.append(dict(stage=stage,repairer=repairer,policy=rule,geometries=len(by_instance),
                    **counts[repairer,rule],vs_incumbent_pct=mean(mean(x[0] for x in v) for v in by_instance.values()),
                    mode_gain_pct=mean(mean(x[1] for x in v) for v in by_instance.values()),
                    mean_oracle_gap_seconds=mean(mean(x[2] for x in v) for v in by_instance.values())))
    return dict(record='ranking_analysis',version='mode-ranking-v1',execution=execution,summary=summaries,decisions=decisions,
                interpretation='geometry-level averages; bootstrap20000 seed20261031 unadjusted; oracle is best in examined top3 only')


def ranking_layout_overlap(rows):
    """Exact endpoint-set overlap, ignoring parent IDs and multiplicity."""
    old={r['instance'].split('::')[0]:r for r in rows if r.get('record')=='run'
         and r.get('version')==STUDY_VERSION and r.get('stage')=='PPO_VALIDATION'
         and r.get('method')=='FIXED_NEW' and r.get('seed')==20261031}
    def geometry(row):
        return {tuple(sorted((tuple(p['start']),tuple(p['end'])))) for p in row['solution']['parents']}
    result=[]
    for row in rows:
        if row.get('record')!='ranking_source' or row.get('version')!='mode-ranking-v1' or row['cmax'] is None:continue
        prior=old[row['instance'].split('::')[0]];a,b=geometry(row),geometry(prior)
        result.append(dict(instance=row['instance'],prior_instance=prior['instance'],N=row['N'],
            unique_segments=len(a),prior_unique_segments=len(b),intersection=len(a&b),union=len(a|b),
            jaccard=len(a&b)/len(a|b)))
    return dict(record='ranking_layout_overlap',version='mode-ranking-v1',layouts=result,
                interpretation='different sheets from reused workbooks are not independent geometry families; '
                               'endpoint sets ignore multiplicity and parent IDs and do not change solver input')


def audit_ranking_study():
    """Save numerical load bounds, cost accounting and selected-candidate replays."""
    from collections import defaultdict
    from statistics import mean
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.certifier import certify_schedule
    from mrta_search.lns import nominal_coordination_features,oriented_route_state
    from mrta_exact.lower_bounds import eligibility_load_lower_bound
    rows=_study_rows()
    if any(r.get('record')=='ranking_audit' and r.get('version')=='mode-ranking-v1' for r in rows):
        print('RANK_AUDIT already saved',flush=True);return
    analysis=summarize_ranking_study(rows)
    sources={r['instance']:r for r in rows if r.get('record')=='ranking_source' and r.get('version')=='mode-ranking-v1'}
    runs={r['instance']:r for r in rows if r.get('record')=='ranking_run'
          and r.get('version')=='mode-ranking-v1' and r['stage']=='NEW_LAYOUT_VALIDATION'}

    def restore(source,trial=None):
        data=source['solution'];parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in data['parents'])
        pp=data['patterns'] if trial is None else trial['patterns']
        patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
            None if p['rail'] is None else Rail(p['rail'])) for p in pp)
        routes=[r['block_ids'] for r in data['routes']] if trial is None else trial['routes']
        return canonicalize(parents,patterns,dict(enumerate(routes)),CONFIG,scope=FORMAL_SCOPE_V2)

    bounds=[]
    for instance,source in sorted(sources.items()):
        if source['cmax'] is None:continue
        solution=restore(source);tick=time.perf_counter()
        bound=eligibility_load_lower_bound(solution.parents,CONFIG)
        features=nominal_coordination_features(solution,source['directions'],CONFIG)
        state=oriented_route_state(tuple(r.block_ids for r in solution.routes),source['directions'],
            block_map(solution,CONFIG,scope=FORMAL_SCOPE_V2),CONFIG)
        processing=max(state.loads);nominal=features['nominal_cmax'];cmax=source['cmax']
        if not bound['lb_process']<=processing+1e-6<=nominal+2e-6<=cmax+3e-6:
            raise ValueError('load/time decomposition inconsistent')
        bounds.append(dict(instance=instance,**bound,cmax=cmax,max_process_load=processing,nominal_cmax=nominal,
            percent_above_bound=100*(cmax/bound['lb_process']-1),process_difference=processing-bound['lb_process'],
            travel_difference=nominal-processing,coordination_difference=cmax-nominal,seconds=time.perf_counter()-tick))

    cost=defaultdict(list)
    for run in runs.values():
        groups=defaultdict(list)
        for t in run['trials']:groups[t['repairer'],t['q'],t['seed']].append(t)
        for (repairer,q,seed),trials in groups.items():
            original=next(t for t in trials if t['mode_rank']==0)
            alternatives=[t for t in trials if t['mode_rank']>0]
            for rule in ('LOAD','PM','EXPOSURE'):
                rank=choose_mode_rank([dict(mode_rank=t['mode_rank'],features=t.get('ranking_features')) for t in alternatives],rule)
                selected=next((t for t in alternatives if t['mode_rank']==rank),None)
                built=([selected] if selected is not None else []) if rule=='LOAD' else alternatives
                construction=original['construction_seconds']+sum(t['construction_seconds'] for t in built)
                feature=0 if rule=='LOAD' else sum(t.get('feature_seconds',0) for t in alternatives)
                evaluated=[original]+([selected] if selected is not None else [])
                dispatch=sum(t['seconds']-t['construction_seconds']-t.get('feature_seconds',0) for t in evaluated)
                cost[repairer,rule].append((construction,feature,dispatch))
    costs=[dict(repairer=repairer,policy=rule,contexts=len(values),
                mean_construction_seconds=mean(v[0] for v in values),mean_feature_seconds=mean(v[1] for v in values),
                mean_dispatch_certification_seconds=mean(v[2] for v in values),
                mean_accounted_seconds=mean(sum(v) for v in values)) for (repairer,rule),values in sorted(cost.items())]
    decisions=[d for d in analysis['decisions'] if d['stage']=='NEW_LAYOUT_VALIDATION']
    lookup={(d['instance'],d['repairer'],d['q'],d['seed'],d['policy']):d for d in decisions}
    replays=[]
    for method in ('NEW','GNN'):
        candidates=[d for d in decisions if d['repairer']==method and d['policy']=='EXPOSURE' and d['cmax']<d['base']-1e-7]
        if not candidates:continue
        # GNN prioritizes a benefit over LOAD; NEW uses its largest mode gain.
        def order(d):
            control=lookup[d['instance'],method,d['q'],d['seed'],'LOAD']['cmax'] if method=='GNN' else d['base']
            return d['cmax']/control-1,d['instance'],d['q'],d['seed']
        decision=min(candidates,key=order);source=sources[decision['instance']]
        trial=next(t for t in runs[decision['instance']]['trials'] if t['repairer']==method and t['q']==decision['q']
                   and t['seed']==decision['seed'] and t['mode_rank']==decision['selected_rank'])
        solution=restore(source,trial)
        result=shared.pilot.oriented.schedule_explicit(solution,trial['directions'],CONFIG)
        check=certify_schedule(solution,result.schedule,CONFIG,scope=FORMAL_SCOPE_V2)
        if not check.certified or abs(result.schedule.cmax-trial['cmax'])>1e-7:
            raise ValueError('ranking selected candidate replay failed')
        replays.append(dict(decision,certified=True,replayed_cmax=result.schedule.cmax,
            load_result=lookup[decision['instance'],method,decision['q'],decision['seed'],'LOAD'],
            mode=trial['mode'],features=trial['ranking_features']))
    if not any(r.get('record')=='ranking_analysis' and r.get('version')=='mode-ranking-v1' for r in rows):
        _study_append(analysis)
    record=dict(record='ranking_audit',version='mode-ranking-v1',bounds=bounds,costs=costs,replays=replays,
        reference_calls=len(replays),note='cost accounting from the audit pool, not an online equal-wall-time benchmark; '
        'LP bound drops travel/conflicts and relaxes parent choices, so its gap is not fully attainable improvement')
    _study_append(record)
    if not any(r.get('record')=='ranking_layout_overlap' and r.get('version')=='mode-ranking-v1' for r in rows):
        _study_append(ranking_layout_overlap(rows))
    print(json.dumps(record,ensure_ascii=False),flush=True)


def run_matched_study(args):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    import multiprocessing,os
    rows=_study_rows()
    neighborhood=args.mode_study=='neighborhood'
    version='parent-neighborhood-v1' if neighborhood else 'matched-state-v1'
    families=('WAIT','Y_PAIR') if neighborhood else ('LEGACY',)
    sources=[r for r in rows if r.get('record')=='run' and r.get('version')==STUDY_VERSION
             and r.get('stage')=='PPO_VALIDATION' and r['method']=='FIXED_NEW' and r['seed']==20261031]
    if len(sources)!=12 or any(r['cmax'] is None for r in sources):
        raise ValueError('requires the twelve existing certified FIXED_NEW incumbents')
    done={(r['instance'],r.get('family','LEGACY')) for r in rows
          if r.get('record')=='matched_run' and r.get('version')==version}
    if not any(r.get('record')=='matched_plan' and r.get('version')==version for r in rows):
        _study_append(dict(record='matched_plan',version=version,source_version=STUDY_VERSION,
            instances=[r['instance'] for r in sources],source='FIXED_NEW final state, seed 20261031; consumed validation reused as development',
            families=families,selectors=['WAIT_ANCHORED','SHARED_Y_PAIR'] if neighborhood else ['RANDOM','TRAVEL','LOAD'],
            seeds='20261031 + context_index (0,1,2) + q',q=[4,8],fallback='LOAD with the same seed, explicitly recorded',
            controls='reuse matched-state-v1 LEGACY; fixed-mode candidates give 72 calls per repairer per family',
            modes='original plus first three load-ranked legal alternatives; actual counts may differ by selected parents',
            shared_y_policy='SETUP/WELD/POST clipped to Y band; cross-rail distinct-parent swept boxes within dx/dy; '
                            'max-phase score: (0.25+criticality)^2*(1+wait/process)/((1+dx/Dx+dy/Dy)*(1+time_gap/phase_scale)); '
                            'sample pair, keep both parents, fill with unchanged spatial/route-neighbour sampler',
            repairers=['NEW','GNN'],acceptance='no state updates; paired diagnostic, not wall-time performance comparison',
            workers=args.workers,model='existing frozen GNN V2',cut_policy='outer boundaries, center, legal midpoint'))
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=_mode_worker_init) as pool:
        jobs=[pool.submit(_matched_worker,r,family,version) for r in sources for family in families
              if (r['instance'],family) not in done]
        for future in as_completed(jobs):
            row=future.result();_study_append(row)
            print('MATCHED',row['family'],row['instance'],len(row['trials']),Counter(t['status'] for t in row['trials']),flush=True)


def summarize_neighborhood_study(rows):
    """Geometry-level descriptive comparisons; failures retain the incumbent."""
    from collections import defaultdict
    from statistics import mean
    runs=[r for r in rows if r.get('record')=='matched_run'
          and r.get('version') in ('matched-state-v1','parent-neighborhood-v1')]
    contexts={}
    for run in runs:
        family=run.get('family','LEGACY')
        for trial in run['trials']:
            key=(family,trial['repairer'],run['instance'],trial['q'],trial['seed'])
            contexts.setdefault(key,[]).append(trial)
    stats={}; scores={}
    for family in ('LEGACY','WAIT','Y_PAIR'):
        for repairer in ('NEW','GNN'):
            selected={k:v for k,v in contexts.items() if k[:2]==(family,repairer)}
            if not selected:continue
            totals=Counter();gains=defaultdict(list);fixed_gains=defaultdict(list);mode_gains=defaultdict(list)
            top1_gains=defaultdict(list);active_gains=defaultdict(list);active_mode_gains=defaultdict(list)
            unique=defaultdict(set)
            for key,trials in selected.items():
                originals=[t for t in trials if t['mode_rank']==0]
                if len(originals)!=1:raise ValueError('missing or repeated original-mode context')
                original=originals[0];incumbent=original['incumbent_cmax']
                fixed=min(incumbent,original['cmax'] if original['certified'] else incumbent)
                top1=min([fixed]+[t['cmax'] for t in trials if t['certified'] and t['mode_rank']==1])
                best=min([fixed]+[t['cmax'] for t in trials if t['certified']])
                totals.update(contexts=1,candidates=len(trials),feasible=sum(t['certified'] for t in trials),
                              reference_calls=sum(t['reference_calls'] for t in trials),
                              errors=sum(t['status']=='EXECUTION_ERROR' for t in trials),
                              original_feasible=int(original['certified']),
                              top1_reference_calls=sum(t['reference_calls'] for t in trials if t['mode_rank']<=1),
                              top1_improved=int(top1<incumbent-1e-7),
                              original_improved=int(fixed<incumbent-1e-7),best_improved=int(best<incumbent-1e-7),
                              mode_improved=int(best<fixed-1e-7),fallbacks=int(bool(original.get('selection_fallback'))))
                unique[key[2]].add((key[3],tuple(original['removed'])))
                fixed_gains[key[2]].append(100*(fixed/incumbent-1))
                gains[key[2]].append(100*(best/incumbent-1))
                top1_gains[key[2]].append(100*(top1/incumbent-1))
                mode_gains[key[2]].append(100*(best/fixed-1))
                if family!='LEGACY' and not original.get('selection_fallback'):
                    active_gains[key[2]].append(100*(best/incumbent-1))
                    active_mode_gains[key[2]].append(100*(best/fixed-1))
                    totals.update(active_contexts=1,active_fixed_improved=int(fixed<incumbent-1e-7),
                                  active_best_improved=int(best<incumbent-1e-7),active_mode_improved=int(best<fixed-1e-7))
                scores[(family,repairer)+key[2:]]=(fixed,top1,best)
            stats[family+'_'+repairer]=dict(totals,unique_partials=sum(map(len,unique.values())),
                fixed_pct=mean(map(mean,fixed_gains.values())),best_top3_pct=mean(map(mean,gains.values())),
                top1_pct=mean(map(mean,top1_gains.values())),active_geometries=len(active_gains),
                active_best_pct=mean(map(mean,active_gains.values())) if active_gains else None,
                active_mode_pct=mean(map(mean,active_mode_gains.values())) if active_mode_gains else None,
                incremental_mode_pct=mean(map(mean,mode_gains.values())),
                geometry_fixed_pct={k:mean(v) for k,v in sorted(fixed_gains.items())},
                geometry_best_pct={k:mean(v) for k,v in sorted(gains.items())})
    pairs=[]
    for a,b in (('WAIT','LEGACY'),('Y_PAIR','LEGACY'),('Y_PAIR','WAIT')):
        for repairer in ('NEW','GNN'):
            for index,endpoint in enumerate(('fixed','top1','best_top3')):
                geometry=defaultdict(list)
                for key,value in scores.items():
                    other=(b,repairer)+key[2:]
                    if key[:2]==(a,repairer) and other in scores:
                        geometry[key[2]].append(100*(value[index]/scores[other][index]-1))
                values=[mean(v) for _,v in sorted(geometry.items())]
                if values:
                    import random
                    rng=random.Random(20261030)
                    boot=sorted(mean(rng.choices(values,k=len(values))) for _ in range(20000))
                    pairs.append(dict(method=a,control=b,repairer=repairer,endpoint=endpoint,
                        geometries=len(values),contexts=sum(map(len,geometry.values())),mean_pct=mean(values),
                        descriptive_ci95=[boot[500],boot[19499]],
                        win=sum(v<-1e-8 for v in values),tie=sum(abs(v)<=1e-8 for v in values),
                        loss=sum(v>1e-8 for v in values),geometry_pct=dict(zip(sorted(geometry),values))))
    new_runs=[r for r in runs if r['version']=='parent-neighborhood-v1']
    return dict(record='neighborhood_analysis',version='parent-neighborhood-v1',run_count=len(runs),
                new_run_count=len(new_runs),new_source_replays=len(new_runs),
                new_candidate_calls=sum(t['reference_calls'] for r in new_runs for t in r['trials']),
                new_candidate_certifications=sum(t['certified'] for r in new_runs for t in r['trials']),
                cumulative_worker_seconds=sum(r['actual_runtime'] for r in new_runs),
                formula='per context percent change, average within geometry then across geometries; '
                        'retain incumbent on failure/worsening; bootstrap 20000 geometry resamples, seed20261030, no multiplicity adjustment',
                summary=stats,pairs=pairs)


if __name__=='__main__':main()
