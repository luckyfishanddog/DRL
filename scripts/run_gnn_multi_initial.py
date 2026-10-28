"""Frozen GNN initial construction: one greedy and three seeded sample paths."""
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
    args=parser.parse_args()
    torch.set_num_threads(1)
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


if __name__=='__main__':main()
