"""Shared parent-atomic GNN construction and repair, with real B32 evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
import run_gnn_joint_repair_pilot as pilot
from mrta_search.gnn_constructor import (
    JointInsertionGNN,joint_geometry,joint_graph,joint_start_state,fixed_split_patterns,
    construct_or_repair_with_gnn,prepare_joint_teacher_steps,
)
from mrta_search.lns import joint_block_candidates
from mrta_reference.model import ScheduleStatus

DATA=ROOT/'data/development/gnn_joint_shared_train_v1.jsonl'
MODEL=ROOT/'models/gnn_joint_shared_v1.pt'
CSV=ROOT/'results/gnn_joint_shared_construction.csv'
REPORT=ROOT/'docs/GNN_JOINT_SHARED_CONSTRUCTION.md'
EMPTY=((),(),(),())
CONFIG=pilot.CONFIG


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=JointInsertionGNN(64,4,normalize_inputs=checkpoint.get('normalize_inputs',False))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model,checkpoint


def load_generation_metadata():
    if REPORT.exists():
        text=REPORT.read_text(encoding='utf-8')
        if '```json\n' in text:
            return json.loads(text.split('```json\n',1)[1].split('\n```',1)[0])
    # The final report replaces the temporary generation summary. Its original
    # metadata remains in the shared checkpoint, so --phase train stays usable.
    if MODEL.exists():
        return torch.load(MODEL,map_location='cpu',weights_only=True)['metadata']
    raise ValueError('generate construction data before training')


def construction_replay(target,directions,*,max_nodes=512,max_seconds=12):
    """Bounded offline label replay, never an inference fallback or optimizer.

    Retained routes are projections of the final teacher routes. Each attempted
    parent inserts its children contiguously in decision time, in block-ID order,
    and only actions returned by the student's unchanged generator are used.
    """
    started=time.perf_counter()
    geometry=joint_geometry(target.parents,target.patterns,CONFIG)
    graph,blocks,patterns=joint_graph(geometry,CONFIG)
    target_routes=tuple(r.block_ids for r in target.routes)
    owners={b:r for r,route in enumerate(target_routes) for b in route}
    ranks={b:k for route in target_routes for k,b in enumerate(route)}
    bits={b:d for route,ds in zip(target_routes,directions) for b,d in zip(route,ds)}
    children={p:tuple(sorted((b for b in blocks.values() if b.parent_id==p),key=lambda b:b.block_id)) for p in patterns}
    empty=joint_start_state(geometry,EMPTY,EMPTY,set(patterns),blocks,CONFIG)
    visited=set();nodes=0;exhausted=False;rejections=Counter()

    def insert_parent(state,parent):
        actions=[]
        for child in children[parent]:
            r=owners[child.block_id]
            position=sum(ranks[b]<ranks[child.block_id] for b in state.routes[r])
            variants=joint_block_candidates(state,child,patterns[parent],blocks,CONFIG,insertion_limit=8)
            matches=[(after,a) for after,a in variants if a.robot==r and a.position==position
                     and a.direction==bits[child.block_id]
                     and all(d==bits[b] for route,ds in zip(after.routes,after.directions) for b,d in zip(route,ds))]
            if not matches:
                rejections['NO_EXECUTABLE_TARGET_INSERTION']+=1
                return None
            state,action=matches[0];actions.append(action)
        return state,tuple(actions)

    def search(state,pending,actions,snapshots):
        nonlocal nodes,exhausted
        if not pending:
            return (state,actions,snapshots) if state.routes==target_routes and state.directions==directions else None
        if nodes>=max_nodes or time.perf_counter()-started>=max_seconds:
            exhausted=True;return None
        key=(state.routes,state.directions)
        if key in visited:return None
        visited.add(key);nodes+=1
        # Prefer the earliest outstanding route tasks. The teacher target is
        # used only here to recover labels, never supplied to GNN inference.
        order=sorted(pending,key=lambda p:(min(ranks[b.block_id] for b in children[p]),
                                          sum(ranks[b.block_id] for b in children[p]),p))
        for parent in order:
            if time.perf_counter()-started>=max_seconds:exhausted=True;break
            inserted=insert_parent(state,parent)
            if inserted is None:continue
            after,new_actions=inserted;remaining=pending-{parent}
            result=search(after,remaining,actions+new_actions,
                          snapshots+((after,tuple(sorted(remaining)),len(actions)+len(new_actions)),))
            if result is not None:return result
            if exhausted:break
        return None

    result=search(empty,set(patterns),(),((empty,tuple(sorted(patterns)),0),))
    info=dict(replay_status='REPLAYED' if result else 'BOUNDED_REPLAY_EXHAUSTED' if exhausted else 'NO_PARENT_ATOMIC_ORDER',
              replay_nodes=nodes,replay_rejections=dict(rejections),replay_runtime_s=time.perf_counter()-started)
    return result,info


def teacher_from_local(record,entry):
    # V2 saved each Beam initial's routes and all changes relative to it. Thus
    # the original explicit bits can be recovered without generating a new Beam.
    parents=pilot.dataset.load_parents(entry)
    target=pilot.record_reference(record)
    if parents!=target.parents or target.patterns!=fixed_split_patterns(parents,CONFIG):
        raise ValueError('existing TRAIN teacher differs from loaded geometry/fixed patterns')
    bits={b:d for route,ds in zip(record['final_routes'],record['final_directions']) for b,d in zip(route,ds)}
    for block,before,after in record['changes']['flips']:
        bits[block]=before
    directions=tuple(tuple(bits[b] for b in route.block_ids) for route in target.routes)
    scheduled=pilot.oriented.schedule_explicit(target,directions,CONFIG)
    if scheduled.schedule.status is not ScheduleStatus.FEASIBLE or abs(scheduled.schedule.cmax-record['original_cmax'])>1e-6:
        raise ValueError('stored Beam initial reconstruction did not recover original feasible Cmax')
    return target,directions,scheduled


def make_construction_record(source,target,directions,scheduled,replayed):
    state,actions,snapshots=replayed
    n=len(target.parents)
    views=[]
    for fraction in (0.,.25,.5,.75):
        placed=int(n*fraction);snapshot,pending,offset=snapshots[placed]
        views.append(dict(placed_parents=placed,requested_fraction=fraction,
                          partial_routes=snapshot.routes,partial_directions=snapshot.directions,
                          pending_parents=pending,action_offset=offset))
    record=dict(entry=source['entry'],parents=source['parents'],patterns=source['patterns'],
                reference_routes=[r.block_ids for r in target.routes],partial_routes=EMPTY,partial_directions=EMPTY,
                pending_parents=tuple(p.parent_id for p in target.parents),actions=[asdict(a) for a in actions],
                final_routes=state.routes,final_directions=state.directions,cmax=scheduled.schedule.cmax,
                original_cmax=scheduled.schedule.cmax,quality='FEASIBLE_CONSTRUCTION',teacher='BEAM_FINAL_PARENT_REPLAY',
                seed=20261027,prefix_states=views)
    # Validate each view's continuation using exactly the common action set.
    geometry=joint_geometry(target.parents,target.patterns,CONFIG)
    for view in views:
        _,_,after=prepare_joint_teacher_steps(geometry,view['partial_routes'],view['partial_directions'],
                   view['pending_parents'],record['actions'][view['action_offset']:],CONFIG,insertion_limit=8)
        if after.routes!=state.routes or after.directions!=state.directions:
            raise ValueError('prefix continuation failed to recover directed teacher')
    return record


def ordered_sources(local,pilot_size):
    unique={}
    for record in local:unique.setdefault(record['entry']['instance_id'],record)
    ordered=sorted(unique.values(),key=lambda r:(r['entry']['N'],r['entry']['instance_id']))
    first=[]
    for i in range(min(pilot_size,len(ordered))):
        record=ordered[i*(len(ordered)-1)//max(1,pilot_size-1)]
        if record not in first:first.append(record)
    return first+[r for r in ordered if r not in first]


def generate_data(local,args):
    whitelist={e['instance_id']:e for group in pilot.dataset.train_sheet_candidates().values() for e in group}
    records=[];infos=[];pilot_stats=None;pilot_checks=None
    with DATA.open('w',encoding='utf-8') as f:
        for source in ordered_sources(local,args.pilot_size):
            name=source['entry']['instance_id'];started=time.perf_counter()
            if source['entry']['split']!='MLP_TRAIN' or name not in whitelist:
                raise ValueError('construction supervision is outside existing TRAIN')
            print('CONSTRUCTION TEACHER',len(infos)+1,name,flush=True)
            target,directions,scheduled=teacher_from_local(source,whitelist[name])
            replayed,info=construction_replay(target,directions)
            info.update(instance_id=name,N=len(target.parents),b32_calls=1,cmax=scheduled.schedule.cmax)
            if replayed is not None:
                record=make_construction_record(source,target,directions,scheduled,replayed)
                records.append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
            info['runtime_s']=time.perf_counter()-started;infos.append(info)
            print('TEACHER RESULT',json.dumps(info),flush=True)
            if len(records)==args.pilot_size and pilot_stats is None:
                names={r['entry']['instance_id'] for r in records}
                subset=[r for r in local if r['entry']['instance_id'] in names]
                _,v2=load_model(pilot.MODEL)
                prepared=prepare_mixed(subset,records)
                model,pilot_stats=pilot.fit_model(subset+records,args,2,initial_checkpoint=v2,prepared=prepared)
                pilot_checks=train_checks(model,subset,records,limit=3)
                print('PILOT COMPLETE',len(records),'replayable geometries',json.dumps(pilot_checks),flush=True)
            REPORT.write_text('# Shared GNN — TRAIN generation\n\n```json\n'+json.dumps(
                dict(generation=infos,pilot_training=pilot_stats,pilot_checks=pilot_checks),ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
    return records,dict(generation=infos,pilot_training=pilot_stats,pilot_checks=pilot_checks)


def prepare_mixed(local,complete):
    graphs,grouped=pilot.prepare_training(local)
    for index,record in enumerate(complete):
        target=pilot.record_reference(record)
        geometry=joint_geometry(target.parents,target.patterns,CONFIG)
        graph,steps,state=prepare_joint_teacher_steps(geometry,EMPTY,EMPTY,record['pending_parents'],record['actions'],CONFIG)
        if state.routes!=tuple(map(tuple,record['final_routes'])) or state.directions!=tuple(map(tuple,record['final_directions'])):
            raise ValueError('full construction replay mismatch')
        name=record['entry']['instance_id'];graphs.setdefault(name,graph)
        # Every completion proportion is already in the full step sequence.
        # Prefix views point into it; repeated suffix steps are never appended.
        grouped[name].extend(steps)
        print('PREPARED CONSTRUCTION',index+1,'/',len(complete),name,len(steps),'steps',flush=True)
    before=sum(map(len,grouped.values()))
    for name,steps in grouped.items():
        unique={}
        for step in steps:
            key=(step['state_routes'],step['state_directions'],step['pending_parent_ids'],step['teacher_action_key'])
            unique.setdefault(key,step)
        grouped[name]=list(unique.values())
    print('MIXED unique state/action steps',before,'->',sum(map(len,grouped.values())),flush=True)
    return graphs,grouped


def neural_initial(parents,model):
    started=time.perf_counter()
    patterns=fixed_split_patterns(parents,CONFIG)
    geometry=joint_geometry(parents,patterns,CONFIG)
    result=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),model,CONFIG,insertion_limit=8)
    scheduled=None
    if result.solution is not None:
        scheduled=pilot.oriented.schedule_explicit(result.solution,result.directions,CONFIG)
    info=dict(constructed=int(result.solution is not None),b32_calls=int(scheduled is not None),
              status='CONSTRUCTION_FAILED' if scheduled is None else scheduled.schedule.status.value,
              cmax=scheduled.schedule.cmax if scheduled is not None and scheduled.schedule.status is ScheduleStatus.FEASIBLE else None,
              runtime_s=time.perf_counter()-started,actions=len(result.actions),failure=result.rejection_reason,
              direction0=sum(a.direction==0 for a in result.actions),direction1=sum(a.direction==1 for a in result.actions),
              flip_actions=sum(bool(a.neighbour_changes) for a in result.actions))
    return scheduled,info


def train_checks(model,local,complete,*,limit=3):
    results=[]
    selected=ordered_sources(complete,limit)[:limit]
    for record in selected:
        parents=pilot.record_reference(record).parents
        _,info=neural_initial(parents,model)
        results.append(dict(check='EMPTY',instance=record['entry']['instance_id'],**info))
    for q in (4,8):
        selected=[];seen=set()
        for record in local:
            if len(record['pending_parents'])==q and record['entry']['instance_id'] not in seen:
                selected.append(record);seen.add(record['entry']['instance_id'])
            if len(selected)==limit:break
        for record in selected:
            start=time.perf_counter()
            result=construct_or_repair_with_gnn(pilot.record_reference(record),record['partial_routes'],record['partial_directions'],
                                              record['pending_parents'],model,CONFIG,insertion_limit=8)
            scheduled=None if result.solution is None else pilot.oriented.schedule_explicit(result.solution,result.directions,CONFIG)
            results.append(dict(check='LOCAL_Q'+str(q),instance=record['entry']['instance_id'],
                constructed=int(result.solution is not None),b32_calls=int(scheduled is not None),
                status='CONSTRUCTION_FAILED' if scheduled is None else scheduled.schedule.status.value,
                cmax=scheduled.schedule.cmax if scheduled is not None and scheduled.schedule.status is ScheduleStatus.FEASIBLE else None,
                runtime_s=time.perf_counter()-start,actions=len(result.actions),failure=result.rejection_reason,
                direction0=sum(a.direction==0 for a in result.actions),direction1=sum(a.direction==1 for a in result.actions),
                flip_actions=sum(bool(a.neighbour_changes) for a in result.actions)))
    for info in results:print('TRAIN CHECK',json.dumps(info),flush=True)
    return results


def save_results(rows,checkpoint):
    fields=('stage','instance','method','status','constructed','initial_cmax','final_cmax','b32_calls','b32_success',
            'search_b32_calls','search_b32_success','iterations','accepted','initial_runtime_s','search_runtime_s',
            'total_runtime_s','actions','direction0','direction1','flip_actions','failure','trace')
    with CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:row.get(k) for k in fields} for row in rows)
    metadata=checkpoint['metadata'];training=checkpoint['training'];local=checkpoint['local_diagnostics']
    generation=metadata['generation'];success=[g for g in generation if g['replay_status']=='REPLAYED']
    pilot_stats=metadata['pilot_training']
    complete=[json.loads(l) for l in DATA.read_text(encoding='utf-8').splitlines() if l]
    def fmt(value):return '—' if value is None else f'{value:.3f}'
    lines=['# 共享 GNN 联合初解构造与瓶颈重构','',
           '本轮只研究固定 WHOLE/mandatory-Y 模式，同一 JointInsertionGNN 从空路线构造和从部分路线修复。两层 SparseGATLayer、hidden=64、heads=4，联合方向和邻块翻转，显式方向一次B32；无独立方向优化、无启发式回退。','',
           '## 1. TRAIN 教师与共享训练','',
           f"仅复用V2的46个TRAIN几何；实际验证 {len(generation)} 个，父焊缝级精确重放成功 {len(success)} 个，失败 {len(generation)-len(success)} 个。未扩充PPO几何；TRAIN加载保留现有17个工作簿白名单与角色检查。DEV、ORACLE与封存数据没有产生监督。",
           '从V2保存的初始路线、最终方向和方向改变记录恢复原Beam初解，重新用B32核对原Cmax。随后仅在joint_block_candidates中寻找保持目标路线投影和显式方向的父级插入动作；每父的全部子块连续处理。每几何最多512个搜索状态/12秒，不能重放即保留失败。没有沿用Beam块级动作或向学生候选注入标签。',
           f"保存完整轨迹 {len(complete)} 条、构造动作 {sum(len(r['actions']) for r in complete)} 个；四种完成比例状态视图 {sum(len(r['prefix_states']) for r in complete)} 个。25%/50%/75%视图是同一动作列表的索引；所有视图均通过真实候选继续重放，训练时不重复累计相同后续动作。",
           '保留原294条q4/q8局部轨迹、1872个动作，与所有构造步骤混合。图和特征仅来自几何、固定模式及当前部分状态。用于纯神经推理的JointGeometry没有routes属性，初始状态四条空路线，pending为全部父焊缝；Beam路线仅在离线恢复标签和基线中使用。',
           f"先以 {0 if pilot_stats is None else pilot_stats['geometries']} 个几何的构造/局部数据进行2轮流程试训并真实调用TRAIN B32，随后丢弃试训权重。正式训练重新加载V2权重，seed={checkpoint['seed']}，AdamW lr=0.001，epochs={checkpoint['epochs']}，batch=16。V2的joint/parent均值和标准差四个buffer逐项保持完全相同，未重新拟合。",
           f"正式监督共 {training['steps']} 个唯一入池步骤、{training['geometries']} 个几何；fit_model时间 {training['runtime_s']:.3f}s（步骤预处理另计）。首/末损失 {training['history'][0]['loss']:.5f}→{training['history'][-1]['loss']:.5f}。CPU单线程，PyTorch {torch.__version__}，运行环境 D:\\pybullet_test\\.venv\\Scripts\\python.exe。没有根据DEV选择模型。",'',
           '|TRAIN几何|N|重放状态|搜索状态|重放秒|全部生成秒|','|---|---:|---|---:|---:|---:|']
    for info in generation:
        lines.append(f"|{info['instance_id']}|{info['N']}|{info['replay_status']}|{info['replay_nodes']}|{info['replay_runtime_s']:.3f}|{info['runtime_s']:.3f}|")
    trial=metadata.get('pilot_checks') or []
    lines+=['',f"教师生成合计 {sum(i['runtime_s'] for i in generation):.3f}s，初解核对B32 {sum(i['b32_calls'] for i in generation)} 次。先导试训的空路线检查B32成功 {sum(c['status']=='FEASIBLE' for c in trial if c['check']=='EMPTY')}/{sum(c['check']=='EMPTY' for c in trial)}；这只用于确认训练和完整解码流程可执行，未据此调整DEV或选择权重。"]
    lines+=['','正式训练前后在相同TRAIN样本上的完整构造与局部补全检查（每种各3例，每个完整候选最多一次B32）：','',
            '|模型|检查|实例|状态|Cmax|方向0/1|含翻转动作|秒|','|---|---|---|---|---:|---:|---:|---:|']
    for label,checks in (('V2',checkpoint.get('v2_checks',[])),('Shared',checkpoint['checks'])):
        for check in checks:
            lines.append(f"|{label}|{check['check']}|{check['instance']}|{check['status']}|{fmt(check['cmax'])}|{check['direction0']}/{check['direction1']}|{check['flip_actions']}|{check['runtime_s']:.3f}|")
    lines+=['',f"共享模型在原1872个局部动作上教师强制准确率：插入 {local['insertion_accuracy']:.2%}、父选择 {local['parent_accuracy']:.2%}；预测方向0/1={local.get('predicted_direction_0',0)}/{local.get('predicted_direction_1',0)}，含翻转动作={local['predicted_flip_actions']}。V2原局部插入73.45%、父选择49.60%、含翻转动作495；拟合变化与真实补全检查分别报告。",'',
            '|epoch|混合损失|混合插入准确率|混合父准确率|','|---:|---:|---:|---:|']
    for item in training['history']:
        diag=item['teacher_forced']
        lines.append(f"|{item['epoch']}|{item['loss']:.5f}|{diag['insertion_accuracy']:.2%}|{diag['parent_accuracy']:.2%}|")
    lines+=['','## 2. 六 DEV 纯初解','',
            '模型固定后才运行DEV。Beam对照保留原实现并如实统计其多个完整候选的B32调用；每个候选仅一次B32。两个GNN各自仅从几何和几何确定的固定模式开始，每个完整构造仅一次B32。三者模式逐例一致。时间含模式/图构建、插入、B32，不含公共Excel读取和统一预加载模型。','',
            '|实例|方法|构造完成|状态|真实Cmax|B32调用|秒|','|---|---|---:|---|---:|---:|---:|']
    for row in rows:
        if row['stage']=='DEV_INIT':
            lines.append(f"|{row['instance']}|{row['method']}|{row['constructed']}|{row['status']}|{fmt(row['final_cmax'])}|{row['b32_calls']}|{row['total_runtime_s']:.3f}|")
    lines+=['','## 3. 三方法24轮端到端调度反馈','',
            'A=Beam+NEW；B=Beam+GNN-Shared；C=GNN-Shared+GNN-Shared。复用iterative_group，seed=20261027，24轮，RANDOM/TRAVEL/LOAD，每三轮切换q4/q8；每轮最多一次B32。严格改善后更新current的路线、方向和B32调度。A/B共用同一个Beam初解；C初解不可调度时直接记失败，不执行修复、不回退。总时间=对应初解耗时+实际24轮搜索耗时；初解实验测得的对象和耗时复用，避免重复评价同一初解。','',
            '|实例|方法|状态|初始→最终Cmax|接纳|搜索B32成功/调用|初解秒|搜索秒|总秒|','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['stage']=='DEV_E2E':
            lines.append(f"|{row['instance']}|{row['method']}|{row['status']}|{fmt(row['initial_cmax'])}→{fmt(row['final_cmax'])}|{row['accepted']}|{row['search_b32_success']}/{row['search_b32_calls']}|{row['initial_runtime_s']:.3f}|{row['search_runtime_s']:.3f}|{row['total_runtime_s']:.3f}|")
    lines+=['','DEADLOCK仅表示给定方向下现有B32未找到调度，不是数学不可行证明。完整路线构造、B32可调度与Cmax质量分别记录；不以失败候选的虚构Cmax参与比较。CSV trace保留全部搜索迭代，并在成功接纳处保存实际改变。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def evaluate(model,checkpoint,args):
    v2,_=load_model(pilot.MODEL)
    by_name={e['sheet_name']:e for e in pilot.entries()[1]}
    rows=[];initials={};initial_info={}
    # Complete every initial comparison before starting any feedback search.
    for name in pilot.oriented.NAMES:
        entry=by_name[name];parents=pilot.dataset.load_parents(entry)
        for method,policy in (('Beam-Init',None),('GNN-V2-Init',v2),('GNN-Shared-Init',model)):
            if policy is None:
                started=time.perf_counter()
                initial,attempts=pilot.oriented.common_initial_beam(parents,CONFIG,seed=args.seed)
                info=dict(constructed=int(initial is not None),status='BEAM_FAILED' if initial is None else initial.schedule.status.value,
                    cmax=None if initial is None else initial.schedule.cmax,runtime_s=time.perf_counter()-started,
                    b32_calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts),
                    b32_success=sum(a['status']=='FEASIBLE' for a in attempts),actions=0,direction0=0,direction1=0,flip_actions=0,
                    failure=json.dumps(Counter(a['status'] for a in attempts)))
                if initial is not None and initial.solution.patterns!=fixed_split_patterns(parents,CONFIG):
                    raise ValueError('Beam and pure GNN fixed patterns differ')
            else:
                initial,info=neural_initial(parents,policy)
                info['b32_success']=int(info['status']=='FEASIBLE')
            initials[name,method]=initial;initial_info[name,method]=info
            row=dict(stage='DEV_INIT',instance=name,method=method,status=info['status'],constructed=info['constructed'],
                     initial_cmax=info['cmax'],final_cmax=info['cmax'],b32_calls=info['b32_calls'],b32_success=info['b32_success'],
                     search_b32_calls=0,search_b32_success=0,iterations=0,accepted=0,initial_runtime_s=info['runtime_s'],
                     search_runtime_s=0.,total_runtime_s=info['runtime_s'],actions=info['actions'],direction0=info['direction0'],
                     direction1=info['direction1'],flip_actions=info['flip_actions'],failure=info['failure'],trace='[]')
            rows.append(row);save_results(rows,checkpoint)
            print('DEV INITIAL',name,method,json.dumps(info),flush=True)
    for name in pilot.oriented.NAMES:
        for label,initial_method,repair_method,policy in (
            ('A:Beam+NEW','Beam-Init','NEW',None),
            ('B:Beam+GNN-Shared','Beam-Init','GNN-Shared',model),
            ('C:GNN-Shared+GNN-Shared','GNN-Shared-Init','GNN-Shared',model)):
            initial=initials[name,initial_method];info=initial_info[name,initial_method]
            row=dict(stage='DEV_E2E',instance=name,method=label,status='INITIAL_'+info['status'],constructed=info['constructed'],
                     initial_cmax=info['cmax'],final_cmax=None,b32_calls=info['b32_calls'],b32_success=info['b32_success'],
                     search_b32_calls=0,search_b32_success=0,iterations=0,accepted=0,initial_runtime_s=info['runtime_s'],
                     search_runtime_s=0.,total_runtime_s=info['runtime_s'],actions=info['actions'],direction0=info['direction0'],
                     direction1=info['direction1'],flip_actions=info['flip_actions'],failure=info['failure'],trace='[]')
            if initial is not None and initial.schedule.status is ScheduleStatus.FEASIBLE:
                search=pilot.iterative_group(by_name[name],initial,repair_method,policy,args)
                row.update(status='FEASIBLE',final_cmax=search['best_cmax'],b32_calls=info['b32_calls']+search['b32_calls'],
                    b32_success=info['b32_success']+search['b32_success'],search_b32_calls=search['b32_calls'],
                    search_b32_success=search['b32_success'],iterations=search['attempts'],accepted=search['accepted'],
                    search_runtime_s=search['runtime_s'],total_runtime_s=info['runtime_s']+search['runtime_s'],
                    actions=info['actions']+search['neural_actions'],direction0=info['direction0']+search['direction0_actions'],
                    direction1=info['direction1']+search['direction1_actions'],flip_actions=info['flip_actions']+search['flip_actions'],
                    failure=search['failures'],trace=search['trace'])
            rows.append(row);save_results(rows,checkpoint)
            print('END TO END',name,label,row['status'],row['final_cmax'],'accepted',row['accepted'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','data','train','eval'),default='all')
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--pilot-size',type=int,default=10)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed);random.seed(args.seed)
    local=[json.loads(l) for l in pilot.DATA.read_text(encoding='utf-8').splitlines() if l]
    if args.phase in ('all','data'):
        complete,metadata=generate_data(local,args)
        if args.phase=='data':return
    else:
        complete=[json.loads(l) for l in DATA.read_text(encoding='utf-8').splitlines() if l]
        metadata=load_generation_metadata() if args.phase=='train' else None
    if args.phase in ('all','train'):
        v2_model,v2=load_model(pilot.MODEL)
        v2_checks=train_checks(v2_model,local,complete,limit=3)
        prepared=prepare_mixed(local,complete)
        model,stats=pilot.fit_model(local+complete,args,args.epochs,initial_checkpoint=v2,prepared=prepared)
        for key in ('joint_mean','joint_scale','parent_mean','parent_scale'):
            if not torch.equal(model.state_dict()[key],v2['state_dict'][key]):
                raise ValueError('V2 normalization changed during continued training')
        local_graphs,local_steps=pilot.prepare_training(local)
        checks=train_checks(model,local,complete,limit=3)
        checkpoint=dict(state_dict=model.state_dict(),hidden_dim=64,heads=4,normalize_inputs=True,seed=args.seed,
                        epochs=args.epochs,training=stats,local_diagnostics=pilot.action_diagnostics(model,local_graphs,local_steps),
                        checks=checks,v2_checks=v2_checks,metadata=metadata,construction_trajectories=len(complete))
        torch.save(checkpoint,MODEL);print('SHARED MODEL SAVED',MODEL,flush=True)
        if args.phase=='train':return
    else:
        model,checkpoint=load_model(MODEL)
    evaluate(model,checkpoint,args)


if __name__=='__main__':main()
