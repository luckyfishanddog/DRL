"""Shared parent-atomic GNN construction and repair, with real B32 evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import csv
import json
import math
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


def _model_from_checkpoint(checkpoint):
    from mrta_search.gnn_constructor import CoordinationJointInsertionGNN,ModeJointInsertionGNN
    model_types={'JointInsertionGNN':JointInsertionGNN,
                 'CoordinationJointInsertionGNN':CoordinationJointInsertionGNN,
                 'ModeJointInsertionGNN':ModeJointInsertionGNN}
    kind=checkpoint.get('model_type','JointInsertionGNN')
    if kind not in model_types:
        raise ValueError('unknown joint construction model: '+str(kind))
    model=model_types[kind](checkpoint.get('hidden_dim',64),checkpoint.get('heads',4),
                           normalize_inputs=checkpoint.get('normalize_inputs',False),
                           opportunity_features=checkpoint.get('opportunity_features',False))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    if kind in ('CoordinationJointInsertionGNN','ModeJointInsertionGNN'):
        model.use_dynamic_context=checkpoint.get('use_dynamic_context',True)
    return model


def _expand_neural_checkpoint(payload):
    if 'research_archive' not in payload:return payload
    import zlib
    if payload.get('research_archive_format')!='zlib-json':raise ValueError('unknown research archive format')
    metadata=json.loads(zlib.decompress(payload['research_archive']).decode('utf-8'))
    full={k:v for k,v in payload.items() if k not in ('research_archive','research_archive_format')}
    full.update(metadata['main'])
    if 'candidate' in metadata:
        full['refinement_candidate']=dict(full['refinement_candidate'],**metadata['candidate'])
    if 'opportunity_candidate' in metadata:
        full['opportunity_cmax_candidate']=dict(full['opportunity_cmax_candidate'],**metadata['opportunity_candidate'])
    if 'early_candidate' in metadata:
        full['early_cmax_candidate']=dict(full['early_cmax_candidate'],**metadata['early_candidate'])
    return full


def read_neural_checkpoint(path,*,include_research=True):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    return _expand_neural_checkpoint(payload) if include_research else payload


def save_neural_checkpoint(checkpoint,path):
    """Keep both exact weight sets; compress diagnostics unused by inference."""
    import zlib
    checkpoint=_expand_neural_checkpoint(checkpoint)
    keep=('state_dict','model_type','hidden_dim','heads','normalize_inputs','use_dynamic_context',
          'scientific_config','mode_head_before_calibration','opportunity_features')
    def split(part):
        return ({k:v for k,v in part.items() if k in keep},
                {k:v for k,v in part.items() if k not in keep and k not in ('refinement_candidate','opportunity_cmax_candidate','early_cmax_candidate')})
    thin,main=split(checkpoint);metadata=dict(main=main)
    if 'refinement_candidate' in checkpoint:
        thin['refinement_candidate'],metadata['candidate']=split(checkpoint['refinement_candidate'])
    if 'opportunity_cmax_candidate' in checkpoint:
        thin['opportunity_cmax_candidate'],metadata['opportunity_candidate']=split(checkpoint['opportunity_cmax_candidate'])
    if 'early_cmax_candidate' in checkpoint:
        thin['early_cmax_candidate'],metadata['early_candidate']=split(checkpoint['early_cmax_candidate'])
    thin['research_archive']=zlib.compress(json.dumps(metadata,ensure_ascii=False,allow_nan=False,
                                                    separators=(',',':')).encode('utf-8'))
    thin['research_archive_format']='zlib-json'
    torch.save(thin,path)


def load_model(path,*,variant=None):
    checkpoint=read_neural_checkpoint(path)
    if variant is not None:checkpoint=checkpoint[variant]
    return _model_from_checkpoint(checkpoint),checkpoint


def load_neural_initial_models(path):
    """Read the existing artifact once and instantiate both portfolio policies."""
    checkpoint=read_neural_checkpoint(path,include_research=False)
    return _model_from_checkpoint(checkpoint),_model_from_checkpoint(checkpoint['refinement_candidate'])


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


def construction_replay(target,directions,*,max_nodes=512,max_seconds=12,config=None):
    """Bounded offline label replay, never an inference fallback or optimizer.

    Retained routes are projections of the final teacher routes. Each attempted
    parent inserts its children contiguously in decision time, in block-ID order,
    and only actions returned by the student's unchanged generator are used.
    """
    config=CONFIG if config is None else config
    started=time.perf_counter()
    geometry=joint_geometry(target.parents,target.patterns,config)
    graph,blocks,patterns=joint_graph(geometry,config)
    target_routes=tuple(r.block_ids for r in target.routes)
    owners={b:r for r,route in enumerate(target_routes) for b in route}
    ranks={b:k for route in target_routes for k,b in enumerate(route)}
    bits={b:d for route,ds in zip(target_routes,directions) for b,d in zip(route,ds)}
    children={p:tuple(sorted((b for b in blocks.values() if b.parent_id==p),key=lambda b:b.block_id)) for p in patterns}
    empty=joint_start_state(geometry,EMPTY,EMPTY,set(patterns),blocks,config)
    visited=set();nodes=0;exhausted=False;rejections=Counter()

    def insert_parent(state,parent):
        actions=[]
        for child in children[parent]:
            r=owners[child.block_id]
            position=sum(ranks[b]<ranks[child.block_id] for b in state.routes[r])
            variants=joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=8)
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


def research_metric_summary(values):
    import statistics
    return dict(count=len(values),minimum=min(values),median=statistics.median(values),
                mean=statistics.fmean(values),maximum=max(values)) if values else dict(count=0)


def research_result_groups(rows):
    """Keep experiment/method/status distinct and missing failed Cmax explicit."""
    import math
    grouped={};nested=0
    for experiment,row in rows:
        if not isinstance(row,dict):nested+=1;continue
        if not any(k in row for k in ('method','algorithm','variant','status','certified','cmax','Cmax_at_60','final_cmax','best_cmax')):
            nested+=1;continue
        method=str(row.get('method',row.get('algorithm',row.get('variant','UNSPECIFIED'))))
        status=str(row.get('status','CERTIFIED' if row.get('certified') is True else 'UNSPECIFIED'))
        key=(str(experiment),method,status)
        group=grouped.setdefault(key,dict(rows=0,metrics={},certified=0))
        group['rows']+=1
        certified=row.get('certified_at_60',row.get('certified'))
        group['certified']+=int(certified is True)
        for field,value in row.items():
            if 'cmax' not in field.lower() or isinstance(value,bool):continue
            try:value=float(value)
            except (TypeError,ValueError):continue
            if math.isfinite(value) and value>0:
                group['metrics'].setdefault(field,[]).append(value)
        # Cmax@60 is authoritative when present, including null failures.
        field=next((k for k in ('Cmax_at_60','cmax','final_cmax','best_cmax') if k in row),None)
        value=row.get(field) if field else None
        if certified is not False:
            try:value=float(value)
            except (TypeError,ValueError):value=None
            if value is not None and math.isfinite(value) and value>0:
                group['metrics'].setdefault('successful_primary_cmax',[]).append(value)
    summaries=[]
    for key,group in sorted(grouped.items()):
        summaries.append(dict(experiment=key[0],method=key[1],status=key[2],rows=group['rows'],
                              certified=group['certified'],metrics={k:research_metric_summary(v)
                              for k,v in group['metrics'].items()}))
    return dict(groups=summaries,nested_summary_rows=nested)


def research_inventory():
    """Inspect all retained experiment containers without creating exports."""
    import gzip
    import sqlite3
    inventory=[]
    csv.field_size_limit(100_000_000)
    for path in sorted((ROOT/'results').rglob('*.csv')):
        counts=Counter();methods=Counter();statuses=Counter();errors=[];analysis_rows=[]
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for raw in csv.DictReader(stream):
                counts['rows']+=1
                try:
                    row=json.loads(raw['payload']) if raw.get('payload') else raw
                except (json.JSONDecodeError,TypeError) as error:
                    errors.append(str(error));continue
                methods[str(row.get('method',row.get('algorithm','UNSPECIFIED')))]+=1
                statuses[str(row.get('status','UNSPECIFIED'))]+=1
                analysis_rows.append((row.get('experiment',row.get('phase',row.get('stage','CSV'))),row))
                details=row.get('details') or {}
                counts['rows_with_directed_states']+=int(any(
                    isinstance(details.get(k),dict) and details[k].get('solution') and details[k].get('directions')
                    for k in ('initial_state','final_state')))
        inventory.append(dict(path=path.relative_to(ROOT).as_posix(),kind='CSV',**counts,
                              methods=dict(methods),statuses=dict(statuses),parse_errors=errors[:3],
                              analysis=research_result_groups(analysis_rows)))
    for path in sorted((ROOT/'results').rglob('*.jsonl.gz')):
        kinds=Counter();rows=0
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():continue
                row=json.loads(line);rows+=1;kinds[str(row.get('record','UNSPECIFIED'))]+=1
        inventory.append(dict(path=path.relative_to(ROOT).as_posix(),kind='GZIP_JSONL',rows=rows,records=dict(kinds)))
    for path in sorted((ROOT/'data/development').glob('*.sqlite')):
        # mode=ro prevents database creation and all SQLite writes.
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True) as connection:
            names=[r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            tables={};analyses={}
            for name in names:
                quoted='"'+name.replace('"','""')+'"'
                tables[name]=connection.execute('SELECT COUNT(*) FROM '+quoted).fetchone()[0]
                columns={r[1] for r in connection.execute('PRAGMA table_info('+quoted+')')}
                if name in ('experiments','results','c2_experiments','trajectories'):
                    field=next((k for k in ('payload','result') if k in columns),None)
                    if field:
                        cohort='experiment' if 'experiment' in columns else ('kind' if 'kind' in columns else None)
                        query='SELECT '+('"'+cohort+'",' if cohort else '')+'"'+field+'" FROM '+quoted
                        records=[]
                        for raw in connection.execute(query):
                            records.append((str(raw[0]) if cohort else name,json.loads(raw[-1])))
                        analyses[name]=research_result_groups(records)
                elif 'status' in columns and 'reference_cmax' in columns:
                    groups={}
                    for status,cmax in connection.execute('SELECT status,reference_cmax FROM '+quoted):
                        groups.setdefault(str(status),[]).append(cmax)
                    analyses[name]=[dict(status=status,rows=len(values),reference_cmax=research_metric_summary(
                        [float(v) for v in values if v is not None and float(v)>0])) for status,values in sorted(groups.items())]
        inventory.append(dict(path=path.relative_to(ROOT).as_posix(),kind='SQLITE',tables=tables,analysis=analyses))
    for filename in ('gnn_experts_fixed_split_v2.json','gnn_joint_repair_train.jsonl',
                     'gnn_joint_repair_train_v2.jsonl','gnn_joint_shared_train_v1.jsonl'):
        path=ROOT/'data/development'/filename
        if not path.exists():
            inventory.append(dict(path=path.relative_to(ROOT).as_posix(),kind='HISTORICAL_GNN',missing=True));continue
        if path.suffix=='.jsonl':
            rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        else:
            payload=json.loads(path.read_text(encoding='utf-8'));rows=payload.get('experts',[]) if isinstance(payload,dict) else payload
        inventory.append(dict(path=path.relative_to(ROOT).as_posix(),kind='HISTORICAL_GNN',rows=len(rows),
                              teachers=dict(Counter(str(r.get('teacher','UNSPECIFIED')) for r in rows))))
    return dict(containers=inventory,policy='All containers inform coverage/diagnosis; scalar-only rows are not action supervision.')


def research_rows(path):
    csv.field_size_limit(100_000_000)
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return [json.loads(raw['payload']) for raw in csv.DictReader(stream) if raw.get('record')=='run']


def research_restore(state,config):
    from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail
    from mrta_reference.solution import canonicalize
    from mrta_reference.scope import FORMAL_SCOPE_V2
    raw=state['solution']
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in raw['parents'])
    patterns=tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
        None if p['rail'] is None else Rail(p['rail'])) for p in raw['patterns'])
    routes={r['robot_id']:tuple(r['block_ids']) for r in raw['routes']}
    solution=canonicalize(parents,patterns,routes,config,scope=FORMAL_SCOPE_V2)
    directions=tuple(tuple(bits) for bits in state['directions'])
    if len(directions)!=4 or any(len(r.block_ids)!=len(bits) or any(d not in (0,1) for d in bits)
                                 for r,bits in zip(solution.routes,directions)):
        raise ValueError('retained research teacher has invalid explicit directions')
    # Do not silently change the historical labels through canonicalization.
    if solution.patterns!=patterns or any(solution.routes[r].block_ids!=routes[r] for r in range(4)):
        raise ValueError('retained research teacher changed under current canonicalization')
    # Renaming parents or reversing their endpoint convention does not make a
    # new physical geometry. Graph indices still require the actual block IDs.
    geometry_key=tuple(sorted(tuple(sorted((p.start,p.end))) for p in solution.parents))
    graph_key=(tuple((p.parent_id,p.start,p.end) for p in solution.parents),solution.patterns)
    directed_key=(graph_key,tuple(r.block_ids for r in solution.routes),directions)
    return solution,directions,geometry_key,graph_key,directed_key


def research_teachers(rows,config,limit=None,*,deployment_only=True):
    """Best per geometry/method, then deduplicate directed solutions across teachers."""
    counts=Counter();best={};geometry_names={}
    for row in rows:
        for label in ('initial_state','final_state'):
            state=(row.get('details') or {}).get(label)
            if not state or not state.get('solution') or state.get('cmax') is None:
                counts['missing_directed_state']+=1;continue
            solution,bits,gkey,graph_key,key=research_restore(state,config)
            counts['directed_states']+=1
            if solution.patterns!=fixed_split_patterns(solution.parents,config):
                counts['nonfixed_pattern_states']+=1
                if deployment_only:
                    counts['excluded_nondeployment_patterns']+=1;continue
            else:counts['fixed_pattern_states']+=1
            geometry_names[gkey]=row['instance']
            item=dict(solution=solution,directions=bits,cmax=float(state['cmax']),geometry_key=gkey,
                      graph_key=graph_key,directed_key=key,instance=row['instance'],teacher=row['method'],
                      source_state=label,seed=row['seed'])
            pair=(gkey,row['method'])
            if pair not in best or (item['cmax'],item['source_state'],item['seed'])<(
                best[pair]['cmax'],best[pair]['source_state'],best[pair]['seed']):best[pair]=item
    ordered=sorted(best.values(),key=lambda r:(len(r['solution'].parents),r['instance'],r['teacher']))
    counts['best_geometry_teacher_pairs']=len(ordered)
    unique={}
    for item in ordered:
        if item['directed_key'] not in unique:
            unique[item['directed_key']]=dict(item,teachers=[item['teacher']])
        else:unique[item['directed_key']]['teachers'].append(item['teacher'])
    selected=list(unique.values())
    counts['unique_directed_teachers']=len(selected)
    if limit is not None:selected=selected[:limit]
    counts['selected_teachers']=len(selected)
    counts['selected_geometries']=len({r['geometry_key'] for r in selected})
    return selected,dict(counts)


def research_certify(scheduled,config):
    from mrta_reference.certifier import certify_schedule
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_reference.geometry import finite_x_split_validator
    from mrta_reference.model import SplitKind
    if scheduled.schedule.status is not ScheduleStatus.FEASIBLE:return False
    validator=(finite_x_split_validator(scheduled.solution.parents,config)
               if any(p.kind is SplitKind.X_SPLIT for p in scheduled.solution.patterns) else None)
    return certify_schedule(scheduled.solution,scheduled.schedule,config,scope=FORMAL_SCOPE_V2,
                            x_split_validator=validator).certified


def research_prepare(teachers,config,*,certify=True):
    """Replay executable labels; merge parent and insertion distributions separately.

    The noncertifying path consumes retained FEASIBLE B32 teacher records. It
    does not schedule them a second time or introduce an independent certifier.
    Historical callers retain their original B32/certification replay by default.
    """
    from mrta_search.lns import joint_action_key
    graphs={};merged={};raw_steps=[];diagnostics=[]
    best_cmax={}
    for item in teachers:
        key=item['geometry_key'];best_cmax[key]=min(best_cmax.get(key,float('inf')),item['cmax'])
    for index,item in enumerate(teachers):
        solution=item['solution'];bits=item['directions'];started=time.perf_counter()
        if certify:
            scheduled=pilot.oriented.schedule_explicit(solution,bits,config)
            if not research_certify(scheduled,config) or abs(scheduled.schedule.cmax-item['cmax'])>1e-6:
                raise ValueError('research teacher failed current B32/certification/Cmax replay: '+item['instance'])
        elif item.get('b32_status')!=ScheduleStatus.FEASIBLE.value or not math.isfinite(item['cmax']) or item['cmax']<=0:
            raise ValueError('retained teacher has no FEASIBLE B32 record: '+item['instance'])
        replay,info=construction_replay(solution,bits,max_nodes=128,max_seconds=2,config=config)
        diag=dict(instance=item['instance'],teachers=item['teachers'],cmax=item['cmax'],
                  source_state=item['source_state'],source_seed=item['seed'],**info)
        diagnostics.append(diag)
        if replay is None:
            print('RESEARCH TEACHER',index+1,'/',len(teachers),item['instance'],info['replay_status'],flush=True);continue
        state,actions,_=replay
        diag['actions']=[asdict(action) for action in actions]
        geometry=joint_geometry(solution.parents,solution.patterns,config)
        graph,steps,after=prepare_joint_teacher_steps(geometry,EMPTY,EMPTY,
            tuple(p.parent_id for p in solution.parents),[asdict(a) for a in actions],config)
        if after.routes!=state.routes or after.directions!=state.directions:
            raise ValueError('research teacher replay changed its directed result')
        gkey=item['graph_key'];graphs.setdefault(gkey,graph);raw_steps.extend(steps)
        weight=best_cmax[item['geometry_key']]/item['cmax']
        for step in steps:
            prefix=(gkey,step['state_routes'],step['state_directions'],step['pending_parent_ids'])
            action_set=tuple(joint_action_key(a) for a in step['candidate_actions'])
            cases=[('insertion',prefix+(action_set,),step['target'],len(action_set))]
            if len(step['parent_ids'])>1:
                cases.append(('parent',prefix+(step['parent_ids'],),step['parent_target'],len(step['parent_ids'])))
            for kind,key,target,count in cases:
                ident=(kind,key)
                if ident not in merged:
                    merged[ident]=dict(kind=kind,graph_key=gkey,step=step,target=torch.zeros(count),weights=[])
                merged[ident]['target'][target]+=weight;merged[ident]['weights'].append(weight)
        diag['runtime_s']=time.perf_counter()-started
        print('RESEARCH TEACHER',index+1,'/',len(teachers),item['instance'],'REPLAYED',len(steps),'steps',flush=True)
    examples=[]
    for example in merged.values():
        example['target']/=example['target'].sum()
        weights=example.pop('weights')
        example['weight']=sum(weights)/len(weights)
        examples.append(example)
    if not examples:raise ValueError('no current-domain executable research teachers were replayable')
    return graphs,examples,raw_steps,diagnostics


def research_train(graphs,examples,raw_steps,args,*,model=None,mode_examples=(),geometry_bounds=None):
    from mrta_search.gnn_constructor import CoordinationJointInsertionGNN
    seed=20261104;torch.manual_seed(seed);rng=random.Random(seed)
    model=CoordinationJointInsertionGNN(64,4,normalize_inputs=True) if model is None else model
    model.fit_input_scaling(raw_steps)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    grouped={key:[] for key in graphs}
    for example in examples:grouped[example['graph_key']].append(example)
    # A certified teacher's Cmax is a valid processing-load ceiling, including
    # robot eligibility and coordination. This auxiliary loss is soft; it is
    # neither an inference mask nor an equal-four-load assumption.
    if geometry_bounds:
        for example in examples:
            if example['kind']!='insertion':continue
            graph=graphs[example['graph_key']];step=example['step']
            gkey=tuple(sorted(tuple(sorted((p[1],p[2]))) for p in example['graph_key'][0]))
            ceiling=geometry_bounds[gkey]
            by_id={b.block_id:b for b in graph.blocks}
            total=max(1.,sum(CONFIG.process_time(b.length) for b in graph.blocks))
            projected=step['features'][:,:4].clone()*total
            for i,action in enumerate(step['candidate_actions']):
                projected[i,action.robot]+=CONFIG.process_time(by_id[action.block_id].length)
            example['load_excess']=torch.relu(projected.max(1).values/ceiling-1.)
    mode_groups={}
    for example in mode_examples:mode_groups.setdefault(example['geometry_key'],[]).append(example)
    history=[];started=time.perf_counter()
    for epoch in range(args.research_epochs):
        model.train();keys=list(grouped);rng.shuffle(keys);total=0.;count=0;load_total=0.
        for key in keys:
            pool=list(grouped[key]);rng.shuffle(pool);graph=graphs[key]
            for offset in range(0,len(pool),16):
                optimizer.zero_grad(set_to_none=True);embeddings=model.encode_static(graph);losses=[]
                for example in pool[offset:offset+16]:
                    step=example['step']
                    context=model.contextualize_state(graph,embeddings,step['state_routes'],step['state_directions'])
                    if example['kind']=='parent':
                        logits=model.parent_logits(graph,context,step['parent_ids'],step['parent_features'])
                    else:logits=model.joint_action_logits(context,step['indices'],step['features'])
                    item=-example['weight']*(example['target']*torch.log_softmax(logits,0)).sum()
                    if 'load_excess' in example:
                        auxiliary=(torch.softmax(logits,0)*example['load_excess']).sum()
                        item=item+auxiliary;load_total+=float(auxiliary.detach())
                    losses.append(item)
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite research training loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();total+=float(loss.detach())*len(losses);count+=len(losses)
        mode_total=mode_count=0
        for key in mode_groups:
            pool=list(mode_groups[key]);rng.shuffle(pool)
            for offset in range(0,len(pool),16):
                batch=pool[offset:offset+16];optimizer.zero_grad(set_to_none=True)
                graph=batch[0]['graph'];emb=model.encode_static(graph);losses=[]
                for example in batch:
                    logits=model.mode_logits(graph,emb,example['parent_ids'],example['features'])
                    losses.append(-(example['target']*torch.log_softmax(logits,0)).sum())
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite mode supervision')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();mode_total+=float(loss.detach())*len(batch);mode_count+=len(batch)
        history.append(dict(epoch=epoch+1,loss=total/count,examples=count,
                            load_auxiliary=load_total/max(1,count),mode_loss=mode_total/max(1,mode_count)))
        print('RESEARCH EPOCH',epoch+1,'/',args.research_epochs,'loss',round(total/count,6),flush=True)
    return model,dict(seed=seed,epochs=args.research_epochs,learning_rate=.001,batch_size=16,
                      gradient_clip=1.,runtime_s=time.perf_counter()-started,history=history,
                      examples=len(examples),graphs=len(graphs),
                      dynamic_context=model.use_dynamic_context if hasattr(model,'use_dynamic_context') else False,
                      mode_examples=len(mode_examples),load_auxiliary=bool(geometry_bounds),
                      multi_target_examples=sum(int(e['target'].count_nonzero())>1 for e in examples))


class _ResearchStaticContext(torch.nn.Module):
    """Inference-only ablation: all trained weights retained, dynamic fusion bypassed."""
    def __init__(self,model):super().__init__();self.model=model
    def encode_static(self,graph):return self.model.encode_static(graph)
    def contextualize_state(self,graph,embeddings,routes,directions):return embeddings
    def parent_logits(self,*args):return self.model.parent_logits(*args)
    def joint_action_logits(self,*args):return self.model.joint_action_logits(*args)


def research_evaluate(model,rows,config,limit,*,training_geometry_keys=()):
    geometries={}
    for row in rows:
        state=(row.get('details') or {}).get('final_state')
        if not state:continue
        solution,_,gkey,_,_=research_restore(state,config)
        geometries.setdefault(gkey,dict(instance=row['instance'],tier=row.get('tier'),parents=solution.parents))
    if set(geometries).intersection(training_geometry_keys):
        raise ValueError('training/evaluation geometry overlap')
    ordered=sorted(geometries.values(),key=lambda r:(len(r['parents']),r['instance']))
    count=min(limit,len(ordered));indices=sorted({i*(len(ordered)-1)//max(1,count-1) for i in range(count)})
    selected=[ordered[i] for i in indices]
    old,_=load_model(MODEL);static=_ResearchStaticContext(model)
    methods=(('SHARED_OLD',old),('NEW_STATIC_CONTEXT_INFERENCE_ABLATION',static),('NEW_DYNAMIC_CONTEXT',model),('RULE_BEAM',None))
    results=[]
    for entry in selected:
        for name,current_model in methods:
            started=time.perf_counter();parents=entry['parents'];calls=0;failure=None
            if current_model is None:
                scheduled,attempts=pilot.oriented.common_initial_beam(parents,config,max_reference_calls=1)
                calls=len(attempts)
            else:
                geometry=joint_geometry(parents,fixed_split_patterns(parents,config),config)
                completion=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),
                                                      current_model,config,insertion_limit=8,decode_mode='greedy')
                failure=completion.rejection_reason
                scheduled=None
                if completion.solution is not None:
                    calls=1;scheduled=pilot.oriented.schedule_explicit(completion.solution,completion.directions,config)
            certified=scheduled is not None and research_certify(scheduled,config)
            result=dict(instance=entry['instance'],tier=entry['tier'],N=len(parents),method=name,
                        status='CONSTRUCTION_FAILED' if scheduled is None else scheduled.schedule.status.value,
                        certified=certified,cmax=scheduled.schedule.cmax if certified else None,
                        runtime_s=time.perf_counter()-started,b32_calls=calls,failure=failure)
            results.append(result);print('RESEARCH EVAL',entry['instance'],name,result['status'],result['cmax'],flush=True)
    return dict(role='Previously consumed formal geometries reused as development holdout; no independent claim.',
                selection='N/instance sorted, equal-index spacing; no outcome selection.',
                static_ablation='Inference-only bypass under identical trained weights; not independently trained.',
                budget='One greedy construction and at most one B32 call per method/geometry; model preload excluded.',rows=results)


def run_research(args):
    from mrta_reference.model import ScientificConfig
    config=ScientificConfig()
    if args.research_epochs<1 or args.research_eval_limit<1 or (
        args.research_train_limit is not None and args.research_train_limit<1):
        raise ValueError('research epochs and limits must be positive')
    torch.set_num_threads(1)
    inventory=research_inventory()
    print('RESEARCH INVENTORY',len(inventory['containers']),'containers',flush=True)
    training_rows=research_rows(ROOT/'results/parent_mode_joint_reconstruction_05.csv')
    evaluation_rows=research_rows(ROOT/'results/final_parent_mode_joint_reconstruction_05.csv')
    teachers,coverage=research_teachers(training_rows,config,args.research_train_limit)
    graphs,examples,steps,replays=research_prepare(teachers,config)
    model,training=research_train(graphs,examples,steps,args)
    evaluation=research_evaluate(model,evaluation_rows,config,args.research_eval_limit,
                                 training_geometry_keys={item['geometry_key'] for item in teachers})
    result=dict(scientific_config=config.scientific_mapping(),inventory=inventory,coverage=coverage,
                training_source='Development 162 rows only; no formal-500, historical-domain or scalar-only supervision.',
                replay=replays,training=training,evaluation=evaluation,checkpoint_saved=None)
    if args.research_save:
        target=Path(args.research_save).resolve()
        if not target.is_relative_to(ROOT.resolve()) or not target.parent.is_dir():
            raise ValueError('research checkpoint must use an existing directory inside DRL')
        if target.exists():raise FileExistsError('research never overwrites an existing checkpoint')
        torch.save(dict(model_type='CoordinationJointInsertionGNN',state_dict=model.state_dict(),hidden_dim=64,heads=4,
                        normalize_inputs=True,use_dynamic_context=model.use_dynamic_context,
                        scientific_config=config.scientific_mapping(),training=training,coverage=coverage,
                        replay=replays,evaluation=evaluation,training_source=result['training_source']),target)
        result['checkpoint_saved']=str(target)
    print('RESEARCH_RESULT '+json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return result


def refinement_split(rows):
    """Keep all methods/seeds/states of a physical geometry in one group."""
    evaluation_names={r['instance'] for r in rows if r.get('tier') in
        ('SMALL','MEDIUM','LARGE','HIGH_WORKLOAD') and
        r['instance'].split('_')[-1] in ('02','04') and not r['instance'].startswith('N')}
    holdout=[r for r in rows if r['instance'] in evaluation_names]
    training=[r for r in rows if r['instance'] not in evaluation_names]
    def keys(group):
        return {research_restore(state,CONFIG)[2] for row in group
                for label in ('initial_state','final_state')
                if (state:=(row.get('details') or {}).get(label))}
    overlap=keys(training)&keys(holdout)
    if overlap:raise ValueError('training/evaluation geometry overlap')
    return training,holdout,dict(rule='Existing formal tier _02/_04 are development holdout; all seeds/methods stay together.',
        training_runs=len(training),holdout_runs=len(holdout),training_geometries=len(keys(training)),
        holdout_geometries=len(keys(holdout)),holdout_names=sorted(evaluation_names))


def refinement_mode_examples(teachers,config,*,best_only=False):
    """Learn actual legal modes, including superior non-fixed MODE/REGION results."""
    from mrta_search.gnn_constructor import legal_mode_inputs
    grouped={}
    for item in teachers:grouped.setdefault(item['geometry_key'],[]).append(item)
    examples=[]
    for key,items in grouped.items():
        graph,candidates,features=legal_mode_inputs(items[0]['solution'].parents,config)
        best=min(item['cmax'] for item in items)
        winners=[item for item in items if item['cmax']<=best+1e-9*max(1.,best)]
        baseline={p.parent_id:p for p in fixed_split_patterns(items[0]['solution'].parents,config)}
        for parent in items[0]['solution'].parents:
            indices=[i for i,p in enumerate(candidates) if p.parent_id==parent.parent_id]
            if len(indices)<2:continue
            local=[candidates[i] for i in indices];target=torch.zeros(len(local))
            for item in winners if best_only else items:
                chosen=next(p for p in item['solution'].patterns if p.parent_id==parent.parent_id)
                if chosen not in local:raise ValueError('stored mode is outside current legal catalog')
                target[local.index(chosen)]+=1. if best_only else best/item['cmax']
            target/=target.sum()
            examples.append(dict(geometry_key=key,graph=graph,parent_ids=tuple(p.parent_id for p in local),
                                 features=features[indices],target=target,
                                 changed=any(local[i]!=baseline[parent.parent_id] for i in target.nonzero().flatten().tolist())))
    return examples


def refinement_calibrate_modes(model,teachers,config,*,epochs=20):
    """Fit only the mode head to best certified configurations, not a vote.

    Many weaker WHOLE solutions must not outvote a better split configuration.
    Balance changed/unchanged parent cases without modifying graph embeddings
    or the trained assignment/insertion parameters.
    """
    examples=refinement_mode_examples(teachers,config,best_only=True);groups={}
    for e in examples:groups.setdefault(e['geometry_key'],[]).append(e)
    positive=sum(e['changed'] for e in examples);negative=len(examples)-positive
    weights={True:len(examples)/(2*positive) if positive else 1.,
             False:len(examples)/(2*negative) if negative else 1.}
    model.eval();embeddings={}
    with torch.no_grad():
        for key,pool in groups.items():embeddings[key]=model.encode_static(pool[0]['graph']).detach()
    optimizer=torch.optim.AdamW(model.mode_head.parameters(),lr=.001)
    rng=random.Random(20261104);history=[]
    for epoch in range(epochs):
        total=count=0
        for key,pool in groups.items():
            pool=list(pool);rng.shuffle(pool)
            for offset in range(0,len(pool),16):
                batch=pool[offset:offset+16];optimizer.zero_grad(set_to_none=True);losses=[]
                for e in batch:
                    logits=model.mode_logits(e['graph'],embeddings[key],e['parent_ids'],e['features'])
                    losses.append(-weights[e['changed']]*(e['target']*torch.log_softmax(logits,0)).sum())
                loss=torch.stack(losses).mean()
                if not torch.isfinite(loss):raise ValueError('nonfinite mode calibration')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.mode_head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();total+=float(loss.detach())*len(batch);count+=len(batch)
        history.append(total/max(1,count))
    model.zero_grad(set_to_none=True)
    return dict(epochs=epochs,examples=len(examples),changed_parent_examples=positive,
                unchanged_parent_examples=negative,loss_history=history,
                objective='Best actual certified complete configuration; changed/unchanged balanced; mode head only.')


def refinement_prefix_examples(teachers,source_model,config):
    """Correct genuine student prefixes with B32-certified complete continuations."""
    from mrta_search.lns import (PartialSearchState,DestroyOperator,RepairOperator,
        repair_partial_state,joint_action_key)
    selected={}
    for item in teachers:selected.setdefault(item['geometry_key'],item)
    graphs={};examples=[];raw_steps=[];diagnostics=[]
    for index,(geometry_key,item) in enumerate(selected.items()):
        parents=item['solution'].parents
        geometry=joint_geometry(parents,fixed_split_patterns(parents,config),config)
        student=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),source_model,config)
        if student.solution is None:
            diagnostics.append(dict(instance=item['instance'],status='STUDENT_CONSTRUCTION_FAILED',reason=student.rejection_reason));continue
        _,student_steps,_=prepare_joint_teacher_steps(geometry,EMPTY,EMPTY,
            tuple(p.parent_id for p in parents),[asdict(a) for a in student.actions],config)
        parent_steps=[step for step in student_steps if step['parent_ids']]
        for fraction in (.25,.5,.75):
            prefix=parent_steps[min(len(parent_steps)-1,int(len(parents)*fraction))]
            routes=prefix['state_routes'];directions=prefix['state_directions'];pending=prefix['pending_parent_ids']
            bits={b:d for route,ds in zip(student.solution.routes,student.directions) for b,d in zip(route.block_ids,ds)}
            bits.update({b:d for route,ds in zip(routes,directions) for b,d in zip(route,ds)})
            current_bits=tuple(tuple(bits[b] for b in route.block_ids) for route in student.solution.routes)
            partial=PartialSearchState(parents,geometry.patterns,routes,pending,student.solution.revision)
            outcomes=[];diag=dict(instance=item['instance'],fraction=fraction,pending=len(pending),attempts=[],
                partial_routes=routes,partial_directions=directions,pending_parent_ids=pending,continuations=[])
            for coordination in (False,True):
                completion=repair_partial_state(student.solution,current_bits,partial,DestroyOperator.RANDOM_REMOVAL,
                    RepairOperator.GREEDY_REPAIR,config,preserve_patterns=True,direction_mode='JOINT',
                    max_insertion_evaluations=4096,insertion_limit=8,seed=20261104+index,
                    coordination_proxy=coordination,direction_versions=2 if coordination else 1)
                if completion.candidate is None:
                    diag['attempts'].append(dict(coordination=coordination,status='REPAIR_FAILED',reason=completion.rejection_reason));continue
                scheduled=pilot.oriented.schedule_explicit(completion.candidate.solution,completion.directions,config)
                valid=research_certify(scheduled,config)
                diag['attempts'].append(dict(coordination=coordination,status=scheduled.schedule.status.value,
                    certified=valid,cmax=scheduled.schedule.cmax if valid else None))
                if valid:outcomes.append((scheduled,completion))
            unique={}
            for scheduled,completion in outcomes:
                key=(tuple(r.block_ids for r in scheduled.solution.routes),scheduled.directions)
                unique.setdefault(key,(scheduled,completion))
            if unique:
                best=min(scheduled.schedule.cmax for scheduled,_ in unique.values());merged={}
                for scheduled,completion in unique.values():
                    diag['continuations'].append(dict(cmax=scheduled.schedule.cmax,
                        actions=[asdict(a) for a in completion.actions],
                        final_routes=[r.block_ids for r in scheduled.solution.routes],final_directions=scheduled.directions))
                    graph,steps,after=prepare_joint_teacher_steps(geometry,routes,directions,pending,
                        [asdict(a) for a in completion.actions],config)
                    if after.routes!=tuple(r.block_ids for r in scheduled.solution.routes) or after.directions!=scheduled.directions:
                        raise ValueError('student-prefix continuation replay mismatch')
                    graph_key=(tuple((p.parent_id,p.start,p.end) for p in parents),geometry.patterns)
                    graphs.setdefault(graph_key,graph);raw_steps.extend(steps)
                    weight=best/scheduled.schedule.cmax
                    for step in steps:
                        state=(step['state_routes'],step['state_directions'],step['pending_parent_ids'])
                        cases=[('insertion',tuple(joint_action_key(a) for a in step['candidate_actions']),step['target'])]
                        if len(step['parent_ids'])>1:cases.append(('parent',step['parent_ids'],step['parent_target']))
                        for kind,pool,target in cases:
                            key=(kind,state,pool)
                            if key not in merged:merged[key]=dict(kind=kind,graph_key=graph_key,step=step,target=torch.zeros(len(pool)),weights=[])
                            merged[key]['target'][target]+=weight;merged[key]['weights'].append(weight)
                for example in merged.values():
                    example['target']/=example['target'].sum();weights=example.pop('weights')
                    example['weight']=sum(weights)/len(weights);examples.append(example)
                diag['certified_continuations']=len(unique)
            else:diag['certified_continuations']=0
            diagnostics.append(diag)
        print('PREFIX TEACHERS',index+1,'/',len(selected),item['instance'],flush=True)
    return graphs,examples,raw_steps,diagnostics


def refinement_evaluate(model,previous_model,rows,config,training_keys):
    from mrta_search.gnn_constructor import select_gnn_patterns
    entries={}
    for row in rows:
        state=(row.get('details') or {}).get('final_state')
        if not state:continue
        solution,_,key,_,_=research_restore(state,config)
        if key in training_keys:raise ValueError('training/evaluation geometry overlap')
        entries.setdefault(key,dict(instance=row['instance'],parents=solution.parents))
    results=[]
    for entry in sorted(entries.values(),key=lambda e:e['instance']):
        parents=entry['parents']
        for method,policy,mode_policy in (('PREVIOUS',previous_model,False),('NEW_FIXED',model,False),('NEW_MODE',model,True),('RULE_B32_1',None,False)):
            started=time.perf_counter();calls=0;constructed=False;scheduled=None;failure=None
            if policy is None:
                scheduled,attempts=pilot.oriented.common_initial_beam(parents,config,max_reference_calls=1)
                calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
                constructed=scheduled is not None or calls>0
                if scheduled is None and attempts:failure=attempts[-1]['status']
            else:
                patterns=select_gnn_patterns(parents,policy,config) if mode_policy else fixed_split_patterns(parents,config)
                geometry=joint_geometry(parents,patterns,config)
                completion=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),policy,config)
                failure=completion.rejection_reason;constructed=completion.solution is not None
                if constructed:
                    scheduled=pilot.oriented.schedule_explicit(completion.solution,completion.directions,config);calls=1
            certified=scheduled is not None and research_certify(scheduled,config)
            loads=[];counts=[];chosen_modes={}
            if scheduled is not None:
                from mrta_reference.solution import block_map
                blocks=block_map(scheduled.solution,config)
                loads=[sum(config.process_time(blocks[b].length) for b in r.block_ids) for r in scheduled.solution.routes]
                counts=[len(r.block_ids) for r in scheduled.solution.routes]
                chosen_modes=dict(Counter(p.kind.value for p in scheduled.solution.patterns))
            status=(scheduled.schedule.status.value if scheduled is not None else
                    failure if policy is None and failure else 'CONSTRUCTION_FAILED')
            result=dict(instance=entry['instance'],method=method,status=status,
                certified=certified,cmax=scheduled.schedule.cmax if certified else None,runtime_s=time.perf_counter()-started,b32_calls=calls,
                constructed=constructed,counts=counts,processing_loads=loads,modes=chosen_modes,failure=failure)
            results.append(result);print('REFINEMENT EVAL',entry['instance'],method,result['status'],result['cmax'],flush=True)
    return results


def opportunity_ranking_loss(model,graphs,imitation,ranking):
    """0.5 IL + 1 pair rank + 0.1 existing soft load loss; no new head."""
    from torch.nn import functional as F
    encoded={};joint_scores={};il=[];load=[];rank=[]
    def embedding(key):
        if key not in encoded:encoded[key]=model.encode_static(graphs[key])
        return encoded[key]
    for example in imitation:
        key=example['graph_key'];step=example['step'];emb=embedding(key)
        if example['kind']=='parent':
            logits=model.parent_logits(graphs[key],emb,step['parent_ids'],step['parent_features'])
        else:logits=model.joint_action_logits(emb,step['indices'],step['features'])
        il.append(-example['weight']*(example['target']*torch.log_softmax(logits,0)).sum())
        if 'load_excess' in example:load.append((torch.softmax(logits,0)*example['load_excess']).sum())
    for state,better,worse in ranking:
        ident=id(state)
        if ident not in joint_scores:
            key=state['graph_key'];emb=embedding(key)
            parent=torch.log_softmax(model.parent_logits(graphs[key],emb,state['parent_ids'],state['parent_features']),0)
            scores=[];conditional={}
            for action in state['options']:
                p=action['parent_index']
                if p not in conditional:
                    conditional[p]=torch.log_softmax(model.joint_action_logits(emb,action['indices'],action['features']),0)
                scores.append(parent[p]+conditional[p][action['action_index']])
            joint_scores[ident]=scores
        scores=joint_scores[ident]
        rank.append(F.softplus(-(scores[better]-scores[worse])))
    zero=next(model.parameters()).new_zeros(())
    il_loss=torch.stack(il).mean() if il else zero
    rank_loss=torch.stack(rank).mean() if rank else zero
    load_loss=torch.stack(load).mean() if load else zero
    return .5*il_loss+rank_loss+.1*load_loss,dict(il=il_loss,rank=rank_loss,load=load_loss)


def expand_opportunity_model(source):
    """Append eight zero-weight inputs to the SAME two existing policy heads."""
    from mrta_search.gnn_constructor import ModeJointInsertionGNN
    if not isinstance(source,ModeJointInsertionGNN) or source.opportunity_features:
        raise ValueError('expand the existing ModeJointInsertionGNN once')
    model=ModeJointInsertionGNN(source.hidden_dim,source.heads,
                              normalize_inputs=source.normalize_inputs,opportunity_features=True)
    old=source.state_dict()
    with torch.no_grad():
        for name,target in model.state_dict().items():
            value=old[name]
            if target.shape==value.shape:target.copy_(value)
            elif name in ('parent_head.0.weight','insertion_head.0.weight'):
                target.zero_();target[:,:value.shape[1]].copy_(value)
            elif name in ('parent_mean','joint_mean','parent_scale','joint_scale'):
                target.fill_(1. if name.endswith('scale') else 0.)
                target[:len(value)].copy_(value)
            else:raise ValueError('unexpected architecture change: '+name)
    model.use_dynamic_context=False
    return model


def neural_single_model_portfolio(parents,model,config,*,seed=20261104,proposals=3,scheduled_cache=None,greedy_decode='greedy'):
    """Independent construction: one model, fixed patterns, no repair or fallback.

    Only the complete greedy/seeded-sample proposals are scheduled. Identical
    complete routes/directions share one B32 result, never a second certificate.
    Returned candidate objects support reporting greedy and trio from one run.
    """
    if proposals not in (1,2,3):raise ValueError('at most three proposals required')
    if greedy_decode not in ('greedy','joint_greedy','parent_joint'):raise ValueError('unknown first-proposal greedy decoder')
    started=time.perf_counter();attempts=[];candidates=[]
    seen={} if scheduled_cache is None else scheduled_cache
    external=set(seen);charged=set();credit=0.;best=None;selected=None
    for index,(decode,draw_seed) in enumerate(((greedy_decode,seed),('sample',seed),('sample',seed+1))[:proposals]):
        tick=time.perf_counter()
        geometry=joint_geometry(parents,fixed_split_patterns(parents,config),config)
        result=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),model,config,
                                          decode_mode=decode,seed=draw_seed)
        attempt=dict(proposal_index=index,decode=decode,seed=draw_seed,status='CONSTRUCTION_FAILED',
                     cmax=None,b32_calls=0,reason=result.rejection_reason,actions=[asdict(a) for a in result.actions])
        candidate=None
        if result.solution is not None:
            key=(result.solution.patterns,tuple(r.block_ids for r in result.solution.routes),result.directions)
            if key not in seen:
                clock=time.perf_counter()
                scheduled=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                seen[key]=(scheduled,time.perf_counter()-clock)
                attempt['b32_calls']=1
            candidate,scheduler_seconds=seen[key]
            reused_seconds=scheduler_seconds if key in external and key not in charged else 0.
            credit+=reused_seconds;charged.add(key)
            attempt['reused_b32_seconds']=reused_seconds
            attempt.update(status=candidate.schedule.status.value,
                cmax=candidate.schedule.cmax if candidate.schedule.status is ScheduleStatus.FEASIBLE else None,
                solution=asdict(result.solution),directions=result.directions)
            if candidate.schedule.status is ScheduleStatus.FEASIBLE and (best is None or candidate.schedule.cmax<best.schedule.cmax):
                best=candidate;selected=index
        attempt['seconds']=time.perf_counter()-tick
        attempt['elapsed']=time.perf_counter()-started+credit
        attempts.append(attempt);candidates.append(candidate)
    return best,dict(attempts=attempts,candidates=candidates,selected_proposal=selected,
                     b32_calls=sum(a['b32_calls'] for a in attempts),runtime_s=time.perf_counter()-started+credit,
                     actual_runtime_s=time.perf_counter()-started,reused_b32_seconds=credit)


def neural_initial_portfolio(parents,previous_model,model,config,*,old_sampling=False,seed=20261104,
                             reference_calls=3,use_learned_modes=False,certify=True,deadline=None,
                             prune_cmax_lb=False,parent_selection_mode='learned',action_selection_mode='gnn',
                             candidate_observer=None):
    """Bounded neural proposals; return the smallest timely feasible Cmax.

    The robust old policy is retained explicitly. No rule solution or teacher
    route is supplied at inference. old_sampling is the equal-call comparator.
    Certification remains enabled by default for historical callers. A deadline
    is absolute perf_counter time; late proposals never replace a timely best.
    selected_proposal identifies the actual winning zero-based menu entry.
    The optional observer receives each constructed result and its recorded
    disposition, including duplicates and safely pruned proposals, without B32.
    """
    from mrta_search.gnn_constructor import select_gnn_patterns
    from mrta_search.lns import oriented_route_state,oriented_route_lower_bound
    from mrta_reference.solution import block_map
    if reference_calls not in (2,3):raise ValueError('portfolio supports two or three proposals')
    started=time.perf_counter();best=None;attempts=[];seen=set();selected_seconds=None;selected_proposal=None
    policies=[(previous_model,False,'greedy',seed)]
    if reference_calls==3:policies.append((previous_model,False,'sample',seed))
    policies.append((previous_model,False,'sample',seed+reference_calls-2) if old_sampling else
                    (model,use_learned_modes,'greedy',seed))
    for proposal_index,(policy,use_modes,decode,proposal_seed) in enumerate(policies):
        if deadline is not None and time.perf_counter()>=deadline:break
        patterns=select_gnn_patterns(parents,policy,config) if use_modes else fixed_split_patterns(parents,config)
        geometry=joint_geometry(parents,patterns,config)
        options=dict(decode_mode=decode,seed=proposal_seed)
        if parent_selection_mode!='learned':options['parent_selection_mode']=parent_selection_mode
        if action_selection_mode!='gnn':options['action_selection_mode']=action_selection_mode
        result=construct_or_repair_with_gnn(geometry,EMPTY,EMPTY,tuple(p.parent_id for p in parents),policy,config,**options)
        attempt=dict(policy='previous' if policy is previous_model else 'refinement',decode=decode,
                     use_learned_modes=use_modes,seed=proposal_seed,
                     status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,reason=result.rejection_reason,
                     admitted=False)
        attempt.update(parent_selection_mode=parent_selection_mode,action_selection_mode=action_selection_mode)
        attempt['proposal_index']=proposal_index
        if deadline is not None and time.perf_counter()>=deadline:
            attempt['status']='CONSTRUCTION_FINISHED_AFTER_DEADLINE'
        elif result.solution is not None:
            key=(result.solution.patterns,tuple(r.block_ids for r in result.solution.routes),result.directions)
            if key in seen:attempt['status']='DUPLICATE'
            else:
                seen.add(key)
                if prune_cmax_lb and best is not None:
                    state=getattr(result,'oriented_state',None)
                    if state is None:
                        state=oriented_route_state(tuple(r.block_ids for r in result.solution.routes),
                                                   result.directions,block_map(result.solution,config),config)
                    bound=oriented_route_lower_bound(state)
                    attempt['lower_bound']=bound
                    if bound>=best.schedule.cmax:
                        attempt.update(status='PRUNED_BY_CMAX_LB',elapsed=time.perf_counter()-started,
                                       late=deadline is not None and time.perf_counter()>deadline)
                        attempts.append(attempt)
                        if candidate_observer is not None:candidate_observer(proposal_index,result,attempt)
                        continue
                scheduled=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                attempt.update(status=scheduled.schedule.status.value,b32_calls=1)
                timely=deadline is None or time.perf_counter()<=deadline
                valid=(scheduled.schedule.status is ScheduleStatus.FEASIBLE and timely and
                       (not certify or research_certify(scheduled,config)))
                completed=time.perf_counter()
                if valid and (deadline is None or completed<=deadline):
                    attempt['admitted']=True
                    attempt['cmax']=scheduled.schedule.cmax
                    if best is None or scheduled.schedule.cmax<best.schedule.cmax:
                        best=scheduled;selected_seconds=completed-started;selected_proposal=proposal_index
        attempt['elapsed']=time.perf_counter()-started
        attempt['late']=deadline is not None and time.perf_counter()>deadline
        attempts.append(attempt)
        if candidate_observer is not None:candidate_observer(proposal_index,result,attempt)
    return best,dict(attempts=attempts,b32_calls=sum(a['b32_calls'] for a in attempts),
                     runtime_s=time.perf_counter()-started,selected_seconds=selected_seconds,
                     selected_proposal=selected_proposal)


def refinement_portfolio_evaluate(model,previous,rows,config,*,reference_calls=3,use_learned_modes=False):
    entries={}
    for row in rows:
        state=(row.get('details') or {}).get('final_state')
        if state and row['instance'] not in entries:entries[row['instance']]=research_restore(state,config)[0].parents
    results=[]
    for name,parents in sorted(entries.items()):
        for method,old_sampling in ((f'PREVIOUS_MULTI{reference_calls}',True),(f'NEW_PORTFOLIO{reference_calls}',False)):
            best,info=neural_initial_portfolio(parents,previous,model,config,old_sampling=old_sampling,
                                             reference_calls=reference_calls,use_learned_modes=use_learned_modes)
            row=dict(instance=name,method=method,status='NO_CERTIFIED_INITIAL' if best is None else 'FEASIBLE',
                     cmax=None if best is None else best.schedule.cmax,**info)
            if best is not None:
                row['selected_state']=dict(solution=json.loads(json.dumps(asdict(best.solution))),
                                          directions=best.directions,cmax=best.schedule.cmax)
            results.append(row);print('PORTFOLIO EVAL',name,method,row['status'],row['cmax'],flush=True)
    return results


def run_refinement(args):
    from mrta_search.gnn_constructor import ModeJointInsertionGNN
    if args.research_epochs<1:raise ValueError('refinement epochs must be positive')
    torch.set_num_threads(1);torch.manual_seed(20261104);config=CONFIG
    checkpoint_path=ROOT/'models/gnn_coordination_constructor_05.pt'
    previous,old_checkpoint=load_model(checkpoint_path)
    rows=research_rows(ROOT/'results/parent_mode_joint_reconstruction_05.csv')+research_rows(ROOT/'results/final_parent_mode_joint_reconstruction_05.csv')
    train_rows,holdout_rows,split=refinement_split(rows)
    teachers,coverage=research_teachers(train_rows,config,args.research_train_limit,deployment_only=False)
    graphs,examples,steps,replays=research_prepare(teachers,config)
    modes=refinement_mode_examples(teachers,config)
    prefix_graphs,prefix_examples,prefix_steps,prefix_diagnostics=refinement_prefix_examples(teachers,previous,config)
    graphs.update(prefix_graphs);examples.extend(prefix_examples);steps.extend(prefix_steps)
    bounds={}
    for item in teachers:bounds[item['geometry_key']]=min(bounds.get(item['geometry_key'],float('inf')),item['cmax'])
    model=ModeJointInsertionGNN(64,4,normalize_inputs=True)
    model,training=research_train(graphs,examples,steps,args,model=model,mode_examples=modes,geometry_bounds=bounds)
    training['mode_calibration']=refinement_calibrate_modes(model,teachers,config)
    evaluation=refinement_evaluate(model,previous,holdout_rows,config,set(bounds))
    portfolio=refinement_portfolio_evaluate(model,previous,holdout_rows,config)
    result=dict(split=split,coverage=coverage,teacher_methods=dict(Counter(m for item in teachers for m in item['teachers'])),
        replays=replays,prefix_diagnostics=prefix_diagnostics,prefix_examples=len(prefix_examples),training=training,evaluation=evaluation,
        portfolio_evaluation=portfolio,
        scope='All retained algorithms including REGION and actual legal nonfixed modes; consumed-development study only.')
    if args.refine_update_checkpoint:
        target=Path(args.refine_update_checkpoint).resolve()
        if target!=checkpoint_path.resolve() or not target.is_file():
            raise ValueError('refinement updates the existing approved checkpoint only; no new file is created')
        old_checkpoint['refinement_candidate']=dict(model_type='ModeJointInsertionGNN',state_dict=model.state_dict(),hidden_dim=64,heads=4,
            normalize_inputs=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),refinement=result)
        save_neural_checkpoint(old_checkpoint,target)
    print('REFINEMENT_RESULT '+json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
    return model,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','data','train','eval','research','refine'),default='all')
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--pilot-size',type=int,default=10)
    parser.add_argument('--research-epochs',type=int,default=20)
    parser.add_argument('--research-train-limit',type=int)
    parser.add_argument('--research-eval-limit',type=int,default=3)
    parser.add_argument('--research-save',help='Explicitly requested new checkpoint path; omitted means no writes.')
    parser.add_argument('--refine-update-checkpoint',help='Update the already approved existing checkpoint; preserve previous weights.')
    args=parser.parse_args()
    if args.phase=='research':return run_research(args)
    if args.phase=='refine':return run_refinement(args)
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
