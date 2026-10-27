"""TRAIN-only supervised joint insertion, followed by paired DEV B32 runs."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import torch
from torch import nn
from mrta_ranker import dataset
from mrta_reference.model import ParentWeld,SplitPattern,SplitKind,Rail,Route,ScientificConfig,ScheduleStatus
from mrta_reference.solution import canonicalize,block_map
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_search.lns import BottleneckSelection,repair_partial_state_oriented,select_bottleneck_parents
from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn,prepare_joint_teacher_steps
import run_joint_oriented_repair_pilot as oriented

DATA = ROOT/'data/development/gnn_joint_repair_train.jsonl'
MODEL = ROOT/'models/gnn_joint_repair_v1.pt'
CSV = ROOT/'results/gnn_joint_repair_pilot.csv'
REPORT = ROOT/'docs/GNN_JOINT_REPAIR_PILOT.md'
CONFIG = ScientificConfig()
SELECTORS = (BottleneckSelection.RANDOM,BottleneckSelection.TRAVEL,BottleneckSelection.LOAD)


def entries():
    all_entries = dataset.representatives_v05()
    train = [e for e in all_entries if e['split']=='MLP_TRAIN']
    dev = [e for e in all_entries if e['split']=='MLP_DEV']
    if len(train)!=17 or len(dev)!=6:
        raise ValueError('expected the existing 17 TRAIN / 6 DEV workbook split')
    # First three cover the existing small/medium/large representative scales.
    first = []
    for target in (35,55,85):
        first.append(min((e for e in train if e not in first),key=lambda e:(abs(e['N']-target),e['instance_id'])))
    return first+[e for e in train if e not in first],dev


def partial_directions(initial,partial):
    bits = {b:d for route,ds in zip(initial.solution.routes,initial.directions) for b,d in zip(route.block_ids,ds)}
    return tuple(tuple(bits[b] for b in route) for route in partial.routes)


def make_record(entry,initial,partial,result,scheduled,source,seed):
    if entry['split']!='MLP_TRAIN':
        raise ValueError('only TRAIN may supply supervision')
    s = initial.solution
    return dict(entry={k:entry[k] for k in ('instance_id','relative_path','sheet_name','split','N')},
                parents=[dict(parent_id=p.parent_id,start=p.start,end=p.end) for p in s.parents],
                patterns=[dict(parent_id=p.parent_id,kind=p.kind.value,t=p.t,point_id=p.point_id,
                               mandatory=p.mandatory,rail=p.rail.value if p.rail else None) for p in s.patterns],
                reference_routes=[r.block_ids for r in s.routes],partial_routes=partial.routes,
                partial_directions=partial_directions(initial,partial),pending_parents=partial.removed_parent_ids,
                actions=[asdict(a) for a in result.actions],final_routes=[r.block_ids for r in result.candidate.solution.routes],
                final_directions=result.directions,cmax=scheduled.schedule.cmax,original_cmax=initial.schedule.cmax,
                quality='STRICT_IMPROVEMENT' if scheduled.schedule.cmax < initial.schedule.cmax-1e-9*max(1,initial.schedule.cmax)
                else 'FEASIBLE_NONIMPROVING',teacher=source,seed=seed)


def record_reference(record):
    parents = tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in record['parents'])
    patterns = tuple(SplitPattern(p['parent_id'],SplitKind(p['kind']),p['t'],p['point_id'],p['mandatory'],
                                  Rail(p['rail']) if p['rail'] else None) for p in record['patterns'])
    return canonicalize(parents,patterns,tuple(Route(r,tuple(route)) for r,route in enumerate(record['reference_routes'])),
                        CONFIG,scope=FORMAL_SCOPE_V2)


def prepare_record(record):
    if record['entry']['split']!='MLP_TRAIN':
        raise ValueError('DEV supervision is forbidden')
    reference = record_reference(record)
    graph,steps,state = prepare_joint_teacher_steps(reference,record['partial_routes'],record['partial_directions'],
                                                    record['pending_parents'],record['actions'],CONFIG,insertion_limit=8)
    if state.routes!=tuple(tuple(r) for r in record['final_routes']) or state.directions!=tuple(tuple(ds) for ds in record['final_directions']):
        raise ValueError('teacher replay did not recover the full directed result')
    normalized = canonicalize(reference.parents,reference.patterns,tuple(Route(r,state.routes[r]) for r in range(4)),
                              CONFIG,scope=FORMAL_SCOPE_V2)
    if normalized.patterns!=reference.patterns or tuple(r.block_ids for r in normalized.routes)!=state.routes:
        raise ValueError('teacher replay changed fixed patterns')
    return graph,steps


def teacher_instance(entry,args,index):
    started = time.perf_counter()
    initial,init_attempts = oriented.common_initial_beam(dataset.load_parents(entry),CONFIG,seed=args.seed)
    info = dict(instance_id=entry['instance_id'],sheet=entry['sheet_name'],N=entry['N'],initial_cmax=None,
                init_calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in init_attempts),
                init_statuses=dict(Counter(a['status'] for a in init_attempts)),teacher_calls=0,teacher_success=0,
                teacher_deadlock=0,repair_failures=0,samples=0,strict=0,supplemental=0,runtime_s=0)
    if initial is None:
        info['runtime_s']=time.perf_counter()-started
        return [],info
    info['initial_cmax']=initial.schedule.cmax
    unique = {}
    for q in (4,8):
        for selection in SELECTORS:
            for attempt in range(args.train_attempts):
                seed=args.seed+index*10000+q*100+attempt
                partial=select_bottleneck_parents(initial.solution,initial.directions,initial.schedule,selection,
                                                  q=q,seed=seed,config=CONFIG)
                successful=[]
                for source in ('OLD','NEW'):
                    result=repair_partial_state_oriented(initial.solution,initial.directions,partial,CONFIG,seed=seed,
                                                        max_insertion_evaluations=4096,insertion_limit=8,
                                                        coordination_proxy=source=='NEW',direction_versions=2 if source=='NEW' else 1)
                    if result.candidate is None:
                        info['repair_failures']+=1
                        continue
                    scheduled=oriented.schedule_explicit(result.candidate.solution,result.directions,CONFIG)
                    info['teacher_calls']+=1
                    if scheduled.schedule.status is ScheduleStatus.FEASIBLE:
                        info['teacher_success']+=1
                        successful.append((scheduled.schedule.cmax,result,scheduled,source))
                    elif scheduled.schedule.status is ScheduleStatus.DEADLOCK:
                        info['teacher_deadlock']+=1
                if not successful:
                    continue
                _,result,scheduled,source=min(successful,key=lambda item:item[0])
                record=make_record(entry,initial,partial,result,scheduled,source,seed)
                record['selection']=selection.value
                # Validation uses precisely the student's action generator and
                # fails loudly on a missing label; no oracle label is injected.
                prepare_record(record)
                key=(tuple(tuple(r) for r in record['final_routes']),tuple(tuple(ds) for ds in record['final_directions']))
                unique.setdefault(key,record)
    strict=[r for r in unique.values() if r['quality']=='STRICT_IMPROVEMENT']
    supplemental=sorted((r for r in unique.values() if r['quality']!='STRICT_IMPROVEMENT'),key=lambda r:r['cmax'])[:2]
    records=strict+supplemental
    info.update(samples=len(records),strict=len(strict),supplemental=len(supplemental),runtime_s=time.perf_counter()-started)
    return records,info


def train_model(records,args,epochs):
    allowed={e['instance_id'] for e in entries()[0]}
    grouped=defaultdict(list)
    graphs={}
    started=time.perf_counter()
    for record in records:
        if record['entry']['instance_id'] not in allowed or record['entry']['split']!='MLP_TRAIN':
            raise ValueError('sample is outside the existing TRAIN whitelist')
        graph,steps=prepare_record(record)
        name=record['entry']['instance_id']
        graphs.setdefault(name,graph)
        grouped[name].extend(steps)
    if not grouped:
        raise ValueError('no replayable TRAIN samples')
    print('TRAIN prepared',len(records),'trajectories',sum(map(len,grouped.values())),'actions',len(grouped),'geometries',flush=True)
    torch.manual_seed(args.seed)
    model=JointInsertionGNN(64,4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    history=[]
    for epoch in range(epochs):
        model.train()
        names=list(grouped)
        random.Random(args.seed+epoch).shuffle(names)
        total_loss=0.0
        count=correct=parent_count=parent_correct=0
        for name in names:
            optimizer.zero_grad()
            graph=graphs[name]
            embeddings=model.encode_static(graph)
            losses=[]
            for step in grouped[name]:
                logits=model.joint_action_logits(embeddings,step['indices'],step['features'])
                loss=nn.functional.cross_entropy(logits.unsqueeze(0),torch.tensor([step['target']]))
                correct+=int(int(logits.argmax())==step['target'])
                count+=1
                if len(step['parent_ids'])>1:
                    parent_logits=model.parent_logits(graph,embeddings,step['parent_ids'],step['parent_features'])
                    loss=loss+nn.functional.cross_entropy(parent_logits.unsqueeze(0),torch.tensor([step['parent_target']]))
                    parent_correct+=int(int(parent_logits.argmax())==step['parent_target'])
                    parent_count+=1
                losses.append(loss)
            batch_loss=torch.stack(losses).mean()
            batch_loss.backward()
            optimizer.step()
            total_loss+=float(batch_loss.detach())*len(losses)
        item=dict(epoch=epoch+1,loss=total_loss/count,insertion_accuracy=correct/count,
                  parent_accuracy=parent_correct/max(1,parent_count),actions=count)
        history.append(item)
        print('EPOCH',epoch+1,json.dumps(item),flush=True)
    return model,dict(history=history,trajectories=len(records),steps=sum(map(len,grouped.values())),
                     geometries=len(grouped),strict=sum(r['quality']=='STRICT_IMPROVEMENT' for r in records),
                     supplemental=sum(r['quality']!='STRICT_IMPROVEMENT' for r in records),runtime_s=time.perf_counter()-started)


def smoke_neural(model,records):
    results=[]
    selected={}
    for record in records:
        selected.setdefault(record['entry']['instance_id'],record)
    for record in list(selected.values())[:3]:
        result=construct_or_repair_with_gnn(record_reference(record),record['partial_routes'],record['partial_directions'],
                                          record['pending_parents'],model,CONFIG)
        status='CONSTRUCTION_FAILED'
        cmax=None
        if result.solution is not None:
            scheduled=oriented.schedule_explicit(result.solution,result.directions,CONFIG)
            status=scheduled.schedule.status.value
            if scheduled.schedule.status is ScheduleStatus.FEASIBLE:
                cmax=scheduled.schedule.cmax
        item=dict(instance=record['entry']['sheet_name'],status=status,cmax=cmax,actions=len(result.actions),
                  failure=result.rejection_reason)
        results.append(item)
        print('TRAIN SMOKE GNN+B32',json.dumps(item),flush=True)
    return results


def generate_data(args):
    records,infos,smoke=[],[],[]
    train,_=entries()
    with DATA.open('w',encoding='utf-8') as f:
        for index,entry in enumerate(train[:args.train_limit]):
            samples,info=teacher_instance(entry,args,index)
            records.extend(samples);infos.append(info)
            for record in samples:
                f.write(json.dumps(record,ensure_ascii=False)+'\n')
            f.flush()
            print('DATA',index+1,json.dumps(info,ensure_ascii=False),flush=True)
            if index==2 and records:
                print('First three TRAIN geometries: smoke supervision and actual neural completion.',flush=True)
                small_model,_=train_model(records,args,1)
                smoke=smoke_neural(small_model,records)
    return records,infos,smoke


def eval_group(entry,initial,partials,method,selection,q,model,args):
    row=dict(instance=entry['sheet_name'],method=method,selection=selection.value,q=q,
             initial_cmax=None if initial is None else initial.schedule.cmax,attempts=len(partials),constructed=0,
             b32_calls=0,b32_success=0,improved=0,best_cmax=None if initial is None else initial.schedule.cmax,
             best_candidate_cmax=None,runtime_s=0.0,construction_failed=0,deadlock=0,infeasible=0,
             neural_actions=0,failures='{}')
    started=time.perf_counter()
    reasons=Counter();examples=[]
    for attempt,partial in enumerate(partials):
        seed=args.seed+oriented.NAMES.index(entry['sheet_name'])*10000+q*100+attempt
        if method=='GNN-JointRepair':
            result=construct_or_repair_with_gnn(initial.solution,partial.routes,partial_directions(initial,partial),
                                              partial.removed_parent_ids,model,CONFIG,insertion_limit=8)
            row['neural_actions']+=len(result.actions)
            solution=result.solution
        else:
            result=repair_partial_state_oriented(initial.solution,initial.directions,partial,CONFIG,seed=seed,
                                                 coordination_proxy=method=='NEW',direction_versions=2 if method=='NEW' else 1,
                                                 insertion_limit=8,max_insertion_evaluations=4096)
            solution=None if result.candidate is None else result.candidate.solution
        if solution is None:
            row['construction_failed']+=1
            reasons[result.rejection_reason]+=1
            continue
        row['constructed']+=1
        scheduled=oriented.schedule_explicit(solution,result.directions,CONFIG)
        row['b32_calls']+=1
        status=scheduled.schedule.status
        if status is ScheduleStatus.FEASIBLE:
            row['b32_success']+=1
            cmax=scheduled.schedule.cmax
            if row['best_candidate_cmax'] is None or cmax<row['best_candidate_cmax']:
                row['best_candidate_cmax']=cmax
            if cmax<initial.schedule.cmax-1e-9*max(1,initial.schedule.cmax):
                row['improved']+=1
                row['best_cmax']=min(row['best_cmax'],cmax)
        elif status is ScheduleStatus.DEADLOCK:
            row['deadlock']+=1
        else:
            row['infeasible']+=1
        if method=='GNN-JointRepair' and (not examples or status is ScheduleStatus.FEASIBLE and scheduled.schedule.cmax<initial.schedule.cmax):
            examples.append(dict(instance=entry['sheet_name'],q=q,selection=selection.value,attempt=attempt,status=status.value,
                                 changes=oriented.summary(initial,scheduled,CONFIG),actions=[asdict(a) for a in result.actions]))
    row['failures']=json.dumps(reasons,sort_keys=True)
    row['runtime_s']=time.perf_counter()-started
    print('DEV',entry['sheet_name'],method,selection.value,q,'success',row['b32_success'],'improved',row['improved'],
          'constructfail',row['construction_failed'],'best',row['best_cmax'],'time',round(row['runtime_s'],3),flush=True)
    return row,examples


def save_results(rows,checkpoint,initials,examples,args):
    with CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats=checkpoint['training']
    lines=['# GNN 联合部分解补全首轮实验','',
           '两层 SparseGATLayer，hidden=64，heads=4；父焊缝头与条件插入头。真实动作包括 (b,r,k,d) 及同一步邻块翻转；父焊缝的两个子块依次放置。',
           '静态图复用焊缝/机器人节点与邻近边，并连接同父子块；动态特征仅来自当前部分路线、端点方向、负载、P+M、插入前后邻块及同轨包络。没有未来 Cmax/WAIT 输入。',
           '合法性通过相同的执行器候选集合落实：机器人资格、X 固定归属、首任务条件、固定模式防合并；每处最多 8 个位置，完整枚举最多 8 个局部方向配置。'
           '训练和推理候选完全一致，教师标签缺失即报错，不注入 oracle 动作。',
           f"只使用 TRAIN 监督；DEV 无梯度更新。AdamW lr=1e-3，seed={checkpoint['seed']}，最终固定 {checkpoint['epochs']} epoch，无 DEV 选 checkpoint。首次三 TRAIN 的 1 epoch 为管线烟测，其权重丢弃后重新初始化进行最终训练。",'',
           '## TRAIN 数据与训练','',
           f"样本 {stats['trajectories']} 条：严格改善 {stats['strict']}、另列非改善可调度样本 {stats['supplemental']}；"
           f"覆盖 {stats['geometries']} 个 TRAIN 几何，教师动作 {stats['steps']}；训练及预处理 {stats['runtime_s']:.3f}s。",
           '每个 TRAIN 的教师尝试为 RANDOM/TRAVEL/LOAD × q4/q8 × '+str(args.train_attempts)+' 次；每个部分解都调用 OLD/NEW，以成功候选中 Cmax 最小者为教师。'
           '按完整路线及方向去重，保留全部严格改善和每几何最多两条较优非改善样本；两类质量标记分开保存。',
           '数据文件保存原始几何和固定模式、部分路线/方向、未放父焊缝、动作及翻转、最终路线/方向、教师 Cmax 与原 Cmax。全部逐步重放并匹配最终状态。','',
           '|TRAIN 工作簿::sheet|N|初解 Cmax|教师 B32/成功|严格样本|补充样本|秒|','|---|---:|---:|---:|---:|---:|---:|']
    for info in checkpoint.get('generation',[]):
        c='—' if info['initial_cmax'] is None else f"{info['initial_cmax']:.3f}"
        lines.append(f"|{info['instance_id']}|{info['N']}|{c}|{info['teacher_calls']}/{info['teacher_success']}|{info['strict']}|{info['supplemental']}|{info['runtime_s']:.2f}|")
    lines+=['',f"首/末 epoch 损失 {stats['history'][0]['loss']:.5f}→{stats['history'][-1]['loss']:.5f}；"
            f"末 epoch 教师强制插入准确率 {100*stats['history'][-1]['insertion_accuracy']:.2f}%，父选择准确率 {100*stats['history'][-1]['parent_accuracy']:.2f}%。"
            '这些不能代替完整补全和 B32 的真实结果。','',
            '三 TRAIN 烟测：'+json.dumps(checkpoint.get('smoke',[]),ensure_ascii=False),'',
            '## DEV 完整补全与 B32','',
            '六个历史已消耗 DEV，仅用于验证。共用现有 Beam 初解、固定模式及 RANDOM/TRAVEL/LOAD 移除集合；q4/q8、每组8次，seed=20261027。'
            '所有方案从同一初解出发，不把某方法的改善初解传给另一方法。OLD/NEW 评分和预算4096不改；GNN 没有启发式回退。'
            '每次完整候选只调用一次 B32，显式传入 R0—R3 全部方向，包括空路线。无独立 DP/refinement、C2/C3/C4、certifier 或精确解。',
            '运行时间包含模型编码、动态候选、神经打分、插入和 B32。构造失败、DEADLOCK 都保留计数；仅 FEASIBLE 可参与 Cmax 比较。'
            '表中最终最佳包含共同初解；CSV 另列候选自身最佳 Cmax，未成功调度时为空。','',
            '|实例|方法|选区|q|初始 Cmax|尝试|构造失败|B32 成功|改善|最终最佳 Cmax|秒|',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        fnum=lambda v:'—' if v is None else f'{v:.3f}'
        lines.append(f"|{row['instance']}|{row['method']}|{row['selection']}|{row['q']}|{fnum(row['initial_cmax'])}|{row['attempts']}|"
                     f"{row['construction_failed']}|{row['b32_success']}|{row['improved']}|{fnum(row['best_cmax'])}|{row['runtime_s']:.2f}|")
    lines+=['','## 结果与典型案例','']
    for method in ('OLD','NEW','GNN-JointRepair'):
        g=[r for r in rows if r['method']==method]
        lines.append(method+'：'+json.dumps({k:sum(r[k] for r in g) for k in ('attempts','constructed','b32_calls','b32_success','improved','construction_failed','deadlock','neural_actions','runtime_s')},ensure_ascii=False))
    selected={c['instance']:c for c in examples}
    for case in [selected[name] for name in ('g21_w035','g33_w055','g45_w082') if name in selected]:
        c=case['changes']
        lines+=['',f"- {case['instance']} {case['selection']} q{case['q']} attempt={case['attempt']}：{case['status']}；"
                f"Cmax {c['cmax_before']}→{c['cmax_after']}；归属变化 {c['transfers']}；重排路线 {c['reordered_routes']}；方向变化 {c['flips']}。",
                f"  加工负载 {c['process_before']}→{c['process_after']}；空走 {c['travel_before']:.3f}→{c['travel_after']:.3f}s；"
                f"WAIT {c['wait_before']:.3f}→"+("—" if c['wait_after'] is None else f"{c['wait_after']:.3f}s")+'（失败调度不报告完整 WAIT）。',
                '  神经动作：'+'; '.join(f"{oriented.short(a['block_id'])}→R{a['robot']}@{a['position']}/{a['direction']}，邻块翻转{a['neighbour_changes']}" for a in case['actions'])+'。']
    lines+=['','## 判断','',
            '网络确实逐步选择父焊缝及合法机器人/位置/局部方向配置，执行后更新路线和方向；第二子块沿同一父任务处理，没有全局先分配后排序。'
            '同一解码接口支持四条空路线，但本轮只训练部分解补全，不声称已学到高质量空路线初解。',
            '结合完整 DEV 结果判断数据覆盖、教师动作拟合和协调失败；不得只看 loss。此版本不追加网络或启发式模块，不用 PPO 补救。',
            '进一步 Cmax 反馈训练可复用相同动作/执行器，但需要先确认本轮失败类型；本轮仅监督学习。','',
            '## 复现','','```powershell',
            "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_joint_repair_pilot.py --phase all --seed 20261027 --epochs 20 --train-attempts 2",
            '```','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def evaluate(model,checkpoint,args):
    _,dev=entries()
    by_name={e['sheet_name']:e for e in dev}
    order=['g33_w055','g45_w082']+[n for n in oriented.NAMES if n not in ('g33_w055','g45_w082')]
    rows,initials,examples=[],{},[]
    for name in order:
        if name not in args.dev_instances:
            continue
        entry=by_name[name]
        initial,attempts=oriented.common_initial_beam(dataset.load_parents(entry),CONFIG,seed=args.seed)
        initials[name]=None if initial is None else initial.schedule.cmax
        print('DEV INITIAL',name,initials[name],flush=True)
        for q in (4,8):
            for selection in SELECTORS:
                partials=[]
                if initial is not None:
                    for attempt in range(args.eval_attempts):
                        seed=args.seed+oriented.NAMES.index(name)*10000+q*100+attempt
                        partials.append(select_bottleneck_parents(initial.solution,initial.directions,initial.schedule,
                                                                 selection,q=q,seed=seed,config=CONFIG))
                for method in ('OLD','NEW','GNN-JointRepair'):
                    row,cases=eval_group(entry,initial,partials,method,selection,q,model,args)
                    rows.append(row)
                    for case in cases:
                        def case_rank(c):
                            gain=(c['status']=='FEASIBLE' and c['changes']['cmax_after']<c['changes']['cmax_before']-1e-6)
                            return (0 if gain else 1 if c['status']=='DEADLOCK' else 2,
                                    c['changes']['cmax_after'] if gain else 0)
                        previous=next((c for c in examples if c['instance']==name),None)
                        if previous is None:
                            examples.append(case)
                        elif case_rank(case)<case_rank(previous):
                            examples[examples.index(previous)]=case
        save_results(rows,checkpoint,initials,examples,args)
    print('Saved',CSV,REPORT,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','data','train','eval'),default='all')
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--train-attempts',type=int,default=2)
    parser.add_argument('--train-limit',type=int,default=17)
    parser.add_argument('--eval-attempts',type=int,default=8)
    parser.add_argument('--dev-instances',nargs='+',choices=oriented.NAMES,default=list(oriented.NAMES))
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed);random.seed(args.seed)
    records=None;generation=[];smoke=[]
    if args.phase in ('all','data'):
        records,generation,smoke=generate_data(args)
        print('DATA FINISHED',len(records),'trajectories',flush=True)
        if args.phase=='data':
            return
    if args.phase in ('all','train'):
        if records is None:
            records=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
        model,stats=train_model(records,args,args.epochs)
        checkpoint=dict(state_dict=model.state_dict(),hidden_dim=64,heads=4,seed=args.seed,epochs=args.epochs,
                        insertion_limit=8,training=stats,generation=generation,smoke=smoke)
        torch.save(checkpoint,MODEL)
        print('MODEL SAVED',MODEL,flush=True)
        if args.phase=='train':
            return
    else:
        checkpoint=torch.load(MODEL,map_location='cpu',weights_only=True)
        model=JointInsertionGNN(checkpoint['hidden_dim'],checkpoint['heads'])
        model.load_state_dict(checkpoint['state_dict'])
    evaluate(model,checkpoint,args)


if __name__=='__main__':
    main()
