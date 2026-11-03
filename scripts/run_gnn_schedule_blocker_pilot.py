"""Fixed-initial GNN-V2 removal pilot on the 20 consumed coordination layouts."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import torch
from mrta_data.ppo_instances import load_ppo_platform_instance,to_parent_welds
from mrta_reference.model import OperationKind,ScheduleStatus,ScientificConfig
from mrta_reference.scheduler import selected_schedule_wait_blockers
from mrta_reference.solution import block_map
from mrta_search.gnn_constructor import JointInsertionGNN,construct_or_repair_with_gnn
from mrta_search.lns import BottleneckSelection,select_bottleneck_parents,select_schedule_blocker_parents
import run_joint_oriented_repair_pilot as oriented
from run_gnn_joint_repair_pilot import partial_directions

CONFIG=ScientificConfig()
SEED=20261103
METHODS=('RANDOM','LOAD','TRAVEL','WAIT_ANCHORED','SCHEDULE_BLOCKER')
CSV=ROOT/'results/gnn_schedule_blocker_pilot.csv'
REPORT=ROOT/'docs/GNN_SCHEDULE_BLOCKER_PILOT.md'


def layouts():
    """Use the existing report's workbook/sheet mapping, not a new manifest."""
    entries=[]
    for line in (ROOT/'docs/FINAL_COORDINATION_COMPARISON.md').read_text(encoding='utf-8').splitlines():
        fields=[v.strip() for v in line.strip('|').split('|')]
        if len(fields)==6 and fields[-1].endswith('.xlsx'):
            name,_,count,_,sheet,filename=fields
            entries.append(dict(instance=name,tier=name.rsplit('_',1)[0],N=int(count),sheet=sheet,filename=filename))
    if len(entries)!=20:raise ValueError('expected the 20 existing final-coordination layouts')
    return entries


def wait_total(schedule):
    return sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT and op.duration>CONFIG.numeric_epsilon)


def event_payload(event,blocks):
    return dict(wait_id=event.waiting.operation_id,waiting_robot=event.waiting.robot_id,
        delayed_operation_id=event.delayed_operation.operation_id,delayed_parent=blocks[event.delayed_block_id].parent_id,
        blockers=[dict(operation_id=op_id,parent=blocks[b].parent_id,forbidden_intervals=spans)
                  for op_id,b,spans in event.blocker_intervals if b in blocks])


def run_group(entry,index,initial,evidence,applicable,reason,method,q,model,extraction_seconds):
    row=dict(instance=entry['instance'],tier=entry['tier'],sheet=entry['sheet'],N=entry['N'],method=method,q=q,
        initial_cmax=None if initial is None else initial.schedule.cmax,
        initial_wait=None if initial is None else wait_total(initial.schedule),
        applicability='APPLICABLE',reason='',attempts=0,b32_calls=0,b32_success=0,improved=0,
        best_cmax=None if initial is None else initial.schedule.cmax,improvement_pct=None,
        runtime_s=extraction_seconds if method=='SCHEDULE_BLOCKER' and q==4 else 0.,trace='[]',best_case='{}')
    if initial is None:
        row.update(applicability='INITIALIZATION_FAILED',reason=reason,best_cmax=None);return row
    if method in ('WAIT_ANCHORED','SCHEDULE_BLOCKER') and not applicable:
        row.update(applicability='NOT_APPLICABLE',reason=reason,best_cmax=None);return row
    blocks=block_map(initial.solution,CONFIG);trace=[];best_case={}
    for attempt in range(4):
        seed=SEED+index*10000+q*100+attempt
        started=time.perf_counter();event=None
        if method=='SCHEDULE_BLOCKER':
            partial,event=select_schedule_blocker_parents(initial.solution,initial.schedule,evidence,q=q,seed=seed,config=CONFIG)
        else:
            partial=select_bottleneck_parents(initial.solution,initial.directions,initial.schedule,
                BottleneckSelection(method),q=q,seed=seed,config=CONFIG)
        if partial is None:raise ValueError(f'{entry["instance"]}/{method}: applicable selector returned no partial')
        result=construct_or_repair_with_gnn(initial.solution,partial.routes,partial_directions(initial,partial),
            partial.removed_parent_ids,model,CONFIG,insertion_limit=8,decode_mode='greedy',parent_selection_mode='learned')
        item=dict(attempt=attempt,seed=seed,removed=partial.removed_parent_ids,actual_q=len(partial.removed_parent_ids),
                  status='CONSTRUCTION_FAILED',candidate_cmax=None,improved=False,rejection_reason=result.rejection_reason)
        if event is not None:item['blocker_evidence']=event_payload(event,blocks)
        row['attempts']+=1
        if result.solution is not None:
            candidate=oriented.schedule_explicit(result.solution,result.directions,CONFIG)
            row['b32_calls']+=1;item['status']=candidate.schedule.status.value
            if candidate.schedule.feasible:
                row['b32_success']+=1;cmax=candidate.schedule.cmax;item['candidate_cmax']=cmax
                if cmax<initial.schedule.cmax-1e-9*max(1,initial.schedule.cmax):
                    row['improved']+=1;item['improved']=True
                    if cmax<row['best_cmax']:
                        row['best_cmax']=cmax
                        best_case=dict(instance=entry['instance'],method=method,q=q,attempt=attempt,
                            removed=partial.removed_parent_ids,changes=oriented.summary(initial,candidate,CONFIG),
                            blocker_evidence=item.get('blocker_evidence'))
        elapsed=time.perf_counter()-started;row['runtime_s']+=elapsed
        item['runtime_s']=elapsed;trace.append(item)
    row.update(improvement_pct=100*(initial.schedule.cmax-row['best_cmax'])/initial.schedule.cmax,
               trace=json.dumps(trace,ensure_ascii=False),best_case=json.dumps(best_case,ensure_ascii=False))
    return row


def save_report(rows,initials):
    valid=[r for r in rows if r['applicability']=='APPLICABLE']
    common={r['instance'] for r in valid if r['method']=='SCHEDULE_BLOCKER'}
    lines=['# GNN联合重构：最终调度直接blocker选区实验','',
        '## 问题与实现','',
        'WAIT_ANCHORED使用等待附近的时间、空间关联。本轮SCHEDULE_BLOCKER只使用原ESS计算给出的直接阻塞操作证据，不声称严格因果归因。',
        '新增默认关闭的只读feasible_dispatch_observer，保留各可调度rollout的操作提交顺序；共同Beam选出初解后，仅按返回schedule.operations取对应提交顺序，相同时间线保留首次记录。没有合并未选中或失败分支。',
        '在该精确提交前缀上，用原earliest_safe_start_optimized重算WAIT的ESS，并要求与实际被延迟操作起点逐位一致；只保留它返回的blocker IDs。没有重新运行调度或B32搜索。MOVE及其前置WAIT按原B索引操作模板映射到目标焊缝，SETUP/WELD/POST/WAIT也映射到各自原模板。',
        'WAIT按原duration×(0.25+completion/max_completion)^2加权抽取；核心先保留被延迟父，再按直接blocker父的禁止起始区间并集覆盖秒数降序、ID升序排序；不足q仅补核心任务的直接路线邻接父，允许实际移除少于q。该覆盖量仅用于删任务次序，不是新的优化目标或等待因果分解。','',
        '## 数据与共同设置','',
        '使用既有FINAL_COORDINATION_COMPARISON表中20个工作簿，原Excel加载器读取。它们已用于GRC比较，本轮属于开发诊断数据；不混入旧六DEV，不读取其他封存数据。',
        '每几何仅构造一次现有common_initial_beam，五方法共用分配、固定拆分、路线、方向和初始Cmax。模型固定gnn_joint_repair_v2.pt，learned父头、greedy、insertion_limit=8；q4/q8各4次；seed=20261103，派生为seed+原表几何索引×10000+q×100+零基attempt。各方法相同派生规则。',
        '独立部分解补全；最多800候选、每个完整候选仅一次B32，不去重增加机会、不更新初解、不执行反馈循环。接纳/改善判据沿用Cnew<Cinitial−1e−9×max(1,Cinitial)。B32为唯一候选评价器。',
        '无正WAIT或无可映射直接blocker时，两种等待策略共同标记NOT_APPLICABLE；Beam失败则五方法均不比较。时间含选区、补全、B32；blocker重建成本每几何仅计一次，记入SCHEDULE_BLOCKER/q4。共同Beam、Excel和模型加载不计入方法时间。','',
        f"共同Beam成功{sum(i['feasible'] for i in initials)}/20，实际初始化B32调用{sum(i['b32_calls'] for i in initials)}次；五策略候选B32实际调用{sum(r['b32_calls'] for r in rows)}次。共同等待/blocker适用几何{len(common)}个。",'',
        '## 实际结果','']
    for title,subset in [('全部适用数据',valid),('共同等待/blocker几何子集',[r for r in valid if r['instance'] in common])]:
        lines += [f'### {title}','','|方法|几何数|补全尝试|B32成功/调用|严格改善|改善几何数|秒|','|---|---:|---:|---:|---:|---:|---:|']
        for method in METHODS:
            rs=[r for r in subset if r['method']==method]
            lines.append(f"|{method}|{len({r['instance'] for r in rs})}|{sum(r['attempts'] for r in rs)}|{sum(r['b32_success'] for r in rs)}/{sum(r['b32_calls'] for r in rs)}|{sum(r['improved'] for r in rs)}|{len({r['instance'] for r in rs if r['improved']})}|{sum(r['runtime_s'] for r in rs):.3f}|")
        lines.append('')
    lines+=['### 每几何最佳Cmax（含共同初值）','','|几何|初始Cmax|WAIT秒|RANDOM|LOAD|TRAVEL|WAIT_ANCHORED|SCHEDULE_BLOCKER|','|---|---:|---:|---:|---:|---:|---:|---:|']
    bests={}
    for info in initials:
        name=info['instance'];bests[name]={}
        values=[]
        for method in METHODS:
            rs=[r for r in valid if r['instance']==name and r['method']==method]
            best=min((r['best_cmax'] for r in rs),default=None);bests[name][method]=best
            values.append('NA' if best is None else f'{best:.6f}')
        start='NA' if info['cmax'] is None else f"{info['cmax']:.6f}"
        wait='NA' if info['wait'] is None else f"{info['wait']:.6f}"
        lines.append('|'+ '|'.join([name,start,wait]+values)+'|')
    lines+=['','每几何/方法/q的成功、改善、最佳Cmax、改善率、耗时与四次轨迹均保存在CSV。NA原因：','']
    lines += [f"- {i['instance']}：{i['reason']}" for i in initials if i['reason']]
    lines+=['','### 几何级配对：SCHEDULE_BLOCKER对照（共同适用子集）','','|对照|胜/平/负|胜的几何|负的几何|','|---|---|---|---|']
    for method in METHODS[:-1]:
        counts=Counter();wins=[];losses=[]
        for name in sorted(common):
            a,b=bests[name]['SCHEDULE_BLOCKER'],bests[name][method];eps=1e-9*max(1,a,b)
            outcome='胜' if a<b-eps else '负' if a>b+eps else '平';counts[outcome]+=1
            if outcome=='胜':wins.append(name)
            if outcome=='负':losses.append(name)
        lines.append(f"|{method}|{counts['胜']}/{counts['平']}/{counts['负']}|{', '.join(wins) or '无'}|{', '.join(losses) or '无'}|")
    cases={}
    for r in valid:
        if r['method']!='SCHEDULE_BLOCKER' or not r['improved']:continue
        case=json.loads(r['best_case']);old=cases.get(r['instance'])
        if old is None or case['changes']['cmax_after']<old['changes']['cmax_after']:cases[r['instance']]=case
    lines+=['','## 真实改善案例','']
    for case in sorted(cases.values(),key=lambda c:(c['changes']['cmax_before']-c['changes']['cmax_after'])/c['changes']['cmax_before'],reverse=True)[:2]:
        c=case['changes'];e=case['blocker_evidence']
        lines += [f"### {case['instance']}，SCHEDULE_BLOCKER/q{case['q']}/attempt{case['attempt']}",'',
            f"WAIT `{e['wait_id']}`，等待R{e['waiting_robot']}，被推迟操作 `{e['delayed_operation_id']}`，父 `{oriented.short(e['delayed_parent'])}`。",
            '实际直接blocker：'+ '; '.join(f"`{b['operation_id']}`→`{oriented.short(b['parent'])}`" for b in e['blockers'])+'。',
            '移除父：`'+', '.join(map(oriented.short,case['removed']))+'`。',
            f"Cmax {c['cmax_before']:.6f}→{c['cmax_after']:.6f}s；WAIT {c['wait_before']:.6f}→{c['wait_after']:.6f}s。",
            '机器人转移：`'+str([(oriented.short(b),a,z) for b,a,z in c['transfers']])+'`。',
            f"保留块相对次序改变的路线：{c['reordered_routes']}。",
            '方向变化：`'+str([(oriented.short(b),a,z) for b,a,z in c['flips']])+'`。','']
    if not cases:lines+=['没有SCHEDULE_BLOCKER严格改善案例。','']
    lines+=['## 解释与决策','',
        '独立统计单位为20个几何，共同等待子集另列；候选不作为独立几何样本，不进行复杂显著性推断。DEADLOCK仅表示现有B32未找到调度。',
        ('本轮SCHEDULE_BLOCKER没有任何严格改善，不支持将其整合进GNN-V2反馈循环。保留原GNN-V2，不根据结果调整参数或继续叠加模块。'
         if not cases else '根据上述几何级配对结果判断是否存在稳定收益；不按候选数量推断几何级优势，也不在本轮继续整合。'),'',
        '复现：在DRL目录执行 `D:\\pybullet_test\\.venv\\Scripts\\python.exe -B scripts/run_gnn_schedule_blocker_pilot.py`。']
    REPORT.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    torch.set_num_threads(1);torch.manual_seed(SEED);random.seed(SEED)
    checkpoint=torch.load(ROOT/'models/gnn_joint_repair_v2.pt',map_location='cpu',weights_only=True)
    model=JointInsertionGNN(checkpoint['hidden_dim'],checkpoint['heads'],normalize_inputs=checkpoint.get('normalize_inputs',False))
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=[];initials=[]
    with CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=None
        for index,entry in enumerate(layouts()):
            source=ROOT/'data/independent/final_coordination'/entry['filename']
            parents=to_parent_welds(load_ppo_platform_instance(source,entry['sheet'],ppo_root=source.parent,instance_id=entry['instance']))
            orders={}
            initial,attempts=oriented.common_initial_beam(parents,CONFIG,seed=SEED,
                feasible_dispatch_observer=lambda result,order:orders.setdefault(result.operations,order))
            calls=sum(a['status'] in {s.value for s in ScheduleStatus} for a in attempts)
            events=();elapsed=0.;reason='';applicable=False
            if initial is None:reason='COMMON_BEAM_FAILED: '+str(Counter(a['status'] for a in attempts))
            else:
                if wait_total(initial.schedule)>0:
                    started=time.perf_counter()
                    events=selected_schedule_wait_blockers(initial.solution,initial.directions,initial.schedule,
                        orders[initial.schedule.operations],CONFIG)
                    elapsed=time.perf_counter()-started
                    applicable=any(e.blocker_intervals for e in events)
                    if not applicable:reason='NO_MAPPABLE_DIRECT_BLOCKER'
                else:reason='NO_POSITIVE_WAIT'
            initials.append(dict(instance=entry['instance'],feasible=initial is not None,
                cmax=None if initial is None else initial.schedule.cmax,wait=None if initial is None else wait_total(initial.schedule),
                b32_calls=calls,reason=reason))
            print('INITIAL',entry['instance'],initials[-1],flush=True)
            for q in (4,8):
                order=METHODS[index%len(METHODS):]+METHODS[:index%len(METHODS)]
                for method in order:
                    row=run_group(entry,index,initial,events,applicable,reason,method,q,model,elapsed)
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row);f.flush();rows.append(row)
                    print('GROUP',entry['instance'],method,q,row['applicability'],row['b32_success'],row['improved'],row['best_cmax'],flush=True)
            save_report(rows,initials)
    print('Saved',CSV,REPORT,flush=True)


if __name__=='__main__':main()
