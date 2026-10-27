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

DATA = ROOT/'data/development/gnn_joint_repair_train_v2.jsonl'
MODEL = ROOT/'models/gnn_joint_repair_v2.pt'
CSV = ROOT/'results/gnn_joint_repair_v2.csv'
REPORT = ROOT/'docs/GNN_JOINT_REPAIR_V2.md'
V1_CSV = ROOT/'results/gnn_joint_repair_pilot.csv'
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
                changes=oriented.summary(initial,scheduled,CONFIG)
                record['changes']={k:changes[k] for k in ('transfers','flips','reordered_routes')}
                # Validation uses precisely the student's action generator and
                # fails loudly on a missing label; no oracle label is injected.
                prepare_record(record)
                unique.setdefault(record_key(record),record)
                print('TEACHER',index+1,entry['sheet_name'],q,selection.value,attempt+1,
                      source,record['quality'],round(record['cmax'],3),flush=True)
            print('TEACHER PROGRESS',index+1,entry['sheet_name'],q,selection.value,
                  'B32',info['teacher_calls'],'feasible',info['teacher_success'],flush=True)
    strict=[r for r in unique.values() if r['quality']=='STRICT_IMPROVEMENT']
    # Keep a small, diverse feasibility supplement. q8, genuine transfers/order
    # changes and local flips provide useful supervision without dominating it.
    pool=[r for r in unique.values() if r['quality']!='STRICT_IMPROVEMENT']
    supplemental=[]
    for q in (8,4):
        candidates=[r for r in pool if len(r['pending_parents'])==q]
        if candidates:
            supplemental.append(min(candidates,key=supplement_rank))
    for record in sorted(pool,key=supplement_rank):
        if len(supplemental)>=min(4,max(2,len(strict))):
            break
        if record not in supplemental:
            supplemental.append(record)
    records=strict+supplemental
    info.update(samples=len(records),strict=len(strict),supplemental=len(supplemental),runtime_s=time.perf_counter()-started)
    return records,info


def record_key(record):
    """Ordinary tuple identity: geometry AND partial state AND directed result."""
    return (tuple((p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in record['parents']),
            tuple((p['parent_id'],p['kind'],p['t'],p['point_id'],p['rail']) for p in record['patterns']),
            tuple(map(tuple,record['partial_routes'])),tuple(map(tuple,record['partial_directions'])),
            tuple(sorted(record['pending_parents'])),tuple(map(tuple,record['final_routes'])),
            tuple(map(tuple,record['final_directions'])))


def supplement_rank(record):
    changes=record['changes']
    return (-bool(changes['transfers']),-bool(changes['reordered_routes']),
            -any(a['neighbour_changes'] for a in record['actions']),record['cmax'],record['seed'])


def prepare_training(records):
    allowed={e['instance_id'] for rows in dataset.train_sheet_candidates().values() for e in rows}
    grouped=defaultdict(list)
    graphs={}
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
    return graphs,grouped


@torch.no_grad()
def action_diagnostics(model,graphs,grouped):
    model.eval()
    counts=Counter()
    for name,steps in grouped.items():
        graph=graphs[name]
        embeddings=model.encode_static(graph)
        for step in steps:
            logits=model.joint_action_logits(embeddings,step['indices'],step['features'])
            pred=int(logits.argmax());target=step['target']
            pa=step['candidate_actions'][pred];ta=step['candidate_actions'][target]
            counts['actions']+=1;counts['correct']+=int(pred==target)
            for field in ('direction','robot','position'):
                counts[field+'_correct']+=int(getattr(pa,field)==getattr(ta,field))
            counts['teacher_direction_'+str(ta.direction)]+=1
            counts['predicted_direction_'+str(pa.direction)]+=1
            counts['teacher_flip_actions']+=bool(ta.neighbour_changes)
            counts['predicted_flip_actions']+=bool(pa.neighbour_changes)
            counts['flip_exact_correct']+=int(pa.neighbour_changes==ta.neighbour_changes)
            counts['teacher_flips']+=len(ta.neighbour_changes)
            counts['predicted_flips']+=len(pa.neighbour_changes)
            if ta.neighbour_changes:
                counts['teacher_flip_recalled']+=bool(pa.neighbour_changes)
            if len(step['parent_ids'])>1:
                pl=model.parent_logits(graph,embeddings,step['parent_ids'],step['parent_features'])
                counts['parent_actions']+=1
                counts['parent_correct']+=int(int(pl.argmax())==step['parent_target'])
    return dict(counts,insertion_accuracy=counts['correct']/counts['actions'],
                parent_accuracy=counts['parent_correct']/max(1,counts['parent_actions']))


def fit_model(records,args,epochs,*,small=False):
    started=time.perf_counter()
    graphs,grouped=prepare_training(records)
    torch.manual_seed(args.seed)
    model=JointInsertionGNN(64,4,normalize_inputs=True)
    model.fit_input_scaling([s for steps in grouped.values() for s in steps])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    history=[];gradient_norms=[]
    for epoch in range(epochs):
        model.train()
        rng=random.Random(args.seed+epoch)
        names=list(grouped)
        rng.shuffle(names)
        total_loss=0.0
        count=correct=parent_count=parent_correct=0
        for name in names:
            graph=graphs[name]
            steps=list(grouped[name]);rng.shuffle(steps)
            # V1 averaged all states of a geometry into one update. Small
            # batches provide many more useful updates at the same epoch count.
            for offset in range(0,len(steps),16):
                optimizer.zero_grad()
                embeddings=model.encode_static(graph)
                losses=[]
                for step in steps[offset:offset+16]:
                    logits=model.joint_action_logits(embeddings,step['indices'],step['features'])
                    loss=nn.functional.cross_entropy(logits.unsqueeze(0),torch.tensor([step['target']]))
                    correct+=int(int(logits.argmax())==step['target']);count+=1
                    if len(step['parent_ids'])>1:
                        pl=model.parent_logits(graph,embeddings,step['parent_ids'],step['parent_features'])
                        loss=loss+nn.functional.cross_entropy(pl.unsqueeze(0),torch.tensor([step['parent_target']]))
                        parent_correct+=int(int(pl.argmax())==step['parent_target']);parent_count+=1
                    losses.append(loss)
                batch_loss=torch.stack(losses).mean()
                if not torch.isfinite(batch_loss):
                    raise ValueError('nonfinite supervised loss')
                batch_loss.backward()
                norms={head:float(sum((p.grad.square().sum() for p in getattr(model,head).parameters()
                                     if p.grad is not None),torch.tensor(0.)).sqrt())
                       for head in ('layer1','layer2','parent_head','insertion_head')}
                if not all(torch.isfinite(torch.tensor(v)) for v in norms.values()):
                    raise ValueError('nonfinite supervised gradient')
                gradient_norms.append(norms)
                optimizer.step()
                total_loss+=float(batch_loss.detach())*len(losses)
        item=dict(epoch=epoch+1,loss=total_loss/count,insertion_accuracy=correct/count,
                  parent_accuracy=parent_correct/max(1,parent_count),actions=count)
        if not small or (epoch+1)%10==0 or epoch==0:
            item['teacher_forced']=action_diagnostics(model,graphs,grouped)
            print('OVERFIT' if small else 'EPOCH',epoch+1,json.dumps(item),flush=True)
        history.append(item)
        if small and 'teacher_forced' in item and min(item['teacher_forced'][k] for k in ('insertion_accuracy','parent_accuracy'))>=0.95:
            break
    diagnostics=action_diagnostics(model,graphs,grouped)
    return model,dict(history=history,teacher_forced=diagnostics,
                     gradient_ranges={h:[min(g[h] for g in gradient_norms),max(g[h] for g in gradient_norms)] for h in gradient_norms[0]},
                     trajectories=len(records),steps=sum(map(len,grouped.values())),
                     geometries=len(grouped),strict=sum(r['quality']=='STRICT_IMPROVEMENT' for r in records),
                     supplemental=sum(r['quality']!='STRICT_IMPROVEMENT' for r in records),runtime_s=time.perf_counter()-started)


def train_model(records,args,epochs):
    # Fixed TRAIN-only implementation check: select a few trajectories for
    # diverse directions, flips, robots and positions, never by DEV outcomes.
    selected=[];covered=set()
    def labels(record):
        return {('direction',a['direction']) for a in record['actions']} | {
            ('robot',a['robot']) for a in record['actions']} | {
            ('position',min(2,a['position'])) for a in record['actions']} | {
            ('flip',bool(a['neighbour_changes'])) for a in record['actions']}
    while len(selected)<4 and len(selected)<len(records):
        record=min((r for r in records if r not in selected),key=lambda r:(-len(labels(r)-covered),r['entry']['instance_id'],r['seed']))
        selected.append(record);covered.update(labels(record))
    if not {('direction',0),('direction',1),('flip',True)}<=covered:
        raise ValueError('TRAIN does not cover the required directed overfit check')
    _,small=fit_model(selected,args,300,small=True)
    small['instances']=[r['entry']['instance_id'] for r in selected]
    small['coverage']=sorted(covered)
    if min(small['teacher_forced'][k] for k in ('insertion_accuracy','parent_accuracy'))<0.95:
        raise ValueError('small real-trajectory fitting failed; inspect features and gradients before full training')
    # Fresh seed and fresh parameters; the overfit model is discarded.
    model,stats=fit_model(records,args,epochs)
    stats['overfit']=small
    return model,stats


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
    records,infos=[],[]
    workbooks=dataset.train_sheet_candidates()
    consumed=defaultdict(int);feasible=Counter()
    generation_started=time.perf_counter()
    with DATA.open('w',encoding='utf-8') as f:
        def run_stage(target):
            for path,candidates in workbooks.items():
                while feasible[path]<target and consumed[path]<5:
                    entry=candidates[consumed[path]];consumed[path]+=1
                    print('TRAIN INITIAL',len(infos)+1,entry['instance_id'],entry['N'],flush=True)
                    samples,info=teacher_instance(entry,args,len(infos))
                    info['relative_path']=path;infos.append(info);records.extend(samples)
                    feasible[path]+=int(info['initial_cmax'] is not None)
                    for record in samples:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')
                    f.flush()
                    REPORT.write_text('# GNN JointRepair V2 — TRAIN generation\n\n```json\n'+
                                      json.dumps(dict(generation=infos,trajectories=len(records),
                                                      runtime_s=time.perf_counter()-generation_started),ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
                    print('DATA',len(infos),json.dumps(info,ensure_ascii=False),'total trajectories',len(records),flush=True)
        run_stage(3)
        strict=sum(r['quality']=='STRICT_IMPROVEMENT' for r in records)
        q8=sum(len(r['pending_parents'])==8 for r in records)
        # Explicitly bounded escalation, decided only from TRAIN coverage.
        if len(records)<150 or strict<40 or q8<40:
            print('TRAIN coverage insufficient; expanding within each workbook to at most five sheets',
                  len(records),strict,q8,flush=True)
            run_stage(5)
    return records,infos,[]


def eval_group(entry,initial,partials,method,selection,q,model,args):
    row=dict(instance=entry['sheet_name'],method=method,selection=selection.value,q=q,
             initial_cmax=None if initial is None else initial.schedule.cmax,attempts=len(partials),constructed=0,
             b32_calls=0,b32_success=0,improved=0,best_cmax=None if initial is None else initial.schedule.cmax,
             best_candidate_cmax=None,runtime_s=0.0,construction_failed=0,deadlock=0,infeasible=0,
             neural_actions=0,direction0_actions=0,direction1_actions=0,flip_actions=0,neighbour_flips=0,failures='{}')
    started=time.perf_counter()
    reasons=Counter();examples=[];trace=[]
    for attempt,partial in enumerate(partials):
        seed=args.seed+oriented.NAMES.index(entry['sheet_name'])*10000+q*100+attempt
        if method.startswith('GNN'):
            result=construct_or_repair_with_gnn(initial.solution,partial.routes,partial_directions(initial,partial),
                                              partial.removed_parent_ids,model,CONFIG,insertion_limit=8)
            row['neural_actions']+=len(result.actions)
            for action in result.actions:
                row['direction'+str(action.direction)+'_actions']+=1
                row['flip_actions']+=bool(action.neighbour_changes)
                row['neighbour_flips']+=len(action.neighbour_changes)
            solution=result.solution
        else:
            result=repair_partial_state_oriented(initial.solution,initial.directions,partial,CONFIG,seed=seed,
                                                 coordination_proxy=method=='NEW',direction_versions=2 if method=='NEW' else 1,
                                                 insertion_limit=8,max_insertion_evaluations=4096)
            solution=None if result.candidate is None else result.candidate.solution
        if solution is None:
            row['construction_failed']+=1
            reasons[result.rejection_reason]+=1
            trace.append(dict(attempt=attempt,status='CONSTRUCTION_FAILED',cmax=None))
            continue
        row['constructed']+=1
        scheduled=oriented.schedule_explicit(solution,result.directions,CONFIG)
        row['b32_calls']+=1
        status=scheduled.schedule.status
        trace.append(dict(attempt=attempt,status=status.value,
                          cmax=scheduled.schedule.cmax if status is ScheduleStatus.FEASIBLE else None))
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
        if method.startswith('GNN') and (not examples or status is ScheduleStatus.FEASIBLE and scheduled.schedule.cmax<initial.schedule.cmax):
            examples.append(dict(instance=entry['sheet_name'],q=q,selection=selection.value,attempt=attempt,status=status.value,
                                 changes=oriented.summary(initial,scheduled,CONFIG),actions=[asdict(a) for a in result.actions]))
    row['failures']=json.dumps(reasons,sort_keys=True)
    row['runtime_s']=time.perf_counter()-started
    row['trace']=json.dumps(trace)
    print('DEV',entry['sheet_name'],method,selection.value,q,'success',row['b32_success'],'improved',row['improved'],
          'constructfail',row['construction_failed'],'best',row['best_cmax'],'time',round(row['runtime_s'],3),flush=True)
    return row,examples


def iterative_group(entry,initial,method,model,args):
    started=time.perf_counter()
    current=initial
    row=dict(experiment='STATEFUL_24',instance=entry['sheet_name'],method=method,selection='RANDOM/TRAVEL/LOAD',
             q='4/8',initial_cmax=initial.schedule.cmax,attempts=24,constructed=0,b32_calls=0,b32_success=0,
             improved=0,accepted=0,best_cmax=initial.schedule.cmax,best_candidate_cmax=None,runtime_s=0.,
             construction_failed=0,deadlock=0,infeasible=0,neural_actions=0,failures='{}',trace='[]',
             direction0_actions=0,direction1_actions=0,flip_actions=0,neighbour_flips=0,
             initial_runtime_s=0.,seed=args.seed,source='CURRENT_RUN')
    trace=[];reasons=Counter()
    for iteration in range(24):
        selection=SELECTORS[iteration%3]
        q=(4,8)[(iteration//3)%2]
        seed=args.seed+oriented.NAMES.index(entry['sheet_name'])*10000+iteration
        # Always select from the accepted incumbent, including its new B32 WAIT.
        partial=select_bottleneck_parents(current.solution,current.directions,current.schedule,selection,
                                          q=q,seed=seed,config=CONFIG)
        before=current.schedule.cmax
        if method.startswith('GNN'):
            result=construct_or_repair_with_gnn(current.solution,partial.routes,partial_directions(current,partial),
                                              partial.removed_parent_ids,model,CONFIG,insertion_limit=8)
            solution=result.solution
            row['neural_actions']+=len(result.actions)
            for action in result.actions:
                row['direction'+str(action.direction)+'_actions']+=1
                row['flip_actions']+=bool(action.neighbour_changes)
                row['neighbour_flips']+=len(action.neighbour_changes)
        else:
            result=repair_partial_state_oriented(current.solution,current.directions,partial,CONFIG,seed=seed,
                       coordination_proxy=True,direction_versions=2,insertion_limit=8,max_insertion_evaluations=4096)
            solution=None if result.candidate is None else result.candidate.solution
        item=dict(iteration=iteration+1,selection=selection.value,q=q,seed=seed,
                  removed=partial.removed_parent_ids,before=before,status='CONSTRUCTION_FAILED',candidate_cmax=None,accepted=False)
        if solution is None:
            row['construction_failed']+=1;reasons[result.rejection_reason]+=1
        else:
            row['constructed']+=1
            candidate=oriented.schedule_explicit(solution,result.directions,CONFIG)
            row['b32_calls']+=1
            item['status']=candidate.schedule.status.value
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:
                row['b32_success']+=1
                cmax=candidate.schedule.cmax;item['candidate_cmax']=cmax
                row['best_candidate_cmax']=cmax if row['best_candidate_cmax'] is None else min(cmax,row['best_candidate_cmax'])
                if cmax<before-1e-9*max(1,before):
                    item['changes']=oriented.summary(current,candidate,CONFIG)
                    item['accepted']=True
                    current=candidate
                    row['accepted']+=1;row['improved']+=1
            elif candidate.schedule.status is ScheduleStatus.DEADLOCK:
                row['deadlock']+=1
            else:
                row['infeasible']+=1
        item['after']=current.schedule.cmax;trace.append(item)
        print('ITERATION',entry['sheet_name'],method,iteration+1,item['status'],round(before,3),
              item['candidate_cmax'],'accepted',item['accepted'],flush=True)
    row.update(best_cmax=current.schedule.cmax,runtime_s=time.perf_counter()-started,
               trace=json.dumps(trace,ensure_ascii=False),failures=json.dumps(reasons))
    return row


def save_results(rows,checkpoint,initials,examples,args):
    with CSV.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stats=checkpoint['training'];diag=stats['teacher_forced'];small=stats['overfit']
    generation=checkpoint['generation']
    records=[json.loads(line) for line in DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    qcounts=Counter(len(r['pending_parents']) for r in records)
    paths={r['entry']['relative_path'] for r in records}
    successful=sum(i['initial_cmax'] is not None for i in generation)
    def aggregates(experiment):
        result=[]
        methods=list(dict.fromkeys(r['method'] for r in rows if r['experiment']==experiment))
        for method in methods:
            group=[r for r in rows if r['experiment']==experiment and r['method']==method]
            result.append((method,group))
        return result
    lines=['# GNN JointRepair V2：扩大 TRAIN 与真实调度反馈实验','',
           '方法：调度反馈驱动的冲突感知联合构造与瓶颈重构。固定拆分模式，方向和邻块翻转由同一次联合插入处理；完整方案仅调用一次 B32，真实 Cmax 严格下降才接纳。',
           '保持两层 SparseGATLayer、hidden=64、heads=4、父选择头和联合候选评分头。没有独立方向阶段或新增启发式算子。','',
           '## 1. TRAIN 数据','',
           f"现有17个 MLP_TRAIN 工作簿中514个 VALID、唯一、formal PASS、10≤N≤90的清单实例可用。实际尝试 {len(generation)} 个 sheet，Beam 成功 {successful} 个，监督覆盖 {stats['geometries']} 个几何、{len(paths)} 个工作簿。",
           '每个工作簿先选最小/中位/最大 N 的三个不同 sheet；同 N 按实例 ID 决定。替补及扩展按 N 与已有空间描述的标准化最小距离选取。全程只依据 TRAIN 几何；每工作簿最多尝试5个。未读取 V2_VALIDATION 或 ID_TEST_SEALED 的 Excel；MLP_DEV 无监督或参数选择用途。',
           f"有效轨迹 {stats['trajectories']}：STRICT_IMPROVEMENT={stats['strict']}，FEASIBLE_NONIMPROVING={stats['supplemental']}；插入动作 {stats['steps']}；q4/q8={qcounts[4]}/{qcounts[8]}。",
           'RANDOM/TRAVEL/LOAD × q4/q8 × 3次。每个部分解均以 OLD/NEW 有向 Regret-2 补全，预算4096、insertion_limit=8，各完整补全仅一次 B32；选可调度且 Cmax 最低的教师。按原始几何、固定模式、部分路线及方向、待补父集合、最终路线及方向组成普通 tuple 去重。',
           '保留全部严格改善；非改善每几何最多2—4条，优先真实归属/次序变化、邻块翻转，并覆盖q8。两类质量单列，不将非改善标成改善。',
           f"教师生成耗时 {sum(i['runtime_s'] for i in generation):.3f}s；教师 B32 {sum(i['teacher_calls'] for i in generation)} 次，成功 {sum(i['teacher_success'] for i in generation)} 次。Beam B32 调用 {sum(i['init_calls'] for i in generation)} 次另计。",'',
           '|TRAIN workbook::sheet|N|Beam Cmax|教师B32/成功|严格/非改善|秒|',
           '|---|---:|---:|---:|---:|---:|']
    for info in generation:
        c='FAIL' if info['initial_cmax'] is None else f"{info['initial_cmax']:.3f}"
        lines.append(f"|{info['instance_id']}|{info['N']}|{c}|{info['teacher_calls']}/{info['teacher_success']}|{info['strict']}/{info['supplemental']}|{info['runtime_s']:.2f}|")
    lines+=['','## 2. 神经动作学习','',
            '检查未发现教师索引或方向语义错位；每个动作在训练/推理共同的 joint_block_candidates 中唯一匹配，并重放至相同最终路线和方向。局部行程特征被总加工时间缩小，且V1每几何每epoch仅一次更新。V2只以TRAIN候选拟合逐特征均值/标准差（最小标准差0.001），保存为模型buffer；训练和推理共用同一处理。每16步小批量更新，交叉熵同时训练父选择与条件插入。未输入未来Cmax、WAIT、完整教师路线或标签。',
            f"小样本真实轨迹 {small['trajectories']} 条/{small['steps']}步，覆盖标签 {small['coverage']}。在 {len(small['history'])} epoch 达到插入准确率 {small['teacher_forced']['insertion_accuracy']:.2%}、父选择 {small['teacher_forced']['parent_accuracy']:.2%}；其模型丢弃。此检查只证明可拟合固定TRAIN动作。",
            f"正式模型重新初始化，seed={checkpoint['seed']}，AdamW lr=0.001，epochs={checkpoint['epochs']}，batch=16；CPU、单线程、PyTorch {torch.__version__}。预处理及训练 {stats['runtime_s']:.3f}s，小样本检查另计 {small['runtime_s']:.3f}s。首/末损失 {stats['history'][0]['loss']:.5f}/{stats['history'][-1]['loss']:.5f}。无DEV checkpoint选择。",
            f"训练后教师强制插入准确率 {diag['insertion_accuracy']:.2%}，父选择 {diag['parent_accuracy']:.2%}；机器人 {diag['robot_correct']/diag['actions']:.2%}、位置 {diag['position_correct']/diag['actions']:.2%}、方向 {diag['direction_correct']/diag['actions']:.2%}。",
            f"教师方向0/1：{diag.get('teacher_direction_0',0)}/{diag.get('teacher_direction_1',0)}；预测方向0/1：{diag.get('predicted_direction_0',0)}/{diag.get('predicted_direction_1',0)}。教师/预测含翻转动作：{diag['teacher_flip_actions']}/{diag['predicted_flip_actions']}；实际邻块翻转数：{diag['teacher_flips']}/{diag['predicted_flips']}。教师含翻转时预测也含翻转 {diag['teacher_flip_recalled']}/{diag['teacher_flip_actions']}；完整翻转元组一致 {diag['flip_exact_correct']}/{diag['actions']}。",
            f"各层梯度范数范围 {stats['gradient_ranges']}。",'',
            '|epoch|loss|训练后插入准确率|父准确率|预测方向1|预测翻转动作|',
            '|---:|---:|---:|---:|---:|---:|']
    for item in stats['history']:
        d=item['teacher_forced']
        lines.append(f"|{item['epoch']}|{item['loss']:.4f}|{d['insertion_accuracy']:.2%}|{d['parent_accuracy']:.2%}|{d.get('predicted_direction_1',0)}|{d['predicted_flip_actions']}|")
    lines+=['','## 3. 固定初解、相同部分解配对实验','',
            '六个原DEV，seed=20261027，q4/q8，RANDOM/TRAVEL/LOAD，每组8次，共288候选/方法。共用同一Beam初解、固定模式、移除集合、合法资格、方向候选和B32。',
            'V1与NEW条件一致时读取原CSV；初解Cmax逐组核对。代码保留V1未归一化默认行为，方向和候选生成器未改。历史运行时间沿用原测量，V2为本次测量；均含补全和B32，不含公共初解。CSV source 列区分历史复用与本次实测。','',
            '|方法|B32成功/尝试|严格改善|补全+B32秒|', '|---|---:|---:|---:|']
    for method,group in aggregates('PAIRED_FIXED'):
        lines.append(f"|{method}|{sum(r['b32_success'] for r in group)}/{sum(r['attempts'] for r in group)}|{sum(r['improved'] for r in group)}|{sum(r['runtime_s'] for r in group):.3f}|")
    vg=[r for r in rows if r['experiment']=='PAIRED_FIXED' and r['method']=='GNN-V2']
    lines+=['',f"V2实际DEV自主补全动作（含最终失败的部分轨迹）：方向0/1={sum(r['direction0_actions'] for r in vg)}/{sum(r['direction1_actions'] for r in vg)}，含邻翻转动作={sum(r['flip_actions'] for r in vg)}，邻块翻转总数={sum(r['neighbour_flips'] for r in vg)}。"]
    lines+=['','|实例|初解Cmax|V1最佳|V2最佳|NEW最佳|V2成功/改善|V2秒|', '|---|---:|---:|---:|---:|---:|---:|']
    def groups(name,experiment,method):
        return [r for r in rows if r['instance']==name and r['experiment']==experiment and r['method']==method]
    for name,initial in initials.items():
        def best(method):
            g=groups(name,'PAIRED_FIXED',method)
            return min(r['best_cmax'] for r in g) if g else initial
        v2=groups(name,'PAIRED_FIXED','GNN-V2')
        lines.append(f"|{name}|{initial:.3f}|{best('GNN-V1'):.3f}|{best('GNN-V2'):.3f}|{best('NEW'):.3f}|{sum(r['b32_success'] for r in v2)}/{sum(r['improved'] for r in v2)}|{sum(r['runtime_s'] for r in v2):.3f}|")
    lines+=['','V2各实例最好改善（没有改善时保存首个可调度/失败例）：','']
    for case in examples:
        changes=case['changes']
        lines.append(f"- {case['instance']} q{case['q']} {case['selection']} attempt={case['attempt']} {case['status']}：Cmax {changes['cmax_before']:.3f}→{changes['cmax_after']}，归属改变 {len(changes['transfers'])}，次序改变路线 {changes['reordered_routes']}，方向变化 {len(changes['flips'])}，WAIT {changes['wait_before']:.3f}→{changes['wait_after']}。")
    lines+=['','## 4. 有状态调度反馈循环（单列）','',
            '每实例每方法24轮；RANDOM/TRAVEL/LOAD依次轮换，每三轮切换q4/q8，固定种子。NEW与V2各维护自己的current；严格改善后下一轮从更新路线、方向和调度重新选瓶颈。每轮最多一次B32，最多24次/实例/方法；不添加重启、SA或其他算子。CSV trace保存全部24轮的移除父集合、前后Cmax、B32状态、接纳与改变明细。','',
            '|实例|方法|成功/调用|接纳次数|初解→最终Cmax|相对48个独立候选最佳|相对24个独立候选最佳|秒|','|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['experiment']!='STATEFUL_24':continue
        paired=groups(row['instance'],'PAIRED_FIXED',row['method'])
        fixed=min(r['best_cmax'] for r in paired)
        independent=[a['cmax'] for r in paired for a in json.loads(r['trace'])
                     if a.get('attempt',99)<4 and a.get('cmax') is not None]
        delta24=f"{row['best_cmax']-min([row['initial_cmax']]+independent):+.3f}" if independent else '历史CSV无逐次记录'
        lines.append(f"|{row['instance']}|{row['method']}|{row['b32_success']}/{row['b32_calls']}|{row['accepted']}|{row['initial_cmax']:.3f}→{row['best_cmax']:.3f}|{row['best_cmax']-fixed:+.3f}|{delta24}|{row['runtime_s']:.3f}|")
    lines+=['','循环24轮，完整配对实验48次/实例；V2另取每个选区/q组合的前4次构成24个独立候选比较，未按结果挑选。NEW历史CSV无逐次记录，不补造24次对照。等次数比较仍只是六个DEV上的观察；多次接纳证明当前方案确实逐轮更新，不能将288次独立候选称作反馈循环。','']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def evaluate(model,checkpoint,args):
    _,dev=entries();by_name={e['sheet_name']:e for e in dev}
    historical=list(csv.DictReader(V1_CSV.open(encoding='utf-8-sig')))
    rows=[];initials={};examples=[];initial_states={};initial_times={}
    fields=('experiment','instance','method','selection','q','initial_cmax','attempts','constructed','b32_calls',
            'b32_success','improved','best_cmax','best_candidate_cmax','runtime_s','construction_failed','deadlock',
            'infeasible','neural_actions','failures','accepted','trace','initial_runtime_s','seed','source')
    fields=fields+('direction0_actions','direction1_actions','flip_actions','neighbour_flips')
    for name in oriented.NAMES:
        if name not in args.dev_instances:continue
        entry=by_name[name];start=time.perf_counter()
        initial,attempts=oriented.common_initial_beam(dataset.load_parents(entry),CONFIG,seed=args.seed)
        initial_time=time.perf_counter()-start
        if initial is None:raise ValueError(f'Existing DEV Beam initial failed: {name}')
        initials[name]=initial.schedule.cmax
        initial_states[name]=initial;initial_times[name]=initial_time
        print('DEV INITIAL',name,initial.schedule.cmax,'seconds',initial_time,flush=True)
        old=[r for r in historical if r['instance']==name and r['method'] in ('NEW','GNN-JointRepair')]
        reusable=(args.seed==20261027 and args.eval_attempts==8 and len(old)==12 and
                  {(r['method'],r['selection'],int(r['q'])) for r in old}==
                  {(m,s.value,q) for m in ('NEW','GNN-JointRepair') for s in SELECTORS for q in (4,8)} and
                  all(abs(float(r['initial_cmax'])-initial.schedule.cmax)<1e-6 for r in old))
        if reusable:
            for original in old:
                row={k:None for k in fields};row.update(original)
                for k in ('q','attempts','constructed','b32_calls','b32_success','improved','construction_failed','deadlock','infeasible','neural_actions'):
                    row[k]=int(row[k])
                for k in ('initial_cmax','best_cmax','best_candidate_cmax','runtime_s'):
                    row[k]=float(row[k]) if row[k] else None
                row.update(experiment='PAIRED_FIXED',method='GNN-V1' if row['method']=='GNN-JointRepair' else 'NEW',
                           accepted=0,trace='[]',initial_runtime_s=initial_time,seed=args.seed,source='V1_CSV_REUSED')
                rows.append(row)
        else:
            old_checkpoint=torch.load(ROOT/'models/gnn_joint_repair_v1.pt',map_location='cpu',weights_only=True)
            old_model=JointInsertionGNN(64,4);old_model.load_state_dict(old_checkpoint['state_dict'])
        for q in (4,8):
            for selection in SELECTORS:
                partials=[select_bottleneck_parents(initial.solution,initial.directions,initial.schedule,selection,q=q,
                          seed=args.seed+oriented.NAMES.index(name)*10000+q*100+attempt,config=CONFIG)
                          for attempt in range(args.eval_attempts)]
                methods=[('GNN-V2',model)] if reusable else [('NEW',None),('GNN-V1',old_model),('GNN-V2',model)]
                for method,policy in methods:
                    row,cases=eval_group(entry,initial,partials,method,selection,q,policy,args)
                    row=dict({k:None for k in fields},**row)
                    row.update(experiment='PAIRED_FIXED',accepted=0,initial_runtime_s=initial_time,
                               seed=args.seed,source='CURRENT_RUN')
                    rows.append(row)
                    if method=='GNN-V2':
                        previous=[c for c in examples if c['instance']==name]
                        feasible_cases=[c for c in previous+cases if c['status']=='FEASIBLE']
                        chosen=min(feasible_cases,key=lambda c:c['changes']['cmax_after']) if feasible_cases else (previous+cases)[0] if previous+cases else None
                        examples=[c for c in examples if c['instance']!=name]+([chosen] if chosen else [])
                save_results(rows,checkpoint,initials,examples,args)
    # Finish all fixed-initial comparisons before starting feedback search.
    for name,initial in initial_states.items():
        entry=by_name[name]
        for method,policy in (('NEW',None),('GNN-V2',model)):
            row=iterative_group(entry,initial,method,policy,args)
            row['initial_runtime_s']=initial_times[name]
            rows.append({k:row[k] for k in fields})
            save_results(rows,checkpoint,initials,examples,args)
    print('Saved',CSV,REPORT,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('all','data','train','eval'),default='all')
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--train-attempts',type=int,default=3)
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
            if REPORT.exists() and '```json\n' in REPORT.read_text(encoding='utf-8'):
                generation=json.loads(REPORT.read_text(encoding='utf-8').split('```json\n',1)[1].split('\n```',1)[0])['generation']
        model,stats=train_model(records,args,args.epochs)
        checkpoint=dict(state_dict=model.state_dict(),hidden_dim=64,heads=4,seed=args.seed,epochs=args.epochs,
                        insertion_limit=8,normalize_inputs=True,training=stats,generation=generation,smoke=smoke)
        torch.save(checkpoint,MODEL)
        print('MODEL SAVED',MODEL,flush=True)
        if args.phase=='train':
            return
    else:
        checkpoint=torch.load(MODEL,map_location='cpu',weights_only=True)
        model=JointInsertionGNN(checkpoint['hidden_dim'],checkpoint['heads'],normalize_inputs=checkpoint.get('normalize_inputs',False))
        model.load_state_dict(checkpoint['state_dict'])
    evaluate(model,checkpoint,args)


if __name__=='__main__':
    main()
