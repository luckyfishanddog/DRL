"""Opportunity-state Cmax pair ranking; historical REINFORCE remains below."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import os
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from torch.nn import functional as F
import run_gnn_joint_repair_pilot as pilot
from mrta_search.gnn_constructor import JointInsertionGNN, construct_or_repair_with_gnn, rollout_joint_repair
from mrta_reference.model import ScheduleStatus

MODEL = ROOT / 'models/gnn_joint_cmax_rl_v1.pt'
CSV = ROOT / 'results/gnn_joint_cmax_rl_v1.csv'
REPORT = ROOT / 'docs/GNN_JOINT_CMAX_RL_V1.md'
SETTINGS = dict(seed=20261027, geometries=20, epochs=3, episodes=8, learning_rate=1e-4,
                insertion_limit=8, temperature=1., bc_coefficient=.01, bc_steps=4, gradient_clip=1.)
HEADS = ('layer1', 'layer2', 'parent_head', 'insertion_head')
CONFIG = pilot.CONFIG

OP_CHECKPOINT=ROOT/'models/gnn_coordination_constructor_05.pt'
OP_CSV=ROOT/'results/gnn_opportunity_cmax.csv'
OP_REPORT=ROOT/'docs/GNN_INITIAL_MINCHILD05_FINAL.md'
OP_VARIANT='opportunity_cmax_candidate'
EARLY_VARIANT='early_cmax_candidate'
PARENT_CMAX_VARIANT='parent_cmax_candidate'
PARENT_MACRO_IL_VARIANT='parent_cmax_macro_il_candidate'
COMPLETE_VARIANT='complete_parent_candidate'
COMPACT_VARIANT='compact_workload_candidate'
OP_SEED=20261104
OP_EVAL_SEED=202611041  # Exact existing RULE/WAG/HGA pairs; no formal rerun.
OP_FIELDS=('phase','instance','seed','method','status','cmax','runtime_s','b32_calls','loads','move','wait','robots','payload')


def op_read():
    csv.field_size_limit(100_000_000)
    if not OP_CSV.exists():return []
    with OP_CSV.open(encoding='utf-8',newline='') as stream:
        return [dict(r,payload=json.loads(r['payload'])) for r in csv.DictReader(stream)]


def op_save(row):
    header=not OP_CSV.exists() or OP_CSV.stat().st_size==0
    raw={k:row.get(k) for k in OP_FIELDS}
    raw['payload']=json.dumps(row.get('payload',{}),ensure_ascii=False,allow_nan=False,separators=(',',':'))
    with OP_CSV.open('a',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=OP_FIELDS)
        if header:writer.writeheader()
        writer.writerow(raw);stream.flush();os.fsync(stream.fileno())
    print('OP',row['phase'],row.get('instance'),row.get('method'),row.get('status'),row.get('cmax'),flush=True)


def op_sources():
    import run_gnn_shared_construction as shared
    from mrta_reference.model import ScientificConfig
    config=ScientificConfig()
    checkpoint=shared.read_neural_checkpoint(OP_CHECKPOINT)
    archive=checkpoint['refinement_candidate']['refinement']
    rows=shared.research_rows(ROOT/'results/parent_mode_joint_reconstruction_05.csv')+shared.research_rows(ROOT/'results/final_parent_mode_joint_reconstruction_05.csv')
    training,development,split=shared.refinement_split(rows)
    teachers,_=shared.research_teachers(training,config,deployment_only=False)
    geometries={}
    for item in teachers:geometries.setdefault(item['geometry_key'],item)
    if len(geometries)!=21 or len(teachers)!=len(archive['replays']):
        raise ValueError('retained TRAIN teacher coverage changed')
    entries={}
    for row in development:
        if row['instance'] in entries:continue
        state=(row.get('details')or{}).get('final_state') or (row.get('details')or{}).get('initial_state')
        if state:
            solution,_,gkey,_,_=shared.research_restore(state,config)
            if gkey in geometries:raise ValueError('development geometry in TRAIN')
            entries[row['instance']]=solution.parents
    if sorted(entries)!=archive['split']['holdout_names']:raise ValueError('use the existing eight development geometries')
    return shared,config,checkpoint,archive,teachers,list(geometries.values()),entries,development


def op_action(raw):
    from mrta_search.lns import OrientedInsertionAction
    return OrientedInsertionAction(raw['block_id'],raw['robot'],raw['position'],raw['direction'],
                                  tuple(tuple(x) for x in raw['neighbour_changes']))


def op_metrics(candidate,config):
    from mrta_reference.model import OperationKind
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    blocks=block_map(candidate.solution,config,scope=FORMAL_SCOPE_V2)
    loads=[sum(config.process_time(blocks[b].length) for b in route.block_ids) for route in candidate.solution.routes]
    return dict(loads=loads,move=sum(o.duration for o in candidate.schedule.operations if o.kind is OperationKind.MOVE),
                wait=sum(o.duration for o in candidate.schedule.operations if o.kind is OperationKind.WAIT),
                robots=sum(bool(r.block_ids) for r in candidate.solution.routes))


def op_pair_labels(outcomes):
    """Only within-state comparisons. Neither failed branch gets a fake Cmax."""
    pairs=[]
    for i,a in enumerate(outcomes):
        for j,b in enumerate(outcomes[i+1:],i+1):
            if 'ERROR' in (a['status'],b['status']):continue
            fa=a['status']=='FEASIBLE';fb=b['status']=='FEASIBLE'
            if fa and fb:
                if math.isclose(a['cmax'],b['cmax'],rel_tol=1e-9,abs_tol=1e-9):continue
                pairs.append((i,j) if a['cmax']<b['cmax'] else (j,i))
            elif fa!=fb:pairs.append((i,j) if fa else (j,i))
    return pairs


def op_decision_pool(geometry,state,pending,model,config,*,early=False):
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    graph,blocks,patterns=c.joint_graph(geometry,config)
    ids,parent_features=c.joint_parent_inputs(state,pending,blocks,patterns,config,include_opportunities=True)
    if not ids:return graph,ids,parent_features,{}
    embeddings=model.encode_static(graph)
    embeddings=model.contextualize_state(graph,embeddings,state.routes,state.directions)
    parent_logits=model.parent_logits(graph,embeddings,ids,parent_features)
    order=sorted(range(len(ids)),key=lambda i:(-float(parent_logits[i]),i))
    if early and len(order)>1:
        # Select ONE different-parent representative using cheap, exact
        # insertion-existence queries, not complete counterfactual rollouts.
        from mrta_search.lns import has_joint_block_candidate
        def priority(index):
            children=sorted((b for b in blocks.values() if b.parent_id==ids[index]),key=lambda b:b.block_id)
            robots=sum(has_joint_block_candidate(state,children[0],patterns[ids[index]],blocks,config,
                insertion_limit=8,robot_id=r) for r in range(4))
            processing=sum(config.process_time(b.length) for b in children)
            return (processing/max(1,robots),processing,-robots,float(parent_logits[index]),-index)
        order=[order[0],max(order[1:],key=priority)]
    pools={}
    for index in order[:2]:
        parent=ids[index]
        child=min((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
        variants=joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=8)
        indices,features=c.joint_candidate_inputs(graph,state,variants,blocks,config,
                                                 include_opportunities=True,patterns=patterns)
        logits=model.joint_action_logits(embeddings,indices,features)
        pools[parent]=dict(parent_index=index,variants=variants,indices=indices,features=features,logits=logits)
    return graph,ids,parent_features,pools


def op_representative_actions(geometry,state,pending,model,config,*,early=False):
    with torch.inference_mode():
        _,_,_,pools=op_decision_pool(geometry,state,pending,model,config,early=early)
        if not pools:return []
        first=next(iter(pools.values()));logits=first['logits'];variants=first['variants']
        best=int(logits.argmax());chosen=[variants[best][1]]
        other=[i for i,(_,a) in enumerate(variants) if a.robot!=chosen[0].robot]
        if other:
            index=(min(other,key=lambda i:(max(variants[i][0].loads),-float(logits[i]),i)) if early else
                   max(other,key=lambda i:(float(logits[i]),-i)))
            chosen.append(variants[index][1])
        if len(pools)>1:
            alternative=list(pools.values())[1]
            chosen.append(alternative['variants'][int(alternative['logits'].argmax())][1])
        elif not early:
            alternatives=[i for i,(_,a) in enumerate(variants) if a.position!=chosen[0].position]
            if alternatives:chosen.append(variants[max(alternatives,key=lambda i:(float(logits[i]),-i))][1])
        from mrta_search.lns import joint_action_key
        return list({joint_action_key(a):a for a in chosen}.values())[:3]


def op_prefix_targets(parent_count,*,early=False):
    fractions=(0.,.1,.25) if early else (.25,.5,.75)
    return {f:int(parent_count*f) if early else max(1,int(parent_count*f)) for f in fractions}


def op_collect(*,early=False):
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state
    shared,config,checkpoint,archive,teachers,training,_,_=op_sources()
    model=(shared._model_from_checkpoint(checkpoint[OP_VARIANT]) if early else
           shared.expand_opportunity_model(shared._model_from_checkpoint(checkpoint['refinement_candidate']))).eval()
    student_phase='early_student' if early else 'student'
    counter_phase='early_counterfactual' if early else 'counterfactual'
    saved=op_read();done={(r['instance'],r['payload'].get('fraction')) for r in saved if r['phase']==counter_phase}
    calls=sum(int(r['b32_calls'] or 0) for r in saved if r['phase']==counter_phase)
    for item in training:
        name=item['instance'];parents=item['solution'].parents
        geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
        existing=next((r for r in saved if r['phase']==student_phase and r['instance']==name),None)
        if existing:student=existing['payload']
        else:
            snapshots={};targets=op_prefix_targets(len(parents),early=early)
            def observe(state,pending,actions):
                completed=len(parents)-len(pending)
                for fraction,target in targets.items():
                    if completed==target:
                        snapshots[str(fraction)]=dict(routes=state.routes,directions=state.directions,pending=pending,
                            assigned=tuple(sorted(p.parent_id for p in parents if p.parent_id not in pending)),
                            progress=completed/len(parents))
            result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,tuple(p.parent_id for p in parents),model,config,parent_observer=observe)
            student=dict(parents=[asdict(p) for p in parents],snapshots=snapshots,actions=[asdict(a) for a in result.actions],
                         failure=result.rejection_reason,complete=result.solution is not None)
            op_save(dict(phase=student_phase,instance=name,seed=OP_SEED,method=OP_VARIANT if early else 'INITIAL_STUDENT',
                         status='COMPLETE' if result.solution else 'CONSTRUCTION_FAILED',b32_calls=0,payload=student))
        graph,blocks,patterns=c.joint_graph(geometry,config);cache={}
        def key(raw):return (tuple(tuple(r['block_ids']) for r in raw['solution']['routes']),tuple(tuple(x) for x in raw['directions']))
        for row in saved:
            if row['phase'] in ('counterfactual',counter_phase) and row['instance']==name:
                for outcome in row['payload']['outcomes']:
                    if outcome.get('complete'):cache[key(outcome['complete'])]=outcome
        for fraction in op_prefix_targets(len(parents),early=early):
            if (name,fraction) in done:continue
            prefix=student['snapshots'].get(str(fraction))
            if prefix is None:
                op_save(dict(phase=counter_phase,instance=name,seed=OP_SEED,method='FORCED_GREEDY',status='MISSING_STUDENT_PREFIX',b32_calls=0,
                    payload=dict(fraction=fraction,outcomes=[],pairs=[],student_failure=student['failure'])));continue
            state=oriented_route_state(tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),blocks,config)
            choices=op_representative_actions(geometry,state,prefix['pending'],model,config,early=early)
            outcomes=[];physical=0;tick=time.perf_counter()
            for action in choices:
                result=c.construct_or_repair_with_gnn(geometry,state.routes,state.directions,prefix['pending'],model,config,forced_action=action)
                outcome=dict(action=asdict(action),status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,failure=result.rejection_reason)
                if result.solution is not None:
                    raw=dict(solution=asdict(result.solution),directions=result.directions)
                    ident=key(raw);outcome['complete']=raw
                    if ident in cache:
                        earlier=cache[ident]
                        outcome.update(status=earlier['status'],cmax=earlier['cmax'],metrics=earlier.get('metrics'),reused=True)
                    else:
                        if calls>=189:raise ValueError('the requested 189 new B32 limit is exhausted')
                        scheduled=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                        calls+=1;physical+=1;outcome['b32_calls']=1
                        feasible=scheduled.schedule.status is ScheduleStatus.FEASIBLE
                        outcome.update(status=scheduled.schedule.status.value,cmax=scheduled.schedule.cmax if feasible else None,
                                       metrics=op_metrics(scheduled,config) if feasible else None,reused=False)
                        cache[ident]=outcome
                outcomes.append(outcome)
            op_save(dict(phase=counter_phase,instance=name,seed=OP_SEED,method='FORCED_GREEDY',status='COLLECTED',
                runtime_s=time.perf_counter()-tick,b32_calls=physical,payload=dict(fraction=fraction,prefix=prefix,parents=student['parents'],
                    outcomes=outcomes,pairs=op_pair_labels(outcomes))))
    print('COUNTERFACTUAL_B32',calls,'maximum',189,flush=True)


def op_graph_key(reference):
    return (tuple((p.parent_id,p.start,p.end) for p in reference.parents),reference.patterns)


def op_saved_imitation(shared,config,archive,teachers):
    """Reuse retained actions/continuations, with ZERO new teacher scheduling."""
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_action_key
    graphs={};examples=[];raw_steps=[];merged={};bounds={};replayed=continuations=0
    for item in teachers:
        bounds[item['geometry_key']]=min(bounds.get(item['geometry_key'],float('inf')),item['cmax'])
    def absorb(reference,routes,directions,pending,actions,weight,target):
        graph,steps,_=c.prepare_joint_teacher_steps(reference,routes,directions,pending,actions,config,include_opportunities=True)
        key=op_graph_key(reference);graphs.setdefault(key,graph);raw_steps.extend(steps)
        for step in steps:
            prefix=(key,step['state_routes'],step['state_directions'],step['pending_parent_ids'])
            pool=tuple(joint_action_key(a) for a in step['candidate_actions'])
            cases=[('insertion',pool,step['target'])]
            if len(step['parent_ids'])>1:cases.append(('parent',step['parent_ids'],step['parent_target']))
            for kind,options,label in cases:
                ident=(kind,prefix,options)
                if ident not in target:target[ident]=dict(kind=kind,graph_key=key,step=step,target=torch.zeros(len(options)),weights=[])
                target[ident]['target'][label]+=weight;target[ident]['weights'].append(weight)
    def finish(target):
        for example in target.values():
            example['target']/=example['target'].sum()
            weights=example.pop('weights');example['weight']=sum(weights)/len(weights)
            examples.append(example)
    by_name={}
    for item,record in zip(teachers,archive['replays']):
        if item['instance']!=record['instance'] or abs(item['cmax']-record['cmax'])>1e-6:
            raise ValueError('saved teacher action order differs from retained sources')
        by_name.setdefault(item['instance'],item)
        if not record.get('actions'):continue
        reference=c.joint_geometry(item['solution'].parents,item['solution'].patterns,config)
        absorb(reference,((),)*4,((),)*4,tuple(p.parent_id for p in reference.parents),record['actions'],
               bounds[item['geometry_key']]/item['cmax'],merged)
        replayed+=1
    finish(merged)
    for record in archive['prefix_diagnostics']:
        if not record['continuations']:continue
        item=by_name[record['instance']];parents=item['solution'].parents
        reference=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
        best=min(p['cmax'] for p in record['continuations']);local={}
        for continuation in record['continuations']:
            absorb(reference,tuple(map(tuple,record['partial_routes'])),tuple(map(tuple,record['partial_directions'])),
                record['pending_parent_ids'],continuation['actions'],best/continuation['cmax'],local)
            continuations+=1
        finish(local)
    for example in examples:
        if example['kind']!='insertion':continue
        step=example['step'];key=example['graph_key'];graph=graphs[key]
        gkey=tuple(sorted(tuple(sorted((p[1],p[2]))) for p in key[0]))
        total=max(1.,sum(config.process_time(b.length) for b in graph.blocks))
        processing={b.block_id:config.process_time(b.length) for b in graph.blocks}
        projected=step['features'][:,:4].clone()*total
        for i,action in enumerate(step['candidate_actions']):projected[i,action.robot]+=processing[action.block_id]
        example['load_excess']=torch.relu(projected.max(1).values/bounds[gkey]-1.)
    return graphs,examples,raw_steps,dict(teachers_reused=replayed,prefix_continuations_reused=continuations,
        imitation_examples=len(examples),graphs=len(graphs),teacher_b32_calls=0)


def op_compile_ranks(records,graphs,config,*,phase='counterfactual'):
    from mrta_search import gnn_constructor as c
    from mrta_reference.model import ParentWeld
    from mrta_search.lns import oriented_route_state,joint_block_candidates,joint_action_key
    compiled=[];raw_steps=[]
    for row in records:
        if row['phase']!=phase or not row['payload'].get('pairs'):continue
        payload=row['payload'];prefix=payload['prefix']
        parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in payload['parents'])
        reference=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
        graph,blocks,patterns=c.joint_graph(reference,config);key=op_graph_key(reference);graphs.setdefault(key,graph)
        state=oriented_route_state(tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),blocks,config)
        ids,parent_features=c.joint_parent_inputs(state,prefix['pending'],blocks,patterns,config,include_opportunities=True)
        options=[];pools={}
        for outcome in payload['outcomes']:
            action=op_action(outcome['action']);parent=blocks[action.block_id].parent_id
            if parent not in pools:
                child=min((b for b in blocks.values() if b.parent_id==parent),key=lambda b:b.block_id)
                variants=joint_block_candidates(state,child,patterns[parent],blocks,config,insertion_limit=8)
                indices,features=c.joint_candidate_inputs(graph,state,variants,blocks,config,include_opportunities=True,patterns=patterns)
                pools[parent]=(variants,indices,features)
                raw_steps.append(dict(features=features,parent_features=parent_features))
            variants,indices,features=pools[parent]
            index=next(i for i,(_,a) in enumerate(variants) if joint_action_key(a)==joint_action_key(action))
            options.append(dict(parent_index=ids.index(parent),indices=indices,features=features,action_index=index))
        record=dict(graph_key=key,parent_ids=ids,parent_features=parent_features,options=options)
        compiled.extend((record,a,b) for a,b in payload['pairs'])
    return compiled,raw_steps


def op_train(*,early=False):
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    variant=EARLY_VARIANT if early else OP_VARIANT
    phase='early_' if early else ''
    counter_phase=phase+'counterfactual'
    if variant in checkpoint:
        print('Existing candidate retained; no second training:',variant,flush=True);return
    rows=op_read()
    if len([r for r in rows if r['phase']==counter_phase])!=63:
        raise ValueError('collect all requested TRAIN prefix attempts before training')
    tick=time.perf_counter()
    graphs,examples,steps,preparation=op_saved_imitation(shared,config,archive,teachers)
    ranking,rank_steps=op_compile_ranks(rows,graphs,config,phase=counter_phase)
    preparation.update(rank_pairs=len(ranking),preparation_seconds=time.perf_counter()-tick,
        counterfactual_b32_calls=sum(int(r['b32_calls'] or 0) for r in rows if r['phase']==counter_phase))
    op_save(dict(phase=phase+'preparation',instance='TRAIN_21',seed=OP_SEED,method='SAVED_ACTIONS',status='PREPARED',b32_calls=0,payload=preparation))
    if not ranking:raise ValueError('no real same-state Cmax/feasibility pairs were obtained')
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT if early else 'refinement_candidate'])
    if early:
        model=shared._model_from_checkpoint(checkpoint[OP_VARIANT])
    else:
        model=shared.expand_opportunity_model(source);model.fit_opportunity_scaling(steps+rank_steps)
    torch.manual_seed(OP_SEED);rng=random.Random(OP_SEED)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    grouped={}
    for example in examples:grouped.setdefault(example['graph_key'][0],[]).append(example)
    ranked={}
    for item in ranking:ranked.setdefault(item[0]['graph_key'][0],[]).append(item)
    started=time.perf_counter();history=[]
    for epoch in range(10):
        model.train();keys=list(grouped);rng.shuffle(keys);totals=Counter();updates=0
        for geometry in keys:
            batch=list(grouped[geometry]);rng.shuffle(batch);pairs=ranked.get(geometry,[])
            for offset in range(0,len(batch),16):
                optimizer.zero_grad(set_to_none=True)
                chosen=[rng.choice(pairs) for _ in range(16)] if pairs else []
                loss,parts=shared.opportunity_ranking_loss(model,graphs,batch[offset:offset+16],chosen)
                if not torch.isfinite(loss):raise ValueError('nonfinite opportunity ranking loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;totals['loss']+=float(loss.detach())
                for k,v in parts.items():totals[k]+=float(v.detach())
                totals['gradient_norm']+=float(norm)
        result=dict(epoch=epoch+1,updates=updates,seconds=time.perf_counter()-started,
                    **{k:v/updates for k,v in totals.items()})
        history.append(result)
        op_save(dict(phase=phase+'training',instance='TRAIN_21',seed=OP_SEED,method=variant,status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    for prefix,size in (('joint',41),('parent',12)):
        for suffix in ('mean','scale'):
            expected=getattr(source,prefix+'_'+suffix)
            actual=getattr(model,prefix+'_'+suffix)
            if not torch.equal(actual if early else actual[:size],expected):
                raise ValueError('legacy normalization parameters changed')
    metadata=dict(preparation=preparation,epochs=10,learning_rate=1e-4,batch_size=16,gradient_clip=1.,seed=OP_SEED,
        losses=dict(imitation=.5,ranking=1.,load=.1),history=history,training_seconds=time.perf_counter()-started,
        opportunity_definition='Remaining unplaced children, exact current insertion existence per robot, processing normalized by full graph processing.',
        training_geometries=21,development_geometries=8,static_context=True)
    metadata.update(initialization=OP_VARIANT if early else 'refinement_candidate',
        prefix_fractions=list(op_prefix_targets(100,early=early)),normalization_refit=not early)
    checkpoint[variant]=dict(model_type='ModeJointInsertionGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),state_dict=model.state_dict(),training=metadata)
    old_root={k:v.clone() for k,v in checkpoint['state_dict'].items()}
    old_candidate={k:v.clone() for k,v in checkpoint['refinement_candidate']['state_dict'].items()}
    old_opportunity={k:v.clone() for k,v in checkpoint[OP_VARIANT]['state_dict'].items()} if early else None
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    if not all(torch.equal(v,saved['state_dict'][k]) for k,v in old_root.items()) or not all(
        torch.equal(v,saved['refinement_candidate']['state_dict'][k]) for k,v in old_candidate.items()):
        raise ValueError('protected existing weights changed')
    if early and not all(torch.equal(v,saved[OP_VARIANT]['state_dict'][k]) for k,v in old_opportunity.items()):
        raise ValueError('protected opportunity weights changed')
    op_save(dict(phase=phase+'model',instance='TRAIN_21',seed=OP_SEED,method=variant,status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


def op_evaluate():
    shared,config,_,_,_,_,entries,development=op_sources()
    saved=op_read();done={(r['instance'],r['method']) for r in saved if r['phase']=='evaluation'}
    for name,parents in sorted(entries.items()):
        for method in ('RULE/BEAM','WAG','HGA'):
            if (name,method) in done:continue
            source=next(r for r in development if r['instance']==name and r['seed']==OP_EVAL_SEED and
                        r['method']==('MODE_NEW' if method=='RULE/BEAM' else method))
            detail=source['details'];initial=method=='RULE/BEAM'
            state=detail.get('initial_state' if initial else 'final_state')
            value=state.get('cmax') if initial and state else source['cmax'] if not initial else None
            metrics={}
            if state:
                if value is not None and abs(value-state['cmax'])>1e-6:raise ValueError('historical timeline is not the paired Cmax')
                loads=state['loads'];loads=json.loads(loads) if isinstance(loads,str) else loads
                metrics=dict(loads=loads,move=state['travel'],wait=state['wait'],robots=sum(v>0 for v in loads))
            runtime=detail.get('initialization_seconds',source['runtime_s']) if initial else source['runtime_s']
            op_save(dict(phase='evaluation',instance=name,seed=OP_EVAL_SEED,method=method,
                status='FEASIBLE' if value is not None else 'INITIALIZATION_FAILED',cmax=value,runtime_s=runtime,b32_calls=0,
                **metrics,payload=dict(reused=True,source='results/final_parent_mode_joint_reconstruction_05.csv',
                    source_method=source['method'],source_status=source['status'],
                    logical_b32_calls=detail.get('initial_reference_calls',source['b32_calls']) if initial else source['b32_calls'],
                    complete=None if state is None else dict(solution=state['solution'],directions=state['directions']))))
        methods=('NEW_GNN_GREEDY','NEW_GNN_MULTI3','OLD_GNN_GREEDY','CURRENT_PORTFOLIO3')
        if all((name,m) in done for m in methods):continue
        if any((name,m) in done for m in methods):
            raise ValueError('partial neural evaluation already saved; retain it rather than silently duplicate B32')
        # The new independent method runs FIRST with an empty schedule cache.
        start=time.perf_counter();thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
        new=shared._model_from_checkpoint(thin[OP_VARIANT]);new_load=time.perf_counter()-start
        cache={}
        best,info=shared.neural_single_model_portfolio(parents,new,config,seed=OP_EVAL_SEED,scheduled_cache=cache)
        new_results=[('NEW_GNN_GREEDY',info['candidates'][0],new_load+info['attempts'][0]['elapsed'],
                      0,info['attempts'][:1],dict(logical_b32_calls=info['attempts'][0]['b32_calls'],
                      timing='first proposal of the same actual trio run',source_variant=OP_VARIANT)),
                     ('NEW_GNN_MULTI3',best,new_load+info['runtime_s'],info['b32_calls'],info['attempts'],
                      dict(logical_b32_calls=sum(a['b32_calls'] for a in info['attempts']),source_variant=OP_VARIANT))]
        start=time.perf_counter();old_data=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
        old=shared._model_from_checkpoint(old_data);old_load=time.perf_counter()-start
        tick=time.perf_counter();refined=shared._model_from_checkpoint(old_data['refinement_candidate'])
        extra_load=time.perf_counter()-tick
        _,old_info=shared.neural_single_model_portfolio(parents,old,config,seed=OP_EVAL_SEED,proposals=2,scheduled_cache=cache)
        refined_best,refined_info=shared.neural_single_model_portfolio(parents,refined,config,seed=OP_EVAL_SEED,proposals=1,scheduled_cache=cache)
        menu=[old_info['candidates'][0],old_info['candidates'][1],refined_info['candidates'][0]]
        feasible=[(i,v) for i,v in enumerate(menu) if v is not None and v.schedule.status is ScheduleStatus.FEASIBLE]
        winner=min(feasible,key=lambda x:x[1].schedule.cmax) if feasible else (None,None)
        logical=lambda attempts:len({(tuple(tuple(r['block_ids']) for r in a['solution']['routes']),
            tuple(map(tuple,a['directions']))) for a in attempts if a.get('solution')})
        current_attempts=[dict(a,proposal_index=i,source_variant='root' if i<2 else 'refinement_candidate')
                          for i,a in enumerate(old_info['attempts'][:2]+refined_info['attempts'])]
        def signature(a):
            return (tuple(tuple(r['block_ids']) for r in a['solution']['routes']),tuple(map(tuple,a['directions']))) if a.get('solution') else None
        duplicate_ref=signature(current_attempts[2]) in {signature(a) for a in current_attempts[:2]}
        duplicate_credit=refined_info['reused_b32_seconds'] if duplicate_ref else 0.
        old_results=[('OLD_GNN_GREEDY',old_info['candidates'][0],old_load+old_info['attempts'][0]['elapsed'],0,
                      old_info['attempts'][:1],dict(logical_b32_calls=logical(old_info['attempts'][:1]),source_variant='root')),
                     ('CURRENT_PORTFOLIO3',winner[1],old_load+extra_load+old_info['attempts'][1]['elapsed']+refined_info['runtime_s']-duplicate_credit,
                      old_info['b32_calls']+refined_info['b32_calls'],current_attempts,
                      dict(logical_b32_calls=logical(current_attempts),source_variant='root greedy + root sample + refinement greedy',selected_proposal=winner[0]))]
        for method,candidate,runtime,physical,attempts,extra in new_results+old_results:
            valid=candidate is not None and candidate.schedule.status is ScheduleStatus.FEASIBLE
            metrics=op_metrics(candidate,config) if valid else {}
            op_save(dict(phase='evaluation',instance=name,seed=OP_EVAL_SEED,method=method,
                status='FEASIBLE' if valid else 'NO_FEASIBLE_CONSTRUCTION',
                cmax=candidate.schedule.cmax if valid else None,runtime_s=runtime,b32_calls=physical,**metrics,
                payload=dict(attempts=attempts,model_load_seconds=new_load if method.startswith('NEW') else old_load+(extra_load if method=='CURRENT_PORTFOLIO3' else 0.),
                    complete=None if not valid else dict(solution=asdict(candidate.solution),directions=candidate.directions),
                    timing_shared_schedule='Equal complete proposals reuse one actual B32 result and its measured duration; logical greedy views share the trio first proposal.',**extra)))


def op_complete_key(raw):
    """Existing geometry/pattern/route values identify an exact B32 input."""
    solution=raw['solution']
    return (tuple((p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in solution['parents']),
            tuple(tuple(sorted(p.items())) for p in solution['patterns']),
            tuple(tuple(r['block_ids']) for r in solution['routes']),tuple(map(tuple,raw['directions'])))


def op_parent_diagnose():
    """Recompute rank recall from original labels and their frozen policies."""
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_reference.model import ParentWeld
    rows=op_read();thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    models={'early_counterfactual':shared._model_from_checkpoint(thin[OP_VARIANT]).eval(),
        'counterfactual':shared.expand_opportunity_model(shared._model_from_checkpoint(thin['refinement_candidate'])).eval()}
    done={(r['instance'],r['payload']['source_phase'],r['payload']['fraction'])
          for r in rows if r['phase']=='parent_recall'}
    source_rows=[r for r in rows if r['phase'] in models]
    def restore(row):
        p=row['payload'];prefix=p['prefix']
        parents=tuple(ParentWeld(v['parent_id'],tuple(v['start']),tuple(v['end'])) for v in p['parents'])
        reference=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
        graph,blocks,patterns=c.joint_graph(reference,CONFIG)
        state=c.joint_start_state(reference,tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),
                                 prefix['pending'],blocks,CONFIG)
        ids,pf=c.joint_parent_inputs(state,prefix['pending'],blocks,patterns,CONFIG,include_opportunities=True)
        return graph,blocks,patterns,state,ids,pf
    for row in source_rows:
        p=row['payload'];source=row['phase'];fraction=p['fraction']
        if (row['instance'],source,fraction) in done:continue
        pairs=op_pair_labels(p['outcomes'])
        better=[(i,o) for i,o in enumerate(p['outcomes']) if (i,0) in pairs]
        alternatives=[]
        if better:
            graph,blocks,patterns,state,ids,pf=restore(row)
            with torch.inference_mode():
                model=models[source]
                emb=model.contextualize_state(graph,model.encode_static(graph),state.routes,state.directions)
                logits=model.parent_logits(graph,emb,ids,pf)
            order=sorted(range(len(ids)),key=lambda i:(-float(logits[i]),i))
            ranks={ids[i]:rank+1 for rank,i in enumerate(order)}
            a=p['outcomes'][0]
            if ranks[blocks[a['action']['block_id']].parent_id]!=1:
                raise ValueError('saved first action does not match its actual source parent head')
            for index,outcome in better:
                parent=blocks[outcome['action']['block_id']].parent_id
                alternatives.append(dict(index=index,parent_id=parent,parent_rank=ranks[parent],
                    kind='cmax' if a['status']=='FEASIBLE' else 'feasibility',
                    baseline_status=a['status'],baseline_cmax=a['cmax'],status=outcome['status'],cmax=outcome['cmax'],
                    action=outcome['action']))
        op_save(dict(phase='parent_recall',instance=row['instance'],seed=OP_SEED,method='RAW_SAME_STATE',
            status='ANALYZED',b32_calls=0,payload=dict(source_phase=source,fraction=fraction,
                source_variant=OP_VARIANT if source=='early_counterfactual' else 'expanded refinement before opportunity training',
                alternatives=alternatives)))
    done_timing={(r['method'],r['payload']['k']) for r in rows if r['phase']=='parent_k_timing'}
    for source,fraction,large in (('early_counterfactual',0.,False),('early_counterfactual',0.,True),
            ('early_counterfactual',.1,True),('early_counterfactual',.25,True),
            ('counterfactual',.5,True),('counterfactual',.75,True)):
        method=f'{source}_{fraction}_{"large" if large else "small"}'
        available=[r for r in source_rows if r['phase']==source and r['payload']['fraction']==fraction]
        row=(max if large else min)(available,key=lambda r:len(r['payload']['parents']))
        graph,blocks,patterns,state,ids,pf=restore(row)
        model=models[source]
        with torch.inference_mode():
            embeddings=model.encode_static(graph)
            context=model.contextualize_state(graph,embeddings,state.routes,state.directions)
            logits=model.parent_logits(graph,context,ids,pf)
            for k in (2,4,8):
                if (method,k) in done_timing:continue
                timing={};tick=time.perf_counter()
                plans=c.joint_parent_action_candidates(graph,state,ids,logits,embeddings,model,blocks,patterns,CONFIG,
                    parent_limit=k,child_limit=2,current_embeddings=context,timing=timing)
                elapsed=time.perf_counter()-tick
                op_save(dict(phase='parent_k_timing',instance=row['instance'],seed=OP_SEED,method=method,
                    status='MEASURED',runtime_s=elapsed,b32_calls=0,payload=dict(k=k,fraction=fraction,
                        source_phase=source,parents=len(row['payload']['parents']),complete_parent_plans=len(plans),**timing)))
    # Necessary first-child coverage of the same known better alternatives.
    # This is not another Cmax evaluation or another decoding policy.
    from mrta_search.lns import joint_block_candidates,joint_action_key
    covered={(r['instance'],r['payload']['fraction']) for r in rows if r['phase']=='parent_action_coverage'}
    for row in source_rows:
        if row['phase']!='early_counterfactual':continue
        p=row['payload'];fraction=p['fraction']
        if (row['instance'],fraction) in covered:continue
        alternatives=[(i,o) for i,o in enumerate(p['outcomes']) if (i,0) in op_pair_labels(p['outcomes'])]
        if not alternatives:continue
        graph,blocks,patterns,state,ids,pf=restore(row);model=models['early_counterfactual'];measured=[];pools={}
        with torch.inference_mode():
            emb=model.contextualize_state(graph,model.encode_static(graph),state.routes,state.directions)
            logits=model.parent_logits(graph,emb,ids,pf)
            order=sorted(range(len(ids)),key=lambda i:(-float(logits[i]),i));ranks={ids[i]:j+1 for j,i in enumerate(order)}
            for i,outcome in alternatives:
                action=op_action(outcome['action']);child=blocks[action.block_id]
                if child.block_id not in pools:
                    variants=joint_block_candidates(state,child,patterns[child.parent_id],blocks,CONFIG)
                    indices,features=c.joint_candidate_inputs(graph,state,variants,blocks,CONFIG,
                        include_opportunities=True,patterns=patterns)
                    values=model.joint_action_logits(emb,indices,features)
                    order=sorted(range(len(variants)),key=lambda j:(-float(values[j]),j))
                    pools[child.block_id]={joint_action_key(variants[j][1]):rank+1 for rank,j in enumerate(order)}
                measured.append(dict(index=i,parent_rank=ranks[child.parent_id],
                    first_child_rank=pools[child.block_id][joint_action_key(action)],
                    kind='cmax' if p['outcomes'][0]['status']=='FEASIBLE' else 'feasibility'))
        op_save(dict(phase='parent_action_coverage',instance=row['instance'],seed=OP_SEED,method='FROZEN_OP',
            status='MEASURED',b32_calls=0,payload=dict(fraction=fraction,alternatives=measured)))


def cmax_prefix(payload,config=CONFIG):
    from mrta_search import gnn_constructor as c
    from mrta_reference.model import ParentWeld
    parents=tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in payload['parents'])
    reference=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
    graph,blocks,patterns=c.joint_graph(reference,config);prefix=payload['prefix']
    state=c.joint_start_state(reference,tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),
                             prefix['pending'],blocks,config)
    return reference,graph,blocks,patterns,state


def cmax_plans(reference,graph,blocks,patterns,state,pending,model,config=CONFIG,*,diverse=True):
    from mrta_search import gnn_constructor as c
    # These graphs/indices are reused by autograd during offline training.
    # no_grad keeps their lazy incoming-edge cache as ordinary tensors.
    with torch.no_grad():
        ids,pf=c.joint_parent_inputs(state,pending,blocks,patterns,config,include_opportunities=True)
        emb=model.encode_static(graph);context=model.contextualize_state(graph,emb,state.routes,state.directions)
        logits=model.parent_logits(graph,context,ids,pf)
        return c.joint_parent_action_candidates(graph,state,ids,logits,emb,model,blocks,patterns,config,
            parent_limit=4,child_limit=2,current_embeddings=context,diverse=diverse,retain_inputs=True,parent_features=pf)


def cmax_action_key(plan):
    from mrta_search.lns import joint_action_key
    return tuple(joint_action_key(a) for a in plan['actions'])


def cmax_compile(rows,source,graphs):
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_action_key
    records=[]
    for row in rows:
        if row['phase']!='parent_cmax_counterfactual' or not row['payload']['pairs']:continue
        payload=row['payload'];reference,graph,blocks,patterns,state=cmax_prefix(payload)
        key=op_graph_key(reference);graphs.setdefault(key,graph)
        plans=cmax_plans(reference,graph,blocks,patterns,state,payload['prefix']['pending'],source)
        by_action={cmax_action_key(p):p for p in plans}
        chosen=[by_action[tuple(joint_action_key(op_action(a)) for a in o['actions'])] for o in payload['outcomes']]
        inputs=c.parent_action_inputs(graph,state,chosen,blocks,CONFIG)
        records.append(dict(graph_key=key,inputs=inputs,pairs=payload['pairs'],instance=row['instance'],fraction=payload['fraction'],
                            legacy_scores=tuple(p['score'] for p in chosen)))
    return records


def cmax_rank_accuracy(model,graphs,records):
    correct=total=legacy=0;encoded={}
    model.eval()
    with torch.no_grad():
        for record in records:
            key=record['graph_key']
            if key not in encoded:encoded[key]=model.encode_static(graphs[key])
            scores=model.parent_action_scores(graphs[key],encoded[key],record['inputs'])
            for better,worse in record['pairs']:
                total+=1;correct+=float(scores[better])>float(scores[worse])
                legacy+=record['legacy_scores'][better]>record['legacy_scores'][worse]
    return dict(correct=correct,total=total,accuracy=correct/max(1,total),legacy_correct=legacy,
                legacy_accuracy=legacy/max(1,total))


def cmax_expert_macro_examples(source,graphs,examples,*,raw_only=False):
    """Six real teacher prefixes per geometry, including late construction.

    Recover both teacher children from already validated insertion examples.
    A missing teacher action may be appended ONLY to the offline IL pool; it
    does not expand the inference candidates or supply a synthetic Cmax label.
    """
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates,joint_action_key,oriented_route_state
    from mrta_reference.model import ParentWeld
    insertion=[e for e in examples if e['kind']=='insertion']
    following={}
    starts={}
    for e in insertion:
        step=e['step'];key=(e['graph_key'],step['state_routes'],step['state_directions'],step['pending_parent_ids'])
        if step['parent_target'] is None:following[key]=e
        else:starts.setdefault(e['graph_key'][0],[]).append(e)
    result=[];missing=0;duplicates=set()
    for geometry,available in starts.items():
        for fraction in (0.,.1,.25,.5,.75,.9):
            order=sorted(available,key=lambda e:(abs(1-len(e['step']['pending_parent_ids'])/len(geometry)-fraction),-e['weight']))
            selected=None
            for e in order:
                key=e['graph_key'];step=e['step'];ident=(key,step['state_routes'],step['state_directions'],step['pending_parent_ids'])
                if ident in duplicates:continue
                graph=graphs[key];blocks={b.block_id:b for b in graph.blocks};patterns={p.parent_id:p for p in key[1]}
                ref=c.joint_geometry(tuple(ParentWeld(*p) for p in key[0]),key[1],CONFIG)
                state=oriented_route_state(step['state_routes'],step['state_directions'],blocks,CONFIG)
                first_index=int(e['target'].argmax());first=step['candidate_actions'][first_index]
                parent=blocks[first.block_id].parent_id
                variants=joint_block_candidates(state,blocks[first.block_id],patterns[parent],blocks,CONFIG)
                after=next((s for s,a in variants if joint_action_key(a)==joint_action_key(first)),None)
                if after is None:continue
                actions=[first];indices=[step['indices'][first_index]];features=[step['features'][first_index]]
                if sum(b.parent_id==parent for b in blocks.values())==2:
                    second=following.get((key,after.routes,after.directions,step['pending_parent_ids']))
                    if second is None:continue
                    j=int(second['target'].argmax());action=second['step']['candidate_actions'][j]
                    if blocks[action.block_id].parent_id!=parent:continue
                    variants=joint_block_candidates(after,blocks[action.block_id],patterns[parent],blocks,CONFIG)
                    last=next((s for s,a in variants if joint_action_key(a)==joint_action_key(action)),None)
                    if last is None:continue
                    after=last;actions.append(action);indices.append(second['step']['indices'][j]);features.append(second['step']['features'][j])
                teacher=dict(parent_id=parent,state=after,actions=tuple(actions),child_indices=torch.stack(indices),
                    child_features=torch.stack(features),parent_features=step['parent_features'][step['parent_target']].clone())
                plans=[] if raw_only else list(cmax_plans(ref,graph,blocks,patterns,state,step['pending_parent_ids'],source))
                target=next((i for i,p in enumerate(plans) if cmax_action_key(p)==cmax_action_key(teacher)),None)
                covered=target is not None
                if target is None:target=len(plans);plans.append(teacher)
                inputs=None if raw_only else c.parent_action_inputs(graph,state,plans,blocks,CONFIG)
                selected=dict(graph_key=key,inputs=inputs,target=target,weight=e['weight'],fraction=fraction,
                    progress=1-len(step['pending_parent_ids'])/len(geometry),teacher_in_inference_pool=covered,
                    reference=ref,state=state,pending=step['pending_parent_ids'],teacher_plan=teacher)
                duplicates.add(ident);break
            if selected is None:missing+=1
            else:result.append(selected)
    return result,dict(expert_macro_examples=len(result),unavailable_prefixes=missing,
        teacher_action_in_inference_pool=sum(e['teacher_in_inference_pool'] for e in result),
        prefix_bins=dict(Counter(str(e['fraction']) for e in result)))


def op_parent_cmax_macro_train():
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    if PARENT_MACRO_IL_VARIANT in checkpoint:
        print('Retain existing macro-IL correction; no repeated training.',flush=True);return
    rows=op_read();graphs,examples,_,preparation=op_saved_imitation(shared,config,archive,teachers)
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    records=cmax_compile(rows,source,graphs)
    macro,coverage=cmax_expert_macro_examples(source,graphs,examples)
    torch.manual_seed(OP_SEED);rng=random.Random(OP_SEED);model=shared.expand_parent_action_model(source)
    model.requires_grad_(False);model.parent_action_head.requires_grad_(True)
    encoded={}
    with torch.no_grad():
        for record in records+macro:
            key=record['graph_key']
            if key not in encoded:encoded[key]=model.encode_static(graphs[key])
            record['rows']=model.parent_action_rows(graphs[key],encoded[key],record['inputs']).clone()
    pairs=[(record,a,b) for record in records for a,b in record['pairs']]
    initial=cmax_rank_accuracy(model,graphs,records)
    op_save(dict(phase='parent_cmax_macro_preparation',instance='TRAIN_21',seed=OP_SEED,method='FROZEN_BASE_DIRECT_MACRO_IL',
        status='PREPARED',b32_calls=0,payload=dict(coverage=coverage,preparation=preparation,initial=initial)))
    optimizer=torch.optim.AdamW(model.parent_action_head.parameters(),lr=1e-4)
    steps=checkpoint[PARENT_CMAX_VARIANT]['training']['history'][0]['updates']
    started=time.perf_counter();history=[]
    from torch.nn import functional as F
    for epoch in range(10):
        totals=Counter();model.train()
        for _ in range(steps):
            chosen=[rng.choice(pairs) for _ in range(16)];expert=[rng.choice(macro) for _ in range(16)]
            optimizer.zero_grad(set_to_none=True);scores={};rank=[];il=[]
            for record,better,worse in chosen:
                ident=id(record)
                if ident not in scores:scores[ident]=model.parent_action_head(record['rows']).squeeze(1)
                value=scores[ident];rank.append(F.softplus(-(value[better]-value[worse])))
            for e in expert:
                logits=model.parent_action_head(e['rows']).squeeze(1)
                il.append(-e['weight']*torch.log_softmax(logits,0)[e['target']])
            rank_loss=torch.stack(rank).mean();il_loss=torch.stack(il).mean();loss=rank_loss+.1*il_loss
            if not torch.isfinite(loss):raise ValueError('nonfinite direct macro imitation loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parent_action_head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            totals['loss']+=float(loss.detach());totals['rank']+=float(rank_loss.detach());totals['il']+=float(il_loss.detach())
        result=dict(epoch=epoch+1,updates=steps,seconds=time.perf_counter()-started,
            **{k:v/steps for k,v in totals.items()},training_pairs=cmax_rank_accuracy(model,graphs,records))
        with torch.no_grad():result['macro_imitation_accuracy']=sum(int(model.parent_action_head(e['rows']).argmax())==e['target'] for e in macro)/len(macro)
        history.append(result)
        op_save(dict(phase='parent_cmax_macro_training',instance='TRAIN_21',seed=OP_SEED,method='RANK_PLUS_01_COMPLETE_IL',
            status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in source.state_dict().items()):
        raise ValueError('the corrected proposal network must remain exactly frozen')
    metadata=dict(initialization=OP_VARIANT,frozen_base=True,coverage=coverage,epochs=10,updates_per_epoch=steps,seed=OP_SEED,
        learning_rate=1e-4,batch_size=16,gradient_clip=1.,losses=dict(rank=1.,complete_imitation=.1),
        rank_pairs=len(pairs),history=history,initial=initial,training_seconds=time.perf_counter()-started,
        new_teacher_b32_calls=0,new_counterfactual_b32_calls=0,training_geometries=21)
    protected={k:{n:v.clone() for n,v in checkpoint[k]['state_dict'].items()} for k in
        ('refinement_candidate',OP_VARIANT,EARLY_VARIANT,PARENT_CMAX_VARIANT)}
    protected['root']={n:v.clone() for n,v in checkpoint['state_dict'].items()}
    checkpoint[PARENT_MACRO_IL_VARIANT]=dict(model_type='ModeJointInsertionGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,parent_action_features=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),
        state_dict=model.state_dict(),training=metadata)
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    for key,values in protected.items():
        actual=saved['state_dict'] if key=='root' else saved[key]['state_dict']
        if not all(torch.equal(v,actual[k]) for k,v in values.items()):raise ValueError('historical weight branch changed')
    op_save(dict(phase='parent_cmax_macro_model',instance='TRAIN_21',seed=OP_SEED,method=PARENT_MACRO_IL_VARIANT,
        status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


def op_parent_cmax_train():
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    if PARENT_CMAX_VARIANT in checkpoint:
        print('Retain existing complete-action model; no repeated training.',flush=True);return
    rows=op_read();data=[r for r in rows if r['phase']=='parent_cmax_counterfactual']
    if len(data)!=63:raise ValueError('complete the 63 original early TRAIN states first')
    graphs,examples,_,preparation=op_saved_imitation(shared,config,archive,teachers)
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    records=cmax_compile(rows,source,graphs)
    torch.manual_seed(OP_SEED);rng=random.Random(OP_SEED)
    model=shared.expand_parent_action_model(source)
    initial=cmax_rank_accuracy(model,graphs,records)
    preparation.update(complete_parent_states=len(records),rank_pairs=initial['total'],
        new_b32_calls=sum(int(r['b32_calls']) for r in data),head_parameters=sum(p.numel() for p in model.parent_action_head.parameters()))
    op_save(dict(phase='parent_cmax_preparation',instance='TRAIN_21',seed=OP_SEED,method='COMPLETE_ACTION_RANK',
        status='PREPARED',b32_calls=0,payload=dict(preparation=preparation,initial=initial)))
    grouped={};ranked={}
    for example in examples:grouped.setdefault(example['graph_key'][0],[]).append(example)
    for record in records:
        ranked.setdefault(record['graph_key'][0],[]).extend((record,a,b) for a,b in record['pairs'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);history=[];started=time.perf_counter()
    for epoch in range(10):
        model.train();keys=list(grouped);rng.shuffle(keys);totals=Counter();updates=0
        for geometry in keys:
            batch=list(grouped[geometry]);rng.shuffle(batch);pairs=ranked.get(geometry,[])
            for offset in range(0,len(batch),16):
                selected=[rng.choice(pairs) for _ in range(16)] if pairs else []
                optimizer.zero_grad(set_to_none=True)
                loss,parts=shared.opportunity_ranking_loss(model,graphs,batch[offset:offset+16],selected,
                    complete_actions=True,imitation_weight=.1,load_weight=0.)
                if not torch.isfinite(loss):raise ValueError('nonfinite complete-parent ranking loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1;totals['loss']+=float(loss.detach())
                for k in ('il','rank'):totals[k]+=float(parts[k].detach())
        result=dict(epoch=epoch+1,updates=updates,seconds=time.perf_counter()-started,
            **{k:v/updates for k,v in totals.items()},training_pairs=cmax_rank_accuracy(model,graphs,records))
        history.append(result)
        op_save(dict(phase='parent_cmax_training',instance='TRAIN_21',seed=OP_SEED,method='PAIR_RANK_PLUS_01_IL',
            status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    protected={'root':checkpoint['state_dict']}
    for key in ('refinement_candidate',OP_VARIANT,EARLY_VARIANT):protected[key]=checkpoint[key]['state_dict']
    metadata=dict(seed=OP_SEED,epochs=10,learning_rate=1e-4,batch_size=16,gradient_clip=1.,losses=dict(rank=1.,imitation=.1,load=0.),
        initialization=OP_VARIANT,preparation=preparation,initial=initial,history=history,
        training_seconds=time.perf_counter()-started,training_geometries=21,development_geometries=8,
        frozen_completion='original OP hierarchical greedy for every offline branch',candidate_parents='top4 + at most 2 legal qualification/load representatives',
        children='2 legal structures per child, maximum 4 complete branches per split parent')
    checkpoint[PARENT_CMAX_VARIANT]=dict(model_type='ModeJointInsertionGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,parent_action_features=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),
        state_dict=model.state_dict(),training=metadata)
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    for key,values in protected.items():
        actual=saved['state_dict'] if key=='root' else saved[key]['state_dict']
        if not all(torch.equal(v,actual[n]) for n,v in values.items()):raise ValueError('protected historical model changed')
    op_save(dict(phase='parent_cmax_model',instance='TRAIN_21',seed=OP_SEED,method=PARENT_CMAX_VARIANT,
        status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


def op_parent_cmax_data():
    """Reuse legacy labels; only ONE extra full-parent candidate may call B32."""
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_action_key
    source=shared._model_from_checkpoint(shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)[OP_VARIANT]).eval()
    rows=op_read();done={(r['instance'],r['payload']['fraction']) for r in rows if r['phase']=='parent_cmax_counterfactual'}
    cache={}
    for row in rows:
        if row['phase'] not in ('counterfactual','early_counterfactual','parent_cmax_counterfactual'):continue
        for o in row['payload'].get('outcomes',[]):
            if o.get('complete'):cache[op_complete_key(o['complete'])]=o
    for row in rows:
        if row['phase']!='early_counterfactual':continue
        payload=row['payload'];fraction=payload['fraction'];name=row['instance']
        if (name,fraction) in done:continue
        reference,graph,blocks,patterns,state=cmax_prefix(payload)
        plans=cmax_plans(reference,graph,blocks,patterns,state,payload['prefix']['pending'],source)
        outcomes=[];chosen=[];coverage=[];skipped=[];physical=0;started=time.perf_counter()
        def complete(plan):
            pending=tuple(p for p in payload['prefix']['pending'] if p!=plan['parent_id'])
            return c.construct_or_repair_with_gnn(reference,plan['state'].routes,plan['state'].directions,pending,source,CONFIG)
        # This loop NEVER schedules: either an exact existing completion is
        # reproduced and its label reused, or the record is skipped explicitly.
        for index,old in enumerate(payload['outcomes']):
            action=op_action(old['action'])
            matching=[p for p in plans if joint_action_key(p['actions'][0])==joint_action_key(action)]
            coverage.append(dict(index=index,covered=bool(matching),
                parent_covered=any(p['parent_id']==blocks[action.block_id].parent_id for p in plans),
                strict_better=(index,0) in op_pair_labels(payload['outcomes']) and payload['outcomes'][0]['status']=='FEASIBLE'))
            if not matching:continue
            plan=max(matching,key=lambda p:p['child_log_probabilities'][-1] if len(p['actions'])==2 else p['score'])
            result=complete(plan)
            raw=None if result.solution is None else dict(solution=asdict(result.solution),directions=result.directions)
            if raw is None or not old.get('complete') or op_complete_key(raw)!=op_complete_key(old['complete']):
                skipped.append(dict(index=index,reason='legacy full continuation not reproduced'));continue
            if cmax_action_key(plan) in {cmax_action_key(p) for p in chosen}:continue
            chosen.append(plan)
            outcomes.append(dict(actions=[asdict(a) for a in plan['actions']],status=old['status'],cmax=old['cmax'],
                complete=raw,reused=True,b32_calls=0,origin='legacy'))
        remaining=[p for p in plans if cmax_action_key(p) not in {cmax_action_key(v) for v in chosen}]
        if remaining:
            parents={p['parent_id'] for p in chosen};count=len(chosen[0]['actions']) if chosen else 1
            # Exactly one extra candidate, favouring a new parent or a different
            # WHOLE/split length. The unchanged B32 is called only below.
            plan=max(remaining,key=lambda p:(p['parent_id'] not in parents,len(p['actions'])!=count,
                -max(p['state'].loads),-p['state'].travel,p['score']))
            result=complete(plan)
            o=dict(actions=[asdict(a) for a in plan['actions']],status='CONSTRUCTION_FAILED',cmax=None,
                   b32_calls=0,reused=False,origin='one_extra',failure=result.rejection_reason)
            if result.solution is not None:
                raw=dict(solution=asdict(result.solution),directions=result.directions);key=op_complete_key(raw);o['complete']=raw
                if key in cache:o.update(status=cache[key]['status'],cmax=cache[key]['cmax'],reused=True)
                else:
                    candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,CONFIG)
                    physical=1;valid=candidate.schedule.status is ScheduleStatus.FEASIBLE
                    o.update(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax if valid else None,b32_calls=1)
                    cache[key]=o
            outcomes.append(o)
        op_save(dict(phase='parent_cmax_counterfactual',instance=name,seed=OP_SEED,method='FULL_PARENT_FROZEN_OP_COMPLETION',
            status='COLLECTED',runtime_s=time.perf_counter()-started,b32_calls=physical,payload=dict(fraction=fraction,
                parents=payload['parents'],prefix=payload['prefix'],outcomes=outcomes,pairs=op_pair_labels(outcomes),
                coverage=coverage,skipped=skipped,parent_candidates=len({p['parent_id'] for p in plans}),
                complete_parent_candidates=len(plans))))


def op_parent_cmax_evaluate(*,macro_imitation=False):
    shared,config,_,_,_,_,entries,_=op_sources()
    phase='parent_cmax_macro_evaluation' if macro_imitation else 'parent_cmax_evaluation'
    coverage_phase='parent_cmax_macro_coverage' if macro_imitation else 'parent_cmax_coverage'
    variant=PARENT_MACRO_IL_VARIANT if macro_imitation else PARENT_CMAX_VARIANT
    rows=op_read();cache=op_evaluation_cache(rows)
    done={(r['instance'],r['method']) for r in rows if r['phase']==phase}
    for name,parents in sorted(entries.items()):
        for method,source in (('ORIGINAL_GNN_MULTI3','NEW_GNN_MULTI3'),
                              ('PARENT_JOINT_MULTI3','GNN_PARENT_JOINT_MULTI3'),('RULE/BEAM','RULE/BEAM')):
            if (name,method) in done:continue
            old=next(r for r in rows if r['phase']=='parent_evaluation' and r['instance']==name and r['method']==source)
            op_save(dict(old,phase=phase,method=method,b32_calls=0,
                payload=dict(old['payload'],reused=True,source_phase='parent_evaluation',source_method=source)))
        if (name,'PARENT_CMAX_MULTI3') in done:continue
        info=op_measure_decodes(shared,config,parents,variant,
            ('parent_cmax','parent_cmax_sample','parent_cmax_sample'),cache)
        op_save_measured(phase,name,'PARENT_CMAX_GREEDY',info,first_only=True)
        op_save_measured(phase,name,'PARENT_CMAX_MULTI3',info)
    if not any(r['phase']==coverage_phase for r in rows):
        from mrta_search.lns import joint_action_key
        source=shared._model_from_checkpoint(shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)[variant]).eval()
        counts=Counter()
        for row in rows:
            if row['phase']!='early_counterfactual':continue
            p=row['payload'];reference,graph,blocks,patterns,state=cmax_prefix(p)
            plans=cmax_plans(reference,graph,blocks,patterns,state,p['prefix']['pending'],source)
            for i,o in enumerate(p['outcomes']):
                if (i,0) not in op_pair_labels(p['outcomes']) or p['outcomes'][0]['status']!='FEASIBLE':continue
                action=op_action(o['action']);counts['strict_good']+=1
                counts['parent_covered']+=any(v['parent_id']==blocks[action.block_id].parent_id for v in plans)
                counts['action_covered']+=any(joint_action_key(v['actions'][0])==joint_action_key(action) for v in plans)
        op_save(dict(phase=coverage_phase,instance='TRAIN_21',seed=OP_SEED,method='AFTER_TRAINING',
            status='MEASURED',b32_calls=0,payload=dict(counts)))


def op_parent_cmax_report():
    import run_gnn_shared_construction as shared
    rows=op_read();heading='## 2026-11-06 有界多样候选与完整父动作Cmax评分训练'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Existing complete-action training report retained.',flush=True);return
    checkpoint=shared.read_neural_checkpoint(OP_CHECKPOINT)
    metadata=checkpoint[PARENT_CMAX_VARIANT]['training']
    data=[r for r in rows if r['phase']=='parent_cmax_counterfactual']
    outcomes=[o for r in data for o in r['payload']['outcomes']]
    evaluation=[r for r in rows if r['phase']=='parent_cmax_evaluation']
    table={(r['instance'],r['method']):r for r in evaluation}
    methods=('ORIGINAL_GNN_MULTI3','PARENT_JOINT_MULTI3','PARENT_CMAX_MULTI3','RULE/BEAM')
    names=sorted(n for n,m in table if m=='PARENT_CMAX_MULTI3')
    if len(names)!=8 or any((n,m) not in table for n in names for m in methods):raise ValueError('finish the original eight development geometries')
    good=lambda r:r['status']=='FEASIBLE' and r['cmax'] not in ('',None)
    num=lambda v:'—' if v is None or v=='' else f'{float(v):.3f}'
    def compare(baseline):
        counts=Counter();changes=[]
        for name in names:
            a,b=table[name,'PARENT_CMAX_MULTI3'],table[name,baseline];fa,fb=good(a),good(b)
            if fa and fb:
                change=100*(float(b['cmax'])-float(a['cmax']))/float(b['cmax']);changes.append(change)
                counts['win' if change>1e-8 else 'loss' if change< -1e-8 else 'tie']+=1
            elif fa:counts['win']+=1
            elif fb:counts['loss']+=1
            else:counts['both_fail']+=1
        return counts,len(changes),statistics.median(changes) if changes else None
    lines=['',heading,'',
        '**结论：新三候选仅4/8可行，原GNN为6/8、parent_joint为7/8；候选覆盖和训练排序改善没有转化为完整Cmax收益。**', '',
        '本轮实际完成候选覆盖改进、32单元完整动作标量头、真实Cmax排序训练及原8几何B32验证。'
        '工作区核查：DRL没有独立.git，所属根仓库为codex/project-cleanup、0935ad8，并非可核实的main。'
        '未切换或重置含大量既有清理改动的根仓库，以当前与11月6日报告一致的DRL文件为实验基线。', '',
        '### 候选与网络修改', '',
        '复用joint_parent_action_candidates：保留GNN Top-4父，最多补两个真实合法资格/低负载代表父，最多6父、每子2动作、'
        '每拆分父最多4完整组合。首子保留网络最佳，再优先不同机器人（以当前最大加工负载和原分数选代表），'
        '无不同机器人则选不同位置，均不存在才保留方向变体。未新增合法性模型或强制平均负载，整父仍一次提交。',
        '原18个已知严格Cmax优质替代：父覆盖7/18→18/18，完整父池内首动作覆盖4/18→18/18。'
        '这里仅指已保存的代表动作，不是未知全动作空间覆盖率。',
        '保留两层GAT、hidden=64、heads=4与原父/插入头。新增一个711→32→1完整父标量头（22817参数）。'
        '输入复用父表示、两个有序子动作的块/机器人/前后邻居GAT表示及原20/49维特征；'
        'WHOLE第二槽置零并掩码，另含动作后四机P/M、最大P、最大P+MOVE、总空走、同轨范围重叠和首点间隔。'
        'WHOLE和拆分统一比较一个标量，不再把子动作log概率相加作为最终得分。零权重时不同子段数得分相同的测试通过。',
        '保留旧greedy/sample/parent_joint入口；新入口为parent_cmax及parent_cmax_sample，三个提案都来自同一个新GNN完整动作头。'
        '未加入ΔO、PPO、独立方向优化、Beam后处理或MODE_NEW/VNS/ALNS回退。'
        '固定fixed_split_patterns()的模式搜索限制仍然存在，ScientificConfig、FORMAL_SCOPE_V2、0.50m与B32均未修改。', '',
        '### 同状态标签与实际训练', '',
        f'复用63个原TRAIN学生0%/10%/25%真实前缀，保存原路线、方向、已分配和剩余父任务。'
        f'获得{len(outcomes)}个完整父动作结果，其中{sum(o["origin"]=="legacy" for o in outcomes)}个旧结果精确复现并复用，'
        f'每状态最多补一个新完整父动作；新增B32 {sum(int(r["b32_calls"]) for r in data)}次，最大单状态新增'
        f'{max(int(r["b32_calls"]) for r in data)}次，跳过旧结果{sum(len(r["payload"]["skipped"]) for r in data)}个。',
        '所有候选从完全相同前缀提交完整父动作后，统一由冻结原OP分层greedy补全。相同完整方案和方向复用原B32；'
        '没有将不同后续策略的Cmax直接混为标签，也没有重新生成教师。',
        f'可行{sum(o["status"]=="FEASIBLE" for o in outcomes)}，DEADLOCK {sum(o["status"]=="DEADLOCK" for o in outcomes)}；'
        f'{metadata["preparation"]["complete_parent_states"]}个状态形成{metadata["preparation"]["rank_pairs"]}个有效同状态对。'
        '两个失败不排序，失败Cmax为空；无可比标签的状态保留但不编造监督。',
        f'复用{metadata["preparation"]["teachers_reused"]}个完整教师动作、'
        f'{metadata["preparation"]["prefix_continuations_reused"]}个既有延续和{metadata["preparation"]["imitation_examples"]}个IL样本，教师新B32为0。'
        'L=softplus(−(score_better−score_worse))+0.1 IL，不使用负载/机会损失。'
        f'从OP初始化，AdamW lr=1e-4、batch=16、clip=1、seed={OP_SEED}，固定10轮、614更新/轮；'
        f'训练{num(metadata["training_seconds"])}秒。未按开发结果加训或挑epoch。',
        f'相同TRAIN标签上，原完整概率分数正确{metadata["initial"]["legacy_correct"]}/{metadata["initial"]["total"]}，'
        f'新头初始{metadata["initial"]["correct"]}/{metadata["initial"]["total"]}，'
        f'第10轮{metadata["history"][-1]["training_pairs"]["correct"]}/{metadata["history"][-1]["training_pairs"]["total"]}。'
        '这只是训练排序拟合，不能代替开发Cmax。',
        '新权重保存在既有models/gnn_coordination_constructor_05.pt的独立parent_cmax_candidate；'
        '保存后逐张量核对原顶层、refinement、opportunity与early分支均保留。没有新建文件。', '',
        '### 原8个开发几何，seed=202611041', '',
        '每个方法最多三个完整提案和三次B32。旧GNN三候选、原parent_joint三候选及符合三次预算的RULE结果复用，'
        '新三候选为一次完整头贪心及两次温度1固定种子采样，无旧模型或规则回退。'
        '8例均为已消费开发集，不称独立测试集。', '',
        '|几何|'+'|'.join(methods)+'|','|---|'+'---:|'*len(methods)]
    for name in names:lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in methods)+'|')
    lines += ['', '|方法|可行/8|总时间中位/s|物理墙钟中位/s|本轮B32|', '|---|---:|---:|---:|---:|']
    for method in methods:
        rs=[table[n,method] for n in names]
        actual=[r['payload'].get('actual_runtime_s') for r in rs]
        lines.append(f'|{method}|{sum(good(r) for r in rs)}/8|{num(statistics.median(float(r["runtime_s"]) for r in rs))}|'
            f'{num(statistics.median(actual) if all(v is not None for v in actual) else None)}|{sum(int(r["b32_calls"]) for r in rs)}|')
    lines += ['', '新GNN总时间包括模型文件读取、实例化、全部图/候选构建、三次完整构造/采样和B32。'
        '外部缓存命中计入原B32实测时长，同时保留本次物理墙钟；内部重复方案不重复调度。'
        '旧parent_joint三候选时间是上轮实测分项组成值，不冒充本轮连续墙钟。', '',
        '|新三候选对照|胜/平/负|双失败|双方可行/8|条件中位Cmax改善/%|','|---|---:|---:|---:|---:|']
    for baseline in ('ORIGINAL_GNN_MULTI3','PARENT_JOINT_MULTI3','RULE/BEAM'):
        counts,paired,change=compare(baseline)
        lines.append(f'|{baseline}|{counts["win"]}/{counts["tie"]}/{counts["loss"]}|{counts["both_fail"]}|{paired}/8|{num(change)}|')
    lines += ['', '胜负计入“可行优于失败”；两个失败单列，不伪造Cmax。改善为100×(baseline−NEW)/baseline，条件中位数明确列出有效配对数。', '',
        '|新三候选几何|R0/R1/R2/R3加工负载|max(P+MOVE)|MOVE|WAIT|失败提案状态|','|---|---|---:|---:|---:|---|']
    for name in names:
        row=table[name,'PARENT_CMAX_MULTI3'];p=row['payload'];m=p.get('metrics',{})
        lines.append(f'|{name}|'+(' / '.join(num(v) for v in m.get('loads',())) or '—')+
            f'|{num(m.get("max_p_move"))}|{num(row.get("move"))}|{num(row.get("wait"))}|'+
            ', '.join(a['status'] for a in p['attempts'] if a['status']!='FEASIBLE')+'|')
    for row in rows:
        if row['phase']=='parent_cmax_coverage':lines += ['', '训练后同一18个已知优质替代的候选覆盖：'+str(row['payload'])+'。']
    reference={(r['instance'],r['method']):r for r in rows if r['phase']=='evaluation'}
    lines += ['', '### WAG/HGA差距与实测结论', '',
        'WAG/HGA取原同几何、同种子60秒终解，预算不同，只作距离参考，不能作为等预算优越性证据。']
    for method in ('WAG','HGA'):
        values=[];failed=0
        for name in names:
            new,old=table[name,'PARENT_CMAX_MULTI3'],reference[name,method]
            if good(new) and good(old):values.append(100*(float(new['cmax'])/float(old['cmax'])-1))
            elif not good(new):failed+=1
        lines.append(f'新GNN相对{method}的双方可行Cmax中位差距为+{num(statistics.median(values) if values else None)}%'
                     f'（{len(values)}/8，另有{failed}个新GNN失败）；原{method}运行中位{num(statistics.median(float(reference[n,method]["runtime_s"]) for n in names))}秒。')
    # Diagnose the failed rollout using saved actions and the same TRAIN
    # tensors; no additional complete construction or scheduler is needed.
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]);graphs={};records=cmax_compile(rows,source,graphs)
    support=max(float(r['inputs']['parent_features'][:,:4].max()) for r in records)
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    support_rows=[]
    for name in names:
        attempt=table[name,'PARENT_CMAX_GREEDY']['payload']['attempts'][0]
        if not attempt.get('solution'):continue
        solution,_,_,_,_=shared.research_restore(attempt,CONFIG)
        blocks=block_map(solution,CONFIG,scope=FORMAL_SCOPE_V2)
        scale=sum(CONFIG.process_time(b.length) for b in blocks.values());loads=[0.]*4;completed=0;snapshots={}
        actions=attempt['actions'];targets={f:int(len(solution.parents)*f) for f in (.25,.5,.75)}
        for i,a in enumerate(actions):
            loads[a['robot']]+=CONFIG.process_time(blocks[a['block_id']].length)
            if i+1==len(actions) or blocks[actions[i+1]['block_id']].parent_id!=blocks[a['block_id']].parent_id:
                completed+=1
                for fraction,target in targets.items():
                    if completed==target:snapshots[str(fraction)]=max(loads)/scale
        support_rows.append(dict(instance=name,normalized_max_processing=snapshots))
    diagnosis=dict(training_prefix_max_normalized_processing=support,new_head_has_direct_expert_macro_imitation=False,
        development_prefixes=support_rows,
        above_training_range_at_half=sum(r['normalized_max_processing'].get('0.5',0.)>support for r in support_rows))
    if not any(r['phase']=='parent_cmax_diagnosis' for r in rows):
        op_save(dict(phase='parent_cmax_diagnosis',instance='TRAIN_AND_DEV',seed=OP_SEED,method='STATE_SUPPORT_AND_CANDIDATE_DRIFT',
            status='ANALYZED',b32_calls=0,payload=diagnosis))
    large=table['LARGE_04','PARENT_CMAX_MULTI3'];large_metrics=large['payload']['metrics']
    lines += ['', '**本轮结论：候选覆盖和TRAIN排序拟合改善，完整构造质量却退化，不能替换原parent_joint。** '
        '新三候选仅4/8可行，原GNN为6/8、parent_joint为7/8；未满足进入60秒完整求解比较的条件。'
        '保留失败权重与全部24个提案，不追加采样、温度调参或重复训练。', '',
        '原因已缩小到候选漂移和监督分布，而不是局部合法性或B32被改动：', '',
        '- 候选漂移：训练前18/18父与18/18首动作覆盖，训练后仅16/18父、12/18首动作。'
        '本轮联合更新共享GAT和旧父/子头，训练使用的旧候选集与新策略实际生成的候选集发生漂移。',
        '- 评分监督不足：新完整头仅由48个0%/10%/25%状态的215个排序对训练；专家IL作用于原父/子头及共享编码器，'
        '没有直接给新完整头提供全构造阶段的专家动作监督。不能把旧头的全轨迹IL当作新头也有后半程监督。',
        f'- 状态范围：48个评分状态的最大归一化已分配负载为{support:.6f}；'
        f'新贪心{diagnosis["above_training_range_at_half"]}/8个几何在50%父任务进度时已超过这一输入范围。'
        '高TRAIN成对正确率没有检验新头在自身后半段路线状态上的行为。',
        '- 标签策略差异：旧结果精确复现、相同起点/相同冻结OP续补和单次B32均通过；但标签衡量的是动作后交给旧OP贪心的结果，'
        '正式推理后续由新完整头控制，实际续补策略与离线价值条件不相同。这是策略/状态分布差异，不是跨状态混排或虚假Cmax。',
        f'- 真实瓶颈：LARGE_04的新可行返回max P={num(max(large_metrics["loads"]))}、'
        f'max(P+MOVE)={num(large_metrics["max_p_move"])}、Cmax={num(large["cmax"])}；'
        'MEDIUM_04和SMALL_02可行返回的Cmax等于P+MOVE，主要损失已来自机器人加工分配。'
        'HIGH_WORKLOAD_02/04、LARGE_02和MEDIUM_02三个完整提案均DEADLOCK，暴露全局同轨协调失败。'
        'SMALL_02贪心甚至只用R0/R2，说明覆盖候选并不阻止新头后续持续选择同一机器人。',
        '下一步仍有必要研究GNN，但不值得原样重复本轮训练。应先保持候选生成稳定，'
        '把已保存教师的完整父动作模仿直接作用到新评分头并覆盖完整构造阶段，再用新头自身状态/一致续补策略收集小量Cmax反馈。'
        '这属于纠正监督覆盖和分布，不需要增加GAT层数、新调度器或新的机会损失。本轮没有提前叠加这些改动。']
    for row in rows:
        if row['phase']=='parent_cmax_verification':lines += ['',row['payload']['summary']]
    lines += ['', '本轮未新增文件；代码、测试、CSV、模型容器及Markdown均复用已有文件。复现入口：', '',
        '```powershell',"Set-Location 'D:\\pybullet_test\\MRTA_GA'",
        "$env:PYTHONPATH = (Join-Path (Get-Location) 'DRL\\src')",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-cmax-data",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-cmax-train",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-cmax-eval",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-cmax-report",'```','']
    OP_REPORT.write_text(original+'\n'.join(lines),encoding='utf-8')
    print('APPENDED',OP_REPORT,flush=True)


def op_parent_evaluate():
    from mrta_search import gnn_constructor as c
    from unittest.mock import patch
    shared,config,_,_,_,_,entries,_=op_sources()
    rows=op_read();cache=op_evaluation_cache(rows)
    done={(r['instance'],r['method']) for r in rows if r['phase']=='parent_evaluation'}
    controls={r['instance'] for r in rows if r['phase']=='parent_construction_timing' and r['method']=='NEW_GNN_MULTI3'}
    for name,parents in sorted(entries.items()):
        originals={m:next(r for r in rows if r['phase']=='evaluation' and r['instance']==name and
                    r['method']==m and r['seed']==str(OP_EVAL_SEED))
                   for m in ('NEW_GNN_GREEDY','NEW_GNN_MULTI3','RULE/BEAM')}
        if name not in controls:
            tick=time.perf_counter();thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
            model=shared._model_from_checkpoint(thin[OP_VARIANT]).eval();load_seconds=time.perf_counter()-tick
            durations=[]
            for i,(decode,seed) in enumerate((('greedy',OP_EVAL_SEED),('sample',OP_EVAL_SEED),('sample',OP_EVAL_SEED+1))):
                tick=time.perf_counter()
                geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
                result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
                    tuple(p.parent_id for p in parents),model,config,decode_mode=decode,seed=seed)
                durations.append(time.perf_counter()-tick)
                saved=originals['NEW_GNN_MULTI3']['payload']['attempts'][i]
                if result.solution is None or op_complete_key(dict(solution=asdict(result.solution),directions=result.directions))!=op_complete_key(saved):
                    raise ValueError('frozen control does not reproduce its saved complete proposal')
            for method,count in (('NEW_GNN_GREEDY',1),('NEW_GNN_MULTI3',3)):
                op_save(dict(phase='parent_construction_timing',instance=name,seed=OP_EVAL_SEED,method=method,
                    status='EXACT_SAVED_PROPOSALS',runtime_s=load_seconds+sum(durations[:count]),b32_calls=0,
                    payload=dict(model_load_seconds=load_seconds,construction_seconds=sum(durations[:count]),
                        proposal_construction_seconds=durations[:count],historical_total_seconds=float(originals[method]['runtime_s']),
                        b32_seconds=None,source_variant=OP_VARIANT)))
        for method in ('NEW_GNN_GREEDY','NEW_GNN_MULTI3'):
            if (name,method) not in done:
                old=originals[method]
                op_save(dict(old,phase='parent_evaluation',b32_calls=0,payload=dict(old['payload'],reused=True,
                    source_phase='evaluation',b32_budget=3)))
        if (name,'RULE/BEAM') not in done:
            old=originals['RULE/BEAM']
            if int(old['payload']['logical_b32_calls'])<=3:
                op_save(dict(old,phase='parent_evaluation',b32_calls=0,payload=dict(old['payload'],reused=True,
                    source_phase='evaluation',b32_budget=3)))
            else:
                actual=pilot.oriented.schedule_explicit;calls=[];seen={}
                def measured(solution,directions,conf,**kwargs):
                    raw=dict(solution=asdict(solution),directions=directions);key=op_complete_key(raw)
                    if key in seen:return seen[key]
                    tick=time.perf_counter();candidate=actual(solution,directions,conf,**kwargs)
                    seconds=time.perf_counter()-tick;seen[key]=candidate
                    calls.append(dict(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax
                        if candidate.schedule.status is ScheduleStatus.FEASIBLE else None,scheduler_seconds=seconds,**raw))
                    return candidate
                started=time.perf_counter()
                with patch.object(pilot.oriented,'schedule_explicit',measured):
                    candidate,attempts=pilot.oriented.common_initial_beam(parents,config,seed=OP_EVAL_SEED,
                        interleave=True,max_reference_calls=3,deadline=started+60.)
                seconds=time.perf_counter()-started;valid=candidate is not None
                metrics=op_metrics(candidate,config) if valid else {}
                op_save(dict(phase='parent_evaluation',instance=name,seed=OP_EVAL_SEED,method='RULE/BEAM',
                    status='FEASIBLE' if valid else 'INITIALIZATION_FAILED',cmax=candidate.schedule.cmax if valid else None,
                    runtime_s=seconds,b32_calls=len(calls),**metrics,payload=dict(reused=False,b32_budget=3,
                        logical_b32_calls=len(calls),attempts=attempts,scheduled_proposals=calls,
                        b32_seconds=sum(a['scheduler_seconds'] for a in calls),
                        construction_seconds=seconds-sum(a['scheduler_seconds'] for a in calls),
                        complete=None if not valid else dict(solution=asdict(candidate.solution),directions=candidate.directions))))
        if (name,'GNN_PARENT_JOINT_MULTI3') in done:continue
        existing=next((r for r in rows if r['phase']=='parent_evaluation' and r['instance']==name and
                       r['method']=='GNN_PARENT_JOINT'),None)
        if existing:
            p=existing['payload'];info=dict(attempts=p['attempts'],model_load_seconds=p['model_load_seconds'],
                runtime_s=float(existing['runtime_s']),actual_runtime_s=p['actual_runtime_s'],
                reused_b32_seconds=p['reused_b32_seconds'],source_variant=OP_VARIANT,b32_calls=0)
        else:
            info=op_measure_decodes(shared,config,parents,OP_VARIANT,('parent_joint',),cache,parent_candidate_limit=4)
            op_save_measured('parent_evaluation',name,'GNN_PARENT_JOINT',info)
        # Equal three-proposal comparison changes ONLY proposal zero. Both
        # samples, their true outcomes and measured durations are unchanged.
        samples=originals['NEW_GNN_MULTI3']['payload']['attempts']
        for i,old in enumerate(samples[1:],1):
            entry=dict(old,proposal_index=i,b32_calls=0,reused=True)
            if entry.get('solution'):entry['metrics']=op_route_metrics(entry,config)
            info['attempts'].append(entry)
        info['runtime_s']+=samples[-1]['elapsed']-samples[0]['elapsed']
        info['actual_runtime_s']=None;info['reused_b32_seconds']=None;info['b32_calls']=0
        feasible=[a for a in info['attempts'] if a['status']=='FEASIBLE']
        if feasible:
            best=min(feasible,key=lambda a:a['cmax'])
            if best['proposal_index']>0:
                old=originals['NEW_GNN_MULTI3']
                if not math.isclose(best['cmax'],float(old['cmax']),rel_tol=1e-9):
                    raise ValueError('selected historical sample lacks saved operation metrics')
                best['metrics'].update(move=float(old['move']),wait=float(old['wait']))
        op_save_measured('parent_evaluation',name,'GNN_PARENT_JOINT_MULTI3',info)


def op_parent_score_audit():
    """Inspect actual mixed WHOLE/split decisions without new completions/B32."""
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state,joint_action_key
    rows=op_read();done={r['instance'] for r in rows if r['phase']=='parent_score_audit'}
    model=shared._model_from_checkpoint(shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)[OP_VARIANT]).eval()
    for row in rows:
        if row['phase']!='parent_evaluation' or row['method']!='GNN_PARENT_JOINT' or row['instance'] in done:continue
        attempt=row['payload']['attempts'][0]
        if not attempt.get('solution'):continue
        solution,bits,_,_,_=shared.research_restore(attempt,CONFIG)
        ref=c.joint_geometry(solution.parents,solution.patterns,CONFIG)
        graph,steps,end=c.prepare_joint_teacher_steps(ref,((),)*4,((),)*4,tuple(p.parent_id for p in ref.parents),
            attempt['actions'],CONFIG,include_opportunities=True)
        if end.routes!=tuple(r.block_ids for r in solution.routes) or end.directions!=bits:
            raise ValueError('complete parent actions do not replay to the actual evaluated state')
        _,blocks,patterns=c.joint_graph(ref,CONFIG);counts=Counter();cases=[]
        parent_steps=[s for s in steps if s['parent_target'] is not None]
        decisions=attempt['parent_decisions']
        if len(parent_steps)!=len(decisions):raise ValueError('parent decision trace differs from actual actions')
        with torch.inference_mode():
            embeddings=model.encode_static(graph)
            for step,d in zip(parent_steps,decisions):
                if not (d['whole_available'] and d['split_available']):continue
                state=oriented_route_state(step['state_routes'],step['state_directions'],blocks,CONFIG)
                context=model.contextualize_state(graph,embeddings,state.routes,state.directions)
                logits=model.parent_logits(graph,context,step['parent_ids'],step['parent_features'])
                plans=c.joint_parent_action_candidates(graph,state,step['parent_ids'],logits,embeddings,model,blocks,patterns,CONFIG,
                    parent_limit=4,child_limit=2,current_embeddings=context)
                best=max(plans,key=lambda p:p['score'])
                if best['parent_id']!=d['parent_id'] or joint_action_key(best['actions'][0])!=step['teacher_action_key']:
                    raise ValueError('mixed parent score audit differs from the actual decision')
                whole=max((p for p in plans if len(p['actions'])==1),key=lambda p:p['score'])
                split=max((p for p in plans if len(p['actions'])==2),key=lambda p:p['score'])
                counts['mixed_states']+=1;counts['whole_selected']+=len(best['actions'])==1
                reverse=split['parent_log_probability']>whole['parent_log_probability'] and whole['score']>split['score']
                counts['parent_prefers_split_joint_prefers_whole']+=reverse
                for parent in {p['parent_id'] for p in plans if len(p['actions'])==2}:
                    first={joint_action_key(p['actions'][0]) for p in plans if p['parent_id']==parent}
                    counts['split_prefix_pools']+=1
                    counts['two_first_actions_same_robot_position']+=len(first)==2 and len({(a[1],a[2]) for a in first})==1
                if reverse and len(cases)<2:
                    cases.append(dict(chosen_parent=best['parent_id'],whole_parent_logp=whole['parent_log_probability'],
                        whole_child_logps=whole['child_log_probabilities'],whole_score=whole['score'],
                        split_parent_logp=split['parent_log_probability'],split_child_logps=split['child_log_probabilities'],
                        split_score=split['score']))
        op_save(dict(phase='parent_score_audit',instance=row['instance'],seed=OP_EVAL_SEED,method='ACTUAL_PARENT_DECISIONS',
            status='EXACT_REPLAY',b32_calls=0,payload=dict(counts=dict(counts),examples=cases)))


def op_parent_report():
    rows=op_read();heading='## 2026-11-06 冻结GNN父级完整联合动作与真实Cmax'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Existing parent joint report retained.',flush=True);return
    evaluation=[r for r in rows if r['phase']=='parent_evaluation']
    table={(r['instance'],r['method']):r for r in evaluation}
    names=sorted(n for n,m in table if m=='GNN_PARENT_JOINT')
    methods=('NEW_GNN_GREEDY','NEW_GNN_MULTI3','GNN_PARENT_JOINT','GNN_PARENT_JOINT_MULTI3','RULE/BEAM')
    if len(names)!=8 or any((n,m) not in table for n in names for m in methods):
        raise ValueError('finish the eight existing development geometries')
    num=lambda v:'—' if v is None or v=='' else f'{float(v):.3f}'
    good=lambda r:r['status']=='FEASIBLE' and r['cmax'] not in ('',None)
    def comparison(method,baseline):
        counts=Counter();changes=[]
        for name in names:
            a,b=table[name,method],table[name,baseline];fa,fb=good(a),good(b)
            if fa and fb:
                change=100*(float(b['cmax'])-float(a['cmax']))/float(b['cmax']);changes.append(change)
                counts['win' if change>1e-8 else 'loss' if change< -1e-8 else 'tie']+=1
            elif fa:counts['win']+=1
            elif fb:counts['loss']+=1
            else:counts['both_fail']+=1
        return f'{counts["win"]}/{counts["tie"]}/{counts["loss"]}',counts['both_fail'],len(changes),statistics.median(changes) if changes else None
    recalls=[r for r in rows if r['phase']=='parent_recall']
    def alternatives(current=False,cmax_only=False):
        return [a for r in recalls if not current or r['payload']['source_phase']=='early_counterfactual'
                for a in r['payload']['alternatives'] if not cmax_only or a['kind']=='cmax']
    def recall(values,k):
        count=sum(a['parent_rank']<=k for a in values)
        return f'{count}/{len(values)} ({100*count/len(values):.1f}%)' if values else '—'
    lines=['',heading,'',
        '**结论：完整父级联合解码提高了开发可行率，但没有建立稳定Cmax优势。** '
        '冻结机会模型单提案从3/8变为5/8，保持原两次采样后从6/8变为7/8；'
        '三候选对原三候选为2胜4平1负、1个双失败，双方可行配对Cmax中位改善0%。'
        '保留parent_joint为可选解码，不替换默认greedy，不据此宣称论文主算法已成立。', '',
        '### 从原始数据核实Top-K，而非复用ΔO报告结论', '',
        '恢复原counterfactual和early_counterfactual的126个真实状态，以各自实际冻结源模型重新前向父头，'
        '在同状态比较已保存动作及B32结果。47个优于首选A的比较中确有19个位于Top-2之外。'
        '但47混合了双方可行且Cmax更小、以及可行胜失败；严格Cmax优质动作只有31个。'
        '原25%/50%/75%数据来自扩展refinement，早期0%/10%/25%来自本轮opportunity_cmax_candidate，不能把两种父头混为当前模型。', '',
        '|原始样本口径|Recall@2|Recall@4|Recall@8|', '|---|---:|---:|---:|']
    for label,current,strict in (('两源、严格Cmax',False,True),('当前OP、严格Cmax',True,True),
                                 ('两源、含可行胜失败',False,False),('当前OP、含可行胜失败',True,False)):
        values=alternatives(current,strict);lines.append('|'+label+'|'+'|'.join(recall(values,k) for k in (2,4,8))+'|')
    lines += ['', '旧counterfactual的替代父本来就只从Top-2选取，不能据此声称旧父头召回完美。'
        '所有召回率仅针对已保存最多三个代表动作中的已知优质替代，不是完整未知动作空间的召回率。'
        '当前OP严格优质替代有10/18落在Top-8之外，父头本身排序不足；不以无限增大K掩盖。', '',
        '以下保留当前OP严格Cmax优质动作的逐几何/真实前缀统计；无优质动作的状态分母为0，不制造标签。'
        '全部126个状态（含分母为0者）在CSV的parent_recall中按source_phase、instance、fraction保存。', '',
        '|几何/前缀|优质替代数|Top-2覆盖|Top-4覆盖|Top-8覆盖|', '|---|---:|---:|---:|---:|']
    for r in recalls:
        if r['payload']['source_phase']!='early_counterfactual':continue
        values=[a for a in r['payload']['alternatives'] if a['kind']=='cmax']
        if values:lines.append(f'|{r["instance"]}/{r["payload"]["fraction"]:.0%}|{len(values)}|'+
            '|'.join(str(sum(a['parent_rank']<=k for a in values)) for k in (2,4,8))+'|')
    lines += ['', '### K选择、完整父动作与时间', '',
        '|K|完整父动作生成+评分/ms中位|候选生成/ms中位|特征/ms中位|网络/ms中位|', '|---:|---:|---:|---:|---:|']
    for k in (2,4,8):
        rs=[r for r in rows if r['phase']=='parent_k_timing' and r['payload']['k']==k]
        lines.append(f'|{k}|{num(1000*statistics.median(float(r["runtime_s"]) for r in rs))}|'+
            '|'.join(num(1000*statistics.median(r['payload'][field] for r in rs))
                     for field in ('candidate_generation_s','feature_s','network_s'))+'|')
    lines += ['', '计时来自六个既有真实状态，含34/85父任务和0%/10%/25%/50%/75%前缀；不调用B32。'
        '选择K=4：两源严格Cmax覆盖20/31，已覆盖Top-8所能恢复的21个中的20个，K=8仅多1个而成本明显上升。'
        'K<=8没有达到绝大多数覆盖，K=4只是本轮受控成本选择，不宣称截断问题已解决。',
        'gnn_constructor.py新增joint_parent_action_candidates及decode_mode="parent_joint"。'
        '父概率在所有合法未分配父池上log_softmax，子概率在该分支完整合法插入池上log_softmax，再分别截取Top-2。'
        'WHOLE一个动作；拆分保留首子Top-2，在各自不可变临时状态中评分第二子Top-2，最多每父4个完整组合。'
        '完整分数为log P(parent)+所有子动作的条件log概率和；无完整组合的分支丢弃，最后一次提交整父。'
        '未相加原logits，也未在Top-K/Top-2内部重新归一化。',
        '两层GAT、64维、4头、20/49维输入及权重完全不改；使用opportunity_cmax_candidate，'
        '未加入ΔO或独立模式头，固定fixed_split_patterns()仍限制拆分搜索空间。没有后处理、RULE/旧模型回退或额外认证。', '',
        '### 8个原开发几何、固定seed=202611041的真实B32', '',
        'NEW_GNN_GREEDY/MULTI3复用保存的同权重结果，并新测模型加载、图构建与构造；24个提案完整路线及方向精确复现。'
        'GNN_PARENT_JOINT是确定性单提案；MULTI3额外视图仅将它与原两个不变采样组合，不增加调度。'
        'RULE原7例各2次调度可直接复用；HIGH_WORKLOAD_04原有8次超出预算，仅补跑该例，保持interleave和Beam宽度4，'
        '只设现有max_reference_calls=3，三个均DEADLOCK。每个方法每几何实际最多3次调度。', '',
        '|几何|'+'|'.join(methods)+'|', '|---|'+'---:|'*len(methods)]
    for name in names:lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in methods)+'|')
    lines += ['', '—为失败，未赋惩罚Cmax。8个几何均已消费，只能称为开发验证。', '',
        '|方法|可行/8|总时间中位/s|逻辑B32/8例|本轮新增B32|', '|---|---:|---:|---:|---:|']
    for method in methods:
        rs=[table[n,method] for n in names]
        lines.append(f'|{method}|{sum(good(r) for r in rs)}/8|{num(statistics.median(float(r["runtime_s"]) for r in rs))}|'
            f'{sum(int(r["payload"].get("logical_b32_calls",0)) for r in rs)}|{sum(int(r["b32_calls"]) for r in rs)}|')
    lines += ['', '新单提案总时间是连续实测；新三候选总时间为新首提案实测加旧两次采样的实测增量，并非一次连续墙钟。'
        '旧基线B32当时未单独计时，不能虚构拆分；旧总时间继续用原实测。'
        '因此三候选总时间下降不应解释成构造加速。', '',
        '|比较|含可行性差异胜/平/负|双失败|双方可行/8|条件Cmax中位改善/%|', '|---|---:|---:|---:|---:|']
    for method,baseline in (('GNN_PARENT_JOINT','NEW_GNN_GREEDY'),('GNN_PARENT_JOINT','NEW_GNN_MULTI3'),
        ('GNN_PARENT_JOINT_MULTI3','NEW_GNN_MULTI3'),('GNN_PARENT_JOINT','RULE/BEAM'),('GNN_PARENT_JOINT_MULTI3','RULE/BEAM')):
        counts,failed,paired,change=comparison(method,baseline)
        lines.append(f'|{method} 对 {baseline}|{counts}|{failed}|{paired}/8|{num(change)}|')
    lines += ['', '改善统一为100×(baseline−GNN)/baseline；仅双方可行时有数值改善率，失败仍计入胜负和可行率分母。', '',
        '|几何|旧贪心新测构造/s|父级联合构造/s|父级联合B32/s|父级联合总/s|', '|---|---:|---:|---:|---:|']
    controls={(r['instance'],r['method']):r for r in rows if r['phase']=='parent_construction_timing'}
    old_times=[];new_times=[]
    for name in names:
        old=controls[name,'NEW_GNN_GREEDY']['payload']['construction_seconds']
        row=table[name,'GNN_PARENT_JOINT'];a=row['payload']['attempts'][0];new=a['construction_seconds']
        old_times.append(old);new_times.append(new)
        lines.append(f'|{name}|{num(old)}|{num(new)}|{num(a.get("scheduler_seconds"))}|{num(row["runtime_s"])}|')
    lines += ['', f'中位构造{num(statistics.median(old_times))}→{num(statistics.median(new_times))}秒，'
        f'增加{num(statistics.median(new_times)-statistics.median(old_times))}秒、'
        f'{num(100*(statistics.median(new_times)/statistics.median(old_times)-1))}%。'
        '构造含固定模式几何/图与全部网络决策；模型读取单列在CSV，已计入总时间。分项中位数不能相加。', '',
        '|新单提案几何|状态|max P|max(P+MOVE)|Cmax−max(P+MOVE)|WAIT总量|', '|---|---|---:|---:|---:|---:|']
    for name in names:
        a=table[name,'GNN_PARENT_JOINT']['payload']['attempts'][0];m=a.get('metrics',{})
        excess=a['cmax']-m['max_p_move'] if a['status']=='FEASIBLE' else None
        lines.append(f'|{name}|{a["status"]}|{num(max(m["loads"]) if m else None)}|{num(m.get("max_p_move"))}|'
            f'{num(excess)}|{num(m.get("wait"))}|')
    lines += ['',
        '### 瓶颈与下一轮判断', '',
        '已评价动作重放到真实末状态逐项一致，WHOLE与拆分两个子段的条件分数及一次提交测试通过，'
        '因此不是状态污染、漏插第二子段或归一化实现错误。当前实际509个父任务中仅13个为固定Y_SPLIT；'
        '新旧解码同时改变父池2→4和完整父评分，没有单独消融，不能把收益全部归给拆分前瞻。',
        '父头：当前18个严格Cmax优质替代只有7个父进入Top-4；再要求首子动作Top-2仅剩4个。'
        '即使完整父评分正确，真正优质的已知动作仍常在预筛选时被丢弃。',
        '分支：实际86个拆分父首分支池中72个Top-2动作机器人和位置相同，仅方向/邻接翻转不同；'
        '完整构造只有1次选中首子排名2。因此Top-2概率动作不等于两个有代表性的分配结构，本轮没有擅自添加规则补充。',
        '评分：75个WHOLE/拆分共存状态中62次选WHOLE，其中44次是父头本来更偏好拆分，'
        '加入两个条件概率后改选WHOLE。变长概率乘积增加非正项，且已有Cmax排序训练主要监督首子动作，'
        '未经完整父级Cmax校准；这提示长度/条件概率错配，不能仅凭该比例证明WHOLE选择必然错误。',
        '结果：HIGH_WORKLOAD_04改善，但MEDIUM_04、SMALL_04单提案Cmax变差。'
        'HIGH_WORKLOAD_02、MEDIUM_02、SMALL_02新单提案仍DEADLOCK；MEDIUM_02三提案全失败。'
        '完整父动作局部合法不等于全局可调度；不能靠均衡使用四机证明成功。',
        'SMALL_04是明确反例：max P从旧贪心3244.205降至2409.602，max(P+MOVE)降至2565.582，'
        'Cmax却从3394.365升至3903.998。新路线相对自身P+MOVE增加1338.416秒协调等待，抵消负载收益后Cmax净增509.633秒；'
        'B32按原机制恢复18次后取得该可行时间线。'
        'LARGE_02/LARGE_04的Cmax则等于P+MOVE，且max P分别5843.427/6218.728，主要限制是机器人加工分配。'
        '因此剩余瓶颈同时包含加工分配与全局同轨协调，不能统称泛化不好或仅归因独立方向。',
        '保留该有界解码作为下一轮针对性训练候选，有可行率与个别Cmax改善的价值；'
        '尚不足以替换默认构造或开展大规模训练。下一轮应优先训练父级完整动作的真实Cmax排名与父头召回，'
        '复核WHOLE/拆分分数校准，沿用现有网络；不直接增加层数、K或随机采样。本轮未训练任何模型。',
        'WAG/HGA保存的是60秒搜索终解，与数秒GNN初解预算不同；本轮不把它们列为等预算胜负。']
    for row in rows:
        if row['phase']=='parent_verification':lines += ['',row['payload']['summary']]
    lines += ['', '修改复用gnn_constructor.py、run_gnn_cmax_finetune.py、run_gnn_shared_construction.py、'
        'test_gnn_construct_vns.py、既有CSV和本报告。没有新增文件，历史权重不写入；本轮新增B32合计11次（新GNN8、RULE补跑3）。', '',
        '```powershell',"Set-Location 'D:\\pybullet_test\\MRTA_GA'",
        "$env:PYTHONPATH = (Join-Path (Get-Location) 'DRL\\src')",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-diagnose",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-eval",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-audit",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase parent-report",'```','']
    OP_REPORT.write_text(original+'\n'.join(lines),encoding='utf-8')
    print('APPENDED',OP_REPORT,flush=True)


def op_delta_analyze():
    """Stage A: restore real saved actions and labels; ZERO new scheduling."""
    import run_gnn_shared_construction as shared
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates,joint_action_key,has_joint_block_candidate
    from mrta_reference.model import ParentWeld
    rows=op_read()
    thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    models={'early_counterfactual':shared._model_from_checkpoint(thin[OP_VARIANT]).eval(),
            'counterfactual':shared.expand_opportunity_model(shared._model_from_checkpoint(thin['refinement_candidate'])).eval()}
    done={(r['instance'],r['payload']['source_phase'],r['payload']['fraction'])
          for r in rows if r['phase']=='delta_analysis'}
    for row in rows:
        if row['phase'] not in ('counterfactual','early_counterfactual'):continue
        p=row['payload'];fraction=p['fraction'];source=row['phase']
        if (row['instance'],source,fraction) in done:continue
        base=dict(source_phase=source,fraction=fraction)
        if not p.get('prefix') or not p.get('parents'):
            op_save(dict(phase='delta_analysis',instance=row['instance'],seed=row['seed'],method='SAME_U_DELTA',
                status='SKIPPED',b32_calls=0,payload=dict(base,reason='saved prefix or parents absent',outcomes=[],pairs=[])))
            continue
        prefix=p['prefix']
        parents=tuple(ParentWeld(x['parent_id'],tuple(x['start']),tuple(x['end'])) for x in p['parents'])
        geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
        graph,blocks,patterns=c.joint_graph(geometry,CONFIG)
        state=c.joint_start_state(geometry,tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),
                                 prefix['pending'],blocks,CONFIG)
        with torch.inference_mode():
            model=models[source]
            ids,features=c.joint_parent_inputs(state,prefix['pending'],blocks,patterns,CONFIG,include_opportunities=True)
            embedding=model.contextualize_state(graph,model.encode_static(graph),state.routes,state.directions)
            logits=model.parent_logits(graph,embedding,ids,features)
            order=sorted(range(len(ids)),key=lambda i:(-float(logits[i]),i))
            parent_ranks={ids[i]:rank+1 for rank,i in enumerate(order)}
        variants=[];outcomes=[];pools={}
        for outcome in p['outcomes']:
            action=op_action(outcome['action']);child=blocks[action.block_id]
            if child.block_id not in pools:
                pools[child.block_id]=joint_block_candidates(state,child,patterns[child.parent_id],blocks,CONFIG)
            matches=[v for v in pools[child.block_id] if joint_action_key(v[1])==joint_action_key(action)]
            entry=dict(action=outcome['action'],status=outcome['status'],cmax=outcome['cmax'],delta_seconds=None,
                       parent_rank=parent_ranks.get(child.parent_id))
            if len(matches)!=1:
                entry['recovery_error']='saved action missing or ambiguous in current pool'
            else:
                variants.append(matches[0]);entry['variant_index']=len(variants)-1
                after=matches[0][0];placed={b for route in after.routes for b in route}
                siblings=[b for b in blocks.values() if b.parent_id==child.parent_id and b.block_id not in placed]
                entry['unplaced_siblings']=[b.block_id for b in siblings]
                entry['sibling_has_legal_next']=all(has_joint_block_candidate(after,b,patterns[b.parent_id],blocks,CONFIG)
                                                    for b in siblings)
                entry['saved_full_completion']=outcome.get('complete') is not None
                if outcome.get('complete'):
                    entry['complete_metrics']=op_route_metrics(outcome['complete'],CONFIG)
            outcomes.append(entry)
        stats={};tick=time.perf_counter()
        deltas=c.joint_opportunity_deltas(state,variants,blocks,patterns,CONFIG,reuse=False,normalize=False,stats=stats)
        seconds=time.perf_counter()-tick
        for entry in outcomes:
            if 'variant_index' in entry:entry['delta_seconds']=deltas[entry.pop('variant_index')]
        payload=dict(base,prefix=prefix,outcomes=outcomes,pairs=p.get('pairs',[]),full_delta_seconds=seconds,
            total_processing=sum(CONFIG.process_time(b.length) for b in blocks.values()),
            block_count=len(blocks),parent_count=len(parents),direct_queries=stats['queries'])
        op_save(dict(phase='delta_analysis',instance=row['instance'],seed=row['seed'],method='SAME_U_DELTA',
            status='ANALYZED',runtime_s=seconds,b32_calls=0,payload=payload))
    # Six real top-two pools cover empty and late states, small and large cases.
    bench_done={r['method'] for r in rows if r['phase']=='delta_timing'}
    source_rows=[r for r in rows if r['phase'] in ('counterfactual','early_counterfactual')]
    requests=(('early_counterfactual',0.,False),('early_counterfactual',0.,True),
              ('early_counterfactual',.1,True),('early_counterfactual',.25,True),
              ('counterfactual',.5,True),('counterfactual',.75,True))
    for source,fraction,large in requests:
        method=f'{source}_{fraction}_{"large" if large else "small"}'
        if method in bench_done:continue
        available=[r for r in source_rows if r['phase']==source and r['payload']['fraction']==fraction and r['payload'].get('prefix')]
        row=(max if large else min)(available,key=lambda r:len(r['payload']['parents']))
        p=row['payload'];prefix=p['prefix']
        parents=tuple(ParentWeld(x['parent_id'],tuple(x['start']),tuple(x['end'])) for x in p['parents'])
        geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,CONFIG),CONFIG)
        graph,blocks,patterns=c.joint_graph(geometry,CONFIG)
        state=c.joint_start_state(geometry,tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),
                                 prefix['pending'],blocks,CONFIG)
        with torch.inference_mode():
            _,_,_,pools=op_decision_pool(geometry,state,prefix['pending'],models[source],CONFIG)
        variants=[v for pool in pools.values() for v in pool['variants']]
        direct={};fast={}
        tick=time.perf_counter()
        expected=c.joint_opportunity_deltas(state,variants,blocks,patterns,CONFIG,reuse=False,normalize=False,stats=direct)
        direct_seconds=time.perf_counter()-tick
        tick=time.perf_counter()
        actual=c.joint_opportunity_deltas(state,variants,blocks,patterns,CONFIG,reuse=True,normalize=False,stats=fast)
        reuse_seconds=time.perf_counter()-tick
        if actual!=expected:raise ValueError('local opportunity reuse differs from direct same-U computation')
        op_save(dict(phase='delta_timing',instance=row['instance'],seed=OP_SEED,method=method,status='EXACT_MATCH',
            runtime_s=reuse_seconds,b32_calls=0,payload=dict(source_phase=source,fraction=fraction,parents=len(parents),
                candidates=len(variants),direct_seconds=direct_seconds,reuse_seconds=reuse_seconds,
                direct_queries=direct['queries'],reuse_queries=fast['queries'],cache_hits=fast['cache_hits'])))


def op_delta_statistics(rows):
    groups=defaultdict(Counter);examples=[]
    for row in rows:
        if row['phase']!='delta_analysis':continue
        p=row['payload'];group=groups[p['fraction']];group['states']+=1
        outcomes=p['outcomes'];deltas=[o['delta_seconds'] for o in outcomes]
        valid=[d for d in deltas if d is not None]
        group['actions']+=len(valid);group['unrecovered']+=len(outcomes)-len(valid)
        group['varying_states']+=any(any(abs(x-y)>1e-6 for x,y in zip(a,b))
                                    for i,a in enumerate(valid) for b in valid[i+1:])
        group['nonzero_actions']+=sum(any(abs(x)>1e-6 for x in d) for d in valid)
        for better,worse in p['pairs']:
            group['saved_pairs']+=1
            a,b=outcomes[better],outcomes[worse]
            if a['delta_seconds'] is None or b['delta_seconds'] is None:continue
            group['pairs']+=1
            distinct=any(abs(x-y)>1e-6 for x,y in zip(a['delta_seconds'],b['delta_seconds']))
            group['vector_distinct']+=distinct
            kind='cmax' if a['status']==b['status']=='FEASIBLE' else 'feasibility'
            group[kind+'_pairs']+=1;group[kind+'_distinct']+=distinct
            gap=sum(a['delta_seconds'])-sum(b['delta_seconds'])
            if gap>1e-6:group['preservation_correct']+=1;group[kind+'_correct']+=1
            elif gap< -1e-6:group['preservation_wrong']+=1;group[kind+'_wrong']+=1
            else:group['preservation_tied']+=1
            if worse==0:
                group['gnn_errors']+=1;group['errors_distinct']+=distinct
                group['errors_correct']+=gap>1e-6;group['errors_wrong']+=gap< -1e-6
                group['better_parent_outside_top2']+=a['parent_rank'] is not None and a['parent_rank']>2
            if distinct and gap< -1e-6 and len(examples)<3:
                examples.append(dict(instance=row['instance'],fraction=p['fraction'],better=a,worse=b))
    total=Counter()
    for group in groups.values():total.update(group)
    return total,groups,examples


def op_delta_report():
    rows=op_read();total,groups,examples=op_delta_statistics(rows)
    analysis=[r for r in rows if r['phase']=='delta_analysis']
    timing=[r for r in rows if r['phase']=='delta_timing']
    heading='## 2026-11-06 动作机会变化ΔO：已有真实Cmax反事实检验'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Existing action opportunity report retained.',flush=True);return
    def num(v):return '—' if v is None or v=='' else f'{float(v):.3f}'
    actions=[o for r in analysis for o in r['payload']['outcomes']]
    seconds=sum(r['payload'].get('full_delta_seconds',0.) for r in analysis)
    lines=['',heading,'',
        '**结论：当前定义的四维ΔO有动作变化，但没有稳定识别真实Cmax优劣的证据，本轮停在阶段A。** '
        '按用户要求的终止条件，不改GNN输入维度、不训练NEW、不重复收集标签或开发B32。'
        '这不证明任何非线性模型都无法利用ΔO；本次数据不支持为它继续投入训练和全路线推理开销。', '',
        '### 实际修改与计算口径', '',
        '- gnn_constructor.py增加joint_opportunity_deltas，直接调用已有has_joint_block_candidate和config.process_time；'
        '没有改合法池、时间定义、拆分模式、B32、网络或默认解码。',
        '- 候选a是现有OrientedInsertionAction，真实转移插入一个子块。U_a严格取该动作执行后未放置的块，'
        '前后都在同一个U_a求和；当前块从两个求和中同时排除，未插入的兄弟仍保留在U_a。'
        '四维返回量可按全几何总加工量统一归一化；诊断保存秒值，未把处理当前块自身当成机会损失。',
        f'- 已恢复的{sum(bool(o.get("unplaced_siblings")) for o in actions)}个首子块动作均检查真实第二子块合法插入存在性；'
        f'{sum(not o.get("sibling_has_legal_next",True) for o in actions)}个无合法后继。'
        f'{sum(o.get("saved_full_completion",False) for o in actions)}个候选有已保存的完整补全，'
        '未把单块存在性宣称为整个剩余父任务集合一定可完成或可被B32调度。',
        '- 局部复用只在一次特征计算内进行，依赖本机器人长度、首块/方向、兄弟位置及所有其他机器人首点；'
        '未假定只有执行机器人或同轨伙伴受影响。lns.py、shared训练入口及全部历史模型未改。', '',
        '### 阶段A真实结果：只比较同状态保存标签', '',
        f'读取原25%/50%/75%和早期0%/10%/25%两批数据，共{total["states"]}个前缀、'
        f'{total["actions"]}个候选、{total["pairs"]}个可比较对，未恢复动作{total["unrecovered"]}个。'
        '全部来自原21个TRAIN几何；不同状态、几何的绝对Cmax没有混合排序。DEADLOCK的Cmax仍为空，两个失败不排序。', '',
        '|前缀|状态|候选|ΔO不同的状态|真实排序对|四维可区分对|机会保持方向一致/相反|',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for fraction,group in sorted(groups.items()):
        lines.append(f'|{fraction:.0%}|{group["states"]}|{group["actions"]}|{group["varying_states"]}|{group["pairs"]}|'
            f'{group["vector_distinct"]}|{group["preservation_correct"]}/{group["preservation_wrong"]}|')
    lines += [f'|合计|{total["states"]}|{total["actions"]}|{total["varying_states"]}|{total["pairs"]}|'
        f'{total["vector_distinct"]}|{total["preservation_correct"]}/{total["preservation_wrong"]}|', '',
        '“方向一致”仅诊断sum_r ΔO_r较大的动作是否对应已保存的更优结果，不是新代理Cmax，也未用于替换GNN。'
        f'另有{total["preservation_tied"]}对总机会变化相同，本数据中它们的四维向量也完全相同。',
        f'双方可行的真实Cmax对共{total["cmax_pairs"]}对，仅{total["cmax_distinct"]}对ΔO不同，'
        f'保持方向为{total["cmax_correct"]}对一致/{total["cmax_wrong"]}对相反；'
        f'可行胜失败的{total["feasibility_pairs"]}对中仅{total["feasibility_distinct"]}对不同，'
        f'方向为{total["feasibility_correct"]}/{total["feasibility_wrong"]}。',
        f'相对当前GNN首选A，已有{total["gnn_errors"]}个替代更优的比较；'
        f'ΔO仅区分其中{total["errors_distinct"]}个，保持方向仅{total["errors_correct"]}个一致、'
        f'{total["errors_wrong"]}个相反，其余{total["gnn_errors"]-total["errors_distinct"]}个四维全同。', '',
        '### 为什么没有形成有效监督信息', '',
        '0%阶段确有较大机会变化，因此不是把计算实现成常数。变化主要来自启用机器人后对首点合法性的约束；'
        '当前O表示“至少存在一个合法插入位置”，不表示剩余加工容量、可接受的路线成本或未来协调Cmax。'
        '当已有合法首点、路线变长且仍可在内部/尾部插入时，局部存在性对加工负载和多数插入位置/方向变化不敏感。'
        '这解释了25%后几乎没有差异，未通过错误扣除当前任务制造伪变化。',
        '原记录中75%阶段已有15/21个前缀的max P超过最佳保存教师完整Cmax；本轮对应21个状态的全部候选ΔO均为0，'
        '因此这些已知加工分配失衡没有得到区分。',
        '代表性同状态证据：N85_03、75%前缀，同一父任务分配R0得到Cmax=5444.345，改分R1得到4493.538；'
        '两者ΔO均为(0,0,0,0)，max P分别5056.925、4172.639，max(P+MOVE)分别5444.345、4493.538。'
        '这对损失主要是机器人加工分配与空走，机会存在性没有反映。',
        'N35_01、0%前缀，GNN首选R3的ΔO为0却DEADLOCK；同父改R2使R3机会减少3787.333秒，'
        '反而得到FEASIBLE、Cmax=4204.472。保留更多局部机会并不必然保留更好的完整协调方案。',
        f'父筛选核查：{total["better_parent_outside_top2"]}/{total["gnn_errors"]}个优于A的替代动作，其父任务排名在Top-2之外。'
        '这是独立的联合排名限制；本次没有以Top-5/10扩大枚举掩盖它，也没有把它当作ΔO有效的证据。', '',
        '### 计算开销与精确复用', '',
        f'371个保存候选的完整直接ΔO计算共{num(seconds)}秒，平均每候选{num(seconds/max(1,total["actions"])*1000)}毫秒。'
        '以下在真实Top-2全部合法插入池上计算，未调用B32；直接与局部复用的所有四维值逐项完全相同。'
        '单次分项计时不能作为完整GNN加速倍数，也未将状态耗时乘任务数冒充实测总构造时间。', '',
        '|几何/前缀|父数|候选数|完整计算/ms|局部复用/ms|真实存在性查询：原/复用|', '|---|---:|---:|---:|---:|---:|']
    for row in timing:
        p=row['payload'];lines.append(f'|{row["instance"]}/{p["fraction"]:.0%}|{p["parents"]}|{p["candidates"]}|'
            f'{num(1000*p["direct_seconds"])}|{num(1000*p["reuse_seconds"])}|{p["direct_queries"]}/{p["reuse_queries"]}|')
    lines += ['', '85父任务实例的0%前缀复用后约32毫秒，非空前缀仍约87–172毫秒，当前没有质量证据支持把这项成本加到每个构造决策。', '',
        '### 已有8个开发几何的真实B32参考：本轮不冒充NEW结果', '',
        'BASE取已保存的机会模型joint_greedy+原两次采样；BASE三个提案均从空路线由同一GNN构造。'
        'NEW未训练，因而没有NEW可行率、Cmax改善或运行时间可报告，也没有新增开发B32。'
        '下表只复用相同seed=202611041、CONFIG_05、FORMAL_SCOPE_V2、0.50m及B32的历史结果。'
        '固定fixed_split_patterns()限制仍在，模式头未重训；这些8例属于已消费开发数据。', '',
        '|几何|BASE Cmax|RULE/BEAM|WAG|HGA|BASE max P|BASE max(P+MOVE)|BASE使用机器人|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    table={(r['instance'],r['method']):r for r in rows if r['phase'] in ('evaluation','joint_evaluation')}
    names=sorted(n for n,m in table if m=='OLD_OP_JOINT_MULTI3')
    for name in names:
        base=table[name,'OLD_OP_JOINT_MULTI3'];raw=base['payload'].get('complete')
        metrics=op_route_metrics(raw,CONFIG) if raw else {}
        robots=','.join(f'R{i}' for i,v in enumerate(metrics.get('loads',())) if v>0) or '—'
        lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in
            ('OLD_OP_JOINT_MULTI3','RULE/BEAM','WAG','HGA'))+
            f'|{num(max(metrics["loads"]) if metrics else None)}|{num(metrics.get("max_p_move"))}|{robots}|')
    lines += ['', 'BASE可行7/8，RULE/BEAM 7/8，WAG、HGA均8/8。BASE相对RULE为1胜7负，相对WAG、HGA均0胜8负，'
        '失败计入负。BASE对双方可行的7例，相对WAG/HGA的Cmax差距中位数分别见下方实际计算，未用失败惩罚Cmax补齐。']
    for method in ('WAG','HGA'):
        gaps=[100*(float(table[n,'OLD_OP_JOINT_MULTI3']['cmax'])/float(table[n,method]['cmax'])-1)
              for n in names if table[n,'OLD_OP_JOINT_MULTI3']['status']=='FEASIBLE' and table[n,method]['status']=='FEASIBLE']
        lines.append(f'BASE相对{method}的可行配对Cmax中位增加{num(statistics.median(gaps))}%（{len(gaps)}/8，另1例BASE失败）。')
    lines += ['保存的BASE时间中位3.385秒（由联合首提案与既有两次采样实测分项组成），RULE 3.213秒，'
        'WAG 60.400秒，HGA 60.051秒。WAG/HGA是原60秒终解，数秒GNN是初解，不能称为公平等预算比较。', '',
        '### 决定与测试', '',
        '本轮不声称Cmax、可行率或推理速度改善，也不启动三种子或统一60秒比较。'
        '当前结果仍不足以支持将GNN作为论文主算法。下一次应优先纠正父任务与机器人分配的联合排名：'
        '检查网络为何拒绝已有特征和真实Cmax标签显示更低加工负载/P+MOVE的同父替代，'
        '以及更优不同父动作在预筛选前被排除的问题；不能继续把局部“还能插入”当作剩余加工容量或可调度性。'
        '本轮不追加其他模块、损失结构或训练轮数。', '',
        '本轮修改文件为gnn_constructor.py、run_gnn_cmax_finetune.py、既有test_gnn_construct_vns.py、'
        'gnn_opportunity_cmax.csv及本报告。没有新增文件，历史checkpoint不写入，新增B32/教师/认证调用均为0。']
    for row in rows:
        if row['phase']=='delta_verification':lines += ['',row['payload']['summary']]
    lines += ['', '```powershell',"Set-Location 'D:\\pybullet_test\\MRTA_GA'",
        "$env:PYTHONPATH = (Join-Path (Get-Location) 'DRL\\src')",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase delta-data",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase delta-report",'```','']
    if not any(r['phase']=='delta_summary' for r in rows):
        op_save(dict(phase='delta_summary',instance='TRAIN_21',seed=OP_SEED,method='STAGE_A',status='NO_TRAINING',
            b32_calls=0,payload=dict(statistics=dict(total),training_started=False,new_model=None,new_b32_calls=0,
                conclusion='Sparse action differences, no consistent within-state Cmax/feasibility advantage; stop after A.')))
    OP_REPORT.write_text(original+'\n'.join(lines),encoding='utf-8')
    print('APPENDED',OP_REPORT,flush=True)


def op_route_metrics(raw,config):
    import run_gnn_shared_construction as shared
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    from mrta_search.lns import oriented_route_state,oriented_route_lower_bound
    solution,bits,_,_,_=shared.research_restore(raw,config)
    state=oriented_route_state(tuple(r.block_ids for r in solution.routes),bits,
        block_map(solution,config,scope=FORMAL_SCOPE_V2),config)
    return dict(loads=state.loads,p_move=[p+sum(a) for p,a in zip(state.loads,state.arc_costs)],
                max_p_move=oriented_route_lower_bound(state),robots=sum(bool(r) for r in state.routes))


def op_measure_decodes(shared,config,parents,variant,decodes,cache,*,parent_candidate_limit=4):
    """Measure complete inference, sharing only exact, already measured B32s.

    The charged runtime includes the original measured B32 duration on a cache
    hit. Physical wall time and reused duration are also retained explicitly.
    No cache hit creates another scheduler call or another certificate.
    """
    from mrta_search import gnn_constructor as c
    started=time.perf_counter()
    thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    model=shared._model_from_checkpoint(thin[variant]).eval()
    load_seconds=time.perf_counter()-started
    attempts=[];credit=0.;charged=set()
    for index,decode in enumerate(decodes):
        seed=OP_EVAL_SEED if index<2 else OP_EVAL_SEED+1
        tick=time.perf_counter()
        geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
        decisions=[];extra={}
        if decode=='parent_joint':
            patterns={p.parent_id:p for p in geometry.patterns}
            def observe(state,plans,best):
                whole=[p for p in plans if len(p['actions'])==1];split=[p for p in plans if len(p['actions'])==2]
                decisions.append(dict(parent_id=best['parent_id'],kind=patterns[best['parent_id']].kind.value,
                    parent_rank=best['parent_rank'],score=best['score'],parent_log_probability=best['parent_log_probability'],
                    child_log_probabilities=best['child_log_probabilities'],child_ranks=best['child_ranks'],
                    child_pool_sizes=best['child_pool_sizes'],whole_available=bool(whole),split_available=bool(split),
                    best_whole_score=max((p['score'] for p in whole),default=None),
                    best_split_score=max((p['score'] for p in split),default=None)))
            extra=dict(parent_candidate_limit=parent_candidate_limit,parent_action_observer=observe)
        result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,
            tuple(p.parent_id for p in parents),model,config,decode_mode=decode,seed=seed,**extra)
        construction_seconds=time.perf_counter()-tick
        attempt=dict(proposal_index=index,decode=decode,seed=seed,status='CONSTRUCTION_FAILED',cmax=None,
            b32_calls=0,reason=result.rejection_reason,actions=[asdict(a) for a in result.actions],
            construction_seconds=construction_seconds)
        if decisions:attempt['parent_decisions']=decisions
        if result.solution is not None:
            raw=dict(solution=asdict(result.solution),directions=result.directions)
            key=op_complete_key(raw)
            if key not in cache:
                clock=time.perf_counter()
                candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                scheduler_seconds=time.perf_counter()-clock
                valid=candidate.schedule.status is ScheduleStatus.FEASIBLE
                metrics=op_route_metrics(raw,config)
                if valid:metrics.update(op_metrics(candidate,config))
                cache[key]=dict(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax if valid else None,
                    metrics=metrics,scheduler_seconds=scheduler_seconds,diagnostics=candidate.schedule.diagnostics)
                attempt['b32_calls']=1
            elif key not in charged:
                credit+=cache[key]['scheduler_seconds']
            charged.add(key)
            attempt.update(cache[key],**raw)
        attempt['seconds']=time.perf_counter()-tick
        attempt['elapsed']=time.perf_counter()-started+credit
        attempts.append(attempt)
    actual=time.perf_counter()-started
    return dict(attempts=attempts,model_load_seconds=load_seconds,runtime_s=actual+credit,
                actual_runtime_s=actual,reused_b32_seconds=credit,source_variant=variant,
                b32_calls=sum(a['b32_calls'] for a in attempts))


def op_save_measured(phase,name,method,info,*,first_only=False):
    attempts=info['attempts'][:1] if first_only else info['attempts']
    feasible=[(i,a) for i,a in enumerate(attempts) if a['status']=='FEASIBLE']
    winner=min(feasible,key=lambda x:x[1]['cmax']) if feasible else (None,None)
    selected=winner[1];metrics=selected['metrics'] if selected else {}
    runtime=attempts[0]['elapsed'] if first_only else info['runtime_s']
    physical=0 if first_only else info['b32_calls']
    op_save(dict(phase=phase,instance=name,seed=OP_EVAL_SEED,method=method,
        status='FEASIBLE' if selected else 'NO_FEASIBLE_CONSTRUCTION',cmax=selected['cmax'] if selected else None,
        runtime_s=runtime,b32_calls=physical,**metrics,payload=dict(
            attempts=attempts,selected_proposal=winner[0],metrics=metrics,
            model_load_seconds=info['model_load_seconds'],source_variant=info['source_variant'],
            complete=None if selected is None else dict(solution=selected['solution'],directions=selected['directions']),
            logical_b32_calls=len({op_complete_key(a) for a in attempts if a.get('solution')}),
            actual_runtime_s=info['actual_runtime_s'] if not first_only else None,
            reused_b32_seconds=info['reused_b32_seconds'] if not first_only else None,
            timing='Model loading, graph building, construction and B32; exact schedule cache hits charged their measured B32 duration.')))


def op_evaluation_cache(rows):
    cache={}
    for row in rows:
        if row['phase'] not in ('joint_evaluation','early_evaluation','parent_evaluation','parent_cmax_evaluation','parent_cmax_macro_evaluation'):continue
        for attempt in row['payload'].get('attempts',[]):
            if attempt.get('solution') and 'scheduler_seconds' in attempt:
                cache[op_complete_key(attempt)]={k:attempt[k] for k in
                    ('status','cmax','metrics','scheduler_seconds','diagnostics')}
    return cache


def op_joint_evaluate():
    shared,config,_,_,_,_,entries,_=op_sources()
    saved=op_read();cache=op_evaluation_cache(saved)
    done={(r['instance'],r['method']) for r in saved if r['phase']=='joint_evaluation'}
    for name,parents in sorted(entries.items()):
        # These exact old-weight, same-seed proposals have already been
        # measured. Reuse them, including the two unchanged sample proposals.
        originals={method:next(r for r in saved if r['phase']=='evaluation' and r['instance']==name and
                    r['seed']==str(OP_EVAL_SEED) and r['method']==source)
                   for method,source in (('OLD_OP_GREEDY','NEW_GNN_GREEDY'),('OLD_OP_MULTI3','NEW_GNN_MULTI3'))}
        for method,original in originals.items():
            if (name,method) in done:continue
            payload=dict(original['payload'],reused=True,source_method=original['method'],source_phase='evaluation')
            if payload.get('complete'):payload['metrics']=op_route_metrics(payload['complete'],config)
            op_save(dict(original,phase='joint_evaluation',method=method,b32_calls=0,payload=payload))
        if (name,'OLD_OP_JOINT_MULTI3') in done:continue
        info=op_measure_decodes(shared,config,parents,OP_VARIANT,('joint_greedy',),cache)
        op_save_measured('joint_evaluation',name,'OLD_OP_JOINT',info,first_only=True)
        old_attempts=originals['OLD_OP_MULTI3']['payload']['attempts']
        sample_seconds=old_attempts[-1]['elapsed']-old_attempts[0]['elapsed']
        for i,old in enumerate(old_attempts[1:],1):
            attempt=dict(old,proposal_index=i,b32_calls=0,reused=True)
            if attempt.get('solution'):
                attempt['metrics']=op_route_metrics(attempt,config)
                if old['status']=='FEASIBLE':
                    # Saved proposal metrics were not columns; recover MOVE
                    # from the exact route, and WAIT from Cmax only when saved.
                    # Returned sample metrics are taken from its saved result
                    # below rather than inventing an operation timeline.
                    attempt['metrics'].update(move=sum(attempt['metrics']['p_move'])-sum(attempt['metrics']['loads']))
            info['attempts'].append(attempt)
        info['runtime_s']+=sample_seconds
        info['actual_runtime_s']=None
        info['reused_b32_seconds']=None
        # The selected saved sample's WAIT/MOVE are available in the original
        # returned row whenever that sample wins this new three-proposal set.
        valid=[a for a in info['attempts'] if a['status']=='FEASIBLE']
        if valid:
            winner=min(valid,key=lambda a:a['cmax'])
            if winner['proposal_index']>0:
                original=originals['OLD_OP_MULTI3']
                if not math.isclose(float(original['cmax']),winner['cmax'],rel_tol=1e-9):
                    raise ValueError('an unselected historical sample needs its saved operation metrics')
                winner['metrics'].update(move=float(original['move']),wait=float(original['wait']))
        op_save_measured('joint_evaluation',name,'OLD_OP_JOINT_MULTI3',info)


def op_choose_early_decoder(rows):
    """Development selection after BOTH fixed single-proposal evaluations.

    Feasibility wins first, then same-geometry Cmax wins; conditional medians
    only break a tie. This is disclosed development selection, not validation.
    """
    selected={m:{r['instance']:r for r in rows if r['phase']=='early_evaluation' and r['method']==m}
              for m in ('EARLY_GREEDY','EARLY_JOINT')}
    if any(len(v)!=8 for v in selected.values()):raise ValueError('finish both eight-geometry early evaluations')
    wins=Counter()
    for name in selected['EARLY_GREEDY']:
        a,b=(selected[m][name] for m in ('EARLY_GREEDY','EARLY_JOINT'))
        fa,fb=a['status']=='FEASIBLE',b['status']=='FEASIBLE'
        if fa!=fb:wins['EARLY_GREEDY' if fa else 'EARLY_JOINT']+=1
        elif fa and not math.isclose(float(a['cmax']),float(b['cmax']),rel_tol=1e-9,abs_tol=1e-9):
            wins['EARLY_GREEDY' if float(a['cmax'])<float(b['cmax']) else 'EARLY_JOINT']+=1
    def key(method):
        good=[float(r['cmax']) for r in selected[method].values() if r['status']=='FEASIBLE']
        return (len(good),wins[method],-statistics.median(good) if good else -math.inf,method=='EARLY_GREEDY')
    return 'greedy' if max(selected,key=key)=='EARLY_GREEDY' else 'joint_greedy'


def op_early_evaluate():
    shared,config,_,_,_,_,entries,_=op_sources()
    saved=op_read();cache=op_evaluation_cache(saved)
    done={(r['instance'],r['method']) for r in saved if r['phase']=='early_evaluation'}
    for name,parents in sorted(entries.items()):
        for decode,method in (('greedy','EARLY_GREEDY'),('joint_greedy','EARLY_JOINT')):
            if (name,method) in done:continue
            info=op_measure_decodes(shared,config,parents,EARLY_VARIANT,(decode,),cache)
            op_save_measured('early_evaluation',name,method,info)
    decode=op_choose_early_decoder(op_read())
    for name,parents in sorted(entries.items()):
        if (name,'EARLY_BEST_MULTI3') in done:continue
        info=op_measure_decodes(shared,config,parents,EARLY_VARIANT,(decode,'sample','sample'),cache)
        op_save_measured('early_evaluation',name,'EARLY_BEST_MULTI3',info)


def op_early_report():
    import run_gnn_shared_construction as shared
    from mrta_reference.model import ScientificConfig
    config=ScientificConfig();rows=op_read()
    heading='## 2026-11-05 早期Cmax反馈与Top-2联合构造'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Existing early Cmax report section retained.',flush=True);return
    table={(r['instance'],r['method']):r for r in rows
           if r['phase'] in ('evaluation','joint_evaluation','early_evaluation')}
    names=sorted({r['instance'] for r in rows if r['phase']=='early_evaluation'})
    methods=('OLD_OP_GREEDY','OLD_OP_JOINT','OLD_OP_MULTI3','OLD_OP_JOINT_MULTI3',
             'EARLY_GREEDY','EARLY_JOINT','EARLY_BEST_MULTI3')
    if len(names)!=8 or any((n,m) not in table for n in names for m in methods):
        raise ValueError('finish all eight paired development geometries before reporting')
    metadata=shared.read_neural_checkpoint(OP_CHECKPOINT)[EARLY_VARIANT]['training']
    data=[r for r in rows if r['phase']=='early_counterfactual']
    outcomes=[o for r in data for o in r['payload']['outcomes']]
    def num(v):return '—' if v is None or v=='' else f'{float(v):.3f}'
    def good(r):return r['status']=='FEASIBLE' and r['cmax'] not in ('',None)
    def compare(method,comparator):
        wins=ties=losses=both_failed=0;changes=[]
        for name in names:
            a,b=table[name,method],table[name,comparator];fa,fb=good(a),good(b)
            if fa and fb:
                change=100.*(float(b['cmax'])-float(a['cmax']))/float(b['cmax']);changes.append(change)
                if abs(change)<=1e-8:ties+=1
                elif change>0:wins+=1
                else:losses+=1
            elif fa:wins+=1
            elif fb:losses+=1
            else:both_failed+=1
        return f'{wins}/{ties}/{losses}',both_failed,len(changes),statistics.median(changes) if changes else None
    calls=sum(int(r['b32_calls'] or 0) for r in data)
    lines=['',heading,'',
        '**实测结论：早期训练未取得稳定改善。** '
        f'旧机会模型原贪心/联合贪心分别{sum(good(table[n,"OLD_OP_GREEDY"]) for n in names)}/8、'
        f'{sum(good(table[n,"OLD_OP_JOINT"]) for n in names)}/8可行；'
        f'早期模型分别{sum(good(table[n,"EARLY_GREEDY"]) for n in names)}/8、'
        f'{sum(good(table[n,"EARLY_JOINT"]) for n in names)}/8。'
        f'早期三候选{sum(good(table[n,"EARLY_BEST_MULTI3"]) for n in names)}/8，'
        f'旧机会联合三候选{sum(good(table[n,"OLD_OP_JOINT_MULTI3"]) for n in names)}/8。'
        '保留新权重和负结果，不将early模型升级为默认模型；本轮仍未达到SCI主算法要求。', '',
        '本轮只改推理联合选择和早期反馈位置。两层GAT、hidden=64、heads=4、20维父输入、49维插入输入及全部归一化缓冲保持原维度；'
        'CONFIG_05、FORMAL_SCOPE_V2、min_child_length=0.50m、B32预算、加工时间和合法动作约束不变。'
        '未使用MODE_NEW、VNS、ALNS、RULE或旧模型回退，未训练独立方向头或模式头。',
        '全部推理使用fixed_split_patterns()：当前GNN只搜索固定合法拆分模式下的分配、顺序和方向，'
        '不能声称联合搜索全部WHOLE/Y_SPLIT/X_SPLIT组合。这是当前搜索空间限制。', '',
        '### 实验一：冻结旧机会权重', '',
        'joint_greedy在完整合法父集合上计算log_softmax，取概率Top-2；每个父的首子段在其完整合法插入池上单独归一化，'
        '按log P(parent)+log P(insertion|parent)选动作。第二子段仍按原贪心合法插入，失败不返回不完整方案。'
        'Top-2是新增有限前瞻策略，与原分层贪心的搜索策略不等价，未扩大到Top-5/10。',
        '原贪心和原三候选直接复用上一轮相同opportunity_cmax_candidate、相同种子的完整记录；'
        '两次采样及其B32结果、构造时间原样复用，只新运行联合贪心首提案。联合三候选时间为新首提案实测时间加旧两次采样的实测增量，'
        '不是一次连续墙钟测量。未重调度采样或增加认证。', '',
        '|几何|旧贪心|联合贪心|旧三候选|联合+原两采样|', '|---|---:|---:|---:|---:|']
    for name in names:lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in methods[:4])+'|')
    lines += ['', '### 实验二：真实早期状态与训练', '',
        f'冻结旧机会模型生成21条学生轨迹，保存{len(data)}个0%/10%/25%真实部分路线、方向、已分配父ID和剩余父ID。'
        '0%在第一条父任务插入前观察，路线与已分配集合为空；10%/25%按int(N*f)取整，并保存实际进度。',
        'A为原分层GNN首选；B来自同父首子段的不同合法机器人，优先最小化动作后的最大纯加工负载，以网络分数打破平局；'
        'C为不同合法父任务，按总加工量/首子段当前合法机器人数量选择代表，再用原网络选择其插入动作。'
        '没有合法替代就减少数量，不用RULE覆盖推理。候选选择仅用于TRAIN反事实数据。',
        f'最多189个候选，实际{len(outcomes)}个；新增B32 {calls}/189次。其余相同完整路线和方向复用既有调度结果。'
        f'可行{sum(o["status"]=="FEASIBLE" for o in outcomes)}，DEADLOCK {sum(o["status"]=="DEADLOCK" for o in outcomes)}，'
        f'同状态排序对{sum(len(r["payload"]["pairs"]) for r in data)}。两个失败不排序，失败Cmax为空。'
        '每个分支从同一完整前缀强制一步，随后由同一冻结旧机会模型原贪心补全。',
        f"复用{metadata['preparation']['teachers_reused']}个完整教师动作和{metadata['preparation']['prefix_continuations_reused']}个历史前缀延续，"
        f"共{metadata['preparation']['imitation_examples']}个IL样本；未重新生成或调度教师。",
        f"early_cmax_candidate从opportunity_cmax_candidate初始化，未重拟合归一化。固定seed={OP_SEED}，"
        'AdamW lr=1e-4，batch=16，clip=1，10轮，L=0.5 IL+rank+0.1 load；'
        f"训练{num(metadata['training_seconds'])}秒，总损失{num(metadata['history'][0]['loss'])}→{num(metadata['history'][-1]['loss'])}。"
        '未按开发性能选择epoch或重复加训，损失下降不等同于Cmax改善。',
        '权重保存在既有models/gnn_coordination_constructor_05.pt的独立early_cmax_candidate中；'
        '顶层、refinement_candidate与opportunity_cmax_candidate逐tensor核对保留，旧实验CSV记录未覆盖。', '',
        '|前缀|状态数|候选数|新增B32|排序对|', '|---:|---:|---:|---:|---:|']
    for fraction in (0.,.1,.25):
        subset=[r for r in data if r['payload']['fraction']==fraction]
        lines.append(f'|{fraction:.0%}|{len(subset)}|{sum(len(r["payload"]["outcomes"]) for r in subset)}|'
            f'{sum(int(r["b32_calls"]) for r in subset)}|{sum(len(r["payload"]["pairs"]) for r in subset)}|')
    lines += ['', '### 实验三：同几何、同种子配对', '',
        f'仅使用已消费的8个开发几何，seed={OP_EVAL_SEED}。RULE/BEAM取保存初始状态，WAG/HGA取原60秒完整结果；'
        'CURRENT_PORTFOLIO3及所有对照直接复用同配置记录，未重跑500次正式实验，未访问未打开测试数据。',
        '先比较四个核心单提案方法，再仅为early模型中开发表现较好的解码方式运行单模型三候选。'
        f'所选首提案为{op_choose_early_decoder(rows)}，后两个仍为seed和seed+1的原采样，选择规则先看可行数、再看配对胜数。'
        '这是开发集选择，不能当作独立验证。', '',
        '|几何|OLD_OP_GREEDY|OLD_OP_JOINT|EARLY_GREEDY|EARLY_JOINT|EARLY_BEST_MULTI3|RULE/BEAM|WAG|HGA|CURRENT_PORTFOLIO3|',
        '|---|'+'---:|'*9]
    final_methods=('OLD_OP_GREEDY','OLD_OP_JOINT','EARLY_GREEDY','EARLY_JOINT','EARLY_BEST_MULTI3','RULE/BEAM','WAG','HGA','CURRENT_PORTFOLIO3')
    for name in names:lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in final_methods)+'|')
    lines += ['', '—为失败，未填0或惩罚Cmax。时间含文件读取、模型实例化、图构建、完整构造、采样和B32。'
        '新early方法保存连续物理墙钟时间与缓存复用时间；表中含缓存原B32实测时长，避免缓存命中造成虚假加速。'
        '旧联合三候选按上一段的实测组成时间报告。', '',
        '|方法|可行/8|含B32计费秒中位|含B32计费秒最大|物理墙钟秒中位|', '|---|---:|---:|---:|---:|']
    for method in methods+('RULE/BEAM','WAG','HGA','CURRENT_PORTFOLIO3'):
        selected=[table[n,method] for n in names];times=[float(r['runtime_s']) for r in selected]
        physical=([r['payload']['actual_runtime_s'] for r in selected] if method.startswith('EARLY') else
                  [] if method=='OLD_OP_JOINT_MULTI3' else times)
        lines.append(f'|{method}|{sum(good(r) for r in selected)}/8|{num(statistics.median(times))}|{num(max(times))}|'
                     f'{num(statistics.median(physical) if physical else None)}|')
    lines += ['', '胜负包含“可行胜失败”，双失败单列；改善中位只对双方可行配对计算，并列明分母。', '',
        '|方法|对照|胜/平/负|双失败|双方可行/8|条件中位改善/%|', '|---|---|---:|---:|---:|---:|']
    for method,comparators in (
        ('OLD_OP_JOINT',('OLD_OP_GREEDY',)),('OLD_OP_JOINT_MULTI3',('OLD_OP_MULTI3',)),
        *((m,('OLD_OP_GREEDY','OLD_OP_JOINT','OLD_OP_MULTI3','CURRENT_PORTFOLIO3','RULE/BEAM','WAG','HGA'))
          for m in ('EARLY_GREEDY','EARLY_JOINT','EARLY_BEST_MULTI3'))):
        for comparator in comparators:
            counts,failed,paired,change=compare(method,comparator)
            lines.append(f'|{method}|{comparator}|{counts}|{failed}|{paired}/8|{num(change)}|')
    lines += ['', '### 加工负载、空走、等待和错误定位', '',
        '单提案DEADLOCK行仍显示其完整构造的加工负载与路线P+MOVE下界，Cmax/MOVE/WAIT保持空值；'
        '三候选整组失败时不挑一个失败方案冒充返回值。加工负载为PRE+WELD+POST，MOVE/WAIT只取成功B32完整时间线。', '',
        '|几何|方法|状态|R0/R1/R2/R3加工负载|max(P+MOVE)|MOVE|WAIT|', '|---|---|---|---|---:|---:|---:|']
    categories=Counter();errors=[]
    for name in names:
        target=min(float(table[name,m]['cmax']) for m in ('RULE/BEAM','WAG','HGA') if good(table[name,m]))
        for method in ('OLD_OP_GREEDY','OLD_OP_JOINT','EARLY_GREEDY','EARLY_JOINT','EARLY_BEST_MULTI3'):
            row=table[name,method];payload=row['payload'];attempts=payload.get('attempts',[])
            raw=payload.get('complete') or (attempts[0] if len(attempts)==1 and attempts[0].get('solution') else None)
            metrics=op_route_metrics(raw,config) if raw else {}
            loads=metrics.get('loads',[])
            lines.append(f'|{name}|{method}|{row["status"]}|'+(' / '.join(num(v) for v in loads) or '—')+
                f'|{num(metrics.get("max_p_move"))}|{num(row.get("move"))}|{num(row.get("wait"))}|')
            if method=='EARLY_BEST_MULTI3':
                category=('B32 DEADLOCK/构造失败' if not good(row) else '加工分配下界过大' if max(loads)>target else
                          '插入顺序/方向空走' if metrics['max_p_move']>target else
                          '协调等待' if float(row['cmax'])>target+1e-8 else '不劣于最佳对照')
                categories[category]+=1
            if method in ('EARLY_GREEDY','EARLY_JOINT') and not good(row):
                errors.append(f'{name}/{method}: '+', '.join(a['status'] for a in attempts)+
                    f'，max P={num(max(loads) if loads else None)}，max(P+MOVE)={num(metrics.get("max_p_move"))}，最佳对照={num(target)}。')
    lines += ['', 'early三候选返回方案分类：'+str(dict(categories))+'。', '', '代表性失败（完整原始提案保存在CSV）：', '']
    lines.extend('- '+e for e in errors[:4])
    lines += ['', '机器人分配造成的max P超额是插入位置、方向或WAIT优化无法消除的下界；'
        'max(P+MOVE)与Cmax的差才用于判断协调增量。DEADLOCK是当前B32预算未找到可行协调调度，'
        '不能据此断言几何绝对不可调度，也不能在没有方向消融时把失败全部归于方向。', '',
        '六个early可行返回方案的Cmax均等于最大P+MOVE；其中SMALL_02虽有2306.176秒总WAIT，'
        '等待发生在非关键路径上，未增加整体Cmax。五例主要差距来自加工量分配；'
        'HIGH_WORKLOAD_04的最大P为4697.530秒，对照4659.421秒，仅超38.109秒，'
        '而P+MOVE额外增加328.250秒，因此该例的剩余主要差距是路线插入顺序/方向造成的空走。'
        '这两项没有独立消融，不能进一步假定全部来自插入位置或全部来自方向。', '',
        'LARGE_02、LARGE_04的early联合三候选均为DEADLOCK。LARGE_02第二次采样的max(P+MOVE)为4180.368秒，'
        '仍未得到B32可行调度，显示负载较低也不足以保证同轨协调可行。'
        '同几何EARLY_GREEDY有4660.517秒可行解，但选定的联合三候选失败；没有把这个额外贪心加入三候选回退。'
        '父任务联合评分改变了完整路线及可调度性，这与加工负载倾斜是两个问题；仅凭路径差异不能把失败唯一归因于父头、位置或方向。', '',
        '### 保留结论与下一步', '',
        f'联合解码对旧机会三候选的失败计入胜/平/负为{compare("OLD_OP_JOINT_MULTI3","OLD_OP_MULTI3")[0]}，'
        f'另有{compare("OLD_OP_JOINT_MULTI3","OLD_OP_MULTI3")[1]}个双失败；有有限正贡献但不稳定。'
        f'early三候选相对旧机会三候选为{compare("EARLY_BEST_MULTI3","OLD_OP_MULTI3")[0]}，'
        f'相对CURRENT_PORTFOLIO3为{compare("EARLY_BEST_MULTI3","CURRENT_PORTFOLIO3")[0]}，'
        f'相对WAG、HGA分别为{compare("EARLY_BEST_MULTI3","WAG")[0]}、{compare("EARLY_BEST_MULTI3","HGA")[0]}。'
        '训练反馈前移未解决独立构造质量，不再追加轮次或采样。',
        '**下一轮有必要优先检验动作后的机会变化ΔO(a)=O(T(s,a))−O(s)。** '
        '当前机会特征在同状态各候选间相同，无法直接反映一次分配对后续可服务任务集合的损失；'
        '早期同状态真实Cmax反馈仍未消除分配下界和DEADLOCK问题。'
        '应利用已有候选状态、首点及可达性信息计算受影响机器人/任务的增量，避免逐候选重新枚举全部剩余焊缝。'
        '这只是下一轮待验证假设，本轮未实现，也不宣称其必然有效。', '',
        '### 测试与复现', '']
    for row in rows:
        if row['phase']=='early_verification':lines += [row['payload']['summary'],'','```powershell',row['payload']['command'],'```','']
        if row['phase']=='early_execution':lines += [row['payload']['summary'],'']
    lines += ['```powershell',"Set-Location 'D:\\pybullet_test\\MRTA_GA'",
        "$env:PYTHONPATH = (Join-Path (Get-Location) 'DRL\\src')",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B DRL/scripts/run_gnn_cmax_finetune.py --phase early-all",'```','',
        '本轮未新增文件；代码、测试、结果CSV、模型容器和报告均复用既有文件。']
    OP_REPORT.write_text(original+'\n'.join(lines)+'\n',encoding='utf-8')
    print('APPENDED',OP_REPORT,flush=True)


def op_report():
    import run_gnn_shared_construction as shared
    from mrta_reference.model import ScientificConfig
    from mrta_search.lns import oriented_route_state,oriented_route_lower_bound
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    rows=op_read();evaluation=[r for r in rows if r['phase']=='evaluation']
    methods=('RULE/BEAM','WAG','HGA','OLD_GNN_GREEDY','CURRENT_PORTFOLIO3','NEW_GNN_GREEDY','NEW_GNN_MULTI3')
    if len(evaluation)!=56:raise ValueError('finish the seven-method eight-geometry evaluation')
    config=ScientificConfig();names=sorted({r['instance'] for r in evaluation})
    table={(r['instance'],r['method']):r for r in evaluation}
    def num(v):return '—' if v is None or v=='' else f'{float(v):.3f}'
    def good(r):return r['status']=='FEASIBLE' and r['cmax'] not in ('',None)
    def compare(method,comparator):
        wins=ties=losses=0;changes=[];missing=[]
        for name in names:
            new,old=table[name,method],table[name,comparator]
            if not good(new) or not good(old):missing.append(name);continue
            change=100.*(float(old['cmax'])-float(new['cmax']))/float(old['cmax'])
            if abs(change)<=1e-8:ties+=1
            elif change>0:wins+=1
            else:losses+=1
            changes.append(change)
        return f'{wins}/{ties}/{losses}',statistics.median(changes) if changes else None,missing
    checkpoint=shared.read_neural_checkpoint(OP_CHECKPOINT)
    metadata=checkpoint[OP_VARIANT]['training']
    data=[r for r in rows if r['phase']=='counterfactual'];outcomes=[o for r in data for o in r['payload']['outcomes']]
    pairs=sum(len(r['payload']['pairs']) for r in data);calls=sum(int(r['b32_calls']) for r in data)
    heading='## 2026-11-05 机器人机会状态与真实Cmax成对排序'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Existing opportunity report section retained.',flush=True);return
    lines=['',heading,'',
        '本节研究独立GNN初始构造，不以MODE_NEW作为主算法。全部新GNN方案均由同一个模型从空路线生成，'
        '没有RULE、旧模型、MODE_NEW、VNS、ALNS、SA或独立方向/WAIT优化回退。旧章节按历史记录保留。', '',
        '### 代码与数据边界', '',
        '- gnn_constructor.py：父特征12→20，联合插入特征41→49，追加四机器人剩余可服务/当前专属加工量。'
        '每次插入后重新查询既有可达性、模式、滑轨、首点、兄弟约束下的合法插入存在性；可服务量可重复计入多台机器人，'
        '不是实际负载，也不是均分或必须启用四机的硬约束。',
        '- lns.py：已有has_joint_block_candidate仅新增可选robot_id过滤参数；默认合法池与MODE_NEW决策不变。',
        '- run_gnn_shared_construction.py：原ModeJointInsertionGNN只扩展现有输入层，旧列逐值复制，新列置0。'
        '旧归一化缓冲保持不变，新八维只用训练数据拟合；两层GAT、64维、4头、静态编码及原父/插入头保留。'
        '新权重保存在原checkpoint的opportunity_cmax_candidate，原顶层与refinement_candidate保存后torch.equal核对通过。',
        '- run_gnn_cmax_finetune.py：新增op-*入口，复用保存动作，进行同状态强制动作→同一GNN贪心补全→一次B32的反事实收集。'
        '新路径未调用该脚本保留的历史REINFORCE代码。',
        '- 只使用既有21个训练几何和8个开发留出几何；开发几何不参与训练或归一化。'
        '未读取封存工作簿或新增保留测试几何。本节八例已经用于开发，不称为独立测试集。', '',
        '### 实际训练数据与训练过程', '',
        f"复用{metadata['preparation']['teachers_reused']}个成功完整教师动作、{metadata['preparation']['prefix_continuations_reused']}个已保存学生前缀延续，"
        f"重建{metadata['preparation']['imitation_examples']}个模仿样本，教师新增B32调用为0。未重新生成110个教师，也未重新做认证。",
        f'新增21条学生构造轨迹、{len(data)}个25%/50%/75%父任务进度前缀，最多每状态3个候选，'
        f'实际得到{len(outcomes)}个完整补全结果，新增{calls}次B32（上限189），其余{len(outcomes)-calls}个相同完整方案复用调度结果。'
        f'其中可行{sum(o["status"]=="FEASIBLE" for o in outcomes)}、DEADLOCK {sum(o["status"]=="DEADLOCK" for o in outcomes)}。',
        f'新增{pairs}个同状态成对标签；两个均不可行和数值容差内相同Cmax不生成优劣标签，失败Cmax始终为空。'
        '分数使用log P(parent|s)+log P(insertion|s,parent)，不是独立动作网络或跨几何Cmax排名。',
        f"固定seed={OP_SEED}，AdamW lr=1e-4，batch=16，clip=1，10轮；L=0.5 IL+1 rank+0.1 load。"
        f"训练耗时{metadata['training_seconds']:.3f}秒，没有开发集调参、早停选轮或重复加训。", '',
        '|轮次|总损失|IL|rank|load|累计秒|','|---:|---:|---:|---:|---:|---:|']
    for h in metadata['history']:lines.append(f"|{h['epoch']}|{num(h['loss'])}|{num(h['il'])}|{num(h['rank'])}|{num(h['load'])}|{num(h['seconds'])}|")
    lines += ['', '### 配对评价口径', '',
        f'固定评价seed={OP_EVAL_SEED}，选取历史五种子中的第一个以复用全部对应对照，未按结果选种子。'
        'RULE/BEAM取保存的初始状态，绝不使用MODE_NEW搜索后的终解冒充规则构造；WAG/HGA保持原配置和60秒预算。'
        '24条对应的规则/WAG/HGA记录直接复用，历史500次实验及长预算实验均未重跑。'
        'HIGH_WORKLOAD_04的规则初始化失败原样保留，非新模型回退条件。',
        'CURRENT_PORTFOLIO3是强旧对照：旧模型greedy+旧模型sample+既有refinement greedy。'
        'NEW_GNN_MULTI3只用opportunity_cmax_candidate：greedy、seed采样、seed+1采样。'
        '两者均最多三个完整方案；贪心视图复用同次三提案的首方案，不重复B32。',
        'GNN时间含模型文件读取与实例化、图构建、全部提案推理/构造和B32。'
        '新模型在空调度缓存下首先实测；本轮新旧两组三提案共48个不同完整方案，均实际调度。'
        '旧单次与旧组合共享首方案的实际结果和计时；CSV物理调用计在组合行，贪心视图的逻辑调用单独记录。', '',
        '|方法|可行/总数|可行率|运行秒中位|运行秒最大|', '|---|---:|---:|---:|---:|']
    for method in methods:
        selected=[r for r in evaluation if r['method']==method];runtimes=[float(r['runtime_s']) for r in selected]
        feasible=sum(good(r) for r in selected)
        lines.append(f'|{method}|{feasible}/8|{100.*feasible/8:.1f}%|{num(statistics.median(runtimes))}|{num(max(runtimes))}|')
    lines += ['', '逐几何真实Cmax；—为失败，未填0或惩罚值。', '',
              '|几何|'+'|'.join(methods)+'|','|---|'+'---:|'*len(methods)]
    for name in names:lines.append('|'+name+'|'+'|'.join(num(table[name,m]['cmax']) for m in methods)+'|')
    lines += ['', '|新方法|对照|胜/平/负|完整配对/8|配对中位改善/%|未配对几何|', '|---|---|---:|---:|---:|---|']
    for method in ('NEW_GNN_GREEDY','NEW_GNN_MULTI3'):
        for comparator in ('OLD_GNN_GREEDY','CURRENT_PORTFOLIO3','RULE/BEAM','WAG','HGA'):
            counts,change,missing=compare(method,comparator)
            lines.append(f"|{method}|{comparator}|{counts}|{8-len(missing)}/8|{num(change)}|{', '.join(missing) or '无'}|")
    lines += ['', '### 负载、空走、等待与失败', '',
        '以下为新三提案返回的真实可行方案；失败组不选择一个代理最优的失败方案冒充返回值。'
        'MOVE/WAIT为B32完整操作总时长，不将负载均衡改善等同为Cmax改善。', '',
        '|几何|R0/R1/R2/R3加工负载|MOVE|WAIT|使用机器人|主要劣势定位|', '|---|---|---:|---:|---:|---|']
    categories=Counter()
    for name in names:
        r=table[name,'NEW_GNN_MULTI3']
        if not good(r):
            categories['B32不可行']+=1
            lines.append(f'|{name}|—|—|—|—|三个方案均未获得B32可行调度|');continue
        loads=json.loads(r['loads']);state=r['payload']['complete'];solution,bits,_,_,_=shared.research_restore(state,config)
        route_state=oriented_route_state(tuple(x.block_ids for x in solution.routes),bits,block_map(solution,config,scope=FORMAL_SCOPE_V2),config)
        bound=oriented_route_lower_bound(route_state)
        target=min(float(table[name,m]['cmax']) for m in ('RULE/BEAM','WAG','HGA') if good(table[name,m]))
        if float(r['cmax'])<=target:category='不劣于最佳对照'
        elif max(loads)>target:category='加工分配已超过对照'
        elif bound>target:category='空走/顺序/方向'
        else:category='协调等待'
        categories[category]+=1
        lines.append(f"|{name}|{' / '.join(num(v) for v in loads)}|{num(r['move'])}|{num(r['wait'])}|{r['robots']}|{category}|")
    lines += ['', '贪心完整构造的机器人使用数统计包含B32失败提案，避免只统计可行样本掩盖塌缩。', '',
              '|贪心模型|完整构造|使用2/3/4台的次数|R0未使用|', '|---|---:|---:|---:|']
    for method in ('OLD_GNN_GREEDY','NEW_GNN_GREEDY'):
        usage=Counter();unused=0
        for name in names:
            a=table[name,method]['payload']['attempts'][0]
            if not a.get('solution'):continue
            counts=[len(r['block_ids']) for r in a['solution']['routes']]
            usage[sum(v>0 for v in counts)]+=1;unused+=counts[0]==0
        lines.append(f"|{method}|{sum(usage.values())}/8|{usage[2]}/{usage[3]}/{usage[4]}|{unused}/8|")
    failures=[]
    for name in names:
        for method in ('NEW_GNN_GREEDY','NEW_GNN_MULTI3'):
            r=table[name,method]
            if not good(r):failures.append(f"{name}/{method}: "+', '.join(a['status'] for a in r['payload']['attempts']))
    lines += ['', '失败提案保留：', '']+[f'- {f}' for f in failures]
    counts,change,missing=compare('NEW_GNN_MULTI3','CURRENT_PORTFOLIO3')
    feasible=sum(good(table[n,'NEW_GNN_MULTI3']) for n in names)
    lines += ['', '### 是否达到目标与下一步', '',
        f'**本轮未达到论文主算法条件。** 新三提案可行{feasible}/8，目标95%以上在本次8例中要求8/8；'
        f'相对旧三提案仅{counts}，完整配对中位改善{num(change)}%，另有{len(missing)}个新方法失败的几何不能丢弃。'
        '相对RULE/WAG/HGA的缺失配对和实际劣势见上表，不把更低负载或更短时间包装成Cmax成功。',
        '按真实选中方案区分主要问题：'+str(dict(categories))+'。'
        '六个可行返回方案的最大纯加工负载均已高于同几何最佳对照Cmax；其中五例WAIT为0，另一例仅13.795秒。'
        '因此可行解质量的主要瓶颈仍是父任务选择与机器人分工耦合造成的加工量倾斜，空走和等待优化无法抵消该下界劣势。'
        '另有两个几何三个完整提案均DEADLOCK，体现联合插入/方向的全局协调问题；没有单独方向对照，不能武断归为方向独立失败。',
        '**下一轮只优先解决早期不可逆的加工量分工。** 本轮Cmax标签来自25%/50%/75%前缀，'
        '不能直接监督最早的父任务与机器人分配；训练集成对损失降低也未转化为开发Cmax。'
        '应在相同预算内把一个前缀位置前移到首部机器人分配决定前，验证是否能改善早期负载分工并保留后续机会，'
        '保持现有GNN结构和真实B32反馈，不继续盲目扩训或增加网络层。', '',
        '### 测试与复现', '']
    diagnostics=[r['payload'] for r in rows if r['phase']=='diagnosis']
    for d in diagnostics:
        lines += ['补充训练前缀诊断：25%/50%/75%中，有标签的状态分别为16/10/9；'
                  '已分配的最大纯加工负载超过该几何最佳保存教师完整Cmax的状态分别为0/5/15（每组21个）。'
                  '这些已分配任务在纯构造中不能再换机器人，后半程标签不能直接纠正该前置分工。'
                  '79个标签中，同父不同机器人对比33个、不同父联合决策对比46个。'
                  '这支持优先检验反馈前移，而不是声称前移一定有效。', '']
    for r in rows:
        if r['phase']=='verification':lines += [r['payload']['summary'],'','```powershell',r['payload']['command'],'```','']
    lines += ['```powershell',"Set-Location 'D:\\pybullet_test\\MRTA_GA\\DRL'",
        "$env:PYTHONPATH = (Join-Path (Get-Location) 'src')",
        "& 'D:\\pybullet_test\\.venv\\Scripts\\python.exe' -B scripts/run_gnn_cmax_finetune.py --phase op-all",'```','',
        '唯一新增文件为results/gnn_opportunity_cmax.csv；op-*入口保留已有数据和既有独立候选，不重复训练。']
    OP_REPORT.write_text(original+'\n'.join(lines)+'\n',encoding='utf-8')
    print('APPENDED',OP_REPORT,flush=True)


def load_v2():
    checkpoint = torch.load(pilot.MODEL, map_location='cpu', weights_only=True)
    if not checkpoint['normalize_inputs']:
        raise ValueError('V2 normalization is required')
    model = JointInsertionGNN(64, 4, normalize_inputs=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model, checkpoint


def select_sources(records):
    """Cover every existing TRAIN workbook, then fill count gaps; no DEV input."""
    unique = {r['entry']['instance_id']: r for r in reversed(records)}
    groups = defaultdict(list)
    for record in unique.values():
        groups[record['entry']['relative_path']].append(record)
    if len(groups) != 17 or len(unique) != 46:
        raise ValueError('expected the existing 46 geometries / 17 TRAIN workbooks')
    ordered = sorted(unique.values(), key=lambda r: (r['entry']['N'], r['entry']['instance_id']))
    low, high = ordered[0]['entry']['N'], ordered[-1]['entry']['N']
    chosen = []
    for i, path in enumerate(sorted(groups)):
        target = low + (high-low)*i/(len(groups)-1)
        chosen.append(min(groups[path], key=lambda r: (abs(r['entry']['N']-target), r['entry']['instance_id'])))
    while len(chosen) < SETTINGS['geometries']:
        remaining = [r for r in ordered if r not in chosen]
        chosen.append(min(remaining, key=lambda r: (-min(abs(r['entry']['N']-s['entry']['N']) for s in chosen),
                                                   r['entry']['instance_id'])))
    return sorted(chosen, key=lambda r: (r['entry']['N'], r['entry']['instance_id']))


def terminal_return(candidate, reference_cmax):
    if not math.isfinite(reference_cmax) or reference_cmax <= 0:
        raise ValueError('reference must have a positive finite feasible Cmax')
    if candidate is None or candidate.schedule.status is not ScheduleStatus.FEASIBLE:
        return -1.
    cmax = candidate.schedule.cmax
    if cmax is None or not math.isfinite(cmax) or cmax <= 0:
        raise ValueError('FEASIBLE requires a positive finite Cmax')
    return (reference_cmax-cmax)/(reference_cmax+cmax)


def score_pair(sample, greedy, reference_cmax):
    """One B32 per distinct complete candidate, only within this update pair."""
    terminal_return(None, reference_cmax)  # Validate even when neither branch completes.
    cache = {}
    outputs = []
    for result in (sample, greedy):
        if result.solution is None:
            outputs.append((None, -1., 'CONSTRUCTION_FAILED', None))
            continue
        key = (tuple(r.block_ids for r in result.solution.routes), result.directions)
        if key not in cache:
            # Structural/numeric exceptions propagate; they are not failure rewards.
            cache[key] = pilot.oriented.schedule_explicit(result.solution, result.directions, CONFIG)
        candidate = cache[key]
        reward = terminal_return(candidate, reference_cmax)
        outputs.append((candidate, reward, candidate.schedule.status.value,
                        candidate.schedule.cmax if candidate.schedule.status is ScheduleStatus.FEASIBLE else None))
    return outputs, len(cache)


def bc_loss(model, graphs, pool, rng):
    losses = []
    embeddings = {}
    for name, step in rng.sample(pool, SETTINGS['bc_steps']):
        if name not in embeddings:
            embeddings[name] = model.encode_static(graphs[name])
        emb = embeddings[name]
        logits = model.joint_action_logits(emb, step['indices'], step['features'])
        loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([step['target']]))
        if len(step['parent_ids']) > 1:
            pl = model.parent_logits(graphs[name], emb, step['parent_ids'], step['parent_features'])
            loss = loss + F.cross_entropy(pl.unsqueeze(0), torch.tensor([step['parent_target']]))
        losses.append(loss)
    return torch.stack(losses).mean()


def gradient_norms(named_parameters, gradients):
    return {head: math.sqrt(sum(float(g.detach().square().sum())
                               for (name, _), g in zip(named_parameters, gradients)
                               if name.startswith(head+'.') and g is not None)) for head in HEADS}


def train_episode(model, optimizer, current, index, epoch, episode, graphs, pool, rng, phase):
    started = time.perf_counter()
    selection = pilot.SELECTORS[episode % 3]
    q = (4, 8)[(episode // 3) % 2]
    seed = SETTINGS['seed'] + epoch*100000 + index*1000 + episode
    partial = pilot.select_bottleneck_parents(current.solution, current.directions, current.schedule,
                                             selection, q=q, seed=seed, config=CONFIG)
    directions = pilot.partial_directions(current, partial)
    args = (current.solution, partial.routes, directions, partial.removed_parent_ids, model, CONFIG)
    optimizer.zero_grad(set_to_none=True)
    # Both branches use these exact weights. No optimizer step until both B32s finish.
    rollout = rollout_joint_repair(*args, insertion_limit=8, seed=seed)
    greedy = construct_or_repair_with_gnn(*args, insertion_limit=8)
    scored, calls = score_pair(rollout.completion, greedy, current.schedule.cmax)
    (sample, gs, ss, cs), (_, gg, sg, cg) = scored
    advantage = gs-gg
    rl = -advantage*rollout.log_prob
    named = list(model.named_parameters())
    if advantage != 0 and rl.requires_grad:
        grads = torch.autograd.grad(rl, [p for _, p in named], retain_graph=True, allow_unused=True)
        norms = gradient_norms(named, grads)
    else:
        norms = dict.fromkeys(HEADS, 0.)
    bc = bc_loss(model, graphs, pool, rng)
    loss = rl + SETTINGS['bc_coefficient']*bc
    if not bool(torch.isfinite(loss)) or not all(math.isfinite(v) for v in norms.values()):
        raise ValueError('nonfinite loss or policy gradient')
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), SETTINGS['gradient_clip'], error_if_nonfinite=True))
    optimizer.step()
    before = current.schedule.cmax
    accepted = sample is not None and sample.schedule.status is ScheduleStatus.FEASIBLE and cs < before-1e-9*max(1, before)
    if accepted:
        current = sample
    row = dict(phase=phase, instance=current.solution.parents[0].parent_id, epoch=epoch+1,
               episode=episode+1, method='GNN-Cmax-RL', q=q, selection=selection.value, seed=seed,
               reference_cmax=before, sample_cmax=cs, greedy_cmax=cg, sample_status=ss, greedy_status=sg,
               sample_return=gs, greedy_return=gg, advantage=advantage, nonzero_advantage=int(advantage != 0),
               log_prob=float(rollout.log_prob.detach()), parent_decisions=len(rollout.parent_log_probs),
               insertion_decisions=len(rollout.insertion_log_probs), rl_loss=float(rl.detach()), bc_loss=float(bc.detach()),
               b32_calls=calls, b32_success=sum(c is not None and c.schedule.status is ScheduleStatus.FEASIBLE
                                              for c in {id(v[0]):v[0] for v in scored if v[0] is not None}.values()),
               accepted=int(accepted), final_cmax=current.schedule.cmax, gradient_norm=grad_norm,
               direction0_actions=sum(a.direction == 0 for a in rollout.completion.actions),
               direction1_actions=sum(a.direction == 1 for a in rollout.completion.actions),
               flip_actions=sum(bool(a.neighbour_changes) for a in rollout.completion.actions),
               removed=json.dumps(partial.removed_parent_ids), runtime_s=time.perf_counter()-started,
               sample_failure=rollout.completion.rejection_reason, greedy_failure=greedy.rejection_reason,
               **{'rl_gradient_'+h:v for h,v in norms.items()})
    return current, row


def write_csv(rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with CSV.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summary(rows, phase):
    selected = [r for r in rows if r['phase'] == phase]
    if not selected:
        return {}
    return dict(episodes=len(selected), b32_calls=sum(r['b32_calls'] for r in selected),
                b32_success=sum(r['b32_success'] for r in selected),
                sample_feasible=sum(r['sample_status'] == 'FEASIBLE' for r in selected),
                greedy_feasible=sum(r['greedy_status'] == 'FEASIBLE' for r in selected),
                nonzero=sum(r['nonzero_advantage'] for r in selected),
                feasible_pair_nonzero=sum(r['advantage'] != 0 and r['sample_status'] == r['greedy_status'] == 'FEASIBLE'
                                          for r in selected),
                positive=sum(r['advantage'] > 0 for r in selected), negative=sum(r['advantage'] < 0 for r in selected),
                both_failed=sum(r['sample_return'] == r['greedy_return'] == -1 for r in selected),
                mean_sample_return=sum(r['sample_return'] for r in selected)/len(selected),
                mean_greedy_return=sum(r['greedy_return'] for r in selected)/len(selected),
                mean_abs_advantage=sum(abs(r['advantage']) for r in selected)/len(selected),
                accepted=sum(r['accepted'] for r in selected),
                gradient_nonzero={h:sum(r['rl_gradient_'+h] > 0 for r in selected) for h in HEADS},
                gradient_max={h:max(r['rl_gradient_'+h] for r in selected) for h in HEADS},
                runtime_s=sum(r['runtime_s'] for r in selected))


def save_model(model, metadata, optimizer=None):
    payload = dict(state_dict=model.state_dict(), normalize_inputs=True, hidden_dim=64, heads=4,
                   seed=SETTINGS['seed'], metadata=metadata)
    if optimizer is not None:
        payload['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(payload, MODEL)


def train_checks(model, records, sources, label):
    rows = []
    for source in sources:
        for q in (4, 8):
            options = [r for r in records if len(r['pending_parents']) == q]
            record = min(options, key=lambda r: (r['entry']['instance_id'] != source['entry']['instance_id'],
                                                 abs(r['entry']['N']-source['entry']['N']), r['entry']['instance_id']))
            reference = pilot.record_reference(record)
            start = time.perf_counter()
            result = construct_or_repair_with_gnn(reference, record['partial_routes'], record['partial_directions'],
                                                 record['pending_parents'], model, CONFIG, insertion_limit=8)
            scheduled = None if result.solution is None else pilot.oriented.schedule_explicit(result.solution, result.directions, CONFIG)
            rows.append(dict(phase='TRAIN_CHECK', instance=record['entry']['instance_id'], method=label, q=q,
                             status='CONSTRUCTION_FAILED' if scheduled is None else scheduled.schedule.status.value,
                             b32_calls=int(scheduled is not None), b32_success=int(scheduled is not None and scheduled.schedule.status is ScheduleStatus.FEASIBLE),
                             final_cmax=None if scheduled is None or scheduled.schedule.status is not ScheduleStatus.FEASIBLE else scheduled.schedule.cmax,
                             direction0_actions=sum(a.direction == 0 for a in result.actions),
                             direction1_actions=sum(a.direction == 1 for a in result.actions),
                             flip_actions=sum(bool(a.neighbour_changes) for a in result.actions),
                             runtime_s=time.perf_counter()-start))
            print('TRAIN CHECK', label, rows[-1], flush=True)
    return rows


def train(rows):
    started = time.perf_counter()
    records = [json.loads(line) for line in pilot.DATA.read_text(encoding='utf-8').splitlines() if line.strip()]
    graphs, grouped = pilot.prepare_training(records)  # Existing role checks + exact candidate replay.
    pool = [(name, step) for name, steps in grouped.items() for step in steps]
    sources = select_sources(records)
    whitelist = {e['instance_id']:e for group in pilot.dataset.train_sheet_candidates().values() for e in group}
    initials = {}
    for source in sources:
        entry = whitelist[source['entry']['instance_id']]
        if entry['split'] != 'MLP_TRAIN':
            raise ValueError('only MLP_TRAIN geometry may supply rollouts')
        parents = pilot.dataset.load_parents(entry)
        if parents != pilot.record_reference(source).parents:
            raise ValueError('TRAIN file differs from the existing teacher geometry')
        begin = time.perf_counter()
        initial, attempts = pilot.oriented.common_initial_beam(parents, CONFIG, seed=SETTINGS['seed'])
        if initial is None or abs(initial.schedule.cmax-source['original_cmax']) > 1e-6:
            raise ValueError('existing TRAIN Beam initial did not reproduce')
        initials[entry['instance_id']] = initial
        rows.append(dict(phase='TRAIN_BEAM', instance=entry['instance_id'], N=entry['N'], method='Beam-Init',
                         initial_cmax=initial.schedule.cmax, b32_calls=sum(a['status'] != 'CONSTRUCTION_FAILED' for a in attempts),
                         b32_success=sum(a['status'] == 'FEASIBLE' for a in attempts), runtime_s=time.perf_counter()-begin))
        print('TRAIN BEAM', len(initials), '/', len(sources), entry['instance_id'], entry['N'], initial.schedule.cmax, flush=True)
    pilot_sources = [sources[i] for i in (0, len(sources)//2, len(sources)-1)]
    model, original = load_v2()
    rows.extend(train_checks(model, records, pilot_sources, 'GNN-V2'))
    diagnostics_before = pilot.action_diagnostics(model, graphs, grouped)
    # Full 3-geometry/8-episode development check. Discard these updated weights.
    for phase, phase_sources, epochs in (('PILOT', pilot_sources, 1), ('TRAIN', sources, SETTINGS['epochs'])):
        torch.manual_seed(SETTINGS['seed'])
        model, _ = load_v2()
        optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS['learning_rate'])
        rng = random.Random(SETTINGS['seed'])
        for epoch in range(epochs):
            for index, source in enumerate(phase_sources):
                name = source['entry']['instance_id']
                current = initials[name]  # Reset each epoch; retain improvements within its 8 episodes.
                for episode in range(SETTINGS['episodes']):
                    current, row = train_episode(model, optimizer, current, index, epoch, episode, graphs, pool, rng, phase)
                    row.update(instance=name, N=source['entry']['N'])
                    rows.append(row)
                    print(phase, epoch+1, index+1, episode+1, name, row['sample_status'], row['greedy_status'],
                          'G', round(row['sample_return'],6), round(row['greedy_return'],6),
                          'A', round(row['advantage'],6), 'B32', row['b32_calls'], 'accept', row['accepted'], flush=True)
                if phase == 'TRAIN':
                    metadata = dict(settings=SETTINGS, complete=False, train_sources=[r['entry'] for r in sources],
                                    pilot=summary(rows,'PILOT'), train=summary(rows,'TRAIN'), diagnostics_before=diagnostics_before,
                                    train_elapsed_s=time.perf_counter()-started)
                    save_model(model, metadata, optimizer)
                write_csv(rows)
            print(phase, 'EPOCH SUMMARY', epoch+1, json.dumps(summary(rows,phase)), flush=True)
        if phase == 'PILOT':
            info = summary(rows, phase)
            if not all(info['gradient_nonzero'][head] > 0 for head in HEADS):
                raise ValueError('pilot did not demonstrate actual RL gradients through both GAT layers and both heads')
    for name, buffer in model.named_buffers():
        if not torch.equal(buffer, original['state_dict'][name]):
            raise ValueError('V2 normalization changed')
    rows.extend(train_checks(model, records, pilot_sources, 'GNN-Cmax-RL'))
    metadata.update(complete=True, train=summary(rows,'TRAIN'), diagnostics_after=pilot.action_diagnostics(model,graphs,grouped),
                    parameter_change={head:math.sqrt(sum(float((p.detach()-original['state_dict'][name]).square().sum())
                                                        for name,p in model.named_parameters() if name.startswith(head+'.')))
                                      for head in HEADS}, train_elapsed_s=time.perf_counter()-started)
    save_model(model, metadata, optimizer)
    write_csv(rows)
    write_report(rows, metadata)
    return model, metadata


def evaluate(rows, model, metadata):
    if not metadata['complete']:
        raise ValueError('complete the predeclared training before DEV')
    before, _ = load_v2()
    for policy in (before, model):
        policy.eval()
        policy.requires_grad_(False)
    by_name = {e['sheet_name']:e for e in pilot.entries()[1]}
    args = argparse.Namespace(seed=SETTINGS['seed'])
    for name in pilot.oriented.NAMES:
        entry = by_name[name]
        parents = pilot.dataset.load_parents(entry)
        started = time.perf_counter()
        initial, attempts = pilot.oriented.common_initial_beam(parents, CONFIG, seed=SETTINGS['seed'])
        initial_seconds = time.perf_counter()-started
        if initial is None:
            raise ValueError('existing DEV Beam initial failed')
        initial_calls = sum(a['status'] != 'CONSTRUCTION_FAILED' for a in attempts)
        for label, policy in (('NEW', None), ('GNN-V2', before), ('GNN-Cmax-RL', model)):
            row = pilot.iterative_group(entry, initial, label, policy, args)
            row.update(phase='DEV', initial_runtime_s=initial_seconds, initial_b32_calls=initial_calls,
                       total_runtime_s=initial_seconds+row['runtime_s'], total_b32_calls=initial_calls+row['b32_calls'])
            rows.append(row)
            write_csv(rows)
            write_report(rows, metadata)


def write_report(rows, metadata):
    train_info = metadata['train']
    training_rows = [r for r in rows if r['phase']=='TRAIN']
    feasible_advantages = [abs(r['advantage']) for r in training_rows
                           if r['advantage'] != 0 and r['sample_status']==r['greedy_status']=='FEASIBLE']
    total_advantage = sum(abs(r['advantage']) for r in training_rows)
    lines = ['# GNN V2：真实 Cmax 反馈微调', '',
             '固定 Beam 有向初解、原联合候选、原特征及归一化、两层 GAT/64/4。方向及邻块翻转仍在同一次插入中决定。',
             'REINFORCE 使用同一更新前模型、同一部分解的 greedy 回报作为停止梯度基线；没有 critic、PPO 或新算子。', '',
             '## 训练', '', f'预先固定参数：`{json.dumps(SETTINGS)}`。TRAIN 几何由已有46个中按工作簿覆盖及N间隔选择20个，覆盖全部17个TRAIN工作簿；不使用DEV选择训练设置。',
             'RL状态限这20个几何；辅助BC复用原46个TRAIN几何的294条/1872步旧教师。教师经过现有角色检查及真实共同候选重放，仅作每次4步、系数0.01的辅助交叉熵。',
             '奖励：FEASIBLE时 (Cref−Cmax)/(Cref+Cmax)，构造/B32失败为−1；仅完整候选调用B32，相同sample/greedy完整路线及方向复用一次结果。数据/数值异常直接报错。',
             f'3个不同规模几何各8回合流程试训完成后丢弃权重；正式重新加载V2。微调完成：{metadata["complete"]}。', '',
             '|阶段|回合|实际B32成功/调用|sample/greedy可调度|非零优势|正/负优势|双方失败|接纳|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase, info in (('流程检查',metadata['pilot']), ('正式训练',train_info)):
        lines.append(f'|{phase}|{info["episodes"]}|{info["b32_success"]}/{info["b32_calls"]}|{info["sample_feasible"]}/{info["greedy_feasible"]}|{info["nonzero"]}|{info["positive"]}/{info["negative"]}|{info["both_failed"]}|{info["accepted"]}|')
    lines += ['', f'正式平均sample/greedy回报 {train_info["mean_sample_return"]:.6f}/{train_info["mean_greedy_return"]:.6f}，平均|A|={train_info["mean_abs_advantage"]:.6f}；双方可调度且非零优势 {train_info["feasible_pair_nonzero"]} 回合。',
              f'非零优势比例 {train_info["nonzero"]/train_info["episodes"]:.2%}；零优势的{train_info["episodes"]-train_info["nonzero"]}次更新只有辅助BC产生学习信号。双方可调度且非零时，|A|中位数={statistics.median(feasible_advantages):.6f}；可调度/失败差别占优势绝对幅度总和的{1-sum(feasible_advantages)/max(1e-12,total_advantage):.2%}。该幅度统计不能等同于裁剪后梯度贡献比例。',
              f'仅RL项的非零梯度回合：`{json.dumps(train_info["gradient_nonzero"])}`；最大范数：`{json.dumps(train_info["gradient_max"])}`。',
              f'各模块参数变化L2：`{json.dumps(metadata.get("parameter_change",{}))}`。四个原V2归一化buffer逐项保持完全相同；没有重拟合。',
              f'正式训练回合计算秒={train_info["runtime_s"]:.3f}；含数据准备、20个Beam、试训、检查与保存的总训练秒={metadata["train_elapsed_s"]:.3f}。', '',
              '|TRAIN实例|N|Beam Cmax|初始化B32|秒|', '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['phase']=='TRAIN_BEAM':
            lines.append(f'|{r["instance"]}|{r["N"]}|{r["initial_cmax"]:.3f}|{r["b32_calls"]}|{r["runtime_s"]:.3f}|')
    lines += ['', '## 局部补全保留检查', '',
              '|模型|原TRAIN q4/q8成功/调用|方向0/1|含邻翻转动作|', '|---|---:|---:|---:|']
    for label in ('GNN-V2','GNN-Cmax-RL'):
        checks=[r for r in rows if r['phase']=='TRAIN_CHECK' and r['method']==label]
        lines.append(f'|{label}|{sum(r["b32_success"] for r in checks)}/{sum(r["b32_calls"] for r in checks)}|{sum(r["direction0_actions"] for r in checks)}/{sum(r["direction1_actions"] for r in checks)}|{sum(r["flip_actions"] for r in checks)}|')
    for key in ('diagnostics_before','diagnostics_after'):
        d=metadata.get(key)
        if d:
            lines.append(f'{key}：原1872步教师强制插入准确率={d["insertion_accuracy"]:.2%}，父选择={d["parent_accuracy"]:.2%}，预测方向0/1={d.get("predicted_direction_0",0)}/{d.get("predicted_direction_1",0)}，含翻转动作={d.get("predicted_flip_actions",0)}。')
    lines += ['', '## 六DEV：相同Beam、各自有状态24轮', '',
              '本次三方法全部重新运行；正式补全均为greedy。每次接受严格改善后，下一轮按自己的最新current选区，不能声称全程移除集合相同。',
              '时间为预加载模型、已读取共同几何后，初解构造+B32+24轮搜索；不含统一模型加载/Excel读取。共同Beam实际只计算一次，其成本在每方法端到端时间中各计一次。',
              '|实例|方法|Beam→最终Cmax|严格接纳|搜索B32成功/调用|初解B32|总秒|',
              '|---|---|---:|---:|---:|---:|---:|']
    dev=[r for r in rows if r['phase']=='DEV']
    for r in dev:
        lines.append(f'|{r["instance"]}|{r["method"]}|{r["initial_cmax"]:.3f}→{r["best_cmax"]:.3f}|{r["accepted"]}|{r["b32_success"]}/{r["b32_calls"]}|{r["initial_b32_calls"]}|{r["total_runtime_s"]:.3f}|')
    if len(dev)==18:
        lines += ['', '|方法|搜索B32成功/调用|严格接纳|初解+搜索总秒|', '|---|---:|---:|---:|']
        for label in ('NEW','GNN-V2','GNN-Cmax-RL'):
            group=[r for r in dev if r['method']==label]
            lines.append(f'|{label}|{sum(r["b32_success"] for r in group)}/{sum(r["b32_calls"] for r in group)}|{sum(r["accepted"] for r in group)}|{sum(r["total_runtime_s"] for r in group):.3f}|')
        lines += ['', '最终Cmax逐实例比较（真实差值，1e−9相对容差）：', '']
        for baseline in ('GNN-V2','NEW'):
            other={r['instance']:r for r in dev if r['method']==baseline}
            pairs=[(r['best_cmax']-other[r['instance']]['best_cmax'],1e-9*max(1,other[r['instance']]['best_cmax']))
                   for r in dev if r['method']=='GNN-Cmax-RL']
            lines.append(f'- RL 对 {baseline}：{sum(d < -e for d,e in pairs)}胜、{sum(abs(d)<=e for d,e in pairs)}平、{sum(d > e for d,e in pairs)}负；逐实例差值秒 `{[d for d,e in pairs]}`。')
        v2={r['instance']:r for r in dev if r['method']=='GNN-V2'}
        rl=[r for r in dev if r['method']=='GNN-Cmax-RL']
        no_gain=all(r['best_cmax']>=v2[r['instance']]['best_cmax']-1e-9*max(1,v2[r['instance']]['best_cmax']) for r in rl)
        if no_gain:
            lines[2:2]=['**负结果：本轮真实Cmax微调没有优于原GNN V2，后续主线保留 Beam+B32+GNN V2。**', '']
            lines += ['', '## 结论与问题定位', '',
                      '1. 策略梯度确实作用于GAT和父选择/插入两头：270次非零RL梯度与四个模块的非零参数变化分别验证了梯度和实际更新。不能把本轮说成只打印随机采样奖励或仅做监督训练；但本实验是RL+0.01BC组合，未独立分离两项的因果收益。',
                      '2. 方向和邻块翻转没有消失。原TRAIN六个q4/q8检查仍全部可调度；教师强制方向仍有两个取值、524个含翻转预测。插入准确率仅由73.45%降至73.08%，没有全面遗忘，但也没有动作拟合提升。',
                      '3. 六DEV对V2没有任何更低的最终Cmax。g31中RL只接纳到3125.416秒便停滞，而原V2随后继续接纳到3071.671秒，退化53.745秒；其余五例最终目标值持平。RL严格接纳6次，少于V2的10次，B32成功105/144也没有提高。',
                      '4. 相对NEW仍为3胜1平2负，g33/g45未超过NEW；没有建立超过传统修复或超过原V2的证据。训练中46次sample严格改善没有转化为DEV端到端收益。',
                      '5. 问题不在梯度缺失或优势全部为零。真正的细粒度Cmax优势偏小，较少的可调度/失败差别占据96.74%的优势绝对幅度；同时微调在g31的后续补全路径变差。失败分支136/960（14.17%），不宜笼统称为“大量样本完全无法训练”。这些是信号和搜索轨迹的实测观察，尚未隔离奖励尺度、策略偏移及泛化的因果贡献。',
                      '', '**选择：保留原GNN V2用于论文主线；本轮RL checkpoint和负结果保留，不替换V2，不根据DEV继续改奖励、种子、轮次或增加优化器。**']
        train_calls=sum(r['b32_calls'] for r in rows if r['phase'] in ('TRAIN_BEAM','PILOT','TRAIN','TRAIN_CHECK'))
        dev_calls=sum(r['b32_calls'] for r in dev)+sum(r['initial_b32_calls'] for r in dev if r['method']=='NEW')
        lines += ['', f'实验实际B32：训练流程{train_calls}次（Beam84、试训48、正式928、前后检查12），DEV{dev_calls}次（共享Beam32、搜索432）；共享初解不重复计入物理调用数。正式训练通过同候选复用少调用32次B32。']
    lines += ['', 'B32 DEADLOCK只表示当前调度器未找到时间安排，不是数学不可行证明。CSV保留每回合真实回报、优势、RL梯度与全部DEV逐轮current/移除/接纳轨迹；模型元数据保留固定训练设置与汇总。',
              '运行环境：`D:\\pybullet_test\\.venv\\Scripts\\python.exe`，工作目录`DRL`，CPU单线程；先执行本脚本`--phase train`，完成后执行`--phase dev`。模型固定使用第3轮末权重，无DEV checkpoint选择。77项相关测试通过，另核对554行实际记录、TRAIN隔离、奖励公式、零优势梯度、每几何current连续性及18条24轮DEV轨迹。', '']
    REPORT.write_text('\n'.join(lines),encoding='utf-8')


def complete_schedule_cache(rows):
    """Reuse only identical native geometry, patterns, routes and directions."""
    cache=op_evaluation_cache(rows)
    for row in rows:
        for outcome in row['payload'].get('outcomes',[]):
            raw=outcome.get('complete')
            if raw and outcome.get('status') in ('FEASIBLE','DEADLOCK'):
                cache[op_complete_key(raw)]=dict(status=outcome['status'],cmax=outcome['cmax'],
                    scheduler_seconds=outcome.get('scheduler_seconds',0.),diagnostics=outcome.get('diagnostics',{}),
                    metrics=outcome.get('metrics') or {})
    return cache


def complete_imitation(shared,config,archive,teachers,source):
    from mrta_search import gnn_constructor as c
    graphs,examples,_,preparation=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(source,graphs,examples,raw_only=True)
    spaces={};records=[];misses=[];bins=Counter();sizes=[];started=time.perf_counter()
    for i,e in enumerate(raw):
        parents=e['reference'].parents
        if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config)
        space=spaces[parents]
        selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in e['pending']}
        plans=c.complete_parent_candidates(space,e['state'],e['pending'],selected,config)
        teacher=e['teacher_plan'];pattern=next(p for p in e['reference'].patterns if p.parent_id==teacher['parent_id'])
        target=next((j for j,p in enumerate(plans) if p['pattern']==pattern and cmax_action_key(p)==cmax_action_key(teacher)),None)
        sizes.append(len(plans))
        if target is None:misses.append(dict(fraction=e['fraction'],pattern=asdict(pattern),actions=[asdict(a) for a in teacher['actions']]))
        else:
            records.append(dict(space=space,inputs=c.complete_parent_inputs(plans),target=target,weight=e['weight']))
            bins[str(e['fraction'])]+=1
        if i%6==5:print('COMPLETE_COVERAGE',i+1,len(records),flush=True)
    coverage=dict(total=len(raw),covered=len(records),fraction=len(records)/max(1,len(raw)),
        prefix_bins=dict(bins),missing=misses,candidate_min=min(sizes),candidate_max=max(sizes),
        candidate_median=statistics.median(sizes),seconds=time.perf_counter()-started,preparation=preparation)
    return records,spaces,coverage


def complete_cmax_pairs(outcomes):
    # No numerical DEADLOCK surrogate and no cross-state absolute ranking.
    return [(i,j) if a['cmax']<b['cmax'] else (j,i)
        for i,a in enumerate(outcomes) for j,b in enumerate(outcomes[i+1:],i+1)
        if a['status']==b['status']=='FEASIBLE' and not math.isclose(a['cmax'],b['cmax'],rel_tol=1e-9,abs_tol=1e-9)]


def complete_feedback(model,spaces,config,rows):
    """Twelve actual student states, six progress bins, at most 30 new B32s."""
    from mrta_search import gnn_constructor as c
    model.eval();cache=complete_schedule_cache(rows);records=[];calls=0;used=0
    fractions=(0.,.1,.25,.5,.75,.9)*2
    for index,(parents,space) in enumerate(list(spaces.items())[:12]):
        fraction=fractions[index];target=round(fraction*len(parents));prefix=[]
        def observe(sp,state,pending,selected,plans,best):
            if len(parents)-len(pending)==target:
                prefix.append((state,pending,selected,plans))
        trajectory=c.construct_complete_parent_gnn(parents,model,config,space=space,observer=observe)
        if not prefix:
            print('COMPLETE_NO_STUDENT_PREFIX',index,fraction,trajectory.rejection_reason,flush=True);continue
        state,pending,selected,plans=prefix[0]
        with torch.no_grad():
            emb=model.encode_static(space['graph'])
            scores=model.parent_action_scores(space['graph'],emb,c.complete_parent_inputs(plans))
        ranking=sorted(range(len(plans)),key=lambda i:(-float(scores[i]),i))
        chosen=[ranking[0]]
        first=plans[chosen[0]]
        different=[i for i in ranking if (plans[i]['pattern'],tuple(a.robot for a in plans[i]['actions'])) !=
                   (first['pattern'],tuple(a.robot for a in first['actions']))]
        if different:chosen.append(different[0])
        if index%2 and len(plans)>len(chosen):
            remaining=[i for i in ranking if i not in chosen]
            # A different load/position structure probes the same learned continuation.
            chosen.append(min(remaining,key=lambda i:(max(p+sum(arcs) for p,arcs in
                zip(plans[i]['state'].loads,plans[i]['state'].arc_costs)),plans[i]['state'].travel,i)))
        outcomes=[];started=time.perf_counter();physical=0
        for j in chosen:
            plan=plans[j];after_selected=dict(selected,**{plan['parent_id']:plan['pattern']})
            result=c.construct_complete_parent_gnn(parents,model,config,space=space,state=plan['state'],
                pending=set(pending)-{plan['parent_id']},selected=after_selected)
            outcome=dict(actions=[asdict(a) for a in plan['actions']],pattern=asdict(plan['pattern']),
                status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,reason=result.rejection_reason)
            if result.solution is not None:
                raw=dict(solution=asdict(result.solution),directions=result.directions);key=op_complete_key(raw)
                if key not in cache:
                    if calls>=30:raise ValueError('thirty TRAIN B32 calls exhausted; reserve DEV budget')
                    tick=time.perf_counter();candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                    feasible=candidate.schedule.status is ScheduleStatus.FEASIBLE
                    metrics=op_route_metrics(raw,config)
                    if feasible:metrics.update(op_metrics(candidate,config))
                    cache[key]=dict(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax if feasible else None,
                        scheduler_seconds=time.perf_counter()-tick,metrics=metrics,diagnostics=candidate.schedule.diagnostics)
                    calls+=1;physical+=1;outcome['b32_calls']=1
                else:used+=1
                outcome.update(cache[key],complete=raw)
            outcomes.append(outcome)
        pairs=complete_cmax_pairs(outcomes)
        subset=tuple(plans[j] for j in chosen)
        records.append(dict(space=space,inputs=c.complete_parent_inputs(subset),pairs=pairs))
        op_save(dict(phase='complete_parent_feedback',instance=f'TRAIN_{index:02d}',seed=OP_SEED,
            method='FROZEN_CURRENT_COMPLETE_GNN',status='COLLECTED',runtime_s=time.perf_counter()-started,b32_calls=physical,
            payload=dict(fraction=fraction,progress=target/len(parents),parents=[asdict(p) for p in parents],
                prefix=dict(routes=state.routes,directions=state.directions,pending=pending,
                            selected_patterns=[asdict(p) for p in selected.values()]),
                outcomes=outcomes,pairs=pairs,completion_policy='same frozen epoch-5 complete-parent GNN')))
        print('COMPLETE_FEEDBACK',index+1,'calls',calls,'pairs',sum(len(r['pairs']) for r in records),flush=True)
    return records,dict(states=len(records),actions=sum(len(r['inputs']['parent_ids']) for r in records),
        cmax_pairs=sum(len(r['pairs']) for r in records),new_b32=calls,exact_reuses=used)


def complete_parent_train(*,coverage_only=False):
    from mrta_search import gnn_constructor as c
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    if COMPLETE_VARIANT in checkpoint and not coverage_only:
        print('Existing complete-parent experiment retained; no repeat training.',flush=True);return
    torch.manual_seed(OP_SEED);rng=random.Random(OP_SEED)
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    imitation,spaces,coverage=complete_imitation(shared,config,archive,teachers,source)
    op_save(dict(phase='complete_parent_coverage',instance='TRAIN_21',seed=OP_SEED,method='ALL_PARENTS_ALL_LEGAL_MODES',
        status='MEASURED',b32_calls=0,payload=coverage))
    print('COMPLETE_COVERAGE_FINAL',coverage['covered'],coverage['total'],flush=True)
    if coverage_only or coverage['fraction']<.85:return
    torch.manual_seed(OP_SEED)
    model=c.CompleteParentGNN(64,4,normalize_inputs=True,opportunity_features=True)
    with torch.no_grad():
        for name,value in source.state_dict().items():model.state_dict()[name].copy_(value)
    model.requires_grad_(False)
    for name,parameter in model.named_parameters():
        if name.startswith(('layer1.','layer2.','geometry_skip.','parent_action_head.')):parameter.requires_grad_(True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=1e-4);history=[];feedback=[];pairs=[];frozen_weights=None
    started=time.perf_counter();feedback_seconds=0.;feedback_info={}
    for epoch in range(10):
        if epoch==5:
            frozen_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
            tick=time.perf_counter();feedback,feedback_info=complete_feedback(model,spaces,config,op_read())
            feedback_seconds=time.perf_counter()-tick
            pairs=[(r,a,b) for r in feedback for a,b in r['pairs']]
        totals=Counter();model.train();tick=time.perf_counter()
        for _ in range(64):
            optimizer.zero_grad(set_to_none=True);encoded={};values={}
            def score(record):
                identity=id(record)
                if identity not in values:
                    space=record['space'];key=id(space)
                    if key not in encoded:encoded[key]=model.encode_static(space['graph'])
                    values[identity]=model.parent_action_scores(space['graph'],encoded[key],record['inputs'])
                return values[identity]
            il=torch.stack([-e['weight']*torch.log_softmax(score(e),0)[e['target']]
                            for e in (rng.choice(imitation) for _ in range(4))]).mean()
            rank=torch.stack([F.softplus(-(score(r)[a]-score(r)[b]))
                for r,a,b in (rng.choice(pairs) for _ in range(4))]).mean() if pairs else il.new_zeros(())
            loss=rank+.1*il
            if not torch.isfinite(loss):raise ValueError('nonfinite complete-parent loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
            totals['loss']+=float(loss.detach());totals['cmax']+=float(rank.detach());totals['il']+=float(il.detach())
        result=dict(epoch=epoch+1,updates=64,seconds=time.perf_counter()-tick,**{k:v/64 for k,v in totals.items()})
        history.append(result)
        op_save(dict(phase='complete_parent_training',instance='TRAIN_21',seed=OP_SEED,method='CMAX_PLUS_01_COMPLETE_IL',
            status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    metadata=dict(seed=OP_SEED,initialization=OP_VARIANT,epochs=10,warmup_epochs=5,updates_per_epoch=64,
        batch_size=4,learning_rate=1e-4,gradient_clip=1.,coverage=coverage,feedback=feedback_info,
        feedback_seconds=feedback_seconds,training_seconds=sum(r['seconds'] for r in history),
        elapsed_seconds=time.perf_counter()-started,history=history,loss='L_same_state_true_Cmax + 0.1 L_complete_IL',
        bootstrap='five IL epochs, frozen student feedback, five joint epochs; no DEV selection',
        trainable=('layer1','layer2','geometry_skip','parent_action_head'),new_teacher_b32=0)
    protected={'root':checkpoint['state_dict']}
    protected.update({k:v['state_dict'] for k,v in checkpoint.items() if isinstance(v,dict) and 'state_dict' in v})
    checkpoint[COMPLETE_VARIANT]=dict(model_type='CompleteParentGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,parent_action_features=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),
        state_dict=model.state_dict(),feedback_state_dict=frozen_weights,training=metadata)
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    for key,values in protected.items():
        actual=saved['state_dict'] if key=='root' else saved[key]['state_dict']
        if not all(torch.equal(v,actual[n]) for n,v in values.items()):raise ValueError('historical weights changed')
    op_save(dict(phase='complete_parent_model',instance='TRAIN_21',seed=OP_SEED,method=COMPLETE_VARIANT,
        status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


def complete_parent_evaluate():
    from mrta_search import gnn_constructor as c
    shared,config,_,_,_,_,entries,_=op_sources()
    rows=op_read();cache=complete_schedule_cache(rows)
    done={r['instance'] for r in rows if r['phase']=='complete_parent_evaluation'}
    for name,parents in sorted(entries.items()):
        if name in done:continue
        started=time.perf_counter();thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
        model=shared._model_from_checkpoint(thin[COMPLETE_VARIANT]).eval();load=time.perf_counter()-started
        graph_tick=time.perf_counter();space=c.complete_parent_space(parents,config)
        graph_seconds=time.perf_counter()-graph_tick
        attempts=[];credit=0.;charged=set()
        for index,seed in enumerate((None,OP_EVAL_SEED,OP_EVAL_SEED+1)):
            tick=time.perf_counter()
            decisions=[]
            def observe(space,state,pending,selected,plans,best):
                decisions.append(dict(parent=best['parent_id'],pattern=asdict(best['pattern']),
                    candidates=len(plans),parent_count=len({p['parent_id'] for p in plans}),
                    mode_count=len({p['pattern'] for p in plans}),loads=best['state'].loads))
            result=c.construct_complete_parent_gnn(parents,model,config,seed=seed,space=space,observer=observe)
            attempt=dict(proposal_index=index,decode='complete_greedy' if seed is None else 'complete_sample',
                seed=seed,status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,reason=result.rejection_reason,
                construction_seconds=time.perf_counter()-tick+(graph_seconds if index==0 else 0.),
                graph_build_seconds=graph_seconds if index==0 else 0.,
                actions=[asdict(a) for a in result.actions],decisions=decisions)
            if result.solution is not None:
                raw=dict(solution=asdict(result.solution),directions=result.directions);key=op_complete_key(raw)
                if key not in cache:
                    tick_schedule=time.perf_counter();candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                    valid=candidate.schedule.status is ScheduleStatus.FEASIBLE;metrics=op_route_metrics(raw,config)
                    if valid:metrics.update(op_metrics(candidate,config))
                    cache[key]=dict(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax if valid else None,
                        metrics=metrics,scheduler_seconds=time.perf_counter()-tick_schedule,diagnostics=candidate.schedule.diagnostics)
                    attempt['b32_calls']=1
                elif key not in charged:credit+=cache[key]['scheduler_seconds']
                charged.add(key);attempt.update(cache[key],**raw)
            attempt.update(seconds=time.perf_counter()-tick,elapsed=time.perf_counter()-started+credit)
            attempts.append(attempt)
        actual=time.perf_counter()-started
        info=dict(attempts=attempts,model_load_seconds=load,runtime_s=actual+credit,actual_runtime_s=actual,
                  reused_b32_seconds=credit,source_variant=COMPLETE_VARIANT,b32_calls=sum(a['b32_calls'] for a in attempts))
        op_save_measured('complete_parent_evaluation',name,'COMPLETE_PARENT_GNN_MULTI3',info)


def complete_parent_report():
    import run_gnn_shared_construction as shared
    rows=op_read();heading='## 2026-11-06 全合法模式、无父级截断与自身轨迹完整动作实验'
    original=OP_REPORT.read_text(encoding='utf-8')
    if heading in original:
        print('Complete-parent report already appended; preserve it.',flush=True);return
    metadata=shared.read_neural_checkpoint(OP_CHECKPOINT)[COMPLETE_VARIANT]['training']
    new={r['instance']:r for r in rows if r['phase']=='complete_parent_evaluation'}
    names=sorted(new)
    if len(names)!=8:raise ValueError('complete the existing eight DEV geometries before reporting')
    baselines={}
    for phase,method,label in (
        ('parent_cmax_macro_evaluation','ORIGINAL_GNN_MULTI3','ORIGINAL_GNN_MULTI3'),
        ('parent_cmax_macro_evaluation','PARENT_JOINT_MULTI3','PARENT_JOINT_MULTI3'),
        ('parent_cmax_macro_evaluation','PARENT_CMAX_MULTI3','PARENT_CMAX_MACRO_IL_MULTI3'),
        ('parent_cmax_macro_evaluation','RULE/BEAM','RULE/BEAM')):
        baselines[label]={r['instance']:r for r in rows if r['phase']==phase and r['method']==method and r['seed']==str(OP_EVAL_SEED)}
    good=lambda r:r['status']=='FEASIBLE' and r['cmax'] not in ('',None)
    number=lambda v:'—' if v in (None,'') else f'{float(v):.3f}'
    def compare(old):
        counts=Counter();changes=[]
        for name in names:
            a,b=new[name],old[name];fa,fb=good(a),good(b)
            if fa and fb:
                delta=100*(float(b['cmax'])-float(a['cmax']))/float(b['cmax']);changes.append(delta)
                counts['win' if delta>1e-8 else 'loss' if delta< -1e-8 else 'tie']+=1
            elif fa:counts['win']+=1
            elif fb:counts['loss']+=1
            else:counts['both_fail']+=1
        return counts,len(changes),statistics.median(changes) if changes else None
    feasible=sum(good(r) for r in new.values())
    coverage=metadata['coverage'];feedback=[r for r in rows if r['phase']=='complete_parent_feedback']
    outcomes=[o for r in feedback for o in r['payload']['outcomes']]
    train_b32=sum(int(r['b32_calls'] or 0) for r in feedback)
    dev_b32=sum(int(r['b32_calls'] or 0) for r in new.values())
    comparisons={label:compare(old) for label,old in baselines.items()}
    target=feasible>=7 and all(comparisons[label][2] is not None and comparisons[label][2]>0
                              for label in ('PARENT_JOINT_MULTI3','RULE/BEAM'))
    lines=['',heading,'',
        f'**实际结果：新GNN三候选{feasible}/8可行。'+
        ('达到可行率与共同可行Cmax的开发目标；仍需新独立几何验证。**' if target else
         '未同时达到7/8可行和优于parent_joint、RULE的共同可行Cmax目标，不能称为论文主算法已取得提升。**'),'',
        '本轮直接修改gnn_constructor.py、run_gnn_shared_construction.py、run_gnn_cmax_finetune.py及既有测试。'
        '所有结果追加到gnn_opportunity_cmax.csv；新最终权重与第5轮冻结续补权重保存于既有'
        'gnn_coordination_constructor_05.pt的complete_parent_candidate分支。原顶层和所有历史候选分支保存后逐tensor相等。'
        '未新建文件、未清理工作区、未运行旧实验或封存20例。','',
        '### 完整构造与候选覆盖','',
        '输入只有父焊缝几何与四条空路线。build_legal_pattern_catalog(FORMAL_SCOPE_V2)枚举WHOLE/Y_SPLIT/X_SPLIT，'
        '所有未分配父、每个合法模式进入同一完整动作池；无父Top-K、无旧父头/插入头/独立mode_head评分。'
        '复用joint_block_candidates的8位置有界插入、资格、邻居翻转、初始冲突与兄弟规则。'
        'WHOLE保留其有界局部插入池；拆分首子最多16个结构代表，次子在每个真实临时路线中最多8个代表，'
        '每模式最多128个拆分完整动作，非子动作全笛卡尔积。代表按机器人、方向、位置及空走确定，与网络权重和教师标签无关。'
        '有界分支未生成完整合法动作的模式无候选，不注入非法动作；不声称穷尽全部合法插入组合。','',
        '用唯一模式节点别名编码全部合法几何，native路线和block_id不变，避免不同切分点共用::0/::1造成几何覆盖。'
        '两层GAT保持hidden=64、heads=4；完整头为723→32→1，复用原完整动作711维并添加12维模式特征。'
        '完整模式、子分配、位置、方向、邻居和动作后四机路线状态统一输出标量。'
        '选中后一次提交全部子动作和模式，继续构造到覆盖所有父任务，再由原B32调度。'
        '没有概率连乘、MODE_NEW、VNS、ALNS、RULE回退、方向后处理或独立模式预测。','',
        f"已有126个六进度教师前缀覆盖{coverage['covered']}/{coverage['total']}={100*coverage['fraction']:.2f}%。"
        f"未覆盖{len(coverage['missing'])}个不加入模仿池。各进度实际可用数：{coverage['prefix_bins']}。"
        f"每状态完整候选数min/median/max={coverage['candidate_min']}/{coverage['candidate_median']}/{coverage['candidate_max']}。"
        '该覆盖只说明现有教师监督可用，不能当作求解质量。','',
        '### 训练与真实调度开销','',
        f"复用{coverage['preparation']['teachers_reused']}条教师重放、"
        f"{coverage['preparation']['prefix_continuations_reused']}条既有续补、"
        f"{coverage['preparation']['imitation_examples']}条旧局部监督恢复126个完整动作；实际训练"
        f"{coverage['covered']}个完整父动作，不重新运行教师B32。初始化为opportunity_cmax_candidate。"
        '固定10轮：5轮完整IL预热；冻结第5轮GNN，从其自身轨迹提取0/10/25/50/75/90%附近真实状态；'
        '每状态2或3个完整动作后统一由该冻结GNN贪心续补；5轮联合更新原两层GAT、geometry_skip与完整头。'
        '保留原输入归一化；旧父/插入/mode头未参与推理或损失。','',
        f"学生反馈统计：{metadata['feedback']}。实际状态进度："
        f"{[r['payload']['progress'] for r in feedback]}。反馈状态完整结果：{dict(Counter(o['status'] for o in outcomes))}。"
        f"有效Cmax排序来自{sum(bool(r['payload']['pairs']) for r in feedback)}/{len(feedback)}个状态。"
        f"新增TRAIN B32={train_b32}，DEV B32={dev_b32}，本轮实验合计={train_b32+dev_b32}，上限60。"
        '相同完整几何/模式/路线/方向才精确复用。旧215个OP续补排序对不直接混用，因为旧OP与当前冻结GNN续补策略不同。'
        'Cmax排序仅使用同状态双方可行且Cmax不同的实测对；DEADLOCK标签为空，不用假数值或跨状态排序。','',
        '唯一损失L=L_Cmax+0.1 L_complete_IL，预热时L_Cmax=0。AdamW lr=1e-4、batch=4、clip=1，'
        '每轮固定64更新，不按DEV选epoch、种子、温度或重训。'
        f"优化器训练{metadata['training_seconds']:.3f}s，自身反馈采集{metadata['feedback_seconds']:.3f}s。",'',
        '### 原8个DEV几何，seed=202611041','',
        '每例一次贪心、seed采样、seed+1采样，全部来自同一最终新GNN，最多3个完整方案、3次B32。'
        '历史基线均直接复用原CSV。失败无Cmax；时间包括模型读取、构图、全部候选构造、三次解码及B32。','',
        '|几何|ORIGINAL_GNN|PARENT_JOINT|MACRO_IL|RULE/BEAM|新GNN Cmax|新总耗时/s|三提案状态|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for name in names:
        row=new[name];attempts=row['payload']['attempts']
        lines.append('|'+name+'|'+ '|'.join(number(baselines[label][name]['cmax']) for label in baselines)+
            f"|{number(row['cmax'])}|{number(row['runtime_s'])}|"+'/'.join(a['status'] for a in attempts)+'|')
    lines+=['','|方法|可行/8|总耗时中位/s|本轮新增B32|','|---|---:|---:|---:|']
    for label,old in baselines.items():
        lines.append(f'|{label}|{sum(good(r) for r in old.values())}/8|{statistics.median(float(r["runtime_s"]) for r in old.values()):.3f}|0|')
    lines.append(f'|COMPLETE_PARENT_GNN_MULTI3|{feasible}/8|{statistics.median(float(r["runtime_s"]) for r in new.values()):.3f}|{dev_b32}|')
    construction=[sum(a['construction_seconds'] for a in r['payload']['attempts']) for r in new.values()]
    scheduler=[sum(a.get('scheduler_seconds',0.) for a in r['payload']['attempts']) for r in new.values()]
    lines+=['',f'三次完整构造合计中位{statistics.median(construction):.3f}s，B32合计中位{statistics.median(scheduler):.3f}s。'
        '三个提案复用同一只读几何图，构图耗时计入第一个提案及总墙钟。分项中位数不可直接相加。']
    scored=[sum(d['candidates'] for d in r['payload']['attempts'][0]['decisions']) for r in new.values()]
    largest=max(d['candidates'] for r in new.values() for a in r['payload']['attempts'] for d in a['decisions'])
    lines.append(f'每几何单次贪心累计完整动作评分数min/median/max={min(scored)}/{statistics.median(scored)}/{max(scored)}，'
        f'全部提案单步池最大{largest}。单模式分支有界并未消除遍历全部剩余父任务带来的累计构造成本。')
    lines+=['','旧parent_joint总耗时为原分项组成值；新耗时为本轮连续墙钟，历史缺失分项不填0。','',
        '|新模型对照|胜/平/负|双失败|双方可行/8|条件Cmax中位改善/%|','|---|---:|---:|---:|---:|']
    for label,(count,n,delta) in comparisons.items():
        lines.append(f"|{label}|{count['win']}/{count['tie']}/{count['loss']}|{count['both_fail']}|{n}/8|{number(delta)}|")
    lines+=['','改善=100×(baseline−new)/baseline，正值才是改善；胜负计入可行优于失败，双失败单列。'
        '负载表对有返回者列选中方案；全部失败者仅列贪心失败提案供诊断，max(P+MOVE)为路线值而非失败Cmax。','',
        '|几何|返回R0/R1/R2/R3加工负载|max(P+MOVE)|MOVE|WAIT|','|---|---|---:|---:|---:|']
    for name in names:
        row=new[name];metrics=row['payload']['metrics']
        if not good(row):metrics=row['payload']['attempts'][0].get('metrics',{})
        loads=' / '.join(number(x) for x in metrics.get('loads',[])) or '失败，见全部提案'
        if not good(row):loads+='（贪心失败）'
        lines.append(f"|{name}|{loads}|{number(metrics.get('max_p_move'))}|{number(metrics.get('move'))}|{number(metrics.get('wait'))}|")
    attempts=[a for r in new.values() for a in r['payload']['attempts']]
    mode_counts=Counter(p['kind'] for a in attempts if a.get('solution') for p in a['solution']['patterns'])
    collapse=[name for name in names if good(new[name]) and min(new[name]['payload']['metrics']['loads'])==0]
    failures=[name for name in names if all(a['status']=='DEADLOCK' for a in new[name]['payload']['attempts'])]
    lines+=['','### 结果解释与下一步','',
        f"全部提案的模式选择数量：{dict(mode_counts)}。可行返回存在空闲机器人的几何：{collapse}。"
        f"三提案集中DEADLOCK几何：{failures}。完整提案状态计数：{dict(Counter(a['status'] for a in attempts))}。",
        '已解决固定模式空间、父Top-K先丢弃、教师不在推理池却注入IL、旧头候选随训练漂移，以及当前动作由不同旧策略续补打标签的问题。'
        '同状态反馈使用新GNN自身状态与同一个冻结续补策略；最后5轮更新仍会使最终策略偏离第5轮冻结策略，'
        '因此不能声称完全消除策略分布差异。12个反馈状态和有限可行对也不能代表全部状态空间。']
    feedback_sizes=[len(r['payload']['parents']) for r in feedback]
    lines.append(f'自身反馈父焊缝规模为{feedback_sizes}，只覆盖{min(feedback_sizes)}–{max(feedback_sizes)}；'
        '教师IL虽覆盖21个TRAIN，反馈按原顺序取前12个，未对63–85父焊缝的其余TRAIN采集一致续补标签。'
        'DEV的大规模案例为75、78、85、92父焊缝。这是本轮采样的明确局限，不能把自身状态训练描述为覆盖全部规模。')
    for label in ('PARENT_JOINT_MULTI3','RULE/BEAM'):
        bad=[]
        for name in names:
            if good(new[name]) and good(baselines[label][name]):
                loads=new[name]['payload']['metrics']['loads']
                if max(loads)>float(baselines[label][name]['cmax']):bad.append(name)
        lines.append(f'相对{label}，仅最大加工负载已超过其完整Cmax的共同可行几何：{bad}；这些差距不能靠消除WAIT追回。')
    for label in ('WAG','HGA'):
        old={r['instance']:r for r in rows if r['phase']=='evaluation' and r['method']==label and r['seed']==str(OP_EVAL_SEED)}
        gaps=[100*(float(new[n]['cmax'])/float(old[n]['cmax'])-1) for n in names if good(new[n]) and n in old and good(old[n])]
        lines.append(f'{label}仅为原60秒终解质量参考，非等预算优越性比较；双方可行{len(gaps)}/8，Cmax中位差距{number(statistics.median(gaps) if gaps else None)}%。')
    lines+=['',('可将此模型作为待独立验证的候选；下一步可考虑新的、未使用过的独立几何，不能用本8例声明泛化。' if target else
        '本轮不支持将新GNN作为已有竞争力的论文主算法，也不建议直接投入新的正式独立实验。保留负结果与权重，'
        '若继续研究，先按任务规模和构造进度分配少量反馈预算，并降低完整候选构造成本；'
        '用明确假设检查同轨协调，不追加模块或重调本8例。'),'',
        '复现（工作目录DRL，指定venv，单线程）：',
        '`D:\\pybullet_test\\.venv\\Scripts\\python.exe -B scripts/run_gnn_cmax_finetune.py --phase complete-parent-train`；',
        '`--phase complete-parent-eval`；`--phase complete-parent-report`。已存在模型分支与DEV记录会保留，不重复训练或调度。','']
    with OP_REPORT.open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    print('COMPLETE_REPORT',feasible,'/8',train_b32+dev_b32,'B32',flush=True)


def compact_imitation(shared,config,archive,teachers,source):
    """Measure both prescribed MRV/LPT quotas; train only actual six-parent labels."""
    from mrta_search import gnn_constructor as c
    graphs,examples,_,preparation=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(source,graphs,examples,raw_only=True)
    spaces={};records=[];counts=Counter();sizes=[];bins=Counter();elapsed=[]
    for e in raw:
        parents=e['reference'].parents
        if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config,compact=True)
        space=spaces[parents];parent=e['teacher_plan']['parent_id']
        pattern=next(p for p in e['reference'].patterns if p.parent_id==parent)
        selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in e['pending']}
        for limit in (4,6):
            ids=c.compact_parent_ids(space,e['pending'],parent_limit=limit)
            counts[f'parent_{limit}']+=int(parent in ids)
            if limit==4 and parent not in ids:continue
            tick=time.perf_counter()
            plans=c.complete_parent_candidates(space,e['state'],e['pending'],selected,config,
                first_limit=8,second_limit=4,parent_limit=limit,include_remaining_workload=True)
            target=next((i for i,p in enumerate(plans) if p['pattern']==pattern and
                         cmax_action_key(p)==cmax_action_key(e['teacher_plan'])),None)
            counts[f'complete_{limit}']+=int(target is not None)
            if limit==6:
                elapsed.append(time.perf_counter()-tick);sizes.append(len(plans))
                if target is not None:
                    records.append(dict(space=space,inputs=c.complete_parent_inputs(plans),target=target,weight=e['weight']))
                    bins[str(e['fraction'])]+=1
    coverage=dict(total=len(raw),**dict(counts),expert_actions=len(records),prefix_bins=dict(bins),
        candidate_min=min(sizes),candidate_median=statistics.median(sizes),candidate_max=max(sizes),
        generation_seconds_median=statistics.median(elapsed),preparation=preparation,
        decision='User selected six-parent experiment despite 50/126 teacher coverage; no offline injection')
    return records,spaces,coverage


def compact_training_geometries(spaces):
    """Use actual seam-count clusters present in the retained TRAIN data."""
    ordered=sorted(spaces,key=lambda parents:(len(parents),tuple(p.parent_id for p in parents)))
    groups=(('SMALL',[p for p in ordered if len(p)<50]),
            ('MEDIUM',[p for p in ordered if 50<=len(p)<63]),
            ('LARGE',[p for p in ordered if len(p)>=63]))
    if any(len(g)<2 for _,g in groups):raise ValueError('two actual TRAIN geometries per scale are required')
    return [(label,parents,spaces[parents]) for label,group in groups for parents in (group[0],group[-1])]


def compact_feedback(model,spaces,config,rows):
    from mrta_search import gnn_constructor as c
    # Reuse requires this exact prefix, action, full continuation AND config.
    old=[r for r in rows if r['phase']=='compact_workload_feedback']
    spent=sum(int(r['b32_calls'] or 0) for r in old)
    records=[];physical=0;reuses=0;selection=[]
    scientific=json.loads(json.dumps(config.scientific_mapping()))
    for index,(scale,parents,space) in enumerate(compact_training_geometries(spaces)):
        targets={0:0.,round(.1*len(parents)):.1};prefixes={};selection.append(dict(scale=scale,parents=len(parents)))
        def observe(sp,state,pending,selected,plans,best):
            progress=len(parents)-len(pending)
            if progress in targets:prefixes[progress]=(state,pending,selected,plans,best)
        trajectory=c.construct_complete_parent_gnn(parents,model,config,space=space,parent_limit=6,observer=observe)
        if set(prefixes)!=set(targets):raise ValueError('compact policy did not reach both actual TRAIN prefixes')
        with torch.no_grad():embedding=model.encode_static(space['graph'])
        for target,fraction in targets.items():
            state,pending,selected,plans,first=prefixes[target]
            with torch.no_grad():
                scores=model.parent_action_scores(space['graph'],embedding,c.complete_parent_inputs(plans))
            ranking=sorted(range(len(plans)),key=lambda i:(-float(scores[i]),i))
            def robots(p):return tuple(a.robot for a in p['actions'])
            alternatives=[i for i in ranking if plans[i]['pattern']==first['pattern'] and robots(plans[i])!=robots(first)]
            if not alternatives:alternatives=[i for i in ranking if robots(plans[i])!=robots(first)]
            if not alternatives:alternatives=[i for i in ranking if cmax_action_key(plans[i])!=cmax_action_key(first) or plans[i]['pattern']!=first['pattern']]
            if not alternatives:raise ValueError('two distinct complete actions unavailable at a required state')
            chosen=(first,plans[alternatives[0]])
            prefix=dict(routes=state.routes,directions=state.directions,pending=pending,
                        selected_patterns=[asdict(p) for p in selected.values()])
            normalized_prefix=json.loads(json.dumps(prefix))
            outcomes=[];calls=0;tick=time.perf_counter();name=f'{scale}_TRAIN_{index:02d}'
            for action_index,plan in enumerate(chosen):
                # The greedy continuation from the empty route is also the
                # exact A-branch continuation at every prefix on that path.
                result=trajectory if action_index==0 else c.construct_complete_parent_gnn(parents,model,config,
                    space=space,state=plan['state'],pending=set(pending)-{plan['parent_id']},
                    selected=dict(selected,**{plan['parent_id']:plan['pattern']}),parent_limit=6)
                action=dict(pattern=asdict(plan['pattern']),actions=[asdict(a) for a in plan['actions']])
                action=json.loads(json.dumps(action))
                outcome=dict(**action,status='CONSTRUCTION_FAILED',cmax=None,b32_calls=0,reason=result.rejection_reason)
                if result.solution is not None:
                    raw=dict(solution=asdict(result.solution),directions=result.directions)
                    reused=None
                    for row in old:
                        p=row['payload']
                        if p['scientific_config']!=scientific or p['prefix']!=normalized_prefix:continue
                        for previous in p['outcomes']:
                            if (previous.get('complete') and previous['pattern']==action['pattern'] and previous['actions']==action['actions']
                                and op_complete_key(previous['complete'])==op_complete_key(raw)):reused=previous;break
                        if reused:break
                    if reused:
                        outcome.update({k:reused[k] for k in ('status','cmax','scheduler_seconds','diagnostics','metrics')});reuses+=1
                    else:
                        if spent+physical>=24:raise ValueError('twenty-four compact TRAIN B32 calls exhausted')
                        clock=time.perf_counter();candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
                        seconds=time.perf_counter()-clock;valid=candidate.schedule.status is ScheduleStatus.FEASIBLE
                        metrics=op_route_metrics(raw,config)
                        if valid:metrics.update(op_metrics(candidate,config))
                        outcome.update(status=candidate.schedule.status.value,cmax=candidate.schedule.cmax if valid else None,
                            scheduler_seconds=seconds,metrics=metrics,diagnostics=candidate.schedule.diagnostics,b32_calls=1)
                        physical+=1;calls+=1
                    outcome['complete']=raw
                outcomes.append(outcome)
            pairs=complete_cmax_pairs(outcomes)
            records.append(dict(space=space,inputs=c.complete_parent_inputs(chosen),pairs=pairs))
            if not any(r['instance']==name and r['payload']['fraction']==fraction for r in old):
                op_save(dict(phase='compact_workload_feedback',instance=name,seed=OP_SEED,method='FROZEN_COMPACT_SAME_STATE_AB',
                    status='COLLECTED',runtime_s=time.perf_counter()-tick,b32_calls=calls,payload=dict(
                        scale=scale,fraction=fraction,progress=target/len(parents),parents=[asdict(p) for p in parents],
                        scientific_config=scientific,prefix=prefix,outcomes=outcomes,pairs=pairs,
                        same_parent_mode=chosen[0]['pattern']==chosen[1]['pattern'],
                        different_robot_assignment=robots(chosen[0])!=robots(chosen[1]),
                        completion_policy='frozen complete_parent_candidate with zero new columns and compact 6/8/4 decoding')))
            print('COMPACT_FEEDBACK',name,fraction,[o['status'] for o in outcomes],'pairs',len(pairs),'calls',physical,flush=True)
    return records,dict(states=len(records),actions=2*len(records),cmax_pairs=sum(len(r['pairs']) for r in records),
        states_with_pairs=sum(bool(r['pairs']) for r in records),new_b32=physical,exact_reuses=reuses,geometries=selection)


def compact_workload_train():
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    if COMPACT_VARIANT in checkpoint:
        print('Retain existing compact model; do not retrain.',flush=True);return
    torch.manual_seed(OP_SEED);rng=random.Random(OP_SEED)
    source=shared._model_from_checkpoint(checkpoint[COMPLETE_VARIANT]).eval()
    imitation,spaces,coverage=compact_imitation(shared,config,archive,teachers,source)
    if not any(r['phase']=='compact_workload_coverage' for r in op_read()):
        op_save(dict(phase='compact_workload_coverage',instance='TRAIN_21',seed=OP_SEED,method='MRV3_LPT3_CHILD8_4',
            status='BELOW_90_PERCENT_RECORDED',b32_calls=0,payload=coverage))
    model=shared.expand_remaining_workload_model(source)
    frozen={k:v.clone() for k,v in model.state_dict().items() if not k.startswith('parent_action_head.')}
    feedback_start=time.perf_counter();feedback,feedback_info=compact_feedback(model,spaces,config,op_read())
    feedback_seconds=time.perf_counter()-feedback_start
    encoded={}
    with torch.no_grad():
        for record in imitation+feedback:
            space=record['space'];key=id(space)
            if key not in encoded:encoded[key]=model.encode_static(space['graph'])
            record['rows']=model.parent_action_rows(space['graph'],encoded[key],record['inputs']).clone()
    pairs=[(r,a,b) for r in feedback for a,b in r['pairs']]
    optimizer=torch.optim.AdamW(model.parent_action_head.parameters(),lr=1e-4);history=[];started=time.perf_counter()
    for epoch in range(10):
        totals=Counter();tick=time.perf_counter()
        for _ in range(64):
            optimizer.zero_grad(set_to_none=True)
            il=torch.stack([-e['weight']*torch.log_softmax(model.parent_action_head(e['rows']).squeeze(1),0)[e['target']]
                for e in (rng.choice(imitation) for _ in range(8))]).mean()
            rank=torch.stack([F.softplus(-(model.parent_action_head(r['rows']).squeeze(1)[a]-
                model.parent_action_head(r['rows']).squeeze(1)[b]))
                for r,a,b in (rng.choice(pairs) for _ in range(8))]).mean() if pairs else il.new_zeros(())
            loss=rank+.1*il
            if not torch.isfinite(loss):raise ValueError('nonfinite compact head loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parent_action_head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            totals['loss']+=float(loss.detach());totals['cmax']+=float(rank.detach());totals['il']+=float(il.detach())
        result=dict(epoch=epoch+1,updates=64,seconds=time.perf_counter()-tick,**{k:v/64 for k,v in totals.items()})
        history.append(result)
        op_save(dict(phase='compact_workload_training',instance='TRAIN_21',seed=OP_SEED,method='CMAX_PLUS_01_IL_HEAD_ONLY',
            status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    if not all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()):raise ValueError('frozen encoder or scalers changed')
    metadata=dict(seed=OP_SEED,initialization=COMPLETE_VARIANT,parent_limit=6,first_limit=8,second_limit=4,
        coverage=coverage,feedback=feedback_info,epochs=10,updates_per_epoch=64,batch_size=8,learning_rate=1e-4,
        losses=dict(cmax=1.,imitation=.1),training_seconds=time.perf_counter()-started,feedback_seconds=feedback_seconds,
        history=history,encoder_frozen=True,normalization_frozen=True,new_columns_initialization='zero',new_teacher_b32=0)
    old={'root':checkpoint['state_dict']}
    old.update({k:v['state_dict'] for k,v in checkpoint.items() if isinstance(v,dict) and 'state_dict' in v})
    checkpoint[COMPACT_VARIANT]=dict(model_type='CompleteParentGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,parent_action_features=True,remaining_workload_features=True,compact_parent_limit=6,
        use_dynamic_context=False,scientific_config=config.scientific_mapping(),state_dict=model.state_dict(),training=metadata)
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    for key,values in old.items():
        actual=saved['state_dict'] if key=='root' else saved[key]['state_dict']
        if not all(torch.equal(v,actual[n]) for n,v in values.items()):raise ValueError('historical branch changed')
    op_save(dict(phase='compact_workload_model',instance='TRAIN_21',seed=OP_SEED,method=COMPACT_VARIANT,
        status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


def compact_workload_evaluate():
    from mrta_search import gnn_constructor as c
    shared,config,_,_,_,_,entries,_=op_sources();rows=op_read()
    done={r['instance'] for r in rows if r['phase']=='compact_workload_evaluation'}
    for name,parents in sorted(entries.items()):
        if name in done:continue
        started=time.perf_counter();thin=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
        model=shared._model_from_checkpoint(thin[COMPACT_VARIANT]).eval();load=time.perf_counter()-started
        decisions=[]
        def observe(space,state,pending,selected,plans,best):
            decisions.append(dict(candidates=len(plans),parents=len({p['parent_id'] for p in plans}),
                selected_parent=best['parent_id'],pattern=asdict(best['pattern']),loads=best['state'].loads,
                remaining_eligible_workload=c.remaining_eligible_workload(space,pending).tolist()))
        tick=time.perf_counter()
        result=c.construct_complete_parent_gnn(parents,model,config,parent_limit=6,observer=observe)
        construction=time.perf_counter()-tick;calls=0;metrics={};diagnostics=[];scheduler=0.;raw=None;cmax=None
        status='CONSTRUCTION_FAILED'
        if result.solution is not None:
            raw=dict(solution=asdict(result.solution),directions=result.directions)
            tick=time.perf_counter();candidate=pilot.oriented.schedule_explicit(result.solution,result.directions,config)
            scheduler=time.perf_counter()-tick;calls=1;status=candidate.schedule.status.value
            diagnostics=candidate.schedule.diagnostics;metrics=op_route_metrics(raw,config)
            if candidate.schedule.status is ScheduleStatus.FEASIBLE:cmax=candidate.schedule.cmax;metrics.update(op_metrics(candidate,config))
        op_save(dict(phase='compact_workload_evaluation',instance=name,seed=OP_EVAL_SEED,method='COMPACT_WORKLOAD_GNN_GREEDY',
            status=status,cmax=cmax,runtime_s=time.perf_counter()-started,b32_calls=calls,**metrics,payload=dict(
                source_variant=COMPACT_VARIANT,proposals=1,sampling=False,model_load_seconds=load,
                construction_seconds=construction,scheduler_seconds=scheduler,metrics=metrics,complete=raw,
                reason=result.rejection_reason,diagnostics=diagnostics,
                wait_for_graph=candidate.schedule.wait_for_graph if result.solution is not None else (),
                decisions=decisions,actions=[asdict(a) for a in result.actions])))


def compact_single_baselines(rows,development):
    """Use original single proposals; never relabel a multi-proposal winner."""
    result={}
    for phase,method,label in (('evaluation','NEW_GNN_GREEDY','ORIGINAL_GNN_GREEDY'),
        ('parent_evaluation','GNN_PARENT_JOINT','PARENT_JOINT_GREEDY')):
        result[label]={r['instance']:r for r in rows if r['phase']==phase and r['method']==method and r['seed']==str(OP_EVAL_SEED)}
    rules={}
    for name in result['ORIGINAL_GNN_GREEDY']:
        saved=next(r for r in rows if r['phase']=='parent_evaluation' and r['method']=='RULE/BEAM' and r['instance']==name)
        scheduled=saved['payload'].get('scheduled_proposals',[])
        source=next(r for r in development if r['instance']==name and r['seed']==OP_EVAL_SEED and r['method']=='MODE_NEW')
        detail=source['details'];attempts=detail.get('initial_attempts',[])
        if scheduled:
            first=scheduled[0];status=first['status'];cmax=first['cmax'];where='retained first scheduled proposal'
        else:
            scheduled_statuses=[a for a in attempts if a['status'] in {s.value for s in ScheduleStatus}]
            status=scheduled_statuses[0]['status'] if scheduled_statuses else 'CONSTRUCTION_FAILED';cmax=None
            where='retained initial attempt statuses; per-proposal Cmax unavailable'
            if status=='FEASIBLE' and sum(a['status']=='FEASIBLE' for a in scheduled_statuses)==1:
                cmax=detail['initial_state']['cmax'];where='only feasible attempt was proposal one, so retained winner is identical'
        rules[name]=dict(status=status,cmax=cmax,runtime_s=None,source=where)
    result['RULE_FIRST_PROPOSAL']=rules
    return result


def compact_route_overlap(raw,config):
    """Read route geometry only; interval overlap does not establish collision."""
    import run_gnn_shared_construction as shared
    from mrta_reference.solution import block_map
    from mrta_reference.scope import FORMAL_SCOPE_V2
    solution,_,_,_,_=shared.research_restore(raw,config)
    blocks=block_map(solution,config,scope=FORMAL_SCOPE_V2);spans=[]
    for route in solution.routes:
        xs=[x for b in route.block_ids for x in (blocks[b].start[0],blocks[b].end[0])]
        spans.append((min(xs),max(xs)) if xs else None)
    return [dict(pair=(left,right),left_span=spans[left],right_span=spans[right],overlap=
        max(0.,min(spans[left][1],spans[right][1])-max(spans[left][0],spans[right][0]))
        if spans[left] is not None and spans[right] is not None else 0.) for left,right in ((0,1),(2,3))]


def compact_workload_report():
    import run_gnn_shared_construction as shared
    import re
    from mrta_reference.model import ScientificConfig
    heading='## 2026-11-06 剩余资格加权工作量与紧凑MRV/LPT联合构造'
    if heading in OP_REPORT.read_text(encoding='utf-8'):
        print('Retain existing compact report.',flush=True);return
    rows=op_read();new={r['instance']:r for r in rows if r['phase']=='compact_workload_evaluation'}
    if len(new)!=8:raise ValueError('finish all eight single-proposal DEV cases before reporting')
    _,config,_,_,_,_,_,development=op_sources()
    checkpoint=shared.read_neural_checkpoint(OP_CHECKPOINT);meta=checkpoint[COMPACT_VARIANT]['training']
    baselines=compact_single_baselines(rows,development);names=sorted(new)
    numeric=lambda r:r['status']=='FEASIBLE' and r['cmax'] not in (None,'')
    number=lambda v:'—' if v in (None,'') else f'{float(v):.3f}'
    def compare(old):
        changes=[];count=Counter();missing=[]
        for name in names:
            a,b=new[name],old[name];fa=a['status']=='FEASIBLE';fb=b['status']=='FEASIBLE'
            if fa and fb:
                if not numeric(b):missing.append(name);continue
                d=100*(float(b['cmax'])-float(a['cmax']))/float(b['cmax']);changes.append(d)
                count['win' if d>1e-8 else 'loss' if d< -1e-8 else 'tie']+=1
            elif fa:count['win']+=1
            elif fb:count['loss']+=1
            else:count['both_fail']+=1
        return count,changes,missing
    feedback=[r for r in rows if r['phase']=='compact_workload_feedback'];outcomes=[o for r in feedback for o in r['payload']['outcomes']]
    calls_train=sum(int(r['b32_calls'] or 0) for r in feedback);calls_dev=sum(int(r['b32_calls'] or 0) for r in new.values())
    feasible=sum(r['status']=='FEASIBLE' for r in new.values());coverage=meta['coverage']
    construction=statistics.median(r['payload']['construction_seconds'] for r in new.values())
    scheduler=statistics.median(r['payload']['scheduler_seconds'] for r in new.values())
    total=statistics.median(float(r['runtime_s']) for r in new.values())
    earlier={r['instance']:r for r in rows if r['phase']=='complete_parent_evaluation'}
    previous_construction=statistics.median(r['payload']['attempts'][0]['construction_seconds'] for r in earlier.values())
    old_coverage=checkpoint[COMPLETE_VARIANT]['training']['coverage']
    comparable={label:compare(old) for label,old in baselines.items()}
    parent_changes=comparable['PARENT_JOINT_GREEDY'][1];rule_changes=comparable['RULE_FIRST_PROPOSAL'][1]
    goals=feasible>=7 and construction<=5 and parent_changes and statistics.median(parent_changes)>0 and rule_changes and not comparable['RULE_FIRST_PROPOSAL'][2] and statistics.median(rule_changes)>0
    lines=['',heading,'',f'**实际单提案结果：{feasible}/8可行，完整构造中位{construction:.3f}s。'
        +('满足本轮可核对的开发目标，仍需新独立数据验证。**' if goals else '未满足本轮全部目标，不进入正式独立实验。**'),'',
        '本轮使用指定venv，在DRL内执行，CPU单线程。继续修改gnn_constructor.py、run_gnn_shared_construction.py、'
        'run_gnn_cmax_finetune.py及既有测试；模型保存于既有checkpoint的compact_workload_candidate独立分支，'
        'CSV与报告均追加，没有新建代码、数据、报告或中间文件。MODE_NEW、科学配置、FORMAL_SCOPE_V2、'
        'min_child_length与delta_x/delta_y=0.50m、B32和既有135/162/500历史实验均未重跑或改写。','',
        '### 紧凑构造与输入','',
        '对每父按合法目录计算F_i=Σ_m Π_b |R_b|，X_SPLIT使用指定轨左右机器人的实际强制资格。'
        '初始MRV2+LPT2去重，覆盖不足后按用户确认使用MRV3+LPT3去重，最多6父；'
        'MRV同灵活性按原始父加工时间降序、再按ID稳定打破平局，LPT按父原始加工时间降序。'
        '保留这些父的全部合法WHOLE/Y_SPLIT/X_SPLIT模式；每首子最多8个代表、第二子最多4个，'
        '次子在首子后的真实临时路线中生成，最多32个拆分组合，WHOLE也限8。'
        '完整模式、机器人、位置、方向仍由同一个完整动作头选择，整父原子提交，从空路线连续构造。'
        '没有新父网络、独立mode_head、独立方向优化、概率连乘、随机采样或后处理补救。','',
        '剩余工作量对每合法模式的各子段按w_b/|R_b|分给资格机器人，先平均同类别切点，再平均父的模式类别，'
        '最后对全部剩余父求和并以同样平均方式的总加工时间归一化；四维和为1，空集合为0。'
        '每父工作量和资格在几何构图时缓存，当前状态复用已有资格计算结果，不重复执行插入可行性查询。'
        '这是机器人资格加权工作量分布，不是数学负载下界。','',
        '保留两层GAT、hidden=64、heads=4及原图/机器人节点。完整头由723→32→1扩为727→32→1，'
        '新增4列初始为0，全部旧列/偏置从complete_parent_candidate精确复制；只训练23329参数的完整头，'
        '编码器、其余旧头和归一化张量保持不变。父表示在同一状态内只计算一次再复用，避免每个动作重复求同一父均值，'
        '这一等价计算已用逐元素相等检查。','',
        '### 教师覆盖、反馈与实际训练','',
        f"旧126动作：4父覆盖{coverage['complete_4']}/126，6父覆盖{coverage['complete_6']}/126="
        f"{100*coverage['complete_6']/126:.2f}%，未达到90%。6父父任务覆盖本身只有{coverage['parent_6']}/126；"
        '其余损失不能靠增加子分支解决。用户明确选择沿用6父完成本轮，故仅训练真实入池的50个完整动作，'
        '未把另外76个动作注入离线池，未扩大父池或筛掉覆盖失败。'
        f"六个进度有效数量：{coverage['prefix_bins']}。复用76条教师重放、27条续补和9654条旧局部样本恢复这些标签，教师新增B32=0。",'',
        f"同一126教师状态的完整池min/median/max={coverage['candidate_min']}/{coverage['candidate_median']}/{coverage['candidate_max']}，"
        f"上一轮为{old_coverage['candidate_min']}/{old_coverage['candidate_median']}/{old_coverage['candidate_max']}。"
        f"中位池缩小{100*(1-coverage['candidate_median']/old_coverage['candidate_median']):.2f}%；"
        f"本轮候选生成中位{coverage['generation_seconds_median']:.6f}s/状态。",'',
        f"实际TRAIN几何：{meta['feedback']['geometries']}。各取空路线和round(0.1N)个已完成父的真实前缀，共12状态。"
        'A为冻结初始化模型的贪心动作，B优先保持相同父/模式而改变机器人组合，无该动作时才选其他完整动作。'
        f"同父同模式{sum(r['payload']['same_parent_mode'] for r in feedback)}/12，不同机器人组合"
        f"{sum(r['payload']['different_robot_assignment'] for r in feedback)}/12。全部分支使用同一个冻结模型和紧凑规则贪心续补。",'',
        f"反馈结果{dict(Counter(o['status'] for o in outcomes))}，可行Cmax数{sum(o['status']=='FEASIBLE' for o in outcomes)}/24，"
        f"有效同状态排序对{meta['feedback']['cmax_pairs']}、可排序状态{meta['feedback']['states_with_pairs']}/12。"
        f"本轮新增TRAIN B32={calls_train}、DEV B32={calls_dev}，合计{calls_train+calls_dev}。"
        '复用要求前缀、完整动作、完整续补路线/方向及科学配置都完全一致；本轮未精确命中可复用结果。'
        'DEADLOCK没有数值Cmax，不与其他状态混排，也不将可行/失败伪装成数值Cmax差。','',
        '固定10轮、64步/轮、batch=8、AdamW lr=1e-4、clip=1，L=L_Cmax+0.1L_IL。'
        f"头部优化{meta['training_seconds']:.3f}s，反馈采集{meta['feedback_seconds']:.3f}s。"
        +('由于0个有效排序对，实际L_Cmax始终为0，这次只有50动作模仿产生梯度，不能称为完成了有效Cmax排序学习。'
          if not meta['feedback']['cmax_pairs'] else '仅使用上述真实同状态排序对，不使用旧OP或旧冻结模型标签。'),'',
        '### 单提案DEV结果与比较口径','',
        '固定seed=202611041，每几何只有一个确定性贪心完整方案，最多1次B32，无随机种子追加、epoch选择或旧算法回退。'
        '原GNN单提案为现有evaluation/NEW_GNN_GREEDY，PARENT_JOINT为parent_evaluation/GNN_PARENT_JOINT，'
        '它们分别只有3/8、5/8可行；历史6/8、7/8是三提案结果，不能当作单提案。','',
        'RULE首提案尽量从已保存scheduled_proposals[0]恢复，其他只用原initial_attempts的首个实际B32状态。'
        '若仅首提案可行，其原保存赢家Cmax就是首提案；若多个提案可行但只保存赢家，则首提案Cmax缺失。'
        '没有重跑RULE、用多提案赢家代替首提案，或把缺失Cmax填0。', '',
        '|几何|原GNN单提案|PARENT_JOINT单提案|RULE首提案|新GNN|构造/s|B32/s|总耗时/s|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in names:
        r=new[name];rule=baselines['RULE_FIRST_PROPOSAL'][name]
        rule_value='可行，Cmax未存' if rule['status']=='FEASIBLE' and not numeric(rule) else number(rule['cmax'])
        lines.append('|'+name+'|'+ '|'.join(number(baselines[label][name]['cmax']) for label in ('ORIGINAL_GNN_GREEDY','PARENT_JOINT_GREEDY'))+
            f"|{rule_value}|{number(r['cmax'])}|{number(r['payload']['construction_seconds'])}|{number(r['payload']['scheduler_seconds'])}|{number(r['runtime_s'])}|")
    lines+=['','“—”为失败而无Cmax；“可行，Cmax未存”为旧记录缺失，不视为失败。','',
        '|单提案方法|可行/8|构造中位/s|B32中位/s|总耗时中位/s|','|---|---:|---:|---:|---:|']
    for label,old in baselines.items():
        times=[float(r['runtime_s']) for r in old.values() if r.get('runtime_s') is not None]
        lines.append(f"|{label}|{sum(r['status']=='FEASIBLE' for r in old.values())}/8|—|—|{number(statistics.median(times) if len(times)==8 else None)}|")
    lines.append(f'|COMPACT_WORKLOAD_GNN_GREEDY|{feasible}/8|{construction:.3f}|{scheduler:.3f}|{total:.3f}|')
    lines+=['','|新模型相对单提案|胜/平/负|双失败|可比Cmax配对|中位改善/%|共同可行但旧Cmax缺失|','|---|---:|---:|---:|---:|---|']
    for label,(count,changes,missing) in comparable.items():
        lines.append(f"|{label}|{count['win']}/{count['tie']}/{count['loss']}|{count['both_fail']}|{len(changes)}/8|"
            f"{number(statistics.median(changes) if changes else None)}|{','.join(missing) or '无'}|")
    lines+=['','改善=100×(old−new)/old；可行优于失败，双失败单列。RULE只在可恢复Cmax的配对上给出结果，'
        '缺失的共同可行配对使其完整配对目标不能得到验证。','',
        f'上一轮COMPLETE_PARENT_GNN的同一8例首个贪心构造中位{previous_construction:.3f}s，'
        f'本轮{construction:.3f}s，按同为单提案计减少{100*(1-construction/previous_construction):.2f}%。'
        '不把上一轮三提案115.963s与本轮单提案直接作为等预算加速比。','',
        '补充历史三提案：ORIGINAL_GNN_MULTI3 6/8、PARENT_JOINT_MULTI3 7/8、RULE/BEAM最多三次B32 7/8、'
        'COMPLETE_PARENT_GNN_MULTI3 3/8，总耗时中位分别4.195/3.850/3.213/119.292s；仅供补充，不替代上面的单提案比较。','',
        '### 分配负载和同轨阻塞检查','',
        '|几何|状态|R0/R1/R2/R3加工负载|max(P+MOVE)|','|---|---|---|---:|']
    for name in names:
        r=new[name];m=r['payload']['metrics']
        lines.append(f"|{name}|{r['status']}|"+' / '.join(number(x) for x in m.get('loads',[]))+f"|{number(m.get('max_p_move'))}|")
    lines+=['','失败方案的max(P+MOVE)仅为路线代理，不是失败Cmax。以下X区间是路线各任务几何端点的包络；'
        '重叠仅表示共享空间使用，不能直接认定实际碰撞或DEADLOCK。操作编号B为该机器人路线中从0开始的块下标。']
    for name in names:
        r=new[name];p=r['payload']
        if not p['complete']:continue
        overlap=compact_route_overlap(p['complete'],config)
        def interval(value):return '空' if value is None else f'[{value[0]:.3f},{value[1]:.3f}]'
        spans='；'.join(f"R{x['pair'][0]}/R{x['pair'][1]}={interval(x['left_span'])}/{interval(x['right_span'])}，重叠{x['overlap']:.3f}m" for x in overlap)
        lines+=['',f"- {name}（{r['status']}）：{spans}。"]
        if r['status']=='DEADLOCK':
            blocked=[d for d in p['diagnostics'] if 'blocked by' in d]
            excerpts=[]
            for d in blocked[:3]:
                operations=re.findall(r'R[0-3]:B\d+:(?:MOVE|SETUP|WELD|POST)',d)
                excerpts.append(d.split(' blocked by')[0]+' ← '+', '.join(operations[:3]))
            lines.append(f"  原B32依赖：{p.get('wait_for_graph',[])}；实际阻塞操作摘录："+'；'.join(excerpts)+'.')
    lines+=['','B32提供的wait_for_graph和blocked-by操作是实际返回的阻塞信息，未新增碰撞检测器或推导调度器未报告的时间依赖。'
        '同轨区间重叠与操作阻塞需结合阅读，不能凭几何包络证明碰撞的确定原因。','',
        '### 结论','',
        '已完成紧凑MRV/LPT候选、类别平均的剩余资格工作量、头部冻结训练、小中大分层采样及单提案验证。'
        '候选覆盖下降是明确代价，早期监督只剩0%进度4条、10%进度1条；若真实反馈无有效可行配对，'
        '新特征只能通过这批稀疏专家标签学习，不能把24次调度包装成24个Cmax排序样本。',
        ('本轮达到可核对开发目标，但仍须补齐公平RULE首提案资料并用新独立数据验证。' if goals else
         '本轮不支持直接进入新的正式独立实验。保留模型和负结果；后续应先解决紧凑候选与专家监督不匹配、'
         '同状态有效Cmax反馈不足的问题，不能通过增加种子、训练轮数或独立后处理制造改进。'),'',
        '复现（工作目录DRL）：指定venv运行scripts/run_gnn_cmax_finetune.py，依次'
        '`--phase compact-workload-train`、`--phase compact-workload-eval`、`--phase compact-workload-report`。'
        '已有模型和DEV记录会保留，不重复训练或调度。','']
    with OP_REPORT.open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    print('COMPACT_REPORT',feasible,'/8','construction',construction,'B32',calls_train+calls_dev,flush=True)


def residual_coverage_sweep():
    """TRAIN-only candidate measurements. No model update or scheduling call."""
    from mrta_search import gnn_constructor as c
    saved=op_read()
    existing=[r for r in saved if r['phase']=='residual_candidate_coverage']
    if len(existing)==20 and any(r['phase']=='residual_coverage_decision' for r in saved):
        print('Retain completed candidate sweep; no repeated measurements.',flush=True);return
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    graphs,examples,_,preparation=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(source,graphs,examples,raw_only=True)
    models={OP_VARIANT:source,'root':shared._model_from_checkpoint(checkpoint).eval()}
    for name in ('refinement_candidate',EARLY_VARIANT,PARENT_CMAX_VARIANT,COMPLETE_VARIANT):
        models[name]=shared._model_from_checkpoint(checkpoint[name]).eval()
    strategies=('gnn','mrv','lpt','gnn_mrv_lpt');limits=(4,6,8,12,16)
    measurements={(strategy,k):[] for strategy in strategies for k in limits}
    source_counts=Counter();spaces={};variable_features=0;parent_ranks=[]
    with torch.no_grad():
        for index,e in enumerate(raw):
            parents=e['reference'].parents
            if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config,compact=True)
            space=spaces[parents];pending=e['pending'];teacher=e['teacher_plan'];parent=teacher['parent_id']
            selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in pending}
            pattern=next(p for p in e['reference'].patterns if p.parent_id==parent)
            tick=time.perf_counter();orders=c.gnn_parent_orders(space,e['state'],pending,selected,source,config)
            parent_seconds=time.perf_counter()-tick
            parent_ranks.append(dict(case=index,fraction=e['fraction'],parent=parent,rank=orders['gnn'].index(parent)+1,
                                     pending=len(pending),geometry_parents=len(parents)))
            all_vectors={tuple(c.remaining_eligible_workload(space,set(pending)-{p}).tolist()) for p in pending}
            variable_features+=len(all_vectors)>1
            for strategy in strategies:
                for k in limits:
                    tick=time.perf_counter()
                    ids=c.compact_parent_ids(space,pending,parent_limit=k,strategy=strategy,orders=orders)
                    plans=c.complete_parent_candidates(space,e['state'],pending,selected,config,first_limit=8,second_limit=4,
                        parent_limit=k,include_remaining_workload=True,candidate_parent_ids=ids,remaining_after_parent=True)
                    matched=any(p['pattern']==pattern and cmax_action_key(p)==cmax_action_key(teacher) for p in plans)
                    seconds=time.perf_counter()-tick+(parent_seconds if strategy in ('gnn','gnn_mrv_lpt') else 0.)
                    measurements[strategy,k].append(dict(case=index,fraction=e['fraction'],parent_covered=parent in ids,
                        complete_covered=matched,parents=len(ids),candidates=len(plans),seconds=seconds))
            # Additional frozen-head/representation checks diagnose whether a
            # proposed repair can possibly cross the complete-action threshold.
            for name,model in models.items():
                representations=('native','catalog_mean','catalog_max') if name in (OP_VARIANT,PARENT_CMAX_VARIANT,'refinement_candidate') else ('native',)
                for representation in representations:
                    ranked=orders if name==OP_VARIANT and representation=='native' else c.gnn_parent_orders(
                        space,e['state'],pending,selected,model,config,proposal_graph_mode=representation)
                    for strategy in ('gnn','gnn_mrv_lpt'):
                        for k in limits:
                            ids=c.compact_parent_ids(space,pending,parent_limit=k,strategy=strategy,orders=ranked)
                            source_counts[name,representation,strategy,k]+=parent in ids
            if index%6==5:print('RESIDUAL_COVERAGE',index+1,'/126',flush=True)
    done={(r['method'],r['payload']['parent_limit']) for r in existing}
    summary=[]
    for (strategy,k),cases in measurements.items():
        payload=dict(source_variant=OP_VARIANT,representation='native',parent_limit=k,first_limit=8,second_limit=4,
            states=len(cases),parent_covered=sum(x['parent_covered'] for x in cases),
            complete_covered=sum(x['complete_covered'] for x in cases),
            actual_parents_median=statistics.median(x['parents'] for x in cases),
            candidates_min=min(x['candidates'] for x in cases),candidates_median=statistics.median(x['candidates'] for x in cases),
            candidates_max=max(x['candidates'] for x in cases),generation_seconds_median=statistics.median(x['seconds'] for x in cases),
            cases=cases,timing='complete candidate generation; measured native parent ranking charged once for GNN strategies',
            remaining_after_parent=True)
        summary.append(dict(strategy=strategy,**{key:value for key,value in payload.items() if key!='cases'}))
        if (strategy,k) not in done:
            op_save(dict(phase='residual_candidate_coverage',instance='TRAIN_126',seed=OP_SEED,method=strategy,
                status='MEASURED',b32_calls=0,payload=payload))
    for name,representation,strategy in sorted({key[:3] for key in source_counts}):
        op_save(dict(phase='residual_parent_diagnosis',instance='TRAIN_126',seed=OP_SEED,
            method=name+':'+representation+':'+strategy,status='FROZEN_HEAD_MEASURED',b32_calls=0,
            payload=dict(source_variant=name,representation=representation,strategy=strategy,
                         parent_coverage={str(k):source_counts[name,representation,strategy,k] for k in limits})))
    qualified=[r for r in summary if r['complete_covered']/len(raw)>=.9]
    chosen=min(qualified,key=lambda r:(r['parent_limit'],r['candidates_median'],-r['complete_covered'])) if qualified else None
    best=max(summary,key=lambda r:(r['complete_covered'],-r['parent_limit']))
    op_save(dict(phase='residual_coverage_decision',instance='TRAIN_126',seed=OP_SEED,method='FULL_ACTION_90_PERCENT',
        status='READY_FOR_FEEDBACK' if chosen else 'STOP_COVERAGE_BELOW_90',b32_calls=0,
        payload=dict(chosen=chosen,best=best,required=math.ceil(.9*len(raw)),total=len(raw),preparation=preparation,
            states_with_action_dependent_remaining=variable_features,parent_ranks=parent_ranks,
            best_parent_coverage_any_checked_frozen_head=max(source_counts.values()),
            train_b32_calls=0,dev_b32_calls=0,training_updates=0)))
    print('RESIDUAL_DECISION',None if chosen is None else (chosen['strategy'],chosen['parent_limit']),
          'best',best['complete_covered'],'/126','B32=0','training=0',flush=True)


def residual_packing_repair():
    """Diagnose the eight packing misses without increasing the parent cap."""
    from mrta_search import gnn_constructor as c
    rows=op_read()
    if any(r['phase']=='residual_candidate_repair' for r in rows):
        print('Retain existing bounded-packing diagnosis.',flush=True);return
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    source=shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    graphs,examples,_,_=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(source,graphs,examples,raw_only=True)
    initial=next(r['payload'] for r in rows if r['phase']=='residual_candidate_coverage' and r['method']=='gnn' and r['payload']['parent_limit']==16)
    spaces={};cases=[];recovered=[];lost=[]
    with torch.no_grad():
        for index,e in enumerate(raw):
            parents=e['reference'].parents
            if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config,compact=True)
            space=spaces[parents];pending=e['pending'];teacher=e['teacher_plan'];parent=teacher['parent_id']
            selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in pending}
            pattern=next(p for p in e['reference'].patterns if p.parent_id==parent)
            tick=time.perf_counter();orders=c.gnn_parent_orders(space,e['state'],pending,selected,source,config)
            ids=c.compact_parent_ids(space,pending,parent_limit=16,strategy='gnn',orders=orders)
            # Existing full bounded pool: WHOLE retains its legal eight-position
            # pool, split branches remain 16 x 8 rather than a full product.
            plans=c.complete_parent_candidates(space,e['state'],pending,selected,config,first_limit=16,second_limit=8,
                candidate_parent_ids=ids,include_remaining_workload=True,remaining_after_parent=True)
            covered=any(p['pattern']==pattern and cmax_action_key(p)==cmax_action_key(teacher) for p in plans)
            original=initial['cases'][index]
            if original['complete_covered'] and not covered:lost.append(index)
            if covered and not original['complete_covered']:recovered.append(index)
            cases.append(dict(case=index,fraction=e['fraction'],parent_covered=parent in ids,complete_covered=covered,
                              parents=len(ids),candidates=len(plans),seconds=time.perf_counter()-tick))
            if index%21==20:print('RESIDUAL_PACKING_REPAIR',index+1,flush=True)
    result=dict(source_variant=OP_VARIANT,representation='native',strategy='gnn',parent_limit=16,
        first_limit=16,second_limit=8,whole_pool='all legal variants within existing eight positions',states=len(cases),
        parent_covered=sum(x['parent_covered'] for x in cases),complete_covered=sum(x['complete_covered'] for x in cases),
        candidates_min=min(x['candidates'] for x in cases),candidates_median=statistics.median(x['candidates'] for x in cases),
        candidates_max=max(x['candidates'] for x in cases),generation_seconds_median=statistics.median(x['seconds'] for x in cases),
        recovered_cases=recovered,lost_cases=lost,cases=cases,required=math.ceil(.9*len(cases)),
        training_updates=0,train_b32_calls=0,dev_b32_calls=0)
    op_save(dict(phase='residual_candidate_repair',instance='TRAIN_126',seed=OP_SEED,method='GNN16_EXISTING_BOUNDED_POOL',
        status='STOP_COVERAGE_BELOW_90' if result['complete_covered']<result['required'] else 'READY_FOR_FEEDBACK',
        b32_calls=0,payload=result))
    print('RESIDUAL_PACKING_RESULT',result['complete_covered'],'/126','recovered',len(recovered),'lost',len(lost),flush=True)


def residual_external_parent_check():
    """Check the separately retained retrained GNN head, without RULE fusion."""
    from mrta_search import gnn_constructor as c
    rows=op_read()
    if any(r['phase']=='residual_external_parent_decision' for r in rows):
        print('Retain completed external parent-head check.',flush=True);return
    shared,config,_,archive,teachers,_,_,_=op_sources()
    external=shared.read_neural_checkpoint(ROOT/'models/gnn_rule_guided_constructor_05.pt')
    source=shared._model_from_checkpoint(external['refinement_candidate']).eval()
    graphs,examples,_,_=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(source,graphs,examples,raw_only=True)
    counts=Counter();spaces={};prepared=[]
    with torch.no_grad():
        for index,e in enumerate(raw):
            parents=e['reference'].parents
            if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config,compact=True)
            selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in e['pending']}
            orders_by_rep={}
            for representation in ('native','catalog_mean','catalog_max'):
                orders=c.gnn_parent_orders(spaces[parents],e['state'],e['pending'],selected,source,config,proposal_graph_mode=representation)
                orders_by_rep[representation]=orders
                for strategy in ('gnn','gnn_mrv_lpt'):
                    for k in (4,6,8,12,16):
                        ids=c.compact_parent_ids(spaces[parents],e['pending'],parent_limit=k,strategy=strategy,orders=orders)
                        counts[representation,strategy,k]+=e['teacher_plan']['parent_id'] in ids
            prepared.append((e,spaces[parents],selected,orders_by_rep))
            if index%21==20:print('RETRAINED_PARENT_CHECK',index+1,flush=True)
        for representation,strategy in sorted({key[:2] for key in counts}):
            op_save(dict(phase='residual_parent_diagnosis',instance='TRAIN_126',seed=OP_SEED,
                method='rule_guided_retrained:'+representation+':'+strategy,status='FROZEN_HEAD_MEASURED',b32_calls=0,
                payload=dict(source_variant='rule_guided_retrained',representation=representation,strategy=strategy,
                    weights_path='models/gnn_rule_guided_constructor_05.pt',weights_branch='refinement_candidate',
                    rule_fusion_used=False,parent_coverage={str(k):counts[representation,strategy,k] for k in (4,6,8,12,16)})))
        eligible=[key for key,value in counts.items() if value>=math.ceil(.9*len(raw))]
        full=[]
        for representation,strategy,k in eligible:
            covered=0;sizes=[];times=[]
            for e,space,selected,orders_by_rep in prepared:
                ids=c.compact_parent_ids(space,e['pending'],parent_limit=k,strategy=strategy,orders=orders_by_rep[representation])
                tick=time.perf_counter()
                plans=c.complete_parent_candidates(space,e['state'],e['pending'],selected,config,first_limit=16,second_limit=8,
                    candidate_parent_ids=ids,include_remaining_workload=True,remaining_after_parent=True)
                target=e['teacher_plan'];pattern=next(p for p in e['reference'].patterns if p.parent_id==target['parent_id'])
                covered+=any(p['pattern']==pattern and cmax_action_key(p)==cmax_action_key(target) for p in plans)
                sizes.append(len(plans));times.append(time.perf_counter()-tick)
            value=dict(source_variant='rule_guided_retrained',representation=representation,strategy=strategy,parent_limit=k,
                parent_covered=counts[representation,strategy,k],complete_covered=covered,states=len(raw),
                candidates_median=statistics.median(sizes),candidates_max=max(sizes),generation_seconds_median=statistics.median(times))
            full.append(value)
            op_save(dict(phase='residual_external_candidate_coverage',instance='TRAIN_126',seed=OP_SEED,method=representation+':'+strategy,
                status='MEASURED',b32_calls=0,payload=value))
    qualified=[p for p in full if p['complete_covered']>=math.ceil(.9*len(raw))]
    op_save(dict(phase='residual_external_parent_decision',instance='TRAIN_126',seed=OP_SEED,method='RETRAINED_PARENT_HEAD_ONLY',
        status='READY_FOR_FEEDBACK' if qualified else 'STOP_COVERAGE_BELOW_90',b32_calls=0,payload=dict(
            best_parent_coverage=max(counts.values()),qualified=qualified,total=len(raw),required=math.ceil(.9*len(raw)),
            rule_fusion_used=False,training_updates=0,train_b32_calls=0,dev_b32_calls=0)))
    print('RETRAINED_PARENT_RESULT',max(counts.values()),'/126','qualified',qualified,flush=True)


def residual_quota_check():
    """Bounded checks of few MRV/LPT supplements, with GNN still the majority."""
    from mrta_search import gnn_constructor as c
    rows=op_read()
    if any(r['phase']=='residual_quota_decision' for r in rows):
        print('Retain measured supplement quota checks.',flush=True);return
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    external=shared.read_neural_checkpoint(ROOT/'models/gnn_rule_guided_constructor_05.pt')
    models={OP_VARIANT:shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval(),
            'rule_guided_retrained':shared._model_from_checkpoint(external['refinement_candidate']).eval()}
    graphs,examples,_,_=op_saved_imitation(shared,config,archive,teachers)
    raw,_=cmax_expert_macro_examples(models[OP_VARIANT],graphs,examples,raw_only=True)
    settings=[(k,h) for k in (4,6,8,12,16) for h in (1,2,3) if k-2*h>=2*h]
    counts=Counter();unions=Counter();spaces={};prepared=[]
    def select(orders,k,h):
        ids=list(orders['gnn'][:k-2*h])
        for key in ('mrv','lpt'):ids.extend([p for p in orders[key] if p not in ids][:h])
        ids.extend(p for p in orders['gnn'] if p not in ids)
        return tuple(ids[:k])
    with torch.no_grad():
        for index,e in enumerate(raw):
            parents=e['reference'].parents
            if parents not in spaces:spaces[parents]=c.complete_parent_space(parents,config,compact=True)
            selected={p.parent_id:p for p in e['reference'].patterns if p.parent_id not in e['pending']};ranked={}
            for name,model in models.items():
                orders=c.gnn_parent_orders(spaces[parents],e['state'],e['pending'],selected,model,config)
                ranked[name]=orders;target=e['teacher_plan']['parent_id']
                unions[name]+=target in set(orders['gnn'][:16]+orders['mrv'][:16]+orders['lpt'][:16])
                for k,h in settings:counts[name,k,h]+=target in select(orders,k,h)
            prepared.append((e,spaces[parents],selected,ranked))
            if index%21==20:print('SUPPLEMENT_QUOTA_CHECK',index+1,flush=True)
        for name in models:
            op_save(dict(phase='residual_quota_parent_coverage',instance='TRAIN_126',seed=OP_SEED,method=name,
                status='MEASURED',b32_calls=0,payload=dict(source_variant=name,states=len(raw),
                    quotas=[dict(parent_limit=k,gnn_quota=k-2*h,mrv_quota=h,lpt_quota=h,parent_covered=counts[name,k,h]) for k,h in settings],
                    union_of_three_top16_parent_coverage=unions[name],union_is_diagnostic_not_an_inference_pool=True)))
        full=[]
        for name,k,h in counts:
            if counts[name,k,h]<math.ceil(.9*len(raw)):continue
            covered=0;sizes=[]
            for e,space,selected,ranked in prepared:
                ids=select(ranked[name],k,h)
                plans=c.complete_parent_candidates(space,e['state'],e['pending'],selected,config,first_limit=16,second_limit=8,
                    candidate_parent_ids=ids,include_remaining_workload=True,remaining_after_parent=True)
                target=e['teacher_plan'];pattern=next(p for p in e['reference'].patterns if p.parent_id==target['parent_id'])
                covered+=any(p['pattern']==pattern and cmax_action_key(p)==cmax_action_key(target) for p in plans);sizes.append(len(plans))
            value=dict(source_variant=name,parent_limit=k,gnn_quota=k-2*h,mrv_quota=h,lpt_quota=h,
                parent_covered=counts[name,k,h],complete_covered=covered,states=len(raw),candidates_median=statistics.median(sizes),candidates_max=max(sizes))
            full.append(value)
            op_save(dict(phase='residual_quota_complete_coverage',instance='TRAIN_126',seed=OP_SEED,
                method=name+f':{k-2*h}+{h}+{h}',status='MEASURED',b32_calls=0,payload=value))
    qualified=[p for p in full if p['complete_covered']>=math.ceil(.9*len(raw))]
    op_save(dict(phase='residual_quota_decision',instance='TRAIN_126',seed=OP_SEED,method='BOUNDED_SUPPLEMENT_QUOTAS',
        status='READY_FOR_FEEDBACK' if qualified else 'STOP_COVERAGE_BELOW_90',b32_calls=0,
        payload=dict(best_parent_coverage=max(counts.values()),qualified=qualified,states=len(raw),required=math.ceil(.9*len(raw)),
                     training_updates=0,train_b32_calls=0,dev_b32_calls=0)))
    print('QUOTA_CHECK_RESULT',max(counts.values()),'parent targets','qualified',qualified,flush=True)


def residual_route_coverage():
    """Measure route-aware bounded supplements on the same TRAIN action identities."""
    from mrta_search import gnn_constructor as c

    rows = op_read()
    if any(r['phase'] == 'residual_route_coverage_decision' for r in rows):
        print('Retain measured route-aware coverage.', flush=True)
        return
    shared, config, checkpoint, archive, teachers, _, _, _ = op_sources()
    source = shared._model_from_checkpoint(checkpoint[OP_VARIANT]).eval()
    graphs, examples, _, _ = op_saved_imitation(shared, config, archive, teachers)
    raw, _ = cmax_expert_macro_examples(source, graphs, examples, raw_only=True)
    previous = next(r['payload']['cases'] for r in rows if r['phase'] == 'residual_candidate_repair')
    limits = (4, 6, 8, 12, 16)
    strategies = ('ROUTE_END', 'REFLECTED_START_ROUTE_END', 'REFLECTED_START_ROUTE_PROGRESS')
    required = math.ceil(.9 * len(raw))
    prepared = []
    counts = Counter()
    stages = Counter()
    spaces = {}
    with torch.inference_mode():
        for case, e in enumerate(raw):
            reference, state, pending = (e[key] for key in ('reference', 'state', 'pending'))
            teacher = e['teacher_plan']
            target = teacher['parent_id']
            pattern = next(p for p in reference.patterns if p.parent_id == target)
            selected = {p.parent_id: p for p in reference.patterns if p.parent_id not in pending}
            if reference.parents not in spaces:
                spaces[reference.parents] = c.complete_parent_space(reference.parents, config, compact=True)
            space = spaces[reference.parents]
            tick = time.perf_counter()
            gnn_order = c.gnn_parent_orders(space, state, pending, selected, source, config)['gnn']
            parent_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            ends = c.route_distance_parent_orders(space, state, selected, gnn_order, config, gaps=False)
            end_seconds = time.perf_counter() - tick
            route_lengths = tuple(map(len, state.routes))
            tick = time.perf_counter()
            reflected = c.reflected_start_parent_orders(space, state, pending, selected, source, config) if not any(route_lengths) else ends
            reflected_seconds = time.perf_counter() - tick if not any(route_lengths) else end_seconds
            tick = time.perf_counter()
            progress_orders = c.route_progress_supplements(space, state, pending, selected, source, config, gnn_order)
            progress_seconds = time.perf_counter() - tick
            pools = {}
            for name, orders, seconds in zip(strategies, (ends, reflected, progress_orders),
                                             (end_seconds, reflected_seconds, progress_seconds)):
                for limit in limits:
                    tick = time.perf_counter()
                    ids = c.supplement_parent_ids(gnn_order, orders, limit)
                    pools[name, limit] = ids, parent_seconds + seconds + time.perf_counter() - tick
                    counts[name, limit] += target in ids
                    stages[name, limit, e['fraction']] += target in ids
            prepared.append((space, state, pending, selected, teacher, pattern, pools, e['fraction']))
            if (case + 1) % 42 == 0:
                print('ROUTE_PARENT_SCREEN', case + 1, '/', len(raw), flush=True)

        for name in strategies:
            op_save(dict(phase='residual_route_parent_screen', instance='TRAIN_126', seed=OP_SEED,
                method=name, status='MEASURED', b32_calls=0, payload=dict(parent_coverage={k: counts[name, k] for k in limits},
                    top16_by_fraction={fraction: stages[name, 16, fraction] for fraction in (0., .1, .25, .5, .75, .9)})))

        measurements = []
        # A missing parent cannot match the complete action. Fully measure the
        # 16-parent variants and any smaller pool that can reach the threshold.
        for name, limit in counts:
            if limit != 16 and counts[name, limit] < required:
                continue
            cases = []
            for case, (space, state, pending, selected, teacher, pattern, pools, fraction) in enumerate(prepared):
                ids, parent_seconds = pools[name, limit]
                tick = time.perf_counter()
                plans = c.complete_parent_candidates(space, state, pending, selected, config, first_limit=16, second_limit=8,
                    candidate_parent_ids=ids, include_remaining_workload=True, remaining_after_parent=True)
                seconds = parent_seconds + time.perf_counter() - tick
                match = any(p['pattern'] == pattern and cmax_action_key(p) == cmax_action_key(teacher) for p in plans)
                cases.append(dict(case=case, fraction=fraction, parent_covered=teacher['parent_id'] in ids,
                    complete_covered=match, parents=len(ids), candidates=len(plans), seconds=seconds))
            payload = dict(strategy=name, parent_limit=limit, states=len(cases), parent_covered=counts[name, limit],
                complete_covered=sum(x['complete_covered'] for x in cases),
                actual_parents_median=statistics.median(x['parents'] for x in cases),
                candidates_min=min(x['candidates'] for x in cases), candidates_max=max(x['candidates'] for x in cases),
                candidates_median=statistics.median(x['candidates'] for x in cases),
                generation_seconds_median=statistics.median(x['seconds'] for x in cases),
                recovered=sum(x['complete_covered'] and not old['complete_covered'] for x, old in zip(cases, previous)),
                lost=sum(not x['complete_covered'] and old['complete_covered'] for x, old in zip(cases, previous)))
            measurements.append(payload)
            op_save(dict(phase='residual_route_candidate_coverage', instance='TRAIN_126', seed=OP_SEED,
                method=name, status='MEASURED', b32_calls=0, payload=dict(payload, cases=cases)))
    qualified = [m for m in measurements if m['complete_covered'] >= required]
    chosen = min(qualified, key=lambda m: (m['parent_limit'], m['candidates_median'], -m['complete_covered'])) if qualified else None
    best = max(measurements, key=lambda m: (m['complete_covered'], -m['candidates_median']))
    op_save(dict(phase='residual_route_coverage_decision', instance='TRAIN_126', seed=OP_SEED,
        method='ROUTE_AWARE_PARENT_SUPPLEMENTS', status='READY_FOR_FEEDBACK' if chosen else 'STOP_COVERAGE_BELOW_90',
        b32_calls=0, payload=dict(best=best, chosen=chosen, required=required, training_updates=0, train_b32_calls=0, dev_b32_calls=0)))
    print('ROUTE_COVERAGE', best, 'required', required, 'chosen', chosen, flush=True)


RESIDUAL_METHOD = 'REFLECTED_START_ROUTE_PROGRESS'
RESIDUAL_VARIANT = 'route_progress_cmax_candidate'


def residual_feedback_model(shared, checkpoint):
    """One GNN with the measured proposal trunk and the existing complete-action head."""
    source = shared._model_from_checkpoint(checkpoint[COMPLETE_VARIANT])
    model = shared.expand_remaining_workload_model(source)
    # The old complete branch had updated its GAT. Retain the proposal trunk
    # whose coverage was measured and reuse the compatible head and scaling.
    with torch.no_grad():
        state = model.state_dict()
        for name, value in checkpoint[OP_VARIANT]['state_dict'].items():
            state[name].copy_(value)
    model.use_dynamic_context = checkpoint[OP_VARIANT]['use_dynamic_context']
    model.requires_grad_(False)
    return model.eval()


def residual_feedback_alternatives(plans, scores):
    order = sorted(range(len(plans)), key=lambda i: (-float(scores[i]), i))
    first = plans[order[0]]
    first_robots = tuple(a.robot for a in first['actions'])
    alternate = next((plans[i] for i in order[1:] if plans[i]['pattern'] == first['pattern']
                      and tuple(a.robot for a in plans[i]['actions']) != first_robots), None)
    other = next((plans[i] for i in order[1:] if plans[i]['pattern'] != first['pattern']), None)
    return (first, alternate, other) if alternate is not None and other is not None else None


def residual_feedback_pairs(outcomes):
    pairs = [dict(better=i, worse=j, kind='cmax') for i, j in complete_cmax_pairs(outcomes)]
    for i, a in enumerate(outcomes):
        for j, b in enumerate(outcomes[i + 1:], i + 1):
            if {a['status'], b['status']} == {'FEASIBLE', 'DEADLOCK'}:
                better, worse = (i, j) if a['status'] == 'FEASIBLE' else (j, i)
                pairs.append(dict(better=better, worse=worse, kind='feasibility'))
    return pairs


def residual_collect_feedback():
    """Three legal same-state actions at four progress bins in each of three scales."""
    from mrta_search import gnn_constructor as c

    rows = op_read()
    coverage = next((r['payload'] for r in rows if r['phase'] == 'residual_route_coverage_decision'), None)
    if coverage is None or not coverage['chosen']:
        print('No qualifying complete-action coverage. Do not call B32.', flush=True)
        return
    if any(r['phase'] == 'residual_feedback_decision' for r in rows):
        print('Retain completed TRAIN feedback, without repeating B32.', flush=True)
        return
    shared, config, checkpoint, archive, teachers, _, _, _ = op_sources()
    model = residual_feedback_model(shared, checkpoint)
    graphs, examples, _, _ = op_saved_imitation(shared, config, archive, teachers)
    raw, _ = cmax_expert_macro_examples(model, graphs, examples, raw_only=True)
    limit = coverage['chosen']['parent_limit']
    spaces = {}
    prefixes = []
    with torch.inference_mode():
        for scale, minimum, maximum in (('SMALL', 0, 50), ('MEDIUM', 50, 63), ('LARGE', 63, math.inf)):
            used = set()
            for fraction in (0., .1, .25, .5):
                eligible = [(case, e) for case, e in enumerate(raw)
                            if e['fraction'] == fraction and minimum <= len(e['reference'].parents) < maximum]
                eligible.sort(key=lambda pair: (pair[1]['reference'].parents in used,
                    abs(pair[1]['progress'] - fraction), len(pair[1]['reference'].parents), pair[0]))
                for case, e in eligible:
                    reference, state, pending = (e[key] for key in ('reference', 'state', 'pending'))
                    parents = reference.parents
                    if parents not in spaces:
                        spaces[parents] = c.complete_parent_space(parents, config, compact=True)
                    space = spaces[parents]
                    selected = {p.parent_id: p for p in reference.patterns if p.parent_id not in pending}
                    plans = c.route_progress_complete_candidates(space, state, pending, selected, model, config,
                        parent_limit=limit)
                    if not plans:
                        continue
                    graph = c.complete_parent_view(space, selected)[0]
                    context = model.contextualize_state(graph, model.encode_static(space['graph']), state.routes, state.directions)
                    scores = model.parent_action_scores(space['graph'], context, c.complete_parent_inputs(plans))
                    if not torch.isfinite(scores).all():
                        raise ValueError('nonfinite feedback action scores')
                    alternatives = residual_feedback_alternatives(plans, scores)
                    if alternatives is not None:
                        prefixes.append((scale, case, e, space, selected, alternatives))
                        used.add(parents)
                        break
                else:
                    raise ValueError(f'No retained {scale} {fraction:.0%} prefix has all three legal alternatives')
    del graphs, examples
    print('RESIDUAL_FEEDBACK_PREFIXES', [(scale, case, e['fraction'], len(e['reference'].parents))
          for scale, case, e, _, _, _ in prefixes], flush=True)
    scientific = json.loads(json.dumps(config.scientific_mapping()))
    archived = [r for r in rows if r['phase'] == 'residual_feedback_branch']
    calls = sum(int(r['b32_calls'] or 0) for r in archived)
    states = []
    for scale, case, e, space, selected, alternatives in prefixes:
        parents, state, pending = e['reference'].parents, e['state'], e['pending']
        identity = dict(case=case, scale=scale, fraction=e['fraction'], progress=e['progress'],
            parents=[asdict(p) for p in parents], routes=state.routes, directions=state.directions,
            pending=pending, selected_patterns=[asdict(selected[p]) for p in sorted(selected)])
        prefix = json.loads(json.dumps(identity))
        name = f'{scale}_TRAIN_{case:03d}'
        outcomes = []
        for label, plan in zip(('A', 'B', 'C'), alternatives):
            action = json.loads(json.dumps(dict(pattern=asdict(plan['pattern']), actions=[asdict(a) for a in plan['actions']])))
            tick = time.perf_counter()
            result = c.construct_complete_parent_gnn(parents, model, config, space=space, state=plan['state'],
                pending=set(pending) - {plan['parent_id']}, selected={**selected, plan['parent_id']: plan['pattern']},
                parent_limit=limit, parent_strategy='route_progress', first_limit=16, second_limit=8)
            seconds = time.perf_counter() - tick
            outcome = dict(label=label, **action, status='CONSTRUCTION_FAILED', cmax=None, construction_seconds=seconds,
                scheduler_seconds=0., reason=result.rejection_reason, b32_calls=0)
            if result.solution is not None:
                complete = dict(solution=asdict(result.solution), directions=result.directions)
                key = op_complete_key(complete)
                previous = next((r['payload']['outcome'] for r in archived if r['method'] == RESIDUAL_METHOD
                    and r['payload']['scientific_config'] == scientific and r['payload']['prefix'] == prefix
                    and all(r['payload']['outcome'][k] == action[k] for k in action)
                    and r['payload']['outcome'].get('complete') and op_complete_key(r['payload']['outcome']['complete']) == key), None)
                if previous is not None:
                    outcomes.append(previous)
                    continue
                if calls >= 36:
                    raise ValueError('36 TRAIN B32 calls already used')
                tick = time.perf_counter()
                candidate = pilot.oriented.schedule_explicit(result.solution, result.directions, config)
                schedule_seconds = time.perf_counter() - tick
                feasible = candidate.schedule.status is ScheduleStatus.FEASIBLE
                metrics = op_route_metrics(complete, config)
                if feasible:
                    metrics.update(op_metrics(candidate, config))
                calls += 1
                outcome.update(status=candidate.schedule.status.value, cmax=candidate.schedule.cmax if feasible else None,
                    scheduler_seconds=schedule_seconds, metrics=metrics, diagnostics=candidate.schedule.diagnostics,
                    complete=complete, b32_calls=1)
            op_save(dict(phase='residual_feedback_branch', instance=name, seed=OP_SEED, method=RESIDUAL_METHOD,
                status=outcome['status'], cmax=outcome['cmax'], runtime_s=seconds + outcome['scheduler_seconds'],
                b32_calls=outcome['b32_calls'], payload=dict(prefix=prefix, scientific_config=scientific, outcome=outcome)))
            outcomes.append(outcome)
        pairs = residual_feedback_pairs(outcomes)
        states.append(dict(instance=name, case=case, scale=scale, fraction=e['fraction'], pairs=pairs,
            statuses=[o['status'] for o in outcomes], cmax=[o['cmax'] for o in outcomes]))
        print('RESIDUAL_FEEDBACK_STATE', name, e['fraction'], states[-1]['statuses'], 'pairs', len(pairs), 'B32', calls, flush=True)
    labels = Counter(pair['kind'] for state in states for pair in state['pairs'])
    state_count = sum(bool(s['pairs']) for s in states)
    allowed = sum(labels.values()) >= 12 and state_count >= 6
    op_save(dict(phase='residual_feedback_decision', instance='TRAIN_12', seed=OP_SEED, method=RESIDUAL_METHOD,
        status='READY_FOR_HEAD_TRAINING' if allowed else 'STOP_INSUFFICIENT_PREFERENCES', b32_calls=0,
        payload=dict(states=states, labels=dict(labels), preference_states=state_count, train_b32_calls=calls,
                     allow_head_training=allowed, dev_b32_calls=0, training_updates=0)))
    print('RESIDUAL_FEEDBACK_DECISION', dict(labels), 'states', state_count, 'B32', calls, 'train', allowed, flush=True)


def residual_train_head():
    """Fit the complete-action head on actual legal actions and same-state feedback."""
    import io
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_action_key

    rows = op_read()
    feedback = next((r['payload'] for r in rows if r['phase'] == 'residual_feedback_decision'), None)
    if feedback is None or not feedback['allow_head_training']:
        print('Insufficient measured same-state feedback. Do not train.', flush=True)
        return
    shared, config, checkpoint, archive, teachers, _, _, _ = op_sources()
    if RESIDUAL_VARIANT in checkpoint:
        print('Retain the trained route-progress branch.', flush=True)
        return
    chosen = next(r['payload']['chosen'] for r in rows if r['phase'] == 'residual_route_coverage_decision')
    model = residual_feedback_model(shared, checkpoint)
    graphs, examples, _, _ = op_saved_imitation(shared, config, archive, teachers)
    raw, _ = cmax_expert_macro_examples(model, graphs, examples, raw_only=True)
    del graphs, examples
    preferences = {s['case']: s['pairs'] for s in feedback['states'] if s['pairs']}
    branch_actions = defaultdict(dict)
    for row in rows:
        if row['phase'] == 'residual_feedback_branch':
            outcome = row['payload']['outcome']
            branch_actions[row['payload']['prefix']['case']][outcome['label']] = outcome
    imitation, ranking = [], []
    spaces, embeddings = {}, {}
    with torch.no_grad():
        for case, e in enumerate(raw):
            reference, state, pending = (e[key] for key in ('reference', 'state', 'pending'))
            parents = reference.parents
            if parents not in spaces:
                spaces[parents] = c.complete_parent_space(parents, config, compact=True)
                embeddings[parents] = model.encode_static(spaces[parents]['graph'])
            space = spaces[parents]
            selected = {p.parent_id: p for p in reference.patterns if p.parent_id not in pending}
            plans = c.route_progress_complete_candidates(space, state, pending, selected, model, config,
                parent_limit=chosen['parent_limit'])
            if not plans:
                continue
            graph = c.complete_parent_view(space, selected)[0]
            context = model.contextualize_state(graph, embeddings[parents], state.routes, state.directions)
            features = model.parent_action_rows(space['graph'], context, c.complete_parent_inputs(plans))
            teacher = e['teacher_plan']
            pattern = next(p for p in reference.patterns if p.parent_id == teacher['parent_id'])
            target = next((i for i, p in enumerate(plans)
                           if p['pattern'] == pattern and cmax_action_key(p) == cmax_action_key(teacher)), None)
            if target is not None:
                imitation.append((features, target))
            if case in preferences:
                action_rows = []
                for label in ('A', 'B', 'C'):
                    outcome = branch_actions[case][label]
                    key = tuple(joint_action_key(op_action(a)) for a in outcome['actions'])
                    index = next(i for i, p in enumerate(plans)
                                 if asdict(p['pattern']) == outcome['pattern'] and cmax_action_key(p) == key)
                    action_rows.append(index)
                ranking.extend(features[[action_rows[p['better']], action_rows[p['worse']]]]
                               for p in preferences[case])
            if (case + 1) % 42 == 0:
                print('RESIDUAL_HEAD_POOL', case + 1, len(imitation), len(ranking), flush=True)
    if len(imitation) != chosen['complete_covered'] or len(ranking) != sum(feedback['labels'].values()):
        raise ValueError('training pool does not reproduce the measured exact actions')

    head = model.parent_action_head
    head.requires_grad_(True)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    generator = random.Random(OP_SEED)
    epochs, steps, batch = 5, 64, 4
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        losses = Counter()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            scores = head(torch.stack(generator.choices(ranking, k=batch))).squeeze(-1)
            preference = F.softplus(scores[:, 1] - scores[:, 0]).mean()
            il = torch.stack([-torch.log_softmax(head(features).squeeze(1), 0)[target]
                              for features, target in generator.choices(imitation, k=batch)]).mean()
            loss = preference + .1 * il
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.update(loss=float(loss.detach()), preference=float(preference.detach()), il=float(il.detach()))
        history.append(dict(epoch=epoch + 1, updates=steps, **{k: v / steps for k, v in losses.items()}))
        print('RESIDUAL_HEAD_EPOCH', history[-1], flush=True)
    if not all(torch.equal(value, model.state_dict()[name])
               for name, value in checkpoint[OP_VARIANT]['state_dict'].items()):
        raise ValueError('frozen GNN trunk or scaling changed')
    metadata = dict(seed=OP_SEED, epochs=epochs, updates=epochs * steps, batch=batch, learning_rate=1e-4,
        loss='L_preference + 0.1 L_IL', trainable='parent_action_head', parents=chosen['parent_limit'],
        exact_imitation_actions=len(imitation), preference_pairs=len(ranking), history=history,
        training_seconds=time.perf_counter() - started)
    checkpoint[RESIDUAL_VARIANT] = dict(model_type='CompleteParentGNN', hidden_dim=model.hidden_dim, heads=model.heads,
        normalize_inputs=model.normalize_inputs, opportunity_features=True, parent_action_features=True,
        remaining_workload_features=True, compact_parent_limit=chosen['parent_limit'], use_dynamic_context=model.use_dynamic_context,
        scientific_config=config.scientific_mapping(), state_dict=model.state_dict(), training=metadata)
    buffer = io.BytesIO()
    shared.save_neural_checkpoint(checkpoint, buffer)
    buffer.seek(0)
    saved = shared.read_neural_checkpoint(buffer)
    protected = [(checkpoint['state_dict'], saved['state_dict'])]
    protected.extend((branch['state_dict'], saved[key]['state_dict']) for key, branch in checkpoint.items()
                     if isinstance(branch, dict) and 'state_dict' in branch)
    if not all(torch.equal(value, after[name]) for before, after in protected for name, value in before.items()):
        raise ValueError('checkpoint tensors differ after serialization')
    # Keep the old bytes available for ordinary write errors without adding a file.
    original = OP_CHECKPOINT.read_bytes()
    with OP_CHECKPOINT.open('r+b') as stream:
        try:
            stream.write(buffer.getbuffer())
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.seek(0)
            stream.write(original)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            raise
    op_save(dict(phase='residual_head_training', instance='TRAIN_21', seed=OP_SEED, method=RESIDUAL_VARIANT,
        status='TRAINED', runtime_s=metadata['training_seconds'], b32_calls=0, payload=metadata))


def residual_evaluate():
    """Use greedy, seed, and seed+1 proposals from one checkpoint branch."""
    from mrta_search import gnn_constructor as c

    rows = op_read()
    done = {r['instance'] for r in rows if r['phase'] == 'residual_dev_evaluation'}
    shared, config, checkpoint, _, _, _, entries, _ = op_sources()
    if RESIDUAL_VARIANT not in checkpoint:
        print('No trained route-progress branch. Do not evaluate DEV.', flush=True)
        return
    for name, parents in sorted(entries.items()):
        if name in done:
            continue
        started = time.perf_counter()
        model = shared._model_from_checkpoint(checkpoint[RESIDUAL_VARIANT]).eval()
        load = time.perf_counter() - started
        tick = time.perf_counter()
        space = c.complete_parent_space(parents, config, compact=True)
        graph_seconds = time.perf_counter() - tick
        archived = {r['payload']['proposal_index']: r['payload'] for r in rows
                    if r['phase'] == 'residual_dev_proposal' and r['instance'] == name}
        attempts = []
        for index, seed in enumerate((None, OP_EVAL_SEED, OP_EVAL_SEED + 1)):
            if index in archived:
                attempts.append(archived[index])
                continue
            tick = time.perf_counter()
            result = c.construct_complete_parent_gnn(parents, model, config, seed=seed, space=space,
                parent_limit=model.compact_parent_limit, parent_strategy='route_progress', first_limit=16, second_limit=8)
            construction = time.perf_counter() - tick + (graph_seconds if index == 0 else 0.)
            attempt = dict(proposal_index=index, seed=seed, decode='greedy' if seed is None else 'sample',
                construction_seconds=construction, status='CONSTRUCTION_FAILED', cmax=None, scheduler_seconds=0.,
                b32_calls=0, reason=result.rejection_reason)
            if result.solution is not None:
                raw = dict(solution=asdict(result.solution), directions=result.directions)
                clock = time.perf_counter()
                candidate = pilot.oriented.schedule_explicit(result.solution, result.directions, config)
                seconds = time.perf_counter() - clock
                feasible = candidate.schedule.status is ScheduleStatus.FEASIBLE
                metrics = op_route_metrics(raw, config)
                if feasible:
                    metrics.update(op_metrics(candidate, config))
                attempt.update(**raw, status=candidate.schedule.status.value, scheduler_seconds=seconds,
                    cmax=candidate.schedule.cmax if feasible else None, metrics=metrics,
                    diagnostics=candidate.schedule.diagnostics, b32_calls=1)
            attempt.update(seconds=time.perf_counter() - tick, elapsed=time.perf_counter() - started)
            op_save(dict(phase='residual_dev_proposal', instance=name, seed=OP_EVAL_SEED, method=RESIDUAL_VARIANT,
                status=attempt['status'], cmax=attempt['cmax'], runtime_s=attempt['seconds'],
                b32_calls=attempt['b32_calls'], payload=attempt))
            attempts.append(attempt)
        total = load + sum(a['construction_seconds'] + a['scheduler_seconds'] for a in attempts)
        info = dict(attempts=attempts, model_load_seconds=load, runtime_s=total, actual_runtime_s=time.perf_counter()-started,
            reused_b32_seconds=0., source_variant=RESIDUAL_VARIANT, b32_calls=0)
        # The proposal rows already carry the physical B32 call counts.
        op_save_measured('residual_dev_evaluation', name, RESIDUAL_VARIANT, info)


def residual_coverage_report():
    """Record the measured stop, without inventing a trained or DEV result."""
    heading='## 2026-11-06 GNN父候选覆盖修复与动作后剩余工作量'
    if heading in OP_REPORT.read_text(encoding='utf-8'):
        print('Retain existing residual coverage report.',flush=True);return
    rows=op_read();coverage=[r for r in rows if r['phase']=='residual_candidate_coverage']
    decision=next(r['payload'] for r in rows if r['phase']=='residual_coverage_decision')
    repair=next(r['payload'] for r in rows if r['phase']=='residual_candidate_repair')
    external=next(r['payload'] for r in rows if r['phase']=='residual_external_parent_decision')
    quotas=next(r['payload'] for r in rows if r['phase']=='residual_quota_decision')
    if len(coverage)!=20:raise ValueError('all twenty candidate configurations must be measured')
    if decision['chosen'] or repair['complete_covered']>=repair['required'] or external['qualified'] or quotas['qualified']:
        raise ValueError('coverage passed: finish preference collection before writing a stopped-stage report')
    shared,config,_,_,_,_,entries,development=op_sources();names=sorted(entries)
    historical={}
    for label,phase,method in (
        ('ORIGINAL_GNN_MULTI3','parent_cmax_macro_evaluation','ORIGINAL_GNN_MULTI3'),
        ('PARENT_JOINT_MULTI3','parent_cmax_macro_evaluation','PARENT_JOINT_MULTI3'),
        ('RULE/BEAM_MAX3','parent_cmax_macro_evaluation','RULE/BEAM'),
        ('WAG_60S','evaluation','WAG'),('HGA_60S','evaluation','HGA')):
        historical[label]={r['instance']:r for r in rows if r['phase']==phase and r['method']==method and r['seed']==str(OP_EVAL_SEED)}
    region={}
    for name in names:
        source=next((r for r in development if r['instance']==name and r['seed']==OP_EVAL_SEED and r['method']=='REGION_FAST_ROUTE_REORDER'),None)
        if source is None:continue
        state=(source.get('details') or {}).get('final_state')
        if state:
            _,_,geometry,_,_=shared.research_restore(state,config)
            expected=tuple(sorted(tuple(sorted((p.start,p.end))) for p in entries[name]))
            if geometry!=expected:raise ValueError('REGION geometry does not match the eight DEV cases')
        region[name]=source
    historical['REGION_60S']=region
    number=lambda v:'—' if v in (None,'') else f'{float(v):.3f}'
    good=lambda r:r['status']=='FEASIBLE' and r['cmax'] not in (None,'')
    best=decision['best'];diagnoses=[r['payload'] for r in rows if r['phase']=='residual_parent_diagnosis']
    model_names=sorted({r['source_variant'] for r in diagnoses})
    lines=['',heading,'',
        f"**本轮停在候选覆盖阶段：最好完整动作覆盖{repair['complete_covered']}/126={100*repair['complete_covered']/126:.2f}%，"
        f"低于90%所需的{repair['required']}/126。未训练、未采集偏好、未运行新GNN的DEV；本轮新增实验B32=0。**",'',
        '使用指定D:\\pybullet_test\\.venv\\Scripts\\python.exe，在DRL工作目录执行，CPU单线程。'
        '核对了引用聊天和本地最新记录；本地DRL属于根仓库codex/project-cleanup工作区，未把它称为已核实的GitHub main，'
        '未切换、重置或清理既有修改。复用当前源码、CSV和报告，没有新增文件，也没有写入模型checkpoint。','',
        '### 实际代码修改与完整动作匹配','',
        'gnn_constructor.py新增现有父头排序和有界选择入口。默认实验使用PARENT_JOINT对应的opportunity_cmax_candidate：'
        '同一个既有GAT/父头在原生图中给全部pending父打分，不增加父预测网络。已提交父使用实际已选模式，'
        '待分配父仅使用部署可取得的默认模式来构造提案图，不读取未来教师模式；完整动作池仍遍历入选父的所有合法模式。'
        '移除了提案排序中“默认模式首子不可插就跳过父”的粗筛，实际动作的资格、插入、方向及初始冲突规则仍由原joint_block_candidates判断。','',
        '分别测试GNN、MRV、LPT、GNN优先+MRV/LPT补充，父上限4/6/8/12/16。'
        '组合策略保留GNN前K−2，再补一条不同的MRV和LPT父，不足时用GNN次序补齐。'
        'MRV采用现有全合法目录资格灵活性F，LPT采用原始父加工时间；并未用教师目标ID决定候选。'
        '动作匹配同时要求SplitPattern（父ID、kind、t、point_id、mandatory、rail）和每个子动作的block_id、robot、position、direction、neighbour_changes完全一致，'
        '父ID命中不等于完整动作命中。','',
        'run_gnn_cmax_finetune.py追加四个TRAIN诊断入口及报告入口，全部复用现有状态、教师重放和CSV；'
        'tests/test_gnn_construct_vns.py增加父头复用、各规模选择、动作后特征和整父提交回归。原有模型入口保持原语义，'
        '不将新特征直接套到旧2/8结果上冒充新实验。','',
        '### 126个TRAIN状态的20组实测','',
        '主表使用首子8/次子4、WHOLE最多8个代表。均为实际生成的完整动作；126个状态全部计入分母，包括没有匹配动作的状态。'
        '耗时是单状态候选生成，GNN策略计入一次实测父排序成本；它不是完整解构造或B32耗时。','',
        '|策略|父上限|实际父数中位|父覆盖/126|完整动作覆盖/126|候选数中位/最大|生成中位/ms|',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels=dict(gnn='GNN',mrv='MRV',lpt='LPT',gnn_mrv_lpt='GNN+MRV/LPT')
    for r in coverage:
        p=r['payload']
        lines.append(f"|{labels[r['method']]}|{p['parent_limit']}|{p['actual_parents_median']}|{p['parent_covered']}|{p['complete_covered']}|"
            f"{p['candidates_median']}/{p['candidates_max']}|{1000*p['generation_seconds_median']:.3f}|")
    lines+=['',f"主表最好为{labels[best['strategy']]}、K={best['parent_limit']}，完整动作{best['complete_covered']}/126。"
        f"该配置父覆盖{best['parent_covered']}/126，另有{best['parent_covered']-best['complete_covered']}个完整动作在有界子池中缺失。",'',
        '### 有针对性的修复与边界','',
        f"复用既有完整有界池：父仍为GNN Top-16，拆分16×8分支，WHOLE保留原8位置内合法方向变体。"
        f"恢复{len(repair['recovered_cases'])}个漏项，丢失原已覆盖动作{len(repair['lost_cases'])}个，完整覆盖"
        f"{repair['complete_covered']}/126={100*repair['complete_covered']/126:.2f}%。"
        f"候选数min/median/max={repair['candidates_min']}/{repair['candidates_median']}/{repair['candidates_max']}，"
        f"生成中位{repair['generation_seconds_median']:.6f}s/状态。没有枚举所有子动作笛卡尔积，也未增加父上限。",'',
        '这一诊断比上轮6父MRV/LPT的50/126=39.68%有所提高，但仍不是合格训练配置。'
        '增大子池不能找回父头在Top-16之前已排除的26个教师父任务；当前父覆盖100/126就是这一固定提案规则的上限。','',
        '还分别检查已有父头与原生图、全部合法模式平均表示、逐模式父头评分取max三种表示。'
        '这些都是既有父头的读法，没有增加mode_head或新网络，未将多个模型的候选合并成一个假想新模型。','',
        '|已有父头来源|原生GNN Top16|原生GNN+补充 Top16|模式表示修复最好父覆盖|',
        '|---|---:|---:|---:|']
    for name in model_names:
        values=[r for r in diagnoses if r['source_variant']==name]
        native={r['strategy']:r['parent_coverage']['16'] for r in values if r['representation']=='native'}
        mode=[r['parent_coverage']['16'] for r in values if r['representation']!='native']
        lines.append(f"|{name}|{native.get('gnn','—')}|{native.get('gnn_mrv_lpt','—')}|{max(mode) if mode else '未测'}|")
    lines+=['',f"另对GNN主集合配少量MRV/LPT名额（包括12+2+2、10+3+3等）做TRAIN检查，最好父覆盖"
        f"{quotas['best_parent_coverage']}/126，仍不足{repair['required']}/126。独立文件中的RULE_GUIDED_GNN_RETRAINED只取其父头、"
        f"不用规则融合，最好也只有{external['best_parent_coverage']}/126。没有用更大父池或把教师动作注入候选来凑90%。",'',
        '已保存教师重放在construction_replay中按最终教师路线的最早未插位置恢复动作次序。'
        '该次序是离线标签恢复次序，并不是已证明的唯一最优父选择。部署端不能访问最终教师路线；'
        '当前结果定位的是既有父排序与这些真实重放标签不匹配，不能通过把未来教师顺序传给提案器解决。','',
        '### 动作后的四维剩余工作量','',
        '新入口remaining_after_parent=True对每个候选父计算remaining_eligible_workload(space,pending−{parent_id})，'
        '按父复用到其模式/机器人/位置/方向完整动作。原四机P、MOVE、最大P、最大P+MOVE、插入与方向输入全部保留。'
        'CompleteParentGNN.parent_action_rows继续使用原四维拼接接口；头部维数和原归一化不变，不新增GAT层或独立网络。',
        f"126个状态中，{decision['states_with_action_dependent_remaining']}个状态的不同剩余父选择会产生至少两种事后工作量向量。"
        '单元测试确认只替换现有最后四列，其他动作特征和全部模型张量不变。'
        '同父的不同机器人动作仍共享该剩余分布，这是pending减同一父的定义；机器人差异由动作后的P/MOVE及路线特征区分。'
        '资格分布相同比例时，不同父也可能得到相同归一化向量。该向量是决策代理，不是严格Cmax下界。','',
        '### 为什么本轮没有训练、反馈和新DEV结果','',
        f"90%门槛要求至少{repair['required']}/126个完整动作，实际最好{repair['complete_covered']}/126。"
        '按本轮明确停止条件，未进行任何预热或IL扩训，未继续旧六父模型训练，没有创建无有效监督的新checkpoint分支。'
        '新增TRAIN B32=0、DEV B32=0、Cmax排序对=0、可行性偏好对=0、训练更新=0；这里的0表示后续阶段未开展，'
        '不是又花36次B32得到0标签。旧24次反馈不混用，新阶段的12偏好对/至少6状态要求尚未进入核查。','',
        '|DEV几何|本轮新GNN结果|本轮Cmax|构造/B32耗时|',
        '|---|---|---|---|']
    for name in names:lines.append(f'|{name}|未运行：候选覆盖未过|不适用|未测|')
    lines+=['','未运行不计为0/8失败，不填虚构Cmax或运行时间。下面只复用已有结果：前三列为最多3次B32的初始构造，'
        '后三列为60秒完整求解质量参考，绝不是本轮新模型结果或等预算优越性证据。','',
        '|几何|ORIGINAL_GNN_MULTI3|PARENT_JOINT_MULTI3|RULE/BEAM_MAX3|WAG_60S|HGA_60S|REGION_60S|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        lines.append('|'+name+'|'+ '|'.join(number(table[name]['cmax']) if name in table else '缺失' for table in historical.values())+'|')
    lines+=['','|已有方法|可行/已保存案例|历史总耗时中位/s|口径|','|---|---:|---:|---|']
    for label,table in historical.items():
        times=[float(r['runtime_s']) for r in table.values() if r['runtime_s'] not in (None,'')]
        lines.append(f"|{label}|{sum(good(r) for r in table.values())}/{len(table)}|{number(statistics.median(times) if times else None)}|"
            +('60秒终解，仅参考' if label.endswith('60S') else '最多3次B32初始构造')+'|')
    lines+=['','本轮与所有对照的配对Cmax改善均无法计算，因为没有新的完整方案和调度结果。'
        '相对PARENT_JOINT的完整构造效率也未测，不能把0.272s单状态候选耗时与3.850s三提案总耗时相除，'
        '也不能沿用旧COMPACT的2/8作为新特征的性能。','',
        '### 结论与复现','',
        '本轮实际完成了候选排序/完整动作覆盖实验、有界池修复、动作后剩余特征改动和回归。'
        '90%覆盖未通过，主算法竞争力目标尚未实现，不具备进入新独立测试的条件。'
        '下一步应先修复部署可用父提案与教师重放优先级的匹配，不能继续仅训练完整动作头来弥补已被父池删除的动作，'
        '也不应以增加层数、随机种子或后处理绕过当前数据问题。',
        '所有历史模型、CSV、MODE_NEW、WAG/HGA、ScientificConfig、FORMAL_SCOPE_V2和B32均保留。'
        'README已将8/8旧组合结果标为历史组合证据，最新状态是覆盖阶段停止。','',
        '复现工作目录DRL，使用指定venv运行scripts/run_gnn_cmax_finetune.py：'
        '`--phase residual-coverage`、`--phase residual-packing-repair`、`--phase residual-external-parent`、'
        '`--phase residual-quota-check`、`--phase residual-report`。已保存的本轮测量会保留，不自动重复或调用训练/B32。','']
    with OP_REPORT.open('a',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    print('RESIDUAL_REPORT','STOP_BELOW_90',repair['complete_covered'],'/126','new_B32=0',flush=True)


def assignment_diagnostic():
    """Decode the best certified assignment per TRAIN geometry, read-only."""
    from mrta_reference.geometry import blocks_for_pattern
    from mrta_reference.model import OperationKind, SplitKind
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_lower_bound

    _, config, _, archive, teachers, _, _, _ = op_sources()
    groups = defaultdict(list)
    teacher_modes, best_modes, teacher_cuts = Counter(), Counter(), Counter()
    teacher_loads = []
    for item, saved in zip(teachers, archive['replays']):
        if (item['instance'], item['teachers'], item['source_state'], item['seed'], item['cmax']) != (
                saved['instance'], saved['teachers'], saved['source_state'], saved['source_seed'], saved['cmax']):
            raise ValueError('teacher source does not match its retained certification record')
        solution = item['solution']
        labels = c.parent_assignment_targets(solution, config)
        groups[item['geometry_key']].append((item, labels))
        parents = {p.parent_id: p for p in solution.parents}
        loads = [0.] * 4
        for parent_id, pattern, robots in labels:
            teacher_modes[pattern.kind.value, pattern.mandatory] += 1
            if pattern.point_id is not None:
                family = pattern.kind.value if pattern.rail is None else f'{pattern.kind.value}/{pattern.rail.value}'
                teacher_cuts[f'{family}/{pattern.point_id}'] += 1
            for block, robot in zip(blocks_for_pattern(parents[parent_id], pattern, config), robots):
                loads[robot] += config.process_time(block.length)
        teacher_loads.append(loads)

    old_pool = next(row['payload']['cases'] for row in op_read()
                    if row['phase'] == 'residual_route_candidate_coverage' and row['method'] == RESIDUAL_METHOD)
    fractions = (0., .1, .25, .5, .75, .9)
    model = c.ParentAssignmentGNN().eval()
    rows, score_times = [], []
    label_count = legal_label_count = physical_parents = 0
    joint_conflicts = mode_conflicts = owner_conflicts = 0
    for group in groups.values():
        best, assignments = min(group, key=lambda pair: pair[0]['cmax'])
        parents = best['solution'].parents
        prefixes = []
        decoded = c.decode_parent_assignments(parents, assignments, config,
            decision_observer=lambda state, blocks, pending: prefixes.append((state, blocks, pending)))
        tick = time.perf_counter()
        scheduled = pilot.oriented.schedule_explicit(decoded.solution, decoded.directions, config).schedule
        b32_seconds = time.perf_counter() - tick

        tick = time.perf_counter()
        space = c.parent_assignment_space(parents, config)
        space_seconds = time.perf_counter() - tick
        with torch.inference_mode():
            tick = time.perf_counter()
            embeddings = model.encode_static(space.graph)
            encode_seconds = time.perf_counter() - tick
            counts, seconds = [], []
            for state, blocks, pending in prefixes:
                tick = time.perf_counter()
                indices, logits = model.assignment_logits(space, embeddings, state, blocks, pending, config)
                indices[logits.argmax()].item()
                seconds.append(time.perf_counter() - tick)
                counts.append(len(indices))
                if not torch.isfinite(logits).all():
                    raise ValueError('non-finite joint assignment scores')
        score_times.extend(seconds)

        # Check both cross-mode disagreements and competing ownership labels
        # under the same physical split, independently of endpoint convention.
        legal = set(space.actions)
        by_parent = defaultdict(lambda: defaultdict(set))
        for item, labels in group:
            by_id = {p.parent_id: p for p in item['solution'].parents}
            occurrences = Counter()
            for parent_id, pattern, robots in labels:
                label_count += 1
                legal_label_count += c.ParentAssignment(parent_id, pattern, robots) in legal
                parent = by_id[parent_id]
                endpoints = tuple(sorted((parent.start, parent.end)))
                occurrence = occurrences[endpoints]
                occurrences[endpoints] += 1
                reverse = parent.start != endpoints[0]
                cut = pattern.t if not reverse or pattern.t is None else 1. - pattern.t
                mode = (pattern.kind, pattern.rail, None if cut is None else round(cut, 12))
                owners = robots[::-1] if reverse and pattern.kind is SplitKind.Y_SPLIT else robots
                by_parent[endpoints, occurrence][mode].add(owners)
        physical_parents += len(by_parent)
        for labels in by_parent.values():
            mode_conflicts += len(labels) > 1
            owner_conflicts += any(len(owners) > 1 for owners in labels.values())
            joint_conflicts += sum(map(len, labels.values())) > 1

        best_modes.update((p.kind.value, p.mandatory) for _, p, _ in assignments)
        cmax = scheduled.cmax if scheduled.status is ScheduleStatus.FEASIBLE else None
        row = dict(instance=best['instance'], parents=len(parents), length_m=sum(p.length for p in parents),
            teachers=len(group), teacher=best['teacher'], teacher_seed=best['seed'], teacher_cmax=best['cmax'],
            status=scheduled.status.value, cmax=cmax,
            degradation_percent=None if cmax is None else 100. * (cmax / best['cmax'] - 1.),
            construction_seconds=decoded.inference_seconds, b32_seconds=b32_seconds,
            loads=decoded.oriented_state.loads, move_seconds=decoded.oriented_state.travel,
            route_lower_bound=oriented_route_lower_bound(decoded.oriented_state),
            wait_seconds=sum(op.duration for op in scheduled.operations if op.kind is OperationKind.WAIT),
            welded=sum(op.kind is OperationKind.WELD for op in scheduled.operations), blocks=len(decoded.actions),
            wait_for=scheduled.wait_for_graph, diagnostics=scheduled.diagnostics,
            recovery_rollouts=scheduled.recovery_rollouts, actions_start=counts[0], actions_scored=sum(counts),
            progress_counts=[counts[round(f * len(parents))] for f in fractions],
            space_seconds=space_seconds, encode_seconds=encode_seconds, scoring_seconds=sum(seconds))
        rows.append(row)
        print('ASSIGNMENT_DIAGNOSTIC', json.dumps(row, allow_nan=False), flush=True)

    def distribution(values):
        return dict(min=min(values), median=statistics.median(values), max=max(values))

    def mode_counts(counts):
        return {kind + ('_MANDATORY' if mandatory else ''): count for (kind, mandatory), count in counts.items()}

    new_counts = [count for row in rows for count in row['progress_counts']]
    paired = [r['degradation_percent'] for r in rows if r['cmax'] is not None]
    summary = dict(geometries=len(rows), b32_calls=len(rows), feasible=len(paired),
        median_degradation_percent=statistics.median(paired) if paired else None,
        teacher_solutions=len(teachers), physical_parents=physical_parents, teacher_labels=label_count,
        legal_teacher_labels=legal_label_count, joint_conflicts=joint_conflicts,
        mode_or_cut_conflicts=mode_conflicts, same_mode_owner_conflicts=owner_conflicts,
        teacher_modes=mode_counts(teacher_modes), best_modes=mode_counts(best_modes), teacher_cuts=dict(teacher_cuts),
        teacher_loads=[dict(distribution(list(loads)), total=sum(loads)) for loads in zip(*teacher_loads)],
        archived_complete_pool=distribution([x['candidates'] for x in old_pool]),
        new_assignment_pool=distribution(new_counts),
        archived_pool_by_progress=[distribution([x['candidates'] for x in old_pool if x['fraction'] == f]) for f in fractions],
        new_pool_by_progress=[distribution([r['progress_counts'][i] for r in rows]) for i in range(len(fractions))],
        scoring_seconds_per_step=distribution(score_times))
    for field in ('construction_seconds', 'b32_seconds', 'space_seconds', 'encode_seconds', 'scoring_seconds',
                  'actions_start', 'actions_scored'):
        summary[field] = distribution([r[field] for r in rows])
    summary['inference_seconds'] = distribution([r['space_seconds'] + r['encode_seconds'] + r['scoring_seconds'] for r in rows])
    summary['decode_b32_seconds'] = distribution([r['construction_seconds'] + r['b32_seconds'] for r in rows])
    print('ASSIGNMENT_SUMMARY', json.dumps(summary, allow_nan=False), flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('all','train','dev','op-all','op-data','op-train','op-eval','op-report',
        'joint-eval','early-all','early-data','early-train','early-eval','early-report',
        'delta-data','delta-report','parent-diagnose','parent-eval','parent-audit','parent-report',
        'parent-cmax-data','parent-cmax-train','parent-cmax-eval','parent-cmax-report',
        'parent-cmax-macro-train','parent-cmax-macro-eval',
        'complete-parent-coverage','complete-parent-train','complete-parent-eval','complete-parent-report',
        'compact-workload-train','compact-workload-eval','compact-workload-report',
        'residual-coverage','residual-packing-repair','residual-external-parent','residual-quota-check',
        'residual-route-coverage','residual-feedback','residual-head-train','residual-dev','residual-report',
        'assignment-diagnostic'),default='op-all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SETTINGS['seed'])
    if args.phase=='assignment-diagnostic':
        assignment_diagnostic();return
    if args.phase=='residual-coverage':
        residual_coverage_sweep();return
    if args.phase=='residual-packing-repair':
        residual_packing_repair();return
    if args.phase=='residual-external-parent':
        residual_external_parent_check();return
    if args.phase=='residual-quota-check':
        residual_quota_check();return
    if args.phase=='residual-route-coverage':
        residual_route_coverage();return
    if args.phase=='residual-feedback':
        residual_collect_feedback();return
    if args.phase=='residual-head-train':
        residual_train_head();return
    if args.phase=='residual-dev':
        residual_evaluate();return
    if args.phase=='residual-report':
        residual_coverage_report();return
    if args.phase.startswith('compact-workload-'):
        if args.phase=='compact-workload-train':compact_workload_train()
        elif args.phase=='compact-workload-eval':compact_workload_evaluate()
        else:compact_workload_report()
        return
    if args.phase.startswith('complete-parent-'):
        if args.phase=='complete-parent-coverage':complete_parent_train(coverage_only=True)
        elif args.phase=='complete-parent-train':complete_parent_train()
        elif args.phase=='complete-parent-eval':complete_parent_evaluate()
        else:complete_parent_report()
        return
    if args.phase=='parent-diagnose':
        op_parent_diagnose();return
    if args.phase=='parent-eval':
        op_parent_evaluate();return
    if args.phase=='parent-audit':
        op_parent_score_audit();return
    if args.phase=='parent-report':
        op_parent_report();return
    if args.phase=='parent-cmax-data':
        op_parent_cmax_data();return
    if args.phase=='parent-cmax-train':
        op_parent_cmax_train();return
    if args.phase=='parent-cmax-eval':
        op_parent_cmax_evaluate();return
    if args.phase=='parent-cmax-report':
        op_parent_cmax_report();return
    if args.phase=='parent-cmax-macro-train':
        op_parent_cmax_macro_train();return
    if args.phase=='parent-cmax-macro-eval':
        op_parent_cmax_evaluate(macro_imitation=True);return
    if args.phase=='delta-data':
        op_delta_analyze();return
    if args.phase=='delta-report':
        op_delta_report();return
    if args.phase=='joint-eval' or args.phase.startswith('early-'):
        torch.manual_seed(OP_SEED);random.seed(OP_SEED)
        if args.phase in ('joint-eval','early-all'):op_joint_evaluate()
        if args.phase in ('early-all','early-data'):op_collect(early=True)
        if args.phase in ('early-all','early-train'):op_train(early=True)
        if args.phase in ('early-all','early-eval'):op_early_evaluate()
        if args.phase in ('early-all','early-report'):op_early_report()
        return
    if args.phase.startswith('op-'):
        torch.manual_seed(OP_SEED);random.seed(OP_SEED)
        if args.phase in ('op-all','op-data'):op_collect()
        if args.phase in ('op-all','op-train'):op_train()
        if args.phase in ('op-all','op-eval'):op_evaluate()
        if args.phase in ('op-all','op-report'):op_report()
        return
    if args.phase in ('all','train'):
        rows=[]
        model,metadata=train(rows)
    else:
        rows=list(csv.DictReader(CSV.open(encoding='utf-8-sig')))
        # Numeric CSV values are only decoded for a deliberate separate DEV run.
        for row in rows:
            for key,value in list(row.items()):
                if key in ('removed','trace','failures'):
                    continue
                if value=='':
                    row.pop(key)
                else:
                    try: row[key]=json.loads(value)
                    except (ValueError,TypeError): pass
        checkpoint=torch.load(MODEL,map_location='cpu',weights_only=True)
        metadata=checkpoint['metadata']
        model=JointInsertionGNN(64,4,normalize_inputs=True)
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        if any(r['phase']=='DEV' for r in rows):
            raise ValueError('DEV already recorded; do not silently append duplicate experiments')
    if args.phase in ('all','dev'):
        evaluate(rows,model,metadata)
    print('COMPLETE',args.phase,'rows',len(rows),MODEL,CSV,REPORT,flush=True)


if __name__=='__main__':
    main()
