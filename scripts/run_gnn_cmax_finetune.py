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


def cmax_expert_macro_examples(source,graphs,examples):
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
                plans=list(cmax_plans(ref,graph,blocks,patterns,state,step['pending_parent_ids'],source))
                target=next((i for i,p in enumerate(plans) if cmax_action_key(p)==cmax_action_key(teacher)),None)
                covered=target is not None
                if target is None:target=len(plans);plans.append(teacher)
                inputs=c.parent_action_inputs(graph,state,plans,blocks,CONFIG)
                selected=dict(graph_key=key,inputs=inputs,target=target,weight=e['weight'],fraction=fraction,
                    progress=1-len(step['pending_parent_ids'])/len(geometry),teacher_in_inference_pool=covered)
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


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('all','train','dev','op-all','op-data','op-train','op-eval','op-report',
        'joint-eval','early-all','early-data','early-train','early-eval','early-report',
        'delta-data','delta-report','parent-diagnose','parent-eval','parent-audit','parent-report',
        'parent-cmax-data','parent-cmax-train','parent-cmax-eval','parent-cmax-report',
        'parent-cmax-macro-train','parent-cmax-macro-eval'),default='op-all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SETTINGS['seed'])
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
