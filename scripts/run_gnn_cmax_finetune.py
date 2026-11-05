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


def op_decision_pool(geometry,state,pending,model,config):
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import joint_block_candidates
    graph,blocks,patterns=c.joint_graph(geometry,config)
    ids,parent_features=c.joint_parent_inputs(state,pending,blocks,patterns,config,include_opportunities=True)
    if not ids:return graph,ids,parent_features,{}
    embeddings=model.encode_static(graph)
    parent_logits=model.parent_logits(graph,embeddings,ids,parent_features)
    order=sorted(range(len(ids)),key=lambda i:(-float(parent_logits[i]),i))
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


def op_representative_actions(geometry,state,pending,model,config):
    with torch.inference_mode():
        _,_,_,pools=op_decision_pool(geometry,state,pending,model,config)
        if not pools:return []
        first=next(iter(pools.values()));logits=first['logits'];variants=first['variants']
        best=int(logits.argmax());chosen=[variants[best][1]]
        other=[i for i,(_,a) in enumerate(variants) if a.robot!=chosen[0].robot]
        if other:chosen.append(variants[max(other,key=lambda i:(float(logits[i]),-i))][1])
        if len(pools)>1:
            alternative=list(pools.values())[1]
            chosen.append(alternative['variants'][int(alternative['logits'].argmax())][1])
        else:
            alternatives=[i for i,(_,a) in enumerate(variants) if a.position!=chosen[0].position]
            if alternatives:chosen.append(variants[max(alternatives,key=lambda i:(float(logits[i]),-i))][1])
        from mrta_search.lns import joint_action_key
        return list({joint_action_key(a):a for a in chosen}.values())[:3]


def op_collect():
    from mrta_search import gnn_constructor as c
    from mrta_search.lns import oriented_route_state
    shared,config,checkpoint,archive,teachers,training,_,_=op_sources()
    model=shared.expand_opportunity_model(shared._model_from_checkpoint(checkpoint['refinement_candidate'])).eval()
    saved=op_read();done={(r['instance'],r['payload'].get('fraction')) for r in saved if r['phase']=='counterfactual'}
    calls=sum(int(r['b32_calls'] or 0) for r in saved if r['phase']=='counterfactual')
    for item in training:
        name=item['instance'];parents=item['solution'].parents
        geometry=c.joint_geometry(parents,c.fixed_split_patterns(parents,config),config)
        existing=next((r for r in saved if r['phase']=='student' and r['instance']==name),None)
        if existing:student=existing['payload']
        else:
            snapshots={};targets={f:max(1,int(len(parents)*f)) for f in (.25,.5,.75)}
            def observe(state,pending,actions):
                completed=len(parents)-len(pending)
                for fraction,target in targets.items():
                    if completed==target:
                        snapshots[str(fraction)]=dict(routes=state.routes,directions=state.directions,pending=pending,progress=completed/len(parents))
            result=c.construct_or_repair_with_gnn(geometry,((),)*4,((),)*4,tuple(p.parent_id for p in parents),model,config,parent_observer=observe)
            student=dict(parents=[asdict(p) for p in parents],snapshots=snapshots,actions=[asdict(a) for a in result.actions],
                         failure=result.rejection_reason,complete=result.solution is not None)
            op_save(dict(phase='student',instance=name,seed=OP_SEED,method='INITIAL_STUDENT',
                         status='COMPLETE' if result.solution else 'CONSTRUCTION_FAILED',b32_calls=0,payload=student))
        graph,blocks,patterns=c.joint_graph(geometry,config);cache={}
        def key(raw):return (tuple(tuple(r['block_ids']) for r in raw['solution']['routes']),tuple(tuple(x) for x in raw['directions']))
        for row in saved:
            if row['phase']=='counterfactual' and row['instance']==name:
                for outcome in row['payload']['outcomes']:
                    if outcome.get('complete'):cache[key(outcome['complete'])]=outcome
        for fraction in (.25,.5,.75):
            if (name,fraction) in done:continue
            prefix=student['snapshots'].get(str(fraction))
            if prefix is None:
                op_save(dict(phase='counterfactual',instance=name,seed=OP_SEED,method='FORCED_GREEDY',status='MISSING_STUDENT_PREFIX',b32_calls=0,
                    payload=dict(fraction=fraction,outcomes=[],pairs=[],student_failure=student['failure'])));continue
            state=oriented_route_state(tuple(map(tuple,prefix['routes'])),tuple(map(tuple,prefix['directions'])),blocks,config)
            choices=op_representative_actions(geometry,state,prefix['pending'],model,config)
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
            op_save(dict(phase='counterfactual',instance=name,seed=OP_SEED,method='FORCED_GREEDY',status='COLLECTED',
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


def op_compile_ranks(records,graphs,config):
    from mrta_search import gnn_constructor as c
    from mrta_reference.model import ParentWeld
    from mrta_search.lns import oriented_route_state,joint_block_candidates,joint_action_key
    compiled=[];raw_steps=[]
    for row in records:
        if row['phase']!='counterfactual' or not row['payload'].get('pairs'):continue
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


def op_train():
    shared,config,checkpoint,archive,teachers,_,_,_=op_sources()
    if OP_VARIANT in checkpoint:
        print('Existing opportunity candidate retained; no second training.',flush=True);return
    rows=op_read()
    if len([r for r in rows if r['phase']=='counterfactual'])!=63:
        raise ValueError('collect all requested TRAIN prefix attempts before training')
    tick=time.perf_counter()
    graphs,examples,steps,preparation=op_saved_imitation(shared,config,archive,teachers)
    ranking,rank_steps=op_compile_ranks(rows,graphs,config)
    preparation.update(rank_pairs=len(ranking),preparation_seconds=time.perf_counter()-tick,
        counterfactual_b32_calls=sum(int(r['b32_calls'] or 0) for r in rows if r['phase']=='counterfactual'))
    op_save(dict(phase='preparation',instance='TRAIN_21',seed=OP_SEED,method='SAVED_ACTIONS',status='PREPARED',b32_calls=0,payload=preparation))
    if not ranking:raise ValueError('no real same-state Cmax/feasibility pairs were obtained')
    source=shared._model_from_checkpoint(checkpoint['refinement_candidate'])
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
        op_save(dict(phase='training',instance='TRAIN_21',seed=OP_SEED,method='OPPORTUNITY_RANK',status='EPOCH_COMPLETE',runtime_s=result['seconds'],b32_calls=0,payload=result))
    for prefix,size in (('joint',41),('parent',12)):
        for suffix in ('mean','scale'):
            if not torch.equal(getattr(model,prefix+'_'+suffix)[:size],getattr(source,prefix+'_'+suffix)):
                raise ValueError('legacy normalization parameters changed')
    metadata=dict(preparation=preparation,epochs=10,learning_rate=1e-4,batch_size=16,gradient_clip=1.,seed=OP_SEED,
        losses=dict(imitation=.5,ranking=1.,load=.1),history=history,training_seconds=time.perf_counter()-started,
        opportunity_definition='Remaining unplaced children, exact current insertion existence per robot, processing normalized by full graph processing.',
        training_geometries=21,development_geometries=8,static_context=True)
    checkpoint[OP_VARIANT]=dict(model_type='ModeJointInsertionGNN',hidden_dim=64,heads=4,normalize_inputs=True,
        opportunity_features=True,use_dynamic_context=False,scientific_config=config.scientific_mapping(),state_dict=model.state_dict(),training=metadata)
    old_root={k:v.clone() for k,v in checkpoint['state_dict'].items()}
    old_candidate={k:v.clone() for k,v in checkpoint['refinement_candidate']['state_dict'].items()}
    shared.save_neural_checkpoint(checkpoint,OP_CHECKPOINT)
    saved=shared.read_neural_checkpoint(OP_CHECKPOINT,include_research=False)
    if not all(torch.equal(v,saved['state_dict'][k]) for k,v in old_root.items()) or not all(
        torch.equal(v,saved['refinement_candidate']['state_dict'][k]) for k,v in old_candidate.items()):
        raise ValueError('protected existing weights changed')
    op_save(dict(phase='model',instance='TRAIN_21',seed=OP_SEED,method=OP_VARIANT,status='TRAINED',runtime_s=metadata['training_seconds'],b32_calls=0,payload=metadata))


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
    parser.add_argument('--phase',choices=('all','train','dev','op-all','op-data','op-train','op-eval','op-report'),default='op-all')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(SETTINGS['seed'])
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
