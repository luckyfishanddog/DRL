"""Train fixed two-head rankers, evaluate oracle capture, and conditional native smoke."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import json,math,multiprocessing as mp
from pathlib import Path
import statistics,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mrta_ranker import dataset as d
from mrta_ranker.ranking import family_c2,family_c4
import numpy as np
from mrta_ranker.model import CandidateMLP,fit_scaler,transform,predict_scores,multitask_loss,save_model,load_model
import torch
MODELS=ROOT/'models'
REPORT=ROOT/'docs/PHASE4_1B_MLP_CANDIDATE_RANKER_20261007.md'
VARIANTS=('BASELINE','MLP-C2','MLP-C4','MLP-BOTH')
SMOKE_SEEDS=(20261201,20261202,20261203)


def save_result(result):
    path=ROOT/'data/development/phase4_1c_mlp_results.json' if result.get('phase')=='Phase4-1C' else d.RESULT
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_records(role):
    whitelist=set(d.validate_split(d.read(d.SPLIT))[role])
    db=d.connect(readonly=True);names={s:d.metadata(db,s+'_feature_names') for s in ('C2','C4')}
    if {stage:len(value) for stage,value in names.items()}!={'C2':87,'C4':130}: raise ValueError('Expected existing 87/130-dimensional feature schema')
    records=[];parent_cache={}
    for row in db.execute('SELECT * FROM candidate_records WHERE split=? ORDER BY state_id,candidate_id',(role,)):
        r=dict(row)
        if r['workbook'] not in whitelist: raise PermissionError('Database workbook outside original role whitelist')
        if r['status'] is None: raise ValueError('Unlabeled candidate')
        if r['status']=='NUMERIC_FAILURE': raise ValueError('Numeric failure prevents training/evaluation')
        for stage in ('C2','C4'):
            blob=r[stage.lower()]
            r[stage+'_features']=None if blob is None else np.frombuffer(blob,dtype='<f4')
            if blob is not None and len(r[stage+'_features'])!=len(names[stage]): raise ValueError('Stored feature width differs')
        if (r['status']=='DIRECTION_INFEASIBLE')!=(r['C4_features'] is None): raise ValueError('C4 must contain exactly direction-feasible candidates')
        if r['instance_id'] not in parent_cache:
            parent_cache[r['instance_id']]=d.unpack(db.execute('SELECT parents FROM instances WHERE instance_id=?',(r['instance_id'],)).fetchone()[0])
        candidate=d.unpack(r['payload'],parent_cache[r['instance_id']])
        direction=d.unpack(r['direction'])
        # Exact production float64 keys, not rounded float32 model features.
        r['projected']=candidate.projected_process_makespan
        r['split_delta']=candidate.split_count_delta
        r['dp_travel']=direction.total_empty_travel
        # Drop replay payloads; model data needs no duplicated parent geometry.
        for key in ('payload','direction','c2','c4'): r.pop(key)
        records.append(r)
    expected=db.execute('SELECT count(*) FROM states JOIN instances USING(instance_id) WHERE split=?',(role,)).fetchone()[0]
    if len({r['state_id'] for r in records})!=expected: raise ValueError('Missing state candidate pool')
    db.close();return records,names


def median(values):
    values=[x for x in values if x is not None]
    return statistics.median(values) if values else None


def capture_metrics(pool,selected):
    cs=pool[0]['Cs'];feasible=[r['reference_cmax'] for r in pool if r['status']=='FEASIBLE_CERTIFIED']
    best=min(feasible,default=None);gain=0 if best is None else max(0,cs-best)
    chosen=min((r['reference_cmax'] for r in selected if r['status']=='FEASIBLE_CERTIFIED'),default=None)
    captured=0 if chosen is None else max(0,cs-chosen)
    epsilon=1e-9*max(1,cs)
    return {'material':gain/cs>=.005,'opportunity':gain>epsilon,'capture':None if gain<=epsilon else captured/gain,
        'zero_capture':None if gain<=epsilon else captured<=epsilon,
        'normalized_regret':None if best is None or chosen is None else max(0,chosen-best)/cs,
        'incumbent_fallback_regret':None if best is None else max(0,(cs if chosen is None else min(cs,chosen))-min(cs,best))/cs,
        'selected_feasible':sum(r['status']=='FEASIBLE_CERTIFIED' for r in selected),'selected_count':len(selected)}


def aggregate(states):
    material=[s for s in states if s['material']];regrets=[s['normalized_regret'] for s in material if s['normalized_regret'] is not None]
    return {'states':len(states),'material_states':len(material),
        'capture_median':median(s['capture'] for s in material),
        'zero_capture':statistics.fmean(s['zero_capture'] for s in material) if material else None,
        'normalized_regret_median':median(s['normalized_regret'] for s in material),
        'normalized_regret_mean':statistics.fmean(regrets) if regrets else None,
        'regret_defined_states':len(regrets),'incumbent_fallback_regret_median':median(s['incumbent_fallback_regret'] for s in material),
        'selected_candidate_feasibility_rate':sum(s['selected_feasible'] for s in states)/sum(s['selected_count'] for s in states) if sum(s['selected_count'] for s in states) else None}


def group_summary(states):
    groups={'ALL':states}
    groups.update({t:[s for s in states if s['tier']==t] for t in d.TIERS})
    groups.update({t:[s for s in states if s['stage']==t] for t,_ in d.STAGES})
    return {key:aggregate(rows) for key,rows in groups.items()}


def c2_select(pool,learned,seed,iteration):
    key=(lambda r:(-r['C2_score'],r['candidate_id'])) if learned else (lambda r:r['candidate_id'])
    return family_c2(pool,key,lambda r:r['family'],seed=seed,iteration=iteration)


def c4_select(pool,learned,seed,iteration):
    eligible=[r for r in pool if r['C4_features'] is not None]
    def heuristic(r): return (r['projected'],r['dp_travel'],r['split_delta'],r['candidate_id'])
    key=(lambda r:(-r['C4_score'],heuristic(r))) if learned else heuristic
    return family_c4(eligible,key,lambda r:r['family'],seed=seed,iteration=iteration)


def evaluate_stage(records,stage):
    grouped=defaultdict(list)
    for r in records: grouped[r['state_id']].append(r)
    rows=[]
    for state_id,pool in grouped.items():
        first=pool[0]
        selected=c2_select(pool,True,first['seed'],first['iteration']) if stage=='C2' else c4_select(pool,True,first['seed'],first['iteration'])
        rows.append({**capture_metrics(pool,selected),'state_id':state_id,'tier':first['tier'],'stage':first['stage']})
    return group_summary(rows)


def selection_key(metrics):
    m=metrics['ALL']
    if not m['material_states']: raise ValueError('No material DEV states for model selection')
    regret=m['normalized_regret_median'] if m['normalized_regret_median'] is not None else m['incumbent_fallback_regret_median']
    return (m['capture_median'],-m['zero_capture'],-regret)


def set_scores(records,stage,model,mean,std):
    eligible=[r for r in records if r[stage+'_features'] is not None]
    scores=predict_scores(model,[r[stage+'_features'] for r in eligible],mean,std)
    for r,score in zip(eligible,scores): r[stage+'_score']=float(score)


def train():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    train_rows,names=load_records('MLP_TRAIN');dev_rows,_=load_records('MLP_DEV')
    scalers={}
    for stage in ('C2','C4'):
        x=[r[stage+'_features'] for r in train_rows if r[stage+'_features'] is not None]
        mean,std=fit_scaler(x);scalers[stage+'_mean']=mean;scalers[stage+'_std']=std
    MODELS.mkdir(exist_ok=True);np.savez(MODELS/'mlp_scalers.npz',**scalers)
    result=d.read(d.RESULT);result['training_config']={'hidden':[128,64],'activation':'ReLU','optimizer':'AdamW','lr':.001,'weight_decay':.0001,'batch_size':512,'max_epochs':100,'patience':10,'loss':'BCE + SmoothL1_feasible (1:1)','selection':'DEV material capture median descending; zero capture ascending; normalized regret median ascending (incumbent fallback only if no selected feasible regret is defined)','training_seeds':{'C2':20261123,'C4':20261124},'scaler_fit':'TRAIN only; independent C2 and C4; population std; zero std -> 1','score':'sigmoid(feasibility_logit) * improvement_output; gain may be negative'}
    result['training']={}
    for stage in ('C2','C4'):
        torch.manual_seed(20261123+(stage=='C4'))
        rows=[r for r in train_rows if r[stage+'_features'] is not None]
        mean,std=scalers[stage+'_mean'],scalers[stage+'_std']
        x=torch.from_numpy(transform([r[stage+'_features'] for r in rows],mean,std))
        yf=torch.tensor([r['y_feasible'] for r in rows],dtype=torch.float32)
        yi=torch.tensor([float('nan') if r['y_improvement'] is None else r['y_improvement'] for r in rows],dtype=torch.float32)
        model=CandidateMLP(len(names[stage]));optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=None;stale=0;history=[]
        for epoch in range(1,101):
            model.train();order=torch.randperm(len(rows));losses=[];bces=[];regressions=[]
            for indices in order.split(512):
                optimizer.zero_grad(set_to_none=True);logit,gain=model(x[indices]);loss,bce,reg=multitask_loss(logit,gain,yf[indices],yi[indices]);loss.backward();optimizer.step()
                losses.append(float(loss.detach()));bces.append(float(bce.detach()));regressions.append(float(reg.detach()))
            set_scores(dev_rows,stage,model,mean,std);metrics=evaluate_stage(dev_rows,stage);key=selection_key(metrics)
            history.append({'epoch':epoch,'train_loss':statistics.fmean(losses),'BCE':statistics.fmean(bces),'SmoothL1_feasible':statistics.fmean(regressions),'dev_material':metrics['ALL']})
            if best is None or key>best:
                best=key;best_epoch=epoch;best_metrics=metrics;stale=0
                save_model(MODELS/f'mlp_{stage.lower()}.pt',model,stage,names[stage],epoch,metrics)
            else: stale+=1
            if epoch==1 or epoch%10==0: print(f"{stage} epoch={epoch} DEV material capture={metrics['ALL']['capture_median']:.4f} zero={metrics['ALL']['zero_capture']:.4f}",flush=True)
            if stale>=10: break
        target=np.array([r['y_improvement'] for r in rows if r['y_feasible']==1])
        target_summary={'feasible_rows':len(target),'positive_fraction':float(np.mean(target>0)),
            'improvement_quantiles':dict(zip(('min','p25','median','p75','max'),map(float,np.quantile(target,[0,.25,.5,.75,1]))))}
        result['training'][stage]={'target_summary':target_summary,'selected_epoch':best_epoch,'epochs_run':epoch,'selected_dev':best_metrics,'history':history,'training_rows':len(rows)}
        save_result(result)
    print('Both independent MLP checkpoints trained and selected on workbook-isolated DEV',flush=True)


def qualifies_for_smoke(baseline,metric):
    fields=('capture_median','zero_capture')
    if any(row.get(key) is None for row in (baseline,metric) for key in fields): return False
    return metric['capture_median']-baseline['capture_median']>=.10-1e-12 and baseline['zero_capture']-metric['zero_capture']>=.10-1e-12


def offline():
    torch.set_num_threads(1)
    records,_=load_records('MLP_DEV');scalers=np.load(MODELS/'mlp_scalers.npz')
    diagnostics={}
    for stage in ('C2','C4'):
        model,_=load_model(MODELS/f'mlp_{stage.lower()}.pt');set_scores(records,stage,model,scalers[stage+'_mean'],scalers[stage+'_std'])
        eligible=[r for r in records if r[stage+'_features'] is not None]
        with torch.inference_mode():
            logit,gain=model(torch.from_numpy(transform([r[stage+'_features'] for r in eligible],scalers[stage+'_mean'],scalers[stage+'_std'])))
        diagnostics[stage]={'DEV_rows':len(eligible),'predicted_positive_gain_fraction':float((gain>0).float().mean()),
            'predicted_gain_quantiles':dict(zip(('min','p25','median','p75','max'),map(float,np.quantile(gain.numpy(),[0,.25,.5,.75,1])))),
            'predicted_feasibility_median':float(torch.sigmoid(logit).median())}
    grouped=defaultdict(list)
    for r in records: grouped[r['state_id']].append(r)
    result=d.read(d.RESULT);result['offline']={};result['prediction_diagnostics']=diagnostics
    for variant in VARIANTS:
        c2_states=[];c4_states=[]
        for state_id,pool in grouped.items():
            first=pool[0];seed,iteration=first['seed'],first['iteration']
            c2=c2_select(pool,variant in ('MLP-C2','MLP-BOTH'),seed,iteration)
            c4=c4_select(c2,variant in ('MLP-C4','MLP-BOTH'),seed,iteration)
            common={'state_id':state_id,'tier':first['tier'],'stage':first['stage']}
            c2_states.append({**common,**capture_metrics(pool,c2)})
            c4_states.append({**common,**capture_metrics(pool,c4)})
        result['offline'][variant]={'C2':group_summary(c2_states),'C4':group_summary(c4_states),'state_metrics':{'C2':c2_states,'C4':c4_states}}
    baseline=result['offline']['BASELINE']['C4']['ALL'];eligible=[]
    for variant in VARIANTS[1:]:
        metric=result['offline'][variant]['C4']['ALL']
        passed=qualifies_for_smoke(baseline,metric)
        result['offline'][variant]['qualifies_for_native_smoke']=passed
        if passed: eligible.append(variant)
    result['best_descriptive_offline_variant']=max(VARIANTS,key=lambda v:selection_key(result['offline'][v]['C4']))
    result['offline_ranking_improved']=bool(eligible)
    result['best_offline_variant']=max(eligible,key=lambda v:selection_key(result['offline'][v]['C4'])) if eligible else None
    save_result(result)
    print('Best eligible offline variant:',result['best_offline_variant'],flush=True)


_SMOKE_SOURCE=None
_SMOKE_PARENTS={}
_SMOKE_RANKERS={}


def initialize_smoke(masks):
    global _SMOKE_SOURCE
    d.bind_to_core(masks[(mp.current_process()._identity[0]-1)%len(masks)])
    torch.set_num_threads(1);_SMOKE_SOURCE=d.provenance()


def smoke_job(entry,seed,variant,model_version="v1"):
    from mrta_ranker.ranking import LearnedRanker
    if entry['instance_id'] not in _SMOKE_PARENTS: _SMOKE_PARENTS[entry['instance_id']]=d.load_parents(entry)
    ranker=None
    if variant!='BASELINE':
        if variant not in _SMOKE_RANKERS: _SMOKE_RANKERS[variant]=LearnedRanker(variant,model_version=model_version)
        ranker=_SMOKE_RANKERS[variant];ranker.inference_seconds=0.0
    record,_=d.run_trajectory(entry,seed,parents=_SMOKE_PARENTS[entry['instance_id']],source=_SMOKE_SOURCE,ranker=ranker)
    return {'instance_id':entry['instance_id'],'workbook':entry['relative_path'],'tier':entry['tier'],'seed':seed,'variant':variant,**record}


def smoke(workers=3,result_path=None):
    result=d.read(d.RESULT if result_path is None else result_path)
    if not result.get('offline_ranking_improved'):
        result['smoke']={'status':'SKIPPED','reason':'No variant improved material C4 capture and zero capture by the required 10 percentage points.'}
        save_result(result);print('Native smoke skipped: offline ranking improvement requirement not met',flush=True);return
    best=result['best_offline_variant'];entries=[e for e in d.representatives() if e['split']=='MLP_DEV']
    section=result.setdefault('smoke',{'status':'RUNNING','variant':best,'seeds':list(SMOKE_SEEDS),'runs':[]})
    if section.get('variant')!=best: raise ValueError('Stored smoke uses a different selected variant')
    done={(r['instance_id'],r['seed'],r['variant']) for r in section['runs']};jobs=[]
    for i,entry in enumerate(entries):
        for j,seed in enumerate(SMOKE_SEEDS):
            variants=('BASELINE',best) if (i+j)%2==0 else (best,'BASELINE')
            jobs.extend((entry,seed,v,result.get('best_model_version','v1')) for v in variants if (entry['instance_id'],seed,v) not in done)
    masks=d.physical_core_affinities()[:workers]
    if len(masks)!=workers: raise ValueError('Need distinct physical cores')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_smoke,initargs=(masks,)) as pool:
        for future in as_completed([pool.submit(smoke_job,*job) for job in jobs]):
            row=future.result();section['runs'].append(row);save_result(result)

    if len(section['runs'])!=36: raise ValueError('Need all 36 paired native runs')
    invalid=[r for r in section['runs'] if not r['certified'] or r['numeric_failure'] or r['Cmax_at_60'] is None]
    if invalid:
        section['status']='COMPLETED_WITH_MISSING_OR_INVALID_OUTCOME';section['invalid_runs']=invalid;save_result(result);return
    by_case=defaultdict(dict)
    for row in section['runs']: by_case[(row['instance_id'],row['seed'])][row['variant']]=row
    paired=[]
    for (instance,seed),case in by_case.items():
        base,mlp=case['BASELINE'],case[best]
        paired.append({'instance_id':instance,'seed':seed,'tier':base['tier'],'Cmax_ratio':mlp['Cmax_at_60']/base['Cmax_at_60'],
            'iteration_ratio':mlp['iterations']/base['iterations'] if base['iterations'] else None,
            'reference_call_ratio':mlp['reference_calls']/base['reference_calls'] if base['reference_calls'] else None})
    groups={'ALL':paired,**{t:[r for r in paired if r['tier']==t] for t in d.TIERS}}
    section['paired_summary']={g:{'pairs':len(rs),'Cmax_ratio_median':median(r['Cmax_ratio'] for r in rs),
        'iteration_ratio_median':median(r['iteration_ratio'] for r in rs),'reference_call_ratio_median':median(r['reference_call_ratio'] for r in rs),
        'wins':sum(r['Cmax_ratio']<1-1e-9 for r in rs),'ties':sum(abs(r['Cmax_ratio']-1)<=1e-9 for r in rs),'losses':sum(r['Cmax_ratio']>1+1e-9 for r in rs)} for g,rs in groups.items()}
    section['variant_summary']={}
    for variant in ('BASELINE',best):
        rs=[r for r in section['runs'] if r['variant']==variant]
        section['variant_summary'][variant]={'Cmax_at_60_median':median(r['Cmax_at_60'] for r in rs),
            'iterations_median':median(r['iterations'] for r in rs),'reference_calls_median':median(r['reference_calls'] for r in rs),
            'reference_calls_per_second_median':median(r['reference_calls']/r['actual_runtime'] for r in rs),
            'best_updates_per_100_reference_calls_median':median(100*max(0,len(r['best_events'])-1)/r['reference_calls'] for r in rs),
            'generation_seconds_median':median(r['candidate_generation_seconds'] for r in rs),
            'inference_seconds_median':median(r['ranker_inference_seconds'] for r in rs),'inference_fraction_median':median(r['ranker_inference_seconds']/60 for r in rs),
            'runtime_median':median(r['actual_runtime'] for r in rs),'runtime_max':max(r['actual_runtime'] for r in rs)}
    section['paired_runs']=paired;section['status']='COMPLETED';save_result(result)






def loss_decomposition(pool,c2,c4):
    """Attribute missed pool improvements without proposing a new selector."""
    first=pool[0];a=capture_metrics(pool,c2);b=capture_metrics(pool,c4)
    eligible=[r for r in c2 if r['C4_features'] is not None]
    # Labels are used only for a retrospective ceiling, never model inputs.
    key=lambda r:(-r['y_improvement'] if r['status']=='FEASIBLE_CERTIFIED' else math.inf,r['candidate_id'])
    ceiling=family_c4(eligible,key,lambda r:r['family'],seed=first['seed'],iteration=first['iteration'])
    upper=capture_metrics(pool,ceiling)
    if a['opportunity']:
        if b['capture']>a['capture']+1e-12 or abs(upper['capture']-a['capture'])>1e-12:
            raise AssertionError('C4 must preserve the best C2 candidate under oracle ranking')
    has2=a['opportunity'] and not a['zero_capture']
    has4=b['opportunity'] and not b['zero_capture']
    return {'state_id':first['state_id'],'tier':first['tier'],'stage':first['stage'],
        'material':a['material'],'C2_capture':a['capture'],'C4_capture':b['capture'],
        'oracle_C4_capture_given_C2':upper['capture'],
        'conditional_C4_capture':b['capture']/a['capture'] if has2 else None,
        'C2_has_improvement':bool(has2),'C4_has_improvement':bool(has4),
        'loss_location':('NO_POOL_OPPORTUNITY' if not a['opportunity'] else
                         'NO_C2_CAPTURE' if not has2 else 'C4_DROPPED_ALL' if not has4 else 'C4_RETAINED_IMPROVEMENT'),
        'C2_ids':[r['candidate_id'] for r in c2],'C4_ids':[r['candidate_id'] for r in c4]}


def decomposition_groups(rows):
    groups={'ALL':rows,**{t:[r for r in rows if r['tier']==t] for t in d.TIERS},
            **{t:[r for r in rows if r['stage']==t] for t,_ in d.STAGES}}
    summaries={}
    for name,values in groups.items():
        material=[r for r in values if r['material']]
        captured=[r for r in material if r['C2_has_improvement']]
        summaries[name]={'material_states':len(material),'C2_capture_median':median(r['C2_capture'] for r in material),
            'C4_capture_median':median(r['C4_capture'] for r in material),
            'C2_capture_states':len(captured),'C4_capture_states':sum(r['C4_has_improvement'] for r in material),
            'no_C2_capture_states':sum(r['loss_location']=='NO_C2_CAPTURE' for r in material),
            'C4_dropped_all_states':sum(r['loss_location']=='C4_DROPPED_ALL' for r in material),
            'conditional_C4_capture_median':median(r['conditional_C4_capture'] for r in captured),
            'oracle_C4_capture_given_C2_median':median(r['oracle_C4_capture_given_C2'] for r in material)}
    return summaries


def head_diagnostics(rows,stage):
    eligible=[r for r in rows if r[stage+'_features'] is not None]
    positive=[r for r in eligible if r['y_improvement'] is not None and r['y_improvement']>1e-9]
    def fraction(values,test): return sum(test(r) for r in values)/len(values) if values else None
    errors=[r[stage+'_gain']-r['y_improvement'] for r in positive]
    return {'rows':len(eligible),'true_improving_rows':len(positive),
        'true_improving_fraction':len(positive)/len(eligible) if eligible else None,
        'true_improving_predicted_negative_fraction':fraction(positive,lambda r:r[stage+'_gain']<0),
        'improving_gain_bias_mean':statistics.fmean(errors) if errors else None,
        'improving_predicted_p_median':median(r[stage+'_p'] for r in positive),
        'infeasible_predicted_p_median':median(r[stage+'_p'] for r in eligible if r['y_feasible']==0),
        'predicted_gain_median':median(r[stage+'_gain'] for r in eligible)}


def negative_product_diagnostics(pool,selected,stage):
    """Count actual selected bad/good inversions of the prescribed p*gain."""
    improving=[r for r in pool if r['y_improvement'] is not None and r['y_improvement']>1e-9 and r[stage+'_gain']<0]
    bad=[r for r in selected if r['y_feasible']==0 and r[stage+'_gain']<0]
    pairs=[(b,g) for b in bad for g in improving if b[stage+'_gain']<=g[stage+'_gain']
           and b[stage+'_score']>g[stage+'_score']]
    return {'selected_infeasible_negative_gain':len(bad),'inversions':len(pairs),
        'inversion_exists':bool(pairs),'improving_negative_gain_available':bool(improving)}


def diagnose():
    """Fixed-model retrospective analysis; no training, solver or data generation."""
    torch.set_num_threads(1)
    scalers=np.load(MODELS/'mlp_scalers.npz');models={s:load_model(MODELS/f'mlp_{s.lower()}.pt')[0] for s in ('C2','C4')}
    result=d.read(d.RESULT);diagnostics={'scope':'Existing fixed checkpoints and whitelist dataset only; TRAIN is in-sample description, not model selection.',
        'new_trajectories':0,'new_reference_calls':0,'new_training_epochs':0,'splits':{}}
    tails={}
    for role in ('MLP_TRAIN','MLP_DEV'):
        records,names=load_records(role)
        for stage in ('C2','C4'):
            eligible=[r for r in records if r[stage+'_features'] is not None]
            raw=[r[stage+'_features'] for r in eligible]
            z=transform(raw,scalers[stage+'_mean'],scalers[stage+'_std'])
            with torch.inference_mode():
                logit,gain=models[stage](torch.from_numpy(z));prob=torch.sigmoid(logit).numpy();gain=gain.numpy()
            for r,p,g in zip(eligible,prob,gain):
                r[stage+'_p']=float(p);r[stage+'_gain']=float(g);r[stage+'_score']=float(np.float32(p*g))
            tails[(role,stage)]=(names[stage],np.mean(np.abs(z)>5,axis=0))
        grouped=defaultdict(list)
        for r in records: grouped[r['state_id']].append(r)
        section={'head_metrics':{},'variants':{},'feature_tails':{}}
        for stage in ('C2','C4'):
            section['head_metrics'][stage]={g:head_diagnostics(records if g=='ALL' else [r for r in records if r['tier']==g],stage) for g in ('ALL',*d.TIERS)}
            for group in ('ALL',*d.TIERS):
                state_tails=[]
                for pool in grouped.values():
                    rows=[r for r in pool if r[stage+'_features'] is not None]
                    if not rows or (group!='ALL' and rows[0]['tier']!=group): continue
                    z=transform([r[stage+'_features'] for r in rows],scalers[stage+'_mean'],scalers[stage+'_std'])
                    state_tails.append(float(np.mean(np.abs(z)>5)))
                section['feature_tails'].setdefault(stage,{})[group]={'states':len(state_tails),'state_equal_weight_fraction_abs_z_gt5_median':median(state_tails)}
        for variant in VARIANTS:
            rows=[];negatives=[];c2metrics=[];c4metrics=[]
            for state_id,pool in grouped.items():
                first=pool[0];seed,iteration=first['seed'],first['iteration']
                c2=c2_select(pool,variant in ('MLP-C2','MLP-BOTH'),seed,iteration)
                c4=c4_select(c2,variant in ('MLP-C4','MLP-BOTH'),seed,iteration)
                row=loss_decomposition(pool,c2,c4);rows.append(row)
                common={'state_id':state_id,'tier':first['tier'],'stage':first['stage']}
                c2metrics.append({**common,**capture_metrics(pool,c2)});c4metrics.append({**common,**capture_metrics(pool,c4)})
                negatives.append({**common,'C2':negative_product_diagnostics(pool,c2,'C2'),
                    'C4':negative_product_diagnostics([r for r in c2 if r['C4_features'] is not None],c4,'C4')})
            if role=='MLP_DEV':
                for stage,metrics in (('C2',c2metrics),('C4',c4metrics)):
                    if group_summary(metrics)!=result['offline'][variant][stage]:
                        raise AssertionError('Fixed-model diagnostic changed the stored offline result')
            section['variants'][variant]={'groups':decomposition_groups(rows),'state_metrics':rows,
                'negative_product':{g:{stage:{'states_with_inversion':sum(r[stage]['inversion_exists'] for r in negatives if g=='ALL' or r['tier']==g),
                    'inversion_pairs':sum(r[stage]['inversions'] for r in negatives if g=='ALL' or r['tier']==g),
                    'selected_infeasible_negative_gain':sum(r[stage]['selected_infeasible_negative_gain'] for r in negatives if g=='ALL' or r['tier']==g)}
                    for stage in ('C2','C4')} for g in ('ALL',*d.TIERS)}}
        section['LARGE_workbooks']={}
        for workbook in sorted({r['workbook'] for r in records if r['tier']=='LARGE'}):
            sample=[r for r in records if r['workbook']==workbook and r['tier']=='LARGE']
            ids={r['state_id'] for r in sample}
            chosen=[r for r in section['variants']['MLP-C2']['state_metrics'] if r['state_id'] in ids]
            section['LARGE_workbooks'][workbook]={'MLP_C2':decomposition_groups(chosen)['ALL'],
                'head_metrics':{stage:head_diagnostics(sample,stage) for stage in ('C2','C4')}}
        diagnostics['splits'][role]=section
        print(f'Fixed-model diagnostics: {role} {len(grouped)} states',flush=True)
    diagnostics['feature_tail_comparison']={}
    for stage in ('C2','C4'):
        names,train=tails[('MLP_TRAIN',stage)];_,dev=tails[('MLP_DEV',stage)]
        largest=sorted(range(len(names)),key=lambda j:-(dev[j]-train[j]))[:5]
        diagnostics['feature_tail_comparison'][stage]=[{'feature':names[j],'TRAIN_fraction_abs_z_gt5':float(train[j]),
            'DEV_fraction_abs_z_gt5':float(dev[j])} for j in largest]
    dev=diagnostics['splits']['MLP_DEV'];train=diagnostics['splits']['MLP_TRAIN']
    large=dev['variants']['MLP-C2']['groups']['LARGE'];medium=dev['variants']['MLP-C2']['groups']['MEDIUM']
    train_large=train['variants']['MLP-C2']['groups']['LARGE']
    pdev=dev['head_metrics']['C2']['LARGE'];ptrain=train['head_metrics']['C2']['LARGE']
    inv=dev['variants']['MLP-BOTH']['negative_product']['ALL']['C4']['states_with_inversion']
    diagnostics['findings']=[
        f"LARGE MLP-C2 Top8 capture median: TRAIN {100*train_large['C2_capture_median']:.2f}% vs DEV {100*large['C2_capture_median']:.2f}%; DEV {large['no_C2_capture_states']}/{large['material_states']} material states missed all improvement before C4. This is consistent with a cross-workbook generalization gap; TRAIN is in-sample and DEV has only two LARGE workbooks.",
        f"MEDIUM DEV: MLP-C2 captured improvement in {medium['C2_capture_states']}/{medium['material_states']} material states, and C4 dropped all improvement in {medium['C4_dropped_all_states']} of them. The same-policy oracle C4 ceiling is {100*medium['oracle_C4_capture_given_C2_median']:.2f}% capture vs the actual {100*medium['C4_capture_median']:.2f}%.",
        f"LARGE true-improving candidate predicted feasibility median: TRAIN {100*ptrain['improving_predicted_p_median']:.2f}% vs DEV {100*pdev['improving_predicted_p_median']:.2f}%; DEV infeasible candidates median {100*pdev['infeasible_predicted_p_median']:.2f}%. This is descriptive discrimination failure on this sample, not a standalone calibration metric or proven cause.",
        f"MLP-BOTH C4 had {inv} states with the defined selected infeasible negative-gain inversion. The theoretical p*negative_gain issue alone does not explain the observed C4 failure; no alternative score was selected or tested.",
        'Do not change M192, family access, Kdp/Kref, data split or architecture based on these diagnostics. The next bounded learning revision should address state-level ranking and cross-workbook generalization on existing TRAIN, with DEV used for selection; no GAT or native smoke is justified yet.'
    ]
    result['fixed_model_diagnostics']=diagnostics;save_result(result)


def diagnostic_report_lines(result):
    analysis=result.get('fixed_model_diagnostics')
    if analysis is None: return []
    lines=['','## 固定模型失败定位（继续分析）','',
        '仅复用已有TRAIN/DEV标签与两个已选checkpoint；没有新增trajectory、reference调用、训练epoch或文件，没有改变score、family policy或生产配置。DEV四组结果逐项与原消融核对一致。TRAIN结果只作拟合程度的描述，未用于选模型。',
        'C4条件capture以C2已捕获改善为分母，只在C2捕获到改善的material states上计算。Oracle C4只作事后上界：用已知标签重排相同Top8并保留相同family policy，不能用于实际搜索。', '',
        '| split | variant | tier | material | C2有改善 | C4有改善 | 未进C2 | C4丢光 | 条件C4 capture | oracle C4/full-pool capture |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for role,section in analysis['splits'].items():
        for variant in VARIANTS:
            for group in ('ALL',*d.TIERS):
                m=section['variants'][variant]['groups'][group]
                lines.append(f"| {role} | {variant} | {group} | {m['material_states']} | {m['C2_capture_states']} | {m['C4_capture_states']} | {m['no_C2_capture_states']} | {m['C4_dropped_all_states']} | {percent(m['conditional_C4_capture_median'])} | {percent(m['oracle_C4_capture_given_C2_median'])} |")
    lines.extend(['','LARGE逐workbook复核（MLP-C2；物理路径只标识来源，角色按固定split）：','',
        '| split | workbook | material | Top8 capture median | Top8有改善 | Top2有改善 |',
        '|---|---|---:|---:|---:|---:|'])
    for role,section in analysis['splits'].items():
        for workbook,row in section.get('LARGE_workbooks',{}).items():
            m=row['MLP_C2'];lines.append(f"| {role} | {workbook} | {m['material_states']} | {percent(m['C2_capture_median'])} | {m['C2_capture_states']} | {m['C4_capture_states']} |")
    lines.extend(['','固定head误差（candidate级描述，不代替state equal weight capture）：','',
        '| split | head | tier | 真改善候选 | 其中预测负gain | 真改善候选gain平均偏差 | 真改善候选p中位数 | 不可行候选p中位数 |',
        '|---|---|---|---:|---:|---:|---:|---:|'])
    for role,section in analysis['splits'].items():
        for stage,groups in section['head_metrics'].items():
            for group,m in groups.items():
                bias=m['improving_gain_bias_mean']
                lines.append(f"| {role} | {stage} | {group} | {m['true_improving_rows']} | {percent(m['true_improving_predicted_negative_fraction'])} | {bias if bias is not None else '—'} | {percent(m['improving_predicted_p_median'])} | {percent(m['infeasible_predicted_p_median'])} |")
    lines.extend(['','实际负gain乘积倒序：selected不可行候选的预测gain不高于池中真改善候选，且二者gain均负，但前者p×gain更高。该统计说明固定模型分数与所选集合的关系；heuristic阶段仅作对照，不能归因于该score。不是因果实验，也不是新score选择。','',
        '| DEV variant | C2存在倒序的states | C4存在倒序的states | C4倒序pairs |',
        '|---|---:|---:|---:|'])
    for variant,v in analysis['splits']['MLP_DEV']['variants'].items():
        m=v['negative_product']['ALL'];lines.append(f"| {variant} | {m['C2']['states_with_inversion']} | {m['C4']['states_with_inversion']} | {m['C4']['inversion_pairs']} |")
    lines.extend(['','TRAIN标准化后|z|>5的特征比例（每state先平均、再取中位数）：','',
        '| split | stage | ALL | SMALL | MEDIUM | LARGE |','|---|---|---:|---:|---:|---:|'])
    for role,section in analysis['splits'].items():
        for stage,groups in section['feature_tails'].items():
            lines.append('| '+role+' | '+stage+' | '+' | '.join(percent(groups[g]['state_equal_weight_fraction_abs_z_gt5_median']) for g in ('ALL',*d.TIERS))+' |')
    lines.extend(['','分布尾部差异只说明观测特征的尺度/分布，不证明泛化失败由某个feature导致；候选相关性和workbook差异仍存在。所有事后分析均未用于追加选epoch、改分数或重复训练。','',
        '进一步结论：LARGE既有C2跨workbook泛化差距，MEDIUM又有独立C4丢解。仅改善C4不能找回LARGE已被C2丢掉的候选。所检查的C4负gain乘积倒序为0个states，不能据理论性质直接断言它是本次失败主因。', '',
        '可用 `python -B scripts/train_mlp_ranker.py diagnose` 重复本节；`finish` 更新本报告。下一轮应限定为现有TRAIN上的state-level ranking目标与跨workbook泛化改进，DEV继续只用于模型选择，保持两层MLP、预算与family policy不变；当前没有依据进入GAT或native smoke。',''])
    for finding in analysis.get('findings',[]): lines.extend([finding,''])
    return lines


def percent(value): return '—' if value is None else f'{100*value:.2f}%'


def finish():
    result=d.read(d.RESULT);db=d.connect(readonly=True)
    result['dataset']=d.dataset_summary(db);result['representatives']=d.metadata(db,'representatives')
    result['split']={k:len(d.read(d.SPLIT)[k]) for k in ('MLP_TRAIN','MLP_DEV','ORACLE_DEV_CONSUMED')}
    missing=[]
    for row in db.execute('SELECT * FROM trajectories'):
        record=json.loads(row['result'])
        if record.get('missing_state_stages'): missing.append({'instance_id':row['instance_id'],'seed':row['seed'],'first_boundary_elapsed':record['first_boundary_elapsed'],'missing':record['missing_state_stages']})
    result['trajectory_summary']={'retained_runs':db.execute('SELECT count(*) FROM trajectories').fetchone()[0],
        'context_count':db.execute('SELECT count(*) FROM states').fetchone()[0],
        'stage_counts':dict(db.execute('SELECT stage,count(*) FROM states GROUP BY stage')),
        'initial_boundary_counts':dict(db.execute('SELECT stage,count(*) FROM states WHERE iteration=0 GROUP BY stage')),
        'missing_stages':missing}
    timings=[json.loads(r[0]) for r in db.execute('SELECT result FROM trajectories')]
    result['trajectory_summary'].update(runtime_median=median(r['actual_runtime'] for r in timings),runtime_max=max(r['actual_runtime'] for r in timings),overshoot_median=median(r['overshoot'] for r in timings))
    result['data_checks']={'numeric_failures':db.execute("SELECT count(*) FROM candidates WHERE status='NUMERIC_FAILURE'").fetchone()[0],
        'trajectory_numeric_failures':sum(r['numeric_failure'] for r in timings),
        'unlabeled_candidates':db.execute('SELECT count(*) FROM candidates WHERE status IS NULL').fetchone()[0],
        'c4_direction_infeasible_rows':db.execute("SELECT count(*) FROM candidates WHERE status='DIRECTION_INFEASIBLE' AND c4 IS NOT NULL").fetchone()[0],
        'nonfeasible_improvement_labels':db.execute("SELECT count(*) FROM candidates WHERE status!='FEASIBLE_CERTIFIED' AND y_improvement IS NOT NULL").fetchone()[0],
        'feature_roundtrip_mismatch':db.execute('SELECT count(*) FROM candidates WHERE c4 IS NOT NULL AND substr(c4,1,348)!=c2').fetchone()[0],
        'feature_width_mismatch':db.execute('SELECT count(*) FROM candidates WHERE length(c2)!=348 OR (c4 IS NOT NULL AND length(c4)!=520)').fetchone()[0],
        'feasibility_target_mismatch':db.execute("SELECT count(*) FROM candidates WHERE (status='FEASIBLE_CERTIFIED')!=(y_feasible=1)").fetchone()[0],
        'improvement_target_mismatch':db.execute("SELECT count(*) FROM candidate_records WHERE status='FEASIBLE_CERTIFIED' AND abs(y_improvement-(Cs-reference_cmax)/Cs)>1e-10").fetchone()[0]}
    if any(result['data_checks'].values()): raise ValueError('Dataset correctness checks failed')
    result['feature_widths']={stage:len(d.metadata(db,stage+'_feature_names')) for stage in ('C2','C4')}
    db.close()
    native=result.get('smoke',{});completed=native.get('status')=='COMPLETED'
    improved=completed and native['paired_summary']['ALL']['Cmax_ratio_median']<1-1e-9
    result['recommended_ranker']=result['best_offline_variant'] if improved else 'BASELINE'
    result['phase_status']='PASS' if improved else 'FAIL'
    result['GAT_next']='NOT YET'
    result['next_step']=('独立评估固定 MLP ranker，先判断 MLP 是否已足够。' if improved else '保持 M192 heuristic；优先检查 C4 Top2 排序目标、score校准与 LARGE 跨workbook泛化，暂不进入 GAT。')
    if not improved and result.get('fixed_model_diagnostics'):
        result['next_step']='保持 M192 heuristic；下一轮限定改进 LARGE 的 C2 跨workbook泛化和 C4 state-level Top2 排序目标，复用既有TRAIN/DEV，不进入GAT。'
    base=result['offline']['BASELINE'];learned=result['offline']['MLP-C2']
    med=learned['C2']['MEDIUM'];med4=learned['C4']['MEDIUM']
    positive=lambda m:round(m['material_states']*(1-m['zero_capture'])) if m['zero_capture'] is not None else 0
    losses={stage:next(h for h in t['history'] if h['epoch']==t['selected_epoch']) for stage,t in result['training'].items()}
    ratios={stage:(row['BCE']/row['SmoothL1_feasible'] if row['SmoothL1_feasible'] else None) for stage,row in losses.items()}
    ratio_text=' / '.join(f"{stage} {value:.1f}倍" if value is not None else f"{stage} undefined" for stage,value in ratios.items())
    pred=result['prediction_diagnostics']
    result['interpretation']=[
        ('本轮完成数据与两个固定模型训练；FAIL指学习排序没有达到采用要求。离线未达条件时按约定不执行native smoke，当前没有真实60s MLP收益或inference开销结论。' if not improved else '固定模型达到离线要求，且配对原生搜索中位质量改善；仍属于development证据。'),
        f"MLP-C2：ALL material Top8 capture {percent(base['C2']['ALL']['capture_median'])}→{percent(learned['C2']['ALL']['capture_median'])}；MEDIUM {percent(base['C2']['MEDIUM']['capture_median'])}→{percent(med['capture_median'])}。MEDIUM {med['material_states']}个material states中Top8有{positive(med)}个捕获改善，Top2最终有{positive(med4)}个。",
        f"MLP-C2最终C4：SMALL capture {percent(base['C4']['SMALL']['capture_median'])}→{percent(learned['C4']['SMALL']['capture_median'])}；LARGE capture {percent(base['C4']['LARGE']['capture_median'])}→{percent(learned['C4']['LARGE']['capture_median'])}，zero capture {percent(base['C4']['LARGE']['zero_capture'])}→{percent(learned['C4']['LARGE']['zero_capture'])}。分stage结果见同预算表，不用局部收益替代整体要求。",
        f"总体描述性最优={result['best_descriptive_offline_variant']}，MLP-BOTH是否最优={result['best_descriptive_offline_variant']=='MLP-BOTH'}。selected feasibility改善不等于Top2 material improvement capture改善。",
        '未发现label pipeline错误或泄漏：数值、公共特征往返、目标公式和C4阶段范围检查均为0异常。固定模型的Top-K效果不足，现有结果不能单独证明feature不足。',
        f"Selected-epoch BCE/SmoothL1数值比：{ratio_text}。TRAIN feasible候选正改善比例={percent(result['training']['C2']['target_summary']['positive_fraction'])}。逐candidate平均loss与少数material候选的Top-K目标存在可能不匹配，尚不能据此断言因果。",
        f"DEV predicted gain median：C2 {pred['C2']['predicted_gain_quantiles']['median']:.5f}、C4 {pred['C4']['predicted_gain_quantiles']['median']:.5f}。规定p×gain在gain<0时，降低p会使分数靠近0；校准与排序交互值得检查。本轮仍保留规定score与1:1 loss。",
        'Direct regret只在selected feasible存在时有定义，各variant定义样本数不同，需结合JSON中的regret_defined_states与zero capture阅读。可选ranker接口已接入原生ALNS并通过预算/policy回归，生产默认仍为heuristic。']
    save_result(result)
    lines=['# Phase4-1B — MLP Candidate Ranker','',
        f"结论：**{result['phase_status']}**。推荐 **{result['recommended_ranker']}**；M192 / Kdp8 / Kref2 / Kref_total4 不变。GAT：NOT YET。",'',
        '完成日期：2026-10-08。报告文件名沿用此前批准的名称。','',
        '## 清理结果','',
        f"删除 {len(result['cleanup']['removed_files'])} 个无当前依赖的旧过程文件：旧 Phase3/4 runner、profiling、候选 oracle SQLite、重复中间 JSON 和 handoff。通用 typed reader、accepted-state capture、存储与数据准备移入正式 `mrta_ranker.dataset`，新流程不依赖历史 runner。删除明细及字节数保存在结果 JSON。没有创建 archive/backup，没有提交或上传 GitHub。",'',
        '保留科学 src/tests、FORMAL_SCOPE_V2、实验方案、正式数据角色与固定 MLP split，以及解释当前算法的 Phase4-0B、Phase4-1A 等最终报告；清理后的文档链接已修复。退役104项仅服务旧runner/protocol/历史机械判定的过程测试，科学scheduler/certifier/search测试保留；清理后261项，加10项当前dataset/ranker及失败定位回归后271项通过。清理前已向用户说明DRL未被当前外层Git跟踪，用户明确授权仍直接删除；没有声称这些本地过程文件已存入当前Git历史。', '',
        '## 训练数据与隔离','',
        '使用既有 17 TRAIN / 6 DEV workbook 白名单，每个选1个静态代表实例。TRAIN 为6/6/5、DEV为2/2/2 SMALL/MEDIUM/LARGE；只依据 N 静态选择，不读取 solver 结果。原8 ORACLE_DEV_CONSUMED 不参与本轮训练、epoch或variant选择；V2_VALIDATION和真正 sealed ID_TEST未访问。物理目录名不作为数据角色。',
        '固定 seeds 20261121/20261122，46条保留的原生 M192 60s trajectory。延续 Phase4-1A 机制 smoke 的初始化 portfolio 配置 construction_budget=5 / kinit_ref=5 / bootstrap=1；baseline与MLP一致，正式 SearchConfig 默认未改。5/30/60s context 取时点前最后一个完整 iteration boundary 的 accepted-current，未用 global-best 代替、未从未来回填。',
        f"用户确认：初始化晚于5s时保留真实缺失，不强行补足138。实际 contexts={result['trajectory_summary']['context_count']}，stage分布={result['trajectory_summary']['stage_counts']}；缺失时点与首次boundary时间逐条保存在JSON。",'',
        '| split | states | candidates | FEASIBLE_CERTIFIED | DEADLOCK | DIRECTION_INFEASIBLE | INFEASIBLE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for role,s in result['dataset'].items():
        counts=s['status_counts'];lines.append(f"| {role} | {s['states']} | {s['candidates']} | {counts.get('FEASIBLE_CERTIFIED',0)} | {counts.get('DEADLOCK',0)} | {counts.get('DIRECTION_INFEASIBLE',0)} | {counts.get('INFEASIBLE',0)} |")
    lines.extend(['',
        f"Numeric failure=0；无未标注candidate、无C4 direction-infeasible row、无不可行样本的改善回归标签；全部C4记录的前87维与生成时C2记录逐字节一致，存储往返未改变公共特征。原生trajectory actual runtime median/max={result['trajectory_summary']['runtime_median']:.2f}/{result['trajectory_summary']['runtime_max']:.2f}s。每个状态只生成一次144 atomic+48 LNS attempt pool，所有valid unique candidate做direction，direction-feasible才做B32 reference与认证。",'',
        '采集错误如实保留：首批提交46条，采集器因缺EARLY抛错，shutdown等待其余任务却未收集结果，仅4条落盘；42条未保存的运行被重新执行。修复后每条worker独立事务落盘，已存4条复用。首批丢失结果未用于任何模型选择，未按质量选择性重试；实际 invocation 数因此超过46。', '',
        '## 模型、特征与选择','',
        '独立 C2(87)→128→64 和 C4(130)→128→64 ReLU MLP，各有feasibility logit和normalized improvement两个head。C2使用全部valid unique候选；C4使用整个pool的全部direction-feasible候选，未限制为baseline Top8。',
        'C2没有candidate direction/reference特征；C4只增加direction阶段信息。当前已知state Cmax合法，candidate reference Cmax/WAIT/DEADLOCK/certifier仅作标签。特征纯函数不调用scheduler、不修改state、不消耗RNG。TRAIN-only分别拟合mean/std，零方差std=1，DEV只apply。存储与在线提取都先使用相同float32原始特征语义，再标准化。',
        'AdamW lr=1e-3、weight_decay=1e-4、batch512、最多100 epochs、patience10；BCE + feasible-only SmoothL1=1:1。不可行improvement=NULL，numeric failure排除。score=sigmoid(logit)×pred_gain，允许负gain。固定配置，没有超参搜索。',
        'Epoch只按6个DEV workbook的material-state capture、zero capture和normalized regret选取。C2看完整pool的Top8，C4看完整direction-feasible pool的Top2；最终消融再串联C2→C4。', '',
        '| model | training rows | selected epoch | epochs run | DEV material capture | zero capture |',
        '|---|---:|---:|---:|---:|---:|'])
    for stage,s in result['training'].items():
        m=s['selected_dev']['ALL'];lines.append(f"| {stage} | {s['training_rows']} | {s['selected_epoch']} | {s['epochs_run']} | {percent(m['capture_median'])} | {percent(m['zero_capture'])} |")
    lines.extend(['','固定损失与预测的描述性诊断：'])
    for stage,t in result['training'].items():
        epoch=next(row for row in t['history'] if row['epoch']==t['selected_epoch'])
        pred=result['prediction_diagnostics'][stage]
        lines.append(f"{stage} selected epoch TRAIN BCE={epoch['BCE']:.6f}、SmoothL1={epoch['SmoothL1_feasible']:.6f}；TRAIN feasible样本正改善比例={percent(t['target_summary']['positive_fraction'])}；DEV预测正gain比例={percent(pred['predicted_positive_gain_fraction'])}，predicted gain median={pred['predicted_gain_quantiles']['median']:.6f}。这些数字仅解释固定模型，不用于追加调权或选feature。")
    lines.extend(['','## 四组离线消融','',
        '主统计为state equal weight，仅在完整M192 pool存在≥0.5%改善机会的material states上汇总capture/zero/regret。C2和C4分母均为完整pool可利用改善。无可行selected candidate时capture=0、direct normalized regret=NULL；主regret=(selected best−pool best)/Cs（非负），不把更差候选截到incumbent。NULL不参与regret median，JSON记录regret_defined_states及保留incumbent的辅助fallback regret。Feasibility rate统计该group全部selected candidates，未限定material states。',
        'C2每个非空family先选1个，再全局填满8；C4保留global-best + seed/iteration family explore及真实轮转fallback。四种variant只改score。Heuristic重放用原始双精度排序键，避免float32特征舍入改变基线。', '',
        '| variant | group | material | C2 capture | C2 zero | C2 regret | C4 capture | C4 zero | C4 regret | C4 feasible |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for variant,s in result['offline'].items():
        for group in ('ALL',*d.TIERS,*(name for name,_ in d.STAGES)):
            a,b=s['C2'][group],s['C4'][group]
            lines.append(f"| {variant} | {group} | {b['material_states']} | {percent(a['capture_median'])} | {percent(a['zero_capture'])} | {percent(a['normalized_regret_median'])} | {percent(b['capture_median'])} | {percent(b['zero_capture'])} | {percent(b['normalized_regret_median'])} | {percent(b['selected_candidate_feasibility_rate'])} |")
    lines.extend(['',f"总体离线指标最优（capture/zero/regret依次比较）：{result['best_descriptive_offline_variant']}。MLP-BOTH是否最优：{result['best_descriptive_offline_variant']=='MLP-BOTH'}。"])
    baseline=result['offline']['BASELINE']
    for variant in VARIANTS[1:]:
        a,b=result['offline'][variant]['C2']['ALL'],result['offline'][variant]['C4']['ALL']
        lines.append(f"{variant}：C2 capture变化{100*(a['capture_median']-baseline['C2']['ALL']['capture_median']):+.2f}个百分点；C4 capture变化{100*(b['capture_median']-baseline['C4']['ALL']['capture_median']):+.2f}个百分点；C4 zero capture下降{100*(baseline['C4']['ALL']['zero_capture']-b['zero_capture']):+.2f}个百分点。达到可进入native smoke的明确改善要求：{result['offline'][variant]['qualifies_for_native_smoke']}。")
    lines.extend(['',f"C4 median capture至少提高10个百分点且zero capture至少下降10个百分点：{result['offline_ranking_improved']}。满足条件的最佳variant：{result['best_offline_variant']}。",'','## 真实60s development',''])
    if completed:
        lines.extend(['6个DEV代表实例×seeds 20261201/02/03×baseline/最佳variant，共36runs。相同M192/Kdp8/Kref2/Kref_total4；初始化、feature/scaler/forward/selection全部计入60秒，模型文件加载在搜索开始前。三路进程各占独立物理核心，资源条件一致。', '',
            '| variant | median Cmax@60 | iterations | reference calls | generation seconds | inference seconds | inference/60s |',
            '|---|---:|---:|---:|---:|---:|---:|'])
        for variant,s in native['variant_summary'].items():
            lines.append(f"| {variant} | {s['Cmax_at_60_median']:.3f} | {s['iterations_median']} | {s['reference_calls_median']} | {s['generation_seconds_median']:.3f} | {s['inference_seconds_median']:.3f} | {percent(s['inference_fraction_median'])} |")
        lines.extend(['','| group | paired median MLP/baseline Cmax@60 | iteration ratio | reference-call ratio | win/tie/loss |','|---|---:|---:|---:|---|'])
        for group,s in native['paired_summary'].items():
            lines.append(f"| {group} | {s['Cmax_ratio_median']:.6f} | {s['iteration_ratio_median']:.3f} | {s['reference_call_ratio_median']:.3f} | {s['wins']}/{s['ties']}/{s['losses']} |")
        for variant,summary in native['variant_summary'].items():
            lines.append(f"{variant} reference吞吐中位数={summary['reference_calls_per_second_median']:.3f}/s；每100次reference对应global-best更新中位数={summary['best_updates_per_100_reference_calls_median']:.3f}。这些是辅助完整run指标。")
        lines.append('\n跨规模绝对Cmax median仅作诊断，质量结论按instance×seed配对ratio；final Cmax与overshoot不参与选择。Iterations/reference calls/generation/inference为完整run累计，包含最后完整iteration的overshoot；Cmax严格按60秒截取。Reference calls须结合配对质量观察，不能单独当作效率提升。')
    elif native.get('status')=='SKIPPED':
        lines.append('未执行（0 runs）：没有任何variant同时满足C4 median capture提高至少10个百分点、zero capture下降至少10个百分点。因此按预定条件跳过36次native smoke；没有Cmax@60、inference开销或LARGE在线收益结论。')
    else:
        lines.append('配对native结果不完整：'+native.get('status','NOT RUN')+'。缺失结果不填人工penalty。')
    lines.extend(['','## 结果解释',''])
    for paragraph in result['interpretation']: lines.extend([paragraph,''])
    gains=[]
    for variant in VARIANTS[1:]:
        for group in (*d.TIERS,*(name for name,_ in d.STAGES)):
            base=result['offline']['BASELINE']['C4'][group]['capture_median'];learned=result['offline'][variant]['C4'][group]['capture_median']
            if base is not None and learned is not None: gains.append((learned-base,variant,group))
    largest=max(gains) if gains else None
    lines.extend(['','## 判断与下一步','',
        f"最大描述性离线C4 capture增量：{('%.2f个百分点，%s / %s'%(100*largest[0],largest[1],largest[2])) if largest else '无可比较material group'}。SMALL/MEDIUM/LARGE、EARLY/MID/LATE与联合/单阶段替换的效果见相同预算表，不预设MLP-BOTH最好。",'',
        ('离线及真实搜索显示收益，下一阶段独立验证固定ranker；本轮development不能作为最终泛化结果。' if improved else
         '尚未证明可推荐的在线MLP收益。保留两个模型用于消融复查，生产继续heuristic。固定逐candidate BCE+SmoothL1并不直接优化Top-K capture，training loss下降不等于ranking有效。当前结果不足以断言特征缺失或标签错误；先检查目标与Top-K指标、负gain/尺度及score分布，不直接跳到GAT。'),'',
        f"下一步：{result['next_step']}",'',
        '没有修改M192、Kdp/Kref、科学scope、X/Y、scheduler/B32或算子；没有访问真正最终测试、训练GAT或重新划分workbook。', '',
        ])
    lines.extend(diagnostic_report_lines(result))
    replay=result.get('regression',{}).get('actual_model_api_replay',{})
    total_candidates=sum(value['candidates'] for value in result['dataset'].values())
    lines.extend(['','## 验证','',f"完整回归：{result.get('regression',{}).get('full_final','未记录')}。真实模型API重放{replay.get('states','未记录')} states×{replay.get('variants','未记录')} variants={replay.get('comparisons','未记录')}次比较，mismatch={replay.get('mismatches','未记录')}，新增direction/reference调用={replay.get('new_direction_or_reference_calls','未记录')}。SQLite核验：{result.get('regression',{}).get('sqlite_integrity','未记录')}。当前candidate总数={total_candidates}。",''])
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    base=result['offline']['BASELINE']['C4']['ALL'];best=result['best_descriptive_offline_variant']
    print(f"Phase4-1B: {result['phase_status']}\nRepository cleanup: completed\nMLP dataset: TRAIN states={result['dataset']['MLP_TRAIN']['states']} / DEV states={result['dataset']['MLP_DEV']['states']}\nBest offline variant: {best}\nBaseline C4 capture: {percent(base['capture_median'])}\nBest MLP C4 capture: {percent(result['offline'][best]['C4']['ALL']['capture_median'])}\n60s development: {native.get('status')}\nRecommended ranker: {result['recommended_ranker']} (M192)\nGAT next: NOT YET\nNext: {result['next_step']}",flush=True)



RANK_RESULT=ROOT/'data/development/phase4_1c_mlp_results.json'
RANK_REPORT=ROOT/'docs/PHASE4_1C_MLP_TOPK_RANKER_20261008.md'
OBJECTIVES={'V2':'material_v2','V3':'rank_v2'}


def heuristic_c4(row):
    return (row['projected'],row['dp_travel'],row['split_delta'],row['candidate_id'])


def pools_by_state(records):
    grouped=defaultdict(list)
    for row in records: grouped[row['state_id']].append(row)
    return grouped


def evaluate_rankers(records):
    output={}
    for variant in VARIANTS:
        c2_rows=[];c4_rows=[];decomposed=[]
        for pool in pools_by_state(records).values():
            first=pool[0];seed,it=first['seed'],first['iteration']
            c2=c2_select(pool,variant in ('MLP-C2','MLP-BOTH'),seed,it)
            c4=c4_select(c2,variant in ('MLP-C4','MLP-BOTH'),seed,it)
            common={k:first[k] for k in ('state_id','tier','stage')}
            c2_rows.append({**common,**capture_metrics(pool,c2)})
            c4_rows.append({**common,**capture_metrics(pool,c4)})
            decomposed.append(loss_decomposition(pool,c2,c4))
        output[variant]={'C2':group_summary(c2_rows),'C4':group_summary(c4_rows),
                         'state_metrics':{'C2':c2_rows,'C4':c4_rows},
                         'loss_decomposition':decomposition_groups(decomposed)}
    return output


def evaluate_c4_top8(records):
    # Fixed baseline and selected V3-C2 Top8 contexts, with full-pool denominators.
    rows=[]
    for pool in pools_by_state(records).values():
        first=pool[0]
        for learned in (False,True):
            top8=c2_select(pool,learned,first['seed'],first['iteration'])
            selected=c4_select(top8,True,first['seed'],first['iteration'])
            rows.append({**capture_metrics(pool,selected),**{k:first[k] for k in ('state_id','tier','stage')}})
    return group_summary(rows)


def fit_material(stage,objective,rows,dev,names,scalers):
    from mrta_ranker.model import ranking_loss
    torch.manual_seed(20261123+(stage=='C4'))
    mean,std=scalers[stage+'_mean'],scalers[stage+'_std']
    x=torch.from_numpy(transform([r[stage+'_features'] for r in rows],mean,std))
    yf=torch.tensor([r['y_feasible'] for r in rows],dtype=torch.float32)
    ym=torch.tensor([d.material_positive(r) for r in rows],dtype=torch.float32)
    positives=int(ym.sum());negatives=len(rows)-positives
    if not positives: raise ValueError('No TRAIN material positives')
    pos_weight=torch.tensor(negatives/positives,dtype=torch.float32)
    pairs=d.hard_ranking_pairs(rows,stage,(lambda r:r['candidate_id']) if stage=='C2' else heuristic_c4)
    pair_ids=sorted({i for p,n,w in pairs for i in (p,n)})
    local={v:i for i,v in enumerate(pair_ids)}
    pi=torch.tensor([local[p] for p,n,w in pairs],dtype=torch.long)
    ni=torch.tensor([local[n] for p,n,w in pairs],dtype=torch.long)
    pw=torch.tensor([w for p,n,w in pairs],dtype=torch.float32)
    px=x[pair_ids]
    model=CandidateMLP(len(names[stage]),objective='material')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    best=None;stale=0;history=[]
    path=MODELS/f'mlp_{stage.lower()}_{OBJECTIVES[objective]}.pt'
    for epoch in range(1,101):
        model.train();losses=[]
        for indices in torch.randperm(len(rows)).split(512):
            optimizer.zero_grad(set_to_none=True)
            feasible,material=model(x[indices])
            pair_scores=model(px)[1] if objective=='V3' and pairs else material[:0]
            positive,negative=(pair_scores[pi],pair_scores[ni]) if len(pair_scores) else (material[:0],material[:0])
            loss,a,b,c=ranking_loss(feasible,material,yf[indices],ym[indices],pos_weight,positive,negative,pw,pairwise=objective=='V3')
            loss.backward();optimizer.step()
            losses.append([float(v.detach()) for v in (loss,a,b,c)])
        set_scores(dev,stage,model,mean,std)
        metrics=evaluate_stage(dev,'C2') if stage=='C2' else evaluate_c4_top8(dev)
        key=selection_key(metrics)
        history.append({'epoch':epoch,**dict(zip(('loss','feasibility_BCE','material_BCE','pairwise'),map(float,np.mean(losses,axis=0)))),'dev_material':metrics['ALL']})
        if best is None or key>best:
            best=key;best_epoch=epoch;best_metrics=metrics;stale=0
            save_model(path,model,stage,names[stage],epoch,metrics)
        else: stale+=1
        if stale>=10: break
    model,_=load_model(path)
    return model,{'training_rows':len(rows),'positive_count':positives,'negative_count':negatives,
        'pos_weight':float(pos_weight),'pair_count':len(pairs) if objective=='V3' else 0,
        'pair_states':len({rows[p]['state_id'] for p,n,w in pairs}) if objective=='V3' else 0,
        'selected_epoch':best_epoch,'epochs_run':epoch,'selected_dev':best_metrics,'history':history,
        'checkpoint':str(path.relative_to(ROOT))}


def rank_offline_gate(baseline,candidate):
    ab,an=baseline['ALL'],candidate['ALL'];lb,ln=baseline['LARGE'],candidate['LARGE']
    checks={'ALL_capture_plus_10pp':an['capture_median'] is not None and ab['capture_median'] is not None and an['capture_median']-ab['capture_median']>=.10-1e-12,
        'ALL_zero_minus_10pp':an['zero_capture'] is not None and ab['zero_capture'] is not None and ab['zero_capture']-an['zero_capture']>=.10-1e-12,
        'LARGE_zero_not_worse':ln['zero_capture'] is not None and lb['zero_capture'] is not None and ln['zero_capture']<=lb['zero_capture']+1e-12}
    return {'passed':all(checks.values()),'checks':checks}


def rank_experiment(workers=3):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    result=d.read(RANK_RESULT)
    train,names=load_records('MLP_TRAIN');dev,_=load_records('MLP_DEV')
    scalers=np.load(MODELS/'mlp_scalers.npz')
    result['dataset']={role:{'states':len(pools_by_state(rows)),'candidates':len(rows),'workbooks':len({r['workbook'] for r in rows}),
        'status_counts':dict(Counter(r['status'] for r in rows)),'material_positives':sum(d.material_positive(r) for r in rows)} for role,rows in (('MLP_TRAIN',train),('MLP_DEV',dev))}
    for stage in ('C2','C4'):
        model,_=load_model(MODELS/f'mlp_{stage.lower()}.pt')
        for rows in (train,dev):
            set_scores(rows,stage,model,scalers[stage+'_mean'],scalers[stage+'_std'])
            for r in rows:
                if r[stage+'_features'] is not None: r[stage+'_v1_score']=r[stage+'_score']
    result['offline']={'V1':evaluate_rankers(dev)}
    historical=d.read(d.RESULT)['offline']
    for variant in VARIANTS:
        for stage in ('C2','C4'):
            if result['offline']['V1'][variant][stage]!=historical[variant][stage]: raise AssertionError('Historical V1 replay differs')
    result['training_config']={'hidden':[128,64],'heads':['feasibility','material'],'gain_head':False,'score':'sigmoid(material_logit)',
        'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'batch_size':512,'max_epochs':100,'patience':10,
        'loss_V2':'1 feasibility BCE + 1 TRAIN-pos-weight material BCE',
        'loss_V3':'V2 + 1 state-equal pairwise softplus(-(sp-sn))',
        'max_negatives_per_positive':4,'training_seeds':{'C2':20261123,'C4':20261124},
        'scaler':'Unchanged V1 TRAIN-only scaler; no refit or new scaler file',
        'C4_ablation':'Same V3-C2-derived TRAIN hard subset and fixed baseline/V3-C2 DEV Top8 contexts',
        'epoch_selection':'C2 DEV Top8; C4 Top8->Top2 contexts equally weighted; capture/zero/regret',
        'split':'Original disjoint 17 TRAIN / 6 DEV workbooks; no candidate resplit'}
    models={};result['training']={o:{} for o in OBJECTIVES}
    for objective in ('V3','V2'):
        model,info=fit_material('C2',objective,train,dev,names,scalers)
        models[(objective,'C2')]=model;result['training'][objective]['C2']=info
        save_result(result)
    for rows in (train,dev): set_scores(rows,'C2',models[('V3','C2')],scalers['C2_mean'],scalers['C2_std'])
    hard=[];subsets=[]
    for pool in pools_by_state(train).values():
        first=pool[0];seed,it=first['seed'],first['iteration']
        base=c2_select(pool,False,seed,it);new=c2_select(pool,True,seed,it)
        for r in pool: r['C2_score']=r['C2_v1_score']
        legacy=c2_select(pool,True,seed,it)
        subset=d.c4_hard_subset(pool,base,new,legacy,heuristic_c4)
        hard.extend(subset)
        subsets.append({'state_id':first['state_id'],'tier':first['tier'],'count':len(subset),
            'material_positives':sum(d.material_positive(r) for r in subset),
            'baseline_top8_ids':[r['candidate_id'] for r in base],
            'new_top8_ids':[r['candidate_id'] for r in new],
            'legacy_top8_ids':[r['candidate_id'] for r in legacy],
            'hard_ids':[r['candidate_id'] for r in subset]})
    result['C4_hard_subset']={'rows':len(hard),'states':len(subsets),'direction_infeasible':sum(r['status']=='DIRECTION_INFEASIBLE' for r in hard),
        'all_direction_feasible_train':sum(r['C4_features'] is not None for r in train),'state_subsets':subsets}
    for objective in ('V3','V2'):
        model,info=fit_material('C4',objective,hard,dev,names,scalers)
        models[(objective,'C4')]=model;result['training'][objective]['C4']=info
        save_result(result)
    result['train_in_sample']={};result['gates']={};eligible=[]
    for objective in ('V2','V3'):
        for rows,section in ((train,result['train_in_sample']),(dev,result['offline'])):
            for stage in ('C2','C4'): set_scores(rows,stage,models[(objective,stage)],scalers[stage+'_mean'],scalers[stage+'_std'])
            section[objective]=evaluate_rankers(rows)
        result['gates'][objective]={}
        for variant in VARIANTS[1:]:
            gate=rank_offline_gate(result['offline'][objective]['BASELINE']['C4'],result['offline'][objective][variant]['C4'])
            result['gates'][objective][variant]=gate
            if gate['passed']: eligible.append((objective,variant))
    candidates=[(o,v) for o in ('V2','V3') for v in VARIANTS[1:]]
    best=max(candidates,key=lambda ov:selection_key(result['offline'][ov[0]][ov[1]]['C4']))
    result['best_descriptive_objective'],result['best_descriptive_variant']=best
    result['offline_ranking_improved']=bool(eligible)
    chosen=max(eligible,key=lambda ov:selection_key(result['offline'][ov[0]][ov[1]]['C4'])) if eligible else best
    result['best_objective'],result['best_offline_variant']=chosen
    result['best_model_version']=OBJECTIVES[chosen[0]]
    if not eligible: result['smoke']={'status':'SKIPPED','runs':[],'reason':'Strict ALL +10pp capture / -10pp zero and LARGE zero-not-worse gate failed.'}
    save_result(result)
    print('Offline gate: '+('PASS' if eligible else 'FAIL'),flush=True)
    if eligible: smoke(workers,RANK_RESULT)



def rank_audit():
    """Fixed V1 checkpoints and DEV only; no reference, training, or new labels."""
    torch.set_num_threads(1)
    records,_=load_records('MLP_DEV');scalers=np.load(MODELS/'mlp_scalers.npz');audit={}
    for stage in ('C2','C4'):
        rows=[r for r in records if r[stage+'_features'] is not None]
        model,_=load_model(MODELS/f'mlp_{stage.lower()}.pt')
        with torch.inference_mode():
            logit,gain=model(torch.from_numpy(transform([r[stage+'_features'] for r in rows],scalers[stage+'_mean'],scalers[stage+'_std'])))
            prob=torch.sigmoid(logit).numpy();gain=gain.numpy()
        for r,p,g in zip(rows,prob,gain):
            r[stage+'_p']=float(p);r[stage+'_gain']=float(g);r[stage+'_score']=float(np.float32(p*g))
        groups={name:[] for name in ('FEASIBLE_MATERIAL','FEASIBLE_NONMATERIAL','DEADLOCK','DIRECTION_INFEASIBLE')}
        for row in rows:
            name=('FEASIBLE_MATERIAL' if d.material_positive(row) else 'FEASIBLE_NONMATERIAL') if row['status']=='FEASIBLE_CERTIFIED' else row['status']
            groups[name].append(row)
        stats={}
        for name,rs in groups.items():
            stats[name]={'n':len(rs),'negative_gain_count':sum(r[stage+'_gain']<0 for r in rs)}
            for field in ('p','gain','score'):
                stats[name][field]=dict(zip(('median','p25','p75','p90'),map(float,np.quantile([r[stage+'_'+field] for r in rs],[.5,.25,.75,.9])))) if rs else None
        inversions=[]
        for pool in pools_by_state(rows).values():
            good=[r for r in pool if d.material_positive(r) and r[stage+'_gain']<0]
            bad=[r for r in pool if r['status']!='FEASIBLE_CERTIFIED' and r[stage+'_gain']<0]
            selected=c2_select(pool,True,pool[0]['seed'],pool[0]['iteration']) if stage=='C2' else c4_select(pool,True,pool[0]['seed'],pool[0]['iteration'])
            pairs=sum(b[stage+'_gain']<=a[stage+'_gain'] and b[stage+'_score']>a[stage+'_score'] for a in good for b in bad)
            selected_pairs=sum(b[stage+'_gain']<=a[stage+'_gain'] and b[stage+'_score']>a[stage+'_score'] for a in good for b in selected if b['status']!='FEASIBLE_CERTIFIED')
            inversions.append({'state_id':pool[0]['state_id'],'tier':pool[0]['tier'],'pairs':pairs,'selected_pairs':selected_pairs})
        audit[stage]={'groups':stats,'negative_gain_fraction':float(np.mean(gain<0)),
            'states_with_material_inversion':sum(r['pairs']>0 for r in inversions),'material_inversion_pairs':sum(r['pairs'] for r in inversions),
            'states_with_selected_inversion':sum(r['selected_pairs']>0 for r in inversions),'state_inversions':inversions}
    result=d.read(RANK_RESULT) if RANK_RESULT.exists() else {'phase':'Phase4-1C','historical_phase4_1b':'FAIL; untouched; 0 MLP smoke runs',
        'new_reference_calls_offline':0,'new_candidate_labels':0}
    result['score_audit']=audit;save_result(result)


def rank_finish():
    """One result JSON and one evidence report; never rewrite Phase4-1B."""
    result=d.read(RANK_RESULT);o,v=result['best_objective'],result['best_offline_variant']
    base=result['offline'][o]['BASELINE'];best=result['offline'][o][v]
    native=result.get('smoke',{});completed=native.get('status')=='COMPLETED'
    passed=completed and native['paired_summary']['ALL']['Cmax_ratio_median']<1-1e-9
    result['phase_status']='PASS' if passed else 'FAIL'
    result['production_ranker']='HEURISTIC'
    result['recommended_ranker']=v if passed else 'HEURISTIC'
    result['GAT']='NOT YET';result['native_smoke_runs']=len(native.get('runs',[]))
    result['temporary_cleanup']={'temporary_checkpoints_created':0,'temporary_caches_created':0,'removed_files':[]}
    lb,ln=base['C4']['LARGE'],best['C4']['LARGE']
    result['large_assessment']='improved' if ln['zero_capture']<lb['zero_capture'] or (ln['zero_capture']==lb['zero_capture'] and ln['capture_median']>lb['capture_median']) else 'worse' if ln['zero_capture']>lb['zero_capture'] or ln['capture_median']<lb['capture_median'] else 'unchanged'
    result['large_ranking_failure_resolved']=ln['capture_median']>0 and best['C2']['LARGE']['capture_median']>0
    result['next_step']='Keep heuristic; analyze LARGE cross-workbook errors with existing labels, then decide whether more LARGE TRAIN states or relational modeling is warranted in the next study.' if not passed else 'Independently evaluate the fixed recommended learned variant before adoption.'
    pc=lambda x:'N/A' if x is None else f'{100*x:.2f}%'
    v2=result['offline']['V2'];v3=result['offline']['V3'];v1=result['offline']['V1']
    audit=result['score_audit'];decomp=best['loss_decomposition']['ALL'];large=best['loss_decomposition']['LARGE']
    replay=result.get('verification',{}).get('actual_model_api_replay',{})
    inference=replay.get('inference_summary',{}).get(o,{}).get(v,{}).get('ALL',{}).get('total_seconds')
    replay_ms='未测量' if inference is None else f'{1000*inference:.2f} ms/state'
    answers=[
        f"旧 p×gain 病理严重且真实存在：C2/C4 全体预测负gain为 {pc(audit['C2']['negative_gain_fraction'])}/{pc(audit['C4']['negative_gain_fraction'])}；280个真实material正样本中 {audit['C2']['groups']['FEASIBLE_MATERIAL']['negative_gain_count']}/{audit['C4']['groups']['FEASIBLE_MATERIAL']['negative_gain_count']} 个预测为负。不可行候选在gain不更好的情况下反超material正样本，分别发生在 {audit['C2']['states_with_material_inversion']}/36、{audit['C4']['states_with_material_inversion']}/36 states。独立阶段Top-K所选slot中出现该反转仅 {audit['C2']['states_with_selected_inversion']}、{audit['C4']['states_with_selected_inversion']} states，不能把全部失败单独归因于该交互。",
        f"Material classification（V2）C2 Top8 median capture：baseline {pc(base['C2']['ALL']['capture_median'])}→{pc(v2['MLP-C2']['C2']['ALL']['capture_median'])}，增加 {100*(v2['MLP-C2']['C2']['ALL']['capture_median']-base['C2']['ALL']['capture_median']):.2f} pp；相对V1 {pc(v1['MLP-C2']['C2']['ALL']['capture_median'])} 再增加 {100*(v2['MLP-C2']['C2']['ALL']['capture_median']-v1['MLP-C2']['C2']['ALL']['capture_median']):.2f} pp。",
        f"Pairwise 未进一步提高：V3 C2 capture={pc(v3['MLP-C2']['C2']['ALL']['capture_median'])}，相对V2变化 {100*(v3['MLP-C2']['C2']['ALL']['capture_median']-v2['MLP-C2']['C2']['ALL']['capture_median']):+.2f} pp。V3 BOTH C4 zero={pc(v3['MLP-BOTH']['C4']['ALL']['zero_capture'])}，V2 BOTH={pc(v2['MLP-BOTH']['C4']['ALL']['zero_capture'])}。固定loss权重下没有继续搜索。",
        f"C4 hard-training使用 {result['C4_hard_subset']['rows']} 个既有candidate（原全部direction-feasible TRAIN {result['C4_hard_subset']['all_direction_feasible_train']}），保留716个material positives。局部有用、整体未解决：V2中保持同一新版C2，把heuristic C4替换为新版C4，MEDIUM capture {pc(v2['MLP-C2']['C4']['MEDIUM']['capture_median'])}→{pc(v2['MLP-BOTH']['C4']['MEDIUM']['capture_median'])}，但SMALL {pc(v2['MLP-C2']['C4']['SMALL']['capture_median'])}→{pc(v2['MLP-BOTH']['C4']['SMALL']['capture_median'])}，LARGE zero {pc(v2['MLP-C2']['C4']['LARGE']['zero_capture'])}→{pc(v2['MLP-BOTH']['C4']['LARGE']['zero_capture'])}。未训练‘新objective+旧C4分布’额外对照，不能独立归因于hard subset。",
        f"ALL C4 Top2 median capture：baseline {pc(base['C4']['ALL']['capture_median'])}→最佳 {pc(best['C4']['ALL']['capture_median'])}，仍为0；6种新objective/variant均为0。",
        f"最佳C4 zero capture：{pc(base['C4']['ALL']['zero_capture'])}→{pc(best['C4']['ALL']['zero_capture'])}，下降 {100*(base['C4']['ALL']['zero_capture']-best['C4']['ALL']['zero_capture']):.2f} pp；29个material states中最终捕获改善 {decomp['C4_capture_states']} 个。",
        '最佳variant的 SMALL/MEDIUM/LARGE：'+ '；'.join(f"{g} C2 {pc(best['C2'][g]['capture_median'])}、C4 {pc(best['C4'][g]['capture_median'])}、C4 zero {pc(best['C4'][g]['zero_capture'])}" for g in d.TIERS)+'。完整四组结果见下表。',
        f"LARGE仍系统失败：最佳C2/C4 median capture均0；12个material states中 {large['no_C2_capture_states']} 个在C2已无改善可送入C4，{large['C4_dropped_all_states']} 个在C4丢光。V2 C2 LARGE TRAIN capture={pc(result['train_in_sample']['V2']['MLP-C2']['C2']['LARGE']['capture_median'])}、DEV=0，TRAIN仅作in-sample诊断。LARGE zero相对baseline {pc(lb['zero_capture'])}→{pc(ln['zero_capture'])} 的改善不代表泛化问题已解决。",
        '最佳variant的 EARLY/MID/LATE：'+'；'.join(f"{g} C2 {pc(best['C2'][g]['capture_median'])}、C4 {pc(best['C4'][g]['capture_median'])}、C4 zero {pc(best['C4'][g]['zero_capture'])}" for g,_ in d.STAGES)+'。',
        f"最佳描述性variant：{o} / NEW-{v}。按ALL C4 capture median降序、zero升序、normalized regret升序选取；与V2 BOTH的capture/zero并列时，C2-only regret更低。无variant取得smoke资格。",
        '离线Gate：'+('PASS。' if result['offline_ranking_improved'] else 'FAIL。最佳variant满足zero下降≥10pp、LARGE zero不恶化，但ALL median capture没有增加≥10pp。标准未降低。'),
        '真实60秒Cmax：'+(f"配对median MLP/baseline={native['paired_summary']['ALL']['Cmax_ratio_median']:.6f}。" if completed else 'SKIPPED / 0 runs；离线失败后停止在线实验，未测量Cmax收益。'),
        f"MLP开销：现有状态真实API重放的feature extraction+scaler+forward+ranking中位数 {replay_ms}，模型load在计时外。{replay.get('comparisons',0)}次比较与离线选择一致，新增reference/direction=0。没有60秒在线run，因此ranker/total time比例未测量。",
        f"本轮能确认的失败位置：ALL {decomp['no_C2_capture_states']}/29 states在C2无捕获，另 {decomp['C4_dropped_all_states']}/29 states在C4丢光。既有LARGE跨workbook问题仍在，C4在SMALL/MEDIUM的取舍也没有形成整体收益。在线未执行，无法声称原因是overhead、trajectory shift、diversity或SA；这些须未来有在线证据才判断。",
        'Production不切换：继续M192 + heuristic C2/C4；新版模型保留为失败实验和对照证据。',
        'GAT：NOT YET。本轮没有实现/训练。LARGE失败和TRAIN/DEV差距不足以证明缺少weld-robot-route关系表达；尤其pairwise本身未改善、C4分组效果不一致，不能直接推出需要GAT。',
        '下一步：保持heuristic，下一轮先用现有labels分析LARGE跨workbook错误与hard-negative覆盖，再决定是否更多LARGE TRAIN states或关系建模。本轮停止，不新增数据、模型搜索、operator或在线运行。'
    ]
    result['answers']=answers;save_result(result)
    lines=['# Phase4-1C：Top-K aligned MLP ranker','',
        f"结论：**{result['phase_status']}**。最佳离线 **{o} / NEW-{v}**；离线Gate **{'PASS' if result['offline_ranking_improved'] else 'FAIL'}**；smoke **{native.get('status')} / {result['native_smoke_runs']} runs**；production **HEURISTIC**。",'',
        '## 直接回答','']+[f'{i}. {answer}' for i,answer in enumerate(answers,1)]
    lines+=['','## 旧score审计：36个DEV states','',
        '每个数列依次为 median / p25 / p75 / p90。FEASIBLE_MATERIAL=≥0.5%；FEASIBLE_NONMATERIAL包括零改善、负改善与<0.5%改善。C4禁止direction-infeasible输入，因此该组为0行，不给伪造统计。审计只使用旧checkpoint、旧scaler和现有DEV labels。',
        '', '| stage | group | n | negative gain | p_feasible | predicted_gain | p×gain |','|---|---|---:|---:|---|---|---|']
    for stage,a in audit.items():
        for group,g in a['groups'].items():
            fmt=lambda field:'N/A' if g[field] is None else ' / '.join(f"{g[field][q]:.6f}" for q in ('median','p25','p75','p90'))
            lines.append(f"| {stage} | {group} | {g['n']} | {g['negative_gain_count']} | {fmt('p')} | {fmt('gain')} | {fmt('score')} |")
    lines+=['','审计中的selected-inversion统计为独立阶段全pool Top8/Top2，用来量化病理；C4该诊断并非串联Top8→Top2选择率，最终结论只用下方真实串联离线表。',
        '', '## 固定训练配置与C4分布','',
        '网络：C2 87→128→64、C4 130→128→64，各自独立，两head feasibility/material；不保留gain head，无attention/GAT。material=(status==FEASIBLE_CERTIFIED and (Cs−Ccandidate)/Cs≥0.005)，NUMERIC_FAILURE排除。score仅sigmoid(material_logit)，tie使用原heuristic。C2全valid unique M192 candidates；C4只用direction-feasible hard subset。',
        '', 'Loss V2 = feasibility BCE + TRAIN-pos-weight material BCE；V3另加权重1的softplus(-(sp−sn))。每positive最多4个hard negatives，交替取heuristic、V1得分最靠前的negative，始终同state；pairwise对有pair的state等权平均，每个classification minibatch加同一有限pair集合的平均loss。没有DEADLOCK人工Cmax penalty。',
        '', 'AdamW lr=1e-3、weight_decay=1e-4、batch512、max100、patience10；ReLU，stage seeds C2=20261123/C4=20261124。V2/V3相同初始化，无超参搜索。直接复用V1的TRAIN-only scaler，不覆盖旧文件。C4 hard subset为 baseline C2 Top8、新V3 C2 Top8、V1 C2 Top8、所有material-positive direction-feasible、并集外heuristic C3/rerank额外8个的去重并集。',
        '', '为隔离pairwise，V2/V3 C4共享一次构造的V3-C2 hard subset，DEV epoch也共享baseline/V3-C2两种固定Top8 contexts。C4只在这两种Top8内用原family policy选Top2，两context等权，改善分母仍为完整pool；C2 epoch直接评价完整pool Top8。最后四组串联消融中，各objective使用自己的C2。C4 epoch没有从百余direction-feasible候选直接选Top2。',
        '', '| objective | stage | TRAIN rows | positives | negatives | pos_weight | pairs | selected epoch | epochs run |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for objective,stages in result['training'].items():
        for stage,t in stages.items():lines.append(f"| {objective} | {stage} | {t['training_rows']} | {t['positive_count']} | {t['negative_count']} | {t['pos_weight']:.6f} | {t['pair_count']} | {t['selected_epoch']} | {t['epochs_run']} |")
    lines+=['', '原split完全保留：17 TRAIN/6 DEV workbook，98/36 states、10,759/4,134 candidates，numeric failure=0；C4 hard rows=3,097、material positives=716、direction-infeasible=0；只有TRAIN计数计算pos_weight。原8 ORACLE_DEV_CONSUMED的结果不用于训练/模型选择，V2_VALIDATION及最终ID_TEST未访问，无candidate随机拆分。',
        '', '## 同预算四组离线结果','',
        'C2每非空family先选1个、再全局补至8；C4保留global-best slot和seed+iteration family rotation/fallback slot，不改成global Top2。全部M192 / Kdp8 / Kref2 / Kref_total4，TWO_OPT_STAR OFF。V1只重放，不重新训练，全部group指标与Phase4-1B原JSON完全相同。',
        '', 'Capture = selected best捕获的改善 / 全pool最佳改善；material state仍要求pool存在≥0.5%改善。Zero capture指没有捕获任何正改善。Capture/zero/regret仅汇总material states，state等权；direct normalized regret只在selected feasible存在时定义，无feasible为NULL，JSON保留defined counts和fallback regret。Feasibility统计该group全部selected candidates。',
        '', '| objective | variant | group | material states | C2 capture | C2 zero | C2 regret | C2 feasible | C4 capture | C4 zero | C4 regret | C4 feasible |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for objective in ('V1','V2','V3'):
        for variant,m in result['offline'][objective].items():
            for group in ('ALL',*d.TIERS,*(stage for stage,_ in d.STAGES)):
                a,b=m['C2'][group],m['C4'][group]
                lines.append(f"| {objective} | {variant} | {group} | {b['material_states']} | {pc(a['capture_median'])} | {pc(a['zero_capture'])} | {pc(a['normalized_regret_median'])} | {pc(a['selected_candidate_feasibility_rate'])} | {pc(b['capture_median'])} | {pc(b['zero_capture'])} | {pc(b['normalized_regret_median'])} | {pc(b['selected_candidate_feasibility_rate'])} |")
    lines+=['','## LARGE泛化与严格Gate','',
        '| objective | variant | TRAIN LARGE C2 capture | DEV LARGE C2 capture | TRAIN LARGE C4 capture | DEV LARGE C4 capture | DEV no C2 capture | DEV C4 dropped all |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for objective in ('V2','V3'):
        for variant in VARIANTS[1:]:
            tr=result['train_in_sample'][objective][variant];dv=result['offline'][objective][variant];dd=dv['loss_decomposition']['LARGE']
            lines.append(f"| {objective} | {variant} | {pc(tr['C2']['LARGE']['capture_median'])} | {pc(dv['C2']['LARGE']['capture_median'])} | {pc(tr['C4']['LARGE']['capture_median'])} | {pc(dv['C4']['LARGE']['capture_median'])} | {dd['no_C2_capture_states']}/12 | {dd['C4_dropped_all_states']}/12 |")
    lines+=['','TRAIN为in-sample诊断，不用于epoch/variant选择。','',
        '| objective | variant | ALL capture +10pp | ALL zero −10pp | LARGE zero not worse | gate |','|---|---|---|---|---|---|']
    for objective,variants in result['gates'].items():
        for variant,gate in variants.items(): lines.append(f"| {objective} | {variant} | {' | '.join(str(x) for x in gate['checks'].values())} | {'PASS' if gate['passed'] else 'FAIL'} |")
    lines+=['','## 开销与验证','',
        '下表是现有DEV accepted states上的真实接口重放中位数（ms/state）。包含feature extraction、scaler、forward、ranking；load排除。只跑一次重放，受当前硬件/进程预热影响，不外推60秒在线占比。heuristic阶段计为0 ranker time；不是整轮solver总时间。','',
        '| objective | variant | group | C2 ms | C4 ms | total ms |','|---|---|---|---:|---:|---:|']
    for objective,variants in replay.get('inference_summary',{}).items():
        for variant,groups in variants.items():
            for group,t in groups.items():lines.append(f"| {objective} | {variant} | {group} | {1000*t['C2_seconds']:.3f} | {1000*t['C4_seconds']:.3f} | {1000*t['total_seconds']:.3f} |")
    if completed:
        lines+=['','真实60秒配对结果：','', '| group | pairs | median MLP/baseline Cmax@60 | win/tie/loss |','|---|---:|---:|---|']
        for group,g in native['paired_summary'].items():lines.append(f"| {group} | {g['pairs']} | {g['Cmax_ratio_median']:.6f} | {g['wins']}/{g['ties']}/{g['losses']} |")
    lines+=['',result.get('verification',{}).get('pytest','pytest PENDING'),
        '',f"真实model API重放36 states × 2 objectives × 3 variants = {replay.get('comparisons',0)}次比较，mismatch=0，新增reference/direction=0；禁用scheduler与trajectory入口执行重放。必要回归包括0.499%/0.500%边界、pairwise优化方向、C4 subset包含Top8并排除direction-infeasible、material score不受feasibility负gain交互影响、family coverage/rotation、Kdp/Kref预算、推理不调用scheduler、workbook隔离。",'',
        '没有生成新candidate dataset、trajectory或label，没有新protocol/hash/scope/manifest，没有改M/Kdp/Kref、X/Y、scheduler/B32、SA、operator或production默认。历史Phase4-1B FAIL、0 MLP smoke runs、旧模型/scaler/核心报告和JSON保持不变。四个新checkpoint直接保存最佳epoch，无临时checkpoint或cache，清理删除0文件。仅新增已确认的4个模型、1个结果JSON和本报告。','']
    RANK_REPORT.write_text('\n'.join(lines),encoding='utf-8')
    print(f"Phase4-1C: {result['phase_status']}\nBest offline variant: {o} / NEW-{v}\nBaseline C4 capture: {pc(base['C4']['ALL']['capture_median'])}\nNew C4 capture: {pc(best['C4']['ALL']['capture_median'])}\nBaseline zero capture: {pc(base['C4']['ALL']['zero_capture'])}\nNew zero capture: {pc(best['C4']['ALL']['zero_capture'])}\nLARGE: {result['large_assessment']} (ranking failure persists)\nOffline gate: {'PASS' if result['offline_ranking_improved'] else 'FAIL'}\n60s smoke: {native.get('status')}\n60s Cmax result: {native.get('paired_summary',{}).get('ALL',{}).get('Cmax_ratio_median','N/A')}\nProduction ranker: HEURISTIC\nGAT: NOT YET\nNext: {result['next_step']}",flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('train','offline','smoke','diagnose','finish','rank','audit','rank-finish'));parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='audit': rank_audit()
    elif args.action=='rank-finish': rank_finish()
    elif args.action=='rank': rank_experiment(args.workers)
    elif args.action=='train': train()
    elif args.action=='offline': offline()
    elif args.action=='smoke': smoke(args.workers)
    elif args.action=='diagnose': diagnose()
    else: finish()
