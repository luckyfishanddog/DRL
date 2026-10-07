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


def save_result(result): d.RESULT.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


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


def smoke_job(entry,seed,variant):
    from mrta_ranker.ranking import LearnedRanker
    if entry['instance_id'] not in _SMOKE_PARENTS: _SMOKE_PARENTS[entry['instance_id']]=d.load_parents(entry)
    ranker=None
    if variant!='BASELINE':
        if variant not in _SMOKE_RANKERS: _SMOKE_RANKERS[variant]=LearnedRanker(variant)
        ranker=_SMOKE_RANKERS[variant];ranker.inference_seconds=0.0
    record,_=d.run_trajectory(entry,seed,parents=_SMOKE_PARENTS[entry['instance_id']],source=_SMOKE_SOURCE,ranker=ranker)
    return {'instance_id':entry['instance_id'],'workbook':entry['relative_path'],'tier':entry['tier'],'seed':seed,'variant':variant,**record}


def smoke(workers=3):
    result=d.read(d.RESULT)
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
            jobs.extend((entry,seed,v) for v in variants if (entry['instance_id'],seed,v) not in done)
    masks=d.physical_core_affinities()[:workers]
    if len(masks)!=workers: raise ValueError('Need distinct physical cores')
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize_smoke,initargs=(masks,)) as pool:
        for future in as_completed([pool.submit(smoke_job,*job) for job in jobs]):
            row=future.result();section['runs'].append(row);save_result(result)
            print(f"60s smoke {len(section['runs'])}/36 {row['variant']} {row['tier']} Cmax@60={row['Cmax_at_60']}",flush=True)
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
        '保留科学 src/tests、FORMAL_SCOPE_V2、实验方案、正式数据角色与固定 MLP split，以及解释当前算法的 Phase4-0B、Phase4-1A 等最终报告；清理后的文档链接已修复。退役104项仅服务旧runner/protocol/历史机械判定的过程测试，科学scheduler/certifier/search测试保留；清理后261项，加9项当前dataset/ranker回归后270项通过。清理前已向用户说明DRL未被当前外层Git跟踪，用户明确授权仍直接删除；没有声称这些本地过程文件已存入当前Git历史。', '',
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
    replay=result.get('regression',{}).get('actual_model_api_replay',{})
    total_candidates=sum(value['candidates'] for value in result['dataset'].values())
    lines.extend(['','## 验证','',f"完整回归：{result.get('regression',{}).get('full_final','未记录')}。真实模型API重放{replay.get('states','未记录')} states×{replay.get('variants','未记录')} variants={replay.get('comparisons','未记录')}次比较，mismatch={replay.get('mismatches','未记录')}，新增direction/reference调用={replay.get('new_direction_or_reference_calls','未记录')}。SQLite核验：{result.get('regression',{}).get('sqlite_integrity','未记录')}。当前candidate总数={total_candidates}。",''])
    REPORT.write_text('\n'.join(lines),encoding='utf-8')
    base=result['offline']['BASELINE']['C4']['ALL'];best=result['best_descriptive_offline_variant']
    print(f"Phase4-1B: {result['phase_status']}\nRepository cleanup: completed\nMLP dataset: TRAIN states={result['dataset']['MLP_TRAIN']['states']} / DEV states={result['dataset']['MLP_DEV']['states']}\nBest offline variant: {best}\nBaseline C4 capture: {percent(base['capture_median'])}\nBest MLP C4 capture: {percent(result['offline'][best]['C4']['ALL']['capture_median'])}\n60s development: {native.get('status')}\nRecommended ranker: {result['recommended_ranker']} (M192)\nGAT next: NOT YET\nNext: {result['next_step']}",flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('train','offline','smoke','finish'));parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.action=='train': train()
    elif args.action=='offline': offline()
    elif args.action=='smoke': smoke(args.workers)
    else: finish()
