"""Fixed LA-4R-EAX-NSGA-II TRAIN screening, then conditional formal splits."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from run_saeax import configure_cpu, load_layouts, read_rows, append_result, paired_summary, baseline_rows, SEEDS, SOURCE as SAEAX_SOURCE
configure_cpu()
from mrta_search.eax_nsga2 import EAXNSGAConfig, run_eax_nsga2

SOURCE = 'LA_4R_EAX_NSGA2_V1'
PILOTS = ('train_15','train_34','train_28')


def experiment_job(job):
    from numba import set_num_threads
    set_num_threads(1)
    stage,(name,parents),seed = job
    print('EAX_START',stage,name,len(parents),seed,flush=True)
    def report(row):print('EAX_PROGRESS',name,seed,json.dumps(row),flush=True)
    started = time.perf_counter()
    try:
        result = run_eax_nsga2(parents,seed=seed,progress_callback=report)
    except Exception as error:
        row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,
            status='ERROR',runtime_s=time.perf_counter()-started,source=SOURCE,termination_reason='ERROR',
            diagnostic=json.dumps(dict(settings=asdict(EAXNSGAConfig()),
                error=f'{type(error).__name__}: {error}'),ensure_ascii=False))
        print('EAX_ERROR',name,seed,row['diagnostic'],flush=True)
        return row
    a = result.accounting
    detail = dict(settings=asdict(EAXNSGAConfig()),accounting=a,progress=result.progress,
        final_candidates=result.final_candidates,
        threshold_rule='Q75_OF_ACTIVE_PARENTAL_DIRECTED_MOVE_DISTANCES',
        adjacency_filter_rule='GAO_4_4_SHORT_SIDE_UNLESS_BOTH_LONG',
        final_coordination_policy='ONE_CANDIDATE_ONE_CALL_NO_FALLBACK')
    if result.solution is not None:
        detail['solution'] = dict(patterns=[dict(parent_id=p.parent_id,kind=p.kind.value,t=p.t,
            point_id=p.point_id,rail=p.rail.value if p.rail else None) for p in result.solution.patterns],
            routes=[list(r.block_ids) for r in result.solution.routes],directions=result.directions)
    loads = a.get('processing_s')
    row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,status=result.status,
        cmax_s=result.cmax,native_cmax_s=result.native_cmax,runtime_s=result.runtime,
        generations=result.generations,valid_children=result.valid_children,native_calls=a['native_calls'],
        coordination_calls=a['coordination_calls'],termination_reason=result.termination_reason,
        max_processing_s=max(loads) if loads else None,move_s=a.get('move_s'),wait_s=a.get('wait_s'),
        source=SOURCE,diagnostic=json.dumps(detail,ensure_ascii=False,allow_nan=False))
    if loads:row.update({f'processing_r{r}_s':p for r,p in enumerate(loads)})
    print('EAX_RESULT',json.dumps({k:v for k,v in row.items() if k!='diagnostic'},ensure_ascii=False),flush=True)
    return row


def method_rows(stage):
    return [r for r in read_rows() if r['stage']==stage and r['source']==SOURCE]


def saeax_rows(stage):
    return {(r['layout_id'],int(r['seed'])):r for r in read_rows()
            if r['stage']==stage and r['source']==SAEAX_SOURCE}


def pilot_decision():
    rows = method_rows('pilot')
    expected = {(name,SEEDS[0]) for name in PILOTS}
    paired = paired_summary(rows,saeax_rows('pilot'))
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    complete = (len(rows)==3 and {(r['layout_id'],int(r['seed'])) for r in rows}==expected and
        all(json.loads(r['diagnostic']).get('accounting',{}).get('search_complete',False) for r in rows))
    feasible = sum(r['status']=='FEASIBLE' for r in rows)
    return dict(complete=complete,feasible=feasible,paired_saeax=paired,runtime_median_s=runtime,
        passed=(complete and feasible==3 and paired['pairs']==3 and paired['wins']>=2 and runtime<=103.59))


def validation_decision():
    rows = method_rows('validation')
    expected = {(name,seed) for name,_ in load_layouts('validation') for seed in SEEDS}
    paired = {method:paired_summary(rows,baseline_rows(method)) for method in ('WAG+VNS','HGA')}
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    complete = (len(rows)==30 and {(r['layout_id'],int(r['seed'])) for r in rows}==expected and
        all(json.loads(r['diagnostic']).get('accounting',{}).get('search_complete',False) for r in rows))
    feasible = sum(r['status']=='FEASIBLE' for r in rows)
    quality = all(p['baseline_runs']==30 and p['pairs']==p['baseline_feasible'] and
        p['median_cmax_change_pct'] is not None and p['median_cmax_change_pct']<0 and p['wins']>p['losses']
        for p in paired.values())
    return dict(complete=complete,feasible=feasible,paired=paired,runtime_median_s=runtime,
        passed=(complete and feasible==30 and quality and runtime<=103.59))


def summarize(stage):
    rows = method_rows(stage)
    if not rows:return None
    feasible = [r for r in rows if r['status']=='FEASIBLE']
    accounts = [json.loads(r['diagnostic']).get('accounting',{}) for r in rows]
    def timings(key):
        values = [a[key] for a in accounts if key in a]
        return median(values) if values else None
    summary = dict(stage=stage,runs=len(rows),feasible=len(feasible),
        statuses={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
        cmax_median_s=median(float(r['cmax_s']) for r in feasible) if feasible else None,
        cmax_mean_s=mean(float(r['cmax_s']) for r in feasible) if feasible else None,
        runtime_median_s=median(float(r['runtime_s']) for r in rows),
        runtime_mean_s=mean(float(r['runtime_s']) for r in rows),
        search_median_s=timings('search_s'),coordination_median_s=timings('coordination_s'),
        SAEAX=paired_summary(rows,saeax_rows(stage)))
    for method in ('WAG+VNS','HGA'):summary[method] = paired_summary(rows,baseline_rows(method))
    if stage=='pilot':summary['decision'] = pilot_decision()
    elif stage=='validation':summary['decision'] = validation_decision()
    print('EAX_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    for r,a in sorted(zip(rows,accounts),key=lambda ra:(ra[0]['layout_id'],ra[0]['seed'])):
        print('EAX_ROW',json.dumps({k:r[k] for k in ('layout_id','seed','status','cmax_s',
            'max_processing_s','move_s','wait_s','runtime_s','generations','valid_children',
            'native_calls','coordination_calls','termination_reason')},ensure_ascii=False),flush=True)
    return summary


def run_experiment(stage,workers):
    if stage in ('validation','test'):
        decision = pilot_decision()
        print('EAX_TRAIN_CHECK',json.dumps(decision),flush=True)
        if not decision['passed']:raise ValueError('TRAIN screening has not passed; no further split is run')
    if stage=='test':
        decision = validation_decision()
        print('EAX_VALIDATION_CHECK',json.dumps(decision),flush=True)
        if not decision['passed']:raise ValueError('VALIDATION has not passed; TEST is not run')
    layouts = load_layouts('train' if stage=='pilot' else stage)
    if stage=='pilot':
        lookup = dict(layouts)
        jobs = [(stage,(name,lookup[name]),SEEDS[0]) for name in PILOTS]
    else:jobs = [(stage,layout,seed) for layout in layouts for seed in SEEDS]
    completed = {(r['layout_id'],int(r['seed'])) for r in method_rows(stage)}
    pending = [job for job in jobs if (job[1][0],job[2]) not in completed]
    print('EAX_BATCH',stage,'completed',len(jobs)-len(pending),'remaining',len(pending),'workers',workers,
        'settings',json.dumps(asdict(EAXNSGAConfig())),flush=True)
    started = time.perf_counter()
    if workers==1:
        for job in pending:append_result(experiment_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(experiment_job,job) for job in pending]
            for future in as_completed(futures):append_result(future.result())
    print('EAX_BATCH_SECONDS',time.perf_counter()-started,flush=True)
    return summarize(stage)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pilot','validation','test','summary'))
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.stage=='summary':
        for stage in ('pilot','validation','test'):summarize(stage)
    else:run_experiment(args.stage,args.workers)
