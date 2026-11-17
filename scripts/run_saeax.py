"""Run SAEAX or BR-SAEAX with natural stopping and append-only results."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def configure_cpu():
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    # Set the thread-pool ceiling before importing Numba. If this module is
    # imported by an already initialized process, only change its active mask
    # in experiment_job, never Numba's immutable pool ceiling.
    if 'numba' not in sys.modules:
        os.environ.setdefault('NUMBA_NUM_THREADS','1')


configure_cpu()

from mrta_reference.model import ParentWeld
from mrta_search.saeax import BR_REPAIR_OPTIONS, SAEAXConfig, run_saeax

RESULTS = ROOT/'results/saeax_validation.csv'
DATA = ROOT/'data/weld_layout'
SEEDS = (20261110,20261111,20261112)
SOURCE = '4R_SAEAX_ADJACENCY_V1'
METHOD_SOURCES = dict(saeax=SOURCE,br='BR_SAEAX_BOTTLENECK_REINSERT_V1')
FIELDS = ('stage','layout_id','parent_count','seed','status','cmax_s','native_cmax_s',
    'runtime_s','generations','valid_children','native_calls','b32_calls','termination_reason',
    'max_processing_s','processing_r0_s','processing_r1_s','processing_r2_s','processing_r3_s',
    'move_s','wait_s','source','diagnostic')
BASELINES = {
    'WAG+VNS':('wag_vns_results_optimized.csv','PARTIAL_LNS_V1',500),
    'HGA':('hga_results_optimized.csv','ORIENTED_ROUTE_VND_NATIVE_SELECT_V1',200000),
}


def load_layouts(split):
    rows = [json.loads(line) for line in (DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    return [(row['layout_id'],tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end']))
            for p in row['parents'])) for row in rows]


def read_rows(path=RESULTS):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def experiment_job(job):
    from numba import set_num_threads
    set_num_threads(1)
    stage,(name,parents),seed,method = job
    source = METHOD_SOURCES[method]
    print('SAEAX_START',method,stage,name,len(parents),seed,flush=True)
    def report(row):
        print('SAEAX_PROGRESS',name,seed,json.dumps(row,ensure_ascii=False),flush=True)
    started = time.perf_counter()
    try:
        result = run_saeax(parents,seed=seed,progress_callback=report,
                           bottleneck_reinsertion=(method=='br'))
    except Exception as error:
        # Keep technical failures in the experimental denominator and measured
        # wall time; do not turn them into a feasible result or silently rerun.
        row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,
            status='ERROR',runtime_s=time.perf_counter()-started,source=source,
            termination_reason='ERROR',diagnostic=json.dumps(dict(error=f'{type(error).__name__}: {error}',
            settings=asdict(SAEAXConfig())),ensure_ascii=False))
        print('SAEAX_ERROR',name,seed,row['diagnostic'],flush=True)
        return row
    a = result.accounting
    loads = a.get('processing_s')
    detail = dict(settings=asdict(SAEAXConfig()),accounting=a,progress=result.progress,
        final_candidates=result.final_candidates,
        checkpoints_not_reached=[g for g in (100,200,400) if not any(p['generation']==g for p in result.progress)])
    if method == 'br':
        detail['reinsertion_options'] = BR_REPAIR_OPTIONS
        detail['reinsertion_slot'] = 'FIRST_EFFECTIVE_CHILD_PER_GENERATION'
    if result.solution is not None:
        detail['solution'] = dict(patterns=[dict(parent_id=p.parent_id,kind=p.kind.value,t=p.t,
            point_id=p.point_id,rail=p.rail.value if p.rail else None) for p in result.solution.patterns],
            routes=[list(r.block_ids) for r in result.solution.routes],directions=result.directions)
    row = dict(stage=stage,layout_id=name,parent_count=len(parents),seed=seed,status=result.status,
        cmax_s=result.cmax,native_cmax_s=result.native_cmax,runtime_s=result.runtime,
        generations=result.generations,valid_children=result.valid_children,native_calls=a['native_calls'],
        b32_calls=a['b32_calls'],termination_reason=result.termination_reason,
        max_processing_s=max(loads) if loads else None,move_s=a.get('move_s'),wait_s=a.get('wait_s'),
        source=source,diagnostic=json.dumps(detail,ensure_ascii=False,allow_nan=False))
    if loads:
        row.update({f'processing_r{r}_s':p for r,p in enumerate(loads)})
    print('SAEAX_RESULT',json.dumps({k:v for k,v in row.items() if k!='diagnostic'},ensure_ascii=False),flush=True)
    return row


def append_result(row):
    exists = RESULTS.exists()
    with RESULTS.open('a' if exists else 'x',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(stage,workers,method='saeax'):
    configure_cpu()
    if method == 'br' and stage in ('validation','test'):
        decision = pilot_decision()
        print('BR_TRAIN_CHECK',json.dumps(decision,ensure_ascii=False),flush=True)
        if not decision['passed']:
            raise ValueError('BR-SAEAX TRAIN criteria have not passed; no new split is run')
    if stage == 'test':
        decision = validation_decision()
        print('BR_VALIDATION_CHECK',json.dumps(decision,ensure_ascii=False),flush=True)
        if method != 'br' or not decision['passed']:
            raise ValueError('BR-SAEAX complete VALIDATION criteria have not passed; TEST is not run')
    if stage == 'pilot':
        ordered = sorted(load_layouts('train'),key=lambda row:(len(row[1]),row[0]))
        layouts = [ordered[i] for i in (0,len(ordered)//2,len(ordered)-1)]
        jobs = [(stage,layout,SEEDS[0],method) for layout in layouts]
    else:
        jobs = [(stage,layout,seed,method) for layout in load_layouts(stage) for seed in SEEDS]
    existing = read_rows()
    completed = {(r['stage'],r['layout_id'],int(r['seed'])) for r in existing
                 if r['source']==METHOD_SOURCES[method]}
    pending = [job for job in jobs if (job[0],job[1][0],job[2]) not in completed]
    print('SAEAX_BATCH',method,stage,'completed',len(jobs)-len(pending),'remaining',len(pending),
          'workers',workers,'settings',json.dumps(asdict(SAEAXConfig())),flush=True)
    started = time.perf_counter()
    if workers == 1:
        for job in pending:
            append_result(experiment_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(experiment_job,job) for job in pending]
            for future in as_completed(futures):
                append_result(future.result())
    print('SAEAX_BATCH_SECONDS',time.perf_counter()-started,flush=True)
    summarize(stage,method)


def baseline_rows(method):
    filename,source,iterations = BASELINES[method]
    result = {}
    for row in read_rows(ROOT/'results/new_dataset_final'/filename):
        if (row['source'] != source or row['iterations'] != str(iterations)
            or row['status'] not in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')
            or row['b32_calls'] != '1'):
            continue
        if method == 'HGA':
            diagnostic = json.loads(row['diagnostic'])
            if diagnostic.get('phase') != 'FULL' or diagnostic.get('run_status') != 'COMPLETED':
                continue
        key = row['layout_id'],int(row['seed'])
        if key in result:
            raise ValueError(f'ambiguous formal baseline row: {method} {key}')
        result[key] = row
    return result


def paired_summary(rows,reference):
    changes,seconds,wins,ties,losses = [],[],0,0,0
    matched = [reference[(r['layout_id'],int(r['seed']))] for r in rows
               if (r['layout_id'],int(r['seed'])) in reference]
    for row in rows:
        other = reference.get((row['layout_id'],int(row['seed'])))
        if other is None or row['status'] != 'FEASIBLE' or other['status'] != 'FEASIBLE':
            continue
        new,old = float(row['cmax_s']),float(other['cmax_s'])
        delta = new-old
        seconds.append(delta)
        changes.append(100*delta/old)
        wins += int(delta < -1.e-6)
        losses += int(delta > 1.e-6)
        ties += int(abs(delta) <= 1.e-6)
    return dict(pairs=len(changes),wins=wins,ties=ties,losses=losses,
        median_cmax_change_pct=median(changes) if changes else None,
        mean_cmax_change_pct=mean(changes) if changes else None,
        median_cmax_change_s=median(seconds) if seconds else None,
        baseline_runs=len(matched),baseline_feasible=sum(r['status']=='FEASIBLE' for r in matched),
        baseline_runtime_median_s=median(float(r['runtime_s']) for r in matched) if matched else None,
        baseline_runtime_mean_s=mean(float(r['runtime_s']) for r in matched) if matched else None)


def method_rows(stage,method):
    return [r for r in read_rows() if r['stage']==stage and r['source']==METHOD_SOURCES[method]]


def pilot_decision():
    rows,old = method_rows('pilot','br'),method_rows('pilot','saeax')
    reference = {(r['layout_id'],int(r['seed'])):r for r in old}
    expected = {(name,SEEDS[0]) for name in ('train_15','train_34','train_28')}
    actual = {(r['layout_id'],int(r['seed'])) for r in rows}
    pairs = [(r,reference[(r['layout_id'],int(r['seed']))]) for r in rows
             if (r['layout_id'],int(r['seed'])) in reference]
    cmax_wins = sum(r['status']==b['status']=='FEASIBLE' and
                    float(r['cmax_s']) < float(b['cmax_s'])-1.e-8 for r,b in pairs)
    pmax_wins = sum(bool(r['max_processing_s']) and
                    float(r['max_processing_s']) < float(b['max_processing_s'])-1.e-8 for r,b in pairs)
    ratios = [float(r['runtime_s'])/float(b['runtime_s']) for r,b in pairs]
    # Stated before pilot execution: a tripling on any matched TRAIN layout
    # or a median above the requested WAG comparison time is clearly excessive.
    # These are post-run research criteria, never solver wall-clock cutoffs.
    reasonable_time = (len(ratios)==3 and max(ratios)<=3. and
                       median(float(r['runtime_s']) for r in rows)<=103.591607)
    return dict(runs=len(rows),complete=(len(rows)==3 and actual==expected),
        feasible=sum(r['status']=='FEASIBLE' for r in rows),cmax_wins=cmax_wins,pmax_wins=pmax_wins,
        runtime_ratios=ratios,reasonable_time=reasonable_time,
        passed=(len(rows)==3 and actual==expected and len(pairs)==3 and
            all(r['status']=='FEASIBLE' for r in rows) and cmax_wins>=2 and pmax_wins>=2 and reasonable_time))


def validation_decision():
    rows = method_rows('validation','br')
    expected = {(name,seed) for name,_ in load_layouts('validation') for seed in SEEDS}
    pairs = {method:paired_summary(rows,baseline_rows(method)) for method in BASELINES}
    complete = len(rows)==30 and {(r['layout_id'],int(r['seed'])) for r in rows}==expected
    feasible = sum(r['status']=='FEASIBLE' for r in rows)
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    quality = (pairs['HGA']['pairs']==30 and all(pairs[m]['baseline_runs']==30 for m in BASELINES) and
        all(pairs[m]['median_cmax_change_pct'] is not None and pairs[m]['median_cmax_change_pct']<0
            and pairs[m]['wins']>pairs[m]['losses'] for m in BASELINES))
    return dict(complete=complete,feasible=feasible,runtime_median_s=runtime,pairs=pairs,
        passed=(complete and feasible==30 and quality and runtime<=103.59))


def summarize(stage,method='saeax'):
    rows = method_rows(stage,method)
    if not rows:
        return
    feasible = [r for r in rows if r['status']=='FEASIBLE']
    summary = dict(method=method,stage=stage,runs=len(rows),feasible=len(feasible),
        deadlocks=sum(r['status']=='DEADLOCK' for r in rows),errors=sum(r['status']=='ERROR' for r in rows),
        runtime_median_s=median(float(r['runtime_s']) for r in rows),
        runtime_mean_s=mean(float(r['runtime_s']) for r in rows),
        early_stagnation=sum(r['termination_reason']=='STAGNATION' for r in rows),
        cmax_median_s=median(float(r['cmax_s']) for r in feasible) if feasible else None)
    for baseline in BASELINES:
        summary[baseline] = paired_summary(rows,baseline_rows(baseline))
    if rows[0]['source'] == METHOD_SOURCES['br']:
        reference = {(r['layout_id'],int(r['seed'])):r for r in method_rows(stage,'saeax')}
        summary['SAEAX'] = paired_summary(rows,reference)
        summary['reinsertion_successes'] = sum(json.loads(r['diagnostic']).get('accounting',{}).get(
            'reinsertion_successes',0) for r in rows)
        if stage == 'pilot':
            summary['decision'] = pilot_decision()
        elif stage == 'validation':
            summary['decision'] = validation_decision()
    print('SAEAX_SUMMARY',json.dumps(summary,ensure_ascii=False),flush=True)
    for row in sorted(rows,key=lambda r:(r['layout_id'],r['seed'])):
        print('SAEAX_ROW',' | '.join(str(row[k]) for k in ('layout_id','seed','status','cmax_s',
            'runtime_s','generations','valid_children','max_processing_s','move_s','wait_s','termination_reason')))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('pilot','validation','test','summary'))
    parser.add_argument('--method',choices=tuple(METHOD_SOURCES),default='saeax')
    parser.add_argument('--workers',type=int,default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage == 'summary':
        for stage in ('pilot','validation','test'):
            summarize(stage,args.method)
    else:
        run_experiment(args.stage,args.workers,args.method)
