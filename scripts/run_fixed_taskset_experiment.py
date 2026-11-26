"""Fixed Euclidean atom experiment; original budgets, data and NATIVE evaluator."""
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
sys.path.insert(0, str(ROOT/'src'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('NUMBA_NUM_THREADS','1')

from mrta_reference.fixed_taskset import FixedTaskset, generate_taskset, expand_routes, job_span
from mrta_reference.model import ParentWeld, ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import canonicalize, block_map
from mrta_baselines.wag_vns import AdaptedWAGConfig, run_adapted_wag_vns, priority_schedule
from mrta_baselines.hga import AdaptedHGAConfig, run_adapted_hga

CONFIG = ScientificConfig()
SEEDS = (20261110,20261111,20261112)
HISTORICAL_RESULTS = ROOT/'results/fixed_euclidean_split'
RESULTS = ROOT/'results/native_priority/fixed_euclidean_split'
REPORT = ROOT/'docs/UNIFIED_NATIVE_CMAX_REPORT.md'
LIMIT = 100_000_000
FIELDS = ('split','layout_id','seed','method','status','cmax_s','native_cmax_s','runtime_s',
          'iterations','termination_reason','coordination_calls','parameters','diagnostic','solution')
WAG_INPUT_ADAPTER = 'FIXED_ATOM_WAG_MERGED_ROUTE_NATIVE_FAST'
HGA_INPUT_ADAPTER = 'FIXED_HGA_ROUTE_CYCLE_VND_NATIVE_SELECT'


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',',':'))


def write_text(path, text):
    if len(text.encode('utf-8')) >= LIMIT:
        raise ValueError(f'file would exceed the user 100MB limit: {path}')
    path.write_text(text, encoding='utf-8')


def read_rows(method, *, include_earlier_adapters=False, historical=False):
    path = (HISTORICAL_RESULTS if historical else RESULTS)/f'{method}.csv'
    if not path.exists():
        return []
    csv.field_size_limit(LIMIT)
    import io
    text = path.read_text(encoding='utf-8')
    # WAG/HGA parents may append concurrently to different CSVs. Each JSON
    # field is escaped onto one physical line; ignore a still-writing tail.
    if text and not text.endswith('\n'):
        text = text.rsplit('\n',1)[0]+'\n'
    rows = list(csv.DictReader(io.StringIO(text)))
    if method in ('wag_vns','hga') and not include_earlier_adapters:
        adapter = WAG_INPUT_ADAPTER if method == 'wag_vns' else HGA_INPUT_ADAPTER
        rows = [r for r in rows if json.loads(r['diagnostic']).get('input_adapter') == adapter]
    return rows


def append_row(method, row):
    import io
    path = RESULTS/f'{method}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    if not path.exists():
        writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if (path.stat().st_size if path.exists() else 0)+len(text.encode('utf-8')) >= LIMIT:
        raise ValueError(f'result file would exceed 100MB: {path}')
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text); stream.flush()


def prepare():
    path = HISTORICAL_RESULTS/'tasksets.json'
    if path.exists():
        print('FIXED_TASKSETS_EXIST', len(load_tasksets()), flush=True)
        return
    records, summaries = [], []
    # No alternative layout generation, no TRAIN-expanded or old task catalog.
    for split in ('train','validation','test'):
        data = ROOT/'data/weld_layout'/f'{split}.jsonl'
        for raw in map(json.loads, data.read_text(encoding='utf-8').splitlines()):
            parents = tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in raw['parents'])
            ts = generate_taskset(parents, CONFIG)  # A failure is fatal; no row is dropped.
            records.append(dict(split=split, layout_id=raw['layout_id'], **ts.record()))
            summaries.append(dict(split=split, layout_id=raw['layout_id'], **ts.statistics))
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_text(path, json_text(records))
    with (RESULTS/'taskset_summary.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=tuple(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    for split in ('train','validation','test'):
        rows = [s for s in summaries if s['split'] == split]
        totals = {k:sum(s[k] for s in rows) for k in rows[0] if k not in ('split','layout_id','shortest_task_m')}
        print('FIXED_TASKSET_SUMMARY',split,len(rows),json_text(totals),
              'shortest',min(s['shortest_task_m'] for s in rows),flush=True)
    update_report()


def load_tasksets():
    return [(row['split'],row['layout_id'],FixedTaskset.from_record(row))
            for row in json.loads((HISTORICAL_RESULTS/'tasksets.json').read_text(encoding='utf-8'))]


def solution_record(solution, directions, config=CONFIG):
    if solution is None or directions is None:
        return None
    blocks = block_map(solution,config)
    atoms = expand_routes(solution.routes)
    jobs = []
    for route in solution.routes:
        for identifier, bit in zip(route.block_ids,directions[route.robot_id]):
            b = blocks[identifier]
            jobs.append(dict(job_id=identifier,robot=route.robot_id,source_parent_id=b.parent_id,
                             u_start=b.u_start,u_end=b.u_end,start=b.start,end=b.end,length=b.length,direction=int(bit)))
    return dict(assignments={atom:r for r in range(4) for atom in atoms[r]},
                merged_jobs=jobs,routes=[list(route.block_ids) for route in solution.routes],
                directions=[list(map(int,bits)) for bits in directions])


def restore_solution(taskset, row):
    payload = json.loads(row['solution']) if isinstance(row['solution'],str) else row['solution']
    solution = canonicalize(taskset.parents,taskset.patterns,dict(enumerate(payload['routes'])),CONFIG,scope=FORMAL_SCOPE_V2)
    if {a:r for r,route in enumerate(expand_routes(solution.routes)) for a in route} != payload['assignments']:
        raise ValueError('saved assignment differs from merged route coverage')
    return solution,tuple(tuple(bits) for bits in payload['directions'])


def baseline_job(item, *, purpose='fixed_taskset_experiment'):
    from numba import set_num_threads
    set_num_threads(1)
    method,split,name,ts,seed = item
    print('FIXED_BASELINE_START',method,split,name,len(ts.tasks),seed,flush=True)
    started = time.perf_counter()
    settings = AdaptedWAGConfig() if method == 'wag_vns' else AdaptedHGAConfig(vnd_objective='ROUTE_CYCLE')
    run = run_adapted_wag_vns if method == 'wag_vns' else run_adapted_hga
    kwargs = dict(seed=seed,time_limit=math.inf,scope=FORMAL_SCOPE_V2,checkpoints=(),
                  certify=False,candidate_evaluator=priority_schedule,fixed_taskset=ts)
    row = dict(split=split,layout_id=name,seed=seed,method=method,parameters=json_text(asdict(settings)))
    try:
        result = run(ts.parents,CONFIG,settings,**kwargs)
        row.update(native_cmax_s=result.schedule.cmax if result.schedule and result.schedule.feasible else None,
                   iterations=result.iterations,termination_reason=result.termination_reason,
                   coordination_calls=result.reference_calls)
        schedule = result.schedule
        expected = settings.imax if method == 'wag_vns' else settings.max_iterations
        if result.iterations != expected:
            row['status'] = 'INCOMPLETE'
        elif result.solution is None or schedule is None:
            row['status'] = result.status.value
        else:
            row.update(status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,
                       solution=json_text(solution_record(result.solution,result.directions)))
        row['diagnostic'] = json_text(dict(search_accounting=result.accounting,
                search_diagnostics=result.diagnostics,search_runtime_s=result.actual_runtime,
                initialization_s=result.initialization_time,search_and_precompute_s=result.search_time,
                final_scheduler_extra_calls=0, final_schedule_reused=schedule is not None,
                robot_completion=() if schedule is None else schedule.robot_completion,
                waiting_s=None if schedule is None else schedule.waiting_time,
                evaluation_policy='NATIVE_PRIORITY_FOUR_ROBOTS',
                formal_diagnostics=() if schedule is None else schedule.diagnostics))
    except Exception as error:
        import traceback
        row.update(status='ERROR',termination_reason='ERROR',diagnostic=json_text(dict(error=repr(error),traceback=traceback.format_exc())))
    if method in ('wag_vns','hga'):
        detail = json.loads(row['diagnostic'])
        detail['input_adapter'] = WAG_INPUT_ADAPTER if method == 'wag_vns' else HGA_INPUT_ADAPTER
        detail['purpose'] = purpose
        detail['single_thread'] = True
        detail['cold_jit_included'] = True
        if purpose == 'single_full_budget_timing':
            detail['runtime_context'] = os.environ.get('FIXED_EXPERIMENT_RUNTIME_CONTEXT','')
        row['diagnostic'] = json_text(detail)
    row['runtime_s'] = time.perf_counter()-started
    print('FIXED_BASELINE_RESULT',json_text({k:v for k,v in row.items() if k not in ('solution','diagnostic')}),flush=True)
    return row


def benchmark(methods, layout_id, seed):
    """Exactly one full run per method, serial, each in a fresh process."""
    if len(methods)>1:
        import subprocess
        for method in methods:
            subprocess.run([sys.executable,'-B','-u','-X','utf8',str(Path(__file__).resolve()),
                'benchmark','--method',method,'--layout',layout_id,'--seed',str(seed)],cwd=ROOT,check=True)
        return
    method = methods[0]
    tasksets = [(split,name,ts) for split,name,ts in load_tasksets() if name==layout_id]
    if len(tasksets)!=1 or tasksets[0][0]!='train':
        raise ValueError('the one-case timing run uses an existing TRAIN layout')
    previous = [r for r in read_rows(method) if r['layout_id']==layout_id and int(r['seed'])==seed
                and json.loads(r['diagnostic']).get('purpose')=='single_full_budget_timing'
                and r['termination_reason']=='ITERATION_LIMIT']
    if previous:
        print('FIXED_BENCHMARK_ALREADY_SAVED',method,json_text({k:previous[-1][k] for k in FIELDS if k not in ('solution','diagnostic')}),flush=True)
        return
    split,name,ts = tasksets[0]
    print('FIXED_BENCHMARK_INPUT',json_text(dict(layout_id=name,seed=seed,parents=len(ts.parents),
        atoms=len(ts.tasks),method=method,workers=1,threads=1,stop='500 VNS' if method=='wag_vns' else '200000 valid offspring')),
        flush=True)
    row = baseline_job((method,split,name,ts,seed),purpose='single_full_budget_timing')
    append_row(method,row); update_report()
    if row['termination_reason']!='ITERATION_LIMIT':
        raise RuntimeError(row['diagnostic'])


def baselines(methods=('wag_vns','hga'), splits=('train','validation','test'), workers=4):
    tasksets = load_tasksets()
    for method in methods:
        existing = read_rows(method)
        keys = [(r['split'],r['layout_id'],int(r['seed'])) for r in existing]
        if len(keys) != len(set(keys)):
            raise ValueError(f'duplicate result keys in {method}')
        required = 500 if method == 'wag_vns' else 200000
        done = {key for key,row in zip(keys,existing) if row['termination_reason'] == 'ITERATION_LIMIT'
                and int(row['iterations']) == required}
        failed = set(keys)-done
        if failed:
            raise RuntimeError(f'{method}: saved failed/incomplete runs require diagnosis before retry: {sorted(failed)}')
        pending = [(method,split,name,ts,seed) for split,name,ts in tasksets if split in splits
                   for seed in SEEDS if (split,name,seed) not in done]
        print('FIXED_BASELINE_PENDING',method,len(pending),'workers',workers,flush=True)
        if workers == 1:
            for item in pending:
                row = baseline_job(item); append_row(method,row); update_report()
                if row['termination_reason'] != 'ITERATION_LIMIT':
                    raise RuntimeError(f'baseline stopped before its required budget: {row["diagnostic"]}')
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(baseline_job,item):item for item in pending}
                for future in as_completed(futures):
                    row = future.result(); append_row(method,row); update_report()
                    if row['termination_reason'] != 'ITERATION_LIMIT':
                        for other in futures:
                            other.cancel()
                        raise RuntimeError(f'baseline stopped before its required budget: {row["diagnostic"]}')


def update_report():
    """Print current native-policy summaries; never rewrite an existing report."""
    summaries=[]
    for method in ('wag_vns','hga'):
        for split in ('train','validation','test'):
            rows=[r for r in read_rows(method) if r['split']==split]
            if not rows:continue
            costs=[float(r['cmax_s']) for r in rows if r['status']=='FEASIBLE' and r['cmax_s']]
            summaries.append(dict(method=method,split=split,runs=len(rows),feasible=len(costs),
                cmax_mean_s=mean(costs) if costs else None,
                cmax_median_s=median(costs) if costs else None,
                runtime_median_s=median(float(r['runtime_s']) for r in rows)))
    summary=dict(evaluation_policy='NATIVE_PRIORITY_FOUR_ROBOTS',summaries=summaries)
    print('FIXED_NATIVE_SUMMARY',json_text(summary),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('prepare','baselines','benchmark','summary'))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--method',choices=('wag_vns','hga','both'),default='both')
    parser.add_argument('--split',choices=('train','validation','test','all'),default='all')
    parser.add_argument('--layout',default='train_03')
    parser.add_argument('--seed',type=int,default=20261111)
    args = parser.parse_args()
    if args.stage=='benchmark':
        benchmark(('wag_vns','hga') if args.method=='both' else (args.method,),args.layout,args.seed)
    if args.stage in ('prepare','all'): prepare()
    if args.stage in ('baselines','all'):
        baselines(('wag_vns','hga') if args.method == 'both' else (args.method,),
                  ('train','validation','test') if args.split == 'all' else (args.split,),args.workers)
    if args.stage in ('summary','all'):
        update_report()
        print(REPORT.read_text(encoding='utf-8').split('复现命令')[0],flush=True)


if __name__ == '__main__':
    main()
