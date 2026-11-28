"""Fixed Euclidean atom experiment; original budgets, data and NATIVE evaluator."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import io
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
os.environ['NUMBA_NUM_THREADS'] = '1'

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
TEACHER_RESULTS = ROOT/'results/native_priority/scale3000_teachers'
TRAIN_DATA = ROOT/'data/weld_layout'
BASE_SEED = 20261110
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


def result_paths(method, directory):
    """The primary CSV and ordinary size-limited continuation files."""
    path = Path(directory)/f'{method}.csv'
    return ([path] if path.exists() else []) + sorted(path.parent.glob(f'{method}_[0-9][0-9][0-9].csv'))


def read_rows(method, *, include_earlier_adapters=False, historical=False, directory=None):
    directory = directory if directory is not None else (HISTORICAL_RESULTS if historical else RESULTS)
    csv.field_size_limit(LIMIT)
    rows = []
    for path in result_paths(method, directory):
        saved = path.read_bytes()
        # JSON strings contain escaped newlines. Ignore a partially written
        # last physical row; append_row removes it before continuing.
        if saved and not saved.endswith(b'\n'):
            saved = saved[:saved.rfind(b'\n')+1]
        text = saved.decode('utf-8')
        rows.extend(csv.DictReader(io.StringIO(text)))
    if method in ('wag_vns','hga') and not include_earlier_adapters:
        adapter = WAG_INPUT_ADAPTER if method == 'wag_vns' else HGA_INPUT_ADAPTER
        selected = []
        for row in rows:
            try:
                if json.loads(row.get('diagnostic') or '{}').get('input_adapter') == adapter:
                    selected.append(row)
            except (ValueError, TypeError, AttributeError):
                continue
        rows = selected
    return rows


def append_row(method, row, *, directory=None):
    directory = RESULTS if directory is None else Path(directory)
    paths = result_paths(method, directory)
    path = paths[-1] if paths else directory/f'{method}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with path.open('r+b') as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            if end:
                stream.seek(end-1)
                if stream.read(1) != b'\n':
                    # Only the interrupted final line is discarded.
                    offset = end
                    while offset:
                        start = max(0, offset-65536)
                        stream.seek(start)
                        chunk = stream.read(offset-start)
                        newline = chunk.rfind(b'\n')
                        if newline >= 0:
                            stream.truncate(start+newline+1)
                            break
                        offset = start
                    else:
                        stream.truncate(0)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    if not path.exists() or not path.stat().st_size:
        writer.writeheader()
    writer.writerow(row)
    text = buffer.getvalue()
    if (path.stat().st_size if path.exists() else 0)+len(text.encode('utf-8')) >= LIMIT:
        if directory != TEACHER_RESULTS:
            raise ValueError(f'result file would exceed 100MB: {path}')
        number = int(path.stem.rsplit('_',1)[1])+1 if path.stem != method else 2
        if number > 999:
            raise ValueError('too many result continuation files')
        path = directory/f'{method}_{number:03d}.csv'
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        writer.writeheader(); writer.writerow(row)
        text = buffer.getvalue()
        if len(text.encode('utf-8')) >= LIMIT:
            raise ValueError(f'a single result row would exceed 100MB: {path}')
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(text); stream.flush()


def _train_file(path):
    layouts = {}
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            name = row['layout_id']
            if name in layouts:
                raise ValueError(f'duplicate TRAIN layout: {name} in {path}')
            if row.get('split', 'train') != 'train':
                raise ValueError(f'non-TRAIN layout in TRAIN input: {name}')
            parents = tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end'])) for p in row['parents'])
            layouts[name] = generate_taskset(parents, CONFIG)
    return layouts


def scale3000_tasksets():
    """Read the two existing TRAIN files, using the unchanged fixed splitter."""
    original = _train_file(TRAIN_DATA/'train.jsonl')
    expanded = _train_file(TRAIN_DATA/'scale1000/train.jsonl')
    if len(original) != 40 or len(expanded) != 3000:
        raise ValueError(f'expected 40 original and 3000 expanded TRAIN layouts, got {len(original)}/{len(expanded)}')
    for name, taskset in original.items():
        if name not in expanded or expanded[name] != taskset:
            raise ValueError(f'scale TRAIN differs from or omits original layout: {name}')
    return list(expanded.items()), set(original)


def hga_training_selection(tasksets, original_ids):
    """Keep the previous 1000 layouts and add 100 from each size stratum."""
    by_name = dict(tasksets)
    if len(tasksets) != len(by_name) or len(by_name) != 3000 or len(original_ids) != 40 or not original_ids <= by_name.keys():
        raise ValueError('HGA selection requires 3000 distinct TRAIN layouts including the 40 originals')
    ranked = sorted(tasksets, key=lambda item: (len(item[1].tasks), item[0]))
    chosen = set(original_ids)
    for layer in range(10):
        group = [name for name,_ in ranked[layer*len(ranked)//10:(layer+1)*len(ranked)//10]
                 if name not in original_ids]
        if len(group) < 196:
            raise ValueError(f'HGA stratum {layer+1} has fewer than 196 non-original layouts')
        # Preserve the previous 96 midpoint samples, including completed runs.
        previous = {group[(2*i+1)*len(group)//192] for i in range(96)}
        remaining = [name for name in group if name not in previous]
        chosen.update(previous)
        chosen.update(remaining[(2*i+1)*len(remaining)//200] for i in range(100))
    if len(chosen) != 2000:
        raise ValueError(f'expected 2000 distinct HGA layouts, got {len(chosen)}')
    return chosen


def complete_budget(row, method):
    try:
        return (row.get('split') == 'train'
                and row.get('method', method) == method
                and row.get('status') in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')
                and row.get('termination_reason') == 'ITERATION_LIMIT'
                and int(row.get('iterations') or -1) == (500 if method == 'wag_vns' else 200000))
    except (ValueError, TypeError):
        return False


def feasible_teacher(row, method):
    """Saved native objective and complete four-robot genes; no evaluation."""
    if not complete_budget(row, method) or row.get('status') != 'FEASIBLE':
        return False
    try:
        value = float(row.get('native_cmax_s') or 'nan')
        payload = json.loads(row.get('solution') or '{}')
        routes, directions = payload['routes'], payload['directions']
        flat = [job for route in routes for job in route]
        return (math.isfinite(value) and value > 0 and len(routes) == len(directions) == 4
                and bool(flat) and len(flat) == len(set(flat))
                and all(isinstance(job,str) and job for job in flat)
                and all(isinstance(route,list) and isinstance(bits,list) and len(route) == len(bits)
                        and all(type(bit) is int and bit in (0,1) for bit in bits)
                        for route,bits in zip(routes,directions)))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def best_completed_rows(method, rows, targets):
    selected = {}
    for row in rows:
        name = row.get('layout_id')
        if name not in targets or not complete_budget(row, method):
            continue
        previous = selected.get(name)
        # A feasible saved native teacher wins across seeds and sources.
        def score_of(item):
            if feasible_teacher(item, method):
                return (0,float(item['native_cmax_s']))
            return (2 if item['status']=='FEASIBLE' else 1,0.)
        score = score_of(row)
        previous_score = score_of(previous) if previous is not None else (3,0.)
        if previous is None or score < previous_score:
            selected[name] = row
    return selected


def teacher_coverage(method, targets):
    """Coverage by layout, independently of seed; DEADLOCK is complete."""
    # Recent native runs are also history: do not repeat the completed probe.
    history = [*read_rows(method, historical=True), *read_rows(method)]
    fresh = read_rows(method, directory=TEACHER_RESULTS)
    historical = best_completed_rows(method, history, targets)
    completed = best_completed_rows(method, [*history,*fresh], targets)
    counts = {status:sum(row['status'] == status for row in completed.values())
              for status in ('FEASIBLE','DEADLOCK','INFEASIBLE','NUMERIC_FAILURE')}
    failed = {row.get('layout_id') for row in [*history,*fresh]
              if row.get('layout_id') in targets and not complete_budget(row,method)} - completed.keys()
    summary = dict(method=method,target_layouts=len(targets),historical_reused=len(historical),
                   new_completed=len(completed.keys()-historical.keys()),complete_budget=len(completed),
                   **counts,valid_feasible_teachers=sum(feasible_teacher(r,method) for r in completed.values()),
                   error_or_incomplete=len(failed),missing=sorted(targets-completed.keys()))
    keys = [(r['layout_id'],r.get('seed')) for r in fresh
            if r.get('layout_id') in targets and complete_budget(r,method)]
    from collections import Counter
    summary['duplicate_complete_keys'] = [list(k) for k,n in Counter(keys).items() if n>1]
    return completed, summary


def scale3000_teachers(methods=('wag_vns','hga'), *, workers=56, check_only=False):
    if workers < 1:
        raise ValueError('workers must be positive')
    tasksets, originals = scale3000_tasksets()
    by_name = dict(tasksets)
    seeds = {name:BASE_SEED+i for i,name in enumerate(sorted(by_name))}
    targets = dict(wag_vns=set(by_name), hga=hga_training_selection(tasksets,originals))
    summaries = []
    for method in methods:  # Sequential batches, each with single-thread workers.
        completed, before = teacher_coverage(method, targets[method])
        before['original_completed'] = len(originals & completed.keys())
        before['new_runs_this_invocation'] = 0
        pending = [(method,'train',name,by_name[name],seeds[name]) for name in sorted(targets[method]-completed.keys())]
        print('TEACHER_BATCH',json_text(dict(**before,pending=len(pending),workers=workers)),flush=True)
        if check_only:
            summaries.append(before)
            continue
        errors = []
        def save(row):
            append_row(method,row,directory=TEACHER_RESULTS)
            if not complete_budget(row,method):
                errors.append(row['layout_id'])
        if workers == 1:
            for item in pending:
                save(baseline_job(item,purpose='scale3000_teachers'))
        elif pending:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(baseline_job,item,purpose='scale3000_teachers'):item for item in pending}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        row = future.result()
                    except Exception as error:
                        row = dict(method=method,split='train',layout_id=item[2],seed=item[4],status='ERROR',
                            termination_reason='ERROR',diagnostic=json_text(dict(error=repr(error),
                            input_adapter=WAG_INPUT_ADAPTER if method=='wag_vns' else HGA_INPUT_ADAPTER)))
                    save(row)
        completed, after = teacher_coverage(method, targets[method])
        after['original_completed'] = len(originals & completed.keys())
        after['new_runs_this_invocation'] = len(pending)
        print('TEACHER_BATCH_DONE',json_text(after),flush=True)
        summaries.append(after)
        if errors or after['missing']:
            raise RuntimeError(f'{method}: incomplete teacher coverage; rerun to retry {sorted(set(errors)|set(after["missing"]))}')
    return summaries


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
    parser.add_argument('stage',choices=('prepare','baselines','benchmark','summary','scale3000-teachers'))
    parser.add_argument('--workers',type=int,default=56)
    parser.add_argument('--method',choices=('wag_vns','hga','both'),default='both')
    parser.add_argument('--split',choices=('train','validation','test','all'))
    parser.add_argument('--layout',default='train_03')
    parser.add_argument('--seed',type=int,default=20261111)
    parser.add_argument('--check-only',action='store_true',help='TRAIN coverage/selection only; no baseline runs or writes')
    args = parser.parse_args()
    if args.stage=='scale3000-teachers':
        if args.split not in (None,'train'):
            parser.error('scale3000-teachers reads TRAIN only')
        scale3000_teachers(('wag_vns','hga') if args.method=='both' else (args.method,),
                           workers=args.workers,check_only=args.check_only)
        return
    if args.check_only:
        parser.error('--check-only is supported by scale3000-teachers')
    if args.stage=='benchmark':
        benchmark(('wag_vns','hga') if args.method=='both' else (args.method,),args.layout,args.seed)
    if args.stage in ('prepare','all'): prepare()
    if args.stage in ('baselines','all'):
        baselines(('wag_vns','hga') if args.method == 'both' else (args.method,),
                  ('train','validation','test') if args.split in (None,'all') else (args.split,),args.workers)
    if args.stage in ('summary','all'):
        update_report()
        print(REPORT.read_text(encoding='utf-8').split('复现命令')[0],flush=True)


if __name__ == '__main__':
    main()
