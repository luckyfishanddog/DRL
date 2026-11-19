"""Fixed RC-IG experiments; append to the existing SAEAX results table."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from run_saeax import (configure_cpu, load_layouts, read_rows, append_result,
    paired_summary, baseline_rows, SEEDS)
configure_cpu()
from mrta_search.rc_ig import RCIGConfig, run_rc_ig, solution_record
from mrta_search.saeax import SAEAXConfig

SOURCE = 'RC_IG_V1'
PILOTS = ('train_15', 'train_34', 'train_28')


def method_rows(stage):
    return [r for r in read_rows() if r['stage'] == stage and r['source'] == SOURCE]


def in_run_pairs(rows):
    references = {}
    for row in rows:
        initial = json.loads(row['diagnostic']).get('initial_saeax')
        if initial is not None:
            references[row['layout_id'], int(row['seed'])] = dict(status=initial['status'],
                cmax_s=initial['cmax_s'], runtime_s=initial['runtime_s'])
    return paired_summary(rows, references)


def pilot_decision():
    rows = method_rows('pilot')
    expected = {(name, SEEDS[0]) for name in PILOTS}
    complete = (len(rows) == 3 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
        and all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows))
    paired = in_run_pairs(rows)
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    return dict(complete=complete, feasible=feasible, paired_in_run_saeax=paired,
        passed=complete and feasible == 3 and paired['pairs'] == 3 and paired['wins'] >= 2)


def validation_decision():
    rows = method_rows('validation')
    expected = {(name, seed) for name, _ in load_layouts('validation') for seed in SEEDS}
    complete = (len(rows) == 30 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
        and all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows))
    paired = {m: paired_summary(rows, baseline_rows(m)) for m in ('WAG+VNS', 'HGA')}
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    quality = all(p['baseline_runs'] == 30 and p['pairs'] == p['baseline_feasible'] > 0
        and p['median_cmax_change_pct'] < 0 and p['wins'] > p['losses'] for p in paired.values())
    time_limit = paired['WAG+VNS']['baseline_runtime_median_s']
    return dict(complete=complete, feasible=feasible, paired=paired, runtime_median_s=runtime,
        passed=complete and feasible == 30 and quality and runtime <= time_limit)


def experiment_job(job):
    from numba import set_num_threads
    set_num_threads(1)
    stage, (name, parents), seed = job
    print('RC_IG_START', stage, name, len(parents), seed, flush=True)
    def report(row):
        print('RC_IG_PROGRESS', name, seed, json.dumps(row), flush=True)
    started = time.perf_counter()
    try:
        result = run_rc_ig(parents, seed=seed, progress_callback=report)
    except Exception as error:
        traceback.print_exc()
        return dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
            status='ERROR', runtime_s=time.perf_counter()-started, source=SOURCE,
            termination_reason='ERROR', diagnostic=json.dumps(dict(settings=asdict(RCIGConfig()),
                error=f'{type(error).__name__}: {error}'), ensure_ascii=False))
    a, initial = result.accounting, result.initial_result
    initial_schedule = initial.schedule
    diagnostic = dict(settings=asdict(RCIGConfig()), accounting=a, iterations=result.iterations,
        progress=result.progress, accepted_actions=result.accepted_actions,
        final_candidate=result.final_candidate, solution=solution_record(result.solution, result.directions),
        initial_saeax=dict(settings=asdict(SAEAXConfig()), status=initial.status, cmax_s=initial.cmax,
            native_cmax_s=initial.native_cmax, runtime_s=initial.runtime, generations=initial.generations,
            valid_children=initial.valid_children, termination_reason=initial.termination_reason,
            accounting=initial.accounting, progress=initial.progress, final_candidates=initial.final_candidates,
            solution=solution_record(initial.solution, initial.directions),
            b32_result=(dict(status=initial_schedule.status.value, cmax_s=initial_schedule.cmax,
                robot_completion=initial_schedule.robot_completion, directions=initial_schedule.directions,
                diagnostics=initial_schedule.diagnostics) if initial_schedule is not None else None)))
    loads = a.get('processing_s')
    # Generic CSV slots are documented: generations is empty; valid_children
    # counts RC-IG's new native evaluations. Actual outer iterations are above.
    row = dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed, status=result.status,
        cmax_s=result.cmax, native_cmax_s=result.native_cmax, runtime_s=result.runtime,
        generations='', valid_children=a['ig_native_calls'], native_calls=a['native_calls'],
        b32_calls=a['b32_calls'], termination_reason=result.termination_reason,
        max_processing_s=max(loads) if loads else None, move_s=a.get('move_s'), wait_s=a.get('wait_s'),
        source=SOURCE, diagnostic=json.dumps(diagnostic, ensure_ascii=False, allow_nan=False))
    if loads:
        row.update({f'processing_r{r}_s': p for r, p in enumerate(loads)})
    print('RC_IG_RESULT', json.dumps({k: v for k, v in row.items() if k != 'diagnostic'}), flush=True)
    return row


def summarize(stage):
    rows = method_rows(stage)
    if not rows:
        return None
    accounts = [json.loads(r['diagnostic']).get('accounting', {}) for r in rows]
    def timing(key):
        values = [a[key] for a in accounts if key in a]
        return median(values) if values else None
    summary = dict(stage=stage, runs=len(rows), feasible=sum(r['status'] == 'FEASIBLE' for r in rows),
        runtime_median_s=median(float(r['runtime_s']) for r in rows),
        runtime_max_s=max(float(r['runtime_s']) for r in rows),
        initialization_median_s=timing('initialization_s'), ig_search_median_s=timing('ig_search_s'),
        native_evaluations=sum(a.get('ig_native_calls', 0) for a in accounts),
        iterations=sum(a.get('iterations', 0) for a in accounts),
        final_fallbacks=sum(a.get('final_fallback', False) for a in accounts),
        in_run_SAEAX=in_run_pairs(rows),
        paired={m: paired_summary(rows, baseline_rows(m)) for m in ('WAG+VNS', 'HGA')})
    summary['decision'] = pilot_decision() if stage == 'pilot' else validation_decision() if stage == 'validation' else None
    print('RC_IG_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    for r, a in sorted(zip(rows, accounts), key=lambda row: (row[0]['layout_id'], row[0]['seed'])):
        print('RC_IG_ROW', json.dumps(dict(layout_id=r['layout_id'], seed=r['seed'], status=r['status'],
            cmax_s=r['cmax_s'], runtime_s=r['runtime_s'], initial_cmax_s=a.get('initial_cmax_s'),
            iterations=a.get('iterations'), ig_native_calls=a.get('ig_native_calls'),
            initial_native_calls=a.get('initial_native_calls'), final_b32_calls=a.get('final_b32_calls'),
            return_reason=a.get('return_reason')), ensure_ascii=False), flush=True)
    return summary


def run_experiment(stage, workers):
    if stage in ('validation', 'test'):
        decision = pilot_decision()
        print('RC_IG_TRAIN_CHECK', json.dumps(decision), flush=True)
        if not decision['passed']:
            raise ValueError('TRAIN screening has not passed; later datasets are not run')
    if stage == 'test':
        decision = validation_decision()
        print('RC_IG_VALIDATION_CHECK', json.dumps(decision), flush=True)
        if not decision['passed']:
            raise ValueError('VALIDATION conditions have not passed; TEST is not run')
    layouts = load_layouts('train' if stage == 'pilot' else stage)
    if stage == 'pilot':
        lookup = dict(layouts)
        jobs = [(stage, (name, lookup[name]), SEEDS[0]) for name in PILOTS]
    else:
        jobs = [(stage, layout, seed) for layout in layouts for seed in SEEDS]
    completed = {(r['layout_id'], int(r['seed'])) for r in method_rows(stage)}
    pending = [j for j in jobs if (j[1][0], j[2]) not in completed]
    print('RC_IG_BATCH', stage, 'completed', len(jobs)-len(pending), 'remaining', len(pending),
        'workers', workers, 'settings', json.dumps(asdict(RCIGConfig())), flush=True)
    started = time.perf_counter()
    if workers == 1:
        for job in pending:
            append_result(experiment_job(job))
    elif pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(experiment_job, job) for job in pending]):
                append_result(future.result())
    print('RC_IG_BATCH_SECONDS', time.perf_counter()-started, flush=True)
    return summarize(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('pilot', 'validation', 'test', 'summary'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.stage == 'summary':
        for stage in ('pilot', 'validation', 'test'):
            summarize(stage)
    else:
        run_experiment(args.stage, args.workers)


if __name__ == '__main__':
    main()
