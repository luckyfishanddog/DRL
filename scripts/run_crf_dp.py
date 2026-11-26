"""Fixed CRF-DP TRAIN pilot, then conditional official VALIDATION; append only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from statistics import median
import time

from run_saeax import (SEEDS, append_result, baseline_rows, configure_cpu,
    load_layouts, paired_summary, read_rows)

configure_cpu()

from mrta_search.crf_dp import CRFDPConfig, run_crf_dp, solution_record
from mrta_search.saeax import SAEAXConfig

SOURCE = 'CRF_DP_V1'
PILOTS = ('train_15', 'train_34', 'train_28')
HISTORY = {'SAEAX': '4R_SAEAX_ADJACENCY_V1', 'BC-ECS': 'BC_ECS_V1', 'RC-IG': 'RC_IG_V1'}
RUNTIME_LIMIT = 103.59


def method_rows(stage):
    rows = [r for r in read_rows() if r['source'] == SOURCE and r['stage'] == stage]
    keys = [(r['layout_id'], int(r['seed'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('ambiguous CRF-DP same-key records')
    return rows


def pilot_references():
    rows = read_rows()
    references = {}
    for name, source in HISTORY.items():
        selected = [r for r in rows if r['source'] == source and r['stage'] == 'pilot'
                    and r['layout_id'] in PILOTS and int(r['seed']) == SEEDS[0]]
        reference = {(r['layout_id'], int(r['seed'])): r for r in selected}
        if len(reference) != len(selected):
            raise ValueError(f'ambiguous historical pilot rows: {name}')
        references[name] = reference
    strongest = {}
    for key in ((name, SEEDS[0]) for name in PILOTS):
        choices = [ref[key] for ref in references.values() if key in ref and ref[key]['status'] == 'FEASIBLE']
        if choices:
            strongest[key] = min(choices, key=lambda r: float(r['cmax_s']))
    return references, strongest


def pilot_decision():
    rows = method_rows('pilot')
    _, strongest = pilot_references()
    pairs = paired_summary(rows, strongest)
    expected = {(name, SEEDS[0]) for name in PILOTS}
    complete = {(r['layout_id'], int(r['seed'])) for r in rows} == expected and len(rows) == 3
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    finished = all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows)
    passed = (complete and finished and feasible == 3 and pairs['pairs'] == 3 and pairs['wins'] >= 2
        and pairs['median_cmax_change_pct'] < 0 and runtime <= RUNTIME_LIMIT)
    return dict(passed=passed, completed=len(rows), feasible=feasible, search_complete=finished,
        strongest_history=pairs, runtime_median_s=runtime, runtime_limit_s=RUNTIME_LIMIT)


def validation_decision():
    rows = method_rows('validation')
    expected = {(name, seed) for name, _ in load_layouts('validation') for seed in SEEDS}
    complete = len(rows) == 30 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
    comparisons = {name: paired_summary(rows, baseline_rows(name)) for name in ('HGA', 'WAG+VNS')}
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    finished = all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows)
    passed = (complete and finished and feasible == 30 and runtime <= RUNTIME_LIMIT and
        all(p['pairs'] == p['baseline_feasible'] and p['baseline_runs'] == 30 and
            p['pairs'] > 0 and p['median_cmax_change_pct'] < 0 and p['wins'] > p['losses']
            for p in comparisons.values()))
    return dict(passed=passed, completed=len(rows), feasible=feasible, search_complete=finished,
        comparisons=comparisons, runtime_median_s=runtime, runtime_limit_s=RUNTIME_LIMIT)


def experiment_job(stage, name, parents, seed):
    from numba import set_num_threads
    set_num_threads(1)
    print('CRF_DP_START', stage, name, len(parents), seed, flush=True)
    def report(row):
        print('CRF_DP_PROGRESS', name, seed, json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    started = time.perf_counter()
    try:
        result = run_crf_dp(parents, seed=seed, progress_callback=report)
        a, initial = result.accounting, result.initial_result
        detail = dict(settings=asdict(CRFDPConfig()), accounting=a, rounds=result.rounds,
            progress=result.progress, accepted_actions=result.accepted_actions,
            final_candidate=result.final_candidate, solution=solution_record(result.solution, result.directions),
            initial_saeax=dict(settings=asdict(SAEAXConfig()), status=initial.status, cmax_s=initial.cmax,
                native_cmax_s=initial.native_cmax, runtime_s=initial.runtime,
                generations=initial.generations, valid_children=initial.valid_children,
                termination_reason=initial.termination_reason, accounting=initial.accounting,
                progress=initial.progress, final_candidates=initial.final_candidates,
                solution=solution_record(initial.solution, initial.directions)))
        loads = a.get('processing_s')
        row = dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
            status=result.status, cmax_s=result.cmax, native_cmax_s=result.native_cmax,
            runtime_s=result.runtime, generations='', valid_children=a['crf_native_calls'],
            native_calls=a['native_calls'], coordination_calls=a['coordination_calls'],
            termination_reason=result.termination_reason, max_processing_s=max(loads) if loads else None,
            move_s=a.get('move_s'), wait_s=a.get('wait_s'), source=SOURCE,
            diagnostic=json.dumps(detail, ensure_ascii=False, allow_nan=False))
        if loads:
            row.update({f'processing_r{r}_s': p for r, p in enumerate(loads)})
    except Exception as error:
        row = dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
            status='ERROR', runtime_s=time.perf_counter()-started, source=SOURCE,
            termination_reason='ERROR', diagnostic=json.dumps(dict(error=f'{type(error).__name__}: {error}'),
                ensure_ascii=False))
        print('CRF_DP_ERROR', name, row['diagnostic'], flush=True)
    print('CRF_DP_RESULT', json.dumps({k: v for k, v in row.items() if k != 'diagnostic'},
        ensure_ascii=False, allow_nan=False), flush=True)
    return row


def summarize(stage):
    rows = method_rows(stage)
    if stage == 'pilot':
        references, strongest = pilot_references()
        comparisons = {name: paired_summary(rows, reference) for name, reference in references.items()}
        comparisons['strongest_same_key'] = paired_summary(rows, strongest)
        decision = pilot_decision()
    else:
        comparisons = {name: paired_summary(rows, baseline_rows(name)) for name in ('HGA', 'WAG+VNS')}
        decision = validation_decision()
    print('CRF_DP_SUMMARY', json.dumps(dict(stage=stage, comparisons=comparisons, decision=decision),
        ensure_ascii=False, allow_nan=False), flush=True)


def run_experiment(stage):
    if stage == 'validation':
        decision = pilot_decision()
        print('CRF_DP_TRAIN_CHECK', json.dumps(decision, ensure_ascii=False), flush=True)
        if not decision['passed']:
            raise ValueError('CRF-DP TRAIN failed; no VALIDATION is run')
    if stage == 'pilot':
        layouts = dict(load_layouts('train'))
        jobs = [(name, layouts[name], SEEDS[0]) for name in PILOTS]
    else:
        jobs = [(name, parents, seed) for name, parents in load_layouts('validation') for seed in SEEDS]
    completed = {(r['layout_id'], int(r['seed'])) for r in method_rows(stage)}
    print('CRF_DP_BATCH', stage, 'settings', json.dumps(asdict(CRFDPConfig())), 'workers', 1, flush=True)
    started = time.perf_counter()
    for name, parents, seed in jobs:
        if (name, seed) not in completed:
            append_result(experiment_job(stage, name, parents, seed))
    print('CRF_DP_BATCH_SECONDS', time.perf_counter()-started, flush=True)
    summarize(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'validation', 'summary'))
    parser.add_argument('--split', choices=('pilot', 'validation'), default='pilot')
    args = parser.parse_args()
    if args.command == 'summary':
        summarize(args.split)
    else:
        run_experiment(args.command)


if __name__ == '__main__':
    main()
