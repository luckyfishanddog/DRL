"""TRAIN pricing fit, paired RCG screening, and conditional formal VALIDATION."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from statistics import median
import time

from run_saeax import (ROOT, RESULTS, FIELDS, SEEDS, baseline_rows,
    configure_cpu, load_layouts, paired_summary, read_rows)

configure_cpu()

import torch
from numba import set_num_threads
from mrta_search.gnn_route_columns import (RCGConfig, fit_pricing_head, load_pricer,
    optimize_route_columns, run_saeax, solution_record)

PILOTS = ('train_30', 'train_13', 'train_36')
FIT_LAYOUTS = ('train_15', 'train_24', 'train_28')
SOURCES = dict(initial='RCG_INITIAL_V1', gnn='GNN_RCG_V1', greedy='GREEDY_RCG_V1')
FIT_SOURCE = 'RCG_PRICING_TRAIN_V1'
ENCODER = ROOT/'models/gnn_weld_layout_final.pt'
EXPOSURE = ('All 40 TRAIN layouts have earlier GNN exposure; selected pilots exclude '
    'the named 20 fine-tuning layouts, six development layouts and SAEAX/BC/RC/CRF pilots. '
    'They are held out from this new pricing-head fit, not historically unseen layouts.')


def method_rows(stage, method='gnn'):
    selected = [r for r in read_rows() if r['source'] == SOURCES[method] and r['stage'] == stage]
    keys = [(r['layout_id'], int(r['seed'])) for r in selected]
    if len(keys) != len(set(keys)):
        raise ValueError(f'ambiguous same-key {method} RCG records')
    return selected


def pilot_decision():
    rows = method_rows('pilot')
    greedy = {(r['layout_id'], int(r['seed'])): r for r in method_rows('pilot', 'greedy')}
    initial = {(r['layout_id'], int(r['seed'])): r for r in method_rows('pilot', 'initial')}
    expected = {(name, SEEDS[0]) for name in PILOTS}
    complete = (len(rows) == 3 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
        and set(greedy) == expected and set(initial) == expected)
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    p_initial, p_greedy = paired_summary(rows, initial), paired_summary(rows, greedy)
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    finished = complete and all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False)
        for r in rows+list(greedy.values()))
    passed = (finished and feasible == 3 and p_initial['pairs'] == 3 and p_initial['wins'] >= 2
        and p_initial['median_cmax_change_pct'] <= -.5 and p_greedy['pairs'] == 3
        and p_greedy['wins'] >= 2 and p_greedy['median_cmax_change_pct'] < 0 and runtime <= 103.59)
    return dict(passed=bool(passed), complete=complete, feasible=feasible, search_complete=finished,
        versus_initial=p_initial, versus_greedy=p_greedy, runtime_median_s=runtime,
        runtime_limit_s=103.59, exposure=EXPOSURE)


def validation_decision():
    rows = method_rows('validation')
    comparisons = {name: paired_summary(rows, baseline_rows(name)) for name in ('HGA', 'WAG+VNS')}
    expected = {(name, seed) for name, _ in load_layouts('validation') for seed in SEEDS}
    complete = len(rows) == 30 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    finished = complete and all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows)
    passed = (finished and feasible == 30 and all(p['baseline_runs'] == 30 and
        p['pairs'] == p['baseline_feasible'] and p['pairs'] > 0 and
        p['median_cmax_change_pct'] < 0 and p['wins'] > p['losses'] for p in comparisons.values()))
    return dict(passed=bool(passed), complete=complete, feasible=feasible, comparisons=comparisons,
        runtime_median_s=median(float(r['runtime_s']) for r in rows) if rows else None)


def audit_train_usage():
    """Read historical evidence, not historical solutions for initialization."""
    evidence = {}
    for path in sorted((ROOT/'results/new_dataset_final').glob('*.csv*')):
        used = set()
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                name = row.get('layout_id', '')
                if name in {f'train_{i:02d}' for i in range(1, 41)}:
                    used.add(name)
        if used:
            evidence[path.name] = sorted(used)
    return dict(exposure=EXPOSURE, truly_unused_train=sorted(set(dict(load_layouts('train')))-
        set().union(*(set(names) for names in evidence.values()))), evidence=evidence)


def historical_references():
    references = {}
    for name in ('HGA', 'WAG+VNS'):
        references[name] = baseline_rows(name)
    allowed = {'4R_SAEAX_ADJACENCY_V1', 'BR_SAEAX_BOTTLENECK_REINSERT_V1',
        'BC_ECS_V1', 'RC_IG_V1', 'CRF_DP_V1'}
    for row in read_rows():
        if row['source'] in allowed:
            references.setdefault(row['source'], {})[row['layout_id'], int(row['seed'])] = row
    return references


def summarize(stage='pilot'):
    rows = method_rows(stage)
    strongest = {}
    for reference in historical_references().values():
        for key, row in reference.items():
            if row['status'] == 'FEASIBLE' and (key not in strongest or float(row['cmax_s']) < float(strongest[key]['cmax_s'])):
                strongest[key] = row
    detail = dict(stage=stage, decision=pilot_decision() if stage == 'pilot' else validation_decision(),
        strongest_official_same_key_history=paired_summary(rows, strongest))
    print('RCG_SUMMARY', json.dumps(detail, ensure_ascii=False, allow_nan=False), flush=True)
    for row in rows:
        a = json.loads(row['diagnostic'])['accounting']
        print('RCG_ROW', json.dumps(dict(layout_id=row['layout_id'], seed=row['seed'], cmax_s=row['cmax_s'],
            runtime_s=row['runtime_s'], initial_cmax_s=a['initial_cmax_s'],
            initial_period_s=max(a['initial_route_periods_s']) if a.get('initial_route_periods_s') else None,
            final_period_s=max(a['final_route_periods_s']) if a.get('final_route_periods_s') else None,
            generated_columns=a['proposed_columns'], added_columns=a['added_columns'],
            lp_calls=a['lp_calls'], milp_calls=a['milp_calls'], coordination_calls=a['coordination_calls'],
            gnn_scored_actions=a['gnn_scored_actions'], gnn_top_choice_differences=a['gnn_top_choice_differences'],
            parent_mode_changes=len(a.get('returned_parent_mode_changes', [])),
            transferred_parents=len(a.get('changes', {})), returned_initial=a.get('returned_initial')),
            ensure_ascii=False, allow_nan=False), flush=True)
    return detail


def _result_row(stage, name, parents, seed, method, result, initial):
    a = result.accounting
    detail = dict(settings=asdict(RCGConfig()), exposure=EXPOSURE, fit_layouts=FIT_LAYOUTS,
        accounting=a, progress=result.progress, final_candidates=result.final_candidates,
        solution=solution_record(result.solution, result.directions),
        initial_saeax=dict(status=initial.status, cmax_s=initial.cmax, runtime_s=initial.runtime,
            native_cmax_s=initial.native_cmax, generations=initial.generations,
            valid_children=initial.valid_children, accounting=initial.accounting,
            solution=solution_record(initial.solution, initial.directions)))
    loads = a.get('processing_s', [])
    row = dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
        status=result.status, cmax_s=result.cmax, native_cmax_s=(initial.native_cmax if result.schedule is initial.schedule else None),
        runtime_s=result.runtime, generations='', valid_children='', native_calls=a['native_calls'],
        coordination_calls=a['coordination_calls'], termination_reason=result.termination_reason, source=SOURCES[method],
        max_processing_s=max(loads) if loads else None, move_s=a.get('move_s'), wait_s=a.get('wait_s'),
        diagnostic=json.dumps(detail, ensure_ascii=False, allow_nan=False))
    row.update({f'processing_r{r}_s': v for r, v in enumerate(loads)})
    return row


def run_experiment(stage):
    if stage == 'validation':
        decision = pilot_decision()
        if not decision['passed']:
            raise ValueError('GNN-RCG TRAIN failed; no VALIDATION is run')
    set_num_threads(1); torch.set_num_threads(1)
    layouts = dict(load_layouts('train' if stage == 'pilot' else 'validation'))
    jobs = [(name, layouts[name], SEEDS[0]) for name in PILOTS] if stage == 'pilot' else [
        (name, parents, seed) for name, parents in layouts.items() for seed in SEEDS]
    gnn_done = {(r['layout_id'], int(r['seed'])) for r in method_rows(stage)}
    greedy_done = {(r['layout_id'], int(r['seed'])) for r in method_rows(stage, 'greedy')}
    initial_done = {(r['layout_id'], int(r['seed'])) for r in method_rows(stage, 'initial')}
    pending = [job for job in jobs if (job[0], job[2]) not in gnn_done or (job[0], job[2]) not in greedy_done]
    if not pending:
        return summarize(stage)
    if any((name, seed) in initial_done for name, _, seed in pending):
        raise RuntimeError('incomplete paired run exists; do not regenerate or reevaluate its initial solution')
    batch_started = time.perf_counter()
    # Open the existing results destination before expensive execution. It is
    # kept open and flushed after each row, avoiding previous lost-run failures.
    with RESULTS.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        def write(row):
            writer.writerow(row); stream.flush()
        policy = load_pricer(ENCODER)
        train = dict(load_layouts('train'))
        def report(row):
            print('RCG_PROGRESS', json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
        fit = fit_pricing_head(policy, [(name, train[name]) for name in FIT_LAYOUTS], progress_callback=report)
        print('RCG_FIT', json.dumps(fit, ensure_ascii=False, allow_nan=False), flush=True)
        write(dict(stage='pricing_fit', layout_id='TRAIN_3', seed=SEEDS[0], source=FIT_SOURCE,
            status='COMPLETED', runtime_s=fit['runtime_s'], native_calls=0, coordination_calls=0,
            diagnostic=json.dumps(fit, ensure_ascii=False, allow_nan=False)))
        for name, parents, seed in pending:
            print('RCG_START', stage, name, len(parents), seed, flush=True)
            t = time.perf_counter()
            initial = run_saeax(parents, seed=seed, progress_callback=lambda row: report(dict(
                phase='SHARED_INITIALIZATION', layout_id=name, **row)))
            init_s = time.perf_counter()-t
            loads = initial.accounting.get('processing_s', [])
            write(dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
                source=SOURCES['initial'], status=initial.status, cmax_s=initial.cmax,
                native_cmax_s=initial.native_cmax, runtime_s=init_s, generations=initial.generations,
                valid_children=initial.valid_children, native_calls=initial.accounting['native_calls'],
                coordination_calls=initial.accounting['coordination_calls'], termination_reason=initial.termination_reason,
                max_processing_s=max(loads) if loads else None,
                diagnostic=json.dumps(dict(solution=solution_record(initial.solution, initial.directions),
                    accounting=initial.accounting, final_candidates=initial.final_candidates),
                    ensure_ascii=False, allow_nan=False)))
            for method, pricer in (('gnn', policy), ('greedy', None)):
                result = optimize_route_columns(parents, initial, policy=pricer,
                    initialization_s=init_s, progress_callback=report)
                row = _result_row(stage, name, parents, seed, method, result, initial)
                write(row)
                print('RCG_RESULT', json.dumps({k: v for k, v in row.items() if k != 'diagnostic'},
                    ensure_ascii=False, allow_nan=False), flush=True)
    print('RCG_BATCH_SECONDS', time.perf_counter()-batch_started, flush=True)
    return summarize(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit', 'summary'))
    parser.add_argument('--split', choices=('pilot', 'validation'), default='pilot')
    args = parser.parse_args()
    if args.command == 'audit':
        print(json.dumps(audit_train_usage(), ensure_ascii=False))
    elif args.command == 'summary':
        summarize(args.split)


if __name__ == '__main__':
    main()
