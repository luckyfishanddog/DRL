"""Fixed TRAIN experiment for joint arc-MILP set selection; append existing CSV."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from statistics import median
import time

from run_saeax import (ROOT, RESULTS, FIELDS, SEEDS, configure_cpu, load_layouts,
    read_rows, paired_summary, baseline_rows)

configure_cpu()

import torch
from numba import set_num_threads
from mrta_search import gnn_joint_mip as j

FIT_LAYOUTS = ('train_08', 'train_35', 'train_25')
PILOTS = ('train_20', 'train_21', 'train_09')
SOURCES = dict(initial='JN_MIP_INITIAL_V1', rule='RULE_JN_MIP_V1', gnn='GNN_JN_MIP_V1')
LABEL_SOURCE, FIT_SOURCE = 'JN_MIP_B32_LABELS_V1', 'JN_MIP_SET_FIT_V1'
ENCODER = ROOT/'models/gnn_weld_layout_cmax_conflict.pt'
EXPOSURE = ('The frozen base encoder has seen all TRAIN layouts historically. '
    'These fit/screen layouts exclude the named recent development and algorithm pilots; '
    'screen layouts are not used for the new set head.')


def method_rows(stage, method='gnn'):
    rows = [r for r in read_rows() if r['source'] == SOURCES[method] and r['stage'] == stage]
    keys = [(r['layout_id'], int(r['seed'])) for r in rows]
    if len(keys) != len(set(keys)): raise ValueError('ambiguous JN-MIP result keys')
    return rows


def pilot_decision():
    rows = method_rows('pilot')
    initial = {(r['layout_id'], int(r['seed'])): r for r in method_rows('pilot', 'initial')}
    rule = {(r['layout_id'], int(r['seed'])): r for r in method_rows('pilot', 'rule')}
    expected = {(name, SEEDS[0]) for name in PILOTS}
    complete = (len(rows) == 3 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
        and set(initial) == set(rule) == expected)
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    vs_initial, vs_rule = paired_summary(rows, initial), paired_summary(rows, rule)
    runtime = median(float(r['runtime_s']) for r in rows) if rows and all(r.get('runtime_s') for r in rows) else None
    ratios = [float(r['runtime_s'])/float(rule[r['layout_id'], int(r['seed'])]['runtime_s']) for r in rows
        if r.get('runtime_s') and (r['layout_id'], int(r['seed'])) in rule]
    finished = complete and all(json.loads(r['diagnostic']).get('accounting', {}).get('search_complete', False) for r in rows)
    passed = (finished and feasible == 3 and vs_initial['pairs'] == 3 and vs_initial['wins'] >= 2
        and vs_initial['median_cmax_change_pct'] <= -.5 and vs_rule['pairs'] == 3
        and vs_rule['wins'] >= 2 and vs_rule['median_cmax_change_pct'] < 0
        and runtime is not None and runtime <= 103.59 and len(ratios) == 3 and max(ratios) <= 3.)
    return dict(passed=bool(passed), complete=complete, feasible=feasible, versus_initial=vs_initial,
        versus_rule=vs_rule, runtime_median_s=runtime, runtime_limit_s=103.59,
        maximum_runtime_ratio_to_rule=max(ratios) if ratios else None, exposure=EXPOSURE)


def summarize(stage='pilot'):
    rows = method_rows(stage)
    if stage == 'pilot': decision = pilot_decision()
    else:
        comparisons = {name: paired_summary(rows, baseline_rows(name)) for name in ('HGA', 'WAG+VNS')}
        feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
        decision = dict(completed=len(rows), feasible=feasible, comparisons=comparisons,
            passed=len(rows) == feasible == 30 and all(p['pairs'] == p['baseline_feasible'] and p['pairs'] > 0
                and p['wins'] > p['losses'] and p['median_cmax_change_pct'] < 0 for p in comparisons.values()))
    print('JN_MIP_SUMMARY', json.dumps(dict(stage=stage, decision=decision), ensure_ascii=False, allow_nan=False), flush=True)
    for method in ('rule', 'gnn'):
        for r in method_rows(stage, method):
            d = json.loads(r['diagnostic']); a = d.get('accounting', {})
            print('JN_MIP_ROW', json.dumps(dict(layout_id=r['layout_id'], method=method, status=r['status'],
                cmax_s=r['cmax_s'], runtime_s=r['runtime_s'], initial_cmax_s=a.get('initial_cmax_s'),
                milp_calls=a.get('milp_calls'), new_b32_calls=a.get('b32_calls'),
                accepted=a.get('accepted'), changes=a.get('changes'), reason=r['termination_reason']),
                ensure_ascii=False, allow_nan=False), flush=True)
    return decision


def result_row(stage, name, parents, seed, method, result, initial):
    plan, a = result.plan, result.accounting
    return dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
        status=plan.schedule.status.value, cmax_s=plan.schedule.cmax, runtime_s=result.runtime,
        native_cmax_s=initial.native_cmax if a['returned_initial'] else None,
        native_calls=initial.accounting['native_calls'],
        b32_calls=initial.accounting['b32_calls']+a['b32_calls'],
        source=SOURCES[method], termination_reason=result.reason,
        max_processing_s=float(max(plan.loads)), move_s=float(sum(plan.costs-plan.loads)),
        wait_s=sum(op.duration for op in plan.schedule.operations if op.kind is j.OperationKind.WAIT),
        **{f'processing_r{r}_s': float(load) for r, load in enumerate(plan.loads)},
        diagnostic=json.dumps(dict(settings=asdict(j.JNConfig()), exposure=EXPOSURE, accounting=a,
            records=result.records, solution=j.solution_record(plan)), ensure_ascii=False, allow_nan=False))


def run_case(stage, name, parents, seed, selector, write, *, paired=True):
    print('JN_MIP_START', stage, name, len(parents), seed, flush=True)
    started = time.perf_counter()
    def report(row): print('JN_MIP_PROGRESS', name, json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    initial, blockers, _ = j.initialize(parents, seed, progress_callback=None)
    if initial.status != 'FEASIBLE':
        elapsed = time.perf_counter()-started
        detail = dict(initial_cmax_s=initial.cmax, initial_status=initial.status,
            solution=j.solution_record(initial), initial_accounting=initial.accounting,
            accounting=dict(initial_cmax_s=initial.cmax, search_complete=False),
            reason='fresh shared initialization is not B32 feasible; no joint search or replacement run')
        for method in (('initial', 'rule', 'gnn') if paired else ('initial', 'gnn')):
            write(dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
                source=SOURCES[method], status=initial.status, cmax_s=initial.cmax, runtime_s=elapsed,
                native_calls=initial.accounting['native_calls'], b32_calls=initial.accounting['b32_calls'],
                termination_reason='INITIALIZATION_FAILED',
                diagnostic=json.dumps(detail, ensure_ascii=False, allow_nan=False)))
        print('JN_MIP_INITIALIZATION_FAILED', name, initial.status, elapsed, flush=True)
        return
    w = j._HGAWorkspace(parents, j.ScientificConfig(), j.FORMAL_SCOPE_V2)
    plan = j.plan_from_initial(w, initial, blockers)
    initialization_s = time.perf_counter()-started
    write(dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed,
        source=SOURCES['initial'], status=initial.status, cmax_s=initial.cmax,
        native_cmax_s=initial.native_cmax, runtime_s=initialization_s,
        native_calls=initial.accounting['native_calls'], b32_calls=initial.accounting['b32_calls'],
        generations=initial.generations, valid_children=initial.valid_children,
        termination_reason=initial.termination_reason,
        diagnostic=json.dumps(dict(solution=j.solution_record(plan), accounting=initial.accounting,
            blocker_events=len(blockers), final_candidates=initial.final_candidates), ensure_ascii=False, allow_nan=False)))
    methods = [('rule', None), ('gnn', selector)] if paired else [('gnn', selector)]
    for method, head in methods:
        if method == 'gnn' and head is None:
            row = dict(stage=stage, layout_id=name, parent_count=len(parents), seed=seed, source=SOURCES['gnn'],
                status='NOT_RUN', termination_reason='NO_TRAINABLE_B32_IMPROVEMENT',
                diagnostic=json.dumps(dict(reason='Joint MILP label stage produced no positive trainable B32 gains; '
                    'do not deploy an untrained set head.', exposure=EXPOSURE)))
        else:
            result = j.optimize(w, plan, initialization_s, selector=head, progress_callback=report)
            row = result_row(stage, name, parents, seed, method, result, initial)
        write(row)
        print('JN_MIP_RESULT', json.dumps({k: v for k, v in row.items() if k != 'diagnostic'},
            ensure_ascii=False, allow_nan=False), flush=True)


def run_experiment():
    set_num_threads(1); torch.set_num_threads(1)
    existing = method_rows('pilot')
    expected = {(name, SEEDS[0]) for name in PILOTS}
    if {(r['layout_id'], int(r['seed'])) for r in existing} == expected:
        return summarize()
    previous = read_rows()
    saved_labels = {r['layout_id']: r for r in previous if r['source'] == LABEL_SOURCE}
    if any(r['source'] == FIT_SOURCE for r in previous):
        raise RuntimeError('trained in-memory selector no longer available; retain results without repeating training')
    train = dict(load_layouts('train'))
    excluded = {'train_'+x for x in ('30','13','36','15','34','28','04','26','16','03','14','10')}
    if set(FIT_LAYOUTS)&set(PILOTS) or excluded&(set(FIT_LAYOUTS)|set(PILOTS)):
        raise ValueError('invalid fit/screen selection')
    started = time.perf_counter()
    with RESULTS.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        def write(row): writer.writerow(row); stream.flush()
        selector = j.load_selector(ENCODER)
        groups = []
        for name in FIT_LAYOUTS:
            if name in saved_labels:
                detail = json.loads(saved_labels[name]['diagnostic'])
                records = detail.get('records', [])
                if records:
                    features = detail.get('set_features')
                    # The first completed pre-fix group has four identical zero
                    # gains and therefore no preference pair needing features.
                    if features is None and len({r.get('actual_gain_s') for r in records}) > 1:
                        raise RuntimeError('saved nonflat labels lack their original before-state features')
                    groups.append((torch.tensor(features, dtype=torch.float32) if features is not None else None, records))
                print('JN_MIP_LABEL_REUSED', name, saved_labels[name]['status'], flush=True)
                continue
            tick = time.perf_counter()
            print('JN_MIP_LABEL_START', name, len(train[name]), flush=True)
            initial, blockers, initialization_s = j.initialize(train[name], SEEDS[0])
            if initial.status != 'FEASIBLE':
                write(dict(stage='training_labels', layout_id=name, parent_count=len(train[name]), seed=SEEDS[0],
                    source=LABEL_SOURCE, status=initial.status, runtime_s=time.perf_counter()-tick,
                    native_calls=initial.accounting['native_calls'], b32_calls=initial.accounting['b32_calls'],
                    termination_reason='INITIALIZATION_FAILED', diagnostic=json.dumps(dict(records=[],
                        initial_accounting=initial.accounting, initial_status=initial.status,
                        initial_solution=j.solution_record(initial), initialization_s=initialization_s,
                        reason='No replacement or repeated initialization', exposure=EXPOSURE),
                        ensure_ascii=False, allow_nan=False)))
                print('JN_MIP_LABEL_INITIALIZATION_FAILED', name, initial.status, flush=True)
                continue
            w = j._HGAWorkspace(train[name], j.ScientificConfig(), j.FORMAL_SCOPE_V2)
            plan = j.plan_from_initial(w, initial, blockers)
            x, records = j.collect_labels(w, plan, selector)
            groups.append((x, records))
            detail = dict(initial_cmax_s=initial.cmax, initial_solution=j.solution_record(plan),
                initial_accounting=initial.accounting, initialization_s=initialization_s,
                blocker_events=len(blockers), records=records, set_features=x.tolist(), exposure=EXPOSURE)
            write(dict(stage='training_labels', layout_id=name, parent_count=len(train[name]), seed=SEEDS[0],
                source=LABEL_SOURCE, status='COMPLETED', runtime_s=time.perf_counter()-tick,
                native_calls=initial.accounting['native_calls'],
                b32_calls=initial.accounting['b32_calls']+sum(r['b32_called'] for r in records),
                diagnostic=json.dumps(detail, ensure_ascii=False, allow_nan=False)))
            print('JN_MIP_LABEL_RESULT', name, json.dumps([{k: v for k, v in r.items() if k != 'solution'}
                for r in records], ensure_ascii=False, allow_nan=False), flush=True)
        fit = j.fit_selector(selector, groups, j.JNConfig().training_epochs)
        write(dict(stage='training_fit', layout_id='TRAIN_3', seed=SEEDS[0], source=FIT_SOURCE,
            status='TRAINED' if fit['trained'] else 'NOT_TRAINED', runtime_s=fit['runtime_s'],
            native_calls=0, b32_calls=0, diagnostic=json.dumps(fit, ensure_ascii=False, allow_nan=False)))
        print('JN_MIP_FIT', json.dumps(fit, ensure_ascii=False, allow_nan=False), flush=True)
        active = selector if fit['trained'] else None
        for name in PILOTS:
            if (name, SEEDS[0]) in {(r['layout_id'], int(r['seed'])) for r in existing}:
                continue
            run_case('pilot', name, train[name], SEEDS[0], active, write)
        decision = summarize()
        if decision['passed']:
            print('JN_MIP_TRAIN_PASSED_FIXED_SETTINGS', flush=True)
            for name, parents in load_layouts('validation'):
                for seed in SEEDS:
                    run_case('validation', name, parents, seed, active, write, paired=False)
            summarize('validation')
    print('JN_MIP_BATCH_SECONDS', time.perf_counter()-started, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('summary',))
    parser.add_argument('--split', choices=('pilot', 'validation'), default='pilot')
    args = parser.parse_args()
    summarize(args.split)


if __name__ == '__main__': main()
