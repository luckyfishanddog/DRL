"""One fixed GNN-CEM TRAIN screen; conditional official splits, simple CSV."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from statistics import median, mean
import time
from types import SimpleNamespace
from unittest.mock import patch

from run_saeax import ROOT, configure_cpu, load_layouts, baseline_rows, SEEDS
configure_cpu()

import numpy as np
import torch
from numba import set_num_threads
from scipy.stats import wilcoxon
from mrta_search import gnn_cem as c
import run_joint_oriented_repair_pilot as common

RESULTS = ROOT/'results/new_dataset_final/gnn_cem.csv'
ENCODER = ROOT/'models/gnn_weld_layout_final.pt'
PILOTS = ('train_12', 'train_05', 'train_07')
FIELDS = ('layout_id', 'seed', 'status', 'initial_cmax', 'final_cmax', 'improvement_pct',
    'runtime_s', 'generations', 'population_size', 'b32_calls', 'accepted_elites',
    'termination_reason', 'stage', 'diagnostic', 'solution')
EXPOSURE = ('The original geometry encoder has historical exposure to all 40 TRAIN layouts. '
    'These three layouts avoid recent algorithm pilots; they are not historically unseen GNN data. '
    'Every instance resets its new CEM policy and performs only the fixed online elite updates.')


def fast_initial(parents, config, seed):
    """Exactly one common RULE/BEAM B32 proposal; retain failures too."""
    started = time.perf_counter(); calls = []
    original = common.reference_schedule_formal
    def observed(solution, cfg, *, orientations, **kwargs):
        item = SimpleNamespace(solution=solution, directions=tuple(tuple(orientations[r]) for r in range(4)), schedule=None)
        calls.append(item)
        item.schedule = original(solution, cfg, orientations=orientations, **kwargs)
        return item.schedule
    with patch.object(common, 'reference_schedule_formal', observed):
        best, attempts = common.common_initial_beam(parents, config, seed=seed, max_reference_calls=1)
    selected = best or (calls[-1] if calls and calls[-1].schedule is not None else None)
    return selected, dict(b32_calls=len(calls), runtime_s=time.perf_counter()-started,
        attempts=attempts, source='COMMON_RULE_BEAM_FIRST_PROPOSAL', max_reference_calls=1)


def read_results():
    if not RESULTS.exists(): return []
    csv.field_size_limit(2**31-1)
    with RESULTS.open(encoding='utf-8-sig', newline='') as stream: return list(csv.DictReader(stream))


def train_decision(rows=None):
    rows = [r for r in (read_results() if rows is None else rows) if r['stage'] == 'pilot']
    expected = {(name, SEEDS[0]) for name in PILOTS}
    complete = len(rows) == 3 and {(r['layout_id'], int(r['seed'])) for r in rows} == expected
    feasible = sum(r['status'] == 'FEASIBLE' for r in rows)
    paired = [r for r in rows if r.get('initial_cmax') and r.get('final_cmax') and r['status'] == 'FEASIBLE']
    gains = [float(r['improvement_pct']) for r in paired]
    wins = sum(float(r['final_cmax']) < float(r['initial_cmax'])-1.e-6 for r in paired)
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    accounts = [json.loads(r['diagnostic']).get('accounting', {}) for r in rows]
    budgets = len(accounts) == 3 and all(a.get('search_complete') and a.get('new_b32_calls', 25) <= 24
        and a.get('sample_attempts', 25) <= 24 for a in accounts)
    passed = (complete and feasible == 3 and len(paired) == 3 and wins >= 2
        and median(gains) >= 1. and runtime <= 120. and budgets)
    return dict(passed=bool(passed), complete=complete, feasible=feasible, pairs=len(paired),
        wins_over_initial=wins, median_improvement_pct=median(gains) if gains else None,
        mean_improvement_pct=mean(gains) if gains else None, runtime_median_s=runtime, budget_ok=budgets)


def compare(rows, method):
    baseline = baseline_rows(method)
    pairs = [(r, baseline[(r['layout_id'], int(r['seed']))]) for r in rows
        if (r['layout_id'], int(r['seed'])) in baseline and r['status'] == 'FEASIBLE'
        and baseline[r['layout_id'], int(r['seed'])]['status'] == 'FEASIBLE']
    changes = [100*(float(r['final_cmax'])-float(b['cmax_s']))/float(b['cmax_s']) for r, b in pairs]
    grouped = {}
    for (r, _), value in zip(pairs, changes): grouped.setdefault(r['layout_id'], []).append(value)
    aggregate = [median(values) for values in grouped.values()]
    pvalue = (float(wilcoxon(aggregate, alternative='less').pvalue) if any(abs(v)>1.e-12 for v in aggregate) else 1.) if len(aggregate)>=8 else None
    return dict(pairs=len(pairs), wins=sum(v < -1.e-8 for v in changes), losses=sum(v > 1.e-8 for v in changes),
        ties=sum(abs(v)<=1.e-8 for v in changes), median_change_pct=median(changes) if changes else None,
        mean_change_pct=mean(changes) if changes else None, layout_aggregates=len(aggregate), wilcoxon_less_p=pvalue)


def formal_decision(rows):
    pairs = {method: compare(rows, method) for method in ('HGA', 'WAG+VNS')}
    pvalues = sorted(v['wilcoxon_less_p'] for v in pairs.values() if v['wilcoxon_less_p'] is not None)
    significant = len(pvalues) == 2 and pvalues[0] <= .025 and pvalues[1] <= .05
    runtime = median(float(r['runtime_s']) for r in rows) if rows else None
    passed = (len(rows) == 30 and all(r['status'] == 'FEASIBLE' for r in rows)
        and pairs['HGA']['pairs'] == 30 and pairs['WAG+VNS']['pairs'] == 22 and significant
        and all(v['median_change_pct'] < 0 and v['wins'] > v['losses'] for v in pairs.values())
        and runtime <= 120.)
    return dict(passed=bool(passed), comparisons=pairs, holm_05_significant=significant, runtime_median_s=runtime)


def summary(stage='pilot'):
    rows = [r for r in read_results() if r['stage'] == stage]
    decision = train_decision(rows) if stage == 'pilot' else formal_decision(rows) if stage == 'validation' else dict(completed=len(rows))
    print('CEM_SUMMARY', json.dumps(dict(stage=stage, decision=decision,
        comparisons={m: compare(rows, m) for m in ('HGA', 'WAG+VNS')}), ensure_ascii=False, allow_nan=False), flush=True)
    for r in rows:
        print('CEM_ROW', json.dumps({k:r[k] for k in FIELDS if k not in ('diagnostic', 'solution')}, ensure_ascii=False), flush=True)
    return decision


def job(stage, name, parents, seed):
    print('CEM_START', stage, name, len(parents), seed, flush=True)
    started, accounting, progress = time.perf_counter(), {}, []
    def report(row):
        progress.append(row)
        print('CEM_PROGRESS', name, json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        result = c.run_cem(parents, ENCODER, fast_initial, seed=seed,
            progress_callback=report, accounting=accounting)
        initial_cmax = result.initial.schedule.cmax if result.initial and result.initial.schedule.feasible else None
        final_cmax = result.best.schedule.cmax if result.best else None
        row = dict(layout_id=name, seed=seed, status='FEASIBLE' if result.best else 'DEADLOCK',
            initial_cmax=initial_cmax, final_cmax=final_cmax,
            improvement_pct=100*(initial_cmax-final_cmax)/initial_cmax if initial_cmax is not None and final_cmax is not None else None,
            runtime_s=result.runtime, generations=result.generations, population_size=8,
            b32_calls=accounting['b32_calls'], accepted_elites=accounting['accepted_elites'],
            termination_reason=result.termination_reason, stage=stage,
            diagnostic=json.dumps(dict(settings=asdict(c.CEMConfig()), exposure=EXPOSURE,
                parent_count=len(parents), accounting=accounting, progress=result.progress,
                initial_solution=c.solution_record(result.initial)), ensure_ascii=False, allow_nan=False),
            solution=json.dumps(c.solution_record(result.best), ensure_ascii=False, allow_nan=False))
    except Exception as error:
        row = dict(layout_id=name, seed=seed, status='ERROR', runtime_s=time.perf_counter()-started,
            population_size=8, b32_calls=accounting.get('b32_calls'), accepted_elites=accounting.get('accepted_elites'),
            termination_reason='ERROR', stage=stage,
            diagnostic=json.dumps(dict(error=f'{type(error).__name__}: {error}', accounting=accounting,
                progress=progress, exposure=EXPOSURE), ensure_ascii=False, allow_nan=False))
        print('CEM_ERROR', name, row['diagnostic'], flush=True)
    print('CEM_RESULT', json.dumps({k:v for k,v in row.items() if k not in ('diagnostic', 'solution')},
        ensure_ascii=False, allow_nan=False), flush=True)
    return row


def run_experiment():
    torch.set_num_threads(1); set_num_threads(1)
    existing = read_results(); done = {(r['stage'], r['layout_id'], int(r['seed'])) for r in existing}
    keys = list(done)
    if len(keys) != len(existing): raise ValueError('ambiguous CEM completed keys')
    started = time.perf_counter()
    # One approved compact CSV, opened BEFORE any expensive work and flushed
    # after each case. No separate log, checkpoint or per-candidate result file.
    exists = RESULTS.exists()
    with RESULTS.open('a' if exists else 'x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists: writer.writeheader(); stream.flush()
        def run_stage(stage, jobs):
            for name, parents, seed in jobs:
                if (stage, name, seed) not in done:
                    writer.writerow(job(stage, name, parents, seed)); stream.flush(); done.add((stage, name, seed))
        train = dict(load_layouts('train'))
        run_stage('pilot', [(name, train[name], SEEDS[0]) for name in PILOTS])
        passed = summary('pilot')['passed']
        if passed:
            run_stage('validation', [(name, parents, seed) for name, parents in load_layouts('validation') for seed in SEEDS])
            if summary('validation')['passed']:
                # Same fixed online CEM optimizer; no cross-instance training,
                # extra epochs, model selection or parameter adjustment on TEST.
                run_stage('test', [(name, parents, seed) for name, parents in load_layouts('test') for seed in SEEDS])
                summary('test')
    print('CEM_BATCH_SECONDS', time.perf_counter()-started, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('pilot', 'summary'))
    parser.add_argument('--split', choices=('pilot', 'validation', 'test'), default='pilot')
    args = parser.parse_args()
    if args.command == 'summary': summary(args.split)
    else: run_experiment()


if __name__ == '__main__': main()
