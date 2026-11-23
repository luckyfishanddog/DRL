"""Run the common-baseline and joint VNext study on the filtered weld layouts."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from itertools import product
from pathlib import Path
import pickle
from random import Random
from statistics import median
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mrta_reference.geometry import optimize_directions, oriented_endpoints, robot_is_eligible
from mrta_reference.model import DeadlockBlockage, Operation, OperationKind, ParentWeld, RobotRoute, ScheduleResult, ScheduleStatus, ScientificConfig, SplitKind
from mrta_reference.scheduler import ScheduleWaitBlockers, build_operation_templates, earliest_safe_start, selected_schedule_wait_blockers
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_baselines import AdaptedHGAConfig, AdaptedWAGConfig, run_adapted_hga_v2, run_adapted_wag_vns_v2
from mrta_baselines.wag_vns import priority_schedule
import mrta_baselines.hga as native_hga
from mrta_search import gnn_constructor as gnn, lns
from mrta_search.direction import _fixed_first_dp
from mrta_search.initialization import build_oriented_insertion_beam
from run_joint_oriented_repair_pilot import initial_reference, schedule_explicit


DATA = ROOT / 'data' / 'weld_layout'
MODEL = ROOT / 'models' / 'gnn_weld_layout.pt'
RESULTS = ROOT / 'results' / 'gnn_weld_layout_comparison.csv'
FINAL_RESULTS = ROOT / 'results' / 'new_dataset_final'
BEAM_VALIDATION_CSV = FINAL_RESULTS / 'gnn_beam2_validation.csv'
BEAM_VALIDATION_FIELDS = ('layout_id', 'method', 'status', 'cmax_s',
                          'processing_r0_s', 'processing_r1_s', 'processing_r2_s', 'processing_r3_s',
                          'move_s', 'runtime_s', 'b32_calls', 'cmax_change_vs_width1_s',
                          'cmax_change_vs_width1_pct', 'diagnostic')
CONFIG = ScientificConfig()
SEED = 20261110
TEST_SEEDS = (20261110, 20261111, 20261112)
CHECKPOINTS = (5., 30., 60.)
BASELINES = {
    'WAG+VNS': (run_adapted_wag_vns_v2, AdaptedWAGConfig, 'wag_vns_results_optimized.csv'),
    'HGA': (run_adapted_hga_v2, AdaptedHGAConfig, 'hga_results_optimized.csv'),
}
BASELINE_ITERATIONS = {'WAG+VNS': 500, 'HGA': 200000}
BASELINE_SOURCES = {'WAG+VNS': 'PARTIAL_LNS_V1', 'HGA': 'ORIENTED_COORDINATED_VND_V1'}
HGA_ROUTE_SOURCE = 'ORIENTED_ROUTE_VND_NATIVE_SELECT_V1'
FORMAL_BASELINE_SOURCES = {**BASELINE_SOURCES, 'HGA': HGA_ROUTE_SOURCE}
EQUAL_TIME_SOURCE = 'EQUAL_TIME_120S'
EQUAL_TIME_LIMIT = 120.
RESULT_FIELDS = ('layout_id', 'seed', 'method', 'status', 'cmax_s', 'runtime_s', 'construct_s',
                 'cmax_5_s', 'cmax_30_s', 'cmax_60_s', 'native_cmax_s', 'native_cmax_5_s',
                 'native_cmax_30_s', 'native_cmax_60_s', 'max_processing_s', 'move_s', 'wait_s',
                 'b32_calls', 'diagnostic', 'source')
OPTIMIZED_FIELDS = ('layout_id', 'seed', 'method', 'status', 'cmax_s', 'native_cmax_s',
                    'processing_r0_s', 'processing_r1_s', 'processing_r2_s', 'processing_r3_s',
                    'move_s', 'wait_s', 'runtime_s', 'iterations', 'b32_calls', 'diagnostic', 'source')
BEAM_SEED = 202611041


def load_layouts(split):
    rows = [json.loads(line) for line in (DATA / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    return [(row['layout_id'], tuple(ParentWeld(p['parent_id'], tuple(p['start']), tuple(p['end']))
                                    for p in row['parents'])) for row in rows]


def rule_beam(parents):
    """Use the existing RULE/BEAM first proposal, with no B32-based selection."""
    started = time.perf_counter()
    failures = []
    beams = {}
    schemes = ('REGION_LOCAL', 'RULE_X_LOCAL')
    for index in range(4):
        for scheme in schemes:
            try:
                if index == 0:
                    reference = initial_reference(parents, CONFIG, scheme)
                    beams[scheme] = reference, build_oriented_insertion_beam(reference, CONFIG, seed=BEAM_SEED)
                if scheme not in beams:
                    continue
                reference, beam = beams[scheme]
                if index < len(beam):
                    state, actions = beam[index]
                    solution = canonicalize(parents, reference.patterns, dict(enumerate(state.routes)),
                                            CONFIG, scope=FORMAL_SCOPE_V2)
                    if solution.patterns != reference.patterns or tuple(r.block_ids for r in solution.routes) != state.routes:
                        raise ValueError('canonicalization changed the fixed-pattern beam')
                    return gnn.JointCompletionResult(solution, state.directions, actions,
                                                     time.perf_counter() - started, oriented_state=state)
            except (ValueError, ArithmeticError) as error:
                failures.append(f'{scheme}: {error}')
    return gnn.JointCompletionResult(None, None, (), time.perf_counter() - started,
                                     rejection_reason='; '.join(failures) or 'EMPTY_BEAM')


def b32(result, *, feasible_dispatch_observer=None):
    return schedule_explicit(result.solution, result.directions, CONFIG,
                             feasible_dispatch_observer=feasible_dispatch_observer).schedule


def final_metrics(schedule):
    if schedule is None or not schedule.feasible:
        return {}
    loads, move, wait = [0.] * 4, 0., 0.
    for op in schedule.operations:
        if op.kind is OperationKind.MOVE:
            move += op.duration
        elif op.kind is OperationKind.WAIT:
            wait += op.duration
        else:
            loads[op.robot_id] += op.duration
    return dict(max_processing_s=max(loads), move_s=move, wait_s=wait)


def route_blocks(solution, config):
    blocks = block_map(solution, config, scope=FORMAL_SCOPE_V2)
    return tuple(tuple(blocks[b] for b in route.block_ids) for route in solution.routes)


def native_result(operations, directions, policy):
    completion = tuple(route[-1].end_time if route else 0. for route in operations)
    return ScheduleResult(ScheduleStatus.FEASIBLE, tuple(op for route in operations for op in route),
                          max(completion), completion, directions=directions, reference_policy_id=policy)


def hga_cycle_time(solution, config):
    """Four-robot adaptation: priority scheduling includes MOVE and WAIT."""
    return priority_schedule(solution, config)


def _wag_x_view(operation, config):
    # Preserve duration and x motion. Disjoint dummy y lanes disable the
    # two-dimensional condition in the shared separation primitive.
    lane = (operation.robot_id % 2) * (config.interference_dy + 1.)
    return replace(operation, start=(operation.start[0], lane), end=(operation.end[0], lane))


def _wag_follow_route(fixed, following, config):
    """Schedule one route behind its fixed-priority same-rail partner."""
    projected = tuple(_wag_x_view(op, config) for op in fixed)
    ready, waited, scheduled = 0., False, []
    for op in following:
        start, _, _ = earliest_safe_start(_wag_x_view(op, config), ready, projected, config)
        if not math.isfinite(start):
            return None
        if start > ready + config.numeric_epsilon:
            waited = True
            scheduled.append(Operation(op.operation_id + ':WAIT', op.robot_id, OperationKind.WAIT,
                                       ready, start, op.start, op.start, op.sequence_index, op.block_id))
        ready = start + op.duration
        scheduled.append(replace(op, start_time=start, end_time=ready))
    return tuple(scheduled), waited, ready


def wag_priority_schedule(solution, config):
    """Shared implementation of the paper priority rule on both rails."""
    return priority_schedule(solution, config)


NATIVE_EVALUATORS = {'WAG+VNS': priority_schedule, 'HGA': priority_schedule}


def baseline_run(parents, method, seed, budget, checkpoints, *, iterations=None, candidate_evaluator=None,
                 formal=False):
    run, make_settings, _ = BASELINES[method]
    settings = make_settings()
    if formal and method == 'HGA':
        settings = replace(settings, vnd_objective='ROUTE_CYCLE')
    if iterations is not None:
        name = 'imax' if method == 'WAG+VNS' else 'max_iterations'
        settings = replace(settings, **{name: iterations})
    return run(parents, CONFIG, settings, seed=seed, time_limit=budget,
               checkpoints=checkpoints, certify=False, candidate_evaluator=candidate_evaluator)


def baseline_job(job, *, time_limit=None):
    (name, parents), method, seed = job
    equal_time = time_limit is not None
    source = EQUAL_TIME_SOURCE if equal_time else FORMAL_BASELINE_SOURCES[method]
    started = time.perf_counter()
    result = baseline_run(parents, method, seed, time_limit if equal_time else math.inf,
                          () if equal_time or method == 'HGA' else CHECKPOINTS,
                          iterations=BASELINE_ITERATIONS[method], candidate_evaluator=NATIVE_EVALUATORS[method],
                          formal=True)
    search_runtime = time.perf_counter() - started
    complete = equal_time or result.iterations == BASELINE_ITERATIONS[method]
    print('BASELINE_STATS', json.dumps(dict(layout_id=name, method=method, seed=seed,
          source=source, iterations=result.iterations,
          accounting={key: value for key, value in result.accounting.items()
                      if value is None or isinstance(value, (int, float))})), flush=True)
    best = result.timely_best
    b32_started = time.perf_counter()
    schedule = schedule_explicit(best.solution, best.directions, CONFIG).schedule if best and complete else None
    b32_runtime = time.perf_counter() - b32_started if schedule is not None else 0.
    if schedule is None:
        diagnostic = f"Native deadlocks={result.accounting['remaining_deadlock']} | {result.termination_reason}"
    else:
        diagnostic = '' if schedule.feasible else ' | '.join(schedule.diagnostics[:2])
    if equal_time:
        diagnostic = json.dumps(dict(phase='EQUAL_TIME', run_status='COMPLETED',
            algorithm_source=FORMAL_BASELINE_SOURCES[method], time_limit_s=time_limit,
            search_runtime_s=result.actual_runtime, search_call_runtime_s=search_runtime,
            initialization_s=result.initialization_time, b32_runtime_s=b32_runtime,
            iterations=result.iterations, termination_reason=result.termination_reason,
            overshoot_s=result.overshoot, timely_best_completed_at_s=best.completed_at if best else None,
            late_best_updates=sum(event.elapsed > time_limit for event in result.best_events),
            native_calls=result.reference_calls, certifier_calls=result.certifier_calls,
            vnd_unique_calls=result.accounting.get('vnd_unique_calls'),
            vnd_cache_hits=result.accounting.get('vnd_cache_hits'),
            vnd_native_candidate_evaluations=result.accounting.get('vnd_native_candidate_evaluations'),
            vnd_pass_cap_hits=result.accounting.get('vnd_pass_cap_hits'),
            vnd_candidate_cap_hits=result.accounting.get('vnd_candidate_cap_hits'),
            b32_status=schedule.status.value if schedule else 'NO_TIMELY_CANDIDATE',
            b32_diagnostic=diagnostic))
    elif method == 'HGA':
        diagnostic = json.dumps(dict(phase='FULL', run_status='COMPLETED' if complete else 'INCOMPLETE',
            search_runtime_s=search_runtime, initialization_s=result.initialization_time,
            vnd_s=result.accounting['local_search_time'], native_coordination_s=result.accounting['scheduler_time'],
            native_calls=result.reference_calls, vnd_candidates=result.accounting['vnd_candidate_operations'],
            vnd_unique_calls=result.accounting['vnd_unique_calls'], vnd_cache_hits=result.accounting['vnd_cache_hits'],
            vnd_native_candidate_evaluations=result.accounting['vnd_native_candidate_evaluations'],
            vnd_pass_cap_hits=result.accounting['vnd_pass_cap_hits'],
            vnd_candidate_cap_hits=result.accounting['vnd_candidate_cap_hits'],
            b32_runtime_s=b32_runtime, b32_diagnostic=diagnostic))
    return dict(layout_id=name, seed=seed, method=method,
                status=('INCOMPLETE' if not complete else
                        schedule.status.value if schedule else
                        'NO_TIMELY_CANDIDATE' if equal_time else 'NO_NATIVE_SOLUTION'),
                cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                runtime_s=time.perf_counter() - started,
                b32_calls=int(schedule is not None), diagnostic=diagnostic,
                source=source, iterations=result.iterations,
                native_cmax_s=best.schedule.cmax if best else None,
                **baseline_metrics(best, schedule))


def baseline_metrics(best, schedule):
    if best is None:
        return {}
    blocks = block_map(best.solution, CONFIG, scope=FORMAL_SCOPE_V2)
    loads = {f'processing_r{r.robot_id}_s': sum(CONFIG.process_time(blocks[b].length) for b in r.block_ids)
             for r in best.solution.routes}
    move = sum(math.dist(oriented_endpoints(blocks[route.block_ids[k - 1]], best.directions[r][k - 1])[1],
                              oriented_endpoints(blocks[route.block_ids[k]], best.directions[r][k])[0]) / CONFIG.empty_speed
               for r, route in enumerate(best.solution.routes) for k in range(1, len(route.block_ids)))
    wait = sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT) if schedule else None
    return dict(**loads, move_s=move, wait_s=wait)


def hga_alpha_sensitivity(*, full_budget=False):
    """Run the screening or full-budget pair, with a fresh HGA per alpha."""
    from numba import set_num_threads
    configure_cpu()
    set_num_threads(1)
    layouts = dict(load_layouts('validation'))
    names = ('validation_07',) if full_budget else ('validation_03', 'validation_04', 'validation_07')
    alphas = (20, 18) if full_budget else (16, 18, 20, 22, 24, 26)
    iterations = 200000 if full_budget else 1000
    settings = AdaptedHGAConfig()
    print('HGA_ALPHA_SETTINGS', settings, 'seed', SEED, 'workers/threads', 1,
          'source', BASELINE_SOURCES['HGA'], 'iterations', iterations, flush=True)
    # Ten warm-up children also compile the mu+lambda population update.
    started = time.perf_counter()
    warm = run_adapted_hga_v2(layouts[names[0]], CONFIG, replace(settings, max_iterations=10),
        seed=SEED, time_limit=math.inf, checkpoints=(), certify=False, candidate_evaluator=priority_schedule)
    print('HGA_ALPHA_WARMUP', warm.iterations, 'seconds', time.perf_counter() - started,
          'excluded from CSV and all measured runs', flush=True)
    del warm
    fields = ('layout_id', 'alpha', 'iterations', 'runtime_s', 'initial_native_cmax',
              'final_native_cmax', 'b32_status', 'b32_cmax', 'vnd_unique_calls',
              'vnd_cache_hits', 'vnd_native_candidate_evaluations', 'b32_runtime_s', 'run_status')
    path = FINAL_RESULTS / 'hga_alpha_sensitivity.csv'
    rows = []
    with path.open('a' if full_budget else 'x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        stream.flush()
        for name in names:
            for alpha in alphas:
                print('HGA_ALPHA_START', name, 'parents', len(layouts[name]), 'alpha', alpha, flush=True)
                started = time.perf_counter()
                result = run_adapted_hga_v2(layouts[name], CONFIG,
                    replace(settings, alpha=alpha, max_iterations=iterations), seed=SEED,
                    time_limit=math.inf, checkpoints=(), certify=False, candidate_evaluator=priority_schedule)
                runtime = time.perf_counter() - started
                best, initial = result.timely_best, result.initial_candidate
                started = time.perf_counter()
                schedule = schedule_explicit(best.solution, best.directions, CONFIG).schedule if best else None
                b32_runtime = time.perf_counter() - started if best else 0.
                row = dict(layout_id=name, alpha=alpha, iterations=result.iterations, runtime_s=runtime,
                    initial_native_cmax=initial.schedule.cmax if initial else None,
                    final_native_cmax=best.schedule.cmax if best else None,
                    b32_status=schedule.status.value if schedule else 'NO_NATIVE_SOLUTION',
                    b32_cmax=schedule.cmax if schedule is not None and schedule.feasible else None,
                    **{key: result.accounting[key] for key in
                       ('vnd_unique_calls', 'vnd_cache_hits', 'vnd_native_candidate_evaluations')},
                    b32_runtime_s=b32_runtime,
                    run_status='COMPLETED' if result.iterations == iterations else 'INCOMPLETE')
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print('HGA_ALPHA_RESULT', json.dumps(row), flush=True)
    return rows


def hga_route_vnd_job(name, parents, iterations, *, final_b32):
    """One fixed-seed route-VND run; append to the existing HGA results."""
    settings = replace(AdaptedHGAConfig(), max_iterations=iterations, vnd_objective='ROUTE_CYCLE')
    print('HGA_ROUTE_START', name, 'parents', len(parents), settings, 'seed', SEED,
          'source', HGA_ROUTE_SOURCE, flush=True)
    started = time.perf_counter()
    result = run_adapted_hga_v2(parents, CONFIG, settings, seed=SEED, time_limit=math.inf,
                              checkpoints=(), certify=False, candidate_evaluator=priority_schedule)
    search_runtime = time.perf_counter() - started
    best = result.timely_best
    b32_started = time.perf_counter()
    schedule = schedule_explicit(best.solution, best.directions, CONFIG).schedule if final_b32 and best else None
    b32_runtime = time.perf_counter() - b32_started if schedule is not None else 0.
    stats = dict(phase='FULL' if final_b32 else 'PERFORMANCE_PILOT',
        run_status='COMPLETED' if result.iterations == iterations else 'INCOMPLETE',
        search_runtime_s=search_runtime, initialization_s=result.initialization_time,
        vnd_s=result.accounting['local_search_time'], native_coordination_s=result.accounting['scheduler_time'],
        crossover_s=result.accounting['crossover_time'], population_management_s=result.accounting['population_management_time'],
        native_calls=result.reference_calls, vnd_candidates=result.accounting['vnd_candidate_operations'],
        vnd_unique_calls=result.accounting['vnd_unique_calls'], vnd_cache_hits=result.accounting['vnd_cache_hits'],
        vnd_native_candidate_evaluations=result.accounting['vnd_native_candidate_evaluations'],
        vnd_pass_cap_hits=result.accounting['vnd_pass_cap_hits'],
        vnd_candidate_cap_hits=result.accounting['vnd_candidate_cap_hits'], b32_runtime_s=b32_runtime)
    if schedule is not None and not schedule.feasible:
        stats['b32_diagnostic'] = ' | '.join(schedule.diagnostics[:2])
    row = dict(layout_id=name, seed=SEED, method='HGA', source=HGA_ROUTE_SOURCE,
        iterations=result.iterations, runtime_s=time.perf_counter() - started,
        native_cmax_s=best.schedule.cmax if best else None,
        status=schedule.status.value if schedule is not None else ('NATIVE_ONLY' if best else 'NO_NATIVE_SOLUTION'),
        cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
        b32_calls=int(schedule is not None), diagnostic=json.dumps(stats), **baseline_metrics(best, schedule))
    write_rows(FINAL_RESULTS / BASELINES['HGA'][2], [row], append=True)
    output = dict(**row, **stats)
    print('HGA_ROUTE_RESULT', json.dumps(output), flush=True)
    return output


def hga_route_vnd_experiment():
    """Pilot first; run the full representatives only at the requested thresholds."""
    from numba import set_num_threads
    configure_cpu()
    set_num_threads(1)
    layouts = dict(load_layouts('validation'))
    settings = replace(AdaptedHGAConfig(), max_iterations=10, vnd_objective='ROUTE_CYCLE')
    started = time.perf_counter()
    warm = run_adapted_hga_v2(layouts['validation_07'], CONFIG, settings, seed=SEED, time_limit=math.inf,
                            checkpoints=(), certify=False, candidate_evaluator=priority_schedule)
    print('HGA_ROUTE_WARMUP', warm.iterations, 'seconds', time.perf_counter() - started,
          'excluded from all measured runs and CSV', flush=True)
    del warm
    rows = [hga_route_vnd_job('validation_07', layouts['validation_07'], 20000, final_b32=False)]
    pilot = rows[-1]
    if pilot['iterations'] != 20000 or pilot['search_runtime_s'] > 180.:
        print('HGA_ROUTE_STOP', 'pilot exceeds 180s or is incomplete; locate the measured bottleneck first', flush=True)
        return rows
    if pilot['search_runtime_s'] > 150.:
        print('HGA_ROUTE_CAUTION', '150-180s pilot; assess measured timings before launching full budget', flush=True)
        return rows
    rows.append(hga_route_vnd_job('validation_07', layouts['validation_07'], 200000, final_b32=True))
    full = rows[-1]
    if full['iterations'] != 200000 or full['runtime_s'] > 1800. or full['status'] != 'FEASIBLE':
        print('HGA_ROUTE_STOP', 'validation_07 did not meet the full-budget time/feasibility target', flush=True)
        return rows
    for name in ('validation_03', 'validation_04'):
        rows.append(hga_route_vnd_job(name, layouts[name], 200000, final_b32=True))
    return rows


def read_rows(path):
    if not path.exists():
        return []
    csv.field_size_limit(2**31-1)
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows, *, append=False):
    fields = OPTIMIZED_FIELDS if isinstance(path.name, str) and path.name.endswith('_optimized.csv') else RESULT_FIELDS
    needs_header = not append or not path.exists() or not path.stat().st_size
    if append and not needs_header:
        with path.open(encoding='utf-8', newline='') as stream:
            header = next(csv.reader(stream))
        if header != list(fields):
            rows = read_rows(path) + list(rows)
            append, needs_header = False, True
    with path.open('a' if append else 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def retained_rows(method):
    """Repeat the retained deterministic measurement across seed labels."""
    for row in read_rows(RESULTS):
        if row['method'] != method:
            continue
        loads = [float(row[f'processing_r{r}_s']) for r in range(4) if row[f'processing_r{r}_s']]
        for seed in TEST_SEEDS:
            yield dict(layout_id=row['layout_id'], method=method, seed=seed, status=row['status'], cmax_s=row['cmax_s'],
                       construct_s=row['construct_s'], b32_calls=0, diagnostic=row['diagnostic'],
                       max_processing_s=max(loads, default=None),
                       move_s=sum(float(row[f'move_r{r}_s']) for r in range(4)) if loads else None,
                       wait_s=sum(float(row[f'wait_r{r}_s']) for r in range(4)) if row['wait_r0_s'] else None,
                       source='RETAINED_DETERMINISTIC')


def completed_baseline_row(row):
    """A completed formal search and its real final B32 evaluation."""
    method = row.get('method')
    try:
        if (method not in FORMAL_BASELINE_SOURCES or row.get('source') != FORMAL_BASELINE_SOURCES[method]
                or int(row['iterations']) != BASELINE_ITERATIONS[method]
                or int(row['seed']) not in TEST_SEEDS or int(row['b32_calls']) != 1
                or row.get('status') not in {status.value for status in ScheduleStatus}):
            return False
        runtime = float(row['runtime_s'])
        if not math.isfinite(runtime) or runtime <= 0.:
            return False
        if row['status'] == 'FEASIBLE' and not (math.isfinite(float(row['cmax_s'])) and float(row['cmax_s']) > 0.):
            return False
        if method == 'HGA':
            detail = json.loads(row['diagnostic'])
            if detail.get('phase') != 'FULL' or detail.get('run_status') != 'COMPLETED':
                return False
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    return True


def completed_equal_time_row(row):
    """Normal finite-budget returns count, including failed final B32 or no timely candidate."""
    method = row.get('method')
    try:
        if method not in BASELINES or row.get('source') != EQUAL_TIME_SOURCE or int(row['seed']) not in TEST_SEEDS:
            return False
        detail = json.loads(row['diagnostic'])
        if (detail.get('phase') != 'EQUAL_TIME' or detail.get('run_status') != 'COMPLETED'
                or detail.get('algorithm_source') != FORMAL_BASELINE_SOURCES[method]
                or detail.get('time_limit_s') != EQUAL_TIME_LIMIT
                or int(row['iterations']) != detail['iterations']
                or not 0 <= int(row['iterations']) <= BASELINE_ITERATIONS[method]):
            return False
        for value in (row['runtime_s'], detail['search_runtime_s']):
            if not math.isfinite(float(value)) or float(value) <= 0.:
                return False
        if not math.isfinite(detail['b32_runtime_s']) or detail['b32_runtime_s'] < 0.:
            return False
        if row['status'] == 'NO_TIMELY_CANDIDATE':
            return (int(row['b32_calls']) == 0 and detail['timely_best_completed_at_s'] is None
                    and detail['b32_runtime_s'] == 0. and not row.get('cmax_s') and not row.get('native_cmax_s'))
        if (row['status'] not in {status.value for status in ScheduleStatus} or int(row['b32_calls']) != 1
                or not 0. <= detail['timely_best_completed_at_s'] <= EQUAL_TIME_LIMIT):
            return False
        if row['status'] == 'FEASIBLE':
            return math.isfinite(float(row['cmax_s'])) and float(row['cmax_s']) > 0.
        return not row.get('cmax_s')
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


def configure_baseline_worker(hga_warmup=None, wag_warmup=None):
    """One compute thread; JIT warm-up is outside every measured job."""
    from numba import set_num_threads
    configure_cpu()
    set_num_threads(1)
    if hga_warmup is not None:
        started = time.perf_counter()
        warm = baseline_run(hga_warmup, 'HGA', SEED, math.inf, (), iterations=10,
                            candidate_evaluator=priority_schedule, formal=True)
        print('BASELINE_HGA_WARMUP', warm.iterations, time.perf_counter() - started,
              'excluded from measured jobs; no B32', flush=True)
    if wag_warmup is not None:
        started = time.perf_counter()
        warm = baseline_run(wag_warmup, 'WAG+VNS', SEED, math.inf, (), iterations=20,
                            candidate_evaluator=priority_schedule, formal=True)
        print('BASELINE_WAG_WARMUP', warm.iterations, time.perf_counter() - started,
              'Move/Swap/LNS', *(warm.accounting[key] for key in ('move_calls', 'swap_calls', 'lns_calls')),
              'excluded from measured jobs; no B32', flush=True)


def run_baselines(workers, only=None, *, split='validation', representative=False, time_limit=None):
    if time_limit is not None and (time_limit != EQUAL_TIME_LIMIT or split != 'test'
                                   or only not in BASELINES or representative):
        raise ValueError('equal-time runs require one method, all TEST layouts and --time-limit 120')
    is_completed = completed_baseline_row if time_limit is None else completed_equal_time_row
    layouts = load_layouts(split)
    seeds = TEST_SEEDS
    if representative:
        ordered = sorted(layouts, key=lambda item: (len(item[1]), item[0]))
        layouts = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
        seeds = (SEED,)
    paths = {method: FINAL_RESULTS / settings[2] for method, settings in BASELINES.items() if only in (None, method)}
    done = set()
    for path in paths.values():
        rows = read_rows(path)
        done.update((row['layout_id'], row['method'], int(row['seed'])) for row in rows
                    if is_completed(row))
    jobs = [(layout, method, seed) for layout in layouts for seed in seeds for method in paths
            if (layout[0], method, seed) not in done]
    expected = {(layout[0], method, seed) for layout in layouts for seed in seeds for method in paths}
    workers = max(1, min(workers, 3)) if 'HGA' in paths or time_limit is not None else workers
    sources = {m: EQUAL_TIME_SOURCE if time_limit is not None else FORMAL_BASELINE_SOURCES[m] for m in paths}
    print('BASELINE_BATCH_START', json.dumps(dict(split=split, workers=workers, expected=len(expected),
          completed=len(expected & done), remaining=len(jobs), sources=sources, time_limit_s=time_limit)),
          flush=True)
    if not jobs:
        return []
    warmup = dict(load_layouts('validation'))['validation_07'] if 'HGA' in paths else None
    wag_warmup = dict(load_layouts('validation'))['validation_07'] if time_limit is not None and 'WAG+VNS' in paths else None
    completed = []
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_baseline_worker,
                             initargs=(warmup, wag_warmup)) as executor:
        futures = {executor.submit(baseline_job, job, time_limit=time_limit): job for job in jobs}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                (name, _), method, seed = futures[future]
                row = dict(layout_id=name, method=method, seed=seed, status='INCOMPLETE',
                           source=sources[method],
                           diagnostic=json.dumps(dict(phase='FULL' if time_limit is None else 'EQUAL_TIME',
                               run_status='INCOMPLETE', time_limit_s=time_limit, error=repr(error))))
            print('BASELINE', json.dumps(row), flush=True)
            write_rows(paths[row['method']], [row], append=True)
            if is_completed(row):
                done.add((row['layout_id'], row['method'], int(row['seed'])))
                completed.append(row)
            print('BASELINE_BATCH_PROGRESS', split, len(expected & done), '/', len(expected), flush=True)
    missing = sorted(expected - done)
    if missing:
        raise RuntimeError(f'{split}: formal runs still missing: {missing}')
    return completed


def wag_deadlock_job(job):
    """Rebuild one current incumbent, then observe the existing dispatch kernel.

    Diagnostic rollouts happen only after the complete 500-round search. They
    never provide feedback to that search and do not change the B32 policy.
    """
    from collections import Counter
    from mrta_reference import scheduler as scheduler
    from mrta_reference.dispatch_recovery import run_dispatch_rollout
    from mrta_reference.geometry import finite_x_split_validator, operations_conflict
    (name, parents), historical = job
    seed = int(historical['seed'])
    started = time.perf_counter()
    result = baseline_run(parents, 'WAG+VNS', seed, math.inf, (),
                          iterations=500, candidate_evaluator=priority_schedule)
    best = result.timely_best
    record = dict(layout_id=name, seed=seed, source=historical['source'], reproduced=False,
                  iterations=result.iterations, search_rerun_iterations=result.iterations,
                  replay_mode='FULL_SEARCH', search_runtime_s=time.perf_counter() - started,
                  historical_native_cmax_s=float(historical['native_cmax_s']),
                  historical_b32_status=historical['status'])
    if best is None:
        record['reason'] = 'no native incumbent; diagnosis skipped'
        print('WAG_DEADLOCK_DIAGNOSIS', json.dumps(record), flush=True)
        return record
    solution, directions, native = best.solution, best.directions, best.schedule
    formal = schedule_explicit(solution, directions, CONFIG).schedule
    delta = native.cmax - record['historical_native_cmax_s']
    record.update(native_cmax_s=native.cmax, native_cmax_delta_s=delta,
                  b32_status=formal.status.value, b32_cmax_s=formal.cmax,
                  reproduced=(result.iterations == 500 and abs(delta) <= 1.e-6
                              and formal.status.value == historical['status']))
    if not record['reproduced']:
        record['reason'] = 'budget, native Cmax or B32 status not reproduced; diagnosis skipped'
        print('WAG_DEADLOCK_DIAGNOSIS', json.dumps(record), flush=True)
        return record
    routes = scheduler.build_robot_routes(solution, CONFIG, dict(enumerate(directions)),
                    x_split_validator=finite_x_split_validator(solution.parents, CONFIG))
    templates = {r.robot_id: scheduler.build_operation_templates(r, CONFIG) for r in routes}
    first_block, first_delay = [], []
    blocked_edges = dict(same=0, cross=0)
    delayed_edges = dict(same=0, cross=0)
    first_cross_delay = []
    intermediate_cycles = []
    reported_cycles = []

    class ObservedKernel(scheduler._OptimizedDispatchKernel):
        def choices(self, state, runtime):
            choices = super().choices(state, runtime)
            available = {c.robot_id for c in choices.ordered_choices}
            blocked = {robot: others for robot, others in choices.blockers_by_robot if robot not in available}
            for robot, others in blocked.items():
                for other in others:
                    blocked_edges['same' if robot // 2 == other // 2 else 'cross'] += 1
            owners = dict(choices.blockers_by_robot)
            for choice in choices.ordered_choices:
                robot = choice.robot_id
                if choice.start_time <= state.completion[robot] + CONFIG.numeric_epsilon:
                    continue
                for other in owners.get(robot, ()):
                    delayed_edges['same' if robot // 2 == other // 2 else 'cross'] += 1
                    if robot // 2 != other // 2 and not first_cross_delay:
                        first_cross_delay.append(dict(depth=state.depth, robot=robot, operation=choice.operation_id,
                            other_robot=other, ready_s=state.completion[robot], earliest_start_s=choice.start_time,
                            reported_blocker_ids=dict(choices.blocker_ids_by_robot).get(robot, ())))
            cycles = scheduler.wait_for_cycles(scheduler.build_wait_for_graph(blocked))
            if cycles and cycles not in intermediate_cycles:
                intermediate_cycles.append(cycles)
            cycles = scheduler.wait_for_cycles(scheduler.build_wait_for_graph(dict(choices.blockers_by_robot)))
            if cycles and cycles not in reported_cycles:
                reported_cycles.append(cycles)
            if not first_delay:
                for choice in choices.ordered_choices:
                    if choice.start_time > state.completion[choice.robot_id] + CONFIG.numeric_epsilon:
                        first_delay.append(dict(depth=state.depth, robot=choice.robot_id,
                            operation=choice.operation_id, ready_s=state.completion[choice.robot_id],
                            earliest_start_s=choice.start_time))
                        break
            if not first_block:
                for robot in range(4):
                    if state.next_index[robot] < len(self.templates[robot]) and robot not in available:
                        op = self.templates[robot][state.next_index[robot]]
                        ids = dict(choices.blocker_ids_by_robot).get(robot, ())
                        first_block.append(dict(depth=state.depth, robot=robot, operation=op.operation_id,
                            kind=op.kind.value, block_id=op.block_id, ready_s=state.completion[robot],
                            start=op.start, end=op.end, duration_s=op.duration,
                            blocker_count=len(ids), blocker_examples=ids[:4],
                            graph=scheduler.build_wait_for_graph(dict(choices.blockers_by_robot))))
                        break
            return choices

    kernel = ObservedKernel(templates, scheduler._remaining_processing_suffix(templates), CONFIG, directions, None)
    outcome = run_dispatch_rollout(kernel, collect_trace=True)
    state = outcome.terminal_state
    choices = scheduler._OptimizedDispatchKernel.choices(kernel, state, kernel.start_runtime(state))
    fixed = {op.operation_id: op for op in state.fixed_operations}
    terminal = []
    for robot, blocker_ids in choices.blocker_ids_by_robot:
        if state.next_index[robot] >= len(templates[robot]):
            continue
        op = templates[robot][state.next_index[robot]]
        ids = [item for item in blocker_ids if item in fixed]
        same = [item for item in ids if fixed[item].robot_id // 2 == robot // 2]
        cross = [item for item in ids if fixed[item].robot_id // 2 != robot // 2]
        terminal.append(dict(robot=robot, next_operation=op.operation_id, kind=op.kind.value,
                             block_id=op.block_id, ready_s=state.completion[robot], start=op.start, end=op.end,
                             same_rail_blockers=len(same), cross_rail_blockers=len(cross),
                             same_examples=same[:4], cross_examples=cross[:4]))
    conflicts = 0
    for i, a in enumerate(native.operations):
        for b in native.operations[i + 1:]:
            if a.robot_id != b.robot_id and operations_conflict(a, b, CONFIG):
                conflicts += 1
    welds = [op for op in native.operations if op.kind is OperationKind.WELD]
    coverage = Counter(op.block_id for op in welds) == Counter(block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2).keys())
    continuous, weld_order = True, True
    for r in range(4):
        ops = [op for op in native.operations if op.robot_id == r]
        continuous &= all(a.end == b.start and a.end_time <= b.start_time + CONFIG.numeric_epsilon
                          for a, b in zip(ops, ops[1:]))
        weld_order &= tuple(op.block_id for op in ops if op.kind is OperationKind.WELD) == solution.routes[r].block_ids
    record.update(
        recovery_rollouts=formal.recovery_rollouts, rollout_budget=formal.rollout_budget,
        frontier_exhausted=formal.frontier_exhausted, baseline_status=outcome.result.status.value,
        baseline_matches_b32_failure=(not formal.feasible and outcome.result.operations == formal.operations),
        terminal_depth=state.depth, next_index=state.next_index,
        template_lengths=[len(templates[r]) for r in range(4)], completion_s=state.completion,
        wait_for_graph=outcome.result.wait_for_graph, wait_for_cycles=scheduler.wait_for_cycles(outcome.result.wait_for_graph),
        first_block=first_block[0] if first_block else None, terminal_blockers=terminal,
        first_delayed_choice=first_delay[0] if first_delay else None,
        blocked_edge_occurrences=blocked_edges, intermediate_wait_cycles=intermediate_cycles,
        delayed_edge_occurrences=delayed_edges, first_cross_delay=first_cross_delay[0] if first_cross_delay else None,
        all_reported_wait_cycles=reported_cycles,
        blocker_finished=[dict(waiting=robot, blocker=other,
                          finished=state.next_index[other] == len(templates[other]))
                          for robot, others in outcome.result.wait_for_graph for other in others],
        native_coverage=coverage, native_inter_robot_conflicts=conflicts,
        native_chain_continuous=continuous, native_weld_order=weld_order,
        native_wait_s=sum(op.duration for op in native.operations if op.kind is OperationKind.WAIT))
    print('WAG_DEADLOCK_DIAGNOSIS', json.dumps(record), flush=True)
    return record


def diagnose_wag_deadlocks(workers):
    historical = [r for r in read_rows(FINAL_RESULTS / BASELINES['WAG+VNS'][2])
                  if r['source'] == BASELINE_SOURCES['WAG+VNS'] and r['layout_id'].startswith('validation_')
                  and r['status'] == 'DEADLOCK']
    layouts = dict(load_layouts('validation'))
    report = ROOT / 'docs' / 'WAG_HGA_BASELINE_REPORT_20261111.md'
    jobs = [((r['layout_id'], layouts[r['layout_id']]), r) for r in historical]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_cpu) as executor:
        records = list(executor.map(wag_deadlock_job, jobs))
    records.sort(key=lambda r: (r['layout_id'], r['seed']))
    lines = ['\n### 13.11 新版PARTIAL_LNS_V1的8次VALIDATION死锁诊断\n',
             '使用当前代码、原始种子对CSV中PARTIAL_LNS_V1的8条失败记录逐条重新执行完整500轮搜索；不读取13.3节的旧路线快照，不修改搜索或B32。只有预算、原生Cmax（容差1e-6）和最终B32状态全部复现后才诊断；未复现者跳过归因。诊断只观察既有派工内核，不反馈搜索。\n',
             '| 排样 | 种子 | 完成轮数 | 重现 | 原生 Cmax/s | 与CSV差值/s | B32 |',
             '|---|---:|---:|---|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['layout_id']} | {r['seed']} | {r['iterations']} | {r['reproduced']} | {r.get('native_cmax_s')} | {r.get('native_cmax_delta_s')} | {r.get('b32_status')} |")
    lines += ['\n逐例诊断（机器人与操作编号沿用代码的0起始编号）：\n']
    for r in records:
        lines.append(f"**{r['layout_id']} / {r['seed']}**：搜索{r['search_runtime_s']:.3f}s。")
        if not r['reproduced']:
            lines += [r['reason'], '']
            continue
        lines.append(f"覆盖完整={r['native_coverage']}，操作链连续={r['native_chain_continuous']}，焊接顺序一致={r['native_weld_order']}，原生机器人间实际冲突={r['native_inter_robot_conflicts']}。")
        lines.append('首次阻塞：`' + json.dumps(r['first_block'], ensure_ascii=False) + '`。')
        terminal = [dict(robot=b['robot'], operation=b['next_operation'], ready_s=b['ready_s'],
                         same=b['same_rail_blockers'], cross=b['cross_rail_blockers'],
                         same_examples=b['same_examples'], cross_examples=b['cross_examples'])
                    for b in r['terminal_blockers']]
        lines.append('终端阻塞：`' + json.dumps(terminal, ensure_ascii=False) + '`。')
        lines.append(f"阻塞边累计同轨/跨轨={r['blocked_edge_occurrences']['same']}/{r['blocked_edge_occurrences']['cross']}；延迟边累计同轨/跨轨={r['delayed_edge_occurrences']['same']}/{r['delayed_edge_occurrences']['cross']}；终端等待图={r['wait_for_graph']}，等待环={r['wait_for_cycles']}，中间完全阻塞等待环={r['intermediate_wait_cycles']}，含可延迟执行操作的关联环={r['all_reported_wait_cycles']}。")
        lines.append(f"阻塞机器人是否完成={r['blocker_finished']}；B32恢复{r['recovery_rollouts']}/{r['rollout_budget']}，前沿耗尽={r['frontier_exhausted']}；诊断原始派工状态={r['baseline_status']}，失败操作时间线与最终B32一致={r['baseline_matches_b32_failure']}。\n")
    lines += ['阻塞边累计是各派工状态返回的关联次数，不是独立冲突数或唯一因果。含可延迟执行操作的关联环不等同于真正等待环。B32失败仅表示该有限恢复策略未找到时间线，不构成数学不可行证明。完整路线快照未保存；13.3节历史11例及其旧快照保持原样。\n']
    with report.open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    print('WAG_DIAGNOSIS_REPORT', report, len(records), flush=True)


FINAL_MODEL = ROOT / 'models' / 'gnn_weld_layout_final.pt'
CMAX_VALUE_MODEL = ROOT / 'models' / 'gnn_weld_layout_cmax_value.pt'
CMAX_VALUE_RESULTS = FINAL_RESULTS / 'gnn_cmax_value_results.csv'
EXPANDED_MODEL = ROOT / 'models' / 'gnn_weld_layout_expanded_dagger.pt'
EXPANDED_RESULTS = FINAL_RESULTS / 'gnn_expanded_dagger.csv'
OWNER_CF_MODEL = ROOT / 'models' / 'gnn_weld_layout_owner_cf.pt'
OWNER_CF_RESULTS = FINAL_RESULTS / 'gnn_owner_cf.csv'
OWNER_CF_FIELDS = ('split', 'stage', 'layout_id', 'method', 'seed', 'parent_id', 'mode',
                   'assigned_parents', 'candidate', 'robot', 'winner', 'preference', 'status', 'cmax_s',
                   'construct_s', 'runtime_s', 'b32_calls', 'processing_r0_s', 'processing_r1_s',
                   'processing_r2_s', 'processing_r3_s', 'max_processing_s', 'move_s', 'diagnostic')
EXPANDED_FIELDS = ('split', 'stage', 'round', 'epoch', 'layout_id', 'method', 'seed', 'status', 'cmax_s',
                   'construct_s', 'runtime_s', 'b32_calls', 'iterations', 'native_cmax_s',
                   'fraction', 'assigned_parents', 'candidate', 'parent_id', 'mode', 'robots',
                   'processing_r0_s', 'processing_r1_s', 'processing_r2_s', 'processing_r3_s',
                   'max_processing_s', 'move_s', 'diagnostic')
CMAX_VALUE_FIELDS = ('split', 'layout_id', 'method', 'epoch', 'fraction', 'assigned_parents', 'candidate',
                     'parent_id', 'mode', 'robots', 'status', 'cmax_s',
                     'processing_r0_s', 'processing_r1_s', 'processing_r2_s', 'processing_r3_s',
                     'max_processing_s', 'move_s', 'runtime_s', 'b32_calls', 'diagnostic')
EPOCHS = 6
VALIDATE_EPOCHS = (2, 4, 6)
JOINT_METHODS = ('GNN_JOINT', 'GNN_JOINT_RECONSTRUCT')
GNN_RESULT_SOURCE = 'COMPLETE_CONSTRUCTION'


def configure_cpu():
    torch.set_num_threads(1)


def joint_model(weights=None, *, cmax_value=False, bounded_assignment=False):
    model = gnn.VNextGNN(workload_lookahead=True, joint_actions=True, cmax_value=cmax_value,
                         bounded_assignment=bounded_assignment)
    if weights is None:
        saved = torch.load(MODEL, map_location='cpu', weights_only=True)
        missing, extra = model.load_state_dict(saved['models']['GNN_VNEXT_WORKLOAD']['state_dict'], strict=False)
        if missing != ['joint_weights'] or extra:
            raise ValueError('unexpected current VNext checkpoint layout')
    else:
        model.load_state_dict(weights)
    return model


def configure_joint_worker(serialized=None):
    configure_cpu()
    global worker_model
    worker_model = joint_model(None if serialized is None else pickle.loads(serialized)).eval()


@dataclass
class Choices:
    indices: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor
    sizes: tuple[int, ...]

    @classmethod
    def from_steps(cls, steps):
        if not steps:
            return None
        return cls(*(torch.cat(columns) for columns in zip(*steps)), tuple(len(step[0]) for step in steps))


@dataclass
class TrainingLayout:
    space: gnn.ParentAssignmentSpace
    assignment: Choices
    insertion: Choices | None
    preferences: tuple
    evidence: dict


def project_routes(routes, directions, kept, blocks):
    pairs = [tuple((b, d) for b, d in zip(route, bits) if b in kept)
             for route, bits in zip(routes, directions)]
    return lns.oriented_route_state(tuple(tuple(b for b, _ in p) for p in pairs),
                                    tuple(tuple(d for _, d in p) for p in pairs), blocks, CONFIG)


def assignment_example(space, model, state, blocks, pending, votes, next_index=None):
    indices, features = model.assignment_features(space, space.graph.node_features, state, blocks, pending, CONFIG)
    target = votes[indices].clone()
    mass = target.sum()
    if not mass > 0:
        return None
    if next_index is not None:
        target += (indices == next_index) * (mass / 3.)
    return indices, features, target / target.sum()


def teacher_examples(space, model, solution, directions, votes, assignment_steps, insertion_steps):
    labels = gnn.parent_assignment_targets(solution, CONFIG)
    labels = sorted(labels, key=lambda a: (-sum(CONFIG.process_time(b.length) for b in space.children[a.pattern]),
                                          a.parent_id))
    blocks = {b.block_id: b for a in labels for b in space.children[a.pattern]}
    nodes = {b.block_id: space.block_nodes[a.pattern, b.block_id] for a in labels for b in space.children[a.pattern]}
    routes = tuple(route.block_ids for route in solution.routes)
    depths = torch.linspace(0, len(labels) - 1, min(8, len(labels))).round().int().tolist()
    for depth in depths:
        selected, pending = labels[:depth], labels[depth:]
        kept = {b.block_id for a in selected for b in space.children[a.pattern]}
        state = project_routes(routes, directions, kept, blocks)
        action = pending[0]
        example = assignment_example(space, model, state, blocks, {a.parent_id for a in pending},
                                     votes, space.action_index[action])
        if example is not None:
            assignment_steps.append(example)
        children = space.children[action.pattern]
        visible = kept | {b.block_id for b in children}
        local = {b: blocks[b] for b in sorted(visible)}
        graph = replace(space.graph, block_index={b: nodes[b] for b in visible})
        for child, robot in zip(children, action.robots):
            kept.add(child.block_id)
            target_state = project_routes(routes, directions, kept, blocks)
            variants = lns.joint_block_candidates(state, child, action.pattern, local, CONFIG,
                                                  robot_id=robot, check_initial_feasibility=False)
            variants = tuple(variants[i] for i in gnn.structural_child_indices(variants, limit=12))
            match = [v.routes == target_state.routes and v.directions == target_state.directions for v, _ in variants]
            if len(variants) > 1 and any(match):
                indices, features = gnn.vnext_insertion_inputs(graph, state, variants, local, CONFIG)
                target = torch.tensor(match, dtype=torch.float32)
                insertion_steps.append((indices, features, target / target.sum()))
            state = target_state


def build_joint(parents, model, **kwargs):
    started = time.perf_counter()
    try:
        return gnn.construct_vnext_gnn(parents, model, CONFIG, **kwargs)
    except (ValueError, ArithmeticError) as error:
        return gnn.JointCompletionResult(None, None, (), time.perf_counter() - started, rejection_reason=str(error))


def plan_key(solution, directions):
    return solution.patterns, solution.routes, directions


def evaluate_once(result, seen, *, blocker_cache=None):
    if result.solution is None:
        return None, False
    key = plan_key(result.solution, result.directions)
    if key in seen:
        return seen[key], False
    if blocker_cache is None:
        schedule = b32(result)
    else:
        orders = {}
        schedule = b32(result, feasible_dispatch_observer=lambda candidate, order:
                       orders.setdefault(candidate.operations, order))
        events = ()
        if schedule.feasible and any(op.kind is OperationKind.WAIT for op in schedule.operations):
            events = selected_schedule_wait_blockers(result.solution, result.directions, schedule,
                                                       orders[schedule.operations], CONFIG)
        blocker_cache[key] = events
    seen[key] = schedule
    return schedule, True


def better_branch(first, second):
    if first is None or second is None:
        return None, None
    feasible = [first.feasible, second.feasible]
    if feasible[0] != feasible[1]:
        return feasible.index(True), 'feasibility'
    if feasible[0] and not math.isclose(first.cmax, second.cmax, rel_tol=1e-9, abs_tol=1e-6):
        return int(second.cmax < first.cmax), 'cmax'
    return None, None


def pack_directed_plan(result):
    if result.solution is None:
        return None
    return dict(patterns=tuple(p.pattern_id for p in result.solution.patterns),
                routes=tuple(r.block_ids for r in result.solution.routes), directions=result.directions)


def unpack_directed_plan(parents, snapshot, space):
    lookup = {p.pattern_id: p for p in space.children}
    solution = canonicalize(parents, tuple(lookup[p] for p in snapshot['patterns']),
                            dict(enumerate(snapshot['routes'])), CONFIG, scope=FORMAL_SCOPE_V2)
    return gnn.JointCompletionResult(solution, snapshot['directions'], (), 0.)


def formal_teacher_records(layout):
    """RULE and the current full-500 WAG; only their final B32 is a teacher label."""
    name, parents = layout
    records, seen = [], {}
    for method in ('RULE/BEAM', 'WAG+VNS'):
        print('FORMAL_TEACHER_START', name, method, flush=True)
        started = time.perf_counter()
        native_cmax, iterations = None, None
        if method == 'RULE/BEAM':
            result = rule_beam(parents)
        else:
            search = baseline_run(parents, method, SEED, math.inf, (), iterations=500,
                                  candidate_evaluator=priority_schedule)
            iterations = search.iterations
            best = search.timely_best
            native_cmax = best.schedule.cmax if best is not None else None
            result = gnn.JointCompletionResult(best.solution if best else None, best.directions if best else None,
                                               (), time.perf_counter() - started,
                                               rejection_reason=search.termination_reason if best is None else None)
        construct_s = time.perf_counter() - started
        schedule, called = evaluate_once(result, seen)
        row = dict(layout_id=name, method=method, seed=SEED, status=schedule.status.value if schedule else 'NO_COMPLETE_SOLUTION',
                   cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                   construct_s=construct_s, runtime_s=time.perf_counter() - started, b32_calls=int(called),
                   iterations=iterations, native_cmax_s=native_cmax,
                   diagnostic=(' | '.join(schedule.diagnostics) if not schedule.feasible else '') if schedule else result.rejection_reason,
                   **beam_validation_metrics(result))
        records.append(dict(row=row, plan=pack_directed_plan(result)))
        print('FORMAL_TEACHER_RESULT', json.dumps(row), flush=True)
    return records, seen


@torch.no_grad()
def collect_teachers(layout, *, records_only=False):
    name, parents = layout
    model = worker_model
    records, seen = formal_teacher_records(layout)
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    teachers = [(record, unpack_directed_plan(parents, record['plan'], space)) for record in records
                if record['row']['status'] == 'FEASIBLE' and record['plan'] is not None
                and (record['row']['method'] != 'WAG+VNS' or record['row']['iterations'] == 500)]
    best_cmax = min((record['row']['cmax_s'] for record, _ in teachers), default=None)
    votes = torch.zeros(len(space.actions))
    assignment_steps, insertion_steps, preferences = [], [], []
    for record, result in teachers:
        solution, directions = result.solution, result.directions
        labels = gnn.parent_assignment_targets(solution, CONFIG)
        teacher_votes = torch.zeros_like(votes)
        teacher_votes[[space.action_index[a] for a in labels]] = best_cmax / record['row']['cmax_s']
        votes += teacher_votes
        if not records_only:
            teacher_examples(space, model, solution, directions, teacher_votes, assignment_steps, insertion_steps)

    depths = {round(len(parents) * .30), round(len(parents) * .65)}
    prefixes = []

    def observe(state, selected, plans, scores):
        if len(selected) not in depths:
            return
        order = torch.argsort(scores, descending=True, stable=True).tolist()
        first = plans[order[0]]
        identity = space.actions[first.assignment_index]
        alternatives = [plans[i] for i in order[1:] if plans[i].assignment_index != first.assignment_index]
        if alternatives:
            second = next((plan for plan in alternatives
                           if space.actions[plan.assignment_index].parent_id == identity.parent_id), alternatives[0])
            prefixes.append((state, dict(selected), (first, second)))

    build_joint(parents, model, space=space, embeddings=embeddings, step_observer=observe)
    branch_calls = branches = 0
    branch_rows, prefix_records = [], []
    preference_types = {}
    for state, selected, plans in prefixes:
        blocks = {b.block_id: b for p in selected.values() for b in space.children[p]}
        example = assignment_example(space, model, state, blocks, set(space.parent_ids) - selected.keys(), votes)
        if example is not None and not records_only:
            assignment_steps.append(example)
        prefix_records.append(dict(patterns=tuple(p.pattern_id for p in selected.values()),
                                   routes=state.routes, directions=state.directions))
        results = []
        for plan in plans:
            started = time.perf_counter()
            action = space.actions[plan.assignment_index]
            result = build_joint(parents, model, space=space, embeddings=embeddings, state=plan.state,
                                 selected={**selected, action.parent_id: action.pattern})
            construct_s = time.perf_counter() - started
            outcome, called = evaluate_once(result, seen)
            branch_calls += int(called)
            branches += 1
            results.append(outcome)
            branch_rows.append(dict(layout_id=name, method='BASE_PREFERENCE_BRANCH', fraction=len(selected) / len(parents),
                                    candidate=len(results), parent_id=action.parent_id, mode=action.pattern.kind.value,
                                    robots=str(action.robots), status=outcome.status.value if outcome else 'NO_COMPLETE_SOLUTION',
                                    cmax_s=outcome.cmax if outcome and outcome.feasible else None,
                                    construct_s=construct_s, runtime_s=time.perf_counter() - started, b32_calls=int(called),
                                    diagnostic=(' | '.join(outcome.diagnostics) if not outcome.feasible else '') if outcome else result.rejection_reason,
                                    **beam_validation_metrics(result)))
        winner, kind = better_branch(*results)
        if winner is None:
            continue
        good, bad = plans[winner], plans[1 - winner]
        preferences.append((torch.tensor((good.assignment_index, bad.assignment_index)),
                            torch.stack((good.assignment_features, bad.assignment_features)),
                            torch.stack((good.outcome, bad.outcome))))
        preference_types[kind] = preference_types.get(kind, 0) + 1

    evidence = dict(name=name, teachers={record['row']['method']: (record['row']['status'], record['row']['cmax_s']) for record in records},
                    prefixes=len(prefixes), branches=branches, branch_b32=branch_calls, preferences=preference_types,
                    assignment_prefixes=len(assignment_steps), insertion_prefixes=len(insertion_steps))
    if records_only:
        return dict(layout_id=name, records=records, preferences=tuple(preferences), prefixes=prefix_records,
                    evidence=evidence, rows=[record['row'] for record in records] + branch_rows)
    return pickle.dumps(TrainingLayout(space, Choices.from_steps(assignment_steps),
                                       Choices.from_steps(insertion_steps), tuple(preferences), evidence))


def ranking_loss(logits, choices):
    return torch.stack([-(target * scores.log_softmax(0)).sum()
                        for scores, target in zip(logits.split(choices.sizes), choices.targets.split(choices.sizes))]).mean()


def repair_prefix(result, space, rng):
    state = result.oriented_state
    labels = gnn.parent_assignment_targets(result.solution, CONFIG)
    blocks = {b.block_id: b for a in labels for b in space.children[a.pattern]}
    selected = {a.parent_id: a.pattern for a in labels}
    heavy = max(range(4), key=state.loads.__getitem__)
    partner = heavy ^ 1
    load_move = torch.tensor([p + sum(arcs) for p, arcs in zip(state.loads, state.arc_costs)])
    candidates = []
    for parent_index, a in enumerate(sorted(labels, key=lambda a: a.parent_id)):
        if heavy not in a.robots:
            continue
        current = space.processing[space.action_index[a]] * 2500.
        choices = space.processing[space.action_parents == parent_index] * 2500.
        alternate = choices[choices[:, heavy] < current[heavy] - 1e-6]
        if not len(alternate):
            continue
        remaining = load_move - current + alternate
        # Prefer transfers that reduce the peak without overloading the peer.
        peak = remaining.amax(dim=1)
        peer = remaining[:, partner]
        adjacent_move = sum(sum(state.arc_costs[heavy][max(0, i-1):i+1])
                            for i, b in enumerate(state.routes[heavy]) if blocks[b].parent_id == a.parent_id)
        score = float((peak + .15 * peer).min()) - .6 * adjacent_move
        candidates.append((score + rng.uniform(0., 70.), a.parent_id))
    if not candidates:
        return None
    removed = {p for _, p in sorted(candidates)[:3]}
    anchors = [b for b in blocks.values() if b.parent_id in removed]
    partner_blocks = [blocks[b] for b in state.routes[partner] if blocks[b].parent_id not in removed]
    if partner_blocks:
        nearby = min(partner_blocks, key=lambda b: min(math.dist(b.start, a.start) for a in anchors))
        removed.add(nearby.parent_id)
    # Keep complete parents when removing a separating block makes siblings adjacent.
    while len(removed) <= 6:
        routes = [[blocks[b] for b in route if blocks[b].parent_id not in removed] for route in state.routes]
        adjacent = {a.parent_id for route in routes for a, b in zip(route, route[1:]) if a.parent_id == b.parent_id}
        if not adjacent:
            kept = {b.block_id for route in routes for b in route}
            partial = project_routes(state.routes, state.directions, kept, blocks)
            return partial, {name: p for name, p in selected.items() if name not in removed}
        removed.update(adjacent)
    return None


@torch.no_grad()
def complete_gnn(parents, model, seed, checkpoints, *, reconstruct, beam_width=1,
                 evaluation_cache=None, result_observer=None):
    if reconstruct and beam_width != 1:
        raise ValueError('the width-two experiment uses construction only')
    started = time.perf_counter()
    repair_rng = Random(seed + 1)
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    calls = repairs = accepted = 0
    seen = {} if evaluation_cache is None else evaluation_cache
    events = []
    best_result = best_schedule = None
    failed_result = failed_schedule = None
    failures = []
    status = 'NO_COMPLETE_SOLUTION'
    diagnostic = ''

    def consider(result):
        nonlocal best_result, best_schedule, failed_result, failed_schedule, calls, status, diagnostic
        if result.solution is None:
            diagnostic = result.rejection_reason or 'NO_COMPLETE_SOLUTION'
            return False
        schedule, called = evaluate_once(result, seen)
        calls += int(called)
        elapsed = time.perf_counter() - started
        if not schedule.feasible:
            if failed_result is None:
                failed_result, failed_schedule = result, schedule
                status = schedule.status.value
                details = schedule.diagnostics[0] if schedule.diagnostics else ''
                if schedule.wait_for_graph:
                    details += ' | blockers=' + str(schedule.wait_for_graph)
                diagnostic = f'{schedule.status.value} {details}'
            failures.append(f'{schedule.status.value}: ' + ' | '.join(schedule.diagnostics))
            return False
        if best_schedule is not None and schedule.cmax >= best_schedule.cmax - 1e-8:
            return False
        best_result, best_schedule = result, schedule
        events.append((elapsed, schedule.cmax))
        return True

    result = build_joint(parents, model, space=space, embeddings=embeddings, beam_width=beam_width)
    initial_seconds = time.perf_counter() - started
    for candidate in result.candidates or (result,):
        consider(candidate)
    if reconstruct and best_result is not None:
        for _ in range(2):
            partial = repair_prefix(best_result, space, repair_rng)
            if partial is None:
                break
            repairs += 1
            state, selected = partial
            repaired = build_joint(parents, model, space=space, embeddings=embeddings, state=state,
                                   selected=selected, rng=repair_rng)
            accepted += int(consider(repaired))

    if best_schedule is None and len(failures) > 1:
        diagnostic += ' | candidates: ' + ' / '.join(failures)
    row = dict(status='FEASIBLE' if best_schedule else status,
               cmax_s=best_schedule.cmax if best_schedule else None, runtime_s=time.perf_counter() - started,
               construct_s=initial_seconds, b32_calls=calls, diagnostic='' if best_schedule else diagnostic,
               source=GNN_RESULT_SOURCE, **final_metrics(best_schedule))
    row.update({f'cmax_{int(t)}_s': min((c for elapsed, c in events if elapsed <= t), default=None)
                for t in checkpoints})
    if result_observer is not None:
        result_observer(best_result or failed_result or result, best_schedule or failed_schedule)
    return row, (1, repairs, accepted)


def beam_validation_metrics(result):
    """Full-plan processing and directed MOVE, also for B32 failures."""
    if result.solution is None:
        return {}
    state = result.oriented_state
    if state is None:
        state = lns.oriented_route_state(tuple(r.block_ids for r in result.solution.routes), result.directions,
                                         block_map(result.solution, CONFIG, scope=FORMAL_SCOPE_V2), CONFIG)
    return dict(**{f'processing_r{r}_s': value for r, value in enumerate(state.loads)}, move_s=state.travel)


def beam_validation_job(layout):
    name, parents = layout
    rows, selected, seen = [], {}, {}
    started = time.perf_counter()
    result = rule_beam(parents)
    schedule, called = evaluate_once(result, seen)
    rows.append(dict(layout_id=name, method='RULE/BEAM',
                     status=schedule.status.value if schedule else 'NO_COMPLETE_SOLUTION',
                     cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                     runtime_s=time.perf_counter() - started, b32_calls=int(called),
                     diagnostic=(' | '.join(schedule.diagnostics) if not schedule.feasible else '')
                     if schedule else result.rejection_reason, **beam_validation_metrics(result)))
    for width in (1, 2):
        def retain(result, schedule):
            selected[width] = result

        row, _ = complete_gnn(parents, worker_model, SEED, (), reconstruct=False, beam_width=width,
                              evaluation_cache=seen, result_observer=retain)
        rows.append(dict(layout_id=name, method=f'GNN_WIDTH{width}',
                         **{key: row[key] for key in ('status', 'cmax_s', 'runtime_s', 'b32_calls', 'diagnostic')},
                         **beam_validation_metrics(selected[width])))
    if rows[1]['cmax_s'] is not None and rows[2]['cmax_s'] is not None:
        delta = rows[2]['cmax_s'] - rows[1]['cmax_s']
        rows[2].update(cmax_change_vs_width1_s=delta,
                       cmax_change_vs_width1_pct=100. * delta / rows[1]['cmax_s'])
    for row in rows:
        print('BEAM_VALIDATION', json.dumps(row), flush=True)
    # Existing insertion actions provide a compact decision trace without log files.
    for width, result in selected.items():
        if result.solution is not None:
            blocks = block_map(result.solution, CONFIG, scope=FORMAL_SCOPE_V2)
            modes = {p.parent_id: p.kind.value for p in result.solution.patterns}
            trace = [(blocks[a.block_id].parent_id, modes[blocks[a.block_id].parent_id],
                      a.robot, a.position, a.direction) for a in result.actions]
            print('BEAM_DECISIONS', name, width, json.dumps(trace[:12]), flush=True)
    return rows


def beam_validation(workers):
    """One deterministic run per method on VALIDATION; no training or time limit."""
    saved = torch.load(FINAL_MODEL, map_location='cpu', weights_only=True)
    weights = pickle.dumps(saved['state_dict'])
    layouts = load_layouts('validation')
    with BEAM_VALIDATION_CSV.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=BEAM_VALIDATION_FIELDS)
        if not stream.tell():
            writer.writeheader()
        done = {(row['layout_id'], row['method']) for row in read_rows(BEAM_VALIDATION_CSV)}
        jobs = [layout for layout in layouts if not all((layout[0], method) in done
                                                       for method in ('RULE/BEAM', 'GNN_WIDTH1', 'GNN_WIDTH2'))]
        with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker,
                                 initargs=(weights,)) as executor:
            for rows in executor.map(beam_validation_job, jobs):
                writer.writerows(row for row in rows if (row['layout_id'], row['method']) not in done)
                stream.flush()
    print('BEAM_VALIDATION_CSV', BEAM_VALIDATION_CSV, flush=True)


def validation_job(layout):
    return complete_gnn(layout[1], worker_model, SEED, (), reconstruct=True)[0]


def validate(model, layouts, workers):
    weights = pickle.dumps(model.state_dict())
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker, initargs=(weights,)) as executor:
        rows = list(executor.map(validation_job, layouts))
    cmax = [r['cmax_s'] for r in rows if r['cmax_s'] is not None]
    return dict(feasible=len(cmax), median_cmax_s=median(cmax) if cmax else None,
                median_runtime_s=median(r['runtime_s'] for r in rows),
                median_construct_s=median(r['construct_s'] for r in rows))


def train(workers):
    started = time.perf_counter()
    training, validation = load_layouts('train'), load_layouts('validation')
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker) as executor:
        samples = []
        for raw in executor.map(collect_teachers, training):
            sample = pickle.loads(raw)
            print('TEACHERS', json.dumps(sample.evidence), flush=True)
            samples.append(sample)
    torch.manual_seed(SEED)
    model = joint_model()
    optimizer = torch.optim.AdamW([{'params': [p for n, p in model.named_parameters() if n != 'joint_weights']},
                                  {'params': [model.joint_weights], 'lr': .02}], lr=3e-4)
    order, rng = list(range(len(samples))), Random(SEED)
    best_key, selected_epoch, selected_state = (-1, -math.inf, -math.inf), None, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(order)
        totals, updates = torch.zeros(3), 0
        for index in order:
            sample = samples[index]
            a, ins = sample.assignment, sample.insertion
            if a is None and ins is None and not sample.preferences:
                continue
            embeddings = model.encode_static(sample.space.graph)
            zero = embeddings.sum() * 0.
            assignment = ranking_loss(model.assignment_scores(model.embed_assignment_rows(
                sample.space, embeddings, a.indices, a.features)), a) if a is not None else zero
            insertion = ranking_loss(model.insertion_logits(embeddings, ins.indices, ins.features), ins) if ins is not None else zero
            ranks = []
            for ids, features, outcome in sample.preferences:
                scores = model.joint_scores(sample.space, embeddings, ids, features, outcome)
                ranks.append(F.softplus(scores[1] - scores[0]))
            preference = torch.stack(ranks).mean() if ranks else assignment.new_zeros(())
            optimizer.zero_grad(set_to_none=True)
            (assignment + .5 * insertion + .7 * preference).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack((assignment.detach(), insertion.detach(), preference.detach()))
            updates += 1
        record = dict(epoch=epoch, **dict(zip(('assignment_loss', 'insertion_loss', 'cmax_rank_loss'),
                                             (totals / max(1, updates)).tolist())))
        if epoch in VALIDATE_EPOCHS:
            result = validate(model.eval(), validation, workers)
            record['validation'] = result
            key = result['feasible'], -(result['median_cmax_s'] or math.inf), -result['median_runtime_s']
            if key > best_key:
                best_key, selected_epoch = key, epoch
                selected_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
        history.append(record)
        print('EPOCH', json.dumps(record), flush=True)
    torch.save(dict(state_dict=selected_state, selected_epoch=selected_epoch, epochs=EPOCHS, seed=SEED,
                    hidden_dim=64, heads=4, workload_lookahead=True, joint_actions=True, history=history,
                    teacher_iterations=500, teacher_methods=('RULE/BEAM', 'WAG+VNS'),
                    validation_stopping='one construction and up to two reconstructions',
                    training=[s.evidence for s in samples], elapsed_s=time.perf_counter() - started), FINAL_MODEL)
    print('MODEL', FINAL_MODEL, 'epoch', selected_epoch, flush=True)


def gnn_test_job(job):
    (name, parents), method, seed = job
    row, stats = complete_gnn(parents, worker_model, seed, CHECKPOINTS,
                              reconstruct=method == JOINT_METHODS[1])
    return dict(layout_id=name, method=method, seed=seed, **row), stats


def test(workers):
    path = FINAL_RESULTS / 'gnn_results.csv'
    if not path.exists():
        retained = [dict(row, method='GNN_VNEXT') for row in retained_rows('GNN_VNEXT_WORKLOAD')]
        write_rows(path, retained)
    done = {(r['layout_id'], r['method'], int(r['seed'])) for r in read_rows(path)
            if r['source'] == GNN_RESULT_SOURCE}
    jobs = [(layout, method, seed) for layout in load_layouts('test') for seed in TEST_SEEDS for method in JOINT_METHODS
            if (layout[0], method, seed) not in done]
    saved = torch.load(FINAL_MODEL, map_location='cpu', weights_only=True)
    weights = pickle.dumps(saved['state_dict'])
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker, initargs=(weights,)) as executor:
        futures = [executor.submit(gnn_test_job, job) for job in jobs]
        for future in as_completed(futures):
            row, stats = future.result()
            write_rows(path, [row], append=True)
            print('TEST', row['method'], row['layout_id'], row['seed'], row['status'], row['cmax_s'],
                  'proposals/repairs/accepted', stats, flush=True)
    write_comparison()


def write_comparison():
    layouts = load_layouts('test')
    hga_layouts = {name for name, _ in layouts}
    rows = [row for name in ('rule_beam_results.csv', 'wag_vns_results.csv', 'hga_results.csv', 'gnn_results.csv')
            for row in read_rows(FINAL_RESULTS / name)
            if (row['method'] not in BASELINES or row['source'].startswith('PAPER_OBJECTIVE_'))
            and (row['method'] not in JOINT_METHODS or row['source'] == GNN_RESULT_SOURCE)
            and (row['method'] != 'HGA' or row['layout_id'] in hga_layouts)]
    rows.sort(key=lambda row: (row['layout_id'], int(row['seed']), row['method']))
    write_rows(FINAL_RESULTS / 'final_comparison.csv', rows)
    print('COMPARISON', len(rows), 'available rows of', 6 * len(layouts) * len(TEST_SEEDS), flush=True)


def summarize_equal_time_baselines():
    """120-second TEST records, with seeds grouped before the layout comparison."""
    layout_ids = {name for name, _ in load_layouts('test')}
    records = [r for _, _, name in BASELINES.values() for r in read_rows(FINAL_RESULTS / name)
               if completed_equal_time_row(r) and r['layout_id'] in layout_ids]
    rows = list({(r['layout_id'], r['method'], int(r['seed'])): r for r in records}.values())
    def average(values):
        return sum(values) / len(values) if values else None
    summaries = []
    for method in BASELINES:
        selected = [r for r in rows if r['method'] == method]
        if not selected:
            continue
        details = [json.loads(r['diagnostic']) for r in selected]
        cmax = [float(r['cmax_s']) for r in selected if r['status'] == 'FEASIBLE']
        times = [float(r['runtime_s']) for r in selected]
        search = [d['search_runtime_s'] for d in details]
        b32_times = [d['b32_runtime_s'] for d in details]
        iterations = [int(r['iterations']) for r in selected]
        overshoot = [d['overshoot_s'] for d in details]
        summary = dict(split='test', method=method, source=EQUAL_TIME_SOURCE,
            algorithm_source=FORMAL_BASELINE_SOURCES[method], time_limit_s=EQUAL_TIME_LIMIT,
            runs=len(selected), expected_runs=len(layout_ids) * len(TEST_SEEDS),
            feasible=len(cmax), feasible_rate=len(cmax) / len(selected),
            deadlock=sum(r['status'] == 'DEADLOCK' for r in selected),
            no_timely_candidate=sum(r['status'] == 'NO_TIMELY_CANDIDATE' for r in selected),
            mean_cmax_s=average(cmax), median_cmax_s=median(cmax) if cmax else None,
            mean_runtime_s=average(times), median_runtime_s=median(times),
            mean_search_runtime_s=average(search), median_search_runtime_s=median(search), max_search_runtime_s=max(search),
            mean_b32_runtime_s=average(b32_times), median_b32_runtime_s=median(b32_times),
            min_iterations=min(iterations), mean_iterations=average(iterations), median_iterations=median(iterations),
            max_iterations=max(iterations), iteration_limit_runs=sum(n == BASELINE_ITERATIONS[method] for n in iterations),
            time_limit_runs=sum(d['termination_reason'] == 'TIME_LIMIT' for d in details),
            mean_overshoot_s=average(overshoot), median_overshoot_s=median(overshoot), max_overshoot_s=max(overshoot),
            late_best_updates=sum(d['late_best_updates'] for d in details),
            b32_calls=sum(int(r['b32_calls']) for r in selected))
        print('BASELINE_EQUAL_TIME_SUMMARY', json.dumps(summary), flush=True)
        summaries.append(summary)
    by_method = {method: {(r['layout_id'], int(r['seed'])): r for r in rows if r['method'] == method}
                 for method in BASELINES}
    common = sorted(by_method['HGA'].keys() & by_method['WAG+VNS'].keys())
    pairs = [(by_method['HGA'][key], by_method['WAG+VNS'][key]) for key in common]
    feasible = [(h, w) for h, w in pairs if h['status'] == w['status'] == 'FEASIBLE']
    delta = [float(h['cmax_s']) - float(w['cmax_s']) for h, w in feasible]
    paired = dict(paired_runs=len(pairs), common_feasible=len(feasible),
        wins=sum(d < -1.e-6 for d in delta), ties=sum(abs(d) <= 1.e-6 for d in delta),
        losses=sum(d > 1.e-6 for d in delta), median_delta_s=median(delta) if delta else None,
        median_change_pct=median(100. * (float(h['cmax_s']) / float(w['cmax_s']) - 1.)
                                 for h, w in feasible) if feasible else None,
        paired_mean_improvement_pct=100. * (sum(float(w['cmax_s']) for _, w in feasible)
            - sum(float(h['cmax_s']) for h, _ in feasible)) / sum(float(w['cmax_s']) for _, w in feasible)
            if feasible else None,
        hga_only_feasible=sum(h['status'] == 'FEASIBLE' and w['status'] != 'FEASIBLE' for h, w in pairs),
        wag_only_feasible=sum(w['status'] == 'FEASIBLE' and h['status'] != 'FEASIBLE' for h, w in pairs),
        both_failed=sum(h['status'] != 'FEASIBLE' and w['status'] != 'FEASIBLE' for h, w in pairs))
    print('BASELINE_EQUAL_TIME_PAIRED_RUNS', json.dumps(paired), flush=True)
    layouts = []
    for name in sorted(layout_ids):
        selected = [(h, w) for h, w in feasible if h['layout_id'] == name]
        h = median(float(h['cmax_s']) for h, _ in selected) if selected else None
        w = median(float(w['cmax_s']) for _, w in selected) if selected else None
        row = dict(layout_id=name, common_feasible_seeds=sorted(int(h['seed']) for h, _ in selected),
            hga_feasible_seeds=sum(r['status'] == 'FEASIBLE' for key, r in by_method['HGA'].items() if key[0] == name),
            wag_feasible_seeds=sum(r['status'] == 'FEASIBLE' for key, r in by_method['WAG+VNS'].items() if key[0] == name),
            hga_median_cmax_s=h, wag_median_cmax_s=w,
            delta_s=h - w if selected else None, change_pct=100. * (h / w - 1.) if selected else None)
        print('BASELINE_EQUAL_TIME_LAYOUT', json.dumps(row), flush=True)
        layouts.append(row)
    available = [r for r in layouts if r['common_feasible_seeds']]
    layout_paired = dict(paired_layouts=len(available), expected_layouts=len(layout_ids),
        all_three_common_feasible=sum(len(r['common_feasible_seeds']) == 3 for r in available),
        wins=sum(r['delta_s'] < -1.e-6 for r in available), ties=sum(abs(r['delta_s']) <= 1.e-6 for r in available),
        losses=sum(r['delta_s'] > 1.e-6 for r in available),
        mean_change_pct=average([r['change_pct'] for r in available]),
        median_change_pct=median(r['change_pct'] for r in available) if available else None,
        median_delta_s=median(r['delta_s'] for r in available) if available else None)
    print('BASELINE_EQUAL_TIME_PAIRED_LAYOUTS', json.dumps(layout_paired), flush=True)
    return dict(summaries=summaries, paired_runs=paired, layouts=layouts, paired_layouts=layout_paired)


def summarize_optimized_baselines(*, time_limit=None):
    """Read the two result files and retained references; never rewrite history."""
    if time_limit is not None:
        if time_limit != EQUAL_TIME_LIMIT:
            raise ValueError('the equal-time comparison uses 120 seconds')
        return summarize_equal_time_baselines()
    records = [r for _, _, name in BASELINES.values() for r in read_rows(FINAL_RESULTS / name)
               if completed_baseline_row(r)]
    rows = list({(r['layout_id'], r['method'], int(r['seed'])): r for r in records}.values())
    references = read_rows(BEAM_VALIDATION_CSV)
    references += read_rows(FINAL_RESULTS / 'rule_beam_results.csv')
    references += [r for r in read_rows(FINAL_RESULTS / 'gnn_results.csv')
                   if r['source'] == GNN_RESULT_SOURCE or r['method'] == 'GNN_VNEXT']
    def average(values):
        return sum(values) / len(values) if values else None
    summaries = []
    for split in ('validation', 'test'):
        layout_ids = {name for name, _ in load_layouts(split)}
        for method in BASELINES:
            selected = [r for r in rows if r['method'] == method and r['layout_id'] in layout_ids]
            if not selected:
                continue
            feasible = [r for r in selected if r['status'] == 'FEASIBLE']
            cmax = [float(r['cmax_s']) for r in feasible]
            times = [float(r['runtime_s']) for r in selected]
            summary = dict(split=split, method=method, source=FORMAL_BASELINE_SOURCES[method],
                           runs=len(selected), expected_runs=len(layout_ids) * len(TEST_SEEDS),
                           feasible=len(feasible), feasible_rate=len(feasible) / len(selected),
                           median_cmax_s=median(cmax) if cmax else None, mean_cmax_s=average(cmax),
                           mean_runtime_s=average(times), median_runtime_s=median(times), max_runtime_s=max(times),
                           iterations=sorted({int(r['iterations']) for r in selected}),
                           b32_calls=sum(int(r['b32_calls']) for r in selected),
                           processing_mean_s=[average([float(r[f'processing_r{i}_s']) for r in feasible]) for i in range(4)],
                           move_mean_s=average([float(r['move_s']) for r in feasible]),
                           wait_mean_s=average([float(r['wait_s']) for r in feasible]))
            print('BASELINE_SUMMARY', json.dumps(summary), flush=True)
            summaries.append(summary)
            comparison_methods = ('RULE/BEAM', 'GNN_WIDTH1') if split == 'validation' else (
                'RULE/BEAM', 'GNN_VNEXT', *JOINT_METHODS)
            for other in comparison_methods:
                pool = [r for r in references if r['method'] == other]
                pairs = []
                for row in feasible:
                    candidates = [r for r in pool if r['layout_id'] == row['layout_id']
                                  and (not r.get('seed') or int(r['seed']) == int(row['seed']))]
                    if candidates and candidates[-1]['status'] == 'FEASIBLE':
                        pairs.append((float(row['cmax_s']), float(candidates[-1]['cmax_s'])))
                delta = [a - b for a, b in pairs]
                print('BASELINE_PAIRED', json.dumps(dict(split=split, method=method, reference=other,
                      common_feasible=len(pairs), wins=sum(d < -1.e-8 for d in delta),
                      ties=sum(abs(d) <= 1.e-8 for d in delta), losses=sum(d > 1.e-8 for d in delta),
                      median_delta_s=median(delta) if pairs else None,
                      median_change_pct=median([100. * (a - b) / b for a, b in pairs]) if pairs else None)), flush=True)
    for split in ('validation', 'test'):
        layout_ids = {name for name, _ in load_layouts(split)}
        by_method = {method: {(r['layout_id'], int(r['seed'])): r for r in rows
                             if r['method'] == method and r['layout_id'] in layout_ids}
                     for method in BASELINES}
        common = sorted(by_method['HGA'].keys() & by_method['WAG+VNS'].keys())
        pairs = [(by_method['HGA'][key], by_method['WAG+VNS'][key]) for key in common]
        feasible = [(h, w) for h, w in pairs if h['status'] == w['status'] == 'FEASIBLE']
        delta = [float(h['cmax_s']) - float(w['cmax_s']) for h, w in feasible]
        paired = dict(split=split, method='HGA', reference='WAG+VNS', paired_runs=len(pairs),
                      common_feasible=len(feasible), wins=sum(d < -1.e-6 for d in delta),
                      ties=sum(abs(d) <= 1.e-6 for d in delta), losses=sum(d > 1.e-6 for d in delta),
                      median_delta_s=median(delta) if delta else None,
                      median_change_pct=median(100. * (float(h['cmax_s']) / float(w['cmax_s']) - 1.)
                                               for h, w in feasible) if feasible else None,
                      hga_only_feasible=sum(h['status'] == 'FEASIBLE' and w['status'] != 'FEASIBLE' for h, w in pairs),
                      wag_only_feasible=sum(w['status'] == 'FEASIBLE' and h['status'] != 'FEASIBLE' for h, w in pairs),
                      both_failed=sum(h['status'] != 'FEASIBLE' and w['status'] != 'FEASIBLE' for h, w in pairs))
        print('BASELINE_FORMAL_PAIRED', json.dumps(paired), flush=True)
        for summary in summaries:
            if summary['split'] == split and summary['method'] == 'HGA':
                summary['paired_vs_wag'] = paired
    return summaries


def value_preferences(schedules):
    """Reuse the existing B32 preference rule, only within one saved prefix."""
    pairs = []
    for first in range(len(schedules)):
        for second in range(first + 1, len(schedules)):
            winner, kind = better_branch(schedules[first], schedules[second])
            if winner is not None:
                indices = (first, second)
                pairs.append((indices[winner], indices[1 - winner], kind))
    return pairs


@torch.no_grad()
def collect_value_branches(layout, *, fractions=(.25, .50, .75), candidate_limit=3, current_policy=False,
                           evaluation_cache=None, retain_evaluations=False):
    name, parents = layout
    started = time.perf_counter()
    model = worker_model
    if model.cmax_value_head is not None and not current_policy:
        raise ValueError('TRAIN continuation labels must use the original frozen GNN')
    model.requires_grad_(False)
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    depths = {math.ceil(len(parents) * fraction): fraction for fraction in fractions}
    prefixes = []

    def observe(state, selected, plans, scores):
        if len(selected) in depths:
            indices = gnn.vnext_value_candidate_indices(space, plans, scores)
            pending = set(space.parent_ids) - selected.keys()
            if candidate_limit == 2 and len(indices) > 1:
                if model.cmax_value_head is None:
                    policy_scores = scores[list(indices)]
                else:
                    inputs = model.cmax_value_inputs(space, embeddings, state, pending,
                                                     tuple(plans[i] for i in indices), scores[list(indices)])
                    policy_scores = model.cmax_values(inputs)
                order = [indices[i] for i in torch.argsort(policy_scores, descending=True, stable=True).tolist()]
                first = space.actions[plans[order[0]].assignment_index]
                alternatives = [i for i in order[1:] if space.actions[plans[i].assignment_index] != first]
                same_parent = [i for i in alternatives if space.actions[plans[i].assignment_index].parent_id == first.parent_id]
                second = next(iter(same_parent or alternatives or order[1:]))
                indices = (order[0], second)
            else:
                indices = indices[:candidate_limit]
            chosen = tuple(plans[i] for i in indices)
            features = model.cmax_value_inputs(space, embeddings, state, pending, chosen, scores[list(indices)])
            prefixes.append((state, dict(selected), chosen, features.cpu()))

    original = build_joint(parents, model, space=space, embeddings=embeddings,
                           beam_width=1, step_observer=observe)
    original_seconds = time.perf_counter() - started
    groups, rows = [], []
    seen = {} if evaluation_cache is None else evaluation_cache
    for state, selected, plans, features in prefixes:
        fraction = depths[len(selected)]
        schedules, descriptions = [], []
        for index, plan in enumerate(plans, 1):
            branch_started = time.perf_counter()
            assignment = space.actions[plan.assignment_index]
            result = build_joint(parents, model, space=space, embeddings=embeddings, beam_width=1,
                                 state=plan.state, selected={**selected, assignment.parent_id: assignment.pattern})
            construct_s = time.perf_counter() - branch_started
            schedule, called = evaluate_once(result, seen)
            elapsed = time.perf_counter() - branch_started
            diagnostic = (' | '.join(schedule.diagnostics) if not schedule.feasible else '') if schedule else result.rejection_reason
            row = dict(split='train', layout_id=name, method='FROZEN_GNN_BRANCH', fraction=fraction,
                       assigned_parents=len(selected), candidate=index, parent_id=assignment.parent_id,
                       mode=assignment.pattern.kind.value, robots=str(assignment.robots),
                       status=schedule.status.value if schedule else 'NO_COMPLETE_SOLUTION',
                       cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                       runtime_s=elapsed, b32_calls=int(called), diagnostic=diagnostic,
                       **beam_validation_metrics(result))
            if current_policy:
                row['construct_s'] = construct_s
            rows.append(row)
            schedules.append(schedule)
            descriptions.append(dict(parent_id=assignment.parent_id, kind=assignment.pattern.kind.value,
                                     rail=assignment.pattern.rail.value if assignment.pattern.rail else None,
                                     t=assignment.pattern.t, point_id=assignment.pattern.point_id, robots=assignment.robots,
                                     actions=[dict(block_id=a.block_id, robot=a.robot, position=a.position,
                                                   direction=a.direction, neighbour_changes=a.neighbour_changes)
                                              for a in plan.actions]))
        groups.append(dict(fraction=fraction, assigned_parents=len(selected), routes=state.routes,
                           directions=state.directions, loads=state.loads, arc_costs=state.arc_costs,
                           selected=[dict(parent_id=p.parent_id, kind=p.kind.value,
                                          rail=p.rail.value if p.rail else None, t=p.t, point_id=p.point_id, mandatory=p.mandatory)
                                     for p in selected.values()],
                           pending=tuple(sorted(set(space.parent_ids) - selected.keys())),
                           features=features, candidates=descriptions, preferences=value_preferences(schedules)))
    sample = dict(layout_id=name, parents=len(parents), groups=groups, rows=rows,
                  original_construct_s=original_seconds, original_complete=original.solution is not None,
                  original_diagnostic=original.rejection_reason, runtime_s=time.perf_counter() - started,
                  policy_has_value=model.cmax_value_head is not None)
    if retain_evaluations:
        sample['evaluations'] = pickle.dumps(seen)
    return sample


def value_reference_rows(split='validation'):
    if split == 'validation':
        return [dict(split=split, **{k: row[k] for k in CMAX_VALUE_FIELDS if k != 'split' and k in row})
                for row in read_rows(BEAM_VALIDATION_CSV) if row['method'] in ('RULE/BEAM', 'GNN_WIDTH1')]
    # TEST is read only after the fixed VALIDATION acceptance conditions pass.
    rule = [r for r in read_rows(FINAL_RESULTS / 'rule_beam_results.csv') if r['method'] == 'RULE/BEAM']
    old = [dict(r, method='GNN_WIDTH1') for r in read_rows(FINAL_RESULTS / 'gnn_results.csv')
           if r['method'] == 'GNN_JOINT' and r['source'] == GNN_RESULT_SOURCE]
    chosen = {}
    for row in (*rule, *old):
        chosen.setdefault((row['layout_id'], row['method']), row)
    return [dict(split=split, **{k: row[k] for k in CMAX_VALUE_FIELDS if k != 'split' and k in row})
            for row in chosen.values()]


def save_cmax_value(saved):
    """The two requested artifacts also retain enough TRAIN data to resume."""
    rows = [row for layout in saved['training'] for row in layout['rows']]
    rows += [row for epoch in saved.get('validation', {}).values() for row in epoch]
    rows += saved.get('test', [])
    for row in rows:
        if all(row.get(f'processing_r{r}_s') not in (None, '') for r in range(4)):
            row['max_processing_s'] = max(float(row[f'processing_r{r}_s']) for r in range(4))
    torch.save(saved, CMAX_VALUE_MODEL)
    with CMAX_VALUE_RESULTS.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CMAX_VALUE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def configure_value_worker(serialized):
    configure_cpu()
    global worker_model
    worker_model = joint_model(pickle.loads(serialized), cmax_value=True).eval()
    worker_model.requires_grad_(False)


def value_validation_job(job):
    (name, parents), split, epoch, seen = job
    chosen = []
    row, _ = complete_gnn(parents, worker_model, SEED, (), reconstruct=False, beam_width=1,
                          evaluation_cache=seen, result_observer=lambda result, schedule: chosen.append(result))
    row = dict(split=split, layout_id=name, method='GNN_CMAX_VALUE', epoch=epoch,
               **{k: row[k] for k in ('status', 'cmax_s', 'runtime_s', 'b32_calls', 'diagnostic')},
               **beam_validation_metrics(chosen[0]))
    print('CMAX_VALUE_EVAL', json.dumps(row), flush=True)
    return row, seen


def validate_value_model(model, epoch, caches, split='validation'):
    layouts = load_layouts(split)
    jobs = [(layout, split, epoch, caches.setdefault(layout[0], {})) for layout in layouts]
    weights = pickle.dumps(model.state_dict())
    # Serial validation keeps measured inference runtimes comparable to Width=1.
    with ProcessPoolExecutor(max_workers=1, initializer=configure_value_worker, initargs=(weights,)) as executor:
        results = list(executor.map(value_validation_job, jobs))
    for layout, (_, seen) in zip(layouts, results):
        caches[layout[0]] = seen
    return [row for row, _ in results]


def cmax_value_summary(rows, references):
    feasible = [row for row in rows if row['status'] == 'FEASIBLE']

    def peak(row):
        if all(row.get(f'processing_r{r}_s') not in ('', None) for r in range(4)):
            return max(float(row[f'processing_r{r}_s']) for r in range(4))
        return float(row['max_processing_s'])

    summary = dict(feasible=len(feasible), runs=len(rows), b32_calls=sum(int(r['b32_calls']) for r in rows),
                   median_cmax_s=median(float(r['cmax_s']) for r in feasible) if feasible else None,
                   median_runtime_s=median(float(r['runtime_s']) for r in rows),
                   median_max_processing_s=median(peak(r) for r in feasible) if feasible else None,
                   median_move_s=median(float(r['move_s']) for r in feasible) if feasible else None)
    paired = {}
    for method in ('RULE/BEAM', 'GNN_WIDTH1'):
        lookup = {r['layout_id']: r for r in references if r['method'] == method}
        pairs = [(row, lookup[row['layout_id']]) for row in feasible
                 if row['layout_id'] in lookup and lookup[row['layout_id']]['status'] == 'FEASIBLE']
        deltas = [float(a['cmax_s']) - float(b['cmax_s']) for a, b in pairs]
        paired[method] = dict(common=len(pairs), wins=sum(d < -1e-6 for d in deltas),
                              ties=sum(abs(d) <= 1e-6 for d in deltas), losses=sum(d > 1e-6 for d in deltas),
                              median_cmax_change_pct=median(100. * (float(a['cmax_s']) / float(b['cmax_s']) - 1.)
                                                            for a, b in pairs) if pairs else None,
                              median_max_processing_delta_s=median(peak(a) - peak(b) for a, b in pairs) if pairs else None,
                              median_move_delta_s=median(float(a['move_s']) - float(b['move_s']) for a, b in pairs) if pairs else None)
    summary['paired'] = paired
    return summary


def value_model_key(summary):
    rule = summary['paired']['RULE/BEAM']
    cmax = rule['median_cmax_change_pct']
    return (summary['feasible'], -(cmax if cmax is not None else math.inf),
            -(summary['median_max_processing_s'] if summary['median_max_processing_s'] is not None else math.inf),
            -(summary['median_move_s'] if summary['median_move_s'] is not None else math.inf),
            -summary['median_runtime_s'])


def cmax_value_experiment(workers):
    if CMAX_VALUE_MODEL.exists():
        saved = torch.load(CMAX_VALUE_MODEL, map_location='cpu', weights_only=True)
    else:
        original = torch.load(FINAL_MODEL, map_location='cpu', weights_only=True)
        saved = dict(base_state_dict=original['state_dict'], training=[], history=[], validation={},
                     selected_epoch=None, completed_epoch=0, epochs=6, seed=SEED)
    done = {row['layout_id'] for row in saved['training']}
    jobs = [layout for layout in load_layouts('train') if layout[0] not in done]
    if jobs:
        weights = pickle.dumps(saved['base_state_dict'])
        with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker, initargs=(weights,)) as executor:
            futures = [executor.submit(collect_value_branches, layout) for layout in jobs]
            for future in as_completed(futures):
                sample = future.result()
                saved['training'].append(sample)
                save_cmax_value(saved)
                print('CMAX_VALUE_TRAIN_DATA', sample['layout_id'], 'groups', len(sample['groups']),
                      'branches', len(sample['rows']), 'B32', sum(r['b32_calls'] for r in sample['rows']),
                      'seconds', sample['runtime_s'], flush=True)
    samples = sorted(saved['training'], key=lambda s: s['layout_id'])
    groups = [group for sample in samples for group in sample['groups']]
    trainable = [group for group in groups if group['preferences']]
    if not trainable:
        raise ValueError('no actual B32 preference pairs were obtained')
    torch.manual_seed(SEED)
    model = joint_model(saved['base_state_dict'])
    model.enable_cmax_value()
    optimizer = torch.optim.AdamW(model.cmax_value_head.parameters(), lr=1e-3, weight_decay=1e-4)
    if saved['completed_epoch']:
        model.load_state_dict(saved['last_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    else:
        inputs = torch.cat([group['features'] for group in groups])
        model.cmax_value_mean.copy_(inputs.mean(0))
        model.cmax_value_scale.copy_(inputs.std(0, unbiased=False).clamp_min(.05))
    caches = {}
    references = value_reference_rows()
    for epoch in range(saved['completed_epoch'] + 1, saved['epochs'] + 1):
        model.train()
        order = list(range(len(trainable)))
        Random(SEED + epoch).shuffle(order)
        losses = []
        for index in order:
            group = trainable[index]
            scores = model.cmax_values(group['features'])
            good, bad = zip(*[(a, b) for a, b, _ in group['preferences']])
            loss = F.softplus(scores[list(bad)] - scores[list(good)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        record = dict(epoch=epoch, ranking_loss=sum(losses) / len(losses))
        if epoch in (2, 4, 6):
            rows = validate_value_model(model.eval(), epoch, caches)
            saved['validation'][epoch] = rows
            summary = cmax_value_summary(rows, references)
            record['validation'] = summary
            if saved['selected_epoch'] is None or value_model_key(summary) > value_model_key(saved['selected_summary']):
                saved['selected_epoch'], saved['selected_summary'] = epoch, summary
                saved['state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved['history'].append(record)
        saved['completed_epoch'] = epoch
        saved['last_state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved['optimizer_state_dict'] = optimizer.state_dict()
        save_cmax_value(saved)
        print('CMAX_VALUE_EPOCH', json.dumps(record), flush=True)
    saved['base_parameters_unchanged'] = all(torch.equal(value, saved['state_dict'][key])
                                             for key, value in saved['base_state_dict'].items())
    rule = saved['selected_summary']['paired']['RULE/BEAM']
    # Use no invented tolerance for "not above RULE": a nonpositive paired median.
    saved['acceptance'] = dict(feasible=saved['selected_summary']['feasible'] >= 7,
                               paired_cmax=rule['median_cmax_change_pct'] is not None and rule['median_cmax_change_pct'] < 0.,
                               wins=rule['common'] > 0 and rule['wins'] >= rule['losses'],
                               processing=rule['median_max_processing_delta_s'] is not None
                                          and rule['median_max_processing_delta_s'] <= CONFIG.numeric_epsilon,
                               runtime=saved['selected_summary']['median_runtime_s'] <= 20.)
    save_cmax_value(saved)
    print('CMAX_VALUE_SELECTED', saved['selected_epoch'], json.dumps(saved['selected_summary']),
          'acceptance', json.dumps(saved['acceptance']), 'frozen_body', saved['base_parameters_unchanged'], flush=True)
    if all(saved['acceptance'].values()) and not saved.get('test'):
        model.load_state_dict(saved['state_dict'])
        model.requires_grad_(False)
        saved['test'] = validate_value_model(model.eval(), saved['selected_epoch'], {}, split='test')
        save_cmax_value(saved)
        print('CMAX_VALUE_TEST', json.dumps(cmax_value_summary(saved['test'], value_reference_rows('test'))), flush=True)


def load_expanded_experiment():
    if EXPANDED_MODEL.exists():
        return torch.load(EXPANDED_MODEL, map_location='cpu', weights_only=True)
    source = torch.load(FINAL_MODEL, map_location='cpu', weights_only=True)
    return dict(initial_state_dict=source['state_dict'], teachers=[], references=[], base_history=[],
                base_validation={}, base_completed_epoch=0, rounds={}, seed=SEED)


def save_expanded_experiment(saved):
    rows = []
    for sample in saved['teachers']:
        rows.extend(dict(row, split='train_expanded', stage='teacher' if row['method'] in ('RULE/BEAM', 'WAG+VNS') else 'base_branch')
                    for row in sample['rows'])
    for reference in saved['references']:
        rows.extend(dict(record['row'], split='validation_fresh', stage='reference') for record in reference['records'])
    for values in saved['base_validation'].values():
        rows.extend(values)
    for data in saved['rounds'].values():
        for sample in data.get('samples', []):
            rows.extend(sample['rows'])
        for values in data.get('validation', {}).values():
            rows.extend(values)
    rows.extend(saved.get('test', []))
    for row in rows:
        if all(row.get(f'processing_r{r}_s') not in (None, '') for r in range(4)):
            row['max_processing_s'] = max(float(row[f'processing_r{r}_s']) for r in range(4))
    torch.save(saved, EXPANDED_MODEL)
    with EXPANDED_RESULTS.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPANDED_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in EXPANDED_FIELDS} for row in rows)


def expanded_teacher_job(layout):
    started = time.perf_counter()
    sample = collect_teachers(layout, records_only=True)
    sample['runtime_s'] = time.perf_counter() - started
    return sample


def expanded_reference_job(layout):
    records, seen = formal_teacher_records(layout)
    return dict(layout_id=layout[0], records=records, seen=pickle.dumps(seen))


def expanded_teachers(workers):
    saved = load_expanded_experiment()
    done = {sample['layout_id'] for sample in saved['teachers']}
    layouts = load_layouts('train_expanded')
    references = load_layouts('validation_fresh')
    if len(layouts) != 120 or len(references) != 20:
        raise ValueError('complete the requested 120/20 expanded packing generation first')
    weights = pickle.dumps(saved['initial_state_dict'])
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_joint_worker, initargs=(weights,)) as executor:
        jobs = [layout for layout in layouts if layout[0] not in done]
        futures = [executor.submit(expanded_teacher_job, layout) for layout in jobs]
        for future in as_completed(futures):
            sample = future.result()
            saved['teachers'].append(sample)
            save_expanded_experiment(saved)
            print('EXPANDED_TEACHERS_SAVED', len(saved['teachers']), '/ 120', sample['layout_id'],
                  json.dumps(sample['evidence']), 'seconds', sample['runtime_s'], flush=True)
        done = {reference['layout_id'] for reference in saved['references']}
        futures = [executor.submit(expanded_reference_job, layout) for layout in references if layout[0] not in done]
        for future in as_completed(futures):
            sample = future.result()
            saved['references'].append(sample)
            save_expanded_experiment(saved)
            print('FRESH_REFERENCES_SAVED', len(saved['references']), '/ 20', sample['layout_id'], flush=True)
    return saved


def expanded_training_layout(layout, record, model):
    """Rebuild the existing demonstrations from saved, B32-feasible teachers."""
    _, parents = layout
    space = gnn.parent_assignment_space(parents, CONFIG)
    feasible = [item for item in record['records'] if item['row']['status'] == 'FEASIBLE'
                and (item['row']['method'] != 'WAG+VNS' or item['row']['iterations'] == 500)]
    best = min((item['row']['cmax_s'] for item in feasible), default=None)
    votes = torch.zeros(len(space.actions))
    assignment_steps, insertion_steps = [], []
    for item in feasible:
        result = unpack_directed_plan(parents, item['plan'], space)
        labels = gnn.parent_assignment_targets(result.solution, CONFIG)
        teacher_votes = torch.zeros_like(votes)
        teacher_votes[[space.action_index[a] for a in labels]] = best / item['row']['cmax_s']
        votes += teacher_votes
        teacher_examples(space, model, result.solution, result.directions, teacher_votes, assignment_steps, insertion_steps)
    lookup = {pattern.pattern_id: pattern for pattern in space.children}
    for prefix in record['prefixes']:
        selected = {lookup[key].parent_id: lookup[key] for key in prefix['patterns']}
        blocks = {b.block_id: b for pattern in selected.values() for b in space.children[pattern]}
        state = lns.oriented_route_state(prefix['routes'], prefix['directions'], blocks, CONFIG)
        example = assignment_example(space, model, state, blocks, set(space.parent_ids) - selected.keys(), votes)
        if example is not None:
            assignment_steps.append(example)
    evidence = dict(record['evidence'], assignment_prefixes=len(assignment_steps), insertion_prefixes=len(insertion_steps))
    return TrainingLayout(space, Choices.from_steps(assignment_steps), Choices.from_steps(insertion_steps),
                          record['preferences'], evidence)


def configure_expanded_worker(serialized):
    configure_cpu()
    from numba import set_num_threads
    set_num_threads(1)
    weights = pickle.loads(serialized)
    global worker_model
    worker_model = joint_model(weights, cmax_value='cmax_value_mean' in weights).eval()
    worker_model.requires_grad_(False)


def expanded_validation_job(job):
    layout, stage, round_index, epoch, cached = job
    selected = []
    seen = pickle.loads(cached) if cached else {}
    row, _ = complete_gnn(layout[1], worker_model, SEED, (), reconstruct=False, beam_width=1,
                          evaluation_cache=seen, result_observer=lambda result, schedule: selected.append(result))
    row = dict(split='test' if stage == 'final_test' else 'validation_fresh', stage=stage, round=round_index, epoch=epoch, layout_id=layout[0],
               method='GNN_BASE' if worker_model.cmax_value_head is None else 'GNN_DAGGER', seed=SEED,
               **{key: row[key] for key in ('status', 'cmax_s', 'construct_s', 'runtime_s', 'b32_calls', 'diagnostic')},
               **beam_validation_metrics(selected[0]), plan=pack_directed_plan(selected[0]))
    print('FRESH_GNN_RESULT', json.dumps({k: v for k, v in row.items() if k != 'plan'}), flush=True)
    return row, pickle.dumps(seen)


def expanded_validate(model, saved, stage, round_index, epoch, split='validation_fresh'):
    layouts = load_layouts(split)
    caches = saved.setdefault('validation_seen', {r['layout_id']: r['seen'] for r in saved['references']})
    jobs = [(layout, stage, round_index, epoch, caches.get(layout[0])) for layout in layouts]
    with ProcessPoolExecutor(max_workers=1, initializer=configure_expanded_worker,
                             initargs=(pickle.dumps(model.state_dict()),)) as executor:
        results = list(executor.map(expanded_validation_job, jobs))
    rows = []
    for layout, (row, cache) in zip(layouts, results):
        caches[layout[0]] = cache
        if split == 'test':
            row['split'], row['stage'] = 'test', 'final_test'
        rows.append(row)
    return rows


def expanded_summary(rows, references):
    feasible = [row for row in rows if row['status'] == 'FEASIBLE']
    def peak(row):
        return max(float(row[f'processing_r{i}_s']) for i in range(4)) if row.get('processing_r0_s') not in (None, '') else float(row['max_processing_s'])
    summary = dict(runs=len(rows), feasible=len(feasible),
                   median_cmax_s=median(float(row['cmax_s']) for row in feasible) if feasible else None,
                   median_construct_s=median(float(row['construct_s']) for row in rows),
                   median_runtime_s=median(float(row['runtime_s']) for row in rows),
                   b32_calls=sum(int(row['b32_calls']) for row in rows), paired={})
    for method in ('WAG+VNS', 'RULE/BEAM'):
        lookup = {row['layout_id']: row for row in references if row['method'] == method}
        pairs = [(row, lookup[row['layout_id']]) for row in feasible
                 if row['layout_id'] in lookup and lookup[row['layout_id']]['status'] == 'FEASIBLE']
        changes = [float(a['cmax_s']) - float(b['cmax_s']) for a, b in pairs]
        summary['paired'][method] = dict(common=len(pairs), wins=sum(d < -1e-6 for d in changes),
            ties=sum(abs(d) <= 1e-6 for d in changes), losses=sum(d > 1e-6 for d in changes),
            median_change_pct=median(100. * (float(a['cmax_s']) / float(b['cmax_s']) - 1.) for a, b in pairs) if pairs else None,
            median_processing_delta_s=median(peak(a) - peak(b) for a, b in pairs) if pairs else None,
            median_move_delta_s=median(float(a['move_s']) - float(b['move_s']) for a, b in pairs) if pairs else None)
    return summary


def expanded_model_key(summary):
    changes = [summary['paired'][method]['median_change_pct'] for method in ('WAG+VNS', 'RULE/BEAM')]
    return (summary['feasible'], *(-(value if value is not None else math.inf) for value in changes),
            -summary['median_construct_s'])


def train_expanded_base(saved):
    if saved['base_completed_epoch'] == EPOCHS:
        return
    torch.manual_seed(SEED)
    model = joint_model(saved['initial_state_dict'])
    layouts = dict(load_layouts('train_expanded'))
    samples = []
    for record in sorted(saved['teachers'], key=lambda r: r['layout_id']):
        sample = expanded_training_layout((record['layout_id'], layouts[record['layout_id']]), record, model)
        record['evidence'] = sample.evidence
        samples.append(sample)
        print('EXPANDED_DEMONSTRATIONS', record['layout_id'], json.dumps(sample.evidence), flush=True)
    optimizer = torch.optim.AdamW([{'params': [p for name, p in model.named_parameters() if name != 'joint_weights']},
                                  {'params': [model.joint_weights], 'lr': .02}], lr=3e-4)
    if saved['base_completed_epoch']:
        model.load_state_dict(saved['base_last_state_dict'])
        optimizer.load_state_dict(saved['base_optimizer_state_dict'])
    references = [record['row'] for reference in saved['references'] for record in reference['records']]
    for epoch in range(saved['base_completed_epoch'] + 1, EPOCHS + 1):
        model.train()
        order = list(range(len(samples)))
        Random(SEED + epoch).shuffle(order)
        totals, updates = torch.zeros(3), 0
        for index in order:
            sample = samples[index]
            a, ins = sample.assignment, sample.insertion
            if a is None and ins is None and not sample.preferences:
                continue
            embeddings = model.encode_static(sample.space.graph)
            zero = embeddings.sum() * 0.
            assignment = ranking_loss(model.assignment_scores(model.embed_assignment_rows(
                sample.space, embeddings, a.indices, a.features)), a) if a is not None else zero
            insertion = ranking_loss(model.insertion_logits(embeddings, ins.indices, ins.features), ins) if ins is not None else zero
            ranks = []
            for ids, features, outcome in sample.preferences:
                scores = model.joint_scores(sample.space, embeddings, ids, features, outcome)
                ranks.append(F.softplus(scores[1] - scores[0]))
            preference = torch.stack(ranks).mean() if ranks else zero
            optimizer.zero_grad(set_to_none=True)
            (assignment + .5 * insertion + .7 * preference).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            totals += torch.stack((assignment.detach(), insertion.detach(), preference.detach()))
            updates += 1
            if updates % 20 == 0:
                print('EXPANDED_BASE_PROGRESS', epoch, updates, '/ 120', flush=True)
        record = dict(epoch=epoch, labelled_layouts=updates,
                      **dict(zip(('assignment_loss', 'insertion_loss', 'cmax_rank_loss'), (totals / max(1, updates)).tolist())))
        if epoch in VALIDATE_EPOCHS:
            rows = expanded_validate(model.eval(), saved, 'base_validation', 0, epoch)
            saved['base_validation'][epoch] = rows
            summary = expanded_summary(rows, references)
            record['validation'] = summary
            if 'base_selected_epoch' not in saved or expanded_model_key(summary) > expanded_model_key(saved['base_selected_summary']):
                saved['base_selected_epoch'], saved['base_selected_summary'] = epoch, summary
                saved['base_state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved['base_history'].append(record)
        saved['base_completed_epoch'] = epoch
        saved['base_last_state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved['base_optimizer_state_dict'] = optimizer.state_dict()
        save_expanded_experiment(saved)
        print('EXPANDED_BASE_EPOCH', json.dumps(record), flush=True)


def stratified_dagger_layouts(records):
    ordered = sorted(records, key=lambda row: (len(row['parents']), row['layout_id']))
    chosen, strata = [], []
    for weld_bin in range(3):
        group = sorted(ordered[weld_bin * 40:(weld_bin + 1) * 40],
                       key=lambda row: (row['rectangle_utilization'], len(row['parts']), row['layout_id']))
        for size_bin in range(2):
            members = group[size_bin * 20:(size_bin + 1) * 20]
            selection = [members[round(i * (len(members) - 1) / 9)] for i in range(10)]
            chosen.extend(row['layout_id'] for row in selection)
            strata.append(dict(weld_bin=weld_bin, size_bin=size_bin, layouts=[row['layout_id'] for row in selection],
                               parent_range=[min(len(row['parents']) for row in selection), max(len(row['parents']) for row in selection)],
                               utilization_range=[min(row['rectangle_utilization'] for row in selection), max(row['rectangle_utilization'] for row in selection)]))
    return chosen, strata


def dagger_branch_job(job):
    layout, round_index, cached = job
    seen = pickle.loads(cached) if cached else {}
    sample = collect_value_branches(layout, fractions=(.05, .15, .50, .85), candidate_limit=2, current_policy=True,
                                    evaluation_cache=seen, retain_evaluations=True)
    for row in sample['rows']:
        row.update(split='train_expanded', stage='dagger_branch', round=round_index, method='GNN_DAGGER_BRANCH')
    print('DAGGER_BRANCH_RESULT', round_index, layout[0], 'prefixes', len(sample['groups']),
          'branches', len(sample['rows']), 'B32', sum(r['b32_calls'] for r in sample['rows']),
          'value_policy', sample['policy_has_value'], 'seconds', sample['runtime_s'], flush=True)
    return sample


def collect_dagger_round(saved, round_index, workers):
    data = saved['rounds'].get(round_index)
    if data is None:
        policy = saved['base_state_dict'] if round_index == 1 else saved['rounds'][1]['state_dict']
        data = dict(policy_state_dict={key: value.clone() for key, value in policy.items()}, samples=[],
                    history=[], validation={}, completed_epoch=0)
        saved['rounds'][round_index] = data
        save_expanded_experiment(saved)
    if 'dagger_layouts' not in saved:
        records = [json.loads(line) for line in (DATA / 'train_expanded.jsonl').read_text(encoding='utf-8').splitlines()]
        saved['dagger_layouts'], saved['dagger_strata'] = stratified_dagger_layouts(records)
    layouts = dict(load_layouts('train_expanded'))
    done = {sample['layout_id'] for sample in data['samples']}
    previous = {sample['layout_id']: sample.get('evaluations') for sample in saved['rounds'].get(round_index - 1, {}).get('samples', [])}
    jobs = [((name, layouts[name]), round_index, previous.get(name)) for name in saved['dagger_layouts'] if name not in done]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_expanded_worker,
                             initargs=(pickle.dumps(data['policy_state_dict']),)) as executor:
        futures = [executor.submit(dagger_branch_job, job) for job in jobs]
        for future in as_completed(futures):
            data['samples'].append(future.result())
            save_expanded_experiment(saved)
            print('DAGGER_SAVED', round_index, len(data['samples']), '/ 60', flush=True)


def train_dagger_round(saved, round_index):
    data = saved['rounds'][round_index]
    if data['completed_epoch'] == EPOCHS:
        return
    source = data['policy_state_dict']
    torch.manual_seed(SEED + round_index)
    model = joint_model(source, cmax_value='cmax_value_mean' in source)
    model.enable_cmax_value()
    groups = [group for index in range(1, round_index + 1)
              for sample in sorted(saved['rounds'][index]['samples'], key=lambda sample: sample['layout_id'])
              for group in sample['groups']]
    trainable = [group for group in groups if group['preferences']]
    if not trainable:
        raise ValueError(f'DAgger round {round_index} obtained no true B32 preferences')
    optimizer = torch.optim.AdamW(model.cmax_value_head.parameters(), lr=1e-3, weight_decay=1e-4)
    if data['completed_epoch']:
        model.load_state_dict(data['last_state_dict'])
        optimizer.load_state_dict(data['optimizer_state_dict'])
    elif round_index == 1:
        features = torch.cat([group['features'] for group in groups])
        model.cmax_value_mean.copy_(features.mean(0))
        model.cmax_value_scale.copy_(features.std(0, unbiased=False).clamp_min(.05))
    references = [record['row'] for reference in saved['references'] for record in reference['records']]
    for epoch in range(data['completed_epoch'] + 1, EPOCHS + 1):
        model.train()
        order = list(range(len(trainable)))
        Random(SEED + round_index * 100 + epoch).shuffle(order)
        losses = []
        for index in order:
            group = trainable[index]
            scores = model.cmax_values(group['features'])
            good, bad = zip(*[(a, b) for a, b, _ in group['preferences']])
            loss = F.softplus(scores[list(bad)] - scores[list(good)]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        record = dict(epoch=epoch, ranking_loss=sum(losses) / len(losses), labelled_prefixes=len(trainable))
        if epoch in VALIDATE_EPOCHS:
            rows = expanded_validate(model.eval(), saved, 'dagger_validation', round_index, epoch)
            data['validation'][epoch] = rows
            summary = expanded_summary(rows, references)
            record['validation'] = summary
            if 'selected_epoch' not in data or expanded_model_key(summary) > expanded_model_key(data['selected_summary']):
                data['selected_epoch'], data['selected_summary'] = epoch, summary
                data['state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        data['history'].append(record)
        data['completed_epoch'] = epoch
        data['last_state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        data['optimizer_state_dict'] = optimizer.state_dict()
        data['base_parameters_unchanged'] = all(torch.equal(value, model.state_dict()[key])
                                                 for key, value in saved['base_state_dict'].items())
        save_expanded_experiment(saved)
        print('DAGGER_EPOCH', round_index, json.dumps(record), 'frozen_body', data['base_parameters_unchanged'], flush=True)


def expanded_dagger_experiment(workers):
    saved = expanded_teachers(workers)
    train_expanded_base(saved)
    for round_index in (1, 2):
        collect_dagger_round(saved, round_index, workers)
        train_dagger_round(saved, round_index)
    candidates = [(0, saved['base_selected_epoch'], saved['base_state_dict'], saved['base_selected_summary'])]
    candidates.extend((index, saved['rounds'][index]['selected_epoch'], saved['rounds'][index]['state_dict'],
                       saved['rounds'][index]['selected_summary']) for index in (1, 2))
    round_index, epoch, weights, summary = max(candidates, key=lambda item: expanded_model_key(item[3]))
    saved['state_dict'] = weights
    saved['selected_round'], saved['selected_epoch'], saved['selected_summary'] = round_index, epoch, summary
    references = [record['row'] for reference in saved['references'] for record in reference['records']]
    counts = {method: sum(row['status'] == 'FEASIBLE' for row in references if row['method'] == method)
              for method in ('RULE/BEAM', 'WAG+VNS')}
    saved['acceptance'] = dict(feasibility=summary['feasible'] >= max(counts.values()),
                               construction_time=summary['median_construct_s'] <= 20.)
    for method, pair in summary['paired'].items():
        saved['acceptance'][method + '_cmax'] = pair['common'] > 0 and pair['median_change_pct'] < 0.
        saved['acceptance'][method + '_wins'] = pair['wins'] >= pair['losses']
    save_expanded_experiment(saved)
    print('EXPANDED_FINAL', json.dumps(summary), 'acceptance', json.dumps(saved['acceptance']), flush=True)
    if all(saved['acceptance'].values()) and not saved.get('test'):
        model = joint_model(saved['state_dict'], cmax_value='cmax_value_mean' in saved['state_dict']).eval()
        model.requires_grad_(False)
        saved['test'] = expanded_validate(model, saved, 'final_test', round_index, epoch, split='test')
        save_expanded_experiment(saved)
        old_wag = [row for row in read_rows(FINAL_RESULTS / 'wag_vns_results_optimized.csv')
                   if row['source'] == BASELINE_SOURCES['WAG+VNS'] and int(row['seed']) == SEED and row['layout_id'].startswith('test_')]
        old_rule = [row for row in read_rows(FINAL_RESULTS / 'rule_beam_results.csv') if int(row['seed']) == SEED]
        print('EXPANDED_TEST', json.dumps(expanded_summary(saved['test'], old_wag + old_rule)), flush=True)


def owner_cf_layouts(source, records):
    teachers = {}
    for sample in source['teachers']:
        feasible = [record for record in sample['records'] if record['row']['status'] == 'FEASIBLE'
                    and record['row']['method'] in ('RULE/BEAM', 'WAG+VNS')
                    and (record['row']['method'] != 'WAG+VNS' or record['row']['iterations'] == 500)]
        if feasible:
            teachers[sample['layout_id']] = min(feasible, key=lambda record: record['row']['cmax_s'])
    ordered = sorted((row for row in records if row['layout_id'] in teachers),
                     key=lambda row: (len(row['parents']), row['layout_id']))
    if len(ordered) < 40:
        raise ValueError(f'only {len(ordered)} TRAIN layouts have saved B32-feasible teachers; need 40')
    chosen, strata = [], []
    for weld_bin in range(4):
        group = sorted(ordered[len(ordered) * weld_bin // 4:len(ordered) * (weld_bin + 1) // 4],
                       key=lambda row: (row['rectangle_utilization'], len(row['parts']), row['layout_id']))
        for size_bin in range(2):
            members = group[len(group) * size_bin // 2:len(group) * (size_bin + 1) // 2]
            def load_spread(row):
                teacher = teachers[row['layout_id']]['row']
                loads = [teacher[f'processing_r{r}_s'] for r in range(4)]
                return max(loads) - min(loads), len(row['parts']), row['layout_id']
            members.sort(key=load_spread)
            selection = [members[round(i * (len(members) - 1) / 4)] for i in range(5)]
            names = [row['layout_id'] for row in selection]
            chosen.extend(names)
            strata.append(dict(weld_bin=weld_bin, size_bin=size_bin, layouts=names,
                               parent_range=[min(len(row['parents']) for row in selection),
                                             max(len(row['parents']) for row in selection)],
                               parts_range=[min(len(row['parts']) for row in selection),
                                            max(len(row['parts']) for row in selection)],
                               utilization_range=[min(row['rectangle_utilization'] for row in selection),
                                                  max(row['rectangle_utilization'] for row in selection)]))
    return chosen, strata, teachers


def owner_cf_candidates(space, embeddings, model, state, selected, assignment):
    """Same WHOLE/prefix, with the original owner and the best legal other owner."""
    if assignment.pattern.kind.value != 'WHOLE':
        return ()
    blocks = {b.block_id: b for pattern in selected.values() for b in space.children[pattern]}
    nodes = {b.block_id: space.block_nodes[pattern, b.block_id]
             for pattern in selected.values() for b in space.children[pattern]}
    indices = tuple(i for i, action in enumerate(space.actions)
                    if action.parent_id == assignment.parent_id and action.pattern == assignment.pattern)
    plans, _ = gnn.vnext_joint_actions(space, embeddings, model, state, blocks, nodes,
                                      set(space.parent_ids) - selected.keys(), CONFIG,
                                      assignment_indices=indices, preserve_prefix=True)
    original = next((plan for plan in plans if space.actions[plan.assignment_index] == assignment), None)
    alternatives = [plan for plan in plans if space.actions[plan.assignment_index].robots != assignment.robots]
    if original is None or not alternatives:
        return ()
    alternative = min(alternatives, key=lambda plan: (max(plan.state.loads), plan.state.travel))
    return original, alternative


@torch.no_grad()
def collect_owner_cf(job):
    layout, teacher, records, cached = job
    name, parents = layout
    started = time.perf_counter()
    model = worker_model
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    result = unpack_directed_plan(parents, teacher['plan'], space)
    labels = sorted(gnn.parent_assignment_targets(result.solution, CONFIG),
                    key=lambda action: (-sum(CONFIG.process_time(b.length) for b in space.children[action.pattern]),
                                        action.parent_id))
    full_blocks = {b.block_id: b for action in labels for b in space.children[action.pattern]}
    routes = tuple(route.block_ids for route in result.solution.routes)
    seen = pickle.loads(cached) if cached else {}
    # These are previously measured B32 outcomes, not the WAG native schedule.
    for record in records:
        row = record['row']
        if record['plan'] is not None and row['status'] in ('FEASIBLE', 'DEADLOCK', 'INFEASIBLE'):
            old = unpack_directed_plan(parents, record['plan'], space)
            seen.setdefault(plan_key(old.solution, old.directions),
                            ScheduleResult(ScheduleStatus(row['status']), cmax=row['cmax_s'],
                                           diagnostics=(row['diagnostic'],) if row['diagnostic'] else ()))
    targets = []
    for depth, assignment in enumerate(labels):
        if assignment.pattern.kind.value != 'WHOLE':
            continue
        owners = {action.robots for action in space.actions
                  if action.parent_id == assignment.parent_id and action.pattern == assignment.pattern}
        if len(owners) < 2:
            continue
        selected = {action.parent_id: action.pattern for action in labels[:depth]}
        kept = {b.block_id for pattern in selected.values() for b in space.children[pattern]}
        state = project_routes(routes, result.directions, kept, full_blocks)
        owner = assignment.robots[0]
        priority = (state.loads[owner] < max(state.loads) - 1e-9,
                    -space.children[assignment.pattern][0].length, -state.loads[owner], assignment.parent_id)
        targets.append((priority, depth, assignment, state, selected))
    groups, rows, skipped = [], [], {}
    for _, depth, assignment, state, selected in sorted(targets, key=lambda item: item[0]):
        plans = owner_cf_candidates(space, embeddings, model, state, selected, assignment)
        if not plans:
            reason = 'NO_TWO_PREFIX_PRESERVING_OWNERS'
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        if (max(plans[1].state.loads), plans[1].state.travel) >= (max(plans[0].state.loads), plans[0].state.travel):
            reason = 'NO_LOAD_OR_MOVE_ADVANTAGE'
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        indices = torch.tensor([plan.assignment_index for plan in plans])
        assignment_rows = model.embed_assignment_rows(space, embeddings, indices,
                                                      torch.stack([plan.assignment_features for plan in plans])).cpu()
        completions, group_rows, descriptions = [], [], []
        for index, plan in enumerate(plans):
            action = space.actions[plan.assignment_index]
            branch_started = time.perf_counter()
            completion = build_joint(parents, model, space=space, embeddings=embeddings, beam_width=1,
                                     state=plan.state, selected={**selected, action.parent_id: action.pattern})
            construct_s = time.perf_counter() - branch_started
            completions.append(completion)
            group_rows.append(dict(split='train_expanded', stage='owner_branch', layout_id=name,
                method='FROZEN_BASE_OWNER_CF', seed=SEED, parent_id=action.parent_id, mode='WHOLE',
                assigned_parents=depth, candidate='AB'[index], robot=action.robots[0],
                construct_s=construct_s, runtime_s=construct_s, b32_calls=0,
                **beam_validation_metrics(completion)))
            descriptions.append(dict(assignment_index=plan.assignment_index, robot=action.robots[0],
                                     actions=[dict(block_id=a.block_id, robot=a.robot, position=a.position,
                                                   direction=a.direction, neighbour_changes=a.neighbour_changes)
                                              for a in plan.actions],
                                     routes=plan.state.routes, directions=plan.state.directions))
        schedules = []
        complete_pair = all(completion.solution is not None for completion in completions)
        for completion, row in zip(completions, group_rows):
            evaluated = time.perf_counter()
            schedule, called = evaluate_once(completion, seen) if complete_pair else (None, False)
            schedules.append(schedule)
            row.update(status=schedule.status.value if schedule else
                       ('NO_COMPLETE_SOLUTION' if completion.solution is None else 'SKIPPED_INCOMPLETE_PAIR'),
                       cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                       b32_calls=int(called), runtime_s=row['runtime_s'] + time.perf_counter() - evaluated,
                       diagnostic=(' | '.join(schedule.diagnostics) if not schedule.feasible else '') if schedule else
                       (completion.rejection_reason or 'counterpart could not complete construction'))
        winner, kind = better_branch(*schedules)
        for row in group_rows:
            row.update(winner='AB'[winner] if winner is not None else None, preference=kind)
        groups.append(dict(parent_id=assignment.parent_id, teacher=teacher['row']['method'],
                           prefix=dict(patterns=tuple(pattern.pattern_id for pattern in selected.values()),
                                       routes=state.routes, directions=state.directions),
                           assignment_rows=assignment_rows, outcomes=torch.stack([plan.outcome for plan in plans]).cpu(),
                           candidates=descriptions, plans=[pack_directed_plan(completion) for completion in completions],
                           preferences=[(winner, 1 - winner, kind)] if winner is not None else []))
        rows.extend(group_rows)
        print('OWNER_CF_PAIR', json.dumps(dict(layout_id=name, parent_id=assignment.parent_id, assigned=depth,
              robots=[row['robot'] for row in group_rows], status=[row['status'] for row in group_rows],
              cmax_s=[row['cmax_s'] for row in group_rows], winner=winner, preference=kind,
              b32_calls=sum(row['b32_calls'] for row in group_rows))), flush=True)
        if len(groups) == 3:
            break
    return dict(layout_id=name, teacher=teacher['row']['method'], groups=groups, rows=rows,
                skipped=skipped, runtime_s=time.perf_counter() - started)


def save_owner_cf(saved):
    rows = [row for sample in saved['training'] for row in sample['rows']] + saved['validation']
    for row in rows:
        if all(row.get(f'processing_r{r}_s') not in (None, '') for r in range(4)):
            row['max_processing_s'] = max(row[f'processing_r{r}_s'] for r in range(4))
    torch.save(saved, OWNER_CF_MODEL)
    with OWNER_CF_RESULTS.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=OWNER_CF_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in OWNER_CF_FIELDS} for row in rows)


def configure_owner_cf_worker(serialized, bounded):
    configure_expanded_worker(serialized)
    worker_model.bounded_assignment = bounded


def train_owner_cf(saved):
    if saved['completed_epoch'] == EPOCHS:
        return
    torch.manual_seed(SEED)
    model = joint_model(saved['state_dict'], bounded_assignment=True)
    model.requires_grad_(False)
    model.policy.requires_grad_(True)
    groups = [group for sample in sorted(saved['training'], key=lambda sample: sample['layout_id'])
              for group in sample['groups'] if group['preferences']]
    if not groups:
        raise ValueError('WHOLE owner counterfactuals produced no true B32 preference; do not train')
    optimizer = torch.optim.AdamW(model.policy.parameters(), lr=3e-4, weight_decay=1e-4)
    if saved['completed_epoch']:
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    for epoch in range(saved['completed_epoch'] + 1, EPOCHS + 1):
        model.train()
        order = list(range(len(groups)))
        Random(SEED + epoch).shuffle(order)
        losses = []
        for index in order:
            group = groups[index]
            scores = model.assignment_contribution(group['assignment_rows']) + group['outcomes'] @ model.joint_weights
            good, bad, _ = group['preferences'][0]
            loss = F.softplus(scores[bad] - scores[good])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        record = dict(epoch=epoch, pairs=len(groups), ranking_loss=sum(losses) / len(losses))
        saved['history'].append(record)
        saved['completed_epoch'] = epoch
        saved['state_dict'] = {key: value.detach().clone() for key, value in model.state_dict().items()}
        saved['optimizer_state_dict'] = optimizer.state_dict()
        saved['frozen_parameters_unchanged'] = all(torch.equal(value, saved['state_dict'][key])
            for key, value in saved['initial_state_dict'].items() if not key.startswith('policy.'))
        save_owner_cf(saved)
        print('OWNER_CF_EPOCH', json.dumps(record), 'frozen', saved['frozen_parameters_unchanged'], flush=True)


def owner_cf_validation_job(job):
    layout, cached = job
    selected = []
    row, _ = complete_gnn(layout[1], worker_model, SEED, (), reconstruct=False, beam_width=1,
                          evaluation_cache=pickle.loads(cached) if cached else {},
                          result_observer=lambda result, schedule: selected.append(result))
    row = dict(split='validation_fresh', stage='owner_validation', layout_id=layout[0], method='GNN_OWNER_CF', seed=SEED,
               **{key: row[key] for key in ('status', 'cmax_s', 'construct_s', 'runtime_s', 'b32_calls', 'diagnostic')},
               **beam_validation_metrics(selected[0]), plan=pack_directed_plan(selected[0]))
    print('OWNER_CF_VALIDATION', json.dumps({key: value for key, value in row.items() if key != 'plan'}), flush=True)
    return row


def owner_cf_experiment(workers):
    source = torch.load(EXPANDED_MODEL, map_location='cpu', weights_only=True)
    if (source['selected_round'], source['selected_epoch']) != (0, 2) or 'cmax_value_mean' in source['state_dict']:
        raise ValueError('owner-cf requires the saved expanded base GNN epoch 2, without a value head')
    records = [json.loads(line) for line in (DATA / 'train_expanded.jsonl').read_text(encoding='utf-8').splitlines()]
    names, strata, teachers = owner_cf_layouts(source, records)
    if OWNER_CF_MODEL.exists():
        saved = torch.load(OWNER_CF_MODEL, map_location='cpu', weights_only=True)
    else:
        weights = {key: value.clone() for key, value in source['state_dict'].items()}
        saved = dict(initial_state_dict=weights, state_dict=weights, bounded_assignment=True,
                     seed=SEED, hidden_dim=64, heads=4, training=[], history=[], validation=[], completed_epoch=0,
                     layouts=names, strata=strata)
        save_owner_cf(saved)
    layouts = dict(load_layouts('train_expanded'))
    samples = {sample['layout_id']: sample for sample in source['teachers']}
    old_evaluations = {sample['layout_id']: sample['evaluations'] for index in (1, 2)
                       for sample in source['rounds'][index]['samples']}
    done = {sample['layout_id'] for sample in saved['training']}
    jobs = [((name, layouts[name]), teachers[name], samples[name]['records'], old_evaluations.get(name))
            for name in saved['layouts'] if name not in done]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_owner_cf_worker,
                             initargs=(pickle.dumps(saved['initial_state_dict']), False)) as executor:
        futures = [executor.submit(collect_owner_cf, job) for job in jobs]
        for future in as_completed(futures):
            sample = future.result()
            saved['training'].append(sample)
            save_owner_cf(saved)
            print('OWNER_CF_SAVED', len(saved['training']), '/ 40', sample['layout_id'],
                  'pairs', sum(bool(group['preferences']) for group in sample['groups']),
                  'seconds', sample['runtime_s'], flush=True)
    train_owner_cf(saved)
    done = {row['layout_id'] for row in saved['validation']}
    cached = source.get('validation_seen', {record['layout_id']: record['seen'] for record in source['references']})
    with ProcessPoolExecutor(max_workers=1, initializer=configure_owner_cf_worker,
                             initargs=(pickle.dumps(saved['state_dict']), True)) as executor:
        for layout in load_layouts('validation_fresh'):
            if layout[0] in done:
                continue
            row = executor.submit(owner_cf_validation_job, (layout, cached.get(layout[0]))).result()
            saved['validation'].append(row)
            save_owner_cf(saved)
    references = [record['row'] for reference in source['references'] for record in reference['records']]
    summary = expanded_summary(saved['validation'], references)
    before = source['base_validation'][2]
    pairs = [(row, next(old for old in before if old['layout_id'] == row['layout_id']))
             for row in saved['validation'] if row['status'] == 'FEASIBLE']
    pairs = [(a, b) for a, b in pairs if b['status'] == 'FEASIBLE']
    changes = [a['cmax_s'] - b['cmax_s'] for a, b in pairs]
    saved['versus_base'] = dict(common=len(pairs), wins=sum(delta < -1e-6 for delta in changes),
                               ties=sum(abs(delta) <= 1e-6 for delta in changes),
                               losses=sum(delta > 1e-6 for delta in changes),
                               median_change_pct=median(100. * (a['cmax_s'] / b['cmax_s'] - 1.) for a, b in pairs) if pairs else None)
    saved['summary'] = summary
    saved['acceptance'] = dict(feasibility=summary['feasible'] >= 19,
                               construction_time=summary['median_construct_s'] <= 20.)
    for method, pair in summary['paired'].items():
        saved['acceptance'][method + '_cmax'] = pair['common'] > 0 and pair['median_change_pct'] < 0.
        saved['acceptance'][method + '_wins'] = pair['wins'] >= pair['losses']
    save_owner_cf(saved)
    print('OWNER_CF_FINAL', json.dumps(summary), 'versus_base', json.dumps(saved['versus_base']),
          'acceptance', json.dumps(saved['acceptance']), flush=True)


FEEDBACK_MODEL = ROOT / 'models' / 'gnn_schedule_feedback.pt'
FEEDBACK_TRAIN_CSV = FINAL_RESULTS / 'gnn_schedule_feedback_training_validation.csv'
FEEDBACK_TEST_CSV = FINAL_RESULTS / 'gnn_schedule_feedback_test.csv'
FEEDBACK_SOURCE = 'SCHEDULE_FEEDBACK_REINFORCE'
FEEDBACK_METHOD = 'GNN_SCHEDULE_FEEDBACK'
FEEDBACK_RECONSTRUCTIONS = 500
FEEDBACK_FULL_SOURCE = 'CMAX_FEEDBACK_ADAPTIVE_500'
FEEDBACK_FIELDS = ('stage', 'epoch', 'layout_id', 'seed', 'method', 'status', 'cmax_s',
    'initial_status', 'initial_cmax_s', 'runtime_s', 'construct_s', 'reconstruction_attempts',
    'reconstructions', 'accepted', 'b32_calls', 'cache_hits', 'deadlocks', 'numeric_failures',
    'incomplete', 'late_candidates', 'baseline_status', 'baseline_cmax_s', 'reward', 'advantage',
    'loss', 'steps', 'gradient_norm', 'max_processing_s', 'move_s', 'wait_s', 'diagnostic', 'source')


def feedback_model(weights=None):
    """Transfer the clean current-TRAIN model, excluding the HGA-taught final model."""
    model = gnn.VNextGNN(workload_lookahead=True, joint_actions=True, schedule_feedback=True)
    if weights is not None:
        model.load_state_dict(weights)
        return model
    saved = torch.load(MODEL, map_location='cpu', weights_only=True)
    training = saved['training']
    if (saved['hidden_dim'], saved['heads'], training['layouts'], training['validation_layouts']) != (64, 4, 40, 10):
        raise ValueError('initial weights do not describe the current weld-layout experiment')
    source = saved['models']['GNN_VNEXT_WORKLOAD']['state_dict']
    target = model.state_dict()
    for key, value in source.items():
        if target[key].shape == value.shape:
            target[key].copy_(value)
        elif key == 'policy.0.weight':
            target[key].zero_()
            target[key][:, :value.shape[1]].copy_(value)
        elif key in ('assignment_mean', 'assignment_scale'):
            target[key][:len(value)].copy_(value)
        else:
            raise ValueError(f'incompatible initial tensor: {key}')
    model.load_state_dict(target)
    return model


def feedback_write(path, rows):
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FEEDBACK_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        for row in rows:
            record = dict(method=FEEDBACK_METHOD, source=FEEDBACK_SOURCE)
            record.update(row)
            writer.writerow(record)
        stream.flush()


def feedback_reward(schedule, scale, *, incumbent=None):
    if schedule is None or schedule.status is ScheduleStatus.NUMERIC_FAILURE:
        return None
    if schedule.status is ScheduleStatus.DEADLOCK:
        return -5.
    if not schedule.feasible:
        return None
    return -schedule.cmax / scale if incumbent is None else (incumbent.cmax - schedule.cmax) / scale


def feedback_improves(candidate, incumbent):
    return candidate is not None and candidate.feasible and (
        incumbent is None or not incumbent.feasible or candidate.cmax < incumbent.cmax)


def feedback_accepts(candidate, current, temperature, rng):
    """Simulated annealing affects the search state; the best stays separate."""
    if candidate is None or not candidate.feasible:
        return False
    delta = candidate.cmax - current.cmax
    return delta <= 0. or rng.random() < math.exp(-delta / temperature)


def feedback_prefix(result, space, feedback, q, seed):
    partial = lns.select_feedback_bottleneck(result.solution, feedback,
        q=min(q, len(result.solution.parents)), seed=seed, assignment_space=space)
    removed = set(partial.removed_parent_ids)
    selected = {p.parent_id: p for p in result.solution.patterns if p.parent_id not in removed}
    blocks = {b.block_id: b for pattern in selected.values() for b in space.children[pattern]}
    return project_routes(result.oriented_state.routes, result.directions, set(blocks), blocks), selected


@torch.no_grad()
def run_feedback_gnn(parents, model, seed, *, budget=None, reconstruction_limit=FEEDBACK_RECONSTRUCTIONS,
                     progress_observer=None):
    """Fixed-policy GNN construction, critical-parent reconstruction and SA.

    Formal runs use 500 attempts without a wall-clock or stagnation cutoff.
    The optional time budget remains available for historical diagnostics only.
    """
    started = time.perf_counter()
    if budget is None and reconstruction_limit is None:
        raise ValueError('construction needs a time budget or a finite reconstruction count')
    deadline = math.inf if budget is None else started + budget
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    seen, blockers = {}, {}
    counts = dict(reconstruction_attempts=0, reconstructions=0, accepted=0, b32_calls=0,
                  cache_hits=0, deadlocks=0, numeric_failures=0, incomplete=0, late_candidates=0)
    current_result = current_schedule = best_result = best_schedule = None
    initial_status, initial_cmax, diagnostic = 'NO_COMPLETE_SOLUTION', None, ''
    last_status = initial_status
    reserve = 1.
    rng = Random(seed)
    initial_attempts = best_improvements = accepted_uphill = accepted_equal = 0
    feasible_repairs = incomplete_repairs = stagnant = 0
    removal_counts = {}

    def consider(result):
        nonlocal diagnostic, last_status, reserve
        if result.solution is None:
            counts['incomplete'] += 1
            diagnostic = result.rejection_reason or 'NO_COMPLETE_SOLUTION'
            last_status = 'TIME_LIMIT' if diagnostic == 'TIME_LIMIT' else 'CONSTRUCTION_FAILED'
            return None
        if time.perf_counter() + reserve >= deadline:
            counts['late_candidates'] += 1
            last_status = 'TIME_LIMIT'
            return None
        evaluation_started = time.perf_counter()
        schedule, called = evaluate_once(result, seen, blocker_cache=blockers)
        reserve = max(reserve, 1.5 * (time.perf_counter() - evaluation_started))
        counts['b32_calls'] += int(called)
        counts['cache_hits'] += int(not called)
        last_status = schedule.status.value
        counts['deadlocks'] += int(schedule.status is ScheduleStatus.DEADLOCK)
        counts['numeric_failures'] += int(schedule.status is ScheduleStatus.NUMERIC_FAILURE)
        if time.perf_counter() > deadline:
            counts['late_candidates'] += 1
            last_status = 'TIME_LIMIT'
            return None
        if not schedule.feasible:
            diagnostic = ' | '.join(schedule.diagnostics)
        return schedule

    # Only GNN sampling may recover a deadlocked greedy initial construction.
    for attempt in range(3):
        if time.perf_counter() + reserve >= deadline:
            break
        initial_attempts += 1
        result = build_joint(parents, model, space=space, embeddings=embeddings,
            rng=None if attempt == 0 else rng, deadline=None if budget is None else deadline - reserve)
        schedule = consider(result)
        if attempt == 0:
            initial_status = schedule.status.value if schedule is not None else last_status
            initial_cmax = schedule.cmax if schedule is not None and schedule.feasible else None
            initial_seconds = time.perf_counter() - started
        if schedule is not None and schedule.feasible:
            current_result = best_result = result
            current_schedule = best_schedule = schedule
            break
        if schedule is None or schedule.status is not ScheduleStatus.DEADLOCK:
            break
    initial_seconds = locals().get('initial_seconds', time.perf_counter() - started)
    initial_incumbent_cmax = best_schedule.cmax if best_schedule is not None else None
    parent_count = len(parents)
    q1 = min(parent_count - 1, max(8, math.ceil(.15 * parent_count)))
    q2 = min(parent_count - 1, max(12, math.ceil(.25 * parent_count)))
    # No per-action B32, direction pass, literature initializer or fallback.
    previous_repair_seconds = 0.
    while (current_schedule is not None
           and (reconstruction_limit is None or counts['reconstruction_attempts'] < reconstruction_limit)
           and time.perf_counter() + reserve + previous_repair_seconds < deadline):
        repair_started = time.perf_counter()
        feedback = lns.schedule_feedback_summary(current_result.solution, current_schedule,
            blockers.get(plan_key(current_result.solution, current_result.directions), ()), CONFIG)
        q = q2 if stagnant >= 25 else q1
        removal_counts[q] = removal_counts.get(q, 0) + 1
        k = counts['reconstruction_attempts']
        counts['reconstruction_attempts'] += 1
        try:
            state, selected = feedback_prefix(current_result, space, feedback, q, rng.randrange(2**32))
            candidate = build_joint(parents, model, space=space, embeddings=embeddings,
                state=state, selected=selected, feedback=feedback,
                deadline=None if budget is None else deadline - reserve)
        except (ValueError, ArithmeticError) as error:
            candidate = gnn.JointCompletionResult(None, None, (), time.perf_counter() - repair_started,
                                                   rejection_reason=str(error))
        counts['reconstructions'] += int(candidate.solution is not None)
        incomplete_repairs += int(candidate.solution is None)
        schedule = consider(candidate)
        feasible_repairs += int(schedule is not None and schedule.feasible)
        if feedback_improves(schedule, best_schedule):
            best_result, best_schedule = candidate, schedule
            best_improvements += 1
            stagnant = 0
        else:
            stagnant += 1
        temperature = .01 * initial_incumbent_cmax * .01 ** (k / 499.)
        if feedback_accepts(schedule, current_schedule, temperature, rng):
            accepted_uphill += int(schedule.cmax > current_schedule.cmax)
            accepted_equal += int(schedule.cmax == current_schedule.cmax)
            current_result, current_schedule = candidate, schedule
            counts['accepted'] += 1
        previous_repair_seconds = time.perf_counter() - repair_started
        if progress_observer is not None and counts['reconstruction_attempts'] % 100 == 0:
            progress_observer(dict(reconstruction_attempts=counts['reconstruction_attempts'],
                reconstructions=counts['reconstructions'], b32_calls=counts['b32_calls'],
                best_cmax_s=best_schedule.cmax, current_cmax_s=current_schedule.cmax,
                q=q, runtime_s=time.perf_counter() - started))
        if budget is not None and candidate.rejection_reason == 'TIME_LIMIT':
            break
    return dict(status='FEASIBLE' if best_schedule is not None else last_status,
        cmax_s=best_schedule.cmax if best_schedule is not None else None,
        initial_status=initial_status, initial_cmax_s=initial_cmax,
        runtime_s=time.perf_counter() - started, construct_s=initial_seconds, **counts,
        diagnostic=json.dumps(dict(initial_incumbent_cmax_s=initial_incumbent_cmax,
            initial_attempts=initial_attempts, best_improvements=best_improvements,
            accepted_uphill=accepted_uphill, accepted_equal=accepted_equal,
            feasible_reconstructions=feasible_repairs, incomplete_reconstructions=incomplete_repairs,
            removal_counts=removal_counts, q1=q1, q2=q2, stagnant=stagnant,
            current_cmax_s=current_schedule.cmax if current_schedule is not None else None,
            best_robot_completion_s=best_schedule.robot_completion if best_schedule is not None else None,
            failure=diagnostic if best_schedule is None else '')), **final_metrics(best_schedule))


def feedback_update(model, space, trace, advantage, optimizer):
    if advantage is None or not trace:
        return None, None
    optimizer.zero_grad(set_to_none=True)
    embeddings = model.encode_static(space.graph)
    log_prob = torch.stack([model.feedback_scores(space, embeddings, inputs).log_softmax(0)[choice]
                            for inputs, choice in trace]).sum()
    loss = -float(advantage) * log_prob
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach()), float(norm)


def feedback_train_layout(model, baseline, layout, epoch, optimizer, *, training_stage='train'):
    name, parents = layout
    space = gnn.parent_assignment_space(parents, CONFIG)
    scale = max(1., sum(CONFIG.process_time(p.length) for p in parents) / 4.)
    seen, blockers, rows = {}, {}, []
    rng = Random(SEED + epoch * 1000 + int(name.rsplit('_', 1)[1]))

    def construct(policy, state=None, selected=None, feedback=None, sample=False):
        trace = []
        with torch.no_grad():
            result = gnn.construct_vnext_gnn(parents, policy, CONFIG, space=space, state=state,
                selected=selected, feedback=feedback, rng=rng if sample else None,
                policy_observer=(lambda inputs, choice: trace.append((inputs, choice))) if sample else None)
        schedule, called = evaluate_once(result, seen, blocker_cache=blockers)
        return result, schedule, trace, int(called)

    def update(phase, sampled, greedy, started, incumbent=None):
        result, schedule, trace, calls = sampled
        _, base_schedule, _, base_calls = greedy
        reward = feedback_reward(schedule, scale, incumbent=incumbent)
        base_reward = feedback_reward(base_schedule, scale, incumbent=incumbent)
        advantage = None if reward is None or base_reward is None else reward - base_reward
        loss, norm = feedback_update(model, space, trace, advantage, optimizer)
        rows.append(dict(stage=training_stage + '_' + phase, epoch=epoch, layout_id=name, seed=SEED,
            status=schedule.status.value if schedule else 'CONSTRUCTION_FAILED',
            cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
            baseline_status=base_schedule.status.value if base_schedule else 'CONSTRUCTION_FAILED',
            baseline_cmax_s=base_schedule.cmax if base_schedule is not None and base_schedule.feasible else None,
            initial_cmax_s=incumbent.cmax if incumbent else None, runtime_s=time.perf_counter() - started,
            b32_calls=calls + base_calls, cache_hits=int(not calls and result.solution is not None),
            deadlocks=sum(s is not None and s.status is ScheduleStatus.DEADLOCK for s in (schedule, base_schedule)),
            numeric_failures=sum(s is not None and s.status is ScheduleStatus.NUMERIC_FAILURE for s in (schedule, base_schedule)),
            reward=reward, advantage=advantage, loss=loss, steps=len(trace), gradient_norm=norm,
            diagnostic=result.rejection_reason or (' | '.join(schedule.diagnostics) if schedule is not None and not schedule.feasible else '')))

    started = time.perf_counter()
    greedy = construct(baseline)
    sampled = construct(model, sample=True)
    update('construct', sampled, greedy, started)
    incumbent_result, incumbent_schedule = (sampled[:2] if feedback_improves(sampled[1], greedy[1]) else greedy[:2])
    if incumbent_schedule is not None and incumbent_schedule.feasible:
        for q in (4, 8):
            started = time.perf_counter()
            feedback = lns.schedule_feedback_summary(incumbent_result.solution, incumbent_schedule,
                blockers.get(plan_key(incumbent_result.solution, incumbent_result.directions), ()), CONFIG)
            state, selected = feedback_prefix(incumbent_result, space, feedback, q, rng.randrange(2**32))
            greedy = construct(baseline, state, selected, feedback)
            sampled = construct(model, state, selected, feedback, sample=True)
            update('repair' + str(q), sampled, greedy, started, incumbent_schedule)
            for result, schedule, _, _ in (greedy, sampled):
                if feedback_improves(schedule, incumbent_schedule):
                    incumbent_result, incumbent_schedule = result, schedule
    return rows


def configure_feedback_worker(serialized):
    configure_cpu()
    global worker_feedback_model
    worker_feedback_model = feedback_model(pickle.loads(serialized)).eval()


def feedback_job(job):
    name, parents, seed, stage, epoch, *limits = job
    print(json.dumps(dict(event='feedback_start', stage=stage, epoch=epoch,
                          layout_id=name, seed=seed)), flush=True)
    options = dict(budget=None, reconstruction_limit=limits[0]) if limits else {}
    options['progress_observer'] = lambda progress: print(json.dumps(dict(
        event='feedback_progress', layout_id=name, seed=seed, **progress)), flush=True)
    row = dict(stage=stage, epoch=epoch, layout_id=name, seed=seed,
               **run_feedback_gnn(parents, worker_feedback_model, seed, **options))
    if limits:
        row['source'] = FEEDBACK_FULL_SOURCE
    return row


def feedback_evaluate(model, layouts, epoch, workers, *, split='validation', reconstruction_limit=None):
    path = FEEDBACK_TEST_CSV if split == 'test' else FEEDBACK_TRAIN_CSV
    seeds = TEST_SEEDS if split == 'test' or reconstruction_limit is not None else (SEED,)
    stage = 'full_validation' if split == 'validation' and reconstruction_limit is not None else split
    extra = () if reconstruction_limit is None else (reconstruction_limit,)
    jobs = [(name, parents, seed, stage, epoch, *extra) for name, parents in layouts for seed in seeds]
    expected = {(job[0], job[2]) for job in jobs}
    retained = [r for r in read_rows(path) if r['stage'] == stage and int(r['epoch']) == epoch
                and (reconstruction_limit is None or r['source'] == FEEDBACK_FULL_SOURCE)
                and (r['layout_id'], int(r['seed'])) in expected]
    completed = {(r['layout_id'], int(r['seed'])) for r in retained}
    jobs = [job for job in jobs if (job[0], job[2]) not in completed]
    print(json.dumps(dict(stage=stage, source=FEEDBACK_FULL_SOURCE if extra else FEEDBACK_SOURCE,
                          completed=len(retained), remaining=len(jobs))), flush=True)
    if not jobs:
        return retained
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_feedback_worker,
                             initargs=(pickle.dumps(model.state_dict()),)) as pool:
        futures = [pool.submit(feedback_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            feedback_write(path, [row])
            retained.append(row)
    return retained


def feedback_summary(rows):
    good = [r for r in rows if r['status'] == 'FEASIBLE']
    return dict(runs=len(rows), feasible=len(good),
        median_cmax_s=median(float(r['cmax_s']) for r in good) if good else None,
        median_runtime_s=median(float(r['runtime_s']) for r in rows) if rows else None,
        min_runtime_s=min((float(r['runtime_s']) for r in rows), default=None),
        max_runtime_s=max((float(r['runtime_s']) for r in rows), default=None),
        total_runtime_s=sum(float(r['runtime_s']) for r in rows),
        b32_calls=sum(int(r.get('b32_calls') or 0) for r in rows))


def feedback_paired(rows, reference):
    first = {(r['layout_id'], int(r['seed'])): r for r in rows}
    second = {(r['layout_id'], int(r['seed'])): r for r in reference}
    changes = []; by_layout = {}; wins = ties = losses = 0
    for key in sorted(first.keys() & second.keys()):
        a, b = first[key], second[key]
        if a['status'] == b['status'] == 'FEASIBLE':
            change = (float(a['cmax_s']) / float(b['cmax_s']) - 1.) * 100.
            changes.append(change)
            by_layout.setdefault(key[0], []).append(change)
            wins += int(change < -1e-7); ties += int(abs(change) <= 1e-7); losses += int(change > 1e-7)
    layout_changes = {name: median(by_layout[name]) if name in by_layout else None
                      for name in sorted({key[0] for key in first})}
    return dict(common=len(changes), wins=wins, ties=ties, losses=losses,
        median_change_pct=median(changes) if changes else None, layout_change_pct=layout_changes,
        improved_layouts=sum(change is not None and change < -1e-7 for change in layout_changes.values()),
        tied_layouts=sum(change is not None and abs(change) <= 1e-7 for change in layout_changes.values()),
        degraded_layouts=sum(change is not None and change > 1e-7 for change in layout_changes.values()))


def train_feedback(workers, rounds, *, expanded=False):
    configure_cpu(); torch.manual_seed(SEED)
    base_layouts = load_layouts('train')
    layouts = load_layouts('train_expanded') if expanded else base_layouts
    validation = load_layouts('validation')
    expected = (120, 11267, 10, 936) if expanded else (40, 3749, 10, 936)
    if (len(layouts), sum(len(p) for _, p in layouts), len(validation), sum(len(p) for _, p in validation)) != expected:
        raise ValueError('the requested packed weld-layout split was not loaded')
    if expanded and layouts[:len(base_layouts)] != base_layouts:
        raise ValueError('expanded packed layouts must preserve the original 40 examples')
    model = feedback_model().eval()
    selected = copy.deepcopy(model.state_dict())
    saved = dict(state_dict=selected, initial_state_dict=selected, selected_epoch=0, seed=SEED,
        hidden_dim=64, heads=4, workload_lookahead=True, joint_actions=True, schedule_feedback=True,
        initialization='models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD',
        excluded_initialization='gnn_weld_layout_final.pt used HGA teachers',
        training_layouts=[name for name, _ in layouts], training_parents=expected[1],
        validation_layouts=[name for name, _ in validation], budget_s=EQUAL_TIME_LIMIT,
        learning_rate=1e-4, deadlock_reward=-5., max_initial_attempts=3, removal_sizes=(4, 8),
        history=[], test_eligible=False, training_complete=False)
    start_epoch = 1
    if expanded:
        saved = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
        if not saved['training_complete'] or FEEDBACK_TEST_CSV.exists():
            raise ValueError('continue before formal TEST and after the current training round has finished')
        start_epoch = saved['completed_epoch'] + 1
        if start_epoch > rounds:
            print('The requested total number of feedback rounds is already complete.', flush=True)
            return saved
        selected = saved['state_dict']
        model = feedback_model(selected).eval()
        saved.setdefault('forty_layout_state_dict', saved['last_state_dict'])
        saved.setdefault('original_training_layouts', saved['training_layouts'])
        saved.update(training_layouts=[name for name, _ in layouts], training_parents=expected[1],
                     training_dataset='data/weld_layout/train_expanded.jsonl',
                     continued_from_selected_epoch=saved['selected_epoch'], training_complete=False)
        selected_rows = [r for r in read_rows(FEEDBACK_TRAIN_CSV)
                         if r['stage'] == 'validation' and int(r['epoch']) == saved['selected_epoch']]
        if len(selected_rows) != len(validation):
            raise ValueError('the selected model must retain its complete original validation results')
    elif FEEDBACK_MODEL.exists():
        previous = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
        if previous.get('training_complete') or previous.get('history'):
            raise ValueError('the recorded feedback training already has results; preserve that experiment')
        # Resume an interrupted initial validation from its unchanged weights.
        saved = previous
        selected = saved['state_dict']
        model = feedback_model(selected).eval()
    torch.save(saved, FEEDBACK_MODEL)
    if not expanded:
        initial_rows = feedback_evaluate(model, validation, 0, workers)
        selected_rows = initial_rows
        saved['history'].append(dict(epoch=0, **feedback_summary(initial_rows)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    for epoch in range(start_epoch, rounds + 1):
        baseline = feedback_model(selected).eval().requires_grad_(False)
        model.train()
        ordered = list(layouts); Random(SEED + epoch).shuffle(ordered)
        for index, layout in enumerate(ordered):
            rows = feedback_train_layout(model, baseline, layout, epoch, optimizer,
                                         training_stage='train_expanded' if expanded else 'train')
            feedback_write(FEEDBACK_TRAIN_CSV, rows)
            print(json.dumps(dict(epoch=epoch, trained=index + 1, layout_id=layout[0], updates=len(rows),
                                  b32_calls=sum(r['b32_calls'] for r in rows)), ensure_ascii=False), flush=True)
        model.eval()
        rows = feedback_evaluate(model, validation, epoch, workers)
        summary = feedback_summary(rows); previous = feedback_summary(selected_rows)
        pair = feedback_paired(rows, selected_rows)
        improved = summary['feasible'] >= previous['feasible'] and pair['common'] > 0 and (
            pair['median_change_pct'] < 0. or (summary['feasible'] > previous['feasible'] and pair['median_change_pct'] <= 0.))
        saved['history'].append(dict(epoch=epoch, training_layouts=len(layouts),
                                     training_parents=expected[1], paired_to_selected=pair, improved=improved, **summary))
        saved['last_state_dict'] = copy.deepcopy(model.state_dict())
        saved['completed_epoch'] = epoch
        if improved:
            selected = copy.deepcopy(model.state_dict()); selected_rows = rows
            saved.update(state_dict=selected, selected_epoch=epoch, test_eligible=True)
        saved.update(training_complete=True)
        torch.save(saved, FEEDBACK_MODEL)
        print(json.dumps(dict(history=saved['history'], test_eligible=saved['test_eligible']), ensure_ascii=False), flush=True)
        if not improved:
            break
    return saved


def test_feedback(workers):
    validation = summarize_feedback(split='validation')
    if not validation['test_eligible']:
        print('The 30-run adaptive VALIDATION did not meet all four requirements; no TEST is run.', flush=True)
        return
    saved = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
    if not saved['training_complete']:
        raise ValueError('formal evaluation requires the completed selected checkpoint')
    model = feedback_model(saved['state_dict']).eval()
    feedback_evaluate(model, load_layouts('test'), saved['selected_epoch'], workers, split='test',
                      reconstruction_limit=FEEDBACK_RECONSTRUCTIONS)


def validate_feedback_complete(workers):
    """Run the selected fixed policy on the original ten layouts, three seeds."""
    saved = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
    if not saved['training_complete']:
        raise ValueError('formal evaluation requires the completed selected checkpoint')
    layouts = load_layouts('validation')
    feedback_evaluate(feedback_model(saved['state_dict']).eval(), layouts, saved['selected_epoch'], workers,
                      reconstruction_limit=FEEDBACK_RECONSTRUCTIONS)
    return summarize_feedback(split='validation')


def summarize_feedback(*, split='test'):
    """Only the new 500-attempt source and completed original WAG/HGA runs."""
    path = FEEDBACK_TEST_CSV if split == 'test' else FEEDBACK_TRAIN_CSV
    stage = 'test' if split == 'test' else 'full_validation'
    rows = [r for r in read_rows(path) if r['source'] == FEEDBACK_FULL_SOURCE and r['stage'] == stage]
    keys = {(r['layout_id'], int(r['seed'])) for r in rows}
    references = {}
    for method, (_, _, filename) in BASELINES.items():
        references[method] = [r for r in read_rows(FINAL_RESULTS / filename)
                              if (r['layout_id'], int(r['seed'])) in keys and completed_baseline_row(r)]
    summary = dict(split=split, source=FEEDBACK_FULL_SOURCE,
                   gnn=feedback_summary(rows), baselines={name: feedback_summary(group) for name, group in references.items()},
                   paired={name: feedback_paired(rows, group) for name, group in references.items()})
    gains = []
    for row in rows:
        if row['status'] != 'FEASIBLE':
            continue
        before = float(row['initial_cmax_s']) if row['initial_cmax_s'] else None
        if row.get('diagnostic', '').startswith('{'):
            before = json.loads(row['diagnostic']).get('initial_incumbent_cmax_s', before)
        if before is not None:
            gains.append((float(row['cmax_s']) / before - 1.) * 100.)
    summary['search'] = dict(initial_greedy_feasible=sum(r['initial_status'] == 'FEASIBLE' for r in rows),
        gain_pairs=len(gains), median_cmax_change_from_first_incumbent_pct=median(gains) if gains else None,
        b32_calls=sum(int(r['b32_calls']) for r in rows), cache_hits=sum(int(r['cache_hits']) for r in rows),
        reconstruction_attempts=sum(int(r['reconstruction_attempts']) for r in rows),
        runs_reaching_500=sum(int(r['reconstruction_attempts']) == FEEDBACK_RECONSTRUCTIONS for r in rows),
        reconstructions=sum(int(r['reconstructions']) for r in rows), accepted=sum(int(r['accepted']) for r in rows),
        incomplete=sum(int(r['incomplete']) for r in rows), deadlocks=sum(int(r['deadlocks']) for r in rows),
        numeric_failures=sum(int(r['numeric_failures']) for r in rows),
        max_runtime_s=max((float(r['runtime_s']) for r in rows), default=None))
    if split == 'validation':
        summary['test_eligible'] = (len(rows) == len(keys) == 30 and summary['gnn']['feasible'] == 30
            and all(len(references[method]) == 30 and pair['common'] > 0
                    and pair['median_change_pct'] < 0. and pair['wins'] > pair['losses']
                    and pair['improved_layouts'] > pair['degraded_layouts']
                    for method, pair in summary['paired'].items()))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


CHILD_MODEL = ROOT / 'models' / 'gnn_child_level_feedback.pt'
CHILD_TRAIN_SOURCE = 'CHILD_BLOCK_B32_PAIRWISE_TRAIN_V1'
CHILD_PILOT_SOURCE = 'CHILD_BLOCK_B32_PAIRWISE_100_V1'
CHILD_FULL_SOURCE = 'CHILD_BLOCK_B32_PAIRWISE_500_V1'
CHILD_METHOD = 'GNN_CHILD_LEVEL_FEEDBACK'


def child_feedback_model(weights=None):
    original = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
    model = feedback_model(original['state_dict']).requires_grad_(False)
    model.enable_child_transfer_head()
    if weights is not None:
        model.load_state_dict(weights)
    return model


def child_counted_evaluation(result, context):
    if result.solution is None:
        context['counts']['incomplete'] += 1
        return None
    schedule, called = evaluate_once(result, context['seen'], blocker_cache=context['blockers'])
    counts = context['counts']
    counts['b32_calls'] += int(called)
    counts['cache_hits'] += int(not called)
    counts['deadlocks'] += int(schedule.status is ScheduleStatus.DEADLOCK)
    counts['numeric_failures'] += int(schedule.status is ScheduleStatus.NUMERIC_FAILURE)
    return schedule


@torch.no_grad()
def child_initial_state(parents, model, seed):
    """The unchanged greedy/at-most-two-samples GNN initial construction."""
    started = time.perf_counter()
    space = gnn.parent_assignment_space(parents, CONFIG)
    context = dict(started=started, space=space, embeddings=model.encode_static(space.graph),
        seen={}, blockers={}, rng=Random(seed), result=None, schedule=None, initial_attempts=0,
        initial_status='NO_COMPLETE_SOLUTION', initial_cmax_s=None, failure='')
    context['counts'] = dict(reconstruction_attempts=0, reconstructions=0, accepted=0, b32_calls=0,
        cache_hits=0, deadlocks=0, numeric_failures=0, incomplete=0, late_candidates=0)
    for attempt in range(3):
        context['initial_attempts'] += 1
        result = build_joint(parents, model, space=space, embeddings=context['embeddings'],
            rng=None if attempt == 0 else context['rng'], deadline=None)
        schedule = child_counted_evaluation(result, context)
        status = schedule.status.value if schedule is not None else 'CONSTRUCTION_FAILED'
        if attempt == 0:
            context.update(initial_status=status, initial_cmax_s=schedule.cmax if schedule and schedule.feasible else None,
                           construct_s=time.perf_counter() - started)
        if schedule is not None and schedule.feasible:
            context.update(result=result, schedule=schedule, failure='')
            break
        context['failure'] = result.rejection_reason or (' | '.join(schedule.diagnostics) if schedule else status)
        if schedule is None or schedule.status is not ScheduleStatus.DEADLOCK:
            break
    context['last_status'] = status
    return context


def child_transfer_result(transfer):
    return gnn.JointCompletionResult(transfer.solution, transfer.state.directions, (transfer.action,), 0.,
                                     oriented_state=transfer.state)


def child_training_indices(transfers, old_scores, limit=4):
    """Old insertion ranking with robot/direction/position coverage, no labels."""
    order = torch.argsort(old_scores, descending=True, stable=True).tolist()
    if not order:
        return ()
    chosen = [order[0]]
    for attribute in (lambda t: t.action.robot, lambda t: t.action.direction, lambda t: t.action.position):
        covered = {attribute(transfers[i]) for i in chosen}
        for i in order:
            if len(chosen) >= limit:
                break
            if attribute(transfers[i]) not in covered:
                chosen.append(i)
                covered.add(attribute(transfers[i]))
    return tuple(dict.fromkeys((*chosen, *order)))[:limit]


def child_preference_pairs(outcomes):
    """Preferences only within one layout and one identical removed-block state."""
    pairs, kinds = [], []
    for first in range(len(outcomes)):
        for second in range(first + 1, len(outcomes)):
            a, b = outcomes[first], outcomes[second]
            if {a['status'], b['status']} - {'FEASIBLE', 'DEADLOCK'}:
                continue
            if a['status'] == b['status'] == 'FEASIBLE':
                if math.isclose(a['cmax_s'], b['cmax_s'], rel_tol=1e-9, abs_tol=1e-6):
                    continue
                better, worse = (first, second) if a['cmax_s'] < b['cmax_s'] else (second, first)
                kinds.append('CMAX')
            elif a['status'] != b['status']:
                better, worse = (first, second) if a['status'] == 'FEASIBLE' else (second, first)
                kinds.append('FEASIBILITY')
            else:
                continue
            pairs.append((better, worse))
    return pairs, kinds


@torch.no_grad()
def child_collect_job(layout):
    name, parents = layout
    model = worker_feedback_model
    context = child_initial_state(parents, model, SEED)
    current, schedule = context['result'], context['schedule']
    data = dict(initial_attempts=context['initial_attempts'], failure=context['failure'],
                block_id=None, candidate_pool=0, candidates=[], features=[], pairs=[], pair_kinds=[])
    if schedule is not None:
        feedback = lns.schedule_feedback_summary(current.solution, schedule,
            context['blockers'].get(plan_key(current.solution, current.directions), ()), CONFIG)
        block_id = lns.select_feedback_child(current.solution, feedback, CONFIG)
        data['block_id'] = block_id
        transfers = lns.child_transfer_candidates(current.solution, current.directions, block_id, CONFIG) if block_id else ()
        data['candidate_pool'] = len(transfers)
        if not transfers:
            data['failure'] = 'NO_MOVABLE_CRITICAL_BLOCK' if block_id is None else 'NO_PATTERN_PRESERVING_TRANSFER'
        rows = gnn.child_transfer_inputs(context['space'], context['embeddings'], current, feedback, transfers, model, CONFIG)
        chosen = child_training_indices(transfers, rows[:, -1])
        for i in chosen:
            transfer = transfers[i]
            result = child_transfer_result(transfer)
            outcome = child_counted_evaluation(result, context)
            context['counts']['reconstruction_attempts'] += 1
            context['counts']['reconstructions'] += 1
            data['candidates'].append(dict(block_id=block_id, source=transfer.move.key.source_robot_id,
                destination=transfer.action.robot, position=transfer.action.position, direction=transfer.action.direction,
                neighbour_changes=transfer.action.neighbour_changes, status=outcome.status.value,
                cmax_s=outcome.cmax if outcome.feasible else None))
            data['features'].append(rows[i].tolist())
        data['pairs'], data['pair_kinds'] = child_preference_pairs(data['candidates'])
    row = dict(stage='child_train_group', epoch=1, layout_id=name, seed=SEED,
        source=CHILD_TRAIN_SOURCE, method=CHILD_METHOD,
        status='FEASIBLE' if schedule is not None else context['last_status'],
        cmax_s=schedule.cmax if schedule is not None else None,
        initial_status=context['initial_status'], initial_cmax_s=context['initial_cmax_s'],
        construct_s=context['construct_s'], runtime_s=time.perf_counter() - context['started'],
        **context['counts'], diagnostic=json.dumps(data))
    return row


def collect_child_preferences(workers):
    layouts = load_layouts('train_expanded')
    retained = [r for r in read_rows(FEEDBACK_TRAIN_CSV)
                if r['source'] == CHILD_TRAIN_SOURCE and r['stage'] == 'child_train_group']
    completed = {(r['layout_id'], int(r['seed'])) for r in retained}
    pending = [layout for layout in layouts if (layout[0], SEED) not in completed]
    original = torch.load(FEEDBACK_MODEL, map_location='cpu', weights_only=True)
    print(json.dumps(dict(event='child_collection',completed=len(retained),remaining=len(pending))),flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=workers, initializer=configure_feedback_worker,
                                 initargs=(pickle.dumps(original['state_dict']),)) as pool:
            futures = [pool.submit(child_collect_job, layout) for layout in pending]
            for future in as_completed(futures):
                row = future.result()
                feedback_write(FEEDBACK_TRAIN_CSV, [row])
                retained.append(row)
                data = json.loads(row['diagnostic'])
                print(json.dumps(dict(event='child_collected',completed=len(retained),layout_id=row['layout_id'],
                    status=row['status'],candidates=len(data['candidates']),pairs=len(data['pairs']),
                    b32_calls=row['b32_calls'],runtime_s=row['runtime_s'])),flush=True)
    return retained


def fit_child_head(model, records):
    """One predeclared fit: 20 passes, AdamW 1e-3, frozen encoder/old heads."""
    groups = [json.loads(row['diagnostic']) for row in sorted(records, key=lambda r:(r['layout_id'],int(r['seed'])))]
    groups = [group for group in groups if group['pairs']]
    rows, pairs, kinds = [], [], []
    for group in groups:
        offset = len(rows)
        rows.extend(group['features'])
        pairs.extend((a + offset, b + offset) for a, b in group['pairs'])
        kinds.extend(group['pair_kinds'])
    summary = dict(train_layouts=len(records),initial_feasible=sum(r['status']=='FEASIBLE' for r in records),
        preference_groups=len(groups),preferences=len(pairs),cmax_preferences=kinds.count('CMAX'),
        feasibility_preferences=kinds.count('FEASIBILITY'),epochs=20,learning_rate=.001,updates=0)
    if not pairs:
        return dict(summary, status='NO_PREFERENCES')
    inputs = torch.tensor(rows, dtype=torch.float32)
    indices = torch.tensor(pairs, dtype=torch.long)
    model.child_transfer_mean.copy_(inputs.mean(0))
    scale = inputs.std(0, unbiased=False)
    model.child_transfer_scale.copy_(torch.where(scale >= .1, scale, torch.ones_like(scale)))
    optimizer = torch.optim.AdamW(model.child_transfer_head.parameters(), lr=.001, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(SEED)
    def loss_for(batch):
        scores = model.child_transfer_scores(inputs)
        return torch.nn.functional.softplus(-(scores[batch[:, 0]] - scores[batch[:, 1]])).mean()
    with torch.no_grad():
        summary['initial_loss'] = float(loss_for(indices))
    for _ in range(20):
        order = torch.randperm(len(indices), generator=generator)
        for batch in order.split(64):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(indices[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.child_transfer_head.parameters(), 1.)
            optimizer.step()
            summary['updates'] += 1
    with torch.no_grad():
        summary['final_loss'] = float(loss_for(indices))
    summary.update(status='TRAINED',head_parameters=sum(p.numel() for p in model.child_transfer_head.parameters()))
    return summary


@torch.no_grad()
def run_child_feedback_gnn(parents, model, seed, *, attempts=500, progress_observer=None):
    context = child_initial_state(parents, model, seed)
    current, current_schedule = context['result'], context['schedule']
    best, best_schedule = current, current_schedule
    c0 = current_schedule.cmax if current_schedule is not None else None
    improvements = uphill = equal = 0
    choices = {}
    if current_schedule is not None:
        for k in range(attempts):
            counts = context['counts']
            counts['reconstruction_attempts'] += 1
            feedback = lns.schedule_feedback_summary(current.solution, current_schedule,
                context['blockers'].get(plan_key(current.solution, current.directions), ()), CONFIG)
            block_id = lns.select_feedback_child(current.solution, feedback, CONFIG,
                                                  seed=context['rng'].randrange(2**32))
            candidate = None
            try:
                transfers = lns.child_transfer_candidates(current.solution, current.directions, block_id, CONFIG) if block_id else ()
                if transfers:
                    inputs = gnn.child_transfer_inputs(context['space'], context['embeddings'], current,
                                                        feedback, transfers, model, CONFIG)
                    scores = model.child_transfer_scores(inputs)
                    if not torch.isfinite(scores).all():
                        raise ArithmeticError('NONFINITE_CHILD_SCORES')
                    candidate = child_transfer_result(transfers[int(scores.argmax())])
                    choices[block_id] = choices.get(block_id, 0) + 1
            except (ValueError, ArithmeticError) as error:
                context['failure'] = str(error)
            if candidate is None:
                counts['incomplete'] += 1
            else:
                counts['reconstructions'] += 1
                schedule = child_counted_evaluation(candidate, context)
                if feedback_improves(schedule, best_schedule):
                    best, best_schedule = candidate, schedule
                    improvements += 1
                temperature = .01 * c0 * .01 ** (k / 499.)
                if feedback_accepts(schedule, current_schedule, temperature, context['rng']):
                    uphill += int(schedule.cmax > current_schedule.cmax)
                    equal += int(schedule.cmax == current_schedule.cmax)
                    current, current_schedule = candidate, schedule
                    counts['accepted'] += 1
            if progress_observer is not None and (k + 1) % 25 == 0:
                progress_observer(dict(attempts=k + 1,best_cmax_s=best_schedule.cmax,
                    b32_calls=counts['b32_calls'],runtime_s=time.perf_counter()-context['started']))
    diagnostic = dict(initial_attempts=context['initial_attempts'],initial_incumbent_cmax_s=c0,
        best_improvements=improvements,accepted_uphill=uphill,accepted_equal=equal,
        current_cmax_s=current_schedule.cmax if current_schedule is not None else None,
        selected_blocks=choices,failure=context['failure'] if best_schedule is None else '')
    return dict(status='FEASIBLE' if best_schedule is not None else context['last_status'],
        cmax_s=best_schedule.cmax if best_schedule is not None else None,
        initial_status=context['initial_status'],initial_cmax_s=context['initial_cmax_s'],
        construct_s=context['construct_s'],runtime_s=time.perf_counter()-context['started'],
        **context['counts'],diagnostic=json.dumps(diagnostic),**final_metrics(best_schedule))


def configure_child_worker(serialized):
    configure_cpu()
    global worker_child_model
    worker_child_model = child_feedback_model(pickle.loads(serialized)).eval()


def child_evaluation_job(job):
    name, parents, seed, attempts, source, stage = job
    print(json.dumps(dict(event='child_start',layout_id=name,seed=seed,attempts=attempts)),flush=True)
    progress = lambda data: print(json.dumps(dict(event='child_progress',layout_id=name,seed=seed,**data)),flush=True)
    return dict(stage=stage,epoch=1,layout_id=name,seed=seed,source=source,method=CHILD_METHOD,
                **run_child_feedback_gnn(parents,worker_child_model,seed,attempts=attempts,progress_observer=progress))


def evaluate_child_feedback(model, layouts, workers, *, formal=False):
    source, stage = (CHILD_FULL_SOURCE,'child_validation') if formal else (CHILD_PILOT_SOURCE,'child_pilot')
    seeds, attempts = (TEST_SEEDS,500) if formal else ((SEED,),100)
    expected = {(name,seed) for name,_ in layouts for seed in seeds}
    retained = [r for r in read_rows(FEEDBACK_TRAIN_CSV) if r['source']==source and r['stage']==stage
                and (r['layout_id'],int(r['seed'])) in expected]
    completed = {(r['layout_id'],int(r['seed'])) for r in retained}
    jobs = [(name,parents,seed,attempts,source,stage) for name,parents in layouts for seed in seeds
            if (name,seed) not in completed]
    if jobs:
        with ProcessPoolExecutor(max_workers=workers,initializer=configure_child_worker,
                                 initargs=(pickle.dumps(model.state_dict()),)) as pool:
            futures = [pool.submit(child_evaluation_job,job) for job in jobs]
            for future in as_completed(futures):
                row=future.result()
                feedback_write(FEEDBACK_TRAIN_CSV,[row]);retained.append(row)
                print(json.dumps(row),flush=True)
    return retained


def child_feedback_comparison(rows):
    keys={(r['layout_id'],int(r['seed'])) for r in rows}
    references={'GNN_PARENT_500':[r for r in read_rows(FEEDBACK_TRAIN_CSV)
        if r['source']==FEEDBACK_FULL_SOURCE and (r['layout_id'],int(r['seed'])) in keys]}
    for method,(_,_,filename) in BASELINES.items():
        references[method]=[r for r in read_rows(FINAL_RESULTS/filename)
            if completed_baseline_row(r) and (r['layout_id'],int(r['seed'])) in keys]
    summary=dict(gnn=feedback_summary(rows),references={name:feedback_summary(group) for name,group in references.items()},
        paired={name:feedback_paired(rows,group) for name,group in references.items()})
    print(json.dumps(summary),flush=True)
    return summary


def child_feedback_experiment(workers):
    """One TRAIN collection/fit, then prescribed pilot and conditional full VAL.

    Return the checkpoint in memory; saving is a separate final action so the
    requested new model file can be reviewed before creating it.
    """
    configure_cpu();torch.manual_seed(SEED)
    if CHILD_MODEL.exists():
        saved=torch.load(CHILD_MODEL,map_location='cpu',weights_only=True)
        model=child_feedback_model(saved['state_dict']).eval()
    else:
        records=collect_child_preferences(workers)
        model=child_feedback_model().eval()
        training=fit_child_head(model,records)
        saved=dict(state_dict=copy.deepcopy(model.state_dict()),seed=SEED,initialization='gnn_schedule_feedback.pt:state_dict',
            base_selected_epoch=0,hidden_dim=64,heads=4,training=training,training_complete=training['status']=='TRAINED')
        feedback_write(FEEDBACK_TRAIN_CSV,[dict(stage='child_train_fit',epoch=1,layout_id='TRAIN_120',seed=SEED,
            source=CHILD_TRAIN_SOURCE,method=CHILD_METHOD,status=training['status'],loss=training.get('final_loss'),
            steps=training['updates'],diagnostic=json.dumps(training))])
        print(json.dumps(dict(event='child_trained',**training)),flush=True)
        if not saved['training_complete']:
            return saved
    ordered=sorted(load_layouts('validation'),key=lambda item:(len(item[1]),item[0]))
    pilot=[ordered[round(i*(len(ordered)-1)/4)] for i in range(5)]
    print(json.dumps(dict(event='child_pilot_layouts',layouts=[(n,len(p)) for n,p in pilot])),flush=True)
    rows=evaluate_child_feedback(model,pilot,workers)
    summary=child_feedback_comparison(rows)
    old,hga=summary['paired']['GNN_PARENT_500'],summary['paired']['HGA']
    passed=(len(rows)==5 and summary['gnn']['feasible']==5 and old['wins']>=4 and
            old['median_change_pct'] is not None and old['median_change_pct']<0 and hga['wins']>=3)
    saved.update(pilot=summary,pilot_passed=passed,pilot_layouts=[name for name,_ in pilot])
    if passed:
        formal_rows=evaluate_child_feedback(model,ordered,workers,formal=True)
        formal=child_feedback_comparison(formal_rows)
        eligible=(len(formal_rows)==30 and formal['gnn']['feasible']==30 and all(
            formal['paired'][name]['median_change_pct'] is not None and
            formal['paired'][name]['median_change_pct']<0 and
            formal['paired'][name]['wins']>formal['paired'][name]['losses'] for name in BASELINES))
        saved.update(formal_validation=formal,test_eligible=eligible)
    else:
        saved['test_eligible']=False
        print('Child pilot failed the predeclared requirements; no full VALIDATION or TEST is run.',flush=True)
    print(json.dumps(dict(event='child_checkpoint_ready',path=str(CHILD_MODEL),pilot_passed=passed,
                          test_eligible=saved['test_eligible'])),flush=True)
    return saved


PPO_DATA = DATA/'scale1000'
PPO_MODELS = ROOT/'models/gnn_ppo_scale'
PPO_RESULTS = FINAL_RESULTS/'gnn_ppo_scale.csv'
PPO_SEED = 20261119
PPO_OPTIONS = dict(epochs=3,rollout_batch=4,update_epochs=4,learning_rate=1e-4,
    gamma=1.,gae_lambda=.95,clip=.2,value_coefficient=.5,entropy_coefficient=.01,max_gradient_norm=.5)
PPO_FIELDS = ('stage','size','epoch','index','layout_id','parent_count','seed','method','status',
    'cmax_s','runtime_s','max_processing_s','move_s','wait_s','b32_calls','steps','reward',
    'loss','gradient_norm','diagnostic','source')


def ppo_write(rows):
    """One existing CSV; an ordinary byte lock permits concurrent training/baseline jobs."""
    import msvcrt
    with PPO_RESULTS.open('a+',encoding='utf-8',newline='') as stream:
        stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        try:
            stream.seek(0,2)
            writer=csv.DictWriter(stream,fieldnames=PPO_FIELDS)
            if stream.tell()==0:writer.writeheader()
            for row in rows:writer.writerow(dict(source='GNN_PPO_SCALE_V1',**row))
            stream.flush()
        finally:
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)


def ppo_layouts(split):
    rows=[json.loads(s) for s in (PPO_DATA/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    return [(r['layout_id'],tuple(ParentWeld(p['parent_id'],tuple(p['start']),tuple(p['end']))
            for p in r['parents'])) for r in rows]


def ppo_model(weights=None):
    torch.manual_seed(PPO_SEED)
    actor=feedback_model() if weights is None else None
    model=gnn.PPOJointInsertionGNN(64,4)
    if weights is None:
        missing,extra=model.load_state_dict(actor.state_dict(),strict=False)
        if set(missing)!={f'value_head.{layer}.{part}' for layer in (0,2) for part in ('weight','bias')} or extra:
            raise ValueError('initial actor parameters do not match the existing joint GNN')
    else:model.load_state_dict(weights)
    return model


def ppo_distribution(scores,mask):
    if not mask.any() or not torch.isfinite(scores[mask]).all():
        raise ArithmeticError('invalid masked joint policy scores')
    return torch.distributions.Categorical(logits=scores.masked_fill(~mask,-math.inf))


def ppo_gae(values,reward,*,gamma=1.,gae_lambda=.95):
    advantage=torch.zeros_like(values);last=values.new_zeros(())
    for i in range(len(values)-1,-1,-1):
        following=values[i+1] if i+1<len(values) else values.new_zeros(())
        delta=(reward if i==len(values)-1 else 0.)+gamma*following-values[i]
        last=delta+gamma*gae_lambda*last
        advantage[i]=last
    return advantage,advantage+values


def ppo_rollout(model,layout,seed):
    started=time.perf_counter();name,parents=layout
    space=gnn.parent_assignment_space(parents,CONFIG)
    trace=[];pending_state={}
    with torch.no_grad():
        embeddings=model.encode_static(space.graph)
        def observe_state(state,selected,plans,scores):
            features=model.partial_value_features(space,state,selected)
            pending_state.update(features=features,value=float(model.state_value(space,embeddings,features)),
                scores=scores.detach().clone(),routes=state.routes,directions=state.directions,
                action_keys=tuple((p.assignment_index,tuple((a.block_id,a.robot,a.position,a.direction,
                    a.neighbour_changes) for a in p.actions)) for p in plans))
        def observe_policy(inputs,choice):
            mask=torch.ones(len(pending_state['scores']),dtype=torch.bool)
            distribution=ppo_distribution(pending_state['scores'],mask)
            trace.append(dict(inputs={k:v.detach().clone() for k,v in inputs.items()},choice=choice,mask=mask,
                old_log_probability=float(distribution.log_prob(torch.tensor(choice))),
                old_probabilities=distribution.probs.detach().clone(),**pending_state))
        try:
            result=gnn.construct_vnext_gnn(parents,model,CONFIG,space=space,embeddings=embeddings,
                rng=Random(seed),step_observer=observe_state,policy_observer=observe_policy)
        except ArithmeticError as error:
            return dict(space=space,trace=trace,reward=None,row=dict(layout_id=name,parent_count=len(parents),
                seed=seed,status='NUMERIC_FAILURE',runtime_s=time.perf_counter()-started,b32_calls=0,
                steps=len(trace),diagnostic=str(error)))
        schedule,called=evaluate_once(result,{})
    reward=feedback_reward(schedule,max(1.,sum(CONFIG.process_time(p.length) for p in parents)/4.))
    status=schedule.status.value if schedule is not None else 'CONSTRUCTION_FAILED'
    row=dict(layout_id=name,parent_count=len(parents),seed=seed,status=status,
        cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
        runtime_s=time.perf_counter()-started,b32_calls=int(called),steps=len(trace),reward=reward,
        diagnostic=result.rejection_reason or (' | '.join(schedule.diagnostics) if schedule is not None and not schedule.feasible else ''),
        **final_metrics(schedule))
    if reward is not None and trace:
        values=torch.tensor([t['value'] for t in trace])
        advantage,returns=ppo_gae(values,reward,gamma=PPO_OPTIONS['gamma'],gae_lambda=PPO_OPTIONS['gae_lambda'])
        for t,a,target in zip(trace,advantage,returns):t.update(advantage=a,return_value=target)
    return dict(space=space,trace=trace,reward=reward,row=row)


def ppo_update(model,episodes,optimizer):
    valid=[e for e in episodes if e['reward'] is not None and e['trace']]
    if not valid:return dict(updates=0,loss=None,gradient_norm=None)
    for episode in valid:
        trace=episode['trace']
        episode['batch_inputs']={k:torch.cat([t['inputs'][k] for t in trace]) for k in trace[0]['inputs']}
        episode['candidate_counts']=[len(t['mask']) for t in trace]
    # Replay the same candidate tensors before any update, including the full normalizer.
    replay_error=0.
    with torch.no_grad():
        for episode in valid:
            embeddings=model.encode_static(episode['space'].graph)
            scores=model.feedback_scores(episode['space'],embeddings,episode['batch_inputs'])
            for step,scores in zip(episode['trace'],scores.split(episode['candidate_counts'])):
                d=ppo_distribution(scores,step['mask'])
                replay_error=max(replay_error,float((d.probs-step['old_probabilities']).abs().max()))
    if replay_error>2e-6:raise RuntimeError(f'joint policy replay probability mismatch: {replay_error}')
    all_advantages=torch.stack([t['advantage'] for e in valid for t in e['trace']])
    center,spread=all_advantages.mean(),all_advantages.std(unbiased=False).clamp_min(1e-8)
    measures=[]
    for _ in range(PPO_OPTIONS['update_epochs']):
        for episode in valid:
            optimizer.zero_grad(set_to_none=True)
            space=episode['space'];embeddings=model.encode_static(space.graph)
            actor_losses=[];targets=[];entropies=[];ratios=[]
            scores=model.feedback_scores(space,embeddings,episode['batch_inputs'])
            for step,scores in zip(episode['trace'],scores.split(episode['candidate_counts'])):
                distribution=ppo_distribution(scores,step['mask'])
                log_probability=distribution.log_prob(torch.tensor(step['choice']))
                ratio=torch.exp(log_probability-step['old_log_probability'])
                advantage=(step['advantage']-center)/spread
                actor_losses.append(-torch.minimum(ratio*advantage,
                    ratio.clamp(1.-PPO_OPTIONS['clip'],1.+PPO_OPTIONS['clip'])*advantage))
                entropies.append(distribution.entropy());ratios.append(ratio)
                targets.append(step['return_value'])
            actor_loss=torch.stack(actor_losses).mean()
            values=model.state_value(space,embeddings,torch.stack([t['features'] for t in episode['trace']]))
            value_loss=F.mse_loss(values,torch.stack(targets))
            entropy=torch.stack(entropies).mean()
            loss=actor_loss+PPO_OPTIONS['value_coefficient']*value_loss-PPO_OPTIONS['entropy_coefficient']*entropy
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite PPO loss; not a B32 DEADLOCK')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),PPO_OPTIONS['max_gradient_norm'])
            if not torch.isfinite(norm):raise ArithmeticError('nonfinite PPO gradient; not a B32 DEADLOCK')
            optimizer.step()
            measures.append((float(loss.detach()),float(norm),float(actor_loss.detach()),
                float(value_loss.detach()),float(entropy.detach()),
                float((torch.stack(ratios).detach()-1.).abs().gt(PPO_OPTIONS['clip']).float().mean())))
    means=[sum(m[i] for m in measures)/len(measures) for i in range(6)]
    return dict(zip(('loss','gradient_norm','actor_loss','value_loss','entropy','clip_fraction'),means),
                updates=len(measures),replay_probability_max_error=replay_error)


def ppo_rollout_job(job):
    configure_cpu()
    serialized,layout,seed=job
    model=ppo_model(pickle.loads(serialized)).eval()
    # A single byte payload avoids one multiprocessing handle per trace tensor.
    return pickle.dumps(ppo_rollout(model,layout,seed),protocol=pickle.HIGHEST_PROTOCOL)


def ppo_train_size(size,workers=4):
    configure_cpu()
    started=time.perf_counter();layouts=ppo_layouts('train')[:size]
    if len(layouts)!=size:raise ValueError('the requested nested TRAIN prefix is not available')
    path=PPO_MODELS/f'ppo_{size}.pt';PPO_MODELS.mkdir(parents=True,exist_ok=True)
    model=ppo_model().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=PPO_OPTIONS['learning_rate'],weight_decay=1e-5)
    saved=dict(size=size,seed=PPO_SEED,settings=dict(PPO_OPTIONS),completed_episodes=0,history=[],
        initial_state_dict=copy.deepcopy(model.state_dict()),training_complete=False,elapsed_s=0.,
        initialization='models/gnn_weld_layout.pt:models.GNN_VNEXT_WORKLOAD',training_dataset=str(PPO_DATA/'train.jsonl'))
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['settings']!=PPO_OPTIONS or saved['size']!=size:raise ValueError('existing PPO run has different settings')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer_state_dict'])
    existing={(r['size'],r['epoch'],r['index']) for r in read_rows(PPO_RESULTS) if r['stage']=='train'}
    ppo_write([r for r in saved['history'] if (str(r['size']),str(r['epoch']),str(r['index'])) not in existing])
    if saved['training_complete']:return saved['history']
    order=[]
    for epoch in range(1,PPO_OPTIONS['epochs']+1):
        indices=list(range(size));Random(PPO_SEED+epoch).shuffle(indices)
        order.extend((epoch,i) for i in indices)
    previous_elapsed=saved['elapsed_s']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while saved['completed_episodes']<len(order):
            first=saved['completed_episodes'];batch=[]
            active=order[first:first+PPO_OPTIONS['rollout_batch']]
            serialized=pickle.dumps(model.state_dict(),protocol=pickle.HIGHEST_PROTOCOL)
            jobs=[(serialized,layouts[index],PPO_SEED+epoch*1000000+index) for epoch,index in active]
            for (epoch,index),payload in zip(active,pool.map(ppo_rollout_job,jobs)):
                episode=pickle.loads(payload)
                episode['row'].update(stage='train',size=size,epoch=epoch,index=index,method='GNN_PPO')
                batch.append(episode)
            update=ppo_update(model,batch,optimizer)
            for episode in batch:
                row=episode['row'];row.update(loss=update['loss'],gradient_norm=update['gradient_norm'])
                row['diagnostic']=json.dumps(dict(outcome=row['diagnostic'],ppo=update),ensure_ascii=False)
                saved['history'].append(row)
            saved.update(completed_episodes=first+len(batch),elapsed_s=previous_elapsed+time.perf_counter()-started,
                state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                training_complete=first+len(batch)==len(order))
            torch.save(saved,path)
            ppo_write([e['row'] for e in batch])
            print('PPO_TRAIN_PROGRESS',json.dumps(dict(size=size,completed=saved['completed_episodes'],total=len(order),
                elapsed_s=saved['elapsed_s'],last_status=batch[-1]['row']['status'],**update)),flush=True)
    return saved['history']


def ppo_train(workers):
    for size in (120,500,1000):ppo_train_size(size,workers)


def ppo_representatives():
    layouts=ppo_layouts('validation')
    if len(layouts)!=100:raise ValueError('select the 20 cases after the full 100 VALIDATION layouts are generated')
    ordered=sorted(layouts,key=lambda row:(len(row[1]),row[0]))
    selected=[ordered[5*i+2] for i in range(20)]
    rows=[r for r in read_rows(PPO_RESULTS) if r['stage']=='selection']
    if rows:
        if [r['layout_id'] for r in rows]!=[name for name,_ in selected]:
            raise ValueError('saved representative cases differ from the original parent-count strata')
    else:
        if any(r['stage'] in ('validation','baseline') for r in read_rows(PPO_RESULTS)):
            raise ValueError('representative cases must be recorded before viewing validation results')
        ppo_write([dict(stage='selection',size=0,epoch=0,index=i,layout_id=name,parent_count=len(parents),
            seed=SEED,method='PARENT_COUNT_STRATUM',diagnostic='middle case in sorted stratum of five')
            for i,(name,parents) in enumerate(selected)])
    return selected


def ppo_baseline_job(job):
    configure_cpu()
    layout,method=job
    row=baseline_job((layout,method,SEED))
    output={k:row[k] for k in ('layout_id','seed','method','status','cmax_s','runtime_s','move_s','wait_s','b32_calls') if k in row}
    loads=[row.get(f'processing_r{r}_s') for r in range(4)]
    output.update(stage='baseline',size=0,parent_count=len(layout[1]),
        max_processing_s=max(loads) if all(p is not None for p in loads) else None,
        diagnostic=json.dumps(row,ensure_ascii=False))
    return output


def ppo_baselines(workers):
    layouts=ppo_representatives()
    done={(r['layout_id'],r['method']) for r in read_rows(PPO_RESULTS) if r['stage']=='baseline'}
    jobs=[(layout,method) for method in ('HGA','WAG+VNS') for layout in layouts if (layout[0],method) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(ppo_baseline_job,job) for job in jobs]):
            row=future.result();ppo_write([row]);print('PPO_BASELINE_RESULT',json.dumps(row),flush=True)


def ppo_inference_job(job):
    configure_cpu();size,layout=job
    started=time.perf_counter()
    if size:
        saved=torch.load(PPO_MODELS/f'ppo_{size}.pt',map_location='cpu',weights_only=True)
        if not saved['training_complete']:raise ValueError('PPO training is incomplete')
        model=ppo_model(saved['state_dict']).eval()
    else:model=ppo_model().eval()
    model_loaded=time.perf_counter()
    name,parents=layout;space=gnn.parent_assignment_space(parents,CONFIG)
    seen={};calls=deadlocks=numeric=0;result=schedule=None;failures=[]
    with torch.no_grad():
        embeddings=model.encode_static(space.graph)
        for attempt in range(3):
            try:
                result=gnn.construct_vnext_gnn(parents,model,CONFIG,space=space,embeddings=embeddings,
                    rng=None if attempt==0 else Random(SEED+attempt))
            except ArithmeticError as error:
                numeric+=1;failures.append(str(error));break
            schedule,called=evaluate_once(result,seen);calls+=int(called)
            if schedule is not None:
                numeric+=schedule.status is ScheduleStatus.NUMERIC_FAILURE
                deadlocks+=schedule.status is ScheduleStatus.DEADLOCK
                if schedule.feasible or schedule.status is ScheduleStatus.NUMERIC_FAILURE:break
    status=('NUMERIC_FAILURE' if numeric else schedule.status.value if schedule is not None else 'CONSTRUCTION_FAILED')
    return dict(stage='validation',size=size,epoch=PPO_OPTIONS['epochs'] if size else 0,layout_id=name,
        parent_count=len(parents),seed=SEED,method='GNN_PPO' if size else 'GNN_INITIAL',status=status,
        cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
        runtime_s=time.perf_counter()-started,b32_calls=calls,
        diagnostic=json.dumps(dict(model_loading_s=model_loaded-started,attempts=attempt+1,
            deadlocks=deadlocks,numeric_failures=numeric,failures=failures,
            schedule_diagnostic=list(schedule.diagnostics) if schedule is not None else None,
            solution=(dict(patterns=[p.pattern_id for p in result.solution.patterns],
                routes=[list(r.block_ids) for r in result.solution.routes],directions=result.directions)
                if result is not None and result.solution is not None else None)),ensure_ascii=False),**final_metrics(schedule))


def ppo_validation(workers,size=None):
    ppo_representatives()
    layouts=ppo_layouts('validation')
    sizes=(size,) if size is not None else (0,120,500,1000)
    done={(int(r['size']),r['layout_id']) for r in read_rows(PPO_RESULTS) if r['stage']=='validation'}
    jobs=[(n,layout) for n in sizes for layout in layouts if (n,layout[0]) not in done]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(ppo_inference_job,job) for job in jobs]):
            row=future.result();ppo_write([row]);print('PPO_VALIDATION_RESULT',json.dumps({k:v for k,v in row.items() if k!='diagnostic'}),flush=True)


def ppo_summary():
    rows=read_rows(PPO_RESULTS);output={}
    for size in (0,120,500,1000):
        selected=[r for r in rows if r['stage']=='validation' and int(r['size'])==size]
        if not selected:continue
        good=[r for r in selected if r['status']=='FEASIBLE']
        info=feedback_summary(selected)
        for key in ('max_processing_s','move_s','wait_s'):
            info['median_'+key]=median(float(r[key]) for r in good) if good else None
        for other in (0,120,500):
            ref=[r for r in rows if r['stage']=='validation' and int(r['size'])==other]
            if other!=size and ref:
                info[f'paired_to_{other}']=feedback_paired(selected,ref)
                lookup={(r['layout_id'],r['seed']):r for r in ref}
                changes=[100*(float(r['max_processing_s'])/float(lookup[r['layout_id'],r['seed']]['max_processing_s'])-1)
                    for r in good if (r['layout_id'],r['seed']) in lookup
                    and lookup[r['layout_id'],r['seed']]['status']=='FEASIBLE']
                info[f'paired_to_{other}']['median_pmax_change_pct']=median(changes) if changes else None
        for method in ('WAG+VNS','HGA'):
            ref=[r for r in rows if r['stage']=='baseline' and r['method']==method]
            if ref:info[method]=dict(summary=feedback_summary(ref),paired=feedback_paired(selected,ref))
        output[size]=info
    chosen=[r for r in rows if r['stage']=='validation' and int(r['size'])==1000]
    representative_ids={r['layout_id'] for r in rows if r['stage']=='selection'}
    subset=[r for r in chosen if r['layout_id'] in representative_ids]
    compared={m:[r for r in rows if r['stage']=='baseline' and r['method']==m] for m in ('WAG+VNS','HGA')}
    complete=len(chosen)==100 and len(subset)==20 and all(len(v)==20 for v in compared.values())
    quality=complete and all(
        sum(r['status']=='FEASIBLE' for r in subset)>=sum(r['status']=='FEASIBLE' for r in ref)
        and sum(r['status']=='FEASIBLE' for r in chosen)*len(ref)>=sum(r['status']=='FEASIBLE' for r in ref)*len(chosen)
        and feedback_paired(subset,ref)['common']>0
        and feedback_paired(subset,ref)['median_change_pct']<0
        and feedback_paired(subset,ref)['wins']>feedback_paired(subset,ref)['losses']
        for ref in compared.values())
    # Predeclared before any new validation outcomes: material Pmax improvement
    # means a paired median reduction of at least 1% versus both initial and 120.
    processing=complete and all(output[1000].get(f'paired_to_{n}',{}).get('median_pmax_change_pct') is not None
        and output[1000][f'paired_to_{n}']['median_pmax_change_pct']<=-1. for n in (0,120))
    efficient=complete and median(float(r['runtime_s']) for r in subset)<=median(float(r['runtime_s']) for r in compared['WAG+VNS'])
    output['test_decision']=dict(complete=complete,quality_passed=quality,
        processing_passed=processing,time_passed=efficient,passed=complete and quality and processing and efficient)
    print('PPO_SCALE_SUMMARY',json.dumps(output,ensure_ascii=False),flush=True)
    return output


def ppo_smoke():
    configure_cpu();model=ppo_model().train()
    lookup=dict(ppo_layouts('train'))
    episodes=[ppo_rollout(model,(name,lookup[name]),PPO_SEED+i) for i,name in enumerate(('train_15','train_34'))]
    optimizer=torch.optim.AdamW(model.parameters(),lr=PPO_OPTIONS['learning_rate'])
    update=ppo_update(model,episodes,optimizer)
    for episode in episodes:
        row=episode['row'];row.update(stage='smoke',size=0,epoch=0,method='GNN_PPO',
            loss=update['loss'],gradient_norm=update['gradient_norm'],diagnostic=json.dumps(update))
        ppo_write([row])
    print('PPO_SMOKE',json.dumps(update),flush=True)


IMITATION_MODEL = ROOT/'models/gnn_weld_layout_imitation.pt'
IMITATION_RESULTS = FINAL_RESULTS/'gnn_imitation_balance_pilot.csv'
IMITATION_CASES = ('scale_validation_0031','scale_validation_0095','scale_validation_0067',
                   'scale_validation_0076','scale_validation_0084','scale_validation_0100')
IMITATION_FIELDS = ('stage','layout_id','parent_count','seed','method','status','cmax_s',
    'robot_task_count_r0','robot_task_count_r1','robot_task_count_r2','robot_task_count_r3',
    'processing_r0_s','processing_r1_s','processing_r2_s','processing_r3_s','max_processing_s',
    'move_s','wait_s','runtime_s','b32_calls','four_robot_cover_possible','wait_for_graph','diagnostic','source')


def imitation_model(weights=None, *, balance=True):
    # The original workload Actor has no Critic or schedule-feedback head.
    model = joint_model(weights)
    model.imitation_balance = balance
    return model


def imitation_hga_selection():
    rows = [json.loads(line) for line in (PPO_DATA/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    ordered = sorted(rows, key=lambda r:(len(r['parents']),r['layout_id']))
    selected = []
    for k in range(4):
        group = sorted(ordered[k*len(rows)//4:(k+1)*len(rows)//4],
                       key=lambda r:(len(r['parts']),r['rectangle_utilization'],r['layout_id']))
        row = group[len(group)//2]
        selected.append(dict(layout_id=row['layout_id'],parent_count=len(row['parents']),
                             part_count=len(row['parts']),rectangle_utilization=row['rectangle_utilization'],
                             parent_count_stratum=k))
    return selected


def imitation_saved():
    if IMITATION_MODEL.exists():
        return torch.load(IMITATION_MODEL,map_location='cpu',weights_only=True)
    source = torch.load(EXPANDED_MODEL,map_location='cpu',weights_only=True)
    expanded = dict(load_layouts('train_expanded'))
    training = dict(ppo_layouts('train'))
    teachers = []
    for sample in source['teachers']:
        name = sample['layout_id']
        if name not in training or expanded[name] != training[name]:
            raise ValueError(f'{name}: saved teacher geometry differs from TRAIN')
        for record in sample['records']:
            row,plan = record['row'],record.get('plan')
            if row['status'] != 'FEASIBLE':
                continue
            if row['method']=='WAG+VNS' and row.get('iterations')!=500:
                continue
            if not plan or not all(k in plan for k in ('patterns','routes','directions')):
                continue
            if not math.isfinite(row['cmax_s']) or len(plan['routes'])!=4 or any(
                    len(r)!=len(d) for r,d in zip(plan['routes'],plan['directions'])):
                raise ValueError(f'{name}: incomplete feasible demonstration')
            teachers.append(dict(layout_id=name,**record))
    model = imitation_model()
    return dict(seed=SEED,initialization='models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD',
                initial_state_dict=copy.deepcopy(model.state_dict()),teachers=teachers,
                hga_selection=imitation_hga_selection(),hga_records=[],training_complete=False,
                history=[],validation={},pilot=[])


def imitation_save(saved):
    torch.save(saved,IMITATION_MODEL)


def imitation_hga_job(layout):
    configure_cpu()
    name,parents = layout
    started = time.perf_counter()
    print('IMITATION_HGA_START',name,len(parents),flush=True)
    result = baseline_run(parents,'HGA',SEED,math.inf,(),iterations=200000,
                          candidate_evaluator=priority_schedule,formal=True)
    if result.iterations!=200000:
        raise ValueError(f'{name}: HGA ended before 200000 effective offspring')
    best = result.timely_best
    plan = gnn.JointCompletionResult(best.solution if best else None,best.directions if best else None,(),
                                   time.perf_counter()-started)
    schedule = b32(plan) if best is not None else None
    row = dict(layout_id=name,method='HGA',seed=SEED,iterations=result.iterations,
               status=schedule.status.value if schedule else 'NO_COMPLETE_SOLUTION',
               cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
               runtime_s=time.perf_counter()-started,b32_calls=int(schedule is not None),
               diagnostic=list(schedule.diagnostics) if schedule is not None else [],
               wait_for_graph=schedule.wait_for_graph if schedule is not None else (),
               **baseline_metrics(best,schedule))
    return dict(layout_id=name,row=row,plan=pack_directed_plan(plan))


def imitation_teachers(workers):
    saved = imitation_saved()
    imitation_save(saved)
    done = {r['layout_id'] for r in saved['hga_records']}
    training = dict(ppo_layouts('train'))
    jobs = [(r['layout_id'],training[r['layout_id']]) for r in saved['hga_selection'] if r['layout_id'] not in done]
    print('IMITATION_TEACHERS_RECOVERED',len(saved['teachers']),
          len({r['layout_id'] for r in saved['teachers']}),'HGA_SELECTION',saved['hga_selection'],flush=True)
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(imitation_hga_job,job) for job in jobs]):
                record = future.result()
                saved['hga_records'].append(record)
                imitation_save(saved)
                print('IMITATION_HGA_RESULT',json.dumps(record['row']),flush=True)
    return saved


@torch.no_grad()
def imitation_evaluate(layout, model, *, method='GNN_IMITATION', seen=None):
    name,parents = layout
    started = time.perf_counter()
    space = gnn.parent_assignment_space(parents,CONFIG)
    forecast_trace = []
    def observe(state, selected, plans, scores):
        chosen = plans[int(scores.argmax())]
        assignment = space.actions[chosen.assignment_index]
        forecast_trace.append(dict(parent_id=assignment.parent_id,mode=assignment.pattern.kind.value,
            owners=assignment.robots,processing_forecast_s=(chosen.assignment_features[-8:-4]*2500.).tolist(),
            processing_after_s=chosen.state.loads,candidate_count=len(plans),
            owner_alternatives=len({space.actions[p.assignment_index].robots for p in plans})))
    result = gnn.construct_vnext_gnn(parents,model.eval(),CONFIG,space=space,
        **(dict(step_observer=observe) if getattr(model,'imitation_balance',False) else {}))
    schedule,called = evaluate_once(result,{} if seen is None else seen)
    metrics = beam_validation_metrics(result)
    loads = [metrics.get(f'processing_r{r}_s') for r in range(4)]
    diagnostic = dict(coverage=space.coverage_diagnostic(),
        rejection=result.rejection_reason,solution=pack_directed_plan(result),
        schedule_diagnostic=list(schedule.diagnostics) if schedule is not None else [],
        baseline_deadlock=schedule.baseline_deadlock if schedule is not None else None,
        recovery_exhausted=schedule.recovery_exhausted if schedule is not None else None,
        wait_for_graph=schedule.wait_for_graph if schedule is not None else ())
    if forecast_trace:
        diagnostic['processing_forecast_trace']=forecast_trace
    if schedule is not None and not schedule.feasible and result.solution is not None:
        blocks = block_map(result.solution,CONFIG,scope=FORMAL_SCOPE_V2)
        templates = [op for r,route in enumerate(result.solution.routes)
            for op in build_operation_templates(RobotRoute(r,tuple(blocks[b] for b in route.block_ids),
                                                           result.directions[r]),CONFIG)]
        diagnostic['operation_templates']=[dict(id=op.operation_id,robot=op.robot_id,kind=op.kind.value,
            start=op.start,end=op.end,duration=op.duration) for op in templates]
        diagnostic['wait_observation']='WAIT accumulated in the committed partial B32 timeline; full Cmax unavailable'
        diagnostic['last_committed_operations']=[dict(id=op.operation_id,robot=op.robot_id,kind=op.kind.value,
            start=op.start,end=op.end,start_s=op.start_time,end_s=op.end_time)
            for r in range(4) for op in tuple(o for o in schedule.operations if o.robot_id==r)[-3:]]
    counts = ({f'robot_task_count_r{r}':len(route.block_ids) for r,route in enumerate(result.solution.routes)}
              if result.solution is not None else {})
    return dict(layout_id=name,seed=SEED,method=method,parent_count=len(parents),
        status=schedule.status.value if schedule is not None else 'CONSTRUCTION_FAILED',
        cmax_s=schedule.cmax if schedule is not None and schedule.feasible else None,
        runtime_s=time.perf_counter()-started,b32_calls=int(called),**counts,**metrics,
        four_robot_cover_possible=space.four_robot_cover_possible,
        max_processing_s=max(loads) if all(p is not None for p in loads) else None,
        wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT)
               if schedule is not None else None,
        wait_for_graph=json.dumps(schedule.wait_for_graph if schedule is not None else ()),
        diagnostic=json.dumps(diagnostic,ensure_ascii=False))


def imitation_best_teachers(saved):
    best = {}
    for record in (*saved['teachers'],*saved['hga_records']):
        if record['row']['status']!='FEASIBLE' or record.get('plan') is None:
            continue
        name = record['layout_id']
        if name not in best or record['row']['cmax_s']<best[name]['row']['cmax_s']:
            best[name] = record
    return best


@torch.no_grad()
def imitation_replay(layout, record, model, *, joint_stride=8):
    """Replay every parent into its exact teacher route and directed position.

    Position enumeration is complete for the bounded coarse assignment pool.
    Structural sampling may bound negatives, but can never remove a positive.
    No B32, native evaluation, or labels from another split are used here.
    """
    name,parents = layout
    space = gnn.parent_assignment_space(parents,CONFIG)
    result = unpack_directed_plan(parents,record['plan'],space)
    routes = tuple(r.block_ids for r in result.solution.routes)
    bits = tuple(tuple(d) for d in result.directions)
    if routes!=tuple(tuple(r) for r in record['plan']['routes']):
        raise ValueError(f'{name}: canonicalization changed the teacher routes')
    labels = sorted(gnn.parent_assignment_targets(result.solution,CONFIG),
                    key=lambda a:(-sum(CONFIG.process_time(b.length) for b in space.children[a.pattern]),a.parent_id))
    all_blocks = block_map(result.solution,CONFIG,scope=FORMAL_SCOPE_V2)
    all_nodes = {b.block_id:space.block_nodes[a.pattern,b.block_id] for a in labels for b in space.children[a.pattern]}
    embeddings = model.encode_static(space.graph)
    state = lns.oriented_route_state(((),)*4,((),)*4,{},CONFIG)
    kept,selected = set(),{}
    assignment_steps,insertion_steps,joint_steps,trace = [],[],[],[]
    rescued = 0
    while labels:
        pending = {a.parent_id for a in labels}
        blocks = {b:all_blocks[b] for b in kept}
        nodes = {b:all_nodes[b] for b in kept}
        indices,features = model.assignment_features(space,embeddings,state,blocks,pending,CONFIG)
        coverage = gnn.coverage_preserving_assignments(space,indices,state,pending)
        indices,features = indices[coverage],features[coverage]
        positions = {int(index):i for i,index in enumerate(indices)}
        prepared = None
        # Same-robot split siblings need their actual separating teacher block
        # to have been inserted first; defer such parents rather than inventing
        # an action absent from the legal atomic-parent space.
        for assignment in labels:
            target = space.action_index[assignment]
            if target not in positions:
                continue
            children = space.children[assignment.pattern]
            local = {**blocks,**{b.block_id:b for b in children}}
            graph = replace(space.graph,block_index={**nodes,**{b.block_id:all_nodes[b.block_id] for b in children}})
            after,visible = state,set(kept)
            fine_steps,actions,retained_features,retained_indices = [],[],[],[]
            local_rescued,risk = 0,0.
            for child,robot in zip(children,assignment.robots):
                visible.add(child.block_id)
                teacher = project_routes(routes,bits,visible,all_blocks)
                variants = lns.joint_block_candidates(after,child,assignment.pattern,local,CONFIG,
                    robot_id=robot,insertion_limit=max(8,len(after.routes[robot])+1),
                    check_initial_feasibility=False,prefer_legal_start_recovery=False)
                matches = [i for i,(v,_) in enumerate(variants) if v.routes==teacher.routes and v.directions==teacher.directions]
                if not matches:
                    break
                chosen = list(gnn.structural_child_indices(variants,limit=12))
                if matches[0] not in chosen:
                    chosen.append(matches[0]);local_rescued+=1
                variants = tuple(variants[i] for i in chosen)
                answer = chosen.index(matches[0])
                ix,inputs = gnn.vnext_insertion_inputs(graph,after,variants,local,CONFIG)
                fine_steps.append((ix,inputs,F.one_hot(torch.tensor(answer),len(variants)).float()))
                retained_features.append(inputs[answer]);retained_indices.append(ix[answer])
                risk = max(risk,float(inputs[answer,-8]+2*inputs[answer,-7]+.5*inputs[answer,-4]))
                after,action = variants[answer]
                actions.append(action)
            if len(actions)==len(children):
                prepared=(assignment,target,after,visible,fine_steps,tuple(actions),
                          tuple(retained_features),tuple(retained_indices),local_rescued,risk)
                break
        if prepared is None:
            raise ValueError(f'{name}: teacher cannot be represented as complete-parent actions; pending={sorted(pending)}')
        assignment,target,after,visible,fine_steps,actions,fine_features,fine_indices,count,risk = prepared
        answer = positions[target]
        assignment_steps.append((indices,features,F.one_hot(torch.tensor(answer),len(indices)).float()))
        insertion_steps.extend(fine_steps)
        rescued+=count
        if len(trace)%joint_stride==0 or len(labels)==1:
            negative,_ = gnn.vnext_joint_actions(space,embeddings,model,state,blocks,nodes,pending,CONFIG,
                                                prefer_start_recovery=False)
            negative = [p for p in negative if not (p.assignment_index==target and
                        p.state.routes==after.routes and p.state.directions==after.directions)]
            expected = features[answer,-8:-4]
            moves = [sum(arcs) for arcs in after.arc_costs]
            lo = [min((min(all_blocks[b].start[0],all_blocks[b].end[0]) for b in r),default=math.inf) for r in after.routes]
            hi = [max((max(all_blocks[b].start[0],all_blocks[b].end[0]) for b in r),default=-math.inf) for r in after.routes]
            overlap = sum(max(0.,hi[r]+CONFIG.interference_dx-lo[r+1]) for r in (0,2))/20.
            peak = float(expected.max())
            outcome = features.new_tensor((peak,peak-float(expected.min()),
                max(p+m for p,m in zip(after.loads,moves))/2500.,(after.travel-state.travel)/45.,overlap,risk))
            positive = gnn.VNextJointAction(target,after,actions,features[answer],outcome,fine_features,fine_indices)
            plans = (*negative,positive)
            joint_steps.append((gnn.vnext_policy_inputs(space,plans,model),len(plans)-1))
        trace.append(dict(parent_id=assignment.parent_id,assignment_index=target,
                          actions=[(a.block_id,a.robot,a.position,a.direction) for a in actions]))
        state,kept = after,visible
        selected[assignment.parent_id]=assignment.pattern
        labels.remove(assignment)
    if state.routes!=routes or state.directions!=bits:
        raise ValueError(f'{name}: teacher-forcing did not restore the exact complete directed solution')
    return dict(layout_id=name,space=space,assignment=Choices.from_steps(assignment_steps),
                insertion=Choices.from_steps(insertion_steps),joint=joint_steps,
                evidence=dict(teacher=record['row']['method'],cmax_s=record['row']['cmax_s'],
                    parent_steps=len(trace),insertion_steps=len(insertion_steps),joint_steps=len(joint_steps),
                    rescued_teacher_insertions=rescued,exact_routes_and_directions=True),trace=trace)


def imitation_choice_loss(logits, choices):
    losses,correct = [],0
    for score,target in zip(logits.split(choices.sizes),choices.targets.split(choices.sizes)):
        losses.append(-(target*score.log_softmax(0)).sum())
        correct+=int(target[score.argmax()]>0)
    return torch.stack(losses).mean(),correct/len(losses)


def imitation_validation_layouts():
    # Six requested pilot cases are kept out of epoch selection.
    ordered = sorted((r for r in ppo_layouts('validation') if r[0] not in IMITATION_CASES),
                     key=lambda r:(len(r[1]),r[0]))
    return [ordered[(2*i+1)*len(ordered)//16] for i in range(8)]


def imitation_train(prepared_samples=None):
    configure_cpu();torch.manual_seed(SEED)
    saved = imitation_saved()
    if saved['training_complete']:
        print('IMITATION_TRAIN_ALREADY_COMPLETE',saved['selected_epoch'],flush=True);return saved
    if len(saved['hga_records'])!=4:
        raise ValueError('finish the four already-selected complete HGA teachers before joint imitation')
    training = dict(ppo_layouts('train'))
    teachers = imitation_best_teachers(saved)
    model = imitation_model(saved.get('last_state_dict',saved['initial_state_dict'])).train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    if 'optimizer_state_dict' in saved:
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    samples = []
    prepared_samples = {} if prepared_samples is None else prepared_samples
    preparation = time.perf_counter()
    for name,record in sorted(teachers.items()):
        sample = prepared_samples.get(name)
        if sample is None or sample['evidence']['cmax_s']!=record['row']['cmax_s']:
            sample = imitation_replay((name,training[name]),record,model)
        samples.append(sample)
        print('IMITATION_REPLAY',len(samples),'/',len(teachers),name,json.dumps(sample['evidence']),flush=True)
    saved['teacher_replay'] = {s['layout_id']:dict(**s['evidence'],trace=s['trace']) for s in samples}
    saved['settings'] = dict(epochs=6,learning_rate=3e-4,weight_decay=1e-4,assignment_loss=1.,
        insertion_loss=1.,joint_loss=.5,joint_stride=8,processing_band_fraction=.01,hidden_dim=64,heads=4,
        selection_layouts=[r[0] for r in imitation_validation_layouts()])
    saved['preparation_s']=time.perf_counter()-preparation
    imitation_save(saved)
    for epoch in range(saved.get('completed_epoch',0)+1,7):
        started = time.perf_counter()
        rng = Random(SEED+epoch)
        order = list(range(len(samples)));rng.shuffle(order)
        statistics = []
        for i in order:
            sample=samples[i];space=sample['space']
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode_static(space.graph)
            choice = sample['assignment']
            assignment_loss,assignment_accuracy = imitation_choice_loss(
                model.assignment_scores(model.embed_assignment_rows(space,embeddings,choice.indices,choice.features)),choice)
            choice = sample['insertion']
            insertion_loss,insertion_accuracy = imitation_choice_loss(
                model.insertion_logits(embeddings,choice.indices,choice.features),choice)
            joint_losses = [F.cross_entropy(gnn.imitation_joint_scores(space,embeddings,model,inputs)[None],
                torch.tensor([target])) for inputs,target in sample['joint']]
            joint_loss = torch.stack(joint_losses).mean()
            loss = assignment_loss+insertion_loss+.5*joint_loss
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
                raise ArithmeticError('nonfinite joint imitation optimization')
            optimizer.step()
            statistics.append((float(assignment_loss.detach()),float(insertion_loss.detach()),
                float(joint_loss.detach()),assignment_accuracy,insertion_accuracy,gradient))
        history = dict(epoch=epoch,runtime_s=time.perf_counter()-started,
            **dict(zip(('assignment_loss','insertion_loss','joint_loss','assignment_accuracy','insertion_accuracy','gradient_norm'),
                       [sum(v[k] for v in statistics)/len(statistics) for k in range(6)])))
        saved['history'].append(history)
        saved.update(completed_epoch=epoch,last_state_dict=copy.deepcopy(model.state_dict()),
                     optimizer_state_dict=optimizer.state_dict())
        print('IMITATION_EPOCH',json.dumps(history),flush=True)
        if epoch in (2,4,6):
            seen = pickle.loads(saved['validation_seen']) if 'validation_seen' in saved else {}
            rows = [imitation_evaluate(layout,model.eval(),method='GNN_IMITATION_SELECTION',
                                      seen=seen.setdefault(layout[0],{}))
                    for layout in imitation_validation_layouts()]
            saved['validation_seen']=pickle.dumps(seen)
            saved['validation'][epoch]=rows
            good=[r for r in rows if r['status']=='FEASIBLE']
            key=(len(good),-median(r['cmax_s'] for r in good) if good else -math.inf)
            if 'selected_key' not in saved or key>tuple(saved['selected_key']):
                saved.update(selected_key=key,selected_epoch=epoch,state_dict=copy.deepcopy(model.state_dict()))
            model.train()
        imitation_save(saved)
    saved['training_complete']=True
    saved['changed_parameter_tensors']={prefix:sum(not torch.equal(v,saved['initial_state_dict'][k])
        for k,v in saved['state_dict'].items() if k.startswith(prefix))
        for prefix in ('layer1.','layer2.','geometry_skip.','policy.','insertion_head.','joint_weights')}
    imitation_save(saved)
    return saved


def imitation_select():
    """Re-evaluate retained epochs after restoring the existing head preference."""
    saved=imitation_saved()
    if saved.get('legal_start_selection_complete'):
        return saved
    if not saved['training_complete']:
        raise ValueError('finish the fixed six training epochs before final selection')
    weights={saved['selected_epoch']:saved['state_dict'],saved['completed_epoch']:saved['last_state_dict']}
    saved['selection_before_legal_start_preference']=saved['validation']
    saved['selection_state_dicts']=weights
    saved['validation']={}
    seen=pickle.loads(saved['validation_seen'])
    best=None
    for epoch,state_dict in sorted(weights.items()):
        model=imitation_model(state_dict).eval()
        rows=[imitation_evaluate(layout,model,method='GNN_IMITATION_SELECTION',seen=seen.setdefault(layout[0],{}))
              for layout in imitation_validation_layouts()]
        saved['validation'][epoch]=rows
        good=[r for r in rows if r['status']=='FEASIBLE']
        key=(len(good),-median(r['cmax_s'] for r in good) if good else -math.inf)
        if best is None or key>best:
            best=key;saved.update(selected_epoch=epoch,selected_key=key,state_dict=copy.deepcopy(state_dict))
        print('IMITATION_FINAL_SELECTION',epoch,len(good),'/',len(rows),key[1],flush=True)
    saved['validation_seen']=pickle.dumps(seen)
    saved['legal_start_selection_complete']=True
    saved['settings'].update(final_selection_epochs=sorted(weights),legal_start_preference=True,
                              teacher_prefix_start_preference=False)
    saved['changed_parameter_tensors']={prefix:sum(not torch.equal(v,saved['initial_state_dict'][k])
        for k,v in saved['state_dict'].items() if k.startswith(prefix))
        for prefix in ('layer1.','layer2.','geometry_skip.','policy.','insertion_head.','joint_weights')}
    imitation_save(saved)
    return saved


def imitation_write(rows):
    with IMITATION_RESULTS.open('a' if IMITATION_RESULTS.exists() else 'w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=IMITATION_FIELDS)
        if stream.tell()==0:
            writer.writeheader()
        writer.writerows({key:row.get(key) for key in IMITATION_FIELDS} for row in rows)


def imitation_references():
    """Reuse the recorded PPO/classical B32 results; evaluate the raw GNN once."""
    layouts = dict(ppo_layouts('validation'))
    done = {(r['layout_id'],r['method']) for r in read_rows(IMITATION_RESULTS)}
    old = read_rows(PPO_RESULTS)
    for name in IMITATION_CASES:
        parents = layouts[name]
        space = None
        for method in ('WAG+VNS','HGA','PPO_1000','GNN_ORIGINAL'):
            if (name,method) in done:
                continue
            if method=='GNN_ORIGINAL':
                original = gnn.VNextGNN(workload_lookahead=True)
                source = torch.load(MODEL,map_location='cpu',weights_only=True)
                original.load_state_dict(source['models']['GNN_VNEXT_WORKLOAD']['state_dict'])
                row = imitation_evaluate((name,parents),original,method=method)
                row['source']='GNN_VNEXT_WORKLOAD original checkpoint; original feature/scoring semantics'
            else:
                matches = [r for r in old if r['layout_id']==name and (
                    r['stage']=='baseline' and r['method']==method or
                    method=='PPO_1000' and r['stage']=='validation' and r['size']=='1000')]
                if len(matches)!=1:
                    raise ValueError(f'{name}: expected exactly one existing {method} reference')
                source = matches[0]
                detail = json.loads(source['diagnostic'])
                row = {k:source.get(k) for k in ('status','cmax_s','runtime_s','move_s','wait_s','max_processing_s')}
                row.update(layout_id=name,parent_count=len(parents),seed=SEED,method=method,b32_calls=0,
                           source='reused gnn_ppo_scale.csv; no repeated search or B32')
                if method=='PPO_1000':
                    space = gnn.parent_assignment_space(parents,CONFIG) if space is None else space
                    plan = unpack_directed_plan(parents,detail['solution'],space)
                    row.update(beam_validation_metrics(plan))
                    row.update({f'robot_task_count_r{r}':len(route.block_ids) for r,route in enumerate(plan.solution.routes)})
                else:
                    row.update({f'processing_r{r}_s':detail.get(f'processing_r{r}_s') for r in range(4)})
                    # These historical rows did not save routes. Task counts
                    # cannot be inferred from workload sums without inventing data.
                    detail['task_count_missing_reason']='historical baseline row has processing totals but no full routes; baseline rerun prohibited'
                row['diagnostic']=json.dumps(detail,ensure_ascii=False)
                row['wait_for_graph']='[]' if source['status']=='FEASIBLE' else None
            row.update(stage='pilot_reference',parent_count=len(parents))
            if all(row.get(f'processing_r{r}_s') not in (None,'') for r in range(4)):
                row['max_processing_s']=max(float(row[f'processing_r{r}_s']) for r in range(4))
            if 'four_robot_cover_possible' not in row:
                space = gnn.parent_assignment_space(parents,CONFIG) if space is None else space
                row['four_robot_cover_possible']=space.four_robot_cover_possible
            imitation_write([row])
            print('IMITATION_REFERENCE',json.dumps({k:v for k,v in row.items() if k!='diagnostic'}),flush=True)


def imitation_pilot():
    saved = imitation_select()
    if not saved['training_complete']:
        raise ValueError('complete joint imitation before the six-case pilot')
    imitation_references()
    layouts = dict(ppo_layouts('validation'))
    model = imitation_model(saved['state_dict']).eval()
    done = {r['layout_id'] for r in read_rows(IMITATION_RESULTS) if r['method']=='GNN_IMITATION'}
    retained = {r['layout_id']:r for r in saved['pilot']}
    for name in IMITATION_CASES:
        if name in done:
            continue
        row = retained.get(name)
        if row is None:
            row = imitation_evaluate((name,layouts[name]),model)
            row.update(stage='pilot',parent_count=len(layouts[name]),source=f'full joint imitation epoch {saved["selected_epoch"]}')
            saved['pilot'].append(row)
            imitation_save(saved)
        imitation_write([row])
        print('IMITATION_PILOT',json.dumps({k:v for k,v in row.items() if k!='diagnostic'}),flush=True)
    saved['pilot_summary']=imitation_summary()
    imitation_save(saved)
    return saved


def imitation_summary():
    rows = read_rows(IMITATION_RESULTS)
    grouped = {method:[r for r in rows if r['method']==method and r['layout_id'] in IMITATION_CASES]
               for method in ('GNN_ORIGINAL','PPO_1000','GNN_IMITATION','WAG+VNS','HGA')}
    def numeric_median(values,key):
        numbers=[float(r[key]) for r in values if r.get(key) not in (None,'')]
        return median(numbers) if numbers else None
    summary = {}
    for method,values in grouped.items():
        good = [r for r in values if r['status']=='FEASIBLE']
        summary[method]=dict(cases=len(values),feasible=len(good),
            median_cmax_s=numeric_median(good,'cmax_s'),median_pmax_s=numeric_median(values,'max_processing_s'),
            median_move_s=numeric_median(values,'move_s'),median_runtime_s=numeric_median(values,'runtime_s'))
    new = grouped['GNN_IMITATION']
    complete = len(new)==6 and all(r['status']=='FEASIBLE' for r in new)
    covered = len(new)==6 and all(all(r.get(f'robot_task_count_r{i}') not in (None,'') and
                  int(r[f'robot_task_count_r{i}'])>0 for i in range(4))
                  for r in new if r['four_robot_cover_possible']=='True')
    paired = {}
    for method in ('GNN_ORIGINAL','PPO_1000','WAG+VNS','HGA'):
        lookup = {r['layout_id']:r for r in grouped[method]}
        pairs = [(r,lookup[r['layout_id']]) for r in new if r['layout_id'] in lookup]
        wins = [r['layout_id'] for r,ref in pairs if r['status']=='FEASIBLE' and
                ref['status']=='FEASIBLE' and float(r['cmax_s'])<float(ref['cmax_s'])-1e-8]
        changes=[100*(float(r['max_processing_s'])/float(ref['max_processing_s'])-1)
                 for r,ref in pairs if r.get('max_processing_s') and ref.get('max_processing_s')]
        paired[method]=dict(wins=wins,non_wins=[r['layout_id'] for r,ref in pairs if r['layout_id'] not in wins],
                           median_pmax_change_pct=median(changes) if changes else None)
    pmax = summary['GNN_IMITATION']['median_pmax_s']
    # Quantify 'clearly lower' before pilot outcomes: at least 10% lower median.
    processing = pmax is not None and all(summary[m]['median_pmax_s'] is not None and
                     pmax<=.9*summary[m]['median_pmax_s'] for m in ('GNN_ORIGINAL','PPO_1000'))
    quality = complete and all(len(paired[m]['wins'])>=4 and
        summary['GNN_IMITATION']['median_cmax_s']<summary[m]['median_cmax_s'] for m in ('WAG+VNS','HGA'))
    output = dict(methods=summary,paired=paired,all_six_feasible=complete,four_robot_coverage=covered,
                  pmax_improved_at_least_10pct=processing,cmax_conditions_met=quality,
                  proceed_to_twenty=bool(complete and covered and processing and quality))
    print('IMITATION_SUMMARY',json.dumps(output),flush=True)
    return output


DEADLOCK_RESULTS = FINAL_RESULTS/'gnn_deadlock_repair_pilot.csv'
DEADLOCK_FIELDS = ('stage','layout_id','parent_count','seed','repair_attempt','repair_round','candidate_rank',
    'origin_attempt','neighbour_layers','removed_parent_ids','status','cmax_s',
    'robot_task_count_r0','robot_task_count_r1','robot_task_count_r2','robot_task_count_r3',
    'processing_r0_s','processing_r1_s','processing_r2_s','processing_r3_s','max_processing_s',
    'same_rail_gap_01_s','same_rail_gap_23_s','move_s','wait_s','runtime_s','construct_s','b32_s','b32_calls',
    'blocked_operation_count','wait_for_graph','deadlock_blockages','diagnostic','reconstruction_failure','solution')


def deadlock_write(rows):
    with DEADLOCK_RESULTS.open('a' if DEADLOCK_RESULTS.exists() else 'w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=DEADLOCK_FIELDS)
        if stream.tell()==0:
            writer.writeheader()
        writer.writerows({k:row.get(k) for k in DEADLOCK_FIELDS} for row in rows)
        stream.flush()


def deadlock_plan(parents, snapshot, space):
    if isinstance(snapshot,str):
        snapshot=json.loads(snapshot)
    snapshot=dict(snapshot,routes=tuple(tuple(r) for r in snapshot['routes']),
                  directions=tuple(tuple(d) for d in snapshot['directions']))
    result=unpack_directed_plan(parents,snapshot,space)
    if tuple(r.block_ids for r in result.solution.routes)!=snapshot['routes']:
        raise ValueError('saved directed plan changed during canonicalization')
    return result


def deadlock_schedule(row):
    def operation(value):
        return None if value is None else Operation(**dict(value,kind=OperationKind(value['kind']),
            start=tuple(value['start']),end=tuple(value['end'])))
    evidence=[]
    for item in json.loads(row.get('deadlock_blockages') or '[]'):
        evidence.append(DeadlockBlockage(**dict(item,
            blocked_operation=operation(item['blocked_operation']),
            blocked_block_ids=tuple(item['blocked_block_ids']),blocked_parent_ids=tuple(item['blocked_parent_ids']),
            blocker_operation_ids=tuple(item['blocker_operation_ids']),
            blocker_operations=tuple(operation(x) for x in item['blocker_operations']),
            blocker_block_ids=tuple(tuple(x) for x in item['blocker_block_ids']),
            blocker_parent_ids=tuple(tuple(x) for x in item['blocker_parent_ids']))))
    diagnostics=json.loads(row.get('diagnostic') or '{}')
    return ScheduleResult(ScheduleStatus(row['status']),
        cmax=float(row['cmax_s']) if row.get('cmax_s') not in (None,'') else None,
        wait_for_graph=tuple((r,tuple(peers)) for r,peers in json.loads(row.get('wait_for_graph') or '[]')),
        diagnostics=tuple(diagnostics.get('schedule_diagnostics',())),deadlock_blockages=tuple(evidence),
        deadlock_evidence_source=diagnostics.get('evidence_source'))


def deadlock_result_row(name, result, schedule, **values):
    # Recompute from all four final directed routes, independently of the
    # forecast and the incremental construction state's cached loads/arcs.
    metrics=beam_validation_metrics(replace(result,oriented_state=None))
    loads=[metrics[f'processing_r{r}_s'] for r in range(4)]
    return dict(layout_id=name,parent_count=len(result.solution.parents),seed=SEED,
        status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,
        **{f'robot_task_count_r{r}':len(route.block_ids) for r,route in enumerate(result.solution.routes)},
        **metrics,max_processing_s=max(loads),same_rail_gap_01_s=abs(loads[0]-loads[1]),
        same_rail_gap_23_s=abs(loads[2]-loads[3]),
        wait_s=sum(op.duration for op in schedule.operations if op.kind is OperationKind.WAIT),
        blocked_operation_count=len(schedule.deadlock_blockages),
        wait_for_graph=json.dumps(schedule.wait_for_graph),
        deadlock_blockages=json.dumps([asdict(item) for item in schedule.deadlock_blockages]),
        diagnostic=json.dumps(dict(schedule_diagnostics=schedule.diagnostics,
            evidence_source=schedule.deadlock_evidence_source,baseline_deadlock=schedule.baseline_deadlock,
            recovery_rollouts=schedule.recovery_rollouts,rollout_budget=schedule.rollout_budget,
            recovery_exhausted=schedule.recovery_exhausted,frontier_exhausted=schedule.frontier_exhausted)),
        solution=json.dumps(pack_directed_plan(result)),**values)


def deadlock_reconstruct(current, partial, model, space, embeddings):
    """Existing joint GNN continuation, retaining all untouched directed tasks."""
    pending=set(partial.removed_parent_ids)
    selected={p.parent_id:p for p in current.solution.patterns if p.parent_id not in pending}
    blocks={b.block_id:b for pattern in selected.values() for b in space.children[pattern]}
    old_bits={b:d for route,bits in zip(current.solution.routes,current.directions)
              for b,d in zip(route.block_ids,bits)}
    bits=tuple(tuple(old_bits[b] for b in route) for route in partial.routes)
    state=lns.oriented_route_state(partial.routes,bits,blocks,CONFIG)
    rebuilt=gnn.construct_vnext_gnn(current.solution.parents,model,CONFIG,state=state,selected=selected,
                                  space=space,embeddings=embeddings,beam_width=2)
    for candidate in rebuilt.candidates or ((rebuilt,) if rebuilt.solution is not None else ()):
        routes=tuple(r.block_ids for r in candidate.solution.routes)
        for robot,route in enumerate(partial.routes):
            kept=set(route)
            if tuple(b for b in routes[robot] if b in kept)!=route:
                raise ValueError('local GNN changed untouched route order or ownership')
            actual=dict(zip(routes[robot],candidate.directions[robot]))
            if any(actual[b]!=old_bits[b] for b in route):
                raise ValueError('local GNN changed an untouched direction')
        if len(candidate.solution.patterns)!=len(current.solution.parents):
            raise ValueError('local GNN did not restore every pending parent')
        if space.four_robot_cover_possible and not all(routes):
            raise ValueError('local GNN lost legal four-robot coverage')
    return rebuilt


def deadlock_initial(name, parents, original, space):
    result=deadlock_plan(parents,json.loads(original['diagnostic'])['solution'],space)
    if original['status']=='FEASIBLE':
        # Reuse 0067 verbatim; no initial reconstruction and no repeated B32.
        row={k:original.get(k) for k in DEADLOCK_FIELDS}
        row.update(stage='initial',repair_attempt=0,repair_round=0,candidate_rank=0,seed=SEED,
            b32_calls=0,runtime_s=0.,construct_s=0.,b32_s=0.,blocked_operation_count=0,
            deadlock_blockages='[]',solution=json.dumps(pack_directed_plan(result)),
            diagnostic=json.dumps(dict(reused='gnn_imitation_balance_pilot.csv',schedule_diagnostics=[])))
        row['same_rail_gap_01_s']=abs(float(row['processing_r0_s'])-float(row['processing_r1_s']))
        row['same_rail_gap_23_s']=abs(float(row['processing_r2_s'])-float(row['processing_r3_s']))
        return row
    started=time.perf_counter()
    schedule=b32(result)
    elapsed=time.perf_counter()-started
    return deadlock_result_row(name,result,schedule,stage='initial',repair_attempt=0,repair_round=0,
        candidate_rank=0,construct_s=0.,b32_s=elapsed,runtime_s=elapsed,b32_calls=1)


def deadlock_repair_case(name, parents, original, model, *, resume_unresolved=False):
    started=time.perf_counter()
    rows=[r for r in read_rows(DEADLOCK_RESULTS) if r['layout_id']==name]
    finals=[r for r in rows if r['stage']=='final']
    if finals and (not resume_unresolved or finals[-1]['status']=='FEASIBLE' or
                   sum(r['stage']=='candidate' for r in rows)>=12):
        return finals[-1]
    space=gnn.parent_assignment_space(parents,CONFIG)
    if not any(r['stage']=='initial' for r in rows):
        row=deadlock_initial(name,parents,original,space)
        deadlock_write([row]);rows.append(row)
        print('DEADLOCK_INITIAL',name,row['status'],'blocked',row['blocked_operation_count'],flush=True)
    evaluated=[r for r in rows if r['stage'] in ('initial','candidate')]
    seen={plan_key(plan.solution,plan.directions) for plan in
          (deadlock_plan(parents,r['solution'],space) for r in evaluated)}
    fresh=[r for r in evaluated if r['stage']=='candidate']
    count=len(fresh)
    feasible=[r for r in evaluated if r['status']=='FEASIBLE']
    if feasible:
        best=min(feasible,key=lambda r:float(r['cmax_s']))
    else:
        deadlocks=[r for r in evaluated if r['status']=='DEADLOCK']
        if not deadlocks:
            best=evaluated[-1]
        else:
            latest=max(int(r.get('repair_round') or 0) for r in deadlocks)
            best=min((r for r in deadlocks if int(r.get('repair_round') or 0)==latest),
                     key=lambda r:(int(r['blocked_operation_count']),float(r['max_processing_s']),float(r['move_s'])))
    embeddings=model.encode_static(space.graph)
    round_index=max((int(r.get('repair_round') or 0) for r in rows),default=0)
    layers=1
    stop='already feasible' if best['status']=='FEASIBLE' else 'new complete candidate budget exhausted'
    while best['status']=='DEADLOCK' and count<12:
        current=deadlock_plan(parents,best['solution'],space)
        schedule=deadlock_schedule(best)
        partial=lns.select_deadlock_blocker_parents(current.solution,schedule,CONFIG,neighbour_layers=layers)
        if partial is None:
            stop='no mapped direct blocker parents';break
        round_index+=1
        construction=time.perf_counter()
        rebuilt=deadlock_reconstruct(current,partial,model,space,embeddings)
        construction=time.perf_counter()-construction
        candidates=(rebuilt.candidates or ((rebuilt,) if rebuilt.solution is not None else ()))[:2]
        next_deadlocks=[];new_count=0
        for rank,candidate in enumerate(candidates):
            key=plan_key(candidate.solution,candidate.directions)
            if key in seen:
                continue
            seen.add(key)
            if count>=12:
                break
            count+=1;new_count+=1
            evaluation=time.perf_counter()
            outcome=b32(candidate)
            evaluation=time.perf_counter()-evaluation
            row=deadlock_result_row(name,candidate,outcome,stage='candidate',repair_attempt=count,
                repair_round=round_index,candidate_rank=rank,origin_attempt=int(best['repair_attempt']),
                neighbour_layers=layers,removed_parent_ids=json.dumps(partial.removed_parent_ids),
                construct_s=construction if new_count==1 else 0.,b32_s=evaluation,
                runtime_s=(construction if new_count==1 else 0.)+evaluation,b32_calls=1)
            deadlock_write([row]);rows.append(row)
            print('DEADLOCK_CANDIDATE',name,count,outcome.status.value,'blocked',row['blocked_operation_count'],
                  'Pmax',row['max_processing_s'],'removed',len(partial.removed_parent_ids),flush=True)
            if outcome.feasible:
                best=row;stop='B32 feasible';break
            if outcome.status is ScheduleStatus.DEADLOCK:
                next_deadlocks.append(row)
        if best['status']=='FEASIBLE':
            break
        if next_deadlocks:
            best=min(next_deadlocks,key=lambda r:(r['blocked_operation_count'],r['max_processing_s'],r['move_s']))
            layers=1
            continue
        reason=(rebuilt.rejection_reason or ('only INFEASIBLE/NUMERIC_FAILURE candidates' if new_count else
                                            'all complete candidates duplicate previous directed plans'))
        deadlock_write([dict(stage='failure',layout_id=name,seed=SEED,repair_attempt=count,repair_round=round_index,
            origin_attempt=best['repair_attempt'],neighbour_layers=layers,removed_parent_ids=json.dumps(partial.removed_parent_ids),
            b32_calls=0,construct_s=construction if not new_count else 0.,reconstruction_failure=reason)])
        if layers==1:
            layers=2
        else:
            stop=reason;break
    final=dict(best,stage='final',origin_attempt=best['repair_attempt'],repair_attempt=count,b32_calls=0,
               runtime_s=time.perf_counter()-started,reconstruction_failure=stop)
    deadlock_write([final])
    print('DEADLOCK_FINAL',json.dumps({k:v for k,v in final.items() if k not in
          ('solution','diagnostic','deadlock_blockages')}),flush=True)
    return final


@torch.no_grad()
def imitation_deadlock_pilot(*, resume_unresolved=False):
    configure_cpu()
    # Open the sole output before expensive evaluations; a denied file write
    # must not consume an unrecorded B32 call or rerun the old initial GNN.
    deadlock_write([])
    saved=torch.load(IMITATION_MODEL,map_location='cpu',weights_only=True)
    model=imitation_model(saved['state_dict'],balance=True).eval()
    source={r['layout_id']:r for r in read_rows(IMITATION_RESULTS) if r['method']=='GNN_IMITATION'}
    layouts=dict(ppo_layouts('validation'))
    final=[deadlock_repair_case(name,layouts[name],source[name],model,resume_unresolved=resume_unresolved)
           for name in IMITATION_CASES]
    print('DEADLOCK_PILOT_SUMMARY',json.dumps(dict(feasible=sum(r['status']=='FEASIBLE' for r in final),
        cases=len(final),new_complete_candidates=sum(int(r['repair_attempt']) for r in final))),flush=True)
    return final


ROUTEAWARE_RESULTS = FINAL_RESULTS/'gnn_routeaware_pilot.csv'
ROUTEAWARE_ROUNDS = 6
ROUTEAWARE_FIELDS = ('stage','layout_id','parent_count','seed','method','round','candidate_rank','origin_round',
    'removed_parent_ids','status','cmax_s',
    *(f'{kind}_r{r}_s' for kind in ('processing','move','wait','completion') for r in range(4)),
    *(f'robot_task_count_r{r}' for r in range(4)),
    'max_processing_s','processing_spread_s','move_s','wait_s','partial_wait_s','metrics_missing_reason',
    'runtime_s','construct_s','b32_s','b32_calls','cache_hits','blocked_operation_count',
    'wait_for_graph','deadlock_blockages','solution','schedule','wait_blockers','feedback','diagnostic')


def routeaware_write(rows, *, path=None):
    path=ROUTEAWARE_RESULTS if path is None else path
    with path.open('a' if path.exists() else 'w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=ROUTEAWARE_FIELDS)
        if stream.tell()==0:
            writer.writeheader()
        writer.writerows({k:row.get(k) for k in ROUTEAWARE_FIELDS} for row in rows)
        stream.flush()


def routeaware_operation(value):
    return Operation(**dict(value,kind=OperationKind(value['kind']),
                            start=tuple(value['start']),end=tuple(value['end'])))


def routeaware_schedule(row):
    schedule=deadlock_schedule(row)
    if not row.get('schedule'):
        return schedule
    data=json.loads(row['schedule'])
    operations=tuple(Operation(op['id'],op['robot'],OperationKind(op['kind']),op['start_time'],op['end_time'],
                               tuple(op['start']),tuple(op['end']),op['sequence'],op['block'])
                     for op in data['operations'])
    return replace(schedule,operations=operations,robot_completion=tuple(data['completion']),
                   directions=tuple(tuple(d) for d in data['directions']),diagnostics=tuple(data['diagnostics']))


def routeaware_wait_blockers(row):
    return tuple(ScheduleWaitBlockers(routeaware_operation(event['waiting']),
        routeaware_operation(event['delayed_operation']),event['delayed_block_id'],
        tuple((op,block,tuple(tuple(interval) for interval in intervals))
              for op,block,intervals in event['blocker_intervals']))
        for event in json.loads(row.get('wait_blockers') or '[]'))


def routeaware_metrics(result,schedule,*,historical=None):
    """Full P/M from B32 templates; full W/C only from a feasible timeline."""
    blocks=block_map(result.solution,CONFIG,scope=FORMAL_SCOPE_V2)
    processing=[0.]*4;move=[0.]*4;wait=[0.]*4
    for r,route in enumerate(result.solution.routes):
        robot_route=RobotRoute(r,tuple(blocks[b] for b in route.block_ids),result.directions[r])
        for op in build_operation_templates(robot_route,CONFIG):
            (move if op.kind is OperationKind.MOVE else processing)[r]+=op.duration
    for op in schedule.operations:
        if op.kind is OperationKind.WAIT:
            wait[op.robot_id]+=op.duration
    metrics=dict(**{f'processing_r{r}_s':p for r,p in enumerate(processing)},
        **{f'move_r{r}_s':m for r,m in enumerate(move)},
        **{f'robot_task_count_r{r}':len(route.block_ids) for r,route in enumerate(result.solution.routes)},
        max_processing_s=max(processing),processing_spread_s=max(processing)-min(processing),move_s=sum(move))
    if schedule.feasible and schedule.operations:
        metrics.update(**{f'wait_r{r}_s':w for r,w in enumerate(wait)},
                       **{f'completion_r{r}_s':c for r,c in enumerate(schedule.robot_completion)},wait_s=sum(wait))
        # Ordinary data consistency check, never another scheduler or Cmax model.
        for r in range(4):
            if not math.isclose(processing[r]+move[r]+wait[r],schedule.robot_completion[r],abs_tol=1e-6):
                raise ValueError('B32 timeline does not match complete P/M/W')
    elif schedule.feasible:
        metrics.update(wait_s=(historical or {}).get('wait_s'),
                       metrics_missing_reason='historical full timeline missing; per-robot WAIT/completion unavailable; no repeated B32')
    else:
        metrics.update(partial_wait_s=sum(wait) if schedule.operations else
                       (historical or {}).get('partial_wait_s',(historical or {}).get('wait_s')),
                       metrics_missing_reason='non-feasible timeline is partial; full WAIT/completion unavailable')
    return metrics


def routeaware_improves(candidate,metrics,incumbent,incumbent_metrics):
    if not candidate.feasible:
        return False
    if incumbent is None or not incumbent.feasible:
        return True
    if candidate.cmax!=incumbent.cmax:
        return candidate.cmax<incumbent.cmax
    if all(m.get(f'wait_r{r}_s') not in (None,'') for m in (metrics,incumbent_metrics) for r in range(4)):
        first=max(float(metrics[f'wait_r{r}_s']) for r in range(4))
        second=max(float(incumbent_metrics[f'wait_r{r}_s']) for r in range(4))
        if first!=second:
            return first<second
    return (float(metrics['move_s']),float(metrics['processing_spread_s'])) < (
            float(incumbent_metrics['move_s']),float(incumbent_metrics['processing_spread_s']))


def routeaware_history(name,parents,space,*,extra_rows=()):
    """Use existing (patterns,routes,directions) keys across both experiments."""
    seen={};blockers={};records={}
    sources=[]
    for path in (PPO_RESULTS,IMITATION_RESULTS):
        for row in read_rows(path):
            if row.get('layout_id')!=name:
                continue
            diagnostic=row.get('diagnostic') or '{}'
            if not diagnostic.lstrip().startswith('{'):
                continue  # Older numeric-failure notes carry no saved plan.
            detail=json.loads(diagnostic)
            if detail.get('solution'):
                sources.append(dict(row,solution=json.dumps(detail['solution']),
                                    diagnostic=json.dumps(dict(schedule_diagnostics=detail.get('schedule_diagnostic',())))))
    sources.extend(r for r in read_rows(DEADLOCK_RESULTS) if r['layout_id']==name and r.get('solution'))
    sources.extend(r for r in read_rows(ROUTEAWARE_RESULTS) if r['layout_id']==name and r.get('solution'))
    sources.extend(r for r in extra_rows if r['layout_id']==name and r.get('solution'))
    for row in sources:
        if row.get('status') not in {status.value for status in ScheduleStatus}:
            continue
        result=deadlock_plan(parents,row['solution'],space)
        key=plan_key(result.solution,result.directions)
        schedule=routeaware_schedule(row)
        # Do not replace saved full timelines with minimal historical records.
        if key in seen and seen[key].operations and not schedule.operations:
            continue
        seen[key]=schedule;records[key]=row
        blockers[key]=routeaware_wait_blockers(row)
    return seen,blockers,records


def routeaware_result_row(name,method,result,schedule,events,*,historical=None,**values):
    return dict(layout_id=name,parent_count=len(result.solution.parents),seed=SEED,method=method,
        status=schedule.status.value,cmax_s=schedule.cmax if schedule.feasible else None,
        **routeaware_metrics(result,schedule,historical=historical),
        blocked_operation_count=len(schedule.deadlock_blockages),
        wait_for_graph=json.dumps(schedule.wait_for_graph),
        deadlock_blockages=json.dumps([asdict(item) for item in schedule.deadlock_blockages]),
        solution=json.dumps(pack_directed_plan(result)),
        schedule=schedule.canonical_json() if schedule.operations else None,
        wait_blockers=json.dumps([asdict(event) for event in events]),
        diagnostic=json.dumps(dict(schedule_diagnostics=schedule.diagnostics,
            evidence_source=schedule.deadlock_evidence_source),ensure_ascii=False),**values)


@torch.no_grad()
def routeaware_case(name,parents,model,method,*,result_path=None,load_first_lambda=None,
                    round_limit=ROUTEAWARE_ROUNDS):
    path=ROUTEAWARE_RESULTS if result_path is None else result_path
    all_rows=read_rows(path)
    previous=[r for r in all_rows if r['layout_id']==name and r['method']==method]
    if any(r['stage']=='final' for r in previous):
        return next(r for r in reversed(previous) if r['stage']=='final')
    started=time.perf_counter()
    space=gnn.parent_assignment_space(parents,CONFIG)
    seen,blockers,records=routeaware_history(name,parents,space,extra_rows=all_rows if result_path is not None else ())
    embeddings=model.encode_static(space.graph)
    model.route_aware=method=='GNN-RouteAware'
    model.load_first_lambda=load_first_lambda
    write=lambda rows:routeaware_write(rows,path=path)
    initial_rows=[r for r in previous if r['stage']=='initial']
    if initial_rows:
        initial_row=initial_rows[-1]
        initial=deadlock_plan(parents,initial_row['solution'],space)
    else:
        construction=time.perf_counter()
        screens=[r for r in previous if r['stage']=='screen' and r.get('solution')]
        if screens:
            initial=deadlock_plan(parents,screens[-1]['solution'],space)
            construction=float(screens[-1]['construct_s'])
        else:
            initial=gnn.construct_vnext_gnn(parents,model,CONFIG,space=space,embeddings=embeddings)
            construction=time.perf_counter()-construction
        if initial.solution is None:
            row=dict(stage='final',layout_id=name,method=method,seed=SEED,status='INFEASIBLE',
                     b32_calls=0,cache_hits=0,runtime_s=time.perf_counter()-started,diagnostic=initial.rejection_reason)
            write([row]);return row
        if method=='GNN-Original':
            old=next(r for r in read_rows(IMITATION_RESULTS) if r['layout_id']==name and r['method']=='GNN_IMITATION')
            old_plan=deadlock_plan(parents,json.loads(old['diagnostic'])['solution'],space)
            if plan_key(initial.solution,initial.directions)!=plan_key(old_plan.solution,old_plan.directions):
                raise ValueError(f'{name}: original construction did not reproduce archived full plan')
        key=plan_key(initial.solution,initial.directions)
        evaluation=time.perf_counter()
        schedule,called=evaluate_once(initial,seen,blocker_cache=blockers)
        elapsed=time.perf_counter()-evaluation
        initial_row=routeaware_result_row(name,method,initial,schedule,blockers.get(key,()),
            historical=records.get(key),stage='initial',round=0,candidate_rank=0,b32_calls=int(called),
            cache_hits=int(not called),construct_s=construction,b32_s=elapsed if called else 0.,
            runtime_s=time.perf_counter()-started+(construction if screens else 0.))
        detail=json.loads(initial_row['diagnostic'])
        detail.update(checkpoint=str(IMITATION_MODEL.relative_to(ROOT)),seed=SEED,
            scientific_parameters=CONFIG.scientific_mapping(),scope='FORMAL_SCOPE_V2',
            reconstruction_rounds=round_limit,beam_width=2,initial_width=1,
            load_first_lambda=load_first_lambda,maximum_reconstruction_attempts=2*round_limit,
            original_full_plan_reproduced=method=='GNN-Original')
        initial_row['diagnostic']=json.dumps(detail,ensure_ascii=False)
        write([initial_row]);previous.append(initial_row);records[key]=initial_row
        print('ROUTEAWARE_INITIAL',name,method,initial_row['status'],initial_row['cmax_s'],'new_B32',int(called),flush=True)
    initial_key=plan_key(initial.solution,initial.directions)
    current=initial;current_row=initial_row;current_schedule=seen[initial_key]
    best=initial if current_schedule.feasible else None
    best_row=initial_row if best else None
    best_schedule=current_schedule if best else None
    rounds=sorted((r for r in previous if r['stage']=='round'),key=lambda r:int(r['round']))
    if rounds:
        current_row=rounds[-1]
        current=deadlock_plan(parents,current_row['solution'],space)
        current_schedule=seen[plan_key(current.solution,current.directions)]
        feasible=[r for r in previous if r.get('status')=='FEASIBLE' and r.get('solution')]
        for row in feasible:
            result=deadlock_plan(parents,row['solution'],space)
            schedule=seen[plan_key(result.solution,result.directions)]
            if routeaware_improves(schedule,row,best_schedule,best_row):
                best,best_row,best_schedule=result,row,schedule
    first_round=int(rounds[-1]['round'])+1 if rounds else 1
    for round_index in range(first_round,round_limit+1):
        round_started=time.perf_counter()
        origin_round=current_row['round']
        key=plan_key(current.solution,current.directions)
        q=min(len(parents),max(6,min(12,round(.08*len(parents))+((round_index-1)//5)%3)))
        if current_schedule.feasible:
            partial,feedback=lns.select_routeaware_bottleneck(current.solution,current.directions,current_schedule,
                blockers.get(key,()),CONFIG,
                q=q if load_first_lambda is not None else min(len(parents),max(4,round(.1*len(parents)))),
                seed=SEED+round_index,assignment_space=space,load_first=load_first_lambda is not None)
        elif current_schedule.status is ScheduleStatus.DEADLOCK:
            partial=lns.select_deadlock_blocker_parents(current.solution,current_schedule,CONFIG,
                **(dict(neighbour_layers=1,max_parents=max(4,q),seed=SEED+round_index)
                   if load_first_lambda is not None else dict(neighbour_layers=1 if round_index%2 else 2)))
            feedback=dict(source='bounded structured B32 DEADLOCK blockers' if load_first_lambda is not None else
                          'structured B32 DEADLOCK blockers',max_parents=q if load_first_lambda is not None else None)
        else:
            partial=None;feedback=dict(source='no valid feasible/deadlock origin')
        construction=time.perf_counter()
        rebuilt=deadlock_reconstruct(current,partial,model,space,embeddings) if partial is not None else None
        construction=time.perf_counter()-construction
        candidates=() if rebuilt is None else (rebuilt.candidates or ((rebuilt,) if rebuilt.solution is not None else ()))[:2]
        outcomes=[]
        for rank,candidate in enumerate(candidates):
            candidate_key=plan_key(candidate.solution,candidate.directions)
            evaluation=time.perf_counter()
            schedule,called=evaluate_once(candidate,seen,blocker_cache=blockers)
            elapsed=time.perf_counter()-evaluation
            row=routeaware_result_row(name,method,candidate,schedule,blockers.get(candidate_key,()),
                historical=records.get(candidate_key),stage='candidate',round=round_index,candidate_rank=rank,
                origin_round=origin_round,removed_parent_ids=json.dumps(partial.removed_parent_ids),
                b32_calls=int(called),cache_hits=int(not called),b32_s=elapsed if called else 0.,
                runtime_s=elapsed,feedback=json.dumps(feedback))
            # A resumed unfinished round reuses its saved evaluation and row.
            saved=next((r for r in previous if r['stage']=='candidate' and int(r['round'])==round_index
                        and int(r['candidate_rank'])==rank),None)
            if saved is None:
                write([row]);previous.append(row)
            else:
                row=saved
            records[candidate_key]=row
            outcomes.append((candidate,schedule,row))
            if routeaware_improves(schedule,row,best_schedule,best_row):
                best,best_schedule,best_row=candidate,schedule,row
            print('ROUTEAWARE_CANDIDATE',name,method,round_index,rank,schedule.status.value,schedule.cmax,
                  'new_B32',int(called),'removed',len(partial.removed_parent_ids),flush=True)
        if best is not None:
            current,current_schedule,current_row=best,best_schedule,best_row
        else:
            deadlocks=[entry for entry in outcomes if entry[1].status is ScheduleStatus.DEADLOCK]
            if deadlocks:
                current,current_schedule,current_row=min(deadlocks,key=lambda entry:(
                    len(entry[1].deadlock_blockages),float(entry[2]['move_s']),float(entry[2]['processing_spread_s'])))
        row=dict(current_row,stage='round',round=round_index,origin_round=origin_round,b32_calls=0,cache_hits=0,
                 construct_s=construction,b32_s=0.,runtime_s=time.perf_counter()-round_started,
                 removed_parent_ids=json.dumps(partial.removed_parent_ids) if partial else '[]',
                 feedback=json.dumps(feedback))
        if not candidates:
            row['diagnostic']=json.dumps(dict(rejection=rebuilt.rejection_reason if rebuilt else 'no mapped feedback'))
        if load_first_lambda is not None:
            # The initial/candidate row already holds the complete timeline.
            # Retain the ordinary full-plan tuple for resume, without copying
            # the same operations into every round checkpoint.
            row.update(schedule='',wait_blockers='')
        write([row]);previous.append(row)
    row=dict(best_row if best is not None else current_row,stage='final',round=round_limit,
        runtime_s=sum(float(r.get('runtime_s') or 0) for r in previous if r['stage'] in ('initial','round')),
        construct_s=sum(float(r.get('construct_s') or 0) for r in previous if r['stage'] in ('initial','round')),
        b32_s=sum(float(r.get('b32_s') or 0) for r in previous if r['stage'] in ('initial','candidate')),
        b32_calls=sum(int(r.get('b32_calls') or 0) for r in previous if r['stage'] in ('initial','candidate')),
        cache_hits=sum(int(r.get('cache_hits') or 0) for r in previous if r['stage'] in ('initial','candidate')))
    write([row])
    print('ROUTEAWARE_FINAL',json.dumps({k:row.get(k) for k in ('layout_id','method','status','cmax_s','move_s',
        'wait_s','runtime_s','b32_calls','cache_hits')}),flush=True)
    return row


def imitation_routeaware_pilot():
    configure_cpu()
    # The only new artifact was explicitly approved by the user. Opening it
    # before B32 also makes a denied destination fail without losing a run.
    routeaware_write([])
    done={(r['layout_id'],r['method']) for r in read_rows(ROUTEAWARE_RESULTS) if r['stage']=='reference'}
    for old in read_rows(IMITATION_RESULTS):
        if old['layout_id'] not in IMITATION_CASES or old['method'] not in ('WAG+VNS','HGA'):
            continue
        if (old['layout_id'],old['method']) in done:
            continue
        detail=json.loads(old['diagnostic'])
        if int(old['seed'])!=SEED or detail['iterations']!=BASELINE_ITERATIONS[old['method']]:
            raise ValueError('historical classical reference has a different seed/budget')
        loads=[float(old[f'processing_r{r}_s']) for r in range(4)]
        row=dict(old,stage='reference',b32_calls=0,cache_hits=1,processing_spread_s=max(loads)-min(loads),
            metrics_missing_reason='historical baseline has no saved full routes/timeline; per-robot MOVE/WAIT/completion and task counts unavailable',
            diagnostic=json.dumps(dict(reference=detail,scientific_parameters=CONFIG.scientific_mapping(),
                scope='FORMAL_SCOPE_V2',budget_note='historical WAG500/HGA200000; different from GNN 6x2; descriptive comparison only'),
                ensure_ascii=False))
        routeaware_write([row])
    torch.manual_seed(SEED)
    saved=torch.load(IMITATION_MODEL,map_location='cpu',weights_only=True)
    model=imitation_model(saved['state_dict']).eval()
    layouts=dict(ppo_layouts('validation'))
    finals=[]
    for method in ('GNN-Original','GNN-RouteAware'):
        for name in IMITATION_CASES:
            finals.append(routeaware_case(name,layouts[name],model,method))
    print('ROUTEAWARE_SUMMARY',json.dumps({method:dict(
        feasible=sum(r['status']=='FEASIBLE' for r in finals if r['method']==method),
        b32_calls=sum(int(r['b32_calls']) for r in finals if r['method']==method))
        for method in ('GNN-Original','GNN-RouteAware')}),flush=True)
    return finals


LOADFIRST_RESULTS=FINAL_RESULTS/'gnn_loadfirst_pilot.csv'
LOADFIRST_LAMBDAS=(.25,.5,1.)
LOADFIRST_ROUNDS=50  # Two full proposal slots per round: at most 100 repairs.


def loadfirst_method(penalty):
    return f'LoadFirst-GNN-lambda{penalty:g}'


def loadfirst_references():
    existing={(r['layout_id'],r['method']) for r in read_rows(LOADFIRST_RESULTS) if r['stage']=='reference'}
    for old in read_rows(ROUTEAWARE_RESULTS):
        if old['stage'] not in ('reference','final') or old['layout_id'] not in IMITATION_CASES:
            continue
        if (old['layout_id'],old['method']) in existing:
            continue
        row=dict(old,stage='reference',b32_calls=0,cache_hits=1)
        detail=json.loads(row.get('diagnostic') or '{}')
        detail['reused_from']='gnn_routeaware_pilot.csv; historical budget, no repeated B32'
        row['diagnostic']=json.dumps(detail,ensure_ascii=False)
        routeaware_write([row],path=LOADFIRST_RESULTS)
        existing.add((old['layout_id'],old['method']))


@torch.no_grad()
def loadfirst_screen(model,layouts):
    """Measure P/M before spending the B32 repair budget; save the initial plan."""
    rows=read_rows(LOADFIRST_RESULTS)
    done={(r['layout_id'],r['method']) for r in rows if r['stage']=='screen'}
    refs={(r['layout_id'],r['method']):r for r in rows if r['stage']=='reference'}
    for penalty in LOADFIRST_LAMBDAS:
        model.load_first_lambda=penalty;model.route_aware=False
        for name in IMITATION_CASES:
            method=loadfirst_method(penalty)
            if (name,method) in done:
                continue
            started=time.perf_counter();violations=[];steps=[]
            def observe(state,selected,plans,scores):
                processing=torch.stack([p.assignment_features[-8:-4]*2500. for p in plans])
                moves=processing.new_tensor([[sum(arcs) for arcs in p.state.arc_costs] for p in plans])
                values=gnn.loadfirst_proxy(processing,moves,penalty)
                gap=float(values.max()-values.min())
                tolerance=max(1.,.001*abs(float(values.min())))
                if gap>tolerance+1e-3:
                    violations.append((len(selected),gap,tolerance))
                steps.append(dict(selected_parent_count=len(selected),candidate_count=len(plans),
                    best_j=float(values.min()),chosen_j=float(values[int(scores.argmax())])))
            result=gnn.construct_vnext_gnn(layouts[name],model,CONFIG,step_observer=observe)
            elapsed=time.perf_counter()-started
            if result.solution is None:
                row=dict(stage='screen',layout_id=name,seed=SEED,method=method,status='CONSTRUCTION_FAILED',
                    b32_calls=0,construct_s=elapsed,diagnostic=json.dumps(dict(rejection=result.rejection_reason)))
            else:
                state=lns.oriented_route_state(tuple(r.block_ids for r in result.solution.routes),result.directions,
                    block_map(result.solution,CONFIG,scope=FORMAL_SCOPE_V2),CONFIG)
                spread=max(state.loads)-min(state.loads)
                row=dict(stage='screen',layout_id=name,parent_count=len(layouts[name]),seed=SEED,method=method,
                    status='UNSCHEDULED',b32_calls=0,construct_s=elapsed,runtime_s=elapsed,
                    **{f'processing_r{r}_s':p for r,p in enumerate(state.loads)},
                    **{f'move_r{r}_s':sum(arcs) for r,arcs in enumerate(state.arc_costs)},
                    **{f'robot_task_count_r{r}':len(route) for r,route in enumerate(state.routes)},
                    max_processing_s=max(state.loads),processing_spread_s=spread,move_s=state.travel,
                    solution=json.dumps(pack_directed_plan(result)),
                    metrics_missing_reason='screen only: B32 has not been called; WAIT/C unknown',
                    diagnostic=json.dumps(dict(load_first_lambda=penalty,steps=steps,j_band_violations=violations,
                        spread_change_vs_routeaware=spread-float(refs[name,'GNN-RouteAware']['processing_spread_s']),
                        move_change_vs_original=state.travel-float(refs[name,'GNN-Original']['move_s']),
                        four_nonempty_routes=all(state.routes))))
            routeaware_write([row],path=LOADFIRST_RESULTS)
            print('LOADFIRST_SCREEN',name,penalty,row['status'],'Pmax',row.get('max_processing_s'),
                'spread',row.get('processing_spread_s'),'MOVE',row.get('move_s'),'J_violations',len(violations),flush=True)
    return [r for r in read_rows(LOADFIRST_RESULTS) if r['stage']=='screen']


def imitation_loadfirst_pilot(*,screen_only=False):
    configure_cpu();torch.manual_seed(SEED)
    routeaware_write([],path=LOADFIRST_RESULTS)
    loadfirst_references()
    model=imitation_model(torch.load(IMITATION_MODEL,map_location='cpu',weights_only=True)['state_dict']).eval()
    layouts=dict(ppo_layouts('validation'))
    screens=loadfirst_screen(model,layouts)
    if screen_only:
        return screens
    finals=[]
    for penalty in LOADFIRST_LAMBDAS:
        for name in IMITATION_CASES:
            finals.append(routeaware_case(name,layouts[name],model,loadfirst_method(penalty),
                result_path=LOADFIRST_RESULTS,load_first_lambda=penalty,round_limit=LOADFIRST_ROUNDS))
    summary={}
    for penalty in LOADFIRST_LAMBDAS:
        good=[r for r in finals if r['method']==loadfirst_method(penalty) and r['status']=='FEASIBLE']
        summary[penalty]=dict(feasible=len(good),mean_feasible_cmax_s=sum(float(r['cmax_s']) for r in good)/len(good)
            if good else None)
    selected=min(LOADFIRST_LAMBDAS,key=lambda p:(-summary[p]['feasible'],
        summary[p]['mean_feasible_cmax_s'] if summary[p]['mean_feasible_cmax_s'] is not None else math.inf,p))
    if not any(r['stage']=='selection' for r in read_rows(LOADFIRST_RESULTS)):
        routeaware_write([dict(stage='selection',layout_id='',seed=SEED,method='LoadFirst-GNN',b32_calls=0,
            diagnostic=json.dumps(dict(selected_lambda=selected,summary=summary,
                rule='maximum feasible cases, then minimum mean feasible Cmax, then smaller lambda')))],path=LOADFIRST_RESULTS)
        routeaware_write([dict(r,stage='selected_final') for r in finals if r['method']==loadfirst_method(selected)],
            path=LOADFIRST_RESULTS)
    print('LOADFIRST_SELECTION',json.dumps(dict(selected_lambda=selected,summary=summary)),flush=True)
    return finals


PRIORITY_RESULTS=FINAL_RESULTS/'gnn_priority_vnd_pilot.csv'
PRIORITY_EVALUATOR='NATIVE_PRIORITY_FOUR_ROBOTS_EXPLICIT_DIRECTIONS'
PRIORITY_FIELDS=('stage','layout_id','seed','method','lambda','evaluator','source_b32_status','source_b32_s',
    'status','cmax_s',*(f'{kind}_r{r}_s' for kind in ('processing','move','wait','completion') for r in range(4)),
    *(f'robot_task_count_r{r}' for r in range(4)),'max_processing_s','processing_spread_s','move_s','wait_s',
    'runtime_s','evaluation_s','native_calls','cache_hits','candidate_index','pass_index','move_id',
    'robots','parent_ids','before_segments','after_segments','accepted','initial_status',
    'initial_pmax_s','initial_spread_s','pmax_limit_s','spread_limit_s','initial_move_s',
    'rejected_pairs','route_cycle_s','progress','solution','schedule','diagnostic')


def priority_write(rows):
    with PRIORITY_RESULTS.open('a' if PRIORITY_RESULTS.exists() else 'w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=PRIORITY_FIELDS)
        if stream.tell()==0:writer.writeheader()
        writer.writerows({k:r.get(k) for k in PRIORITY_FIELDS} for r in rows)
        stream.flush()


def priority_saved_schedule(row):
    data=json.loads(row['schedule'])
    operations=tuple(Operation(x['id'],x['robot'],OperationKind(x['kind']),x['start_time'],x['end_time'],
        tuple(x['start']),tuple(x['end']),x['sequence'],x['block']) for x in data['operations'])
    return ScheduleResult(ScheduleStatus(data['status']),operations,data['cmax'],tuple(data['completion']),
        diagnostics=tuple(data['diagnostics']),directions=tuple(tuple(d) for d in data['directions']),
        source='NATIVE_PRIORITY',reference_policy_id=PRIORITY_EVALUATOR)


def priority_cache(name,parents,space):
    cache={}
    for row in read_rows(PRIORITY_RESULTS):
        if row['layout_id']!=name or not row.get('solution') or not row.get('schedule'):continue
        plan=deadlock_plan(parents,row['solution'],space)
        cache[plan_key(plan.solution,plan.directions)]=(priority_saved_schedule(row),json.loads(row['progress']))
    return cache


def priority_once(plan,cache):
    key=plan_key(plan.solution,plan.directions)
    if key in cache:return (*cache[key],False,0.)
    progress={};started=time.perf_counter()
    schedule=priority_schedule(plan.solution,CONFIG,directions=plan.directions,progress=progress)
    elapsed=time.perf_counter()-started
    if schedule.directions!=plan.directions:raise ValueError('priority evaluator changed explicit GNN directions')
    cache[key]=(schedule,progress)
    return schedule,progress,True,elapsed


def priority_result_row(name,plan,schedule,progress,**values):
    metrics=routeaware_metrics(plan,schedule)
    # Only the existing model/template costs are reused here, not its B32
    # evaluator or DEADLOCK metadata. Native failed timelines are unavailable.
    metrics={k:v for k,v in metrics.items() if k in PRIORITY_FIELDS}
    for kind in ('wait','completion'):
        for r in range(4):metrics.setdefault(f'{kind}_r{r}_s',None)
    metrics.setdefault('wait_s',None)
    loads=[metrics[f'processing_r{r}_s'] for r in range(4)]
    cycle=max(loads[r]+metrics[f'move_r{r}_s'] for r in range(4))
    return dict(layout_id=name,evaluator=PRIORITY_EVALUATOR,status=schedule.status.value,
        cmax_s=schedule.cmax if schedule.feasible else None,**metrics,
        rejected_pairs=progress['rejected_pairs'],route_cycle_s=cycle,progress=json.dumps(progress),
        solution=json.dumps(pack_directed_plan(plan)),schedule=schedule.canonical_json(),**values)


def priority_history_diagnosis():
    """Rescore saved complete LoadFirst genes; no GNN, B32 or certification."""
    configure_cpu();priority_write([])
    layouts=dict(ppo_layouts('validation'))
    spaces={n:gnn.parent_assignment_space(layouts[n],CONFIG) for n in IMITATION_CASES}
    caches={n:priority_cache(n,layouts[n],spaces[n]) for n in IMITATION_CASES}
    done={n:set() for n in IMITATION_CASES};catalog={n:{} for n in IMITATION_CASES}
    for row in read_rows(PRIORITY_RESULTS):
        if row['stage']=='history' and row.get('solution'):
            plan=deadlock_plan(layouts[row['layout_id']],row['solution'],spaces[row['layout_id']])
            done[row['layout_id']].add(plan_key(plan.solution,plan.directions))
    if LOADFIRST_RESULTS.exists():
        csv.field_size_limit(2**31-1)
        with LOADFIRST_RESULTS.open(encoding='utf-8',newline='') as stream:
            for old in csv.DictReader(stream):
                name=old['layout_id']
                if name not in catalog or old['stage'] not in ('initial','candidate') or not old.get('solution'):continue
                plan=deadlock_plan(layouts[name],old['solution'],spaces[name]);key=plan_key(plan.solution,plan.directions)
                if key not in catalog[name]:catalog[name][key]=(plan,old)
                elif old['b32_calls']=='1':catalog[name][key]=(plan,old)
    else:
        for old in read_rows(ROUTEAWARE_RESULTS):
            name=old['layout_id']
            if name in catalog and old['stage']=='initial' and old.get('solution'):
                plan=deadlock_plan(layouts[name],old['solution'],spaces[name])
                catalog[name][plan_key(plan.solution,plan.directions)]=(plan,old)
    count=0
    for name in IMITATION_CASES:
        for key,(plan,old) in catalog[name].items():
            if key in done[name]:continue
            schedule,progress,called,elapsed=priority_once(plan,caches[name])
            priority_write([priority_result_row(name,plan,schedule,progress,stage='history',seed=SEED,
                method='LoadFirst-history',**{'lambda':old['method'].split('lambda')[-1] if 'lambda' in old['method'] else ''},
                source_b32_status=old['status'],source_b32_s=old.get('b32_s'),
                evaluation_s=elapsed,native_calls=int(called),cache_hits=int(not called),
                diagnostic=json.dumps(dict(source_method=old['method'],source_round=old.get('round'))))])
            count+=1
            if count%100==0:print('PRIORITY_HISTORY',count,name,flush=True)
    history=[r for r in read_rows(PRIORITY_RESULTS) if r['stage']=='history']
    summary={name:dict(total=sum(r['layout_id']==name for r in history),
        feasible=sum(r['layout_id']==name and r['status']=='FEASIBLE' for r in history),
        b32_deadlock_to_feasible=sum(r['layout_id']==name and r['source_b32_status']=='DEADLOCK' and r['status']=='FEASIBLE' for r in history))
        for name in IMITATION_CASES}
    print('PRIORITY_HISTORY_SUMMARY',json.dumps(summary),flush=True)
    return history


def priority_native_better(schedule,progress,cycle,current,current_progress,current_cycle):
    if current.feasible:
        return schedule.feasible and schedule.cmax<current.cmax-1e-8
    if schedule.feasible:return True
    return (progress['rejected_pairs'],cycle)<(current_progress['rejected_pairs'],current_cycle-1e-8)


def priority_load_allowed(loads,limits):
    return (max(loads)<=limits['pmax_limit_s']+1e-8 and
            max(loads)-min(loads)<=limits['spread_limit_s']+1e-8)


def priority_oriented_candidates(plan,workspace,rng,*,deadlock=True):
    """Reuse HGA's candidate kernel and all six oriented segment operators."""
    w=workspace;routes=w.encode(plan.solution)
    bits=[np.asarray(d,np.int64) for d in plan.directions]
    for r in range(4):
        native_hga._refresh_oriented_route(r,routes[r],bits[r],w.travel,w.processing,w.eligible,w.oriented_scratch)
    costs=[native_hga._oriented_route_cost(route,d,w.travel,w.processing) for route,d in zip(routes,bits)]
    active=np.concatenate(routes);alpha=min(20,max(1,len(active)-1));active_set=set(active.tolist())
    near=np.full((len(w.processing),alpha),-1,np.int64)
    for u in active:
        values=[v for v in w.nearest[u] if v!=u and v in active_set][:alpha]
        near[u,:len(values)]=values
    actions=np.empty((6,len(active)*alpha,2),np.int64);counts=np.zeros(6,np.int64)
    scratch=w.oriented_scratch;robot,position=scratch[5],scratch[6]
    native_hga._collect_oriented_candidates(active,near,robot,position,scratch[2],int(np.argmax(costs)),actions,counts)
    groups={m:[] for m in (4,5,6,1,2,3)}
    for move in groups:
        pairs=set()
        for u,v in actions[move-1,:counts[move-1]]:
            u,v=int(u),int(v)
            if move in (2,3,4,6):
                pair=tuple(sorted((u,v)))
                if pair in pairs:continue
                pairs.add(pair)
            ru,iu,rv,iv=int(robot[u]),int(position[u]),int(robot[v]),int(position[v])
            a,b=native_hga._move_segments(routes,move,ru,iu,rv,iv)
            ca=native_hga._oriented_segment_cost(a,ru,scratch,w.travel,w.parent_index)
            cb=0. if move==4 else native_hga._oriented_segment_cost(b,rv,scratch,w.travel,w.parent_index)
            if not math.isfinite(ca+cb):continue
            cycle=max(ca,cb,*(costs[r] for r in range(4) if r not in (ru,rv)))
            loads=[sum(w.processing[n] for n in route) for route in routes]
            loads[ru]=sum(float(scratch[7][source,hi]-scratch[7][source,lo]) for source,lo,hi,_ in a)
            if move!=4:loads[rv]=sum(float(scratch[7][source,hi]-scratch[7][source,lo]) for source,lo,hi,_ in b)
            groups[move].append((cycle,rng.random(),(move,u,v,ru,iu,rv,iv),tuple(loads),a,b))
        # Cycle cost is only a DEADLOCK search hint. A feasible incumbent is
        # improved solely by actual priority_schedule Cmax, without that hint.
        groups[move].sort(key=(lambda item:(item[0],item[1])) if deadlock else (lambda item:item[1]))
    return routes,bits,groups


def priority_materialize(plan,w,routes,bits,item):
    _,_,action,loads,a,b=item;move,u,v,ru,iu,rv,iv=action
    rr=list(routes);dd=list(bits)
    for r,segments in ((ru,a),) if move==4 else ((ru,a),(rv,b)):
        rr[r]=native_hga._materialize_segments(segments,routes)
        dd[r]=native_hga._materialize_directions(segments,bits)
    if not all(len(route) for route in rr):return None
    solution=w.solution(rr,w.initial_selected)
    expected=tuple(tuple(w.ids[n] for n in route) for route in rr)
    if tuple(r.block_ids for r in solution.routes)!=expected:return None
    if solution.patterns!=plan.solution.patterns:raise ValueError('route VND changed frozen GNN patterns')
    return gnn.JointCompletionResult(solution,tuple(tuple(int(b) for b in route) for route in dd),(),0.)


def priority_vnd_case(name,seed,initial,space,cache,*,budget=100):
    saved=[r for r in read_rows(PRIORITY_RESULTS) if r['layout_id']==name and r['seed']==str(seed) and r['method']=='GNN-Priority-VND']
    final=next((r for r in reversed(saved) if r['stage']=='final'),None)
    if final:return final
    started=time.perf_counter();rng=Random(seed)
    w=native_hga._HGAWorkspace(initial.solution.parents,CONFIG,FORMAL_SCOPE_V2,patterns=initial.solution.patterns)
    initial_schedule,initial_progress,called,elapsed=priority_once(initial,cache)
    initial_row=priority_result_row(name,initial,initial_schedule,initial_progress,stage='initial',seed=seed,
        method='GNN-Priority-VND',native_calls=int(called),cache_hits=int(not called),evaluation_s=elapsed)
    pmax=initial_row['max_processing_s'];spread=initial_row['processing_spread_s']
    limits=dict(initial_status=initial_schedule.status.value,initial_pmax_s=pmax,initial_spread_s=spread,
        pmax_limit_s=1.01*pmax,spread_limit_s=spread+20.,initial_move_s=initial_row['move_s'])
    current=initial;current_schedule=initial_schedule;current_progress=initial_progress
    current_row=initial_row;cycle=initial_row['route_cycle_s']
    visited={plan_key(initial.solution,initial.directions)};counter=0;pass_index=0
    old_initial=next((r for r in saved if r['stage']=='initial'),None)
    calls=int(old_initial['native_calls']) if old_initial else int(called)
    hits=int(old_initial['cache_hits']) if old_initial else int(not called)
    if not saved:priority_write([dict(initial_row,**limits)])
    stop='100 complete neighbour candidate slots exhausted';pruned=Counter()
    while counter<budget:
        pass_index+=1;routes,bits,groups=priority_oriented_candidates(current,w,rng,deadlock=not current_schedule.feasible)
        improvement=None;best_deadlock=None
        order=(4,5,6,1,2,3)
        for order_index,move in enumerate(order):
            allowance=max(1,(budget-counter)//(len(order)-order_index));used=0
            for item in groups[move]:
                if used>=allowance or counter>=budget:break
                loads=item[3]
                if not priority_load_allowed(loads,limits):
                    pruned['load']+=1;continue
                try:candidate=priority_materialize(current,w,routes,bits,item)
                except ValueError:
                    pruned['canonical']+=1;continue
                if candidate is None:pruned['nonempty_or_order']+=1;continue
                key=plan_key(candidate.solution,candidate.directions)
                if key in visited:pruned['duplicate']+=1;continue
                visited.add(key);counter+=1;used+=1
                schedule,progress,called,elapsed=priority_once(candidate,cache)
                old=next((r for r in saved if r['stage']=='candidate' and int(r['candidate_index'])==counter),None)
                calls+=int(old['native_calls']) if old else int(called)
                hits+=int(old['cache_hits']) if old else int(not called)
                robots=tuple(sorted({item[2][3],item[2][5]}))
                before=[dict(robot=r,route=current.solution.routes[r].block_ids,directions=current.directions[r]) for r in robots]
                after=[dict(robot=r,route=candidate.solution.routes[r].block_ids,directions=candidate.directions[r]) for r in robots]
                old_genes={b:(r,d) for r,route in enumerate(current.solution.routes)
                           for b,d in zip(route.block_ids,current.directions[r])}
                new_genes={b:(r,d) for r,route in enumerate(candidate.solution.routes)
                           for b,d in zip(route.block_ids,candidate.directions[r])}
                affected={b for b in old_genes if old_genes[b]!=new_genes[b]}
                affected.update(w.ids[n] for n in item[2][1:3])
                parents=sorted({b.split('::')[0] for b in affected})
                row=priority_result_row(name,candidate,schedule,progress,stage='candidate',seed=seed,method='GNN-Priority-VND',
                    candidate_index=counter,pass_index=pass_index,move_id=f'M{move}',robots=json.dumps(robots),parent_ids=json.dumps(parents),
                    before_segments=json.dumps(before),after_segments=json.dumps(after),native_calls=int(called),cache_hits=int(not called),
                    evaluation_s=elapsed,diagnostic=json.dumps(dict(action=item[2],before_pmax=current_row['max_processing_s'],
                        before_spread=current_row['processing_spread_s'],before_move=current_row['move_s'],before_status=current_schedule.status.value)))
                better=priority_native_better(schedule,progress,row['route_cycle_s'],current_schedule,current_progress,cycle)
                row['accepted']=int(better and schedule.feasible)
                if old is None:priority_write([dict(row,**limits)])
                if schedule.feasible and better:
                    improvement=(candidate,schedule,progress,row);break
                if not schedule.feasible and better and (best_deadlock is None or
                    (progress['rejected_pairs'],row['route_cycle_s'])<(best_deadlock[2]['rejected_pairs'],best_deadlock[3]['route_cycle_s'])):
                    best_deadlock=(candidate,schedule,progress,row)
            if improvement is not None or counter>=budget:break
            if move==6 and best_deadlock is not None:break
        chosen=improvement or best_deadlock
        if chosen is None:stop='no improvement in the bounded ordered neighbourhood pass';break
        was_deadlock=not current_schedule.feasible
        current,current_schedule,current_progress,current_row=chosen;cycle=current_row['route_cycle_s']
        if was_deadlock and current_schedule.feasible:
            priority_write([dict(current_row,stage='repair',runtime_s=time.perf_counter()-started,**limits)])
        priority_write([dict(current_row,stage='accepted',accepted=1,**limits)])
    final=dict(current_row,stage='final',seed=seed,method='GNN-Priority-VND',runtime_s=time.perf_counter()-started,
        native_calls=calls,cache_hits=hits,candidate_index=counter,diagnostic=json.dumps(dict(stop=stop,pruned=dict(pruned),
        evaluation='priority_schedule only',seed_role='neighbour tie ordering; deterministic saved GNN initial genes reused')),**limits)
    priority_write([final])
    print('PRIORITY_VND_FINAL',json.dumps({k:final.get(k) for k in ('layout_id','seed','status','cmax_s','candidate_index','native_calls','move_s','wait_s','runtime_s')}),flush=True)
    return final


def priority_pilot():
    priority_history_diagnosis()
    layouts=dict(ppo_layouts('validation'));source=read_rows(LOADFIRST_RESULTS)
    starts={r['layout_id']:r for r in source if r['stage']=='screen' and r['method']==loadfirst_method(.25)}
    originals={r['layout_id']:r for r in read_rows(IMITATION_RESULTS) if r['method']=='GNN_IMITATION'}
    native_refs={}
    for r in read_rows(PPO_RESULTS):
        if r['stage']=='baseline' and r['layout_id'] in IMITATION_CASES:
            detail=json.loads(r['diagnostic']);native_refs[r['layout_id'],int(r['seed']),r['method']]=detail
    done={(r['stage'],r['layout_id'],r['seed'],r['method']) for r in read_rows(PRIORITY_RESULTS)}
    finals=[]
    for name in IMITATION_CASES:
        space=gnn.parent_assignment_space(layouts[name],CONFIG);cache=priority_cache(name,layouts[name],space)
        initial=deadlock_plan(layouts[name],starts[name]['solution'],space)
        original=deadlock_plan(layouts[name],json.loads(originals[name]['diagnostic'])['solution'],space)
        for seed in TEST_SEEDS:
            for method,plan in (('GNN-Original',original),('GNN-LoadFirst',initial)):
                if ('reference',name,str(seed),method) in done:continue
                schedule,progress,called,elapsed=priority_once(plan,cache)
                priority_write([priority_result_row(name,plan,schedule,progress,stage='reference',seed=seed,method=method,
                    evaluation_s=elapsed,native_calls=int(called),cache_hits=int(not called),diagnostic='saved deterministic GNN genes, explicit directions')])
            for method in ('HGA','WAG+VNS'):
                if ('reference',name,str(seed),method) in done:continue
                detail=native_refs.get((name,seed,method))
                priority_write([dict(stage='reference',layout_id=name,seed=seed,method=method,evaluator='historical priority_schedule native_cmax_s',
                    status='FEASIBLE' if detail and detail.get('native_cmax_s') not in (None,'') else 'MISSING',
                    cmax_s=detail.get('native_cmax_s') if detail else None,native_calls=0,
                    diagnostic=json.dumps(dict(source='gnn_ppo_scale.csv native_cmax_s',
                        reason='historical native scalar only; no saved routes/directions; missing seeds left blank; no full baseline search')))])
            finals.append(priority_vnd_case(name,seed,initial,space,cache))
    print('PRIORITY_PILOT_SUMMARY',json.dumps(dict(feasible=sum(r['status']=='FEASIBLE' for r in finals),total=len(finals))),flush=True)
    return finals


CONFLICT_RESULTS = FINAL_RESULTS/'gnn_conflict_core_pilot.csv'
CONFLICT_LOAD_RESULTS = FINAL_RESULTS/'gnn_conflict_load_compensation_pilot.csv'
CONFLICT_METHOD = 'GNN-Conflict-Core-LNS'
CONFLICT_LOAD_METHOD = 'GNN-Conflict-Load-Compensation-LNS'
CONFLICT_FIELDS = (*ROUTEAWARE_FIELDS, 'candidate_id', 'origin_id', 'operator', 'gnn_score',
    'load_allowed', 'load_limits', 'mode_changes', 'assignment_changes', 'rebuild_actions',
    'successful_repair', 'pool_ids')
CONFLICT_LOAD_FIELDS = (*CONFLICT_FIELDS, 'transfer_count', 'transfer_details')


def conflict_write(rows, *, path=None):
    path = CONFLICT_RESULTS if path is None else path
    fields = CONFLICT_LOAD_FIELDS if path == CONFLICT_LOAD_RESULTS else CONFLICT_FIELDS
    with path.open('a' if path.exists() else 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows({k: row.get(k) for k in fields} for row in rows)


def conflict_load_limits(name, references):
    """Case-specific processing protection, independent of the frozen initial genes."""
    vectors = {r['method']: [float(r[f'processing_r{i}_s']) for i in range(4)]
               for r in references if r['layout_id'] == name
               and r['method'] in ('HGA', 'WAG+VNS', 'GNN_IMITATION')}
    if len(vectors) != 3:
        raise ValueError(f'{name}: missing actual HGA/WAG/GNN processing references')
    peaks = [max(p) for p in vectors.values()]
    return dict(pmax_limit_s=1.05*max(peaks),
                spread_limit_s=max(max(max(p)-min(p) for p in vectors.values()), .10*median(peaks)),
                temporary_pmax_limit_s=1.20*1.05*max(peaks), reference_processing=vectors,
                rule='final peak <= 1.05*largest reference peak; spread <= max(reference spreads, 10% median peak)')


def conflict_load_allowed(row, limits):
    if limits.get('objective') == 'Cmax':
        return True  # No processing-load hard constraint exists in the formal model.
    return (float(row['max_processing_s']) <= limits['pmax_limit_s']+1e-7
            and float(row['processing_spread_s']) <= limits['spread_limit_s']+1e-7)


def conflict_core_partial(current, schedule, model, space, embeddings, *, seed, q):
    """Rank actual blocked/direct-blocker cores with the existing assignment GNN."""
    pairs = []
    known = set(space.parent_ids)
    for event in schedule.deadlock_blockages:
        for operation, parents in zip(event.blocker_operations, event.blocker_parent_ids):
            core = (set(event.blocked_parent_ids) | set(parents)) & known
            if operation is not None and set(parents)&known and set(event.blocked_parent_ids)&known:
                pairs.append((event, operation, core))
    if not pairs:
        return None, dict(source='B32 DEADLOCK', reason='no mapped direct blocker core')
    union = set().union(*(core for _, _, core in pairs))
    partial = lns._remove_parent_set(current.solution, union)
    patterns = {p.parent_id: p for p in current.solution.patterns if p.parent_id not in union}
    blocks = {b.block_id: b for p in patterns.values() for b in space.children[p]}
    old_bits = {b: d for route, bits in zip(current.solution.routes, current.directions)
                for b, d in zip(route.block_ids, bits)}
    state = lns.oriented_route_state(partial.routes, tuple(tuple(old_bits[b] for b in route)
                                           for route in partial.routes), blocks, CONFIG)
    indices, logits = model.assignment_logits(space, embeddings, state, blocks, union, CONFIG)
    priorities = {p: -math.inf for p in union}
    for index, score in zip(indices.tolist(), logits.tolist()):
        parent = space.actions[index].parent_id
        priorities[parent] = max(priorities[parent], score)
    pairs.sort(key=lambda item: (-sum(priorities[p] for p in item[2])/len(item[2]),
                                 item[0].robot_id, item[1].operation_id))
    # Rotate among GNN-ranked real cores. This also handles acyclic wait-for graphs.
    event, operation, core = pairs[seed % len(pairs)]
    removed = set(core)
    blocks = block_map(current.solution, CONFIG, scope=FORMAL_SCOPE_V2)
    neighbors = []
    for route in current.solution.routes:
        for pos, block in enumerate(route.block_ids):
            if blocks[block].parent_id in core:
                for offset in (-1, 1, -2, 2):
                    if 0 <= pos+offset < len(route.block_ids):
                        neighbors.append(blocks[route.block_ids[pos+offset]].parent_id)
    for parent in dict.fromkeys(neighbors):
        if len(removed) >= max(q, len(core)):
            break
        removed.add(parent)
    return lns._remove_parent_set(current.solution, removed), dict(source='structured B32 DEADLOCK',
        blocked_robot=event.robot_id, blocked_operation=asdict(event.blocked_operation),
        blocked_parents=event.blocked_parent_ids, blocked_MOVE_endpoints=event.blocked_block_ids,
        blocker_operation=asdict(operation), blocker_parents=next(parents for op, parents in
            zip(event.blocker_operations, event.blocker_parent_ids) if op is operation),
        core=sorted(core), gnn_core_priority={p: priorities[p] for p in core},
        evidence_source=schedule.deadlock_evidence_source)


CYCLE_MODEL = ROOT/'models/gnn_weld_layout_cycle_transfer.pt'
CYCLE_RESULTS = FINAL_RESULTS/'gnn_cycle_transfer_pilot.csv'
CYCLE_METHOD = 'GNN-Cycle-Transfer-LNS'
CYCLE_FIELDS = (*CONFLICT_LOAD_FIELDS, 'cycle_lb_s')
CYCLE_EPS = 1e-7


def cycle_route_state(plan):
    return lns.oriented_route_state(tuple(r.block_ids for r in plan.solution.routes), plan.directions,
        block_map(plan.solution, CONFIG, scope=FORMAL_SCOPE_V2), CONFIG)


def cycle_values(state):
    cycles = tuple(p+sum(arcs) for p, arcs in zip(state.loads, state.arc_costs))
    return cycles, max(cycles), max(state.loads), max(state.loads)-min(state.loads)


def cycle_joint_plan(space, embeddings, model, current, transfer, feedback=None):
    """Express an existing legal child insertion as the ordinary VNext joint action."""
    parent = transfer.action.block_id.split('::', 1)[0]
    before = {a.parent_id: a for a in gnn.parent_assignment_targets(current, CONFIG)}[parent]
    after_owners = gnn.assignment_targets(transfer.solution)
    robots = tuple(after_owners[b.block_id] for b in space.children[before.pattern])
    target = gnn.ParentAssignment(parent, before.pattern, robots)
    target_index = space.action_index[target]
    blocks = block_map(current, CONFIG, scope=FORMAL_SCOPE_V2)
    patterns = {p.parent_id: p for p in current.patterns}
    nodes = {b.block_id: space.block_nodes[patterns[b.parent_id], b.block_id] for b in blocks.values()}
    pending = {parent}
    indices, features = model.assignment_features(space, embeddings, transfer.partial, blocks, pending, CONFIG)
    positions = {index: i for i, index in enumerate(indices.tolist())}
    position = positions[target_index]
    graph = replace(space.graph, block_index=nodes)
    ix, inputs = gnn.vnext_insertion_inputs(graph, transfer.partial,
        ((transfer.state, transfer.action),), blocks, CONFIG)
    moves = features.new_tensor([sum(arcs) for arcs in transfer.state.arc_costs])
    assignment = features[position]
    peak = float(assignment[-8:-4].max())
    nominal = max(p+m for p, m in zip(transfer.state.loads, moves.tolist()))/2500.
    lo = [min((min(blocks[b].start[0], blocks[b].end[0]) for b in route), default=math.inf)
          for route in transfer.state.routes]
    hi = [max((max(blocks[b].start[0], blocks[b].end[0]) for b in route), default=-math.inf)
          for route in transfer.state.routes]
    overlap = sum(max(0., hi[r]+CONFIG.interference_dx-lo[r+1]) for r in (0, 2))/20.
    feedback = feedback or {}
    risk = (feedback.get('parent_wait', {}).get(parent, 0.) +
            feedback.get('parent_blocked', {}).get(parent, 0.))/2500.
    outcome = features.new_tensor((peak, peak-float(assignment[-8:-4].min()), nominal,
        (transfer.state.travel-transfer.partial.travel)/45., overlap, risk))
    return gnn.VNextJointAction(target_index, transfer.state, (transfer.action,), assignment,
                                outcome, (inputs[0].detach(),), (ix[0].detach(),))


@torch.no_grad()
def cycle_score_candidates(space, embeddings, model, candidates):
    flat = [plan for candidate in candidates for plan in candidate['joint_plans']]
    if not flat:
        return
    scores = gnn.imitation_joint_scores(space, embeddings, model, flat)
    offset = 0
    for candidate in candidates:
        count = len(candidate['joint_plans'])
        candidate['gnn_score'] = float(scores[offset:offset+count].sum())/count
        offset += count


def cycle_candidate_group(kind, moves, candidates, source, destination):
    for candidate in candidates:
        candidate.update(kind=kind, moves=moves, source_robot=source, destination_robot=destination)
    return dict(kind=kind, source_robot=source, destination_robot=destination,
                moves=moves, candidates=candidates)


def cycle_transfer_groups(current, space, embeddings, model, *, kind, feedback=None,
                           training=False, limits=None, max_blocks=None, max_partners=None,
                           max_swaps=12):
    """Enumerate bottleneck tasks/receivers, then score legal joint insertions."""
    solution = current.solution
    state = cycle_route_state(current)
    cycles, incumbent_lb, _, _ = cycle_values(state)
    heavy = max(range(4), key=lambda r: (cycles[r], -r))
    groups = []
    working_blocks = block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2)
    patterns = {p.parent_id: p for p in solution.patterns}
    source_blocks = [b for b in state.routes[heavy]
        if patterns[b.split('::', 1)[0]].kind is not SplitKind.X_SPLIT]
    source_blocks.sort(key=lambda b: (-CONFIG.process_time(working_blocks[b].length), b))
    if max_blocks is not None:
        source_blocks = source_blocks[:max_blocks]

    def allowed_candidate(lb, peak, spread):
        # Safe pruning only: an exact directed route-cycle lower bound that already
        # reaches the best measured B32 Cmax cannot improve that Cmax.  A candidate
        # is never rejected merely because its LB does not fall below the current
        # LB, because WAIT can shrink while the directed P+MOVE cycle does not.
        if limits is not None and limits.get('incumbent_cmax_s') is not None \
                and lb >= float(limits['incumbent_cmax_s'])-CYCLE_EPS:
            return False
        return limits is None or (peak <= limits['pmax_limit_s']+CYCLE_EPS
                                  and spread <= limits['spread_limit_s']+CYCLE_EPS)

    if kind == 'relocate':
        for block_id in source_blocks:
            parent = block_id.split('::', 1)[0]
            destinations = sorted({t.move.key.destination_robot_id for t in lns.child_transfer_candidates(
                solution, current.directions, block_id, CONFIG)})
            for destination in destinations:
                moves = [dict(parent_id=parent, block_id=block_id, source=heavy, destination=destination)]
                candidates = []
                for transfer in lns.child_transfer_candidates(solution, current.directions, block_id, CONFIG,
                                                               destination_robot=destination):
                    _, lb, peak, spread = cycle_values(transfer.state)
                    if not training and not allowed_candidate(lb, peak, spread):
                        continue
                    candidates.append(dict(plan=gnn.JointCompletionResult(transfer.solution,
                        transfer.state.directions, (transfer.action,), 0., oriented_state=transfer.state),
                        lb=lb, peak=peak, spread=spread,
                        joint_plans=(cycle_joint_plan(space, embeddings, model, solution, transfer, feedback),)))
                cycle_score_candidates(space, embeddings, model, candidates)
                candidates.sort(key=lambda c: (-c['gnn_score'], c['lb']))
                selected = candidates if training else candidates[:4]
                if selected:
                    groups.append(cycle_candidate_group(kind, moves, selected, heavy, destination))
    elif kind == 'swap':
        # Fixed-position exchange gives an exact route-cycle ordering over every
        # bottleneck-task/receiver-task pair. The leading pairs then retain every
        # legal GNN insertion/direction variant. No pair is selected by a rule
        # score; the GNN jointly orders the expanded complete candidates.
        direct = []
        owners = {b: (r, i) for r, route in enumerate(state.routes) for i, b in enumerate(route)}
        bits = {b: d for route, vector in zip(state.routes, state.directions)
                for b, d in zip(route, vector)}
        for block_id in source_blocks:
            parent = block_id.split('::', 1)[0]
            for destination in range(4):
                if destination == heavy:
                    continue
                for other_id in state.routes[destination]:
                    other_parent = other_id.split('::', 1)[0]
                    if other_parent == parent or patterns[other_parent].kind is SplitKind.X_SPLIT:
                        continue
                    if not (robot_is_eligible(working_blocks[block_id], destination, CONFIG)
                            and robot_is_eligible(working_blocks[other_id], heavy, CONFIG)):
                        continue
                    routes = [list(route) for route in state.routes]
                    source_pos = owners[block_id][1]
                    other_pos = owners[other_id][1]
                    routes[heavy][source_pos], routes[destination][other_pos] = other_id, block_id
                    vectors = [list(vector) for vector in state.directions]
                    vectors[heavy][source_pos], vectors[destination][other_pos] = bits[other_id], bits[block_id]
                    direct_state = lns.oriented_route_state(routes, vectors, working_blocks, CONFIG)
                    _, lb, peak, spread = cycle_values(direct_state)
                    if not training and not allowed_candidate(lb, peak, spread):
                        continue
                    direct.append((lb, block_id, other_id, destination, other_parent))
        # No fixed-position LB pre-ranking: the enumeration order stays structural
        # and max_swaps is only an explicit bound on how many pairs are expanded,
        # not an LB filter applied before the GNN scores the legal candidates.
        for lb, block_id, other_id, destination, other_parent in direct[:max_swaps]:
            parent = block_id.split('::', 1)[0]
            first_moves = [dict(parent_id=parent, block_id=block_id, source=heavy, destination=destination)]
            first = []
            for transfer in lns.child_transfer_candidates(solution, current.directions, block_id, CONFIG,
                                                           destination_robot=destination):
                first.append(dict(plan=gnn.JointCompletionResult(transfer.solution, transfer.state.directions,
                    (transfer.action,), 0., oriented_state=transfer.state), lb=lb,
                    joint_plans=(cycle_joint_plan(space, embeddings, model, solution, transfer, feedback),)))
            cycle_score_candidates(space, embeddings, model, first)
            first.sort(key=lambda c: (-c['gnn_score'], c['plan'].oriented_state.travel))
            for left in first[:2]:
                moves = [*first_moves, dict(parent_id=other_parent, block_id=other_id,
                                            source=destination, destination=heavy)]
                second = []
                for transfer in lns.child_transfer_candidates(left['plan'].solution, left['plan'].directions,
                    other_id, CONFIG, destination_robot=heavy):
                    _, final_lb, peak, spread = cycle_values(transfer.state)
                    if not training and not allowed_candidate(final_lb, peak, spread):
                        continue
                    second.append(dict(plan=gnn.JointCompletionResult(transfer.solution,
                        transfer.state.directions, (*left['plan'].actions, transfer.action), 0.,
                        oriented_state=transfer.state), lb=final_lb, peak=peak, spread=spread,
                        joint_plans=(*left['joint_plans'], cycle_joint_plan(space, embeddings, model,
                            left['plan'].solution, transfer, feedback))))
                cycle_score_candidates(space, embeddings, model, second)
                second.sort(key=lambda c: (-c['gnn_score'], c['lb']))
                selected = second if training else second[:2]
                if selected:
                    groups.append(cycle_candidate_group(kind, moves, selected, heavy, destination))
    else:
        raise ValueError('cycle transfer kind must be relocate or swap')
    if training:
        for group in groups:
            group['candidates'].sort(key=lambda c: (c['lb'], -c['gnn_score']))
            group['candidates'] = group['candidates'][:6]
    return groups


def cycle_group_loss(space, embeddings, model, group):
    scores = []
    for candidate in group['candidates']:
        inputs = gnn.vnext_policy_inputs(space, candidate['joint_plans'], model)
        steps = gnn.imitation_joint_scores(space, embeddings, model, inputs)
        scores.append(steps.mean())
    if not scores:
        return None, 0, 0
    scores = torch.stack(scores)
    cycles = [c['lb'] for c in group['candidates']]
    reference = max(max(cycles, default=0.), 1.)
    losses, weights, correct, pairs = [], [], 0, 0
    for i in range(len(cycles)):
        for j in range(i+1, len(cycles)):
            if abs(cycles[i]-cycles[j]) <= 1e-6:
                continue
            good, bad = (i, j) if cycles[i] < cycles[j] else (j, i)
            weight = abs(cycles[i]-cycles[j])/reference
            losses.append(F.softplus(scores[bad]-scores[good]))
            weights.append(weight)
            correct += int(scores[good] > scores[bad])
            pairs += 1
    if not pairs:
        return None, 0, 0
    weights = scores.new_tensor(weights)
    loss = (torch.stack(losses)*weights).sum()/weights.sum().clamp_min(1e-12)
    return loss, pairs, correct


def imitation_cycle_train():
    """TRAIN-only directed-cycle ranking on the unchanged joint actor."""
    configure_cpu()
    torch.manual_seed(SEED)
    source = torch.load(IMITATION_MODEL, map_location='cpu', weights_only=True)
    saved = torch.load(CYCLE_MODEL, map_location='cpu', weights_only=True) if CYCLE_MODEL.exists() else {}
    if saved.get('training_complete'):
        print('CYCLE_TRAIN_ALREADY_COMPLETE', saved.get('completed_epoch'), flush=True)
        return saved
    initial = saved.get('initial_state_dict', copy.deepcopy(source['state_dict']))
    weights = saved.get('last_state_dict', initial)
    model = imitation_model(weights).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    if saved.get('optimizer_state_dict'):
        optimizer.load_state_dict(saved['optimizer_state_dict'])
    teachers = imitation_best_teachers(imitation_saved())
    training = dict(ppo_layouts('train'))
    # A fixed size-stratified set of existing TRAIN instances, not a new dataset
    # and never VALIDATION/TEST. It bounds ranking collection without changing
    # the available legal receivers or using B32 labels.
    teacher_names=sorted(teachers,key=lambda n:(len(training[n]),n))
    teacher_names=[teacher_names[round(i*(len(teacher_names)-1)/19)] for i in range(20)]
    prepared = []
    with torch.no_grad():
        for name in teacher_names:
            record=teachers[name]
            space = gnn.parent_assignment_space(training[name], CONFIG)
            current = unpack_directed_plan(training[name], record['plan'], space)
            embeddings = model.encode_static(space.graph)
            groups = []
            for kind in ('relocate', 'swap'):
                groups.extend(cycle_transfer_groups(current, space, embeddings, model,
                    kind=kind, training=True, max_blocks=4, max_partners=3))
            sizes = [cycle_group_loss(space, embeddings, model, group)[1] for group in groups]
            kept = [group for group, pairs in zip(groups, sizes) if pairs]
            pairs = sum(sizes)
            prepared.append((name, space, kept))
            print('CYCLE_TRAIN_GROUPS', name, len(kept), 'pairs', pairs, flush=True)
    if not any(groups for _, _, groups in prepared):
        raise ValueError('no usable directed-cycle ranking preference in TRAIN')
    history = list(saved.get('history', ()))
    for epoch in range(saved.get('completed_epoch', 0)+1, 4):
        started = time.perf_counter()
        rng = Random(SEED+epoch)
        order = list(range(len(prepared)))
        rng.shuffle(order)
        epoch_loss, epoch_pairs, epoch_correct, updates = [], 0, 0, 0
        for index in order:
            name, space, groups = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode_static(space.graph)
            values = [cycle_group_loss(space, embeddings, model, group) for group in groups]
            losses = [value[0] for value in values if value[0] is not None]
            if not losses:
                continue
            loss = torch.stack(losses).mean()
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
                raise ArithmeticError('nonfinite directed-cycle ranking optimization')
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            epoch_pairs += sum(value[1] for value in values)
            epoch_correct += sum(value[2] for value in values)
            updates += 1
        history.append(dict(epoch=epoch, loss=sum(epoch_loss)/max(1, len(epoch_loss)),
                            ranking_accuracy=epoch_correct/max(1, epoch_pairs),
                            pairs=epoch_pairs, updates=updates, learning_rate=1e-4,
                            runtime_s=time.perf_counter()-started))
        saved.update(initial_state_dict=initial, state_dict=copy.deepcopy(model.state_dict()),
                     last_state_dict=copy.deepcopy(model.state_dict()), optimizer_state_dict=optimizer.state_dict(),
                     completed_epoch=epoch, training_complete=epoch == 3, history=history,
                     training_split='TRAIN_ONLY', teacher_layouts=[name for name, _, _ in prepared],
                     cycle_metric='max directed processing+MOVE; no WAIT; no B32 labels',
                     network='unchanged VNextGNN assignment/joint/insertion scoring')
        torch.save(saved, CYCLE_MODEL)
        print('CYCLE_TRAIN_EPOCH', json.dumps(history[-1]), flush=True)
    return saved


def cycle_write(rows):
    with CYCLE_RESULTS.open('a' if CYCLE_RESULTS.exists() else 'w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=CYCLE_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows({key: row.get(key) for key in CYCLE_FIELDS} for row in rows)


def cycle_origin_feedback(current, schedule, events):
    if schedule.feasible and schedule.operations:
        summary = lns.schedule_feedback_summary(current.solution, schedule, events, CONFIG)
        return dict(parent_wait=summary['parent_wait'], parent_blocked=summary['parent_blocked'])
    parent_blocked = Counter()
    for event in schedule.deadlock_blockages:
        for parent in event.blocked_parent_ids:
            parent_blocked[parent] += 1
        for parents in event.blocker_parent_ids:
            for parent in parents:
                parent_blocked[parent] += 1
    return dict(parent_wait={}, parent_blocked=dict(parent_blocked))


@torch.no_grad()
def cycle_transfer_case(name, parents, seed, model, references, *, budget=100, max_attempts=500):
    saved = [r for r in read_rows(CYCLE_RESULTS) if r['layout_id'] == name and int(r['seed']) == seed]
    final = next((r for r in reversed(saved) if r['stage'] == 'final'), None)
    if final is not None:
        return final
    started = time.perf_counter()
    limits = conflict_load_limits(name, references)
    space = gnn.parent_assignment_space(parents, CONFIG)
    history = [*read_rows(CONFLICT_LOAD_RESULTS), *read_rows(CONFLICT_RESULTS), *read_rows(CYCLE_RESULTS)]
    seen, blockers, records = routeaware_history(name, parents, space, extra_rows=history)
    embeddings = model.encode_static(space.graph)
    warm = next(r for r in read_rows(CONFLICT_LOAD_RESULTS)
                if r['layout_id'] == name and r['stage'] == 'final' and int(r['seed']) == seed)
    original = deadlock_plan(parents, warm['solution'], space)
    calls = sum(int(r.get('b32_calls') or 0) for r in saved if r['stage'] in ('initial', 'candidate'))
    hits = 0
    attempts = sum(r['stage'] == 'round' for r in saved)
    visited, all_entries, failures = set(), [], Counter()
    saved_ids = {r['candidate_id']: r for r in saved if r['stage'] in ('initial', 'candidate')}
    exhausted = set()

    def evaluate(plan, candidate_id, *, origin=None, removed=(), feedback=None, construction=None,
                 stage='candidate', operator='cycle-transfer'):
        nonlocal calls, hits
        key = plan_key(plan.solution, plan.directions)
        if key in visited or key not in seen and calls >= budget:
            return None
        visited.add(key)
        transfers = conflict_actual_transfers(plan, dict(moves=(feedback or {}).get('moves', ())),
                                              None if origin is None else origin[0])
        if origin is not None and not transfers:
            failures['requested ownership not completed'] += 1
            return None
        _, lb, _, _ = cycle_values(cycle_route_state(plan))
        evaluation = time.perf_counter()
        schedule, called = evaluate_once(plan, seen, blocker_cache=blockers)
        elapsed = time.perf_counter()-evaluation
        calls += int(called)
        hits += int(not called)
        modes, owners = ({}, {}) if origin is None else conflict_changes(origin[0], plan, removed)
        row = routeaware_result_row(name, CYCLE_METHOD, plan, schedule, blockers.get(key, ()),
            historical=records.get(key), stage=stage, candidate_id=candidate_id,
            origin_id='' if origin is None else origin[2]['candidate_id'], operator=operator,
            removed_parent_ids=json.dumps(removed), feedback=json.dumps(feedback or {}),
            construct_s=plan.inference_seconds, b32_s=elapsed if called else 0.,
            runtime_s=plan.inference_seconds+elapsed, b32_calls=int(called), cache_hits=int(not called),
            gnn_score=(construction or {}).get('gnn_score'), load_limits=json.dumps(limits),
            mode_changes=json.dumps(modes), assignment_changes=json.dumps(owners),
            rebuild_actions=json.dumps((construction or {}).get('actions', [])),
            successful_repair=int(origin is not None and origin[1].status is ScheduleStatus.DEADLOCK and schedule.feasible),
            transfer_count=int(bool(transfers)), transfer_details=json.dumps(transfers), cycle_lb_s=lb)
        row.update(seed=seed, load_allowed=int(conflict_load_allowed(row, limits)))
        previous = saved_ids.get(candidate_id)
        if previous is None:
            cycle_write([row])
        else:
            if json.loads(previous['solution']) != json.loads(row['solution']):
                raise ValueError('resumed cycle proposal differs from saved proposal')
            row = previous
        records[key] = row
        entry = (plan, schedule, row)
        all_entries.append(entry)
        print('CYCLE_CANDIDATE', name, candidate_id, schedule.status.value, schedule.cmax,
              'LB', round(lb, 3), 'B32', calls, 'load_ok', row['load_allowed'], flush=True)
        return entry

    initial = evaluate(original, 'warm-start', stage='initial', operator='saved-load-compensation-best')
    if initial is None:
        raise ValueError(f'{name}: unable to restore the previous feasible incumbent')
    # Restore already evaluated cycle candidates after a process interruption.
    for previous in saved:
        if previous['stage'] != 'candidate' or previous['solution'] in (None, ''):
            continue
        plan = deadlock_plan(parents, previous['solution'], space)
        key = plan_key(plan.solution, plan.directions)
        if key in visited or key not in seen:
            continue
        schedule = seen[key]
        entry = (plan, schedule, previous)
        visited.add(key)
        all_entries.append(entry)
    pool = conflict_pool([], [initial])
    while calls < budget and attempts < max_attempts:
        origin = pool[(attempts % len(pool))]
        attempts += 1
        kind = 'relocate' if attempts % 2 else 'swap'
        current, schedule, row = origin
        origin_key = plan_key(current.solution, current.directions)
        if (origin_key, kind) in exhausted:
            cycle_write([dict(stage='round', layout_id=name, seed=seed, method=CYCLE_METHOD,
                round=attempts, b32_calls=0, cache_hits=0, runtime_s=time.perf_counter()-started,
                diagnostic=json.dumps(dict(skip='deterministic origin/kind already exhausted')))])
            continue
        feedback = cycle_origin_feedback(current, schedule, blockers.get(origin_key, ()))
        groups = cycle_transfer_groups(current, space, embeddings, model, kind=kind,
            feedback=feedback, training=False, limits=dict(limits, incumbent_cmax_s=incumbent_cmax),
            max_blocks=None, max_swaps=16)
        candidates = [dict(group=group, **candidate) for group in groups for candidate in group['candidates']]
        admissible = [e for e in all_entries if int(e[2]['load_allowed']) and e[1].feasible]
        incumbent_cmax = min((e[1].cmax for e in admissible), default=math.inf)
        candidates = [c for c in candidates if c['lb'] < incumbent_cmax-CYCLE_EPS
                      and plan_key(c['plan'].solution,c['plan'].directions) not in visited]
        candidates.sort(key=lambda c: (-c['gnn_score'], c['lb']))
        new = []
        for index, candidate in enumerate(candidates[:2]):
            if calls >= budget:
                break
            group = candidate['group']
            moves = group['moves']
            removed = tuple(dict.fromkeys(move['parent_id'] for move in moves))
            construction = dict(gnn_score=candidate['gnn_score'],
                actions=[dict(block_id=a.block_id, robot=a.robot, position=a.position, direction=a.direction,
                              neighbour_changes=a.neighbour_changes) for a in candidate['plan'].actions])
            detail = dict(compensation=kind, moves=moves, source_robot=group['source_robot'],
                          destination_robot=group['destination_robot'], candidate_lb_s=candidate['lb'],
                          incumbent_cmax_s=incumbent_cmax, cycle_feedback=feedback)
            entry = evaluate(candidate['plan'], f'attempt{attempts}-{index}', origin=origin, removed=removed,
                feedback=detail, construction=construction, operator='cycle-'+kind)
            if entry is not None:
                new.append(entry)
        if not new:
            exhausted.add((origin_key, kind))
        pool = conflict_pool(pool, new)
        cycle_write([dict(stage='round', layout_id=name, seed=seed, method=CYCLE_METHOD, round=attempts,
            pool_ids=json.dumps([e[2]['candidate_id'] for e in pool]), b32_calls=0, cache_hits=0,
            runtime_s=time.perf_counter()-started, diagnostic=json.dumps(dict(kind=kind,
                candidates_after_lb=len(candidates), failures=failures)))])
    admissible = [e for e in all_entries if int(e[2]['load_allowed'])]
    good = [e for e in admissible if e[1].feasible]
    best = min(good, key=lambda e: e[1].cmax) if good else max(admissible or all_entries, key=conflict_deadlock_progress)
    modes, owners = conflict_changes(original, best[0], space.parent_ids)
    baseline_hga = next(float(r['cmax_s']) for r in references
                        if r['layout_id'] == name and r['method'] == 'HGA' and r['status'] == 'FEASIBLE')
    evaluated = [r for r in read_rows(CYCLE_RESULTS) if r['layout_id'] == name and int(r['seed']) == seed
                 and r['stage'] in ('initial', 'candidate')]
    minimum_lb = min(float(r['cycle_lb_s']) for r in evaluated)
    final = dict(best[2], stage='final', runtime_s=time.perf_counter()-started, b32_calls=calls,
        cache_hits=hits, origin_id=best[2]['candidate_id'], mode_changes=json.dumps(modes),
        assignment_changes=json.dumps(owners), diagnostic=json.dumps(dict(
            stop='B32 budget exhausted' if calls >= budget else '500 attempts/legal origin candidates exhausted',
            attempts=attempts, unique_visited=len(visited), failures=failures,
            feasible_candidates=sum(e[1].feasible for e in all_entries), admissible_feasible=len(good),
            successful_transfers=sum(int(e[2].get('transfer_count') or 0) for e in all_entries),
            minimum_cycle_lb_s=minimum_lb, lb_below_hga=minimum_lb < baseline_hga-CYCLE_EPS,
            evaluation='directed P+MOVE screening; original FORMAL_SCOPE_V2 B32; no priority_schedule', seed=seed)))
    cycle_write([final])
    print('CYCLE_FINAL', json.dumps({key: final.get(key) for key in
        ('layout_id', 'status', 'cmax_s', 'cycle_lb_s', 'b32_calls', 'runtime_s')}), flush=True)
    return final


def imitation_cycle_transfer_pilot():
    saved = imitation_cycle_train()
    references = read_rows(IMITATION_RESULTS)
    layouts = dict(ppo_layouts('validation'))
    model = imitation_model(saved['state_dict']).eval()
    model.requires_grad_(False)
    model.load_first_lambda = None
    model.route_aware = False
    finals = [cycle_transfer_case(name, layouts[name], SEED, model, references) for name in IMITATION_CASES]
    feasible = sum(r['status'] == 'FEASIBLE' and int(r['load_allowed']) for r in finals)
    print('CYCLE_PHASE1_SUMMARY', json.dumps(dict(feasible=feasible, cases=6,
        b32_calls=sum(int(r['b32_calls']) for r in finals),
        lb_below_hga=sum(bool(json.loads(r['diagnostic']).get('lb_below_hga')) for r in finals))), flush=True)
    return finals


def conflict_compensation(current, space, removed, protected, *, q, kind, seed):
    """Select one legal Relocate or two-sided Swap inside the existing core."""
    solution = current.solution
    current_actions = {a.parent_id: a for a in gnn.parent_assignment_targets(solution, CONFIG)}
    state = lns.oriented_route_state(tuple(r.block_ids for r in solution.routes), current.directions,
                                    block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2), CONFIG)
    loads = list(state.loads)
    heavy = max(range(4), key=lambda r: (loads[r], -r))
    processing = {p: tuple(float(x)*2500. for x in space.processing[space.action_index[a]])
                  for p, a in current_actions.items()}

    def alternative(parent, source, destination):
        action = current_actions[parent]
        if action.pattern.kind is SplitKind.X_SPLIT:
            return None
        changing = [i for i, robot in enumerate(action.robots) if robot == source]
        if len(changing) != 1:
            return None
        robots = list(action.robots)
        index = changing[0]
        robots[index] = destination
        candidate = gnn.ParentAssignment(parent, action.pattern, tuple(robots))
        if candidate == action or candidate not in space.action_index:
            return None
        block = space.children[action.pattern][index]
        work = float(space.processing[space.action_index[action], source])*2500.
        return candidate, block.block_id, work

    def score(targets):
        updated = loads.copy()
        for parent, action in targets.items():
            for robot in range(4):
                updated[robot] += (float(space.processing[space.action_index[action], robot])*2500.
                                   - processing[parent][robot])
        return updated, (max(updated), max(updated)-min(updated))

    ranked = []
    if kind == 'relocate':
        for parent in sorted(current_actions):
            work = processing[parent][heavy]
            if work <= CONFIG.numeric_epsilon:
                continue
            for destination in range(4):
                if destination == heavy or loads[destination] >= loads[heavy]:
                    continue
                change = alternative(parent, heavy, destination)
                if change is None:
                    continue
                action, block, _ = change
                targets = {parent: action}
                updated, key = score(targets)
                if key[0] < max(loads)-1e-7 or key[1] < max(loads)-min(loads)-1e-7:
                    ranked.append(((*key, -work, parent, block, destination), targets,
                                   [dict(parent_id=parent, block_id=block, source=heavy, destination=destination)],
                                   heavy, destination, updated))
    elif kind == 'swap':
        for destination in range(4):
            if destination == heavy or loads[destination] >= loads[heavy]:
                continue
            for left in sorted(current_actions):
                if processing[left][heavy] <= CONFIG.numeric_epsilon:
                    continue
                first = alternative(left, heavy, destination)
                if first is None:
                    continue
                for right in sorted(current_actions):
                    if right == left or processing[right][destination] <= CONFIG.numeric_epsilon:
                        continue
                    second = alternative(right, destination, heavy)
                    if second is None:
                        continue
                    targets = {left: first[0], right: second[0]}
                    updated, key = score(targets)
                    pair_gap = abs(updated[heavy]-updated[destination])
                    if key[0] < max(loads)-1e-7 or key[1] < max(loads)-min(loads)-1e-7:
                        ranked.append(((*key, pair_gap, left, right, destination), targets,
                                       [dict(parent_id=left, block_id=first[1], source=heavy, destination=destination),
                                        dict(parent_id=right, block_id=second[1], source=destination, destination=heavy)],
                                       heavy, destination, updated))
    else:
        raise ValueError('compensation kind must be relocate or swap')
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    _, targets, moves, source, destination, updated = ranked[seed % min(3, len(ranked))]
    required = set(protected) | set(targets)
    limit = max(q, len(protected))
    if len(required) > limit:
        return None
    ordered = list(dict.fromkeys(removed))
    retained = [p for p in ordered if p in required]
    retained.extend(p for p in ordered if p not in required)
    removed = tuple(dict.fromkeys((*retained, *(p for p in targets if p not in retained))))[:limit]
    if not required <= set(removed):
        return None
    return removed, targets, dict(compensation=kind, source_robot=source, destination_robot=destination,
                                  processing_before=loads, processing_after=updated, moves=moves)


def conflict_actual_transfers(plan, compensation, origin=None):
    owners = gnn.assignment_targets(plan.solution)
    before = None if origin is None else gnn.assignment_targets(origin.solution)
    actual = []
    for move in (compensation or {}).get('moves', ()):
        source, destination = move['source'], move['destination']
        if owners.get(move['block_id']) != destination or before is not None and before.get(move['block_id']) != source:
            return []
        actual.append(dict(move, actual_source=source, actual_destination=destination))
    return actual


@torch.no_grad()
def conflict_construct(parents, model, space, embeddings, limits, *, rng=None, current=None,
                       removed=(), forced=None, targets=None):
    """Continue the old joint actor over all legal modes and owners of removed parents."""
    started = time.perf_counter()
    pending = set(space.parent_ids) if current is None else set(removed)
    selected = {} if current is None else {p.parent_id: p for p in current.solution.patterns
                                          if p.parent_id not in pending}
    blocks = {b.block_id: b for p in selected.values() for b in space.children[p]}
    nodes = {b.block_id: space.block_nodes[p, b.block_id] for p in selected.values() for b in space.children[p]}
    routes = ((),)*4 if current is None else lns._remove_parent_set(current.solution, pending).routes
    fixed = {} if current is None else {b: d for route, bits in zip(current.solution.routes, current.directions)
                                       for b, d in zip(route.block_ids, bits) if b in blocks}
    bits = tuple(tuple(fixed[b] for b in route) for route in routes)
    state = lns.oriented_route_state(routes, bits, blocks, CONFIG)
    actions, learned = [], []
    forced_indices = None if forced is None else tuple(space.action_index[a] for a in forced)
    target_indices = {p: (space.action_index[a],) for p, a in (targets or {}).items()}
    while pending:
        plans, scores = gnn.vnext_joint_actions(space, embeddings, model, state, blocks, nodes, pending, CONFIG,
            conflict_core=True, fixed_directions=fixed, assignment_indices=forced_indices,
            assignment_targets=target_indices)
        keep = [i for i, plan in enumerate(plans)
                if max(plan.state.loads) <= limits['temporary_pmax_limit_s']+1e-7
                and float(plan.assignment_features[-8:-4].max())*2500 <= limits['temporary_pmax_limit_s']+1e-3]
        if not keep:
            return None, dict(reason='no joint action within temporary peak limit', pending=sorted(pending))
        keep = [i for i in keep if space.actions[plans[i].assignment_index].parent_id not in target_indices
                or plans[i].assignment_index in target_indices[space.actions[plans[i].assignment_index].parent_id]]
        if not keep:
            return None, dict(reason='directed compensation action has no legal joint insertion', pending=sorted(pending))
        keep.sort(key=lambda i: -float(scores[i]))
        index = keep[0]
        if rng is not None and len(keep) > 1:
            top = keep[:12]
            values = scores[top]
            scale = max(1., float(values.std(unbiased=False)))
            index = rng.choices(top, weights=((values-values.max())/scale).softmax(0).tolist())[0]
        plan = plans[index]
        assignment = space.actions[plan.assignment_index]
        learned.append(float(scores[index]))
        state = plan.state
        for block in space.children[assignment.pattern]:
            blocks[block.block_id] = block
            nodes[block.block_id] = space.block_nodes[assignment.pattern, block.block_id]
        selected[assignment.parent_id] = assignment.pattern
        pending.remove(assignment.parent_id)
        actions.extend(plan.actions)
    try:
        if not all(state.routes):
            raise ValueError('LEGAL_FOUR_ROBOT_COVER_NOT_COMPLETED')
        solution = canonicalize(parents, selected.values(), dict(enumerate(state.routes)), CONFIG, scope=FORMAL_SCOPE_V2)
        if tuple(r.block_ids for r in solution.routes) != state.routes or set(solution.patterns) != set(selected.values()):
            raise ValueError('canonicalization changed the directed joint construction')
        if any(selected[parent] != action.pattern or
               tuple(gnn.assignment_targets(solution)[b.block_id] for b in space.children[action.pattern]) != action.robots
               for parent, action in (targets or {}).items()):
            raise ValueError('canonical solution did not complete the directed compensation ownership')
        # Retained tasks keep mode, owner, relative order and direction.
        for robot, route in enumerate(routes):
            kept = set(route)
            if tuple(b for b in state.routes[robot] if b in kept) != route:
                raise ValueError('reconstruction changed retained route order')
            actual = dict(zip(state.routes[robot], state.directions[robot]))
            if any(actual[b] != fixed[b] for b in route):
                raise ValueError('reconstruction changed retained directions')
    except ValueError as error:
        return None, dict(reason=str(error))
    result = gnn.JointCompletionResult(solution, state.directions, tuple(actions), time.perf_counter()-started,
                                      oriented_state=state)
    return result, dict(gnn_score=sum(learned)/max(1, len(learned)), actions=[asdict(a) for a in actions],
                        directed_assignments=[dict(parent_id=p, pattern_id=a.pattern.pattern_id, robots=a.robots)
                                              for p, a in (targets or {}).items()])


def conflict_initial_hints(parents, original, space):
    """Only unsearched region/WAG assignment groups; never their final answers."""
    from mrta_baselines.wag_vns import generate_wag_assignments
    catalog = {p: tuple(pattern for pattern in space.children if pattern.parent_id == p) for p in space.parent_ids}
    minimal = native_hga.initial_patterns(parents, CONFIG, catalog=catalog)
    hints = []
    for label, patterns in (('minimal-modes', minimal), ('GNN-modes', original.solution.patterns)):
        try:
            region = native_hga.initialize_hga_region_seed(parents, CONFIG, patterns=patterns, scope=FORMAL_SCOPE_V2)
        except ValueError:
            # That initializer predates prescribed X-child owners. Discard its
            # illegal hint; the WAG generator below already respects them. Do
            # not modify either official baseline or relax canonicalization.
            region = None
        if region is not None:
            hints.append((label+'-region', gnn.parent_assignment_targets(region, CONFIG)))
        for index, assignment in enumerate(generate_wag_assignments(parents, patterns, CONFIG, max_variants=2,
                                                                   scope=FORMAL_SCOPE_V2)):
            hints.append((f'{label}-WAG-group{index}', gnn.parent_assignment_targets(assignment, CONFIG)))
    return hints


def conflict_deadlock_progress(entry):
    """Observed dispatch cursor fractions, used only to retain DEADLOCK diversity."""
    plan, schedule, _ = entry
    lengths = [max(1, 4*len(route.block_ids)-1) for route in plan.solution.routes]
    fractions = [event.next_template_index/lengths[event.robot_id] for event in schedule.deadlock_blockages]
    return min(fractions, default=0.), sum(fractions)/max(1, len(fractions))


def conflict_load_balance(entry):
    row = entry[2]
    limits = json.loads(row.get('load_limits') or '{}')
    if limits.get('objective') == 'Cmax':
        return (0., 0.)
    peak, spread = float(row.get('max_processing_s') or 0.), float(row.get('processing_spread_s') or 0.)
    peak_ratio = peak / max(limits.get('pmax_limit_s', 1.), 1e-12)
    spread_ratio = spread / max(limits.get('spread_limit_s', 1.), 1e-12)
    return max(peak_ratio, spread_ratio), max(0., peak_ratio-1.) + max(0., spread_ratio-1.)


def conflict_pool(entries, latest, *, width=6):
    valid = [e for e in (*entries, *latest) if e[1].status in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK)]
    by_key = {plan_key(e[0].solution, e[0].directions): e for e in valid}
    latest_keys = [plan_key(e[0].solution, e[0].directions) for e in reversed(latest)
                   if plan_key(e[0].solution, e[0].directions) in by_key]
    feasible_ok = [e for e in by_key.values() if e[1].feasible and int(e[2].get('load_allowed') or 0)]
    deadlock_ok = [e for e in by_key.values() if e[1].status is ScheduleStatus.DEADLOCK
                   and int(e[2].get('load_allowed') or 0)]
    deadlocks = [e for e in by_key.values() if e[1].status is ScheduleStatus.DEADLOCK]
    feasible_bad = [e for e in by_key.values() if e[1].feasible and not int(e[2].get('load_allowed') or 0)]
    ranked = sorted(by_key.values(), key=lambda e: (0, e[1].cmax) if e[1].feasible else
                    (1, *(-x for x in conflict_deadlock_progress(e))))
    priority = [
        sorted(feasible_ok, key=lambda e: e[1].cmax),
        sorted(feasible_ok, key=lambda e: (conflict_load_balance(e)[0], e[1].cmax)),
        sorted(deadlock_ok, key=lambda e: tuple(-x for x in conflict_deadlock_progress(e))),
        sorted(deadlocks, key=lambda e: (conflict_load_balance(e)[1], *(-x for x in conflict_deadlock_progress(e)))),
        sorted(feasible_bad, key=lambda e: e[1].cmax),
        [by_key[latest_keys[0]]] if latest_keys else [],
    ]
    chosen, keys, signatures = [], set(), set()

    def signature(entry):
        plan, schedule, _ = entry
        return (plan.solution.patterns, tuple((e.robot_id, e.blocked_parent_ids, e.blocker_parent_ids)
                                              for e in schedule.deadlock_blockages))

    def add(entry, *, diverse=False):
        if entry is None or len(chosen) >= width:
            return
        key = plan_key(entry[0].solution, entry[0].directions)
        if key in keys or diverse and signature(entry) in signatures:
            return
        chosen.append(entry)
        keys.add(key)
        signatures.add(signature(entry))

    for category in priority:
        for entry in category:
            before = len(chosen)
            add(entry)
            if len(chosen) != before:
                break
    for entry in (*ranked, *(by_key[k] for k in latest_keys)):
        add(entry, diverse=True)
        if len(chosen) >= width:
            break
    return chosen


def conflict_changes(before, after, removed):
    old = {p.parent_id: p.pattern_id for p in before.solution.patterns}
    modes = {p.parent_id: dict(before=old[p.parent_id], after=p.pattern_id) for p in after.solution.patterns
             if old[p.parent_id] != p.pattern_id}
    def owners(plan):
        return {p: [(b, r) for r, route in enumerate(plan.solution.routes) for b in route.block_ids
                    if b.split('::')[0] == p] for p in removed}
    a, b = owners(before), owners(after)
    return modes, {p: dict(before=a[p], after=b[p]) for p in removed if a[p] != b[p]}


@torch.no_grad()
def conflict_core_case(name, parents, seed, model, references, *, budget=100, result_path=CONFLICT_RESULTS,
                       method=CONFLICT_METHOD, load_compensation=False):
    def write(rows):
        if result_path == CONFLICT_RESULTS:
            conflict_write(rows)
        else:
            conflict_write(rows, path=result_path)
    saved = [r for r in read_rows(result_path) if r['layout_id'] == name and int(r['seed']) == seed]
    final = next((r for r in reversed(saved) if r['stage'] == 'final'), None)
    if final is not None:
        return final
    started = time.perf_counter()
    limits = conflict_load_limits(name, references)
    space = gnn.parent_assignment_space(parents, CONFIG)
    history = [*read_rows(LOADFIRST_RESULTS), *read_rows(CONFLICT_RESULTS), *read_rows(result_path)]
    seen, blockers, records = routeaware_history(name, parents, space, extra_rows=history)
    embeddings = model.encode_static(space.graph)
    old = next(r for r in references if r['layout_id'] == name and r['method'] == 'GNN_IMITATION')
    original = deadlock_plan(parents, json.loads(old['diagnostic'])['solution'], space)
    calls = sum(int(r.get('b32_calls') or 0) for r in saved if r['stage'] in ('initial', 'candidate'))
    hits = 0
    visited, all_entries, failures = set(), [], Counter()
    saved_ids = {r['candidate_id']: r for r in saved if r['stage'] in ('initial', 'candidate')}

    def evaluate(plan, candidate_id, *, origin=None, removed=(), feedback=None, construction=None, stage='candidate', operator='core-rebuild'):
        nonlocal calls, hits
        key = plan_key(plan.solution, plan.directions)
        if key in visited or (key not in seen and calls >= budget):
            return None
        visited.add(key)
        evaluation = time.perf_counter()
        transfers = (conflict_actual_transfers(plan, feedback, None if origin is None else origin[0])
                     if (feedback or {}).get('compensation') in ('relocate', 'swap') else [])
        if (feedback or {}).get('compensation') in ('relocate', 'swap') and not transfers:
            failures['compensation did not change the requested ownership'] += 1
            return None
        schedule, called = evaluate_once(plan, seen, blocker_cache=blockers)
        elapsed = time.perf_counter()-evaluation
        calls += int(called)
        hits += int(not called)
        modes, owners = ({}, {}) if origin is None else conflict_changes(origin[0], plan, removed)
        row = routeaware_result_row(name, method, plan, schedule, blockers.get(key, ()),
            historical=records.get(key), stage=stage, candidate_id=candidate_id,
            origin_id='' if origin is None else origin[2]['candidate_id'], operator=operator,
            removed_parent_ids=json.dumps(removed), feedback=json.dumps(feedback or {}),
            construct_s=plan.inference_seconds, b32_s=elapsed if called else 0.,
            runtime_s=plan.inference_seconds+elapsed, b32_calls=int(called), cache_hits=int(not called),
            gnn_score=(construction or {}).get('gnn_score'), load_limits=json.dumps(limits),
            mode_changes=json.dumps(modes), assignment_changes=json.dumps(owners),
            rebuild_actions=json.dumps((construction or {}).get('actions', [])),
            successful_repair=int(origin is not None and origin[1].status is ScheduleStatus.DEADLOCK and schedule.feasible),
            transfer_count=int(bool(transfers)), transfer_details=json.dumps(transfers))
        row['seed'] = seed
        row['load_allowed'] = int(conflict_load_allowed(row, limits))
        previous = saved_ids.get(candidate_id)
        if previous is None:
            write([row])
        else:
            if json.loads(previous['solution']) != json.loads(row['solution']):
                raise ValueError('resumed seeded proposal differs from saved proposal')
            row = previous
        records[key] = row
        entry = (plan, schedule, row)
        all_entries.append(entry)
        print('CONFLICT_CANDIDATE', name, seed, candidate_id, schedule.status.value, schedule.cmax,
              'Pmax', round(float(row['max_processing_s']), 3), 'load_ok', row['load_allowed'],
              'B32', calls, 'modes', len(modes), 'core', len(removed), flush=True)
        return entry

    initial = evaluate(original, 'initial0', stage='initial', operator='saved-GNN-initial')
    candidates = [initial] if initial else []
    proposal, detail = conflict_construct(parents, model, space, embeddings, limits)
    if proposal is not None:
        entry = evaluate(proposal, 'initial1', stage='initial', operator='joint-GNN', construction=detail)
        if entry: candidates.append(entry)
    else:
        failures[detail['reason']] += 1
    for index, (label, hints) in enumerate(conflict_initial_hints(parents, original, space), start=2):
        proposal, detail = conflict_construct(parents, model, space, embeddings, limits,
                                              rng=Random(seed+index), forced=hints)
        if proposal is not None:
            entry = evaluate(proposal, f'initial{index}', stage='initial', operator=label, construction=detail)
            if entry: candidates.append(entry)
        else:
            failures[detail['reason']] += 1
    # At most eight initial plans. Their GNN construction scores choose the
    # initial search origins; B32 feasibility/Cmax ranks the retained pool.
    candidates.sort(key=lambda e: -float(e[2].get('gnn_score') or -1e9))
    pool = conflict_pool([], candidates)
    rounds = 0
    while calls < budget and rounds < 40:
        rounds += 1
        origin = pool[(rounds-1) % len(pool)]
        current, schedule, row = origin
        q = min(len(parents), 12, max(4, round(.08*len(parents)))+2*((rounds-1)%3))
        new = []
        for rank in range(3 if load_compensation else 4):
            if calls >= budget:
                break
            pick_seed = seed+rounds*7+rank
            if schedule.status is ScheduleStatus.DEADLOCK:
                partial, feedback = conflict_core_partial(current, schedule, model, space, embeddings, seed=pick_seed, q=q)
            elif schedule.feasible:
                partial, feedback = lns.select_routeaware_bottleneck(current.solution, current.directions, schedule,
                    blockers.get(plan_key(current.solution, current.directions), ()), CONFIG, q=q, seed=pick_seed,
                    assignment_space=space)
            else:
                partial, feedback = None, dict(reason='no structured B32 feedback')
            if partial is None:
                failures[feedback.get('reason', 'empty feedback core')] += 1
                continue
            removed, targets = partial.removed_parent_ids, None
            if load_compensation and rank:
                kind = 'relocate' if rank == 1 else 'swap'
                compensated = conflict_compensation(current, space, removed,
                    set(feedback.get('core', ())) if schedule.status is ScheduleStatus.DEADLOCK else set(),
                    q=q, kind=kind, seed=pick_seed)
                if compensated is None:
                    failures['no improving legal '+kind] += 1
                    continue
                removed, targets, compensation = compensated
                feedback = dict(feedback, **compensation)
            proposal, detail = conflict_construct(parents, model, space, embeddings, limits,
                rng=None if rank == 0 else Random(pick_seed), current=current, removed=removed, targets=targets)
            if proposal is None:
                failures[detail['reason']] += 1
                continue
            entry = evaluate(proposal, f'round{rounds}-{rank}', origin=origin, removed=removed,
                feedback=feedback, construction=detail,
                operator='core-rebuild' if not load_compensation or not rank else ('core-relocate' if rank == 1 else 'core-swap'))
            if entry: new.append(entry)
            else: failures['duplicate complete plan'] += 1
        pool = conflict_pool(pool, new)
        write([dict(stage='round', layout_id=name, seed=seed, method=method, round=rounds,
            pool_ids=json.dumps([e[2]['candidate_id'] for e in pool]), b32_calls=0, cache_hits=0,
            runtime_s=time.perf_counter()-started, diagnostic=json.dumps(dict(failures=failures)))])
    admissible = [e for e in all_entries if conflict_load_allowed(e[2], limits)]
    good = [e for e in admissible if e[1].feasible]
    best = min(good, key=lambda e: e[1].cmax) if good else max(admissible or all_entries, key=conflict_deadlock_progress)
    modes, owners = conflict_changes(original, best[0], space.parent_ids)
    final = dict(best[2], stage='final', runtime_s=time.perf_counter()-started, b32_calls=calls, cache_hits=hits,
        origin_id=best[2]['candidate_id'], mode_changes=json.dumps(modes), assignment_changes=json.dumps(owners),
        diagnostic=json.dumps(dict(stop='B32 budget exhausted' if calls >= budget else '40 bounded candidate-pool rounds exhausted',
            rounds=rounds, unique_visited=len(visited), failures=failures, initial_candidate='initial0',
            feasible_candidates=sum(e[1].feasible for e in all_entries), admissible_feasible=len(good),
            successful_local_repairs=[e[2]['candidate_id'] for e in all_entries if int(e[2]['successful_repair'])],
            successful_transfers=sum(int(e[2].get('transfer_count') or 0) for e in all_entries),
            evaluation='original FORMAL_SCOPE_V2 B32; no priority_schedule', seed=seed)))
    write([final])
    print('CONFLICT_FINAL', json.dumps({k: final.get(k) for k in
        ('layout_id', 'seed', 'status', 'cmax_s', 'max_processing_s', 'processing_spread_s', 'b32_calls', 'runtime_s')}), flush=True)
    return final


def imitation_conflict_core_pilot():
    configure_cpu()
    conflict_write([])
    references = read_rows(IMITATION_RESULTS)
    layouts = dict(ppo_layouts('validation'))
    model = imitation_model(torch.load(IMITATION_MODEL, map_location='cpu', weights_only=True)['state_dict']).eval()
    model.requires_grad_(False)
    # No LoadFirst/Pareto branch or new learned parameters in this experiment.
    model.load_first_lambda = None
    model.route_aware = False
    finals = [conflict_core_case(name, layouts[name], SEED, model, references) for name in IMITATION_CASES]
    feasible = sum(r['status'] == 'FEASIBLE' and int(r['load_allowed']) for r in finals)
    print('CONFLICT_PHASE1_SUMMARY', json.dumps(dict(feasible=feasible, cases=6,
        target_met=feasible >= 5, b32_calls=sum(int(r['b32_calls']) for r in finals))), flush=True)
    if feasible >= 5:
        for seed in TEST_SEEDS[1:]:
            finals.extend(conflict_core_case(name, layouts[name], seed, model, references) for name in IMITATION_CASES)
    return finals


def imitation_conflict_load_pilot():
    configure_cpu()
    conflict_write([], path=CONFLICT_LOAD_RESULTS)
    references = read_rows(IMITATION_RESULTS)
    layouts = dict(ppo_layouts('validation'))
    model = imitation_model(torch.load(IMITATION_MODEL, map_location='cpu', weights_only=True)['state_dict']).eval()
    model.requires_grad_(False)
    model.load_first_lambda = None
    model.route_aware = False
    finals = [conflict_core_case(name, layouts[name], SEED, model, references,
                                result_path=CONFLICT_LOAD_RESULTS, method=CONFLICT_LOAD_METHOD,
                                load_compensation=True) for name in IMITATION_CASES]
    feasible = sum(r['status'] == 'FEASIBLE' and int(r['load_allowed']) for r in finals)
    print('CONFLICT_LOAD_PHASE1_SUMMARY', json.dumps(dict(feasible=feasible, cases=6,
        b32_calls=sum(int(r['b32_calls']) for r in finals),
        successful_transfers=sum(json.loads(r['diagnostic']).get('successful_transfers', 0) for r in finals))), flush=True)
    return finals


# ---------------------------------------------------------------------------
# Independent GNN main algorithm: B32-feedback-driven three-parent joint
# reconstruction.  Initialisation is the original GNN_VNEXT_WORKLOAD actor only;
# no imitation/cycle checkpoint, no HGA/WAG teacher, hint, warm start or fallback.
# The single move operator removes a measured core and lets the same joint actor
# choose modes, owners, insertion order/positions and directions together.
# ---------------------------------------------------------------------------
JOINT3_LAYOUTS = tuple(f'validation_{index:02d}' for index in range(1, 11))
JOINT3_MODEL = ROOT/'models/gnn_weld_layout_joint3.pt'
JOINT3_RESULTS = FINAL_RESULTS/'gnn_joint3_rebuild.csv'
JOINT3_TRAINING_RESULTS = FINAL_RESULTS/'gnn_joint3_training.csv'
JOINT3_METHOD = 'GNN-Joint3-B32-Feedback'
JOINT3_CORE = 3
JOINT3_BEAM = 3
JOINT3_TRAIN_BEAM = 4
JOINT3_BUDGET = 100
JOINT3_ATTEMPTS = 500
JOINT3_STALE = 128
JOINT3_SEEDS = TEST_SEEDS
JOINT3_EPOCHS = 3
JOINT3_LEARNING_RATE = 1e-4
JOINT3_TRAIN_LAYOUTS = 20
JOINT3_TRAIN_CORES = 4
JOINT3_EPS = CYCLE_EPS
JOINT3_HGA_REFERENCE = FINAL_RESULTS/'hga_results_optimized.csv'
JOINT3_WAG_REFERENCE = FINAL_RESULTS/'wag_vns_results_optimized.csv'
JOINT3_HGA_SOURCE = HGA_ROUTE_SOURCE
JOINT3_WAG_SOURCE = BASELINE_SOURCES['WAG+VNS']
JOINT3_FIELDS = (*CONFLICT_LOAD_FIELDS, 'beam_width', 'core_size', 'route_lb_s', 'incumbent_cmax_s',
                 'feedback_features', 'load_allowed_three_ref', 'load_limits_three_ref', 'run_id')
JOINT3_TRAINING_FIELDS = ('stage', 'layout_id', 'parent_count', 'seed', 'method', 'epoch', 'loss',
    'ranking_accuracy', 'pairs', 'updates', 'gradient_norm', 'learning_rate', 'candidates', 'b32_calls',
    'construct_s', 'b32_s', 'runtime_s', 'diagnostic')
# Load-keeping / deadlock-repair stage: same operator, no new network, no new scheduler.
JOINT3B_METHOD = 'GNN-Joint3-LoadKeep-Repair'
JOINT3B_RESULTS = FINAL_RESULTS/'gnn_joint3_loadkeep.csv'
JOINT3B_TOLERANCE = 1.0
JOINT3B_TRAIN_CASES = ('train_52', 'scale_train_0482', 'scale_train_0973',
                       'scale_train_0214', 'scale_train_0617', 'scale_train_0268')
JOINT3B_FIELDS = (*JOINT3_FIELDS, 'variant', 'phase', 'final_choice', 'r_c_pct',
                  'alt_cmax_balanced_s', 'alt_cmax_best_s', 'alt_cmax_balance_s',
                  'deadlock_origins', 'deadlock_repairs', 'cmax_tolerance_pct')
# Joint4: GNN-guided bottleneck cross-robot parent re-insertion on the same actor.
JOINT4_METHOD = 'GNN-Joint4-Bottleneck-Reassign'
JOINT4_MODEL = ROOT/'models/gnn_weld_layout_joint4.pt'
JOINT4_RESULTS = FINAL_RESULTS/'gnn_joint4_reassign.csv'
JOINT4_CORE = 4
JOINT4_BEAM = 4
JOINT4_B32_PER_ROUND = 2
JOINT4_EPOCHS = 3
JOINT4_TRAIN_LAYOUTS = 20
JOINT4_TRAIN_CORES = 4
JOINT4_FIELDS = (*JOINT3B_FIELDS, 'native_s', 'native_status', 'transfer_robot', 'receiver_robot',
                 'core_kind', 'cross_robot_changes', 'epoch', 'loss', 'ranking_accuracy', 'pairs',
                 'updates')
CMAX_MODEL = ROOT/'models/gnn_weld_layout_cmax_conflict.pt'
CMAX_RESULTS = FINAL_RESULTS/'gnn_cmax_conflict.csv'
CMAX_METHOD = 'Cmax-Conflict-GNN-MultiRobot'
CMAX_BEAM = 8
JOINT3_FORBIDDEN_SOURCES = frozenset((
    'IMITATION_MODEL', 'CYCLE_MODEL', 'EXPANDED_MODEL', 'IMITATION_RESULTS', 'CYCLE_RESULTS',
    'CONFLICT_RESULTS', 'CONFLICT_LOAD_RESULTS', 'LOADFIRST_RESULTS', 'ROUTEAWARE_RESULTS', 'PPO_RESULTS',
    'imitation_saved', 'imitation_best_teachers', 'imitation_cycle_train', 'imitation_train',
    'imitation_model', 'imitation_replay', 'imitation_evaluate', 'conflict_initial_hints',
    'conflict_compensation', 'conflict_construct', 'conflict_core_case', 'cycle_transfer_case',
    'routeaware_history', 'baseline_run', 'native_hga', 'generate_wag_assignments'))


def joint3_model(weights=None):
    """Original workload Actor plus four zero-initialised measured-feedback slots."""
    model = gnn.VNextGNN(workload_lookahead=True, joint_actions=True, feedback_outcomes=True)
    model.imitation_balance = False
    model.load_first_lambda = None
    model.route_aware = False
    if weights is None:
        source = torch.load(MODEL, map_location='cpu', weights_only=True)
        missing, extra = model.load_state_dict(source['models']['GNN_VNEXT_WORKLOAD']['state_dict'], strict=False)
        if missing != ['joint_weights'] or extra:
            raise ValueError('joint3 must start from the original GNN_VNEXT_WORKLOAD actor only')
        return model
    if weights['layer1.linear.weight'].shape[1] > gnn.NODE_DIM:
        model.enable_conflict_graph()
    model.load_state_dict(weights)
    return model


def joint3_called_names(function):
    import ast, inspect, textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    return names


def joint3_provenance():
    """Static and checkpoint audit of the model and every plan source."""
    functions = (joint3_model, joint3_feedback, joint3_feedback_features, joint3_load_limits,
                 joint3_training_limits, joint3_references, joint3_core, joint3_reconstruct,
                 joint3_case, joint3_train, joint3_validation)
    violations = {}
    for function in functions:
        hits = sorted(joint3_called_names(function) & JOINT3_FORBIDDEN_SOURCES)
        if hits:
            violations[function.__name__] = hits
    if violations:
        raise ValueError(f'joint3 audit found forbidden teacher/fallback sources: {violations}')
    reconstruct = joint3_called_names(joint3_reconstruct)
    case = joint3_called_names(joint3_case)
    train = joint3_called_names(joint3_train)
    shared = dict(reconstruct_uses_feedback_features='joint3_feedback_features' in reconstruct,
                  inference_uses_same_feedback='joint3_feedback' in case and 'joint3_reconstruct' in case,
                  training_uses_same_feedback='joint3_feedback' in train and 'joint3_reconstruct' in train)
    if not all(shared.values()):
        raise ValueError(f'joint3 B32 feedback is not shared by training and inference: {shared}')
    source = torch.load(MODEL, map_location='cpu', weights_only=True)
    state = source['models']['GNN_VNEXT_WORKLOAD']['state_dict']
    return dict(algorithm=JOINT3_METHOD, initialization='models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD',
        initialization_parameters=len(state), new_model=str(JOINT3_MODEL.relative_to(ROOT)),
        own_records=str(JOINT3_RESULTS.relative_to(ROOT)), forbidden_sources=sorted(JOINT3_FORBIDDEN_SOURCES),
        violations=violations, feedback_shared=shared,
        core='lns.select_routeaware_bottleneck (FEASIBLE) | conflict_core_partial (DEADLOCK)',
        reconstruction='gnn.vnext_joint_actions joint beam; mode+owner+position+order+direction together',
        labels='real B32 feasibility/Cmax; no HGA/WAG teacher and no HGA/WAG candidate source',
        load_rule='frozen HGA/WAG 1.05x ceiling; three-reference variant reported only')


def joint3_references():
    """Frozen HGA/WAG+VNS rows: only the fixed load ceiling and the final pairing."""
    references = {}
    for method, path, source in (('HGA', JOINT3_HGA_REFERENCE, JOINT3_HGA_SOURCE),
                                 ('WAG+VNS', JOINT3_WAG_REFERENCE, JOINT3_WAG_SOURCE)):
        for row in read_rows(path):
            if row.get('source') != source or row.get('status') != 'FEASIBLE':
                continue
            if row.get('layout_id') not in JOINT3_LAYOUTS:
                continue
            if not all(row.get(f'processing_r{robot}_s') not in (None, '') for robot in range(4)):
                continue
            values = [float(row[f'processing_r{robot}_s']) for robot in range(4)]
            record = dict(method=method, layout_id=row['layout_id'], seed=int(row['seed']),
                          cmax_s=float(row['cmax_s']), runtime_s=float(row.get('runtime_s') or 0.),
                          processing=values, peak=max(values), spread=max(values)-min(values))
            key = (row['layout_id'], method)
            # Duplicate rows of the same official frozen source keep the best Cmax.
            if key not in references or record['cmax_s'] < references[key]['cmax_s']:
                references[key] = record
    # validation_07 has no feasible frozen WAG+VNS row for any seed; that method
    # then contributes a feasibility result only and cannot supply a load ceiling.
    if not any((layout, 'HGA') in references for layout in JOINT3_LAYOUTS):
        raise ValueError('the frozen HGA reference is missing')
    return references


def joint3_load_limits(name, references, gnn_peak, gnn_spread=None):
    """The fixed formal ceiling; the three-reference variant is recorded, never used to accept."""
    methods = [method for method in ('HGA', 'WAG+VNS') if (name, method) in references]
    peaks, spreads, vectors = [], [], {}
    for method in methods:
        record = references[(name, method)]
        peaks.append(record['peak'])
        spreads.append(record['spread'])
        vectors[method] = record['processing']
    base = max(peaks)
    three = max(base, float(gnn_peak))
    gnn_spread = 0. if gnn_spread is None else float(gnn_spread)
    return dict(pmax_limit_s=1.05*base, spread_limit_s=max(max(spreads), .10*median(peaks)),
        temporary_pmax_limit_s=1.20*1.05*base, reference_processing=vectors,
        reference_peaks={method: references[(name, method)]['peak'] for method in methods},
        rule='final peak <= 1.05*largest feasible frozen HGA/WAG peak; '
             'spread <= max(frozen spreads, 10% median peak)',
        three_reference=dict(pmax_limit_s=1.05*three,
            spread_limit_s=max(max(spreads), gnn_spread, .10*median((*peaks, float(gnn_peak)))),
            temporary_pmax_limit_s=1.20*1.05*three, gnn_peak=float(gnn_peak), gnn_spread=gnn_spread,
            rule='same formula with this experiment original GNN construction as third reference; reported only'))


def joint3_relative_limits(state):
    """TRAIN-only relative guard (the training-label rule); used because TRAIN has no frozen reference."""
    peak = max(state.loads)
    spread = peak-min(state.loads)
    rule = 'TRAIN-only relative guard: Pmax <= 1.01*initial Pmax; spread <= initial spread+20 s'
    limits = dict(pmax_limit_s=1.01*peak, spread_limit_s=spread+20.,
                  temporary_pmax_limit_s=1.20*1.01*peak, reference_processing={},
                  reference_peaks={}, rule=rule)
    limits['three_reference'] = dict(limits, rule=rule)
    return limits


def joint3_training_limits(state):
    """TRAIN-only relative protection, the earlier formal Pmax<=1.01*initial variant."""
    peak = max(state.loads)
    return dict(pmax_limit_s=1.01*peak, spread_limit_s=max(state.loads)-min(state.loads)+20.,
                temporary_pmax_limit_s=1.20*1.01*peak)


def joint3_feedback(current, schedule, events, removed):
    """Measured B32 evidence, identical in training and inference."""
    removed = set(removed)
    observed_nodes = None
    pairs = []
    if schedule.feasible:
        pairs = [(event.delayed_block_id, blocker)
                 for event in events for _, blocker, intervals in event.blocker_intervals if intervals]
    else:
        pairs = [(blocked, blocker) for event in schedule.deadlock_blockages
                 for blocked in event.blocked_block_ids for group in event.blocker_block_ids
                 for blocker in group]
    temporal = dict(blocked_pairs=list(dict.fromkeys(pairs)),
                    completion=list(schedule.robot_completion) if schedule.feasible else [0.]*4,
                    robot_wait=[sum(op.duration for op in schedule.operations
                                    if op.robot_id == robot and op.kind is OperationKind.WAIT)
                                for robot in range(4)])
    if schedule.feasible and schedule.operations:
        summary = lns.schedule_feedback_summary(current.solution, schedule, events, CONFIG)
        return dict(source='B32 feasible WAIT/blocked intervals',
            **temporal, critical_robot=int(summary['critical_robot']),
            parent_wait={p: float(w) for p, w in summary['parent_wait'].items()
                         if w > CONFIG.numeric_epsilon},
            parent_blocked={p: float(w) for p, w in summary['parent_blocked'].items()
                            if w > CONFIG.numeric_epsilon},
            parent_critical_wait={p: float(w) for p, w in summary['parent_critical_wait'].items()
                                  if w > CONFIG.numeric_epsilon},
            parent_critical_blocked={p: float(w) for p, w in summary['parent_critical_blocked'].items()
                                     if w > CONFIG.numeric_epsilon},
            parent_processing={p: [float(x) for x in summary['parent_processing'][p]]
                               for p in removed if p in summary['parent_processing']})
    blocked = Counter()
    for event in schedule.deadlock_blockages:
        for parent in event.blocked_parent_ids:
            blocked[parent] += 1
        for parents in event.blocker_parent_ids:
            blocked.update(parents)
    # Processing is geometry, not a B32 label, so the same parent work vector is
    # available on a DEADLOCK origin and keeps the blocked-work interaction real.
    blocks = block_map(current.solution, CONFIG, scope=FORMAL_SCOPE_V2)
    processing = {parent: [0.]*4 for parent in removed}
    for robot, route in enumerate(current.solution.routes):
        for block in route.block_ids:
            parent = block.split('::', 1)[0]
            if parent in processing:
                processing[parent][robot] += CONFIG.process_time(blocks[block].length)
    return dict(source='B32 DEADLOCK blocked/blocker evidence', **temporal, critical_robot=None, parent_wait={},
                parent_blocked=dict(blocked), parent_critical_wait={}, parent_critical_blocked={},
                parent_processing=processing)


def joint3_feedback_features(current, solution, state, feedback, removed):
    """Candidate-specific B32 feedback interactions, never all-zero during training."""
    removed = set(removed)
    critical = feedback.get('critical_robot')
    if critical is None:
        cycles = [load + sum(arcs) for load, arcs in zip(state.loads, state.arc_costs)]
        critical = max(range(4), key=lambda robot: (cycles[robot], -robot))
    blocks = block_map(solution, CONFIG, scope=FORMAL_SCOPE_V2)
    owners = {block: robot for robot, route in enumerate(state.routes) for block in route}
    before = {block: robot for robot, route in enumerate(current.solution.routes) for block in route.block_ids}
    processing = {parent: [0.]*4 for parent in removed}
    for block, robot in owners.items():
        parent = block.split('::', 1)[0]
        if parent in processing:
            processing[parent][robot] += CONFIG.process_time(blocks[block].length)
    origin = feedback.get('parent_processing') or {}

    def moved(parents):
        total = 0.
        for parent in parents:
            if parent not in removed or parent not in origin:
                continue
            total += max(0., float(origin[parent][critical]) - processing[parent][critical])
        return total / 2500.

    waits = [parent for parent in feedback.get('parent_wait', {}) if parent in removed]
    blocked = [parent for parent in feedback.get('parent_blocked', {}) if parent in removed]
    cycle = (state.loads[critical] + sum(state.arc_costs[critical])) / 2500.
    changed = sum(1 for block, robot in owners.items() if robot == critical
                  and block.split('::', 1)[0] in removed and before.get(block) != critical) / 4.
    return [moved(waits), moved(blocked), cycle, changed]


def joint3_core(current, schedule, model, space, embeddings, blockers, *, seed, q):
    """FEASIBLE: route-aware bottleneck; DEADLOCK: every measured blocked/blocker parent."""
    if schedule.status is ScheduleStatus.DEADLOCK:
        partial, feedback = conflict_core_partial(current, schedule, model, space, embeddings, seed=seed, q=q)
        return partial, dict(feedback, operator='conflict_core_partial')
    if schedule.feasible:
        partial, feedback = lns.select_routeaware_bottleneck(current.solution, current.directions, schedule,
            blockers, CONFIG, q=q, seed=seed, assignment_space=space)
        return partial, dict(feedback, operator='select_routeaware_bottleneck')
    return None, dict(reason='B32 returned neither FEASIBLE nor DEADLOCK')


def joint3_route_lb(plan):
    """Exact directed route-cycle lower bound max_r(P_r+MOVE_r) for a complete plan."""
    return cycle_values(cycle_route_state(plan))[1]


def joint3_safe_prune(lb, incumbent_cmax):
    """Safe pruning: only an exact route-cycle LB that already reaches the measured Cmax."""
    return math.isfinite(incumbent_cmax) and lb >= incumbent_cmax


def incumbent_cmax(entries):
    """Only measured, feasible Cmax values supply an incumbent bound."""
    return min((entry[1].cmax for entry in entries
                if entry[1].feasible and entry[1].cmax is not None), default=math.inf)


def joint3_assignment_changes(before, after):
    """Actual weld-length ownership changes, excluding split-only task renaming."""
    old, new = (block_map(plan.solution, CONFIG, scope=FORMAL_SCOPE_V2) for plan in (before, after))
    old_owner = {b:r for r,route in enumerate(before.solution.routes) for b in route.block_ids}
    new_owner = {b:r for r,route in enumerate(after.solution.routes) for b in route.block_ids}
    changes = {}
    for block in old.values():
        axis = int(abs(block.end[1]-block.start[1]) > abs(block.end[0]-block.start[0]))
        lo, hi = sorted((block.start[axis],block.end[axis]))
        for child in new.values():
            if child.parent_id != block.parent_id or old_owner[block.block_id] == new_owner[child.block_id]:
                continue
            lower, upper = sorted((child.start[axis],child.end[axis]))
            overlap = min(hi,upper)-max(lo,lower)
            if overlap > CONFIG.numeric_epsilon:
                length = overlap/(hi-lo)*block.length
                changes.setdefault(block.parent_id,[]).append(dict(
                    source=old_owner[block.block_id],destination=new_owner[child.block_id],length_m=length))
    return changes


@torch.no_grad()
def joint3_reconstruct(parents, model, space, embeddings, current, removed, limits, feedback, *,
                       width=JOINT3_BEAM):
    """Bounded joint beam over whole removed parents; no rule fixes a migration target."""
    if getattr(model, 'conflict_graph', False) and not 3 <= len(set(removed)) <= 6:
        return [], dict(reason='Cmax reconstruction requires 3 to 6 distinct whole parents',
                        legal=0, truncated=0)
    started = time.perf_counter()
    pending = set(removed)
    selected = {p.parent_id: p for p in current.solution.patterns if p.parent_id not in pending}
    blocks = {b.block_id: b for p in selected.values() for b in space.children[p]}
    nodes = {b.block_id: space.block_nodes[p, b.block_id] for p in selected.values() for b in space.children[p]}
    routes = lns._remove_parent_set(current.solution, pending).routes
    fixed = {b: d for route, bits in zip(current.solution.routes, current.directions)
             for b, d in zip(route.block_ids, bits) if b in blocks}
    state = lns.oriented_route_state(routes, tuple(tuple(fixed[b] for b in route) for route in routes),
                                     blocks, CONFIG)
    origin_assignment = {action.parent_id: action
                         for action in gnn.parent_assignment_targets(current.solution, CONFIG)}
    feedback = dict(feedback, observed_nodes={b.block_id: space.block_nodes[p, b.block_id]
                    for p in current.solution.patterns for b in space.children[p]})
    beam = [dict(state=state, blocks=dict(blocks), nodes=dict(nodes), selected=dict(selected), actions=(),
                 plans=(), logits=(), pending=set(pending))]
    truncated = 0
    for _ in range(len(removed)):
        expanded = []
        for node in beam:
            plans, scores = gnn.vnext_joint_actions(space, embeddings, model, node['state'], node['blocks'],
                node['nodes'], node['pending'], CONFIG, conflict_core=True, feedback=feedback)
            seen_structures = set()
            for plan, score in zip(plans, scores.tolist()):
                assignment = space.actions[plan.assignment_index]
                if assignment.parent_id not in node['pending'] or not math.isfinite(score):
                    continue
                # Keep one representative per (parent, legal mode, robot ownership):
                # direction-only duplicates are not a distinct joint structure.
                structure = (assignment.parent_id, assignment.pattern.pattern_id, assignment.robots)
                if structure in seen_structures:
                    continue
                seen_structures.add(structure)
                before = origin_assignment[assignment.parent_id]
                # Beam diversity: a rebuild that changes a parent mode or its robot
                # ownership is kept ahead of a rebuild that restores the origin, so
                # the limited beam is not three copies of the incumbent plan.
                changed = int(assignment.pattern.pattern_id != before.pattern.pattern_id
                              or assignment.robots != before.robots)
                children = space.children[assignment.pattern]
                expanded.append(dict(state=plan.state, changes=node.get('changes', 0)+changed,
                    blocks={**node['blocks'], **{b.block_id: b for b in children}},
                    nodes={**node['nodes'], **{b.block_id: space.block_nodes[assignment.pattern, b.block_id]
                                               for b in children}},
                    selected={**node['selected'], assignment.parent_id: assignment.pattern},
                    actions=(*node['actions'], *plan.actions), plans=(*node['plans'], plan),
                    logits=(*node['logits'], score), pending=node['pending']-{assignment.parent_id}))
        if not expanded:
            return [], dict(reason='no legal joint insertion',
                            removed=list(removed), truncated=truncated)
        unique = {}
        for node in expanded:
            key = (node['state'].routes, node['state'].directions)
            best = unique.get(key)
            if best is None or sum(node['logits']) > sum(best['logits']):
                unique[key] = node
        ranked = sorted(unique.values(), key=lambda node: (
            -sum(node['logits'])/max(1, len(node['logits'])), node['state'].travel))
        truncated += max(0, len(ranked)-max(1, width))
        # Keep changed structures first but always retain one unchanged completion:
        # that keeps a legal rebuild in the beam even if every mode/owner change of
        # this core is rejected by canonicalization.
        size = max(1, width)
        # Mode/owner structure, rather than direction-only or insertion-order
        # copies, receives a beam slot. Robot-flow diversity preserves cyclic
        # 3/4-robot exchanges while all decisions still use this same GNN.
        groups = {}
        for node in ranked:
            structure = tuple(sorted((space.actions[p.assignment_index].parent_id,
                space.actions[p.assignment_index].pattern.pattern_id,
                space.actions[p.assignment_index].robots) for p in node['plans']))
            groups.setdefault(structure, node)
        representatives = list(groups.values())
        beam, flows = [], set()
        for node in representatives:
            flow = tuple(sorted({(old, new) for p in node['plans']
                for old in origin_assignment[space.actions[p.assignment_index].parent_id].robots
                for new in space.actions[p.assignment_index].robots if old != new}))
            if flow not in flows:
                beam.append(node); flows.add(flow)
            if len(beam) >= size:
                break
        for node in representatives:
            if all(node is not kept for kept in beam):
                beam.append(node)
            if len(beam) >= size:
                break
        ordered = beam
    results, seen_plans, rejected = [], set(), Counter()
    for node in ordered[:2*max(1, width)]:
        try:
            if not all(node['state'].routes):
                raise ValueError('LEGAL_FOUR_ROBOT_COVER_NOT_COMPLETED')
            solution = canonicalize(parents, node['selected'].values(), dict(enumerate(node['state'].routes)),
                                    CONFIG, scope=FORMAL_SCOPE_V2)
            if tuple(r.block_ids for r in solution.routes) != node['state'].routes or \
                    set(solution.patterns) != set(node['selected'].values()):
                raise ValueError('canonicalization changed the directed joint construction')
            for robot, route in enumerate(routes):
                kept = set(route)
                if tuple(b for b in node['state'].routes[robot] if b in kept) != route:
                    raise ValueError('reconstruction changed retained route order')
            # The native joint insertion decides a new child's direction together with
            # the directions of its immediate neighbours.  That is allowed only inside
            # the reconstructed core's new neighbourhood; no retained parent elsewhere
            # may be re-optimised, and there is no separate direction pass.
            touched = set()
            for parent in removed:
                touched.update(block.block_id for block in space.children[node['selected'][parent]])
            allowed = set(touched)
            for route in node['state'].routes:
                for index, block in enumerate(route):
                    if block in touched:
                        for offset in (-1, 1):
                            if 0 <= index+offset < len(route):
                                allowed.add(route[index+offset])
            actual = {block: direction for route, bits in zip(node['state'].routes, node['state'].directions)
                      for block, direction in zip(route, bits)}
            outside = {block for block in fixed if block in actual and actual[block] != fixed[block]} - allowed
            if outside:
                raise ValueError('direction change outside the reconstructed core neighbourhood')
        except ValueError as error:
            rejected[str(error)] += 1
            truncated += 1
            continue
        key = (solution.patterns, solution.routes, node['state'].directions)
        if key in seen_plans:
            truncated += 1
            continue
        seen_plans.add(key)
        extra = joint3_feedback_features(current, solution, node['state'], feedback, removed)
        patched = tuple(plan._replace(outcome=torch.cat(
            (plan.outcome[:6], plan.outcome.new_tensor(extra))))
            for plan in node['plans'])
        # The terminal feedback slots are identical for each step. Reuse the
        # already computed per-step GNN scores; replay uses those same graphs.
        score = (sum(node['logits'])/max(1, len(node['logits']))
                 + float(model.joint_weights[6:] @ model.joint_weights.new_tensor(extra)))
        result = gnn.JointCompletionResult(solution, node['state'].directions, node['actions'],
                                           time.perf_counter()-started, oriented_state=node['state'])
        results.append(dict(plan=result, score=score, joint_plans=patched, feedback_features=extra,
            actions=[dict(block_id=a.block_id, robot=a.robot, position=a.position, direction=a.direction,
                          neighbour_changes=list(a.neighbour_changes)) for plan in node['plans'] for a in plan.actions],
            lb=joint3_route_lb(result)))
    results.sort(key=lambda item: (-item['score'], item['lb'], item['plan'].solution.patterns))
    return results, dict(reason='ok' if results else 'no legal joint completion', removed=sorted(removed),
                         truncated=truncated, legal=len(results), rejected=dict(rejected))


def joint3_pool(entries, latest, *, width=6):
    """Retain distinct real states: load-qualified Cmax, load margin, best Cmax, deadlock progress."""
    by_key = {}
    for entry in (*entries, *latest):
        if entry[1].status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
            continue
        by_key[plan_key(entry[0].solution, entry[0].directions)] = entry
    feasible = [entry for entry in by_key.values() if entry[1].feasible]
    feasible_ok = [entry for entry in feasible if int(entry[2].get('load_allowed') or 0)]
    deadlocks = [entry for entry in by_key.values() if entry[1].status is ScheduleStatus.DEADLOCK]
    latest_keys = [plan_key(entry[0].solution, entry[0].directions) for entry in reversed(latest)
                   if plan_key(entry[0].solution, entry[0].directions) in by_key]
    categories = [
        sorted(feasible_ok, key=lambda entry: entry[1].cmax),
        sorted(feasible_ok, key=lambda entry: (conflict_load_balance(entry)[0], entry[1].cmax)),
        sorted(feasible, key=lambda entry: entry[1].cmax),
        sorted(feasible, key=lambda entry: (conflict_load_balance(entry)[0], entry[1].cmax)),
        sorted(deadlocks, key=lambda entry: tuple(-value for value in conflict_deadlock_progress(entry))),
        [by_key[key] for key in latest_keys[:1]],
    ]
    cursors = [0]*len(categories)
    chosen, keys = [], set()
    while len(chosen) < width:
        added = False
        for index, category in enumerate(categories):
            while cursors[index] < len(category):
                entry = category[cursors[index]]
                cursors[index] += 1
                key = plan_key(entry[0].solution, entry[0].directions)
                if key in keys:
                    continue
                chosen.append(entry)
                keys.add(key)
                added = True
                break
            if len(chosen) >= width:
                break
        if not added:
            break
    return chosen


def joint3b_ratio(entry):
    """Normalized load pressure max(Pmax/limit, spread/limit); <=1 means load-qualified."""
    return conflict_load_balance(entry)[0]


def joint3b_deadlock_signature(entry):
    """Measured blocked-parent set of a DEADLOCK state, used only for core diversity."""
    return frozenset(parent for event in entry[1].deadlock_blockages
                     for parent in event.blocked_parent_ids)


def joint3b_pool(entries, latest, *, width=6, signature=joint3b_deadlock_signature):
    """Six protected states: a DEADLOCK can never displace a FEASIBLE state."""
    by_key = {}
    for entry in (*entries, *latest):
        if entry[1].status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
            continue
        by_key[plan_key(entry[0].solution, entry[0].directions)] = entry
    feasible = [entry for entry in by_key.values() if entry[1].feasible]
    balanced = [entry for entry in feasible if int(entry[2].get('load_allowed') or 0)]
    deadlocks = sorted((entry for entry in by_key.values()
                        if entry[1].status is ScheduleStatus.DEADLOCK),
                       key=lambda entry: tuple(-value for value in conflict_deadlock_progress(entry)))
    first = deadlocks[0] if deadlocks else None
    first_signature = signature(first) if first is not None else None
    other = next((entry for entry in deadlocks if signature(entry) != first_signature), None)
    latest_keys = [plan_key(entry[0].solution, entry[0].directions) for entry in reversed(latest)
                   if plan_key(entry[0].solution, entry[0].directions) in by_key]
    categories = [
        sorted(balanced, key=lambda entry: entry[1].cmax),
        sorted(feasible, key=lambda entry: entry[1].cmax),
        sorted(feasible, key=lambda entry: (joint3b_ratio(entry), entry[1].cmax)),
        [first] if first is not None else [],
        [other] if other is not None else [],
        [by_key[key] for key in latest_keys[:1]],
    ]
    fallback = [*sorted(feasible, key=lambda entry: entry[1].cmax), *deadlocks,
                *sorted(by_key.values(), key=lambda entry: (0, entry[1].cmax) if entry[1].feasible
                        else (1, *(-value for value in conflict_deadlock_progress(entry))))]
    chosen, keys = [], set()

    def add(entry):
        if entry is None or len(chosen) >= width:
            return False
        key = plan_key(entry[0].solution, entry[0].directions)
        if key in keys:
            return False
        chosen.append(entry)
        keys.add(key)
        return True

    cursors = [0]*len(categories)
    while len(chosen) < width:
        added = False
        for index, category in enumerate(categories):
            while cursors[index] < len(category):
                if add(category[cursors[index]]):
                    added = True
                cursors[index] += 1
                if added or len(chosen) >= width:
                    break
            if len(chosen) >= width:
                break
        if not added:
            break
    for entry in fallback:
        if len(chosen) >= width:
            break
        add(entry)
    return chosen


def joint3b_deadlock_choice(current, schedule, model, space, embeddings, *, seed, q):
    """Rank real B32 blocked/blocker cores by measured coverage; the GNN only breaks ties."""
    known = set(space.parent_ids)
    all_blocks = block_map(current.solution, CONFIG, scope=FORMAL_SCOPE_V2)
    records = []
    for event in schedule.deadlock_blockages:
        for operation, parents in zip(event.blocker_operations, event.blocker_parent_ids):
            blocked = set(event.blocked_parent_ids) & known
            blockers = set(parents) & known
            if operation is None or not blocked or not blockers:
                continue
            records.append(dict(event=event, operation=operation, core=blocked | blockers,
                                blocked=blocked, blockers=blockers))
    if not records:
        return None, dict(source='B32 DEADLOCK', reason='no mapped blocked/blocker parent core')
    union = set().union(*(record['core'] for record in records))
    partial = lns._remove_parent_set(current.solution, union)
    kept = {p.parent_id: p for p in current.solution.patterns if p.parent_id not in union}
    kept_blocks = {b.block_id: b for p in kept.values() for b in space.children[p]}
    old_bits = {b: d for route, bits in zip(current.solution.routes, current.directions)
                for b, d in zip(route.block_ids, bits)}
    state = lns.oriented_route_state(partial.routes,
        tuple(tuple(old_bits[b] for b in route) for route in partial.routes), kept_blocks, CONFIG)
    indices, logits = model.assignment_logits(space, embeddings, state, kept_blocks, union, CONFIG)
    priorities = {parent: -math.inf for parent in union}
    for index, score in zip(indices.tolist(), logits.tolist()):
        parent = space.actions[index].parent_id
        priorities[parent] = max(priorities[parent], score)

    def gnn_priority(record):
        return sum(priorities[parent] for parent in record['core'])/len(record['core'])

    records.sort(key=lambda record: (-len(record['core']), -len(record['blocked']),
                                     -gnn_priority(record), record['event'].robot_id,
                                     record['operation'].operation_id))
    tier = [record for record in records
            if (len(record['core']), len(record['blocked']))
            == (len(records[0]['core']), len(records[0]['blocked']))]
    chosen = tier[seed % len(tier)]
    removed = set(chosen['core'])
    # Necessary neighbours come only from the blocked MOVE endpoints, not from
    # every core parent, so the reconstruction does not grow without evidence.
    blocked_blocks = set(chosen['event'].blocked_block_ids)
    neighbours = []
    for route in current.solution.routes:
        for position, block in enumerate(route.block_ids):
            if block in blocked_blocks:
                for offset in (-1, 1):
                    if 0 <= position+offset < len(route.block_ids):
                        neighbours.append(all_blocks[route.block_ids[position+offset]].parent_id)
    for parent in dict.fromkeys(neighbours):
        if len(removed) >= max(q, len(chosen['core'])):
            break
        removed.add(parent)
    if not removed:
        return None, dict(source='B32 DEADLOCK', reason='empty measured deadlock core')
    return lns._remove_parent_set(current.solution, removed), dict(
        source='structured B32 DEADLOCK (blocked/blocker coverage first)',
        operator='joint3b_deadlock_core', blocked_robot=chosen['event'].robot_id,
        blocked_operation=asdict(chosen['event'].blocked_operation),
        blocked_parents=sorted(chosen['blocked']), blocker_parents=sorted(chosen['blockers']),
        blocked_MOVE_endpoints=chosen['event'].blocked_block_ids,
        blocker_operation=asdict(chosen['operation']), core=sorted(chosen['core']),
        gnn_core_priority={parent: priorities[parent] for parent in chosen['core']},
        evidence_coverage=len(chosen['core']), blocked_count=len(chosen['blocked']),
        necessary_neighbours=sorted(removed-chosen['core']),
        evidence_source=schedule.deadlock_evidence_source)


def joint3b_core(current, schedule, model, space, embeddings, blockers, *, seed, q):
    if schedule.status is ScheduleStatus.FEASIBLE:
        partial, feedback = lns.select_routeaware_bottleneck(current.solution, current.directions, schedule,
            blockers, CONFIG, q=q, seed=seed, assignment_space=space)
        return partial, dict(feedback, operator='select_routeaware_bottleneck')
    if schedule.status is ScheduleStatus.DEADLOCK:
        return joint3b_deadlock_choice(current, schedule, model, space, embeddings, seed=seed, q=q)
    return None, dict(reason='B32 returned neither FEASIBLE nor DEADLOCK')


def joint3b_select_final(entries, tolerance=JOINT3B_TOLERANCE):
    """Return the true minimum feasible B32 Cmax; load is diagnostic only."""
    feasible = [entry for entry in entries if entry[1].feasible]
    balanced = [entry for entry in feasible if int(entry[2].get('load_allowed') or 0)]
    best_balanced = min(balanced, key=lambda entry: entry[1].cmax) if balanced else None
    best_cmax = min(feasible, key=lambda entry: entry[1].cmax) if feasible else None
    best_balance = min(feasible, key=lambda entry: (joint3b_ratio(entry), entry[1].cmax)) if feasible else None
    ratio = ((best_balanced[1].cmax-best_cmax[1].cmax)/best_cmax[1].cmax*100.
             if best_balanced is not None and best_cmax is not None else None)
    chosen, choice = best_cmax, 'cmax' if best_cmax is not None else None
    return dict(chosen=chosen, choice=choice, best_balanced=best_balanced, best_cmax=best_cmax,
                best_balance=best_balance, r_c_pct=ratio)


def joint3b_write(rows):
    """The new-stage CSV; the Joint3 CSV is never written by this path."""
    import msvcrt
    if not rows:
        return
    with JOINT3B_RESULTS.open('a+', encoding='utf-8', newline='') as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            stream.seek(0, 2)
            writer = csv.DictWriter(stream, fieldnames=JOINT3B_FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in JOINT3B_FIELDS})
            stream.flush()
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def joint3b_failure_final(name, parent_count, seed, entries, *, calls, started, run, phase,
                          method=JOINT3B_METHOD, variant='joint3b'):
    """Explicit failure row: no DEADLOCK is ever reported as a successful final."""
    deadlocks = [entry for entry in entries if entry[1].status is ScheduleStatus.DEADLOCK]
    best = max(deadlocks, key=conflict_deadlock_progress) if deadlocks else None
    return dict(stage='final', layout_id=name, seed=seed, method=method, variant=variant,
        phase=phase, parent_count=parent_count, status='NO_FEASIBLE_SOLUTION', cmax_s=None,
        load_allowed=0, load_allowed_three_ref=0, b32_calls=calls,
        runtime_s=time.perf_counter()-started, candidate_id='final', run_id=run,
        final_choice='none', removed_parent_ids='[]', feedback='{}',
        diagnostic=json.dumps(dict(reason='no B32 FEASIBLE candidate inside the budget',
            final_status='NO_FEASIBLE_SOLUTION', evaluated=len(entries),
            deadlock_candidates=len(deadlocks),
            best_deadlock=None if best is None else dict(
                candidate_id=best[2]['candidate_id'], progress=conflict_deadlock_progress(best),
                blocked_parents=sorted(joint3b_deadlock_signature(best)),
                blocked_operations=len(best[1].deadlock_blockages),
                deadlock_blockages=json.loads(best[2]['deadlock_blockages'] or '[]'),
                feedback=json.loads(best[2]['feedback'] or '{}'))), ensure_ascii=False))


def joint3_write(rows):
    """One existing CSV; an ordinary byte lock permits concurrent validation workers."""
    import msvcrt
    if not rows:
        return
    with JOINT3_RESULTS.open('a+', encoding='utf-8', newline='') as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            stream.seek(0, 2)
            writer = csv.DictWriter(stream, fieldnames=JOINT3_FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in JOINT3_FIELDS})
            stream.flush()
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def joint3_training_write(rows):
    if not rows:
        return
    with JOINT3_TRAINING_RESULTS.open('a' if JOINT3_TRAINING_RESULTS.exists() else 'w',
                                      encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=JOINT3_TRAINING_FIELDS)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerows({key: row.get(key) for key in JOINT3_TRAINING_FIELDS} for row in rows)
        stream.flush()


@torch.no_grad()
def joint3_case(name, parents, seed, model, references, *, budget=JOINT3_BUDGET,
                max_attempts=JOINT3_ATTEMPTS, limits=None, write=True,
                variant='joint3', sink='joint3', phase='validation',
                tolerance=JOINT3B_TOLERANCE, limit_mode='reference'):
    cmax_driven = getattr(model, 'conflict_graph', False)
    method = (CMAX_METHOD if cmax_driven else JOINT4_METHOD if variant == 'joint4' else
              JOINT3B_METHOD if variant == 'joint3b' else JOINT3_METHOD)
    writer = (cmax_write if sink == 'cmax' else joint4_write if sink == 'joint4' else
              joint3b_write if sink == 'loadkeep' else joint3_write)
    pool_fn = (joint4_pool if variant == 'joint4' else
               joint3b_pool if variant == 'joint3b' else joint3_pool)
    core_fn = (joint4_core if variant == 'joint4' else
               joint3b_core if variant == 'joint3b' else joint3_core)
    results_path = (CMAX_RESULTS if sink == 'cmax' else JOINT4_RESULTS if sink == 'joint4' else
                    JOINT3B_RESULTS if sink == 'loadkeep' else JOINT3_RESULTS)
    finished = [row for row in read_rows(results_path)
                if row.get('layout_id') == name and row.get('method') == method
                and row.get('stage') == 'final' and row.get('phase') == phase
                and int(row.get('seed') or 0) == seed]
    if finished:
        return finished[-1]
    started = time.perf_counter()
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    seen, blockers, visited, entries, pending_rows = {}, {}, set(), [], []
    calls, attempts, stale = 0, 0, 0
    deadlock_origins, deadlock_repairs = 0, 0
    native_cache = {}
    core_q = JOINT4_CORE if variant == 'joint4' else JOINT3_CORE
    run = f'{name}-{seed}-{int(started)}'
    failures, core_sizes = Counter(), Counter()

    def flush():
        if write and pending_rows:
            writer(pending_rows)
            pending_rows.clear()

    initial = build_joint(parents, model, space=space, embeddings=embeddings)
    if initial.solution is None:
        final = dict(stage='final', layout_id=name, seed=seed, method=method,
            variant=variant, phase=phase, parent_count=len(parents), status='CONSTRUCTION_FAILED',
            cmax_s=None, load_allowed=0, runtime_s=time.perf_counter()-started, b32_calls=0,
            candidate_id='final', run_id=run,
            diagnostic=json.dumps(dict(reason=initial.rejection_reason, algorithm=method)))
        if write:
            writer([final])
        return final
    gnn_state = cycle_route_state(initial)
    gnn_peak = max(gnn_state.loads)
    gnn_spread = gnn_peak-min(gnn_state.loads)
    limits = (cmax_limits() if cmax_driven else joint3_relative_limits(gnn_state) if limit_mode == 'relative' else
              joint3_load_limits(name, references, gnn_peak, gnn_spread)) if limits is None else limits

    def current_incumbent_cmax():
        return incumbent_cmax(entries)

    def evaluate(result, candidate_id, *, origin=None, removed=(), feedback=None, construction=None,
                 stage='candidate', operator='joint3'):
        nonlocal calls
        key = plan_key(result.solution, result.directions)
        if key in visited:
            return None
        visited.add(key)
        lb = joint3_route_lb(result)
        evaluation = time.perf_counter()
        schedule, called = evaluate_once(result, seen, blocker_cache=blockers)
        elapsed = time.perf_counter()-evaluation
        calls += int(called)
        if schedule is None or schedule.status not in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK):
            return None
        modes, owners = ({}, {}) if origin is None else conflict_changes(origin[0], result, removed)
        actual_transfers = (joint3_assignment_changes(origin[0],result)
                            if origin is not None and cmax_driven else {})
        cross_robot = (len(actual_transfers) if cmax_driven else sum(1 for change in owners.values()
                          if {robot for _, robot in change['before']}
                          != {robot for _, robot in change['after']}))
        row = routeaware_result_row(name, method, result, schedule, blockers.get(key, ()),
            stage=stage, candidate_id=candidate_id,
            origin_id='' if origin is None else origin[2]['candidate_id'], operator=operator,
            removed_parent_ids=json.dumps(list(removed)), feedback=json.dumps(feedback or {}),
            construct_s=result.inference_seconds, b32_s=elapsed if called else 0.,
            runtime_s=time.perf_counter()-started, b32_calls=int(called),
            gnn_score=(construction or {}).get('score'), load_limits=json.dumps(limits),
            mode_changes=json.dumps(modes), assignment_changes=json.dumps(owners),
            rebuild_actions=json.dumps((construction or {}).get('actions', [])),
            route_lb_s=lb, incumbent_cmax_s=current_incumbent_cmax(),
            beam_width=CMAX_BEAM if cmax_driven else JOINT4_BEAM if variant == 'joint4' else JOINT3_BEAM,
            core_size=len(removed),
            feedback_features=json.dumps((construction or {}).get('feedback_features')),
            native_s=(construction or {}).get('native_cmax'),
            native_status=(construction or {}).get('native_status'),
            transfer_robot=(construction or {}).get('transfer_robot'),
            receiver_robot=(construction or {}).get('receiver_robot'),
            core_kind=(construction or {}).get('core_kind'),
            cross_robot_changes=cross_robot,
            load_allowed_three_ref=None, load_limits_three_ref=json.dumps(limits['three_reference']),
            successful_repair=int(origin is not None and origin[1].status is ScheduleStatus.DEADLOCK
                                  and schedule.feasible),
            transfer_count=len(actual_transfers), transfer_details=json.dumps(actual_transfers),
            pool_ids='', run_id=run)
        row.update(seed=seed, load_allowed=int(conflict_load_allowed(row, limits)),
                   load_allowed_three_ref=int(conflict_load_allowed(row, limits['three_reference'])),
                   variant=variant, phase=phase)
        if stage != 'final':
            row['schedule'] = None
            row['wait_blockers'] = None
        entry = (result, schedule, row)
        entries.append(entry)
        if write:
            pending_rows.append(row)
        return entry

    initial_native = joint4_native_value(initial) if variant == 'joint4' and not cmax_driven else (None, None)
    initial_entry = evaluate(initial, 'initial0', stage='initial', operator='original-GNN-construction',
                             construction=dict(native_cmax=initial_native[0],
                                               native_status=initial_native[1],
                                               core_kind='initial') if variant == 'joint4' else None)
    if variant == 'joint4' and initial_entry is not None:
        native_cache[plan_key(initial.solution, initial.directions)] = initial_native[0]
    if initial_entry is None:
        raise ValueError(f'{name}: the original GNN construction has no B32 schedule')
    print('JOINT3_INITIAL', json.dumps(dict(layout_id=name, seed=seed, status=initial_entry[1].status.value,
        cmax_s=initial_entry[1].cmax if initial_entry[1].feasible else None,
        pmax_s=float(initial_entry[2]['max_processing_s']),
        pmax_limit_s=limits['pmax_limit_s'], pmax_limit_three_ref_s=limits['three_reference']['pmax_limit_s'],
        load_allowed=initial_entry[2]['load_allowed'],
        load_allowed_three_ref=initial_entry[2]['load_allowed_three_ref'])), flush=True)
    flush()
    pool = pool_fn([], [initial_entry])
    expanded = set()
    while calls < budget and attempts < max_attempts and stale < JOINT3_STALE:
        attempts += 1
        origin = core = feedback = removed = None
        if variant in ('joint3b', 'joint4'):
            # Prefer a pool state whose measured core for this round is still
            # unexpanded, so the same deadlock measurement is not repaired twice.
            for offset in range(len(pool)):
                candidate = pool[(attempts-1+offset) % len(pool)]
                candidate_key = plan_key(candidate[0].solution, candidate[0].directions)
                candidate_core, candidate_feedback = core_fn(candidate[0], candidate[1], model, space,
                    embeddings, blockers.get(candidate_key, ()), seed=seed+attempts*7919,
                    q=3+(attempts-1)%4 if cmax_driven else core_q)
                if candidate_core is None:
                    continue
                candidate_removed = tuple(candidate_core.removed_parent_ids)
                if (candidate_key, tuple(sorted(candidate_removed))) in expanded:
                    continue
                origin, core, feedback, removed = (candidate, candidate_core, candidate_feedback,
                                                   candidate_removed)
                break
            if origin is None:
                failures['every measured core of the current pool is expanded'] += 1
                stale += 1
                continue
        else:
            origin = pool[(attempts-1) % len(pool)]
            current, schedule, _ = origin
            origin_key = plan_key(current.solution, current.directions)
            core, feedback = core_fn(current, schedule, model, space, embeddings,
                blockers.get(origin_key, ()), seed=seed+attempts*7919, q=core_q)
            if core is None:
                failures[feedback.get('reason', 'empty measured core')] += 1
                stale += 1
                continue
            removed = tuple(core.removed_parent_ids)
        current, schedule, origin_row = origin
        origin_key = plan_key(current.solution, current.directions)
        expansion = (origin_key, tuple(sorted(removed)))
        if expansion in expanded:
            # This measured core of this state was already rebuilt; do not spend
            # reconstruction time on a repeat, and do not relax any filter.
            failures['measured core already expanded'] += 1
            stale += 1
            continue
        expanded.add(expansion)
        core_sizes[len(removed)] += 1
        if schedule.status is ScheduleStatus.DEADLOCK:
            deadlock_origins += 1
        measured = joint3_feedback(current, schedule, blockers.get(origin_key, ()), removed)
        beam_width = CMAX_BEAM if cmax_driven else JOINT4_BEAM if variant == 'joint4' else JOINT3_BEAM

        def rebuild(removed_set):
            local = joint3_feedback(current, schedule, blockers.get(origin_key, ()), removed_set)
            plans, detail_ = joint3_reconstruct(parents, model, space, embeddings, current, removed_set,
                                                limits, local, width=beam_width)
            return plans, detail_, local

        candidates, detail, measured = rebuild(removed)
        if variant == 'joint4' and not candidates and len(removed) > 3:
            # A five-parent core can be unreconstructable on some layouts (a sibling
            # pair would merge and break the directed encoding).  Shrink the measured
            # core instead of dropping the round; the operator is unchanged.
            for size in (4, 3):
                removed = tuple(removed[:size])
                candidates, detail, measured = rebuild(removed)
                if candidates:
                    failures[f'core shrank to {size} parents'] += 1
                    break
        if not candidates:
            failures[detail.get('reason', 'no joint beam completion')] += 1
            stale += 1
            continue
        incumbent = current_incumbent_cmax()
        # Safe pruning only: the exact route-cycle lower bound already reaches the
        # best measured feasible B32 Cmax. No LB-versus-LB filter is applied.
        admissible = [candidate for candidate in candidates
                      if not joint3_safe_prune(candidate['lb'], incumbent)
                      and plan_key(candidate['plan'].solution, candidate['plan'].directions) not in visited]
        admissible.sort(key=lambda candidate: (-candidate['score'], candidate['lb']))
        if variant == 'joint4' and not cmax_driven:
            # The existing native coordinator screens the GNN proposals; only the
            # leading distinct structures spend one of the 100 B32 evaluations.
            if origin_key not in native_cache:
                native_cache[origin_key] = joint4_native_value(current)[0]
            admissible = joint4_screen(admissible, native_cache[origin_key])[:JOINT4_B32_PER_ROUND]
        elif cmax_driven:
            admissible = admissible[:JOINT4_B32_PER_ROUND]
        core_info = dict(transfer_robot=feedback.get('transfer_robot'),
                         receiver_robot=feedback.get('receiver_robot'),
                         core_kind=feedback.get('core_kind', feedback.get('operator')))
        new = []
        for index, candidate in enumerate(admissible):
            if calls >= budget:
                break
            entry = evaluate(candidate['plan'], f'{run}-r{attempts}-{index}', origin=origin, removed=removed,
                feedback=dict(measured, involved_robots=feedback.get('involved_robots'),
                              receiver_robots=feedback.get('receiver_robots'),
                              candidate_lb_s=candidate['lb'], incumbent_cmax_s=incumbent,
                              beam_truncated=detail['truncated'], beam_legal=detail['legal']),
                construction=dict(candidate, **core_info),
                operator='deadlock-core-rebuild' if schedule.status is ScheduleStatus.DEADLOCK
                else 'bottleneck-core-rebuild')
            if entry is not None:
                new.append(entry)
        if not new:
            failures['beam candidates pruned or duplicate'] += 1
            stale += 1
        else:
            stale = 0
            pool = pool_fn(pool, new)
            deadlock_repairs += sum(int(entry[2].get('successful_repair') or 0) for entry in new)
        print('JOINT3_ROUND', json.dumps(dict(layout_id=name, seed=seed, round=attempts,
            core=list(removed), core_size=len(removed), candidates=len(candidates),
            admissible=len(admissible), b32_calls=calls, incumbent_s=None if not math.isfinite(incumbent) else incumbent,
            pool=[entry[2]['candidate_id'] for entry in pool])), flush=True)
        flush()
    state = joint3b_select_final(entries, tolerance)
    if variant in ('joint3b', 'joint4'):
        if state['chosen'] is None:
            # Final deadlock protection: an explicit failure, never a DEADLOCK final.
            final = joint3b_failure_final(name, len(parents), seed, entries, calls=calls,
                                          started=started, run=run, phase=phase,
                                          method=method, variant=variant)
            if write:
                writer([final])
            print('JOINT3B_FINAL', json.dumps({key: final.get(key) for key in
                ('layout_id', 'seed', 'status', 'cmax_s', 'b32_calls', 'runtime_s')}), flush=True)
            return final
        best = state['chosen']
        operator = 'joint3b-final-'+str(state['choice'])
    else:
        admissible = [entry for entry in entries
                      if entry[1].feasible and int(entry[2].get('load_allowed') or 0)]
        feasible_entries = [entry for entry in entries if entry[1].feasible]
        best = (min(admissible, key=lambda entry: entry[1].cmax) if admissible else
                min(feasible_entries, key=lambda entry: entry[1].cmax) if feasible_entries else
                max(entries, key=conflict_deadlock_progress))
        operator = 'joint3-final'
    best_key = plan_key(best[0].solution, best[0].directions)
    modes, owners = conflict_changes(initial, best[0], space.parent_ids)
    actual_transfers = joint3_assignment_changes(initial,best[0]) if cmax_driven else {}
    final = routeaware_result_row(name, method, best[0], best[1], blockers.get(best_key, ()),
        stage='final', candidate_id='final', origin_id=best[2]['candidate_id'],
        operator=operator, removed_parent_ids=best[2]['removed_parent_ids'],
        feedback=best[2]['feedback'], construct_s=best[0].inference_seconds, b32_s=0.,
        runtime_s=time.perf_counter()-started, b32_calls=calls, gnn_score=best[2]['gnn_score'],
        load_limits=json.dumps(limits), mode_changes=json.dumps(modes),
        assignment_changes=json.dumps(owners), rebuild_actions=best[2]['rebuild_actions'],
        route_lb_s=best[2]['route_lb_s'], incumbent_cmax_s=best[2]['incumbent_cmax_s'],
        beam_width=CMAX_BEAM if cmax_driven else JOINT3_BEAM, core_size=best[2]['core_size'],
        feedback_features=best[2]['feedback_features'], load_allowed_three_ref=best[2]['load_allowed_three_ref'],
        load_limits_three_ref=best[2]['load_limits_three_ref'], successful_repair=best[2]['successful_repair'],
        transfer_count=len(actual_transfers), transfer_details=json.dumps(actual_transfers),
        cross_robot_changes=len(actual_transfers), pool_ids='', run_id=run, variant=variant, phase=phase,
        final_choice='joint3-balanced-first' if variant == 'joint3' else state['choice'],
        r_c_pct=state['r_c_pct'],
        alt_cmax_balanced_s=None if state['best_balanced'] is None else state['best_balanced'][1].cmax,
        alt_cmax_best_s=None if state['best_cmax'] is None else state['best_cmax'][1].cmax,
        alt_cmax_balance_s=None if state['best_balance'] is None else state['best_balance'][1].cmax,
        deadlock_origins=deadlock_origins, deadlock_repairs=deadlock_repairs,
        cmax_tolerance_pct=0.)
    final.update(seed=seed, load_allowed=int(conflict_load_allowed(final, limits)),
                 load_allowed_three_ref=int(conflict_load_allowed(final, limits['three_reference'])))
    evaluated = len(entries)
    deadlocks = sum(entry[1].status is ScheduleStatus.DEADLOCK for entry in entries)
    feasible = sum(entry[1].feasible for entry in entries)
    load_allowed = sum(int(entry[2].get('load_allowed') or 0) for entry in entries)
    balanced_exists = state['best_balanced'] is not None
    final['diagnostic'] = json.dumps(dict(algorithm=method, run_id=run, variant=variant, budget=budget,
        max_attempts=max_attempts, attempts=attempts,
        stop='B32 budget exhausted' if calls >= budget else
             'outer attempts exhausted' if attempts >= max_attempts else 'no new legal core candidate',
        evaluated=evaluated, unique_plans=len(visited), b32_calls=calls,
        deadlock_candidates=deadlocks, deadlock_rate=deadlocks/max(1, evaluated),
        feasible_candidates=feasible, feasible_rate=feasible/max(1, evaluated),
        load_allowed_candidates=load_allowed, load_allowed_rate=load_allowed/max(1, evaluated),
        deadlock_origins=deadlock_origins, deadlock_repairs=deadlock_repairs,
        deadlock_repair_rate=deadlock_repairs/max(1, deadlock_origins),
        final_feasible=int(best[1].feasible), final_from_initial=best[0] is initial,
        final_choice=final['final_choice'], r_c_pct=state['r_c_pct'],
        choice_at_0p5=('balanced' if balanced_exists and state['r_c_pct'] is not None
                       and state['r_c_pct'] <= 0.5+1e-9 else 'cmax'),
        choice_at_1p0=('balanced' if balanced_exists and state['r_c_pct'] is not None
                       and state['r_c_pct'] <= 1.0+1e-9 else 'cmax'),
        alt_cmax_balanced_s=final['alt_cmax_balanced_s'],
        alt_cmax_best_s=final['alt_cmax_best_s'],
        alt_cmax_balance_s=final['alt_cmax_balance_s'],
        core_size_counts=dict(sorted(core_sizes.items())), failures=failures,
        final_core=json.loads(final['removed_parent_ids']), final_feedback=json.loads(final['feedback']),
        final_feedback_features=json.loads(final['feedback_features'] or 'null'),
        final_max_p_plus_move_s=final['route_lb_s'], gnn_reference_peak_s=gnn_peak,
        load_rule=limits['rule'], load_rule_three_ref=limits['three_reference']['rule'],
        evaluation='original FORMAL_SCOPE_V2 B32; no priority_schedule; no HGA/WAG candidate source',
        construction_source='original GNN_VNEXT_WORKLOAD'), ensure_ascii=False)
    if write:
        writer([final])
    print('JOINT3_FINAL', json.dumps({key: final.get(key) for key in
        ('layout_id', 'seed', 'method', 'status', 'cmax_s', 'max_processing_s', 'load_allowed',
         'load_allowed_three_ref', 'final_choice', 'r_c_pct', 'b32_calls', 'runtime_s')}), flush=True)
    return final


def joint3_group_loss(scores, ranks):
    losses, weights, correct, pairs = [], [], 0, 0
    for first in range(len(ranks)):
        for second in range(first+1, len(ranks)):
            a, b = ranks[first], ranks[second]
            if a == b:
                continue
            good, bad = (first, second) if a < b else (second, first)
            if a[0] != b[0]:
                weight = .25
            else:
                weight = abs(a[1]-b[1])/max(1., min(a[1], b[1]))
            losses.append(F.softplus(scores[bad]-scores[good]))
            weights.append(max(weight, 1e-3))
            correct += int(scores[good] > scores[bad])
            pairs += 1
    if not pairs:
        return None, 0, 0
    weights = scores.new_tensor(weights)
    return (torch.stack(losses)*weights).sum()/weights.sum().clamp_min(1e-12), pairs, correct


def joint3_train():
    """TRAIN-only pairwise ranking on the existing joint head, labelled by real B32."""
    configure_cpu()
    torch.manual_seed(SEED)
    if JOINT3_MODEL.exists():
        complete = torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)
        if complete.get('training_complete'):
            print('JOINT3_TRAIN_ALREADY_COMPLETE', complete.get('completed_epoch'), flush=True)
            return complete
    model = joint3_model().train()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT3_LEARNING_RATE, weight_decay=1e-4)
    references = joint3_references()
    training = dict(ppo_layouts('train'))
    ordered = sorted(training, key=lambda name: (len(training[name]), name))
    names = [ordered[round(index*(len(ordered)-1)/(JOINT3_TRAIN_LAYOUTS-1))]
             for index in range(JOINT3_TRAIN_LAYOUTS)]
    training_started = time.perf_counter()
    prepared, groups, calls = [], 0, 0
    construct_s = b32_s = 0.
    with torch.no_grad():
        for name in names:
            parents = training[name]
            started = time.perf_counter()
            space = gnn.parent_assignment_space(parents, CONFIG)
            embeddings = model.encode_static(space.graph)
            initial = build_joint(parents, model, space=space, embeddings=embeddings)
            construct_s += time.perf_counter()-started
            if initial.solution is None:
                print('JOINT3_TRAIN_SKIP', name, initial.rejection_reason, flush=True)
                continue
            seen, blockers = {}, {}
            evaluation = time.perf_counter()
            schedule, called = evaluate_once(initial, seen, blocker_cache=blockers)
            b32_s += time.perf_counter()-evaluation
            calls += int(called)

            def collect(origin, origin_schedule, stage, limit=JOINT3_TRAIN_CORES):
                nonlocal calls, b32_s, groups
                state = cycle_route_state(origin)
                limits = joint3_training_limits(state)
                key = plan_key(origin.solution, origin.directions)
                cores = []
                for index in range(12):
                    core, _ = joint3_core(origin, origin_schedule, model, space, embeddings,
                        blockers.get(key, ()), seed=SEED+index*7919+len(parents)+len(prepared), q=JOINT3_CORE)
                    if core is None:
                        continue
                    removed = tuple(core.removed_parent_ids)
                    if removed in cores:
                        continue
                    cores.append(removed)
                    if len(cores) >= limit:
                        break
                labelled, best = [], None
                for removed in cores:
                    measured = joint3_feedback(origin, origin_schedule, blockers.get(key, ()), removed)
                    candidates, _ = joint3_reconstruct(parents, model, space, embeddings, origin, removed,
                                                       limits, measured, width=JOINT3_TRAIN_BEAM)
                    for candidate in candidates:
                        candidate_key = plan_key(candidate['plan'].solution, candidate['plan'].directions)
                        if candidate_key in seen:
                            continue
                        evaluation = time.perf_counter()
                        candidate_schedule, called = evaluate_once(candidate['plan'], seen, blocker_cache=blockers)
                        b32_s += time.perf_counter()-evaluation
                        calls += int(called)
                        row = routeaware_result_row(name, JOINT3_METHOD, candidate['plan'], candidate_schedule,
                            blockers.get(candidate_key, ()))
                        row.update(load_allowed=int(conflict_load_allowed(row, limits)))
                        cls = 0 if candidate_schedule.feasible and int(row['load_allowed']) else \
                            1 if candidate_schedule.feasible else 2
                        labelled.append(dict(plan=candidate['plan'], joint_plans=candidate['joint_plans'],
                            rank=(cls, float(candidate_schedule.cmax) if candidate_schedule.feasible else 1e9),
                            status=candidate_schedule.status.value, load_allowed=int(row['load_allowed']),
                            feedback_features=candidate['feedback_features']))
                        if candidate_schedule.feasible and (best is None or candidate_schedule.cmax < best[1].cmax):
                            best = (candidate['plan'], candidate_schedule)
                distinct = len({item['rank'] for item in labelled})
                if distinct > 1:
                    prepared.append((space, labelled))
                    groups += 1
                print('JOINT3_TRAIN_GROUP', json.dumps(dict(layout_id=name, stage=stage,
                    parents=len(parents), cores=len(cores), candidates=len(labelled), classes=distinct,
                    best_s=None if best is None else best[1].cmax,
                    feedback_features=[item['feedback_features'] for item in labelled])), flush=True)
                return best

            best = collect(initial, schedule, 'initial',
                           limit=8 if schedule.status is ScheduleStatus.DEADLOCK else JOINT3_TRAIN_CORES)
            # A repaired FEASIBLE state is the informative ranking origin: its
            # candidates differ in real Cmax and in the load class.
            if best is not None:
                collect(best[0], best[1], 'repaired')
    if not prepared:
        raise ValueError('no usable real-B32 ranking preference in TRAIN')
    history = []
    for epoch in range(1, JOINT3_EPOCHS+1):
        started = time.perf_counter()
        order = list(range(len(prepared)))
        Random(SEED+epoch).shuffle(order)
        epoch_loss, epoch_pairs, epoch_correct, updates, gradients = [], 0, 0, 0, []
        for index in order:
            space, labelled = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode_static(space.graph)
            scores = torch.stack([gnn.imitation_joint_scores(space, embeddings, model,
                item['joint_plans']).mean() for item in labelled])
            loss, pairs, correct = joint3_group_loss(scores, [item['rank'] for item in labelled])
            if loss is None:
                continue
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
                raise ArithmeticError('nonfinite joint3 ranking optimization')
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            epoch_pairs += pairs
            epoch_correct += correct
            updates += 1
            gradients.append(gradient)
        record = dict(stage='epoch', seed=SEED, method=JOINT3_METHOD, epoch=epoch,
            loss=sum(epoch_loss)/max(1, len(epoch_loss)), ranking_accuracy=epoch_correct/max(1, epoch_pairs),
            pairs=epoch_pairs, updates=updates, gradient_norm=max(gradients, default=0.),
            learning_rate=JOINT3_LEARNING_RATE, candidates=sum(len(labelled) for _, labelled in prepared),
            b32_calls=calls, construct_s=construct_s, b32_s=b32_s,
            runtime_s=time.perf_counter()-training_started,
            parent_count=len(names), diagnostic=json.dumps(dict(training_split='TRAIN_ONLY',
                groups=groups, layouts=len(names), feedback_outcomes=True,
                label='load-qualified FEASIBLE > FEASIBLE by real B32 Cmax > DEADLOCK')))
        history.append(record)
        joint3_training_write([record])
        torch.save(dict(seed=SEED, initialization='models/gnn_weld_layout.pt:GNN_VNEXT_WORKLOAD',
            state_dict=copy.deepcopy(model.state_dict()), initial_state_dict=initial_state,
            optimizer_state_dict=optimizer.state_dict(), completed_epoch=epoch,
            training_complete=epoch == JOINT3_EPOCHS, history=history, training_split='TRAIN_ONLY',
            training_layouts=names, feedback_outcomes=True, b32_calls=calls,
            construct_s=construct_s, b32_s=b32_s), JOINT3_MODEL)
        print('JOINT3_TRAIN_EPOCH', json.dumps({key: record[key] for key in
            ('epoch', 'loss', 'ranking_accuracy', 'pairs', 'updates', 'b32_calls', 'runtime_s')}), flush=True)
    return torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)


def joint3_job(payload):
    configure_cpu()
    name, parents, seed, weights = payload
    model = joint3_model(weights).eval()
    model.requires_grad_(False)
    return joint3_case(name, parents, seed, model, joint3_references())


def joint3_validation(workers=4, seeds=JOINT3_SEEDS):
    if not JOINT3_MODEL.exists():
        raise ValueError('train and save the joint3 actor before VALIDATION')
    saved = torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)
    if not saved.get('training_complete'):
        raise ValueError('joint3 training is incomplete')
    layouts = dict(load_layouts('validation'))
    if set(layouts) != set(JOINT3_LAYOUTS):
        raise ValueError('the official ten-layout VALIDATION split changed')
    jobs = [(name, layouts[name], seed, saved['state_dict'])
            for name in JOINT3_LAYOUTS for seed in seeds]
    finals = []
    if workers <= 1:
        finals = [joint3_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(joint3_job, job) for job in jobs]):
                finals.append(future.result())
    summary = joint3_summary()
    print('JOINT3_VALIDATION_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    return finals


def joint3b_job(payload):
    configure_cpu()
    name, parents, seed, variant, phase, weights, budget, limit_mode = payload
    model = joint3_model(weights).eval()
    model.requires_grad_(False)
    return joint3_case(name, parents, seed, model, joint3_references(), budget=budget,
                       variant=variant, sink='loadkeep', phase=phase, limit_mode=limit_mode)


def joint3b_train_stage(workers=1, seeds=(SEED,), budget=JOINT3_BUDGET):
    """Stage 1: six TRAIN layouts, original Joint3 against the load-keeping variant."""
    layouts = dict(ppo_layouts('train'))
    missing = [name for name in JOINT3B_TRAIN_CASES if name not in layouts]
    if missing:
        raise ValueError(f'missing TRAIN layouts: {missing}')
    weights = torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)['state_dict']
    jobs = [(name, layouts[name], seed, variant, 'train', weights, budget, 'relative')
            for name in JOINT3B_TRAIN_CASES for seed in seeds for variant in ('joint3', 'joint3b')]
    if workers <= 1:
        finals = [joint3b_job(job) for job in jobs]
    else:
        finals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(joint3b_job, job) for job in jobs]):
                finals.append(future.result())
    summary = joint3b_summary(phase='train')
    print('JOINT3B_TRAIN_STAGE', json.dumps(summary, ensure_ascii=False), flush=True)
    return finals


def joint3b_validation(workers=4, seeds=JOINT3_SEEDS):
    saved = torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)
    if not saved.get('training_complete'):
        raise ValueError('joint3 training is incomplete')
    layouts = dict(load_layouts('validation'))
    if set(layouts) != set(JOINT3_LAYOUTS):
        raise ValueError('the official ten-layout VALIDATION split changed')
    jobs = [(name, layouts[name], seed, 'joint3b', 'validation', saved['state_dict'], JOINT3_BUDGET,
             'reference')
            for name in JOINT3_LAYOUTS for seed in seeds]
    if workers <= 1:
        finals = [joint3b_job(job) for job in jobs]
    else:
        finals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(joint3b_job, job) for job in jobs]):
                finals.append(future.result())
    summary = joint3b_summary(phase='validation')
    print('JOINT3B_VALIDATION_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    return finals


def joint4_deadlock_signature(entry):
    """Full B32 deadlock identity: blocked/blocker parents, robots, operations and MOVE endpoints."""
    parts = []
    for event in entry[1].deadlock_blockages:
        blocked = event.blocked_operation
        parts.append((
            event.robot_id,
            getattr(blocked, 'kind', None),
            getattr(blocked, 'robot_id', None),
            tuple(sorted(event.blocked_parent_ids)),
            tuple(sorted(event.blocked_block_ids)),
            tuple(sorted(event.blocker_operation_ids)),
            tuple(sorted(event.blocker_block_ids)),
            tuple(tuple(sorted(parents)) for parents in event.blocker_parent_ids),
        ))
    return frozenset(parts)


def joint4_pool(entries, latest, *, width=6):
    """Cmax-ranked feasible states plus two distinct measured deadlock cores."""
    by_key = {plan_key(entry[0].solution, entry[0].directions): entry
              for entry in (*entries, *latest)
              if entry[1].status in (ScheduleStatus.FEASIBLE, ScheduleStatus.DEADLOCK)}
    feasible = sorted((e for e in by_key.values() if e[1].feasible), key=lambda e: e[1].cmax)
    deadlocks = sorted((e for e in by_key.values() if not e[1].feasible),
                       key=conflict_deadlock_progress, reverse=True)
    diverse, signatures = [], set()
    for entry in deadlocks:
        signature = joint4_deadlock_signature(entry)
        if signature not in signatures:
            diverse.append(entry); signatures.add(signature)
    chosen = feasible[:max(1, width-2)] + diverse[:2]
    for entry in (*feasible, *diverse, *deadlocks):
        if len(chosen) >= width:
            break
        if all(entry is not kept for kept in chosen):
            chosen.append(entry)
    return chosen[:width]


def joint4_actions_by_parent(space):
    grouped = {}
    for action in space.actions:
        grouped.setdefault(action.parent_id, []).append(action)
    return grouped


def joint4_bottleneck_core(current, schedule, summary, space, state, *, seed, q):
    """3--6 whole parents linking the measured critical route to 2--4 robots."""
    q = min(len(current.solution.parents), max(3, min(6, q)))
    completion = list(schedule.robot_completion)
    critical = max(range(4), key=lambda r: (completion[r], -r))
    grouped = joint4_actions_by_parent(space)
    routes = [tuple(dict.fromkeys(b.split('::', 1)[0] for b in route.block_ids))
              for route in current.solution.routes]
    current_owners = {p: {r for r in range(4) if p in routes[r]} for p in space.parent_ids}

    def movable(parent):
        return any(set(a.robots) != current_owners[parent] for a in grouped.get(parent, ()))

    def contribution(parent, robot):
        costs = summary.get('parent_costs', {}).get(parent)
        cost = (float(costs[robot]) if costs is not None else
                float(summary['parent_processing'].get(parent, (0.,)*4)[robot])
                + float(summary['parent_move'].get(parent, 0.)))
        return cost + float(summary['parent_critical_blocked'].get(parent, 0.))

    ranked = {r: sorted((p for p in routes[r] if movable(p)),
                        key=lambda p: (-contribution(p, r), p)) for r in range(4)}
    if not ranked[critical]:
        ranked[critical] = sorted(routes[critical], key=lambda p: (-contribution(p, critical), p))
    if not ranked[critical]:
        return None, dict(reason='critical route has no parent')
    rotation = (seed//7919) % min(5, len(ranked[critical]))
    lead = ranked[critical][:5]
    removed = [lead[rotation]]
    connected = {critical}
    receiver_parents = []
    # Receiving capacity is inferred from the unchanged legal catalog. No
    # single destination, rail, mode or owner tuple is fixed for reconstruction.
    receivers = sorted((r for r in range(4) if r != critical and ranked[r]),
                       key=lambda r: (completion[r], r))
    for robot in receivers:
        if len(removed) >= min(q, 4):
            break
        candidates = ranked[robot]
        compatible = [p for p in candidates if any(
            owner in connected for a in grouped.get(p, ()) for owner in a.robots)
            or any(robot in a.robots for p0 in removed for a in grouped.get(p0, ()))]
        if not compatible:
            continue
        band = compatible[:5]
        parent = band[(seed//7+robot) % len(band)]
        if parent not in removed:
            removed.append(parent); receiver_parents.append(parent)
        connected.add(robot)
    # Actual critical WAIT and expensive MOVE endpoints can join the same set.
    evidence = sorted(set(summary['parent_critical_wait']) | set(summary['parent_critical_blocked']),
                      key=lambda p: (-(summary['parent_critical_wait'].get(p, 0.)
                                       + summary['parent_critical_blocked'].get(p, 0.)), p))
    expensive = []
    route = current.solution.routes[critical].block_ids
    for arc in sorted(range(len(state.arc_costs[critical])),
                      key=lambda i: -state.arc_costs[critical][i])[:2]:
        for position in (arc-1, arc):
            if 0 <= position < len(route):
                expensive.append(route[position].split('::', 1)[0])
    fill = [*evidence, *expensive, *ranked[critical]]
    fill += [p for robot in (*receivers, critical) for p in ranked[robot]]
    for parent in fill:
        if len(removed) >= q:
            break
        if parent not in removed:
            removed.append(parent)
    if len(removed) < 3:
        return None, dict(reason='fewer than three reconstructable parents')
    robots = sorted({r for r in range(4) for p in removed if p in routes[r]})
    return lns._remove_parent_set(current.solution, removed), dict(
        source='B32 critical completion, route contributions, MOVE and blocked/blocker evidence',
        operator='joint4_bottleneck_core', transfer_robot=critical,
        receiver_robots=[r for r in robots if r != critical], involved_robots=robots,
        transfer_parents=[p for p in removed if p in routes[critical]],
        receiver_parents=receiver_parents, core=removed, completion_s=completion,
        completion_slack_s=[max(completion)-c for c in completion],
        cycle_s=[p+sum(m) for p, m in zip(state.loads, state.arc_costs)],
        wait_s=list(summary['robot_wait']), core_kind='multi-robot-joint-reconstruction')




def joint4_core(current, schedule, model, space, embeddings, blockers, *, seed, q):
    if schedule.status is ScheduleStatus.DEADLOCK:
        return joint3b_core(current, schedule, model, space, embeddings, blockers, seed=seed, q=q)
    if schedule.status is not ScheduleStatus.FEASIBLE:
        return None, dict(reason='B32 returned neither FEASIBLE nor DEADLOCK')
    summary = lns.schedule_feedback_summary(current.solution, schedule, blockers, CONFIG)
    state = cycle_route_state(current)
    partial, feedback = joint4_bottleneck_core(current, schedule, summary, space, state, seed=seed, q=q)
    if partial is None:
        route_partial, route_feedback = lns.select_routeaware_bottleneck(current.solution,
            current.directions, schedule, blockers, CONFIG, q=min(q, 3), seed=seed, assignment_space=space)
        return route_partial, dict(route_feedback, operator='joint4_fallback_routeaware',
                                   fallback_reason=feedback.get('reason'))
    return partial, feedback


def joint4_screen(candidates, origin_native):
    """Reserve one B32 slot for the GNN choice when native and GNN disagree.

    The native priority scheduler can report DEADLOCK for plans accepted by B32.
    Since the caller evaluates only the first two candidates with B32, blindly
    ranking every native-feasible candidate first can discard a valid GNN choice.
    This retains all candidates and does not use native as a feasibility filter.
    """
    for candidate in candidates:
        native = priority_schedule(candidate['plan'].solution, CONFIG, candidate['plan'].directions)
        candidate['native_status'] = native.status.value
        candidate['native_cmax'] = (native.cmax if native.status is ScheduleStatus.FEASIBLE
                                    and native.cmax is not None else None)
    if not candidates:
        return candidates
    if origin_native is not None:
        native_order = sorted(candidates, key=lambda candidate: (
            candidate['native_cmax'] is None,
            candidate['native_cmax'] if candidate['native_cmax'] is not None else math.inf,
            -candidate['score']))
    else:
        native_order = sorted(candidates, key=lambda candidate: (
            candidate['native_cmax'] is None, -candidate['score']))
    gnn_order = sorted(candidates, key=lambda candidate: (-candidate['score'], candidate['lb']))
    # First B32 slot: native-ranked best. Second: the distinct top GNN choice.
    # If both rankings agree, use the next GNN candidate instead of a duplicate.
    selected = [native_order[0]]
    second = next((item for item in gnn_order if item is not selected[0]), None)
    if second is not None:
        selected.append(second)
    return selected + [item for item in native_order
                       if all(item is not chosen for chosen in selected)]


def joint4_native_value(plan):
    native = priority_schedule(plan.solution, CONFIG, plan.directions)
    if native.status is ScheduleStatus.FEASIBLE and native.cmax is not None:
        return float(native.cmax), native.status.value
    return None, native.status.value


def joint4_training_rank(row, initial_peak, initial_spread):
    """Feasibility class and actual B32 Cmax, without a load objective."""
    if row['status'] != 'FEASIBLE' or row.get('cmax_s') in (None, ''):
        return (1, 0.0, None)
    return (0, 0.0, float(row['cmax_s']))


def joint4_group_loss(scores, ranks):
    """Pairwise Cmax loss; DEADLOCK supplies only an infeasibility label."""
    losses, weights, correct, pairs = [], [], 0, 0
    for first in range(len(ranks)):
        for second in range(first+1, len(ranks)):
            a, b = ranks[first], ranks[second]
            if a == b:
                continue
            good, bad = (first, second) if a < b else (second, first)
            if a[0] != b[0]:
                weight = .25
            else:
                reference = max(1., min(a[2], b[2]))
                weight = abs(a[2]-b[2])/reference
            losses.append(F.softplus(scores[bad]-scores[good]))
            weights.append(max(weight, 1e-3))
            correct += int(scores[good] > scores[bad])
            pairs += 1
    if not pairs:
        return None, 0, 0
    weights = scores.new_tensor(weights)
    return (torch.stack(losses)*weights).sum()/weights.sum().clamp_min(1e-12), pairs, correct


def joint4_write(rows, *, path=JOINT4_RESULTS):
    """The Joint4 CSV; Joint3/Joint3B CSV files are never written by this path."""
    import msvcrt
    if not rows:
        return
    with path.open('a+', encoding='utf-8', newline='') as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        try:
            stream.seek(0, 2)
            writer = csv.DictWriter(stream, fieldnames=JOINT4_FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in JOINT4_FIELDS})
            stream.flush()
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def cmax_write(rows):
    joint4_write(rows, path=CMAX_RESULTS)


def cmax_limits():
    rule = 'minimize actual feasible B32 Cmax; no processing-load hard constraints'
    diagnostic = dict(objective='Cmax', pmax_limit_s=None, spread_limit_s=None,
                      temporary_pmax_limit_s=None, rule=rule)
    return dict(diagnostic, three_reference=dict(diagnostic))


def cmax_train_split():
    layouts = dict(load_layouts('train'))
    ordered = sorted(layouts, key=lambda name: (len(layouts[name]), name))
    names = [ordered[round(i*(len(ordered)-1)/19)] for i in range(20)]
    inherited = torch.load(JOINT4_MODEL, map_location='cpu', weights_only=True)['training_layouts']
    remaining = [name for name in ordered if name not in names and name not in inherited]
    holdout = [remaining[round(i*(len(remaining)-1)/5)] for i in range(6)]
    return names, holdout


def joint3b_summary(phase='train'):
    """Paired comparison of the two variants on the same layouts and seeds."""
    rows = [row for row in read_rows(JOINT3B_RESULTS) if row.get('phase') == phase]
    finals = {(row['layout_id'], int(row['seed']), row['variant']): row
              for row in rows if row.get('stage') == 'final'}
    candidates = {}
    for row in rows:
        if row.get('stage') in ('initial', 'candidate'):
            candidates.setdefault((row['layout_id'], int(row['seed']), row.get('variant')), []).append(row)
    cases = sorted({(key[0], key[1]) for key in finals})
    output = dict(phase=phase, cases=len(cases), variants={}, paired={}, tradeoff={})
    for variant in ('joint3', 'joint3b'):
        chosen = [row for key, row in finals.items() if key[2] == variant]
        feasible = [row for row in chosen if row['status'] == 'FEASIBLE' and row.get('cmax_s') not in (None, '')]
        load_ok = [row for row in feasible if int(row.get('load_allowed') or 0)]
        rebuilt = [row for key, group in candidates.items() if key[2] == variant for row in group]
        output['variants'][variant] = dict(
            finals=len(chosen), feasible=len(feasible), load_allowed=len(load_ok),
            deadlock_finals=sum(1 for row in chosen if row['status'] == 'DEADLOCK'),
            explicit_failures=sum(1 for row in chosen
                                  if row['status'] not in ('FEASIBLE', 'DEADLOCK')),
            candidate_count=len(rebuilt),
            deadlock_candidates=sum(1 for row in rebuilt if row['status'] == 'DEADLOCK'),
            deadlock_repairs=sum(int(row.get('successful_repair') or 0) for row in rebuilt),
            median_cmax_s=median([float(row['cmax_s']) for row in feasible]) if feasible else None,
            median_runtime_s=median([float(row['runtime_s']) for row in chosen]) if chosen else None,
            median_b32_calls=median([int(row['b32_calls']) for row in chosen]) if chosen else None)
    deltas = []
    for name, seed in cases:
        old = finals.get((name, seed, 'joint3'))
        new = finals.get((name, seed, 'joint3b'))
        if old is None or new is None:
            continue
        if old['status'] == 'FEASIBLE' and new['status'] == 'FEASIBLE' \
                and old.get('cmax_s') not in (None, '') and new.get('cmax_s') not in (None, ''):
            old_cmax, new_cmax = float(old['cmax_s']), float(new['cmax_s'])
            deltas.append(dict(layout_id=name, seed=seed, joint3_s=old_cmax, loadkeep_s=new_cmax,
                               delta_pct=(new_cmax-old_cmax)/old_cmax*100.))
    output['paired'] = dict(pairs=len(deltas),
                            median_pct=median([item['delta_pct'] for item in deltas]) if deltas else None,
                            wins=sum(1 for item in deltas if item['delta_pct'] < -1e-9),
                            losses=sum(1 for item in deltas if item['delta_pct'] > 1e-9),
                            ties=sum(1 for item in deltas if abs(item['delta_pct']) <= 1e-9),
                            detail=deltas)
    ratios = [float(row['r_c_pct']) for key, row in finals.items()
              if key[2] == 'joint3b' and row.get('r_c_pct') not in (None, '')]
    output['tradeoff'] = dict(cases_with_r_c=len(ratios),
                              median_r_c_pct=median(ratios) if ratios else None,
                              max_r_c_pct=max(ratios) if ratios else None,
                              within_0p5=sum(1 for value in ratios if value <= 0.5+1e-9),
                              within_1p0=sum(1 for value in ratios if value <= 1.0+1e-9),
                              above_1p0=sum(1 for value in ratios if value > 1.0+1e-9))
    if phase == 'validation':
        references = joint3_references()
        old_csv = {(row['layout_id'], int(row['seed'])): row for row in read_rows(JOINT3_RESULTS)
                   if row.get('stage') == 'final' and row.get('method') == JOINT3_METHOD}
        old_deltas = []
        for key, row in finals.items():
            if key[2] != 'joint3b' or row['status'] != 'FEASIBLE' or row.get('cmax_s') in (None, ''):
                continue
            old = old_csv.get((key[0], key[1]))
            if old is None or old['status'] != 'FEASIBLE' or old.get('cmax_s') in (None, ''):
                continue
            old_deltas.append((float(row['cmax_s'])-float(old['cmax_s']))/float(old['cmax_s'])*100.)
        output['vs_joint3'] = dict(pairs=len(old_deltas),
                                   median_pct=median(old_deltas) if old_deltas else None,
                                   wins=sum(1 for value in old_deltas if value < -1e-9),
                                   losses=sum(1 for value in old_deltas if value > 1e-9))
        for tag, method in (('HGA', 'HGA'), ('WAG', 'WAG+VNS')):
            values = []
            for key, row in finals.items():
                if key[2] != 'joint3b' or row['status'] != 'FEASIBLE' or row.get('cmax_s') in (None, ''):
                    continue
                reference = references.get((key[0], method))
                if reference is None:
                    continue
                values.append((float(row['cmax_s'])-reference['cmax_s'])/reference['cmax_s']*100.)
            output[tag] = dict(pairs=len(values), median_pct=median(values) if values else None,
                               wins=sum(1 for value in values if value < -1e-9),
                               losses=sum(1 for value in values if value > 1e-9))
    return output


def joint4_job(payload):
    configure_cpu()
    name, parents, seed, variant, phase, weights, budget, limit_mode = payload
    model = joint4_model(weights).eval()
    model.requires_grad_(False)
    return joint3_case(name, parents, seed, model, joint3_references(), budget=budget,
                       variant=variant, sink='joint4', phase=phase, limit_mode=limit_mode)


def joint4_model(weights=None):
    """Same VNext actor; Joint4 fine-tunes the Joint3 weights, never a new network."""
    if weights is not None:
        return joint3_model(weights)
    source = torch.load(JOINT3_MODEL, map_location='cpu', weights_only=True)
    return joint3_model(source['state_dict'])


def joint4_train_stage(workers=1, seeds=(SEED,), budget=JOINT3_BUDGET):
    """Stage 1: six TRAIN layouts, Joint3B against the bottleneck cross-robot Joint4."""
    layouts = dict(ppo_layouts('train'))
    missing = [name for name in JOINT3B_TRAIN_CASES if name not in layouts]
    if missing:
        raise ValueError(f'missing TRAIN layouts: {missing}')
    weights = torch.load(JOINT4_MODEL if JOINT4_MODEL.exists() else JOINT3_MODEL,
                         map_location='cpu', weights_only=True)['state_dict']
    jobs = [(name, layouts[name], seed, variant, 'train', weights, budget, 'relative')
            for name in JOINT3B_TRAIN_CASES for seed in seeds for variant in ('joint3b', 'joint4')]
    if workers <= 1:
        finals = [joint4_job(job) for job in jobs]
    else:
        finals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(joint4_job, job) for job in jobs]):
                finals.append(future.result())
    summary = joint4_summary(phase='train')
    print('JOINT4_TRAIN_STAGE', json.dumps(summary, ensure_ascii=False), flush=True)
    return finals


def joint4_validation(workers=4, seeds=JOINT3_SEEDS):
    saved = torch.load(JOINT4_MODEL, map_location='cpu', weights_only=True)
    if not saved.get('training_complete'):
        raise ValueError('joint4 training is incomplete')
    layouts = dict(load_layouts('validation'))
    if set(layouts) != set(JOINT3_LAYOUTS):
        raise ValueError('the official ten-layout VALIDATION split changed')
    jobs = [(name, layouts[name], seed, 'joint4', 'validation', saved['state_dict'],
             JOINT3_BUDGET, 'reference')
            for name in JOINT3_LAYOUTS for seed in seeds]
    if workers <= 1:
        finals = [joint4_job(job) for job in jobs]
    else:
        finals = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(joint4_job, job) for job in jobs]):
                finals.append(future.result())
    summary = joint4_summary(phase='validation')
    print('JOINT4_VALIDATION_SUMMARY', json.dumps(summary, ensure_ascii=False), flush=True)
    return finals


def joint4_summary(phase='train'):
    rows = [row for row in read_rows(JOINT4_RESULTS) if row.get('phase') == phase]
    finals = {(row['layout_id'], int(row['seed']), row['variant']): row
              for row in rows if row.get('stage') == 'final'}
    candidates = {}
    for row in rows:
        if row.get('stage') in ('initial', 'candidate'):
            candidates.setdefault((row['layout_id'], int(row['seed']), row.get('variant')), []).append(row)
    output = dict(phase=phase, variants={}, paired={}, tradeoff={})
    for variant in ('joint3b', 'joint4'):
        chosen = [row for key, row in finals.items() if key[2] == variant]
        feasible = [row for row in chosen if row['status'] == 'FEASIBLE' and row.get('cmax_s') not in (None, '')]
        rebuilt = [row for key, group in candidates.items() if key[2] == variant for row in group]
        cross = [int(row.get('cross_robot_changes') or 0) for row in rebuilt]
        native_known = [row for row in rebuilt if row.get('native_status') not in (None, '')]
        output['variants'][variant] = dict(
            finals=len(chosen), feasible=len(feasible),
            load_allowed=sum(1 for row in feasible if int(row.get('load_allowed') or 0)),
            deadlock_finals=sum(1 for row in chosen if row['status'] == 'DEADLOCK'),
            explicit_failures=sum(1 for row in chosen if row['status'] not in ('FEASIBLE', 'DEADLOCK')),
            candidate_count=len(rebuilt),
            deadlock_candidates=sum(1 for row in rebuilt if row['status'] == 'DEADLOCK'),
            deadlock_repairs=sum(int(row.get('successful_repair') or 0) for row in rebuilt),
            cross_robot_candidates=sum(1 for value in cross if value > 0),
            median_cross_robot=median(cross) if cross else None,
            native_feasible=sum(1 for row in native_known if row.get('native_status') == 'FEASIBLE'),
            native_known=len(native_known),
            median_cmax_s=median([float(row['cmax_s']) for row in feasible]) if feasible else None,
            median_pmax_s=median([float(row['max_processing_s']) for row in feasible]) if feasible else None,
            median_spread_s=median([float(row['processing_spread_s']) for row in feasible]) if feasible else None,
            median_move_s=median([float(row['move_s']) for row in feasible]) if feasible else None,
            median_wait_s=median([float(row['wait_s']) for row in feasible]) if feasible else None,
            median_runtime_s=median([float(row['runtime_s']) for row in chosen]) if chosen else None,
            median_b32_calls=median([int(row['b32_calls']) for row in chosen]) if chosen else None)
    deltas = []
    for name, seed in sorted({(key[0], key[1]) for key in finals}):
        old = finals.get((name, seed, 'joint3b'))
        new = finals.get((name, seed, 'joint4'))
        if old is None or new is None:
            continue
        if old['status'] == 'FEASIBLE' and new['status'] == 'FEASIBLE' \
                and old.get('cmax_s') not in (None, '') and new.get('cmax_s') not in (None, ''):
            old_cmax, new_cmax = float(old['cmax_s']), float(new['cmax_s'])
            deltas.append(dict(layout_id=name, seed=seed, joint3b_s=old_cmax, joint4_s=new_cmax,
                               delta_pct=(new_cmax-old_cmax)/old_cmax*100.))
    output['paired'] = dict(pairs=len(deltas),
                            median_pct=median([item['delta_pct'] for item in deltas]) if deltas else None,
                            wins=sum(1 for item in deltas if item['delta_pct'] < -1e-9),
                            losses=sum(1 for item in deltas if item['delta_pct'] > 1e-9),
                            ties=sum(1 for item in deltas if abs(item['delta_pct']) <= 1e-9),
                            detail=deltas)
    ratios = [float(row['r_c_pct']) for key, row in finals.items()
              if key[2] == 'joint4' and row.get('r_c_pct') not in (None, '')]
    output['tradeoff'] = dict(cases_with_r_c=len(ratios),
                              median_r_c_pct=median(ratios) if ratios else None,
                              within_0p5=sum(1 for value in ratios if value <= 0.5+1e-9),
                              within_1p0=sum(1 for value in ratios if value <= 1.0+1e-9),
                              above_1p0=sum(1 for value in ratios if value > 1.0+1e-9))
    if phase == 'validation':
        references = joint3_references()
        for tag, method in (('HGA', 'HGA'), ('WAG', 'WAG+VNS')):
            values, wins, losses, deadlock_seeds = [], 0, 0, 0
            wag_rows = read_rows(JOINT3_WAG_REFERENCE) if tag == 'WAG' else ()
            for key, row in finals.items():
                if key[2] != 'joint4':
                    continue
                reference = references.get((key[0], method))
                feasible = row['status'] == 'FEASIBLE' and row.get('cmax_s') not in (None, '')
                if not feasible:
                    losses += 1
                    continue
                if reference is None:
                    wins += 1
                    deadlock_seeds += 1
                    continue
                if tag == 'WAG' and any(other['layout_id'] == key[0] and int(other['seed']) == key[1]
                                        and other.get('source') == JOINT3_WAG_SOURCE
                                        and other['status'] != 'FEASIBLE' for other in wag_rows):
                    wins += 1
                    deadlock_seeds += 1
                    continue
                delta = (float(row['cmax_s'])-reference['cmax_s'])/reference['cmax_s']*100.
                values.append(delta)
                wins += int(delta < -1e-9)
                losses += int(delta > 1e-9)
            output[tag] = dict(pairs=len(values), median_pct=median(values) if values else None,
                               wins=wins, losses=losses, feasibility_wins=deadlock_seeds)
    return output


def joint4_fit(workers=1, *, cmax_driven=False):
    """TRAIN-only pairwise fit of the existing joint head with the Joint4 labels."""
    configure_cpu()
    torch.manual_seed(SEED)
    model_path = CMAX_MODEL if cmax_driven else JOINT4_MODEL
    writer = cmax_write if cmax_driven else joint4_write
    if model_path.exists():
        complete = torch.load(model_path, map_location='cpu', weights_only=True)
        if complete.get('training_complete'):
            print('JOINT4_FIT_ALREADY_COMPLETE', complete.get('completed_epoch'), flush=True)
            return complete
    model = (joint4_model(torch.load(JOINT4_MODEL, map_location='cpu', weights_only=True)['state_dict'])
             .enable_conflict_graph() if cmax_driven else joint4_model()).train()
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=JOINT3_LEARNING_RATE, weight_decay=1e-4)
    training = dict(load_layouts('train') if cmax_driven else ppo_layouts('train'))
    ordered = sorted(training, key=lambda name: (len(training[name]), name))
    names = [ordered[round(index*(len(ordered)-1)/(JOINT4_TRAIN_LAYOUTS-1))]
             for index in range(JOINT4_TRAIN_LAYOUTS)]
    if cmax_driven:
        names, holdout = cmax_train_split()
        print('CMAX_FIXED_SPLIT', json.dumps(dict(training=names, independent=holdout)), flush=True)
    started = time.perf_counter()
    prepared, calls, good_total, feasible_total = [], 0, 0, 0
    groups, construct_s, b32_s = 0, 0., 0.
    label_rows = []
    with torch.no_grad():
        for name in names:
            parents = training[name]
            clock = time.perf_counter()
            space = gnn.parent_assignment_space(parents, CONFIG)
            embeddings = model.encode_static(space.graph)
            initial = build_joint(parents, model, space=space, embeddings=embeddings)
            construct_s += time.perf_counter()-clock
            if initial.solution is None:
                continue
            state = cycle_route_state(initial)
            limits = cmax_limits() if cmax_driven else joint3_relative_limits(state)
            initial_peak, initial_spread = max(state.loads), max(state.loads)-min(state.loads)
            seen, blockers = {}, {}
            clock = time.perf_counter()
            schedule, called = evaluate_once(initial, seen, blocker_cache=blockers)
            b32_s += time.perf_counter()-clock
            calls += int(called)

            if cmax_driven:
                writer([routeaware_result_row(name, CMAX_METHOD, initial, schedule,
                    blockers.get(plan_key(initial.solution,initial.directions),()),
                    stage='training_initial', phase='fit_origin_nodes', b32_calls=int(called),
                    runtime_s=time.perf_counter()-started)])

            def collect(origin, origin_schedule, limit):
                nonlocal calls, b32_s, groups, good_total, feasible_total
                key = plan_key(origin.solution, origin.directions)
                cores = []
                for index in range(16):
                    core, _ = joint4_core(origin, origin_schedule, model, space, embeddings,
                        blockers.get(key, ()), seed=SEED+index*7919+len(parents)+len(prepared),
                        q=((3,4,6)[index%3] if limit == 3 else (5,4)[index%2])
                          if cmax_driven else JOINT4_CORE)
                    if core is None:
                        continue
                    removed = tuple(core.removed_parent_ids)
                    if removed in cores:
                        continue
                    cores.append(removed)
                    if len(cores) >= limit:
                        break
                labelled = []
                for removed in cores:
                    measured = joint3_feedback(origin, origin_schedule, blockers.get(key, ()), removed)
                    candidates, _ = joint3_reconstruct(parents, model, space, embeddings, origin, removed,
                                                       limits, measured, width=CMAX_BEAM if cmax_driven else JOINT4_BEAM)
                    for candidate in candidates:
                        candidate_key = plan_key(candidate['plan'].solution, candidate['plan'].directions)
                        if candidate_key in seen:
                            continue
                        clock = time.perf_counter()
                        candidate_schedule, called = evaluate_once(candidate['plan'], seen,
                                                                    blocker_cache=blockers)
                        b32_s += time.perf_counter()-clock
                        calls += int(called)
                        row = routeaware_result_row(name, JOINT4_METHOD, candidate['plan'], candidate_schedule,
                            blockers.get(candidate_key, ()))
                        row.update(load_allowed=int(conflict_load_allowed(row, limits)))
                        if cmax_driven:
                            row.update(stage='training_candidate', phase='fit_origin_nodes', seed=SEED,
                                       method=CMAX_METHOD, removed_parent_ids=json.dumps(removed),
                                       gnn_score=candidate['score'],
                                       feedback=json.dumps(measured), route_lb_s=candidate['lb'],
                                       b32_calls=int(called))
                            writer([row])
                        rank = joint4_training_rank(row, initial_peak, initial_spread)
                        labelled.append(dict(plan=candidate['plan'], joint_plans=candidate['joint_plans'],
                                             rank=rank, status=row['status'],
                                             load_allowed=int(row['load_allowed']),
                                             schedule=candidate_schedule))
                if len({item['rank'] for item in labelled}) > 1:
                    prepared.append((space, labelled))
                    groups += 1
                for item in labelled:
                    if item['status'] == 'FEASIBLE':
                        feasible_total += 1
                        good_total += int(item['rank'][0] == 0)
                label_rows.append(dict(layout_id=name, stage='label', candidates=len(labelled),
                    good=sum(1 for item in labelled if item['rank'][0] == 0),
                    feasible=sum(1 for item in labelled if item['status'] == 'FEASIBLE'),
                    cores=len(cores), diagnostic=json.dumps(dict(
                        initial_peak=initial_peak, initial_spread=initial_spread))))
                print('JOINT4_FIT_GROUP', json.dumps({key_: label_rows[-1][key_] for key_ in
                    ('layout_id', 'candidates', 'good', 'feasible', 'cores')}), flush=True)
                best = None
                for item in labelled:
                    if item['status'] != 'FEASIBLE':
                        continue
                    if best is None or item['rank'][2] < best['rank'][2]:
                        best = item
                return best

            best = collect(initial, schedule, limit=3)
            if best is not None:
                collect(best['plan'], best['schedule'], limit=2)
            label_rows.append(dict(layout_id=name, stage='layout', candidates=0, good=good_total,
                feasible=feasible_total, cores=0, diagnostic=json.dumps(dict(
                    prepared_groups=groups, b32_calls=calls))))
    if not prepared:
        raise ValueError('no usable Joint4 ranking preference in TRAIN')
    history = []
    for epoch in range(1, JOINT4_EPOCHS+1):
        clock = time.perf_counter()
        order = list(range(len(prepared)))
        Random(SEED+epoch).shuffle(order)
        epoch_loss, epoch_pairs, epoch_correct, updates, gradients = [], 0, 0, 0, []
        for index in order:
            space, labelled = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            embeddings = model.encode_static(space.graph)
            replay_cache = {}
            scores = torch.stack([gnn.imitation_joint_scores(space, embeddings, model,
                item['joint_plans'], graph_cache=replay_cache).mean() for item in labelled])
            loss, pairs, correct = joint4_group_loss(scores, [item['rank'] for item in labelled])
            if loss is None:
                continue
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
            if not math.isfinite(float(loss.detach())) or not math.isfinite(gradient):
                raise ArithmeticError('nonfinite Joint4 ranking optimization')
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            epoch_pairs += pairs
            epoch_correct += correct
            updates += 1
            gradients.append(gradient)
        record = dict(stage='epoch', seed=SEED, method=JOINT4_METHOD, epoch=epoch,
            loss=sum(epoch_loss)/max(1, len(epoch_loss)),
            ranking_accuracy=epoch_correct/max(1, epoch_pairs), pairs=epoch_pairs, updates=updates,
            gradient_norm=max(gradients, default=0.), learning_rate=JOINT3_LEARNING_RATE,
            candidates=sum(len(labelled) for _, labelled in prepared),
            b32_calls=calls, construct_s=construct_s, b32_s=b32_s,
            runtime_s=time.perf_counter()-started,
            diagnostic=json.dumps(dict(training_split='TRAIN_ONLY', groups=groups,
                layouts=len(names), positives=good_total, feasible=feasible_total,
                positive_share=good_total/max(1, feasible_total),
                label='FEASIBLE ordered directly by true B32 Cmax; DEADLOCK has no Cmax')))
        history.append(record)
        if cmax_driven:
            record.update(method=CMAX_METHOD, phase='fit_origin_nodes')
        writer([record])
        torch.save(dict(seed=SEED, initialization=('models/gnn_weld_layout_joint4.pt:state_dict'
            if cmax_driven else 'models/gnn_weld_layout_joint3.pt:state_dict'),
            state_dict=copy.deepcopy(model.state_dict()), initial_state_dict=initial_state,
            optimizer_state_dict=optimizer.state_dict(), completed_epoch=epoch,
            training_complete=epoch == JOINT4_EPOCHS, history=history, training_split='TRAIN_ONLY',
            training_layouts=names, feedback_outcomes=True, b32_calls=calls,
            construct_s=construct_s, b32_s=b32_s, label_rows=label_rows,
            independent_layouts=holdout if cmax_driven else [],
            objective='FEASIBLE_B32_CMAX', graph_relations=list(gnn.GRAPH_RELATIONS)), model_path)
        print('JOINT4_FIT_EPOCH', json.dumps({key_: record[key_] for key_ in
            ('epoch', 'loss', 'ranking_accuracy', 'pairs', 'updates', 'b32_calls')}), flush=True)
    return torch.load(model_path, map_location='cpu', weights_only=True)


def cmax_summary(phase='independent'):
    rows = [row for row in read_rows(CMAX_RESULTS) if row.get('stage') == 'final']
    chosen = {(r['layout_id'], int(r['seed'])): r for r in rows
              if r.get('phase') == phase and r.get('method') == CMAX_METHOD}
    if phase == 'independent':
        expected = [(name, SEED) for name in cmax_train_split()[1]]
        references = {'Joint4': {(r['layout_id'], int(r['seed'])): r for r in rows
                                if r.get('phase') == 'original_independent'}}
    else:
        expected = [(name, seed) for name, _ in load_layouts(phase) for seed in JOINT3_SEEDS]
        references = {}
        for method, path, source in (('HGA', JOINT3_HGA_REFERENCE, JOINT3_HGA_SOURCE),
                                    ('WAG+VNS', JOINT3_WAG_REFERENCE, JOINT3_WAG_SOURCE)):
            reference = {}
            for row in read_rows(path):
                if row.get('source') != source or not completed_baseline_row(row):
                    continue
                key = (row['layout_id'], int(row['seed']))
                if key in expected:
                    reference[key] = row
            references[method] = reference
    output = dict(phase=phase, expected=len(expected), completed=sum(k in chosen for k in expected),
                  feasible=sum(k in chosen and chosen[k]['status'] == 'FEASIBLE' for k in expected),
                  paired={}, detail=[])
    for method, reference in references.items():
        deltas, wins, losses, ties, feasibility_wins = [], 0, 0, 0, 0
        for key in expected:
            new, old = chosen.get(key), reference.get(key)
            if not new or not old:
                continue
            if new['status'] == old['status'] == 'FEASIBLE':
                delta = 100.*(float(new['cmax_s'])/float(old['cmax_s'])-1.)
                deltas.append(delta)
                wins += delta < -1e-9; losses += delta > 1e-9; ties += abs(delta) <= 1e-9
                output['detail'].append(dict(layout_id=key[0], seed=key[1], reference=method,
                    original_cmax_s=float(old['cmax_s']), cmax_s=float(new['cmax_s']), delta_pct=delta,
                    original_runtime_s=float(old['runtime_s']), runtime_s=float(new['runtime_s']),
                    original_b32=int(old['b32_calls']), b32_calls=int(new['b32_calls'])))
            elif new['status'] == 'FEASIBLE' and old['status'] == 'DEADLOCK':
                feasibility_wins += 1
        output['paired'][method] = dict(pairs=len(deltas), median_delta_pct=median(deltas) if deltas else None,
            wins=int(wins), losses=int(losses), ties=int(ties), feasibility_wins=feasibility_wins)
    if phase == 'independent':
        paired = output['paired']['Joint4']
        output['passed'] = (output['completed'] == output['feasible'] == 6 and paired['pairs'] == 6
            and paired['median_delta_pct'] <= -1. and paired['wins'] >= 4)
    else:
        output['passed'] = (output['completed'] == output['feasible'] == len(expected)
            and all(set(expected) <= set(reference) for reference in references.values())
            and all(p['median_delta_pct'] is not None and p['median_delta_pct'] < 0.
                    and p['wins'] > p['losses'] for p in output['paired'].values()))
    return output


def cmax_evaluate(phase='independent'):
    """The requested sequential experiment stages, using only the existing B32."""
    if phase == 'validation' and not cmax_summary('independent')['passed']:
        print('CMAX_STOP: independent TRAIN comparison did not pass', flush=True)
        return []
    if phase == 'test' and not cmax_summary('validation')['passed']:
        print('CMAX_STOP: VALIDATION comparison did not pass', flush=True)
        return []
    saved = torch.load(CMAX_MODEL, map_location='cpu', weights_only=True)
    if not saved.get('training_complete'):
        raise ValueError('Cmax training is incomplete')
    model = joint4_model(saved['state_dict']).eval()
    model.requires_grad_(False)
    layouts = dict(load_layouts('train' if phase == 'independent' else phase))
    names = saved['independent_layouts'] if phase == 'independent' else list(layouts)
    seeds = (SEED,) if phase == 'independent' else JOINT3_SEEDS
    finals = [joint3_case(name, layouts[name], seed, model, {}, budget=100,
        variant='joint4', sink='cmax', phase=phase, limit_mode='relative')
        for name in names for seed in seeds]
    print('CMAX_COMPARISON', json.dumps(cmax_summary(phase), ensure_ascii=False), flush=True)
    return finals


def joint3_summary():
    """Thirty paired VALIDATION comparisons against the frozen HGA/WAG+VNS results."""
    references = joint3_references()
    finals = {(row['layout_id'], int(row['seed'])): row for row in read_rows(JOINT3_RESULTS)
              if row.get('stage') == 'final' and row.get('method') == JOINT3_METHOD}
    missing = [(name, seed) for name in JOINT3_LAYOUTS for seed in JOINT3_SEEDS
               if (name, seed) not in finals]
    result = dict(algorithm=JOINT3_METHOD, expected=len(JOINT3_LAYOUTS)*len(JOINT3_SEEDS),
                  completed=len(JOINT3_LAYOUTS)*len(JOINT3_SEEDS)-len(missing), missing=missing)
    wag_rows = read_rows(JOINT3_WAG_REFERENCE)
    for tag, method in (('HGA', 'HGA'), ('WAG', 'WAG+VNS')):
        deltas, wins, losses, ties, feasibility_wins, no_reference = [], 0, 0, 0, 0, 0
        for name in JOINT3_LAYOUTS:
            reference = references.get((name, method))
            for seed in JOINT3_SEEDS:
                row = finals.get((name, seed))
                if row is None:
                    continue
                feasible = row['status'] == 'FEASIBLE' and row.get('cmax_s') not in (None, '')
                if not feasible:
                    losses += 1
                    continue
                if reference is None:
                    # No feasible frozen row exists for this layout: a feasible final
                    # is a feasibility result, never a fabricated Cmax pair.
                    wins += 1
                    feasibility_wins += 1
                    no_reference += 1
                    continue
                if tag == 'WAG':
                    deadlock_seed = [other for other in wag_rows if other['layout_id'] == name
                                     and int(other['seed']) == seed
                                     and other.get('source') == JOINT3_WAG_SOURCE
                                     and other['status'] != 'FEASIBLE']
                    if deadlock_seed:
                        wins += 1
                        feasibility_wins += 1
                        continue
                delta = (float(row['cmax_s'])-reference['cmax_s'])/reference['cmax_s']*100.
                deltas.append(delta)
                wins += int(delta < -1e-9)
                losses += int(delta > 1e-9)
                ties += int(abs(delta) <= 1e-9)
        result[tag] = dict(pairs=len(deltas), median_pct=median(deltas) if deltas else None,
                           wins=wins, losses=losses, ties=ties, feasibility_wins=feasibility_wins,
                           pairs_without_frozen_reference=no_reference,
                           reference_cmax_s={name: references[(name, method)]['cmax_s']
                                             for name in JOINT3_LAYOUTS if (name, method) in references},
                           reference_runtime_s=median([references[(name, method)]['runtime_s']
                                                       for name in JOINT3_LAYOUTS
                                                       if (name, method) in references]))
    rows = [finals[(name, seed)] for name in JOINT3_LAYOUTS for seed in JOINT3_SEEDS
            if (name, seed) in finals]
    feasible = [row for row in rows if row['status'] == 'FEASIBLE']
    load_ok = [row for row in feasible if int(row.get('load_allowed') or 0)]
    load_ok_three = [row for row in feasible if int(row.get('load_allowed_three_ref') or 0)]
    deadlocks = sum(1 for row in rows if row['status'] == 'DEADLOCK')
    evaluated = sum(int(json.loads(row['diagnostic']).get('evaluated') or 0) for row in rows) \
        if rows and rows[0].get('diagnostic', '').startswith('{') else None
    candidate_deadlocks = sum(int(json.loads(row['diagnostic']).get('deadlock_candidates') or 0)
                              for row in rows if row.get('diagnostic', '').startswith('{'))
    candidate_total = sum(int(json.loads(row['diagnostic']).get('evaluated') or 0)
                          for row in rows if row.get('diagnostic', '').startswith('{'))
    result.update(feasible=len(feasible), deadlock_finals=deadlocks,
                  load_allowed_strict=len(load_ok), load_allowed_three_reference=len(load_ok_three),
                  final_feasible_rate=len(feasible)/max(1, len(rows)),
                  candidate_deadlock_rate=candidate_deadlocks/max(1, candidate_total),
                  median_runtime_s=median([float(row['runtime_s']) for row in rows]) if rows else None,
                  median_b32_calls=median([int(row['b32_calls']) for row in rows]) if rows else None,
                  duration_seconds={})
    thresholds = dict(final_feasible_load_ok=len(load_ok) == 30,
                      hga_median_below_zero=(result['HGA']['median_pct'] is not None
                                             and result['HGA']['median_pct'] < 0),
                      wag_median_below_zero=(result['WAG']['median_pct'] is not None
                                             and result['WAG']['median_pct'] < 0),
                      hga_wins_more=result['HGA']['wins'] > result['HGA']['losses'],
                      wag_wins_more=result['WAG']['wins'] > result['WAG']['losses'])
    result['criteria'] = thresholds
    result['passed'] = all(thresholds.values())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baselines', 'train', 'test', 'summary', 'beam-validation', 'baseline-summary', 'cmax-value', 'wag-deadlock-diagnosis', 'hga-alpha-sensitivity', 'hga-alpha-full-budget', 'hga-route-vnd', 'expanded-teachers', 'expanded-dagger', 'owner-cf', 'feedback-train', 'feedback-continue', 'feedback-full-validation', 'feedback-test', 'feedback-summary', 'child-feedback','ppo-scale-smoke','ppo-scale-train','ppo-scale-baselines','ppo-scale-validation','ppo-scale-summary','imitation-teachers','imitation-train','imitation-select','imitation-references','imitation-pilot','imitation-summary','imitation-deadlock-pilot','imitation-routeaware-pilot','imitation-loadfirst-screen','imitation-loadfirst-pilot','imitation-priority-diagnosis','imitation-priority-vnd','imitation-conflict-core-pilot','imitation-conflict-load-pilot','imitation-cycle-transfer-pilot','joint3-audit','joint3-train','joint3-validation','joint3-summary','joint3b-train','joint3b-validation','joint3b-summary','joint4-fit','joint4-train','joint4-validation','joint4-summary','cmax-fit','cmax-independent','cmax-validation','cmax-test','cmax-summary'))
    parser.add_argument('--size',type=int,choices=(0,120,500,1000),help='PPO validation model; 0 is the unchanged initialization')
    parser.add_argument('--workers', type=int, default=4, help='parallel runs or TRAIN teacher processes')
    parser.add_argument('--resume-unresolved',action='store_true',help='continue unresolved deadlock cases within the original 12-candidate budget')
    parser.add_argument('--method', choices=tuple(BASELINES), help='limit the baseline stage to one method')
    parser.add_argument('--split', choices=('train', 'validation', 'test'), default='validation', help='baseline dataset')
    parser.add_argument('--representative', action='store_true', help='small/median/large layouts, first seed only')
    parser.add_argument('--rounds', type=int, choices=(1, 2, 3), default=3, help='maximum feedback rounds; stop if validation does not improve')
    parser.add_argument('--time-limit', type=float, choices=(EQUAL_TIME_LIMIT,),
                        help='120-second TEST comparison; omit for the original full iteration budget')
    args = parser.parse_args()
    if args.time_limit is not None and args.stage not in ('baselines', 'baseline-summary'):
        parser.error('--time-limit is only supported by baselines and baseline-summary')
    configure_cpu()
    if args.stage == 'cmax-fit':
        joint4_fit(args.workers, cmax_driven=True)
    elif args.stage in ('cmax-independent', 'cmax-validation', 'cmax-test'):
        cmax_evaluate(args.stage.removeprefix('cmax-'))
    elif args.stage == 'cmax-summary':
        print('CMAX_SUMMARY', json.dumps(cmax_summary('independent' if args.split == 'train' else args.split),
                                       ensure_ascii=False), flush=True)
    elif args.stage == 'joint3-audit':
        print('JOINT3_AUDIT', json.dumps(joint3_provenance(), ensure_ascii=False), flush=True)
    elif args.stage == 'joint3-train':
        joint3_train()
    elif args.stage == 'joint3-validation':
        joint3_validation(args.workers)
    elif args.stage == 'joint3-summary':
        print('JOINT3_SUMMARY', json.dumps(joint3_summary(), ensure_ascii=False), flush=True)
    elif args.stage == 'joint3b-train':
        joint3b_train_stage(args.workers)
    elif args.stage == 'joint3b-validation':
        joint3b_validation(args.workers)
    elif args.stage == 'joint3b-summary':
        print('JOINT3B_SUMMARY', json.dumps(joint3b_summary(phase=args.split), ensure_ascii=False), flush=True)
    elif args.stage == 'joint4-fit':
        joint4_fit(args.workers)
    elif args.stage == 'joint4-train':
        joint4_train_stage(args.workers)
    elif args.stage == 'joint4-validation':
        joint4_validation(args.workers)
    elif args.stage == 'joint4-summary':
        print('JOINT4_SUMMARY', json.dumps(joint4_summary(phase=args.split), ensure_ascii=False), flush=True)
    elif args.stage == 'imitation-cycle-transfer-pilot':imitation_cycle_transfer_pilot()
    elif args.stage == 'imitation-conflict-load-pilot':imitation_conflict_load_pilot()
    elif args.stage == 'imitation-conflict-core-pilot':imitation_conflict_core_pilot()
    elif args.stage in ('imitation-priority-vnd','imitation-deadlock-pilot'):priority_pilot()
    elif args.stage == 'imitation-priority-diagnosis':priority_history_diagnosis()
    elif args.stage == 'imitation-loadfirst-screen':imitation_loadfirst_pilot(screen_only=True)
    elif args.stage == 'imitation-loadfirst-pilot':imitation_loadfirst_pilot()
    elif args.stage == 'imitation-routeaware-pilot':imitation_routeaware_pilot()
    elif args.stage == 'imitation-teachers':imitation_teachers(args.workers)
    elif args.stage == 'imitation-train':imitation_train()
    elif args.stage == 'imitation-select':imitation_select()
    elif args.stage == 'imitation-references':imitation_references()
    elif args.stage == 'imitation-pilot':imitation_pilot()
    elif args.stage == 'imitation-summary':imitation_summary()
    elif args.stage == 'ppo-scale-smoke':ppo_smoke()
    elif args.stage == 'ppo-scale-train':ppo_train(args.workers)
    elif args.stage == 'ppo-scale-baselines':ppo_baselines(args.workers)
    elif args.stage == 'ppo-scale-validation':ppo_validation(args.workers,args.size)
    elif args.stage == 'ppo-scale-summary':ppo_summary()
    elif args.stage == 'child-feedback':
        saved=child_feedback_experiment(args.workers)
        if not CHILD_MODEL.exists() and saved['training_complete']:
            with CHILD_MODEL.open('xb') as stream:
                torch.save(saved,stream)
    elif args.stage == 'feedback-train':
        train_feedback(args.workers, args.rounds)
    elif args.stage == 'feedback-continue':
        train_feedback(args.workers, args.rounds, expanded=True)
    elif args.stage == 'feedback-full-validation':
        validate_feedback_complete(args.workers)
    elif args.stage == 'feedback-summary':
        summarize_feedback(split=args.split)
    elif args.stage == 'feedback-test':
        test_feedback(args.workers)
    elif args.stage == 'baselines':
        run_baselines(args.workers, args.method, split=args.split, representative=args.representative,
                      time_limit=args.time_limit)
    elif args.stage == 'hga-alpha-sensitivity':
        hga_alpha_sensitivity()
    elif args.stage == 'hga-alpha-full-budget':
        hga_alpha_sensitivity(full_budget=True)
    elif args.stage == 'hga-route-vnd':
        hga_route_vnd_experiment()
    elif args.stage == 'train':
        train(args.workers)
    elif args.stage == 'test':
        test(args.workers)
    elif args.stage == 'beam-validation':
        beam_validation(args.workers)
    elif args.stage == 'baseline-summary':
        summarize_optimized_baselines(time_limit=args.time_limit)
    elif args.stage == 'wag-deadlock-diagnosis':
        diagnose_wag_deadlocks(args.workers)
    elif args.stage == 'cmax-value':
        cmax_value_experiment(args.workers)
    elif args.stage == 'expanded-teachers':
        expanded_teachers(args.workers)
    elif args.stage == 'expanded-dagger':
        expanded_dagger_experiment(args.workers)
    elif args.stage == 'owner-cf':
        owner_cf_experiment(args.workers)
    else:
        write_comparison()


if __name__ == '__main__':
    main()
