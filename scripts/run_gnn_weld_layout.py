"""Run the common-baseline and joint VNext study on the filtered weld layouts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import copy
from dataclasses import dataclass, replace
import json
import math
from itertools import product
from pathlib import Path
import pickle
from random import Random
from statistics import median
import sys
import time

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mrta_reference.geometry import optimize_directions, oriented_endpoints
from mrta_reference.model import Operation, OperationKind, ParentWeld, RobotRoute, ScheduleResult, ScheduleStatus, ScientificConfig
from mrta_reference.scheduler import build_operation_templates, earliest_safe_start, selected_schedule_wait_blockers
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_baselines import AdaptedHGAConfig, AdaptedWAGConfig, run_adapted_hga_v2, run_adapted_wag_vns_v2
from mrta_baselines.wag_vns import priority_schedule
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baselines', 'train', 'test', 'summary', 'beam-validation', 'baseline-summary', 'cmax-value', 'wag-deadlock-diagnosis', 'hga-alpha-sensitivity', 'hga-alpha-full-budget', 'hga-route-vnd', 'expanded-teachers', 'expanded-dagger', 'owner-cf', 'feedback-train', 'feedback-continue', 'feedback-full-validation', 'feedback-test', 'feedback-summary'))
    parser.add_argument('--workers', type=int, default=4, help='parallel runs or TRAIN teacher processes')
    parser.add_argument('--method', choices=tuple(BASELINES), help='limit the baseline stage to one method')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation', help='baseline dataset')
    parser.add_argument('--representative', action='store_true', help='small/median/large layouts, first seed only')
    parser.add_argument('--rounds', type=int, choices=(1, 2, 3), default=3, help='maximum feedback rounds; stop if validation does not improve')
    parser.add_argument('--time-limit', type=float, choices=(EQUAL_TIME_LIMIT,),
                        help='120-second TEST comparison; omit for the original full iteration budget')
    args = parser.parse_args()
    if args.time_limit is not None and args.stage not in ('baselines', 'baseline-summary'):
        parser.error('--time-limit is only supported by baselines and baseline-summary')
    configure_cpu()
    if args.stage == 'feedback-train':
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
