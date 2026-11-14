"""Run the common-baseline and joint VNext study on the filtered weld layouts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
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
from mrta_reference.scheduler import build_operation_templates, earliest_safe_start
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


def b32(result):
    return schedule_explicit(result.solution, result.directions, CONFIG).schedule


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


def baseline_run(parents, method, seed, budget, checkpoints, *, iterations=None, candidate_evaluator=None):
    run, make_settings, _ = BASELINES[method]
    settings = make_settings()
    if iterations is not None:
        name = 'imax' if method == 'WAG+VNS' else 'max_iterations'
        settings = replace(settings, **{name: iterations})
    return run(parents, CONFIG, settings, seed=seed, time_limit=budget,
               checkpoints=checkpoints, certify=False, candidate_evaluator=candidate_evaluator)


def baseline_job(job):
    (name, parents), method, seed = job
    started = time.perf_counter()
    result = baseline_run(parents, method, seed, math.inf, CHECKPOINTS,
                          iterations=BASELINE_ITERATIONS[method], candidate_evaluator=NATIVE_EVALUATORS[method])
    print('BASELINE_STATS', json.dumps(dict(layout_id=name, method=method, seed=seed,
          source=BASELINE_SOURCES[method], iterations=result.iterations,
          accounting={key: value for key, value in result.accounting.items()
                      if value is None or isinstance(value, (int, float))})), flush=True)
    best = result.timely_best
    schedule = schedule_explicit(best.solution, best.directions, CONFIG).schedule if best else None
    if schedule is None:
        diagnostic = f"Native deadlocks={result.accounting['remaining_deadlock']} | {result.termination_reason}"
    else:
        diagnostic = '' if schedule.feasible else ' | '.join(schedule.diagnostics[:2])
    return dict(layout_id=name, seed=seed, method=method,
                status=schedule.status.value if schedule else 'NO_NATIVE_SOLUTION',
                cmax_s=schedule.cmax if schedule else None, runtime_s=time.perf_counter() - started,
                b32_calls=int(schedule is not None), diagnostic=diagnostic,
                source=BASELINE_SOURCES[method], iterations=result.iterations,
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


def hga_alpha_sensitivity():
    """Sequential one-seed screening; each alpha starts a fresh native HGA."""
    from numba import set_num_threads
    configure_cpu()
    set_num_threads(1)
    layouts = dict(load_layouts('validation'))
    names = ('validation_03', 'validation_04', 'validation_07')
    alphas = (16, 18, 20, 22, 24, 26)
    settings = AdaptedHGAConfig()
    print('HGA_ALPHA_SETTINGS', settings, 'seed', SEED, 'workers/threads', 1, flush=True)
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
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for name in names:
            for alpha in alphas:
                print('HGA_ALPHA_START', name, 'parents', len(layouts[name]), 'alpha', alpha, flush=True)
                started = time.perf_counter()
                result = run_adapted_hga_v2(layouts[name], CONFIG,
                    replace(settings, alpha=alpha, max_iterations=1000), seed=SEED,
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
                    run_status='COMPLETED' if result.iterations == 1000 else 'INCOMPLETE')
                writer.writerow(row)
                stream.flush()
                rows.append(row)
                print('HGA_ALPHA_RESULT', json.dumps(row), flush=True)
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


def run_baselines(workers, only=None, *, split='validation', representative=False):
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
                    if row['source'] == BASELINE_SOURCES[row['method']])
    jobs = [(layout, method, seed) for layout in layouts for seed in seeds for method in paths
            if (layout[0], method, seed) not in done]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_cpu) as executor:
        futures = [executor.submit(baseline_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print('BASELINE', json.dumps(row), flush=True)
            write_rows(paths[row['method']], [row], append=True)


def wag_deadlock_job(job):
    """Replay one historical incumbent, then observe the existing dispatch kernel.

    Diagnostic rollouts happen only after the complete 500-round search. They
    never provide feedback to that search and do not change the B32 policy.
    """
    from collections import Counter
    from mrta_reference import scheduler as scheduler
    from mrta_reference.dispatch_recovery import run_dispatch_rollout
    from mrta_reference.geometry import finite_x_split_validator, operations_conflict
    (name, parents), historical, *saved = job
    seed = int(historical['seed'])
    if saved and saved[0] is not None:
        from mrta_reference.geometry import build_legal_pattern_catalog
        catalog = build_legal_pattern_catalog(parents, CONFIG, FORMAL_SCOPE_V2)
        lookup = {p.pattern_id: p for options in catalog.values() for p in options}
        snapshot = saved[0]
        solution = canonicalize(parents, tuple(lookup[p] for p in snapshot['patterns']),
                                dict(enumerate(snapshot['routes'])), CONFIG, scope=FORMAL_SCOPE_V2)
        directions = tuple(tuple(ds) for ds in snapshot['directions'])
        native, iterations, rerun = priority_schedule(solution, CONFIG), snapshot['iterations'], 0
    else:
        result = baseline_run(parents, 'WAG+VNS', seed, math.inf, (),
                              iterations=500, candidate_evaluator=priority_schedule)
        best = result.timely_best
        if best is None:
            return dict(layout_id=name, seed=seed, reproduced=False, reason='no native incumbent')
        solution, directions, native = best.solution, best.directions, best.schedule
        iterations, rerun = result.iterations, result.iterations
    formal = schedule_explicit(solution, directions, CONFIG).schedule
    reproduced = (iterations == 500 and formal.status.value == historical['status']
                  and abs(native.cmax - float(historical['native_cmax_s'])) <= 1.e-6)
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
    first_tasks = [dict(robot=r.robot_id, block=r.blocks[0].block_id, direction=r.orientations[0],
                        start=oriented_endpoints(r.blocks[0], r.orientations[0])[0],
                        end=oriented_endpoints(r.blocks[0], r.orientations[0])[1]) for r in routes if r.blocks]
    record = dict(layout_id=name, seed=seed, reproduced=reproduced, iterations=iterations,
        search_rerun_iterations=rerun, replay_mode='FULL_SEARCH' if rerun else 'SAVED_INCUMBENT',
        native_cmax_s=native.cmax, historical_native_cmax_s=float(historical['native_cmax_s']),
        b32_status=formal.status.value, b32_cmax_s=formal.cmax,
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
        first_tasks=first_tasks, native_coverage=coverage, native_inter_robot_conflicts=conflicts,
        native_chain_continuous=continuous, native_weld_order=weld_order,
        native_wait_s=sum(op.duration for op in native.operations if op.kind is OperationKind.WAIT),
        # Retain the reconstructed incumbent in the existing report so the
        # diagnostic can be independently replayed without another WAG search.
        patterns=[p.pattern_id for p in solution.patterns], routes=[r.block_ids for r in solution.routes],
        directions=directions)
    print('WAG_DEADLOCK_DIAGNOSIS', json.dumps({k: v for k, v in record.items()
          if k not in ('patterns', 'routes', 'directions')}), flush=True)
    return record


def diagnose_wag_deadlocks(workers):
    historical = [r for r in read_rows(FINAL_RESULTS / BASELINES['WAG+VNS'][2])
                  if r['source'] == 'OPTIMIZED_ITERATIONS' and r['layout_id'].startswith('validation_')
                  and r['status'] == 'DEADLOCK']
    layouts = dict(load_layouts('validation'))
    report = ROOT / 'docs' / 'WAG_HGA_BASELINE_REPORT_20261111.md'
    text = report.read_text(encoding='utf-8')
    start_marker = '<!-- WAG_DEADLOCK_CASES_START -->'
    snapshots = {}
    if start_marker in text:
        raw = text.split(start_marker, 1)[1].split('```json', 1)[1].split('```', 1)[0]
        snapshots = {(r['layout_id'], r['seed']): r for r in json.loads(raw) if r.get('reproduced')}
    jobs = [((r['layout_id'], layouts[r['layout_id']]), r,
             snapshots.get((r['layout_id'], int(r['seed'])))) for r in historical]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_cpu) as executor:
        records = list(executor.map(wag_deadlock_job, jobs))
    records.sort(key=lambda r: (r['layout_id'], r['seed']))
    lines = ['\n### 13.3 WAG 历史 11 次死锁的逐例诊断\n',
             '以下是原 WAG 完成 500 轮后的固定最优路线重放；诊断不反馈搜索。首次从完整 500 轮重建最优路线，随后从本报告保存的模式、路线和方向重放派工，避免 LNS 修正后拿新最优路线冒充旧实例。`search_rerun_iterations=0` 表示只重放已有旧方案。`reproduced` 核对原生 Cmax、最终状态和原搜索预算。B32 未找到时间线不构成数学不可行证明。阻塞边累计值是各派工状态 ESS 返回关联的累计次数，不能解释为独立冲突事件或唯一因果。\n',
             '| 排样 | 种子 | 重现 | 原生 Cmax/s | B32 | 同轨/跨轨阻塞数 | 等待环 | 原生时间线冲突数 |',
             '|---|---:|---|---:|---|---|---|---:|']
    for r in records:
        blocks = r.get('terminal_blockers', ())
        same, cross = sum(x['same_rail_blockers'] for x in blocks), sum(x['cross_rail_blockers'] for x in blocks)
        lines.append(f"| {r['layout_id']} | {r['seed']} | {r['reproduced']} | {r.get('native_cmax_s', 0):.4f} | {r.get('b32_status')} | {same}/{cross} | {r.get('wait_for_cycles')} | {r.get('native_inter_robot_conflicts')} |")
    lines += ['\n逐例首任务、方向、首次阻塞、终端等待图和重放方案：\n',
              '<!-- WAG_DEADLOCK_CASES_START -->', '```json', json.dumps(records, ensure_ascii=False, indent=2),
              '```', '<!-- WAG_DEADLOCK_CASES_END -->', '']
    marker = '\n### 13.3 WAG 历史 11 次死锁的逐例诊断'
    # This section is generated evidence in an existing report, not a new file.
    if marker in text:
        end_marker = '<!-- WAG_DEADLOCK_CASES_END -->'
        end = text.index(end_marker, text.index(marker)) + len(end_marker)
        text = text[:text.index(marker)] + '\n' + '\n'.join(lines) + text[end:]
    else:
        text = text.rstrip() + '\n' + '\n'.join(lines)
    report.write_text(text, encoding='utf-8')
    print('WAG_DIAGNOSIS_REPORT', report, len(records), flush=True)


FINAL_MODEL = ROOT / 'models' / 'gnn_weld_layout_final.pt'
CMAX_VALUE_MODEL = ROOT / 'models' / 'gnn_weld_layout_cmax_value.pt'
CMAX_VALUE_RESULTS = FINAL_RESULTS / 'gnn_cmax_value_results.csv'
CMAX_VALUE_FIELDS = ('split', 'layout_id', 'method', 'epoch', 'fraction', 'assigned_parents', 'candidate',
                     'parent_id', 'mode', 'robots', 'status', 'cmax_s',
                     'processing_r0_s', 'processing_r1_s', 'processing_r2_s', 'processing_r3_s',
                     'max_processing_s', 'move_s', 'runtime_s', 'b32_calls', 'diagnostic')
TRAIN_BUDGET = 12.
EPOCHS = 6
VALIDATE_EPOCHS = (2, 4, 6)
JOINT_METHODS = ('GNN_JOINT', 'GNN_JOINT_RECONSTRUCT')
GNN_RESULT_SOURCE = 'COMPLETE_CONSTRUCTION'


def configure_cpu():
    torch.set_num_threads(1)


def joint_model(weights=None, *, cmax_value=False):
    model = gnn.VNextGNN(workload_lookahead=True, joint_actions=True, cmax_value=cmax_value)
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


def evaluate_once(result, seen):
    if result.solution is None:
        return None, False
    key = plan_key(result.solution, result.directions)
    if key in seen:
        return seen[key], False
    schedule = b32(result)
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


@torch.no_grad()
def collect_teachers(layout):
    name, parents = layout
    model = worker_model
    teachers, seen = [], {}
    beam = rule_beam(parents)
    schedule, _ = evaluate_once(beam, seen)
    if beam.solution is not None:
        teachers.append(('RULE/BEAM', beam.solution, beam.directions, schedule))
    for method in BASELINES:
        search = baseline_run(parents, method, SEED, TRAIN_BUDGET, (TRAIN_BUDGET,))
        best = search.timely_best
        if best is not None:
            teachers.append((method, best.solution, best.directions, best.schedule))
            seen[plan_key(best.solution, best.directions)] = best.schedule

    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    best_cmax = min((s.cmax for _, _, _, s in teachers if s.feasible), default=None)
    votes = torch.zeros(len(space.actions))
    assignment_steps, insertion_steps, preferences = [], [], []
    for _, solution, directions, schedule in teachers:
        labels = gnn.parent_assignment_targets(solution, CONFIG)
        teacher_votes = torch.zeros_like(votes)
        teacher_votes[[space.action_index[a] for a in labels]] = best_cmax / schedule.cmax if schedule.feasible else .35
        votes += teacher_votes
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
    preference_types = {}
    for state, selected, plans in prefixes:
        blocks = {b.block_id: b for p in selected.values() for b in space.children[p]}
        example = assignment_example(space, model, state, blocks, set(space.parent_ids) - selected.keys(), votes)
        if example is not None:
            assignment_steps.append(example)
        results = []
        for plan in plans:
            action = space.actions[plan.assignment_index]
            result = build_joint(parents, model, space=space, embeddings=embeddings, state=plan.state,
                                 selected={**selected, action.parent_id: action.pattern})
            outcome, called = evaluate_once(result, seen)
            branch_calls += int(called)
            branches += 1
            results.append(outcome)
        winner, kind = better_branch(*results)
        if winner is None:
            continue
        good, bad = plans[winner], plans[1 - winner]
        preferences.append((torch.tensor((good.assignment_index, bad.assignment_index)),
                            torch.stack((good.assignment_features, bad.assignment_features)),
                            torch.stack((good.outcome, bad.outcome))))
        preference_types[kind] = preference_types.get(kind, 0) + 1

    evidence = dict(name=name, teachers={m: (s.status.value, s.cmax) for m, _, _, s in teachers},
                    prefixes=len(prefixes), branches=branches, branch_b32=branch_calls, preferences=preference_types,
                    assignment_prefixes=len(assignment_steps), insertion_prefixes=len(insertion_steps))
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
        totals = torch.zeros(3)
        for index in order:
            sample = samples[index]
            embeddings = model.encode_static(sample.space.graph)
            a, ins = sample.assignment, sample.insertion
            rows = model.embed_assignment_rows(sample.space, embeddings, a.indices, a.features)
            assignment = ranking_loss(model.assignment_scores(rows), a)
            insertion = (ranking_loss(model.insertion_logits(embeddings, ins.indices, ins.features), ins)
                         if ins is not None else assignment.new_zeros(()))
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
        record = dict(epoch=epoch, **dict(zip(('assignment_loss', 'insertion_loss', 'cmax_rank_loss'),
                                             (totals / len(samples)).tolist())))
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
                    teacher_budget_s=TRAIN_BUDGET, validation_stopping='one construction and up to two reconstructions',
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


def summarize_optimized_baselines():
    """Read the two result files and retained references; never rewrite history."""
    rows = [r for _, _, name in BASELINES.values() for r in read_rows(FINAL_RESULTS / name)
            if r['source'] == BASELINE_SOURCES[r['method']]]
    references = read_rows(BEAM_VALIDATION_CSV)
    references += read_rows(FINAL_RESULTS / 'rule_beam_results.csv')
    references += [r for r in read_rows(FINAL_RESULTS / 'gnn_results.csv')
                   if r['source'] == GNN_RESULT_SOURCE or r['method'] == 'GNN_VNEXT']
    def average(values):
        return sum(values) / len(values) if values else None
    summaries = []
    for split in ('validation', 'test'):
        for method in BASELINES:
            selected = [r for r in rows if r['method'] == method and r['layout_id'].startswith(split + '_')]
            if not selected:
                continue
            feasible = [r for r in selected if r['status'] == 'FEASIBLE']
            cmax = [float(r['cmax_s']) for r in feasible]
            times = [float(r['runtime_s']) for r in selected]
            summary = dict(split=split, method=method, runs=len(selected), feasible=len(feasible),
                           median_cmax_s=median(cmax) if cmax else None, mean_cmax_s=average(cmax),
                           mean_runtime_s=average(times), max_runtime_s=max(times),
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
def collect_value_branches(layout):
    name, parents = layout
    started = time.perf_counter()
    model = worker_model
    if model.cmax_value_head is not None:
        raise ValueError('TRAIN continuation labels must use the original frozen GNN')
    model.requires_grad_(False)
    space = gnn.parent_assignment_space(parents, CONFIG)
    embeddings = model.encode_static(space.graph)
    depths = {math.ceil(len(parents) * fraction): fraction for fraction in (.25, .50, .75)}
    prefixes = []

    def observe(state, selected, plans, scores):
        if len(selected) in depths:
            indices = gnn.vnext_value_candidate_indices(space, plans, scores)
            chosen = tuple(plans[i] for i in indices)
            pending = set(space.parent_ids) - selected.keys()
            features = model.cmax_value_inputs(space, embeddings, state, pending, chosen, scores[list(indices)])
            prefixes.append((state, dict(selected), chosen, features.cpu()))

    original = build_joint(parents, model, space=space, embeddings=embeddings,
                           beam_width=1, step_observer=observe)
    original_seconds = time.perf_counter() - started
    groups, rows, seen = [], [], {}
    for state, selected, plans, features in prefixes:
        fraction = depths[len(selected)]
        schedules, descriptions = [], []
        for index, plan in enumerate(plans, 1):
            branch_started = time.perf_counter()
            assignment = space.actions[plan.assignment_index]
            result = build_joint(parents, model, space=space, embeddings=embeddings, beam_width=1,
                                 state=plan.state, selected={**selected, assignment.parent_id: assignment.pattern})
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
    return dict(layout_id=name, parents=len(parents), groups=groups, rows=rows,
                original_construct_s=original_seconds, original_complete=original.solution is not None,
                original_diagnostic=original.rejection_reason, runtime_s=time.perf_counter() - started)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baselines', 'train', 'test', 'summary', 'beam-validation', 'baseline-summary', 'cmax-value', 'wag-deadlock-diagnosis', 'hga-alpha-sensitivity'))
    parser.add_argument('--workers', type=int, default=7, help='parallel runs or TRAIN teacher processes')
    parser.add_argument('--method', choices=tuple(BASELINES), help='limit the baseline stage to one method')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation', help='baseline dataset')
    parser.add_argument('--representative', action='store_true', help='small/median/large layouts, first seed only')
    args = parser.parse_args()
    configure_cpu()
    if args.stage == 'baselines':
        run_baselines(args.workers, args.method, split=args.split, representative=args.representative)
    elif args.stage == 'hga-alpha-sensitivity':
        hga_alpha_sensitivity()
    elif args.stage == 'train':
        train(args.workers)
    elif args.stage == 'test':
        test(args.workers)
    elif args.stage == 'beam-validation':
        beam_validation(args.workers)
    elif args.stage == 'baseline-summary':
        summarize_optimized_baselines()
    elif args.stage == 'wag-deadlock-diagnosis':
        diagnose_wag_deadlocks(args.workers)
    elif args.stage == 'cmax-value':
        cmax_value_experiment(args.workers)
    else:
        write_comparison()


if __name__ == '__main__':
    main()
