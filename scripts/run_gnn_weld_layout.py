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
                source='OPTIMIZED_ITERATIONS', iterations=result.iterations,
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
                    if row['source'] == 'OPTIMIZED_ITERATIONS')
    jobs = [(layout, method, seed) for layout in layouts for seed in seeds for method in paths
            if (layout[0], method, seed) not in done]
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_cpu) as executor:
        futures = [executor.submit(baseline_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            print('BASELINE', json.dumps(row), flush=True)
            write_rows(paths[row['method']], [row], append=True)


FINAL_MODEL = ROOT / 'models' / 'gnn_weld_layout_final.pt'
TRAIN_BUDGET = 12.
EPOCHS = 6
VALIDATE_EPOCHS = (2, 4, 6)
JOINT_METHODS = ('GNN_JOINT', 'GNN_JOINT_RECONSTRUCT')
GNN_RESULT_SOURCE = 'COMPLETE_CONSTRUCTION'


def configure_cpu():
    torch.set_num_threads(1)


def joint_model(weights=None):
    model = gnn.VNextGNN(workload_lookahead=True, joint_actions=True)
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
            if r['source'] == 'OPTIMIZED_ITERATIONS']
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('baselines', 'train', 'test', 'summary', 'beam-validation', 'baseline-summary'))
    parser.add_argument('--workers', type=int, default=7, help='parallel runs or TRAIN teacher processes')
    parser.add_argument('--method', choices=tuple(BASELINES), help='limit the baseline stage to one method')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation', help='baseline dataset')
    parser.add_argument('--representative', action='store_true', help='small/median/large layouts, first seed only')
    args = parser.parse_args()
    configure_cpu()
    if args.stage == 'baselines':
        run_baselines(args.workers, args.method, split=args.split, representative=args.representative)
    elif args.stage == 'train':
        train(args.workers)
    elif args.stage == 'test':
        test(args.workers)
    elif args.stage == 'beam-validation':
        beam_validation(args.workers)
    elif args.stage == 'baseline-summary':
        summarize_optimized_baselines()
    else:
        write_comparison()


if __name__ == '__main__':
    main()
