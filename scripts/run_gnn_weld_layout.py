"""Train and compare two VNext constructors on the filtered weld layouts."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass, replace
import json
import math
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

from mrta_reference.model import OperationKind, ParentWeld, ScientificConfig
from mrta_reference.scope import FORMAL_SCOPE_V2
from mrta_reference.solution import block_map, canonicalize
from mrta_search import gnn_constructor as gnn, lns
from mrta_search.initialization import build_oriented_insertion_beam
from run_gnn_cmax_finetune import vnext_rank_targets
from run_joint_oriented_repair_pilot import initial_reference, schedule_explicit


DATA = ROOT / 'data' / 'weld_layout'
MODEL = ROOT / 'models' / 'gnn_weld_layout.pt'
RESULTS = ROOT / 'results' / 'gnn_weld_layout_comparison.csv'
CONFIG = ScientificConfig()
SEED = 20261110
BEAM_SEED = 202611041
EPOCHS = 8
VALIDATE_EPOCHS = (2, 4, 6, 8)
METHODS = {'GNN_VNEXT': False, 'GNN_VNEXT_WORKLOAD': True}


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


def same_plan(first, second):
    return first.solution == second.solution and first.directions == second.directions


@dataclass
class Choices:
    indices: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor
    sizes: tuple[int, ...]

    @classmethod
    def from_steps(cls, steps):
        return cls(*(torch.cat(columns) for columns in zip(*steps)), tuple(len(step[0]) for step in steps))


@dataclass
class TrainingLayout:
    name: str
    space: gnn.ParentAssignmentSpace
    assignment: tuple[Choices, Choices]
    insertion: Choices
    preference: tuple[int, int] | None
    assignment_quality: float
    insertion_quality: float
    teacher_outcomes: tuple[tuple[str, float | None], ...]


def compare_teacher_actions(beam, prefixes, beam_cmax, b_cmax):
    """Find an actual divergent insertion from the same directed prefix."""
    if beam_cmax is None or b_cmax is None or math.isclose(beam_cmax, b_cmax, rel_tol=1e-9, abs_tol=1e-6):
        return None
    blocks = block_map(beam.solution, CONFIG, scope=FORMAL_SCOPE_V2)
    state = lns.oriented_route_state(((),) * 4, ((),) * 4, blocks, CONFIG)
    for action in beam.actions:
        key = state.routes, state.directions, action.block_id, action.robot
        if key in prefixes:
            offset, options, selected = prefixes[key]
            if action in options and action != options[selected]:
                beam_index, b_index = offset + options.index(action), offset + selected
                return (beam_index, b_index) if beam_cmax < b_cmax else (b_index, beam_index)
        variants, _ = lns.oriented_insert_variants(state, blocks[action.block_id], action.robot,
                                                  action.position, blocks, CONFIG, keep=8)
        state = next(after for after, candidate in variants if candidate == action)
    return None


@torch.no_grad()
def collect_teachers(layout):
    name, parents = layout
    beam = rule_beam(parents)
    if beam.solution is None:
        raise ValueError(f'{name}: RULE/BEAM has no complete assignment ({beam.rejection_reason})')
    beam_schedule = b32(beam)
    assignments = gnn.parent_assignment_targets(beam.solution, CONFIG)
    space = gnn.parent_assignment_space(parents, CONFIG)
    policies = tuple(gnn.VNextGNN(workload_lookahead=enabled) for enabled in METHODS.values())
    teacher_ids = {a.parent_id: space.action_index[a] for a in assignments}
    teacher_mask = torch.zeros(len(space.actions), dtype=torch.bool)
    teacher_mask[list(teacher_ids.values())] = True
    selected_patterns = {a.parent_id: a.pattern for a in assignments}
    all_blocks = {b.block_id: b for p in selected_patterns.values() for b in space.children[p]}
    all_nodes = {b.block_id: space.block_nodes[p, b.block_id] for p in selected_patterns.values() for b in space.children[p]}
    assignment_steps = ([], [])
    insertion_steps, prefixes = [], {}
    prefix = None
    offset = 0
    # Sample prefixes over the whole episode, including the empty and last states.
    depths = set(torch.linspace(0, len(parents) - 1, min(24, len(parents))).round().int().tolist())

    def before_parent(state, blocks, pending):
        nonlocal prefix
        prefix = (state, blocks, pending) if len(parents) - len(pending) in depths else None

    def before_insertion(state, pattern, variants, scores):
        nonlocal prefix, offset
        if prefix is not None:
            for policy, steps in zip(policies, assignment_steps):
                indices, features = policy.assignment_features(space, space.graph.node_features, *prefix, CONFIG)
                votes = teacher_mask[indices].float()
                if votes.sum() > 0:
                    # Match ownership for all pending parents, with extra weight on B's next parent.
                    votes += (indices == teacher_ids[pattern.parent_id]) * votes.sum() / 3
                    steps.append((indices, features, votes / votes.sum()))
            prefix = None
        if len(variants) < 2:
            return
        visible = sorted({b for route in state.routes for b in route} |
                         {b.block_id for b in space.children[pattern]})
        graph = replace(space.graph, block_index={b: all_nodes[b] for b in visible})
        indices, features = gnn.vnext_insertion_inputs(graph, state, variants,
                                                      {b: all_blocks[b] for b in visible}, CONFIG)
        actions = tuple(a for _, a in variants)
        selected = min(range(len(scores)), key=scores.__getitem__)
        key = state.routes, state.directions, actions[0].block_id, actions[0].robot
        prefixes[key] = offset, actions, selected
        insertion_steps.append((indices, features, vnext_rank_targets(scores)))
        offset += len(variants)

    method_b = gnn.decode_parent_assignments(parents, assignments, CONFIG, coordination_aware=True,
                                            decision_observer=before_parent, insertion_observer=before_insertion)
    b_schedule = beam_schedule if same_plan(beam, method_b) else b32(method_b)
    preference = compare_teacher_actions(beam, prefixes, beam_schedule.cmax, b_schedule.cmax)
    work_bound = sum(CONFIG.process_time(p.length) for p in parents) / 4
    feasible = [s.cmax for s in (beam_schedule, b_schedule) if s.feasible]
    assignment_quality = work_bound / min(feasible) if feasible else 0.
    insertion_quality = work_bound / b_schedule.cmax if b_schedule.feasible else 0.
    # Serialize in memory so Windows workers do not need a tensor-sharing directory.
    return pickle.dumps(TrainingLayout(name, space, tuple(Choices.from_steps(s) for s in assignment_steps),
                                       Choices.from_steps(insertion_steps), preference,
                                       assignment_quality, insertion_quality,
                                       tuple((s.status.value, s.cmax) for s in (beam_schedule, b_schedule))))


def configure_cpu():
    torch.set_num_threads(1)


def prepare_training(layouts, workers):
    def unpack(serialized):
        samples = []
        for data in serialized:
            sample = pickle.loads(data)
            samples.append(sample)
            print('TEACHERS', sample.name, sample.teacher_outcomes, 'pair', sample.preference is not None, flush=True)
        return samples

    if workers == 1:
        return unpack(map(collect_teachers, layouts))
    with ProcessPoolExecutor(max_workers=workers, initializer=configure_cpu) as executor:
        return unpack(executor.map(collect_teachers, layouts))


@torch.no_grad()
def normalize_inputs(model, samples, variant):
    total = torch.zeros_like(model.assignment_mean)
    squares = torch.zeros_like(total)
    count = 0
    for sample in samples:
        choices = sample.assignment[variant]
        embeddings = model.encode_static(sample.space.graph)
        rows = model.embed_assignment_rows(sample.space, embeddings, choices.indices, choices.features)
        total += rows.sum(0)
        squares += rows.square().sum(0)
        count += len(rows)
    model.assignment_mean.copy_(total / count)
    model.assignment_scale.copy_((squares / count - model.assignment_mean.square()).clamp_min(.05**2).sqrt())
    fine = torch.cat([sample.insertion.features for sample in samples])
    model.insertion_mean.copy_(fine.mean(0))
    model.insertion_scale.copy_(fine.std(0, unbiased=False).clamp_min(.05))


def ranking_loss(logits, choices):
    return torch.stack([-(target * scores.log_softmax(0)).sum()
                        for scores, target in zip(logits.split(choices.sizes), choices.targets.split(choices.sizes))]).mean()


def construct_gnn(parents, model):
    started = time.perf_counter()
    try:
        return gnn.construct_vnext_gnn(parents, model, CONFIG)
    except (ValueError, ArithmeticError) as error:
        return gnn.JointCompletionResult(None, None, (), time.perf_counter() - started, rejection_reason=str(error))


def validate(model, layouts):
    cmax = []
    for _, parents in layouts:
        result = construct_gnn(parents, model)
        if result.solution is not None:
            schedule = b32(result)
            if schedule.feasible:
                cmax.append(schedule.cmax)
    return dict(feasible=len(cmax), median_cmax_s=median(cmax) if cmax else None)


def fit(name, variant, samples, validation):
    torch.manual_seed(SEED)
    model = gnn.VNextGNN(workload_lookahead=METHODS[name])
    normalize_inputs(model, samples, variant)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    order = list(range(len(samples)))
    rng = Random(SEED)
    best_key, best_state, selected_epoch = (-1, -math.inf), None, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(order)
        losses = torch.zeros(3)
        for index in order:
            sample = samples[index]
            embeddings = model.encode_static(sample.space.graph)
            coarse, fine = sample.assignment[variant], sample.insertion
            rows = model.embed_assignment_rows(sample.space, embeddings, coarse.indices, coarse.features)
            assignment = ranking_loss(model.assignment_scores(rows), coarse)
            logits = model.insertion_logits(embeddings, fine.indices, fine.features)
            insertion = ranking_loss(logits, fine)
            cmax = sample.assignment_quality * assignment + sample.insertion_quality * insertion
            if sample.preference is not None:
                better, worse = sample.preference
                cmax = cmax + F.softplus(logits[worse] - logits[better])
            loss = assignment + insertion + .5 * cmax
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses += torch.stack((assignment.detach(), insertion.detach(), cmax.detach()))
        record = dict(epoch=epoch, **dict(zip(('assignment_loss', 'insertion_loss', 'cmax_loss'),
                                             (losses / len(samples)).tolist())))
        if epoch in VALIDATE_EPOCHS:
            record['validation'] = validate(model.eval(), validation)
            result = record['validation']
            key = result['feasible'], -(result['median_cmax_s'] or math.inf)
            if key > best_key:
                best_key, selected_epoch = key, epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(record)
        print('EPOCH', name, json.dumps(record), flush=True)
    return dict(state_dict=best_state, selected_epoch=selected_epoch, history=history)


def train(workers):
    started = time.perf_counter()
    training, validation = load_layouts('train'), load_layouts('validation')
    samples = prepare_training(training, workers)
    info = dict(layouts=len(samples), validation_layouts=len(validation),
                assignment_prefixes=sum(len(s.assignment[0].sizes) for s in samples),
                insertion_prefixes=sum(len(s.insertion.sizes) for s in samples),
                cmax_pairs=sum(s.preference is not None for s in samples),
                teachers={s.name: s.teacher_outcomes for s in samples},
                teacher_seconds=time.perf_counter() - started)
    print('TRAIN_DATA', json.dumps({k: v for k, v in info.items() if k != 'teachers'}), flush=True)
    models = {name: fit(name, variant, samples, validation) for variant, name in enumerate(METHODS)}
    torch.save(dict(seed=SEED, epochs=EPOCHS, hidden_dim=64, heads=4, training=info, models=models,
                    elapsed_s=time.perf_counter() - started), MODEL)
    print('MODEL', MODEL, flush=True)


def test_row(name, method, result, schedule, calls):
    state = result.oriented_state
    row = dict(layout_id=name, method=method, construct_s=result.inference_seconds,
               status=schedule.status.value if schedule else 'CONSTRUCTION_FAILED',
               cmax_s=schedule.cmax if schedule else None, b32_calls=calls,
               deadlock_edges=json.dumps(schedule.wait_for_graph) if schedule else '[]',
               diagnostic=' | '.join(schedule.diagnostics) if schedule else result.rejection_reason)
    waits = [0.] * 4
    if schedule:
        for op in schedule.operations:
            if op.kind is OperationKind.WAIT:
                waits[op.robot_id] += op.duration
    for r in range(4):
        row[f'processing_r{r}_s'] = state.loads[r] if state else None
        row[f'move_r{r}_s'] = sum(state.arc_costs[r]) if state else None
        row[f'wait_r{r}_s'] = waits[r] if schedule else None
    return row


def test():
    if RESULTS.exists():
        raise FileExistsError(f'TEST results already exist: {RESULTS}')
    saved = torch.load(MODEL, map_location='cpu', weights_only=True)
    models = {}
    for name, workload in METHODS.items():
        model = gnn.VNextGNN(saved['hidden_dim'], saved['heads'], workload_lookahead=workload)
        model.load_state_dict(saved['models'][name]['state_dict'])
        models[name] = model.eval()
    rows = []
    for name, parents in load_layouts('test'):
        evaluated = []
        for method in ('RULE/BEAM', *models):
            result = rule_beam(parents) if method == 'RULE/BEAM' else construct_gnn(parents, models[method])
            schedule, calls = None, 0
            if result.solution is not None:
                schedule = next((s for previous, s in evaluated if same_plan(previous, result)), None)
                if schedule is None:
                    schedule, calls = b32(result), 1
                    evaluated.append((result, schedule))
            row = test_row(name, method, result, schedule, calls)
            rows.append(row)
            print('TEST', name, method, row['status'], row['cmax_s'], flush=True)
    with RESULTS.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print('RESULTS', RESULTS, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'test'))
    parser.add_argument('--workers', type=int, default=4, help='TRAIN-only offline teacher processes')
    args = parser.parse_args()
    configure_cpu()
    if args.stage == 'train':
        train(args.workers)
    else:
        test()


if __name__ == '__main__':
    main()
